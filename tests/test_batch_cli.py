import io
import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import boto3
from botocore.awsrequest import AWSResponse
from botocore.config import Config
from click.testing import CliRunner

from sutro import _batch as batch
from sutro.cli import cli
from sutro.sdk import Sutro


BASE = "https://deployment.example.test/v1"
KEY = "sk_unit_test_not_a_credential"
OWNER = "22222222-2222-4222-8222-222222222222"
DATASET = "s3://customer-data/source.jsonl"
SOURCE = {
    "bucket": "customer-data",
    "key": "source.jsonl",
    "etag": '"etag-original"',
    "size_bytes": 123,
    "version_id": "version-original",
}
JOB = "job-11111111-1111-4111-8111-111111111111"


class Response:
    def __init__(self, status=200, data=None, raw=b""):
        self.status_code, self.data, self.raw = status, data, io.BytesIO(raw)

    def json(self):
        return self.data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeClient:
    api_url, api_key = BASE, KEY
    batch_run_function = Sutro.batch_run_function

    def __init__(self, output):
        self.output = output
        self.calls, self.downloads, self.submissions = [], [], {}
        self.capable = True
        self.post_mode = None
        self.statuses = ["SUCCEEDED"]
        self.download_mode = None
        self.accepted_job = JOB
        self.lookup_mode = None
        self.ingesting_reads = 0
        self.owner_id = OWNER
        self.request_fingerprints = {}

    def do_request(self, method, endpoint, **kwargs):
        self.calls.append((method, endpoint, kwargs))
        if endpoint == "batch-capabilities":
            return Response(
                data={
                    "idempotent_submission": 1 if self.capable else 0,
                    "submission_owner_id": self.owner_id,
                    "input_formats": ["csv", "parquet", "jsonl"],
                    "result_formats": ["parquet"],
                }
            )
        if endpoint.startswith("batch-submissions/"):
            key = endpoint.rsplit("/", 1)[1]
            if self.lookup_mode:
                mode, self.lookup_mode = self.lookup_mode, None
                if mode == "retryable":
                    error_response = Response(503, {"retryable": True})
                    raise requests.HTTPError(response=error_response)
                if mode == "retryable_200":
                    return Response(
                        data={
                            "status": "RETRYABLE",
                            "retryable": True,
                            "code": "input_unavailable",
                            "idempotency_key": key,
                        }
                    )
                if mode == "failed_200":
                    return Response(
                        data={
                            "status": "FAILED",
                            "retryable": False,
                            "code": "invalid_jsonl",
                            "idempotency_key": key,
                        }
                    )
                return Response(
                    202,
                    {"status": "INGESTING", "retryable": True, "idempotency_key": key},
                )
            if key in self.submissions and self.ingesting_reads:
                self.ingesting_reads -= 1
                return Response(
                    data={
                        "status": "INGESTING",
                        "retryable": False,
                        "idempotency_key": key,
                    }
                )
            if key not in self.submissions:
                return Response(404)
            return Response(
                data={"status": "ACCEPTED", "idempotency_key": key, "results": self.submissions[key]}
            )
        if endpoint == "batch-inference":
            payload = kwargs["json"]
            receipt_path = Path(str(self.output) + ".sutro.json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["idempotency_key"] == payload["idempotency_key"]
                assert receipt["input_source"] == payload["input_source"]
                assert "signed-secret" not in json.dumps(receipt)
            mode, self.post_mode = self.post_mode, None
            if mode == "unaccepted-timeout":
                raise requests.Timeout("https://s3.invalid/?signed-secret")
            if mode == "failed":
                return Response(422, {"code": "invalid_jsonl"})
            key = payload["idempotency_key"]
            fingerprint = {k: v for k, v in payload.items() if k not in {"inputs", "idempotency_key"}}
            if key in self.request_fingerprints and self.request_fingerprints[key] != fingerprint:
                if mode == "conflict-timeout":
                    raise requests.Timeout("lost conflict response")
                return Response(409)
            self.request_fingerprints[key] = fingerprint
            self.submissions[payload["idempotency_key"]] = self.accepted_job
            if mode == "accepted-timeout":
                raise requests.Timeout("https://s3.invalid/?signed-secret")
            if mode == "interrupt":
                raise KeyboardInterrupt()
            if mode == "ingesting":
                return Response(
                    202,
                    {
                        "status": "INGESTING",
                        "retryable": False,
                        "idempotency_key": payload["idempotency_key"],
                    },
                )
            return Response(data={"results": self.accepted_job})
        if endpoint.startswith("job-status/"):
            status = (
                self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            )
            return Response(data={"job_status": {endpoint.split("/", 1)[1]: status}})
        if endpoint.startswith("stream-job-progress/"):
            return Response(raw=b'{"update_type":"progress","result":2}\n')
        raise AssertionError((method, endpoint, kwargs))

    def download_job_results(self, job_id, output_path, include_inputs=False):
        self.downloads.append((job_id, output_path, include_inputs))
        if self.download_mode == "interrupted":
            self.download_mode = None
            Path(output_path + ".part").write_bytes(b"partial")
            return None
        rows = {
            job_id: ['{"label":"positive"}', '{"label":"negative"}'],
            "record_id": ["a", "b"],
            "confidence_score": [1.0, 0.8],
        }
        if include_inputs:
            rows["inputs"] = ["good", "bad"]
        pq.write_table(pa.table(rows), output_path)
        return output_path


class BatchCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / "results.parquet"
        self.client = FakeClient(self.output)
        self.messages = []
        for name, replacement in (
            ("_source_identity", MagicMock(return_value=(SOURCE.copy(), "us-west-2"))),
            (
                "_signed_input",
                MagicMock(return_value="https://s3.invalid/source.jsonl?signed-secret"),
            ),
            ("_progress", MagicMock()),
        ):
            patcher = patch.object(batch, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
            setattr(self, name, replacement)
        sleep = patch.object(batch.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        network = patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("No network in tests"),
        )
        network.start()
        self.addCleanup(network.stop)

    def run_batch(self, **options):
        return batch.run_batch(
            self.client,
            function="sentiment",
            dataset=DATASET,
            output_file_name=self.output,
            emit=self.messages.append,
            **options,
        )

    def posts(self):
        return [
            kwargs["json"]
            for method, _, kwargs in self.client.calls
            if method == "POST"
        ]

    def submit_sdk(self, **changes):
        options = {
            "name": "sentiment",
            "data": "https://s3.invalid/source.jsonl?signed-secret",
            "idempotency_key": "44444444-4444-4444-8444-444444444444",
            "input_source": SOURCE,
        }
        return self.client.batch_run_function(**{**options, **changes})

    def receipt(self):
        return json.loads(Path(str(self.output) + ".sutro.json").read_text())

    def test_success_submits_p1_and_writes_normalized_parquet(self):
        self.assertEqual(self.run_batch(id_column="record_id"), JOB)
        self.assertEqual(len(self.posts()), 1)
        payload = self.posts()[0]
        self.assertEqual(payload["job_priority"], 1)
        self.assertEqual(payload["input_source"], SOURCE)
        self.assertEqual(payload["id_column_name"], "record_id")
        self.assertFalse(payload["truncate_rows"])
        self.assertLessEqual(len(payload["name"]), 45)
        self.assertEqual(
            pq.read_table(self.output).to_pydict(),
            {
                "record_id": ["a", "b"],
                "confidence_score": [1.0, 0.8],
                "label": ["positive", "negative"],
            },
        )
        self.assertEqual(self.receipt()["status"], "DOWNLOADED")
        self.assertNotIn("signed-secret", "\n".join(self.messages))
        self.assertNotIn(KEY, Path(str(self.output) + ".sutro.json").read_text())

    def test_repeated_completed_command_is_no_network_no_post(self):
        self.run_batch()
        before = len(self.client.calls)
        self.run_batch()
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(len(self.posts()), 1)
        self._source_identity.assert_called_once()

    def test_uncertain_accepted_post_is_looked_up_not_repeated(self):
        self.client.post_mode = "accepted-timeout"
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)

    def test_uncertain_unaccepted_post_reuses_key_and_frozen_source(self):
        self.client.post_mode = "unaccepted-timeout"
        self.run_batch()
        self.assertEqual(len(self.posts()), 2)
        self.assertEqual(
            self.posts()[0]["idempotency_key"], self.posts()[1]["idempotency_key"]
        )
        self.assertEqual(
            self.posts()[0]["input_source"], self.posts()[1]["input_source"]
        )
        self._source_identity.assert_called_once()
        self.assertNotIn("signed-secret", "\n".join(self.messages))

    def test_async_ingestion_is_polled(self):
        self.client.post_mode = "ingesting"
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)
        self.assertTrue(any("INGESTING" in message for message in self.messages))

    def test_active_ingestion_status_200_is_polled_without_resubmission(self):
        self.client.post_mode = "ingesting"
        self.client.ingesting_reads = 2
        self.assertEqual(self.run_batch(), JOB)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.client.ingesting_reads, 0)
        self.assertEqual(self.receipt()["status"], "DOWNLOADED")

    def test_lost_submission_response_polls_active_ingestion(self):
        self.client.post_mode = "accepted-timeout"
        self.client.ingesting_reads = 2
        self.assertEqual(self.run_batch(), JOB)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.client.ingesting_reads, 0)

    def test_expired_ingestion_lease_can_repost_same_key(self):
        self.client.lookup_mode = "expired"
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)

    def test_retryable_503_can_repost_same_key(self):
        self.client.lookup_mode = "retryable"
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)

    def test_retryable_200_can_repost_same_key(self):
        self.client.lookup_mode = "retryable_200"
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)

    def test_failed_status_prints_server_code_and_new_run_hint(self):
        self.client.lookup_mode = "failed_200"
        with self.assertRaisesRegex(
            batch.ApiError, r"HTTP 422 \[invalid_jsonl\].*--new-run"
        ):
            self.run_batch()
        self.assertFalse(self.posts())

    def test_ctrl_c_receipt_resumes_accepted_job_without_aws(self):
        self.client.post_mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch()
        saved_key = self.receipt()["idempotency_key"]
        self.run_batch()
        self.assertEqual(self.receipt()["idempotency_key"], saved_key)
        self.assertEqual(len(self.posts()), 1)
        self._source_identity.assert_called_once()

    def test_failed_job_is_not_resubmitted(self):
        self.client.statuses = ["FAILED"]
        for _ in range(2):
            with self.assertRaisesRegex(batch.BatchError, "Job FAILED"):
                self.run_batch()
        self.assertEqual(len(self.posts()), 1)
        self.assertFalse(self.client.downloads)
        self.assertFalse(
            any("cancel" in endpoint for _, endpoint, _ in self.client.calls)
        )

    def test_terminal_and_partial_jobs_stop_without_download_or_resubmission(self):
        for status in ("CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED"):
            with self.subTest(status=status):
                self.client.statuses = [status]
                with patch.object(batch, "_pause", side_effect=AssertionError("terminal job polled")):
                    with self.assertRaisesRegex(batch.BatchError, f"Job {status}"):
                        self.run_batch()
                self.assertEqual(self.receipt()["status"], status)
                self.assertFalse(self.client.downloads)
        self.assertEqual(len(self.posts()), 1)

    def test_result_retry_refreshes_the_server_artifact(self):
        downloads = []

        def download(job_id, output_path, include_inputs=False):
            downloads.append(job_id)
            values = ['{"record_id":"collision"}'] if len(downloads) == 1 else ['{"label":"current"}']
            pq.write_table(pa.table({job_id: values, "record_id": ["source"]}), output_path)
            return output_path

        self.client.download_job_results = download
        with self.assertRaisesRegex(batch.BatchError, "--no-unpack-json"):
            self.run_batch()
        self.run_batch(unpack_json=False)
        self.assertEqual(len(downloads), 2)
        self.assertEqual(pq.read_table(self.output)["inference_result"].to_pylist(), ['{"label":"current"}'])
        self.assertEqual(len(self.posts()), 1)

    def test_status_polling_completes_when_progress_is_unavailable(self):
        self.client.statuses = ["RUNNING", "SUCCEEDED"]
        self.run_batch()
        self.assertTrue(any("RUNNING" in message for message in self.messages))
        self.assertEqual(len(self.posts()), 1)

    def test_timeout_keeps_receipt_and_resume_does_not_submit(self):
        self.client.statuses = ["RUNNING"]
        clock = [0]
        with (
            patch.object(batch.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(
                batch.time,
                "sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
        ):
            with self.assertRaisesRegex(batch.BatchError, "time limit"):
                self.run_batch(timeout=1)
        self.assertEqual(self.receipt()["job_id"], JOB)
        self.client.statuses = ["SUCCEEDED"]
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)
        self.assertFalse(
            any("cancel" in endpoint for _, endpoint, _ in self.client.calls)
        )

    def test_receipt_write_failure_prevents_post(self):
        with patch.object(batch, "_save_receipt", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_batch()
        self.assertFalse(self.posts())

    def test_collision_can_be_resumed_as_raw_without_new_inference(self):
        def download(job_id, output_path, include_inputs=False):
            pq.write_table(
                pa.table(
                    {job_id: ['{"record_id":"generated"}'], "record_id": ["original"]}
                ),
                output_path,
            )
            return output_path

        self.client.download_job_results = download
        with self.assertRaisesRegex(batch.BatchError, "--no-unpack-json"):
            self.run_batch()
        self.assertFalse(self.output.exists())
        self.run_batch(unpack_json=False)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(
            pq.read_table(self.output)["record_id"].to_pylist(), ["original"]
        )

    def test_download_failure_resumes_same_job(self):
        self.client.download_mode = "interrupted"
        with self.assertRaisesRegex(batch.BatchError, "download incomplete"):
            self.run_batch()
        self.run_batch()
        self.assertEqual(len(self.posts()), 1)
        self._source_identity.assert_called_once()

    def test_existing_output_rejected_before_submission(self):
        self.output.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(batch.BatchError, "Output already exists"):
            self.run_batch()
        self.assertEqual(self.output.read_bytes(), b"do not overwrite")
        self.assertFalse(self.client.calls)
        self._source_identity.assert_not_called()

    def test_old_backend_is_rejected_without_post_or_receipt(self):
        self.client.capable = False
        with self.assertRaisesRegex(batch.BatchError, "required batch capabilities"):
            self.run_batch()
        self.assertFalse(self.posts())
        self.assertFalse(Path(str(self.output) + ".sutro.json").exists())
        self._source_identity.assert_not_called()

    def test_different_request_needs_explicit_new_run(self):
        self.run_batch()
        with self.assertRaisesRegex(batch.BatchError, "different run"):
            self.run_batch(id_column="different")
        self.assertEqual(len(self.posts()), 1)

    def test_explicit_new_run_gets_new_key_and_archives_receipt(self):
        self.run_batch()
        old_key = self.receipt()["idempotency_key"]
        self.run_batch(new_run=True, overwrite=True)
        self.assertEqual(len(self.posts()), 2)
        self.assertNotEqual(self.receipt()["idempotency_key"], old_key)
        self.assertTrue(Path(str(self.output) + ".sutro.json." + old_key).exists())

    def test_output_options_can_change_without_resubmission(self):
        self.run_batch()
        self.run_batch(unpack_json=False, include_inputs=True, overwrite=True)
        self.assertEqual(len(self.posts()), 1)
        columns = pq.read_table(self.output).column_names
        self.assertIn("inference_result", columns)
        self.assertIn("inputs", columns)

    def test_read_only_resume_uses_neither_capabilities_nor_aws(self):
        batch.run_batch(
            self.client,
            resume_job_id=JOB,
            output_file_name=self.output,
            emit=self.messages.append,
        )
        self.assertFalse(self.posts())
        self.assertFalse(
            any(
                endpoint == "batch-capabilities" for _, endpoint, _ in self.client.calls
            )
        )
        self._source_identity.assert_not_called()
        self._signed_input.assert_not_called()

    def test_different_deployment_receipt_rejected(self):
        self.run_batch()
        self.client.api_url = "https://other.example.test/v1"
        with self.assertRaisesRegex(batch.BatchError, "different deployment"):
            self.run_batch()
        self.assertEqual(len(self.posts()), 1)

    def test_rotated_api_key_can_resume_same_deployment(self):
        self.run_batch()
        before = len(self.client.calls)
        self.client.api_key = "sk_rotated_unit_test_not_a_credential"
        self.assertEqual(self.run_batch(), JOB)
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(self.receipt()["identity"], {"api_url": BASE, "owner_id": OWNER})

    def test_owner_switch_cannot_replay_an_unconfirmed_receipt(self):
        self.client.post_mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch()
        self.client.owner_id = "33333333-3333-4333-8333-333333333333"
        self.client.submissions.clear()  # Owner-scoped lookup cannot see the original.
        self._signed_input.reset_mock()
        with self.assertRaisesRegex(batch.BatchError, "different owner"):
            self.run_batch()
        self.assertEqual(len(self.posts()), 1)
        self._signed_input.assert_not_called()

    def test_same_owner_rotated_key_recovers_unconfirmed_receipt(self):
        self.client.post_mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch()
        self.client.api_key = "sk_rotated_unit_test_not_a_credential"
        self.assertEqual(self.run_batch(), JOB)
        self.assertEqual(len(self.posts()), 1)

    def test_legacy_unconfirmed_receipt_404_cannot_submit(self):
        self.client.post_mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch()
        receipt = self.receipt()
        receipt.update(version=2, identity={"api_url": BASE})
        Path(str(self.output) + ".sutro.json").write_text(json.dumps(receipt))
        self.client.submissions.clear()
        with self.assertRaisesRegex(batch.BatchError, "no owner identity"):
            self.run_batch()
        self.assertEqual(len(self.posts()), 1)

    def test_legacy_visible_submission_binds_owner_without_resubmitting(self):
        self.client.post_mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch()
        receipt = self.receipt()
        receipt.update(version=2, identity={"api_url": BASE})
        Path(str(self.output) + ".sutro.json").write_text(json.dumps(receipt))
        self.assertEqual(self.run_batch(), JOB)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.receipt()["identity"]["owner_id"], OWNER)

    def test_fallback_interruption_keeps_final_path_absent_and_resumes(self):
        with patch.object(batch.os, "link", side_effect=OSError("unsupported")), patch.object(
            batch, "_publish_no_replace", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.receipt()["status"], "PUBLISHING")
        self.assertEqual(self.run_batch(), JOB)
        self.assertEqual(len(self.posts()), 1)

    def test_publication_cannot_overwrite_a_racing_file(self):
        source = self.directory / "normalized.parquet"
        source.write_bytes(b"new output")
        self.output.write_bytes(b"existing output")
        with self.assertRaises(FileExistsError):
            batch._publish_no_replace(source, self.output)
        with patch.object(batch.os, "link", side_effect=OSError("unsupported")), self.assertRaises(FileExistsError):
            batch._publish_no_replace(source, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing output")
        self.assertTrue(source.exists())

    def test_version_one_receipt_migrates_without_resubmission(self):
        self.run_batch()
        receipt_path = Path(str(self.output) + ".sutro.json")
        receipt = self.receipt()
        receipt.update(
            version=1,
            identity={"api_url": BASE, "key_sha256": "obsolete-key-fingerprint"},
        )
        receipt_path.write_text(json.dumps(receipt))
        before = len(self.client.calls)

        self.client.api_key = "sk_rotated_unit_test_not_a_credential"
        self.assertEqual(self.run_batch(), JOB)

        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(self.receipt()["version"], 2)
        self.assertEqual(self.receipt()["identity"], {"api_url": BASE})

    def test_publication_renames_when_hard_links_are_unavailable(self):
        with patch.object(batch.os, "link", side_effect=OSError("unsupported")):
            self.assertEqual(self.run_batch(id_column="record_id"), JOB)
        self.assertTrue(self.output.exists())
        self.assertEqual(pq.read_table(self.output).num_rows, 2)

    def test_concurrent_same_output_fails_without_post(self):
        with batch._output_lock(self.output):
            with self.assertRaisesRegex(batch.BatchError, "Another process"):
                self.run_batch()
        self.assertFalse(self.client.calls)

    def test_publication_crash_recognizes_already_written_output(self):
        save = batch._save_receipt

        def fail_last_save(path, receipt):
            if receipt.get("status") == "DOWNLOADED":
                raise OSError("simulated crash")
            return save(path, receipt)

        with patch.object(batch, "_save_receipt", side_effect=fail_last_save):
            with self.assertRaises(OSError):
                self.run_batch()
        self.assertTrue(self.output.exists())
        self.assertEqual(self.receipt()["status"], "PUBLISHING")
        previous_calls = len(self.client.calls)
        self.run_batch()
        self.assertEqual(len(self.client.calls), previous_calls)
        self.assertEqual(self.receipt()["status"], "DOWNLOADED")

    def test_click_command_and_resume_flags(self):
        runner = CliRunner()
        with (
            patch.dict(os.environ, {"SUTRO_API_URL": BASE, "SUTRO_API_KEY": KEY}),
            patch("sutro.cli.get_sdk", return_value=self.client),
        ):
            result = runner.invoke(
                cli,
                [
                    "batch-run",
                    "--function",
                    "sentiment",
                    "--dataset",
                    DATASET,
                    "-o",
                    str(self.output),
                    "--id-column",
                    "record_id",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            result = runner.invoke(cli, ["jobs", "resume", JOB, "-o", str(self.output)])
            self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(self.posts()), 1)

    def test_existing_sdk_submission_recovers_lost_response_without_local_state(self):
        self.client.post_mode = "accepted-timeout"
        for _ in range(2):
            self.assertEqual(self.submit_sdk(), JOB)
        self.assertEqual(len(self.posts()), 3)
        self.assertEqual(len(self.client.submissions), 1)
        self.assertEqual(self.client.downloads, [])
        self.assertEqual(list(self.directory.iterdir()), [])
        self._source_identity.assert_not_called()
        self._signed_input.assert_not_called()
        self.assertEqual(self.posts()[0]["job_priority"], 0)

    def test_existing_sdk_submission_waits_for_ingestion_only(self):
        self.client.post_mode = "ingesting"
        self.client.ingesting_reads = 1
        self.assertEqual(self.submit_sdk(
            job_priority=1, id_column="record_id",
            job_name="named-run", description="description",
        ), JOB)
        self.assertFalse(any(endpoint.startswith("job-status/") for _, endpoint, _ in self.client.calls))
        self.assertEqual(self.client.downloads, [])
        self.assertEqual(self.posts()[0]["id_column_name"], "record_id")
        self.assertEqual(self.posts()[0]["name"], "named-run")

    def test_existing_sdk_rejects_unsupported_idempotent_inputs_before_requests(self):
        for changes in (
            {"input_source": None}, {"idempotency_key": None},
            {"data": [{"text": "hello"}]}, {"dry_run": True},
            {"stay_attached": True}, {"submission_timeout": 0},
            {"submission_timeout": float("inf")}, {"submission_timeout": float("nan")},
            {"idempotency_key": "invalid"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.submit_sdk(**changes)
        self.assertEqual(self.client.calls, [])

    def test_existing_sdk_old_deployment_never_receives_submission(self):
        self.client.capable = False
        self.assertIsNone(self.submit_sdk())
        self.assertEqual(self.posts(), [])

    def test_existing_sdk_interruption_recovers_same_key_without_a_receipt(self):
        self.client.post_mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.submit_sdk()
        self.assertEqual(self.submit_sdk(), JOB)
        self.assertEqual(len(self.posts()), 2)
        self.assertEqual(len(self.client.submissions), 1)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_sdk_ingestion_timeout_can_recover_without_another_job(self):
        self.client.post_mode = "ingesting"
        with patch.object(batch, "_pause", side_effect=batch.BatchError("Client time limit reached")):
            self.assertIsNone(self.submit_sdk())
        self.assertEqual(self.submit_sdk(), JOB)
        self.assertEqual(len(self.posts()), 2)
        self.assertEqual(len(self.client.submissions), 1)

    def test_existing_sdk_terminal_submission_does_not_repost(self):
        self.client.post_mode = "failed"
        self.assertIsNone(self.submit_sdk())
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.client.submissions, {})

    def test_public_sdk_has_no_file_workflow_or_batch_error_exports(self):
        import sutro
        for name in ("batch_run", "resume_job", "SutroBatchError"):
            self.assertFalse(hasattr(sutro, name), name)
            self.assertFalse(hasattr(Sutro, name), name)

    def test_existing_sdk_replay_checks_request_identity_with_the_server(self):
        self.assertEqual(self.submit_sdk(), JOB)
        self.assertEqual(self.submit_sdk(data="https://s3.invalid/source.jsonl?signature=fresh"), JOB)
        for changes in (
            {"name": "different-function"}, {"job_priority": 1},
            {"input_source": {**SOURCE, "etag": '"changed"'}},
            {"id_column": "different-id"}, {"job_name": "different-name"},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(self.submit_sdk(**changes))
        self.assertEqual(len(self.client.submissions), 1)

    def test_existing_sdk_lost_conflict_response_cannot_recover_a_different_request(self):
        self.assertEqual(self.submit_sdk(), JOB)
        self.client.post_mode = "conflict-timeout"
        self.assertIsNone(self.submit_sdk(name="different-function"))
        self.assertEqual(len(self.client.submissions), 1)

    def test_legacy_sdk_wait_stops_on_partial_and_retries_transient_reads(self):
        client = object.__new__(Sutro)
        with patch.object(client, "_fetch_job_status", side_effect=[
            requests.ConnectionError("private-url"), "RUNNING", "PARTIALLY_SUCCEEDED",
        ]) as status, patch.object(client, "get_job_results") as results:
            self.assertIsNone(client.await_job_completion(JOB))
        self.assertEqual(status.call_count, 3)
        results.assert_not_called()


class LocalHttpSmokeTests(unittest.TestCase):
    def test_existing_sdk_replays_idempotent_post_over_http_without_file_workflow(self):
        key = "44444444-4444-4444-8444-444444444444"
        contracts, calls = {}, []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, payload, status=200):
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                calls.append(("GET", self.path))
                if self.headers.get("Authorization") != "Key " + KEY:
                    return self.reply({}, 401)
                if self.path == "/v1/batch-capabilities":
                    return self.reply({"idempotent_submission": 1, "submission_owner_id": OWNER,
                                       "input_formats": ["jsonl"], "result_formats": ["parquet"]})
                self.reply({}, 404)

            def do_POST(self):
                calls.append(("POST", self.path))
                assert self.path == "/v1/batch-inference"
                assert self.headers.get("Authorization") == "Key " + KEY
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fingerprint = {k: v for k, v in payload.items() if k != "inputs"}
                if key in contracts and contracts[key] != fingerprint:
                    return self.reply({}, 409)
                if key not in contracts:
                    contracts[key] = fingerprint
                    self.close_connection = True  # Acceptance acknowledgement is lost.
                    return
                self.reply({"status": "ACCEPTED", "idempotency_key": key, "results": JOB})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = Sutro(api_url=f"http://127.0.0.1:{server.server_port}", api_key=KEY)
            options = {"name": "sentiment", "data": "https://s3.invalid/source.jsonl?signature=old",
                       "idempotency_key": key, "input_source": SOURCE, "id_column": "record_id"}
            with patch.object(batch, "_pause"), patch.object(client, "download_job_results") as download:
                self.assertEqual(client.batch_run_function(**options), JOB)
                self.assertEqual(client.batch_run_function(**{**options, "data": "https://s3.invalid/source.jsonl?signature=fresh"}), JOB)
                self.assertIsNone(client.batch_run_function(**{**options, "name": "different-function"}))
                download.assert_not_called()
            self.assertEqual(len(contracts), 1)
            self.assertEqual(contracts[key]["job_priority"], 0)
            self.assertEqual(contracts[key]["id_column_name"], "record_id")
            self.assertEqual(sum(method == "POST" for method, _ in calls), 4)
            self.assertFalse(any(path.startswith("/v1/job-status") for _, path in calls))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_interrupted_cli_resumes_over_http_and_downloads_without_api_credentials(self):
        # This is a loopback contract smoke. S3 inspection/signing and inference
        # are simulated; SDK requests, receipts and Parquet transfer are real.
        buffer = io.BytesIO()
        pq.write_table(pa.table({JOB: [json.dumps({"content": {"label": "yes"}, "reason": "line one\nline two"})], "record_id": [7]}), buffer)
        artifact = buffer.getvalue()
        submissions = []
        object_authorizations = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, data, status=200):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self):
                object_authorizations.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Length", str(len(artifact)))
                self.send_header("ETag", '"fixture"')
                self.end_headers()

            def do_GET(self):
                if self.path.startswith("/object"):
                    object_authorizations.append(self.headers.get("Authorization"))
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(artifact)))
                    self.send_header("ETag", '"fixture"')
                    self.end_headers()
                    self.wfile.write(artifact)
                    return
                if self.headers.get("Authorization") != "Key " + KEY:
                    self.reply({}, 401)
                elif self.path == "/v1/batch-capabilities":
                    self.reply({"idempotent_submission": 1, "submission_owner_id": OWNER, "input_formats": ["jsonl"], "result_formats": ["parquet"]})
                elif self.path.startswith("/v1/batch-submissions/"):
                    self.reply({"status": "ACCEPTED", "idempotency_key": submissions[0]["idempotency_key"], "results": JOB} if submissions else {}, 200 if submissions else 404)
                elif self.path.startswith("/v1/job-status/"):
                    self.reply({"job_status": {JOB: "SUCCEEDED"}})
                elif self.path.startswith("/v1/jobs/" + JOB + "/results-url"):
                    url = f"http://127.0.0.1:{self.server.server_port}/object?signature=private-fixture"
                    self.reply({"artifact": {"filename": "results.parquet", "size_bytes": len(artifact)}, "urls": {"get": url, "head": url}})
                else:
                    self.reply({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                submissions.append(body)
                self.reply({"status": "INGESTING", "idempotency_key": body["idempotency_key"]}, 202)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "SUTRO_API_URL": f"http://127.0.0.1:{server.server_port}", "SUTRO_API_KEY": KEY,
            }), patch.object(batch, "_source_identity", return_value=(SOURCE, "us-west-2")), patch.object(batch, "_signed_input", return_value="https://s3.invalid/?private-fixture"), patch.object(batch, "_progress"):
                output = Path(directory) / "result.parquet"
                command = ["batch-run", "--function", "sentiment", "--dataset", DATASET, "-o", str(output), "--timeout", "5"]
                runner = CliRunner()
                with patch.object(batch, "_pause", side_effect=KeyboardInterrupt):
                    interrupted = runner.invoke(cli, command)
                self.assertEqual(interrupted.exit_code, 130, interrupted.output)
                receipt = json.loads(Path(str(output) + ".sutro.json").read_text())
                resumed = runner.invoke(cli, command)
                self.assertEqual(resumed.exit_code, 0, resumed.output)
                self.assertEqual(len(submissions), 1)
                self.assertEqual(receipt["idempotency_key"], submissions[0]["idempotency_key"])
                self.assertNotIn("private-fixture", resumed.output)
                self.assertEqual(pq.read_table(output).to_pylist(), [{"content": {"label": "yes"}, "reason": "line one\nline two", "record_id": 7}])
                self.assertEqual(object_authorizations, [None, None])
                # Recover through the ordinary Python client on another output
                # path. This must only use the existing job, even without AWS.
                recovered = Path(directory) / "recovered.parquet"
                with patch.object(batch, "_source_identity", side_effect=AssertionError("No source credentials")), patch.object(batch, "_signed_input", side_effect=AssertionError("No source credentials")):
                    client = Sutro()
                    self.assertEqual(client.download_job_results(JOB, output_path=str(recovered)), str(recovered))
                self.assertEqual(pq.read_table(recovered).to_pylist(), [{JOB: json.dumps({"content": {"label": "yes"}, "reason": "line one\nline two"}), "record_id": 7}])
                self.assertEqual(len(submissions), 1)
                self.assertEqual(object_authorizations, [None] * 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class SourceAndTransportTests(unittest.TestCase):
    def test_s3_region_survives_object_redirect_without_head_object_region_header(self):
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            config=Config(signature_version="s3v4"),
        )
        requests_seen = []

        def send(request):
            requests_seen.append(request.url)
            raw = MagicMock()
            raw.stream.return_value = iter(())
            if len(requests_seen) == 1:
                return AWSResponse(
                    request.url, 301, {"x-amz-bucket-region": "us-west-2"}, raw
                )
            if request.url.endswith("source.jsonl"):
                return AWSResponse(
                    request.url, 200, {"ETag": '"e"', "Content-Length": "42"}, raw
                )
            # GetObject-only credentials may not permit ListBucket, but AWS
            # still returns the bucket region in this response.
            return AWSResponse(
                request.url, 403, {"x-amz-bucket-region": "us-west-2"}, raw
            )

        with (
            patch.object(client._endpoint.http_session, "send", side_effect=send),
            patch.object(batch, "_s3_client", return_value=client),
        ):
            source, region = batch._source_identity(DATASET)
        self.assertEqual(client.meta.region_name, "us-east-1")
        self.assertIn("s3.us-west-2.amazonaws.com", requests_seen[1])
        self.assertEqual(source["size_bytes"], 42)
        self.assertEqual(region, "us-west-2")

    def test_s3_head_pins_identity_and_region(self):
        client = MagicMock()
        client.head_object.return_value = {
            "ETag": '"e"',
            "ContentLength": 42,
            "VersionId": "v1",
            "ResponseMetadata": {"HTTPHeaders": {"x-amz-bucket-region": "us-west-2"}},
        }
        with patch.object(batch, "_s3_client", return_value=client):
            source, region = batch._source_identity(DATASET)
        self.assertEqual(
            source,
            {
                "bucket": "customer-data",
                "key": "source.jsonl",
                "etag": '"e"',
                "size_bytes": 42,
                "version_id": "v1",
            },
        )
        self.assertEqual(region, "us-west-2")
        client.head_bucket.assert_not_called()

    def test_missing_bucket_region_fails_before_signing(self):
        client = MagicMock()
        client.head_object.return_value = {"ETag": '"e"', "ContentLength": 42}
        client.head_bucket.return_value = {"ResponseMetadata": {"HTTPHeaders": {}}}
        with patch.object(batch, "_s3_client", return_value=client):
            with self.assertRaisesRegex(batch.BatchError, "bucket region"):
                batch._source_identity(DATASET)
        client.generate_presigned_url.assert_not_called()

    def test_signed_url_uses_saved_version_and_region(self):
        client = MagicMock()
        with patch.object(batch, "_s3_client", return_value=client) as factory:
            batch._signed_input(SOURCE, "us-west-2")
        factory.assert_called_once_with("us-west-2")
        client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={
                "Bucket": "customer-data",
                "Key": "source.jsonl",
                "VersionId": "version-original",
            },
            ExpiresIn=3600,
        )

    def test_bad_dataset_formats_fail_before_cloud_calls(self):
        for dataset in (
            "./data.csv",
            "https://data.example/a.csv",
            "s3://bucket/",
            "s3://bucket/*.csv",
            "s3://bucket/a.csv.gz",
        ):
            with self.subTest(dataset=dataset), self.assertRaises(batch.BatchError):
                batch._s3_location(dataset)

    def test_batch_transport_disables_redirects_and_retries_and_sets_timeout(self):
        client = MagicMock()
        client.do_request.return_value = Response(data={})
        with patch.object(batch.time, "monotonic", return_value=0):
            batch._json_request(client, "POST", "batch-inference", 45, {})
        call = client.do_request
        self.assertFalse(call.call_args.kwargs["allow_redirects"])
        self.assertEqual(call.call_args.kwargs["max_retries"], 0)
        self.assertEqual(call.call_args.kwargs["timeout"], (5, 45))
        self.assertIn("SutroBatch/", call.call_args.kwargs["headers"]["User-Agent"])

    def test_progress_only_prints_numeric_counters_and_falls_back_on_eof(self):
        client = MagicMock()
        client.do_request.return_value = Response(
            raw=(
                b'{"update_type":"progress","result":42}\n'
                b'{"update_type":"tokens","result":{"input_tokens":500,'
                b'"output_tokens":20,"secret":"do not print"}}\n'
                b'{"update_type":"sample","result":"private data"}\n'
            )
        )
        messages = []
        batch._progress(
            client,
            JOB,
            batch.time.monotonic() + 60,
            batch.threading.Event(),
            messages.append,
        )
        self.assertIn('"rows_completed": 42', "\n".join(messages))
        self.assertIn('"output_tokens": 20', "\n".join(messages))
        self.assertNotIn("private data", "\n".join(messages))
        self.assertNotIn("do not print", "\n".join(messages))
        self.assertIn("status polling continues", messages[-1])
        self.assertEqual(client.do_request.call_count, 2)
        for call in client.do_request.call_args_list:
            self.assertEqual(call.kwargs["timeout"], (5, None))


if __name__ == "__main__":
    unittest.main()
