import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl

from sutro.sdk import Sutro


def sutro_client() -> Sutro:
    client = object.__new__(Sutro)
    client.api_key = "test-api-key"
    client.base_url = "https://api.sutro.sh"
    client.serving_base_url = "https://serve.sutro.sh"
    return client


class TestIdColumn(unittest.TestCase):
    def test_infer_sends_id_column_name_for_download_url(self):
        client = sutro_client()
        response = MagicMock(status_code=200)
        response.json.return_value = {"results": "job-123"}

        with patch.object(client, "do_request", return_value=response) as do_request:
            job_id = client.infer(
                "https://example.com/inputs.parquet?signature=abc",
                column="prompt",
                id_column="experience_id",
                stay_attached=False,
            )

        self.assertEqual(job_id, "job-123")
        payload = do_request.call_args.kwargs["json"]
        self.assertEqual(payload["column_name"], "prompt")
        self.assertEqual(payload["id_column_name"], "experience_id")

    def test_infer_rejects_id_column_for_non_url_input(self):
        client = sutro_client()

        with self.assertRaisesRegex(
            ValueError,
            r"id_column is only supported for HTTP\(S\) download URL inputs",
        ):
            client.infer(
                ["prompt"],
                id_column="experience_id",
                stay_attached=False,
            )

    def test_batch_run_function_passes_download_url_and_id_column_through(self):
        client = sutro_client()
        url = "https://example.com/function-inputs.parquet?signature=abc"

        with patch.object(client, "infer", return_value=None) as infer:
            client.batch_run_function(
                name="experience-facets",
                data=url,
                id_column="experience_id",
            )

        self.assertEqual(infer.call_args.kwargs["data"], url)
        self.assertEqual(
            infer.call_args.kwargs["id_column"],
            "experience_id",
        )

    def test_batch_run_function_rejects_attached_mode_for_download_url(self):
        client = sutro_client()
        url = "https://example.com/function-inputs.parquet?signature=abc"

        with (
            patch.object(client, "infer") as infer,
            self.assertRaisesRegex(
                ValueError,
                r"stay_attached=True is not supported for HTTP\(S\) Function inputs",
            ),
        ):
            client.batch_run_function(
                name="experience-facets",
                data=url,
                stay_attached=True,
            )

        infer.assert_not_called()

    def test_get_job_results_preserves_id_column(self):
        client = sutro_client()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": {
                "outputs": ["result-1", "result-2"],
                "experience_id": [101, 102],
            }
        }

        with (
            patch.object(client, "do_request", return_value=response),
            patch("sutro.sdk._has_open_batch_traces", return_value=False),
        ):
            results = client.get_job_results(
                "job-123",
                disable_cache=True,
                unpack_json=False,
            )

        self.assertEqual(
            results.columns,
            ["experience_id", "inference_result"],
        )
        self.assertEqual(results["experience_id"].to_list(), [101, 102])

    def test_get_job_results_rejects_id_and_structured_output_field_collision(self):
        client = sutro_client()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": {
                "outputs": [
                    '{"customer_id": "generated-1"}',
                    '{"customer_id": "generated-2"}',
                ],
                "customer_id": ["customer-1", "customer-2"],
            }
        }

        with (
            patch.object(client, "do_request", return_value=response),
            patch("sutro.sdk._has_open_batch_traces", return_value=False),
            self.assertRaisesRegex(
                ValueError,
                "customer_id.*Set unpack_json=False",
            ),
        ):
            client.get_job_results(
                "job-123",
                disable_cache=True,
            )

    def test_get_job_results_preserves_id_when_unpacking_structured_output(self):
        client = sutro_client()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": {
                "outputs": ['{"label": "one"}', '{"label": "two"}'],
                "customer_id": ["customer-1", "customer-2"],
            }
        }

        with (
            patch.object(client, "do_request", return_value=response),
            patch("sutro.sdk._has_open_batch_traces", return_value=False),
        ):
            results = client.get_job_results(
                "job-123",
                disable_cache=True,
            )

        self.assertEqual(results.columns, ["customer_id", "label"])
        self.assertEqual(
            results["customer_id"].to_list(),
            ["customer-1", "customer-2"],
        )
        self.assertEqual(results["label"].to_list(), ["one", "two"])

    def test_get_job_results_does_not_overwrite_id_with_json_scratch_column(self):
        client = sutro_client()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": {
                "outputs": ['{"label": "one"}', '{"label": "two"}'],
                "output_column_json_decoded": ["customer-1", "customer-2"],
            }
        }

        with (
            patch.object(client, "do_request", return_value=response),
            patch("sutro.sdk._has_open_batch_traces", return_value=False),
        ):
            results = client.get_job_results(
                "job-123",
                disable_cache=True,
            )

        self.assertEqual(
            results.columns,
            ["output_column_json_decoded", "label"],
        )
        self.assertEqual(
            results["output_column_json_decoded"].to_list(),
            ["customer-1", "customer-2"],
        )
        self.assertEqual(results["label"].to_list(), ["one", "two"])

    def test_get_job_results_uses_cache_with_id_column(self):
        client = sutro_client()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "job-123.snappy.parquet"
            pl.DataFrame(
                {
                    "experience_id": [101, 102],
                    "inference_result": ["result-1", "result-2"],
                }
            ).write_parquet(cache_path)

            with (
                patch(
                    "sutro.sdk.os.path.expanduser",
                    return_value=str(cache_path),
                ),
                patch("sutro.sdk._has_open_batch_traces", return_value=False),
                patch.object(client, "do_request") as do_request,
            ):
                results = client.get_job_results(
                    "job-123",
                    unpack_json=False,
                )

        do_request.assert_not_called()
        self.assertEqual(
            results.columns,
            ["experience_id", "inference_result"],
        )
