import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow.parquet as pq

from sutro.sdk import Sutro


class TestJobResultsCache(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.client = object.__new__(Sutro)
        self.client.api_url = "https://offline.example.test"
        self.client.api_key = "offline"
        directory = contexts.enter_context(tempfile.TemporaryDirectory())
        self.cache_path = Path(directory) / "results.parquet"
        contexts.enter_context(
            patch.object(
                self.client,
                "_job_results_cache_file_path",
                return_value=str(self.cache_path),
            )
        )
        contexts.enter_context(
            patch.object(self.client, "_remove_legacy_job_results_cache_file")
        )
        self.traces = contexts.enter_context(
            patch("sutro.sdk._has_open_batch_traces", return_value=False)
        )
        contexts.enter_context(patch("sutro.sdk.yaspin", return_value=MagicMock()))
        self.response = MagicMock()
        self.response.json.return_value = {
            "results": {"outputs": ['{"label":"pass"}'], "customer_id": [101]}
        }
        self.request = contexts.enter_context(
            patch.object(self.client, "do_request", return_value=self.response)
        )

    def get_results(self, **kwargs):
        return self.client.get_job_results("offline-job", **kwargs)

    def assert_cache_hit(self, **kwargs):
        self.request.reset_mock()
        self.request.side_effect = AssertionError(
            "Cache lookup must not request the API"
        )
        result = self.get_results(**kwargs)
        self.request.assert_not_called()
        return result

    def test_custom_output_names_share_the_same_raw_cache(self):
        self.get_results(output_column="first_result")
        for output_column in (
            "inference_result",
            "next_result",
            "outputs",
            "*",
            "^output$",
        ):
            with self.subTest(column=output_column):
                result = self.assert_cache_hit(
                    output_column=output_column, unpack_json=False
                )
                self.assertEqual(result[output_column].to_list(), ['{"label":"pass"}'])
                self.assertEqual(result["customer_id"].to_list(), [101])

    def test_source_id_named_inference_result_is_never_parsed_as_output(self):
        source_id = '{"source_key":"customer-id"}'
        self.response.json.return_value = {
            "results": {
                "outputs": ['{"label":"pass"}'],
                "inference_result": [source_id],
            }
        }
        expected = [{"inference_result": source_id, "label": "pass"}]
        self.assertEqual(
            self.get_results(output_column="model_result").to_dicts(), expected
        )
        self.assertEqual(
            self.assert_cache_hit(output_column="other_result").to_dicts(), expected
        )
        with self.assertRaisesRegex(ValueError, "output_column"):
            self.get_results()
        self.request.assert_not_called()
        # A rejected destination must not destroy either cached source column.
        raw = self.assert_cache_hit(output_column="model_result", unpack_json=False)
        self.assertEqual(raw["inference_result"].to_list(), [source_id])
        self.assertEqual(raw["model_result"].to_list(), ['{"label":"pass"}'])

    def test_fresh_output_name_collision_has_the_same_error(self):
        self.response.json.return_value = {
            "results": {"outputs": ["answer"], "inference_result": ["source-id"]}
        }
        with self.assertRaisesRegex(ValueError, "output_column"):
            self.get_results(disable_cache=True)

    def test_unpacking_does_not_change_cached_raw_outputs(self):
        expected = self.get_results()
        cached = pl.read_parquet(self.cache_path)
        self.assertEqual(cached["outputs"].to_list(), ['{"label":"pass"}'])
        self.assertNotIn("label", cached.columns)
        self.assertTrue(self.assert_cache_hit().equals(expected))
        raw = self.assert_cache_hit(unpack_json=False)
        self.assertEqual(raw["inference_result"].to_list(), ['{"label":"pass"}'])

    def test_output_column_cannot_impersonate_requested_optional_data(self):
        for field, flag in (
            ("inputs", "include_inputs"),
            ("cumulative_logprobs", "include_cumulative_logprobs"),
        ):
            with self.subTest(field=field):
                self.cache_path.unlink(missing_ok=True)
                self.request.reset_mock()
                self.response.json.return_value = {"results": {"outputs": ["answer"]}}
                self.get_results(output_column=field)
                self.response.json.return_value = {
                    "results": {"outputs": ["answer"], field: ["actual optional data"]}
                }
                self.request.reset_mock()
                with self.assertRaisesRegex(ValueError, "output_column"):
                    self.get_results(output_column=field, **{flag: True})
                self.request.assert_called_once()
                self.assertTrue(self.request.call_args.kwargs["json"][flag])

    def test_requested_optional_capabilities_can_satisfy_smaller_requests(self):
        self.response.json.return_value = {
            "results": {
                "outputs": ["answer"],
                "inputs": ["prompt"],
                "cumulative_logprobs": [-1.2],
                "confidence_score": [0.6],
                "customer_id": [101],
            }
        }
        self.get_results(include_inputs=True, include_cumulative_logprobs=True)
        smaller = self.assert_cache_hit()
        self.assertEqual(
            smaller.columns, ["customer_id", "inference_result", "confidence_score"]
        )
        full = self.assert_cache_hit(
            include_inputs=True, include_cumulative_logprobs=True
        )
        self.assertEqual(full["inputs"].to_list(), ["prompt"])
        self.assertEqual(full["cumulative_logprobs"].to_list(), [-1.2])

    def test_disabled_cached_diagnostics_can_be_used_as_output_destinations(self):
        for field, flag in (
            ("inputs", "include_inputs"),
            ("cumulative_logprobs", "include_cumulative_logprobs"),
        ):
            with self.subTest(field=field):
                self.cache_path.unlink(missing_ok=True)
                self.request.side_effect = None
                self.response.json.return_value = {
                    "results": {"outputs": ["answer"], field: ["optional data"]}
                }
                self.get_results(**{flag: True})
                cached = self.assert_cache_hit(output_column=field)
                self.assertEqual(cached.to_dicts(), [{field: "answer"}])
                self.request.side_effect = None
                self.response.json.return_value = {"results": {"outputs": ["answer"]}}
                fresh = self.get_results(output_column=field, disable_cache=True)
                self.assertTrue(cached.equals(fresh))
                # Filtering a caller's view must retain the full cache for later use.
                full = self.assert_cache_hit(**{flag: True})
                self.assertEqual(full[field].to_list(), ["optional data"])

    def test_unrequested_optional_names_remain_carried_metadata(self):
        for field in ("inputs", "cumulative_logprobs"):
            with self.subTest(field=field):
                self.cache_path.unlink(missing_ok=True)
                self.request.side_effect = None
                self.response.json.return_value = {
                    "results": {"outputs": ["answer"], field: ["customer-id"]}
                }
                fresh = self.get_results(output_column="model_result")
                self.assertEqual(fresh[field].to_list(), ["customer-id"])
                cached = self.assert_cache_hit(output_column="other_result")
                self.assertEqual(cached[field].to_list(), ["customer-id"])
                with self.assertRaisesRegex(ValueError, "output_column"):
                    self.get_results(output_column=field)
                self.request.assert_not_called()
                self.assertEqual(
                    pl.read_parquet(self.cache_path)[field].to_list(), ["customer-id"]
                )

    def test_missing_optional_columns_do_not_force_repeated_requests(self):
        self.get_results(include_inputs=True, include_cumulative_logprobs=True)
        self.assert_cache_hit(include_inputs=True, include_cumulative_logprobs=True)

    def test_larger_optional_requests_are_cache_misses(self):
        self.get_results()
        self.response.json.return_value = {
            "results": {
                "outputs": ["answer"],
                "inputs": ["prompt"],
                "customer_id": [101],
            }
        }
        self.request.reset_mock()
        result = self.get_results(include_inputs=True)
        self.request.assert_called_once()
        self.assertEqual(result["inputs"].to_list(), ["prompt"])
        self.response.json.return_value = {
            "results": {
                "outputs": ["answer"],
                "cumulative_logprobs": [-1.2],
                "customer_id": [101],
            }
        }
        self.request.reset_mock()
        result = self.get_results(include_cumulative_logprobs=True)
        self.request.assert_called_once()
        self.assertEqual(result["cumulative_logprobs"].to_list(), [-1.2])

    def test_unmarked_legacy_cache_is_a_miss_even_if_column_names_match(self):
        pl.DataFrame({"inference_result": ['{"source_key":"id"}']}).write_parquet(
            self.cache_path
        )
        result = self.get_results()
        self.request.assert_called_once()
        self.assertEqual(result["label"].to_list(), ["pass"])

    def test_disable_cache_does_not_read_or_write_corrupt_cache(self):
        self.cache_path.write_bytes(b"not parquet")
        with patch(
            "sutro.sdk.pq.read_schema", side_effect=AssertionError("Cache is disabled")
        ) as schema:
            result = self.get_results(disable_cache=True)
        schema.assert_not_called()
        self.request.assert_called_once()
        self.assertEqual(result["label"].to_list(), ["pass"])
        self.assertEqual(self.cache_path.read_bytes(), b"not parquet")

    def test_malformed_or_unknown_cache_metadata_is_a_miss(self):
        for marker in (b"not JSON", b"[]", b'{"version":999}', b"null"):
            with self.subTest(marker=marker):
                table = pl.DataFrame(
                    {"outputs": ["stale"], "inference_result": ["stale"]}
                ).to_arrow()
                pq.write_table(
                    table.replace_schema_metadata({b"sutro.job_results": marker}),
                    self.cache_path,
                )
                self.request.reset_mock()
                result = self.get_results()
                self.request.assert_called_once()
                self.assertEqual(result["label"].to_list(), ["pass"])

    def test_unreadable_cache_metadata_is_a_miss(self):
        self.cache_path.write_bytes(b"not parquet")
        for error in (PermissionError("unreadable"), ValueError("bad parquet")):
            with self.subTest(error=error):
                self.request.reset_mock()
                with patch("sutro.sdk.pq.read_schema", side_effect=error):
                    result = self.get_results()
                self.request.assert_called_once()
                self.assertEqual(result["label"].to_list(), ["pass"])

    def test_cached_traces_receive_raw_outputs_and_never_customer_ids(self):
        source_id = '{"source_key":"customer-id"}'
        outputs = ['{"label":"pass"}']
        self.response.json.return_value = {
            "results": {"outputs": outputs, "inference_result": [source_id]}
        }
        self.get_results(output_column="model_result")
        self.traces.return_value = True
        with (
            patch.object(self.client, "_fetch_job", return_value={"name": "test"}),
            patch("sutro.sdk._complete_batch_traces") as complete,
        ):
            result = self.assert_cache_hit(output_column="next_result")
        self.assertEqual(result["inference_result"].to_list(), [source_id])
        self.assertEqual(complete.call_args.kwargs["outputs"], outputs)
        self.assertEqual(complete.call_args.kwargs["num_rows"], 1)

    def test_fresh_traces_receive_raw_outputs(self):
        self.traces.return_value = True
        with (
            patch.object(self.client, "_fetch_job", return_value={}),
            patch("sutro.sdk._complete_batch_traces") as complete,
        ):
            self.get_results(output_column="model_result")
        self.assertEqual(complete.call_args.kwargs["outputs"], ['{"label":"pass"}'])

    def test_cache_marker_records_provenance_and_request_flags(self):
        self.get_results(output_column="custom", include_inputs=True)
        marker = json.loads(
            pq.read_schema(self.cache_path).metadata[b"sutro.job_results"]
        )
        self.assertEqual(
            marker,
            {
                "version": 1,
                "output_key": "outputs",
                "include_inputs": True,
                "include_cumulative_logprobs": False,
            },
        )
