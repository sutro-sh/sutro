import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from sutro.sdk import Sutro


class TestJobResultsMerge(unittest.TestCase):
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
        contexts.enter_context(
            patch("sutro.sdk._has_open_batch_traces", return_value=False)
        )
        contexts.enter_context(patch("sutro.sdk.yaspin", return_value=MagicMock()))

    def get_results(self, original, outputs, metadata=None, **kwargs):
        response = MagicMock()
        response.json.return_value = {
            "results": {"outputs": outputs, **(metadata or {})}
        }
        with patch.object(self.client, "do_request", return_value=response):
            return self.client.get_job_results(
                "offline-job", with_original_df=original, disable_cache=True, **kwargs
            )

    def test_diagnostics_and_inputs_cannot_replace_customer_columns(self):
        for frame in (pl.DataFrame, pd.DataFrame):
            for unpack_json in (False, True):
                for field, generated in (
                    ("inputs", "rendered prompt"),
                    ("confidence_score", 0.6),
                    ("cumulative_logprobs", -3.2),
                ):
                    with self.subTest(
                        frame=frame, field=field, unpack_json=unpack_json
                    ):
                        original = frame({field: ["customer value"]})
                        before = (
                            original.clone()
                            if frame is pl.DataFrame
                            else original.copy()
                        )
                        with self.assertRaisesRegex(ValueError, field):
                            self.get_results(
                                original,
                                ['{"label":"pass"}'],
                                {field: [generated]},
                                include_inputs=True,
                                include_cumulative_logprobs=True,
                                unpack_json=unpack_json,
                            )
                        if frame is pl.DataFrame:
                            self.assertTrue(original.equals(before))
                        else:
                            pd.testing.assert_frame_equal(original, before)

    def test_raw_output_destinations_cannot_replace_original_columns(self):
        for frame in (pl.DataFrame, pd.DataFrame):
            for output_column in ("inference_result", "prediction", "*", "^output$"):
                for output in ("plain text", "{}", '{"label":"pass"}'):
                    for unpack_json in (False, True):
                        if unpack_json and output.startswith('{"label"'):
                            continue
                        with self.subTest(
                            frame=frame,
                            column=output_column,
                            output=output,
                            unpack_json=unpack_json,
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "original DataFrame"
                            ):
                                self.get_results(
                                    frame({output_column: ["customer value"]}),
                                    [output],
                                    output_column=output_column,
                                    unpack_json=unpack_json,
                                )

    def test_equal_generated_columns_are_still_conflicts(self):
        for field in (
            "inference_result",
            "inputs",
            "confidence_score",
            "cumulative_logprobs",
        ):
            with self.subTest(field=field):
                metadata = {} if field == "inference_result" else {field: ["same"]}
                with self.assertRaisesRegex(ValueError, field):
                    self.get_results(
                        pl.DataFrame({field: ["same"]}),
                        ["same"],
                        metadata,
                        include_inputs=True,
                        include_cumulative_logprobs=True,
                    )

    def test_identical_carried_ids_preserve_original_columns(self):
        originals = (
            pl.DataFrame({"customer_id": [101, None, 103], "source": ["a", "b", "c"]}),
            pd.DataFrame(
                {
                    "customer_id": pd.array([101, None, 103], dtype="Int64"),
                    "source": pd.Categorical(["a", "b", "c"]),
                },
                index=pd.Index([4, 8, 12], name="source_row"),
            ),
        )
        for original in originals:
            with self.subTest(frame=type(original)):
                result = self.get_results(
                    original,
                    ['{"label":"pass"}'] * 3,
                    {"customer_id": [101, None, 103], "new_metadata": ["x"] * 3},
                )
                self.assertEqual(result["label"].to_list(), ["pass"] * 3)
                if isinstance(original, pd.DataFrame):
                    pd.testing.assert_frame_equal(result[original.columns], original)
                else:
                    self.assertTrue(result[original.columns].equals(original))

    def test_changed_or_coerced_metadata_is_rejected(self):
        for original_values, returned_values in (
            ([101, 102], [102, 101]),
            ([101, 102], [101, 999]),
            ([101, 102], [101.0, 102.0]),
            ([101, 102], ["101", "102"]),
            ([101, None], [101, 102]),
        ):
            for frame in (pl.DataFrame, pd.DataFrame):
                with self.subTest(
                    frame=frame, original=original_values, returned=returned_values
                ):
                    with self.assertRaisesRegex(ValueError, "customer_id"):
                        self.get_results(
                            frame({"customer_id": original_values}),
                            ["answer"] * 2,
                            {"customer_id": returned_values},
                        )

    def test_disabled_optional_columns_do_not_conflict(self):
        original = pl.DataFrame({"inputs": ["customer"], "cumulative_logprobs": [99]})
        result = self.get_results(
            original,
            ["answer"],
        )
        self.assertTrue(result[original.columns].equals(original))
        self.assertEqual(result["inference_result"].to_list(), ["answer"])

    def test_unrequested_optional_names_are_validated_as_carried_metadata(self):
        for field in ("inputs", "cumulative_logprobs"):
            with self.subTest(field=field):
                original = pl.DataFrame({field: ["customer-id"]})
                with self.assertRaisesRegex(ValueError, field):
                    self.get_results(original, ["answer"], {field: ["different-id"]})
                result = self.get_results(
                    original, ["answer"], {field: ["customer-id"]}
                )
                self.assertTrue(result[[field]].equals(original))
                self.assertEqual(result["inference_result"].to_list(), ["answer"])

    def test_expansion_preserves_original_raw_output_name(self):
        for frame in (pl.DataFrame, pd.DataFrame):
            with self.subTest(frame=frame):
                result = self.get_results(
                    frame({"inference_result": ["customer value"]}),
                    ['{"label":"pass"}'],
                )
                self.assertEqual(
                    result["inference_result"].to_list(), ["customer value"]
                )
                self.assertEqual(result["label"].to_list(), ["pass"])

    def test_renaming_raw_output_does_not_bypass_diagnostic_conflict(self):
        with self.assertRaisesRegex(ValueError, "confidence_score"):
            self.get_results(
                pl.DataFrame({"confidence_score": [99.0]}),
                ['{"label":"pass"}'],
                {"confidence_score": [0.6]},
                output_column="prediction",
                unpack_json=False,
            )

    def test_raw_output_name_cannot_replace_response_metadata(self):
        for output_column in (
            "customer_id",
            "inputs",
            "confidence_score",
            "cumulative_logprobs",
        ):
            with self.subTest(column=output_column):
                with self.assertRaisesRegex(ValueError, "output_column"):
                    self.get_results(
                        None,
                        ["answer"],
                        {output_column: ["metadata"]},
                        output_column=output_column,
                    )

    def test_custom_output_name_matching_optional_column_is_not_selected_twice(self):
        for output_column in ("inputs", "confidence_score", "cumulative_logprobs"):
            with self.subTest(column=output_column):
                result = self.get_results(
                    None,
                    ["answer"],
                    output_column=output_column,
                    include_inputs=True,
                    include_cumulative_logprobs=True,
                )
                self.assertEqual(result.to_dicts(), [{output_column: "answer"}])

    def test_cached_results_are_checked_before_merging(self):
        response = MagicMock()
        response.json.return_value = {
            "results": {"outputs": ["answer"], "confidence_score": [0.6]}
        }
        with patch.object(self.client, "do_request", return_value=response):
            self.client.get_job_results("offline-job")
        with patch.object(
            self.client, "do_request", side_effect=AssertionError("offline")
        ):
            with self.assertRaisesRegex(ValueError, "confidence_score"):
                self.client.get_job_results(
                    "offline-job",
                    with_original_df=pl.DataFrame({"confidence_score": [99.0]}),
                )

    def test_mismatched_row_counts_cannot_broadcast_results(self):
        for frame in (pl.DataFrame, pd.DataFrame):
            with self.subTest(frame=frame):
                with self.assertRaisesRegex(ValueError, "row count"):
                    self.get_results(frame({"source": ["a", "b"]}), ["one result"])
