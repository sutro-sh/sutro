import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from sutro.sdk import Sutro


class TestJobResultsJson(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.client = object.__new__(Sutro)
        self.client.api_url = "https://harmonize.example.test"
        self.client.api_key = "test-api-key"
        self.client.serving_base_url = None
        cache_dir = contexts.enter_context(tempfile.TemporaryDirectory())
        contexts.enter_context(
            patch.object(
                self.client,
                "_job_results_cache_file_path",
                return_value=str(Path(cache_dir) / "job.parquet"),
            )
        )
        contexts.enter_context(
            patch.object(self.client, "_remove_legacy_job_results_cache_file")
        )
        contexts.enter_context(
            patch("sutro.sdk._has_open_batch_traces", return_value=False)
        )

    def get_results(self, outputs, metadata=None, **kwargs):
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": {"outputs": outputs, **(metadata or {})}
        }
        with patch.object(self.client, "do_request", return_value=response):
            return self.client.get_job_results("job", disable_cache=True, **kwargs)

    def test_reasoning_output_cannot_replace_source_id_or_metadata(self):
        for field in ("customer_id", "source_record"):
            with self.subTest(field=field):
                outputs = [
                    json.dumps(
                        {
                            "content": {field: "model-generated-456", "label": "pass"},
                            "reasoning_content": "model rationale",
                        }
                    )
                ]
                metadata = {field: ["source-record-123"]}
                with self.assertRaisesRegex(
                    ValueError, rf"{field}.*Set unpack_json=False"
                ):
                    self.get_results(outputs, metadata)

                raw = self.get_results(outputs, metadata, unpack_json=False)
                self.assertEqual(raw[field].to_list(), ["source-record-123"])
                self.assertEqual(raw["inference_result"].to_list(), outputs)

    def test_json_property_names_are_literal_when_expanding(self):
        content = {"*": 1, "^label$": 2, "^.*$": 3, "label": "ok"}
        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning):
                output = (
                    {"content": content, "reasoning_content": "rationale"}
                    if reasoning
                    else content
                )
                outputs = [json.dumps(output)]
                results = self.get_results(outputs, {"customer_id": ["source-id"]})
                expected = {"customer_id": "source-id", **content}
                if reasoning:
                    expected["reasoning_content"] = "rationale"
                self.assertEqual(results.to_dicts(), [expected])
                raw = self.get_results(
                    outputs, {"customer_id": ["source-id"]}, unpack_json=False
                )
                self.assertEqual(raw["inference_result"].to_list(), outputs)

    def test_literal_property_names_still_check_metadata_collisions(self):
        for reasoning in (False, True):
            for name in ("*", "^label$"):
                with self.subTest(reasoning=reasoning, name=name):
                    content = {name: "generated", "label": "ok"}
                    output = (
                        {"content": content, "reasoning_content": "rationale"}
                        if reasoning
                        else content
                    )
                    with self.assertRaisesRegex(ValueError, "Set unpack_json=False"):
                        self.get_results([json.dumps(output)], {name: ["original"]})
                    raw = self.get_results(
                        [json.dumps(output)], {name: ["original"]}, unpack_json=False
                    )
                    self.assertEqual(raw[name].to_list(), ["original"])

    def test_custom_output_names_are_literal_when_expanding(self):
        for output_column in ("*", "^output$"):
            for reasoning in (False, True):
                with self.subTest(output_column=output_column, reasoning=reasoning):
                    content = {"label": "ok"}
                    output = (
                        {"content": content, "reasoning_content": "rationale"}
                        if reasoning
                        else content
                    )
                    results = self.get_results(
                        [json.dumps(output)],
                        {"customer_id": ["source-id"]},
                        output_column=output_column,
                    )
                    expected = {"customer_id": "source-id", "label": "ok"}
                    if reasoning:
                        expected["reasoning_content"] = "rationale"
                    self.assertEqual(results.to_dicts(), [expected])

    def test_reasoning_destinations_cannot_replace_raw_output_or_reasoning(self):
        for field, output_column in (
            ("reasoning_content", "inference_result"),
            ("inference_result", "inference_result"),
            ("custom_output", "custom_output"),
            ("content", "content"),
            ("label", "reasoning_content"),
        ):
            with self.subTest(field=field, output_column=output_column):
                outputs = [
                    json.dumps(
                        {
                            "content": {field: "generated value"},
                            "reasoning_content": "model rationale",
                        }
                    )
                ]
                with self.assertRaisesRegex(ValueError, "Set unpack_json=False"):
                    self.get_results(outputs, output_column=output_column)
                raw = self.get_results(
                    outputs, output_column=output_column, unpack_json=False
                )
                self.assertEqual(raw[output_column].to_list(), outputs)

    def test_reasoning_expansion_cannot_replace_inputs_or_diagnostics(self):
        for field in (
            "inputs",
            "confidence_score",
            "cumulative_logprobs",
            "reasoning_content",
        ):
            with self.subTest(field=field):
                content = (
                    {"label": "pass"} if field == "reasoning_content" else {field: 7}
                )
                outputs = [
                    json.dumps({"content": content, "reasoning_content": "rationale"})
                ]
                options = {
                    "include_inputs": True,
                    "include_cumulative_logprobs": True,
                }
                metadata = {field: ["original value"]}
                with self.assertRaisesRegex(
                    ValueError, rf"{field}.*Set unpack_json=False"
                ):
                    self.get_results(outputs, metadata, **options)
                raw = self.get_results(outputs, metadata, unpack_json=False, **options)
                self.assertEqual(raw[field].to_list(), ["original value"])
                self.assertEqual(raw["inference_result"].to_list(), outputs)

    def test_reasoning_nested_content_field_is_preserved(self):
        outputs = [
            json.dumps(
                {
                    "content": {"content": "answer", "label": "pass"},
                    "reasoning_content": "model rationale",
                }
            )
        ]
        results = self.get_results(outputs, {"customer_id": ["source-record-123"]})
        self.assertEqual(
            results.to_dicts(),
            [
                {
                    "customer_id": "source-record-123",
                    "reasoning_content": "model rationale",
                    "content": "answer",
                    "label": "pass",
                }
            ],
        )

    def test_generated_content_cannot_replace_content_metadata(self):
        outputs = [
            '{"content": {"content": "generated"}, "reasoning_content": "rationale"}'
        ]
        with self.assertRaisesRegex(ValueError, "content.*Set unpack_json=False"):
            self.get_results(outputs, {"content": ["original metadata"]})

    def test_reasoning_can_preserve_metadata_named_content(self):
        outputs = ['{"content": {"label": "pass"}, "reasoning_content": "rationale"}']
        results = self.get_results(outputs, {"content": ["original metadata"]})
        self.assertEqual(results["content"].to_list(), ["original metadata"])
        self.assertEqual(results["label"].to_list(), ["pass"])

    def test_collision_in_later_rows_is_rejected(self):
        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning):
                objects = [{"label": "pass"}] * 101 + [{"customer_id": "generated"}]
                if reasoning:
                    objects = [
                        {"content": obj, "reasoning_content": "rationale"}
                        for obj in objects
                    ]
                outputs = [json.dumps(obj) for obj in objects]
                metadata = {"customer_id": ["source-record-123"] * len(outputs)}
                with self.assertRaisesRegex(
                    ValueError, "customer_id.*Set unpack_json=False"
                ):
                    self.get_results(outputs, metadata)

    def test_expansion_cannot_replace_original_dataframe_metadata(self):
        for original in (
            pl.DataFrame({"customer_id": ["source-record-123"]}),
            pd.DataFrame({"customer_id": ["source-record-123"]}),
        ):
            for output in (
                '{"customer_id": "generated"}',
                '{"content": {"customer_id": "generated"}, "reasoning_content": "r"}',
            ):
                with self.subTest(dataframe=type(original), output=output):
                    with self.assertRaisesRegex(
                        ValueError, "customer_id.*Set unpack_json=False"
                    ):
                        self.get_results([output], with_original_df=original)

    def test_normal_structured_outputs_preserve_ids_and_all_fields(self):
        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning):
                objects = [{"label": "pass"}, {"label": "fail", "score": 0.5}]
                if reasoning:
                    objects = [
                        {"content": obj, "reasoning_content": "rationale"}
                        for obj in objects
                    ]
                results = self.get_results(
                    [json.dumps(obj) for obj in objects],
                    {"customer_id": ["id-1", "id-2"]},
                )
                self.assertEqual(results["customer_id"].to_list(), ["id-1", "id-2"])
                self.assertEqual(results["label"].to_list(), ["pass", "fail"])
                self.assertEqual(results["score"].to_list(), [None, 0.5])
                self.assertNotIn("inference_result", results.columns)
                if reasoning:
                    self.assertEqual(
                        results["reasoning_content"].to_list(),
                        ["rationale", "rationale"],
                    )

    def test_plain_text_reasoning_content_unpacks_normally(self):
        outputs = ['{"content": "answer", "reasoning_content": "rationale"}']
        results = self.get_results(outputs)
        self.assertEqual(
            results.to_dicts(),
            [{"content": "answer", "reasoning_content": "rationale"}],
        )

    def test_empty_objects_and_null_outputs_preserve_raw_rows(self):
        for outputs in (
            ["{}", "{}"],
            ["{}", "null"],
            ["null", "null"],
            [None, None],
            [],
        ):
            for include_metadata in (False, True):
                with self.subTest(outputs=outputs, metadata=include_metadata):
                    metadata = (
                        {"customer_id": list(range(len(outputs)))}
                        if include_metadata
                        else None
                    )
                    results = self.get_results(outputs, metadata)
                    raw = self.get_results(outputs, metadata, unpack_json=False)
                    self.assertTrue(results.equals(raw))
                    self.assertEqual(results.height, len(outputs))
                    self.assertEqual(results["inference_result"].to_list(), outputs)
                    if include_metadata:
                        self.assertEqual(
                            results["customer_id"].to_list(), metadata["customer_id"]
                        )

    def test_empty_objects_preserve_raw_rows_with_original_dataframe(self):
        outputs = ["{}", "{}"]
        for original in (
            pl.DataFrame({"source_id": ["source-1", "source-2"]}),
            pd.DataFrame({"source_id": ["source-1", "source-2"]}),
        ):
            for metadata in (None, {"customer_id": ["id-1", "id-2"]}):
                with self.subTest(dataframe=type(original), metadata=metadata):
                    results = self.get_results(
                        outputs, metadata, with_original_df=original
                    )
                    self.assertEqual(len(results), len(outputs))
                    self.assertEqual(results["inference_result"].to_list(), outputs)
                    self.assertEqual(
                        results["source_id"].to_list(), ["source-1", "source-2"]
                    )

    def test_empty_reasoning_content_keeps_reasoning_rows(self):
        outputs = [
            '{"content": {}, "reasoning_content": "first rationale"}',
            '{"content": {}, "reasoning_content": "second rationale"}',
        ]
        results = self.get_results(outputs)
        self.assertEqual(
            results.to_dicts(),
            [
                {"reasoning_content": "first rationale"},
                {"reasoning_content": "second rationale"},
            ],
        )

    def test_empty_and_null_outputs_allow_fields_in_later_rows(self):
        outputs = [None, "{}", '{"label": "pass"}', "null"]
        results = self.get_results(outputs)
        self.assertEqual(results.height, len(outputs))
        self.assertEqual(results["label"].to_list(), [None, None, "pass", None])
        self.assertNotIn("inference_result", results.columns)

    def test_failed_json_decoding_keeps_raw_outputs_and_metadata(self):
        for outputs in (
            ["plain text"],
            ['{"label": "pass"}', "invalid JSON"],
            ["42"],
            ['["answer"]'],
        ):
            with self.subTest(outputs=outputs):
                metadata = {"customer_id": ["source-record-123"] * len(outputs)}
                results = self.get_results(outputs, metadata)
                self.assertEqual(results.columns, ["customer_id", "inference_result"])
                self.assertEqual(
                    results["customer_id"].to_list(), metadata["customer_id"]
                )
                self.assertEqual(results["inference_result"].to_list(), outputs)
