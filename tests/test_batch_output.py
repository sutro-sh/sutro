import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from sutro._batch_output import ResultShapeError, normalize_results


class TestBatchOutput(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "raw.parquet"
        self.destination = Path(self.temporary.name) / "normalized.parquet"
        self.batch_size = patch("sutro._batch_output._BATCH_SIZE", 2)
        self.batch_size.start()
        self.addCleanup(self.batch_size.stop)

    def write(self, outputs, **columns):
        pq.write_table(
            pa.table({"outputs": pa.array(outputs, type=pa.string()), **columns}),
            self.source,
            row_group_size=3,
        )

    def normalize(self, **kwargs):
        normalize_results(self.source, self.destination, **kwargs)
        return pq.read_table(self.destination)

    def test_customer_fields_and_multiline_text_are_preserved(self):
        customer = {
            "content": {"label": "customer"},
            "reasoning_content": "customer reasoning",
            "reason": "line one\nline two",
            "rationale": r"literal\ntext",
            "analysis": "customer analysis",
            "error": "customer error",
        }
        self.write([json.dumps(customer)], inputs=["real\nnewline and literal \\n"])
        result = self.normalize(include_inputs=True)
        self.assertEqual(result.to_pylist(), [{"inputs": "real\nnewline and literal \\n", **customer}])

    def test_multiple_batches_preserve_ids_confidence_order_and_late_fields(self):
        outputs = [
            None,
            '{"label":"positive","nested":{"x":1}}',
            '{"label":"negative","nested":null}',
            '{"label":"neutral","nested":{"y":[1,2]}}',
            '{"label":"positive","late":true}',
        ]
        ids = [9007199254740993, 8, 7, 6, 5]
        confidence = [None, 0.8, 0.7, 0.6, 0.9]
        self.write(
            outputs,
            customer_id=ids,
            confidence_score=confidence,
            inputs=["private"] * 5,
            cumulative_logprobs=[-1.0] * 5,
        )

        result = self.normalize()

        self.assertEqual(result.num_rows, 5)
        self.assertEqual(
            result.column_names,
            ["customer_id", "confidence_score", "label", "late", "nested"],
        )
        self.assertEqual(result["customer_id"].to_pylist(), ids)
        self.assertEqual(result["confidence_score"].to_pylist(), confidence)
        self.assertEqual(result["label"].to_pylist()[0], None)
        self.assertEqual(result["late"].to_pylist(), [None, None, None, None, True])
        self.assertEqual(
            result["nested"].to_pylist(),
            [None, {"x": 1, "y": None}, None, {"x": None, "y": [1, 2]}, None],
        )

    def test_include_inputs_and_raw_output_preserve_all_requested_values(self):
        outputs = ['{"label":"x"}', None, "plain", "[1,2]"]
        inputs = ["one", "two", "three", "four"]
        self.write(outputs, inputs=inputs, customer_id=[4, 3, 2, 1])

        result = self.normalize(include_inputs=True, unpack_json=False)

        self.assertEqual(
            result.column_names, ["inputs", "customer_id", "inference_result"]
        )
        self.assertEqual(result["inputs"].to_pylist(), inputs)
        self.assertEqual(result["inference_result"].to_pylist(), outputs)

    def test_fallback_retains_every_original_output_across_batches(self):
        invalid_or_mixed = [
            "plain text",
            "{broken",
            "[1,2]",
            "42",
            '"text"',
            '{"label":42}',
            '{"label":{"nested":1}}',
            '{"label":"x","label":"y"}',
            '{"other":NaN}',
            '{"other":1e999}',
            '{"other":{"nested":[-1e999]}}',
            '{"empty":{}}',
            '{"large":18446744073709551616}',
        ]
        for final in invalid_or_mixed:
            with self.subTest(final=final):
                outputs = ['{"label":"x"}', None, '{"label":"y"}', final]
                self.write(outputs, customer_id=[1, 2, 3, 4])
                result = self.normalize()
                self.assertEqual(result["inference_result"].to_pylist(), outputs)
                self.assertEqual(result["customer_id"].to_pylist(), [1, 2, 3, 4])
                self.destination.unlink()

    def test_late_collision_is_rejected_without_writing_or_losing_raw_source(self):
        for field in ["customer_id", "confidence_score", "inference_result"]:
            with self.subTest(field=field):
                outputs = ['{"label":"x"}'] * 3 + [json.dumps({field: "generated"})]
                self.write(
                    outputs, customer_id=[1, 2, 3, 4], confidence_score=[0.9] * 4
                )
                with self.assertRaisesRegex(
                    ResultShapeError, field + ".*unpack_json=False"
                ):
                    self.normalize()
                self.assertFalse(self.destination.exists())
                self.assertEqual(
                    pq.read_table(self.source)["outputs"].to_pylist(), outputs
                )
                self.assertEqual(
                    self.normalize(unpack_json=False)["inference_result"].to_pylist(),
                    outputs,
                )
                self.destination.unlink()

    def test_empty_all_null_and_empty_objects_keep_the_raw_column(self):
        for outputs in [[], [None, None, None], ["{}", "null", None, "{}"]]:
            with self.subTest(outputs=outputs):
                self.write(outputs)
                result = self.normalize()
                self.assertEqual(result["inference_result"].to_pylist(), outputs)
                self.destination.unlink()

    def test_json_nulls_and_empty_objects_support_fields_that_appear_late(self):
        self.write(["{}", "null", None, '{"label":"x"}'])
        self.assertEqual(self.normalize()["label"].to_pylist(), [None, None, None, "x"])

    def test_schema_and_fallback_are_independent_of_batch_size(self):
        for outputs in [
            ['{"a":1}', '{"z":"yes"}', '{"b":true}'],
            ['{"number":1}', '{"number":1.5}', '{"number":2}'],
        ]:
            self.write(outputs)
            expected = None
            for size in [1, 2, 100]:
                with patch("sutro._batch_output._BATCH_SIZE", size):
                    result = self.normalize()
                if expected is not None:
                    self.assertEqual(result.schema, expected.schema)
                    self.assertEqual(result.to_pydict(), expected.to_pydict())
                expected = result
                self.destination.unlink()

    def test_plural_confidence_metadata_is_preserved(self):
        self.write(["plain", "text"], confidence_scores=[0.8, None])
        self.assertEqual(self.normalize()["confidence_scores"].to_pylist(), [0.8, None])

    def test_download_artifact_uses_explicit_job_id_output_column(self):
        job_id = "job-00000000-0000-0000-0000-000000000001"
        # Matches create_lazy_result_dataframes_for_job +
        # combine_job_result_dataframes with include_inputs=True.
        pq.write_table(
            pa.table(
                {
                    "inputs": ["prompt one", "prompt two", "prompt three"],
                    "customer_id": [101, 102, 103],
                    job_id: ['{"label":"a"}', None, '{"label":"b","late":true}'],
                    "confidence_score": [0.8, None, 0.9],
                }
            ),
            self.source,
        )
        result = self.normalize(source_output_column=job_id, include_inputs=True)
        self.assertEqual(
            result.column_names,
            ["inputs", "customer_id", "confidence_score", "label", "late"],
        )
        self.assertEqual(result["customer_id"].to_pylist(), [101, 102, 103])
        self.assertEqual(result["confidence_score"].to_pylist(), [0.8, None, 0.9])
        self.assertEqual(result["label"].to_pylist(), ["a", None, "b"])
        self.assertEqual(result["late"].to_pylist(), [None, None, True])
        self.destination.unlink()
        raw = self.normalize(source_output_column=job_id, unpack_json=False)
        self.assertEqual(
            raw["inference_result"].to_pylist(),
            ['{"label":"a"}', None, '{"label":"b","late":true}'],
        )

    def test_source_output_column_is_never_guessed(self):
        job_id = "job-00000000-0000-0000-0000-000000000001"
        self.write(["metadata"], **{job_id: ["generated"]})
        result = self.normalize(source_output_column=job_id)
        self.assertEqual(result["outputs"].to_pylist(), ["metadata"])
        self.assertEqual(result["inference_result"].to_pylist(), ["generated"])
        self.destination.unlink()
        pq.write_table(pa.table({job_id: ["generated"]}), self.source)
        with self.assertRaisesRegex(ResultShapeError, "missing the expected output"):
            self.normalize()
        self.assertFalse(self.destination.exists())
        with self.assertRaisesRegex(ResultShapeError, "missing the expected output"):
            self.normalize(source_output_column="job-wrong")
        self.assertFalse(self.destination.exists())

    def test_native_list_results_preserve_values_and_metadata(self):
        job_id = "job-00000000-0000-0000-0000-000000000001"
        outputs = [["alpha, beta", 'quoted "text"', None], [], None, ["alpha", "beta"]]
        metadata = [{"tags": ["x, y", None]}, None, {"tags": []}, {"tags": ["z"]}]
        pq.write_table(
            pa.table(
                {
                    "customer_id": [9007199254740993, 2, 3, 4],
                    "metadata": metadata,
                    job_id: outputs,
                    "confidence_score": [0.9, None, 0.2, 0.8],
                }
            ),
            self.source,
        )
        for unpack_json in (True, False):
            result = self.normalize(
                source_output_column=job_id, unpack_json=unpack_json
            )
            self.assertEqual(result["inference_result"].to_pylist(), outputs)
            self.assertEqual(result["metadata"].to_pylist(), metadata)
            self.assertEqual(
                result["customer_id"].to_pylist(), [9007199254740993, 2, 3, 4]
            )
            self.assertEqual(
                result["confidence_score"].to_pylist(), [0.9, None, 0.2, 0.8]
            )
            self.destination.unlink()

    def test_does_not_overwrite_destination_or_source(self):
        self.write(["original"])
        self.destination.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.normalize()
        self.assertEqual(self.destination.read_bytes(), b"existing")
        with self.assertRaises(FileExistsError):
            normalize_results(self.source, self.source)
        self.assertEqual(
            pq.read_table(self.source)["outputs"].to_pylist(), ["original"]
        )

    def test_raw_column_rename_collision_and_missing_outputs_are_rejected(self):
        for data in [
            {"outputs": ["original"], "inference_result": ["metadata"]},
            {"missing_outputs": ["original"]},
        ]:
            pq.write_table(pa.table(data), self.source)
            with self.assertRaises(ResultShapeError):
                self.normalize()
            self.assertFalse(self.destination.exists())

    def test_reader_and_writer_use_bounded_batches_not_full_table_reads(self):
        self.write(['{"label":"x"}'] * 9)
        original = pq.ParquetFile.iter_batches
        observed = []

        def batches(parquet, **kwargs):
            self.assertEqual(kwargs["batch_size"], 2)
            for batch in original(parquet, **kwargs):
                observed.append(batch.num_rows)
                yield batch

        with (
            patch.object(pq.ParquetFile, "iter_batches", batches),
            patch.object(
                pq.ParquetFile, "read", side_effect=AssertionError("full read")
            ),
            patch.object(pq, "read_table", side_effect=AssertionError("full read")),
        ):
            normalize_results(self.source, self.destination)
        self.assertEqual(observed, [2, 2, 2, 2, 1] * 2)
        self.assertEqual(pq.read_table(self.destination).num_rows, 9)


if __name__ == "__main__":
    unittest.main()
