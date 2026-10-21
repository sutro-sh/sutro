"""Internal bounded-memory normalization for CLI result files."""

import json
import math
from os import PathLike
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_BATCH_SIZE = 4096


class ResultShapeError(ValueError):
    """Safe result-schema validation failure, containing no result values."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _decode(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Output is not a JSON string")
    result = json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    if result is not None and not isinstance(result, dict):
        raise ValueError("Output is not a JSON object")
    return result


def _has_empty_struct(data_type):
    if pa.types.is_struct(data_type):
        return not data_type.num_fields or any(
            _has_empty_struct(field.type) for field in data_type
        )
    if pa.types.is_list(data_type):
        return _has_empty_struct(data_type.value_type)
    return False


def _structured_schema(parquet, source_output_column):
    """Scan only outputs; retain schema, never the full result set in memory."""
    schema = pa.schema([])
    try:
        for batch in parquet.iter_batches(
            batch_size=_BATCH_SIZE, columns=[source_output_column]
        ):
            for raw in batch.column(0).to_pylist():
                value = _decode(raw)
                if value is None:
                    continue
                inferred = pa.scalar(value)
                if inferred.as_py() != value:
                    return None
                # Infer each row independently so mixed types have the same
                # fallback policy regardless of batch/row-group boundaries.
                schema = pa.unify_schemas([schema, pa.schema(inferred.type)])
    except (ValueError, TypeError, OverflowError, RecursionError, pa.ArrowException):
        return None
    if not schema or any(_has_empty_struct(field.type) for field in schema):
        return None
    return pa.schema(sorted(schema, key=lambda field: field.name))


def normalize_results(
    source_path: str | PathLike,
    destination_path: str | PathLike,
    *,
    unpack_json: bool = True,
    include_inputs: bool = False,
    include_cumulative_logprobs: bool = False,
    source_output_column: str = "outputs",
) -> None:
    """Write SDK-shaped Parquet without overwriting an existing destination.

    Preserve row order, IDs, metadata and confidence. Rename the exact
    ``source_output_column`` to ``inference_result``. Pass the public job ID
    for downloaded result artifacts; the default matches JSON-API column names.
    Never guess the output column or join the original dataset. Omit inputs by
    default and omit cumulative logprobs, matching ``get_job_results`` defaults.

    Structured output is unpacked at the top level only, using a full schema
    scan followed by batched writing. Missing fields become null, nested values
    remain nested. Any malformed/non-object JSON, incompatible field types or
    unrepresentable values retain the raw output column for the entire file.
    Collisions raise before writing; retry with ``unpack_json=False``.

    Memory is bounded by a row batch plus the output schema. The caller owns
    the destination (normally a temporary file) and cleanup after write errors.
    """
    destination = Path(destination_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Result destination already exists")
    with pq.ParquetFile(source_path) as parquet:
        source_schema = parquet.schema_arrow
        names = source_schema.names
        if len(set(names)) != len(names):
            raise ResultShapeError("Result artifact contains duplicate column names")
        if source_output_column not in names:
            raise ResultShapeError(
                "Result artifact is missing the expected output column"
            )
        if "inference_result" in names:
            raise ResultShapeError("Result artifact already contains inference_result")

        standard = {
            "inputs",
            source_output_column,
            "cumulative_logprobs",
            "confidence_score",
        }
        columns = ["inputs"] if include_inputs and "inputs" in names else []
        columns += [name for name in names if name not in standard]
        columns += [source_output_column]
        if include_cumulative_logprobs and "cumulative_logprobs" in names:
            columns.append("cumulative_logprobs")
        if "confidence_score" in names:
            columns.append("confidence_score")
        fields = [
            source_schema.field(name).with_name(
                "inference_result" if name == source_output_column else name
            )
            for name in columns
        ]
        structured = (
            _structured_schema(parquet, source_output_column) if unpack_json else None
        )
        if structured is not None:
            conflicts = sorted(set(structured.names) & {field.name for field in fields})
            if conflicts:
                raise ResultShapeError(
                    "Cannot unpack structured output fields that conflict with "
                    "existing result columns: "
                    + ", ".join(conflicts)
                    + ". Set unpack_json=False to preserve metadata and raw output."
                )
            fields = [field for field in fields if field.name != "inference_result"]
            fields.extend(structured)
        output_schema = pa.schema(fields)

        # Exclusive creation closes the race after the preflight check above.
        with destination.open("xb") as sink:
            with pq.ParquetWriter(sink, output_schema, compression="snappy") as writer:
                for batch in parquet.iter_batches(
                    batch_size=_BATCH_SIZE, columns=columns
                ):
                    arrays = dict(zip(columns, batch.columns))
                    outputs = arrays.pop(source_output_column)
                    if structured is None:
                        arrays["inference_result"] = outputs
                    else:
                        decoded = pa.array(
                            [_decode(value) for value in outputs.to_pylist()],
                            type=pa.struct(structured),
                        )
                        # flatten() propagates the parent null bitmap; field()
                        # alone would turn null outputs into empty/zero values.
                        arrays.update(zip(structured.names, decoded.flatten()))
                    writer.write_batch(
                        pa.RecordBatch.from_arrays(
                            [arrays[field.name] for field in output_schema],
                            schema=output_schema,
                        )
                    )
