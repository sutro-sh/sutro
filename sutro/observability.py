from typing import Optional, Dict, Any, List, Callable

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import requests
from langsmith import traceable, get_current_run_tree, Client

logger = logging.getLogger(__name__)

# Fixed namespace for deterministic child run IDs.
_SUTRO_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def _row_run_id(job_id: str, row_index: int) -> uuid.UUID:
    """Deterministic UUID for a batch row trace."""
    return uuid.uuid5(_SUTRO_NAMESPACE, f"{job_id}-{row_index}")


def _ts(dt: datetime) -> str:
    """Format a datetime as a LangSmith dotted_order timestamp."""
    return dt.strftime("%Y%m%dT%H%M%S%fZ")


def _try_parse_json(value: Any, fallback_key: str) -> dict:
    """Try to parse a value as JSON. If it's already a dict, return it.
    If it's a JSON string, parse it. Otherwise wrap it in a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {fallback_key: value}


def _is_langsmith_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled via environment variable."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def _create_batch_traces(
    function_name: str,
    job_id: str,
    input_data: List[dict],
    langsmith_metadata: Optional[Dict[str, Any]] = None,
    langsmith_tags: Optional[List[str]] = None,
) -> bool:
    """
    Create one top-level LangSmith trace per row in a batch function job.

    Each row gets its own independent trace (like run_function would produce),
    linked by shared sutro_job_id metadata for filtering. Uses deterministic
    UUIDs so traces can be updated with outputs at retrieval time.

    Uses batch_ingest_runs for efficient bulk creation.

    Returns True if traces were created successfully.
    """
    if not _is_langsmith_tracing_enabled():
        return False

    try:
        client = Client()

        metadata = {
            "ls_provider": "sutro",
            "ls_model_name": function_name,
            "ls_method": "traceable",
            "sutro_job_id": job_id,
        }
        if langsmith_metadata:
            metadata.update(langsmith_metadata)

        now = datetime.now(timezone.utc)
        project_name = os.environ.get("LANGSMITH_PROJECT", "default")

        logger.debug("[langsmith] Building %d trace dicts...", len(input_data))

        runs = []
        for i, inp in enumerate(input_data):
            run_id = _row_run_id(job_id, i)
            run_ts = _ts(now)
            run = {
                "id": str(run_id),
                "session_name": project_name,
                "name": "Sutro Function",
                "run_type": "llm",
                "inputs": {"input_data": inp},
                "extra": {"metadata": metadata},
                "start_time": now.isoformat(),
                "trace_id": str(run_id),
                "dotted_order": f"{run_ts}{str(run_id)}",
            }
            if langsmith_tags:
                run["tags"] = langsmith_tags
            runs.append(run)

        logger.debug("[langsmith] Calling batch_ingest_runs for %d traces...", len(runs))
        client.batch_ingest_runs(create=runs)
        logger.debug("[langsmith] batch_ingest_runs returned")

        return True
    except Exception as e:
        print(f"Warning: Failed to create LangSmith traces for batch job {job_id}: {e}")
        return False


def _has_open_batch_traces(job_id: str) -> bool:
    """
    Check if open (incomplete) LangSmith traces exist for this batch job_id.

    Returns True if at least one open trace is found, False otherwise.
    """
    if not _is_langsmith_tracing_enabled():
        return False

    try:
        client = Client()
        project_name = os.environ.get("LANGSMITH_PROJECT", "default")

        # Check just the first row's trace to see if it exists and is open
        first_run_id = _row_run_id(job_id, 0)
        runs = list(
            client.list_runs(
                project_name=project_name,
                run_ids=[str(first_run_id)],
                limit=1,
            )
        )

        if not runs:
            return False

        # If it already has an end_time, traces were already completed
        return runs[0].end_time is None
    except Exception as e:
        print(f"Warning: Failed to check LangSmith traces for batch job {job_id}: {e}")
        return False


def _complete_batch_traces(
    job_id: str,
    num_rows: int,
    outputs: List[Any],
    job_details: Optional[dict] = None,
):
    """
    Update batch row traces with outputs and mark them as complete.

    Uses deterministic UUIDs matching those created in _create_batch_traces.
    Uses batch_ingest_runs for efficient bulk updates.

    Args:
        job_id: The Sutro batch job ID.
        num_rows: Total number of rows in the batch.
        outputs: Per-row outputs from the batch results.
        job_details: Optional job dict from GET /jobs/{job_id} with token counts.
    """
    try:
        client = Client()

        now = datetime.now(timezone.utc)

        # Compute per-row token estimates
        per_row_usage = None
        if job_details and num_rows > 0:
            total_input = job_details.get("input_tokens")
            total_output = job_details.get("output_tokens")
            if total_input is not None or total_output is not None:
                per_row_usage = {}
                if total_input is not None:
                    per_row_usage["input_tokens"] = total_input // num_rows
                if total_output is not None:
                    per_row_usage["output_tokens"] = total_output // num_rows
                if total_input is not None and total_output is not None:
                    per_row_usage["total_tokens"] = (
                        per_row_usage["input_tokens"] + per_row_usage["output_tokens"]
                    )

        logger.debug("[langsmith] Building %d update dicts...", len(outputs))

        updates = []
        for i, out in enumerate(outputs):
            run_id = _row_run_id(job_id, i)
            parsed_out = _try_parse_json(out, "output")
            update = {
                "id": str(run_id),
                "trace_id": str(run_id),
                "dotted_order": f"{_ts(now)}{str(run_id)}",
                "outputs": parsed_out,
                "end_time": now.isoformat(),
            }
            if per_row_usage:
                update["extra"] = {
                    "metadata": {
                        "ls_method": "traceable",
                        "usage_metadata": per_row_usage,
                    }
                }
            updates.append(update)

        logger.debug("[langsmith] Calling batch_ingest_runs for %d updates...", len(updates))
        client.batch_ingest_runs(update=updates)
        logger.debug("[langsmith] batch_ingest_runs (update) returned")
    except Exception as e:
        print(f"Warning: Failed to complete LangSmith traces for batch job {job_id}: {e}")


def _traced_run(
    function_name: str,
    api_call: Callable[[dict], dict],
    input_data: dict,
    langsmith_metadata: Optional[Dict[str, Any]] = None,
    langsmith_tags: Optional[List[str]] = None,
) -> dict:
    """
    Execute an API call wrapped in a LangSmith trace.

    The trace captures real wall-clock latency, token usage, and any errors
    from the actual call. Use this when you want accurate latency metrics.

    Args:
        function_name: The Function name (e.g. "clay-bert"). Used as ls_model_name.
        api_call: A callable that performs the API request and returns
            the result dict. It should take in the input_data dict that the model
             expects to receive. Exceptions will be captured in the trace and re-raised.
        langsmith_metadata: Extra metadata merged into the trace.
        langsmith_tags: Tags attached to the trace for filtering.

    Returns:
        dict: The result from api_call().
    """
    metadata = {
        "ls_provider": "sutro",
        "ls_model_name": function_name,
    }
    if langsmith_metadata:
        metadata.update(langsmith_metadata)

    traceable_kwargs = {
        "run_type": "llm",
        "name": "Sutro Function",
        "metadata": metadata,
    }
    if langsmith_tags:
        traceable_kwargs["tags"] = langsmith_tags

    @traceable(**traceable_kwargs)
    def _traced_call(input_data: dict) -> dict:
        try:
            result = api_call(input_data)
        except requests.HTTPError as e:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                try:
                    error_json = e.response.json()
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_json = {}
                run_tree.add_outputs(
                    {
                        "error": {
                            "status_code": e.response.status_code,
                            "detail": error_json.get("detail"),
                        }
                    }
                )
            raise  # Re-raise so @traceable captures the exception

        _attach_run_metadata(result)
        return result

    return _traced_call(input_data)


def _attach_run_metadata(result: dict):
    """Attach token usage and run ID to the current LangSmith run tree."""
    run_tree = get_current_run_tree()
    if run_tree is None:
        return

    usage = result.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if input_tokens is not None or output_tokens is not None:
        token_usage = {}
        if input_tokens is not None:
            token_usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            token_usage["output_tokens"] = output_tokens
        if input_tokens is not None and output_tokens is not None:
            token_usage["total_tokens"] = input_tokens + output_tokens
        run_tree.set(usage_metadata=token_usage)

    run_id = result.get("run_id")
    if run_id:
        run_tree.add_metadata({"sutro_run_id": run_id})
