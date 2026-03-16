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
# uuid5(SUTRO_NAMESPACE, f"{job_id}-{row_index}") produces the same UUID
# at submission time and retrieval time, so no state needs to be stored.
_SUTRO_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def _child_run_id(job_id: str, row_index: int) -> uuid.UUID:
    return uuid.uuid5(_SUTRO_NAMESPACE, f"{job_id}-{row_index}")


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
    Create a parent LangSmith trace and child traces for a batch function job.

    Called at job submission time so that traces are immediately visible in
    LangSmith while the batch job is running. All traces are left open
    (no end_time) until results are retrieved via _complete_batch_traces.

    Child traces use deterministic UUIDs based on job_id + row index so they
    can be updated at retrieval time without storing any state.

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
        parent_id = uuid.uuid4()

        parent_kwargs = {
            "name": "Sutro Batch Function",
            "run_type": "chain",
            "id": parent_id,
            "inputs": {
                "function_name": function_name,
                "num_rows": len(input_data),
                "job_id": job_id,
            },
            "extra": {"metadata": metadata},
            "start_time": now,
        }
        if langsmith_tags:
            parent_kwargs["tags"] = langsmith_tags

        client.create_run(**parent_kwargs)

        for i, inp in enumerate(input_data):
            child_kwargs = {
                "name": "Sutro Function",
                "run_type": "llm",
                "id": _child_run_id(job_id, i),
                "parent_run_id": parent_id,
                "inputs": {"input_data": inp},
                "extra": {"metadata": metadata},
                "start_time": now,
            }
            if langsmith_tags:
                child_kwargs["tags"] = langsmith_tags
            client.create_run(**child_kwargs)

        return True
    except Exception:
        logger.warning(
            "Failed to create LangSmith traces for batch job %s",
            job_id,
            exc_info=True,
        )
        return False


def _find_batch_parent_trace(job_id: str) -> Optional[dict]:
    """
    Query LangSmith for a parent trace associated with this batch job_id.

    Returns a dict with the parent run's id, metadata, and tags if found,
    or None if tracing is disabled or no parent trace exists.
    """
    if not _is_langsmith_tracing_enabled():
        return None

    try:
        client = Client()

        project_name = os.environ.get("LANGSMITH_PROJECT", "default")
        runs = list(
            client.list_runs(
                project_name=project_name,
                filter=f'and(has(metadata, \'{{"sutro_job_id": "{job_id}"}}\'), eq(name, "Sutro Batch Function"))',
                limit=1,
            )
        )

        if not runs:
            return None

        parent_run = runs[0]
        # If the parent trace already has an end_time, child traces were
        # already completed on a previous get_job_results call — skip
        if parent_run.end_time is not None:
            return None

        return {
            "id": parent_run.id,
            "metadata": parent_run.extra.get("metadata", {})
            if parent_run.extra
            else {},
            "tags": parent_run.tags or [],
        }
    except Exception:
        logger.warning(
            "Failed to query LangSmith for batch parent trace for job %s",
            job_id,
            exc_info=True,
        )
        return None


def _complete_batch_traces(
    job_id: str,
    parent_trace: dict,
    num_rows: int,
    outputs: List[Any],
    job_details: Optional[dict] = None,
):
    """
    Update child traces with outputs and mark the parent trace as complete.

    Child traces are identified by deterministic UUIDs matching those created
    in _create_batch_traces. Only outputs and token usage are updated.

    Args:
        job_id: The Sutro batch job ID.
        parent_trace: Dict returned by _find_batch_parent_trace.
        num_rows: Total number of rows in the batch (used for deterministic IDs).
        outputs: Per-row outputs from the batch results.
        job_details: Optional job dict from GET /jobs/{job_id} with token counts.
    """
    from langsmith.utils import LangSmithConflictError

    try:
        client = Client()

        metadata = parent_trace["metadata"]
        now = datetime.now(timezone.utc)

        # Compute per-row token estimates by splitting aggregate evenly
        # since the only way to show total token count in the LangSmith dashboard
        # is via aggregating the child runs of the parent trace
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

        for i, out in enumerate(outputs):
            parsed_out = _try_parse_json(out, "output")
            child_metadata = dict(metadata)
            if per_row_usage:
                child_metadata["usage_metadata"] = per_row_usage
            try:
                client.update_run(
                    _child_run_id(job_id, i),
                    end_time=now,
                    outputs=parsed_out,
                    extra={"metadata": child_metadata},
                )
            except Exception as e:
                if isinstance(e, LangSmithConflictError):
                    continue  # Already completed, skip
                logger.warning(
                    "Failed to update LangSmith child trace %d for batch job %s",
                    i,
                    job_id,
                    exc_info=True,
                )

        # Always try to complete the parent, even if some children failed.
        # The SDK may flush buffered child writes during this call, causing
        # a 409 conflict from a *child* that masks the parent update. Retry
        # a few times to work around this.
        for attempt in range(3):
            try:
                client.update_run(
                    parent_trace["id"],
                    end_time=now,
                    outputs={"job_id": job_id},
                )
                break
            except Exception as e:
                if isinstance(e, LangSmithConflictError):
                    continue  # Likely a buffered child conflict, retry
                logger.warning(
                    "Failed to complete LangSmith parent trace for batch job %s",
                    job_id,
                    exc_info=True,
                )
                break
    except Exception:
        logger.warning(
            "Failed to complete LangSmith traces for batch job %s",
            job_id,
            exc_info=True,
        )


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
