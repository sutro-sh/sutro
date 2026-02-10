import os
from typing import Optional, Dict, Any, List, Callable

from langsmith import traceable, get_current_run_tree


# TODO(cooper) expand this more or make it more ergonomic to use
# with batch style results. Perhaps there is a more native way in LangSmith
# to handle these?
def save_trace_to_langsmith(
    function_name: str,
    result: dict,
    langsmith_metadata: Optional[Dict[str, Any]] = None,
    langsmith_tags: Optional[List[str]] = None,
):
    """
    Log a single inference result to LangSmith as a trace (post-hoc). Intended to be
    used with results from a batch style job.

    Useful for logging batch results or results you already have in hand.
    Latency will NOT reflect the real API call duration.

    Args:
        function_name: The Function name (e.g. "clay-bert"). Used as ls_model_name.
        result: A result dict from the /functions/run endpoint.
        langsmith_metadata: Extra metadata merged into the trace.
        langsmith_tags: Tags attached to the trace for filtering.
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

    # Note latency with this won't reflect the real time it took to process
    # the request represented here
    @traceable(**traceable_kwargs)
    def _log_result() -> dict:
        _attach_run_metadata(result)
        return result

    _log_result()


def traced_run(
    function_name: str,
    api_call: Callable[[], dict],
    langsmith_metadata: Optional[Dict[str, Any]] = None,
    langsmith_tags: Optional[List[str]] = None,
) -> dict:
    """
    Execute an API call wrapped in a LangSmith trace. Intended to be used with
    singleton requests where we can wrap the code that makes the request and
    receives the response.

    The trace captures real wall-clock latency, token usage, and any errors
    from the actual call. Use this when you want accurate latency metrics.

    Args:
        function_name: The Function name (e.g. "clay-bert"). Used as ls_model_name.
        api_call: A zero-arg callable that performs the API request and returns
            the result dict. Exceptions will be captured in the trace and re-raised.
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
    def _traced_call() -> dict:
        result = api_call()
        _attach_run_metadata(result)
        return result

    return _traced_call()


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