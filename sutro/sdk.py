import requests
import pandas as pd
import polars as pl
import hashlib
import json
from typing import Union, List, Optional, Dict, Any, Type
import os
import warnings

from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners
from colorama import init
import time
from pydantic import BaseModel
import pyarrow.parquet as pq
import shutil
from sutro.common import (
    ModelOptions,
    prepare_input_data,
    normalize_output_schema,
    to_colored_text,
    fancy_tqdm,
    BASE_OUTPUT_COLOR,
)
from sutro.interfaces import JobStatus
from sutro.observability import (
    _is_langsmith_tracing_enabled,
    _create_batch_traces,
    _has_open_batch_traces,
    _complete_batch_traces,
    _traced_run,
)
from sutro.templates.classification import ClassificationTemplates
from sutro.templates.embed import EmbeddingTemplates
from sutro.templates.evals import EvalTemplates
from sutro.validation import (
    DIRECT_TENSOR_FACTORY_API_ERROR,
    check_version,
    normalize_api_url,
    resolve_environment_api_configuration_with_context,
    resolve_api_configuration_with_context,
)

JOB_NAME_CHAR_LIMIT = 45
JOB_DESCRIPTION_CHAR_LIMIT = 512

# (connect, read) timeouts for direct requests to presigned URLs. The read
# timeout bounds inactivity between streamed chunks, not the whole download.
DOWNLOAD_REQUEST_TIMEOUT = (10, 120)

# (connect, read) timeouts for a real-time Function run. The server answers or
# fails the request within its own 90 s deadline; the read timeout sits just
# above it so a stalled connection cannot leave the caller waiting forever.
FUNCTION_RUN_REQUEST_TIMEOUT = (10, 100)

# Initialize colorama (required for Windows)
init()


SPINNER = Spinners.dots14


# Isn't fully support in all terminals unfortunately. We should switch to Rich
# at some point, but even Rich links aren't clickable on MacOS Terminal


class SutroConfigurationError(RuntimeError):
    """Raised when the SDK is used without deployment API configuration."""


class SutroRateLimitError(requests.HTTPError):
    """Raised when a Function run is rate limited (HTTP 429).

    ``retry_after`` is the number of seconds the deployment asked the caller to
    wait, taken from the ``Retry-After`` response header.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        detail: Optional[str] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        response: Optional[requests.Response] = None,
    ):
        super().__init__(message, response=response)
        self.retry_after = retry_after
        self.detail = detail
        self.code = code
        self.request_id = request_id


class SutroValidationError(requests.HTTPError):
    """Raised when a Function run is rejected as invalid (HTTP 422).

    ``detail`` names the offending input field or the reason the Function
    cannot run, and ``code`` is the deployment's stable error code.
    """

    def __init__(
        self,
        message: str,
        *,
        detail: Optional[str] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        response: Optional[requests.Response] = None,
    ):
        super().__init__(message, response=response)
        self.detail = detail
        self.code = code
        self.request_id = request_id


class FunctionRunResult(dict):
    """The result of a single :meth:`Sutro.run_function` call.

    The full response payload is available as a dict; the documented fields are
    also exposed as attributes.
    """

    @property
    def output(self) -> Any:
        """The Function's answer, parsed against its output schema."""
        return self.get("output")

    @property
    def confidence(self) -> Optional[float]:
        """The confidence score for this answer, between 0 and 1."""
        return self.get("confidence")

    @property
    def usage(self) -> Dict[str, Any]:
        """Token counts and cost for the request."""
        return self.get("usage") or {}

    @property
    def request_id(self) -> Optional[str]:
        """The deployment's ID for this request, for support and log lookups."""
        return self.get("request_id")

    @property
    def function(self) -> Dict[str, Any]:
        """The Function name, model, and model source."""
        return self.get("function") or {}


def _function_run_input(input_data: Union[dict, str, BaseModel]) -> Any:
    """Normalize a ``run_function`` input into the JSON the API accepts."""
    if isinstance(input_data, BaseModel):
        return input_data.model_dump(mode="json")
    if isinstance(input_data, (dict, str)):
        return input_data
    raise TypeError(
        "run_function() input must be a dict, a string, or a pydantic model, "
        f"not {type(input_data).__name__}."
    )


def _error_body(response: Optional[requests.Response]) -> Dict[str, Any]:
    if response is None:
        return {}
    try:
        body = response.json()
    except (TypeError, ValueError, requests.exceptions.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _retry_after_seconds(response: Optional[requests.Response]) -> Optional[float]:
    """Seconds to wait, from the ``Retry-After`` header when it is a delay."""
    if response is None:
        return None
    header = (response.headers or {}).get("Retry-After")
    if header is None:
        return None
    try:
        return float(str(header).strip())
    except ValueError:
        # An HTTP-date Retry-After is legal but Sutro never sends one.
        return None


def _function_run_error(error: requests.HTTPError) -> requests.HTTPError:
    """Map a Function run failure onto the exception the caller should see."""
    response = error.response
    status_code = response.status_code if response is not None else None
    body = _error_body(response)
    detail = body.get("detail") or str(error)
    code = body.get("code")
    request_id = body.get("request_id")

    if status_code == 429:
        return SutroRateLimitError(
            detail,
            retry_after=_retry_after_seconds(response),
            detail=detail,
            code=code,
            request_id=request_id,
            response=response,
        )
    if status_code == 422:
        return SutroValidationError(
            detail,
            detail=detail,
            code=code,
            request_id=request_id,
            response=response,
        )

    error.detail = detail
    error.code = code
    error.request_id = request_id
    return error


class Sutro(EmbeddingTemplates, ClassificationTemplates, EvalTemplates):
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        serving_base_url: str = None,
        *,
        api_url: str = None,
    ):
        if base_url is not None:
            warnings.warn(
                "base_url is deprecated; use api_url instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if api_url is not None and base_url is not None:
            normalized_api_url = normalize_api_url(api_url)
            normalized_base_url = normalize_api_url(base_url)
            if normalized_api_url != normalized_base_url:
                raise ValueError(
                    "api_url and deprecated base_url must refer to the same URL."
                )
            configured_api_url = normalized_api_url
        else:
            configured_api_url = api_url if api_url is not None else base_url
            if configured_api_url is not None:
                configured_api_url = normalize_api_url(configured_api_url)

        if api_key is not None and configured_api_url is not None:
            # A complete explicit pair must not depend on ambient config.
            resolved_api_key, resolved_api_url, resolved_api_url_error = (
                None,
                None,
                None,
            )
        elif api_key is not None:
            # A caller-supplied key may pair with an environment URL because
            # the environment is an intentional per-process override. Never
            # borrow a persisted URL here: it may be stale and belong to the
            # persisted key that the caller explicitly replaced.
            resolved_api_key, resolved_api_url, resolved_api_url_error = (
                resolve_environment_api_configuration_with_context()
            )
        else:
            resolved_api_key, resolved_api_url, resolved_api_url_error = (
                resolve_api_configuration_with_context()
            )

        if configured_api_url is None:
            # An explicit key can safely use the URL resolved from the same
            # environment/config snapshot. Invalid inherited URLs remain
            # non-routable and surface their original error on first use.
            self._api_url = resolved_api_url
            self._api_url_error = resolved_api_url_error
            self.api_key = api_key if api_key is not None else resolved_api_key
        else:
            self._api_url = configured_api_url
            self._api_url_error = None
            if api_key is not None:
                self.api_key = api_key
            elif configured_api_url == resolved_api_url:
                # Reuse a fallback key only when its URL proves it belongs to
                # the exact deployment explicitly selected by the caller.
                self.api_key = resolved_api_key
            else:
                self.api_key = None

        if serving_base_url is not None:
            warnings.warn(
                "serving_base_url is deprecated; synchronous run_function() is "
                "not available through Sutro deployments.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.serving_base_url = serving_base_url
        check_version("sutro")

    @property
    def api_url(self) -> Optional[str]:
        """Canonical Sutro deployment ``/v1`` prefix used for all requests."""
        return self._api_url

    @api_url.setter
    def api_url(self, api_url: str):
        normalized_api_url = normalize_api_url(api_url)
        current_api_url = getattr(self, "_api_url", None)
        if normalized_api_url != current_api_url:
            # Deployment keys are scoped. Changing the destination invalidates
            # the prior pairing so it cannot be sent to a different deployment.
            self.api_key = None
        self._api_url = normalized_api_url
        self._api_url_error = None

    @property
    def base_url(self) -> Optional[str]:
        """Deprecated alias for :attr:`api_url`."""
        return self.api_url

    @base_url.setter
    def base_url(self, base_url: str):
        warnings.warn(
            "base_url is deprecated; use api_url instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.api_url = base_url

    def set_api_key(self, api_key: str):
        """
        Set the API key for the Sutro API.

        This method allows you to set the API key for the Sutro API.
        The API key is used to authenticate requests to the API.

        Args:
            api_key (str): The API key to set.

        Returns:
            None
        """
        self.api_key = api_key

    def set_api_url(self, api_url: str):
        """Set the Sutro deployment URL used by the Sutro API.

        The value may be either the deployment origin or its ``/v1`` API
        prefix. It is normalized to ``<origin>/v1``. Changing deployments
        clears the current API key; call :meth:`set_api_key` with a key issued
        by the new deployment before making a request.
        """
        self.api_url = api_url

    def set_base_url(self, base_url: str):
        """Deprecated alias for :meth:`set_api_url`."""
        warnings.warn(
            "set_base_url() is deprecated; use set_api_url() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_api_url(base_url)

    def set_serving_base_url(self, serving_base_url: str):
        """
        Store the deprecated serving URL for source compatibility.

        Synchronous Function execution is not available through Sutro yet,
        so this value is not used for requests.

        Args:
            serving_base_url (str): The serving base URL to set.
        """
        warnings.warn(
            "set_serving_base_url() is deprecated; synchronous run_function() "
            "is not available through Sutro deployments.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.serving_base_url = serving_base_url

    def do_request(
        self,
        method: str,
        endpoint: str,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        max_retries: int = 5,
        **kwargs: Any,
    ):
        """
        Helper to make authenticated requests.
        """
        api_url = self.api_url if base_url_override is None else base_url_override
        if api_url is None:
            if self._api_url_error is not None:
                raise SutroConfigurationError(self._api_url_error)
            raise SutroConfigurationError(
                "Sutro API URL is not configured. Set SUTRO_API_URL to your "
                "Sutro deployment URL (for example, "
                "https://sutro.example.com)."
            )
        api_url = normalize_api_url(api_url)
        if base_url_override is not None:
            if api_url != self.api_url and api_key_override is None:
                raise SutroConfigurationError(
                    "Overriding the Sutro API URL requires api_key_override so "
                    "a key scoped to one deployment is never sent to another."
                )
        url = api_url.rstrip("/") + "/" + endpoint.lstrip("/")

        key = self.api_key if api_key_override is None else api_key_override
        if not isinstance(key, str) or not key.strip():
            raise SutroConfigurationError(
                "Sutro API key is not configured. Create a key in your Sutro "
                "deployment's API Keys panel and set SUTRO_API_KEY."
            )
        headers = {"Authorization": f"Key {key}"}

        # Merge with any headers passed in kwargs
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        # Helper to make the actual HTTP request
        def _make_request():
            method_upper = method.upper()
            if method_upper == "GET":
                return requests.get(url, headers=headers, **kwargs)
            elif method_upper == "POST":
                return requests.post(url, headers=headers, **kwargs)
            elif method_upper == "PUT":
                return requests.put(url, headers=headers, **kwargs)
            elif method_upper == "DELETE":
                return requests.delete(url, headers=headers, **kwargs)
            elif method_upper == "PATCH":
                return requests.patch(url, headers=headers, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        def _raise_direct_tensor_factory_configuration_error(
            error: requests.HTTPError,
        ) -> None:
            response = error.response
            if response is None or response.status_code != 410:
                return
            try:
                response_data = response.json()
            except (TypeError, ValueError):
                return
            detail = (
                response_data.get("detail")
                if isinstance(response_data, dict)
                else None
            )
            if detail == DIRECT_TENSOR_FACTORY_API_ERROR:
                raise SutroConfigurationError(detail) from error

        # Make initial request
        try:
            response = _make_request()
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            status_code = (
                e.response.status_code if e.response is not None else None
            )
            _raise_direct_tensor_factory_configuration_error(e)

            # Only retry on Cloudflare 524 timeout errors when retries are enabled.
            if status_code != 524 or max_retries <= 0:
                raise

            for attempt in range(max_retries):
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                retry_message = (
                    f"⚠️  Cloudflare timeout (524). Retrying in {wait_time}s... "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                print(to_colored_text(retry_message))
                time.sleep(wait_time)

                try:
                    response = _make_request()
                    response.raise_for_status()
                    return response
                except requests.HTTPError as retry_error:
                    retry_status_code = (
                        retry_error.response.status_code
                        if retry_error.response is not None
                        else None
                    )
                    _raise_direct_tensor_factory_configuration_error(retry_error)
                    # If not a 524 or this was the last retry, raise the error
                    if retry_status_code != 524 or attempt == max_retries - 1:
                        raise
                    # Otherwise continue to next retry attempt

    def _run_one_batch_inference(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: ModelOptions,
        column: Union[str, List[str]],
        output_column: str,
        job_priority: int,
        json_schema: Dict[str, Any],
        sampling_params: dict,
        system_prompt: str,
        cost_estimate: bool,
        stay_attached: Optional[bool],
        random_seed_per_input: bool,
        truncate_rows: bool,
        name: str,
        description: str,
        id_column: Optional[str],
    ):
        # Validate name and description lengths
        if name is not None and len(name) > JOB_NAME_CHAR_LIMIT:
            raise ValueError(
                f"Job name cannot exceed {JOB_NAME_CHAR_LIMIT} characters."
            )
        if description is not None and len(description) > JOB_DESCRIPTION_CHAR_LIMIT:
            raise ValueError(
                f"Job description cannot exceed {JOB_DESCRIPTION_CHAR_LIMIT} characters."
            )

        if id_column is not None and not (
            isinstance(data, str)
            and data.startswith(("https://", "http://"))
        ):
            raise ValueError(
                "id_column is only supported for HTTP(S) download URL inputs."
            )

        input_data, column_name = prepare_input_data(data, column)
        payload = {
            "model": model,
            "inputs": input_data,
            "job_priority": job_priority,
            "json_schema": json_schema,
            "system_prompt": system_prompt,
            "cost_estimate": cost_estimate,
            "sampling_params": sampling_params,
            "random_seed_per_input": random_seed_per_input,
            "truncate_rows": truncate_rows,
            "name": name,
            "description": description,
        }
        if column_name is not None:
            payload["column_name"] = column_name
        if id_column is not None:
            payload["id_column_name"] = id_column

        # There are two gotchas with yaspin:
        # 1. Can't use print while in spinner is running
        # 2. When writing to stdout via spinner.fail, spinner.write etc, there is a pretty strict
        # limit for content length in jupyter notebooks, where it wisll give an error about:
        # Terminal size {self._terminal_width} is too small to display spinner with the given settings.
        # https://github.com/pavdmyt/yaspin/blob/9c7430b499ab4611888ece39783a870e4a05fa45/yaspin/core.py#L568-L571
        job_id = None
        t = f"Creating {'[cost estimate] ' if cost_estimate else ''}priority {job_priority} job"
        spinner_text = to_colored_text(t)

        try:
            with yaspin(SPINNER, text=spinner_text, color=BASE_OUTPUT_COLOR) as spinner:
                try:
                    # A 524 does not tell us whether the server created the job.
                    # Retrying this non-idempotent submission could create duplicates.
                    response = self.do_request(
                        "POST",
                        "batch-inference",
                        max_retries=0,
                        json=payload,
                    )
                    response_data = response.json()
                except requests.HTTPError as e:
                    response = e.response
                    response_data = response.json()
                if response.status_code != 200:
                    spinner.write(
                        to_colored_text(f"Error: {response.status_code}", state="fail")
                    )
                    spinner.stop()
                    print(to_colored_text(response_data, state="fail"))
                    return None
                else:
                    job_id = response_data["results"]
                    if cost_estimate:
                        spinner.write(
                            to_colored_text(
                                f"Awaiting cost estimates with job ID: {job_id}. You can safely detach and retrieve the cost estimates later."
                            )
                        )
                        spinner.stop()
                        self.await_job_completion(
                            job_id, obtain_results=False, is_cost_estimate=True
                        )
                        cost_estimate = self._get_job_cost_estimate(job_id)
                        spinner.write(
                            to_colored_text(
                                f"✔ Cost estimates retrieved for job {job_id}: ${cost_estimate}",
                                state="success",
                            )
                        )
                        return job_id
                    else:
                        name_text = f" and name {name}" if name is not None else ""
                        spinner.write(
                            to_colored_text(
                                f"🛠 Priority {job_priority} Job created with ID: {job_id}{name_text}",
                                state="success",
                            )
                        )
                        spinner.write(to_colored_text(f"Model: {model}"))
                        if not stay_attached:
                            # TODO: Restore a deployment-local batch UI link when
                            # Sutro deployments expose one.
                            spinner.write(
                                to_colored_text(
                                    f"Use `so.get_job_status('{job_id}')` to check "
                                    "the status of the job."
                                )
                            )
                            return job_id
        except KeyboardInterrupt:
            pass
        finally:
            if spinner:
                spinner.stop()

        success = False
        if stay_attached and job_id is not None:
            spinner.write(
                to_colored_text(
                    "Awaiting job start...",
                )
            )
            # TODO: Restore a deployment-local batch UI link when Sutro
            # deployments expose one.
            started = self._await_job_start(job_id)
            if not started:
                failure_reason = self._get_failure_reason(job_id)
                spinner.write(
                    to_colored_text(
                        f"Failure reason: {failure_reason['message']}", "fail"
                    )
                )
                return None

            pbar = None

            try:
                with self.do_request(
                    "GET",
                    f"/stream-job-progress/{job_id}",
                    stream=True,
                ) as streaming_response:
                    streaming_response.raise_for_status()
                    spinner = yaspin(
                        SPINNER,
                        text=to_colored_text("Awaiting status updates..."),
                        color=BASE_OUTPUT_COLOR,
                    )
                    spinner.start()

                    token_state = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens_processed_per_second": 0,
                    }

                    for line in streaming_response.iter_lines():
                        if line:
                            try:
                                json_obj = json.loads(line)
                            except json.JSONDecodeError:
                                print("Error: ", line, flush=True)
                                continue

                            if json_obj["update_type"] == "progress":
                                if pbar is None:
                                    spinner.stop()
                                    postfix = "Input tokens processed: 0"
                                    pbar = fancy_tqdm(
                                        total=len(input_data),
                                        desc="Progress",
                                        style=1,
                                        postfix=postfix,
                                    )
                                if json_obj["result"] > pbar.n:
                                    pbar.update(json_obj["result"] - pbar.n)
                                    pbar.refresh()
                                if json_obj["result"] == len(input_data):
                                    success = True
                            elif json_obj["update_type"] == "tokens":
                                # Update only the values that are present in this update
                                # Currently, the way the progress stream endpoint is defined,
                                # its possible to have updates come in that only have 1 or 2 fields
                                new = {
                                    k: v
                                    for k, v in json_obj.get("result", {}).items()
                                    if k in token_state and v >= token_state[k]
                                }
                                token_state.update(new)

                                if pbar is not None:
                                    pbar.postfix = f"Input tokens processed: {token_state['input_tokens']}, Output tokens generated: {token_state['output_tokens']}, Total tokens/s: {token_state['total_tokens_processed_per_second']}"
                                    pbar.refresh()

            except KeyboardInterrupt:
                pass
            finally:
                # Need to clean these up on keyboard exit otherwise it causes
                # an error
                if pbar is not None:
                    pbar.close()
                if spinner is not None:
                    spinner.stop()
            if success:
                spinner.text = to_colored_text(
                    "✔ Job succeeded. Obtaining results...", state="success"
                )
                spinner.start()

                # TODO: we implment retries in cases where the job hasn't written results yet
                # it would be better if we could receive a fully succeeded status from the job
                # and not have such a race condition
                max_retries = 20  # winds up being 100 seconds cumulative delay
                retry_delay = 5  # initial delay in seconds
                job_results_response = None
                for _ in range(max_retries):
                    try:
                        job_results_response = self.do_request(
                            "POST",
                            "job-results",
                            json={
                                "job_id": job_id,
                            },
                        )
                        break
                    except requests.HTTPError:
                        time.sleep(retry_delay)
                        continue

                if not job_results_response or job_results_response.status_code != 200:
                    spinner.write(
                        to_colored_text(
                            "Job succeeded, but results are not yet available. Use `so.get_job_results('{job_id}')` to obtain results.",
                            state="fail",
                        )
                    )
                    spinner.stop()
                    return None

                results = job_results_response.json()["results"]["outputs"]

                if isinstance(data, (pd.DataFrame, pl.DataFrame)):
                    if isinstance(data, pd.DataFrame):
                        data[output_column] = results
                    elif isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.Series(output_column, results))
                    print(data)
                    spinner.write(
                        to_colored_text(
                            f"✔ Displaying result preview. You can join the results on the original dataframe with `so.get_job_results('{job_id}', with_original_df=<original_df>)`",
                            state="success",
                        )
                    )
                else:
                    print(results)
                    spinner.write(
                        to_colored_text(
                            f"✔ Job results received. You can re-obtain the results with `so.get_job_results('{job_id}')`",
                            state="success",
                        )
                    )
                spinner.stop()

                return job_id
            return None
        return None

    def infer(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: ModelOptions = "gpt-oss-20b",
        name: Optional[str] = None,
        description: Optional[str] = None,
        column: Union[str, List[str]] = None,
        output_column: str = "inference_result",
        job_priority: int = 0,
        output_schema: Union[Dict[str, Any], Type[BaseModel]] = None,
        sampling_params: dict = None,
        system_prompt: str = None,
        dry_run: bool = False,
        stay_attached: Optional[bool] = None,
        random_seed_per_input: bool = False,
        truncate_rows: bool = True,
        id_column: Optional[str] = None,
    ):
        """
        Run inference on the provided data.

        This method allows you to run inference on the provided data using the Sutro API.
        It supports various data types such as lists, DataFrames (Polars or Pandas), file paths and download URLs.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            model (ModelOptions, optional): The model to use for inference. Defaults to "gemma-3-12b-it".
            name (str, optional): A job name for experiment/metadata tracking purposes. Defaults to None.
            description (str, optional): A job description for experiment/metadata tracking purposes. Defaults to None.
            column (Union[str, List[str]], optional): The column name to use for inference. Required if data is a DataFrame or file path. If a list is supplied, it will concatenate the columns of the list into a single column, accepting separator strings.
            output_column (str, optional): The column name to store the inference results in if the input is a DataFrame. Defaults to "inference_result".
            job_priority (int, optional): The priority of the job. Defaults to 0.
            output_schema (Union[Dict[str, Any], BaseModel], optional): A structured schema for the output.
                Can be either a dictionary representing a JSON schema or a class that inherits from Pydantic BaseModel. Defaults to None.
            sampling_params: (dict, optional): The sampling parameters to use at generation time, ie temperature, top_p etc.
            system_prompt (str, optional): A system prompt to add to all inputs. This allows you to define the behavior of the model. Defaults to None.
            dry_run (bool, optional): If True, the method will return cost estimates instead of running inference. Defaults to False.
            stay_attached (bool, optional): If True, the method will stay attached to the job until it is complete. Defaults to True for prototyping jobs, False otherwise.
            random_seed_per_input (bool, optional): If True, the method will use a different random seed for each input. Defaults to False.
            truncate_rows (bool, optional): If True, any rows that have a token count exceeding the context window length of the selected model will be truncated to the max length that will fit within the context window. Defaults to True.
            id_column (str, optional): ID column to carry into results for
                HTTP(S) CSV or Parquet download URL inputs. Defaults to None.

        Returns:
            str: The ID of the inference job.

        """
        # Default stay_attached to True for prototyping jobs (priority 0)
        if stay_attached is None:
            stay_attached = job_priority == 0

        json_schema = None
        if output_schema:
            # Convert BaseModel to dict if needed
            json_schema = normalize_output_schema(output_schema)

        return self._run_one_batch_inference(
            data,
            model,
            column,
            output_column,
            job_priority,
            json_schema,
            sampling_params,
            system_prompt,
            dry_run,
            stay_attached,
            random_seed_per_input,
            truncate_rows,
            name,
            description,
            id_column,
        )

    def run_function(
        self,
        name: str,
        input_data: Union[dict, str, BaseModel],
        langsmith_metadata: Optional[Dict[str, Any]] = None,
        langsmith_tags: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> FunctionRunResult:
        """
        Run a published Sutro Function on a single input and wait for the answer.

        The Function supplies its prompt, model, output schema, and generation
        defaults. Use :meth:`batch_run_function` instead for large tables: this
        method makes one synchronous request per call and is rate limited.

        Automatically traces to LangSmith when ``LANGSMITH_TRACING=true`` is set.

        Args:
            name (str): The name of the Sutro Function to run.
            input_data (dict | str | BaseModel): The Function's input fields.
                Keys must match the Function's configured inputs. Image and PDF
                fields take an :class:`sutro.Asset` or :class:`sutro.Image`
                value. A bare string is accepted when the Function has exactly
                one text input.
            langsmith_metadata (dict, optional): Additional metadata to attach to
                the LangSmith trace. Only used when tracing is enabled.
            langsmith_tags (list, optional): Tags to attach to the LangSmith trace
                for filtering. Only used when tracing is enabled.
            timeout_seconds (float, optional): How long to wait for the answer.
                Defaults to just above the deployment's standard 90 s request
                deadline; set it above a deployment's own deadline when that
                has been raised.

        Returns:
            FunctionRunResult: The response payload, with ``output``,
            ``confidence``, ``usage``, ``request_id``, and ``function``
            available as attributes.

        Raises:
            SutroRateLimitError: The deployment is rate limiting this API key.
                Wait ``retry_after`` seconds and try again.
            SutroValidationError: The input does not match the Function, or the
                Function cannot be run in real time on its current model.
            requests.HTTPError: Any other API failure. ``detail``, ``code``, and
                ``request_id`` are attached to the exception.
        """
        payload_input = _function_run_input(input_data)
        timeout = (
            FUNCTION_RUN_REQUEST_TIMEOUT
            if timeout_seconds is None
            else (FUNCTION_RUN_REQUEST_TIMEOUT[0], timeout_seconds)
        )

        def _call(request_input: Any) -> Dict[str, Any]:
            try:
                # No automatic retries: the caller is waiting, and a Function
                # run is not idempotent from the usage ledger's point of view.
                response = self.do_request(
                    "POST",
                    f"functions/{name}/run",
                    json={"input": request_input},
                    max_retries=0,
                    timeout=timeout,
                )
            except requests.HTTPError as e:
                raise _function_run_error(e) from None
            return response.json()

        if _is_langsmith_tracing_enabled():
            payload = _traced_run(
                name,
                _call,
                payload_input,
                langsmith_metadata,
                langsmith_tags,
            )
        else:
            payload = _call(payload_input)

        return FunctionRunResult(payload)

    def batch_run_function(
        self,
        name: str,
        data: Union[List[dict], pl.DataFrame, pd.DataFrame, str],
        job_priority: int | None = 0,
        output_column: str = "inference_result",
        dry_run: bool = False,
        stay_attached: bool = False,
        job_name: Optional[str] = None,
        description: Optional[str] = None,
        langsmith_metadata: Optional[Dict[str, Any]] = None,
        langsmith_tags: Optional[List[str]] = None,
        id_column: Optional[str] = None,
    ):
        """
        Run a Sutro Function on a large table, dataframe, or file using batch processing.

        This is a convenience method for running batch inference with Functions.
        The job priority defaults to 0 and can be changed with ``job_priority``.

        Automatically traces to LangSmith when LANGSMITH_TRACING=true is set, not a dry run and stay_attached=False.
        A parent trace is created at job submission time, and child traces (one per row) are added when results
        are retrieved via get_job_results().

        Args:
            name (str): The name of the Sutro Function to use.
            data (Union[List[dict], pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
                Accepts a list of dictionaries, a DataFrame, a path to a parquet/CSV file,
                or an HTTP(S) download URL. Dictionary keys or table columns must match
                the function's expected schema.
            output_column (str, optional): The column name to store the inference results.
                Defaults to "inference_result".
            dry_run (bool, optional): If True, return cost estimates instead of running inference.
                Defaults to False.
            stay_attached (bool, optional): If True, the SDK will stay attached to the job and
                stream progress updates. Not compatible with HTTP(S) inputs or LangSmith
                tracing. Defaults to False.
            job_name (str, optional): A job name for experiment/metadata tracking purposes.
                Defaults to None.
            description (str, optional): A job description for experiment/metadata tracking purposes.
                Defaults to None.
            langsmith_metadata (dict, optional): Additional metadata to attach to the LangSmith trace.
                Only used when tracing is enabled.
            langsmith_tags (list, optional): Tags to attach to the LangSmith trace for filtering.
                Only used when tracing is enabled.
            id_column (str, optional): ID column to carry into results for an
                HTTP(S) CSV or Parquet download URL. Defaults to None.

        Returns:
            str: The ID of the batch job.
        """
        is_remote_input = isinstance(data, str) and data.startswith(
            ("https://", "http://")
        )
        if stay_attached and is_remote_input:
            raise ValueError(
                "stay_attached=True is not supported for HTTP(S) Function inputs."
            )

        # Convert DataFrames/files to list of dicts for function calls
        if stay_attached and _is_langsmith_tracing_enabled():
            raise ValueError(
                "Attached mode is not compatablie with LangSmith tracing. "
                "Please set one of stay_attached OR LANGSMITH_TRACING env var."
            )

        if isinstance(data, pd.DataFrame):
            input_data = data.to_dict(orient="records")
        elif isinstance(data, pl.DataFrame):
            input_data = data.to_dicts()
        elif isinstance(data, str):
            if is_remote_input:
                input_data = data
            else:
                file_ext = os.path.splitext(data)[1].lower()
                if file_ext == ".csv":
                    input_data = pl.read_csv(data).to_dicts()
                elif file_ext == ".parquet":
                    input_data = pl.read_parquet(data).to_dicts()
                else:
                    raise ValueError(
                        f"Unsupported file type: {file_ext}. Use .csv or .parquet"
                    )
        elif isinstance(data, list):
            input_data = data
        else:
            raise ValueError(
                "The only acceptable arguments for the `data` parameter are List[dict],"
                " pl.DataFrame, pd.DataFrame, or str where str is a filepath to a "
                "Parquet or CSV file"
            )

        job_id = self.infer(
            data=input_data,
            model=name,
            name=job_name,
            description=description,
            output_column=output_column,
            job_priority=job_priority,
            dry_run=dry_run,
            stay_attached=stay_attached,
            truncate_rows=False,
            id_column=id_column,
        )

        if (
            job_id
            and not dry_run
            and not stay_attached
            and not isinstance(input_data, str)
        ):
            traced = _create_batch_traces(
                function_name=name,
                job_id=job_id,
                input_data=input_data,
                langsmith_metadata=langsmith_metadata,
                langsmith_tags=langsmith_tags,
            )
            if traced:
                print(
                    to_colored_text(
                        f"📊 LangSmith tracing enabled — {len(input_data):,} traces created for {job_id}"
                    )
                )

        return job_id

    def infer_per_model(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        models: List[ModelOptions],
        names: List[str] = None,
        descriptions: List[str] = None,
        column: Union[str, List[str]] = None,
        output_column: str = "inference_result",
        job_priority: int = 0,
        output_schema: Union[Dict[str, Any], Type[BaseModel]] = None,
        sampling_params: dict = None,
        system_prompt: str = None,
        dry_run: bool = False,
        random_seed_per_input: bool = False,
        truncate_rows: bool = True,
        id_column: Optional[str] = None,
    ):
        """
        Run inference on the provided data, across multiple models. This method is often useful to sampling outputs from multiple models across the same data and compare the job_ids.

        For input data, it supports various data types such as lists, DataFrames (Polars or Pandas), file paths and download URLs.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            models (Union[ModelOptions, List[ModelOptions]], optional): The models to use for inference. Fans out each model to its own seperate job, over the same data.
            names (Union[str, List[str]], optional): A job name for experiment/metadata tracking purposes. If using a list of models, you must pass a list of names with length equal to the number of models, or None. Defaults to None.
            descriptions (Union[str, List[str]], optional): A job description for experiment/metadata tracking purposes. If using a list of models, you must pass a list of descriptions with length equal to the number of models, or None. Defaults to None.
            column (Union[str, List[str]], optional): The column name to use for inference. Required if data is a DataFrame or file path. If a list is supplied, it will concatenate the columns of the list into a single column, accepting separator strings.
            output_column (str, optional): The column name to store the inference job_ids in if the input is a DataFrame. Defaults to "inference_result".
            job_priority (int, optional): The priority of the job. Defaults to 0.
            output_schema (Union[Dict[str, Any], BaseModel], optional): A structured schema for the output.
                Can be either a dictionary representing a JSON schema or a class that inherits from Pydantic BaseModel. Defaults to None.
            sampling_params: (dict, optional): The sampling parameters to use at generation time, ie temperature, top_p etc.
            system_prompt (str, optional): A system prompt to add to all inputs. This allows you to define the behavior of the model. Defaults to None.
            dry_run (bool, optional): If True, the method will return cost estimates instead of running inference. Defaults to False.
            stay_attached (bool, optional): If True, the method will stay attached to the job until it is complete. Defaults to True for prototyping jobs, False otherwise.
            random_seed_per_input (bool, optional): If True, the method will use a different random seed for each input. Defaults to False.
            truncate_rows (bool, optional): If True, any rows that have a token count exceeding the context window length of the selected model will be truncated to the max length that will fit within the context window. Defaults to True.
            id_column (str, optional): ID column to carry into results for
                HTTP(S) CSV or Parquet download URL inputs. Defaults to None.

        Returns:
            str: The ID of the inference job.

        """
        if isinstance(names, list):
            if len(names) != len(models):
                raise ValueError(
                    "names parameter must be the same length as the models parameter."
                )
        elif names is None:
            names = [None] * len(models)
        else:
            raise ValueError(
                "names parameter must be  a list or None if using a list of models"
            )

        if isinstance(descriptions, list):
            if len(descriptions) != len(models):
                raise ValueError(
                    "descriptions parameter must be the same length as the models"
                    " parameter."
                )
        elif descriptions is None:
            descriptions = [None] * len(models)
        else:
            raise ValueError(
                "descriptions parameter must be a list or None if using a list of "
                "models"
            )

        json_schema = None
        if output_schema:
            # Convert BaseModel to dict if needed
            json_schema = normalize_output_schema(output_schema)

        def start_job(
            model_singleton: ModelOptions,
            name_singleton: str | None,
            description_singleton: str | None,
        ):
            return self._run_one_batch_inference(
                data,
                model_singleton,
                column,
                output_column,
                job_priority,
                json_schema,
                sampling_params,
                system_prompt,
                dry_run,
                False,
                random_seed_per_input,
                truncate_rows,
                name_singleton,
                description_singleton,
                id_column,
            )

        job_ids = [
            start_job(model, name, description)
            for model, name, description in zip(
                models, names, descriptions, strict=True
            )
        ]

        return job_ids

    def attach(self, job_id):
        """
        Attach to an existing job and stream its progress.

        Args:
            job_id (str): The ID of the job to attach to
        """

        s = requests.Session()
        pbar = None

        with yaspin(
            SPINNER,
            text=to_colored_text("Looking for job..."),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            # Fetch the specific job we want to attach to
            job = self._fetch_job(job_id)

            if not job:
                spinner.write(to_colored_text(f"Job {job_id} not found", state="fail"))
                return

            match job.get("status"):
                case "SUCCEEDED":
                    spinner.write(
                        to_colored_text(
                            f"Job already completed. You can obtain the results with `sutro jobs results {job_id}`"
                        )
                    )
                    return
                case "FAILED":
                    spinner.write(
                        to_colored_text("❌ Job is in failed state.", state="fail")
                    )
                    return
                case "CANCELLED":
                    spinner.write(
                        to_colored_text("❌ Job was cancelled.", state="fail")
                    )
                    return
                case _:
                    spinner.write(to_colored_text("✔ Job found!", state="success"))

        total_rows = job["num_rows"]
        success = False

        try:
            with self.do_request(
                "GET",
                f"/stream-job-progress/{job_id}",
                stream=True,
            ) as streaming_response:
                streaming_response.raise_for_status()
                spinner = yaspin(
                    SPINNER,
                    text=to_colored_text("Awaiting status updates..."),
                    color=BASE_OUTPUT_COLOR,
                )
                # TODO: Restore a deployment-local batch UI link when Sutro
                # deployments expose one.
                spinner.start()
                for line in streaming_response.iter_lines():
                    if line:
                        try:
                            json_obj = json.loads(line)
                        except json.JSONDecodeError:
                            print("Error: ", line, flush=True)
                            continue

                        if json_obj["update_type"] == "progress":
                            if pbar is None:
                                spinner.stop()
                                postfix = "Input tokens processed: 0"
                                pbar = fancy_tqdm(
                                    total=total_rows,
                                    desc="Progress",
                                    style=1,
                                    postfix=postfix,
                                )
                            if json_obj["result"] > pbar.n:
                                pbar.update(json_obj["result"] - pbar.n)
                                pbar.refresh()
                            if json_obj["result"] == total_rows:
                                pbar.close()
                                success = True
                        elif json_obj["update_type"] == "tokens":
                            if pbar is not None:
                                pbar.postfix = f"Input tokens processed: {json_obj['result']['input_tokens']}, Tokens generated: {json_obj['result']['output_tokens']}, Total tokens/s: {json_obj['result'].get('total_tokens_processed_per_second')}"
                                pbar.refresh()

                if success:
                    spinner.write(
                        to_colored_text(
                            f"✔ Job succeeded. Use `sutro jobs results {job_id}` to obtain results.",
                            state="success",
                        )
                    )
                    spinner.stop()
        except KeyboardInterrupt:
            pass
        finally:
            if pbar:
                pbar.close()
            if spinner:
                spinner.stop()

    def fancy_tqdm(
        self,
        total: int,
        desc: str = "Progress",
        color: str = BASE_OUTPUT_COLOR,
        style=1,
        postfix: str = None,
    ):
        """
        Creates a customized tqdm progress bar with different styling options.

        Args:
            total (int): Total iterations
            desc (str): Description for the progress bar
            color (str): Color of the progress bar (green, blue, red, yellow, magenta)
            style (int): Style preset (1-4)
            postfix (str): Postfix for the progress bar
        """

        # Style presets
        style_presets = {
            1: {
                "bar_format": "{l_bar}{bar:30}| {n_fmt}/{total_fmt} | {percentage:3.0f}% {postfix}",
                "ascii": "░▒█",
            },
            2: {
                "bar_format": "╢{l_bar}{bar:30}╟ {percentage:3.0f}%",
                "ascii": "▁▂▃▄▅▆▇█",
            },
            3: {
                "bar_format": "{desc}: |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
                "ascii": "◯◔◑◕●",
            },
            4: {
                "bar_format": "⏳ {desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}",
                "ascii": "⬜⬛",
            },
            5: {
                "bar_format": "⏳ {desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}",
                "ascii": "▏▎▍▌▋▊▉█",
            },
        }

        # Get style configuration
        style_config = style_presets.get(style, style_presets[1])

        return tqdm(
            total=total,
            desc=desc,
            colour=color,
            bar_format=style_config["bar_format"],
            ascii=style_config["ascii"],
            ncols=80,
            dynamic_ncols=True,
            smoothing=0.3,
            leave=True,
            postfix=postfix,
        )

    def list_jobs(self):
        """
        List all jobs.

        This method retrieves a list of all jobs associated with the API key.

        Returns:
            list: A list of job details, or None if the request fails.
        """
        with yaspin(
            SPINNER, text=to_colored_text("Fetching jobs"), color=BASE_OUTPUT_COLOR
        ) as spinner:
            try:
                return self._list_all_jobs_for_user()
            except requests.HTTPError as e:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(e.response.json(), state="fail"))
                return None

    def _list_all_jobs_for_user(self):
        response = self.do_request("GET", "list-jobs")
        return response.json()["jobs"]

    def _fetch_job(self, job_id):
        """
        Helper function to fetch a single job.
        """
        try:
            response = self.do_request("GET", f"jobs/{job_id}")
            return response.json().get("job")
        except requests.HTTPError:
            return None

    def _get_job_cost_estimate(self, job_id: str):
        """
        Get the cost estimate for a job.
        """
        job = self._fetch_job(job_id)
        if not job:
            return None

        return job.get("cost_estimate")

    def _get_failure_reason(self, job_id: str):
        """
        Get the failure reason for a job.
        """
        job = self._fetch_job(job_id)
        if not job:
            return None
        return job.get("failure_reason")

    def _fetch_job_status(self, job_id: str):
        """
        Core logic to fetch job status from the API.

        Args:
            job_id (str): The ID of the job to retrieve the status for.

        Returns:
            dict: The response JSON from the API.

        Raises:
            requests.HTTPError: If the API returns a non-200 status code.
        """
        response = self.do_request("GET", f"job-status/{job_id}")
        return response.json()["job_status"][job_id]

    def get_job_status(self, job_id: str):
        """
        Get the status of a job by its ID.

        This method retrieves the status of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the status for.

        Returns:
            str: The status of the job.
        """
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Checking job status with ID: {job_id}"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            try:
                response_data = self._fetch_job_status(job_id)
                spinner.write(
                    to_colored_text("✔ Job status retrieved!", state="success")
                )
                return response_data
            except requests.HTTPError as e:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(e.response.json(), state="fail"))
                return None

    def get_job_results(
        self,
        job_id: str,
        include_inputs: bool = False,
        include_cumulative_logprobs: bool = False,
        with_original_df: pl.DataFrame | pd.DataFrame = None,
        output_column: str = "inference_result",
        disable_cache: bool = False,
        unpack_json: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Get the results of a job by its ID.

        This method retrieves the results of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the results for.
            include_inputs (bool, optional): Whether to include the inputs in the results. Defaults to False.
            include_cumulative_logprobs (bool, optional): Whether to include the cumulative logprobs in the results. Defaults to False.
            with_original_df (pd.DataFrame | pl.DataFrame, optional): Original DataFrame to concatenate with results. Defaults to None.
            output_column (str, optional): Name of the output column. Defaults to "inference_result".
            disable_cache (bool, optional): Whether to disable the cache. Defaults to False.
            unpack_json (bool, optional): If the output_column is formatted as a JSON string, decides whether to unpack the top level JSON fields in the results into separate columns. Defaults to True.

        Returns:
            Union[pl.DataFrame, pd.DataFrame]: The results as a DataFrame. By default, returns polars.DataFrame; when with_original_df is an instance of pandas.DataFrame, returns pandas.DataFrame.
        """

        cache_file_path = self._job_results_cache_file_path(job_id)
        self._remove_legacy_job_results_cache_file(job_id)
        expected_columns = {output_column}
        if include_inputs:
            expected_columns.add("inputs")
        if include_cumulative_logprobs:
            expected_columns.add("cumulative_logprobs")
        cached_contains_expected_columns = False
        if os.path.exists(cache_file_path):
            cached_columns = set(pq.read_schema(cache_file_path).names)
            cached_contains_expected_columns = expected_columns.issubset(
                cached_columns
            )

        # Check if open LangSmith traces exist for this job (created at
        # submission time via batch_run_function). If so, we'll complete
        # them with outputs once results are available.
        has_open_traces = _has_open_batch_traces(job_id)

        raw_outputs = None
        if not disable_cache and cached_contains_expected_columns:
            with yaspin(
                SPINNER,
                text=to_colored_text(f"Loading results from cache: {cache_file_path}"),
                color=BASE_OUTPUT_COLOR,
            ) as spinner:
                results_df = pl.read_parquet(cache_file_path)
                spinner.write(
                    to_colored_text("✔ Results loaded from cache", state="success")
                )
                if has_open_traces:
                    # The cache stores the renamed output column, not the raw
                    # API "outputs" key.
                    raw_outputs = (
                        results_df[output_column].to_list()
                        if output_column in results_df.columns
                        else None
                    )
        else:
            payload = {
                "job_id": job_id,
                "include_inputs": include_inputs,
                "include_cumulative_logprobs": include_cumulative_logprobs,
            }
            with yaspin(
                SPINNER,
                text=to_colored_text(f"Gathering results from job: {job_id}"),
                color=BASE_OUTPUT_COLOR,
            ) as spinner:
                try:
                    # TODO(cooper) refactor to use /jobs/{job_id}/results endpoint
                    #  (more resource efficient)
                    response = self.do_request("POST", "job-results", json=payload)
                    response_data = response.json()
                    spinner.write(
                        to_colored_text("✔ Job results retrieved", state="success")
                    )
                except requests.HTTPError as e:
                    spinner.write(
                        to_colored_text(
                            f"Bad status code: {e.response.status_code}", state="fail"
                        )
                    )
                    spinner.stop()
                    print(to_colored_text(e.response.json(), state="fail"))
                    return None

            if has_open_traces:
                raw_outputs = response_data["results"].get("outputs", [])

            results_df = pl.DataFrame(response_data["results"])
            results_df = results_df.rename({"outputs": output_column})

            if not disable_cache:
                os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                results_df.write_parquet(cache_file_path, compression="snappy")
                spinner.write(
                    to_colored_text("✔ Results saved to cache", state="success")
                )

        # Complete LangSmith traces with outputs regardless of source is cache or API request
        if has_open_traces and raw_outputs is not None:
            print(to_colored_text(f"📊 Completing LangSmith traces for {job_id}..."))
            job_details = self._fetch_job(job_id)
            _complete_batch_traces(
                job_id=job_id,
                num_rows=len(raw_outputs),
                outputs=raw_outputs,
                job_details=job_details,
            )
            print(to_colored_text(f"📊 LangSmith traces completed for {job_id}"))
        standard_columns = {
            "inputs",
            output_column,
            "cumulative_logprobs",
            "confidence_score",
        }
        metadata_columns = [
            column for column in results_df.columns if column not in standard_columns
        ]

        # Order inputs first, followed by user metadata, output, and diagnostics.
        column_config = [
            ("inputs", include_inputs),
            *((column, True) for column in metadata_columns),
            (output_column, True),
            ("cumulative_logprobs", include_cumulative_logprobs),
            ("confidence_score", "confidence_score" in results_df.columns),
        ]

        columns_to_keep = [
            col
            for col, include in column_config
            if include and col in results_df.columns
        ]

        results_df = results_df.select(columns_to_keep)

        if unpack_json:
            try:
                first_row = json.loads(
                    results_df.head(1)[output_column][0]
                )  # checks if the first row can be json decoded
                json_decoded_fields = first_row.keys()
            except Exception:
                # if the first row cannot be json decoded, do nothing
                pass
            else:
                conflicting_fields = sorted(
                    set(json_decoded_fields) & set(results_df.columns)
                )
                if conflicting_fields:
                    raise ValueError(
                        "Cannot unpack structured output fields that conflict with "
                        "existing result columns: "
                        f"{', '.join(conflicting_fields)}. Set unpack_json=False "
                        "to preserve the metadata and raw structured output."
                    )

                try:
                    results_df = results_df.map_columns(
                        output_column, lambda s: s.str.json_decode()
                    )
                    for field in json_decoded_fields:
                        results_df = results_df.with_columns(
                            pl.col(output_column).struct.field(field).alias(field)
                        )
                    if sorted(list(set(json_decoded_fields))) == [
                        "content",
                        "reasoning_content",
                    ]:  # if it's a reasoning model, we need to unpack the content field
                        content_keys = results_df.head(1)["content"][0].keys()
                        for key in content_keys:
                            results_df = results_df.with_columns(
                                pl.col("content").struct.field(key).alias(key)
                            )
                        results_df = results_df.drop("content")
                    results_df = results_df.drop(output_column)
                except Exception:
                    # if the output column cannot be unpacked, do nothing
                    pass

        # Handle concatenation with original DataFrame
        if with_original_df is not None:
            if isinstance(with_original_df, pd.DataFrame):
                # Convert to polars for consistent handling
                original_pl = pl.from_pandas(with_original_df)

                combined_df = original_pl.with_columns(results_df)

                # Convert back to pandas to match input type
                return combined_df.to_pandas()

            elif isinstance(with_original_df, pl.DataFrame):
                return with_original_df.with_columns(results_df)

        # Return pd.DataFrame type when appropriate
        if with_original_df is None and isinstance(with_original_df, pd.DataFrame):
            return results_df.to_pandas()

        return results_df

    def results_download_url(
        self,
        job_id: str,
        include_inputs: bool = False,
        include_cumulative_logprobs: bool = False,
        expires_in_seconds: int = 3600,
    ) -> Optional[dict]:
        """
        Get presigned download URLs for a job's results artifact.

        The backend materializes (or reuses) a single Parquet artifact
        containing the job's results and returns presigned URLs for it.
        Useful when you want to hand the download off to another system,
        issue partial Range reads, or download from a different machine.
        To simply save the results locally, use `download_job_results()`.

        Args:
            job_id (str): The ID of the job to retrieve results for.
            include_inputs (bool, optional): Whether to include the inputs in the results. Defaults to False.
            include_cumulative_logprobs (bool, optional): Whether to include the cumulative logprobs in the results. Defaults to False.
            expires_in_seconds (int, optional): How long the presigned URLs remain valid, up to 7 days. Defaults to 3600 (1 hour).

        Returns:
            Optional[dict]: Payload with `artifact` metadata (`filename`, `size_bytes`, ...) and presigned
            `urls` (`get` for downloading, `head` for metadata). The `get` URL supports HTTP Range
            requests. Returns None if the request fails.
        """
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Preparing results download for job: {job_id}"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            try:
                response = self.do_request(
                    "GET",
                    f"jobs/{job_id}/results-url",
                    params={
                        "format": "parquet",
                        "include_inputs": include_inputs,
                        "include_cumulative_logprobs": include_cumulative_logprobs,
                        "expires_in_seconds": expires_in_seconds,
                    },
                )
                spinner.write(
                    to_colored_text("✔ Results download URL ready", state="success")
                )
                return response.json()
            except requests.HTTPError as e:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                try:
                    detail = e.response.json()
                except ValueError:
                    # Intermediaries (e.g. Cloudflare) return HTML error pages.
                    detail = e.response.text
                print(to_colored_text(detail, state="fail"))
                return None
            except requests.RequestException as e:
                spinner.write(to_colored_text(f"Request failed: {e}", state="fail"))
                return None

    def download_job_results(
        self,
        job_id: str,
        output_path: Optional[str] = None,
        include_inputs: bool = False,
        include_cumulative_logprobs: bool = False,
        resume: bool = True,
        expires_in_seconds: int = 3600,
    ) -> Optional[str]:
        """
        Download a job's results as a Parquet file on local disk.

        Fetches presigned download URLs via `results_download_url()` and
        streams the artifact to disk with a progress bar. An interrupted
        download leaves a `.part` file behind; when `resume` is True,
        rerunning picks up where it left off via an HTTP Range request,
        as long as the artifact is unchanged server-side (validated by
        ETag). If the artifact changed, or the server doesn't expose an
        ETag to validate against, the download restarts from scratch.

        Args:
            job_id (str): The ID of the job to download results for.
            output_path (str, optional): Where to write the Parquet file. May be a directory
                (the server-provided artifact filename is used) or a full file path.
                Defaults to the artifact filename in the current directory.
            include_inputs (bool, optional): Whether to include the inputs in the results. Defaults to False.
            include_cumulative_logprobs (bool, optional): Whether to include the cumulative logprobs in the results. Defaults to False.
            resume (bool, optional): Whether to resume a partial download if one exists. Defaults to True.
            expires_in_seconds (int, optional): How long the presigned URLs remain valid, up to 7 days. Defaults to 3600 (1 hour).

        Returns:
            Optional[str]: The local path of the downloaded Parquet file, ready for
            `pl.read_parquet()`. Returns None if the request fails.
        """
        payload = self.results_download_url(
            job_id,
            include_inputs=include_inputs,
            include_cumulative_logprobs=include_cumulative_logprobs,
            expires_in_seconds=expires_in_seconds,
        )
        if payload is None:
            return None

        filename = payload["artifact"]["filename"]
        if output_path is None:
            destination = filename
        elif output_path.endswith(("/", os.sep)) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            destination = os.path.join(output_path, filename)
        else:
            destination = output_path
            parent_dir = os.path.dirname(destination)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        part_path = destination + ".part"
        etag_path = destination + ".part.etag"

        try:
            head_response = requests.head(
                payload["urls"]["head"], timeout=DOWNLOAD_REQUEST_TIMEOUT
            )
            head_response.raise_for_status()
            # Some S3-compatible gateways omit ETag; without one, resume
            # can't be validated and the download restarts from scratch.
            etag = head_response.headers.get("ETag")
            total_bytes = int(head_response.headers["Content-Length"])

            start_byte = 0
            if (
                resume
                and etag
                and os.path.exists(part_path)
                and os.path.exists(etag_path)
            ):
                with open(etag_path) as f:
                    stored_etag = f.read()
                part_size = os.path.getsize(part_path)
                if stored_etag == etag and part_size <= total_bytes:
                    start_byte = part_size

            if start_byte == 0 and os.path.exists(part_path):
                # Discard a stale partial before stamping the new ETag so a
                # failed download can't later resume-append bytes from a
                # different artifact onto it.
                os.remove(part_path)

            if etag:
                with open(etag_path, "w") as f:
                    f.write(etag)
            elif os.path.exists(etag_path):
                os.remove(etag_path)

            while start_byte < total_bytes:
                headers = None
                if start_byte > 0:
                    # If-Range makes the server return the full body instead
                    # of the requested range if the artifact was replaced
                    # after the HEAD check above.
                    headers = {"Range": f"bytes={start_byte}-", "If-Range": etag}
                with requests.get(
                    payload["urls"]["get"],
                    headers=headers,
                    stream=True,
                    timeout=DOWNLOAD_REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    if start_byte > 0 and response.status_code != 206:
                        # If-Range mismatch or ignored Range: this response is
                        # the full body of the current artifact, so stream it
                        # from scratch instead of appending.
                        start_byte = 0
                    elif (
                        start_byte > 0
                        and response.headers.get("ETag", etag) != etag
                    ):
                        # A 206 for a replaced object; re-request in full.
                        start_byte = 0
                        continue
                    if start_byte == 0:
                        # The response is authoritative for the object being
                        # streamed: after a restart, validate against the
                        # replacement's metadata rather than the stale HEAD.
                        etag = response.headers.get("ETag", etag)
                        if "Content-Length" in response.headers:
                            total_bytes = int(response.headers["Content-Length"])
                        if etag:
                            with open(etag_path, "w") as f:
                                f.write(etag)
                    with (
                        open(part_path, "ab" if start_byte > 0 else "wb") as f,
                        tqdm(
                            total=total_bytes,
                            initial=start_byte,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=to_colored_text(f"Downloading {filename}"),
                            colour=BASE_OUTPUT_COLOR,
                        ) as pbar,
                    ):
                        for chunk in response.iter_content(
                            chunk_size=8 * 1024 * 1024
                        ):
                            f.write(chunk)
                            pbar.update(len(chunk))
                break
        except requests.HTTPError as e:
            print(
                to_colored_text(
                    f"Download failed with status code: {e.response.status_code}",
                    state="fail",
                )
            )
            return None
        except requests.RequestException as e:
            # Mid-stream failures (connection resets, read timeouts) leave the
            # .part file in place; rerunning resumes from where it left off.
            # Only the exception type is printed: str(e) embeds the request
            # URL, and presigned query params grant access to the results.
            print(
                to_colored_text(
                    f"Download interrupted ({type(e).__name__}); rerun to resume.",
                    state="fail",
                )
            )
            return None

        # A stream can end cleanly but short (e.g. an ETag-less gateway
        # serving a replaced object); never publish a truncated file.
        part_size = os.path.getsize(part_path)
        if part_size != total_bytes:
            print(
                to_colored_text(
                    f"Download incomplete ({part_size} of {total_bytes} bytes); "
                    "rerun to retry.",
                    state="fail",
                )
            )
            return None

        os.replace(part_path, destination)
        if etag:
            os.remove(etag_path)
        print(
            to_colored_text(f"✔ Results downloaded to {destination}", state="success")
        )
        return destination

    def _job_results_cache_file_path(self, job_id: str) -> str:
        """Return a cache path scoped to this credential pair and job."""
        if self.api_url is None:
            if self._api_url_error is not None:
                raise SutroConfigurationError(self._api_url_error)
            raise SutroConfigurationError(
                "Sutro API URL is not configured. Set SUTRO_API_URL to your "
                "Sutro deployment URL (for example, "
                "https://sutro.example.com)."
            )
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise SutroConfigurationError(
                "Sutro API key is not configured. Create a key in your Sutro "
                "deployment's API Keys panel and set SUTRO_API_KEY."
            )
        credential_digest = hashlib.sha256(
            f"{self.api_url}\0{self.api_key}".encode("utf-8")
        ).hexdigest()
        job_digest = hashlib.sha256(str(job_id).encode("utf-8")).hexdigest()
        return os.path.expanduser(
            "~/.sutro/job-results/"
            f"{credential_digest}-{job_digest}.snappy.parquet"
        )

    def _remove_legacy_job_results_cache_file(self, job_id: str) -> None:
        """Delete an old unscoped cache entry without ever reusing its data.

        Legacy cache files contain no deployment or credential provenance, so
        migrating their contents into the scoped cache could disclose another
        deployment's results. Remove only the exact, top-level filename for
        this job; unsafe job IDs are ignored rather than interpreted as paths.
        """
        job_id_text = str(job_id)
        legacy_filename = f"{job_id_text}.snappy.parquet"
        if (
            "\0" in legacy_filename
            or os.path.basename(legacy_filename) != legacy_filename
        ):
            return

        legacy_path = os.path.join(
            os.path.expanduser("~/.sutro/job-results"), legacy_filename
        )
        try:
            os.unlink(legacy_path)
        except (FileNotFoundError, IsADirectoryError):
            return
        except OSError:
            warnings.warn(
                "Unable to remove a legacy unscoped job-results cache file. "
                "Run `sutro cache clear` to remove old cache artifacts.",
                RuntimeWarning,
                stacklevel=2,
            )

    def cancel_job(self, job_id: str):
        """
        Cancel a job by its ID.

        This method allows you to cancel a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to cancel.

        Returns:
            dict: The status of the job.
        """
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Cancelling job: {job_id}"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            try:
                response = self.do_request("GET", f"job-cancel/{job_id}")
                spinner.write(to_colored_text("✔ Job cancelled", state="success"))
                return response.json()
            except requests.HTTPError as e:
                spinner.write(to_colored_text("Failed to cancel job", state="fail"))
                spinner.stop()
                print(to_colored_text(e.response.json(), state="fail"))
                return None

    def try_authentication(self, api_key: str):
        """
        Validate an API key with the configured Sutro deployment.

        This method allows you to authenticate with the API key.

        Args:
            api_key (str): The API key to authenticate with.

        Returns:
            dict: The status of the authentication.
        """
        with yaspin(
            SPINNER, text=to_colored_text("Checking API key"), color=BASE_OUTPUT_COLOR
        ) as spinner:
            try:
                response = self.do_request(
                    "GET", "auth/check", api_key_override=api_key
                )

                spinner.write(to_colored_text("✔"))
                return response.json()
            except requests.HTTPError as e:
                spinner.write(
                    to_colored_text(
                        f"API key failed to authenticate: {e.response.status_code}",
                        state="fail",
                    )
                )
                return None

    def get_quotas(self):
        with yaspin(
            SPINNER, text=to_colored_text("Fetching quotas"), color=BASE_OUTPUT_COLOR
        ) as spinner:
            try:
                response = self.do_request("GET", "get-quotas")
                return response.json()["quotas"]
            except requests.HTTPError as e:
                spinner.fail(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                print(to_colored_text(f"Error: {e.response.json()}", state="fail"))
                return None

    def await_job_completion(
        self,
        job_id: str,
        timeout: Optional[int] = 7200,
        obtain_results: bool = True,
        output_column: str = "inference_result",
        is_cost_estimate: bool = False,
    ) -> pl.DataFrame | None:
        """
        Waits for job completion to occur and then returns the results upon
        a successful completion.

        Prints out the job's status every 5 seconds.

        Args:
            job_id (str): The ID of the job to await.
            timeout (Optional[int]): The max time in seconds the function should wait for job results for. Default is 7200 (2 hours).

        Returns:
            pl.DataFrame: The results of the job in a polars DataFrame.
        """
        POLL_INTERVAL = 5

        results: pl.DataFrame | None = None
        start_time = time.time()
        with yaspin(
            SPINNER,
            text=to_colored_text("Awaiting job completion"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            # TODO: Restore a deployment-local batch UI link when Sutro
            # deployments expose one.
            while (time.time() - start_time) < timeout:
                try:
                    status = self._fetch_job_status(job_id)
                except requests.HTTPError as e:
                    spinner.write(
                        to_colored_text(
                            f"Bad status code: {e.response.status_code}", state="fail"
                        )
                    )
                    spinner.stop()
                    print(to_colored_text(e.response.json(), state="fail"))
                    return None

                spinner.text = to_colored_text(f"Job status is {status} for {job_id}")

                if status == JobStatus.SUCCEEDED:
                    spinner.stop()  # Stop this spinner as `get_job_results` has its own spinner text
                    if obtain_results:
                        spinner.write(
                            to_colored_text(
                                "Job completed! Retrieving results...", "success"
                            )
                        )
                        results = self.get_job_results(
                            job_id, output_column=output_column
                        )
                    break
                if status == JobStatus.FAILED:
                    spinner.write(to_colored_text("Job has failed", "fail"))
                    return None
                if status == JobStatus.CANCELLED:
                    spinner.write(to_colored_text("Job has been cancelled"))
                    return None

                time.sleep(POLL_INTERVAL)

        return results

    def _clear_job_results_cache(self):  # only to be called by the CLI
        """
        Clears the cache for a job results.
        """
        if os.path.exists(os.path.expanduser("~/.sutro/job-results")):
            shutil.rmtree(os.path.expanduser("~/.sutro/job-results"))

    def _show_cache_contents(self):
        """
        Shows the contents and size of each file in the job results cache.
        """
        # get the size of the job-results directory
        with yaspin(
            SPINNER,
            text=to_colored_text("Retrieving job results cache contents"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            if not os.path.exists(os.path.expanduser("~/.sutro/job-results")):
                spinner.write(to_colored_text("No job results cache found", "success"))
                return
            total_size = 0
            for file in os.listdir(os.path.expanduser("~/.sutro/job-results")):
                size = (
                    os.path.getsize(os.path.expanduser(f"~/.sutro/job-results/{file}"))
                    / 1024
                    / 1024
                    / 1024
                )
                total_size += size
                spinner.write(to_colored_text(f"File: {file} - Size: {size} GB"))
            spinner.write(
                to_colored_text(
                    f"Total size of results cache at ~/.sutro/job-results: {total_size} GB",
                    "success",
                )
            )

    def _await_job_start(self, job_id: str, timeout: Optional[int] = 7200):
        """
        Waits for job start to occur and then returns the results upon
        a successful start.

        """
        POLL_INTERVAL = 5

        start_time = time.time()
        with yaspin(
            SPINNER,
            text=to_colored_text("Awaiting job completion"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            while (time.time() - start_time) < timeout:
                try:
                    status = self._fetch_job_status(job_id)
                except requests.HTTPError as e:
                    spinner.write(
                        to_colored_text(
                            f"Bad status code: {e.response.status_code}", state="fail"
                        )
                    )
                    spinner.stop()
                    print(to_colored_text(e.response.json(), state="fail"))
                    return None

                spinner.text = to_colored_text(f"Job status is {status} for {job_id}")

                if status == JobStatus.RUNNING or status == JobStatus.STARTING:
                    return True
                if status == JobStatus.FAILED:
                    return False
                if status == JobStatus.CANCELLED:
                    return False

                time.sleep(POLL_INTERVAL)

        return False
