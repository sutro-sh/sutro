import requests
import pandas as pd
import polars as pl
import json
from typing import Union, List, Optional, Dict, Any, Type
import os
from yaspin import yaspin
from yaspin.spinners import Spinners
from colorama import init
import time
from pydantic import BaseModel
import pyarrow.parquet as pq
import shutil
from sutro.common import (
    ModelOptions,
    handle_data_helper,
    normalize_output_schema,
    to_colored_text,
    fancy_tqdm,
    make_clickable_link,
    BASE_OUTPUT_COLOR,
)
from sutro.interfaces import JobStatus
from sutro.templates.classification import ClassificationTemplates
from sutro.templates.embed import EmbeddingTemplates
from sutro.templates.evals import EvalTemplates
from sutro.validation import check_version, check_for_api_key

JOB_NAME_CHAR_LIMIT = 45
JOB_DESCRIPTION_CHAR_LIMIT = 512

# Initialize colorama (required for Windows)
init()


SPINNER = Spinners.dots14


# Isn't fully support in all terminals unfortunately. We should switch to Rich
# at some point, but even Rich links aren't clickable on MacOS Terminal


class Sutro(EmbeddingTemplates, ClassificationTemplates, EvalTemplates):
    def __init__(self, api_key: str = None, base_url: str = "https://api.sutro.sh/", serving_base_url: str = "https://serve.sutro.sh/"):
        self.api_key = api_key or check_for_api_key()
        self.base_url = base_url
        self.serving_base_url = serving_base_url
        check_version("sutro")

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

    def set_base_url(self, base_url: str):
        """
        Set the base URL for the Sutro API.

        This method allows you to set the base URL for the Sutro API.
        The base URL is used to authenticate requests to the API.

        Args:
            base_url (str): The base URL to set.
        """
        self.base_url = base_url

    def set_serving_base_url(self, serving_base_url: str):
        """
        Set the serving base URL for the Sutro API.

        This method allows you to set the serving base URL for the Sutro API.
        The serving base URL is used for function execution requests.

        Args:
            serving_base_url (str): The serving base URL to set.
        """
        self.serving_base_url = serving_base_url

    def do_request(
        self,
        method: str,
        endpoint: str,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Helper to make authenticated requests.
        """
        key = self.api_key if not api_key_override else api_key_override
        headers = {"Authorization": f"Key {key}"}

        # Merge with any headers passed in kwargs
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        base_url = base_url_override if base_url_override else self.base_url
        url = f"{base_url}/{endpoint.lstrip('/')}"

        # Explicit method dispatch
        method = method.upper()
        if method == "GET":
            response = requests.get(url, headers=headers, **kwargs)
        elif method == "POST":
            response = requests.post(url, headers=headers, **kwargs)
        elif method == "PUT":
            response = requests.put(url, headers=headers, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, **kwargs)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response

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

        input_data = handle_data_helper(data, column)
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
                    response = self.do_request("POST", "batch-inference", json=payload)
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
                            clickable_link = make_clickable_link(
                                f"https://app.sutro.sh/jobs/{job_id}"
                            )
                            spinner.write(
                                to_colored_text(
                                    f"Use `so.get_job_status('{job_id}')` to check the status of the job, or monitor progress at {clickable_link}"
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
            clickable_link = make_clickable_link(f"https://app.sutro.sh/jobs/{job_id}")
            spinner.write(
                to_colored_text(f"Progress can also be monitored at: {clickable_link}")
            )
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
        model: ModelOptions = "gemma-3-12b-it",
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
    ):
        """
        Run inference on the provided data.

        This method allows you to run inference on the provided data using the Sutro API.
        It supports various data types such as lists, DataFrames (Polars or Pandas), file paths and datasets.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            model (ModelOptions, optional): The model to use for inference. Defaults to "gemma-3-12b-it".
            name (str, optional): A job name for experiment/metadata tracking purposes. Defaults to None.
            description (str, optional): A job description for experiment/metadata tracking purposes. Defaults to None.
            column (Union[str, List[str]], optional): The column name to use for inference. Required if data is a DataFrame, file path, or dataset. If a list is supplied, it will concatenate the columns of the list into a single column, accepting separator strings.
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
        )

    def run_function(self, model_id: str, input_data: Union[dict, BaseModel]):
        """
        Run inference using the /functions/run endpoint for immediate model execution.
        
        Args:
            model_id (str): The model name to use (e.g., "clay-bert", "clay-judge")
            input_data (Union[dict, BaseModel]): The input data to send to the model.
                Can be a dictionary or a Pydantic model instance
        
        Returns:
            dict: Standardized response with structure:
                {
                    "response": str,        # The predicted class/label
                    "confidence": float,    # Confidence score (0.0-1.0)
                    "predictions": [        # All predictions sorted by confidence
                        {"label": str, "confidence": float},
                        ...
                    ]
                }
        """
        # Convert Pydantic model to dict if needed
        if isinstance(input_data, BaseModel):
            input_data = input_data.model_dump()
        
        payload = {
            "model_id": model_id,
            "input_data": input_data
        }
        
        try:
            response = self.do_request("POST", "functions/run", base_url_override=self.serving_base_url, json=payload)
            return response.json()
        except requests.HTTPError as e:
            print(to_colored_text(f"Error: {e.response.status_code}", state="fail"))
            try:
                error_response = e.response.json()
                print(to_colored_text(error_response, state="fail"))
            except (ValueError, requests.exceptions.JSONDecodeError):
                print(to_colored_text(f"Response body: {e.response.text}", state="fail"))
            return None

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
    ):
        """
        Run inference on the provided data, across multiple models. This method is often useful to sampling outputs from multiple models across the same dataset and compare the job_ids.

        For input data, it supports various data types such as lists, DataFrames (Polars or Pandas), file paths and datasets.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            models (Union[ModelOptions, List[ModelOptions]], optional): The models to use for inference. Fans out each model to its own seperate job, over the same dataset.
            names (Union[str, List[str]], optional): A job name for experiment/metadata tracking purposes. If using a list of models, you must pass a list of names with length equal to the number of models, or None. Defaults to None.
            descriptions (Union[str, List[str]], optional): A job description for experiment/metadata tracking purposes. If using a list of models, you must pass a list of descriptions with length equal to the number of models, or None. Defaults to None.
            column (Union[str, List[str]], optional): The column name to use for inference. Required if data is a DataFrame, file path, or dataset. If a list is supplied, it will concatenate the columns of the list into a single column, accepting separator strings.
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
                clickable_link = make_clickable_link(
                    f"https://app.sutro.sh/jobs/{job_id}"
                )
                spinner.write(
                    to_colored_text(
                        f"Progress can also be monitored at: {clickable_link}"
                    )
                )
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

        file_path = os.path.expanduser(f"~/.sutro/job-results/{job_id}.snappy.parquet")
        expected_num_columns = 1 + include_inputs + include_cumulative_logprobs
        contains_expected_columns = False
        if os.path.exists(file_path):
            num_columns = pq.read_table(file_path).num_columns
            contains_expected_columns = num_columns == expected_num_columns

        if disable_cache == False and contains_expected_columns:
            with yaspin(
                SPINNER,
                text=to_colored_text(f"Loading results from cache: {file_path}"),
                color=BASE_OUTPUT_COLOR,
            ) as spinner:
                results_df = pl.read_parquet(file_path)
                spinner.write(
                    to_colored_text("✔ Results loaded from cache", state="success")
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

            results_df = pl.DataFrame(response_data["results"])

            results_df = results_df.rename({"outputs": output_column})

            if not disable_cache:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                results_df.write_parquet(file_path, compression="snappy")
                spinner.write(
                    to_colored_text("✔ Results saved to cache", state="success")
                )

        # Ordering inputs col first seems most logical/useful
        column_config = [
            ("inputs", include_inputs),
            (output_column, True),
            ("cumulative_logprobs", include_cumulative_logprobs),
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
                results_df = results_df.map_columns(
                    output_column, lambda s: s.str.json_decode()
                )
                results_df = results_df.with_columns(
                    pl.col(output_column).alias("output_column_json_decoded")
                )
                json_decoded_fields = first_row.keys()
                for field in json_decoded_fields:
                    results_df = results_df.with_columns(
                        pl.col("output_column_json_decoded")
                        .struct.field(field)
                        .alias(field)
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
                results_df = results_df.drop(
                    [output_column, "output_column_json_decoded"]
                )
            except Exception:
                # if the first row cannot be json decoded, do nothing
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

    def create_dataset(self):
        """
        Create a new dataset.

        This method creates a new empty dataset and returns its ID.

        Returns:
            str: The ID of the new dataset.
        """
        with yaspin(
            SPINNER, text=to_colored_text("Creating dataset"), color=BASE_OUTPUT_COLOR
        ) as spinner:
            try:
                response = self.do_request("GET", "create-dataset")
                dataset_id = response.json()["dataset_id"]
                spinner.write(
                    to_colored_text(
                        f"✔ Dataset created with ID: {dataset_id}", state="success"
                    )
                )
                return dataset_id
            except requests.HTTPError as e:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(e.response.json(), state="fail"))
                return None

    def upload_to_dataset(
        self,
        dataset_id: Union[List[str], str] = None,
        file_paths: Union[List[str], str] = None,
        verify_ssl: bool = True,
    ):
        """
        Upload data to a dataset.

        This method uploads files to a dataset. Accepts a dataset ID and file paths. If only a single parameter is provided, it will be interpreted as the file paths.

        Args:
            dataset_id (str): The ID of the dataset to upload to. If not provided, a new dataset will be created.
            file_paths (Union[List[str], str]): A list of paths to the files to upload, or a single path to a collection of files.
            verify_ssl (bool): Whether to verify SSL certificates. Set to False to bypass SSL verification for troubleshooting.

        Returns:
            dict: The response from the API.
        """
        # when only a single parameter is provided, it is interpreted as the file paths
        if file_paths is None and dataset_id is not None:
            file_paths = dataset_id
            dataset_id = None

        if file_paths is None:
            raise ValueError("File paths must be provided")

        if dataset_id is None:
            dataset_id = self.create_dataset()

        if isinstance(file_paths, str):
            # check if the file path is a directory
            if os.path.isdir(file_paths):
                file_paths = [
                    os.path.join(file_paths, f) for f in os.listdir(file_paths)
                ]
                if len(file_paths) == 0:
                    raise ValueError("No files found in the directory")
            else:
                file_paths = [file_paths]

        with yaspin(
            SPINNER,
            text=to_colored_text(f"Uploading files to dataset: {dataset_id}"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            count = 0
            for file_path in file_paths:
                file_name = os.path.basename(file_path)

                files = {
                    "file": (
                        file_name,
                        open(file_path, "rb"),
                        "application/octet-stream",
                    )
                }

                payload = {
                    "dataset_id": dataset_id,
                }

                count += 1
                spinner.write(
                    to_colored_text(
                        f"Uploading file {count}/{len(file_paths)} to dataset: {dataset_id}"
                    )
                )

                try:
                    self.do_request(
                        "POST",
                        "/upload-to-dataset",
                        data=payload,
                        files=files,
                        verify=verify_ssl,
                    )
                except requests.exceptions.RequestException as e:
                    # Stop spinner before showing error to avoid terminal width error
                    spinner.stop()
                    print(to_colored_text(f"Upload failed: {str(e)}", state="fail"))
                    return None

            spinner.write(
                to_colored_text(
                    f"✔ {count} files successfully uploaded to dataset", state="success"
                )
            )
        return dataset_id

    def list_datasets(self):
        with yaspin(
            SPINNER,
            text=to_colored_text("Retrieving datasets"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            try:
                response = self.do_request("POST", "list-datasets")
                spinner.write(to_colored_text("✔ Datasets retrieved", state="success"))
                return response.json()["datasets"]
            except requests.HTTPError as e:
                spinner.fail(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                print(to_colored_text(f"Error: {e.response.json()}", state="fail"))
                return None

    def list_dataset_files(self, dataset_id: str):
        payload = {
            "dataset_id": dataset_id,
        }
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Listing files in dataset: {dataset_id}"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            try:
                response = self.do_request("POST", "list-dataset-files", json=payload)
                spinner.write(
                    to_colored_text(
                        f"✔ Files listed in dataset: {dataset_id}", state="success"
                    )
                )
                return response.json()["files"]
            except requests.HTTPError as e:
                spinner.fail(
                    to_colored_text(
                        f"Bad status code: {e.response.status_code}", state="fail"
                    )
                )
                print(to_colored_text(f"Error: {e.response.json()}", state="fail"))
                return None

    def download_from_dataset(
        self,
        dataset_id: str,
        files: Union[List[str], str] = None,
        output_path: str = None,
    ):
        if files is None:
            files = self.list_dataset_files(dataset_id)
        elif isinstance(files, str):
            files = [files]

        if not files:
            print(
                to_colored_text(
                    f"Couldn't find files for dataset ID: {dataset_id}", state="fail"
                )
            )
            return

        # if no output path is provided, save the files to the current working directory
        if output_path is None:
            output_path = os.getcwd()

        with yaspin(
            SPINNER,
            text=to_colored_text(f"Downloading files from dataset: {dataset_id}"),
            color=BASE_OUTPUT_COLOR,
        ) as spinner:
            count = 0
            for file in files:
                spinner.text = to_colored_text(
                    f"Downloading file {count + 1}/{len(files)} from dataset: {dataset_id}"
                )

                try:
                    payload = {
                        "dataset_id": dataset_id,
                        "file_name": file,
                    }
                    response = self.do_request(
                        "POST", "download-from-dataset", json=payload
                    )

                    file_content = response.content
                    with open(os.path.join(output_path, file), "wb") as f:
                        f.write(file_content)

                    count += 1
                except requests.HTTPError as e:
                    spinner.fail(
                        to_colored_text(
                            f"Bad status code: {e.response.status_code}", state="fail"
                        )
                    )
                    print(to_colored_text(f"Error: {e.response.json()}", state="fail"))
                    return
            spinner.write(
                to_colored_text(
                    f"✔ {count} files successfully downloaded from dataset: {dataset_id}",
                    state="success",
                )
            )

    def try_authentication(self, api_key: str):
        """
        Try to authenticate with the API key.

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
                response = self.do_request("GET", "try-authentication", api_key)

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
            if not is_cost_estimate:
                clickable_link = make_clickable_link(
                    f"https://app.sutro.sh/jobs/{job_id}"
                )
                spinner.write(
                    to_colored_text(
                        f"Progress can also be monitored at: {clickable_link}"
                    )
                )
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
