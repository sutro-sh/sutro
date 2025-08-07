import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum

import requests
import pandas as pd
import polars as pl
import json
from typing import Union, List, Optional, Literal, Generator, Dict, Any
import os
import sys
from yaspin import yaspin
from yaspin.spinners import Spinners
from colorama import init, Fore, Back, Style
from tqdm import tqdm
import time
from pydantic import BaseModel
import json


class JobStatus(str, Enum):
    """Job statuses that will be returned by the API & SDK"""

    UNKNOWN = "UNKNOWN"
    QUEUED = "QUEUED"  # Job is waiting to start
    STARTING = "STARTING"  # Job is in the process of starting up
    RUNNING = "RUNNING"  # Job is actively running
    SUCCEEDED = "SUCCEEDED"  # Job completed successfully
    CANCELLING = "CANCELLING"  # Job is in the process of being canceled
    CANCELLED = "CANCELLED"  # Job was canceled by the user
    FAILED = "FAILED"  # Job failed

    @classmethod
    def terminal_statuses(cls) -> list["JobStatus"]:
        return [
            cls.SUCCEEDED,
            cls.FAILED,
            cls.CANCELLING,
            cls.CANCELLED,
        ]

    def is_terminal(self) -> bool:
        return self in self.terminal_statuses()

# Initialize colorama (required for Windows)
init()


# This is how yaspin defines is_jupyter logic
def is_jupyter() -> bool:
    return not sys.stdout.isatty()


# `color` param not supported in Jupyter notebooks
YASPIN_COLOR = None if is_jupyter() else "blue"
SPINNER = Spinners.dots14

# Models available for inference.  Keep in sync with the backend configuration
# so users get helpful autocompletion when selecting a model.
ModelOptions = Literal[
    "llama-3.2-3b",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "llama-3.3-70b",
    "qwen-3-4b",
    "qwen-3-32b",
    "qwen-3-4b-thinking",
    "qwen-3-32b-thinking",
    "gemma-3-4b-it",
    "gemma-3-27b-it",
    "multilingual-e5-large-instruct",
    "gte-qwen2-7b-instruct",
]


def to_colored_text(
    text: str, state: Optional[Literal["success", "fail"]] = None
) -> str:
    """
    Apply color to text based on state.

    Args:
        text (str): The text to color
        state (Optional[Literal['success', 'fail']]): The state that determines the color.
            Options: 'success', 'fail', or None (default blue)

    Returns:
        str: Text with appropriate color applied
    """
    match state:
        case "success":
            return f"{Fore.GREEN}{text}{Style.RESET_ALL}"
        case "fail":
            return f"{Fore.RED}{text}{Style.RESET_ALL}"
        case _:
            # Default to blue for normal/processing states
            return f"{Fore.BLUE}{text}{Style.RESET_ALL}"

# Isn't fully support in all terminals unfortunately. We should switch to Rich
# at some point, but even Rich links aren't clickable on MacOS Terminal
def make_clickable_link(url, text=None):
    """
    Create a clickable link for terminals that support OSC 8 hyperlinks.
    Falls back to plain text for terminals that don't support it.
    """
    if text is None:
        text = url
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"

class Sutro:
    def __init__(
        self, api_key: str = None, base_url: str = "https://api.sutro.sh/"
    ):
        self.api_key = api_key or self.check_for_api_key()
        self.base_url = base_url

    def check_for_api_key(self):
        """
        Check for an API key in the user's home directory.

        This method looks for a configuration file named 'config.json' in the
        '.sutro' directory within the user's home directory.
        If the file exists, it attempts to read the API key from it.

        Returns:
            str or None: The API key if found in the configuration file, or None if not found.

        Note:
            The expected structure of the config.json file is:
            {
                "api_key": "your_api_key_here"
            }
        """
        CONFIG_DIR = os.path.expanduser("~/.sutro")
        CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            return config.get("api_key")
        else:
            return None

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

    def handle_data_helper(
        self, data: Union[List, pd.DataFrame, pl.DataFrame, str], column: str = None
    ):
        if isinstance(data, list):
            input_data = data
        elif isinstance(data, (pd.DataFrame, pl.DataFrame)):
            if column is None:
                raise ValueError("Column name must be specified for DataFrame input")
            input_data = data[column].to_list()
        elif isinstance(data, str):
            if data.startswith("dataset-"):
                input_data = data + ":" + column
            else:
                file_ext = os.path.splitext(data)[1].lower()
                if file_ext == ".csv":
                    df = pl.read_csv(data)
                elif file_ext == ".parquet":
                    df = pl.read_parquet(data)
                elif file_ext in [".txt", ""]:
                    with open(data, "r") as file:
                        input_data = [line.strip() for line in file]
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")

                if file_ext in [".csv", ".parquet"]:
                    if column is None:
                        raise ValueError(
                            "Column name must be specified for CSV/Parquet input"
                        )
                    input_data = df[column].to_list()
        else:
            raise ValueError(
                "Unsupported data type. Please provide a list, DataFrame, or file path."
            )

        return input_data

    def set_base_url(self, base_url: str):
        """
        Set the base URL for the Sutro API.

        This method allows you to set the base URL for the Sutro API.
        The base URL is used to authenticate requests to the API.

        Args:
            base_url (str): The base URL to set.
        """
        self.base_url = base_url

    def _run_one_batch_inference(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: ModelOptions,
        column: str,
        output_column: str,
        job_priority: int,
        json_schema: Dict[str, Any],
        sampling_params: dict,
        system_prompt: str,
        cost_estimate: bool,
        stay_attached: Optional[bool],
        random_seed_per_input: bool,
        truncate_rows: bool
    ):
        input_data = self.handle_data_helper(data, column)
        endpoint = f"{self.base_url}/batch-inference"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "inputs": input_data,
            "job_priority": job_priority,
            "json_schema": json_schema,
            "system_prompt": system_prompt,
            "cost_estimate": cost_estimate,
            "sampling_params": sampling_params,
            "random_seed_per_input": random_seed_per_input,
            "truncate_rows": truncate_rows
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
            with yaspin(SPINNER, text=spinner_text, color=YASPIN_COLOR) as spinner:
                response = requests.post(
                    endpoint, data=json.dumps(payload), headers=headers
                )
                response_data = response.json()
                if response.status_code != 200:
                    spinner.write(
                        to_colored_text(f"Error: {response.status_code}", state="fail")
                    )
                    spinner.stop()
                    print(to_colored_text(response.json(), state="fail"))
                    return None
                else:
                    job_id = response_data["results"]
                    if cost_estimate:
                        spinner.write(
                            to_colored_text(f"Awaiting cost estimates with job ID: {job_id}. You can safely detach and retrieve the cost estimates later.")
                        )
                        spinner.stop()
                        self.await_job_completion(job_id, obtain_results=False, is_cost_estimate=True)
                        cost_estimate = self._get_job_cost_estimate(job_id)
                        spinner.write(
                            to_colored_text(f"✔ Cost estimates retrieved for job {job_id}: ${cost_estimate}", state="success")
                        )
                        return job_id
                    else:
                        spinner.write(
                            to_colored_text(
                                f"🛠 Priority {job_priority} Job created with ID: {job_id}.",
                                state="success",
                            )
                        )
                        if not stay_attached:
                            clickable_link = make_clickable_link(f'https://app.sutro.sh/jobs/{job_id}')
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
            spinner.write(to_colored_text("Awaiting job start...", ))
            clickable_link = make_clickable_link(f'https://app.sutro.sh/jobs/{job_id}')
            spinner.write(to_colored_text(f'Progress can also be monitored at: {clickable_link}'))
            started = self._await_job_start(job_id)
            if not started:
                failure_reason = self._get_failure_reason(job_id)
                spinner.write(to_colored_text(f"Failure reason: {failure_reason['message']}", "fail"))
                return None
            s = requests.Session()
            pbar = None

            try:
                with requests.get(
                        f"{self.base_url}/stream-job-progress/{job_id}",
                        headers=headers,
                        stream=True,
                ) as streaming_response:
                    streaming_response.raise_for_status()
                    spinner = yaspin(
                        SPINNER,
                        text=to_colored_text("Awaiting status updates..."),
                        color=YASPIN_COLOR,
                    )
                    spinner.start()

                    token_state = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                            'total_tokens_processed_per_second': 0
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
                                    postfix = f"Input tokens processed: 0"
                                    pbar = self.fancy_tqdm(
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
                                    k: v for k, v in json_obj.get('result', {}).items()
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

                payload = {
                    "job_id": job_id,
                }

                # TODO: we implment retries in cases where the job hasn't written results yet
                # it would be better if we could receive a fully succeeded status from the job
                # and not have such a race condition
                max_retries = 20 # winds up being 100 seconds cumulative delay
                retry_delay = 5 # initial delay in seconds

                for _ in range(max_retries):
                    time.sleep(retry_delay)

                    job_results_response = s.post(
                        f"{self.base_url}/job-results",
                        headers=headers,
                        data=json.dumps(payload),
                    )
                    if job_results_response.status_code == 200:
                        break

                if job_results_response.status_code != 200:
                    spinner.write(
                        to_colored_text(
                            "Job succeeded, but results are not yet available. Use `so.get_job_results('{job_id}')` to obtain results.",
                            state="fail",
                        )
                    )
                    spinner.stop()
                    return None

                results = job_results_response.json()["results"]["outputs"]

                spinner.write(
                    to_colored_text(
                        f"✔ Job results received. You can re-obtain the results with `so.get_job_results('{job_id}')`",
                        state="success",
                    )
                )
                spinner.stop()

                if isinstance(data, (pd.DataFrame, pl.DataFrame)):
                    if isinstance(data, pd.DataFrame):
                        data[output_column] = results
                    elif isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.Series(output_column, results))
                    return data

                return results
            return None
        return None

    def infer(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: Union[ModelOptions, List[ModelOptions]] = "llama-3.1-8b",
        column: str = None,
        output_column: str = "inference_result",
        job_priority: int = 0,
        output_schema: Union[Dict[str, Any], BaseModel] = None,
        sampling_params: dict = None,
        system_prompt: str = None,
        dry_run: bool = False,
        stay_attached: Optional[bool] = None,
        random_seed_per_input: bool = False,
        truncate_rows: bool = False
    ):
        """
        Run inference on the provided data.

        This method allows you to run inference on the provided data using the Sutro API.
        It supports various data types such as lists, pandas DataFrames, polars DataFrames, file paths and datasets.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            model (Union[ModelOptions, List[ModelOptions]], optional): The model(s) to use for inference. Defaults to "llama-3.1-8b". You can pass a single model or a list of models. In the case of a list, the inference will be run in parallel for each model and stay_attached will be set to False.
            column (str, optional): The column name to use for inference. Required if data is a DataFrame, file path, or dataset.
            output_column (str, optional): The column name to store the inference results in if the input is a DataFrame. Defaults to "inference_result".
            job_priority (int, optional): The priority of the job. Defaults to 0.
            output_schema (Union[Dict[str, Any], BaseModel], optional): A structured schema for the output.
                Can be either a dictionary representing a JSON schema or a pydantic BaseModel. Defaults to None.
            sampling_params: (dict, optional): The sampling parameters to use at generation time, ie temperature, top_p etc.
            system_prompt (str, optional): A system prompt to add to all inputs. This allows you to define the behavior of the model. Defaults to None.
            dry_run (bool, optional): If True, the method will return cost estimates instead of running inference. Defaults to False.
            stay_attached (bool, optional): If True, the method will stay attached to the job until it is complete. Defaults to True for prototyping jobs, False otherwise.
            random_seed_per_input (bool, optional): If True, the method will use a different random seed for each input. Defaults to False.
            truncate_rows (bool, optional): If True, any rows that have a token count exceeding the context window length of the selected model will be truncated to the max length that will fit within the context window. Defaults to False.

        Returns:
            Union[List, pd.DataFrame, pl.DataFrame, str]: The results of the inference.

        """
        if isinstance(model, list) == False:
            model_list = [model]
            stay_attached = stay_attached if stay_attached is not None else job_priority == 0
        else:
            model_list = model
            stay_attached = False

        # Convert BaseModel to dict if needed
        if output_schema is not None:
            if hasattr(output_schema, 'model_json_schema'):  # Check for pydantic Model interface
                json_schema = output_schema.model_json_schema()
            elif isinstance(output_schema, dict):
                json_schema = output_schema
            else:
                raise ValueError("Invalid output schema type. Must be a dictionary or a pydantic Model.")
        else:
            json_schema = None

        for model in model_list:
            res = self._run_one_batch_inference(
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
                truncate_rows
            )
            if stay_attached:
                return res


    def attach(self, job_id):
        """
        Attach to an existing job and stream its progress.

        Args:
            job_id (str): The ID of the job to attach to
        """

        s = requests.Session()
        payload = {
            "job_id": job_id,
        }
        pbar = None

        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

        with yaspin(
                SPINNER,
                text=to_colored_text("Looking for job..."),
                color=YASPIN_COLOR,
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
                    spinner.write(to_colored_text("❌ Job is in failed state.", state="fail"))
                    return
                case "CANCELLED":
                    spinner.write(to_colored_text("❌ Job was cancelled.", state="fail"))
                    return
                case _:
                    spinner.write(to_colored_text("✔ Job found!", state="success"))

        total_rows = job["num_rows"]
        success = False

        try:
            with s.get(
                    f"{self.base_url}/stream-job-progress/{job_id}",
                    headers=headers,
                    stream=True,
            ) as streaming_response:
                streaming_response.raise_for_status()
                spinner = yaspin(
                    SPINNER,
                    text=to_colored_text("Awaiting status updates..."),
                    color=YASPIN_COLOR,
                )
                clickable_link = make_clickable_link(f'https://app.sutro.sh/jobs/{job_id}')
                spinner.write(to_colored_text(f'Progress can also be monitored at: {clickable_link}'))
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
                                postfix = f"Input tokens processed: 0"
                                pbar = self.fancy_tqdm(
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
        color: str = "blue",
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
            list: A list of job details.
        """
        endpoint = f"{self.base_url}/list-jobs"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

        with yaspin(
            SPINNER, text=to_colored_text("Fetching jobs"), color=YASPIN_COLOR
        ) as spinner:
            response = requests.get(endpoint, headers=headers)
            if response.status_code != 200:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(response.json(), state="fail"))
                return
        return response.json()["jobs"]

    def _list_jobs_helper(self):
        """
        Helper function to list jobs.
        """
        endpoint = f"{self.base_url}/list-jobs˚"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(endpoint, headers=headers)
        if response.status_code != 200:
            return None
        return response.json()["jobs"]

    def _fetch_job(self, job_id):
        """
        Helper function to fetch a single job.
        """
        endpoint = f"{self.base_url}/jobs/{job_id}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(endpoint, headers=headers)
        if response.status_code != 200:
            return None
        return response.json().get('job')

    def _get_job_cost_estimate(self, job_id: str):
        """
        Get the cost estimate for a job.
        """
        job = self._fetch_job(job_id)
        if not job:
            return None

        return job.get('cost_estimate')
    
    def _get_failure_reason(self, job_id: str):
        """
        Get the failure reason for a job.
        """
        job = self._fetch_job(job_id)
        if not job:
            return None
        return job.get('failure_reason')

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
        endpoint = f"{self.base_url}/job-status/{job_id}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()

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
                color=YASPIN_COLOR,
        ) as spinner:
            try:
                response_data = self._fetch_job_status(job_id)
                spinner.write(to_colored_text("✔ Job status retrieved!", state="success"))
                return response_data["job_status"][job_id]
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
    ):
        """
        Get the results of a job by its ID.

        This method retrieves the results of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the results for.
            include_inputs (bool, optional): Whether to include the inputs in the results. Defaults to False.
            include_cumulative_logprobs (bool, optional): Whether to include the cumulative logprobs in the results. Defaults to False.
            with_original_df (pd.DataFrame | pl.DataFrame, optional): Original DataFrame to concatenate with results. Defaults to None.
            output_column (str, optional): Name of the output column. Defaults to "inference_result".

        Returns:
            Union[pl.DataFrame, pd.DataFrame]: The results as a DataFrame. By default, returns polars.DataFrame; when with_original_df is an instance of pandas.DataFrame, returns pandas.DataFrame.
        """
        endpoint = f"{self.base_url}/job-results"
        payload = {
            "job_id": job_id,
            "include_inputs": include_inputs,
            "include_cumulative_logprobs": include_cumulative_logprobs,
        }
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
                SPINNER,
                text=to_colored_text(f"Gathering results from job: {job_id}"),
                color=YASPIN_COLOR,
        ) as spinner:
            response = requests.post(
                endpoint, data=json.dumps(payload), headers=headers
            )
            if response.status_code != 200:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(response.json(), state="fail"))
                return None

            spinner.write(
                to_colored_text("✔ Job results retrieved", state="success")
            )

        response_data = response.json()
        results_df = pl.DataFrame(response_data["results"])

        results_df = results_df.rename({'outputs': output_column})

        # Ordering inputs col first seems most logical/useful
        column_config = [
            ('inputs', include_inputs),
            (output_column, True),
            ('cumulative_logprobs', include_cumulative_logprobs),
        ]

        columns_to_keep = [col for col, include in column_config
                           if include and col in results_df.columns]

        results_df = results_df.select(columns_to_keep)

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
        endpoint = f"{self.base_url}/job-cancel/{job_id}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Cancelling job: {job_id}"),
            color=YASPIN_COLOR,
        ) as spinner:
            response = requests.get(endpoint, headers=headers)
            if response.status_code == 200:
                spinner.write(to_colored_text("✔ Job cancelled", state="success"))
            else:
                spinner.write(to_colored_text("Failed to cancel job", state="fail"))
                spinner.stop()
                print(to_colored_text(response.json(), state="fail"))
                return
        return response.json()

    def create_dataset(self):
        """
        Create a new dataset.

        This method creates a new empty dataset and returns its ID.

        Returns:
            str: The ID of the new dataset.
        """
        endpoint = f"{self.base_url}/create-dataset"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER, text=to_colored_text("Creating dataset"), color=YASPIN_COLOR
        ) as spinner:
            response = requests.get(endpoint, headers=headers)
            if response.status_code != 200:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(response.json(), state="fail"))
                return
            dataset_id = response.json()["dataset_id"]
            spinner.write(
                to_colored_text(f"✔ Dataset created with ID: {dataset_id}", state="success")
            )
        return dataset_id

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

        endpoint = f"{self.base_url}/upload-to-dataset"

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
            color=YASPIN_COLOR,
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

                headers = {
                    "Authorization": f"Key {self.api_key}"}

                count += 1
                spinner.write(
                    to_colored_text(
                        f"Uploading file {count}/{len(file_paths)} to dataset: {dataset_id}"
                    )
                )

                try:
                    response = requests.post(
                        endpoint, headers=headers, data=payload, files=files
                    )
                    if response.status_code != 200:
                        # Stop spinner before showing error to avoid terminal width error
                        spinner.stop()
                        print(
                            to_colored_text(
                                f"Error: HTTP {response.status_code}", state="fail"
                            )
                        )
                        print(to_colored_text(response.json(), state="fail"))
                        return

                except requests.exceptions.RequestException as e:
                    # Stop spinner before showing error to avoid terminal width error
                    spinner.stop()
                    print(to_colored_text(f"Upload failed: {str(e)}", state="fail"))
                    return

            spinner.write(
                to_colored_text(
                    f"✔ {count} files successfully uploaded to dataset", state="success"
                )
            )
        return dataset_id

    def list_datasets(self):
        endpoint = f"{self.base_url}/list-datasets"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER, text=to_colored_text("Retrieving datasets"), color=YASPIN_COLOR
        ) as spinner:
            response = requests.post(endpoint, headers=headers)
            if response.status_code != 200:
                spinner.fail(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                print(to_colored_text(f"Error: {response.json()}", state="fail"))
                return
            spinner.write(to_colored_text("✔ Datasets retrieved", state="success"))
        return response.json()["datasets"]

    def list_dataset_files(self, dataset_id: str):
        endpoint = f"{self.base_url}/list-dataset-files"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "dataset_id": dataset_id,
        }
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Listing files in dataset: {dataset_id}"),
            color=YASPIN_COLOR,
        ) as spinner:
            response = requests.post(
                endpoint, headers=headers, data=json.dumps(payload)
            )
            if response.status_code != 200:
                spinner.fail(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                print(to_colored_text(f"Error: {response.json()}", state="fail"))
                return
            spinner.write(
                to_colored_text(f"✔ Files listed in dataset: {dataset_id}", state="success")
            )
        return response.json()["files"]

    def download_from_dataset(
        self,
        dataset_id: str,
        files: Union[List[str], str] = None,
        output_path: str = None,
    ):
        endpoint = f"{self.base_url}/download-from-dataset"

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
            color=YASPIN_COLOR,
        ) as spinner:
            count = 0
            for file in files:
                headers = {
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "dataset_id": dataset_id,
                    "file_name": file,
                }
                spinner.text = to_colored_text(
                    f"Downloading file {count + 1}/{len(files)} from dataset: {dataset_id}"
                )
                response = requests.post(
                    endpoint, headers=headers, data=json.dumps(payload)
                )
                if response.status_code != 200:
                    spinner.fail(
                        to_colored_text(
                            f"Bad status code: {response.status_code}", state="fail"
                        )
                    )
                    print(to_colored_text(f"Error: {response.json()}", state="fail"))
                    return
                file_content = response.content
                with open(os.path.join(output_path, file), "wb") as f:
                    f.write(file_content)
                count += 1
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
        endpoint = f"{self.base_url}/try-authentication"
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER, text=to_colored_text("Checking API key"), color=YASPIN_COLOR
        ) as spinner:
            response = requests.get(endpoint, headers=headers)
            if response.status_code == 200:
                spinner.write(to_colored_text("✔"))
            else:
                spinner.write(
                    to_colored_text(
                        f"API key failed to authenticate: {response.status_code}",
                        state="fail",
                    )
                )
                return
        return response.json()

    def get_quotas(self):
        endpoint = f"{self.base_url}/get-quotas"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER, text=to_colored_text("Fetching quotas"), color=YASPIN_COLOR
        ) as spinner:
            response = requests.get(endpoint, headers=headers)
            if response.status_code != 200:
                spinner.fail(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                print(to_colored_text(f"Error: {response.json()}", state="fail"))
                return
        return response.json()["quotas"]

    def await_job_completion(self, job_id: str, timeout: Optional[int] = 7200, obtain_results: bool = True, is_cost_estimate: bool=False) -> list | None:
        """
        Waits for job completion to occur and then returns the results upon
        a successful completion.

        Prints out the job's status every 5 seconds.

        Args:
            job_id (str): The ID of the job to await.
            timeout (Optional[int]): The max time in seconds the function should wait for job results for. Default is 7200 (2 hours).

        Returns:
            list: The results of the job.
        """
        POLL_INTERVAL = 5

        results = None
        start_time = time.time()
        with yaspin(
            SPINNER, text=to_colored_text("Awaiting job completion"), color=YASPIN_COLOR
        ) as spinner:
            if not is_cost_estimate:
                clickable_link = make_clickable_link(f'https://app.sutro.sh/jobs/{job_id}')
                spinner.write(to_colored_text(f'Progress can also be monitored at: {clickable_link}'))
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
                    spinner.stop() # Stop this spinner as `get_job_results` has its own spinner text
                    if obtain_results:
                        spinner.write(to_colored_text("Job completed! Retrieving results...", "success"))
                        results = self.get_job_results(job_id)
                    break
                if status == JobStatus.FAILED:
                    spinner.write(to_colored_text("Job has failed", "fail"))
                    return None
                if status == JobStatus.CANCELLED:
                    spinner.write(to_colored_text("Job has been cancelled"))
                    return None


                time.sleep(POLL_INTERVAL)

        return results
    
    def _await_job_start(self, job_id: str, timeout: Optional[int] = 7200):
        """
        Waits for job start to occur and then returns the results upon
        a successful start.
        
        """
        POLL_INTERVAL = 5

        start_time = time.time()
        with yaspin(
                SPINNER, text=to_colored_text("Awaiting job completion"), color=YASPIN_COLOR
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
            