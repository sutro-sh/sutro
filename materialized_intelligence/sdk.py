import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import requests
import pandas as pd
import polars as pl
import json
from typing import Union, List, Optional, Literal, Generator
import os
import sys
from yaspin import yaspin
from yaspin.spinners import Spinners
from colorama import init, Fore, Back, Style
from tqdm import tqdm
import time


# Initialize colorama (required for Windows)
init()


# This is how yaspin defines is_jupyter logic
def is_jupyter() -> bool:
    return not sys.stdout.isatty()


# `color` param not supported in Jupyter notebooks
YASPIN_COLOR = None if is_jupyter() else "blue"
SPINNER = Spinners.dots14


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


class MaterializedIntelligence:
    def __init__(
        self, api_key: str = None, base_url: str = "https://api.materialized.dev/"
    ):
        self.api_key = api_key or self.check_for_api_key()
        self.base_url = base_url
        self.HEARTBEAT_INTERVAL_SECONDS = 15  # Keep in sync w what the backend expects

    def check_for_api_key(self):
        """
        Check for an API key in the user's home directory.

        This method looks for a configuration file named 'config.json' in the
        '.materialized_intelligence' directory within the user's home directory.
        If the file exists, it attempts to read the API key from it.

        Returns:
            str or None: The API key if found in the configuration file, or None if not found.

        Note:
            The expected structure of the config.json file is:
            {
                "api_key": "your_api_key_here"
            }
        """
        CONFIG_DIR = os.path.expanduser("~/.materialized_intelligence")
        CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            return config.get("api_key")
        else:
            return None

    def set_api_key(self, api_key: str):
        """
        Set the API key for the Materialized Intelligence API.

        This method allows you to set the API key for the Materialized Intelligence API.
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
            if data.startswith("stage-"):
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
        Set the base URL for the Materialized Intelligence API.

        This method allows you to set the base URL for the Materialized Intelligence API.
        The base URL is used to authenticate requests to the API.

        Args:
            base_url (str): The base URL to set.
        """
        self.base_url = base_url

    def infer(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: str = "llama-3.1-8b",
        column: str = None,
        output_column: str = "inference_result",
        job_priority: int = 0,
        json_schema: dict = None,
        sampling_params: dict = None,
        num_workers: int = 1,
        system_prompt: str = None,
        dry_run: bool = False,
        stay_attached: bool = False,
    ):
        """
        Run inference on the provided data.

        This method allows you to run inference on the provided data using the Materialized Intelligence API.
        It supports various data types such as lists, pandas DataFrames, polars DataFrames, file paths and stages.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            model (str, optional): The model to use for inference. Defaults to "llama-3.1-8b".
            column (str, optional): The column name to use for inference. Required if data is a DataFrame, file path, or stage.
            output_column (str, optional): The column name to store the inference results in if input is a DataFrame. Defaults to "inference_result".
            job_priority (int, optional): The priority of the job. Defaults to 0.
            json_schema (dict, optional): A JSON schema for the output. Defaults to None.
            system_prompt (str, optional): A system prompt to add to all inputs. This allows you to define the behavior of the model. Defaults to None.
            dry_run (bool, optional): If True, the method will return cost estimates instead of running inference. Defaults to False.
            stay_attached (bool, optional): If True, the method will stay attached to the job until it is complete. Defaults to True for prototyping jobs, False otherwise.

        Returns:
            Union[List, pd.DataFrame, pl.DataFrame, str]: The results of the inference.

        """
        input_data = self.handle_data_helper(data, column)
        stay_attached = stay_attached or job_priority == 0

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
            "dry_run": dry_run,
            "sampling_params": sampling_params,
        }
        if dry_run:
            spinner_text = to_colored_text("Retrieving cost estimates...")
        else:
            t = f"Creating priority {job_priority} job"
            spinner_text = to_colored_text(t)

        # There are two gotchas with yaspin:
        # 1. Can't use print while in spinner is running
        # 2. When writing to stdout via spinner.fail, spinner.write etc, there is a pretty strict
        # limit for content length in jupyter notebooks, where it wisll give an error about:
        # Terminal size {self._terminal_width} is too small to display spinner with the given settings.
        # https://github.com/pavdmyt/yaspin/blob/9c7430b499ab4611888ece39783a870e4a05fa45/yaspin/core.py#L568-L571
        job_id = None
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
                return
            else:
                if dry_run:
                    spinner.write(
                        to_colored_text("✔ Cost estimates retrieved", state="success")
                    )
                    return response_data["results"]
                else:
                    job_id = response_data["results"]
                    spinner.write(
                        to_colored_text(
                            f"🛠️  Priority {job_priority} Job created with ID: {job_id}",
                            state="success",
                        )
                    )
                    if not stay_attached:
                        spinner.write(
                            to_colored_text(
                                f"Use `mi.get_job_status('{job_id}')` to check the status of the job."
                            )
                        )
                        return job_id

        success = False
        if stay_attached and job_id is not None:
            s = requests.Session()
            payload = {
                "job_id": job_id,
            }
            pbar = None

            # Register for stream and get session token
            session_token = self.register_stream_listener(job_id)

            # Use the heartbeat session context manager
            with self.stream_heartbeat_session(job_id, session_token) as s:
                with s.get(
                        f"{self.base_url}/stream-job-progress/{job_id}?request_session_token={session_token}",
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
                                    pbar.close()
                                    success = True
                            elif json_obj["update_type"] == "tokens":
                                if pbar is not None:
                                    pbar.postfix = f"Input tokens processed: {json_obj['result']['input_tokens']}, Tokens generated: {json_obj['result']['output_tokens']}, Total tokens/s: {json_obj['result'].get('total_tokens_processed_per_second')}"
                                    pbar.refresh()
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
                            "Job succeeded, but results are not yet available. Use `mi.get_job_results('{job_id}')` to obtain results.",
                            state="fail",
                        )
                    )
                    spinner.stop()
                    return

                results = job_results_response.json()["results"]

                spinner.write(
                    to_colored_text(
                        f"✔ Job results received. You can re-obtain the results with `mi.get_job_results('{job_id}')`",
                        state="success",
                    )
                )
                spinner.stop()

                if isinstance(data, (pd.DataFrame, pl.DataFrame)):
                    sample_n = 1 if sampling_params is None else sampling_params["n"]
                    if sample_n > 1:
                        results = [
                            results[i : i + sample_n]
                            for i in range(0, len(results), sample_n)
                        ]
                    if isinstance(data, pd.DataFrame):
                        data[output_column] = results
                    elif isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.Series(output_column, results))
                    return data

                return results

    def register_stream_listener(self, job_id: str) -> str:
        """Register a new stream listener and get a session token."""
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with requests.post(
                f"{self.base_url}/register-stream-listener/{job_id}",
                headers=headers,
        ) as response:
            response.raise_for_status()
            data = response.json()
            return data["request_session_token"]

    # This is a best effort action and is ok if it sometimes doesn't complete etc
    def unregister_stream_listener(self, job_id: str, session_token: str):
        """Explicitly unregister a stream listener."""
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with requests.post(
                f"{self.base_url}/unregister-stream-listener/{job_id}",
                headers=headers,
                json={"request_session_token": session_token},
        ) as response:
            response.raise_for_status()

    def start_heartbeat(
            self,
            job_id: str,
            session_token: str,
            session: requests.Session,
            stop_event: threading.Event
    ):
        """Send heartbeats until stopped."""
        while not stop_event.is_set():
            try:
                headers = {
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                }
                response = session.post(
                    f"{self.base_url}/stream-heartbeat/{job_id}",
                    headers=headers,
                    params={"request_session_token": session_token},
                )
                response.raise_for_status()
            except Exception as e:
                if not stop_event.is_set():  # Only log if we weren't stopping anyway
                    print(f"Heartbeat failed for job {job_id}: {e}")

            # Use time.sleep instead of asyncio.sleep since this is synchronous
            time.sleep(self.HEARTBEAT_INTERVAL_SECONDS)

    @contextmanager
    def stream_heartbeat_session(self, job_id: str, session_token: str) -> Generator[requests.Session, None, None]:
        """Context manager that handles session registration and heartbeat."""
        session = requests.Session()
        stop_heartbeat = threading.Event()

        # Run this concurrently in a thread so we can not block main SDK path/behavior
        # but still run heartbeat requests
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.start_heartbeat,
                job_id,
                session_token,
                session,
                stop_heartbeat
            )

            try:
                yield session
            finally:
                # Signal stop and cleanup
                stop_heartbeat.set()
                # Wait for heartbeat to finish with timeout
                try:
                    future.result(timeout=1.0)
                except TimeoutError:
                    pass
                self.unregister_stream_listener(job_id, session_token)
                session.close()

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
            # Get job information from list-jobs endpoint
            # TODO(cooper) we should add a get jobs endpoint:
            # GET /jobs/{job_id}
            jobs_response = s.get(
                f"{self.base_url}/list-jobs",
                headers=headers
            )
            jobs_response.raise_for_status()

            # Find the specific job we want to attach to
            job = next(
                (job for job in jobs_response.json()["jobs"] if job["job_id"] == job_id),
                None
            )

            if not job:
                spinner.write(to_colored_text(f"Job {job_id} not found", state="fail"))
                return

            match job.get("status"):
                case "SUCCEEDED":
                    spinner.write(
                        to_colored_text(
                            f"Job already completed. You can obtain the results with `mi jobs results {job_id}`"
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

        session_token = self.register_stream_listener(job_id)

        with self.stream_heartbeat_session(job_id, session_token) as s:
            with s.get(
                    f"{self.base_url}/stream-job-progress/{job_id}?request_session_token={session_token}",
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
                            f"✔ Job succeeded. Use `mi jobs results {job_id}` to obtain results.",
                            state="success",
                        )
                    )
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

    def get_job_status(self, job_id: str):
        """
        Get the status of a job by its ID.

        This method retrieves the status of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the status for.

        Returns:
            str: The status of the job.
        """
        endpoint = f"{self.base_url}/job-status/{job_id}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Checking job status with ID: {job_id}"),
            color=YASPIN_COLOR,
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
            spinner.write(to_colored_text("✔ Job status retrieved!", state="success"))
        return response.json()["job_status"][job_id]

    def get_job_results(
        self,
        job_id: str,
        include_inputs: bool = False,
        include_cumulative_logprobs: bool = False,
    ):
        """
        Get the results of a job by its ID.

        This method retrieves the results of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the results for.
            include_inputs (bool, optional): Whether to include the inputs in the results. Defaults to False.
            include_cumulative_logprobs (bool, optional): Whether to include the cumulative logprobs in the results. Defaults to False.

        Returns:
            list: The results of the job.
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
            if response.status_code == 200:
                spinner.write(
                    to_colored_text("✔ Job results retrieved", state="success")
                )
            else:
                spinner.write(
                    to_colored_text(
                        f"Bad status code: {response.status_code}", state="fail"
                    )
                )
                spinner.stop()
                print(to_colored_text(response.json(), state="fail"))
                return
        return response.json()["results"]

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

    def create_stage(self):
        """
        Create a new stage.

        This method creates a new stage and returns its ID.

        Returns:
            str: The ID of the new stage.
        """
        endpoint = f"{self.base_url}/create-stage"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER, text=to_colored_text("Creating stage"), color=YASPIN_COLOR
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
            stage_id = response.json()["stage_id"]
            spinner.write(
                to_colored_text(f"✔ Stage created with ID: {stage_id}", state="success")
            )
        return stage_id

    def upload_to_stage(
        self,
        stage_id: Union[List[str], str] = None,
        file_paths: Union[List[str], str] = None,
        verify_ssl: bool = True,
    ):
        """
        Upload data to a stage.

        This method uploads files to a stage. Accepts a stage ID and file paths. If only a single parameter is provided, it will be interpreted as the file paths.

        Args:
            stage_id (str): The ID of the stage to upload to. If not provided, a new stage will be created.
            file_paths (Union[List[str], str]): A list of paths to the files to upload, or a single path to a collection of files.
            verify_ssl (bool): Whether to verify SSL certificates. Set to False to bypass SSL verification for troubleshooting.

        Returns:
            dict: The response from the API.
        """
        # when only a single parameter is provided, it is interpreted as the file paths
        if file_paths is None and stage_id is not None:
            file_paths = stage_id
            stage_id = None

        if file_paths is None:
            raise ValueError("File paths must be provided")

        if stage_id is None:
            stage_id = self.create_stage()

        endpoint = f"{self.base_url}/upload-to-stage"

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
            text=to_colored_text(f"Uploading files to stage: {stage_id}"),
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
                    "stage_id": stage_id,
                }

                headers = {
                    "Authorization": f"Key {self.api_key}"}

                count += 1
                spinner.write(
                    to_colored_text(
                        f"Uploading file {count}/{len(file_paths)} to stage: {stage_id}"
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
                    f"✔ {count} files successfully uploaded to stage", state="success"
                )
            )
        return stage_id

    def list_stages(self):
        endpoint = f"{self.base_url}/list-stages"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with yaspin(
            SPINNER, text=to_colored_text("Retrieving stages"), color=YASPIN_COLOR
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
            spinner.write(to_colored_text("✔ Stages retrieved", state="success"))
        return response.json()["stages"]

    def list_stage_files(self, stage_id: str):
        endpoint = f"{self.base_url}/list-stage-files"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "stage_id": stage_id,
        }
        with yaspin(
            SPINNER,
            text=to_colored_text(f"Listing files in stage: {stage_id}"),
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
                to_colored_text(f"✔ Files listed in stage: {stage_id}", state="success")
            )
        return response.json()["files"]

    def download_from_stage(
        self,
        stage_id: str,
        files: Union[List[str], str] = None,
        output_path: str = None,
    ):
        endpoint = f"{self.base_url}/download-from-stage"

        if files is None:
            files = self.list_stage_files(stage_id)
        elif isinstance(files, str):
            files = [files]

        if not files:
            print(
                to_colored_text(
                    f"Couldn't find files for stage ID: {stage_id}", state="fail"
                )
            )
            return

        # if no output path is provided, save the files to the current working directory
        if output_path is None:
            output_path = os.getcwd()

        with yaspin(
            SPINNER,
            text=to_colored_text(f"Downloading files from stage: {stage_id}"),
            color=YASPIN_COLOR,
        ) as spinner:
            count = 0
            for file in files:
                headers = {
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "stage_id": stage_id,
                    "file_name": file,
                }
                spinner.text = to_colored_text(
                    f"Downloading file {count + 1}/{len(files)} from stage: {stage_id}"
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
                    f"✔ {count} files successfully downloaded from stage: {stage_id}",
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
