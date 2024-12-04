import requests
import pandas as pd
import polars as pl
import json
from typing import Union, List
import os
from halo import Halo

class MaterializedIntelligence:
    def __init__(self, api_key: str = None, base_url: str = "https://api.materialized.dev/"):
        self.api_key = api_key or self.check_for_api_key()
        self.base_url = base_url

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
            with open(CONFIG_FILE, 'r') as f:
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

    def handle_data_helper(self, data: Union[List, pd.DataFrame, pl.DataFrame, str], column: str = None):
        if isinstance(data, list):
            input_data = data
        elif isinstance(data, (pd.DataFrame, pl.DataFrame)):
            if column is None:
                raise ValueError("Column name must be specified for DataFrame input")
            input_data = data[column].to_list()
        elif isinstance(data, str):
            if data.startswith("stage-"):
                input_data = data + ':' + column
            else:
                file_ext = os.path.splitext(data)[1].lower()
                if file_ext == '.csv':
                    df = pl.read_csv(data)
                elif file_ext == '.parquet':
                    df = pl.read_parquet(data)
                elif file_ext in ['.txt', '']:
                    with open(data, 'r') as file:
                        input_data = [line.strip() for line in file]
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
                
                if file_ext in ['.csv', '.parquet']:
                    if column is None:
                        raise ValueError("Column name must be specified for CSV/Parquet input")
                    input_data = df[column].to_list()
        else:
            raise ValueError("Unsupported data type. Please provide a list, DataFrame, or file path.")
        
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
    def infer(self, 
        data: Union[List, pd.DataFrame, pl.DataFrame, str], 
        model: str = "llama-3.1-8b",
        column: str = None, 
        output_column: str = "inference_result",
        job_priority: int = 0,
        json_schema: str = None,
        num_workers: int = 1,
        system_prompt: str = None,
        dry_run: bool = False
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
            json_schema (str, optional): A JSON schema for the output. Defaults to None.
            system_prompt (str, optional): A system prompt to add to all inputs. This allows you to define the behavior of the model. Defaults to None.
            dry_run (bool, optional): If True, the method will return cost estimates instead of running inference. Defaults to False.

        Returns:
            Union[List, pd.DataFrame, pl.DataFrame, str]: The results of the inference.
        
        """
        input_data = self.handle_data_helper(data, column)

        endpoint = f"{self.base_url}/batch-inference"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "inputs": input_data, 
            "job_priority": job_priority,
            "json_schema": json_schema,
            "num_workers": num_workers,
            "system_prompt": system_prompt,
            "dry_run": dry_run
        }
        if dry_run:
            spinner_text = "Retrieving cost estimates..."
        else:   
            spinner_text = "Materializing results" if job_priority == 0 else f"Creating priority {job_priority} job"
        spinner = Halo(text=spinner_text, spinner="dots", text_color="blue")
        spinner.start()
        response = requests.post(endpoint, data=json.dumps(payload), headers=headers)
        spinner.stop()
        response_data = response.json()
        if response.status_code != 200:
            message = response_data.get("metadata", "Unknown error")["message"]
            spinner.fail(f"Error: {response.status_code} - {message}")
            return 
        else:
            if dry_run:
                spinner.succeed("Cost estimates retrieved")
                return response_data["results"]
            elif job_priority != 0:
                job_id = response_data["results"]
                spinner.text_color = "yellow"
                spinner.stop_and_persist(symbol="🛠️ ", text=f"Priority {job_priority} Job created with ID: {job_id}. Use `mi.get_job_status('{job_id}')` to check the status of the job.")
                return job_id
            else:
                spinner.text_color = "green"
                job_id = response_data["metadata"]["job_id"]
                spinner.succeed(f"Materialized results received. You can re-obtain the results with `mi.get_job_results('{job_id}')`.")

                results = response_data["results"]

                if isinstance(data, (pd.DataFrame, pl.DataFrame)):
                    if isinstance(data, pd.DataFrame):
                        data[output_column] = results
                    elif isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.Series(output_column, results))
                    return data
                
                return results

    def list_jobs(self):
        """
        List all jobs.

        This method retrieves a list of all jobs associated with the API key.

        Returns:
            list: A list of job details.
        """
        endpoint = f"{self.base_url}/list-jobs"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with Halo(text="Fetching jobs", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()['jobs']

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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with Halo(text=f"Checking job status with ID: {job_id}", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()['job_status'][job_id]
    
    def get_job_results(self, job_id: str, include_inputs: bool = False, include_cumulative_logprobs: bool = False):
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
            "include_cumulative_logprobs": include_cumulative_logprobs
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        spinner = Halo(text=f"Gathering results from job: {job_id}", spinner="dots", text_color="blue")
        spinner.start()
        response = requests.post(endpoint, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            spinner.succeed("Job results retrieved")
        else:
            spinner.fail("No data available for the specified job")
        return response.json()['results']

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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        spinner = Halo(text=f"Cancelling job: {job_id}", spinner="dots", text_color="blue")
        spinner.start()
        response = requests.get(endpoint, headers=headers)
        if response.status_code == 200:
            spinner.succeed("Job cancelled")
        else:
            spinner.fail("Failed to cancel job")
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        spinner = Halo(text="Creating stage", spinner="dots", text_color="blue")
        spinner.start()
        response = requests.get(endpoint, headers=headers)
        if response.status_code != 200:
            spinner.fail(f"Error: {response.json()['message']}")
            return
        stage_id = response.json()['stage_id']
        spinner.succeed(f"Stage created with ID: {stage_id}")
        return stage_id

    def upload_to_stage(self, stage_id: Union[List[str], str] = None, file_paths: Union[List[str], str] = None):
        """
        Upload data to a stage.

        This method uploads files to a stage. Accepts a stage ID and file paths. If only a single parameter is provided, it will be interpreted as the file paths.

        Args:
            stage_id (str): The ID of the stage to upload to. If not provided, a new stage will be created.
            file_paths (Union[List[str], str]): A list of paths to the files to upload, or a single path to a collection of files.

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
                file_paths = [os.path.join(file_paths, f) for f in os.listdir(file_paths)]
                if len(file_paths) == 0:
                    raise ValueError("No files found in the directory")
            else:
                file_paths = [file_paths]

        spinner = Halo(text=f"Uploading files to stage: {stage_id}", spinner="dots", text_color="blue")
        spinner.start()
        for count, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            
            files = {
                "file": (file_name, open(file_path, "rb"), "application/octet-stream")
            }
            
            payload = {
                "stage_id": stage_id,
            }
            
            headers = { 
                "Authorization": f"Bearer {self.api_key}"
            }

            spinner.text = f"Uploading file {count + 1}/{len(file_paths)} to stage: {stage_id}"
            response = requests.post(endpoint, headers=headers, data=payload, files=files)
            if response.status_code != 200:
                spinner.fail(f"Error: {response.json()['message']}")
                return
            
            count += 1
        spinner.succeed(f"{count} files successfully uploaded to stage: {stage_id}")
        return stage_id

    def list_stages(self):
        endpoint = f"{self.base_url}/list-stages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        spinner = Halo(text="Retrieving stages", spinner="dots", text_color="blue")
        spinner.start()
        response = requests.post(endpoint, headers=headers)
        if response.status_code != 200:
            spinner.fail(f"Error: {response.json()['message']}")
            return
        spinner.succeed("Stages retrieved")
        return response.json()['stages']

    def list_stage_files(self, stage_id: str):
        endpoint = f"{self.base_url}/list-stage-files"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "stage_id": stage_id,
        }
        spinner = Halo(text=f"Listing files in stage: {stage_id}", spinner="dots", text_color="blue")
        spinner.start()
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            spinner.fail(f"Error: {response.json()['message']}")
            return
        spinner.succeed(f"Files listed in stage: {stage_id}")
        return response.json()['files']
    
    def download_from_stage(self, stage_id: str, files: Union[List[str], str] = None, output_path: str = None):
        endpoint = f"{self.base_url}/download-from-stage"

        if files is None:
            files = self.list_stage_files(stage_id)
        elif isinstance(files, str):
            files = [files]

        # if no output path is provided, save the files to the current working directory
        if output_path is None:
            output_path = os.getcwd()

        spinner = Halo(text=f"Downloading files from stage: {stage_id}", spinner="dots", text_color="blue")
        spinner.start()
        for count, file in enumerate(files):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "stage_id": stage_id,
                "file_name": file,
            }
            spinner.text = f"Downloading file {count + 1}/{len(files)} from stage: {stage_id}"
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                spinner.fail(f"Error: {response.json()['message']}")
                return
            file_content = response.content
            with open(os.path.join(output_path, file), "wb") as f:
                f.write(file_content)
            count += 1
        spinner.succeed(f"{count} files successfully downloaded from stage: {stage_id}")
    
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
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        spinner = Halo(text=f"Checking API key.", spinner="dots", text_color="blue")
        spinner.start()
        response = requests.get(endpoint, headers=headers)
        if response.status_code == 200:
            spinner.succeed()
        else:
            spinner.fail()
        return response.json()
    
    def get_quotas(self):
        endpoint = f"{self.base_url}/get-quotas"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with Halo(text="Fetching quotas", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()['quotas']
        