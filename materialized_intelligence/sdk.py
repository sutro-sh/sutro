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
        output_path: str = None, 
        output_column: str = "inference_result",
        job_priority: int = 0,
        json_schema: str = None,
        num_workers: int = 1,
        prompt_prefix: str = None,
        dry_run: bool = False
    ):
        """
        Run inference on the provided data.

        This method allows you to run inference on the provided data using the Materialized Intelligence API.
        It supports various data types such as lists, pandas DataFrames, polars DataFrames, and file paths.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to run inference on.
            model (str, optional): The model to use for inference. Defaults to "llama-3.1-8b".
            column (str, optional): The column name to use for inference. Required if data is a DataFrame.
            output_path (str, optional): The path to save the output to. If not specified, the results will be returned as a list.
            output_column (str, optional): The column name to store the inference results in. Defaults to "inference_result".
            job_priority (int, optional): The priority of the job. Defaults to 0.
            json_schema (str, optional): A JSON schema for the output. Defaults to None.
            num_workers (int, optional): The number of workers to use for inference. Defaults to 1.
            prompt_prefix (str, optional): A prefix prompt to add to all inputs. Defaults to None.
            dry_run (bool, optional): If True, the method will return cost estimates instead of running inference. Defaults to False.

        Returns:
            Union[List, pd.DataFrame, pl.DataFrame, str]: The results of the inference.
        
        """
        if isinstance(data, list):
            input_data = data
        elif isinstance(data, (pd.DataFrame, pl.DataFrame)):
            if column is None:
                raise ValueError("Column name must be specified for DataFrame input")
            input_data = data[column].to_list()
        elif isinstance(data, str):
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

        if prompt_prefix:
            if prompt_prefix[-1] != ' ':
                prompt_prefix = prompt_prefix + ' '
            input_data = [prompt_prefix + input_item for input_item in input_data]

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
                spinner.succeed("Materialized results received.")

                results = response_data["results"]

                if isinstance(data, (pd.DataFrame, pl.DataFrame)):
                    if isinstance(data, pd.DataFrame):
                        data[output_column] = results
                    elif isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.Series(output_column, results))
                    return data
                elif isinstance(data, str) and output_path:
                    file_ext = os.path.splitext(output_path)[1].lower()
                    if file_ext == '.csv':
                        pl.DataFrame({"input": input_data, output_column: results}).write_csv(output_path)
                    elif file_ext == '.parquet':
                        pl.DataFrame({"input": input_data, output_column: results}).write_parquet(output_path)
                    else:
                        with open(output_path, 'w') as file:
                            for input_item, result in zip(input_data, results):
                                file.write(f"{input_item}\t{result}\n")
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
        with Halo(text="Fetching all jobs", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()['jobs']

    def get_job_status(self, job_id: str):
        """
        Get the status of a job by its ID.

        This method retrieves the status of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the status for.

        Returns:
            dict: The status of the job.
        """
        endpoint = f"{self.base_url}/job-status/{job_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with Halo(text=f"Checking job status with ID: {job_id}", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()
    
    def get_job_results(self, job_id: str, include_inputs: bool = False):
        """
        Get the results of a job by its ID.

        This method retrieves the results of a job using its unique identifier.

        Args:
            job_id (str): The ID of the job to retrieve the results for.

        Returns:
            list: The results of the job.
        """
        endpoint = f"{self.base_url}/job-results"
        payload = {
            "job_id": job_id,
            "include_inputs": include_inputs
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