import requests
import pandas as pd
import polars as pl
import json
from typing import Union, List
import os
from halo import Halo

class MaterializedIntelligence:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or self.check_for_api_key()
        self.base_url = "https://staging.api.materialized.dev/"

    def check_for_api_key(self):
        CONFIG_DIR = os.path.expanduser("~/.materialized_intelligence")
        CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get("api_key")
        else:
            return None
        
    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def infer(self, 
        data: Union[List, pd.DataFrame, pl.DataFrame, str], 
        model: str = "llama-3.1-8b",
        column: str = None, 
        output_path: str = None, 
        output_column: str = "inference_result",
        job_priority: int = 0,
        json_schema: str = None,
        num_workers: int = 1,
        dry_run: bool = False
    ):
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
        endpoint = f"{self.base_url}/list-jobs"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with Halo(text="Fetching all jobs", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()['jobs']

    def get_job_status(self, job_id: str):
        endpoint = f"{self.base_url}/job-status/{job_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with Halo(text=f"Checking job status with ID: {job_id}", spinner="dots", text_color="blue"):
            response = requests.get(endpoint, headers=headers)
        return response.json()
    
    def get_job_results(self, job_id: str):
        endpoint = f"{self.base_url}/job-results"
        payload = {
            "job_id": job_id
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
            spinner.fail("Failed to retrieve job results")
        return response.json()['results']

    def cancel_job(self, job_id: str):
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