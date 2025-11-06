from typing import Union, List
import polars as pl
import pandas as pd
from ..common import EmbeddingModelOptions
from ..interfaces import BaseSutroClient


class EmbeddingTemplates(BaseSutroClient):
    def embed(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: EmbeddingModelOptions = "qwen-3-embedding-0.6b",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        output_column: str = "inference_result",
        column: Union[str, List[str]] = None,
        truncate_rows: bool = True,
    ):
        """
        A simple template style function to generate embeddings for the provided data, with Sutro. The intention is that the implemented code should be very easy to extend further, while showing a basic structure for large scale embedding generation with Sutro.

        This method allows you to generate vector embeddings for the provided data using Sutro.
        It supports various options for inputting data, such as lists, DataFrames (Polars or Pandas), file paths and datasets.
        The method will wait for the embedding job to complete before returning the results.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to generate embeddings for.
            model (ModelOptions, optional): The embedding model to use. Defaults to "qwen-3-embedding-0.6b"; a model we chose as its small & fast, yet performs well on a variety of tasks.
            job_priority (int, optional): The priority of the job. Defaults to 0.
            name (Union[str, List[str]], optional): A job name for experiment/metadata tracking purposes. Defaults to None.
            description (Union[str, List[str]], optional): A job description for experiment/metadata tracking purposes. Defaults to None.
            output_column (str, optional): The column name to store the embedding results in if the input is a DataFrame. Defaults to "inference_result".
            column (Union[str, List[str]], optional): The column name to use for embedding generation. Required if data is a DataFrame, file path, or dataset. If a list is supplied, it will concatenate the columns of the list into a single column, accepting separator strings.
            truncate_rows (bool, optional): If True, any rows that have a token count exceeding the context window length of the selected model will be truncated to the max length that will fit within the context window. Defaults to True.

        Returns:
            The completed embedding results for the provided data.

        """
        job_id = self.infer(
            data,
            model,
            name,
            description,
            column,
            output_column,
            job_priority,
            truncate_rows=truncate_rows,
            stay_attached=False,
        )

        return self.await_job_completion(job_id)
