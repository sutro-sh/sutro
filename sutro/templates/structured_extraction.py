import json
from typing import Union, List, Dict, Any, Type
import polars as pl
import pandas as pd
from pydantic import BaseModel

from ..common import ModelOptions
from ..interfaces import BaseSutroClient


def normalize_to_dict(
    schema_input: Union[type[BaseModel], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract field information from either a Pydantic model or JSON schema.
    Returns a normalized schema dictionary.
    """
    if isinstance(schema_input, type) and issubclass(schema_input, BaseModel):
        # It's a Pydantic model - convert to JSON schema
        return schema_input.model_json_schema()
    else:
        # It's already a JSON schema
        return schema_input


class StructuredExtractionTemplates(BaseSutroClient):
    def structure(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        output_schema: Union[Dict[str, Any], Type[BaseModel]],
        model: ModelOptions = "gemma-3-12b-it",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        output_column: str = "inference_result",
        column: Union[str, List[str]] = None,
        truncate_rows: bool = True,
    ):
        """
        Extract structured information from unstructured text data using Sutro. The extracted text can be abstractive
        ("create a detailed title and description for this SEC document") or extractive ("return all quotes mentioning
        San Diego, CA") in nature.

        The method supports various input formats including lists, DataFrames (Polars or Pandas),
        file paths, and Sutro Datasets. It will wait for the job to complete before returning results.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to extract from. Each row
                should contain text with information to be extracted according to the output schema.
            output_schema (Union[Dict[str, Any], Type[BaseModel]]): The schema defining what information
                to extract and its structure. Can be either:
                - A Pydantic BaseModel class defining the expected output structure
                - A JSON schema dictionary specifying the fields and types
                It is reccomended to add rich `description` fields to each desired field to extract;
                this helps the LLM navigate any ambiguity or nuance that may be present.
            model (ModelOptions, optional): The LLM to use. Defaults to "gemma-3-12b-it"; a model
                chosen for its balance of performance and efficiency across diverse extraction tasks.
            job_priority (int, optional): The priority of the job. Defaults to 0.
            name (Union[str, List[str]], optional): A job name for experiment/metadata tracking purposes.
                Defaults to None.
            description (Union[str, List[str]], optional): A job description for experiment/metadata
                tracking purposes. Defaults to None.
            output_column (str, optional): The column name to store the extraction results in if the
                input is a DataFrame. Defaults to "inference_result".
            column (Union[str, List[str]], optional): The column name to use for extraction. Required
                if data is a DataFrame, file path, or dataset. If a list is supplied, it will concatenate
                the columns into a single column, accepting separator strings.
            truncate_rows (bool, optional): If True, any rows exceeding the model's context window
                will be truncated to fit. Defaults to True.

        Returns:
            The completed extraction results as where each row or list element is a structured JSON string matching
            the provided output_schema.
        """
        schema = normalize_to_dict(output_schema)

        system_prompt = f"""You are an expert at carefully reading and extracting information from text with high accuracy and attention to nuance.

## Task

When provided with text, you will extract structured information matching a predefined schema. Your goal is to produce high-quality extractions that capture the true meaning and relevant details from the source material.

## Extraction Guidelines

1. **Read thoroughly** - Consider the full context before extracting any information
2. **Preserve meaning** - Capture intent and nuance, not just surface-level keywords
3. **Use context for clarity** - When information is ambiguous, use surrounding context to make informed judgments
4. **Preserve exact wording** - When aiming to extract specific terms, quotes, or phrases, keep the original language
5. **Infer reasonably** - Some fields may require reading between the lines based on clear contextual clues

## Output Schema

You must return valid JSON that strictly matches this schema:

{json.dumps(schema, indent=2)}

## Output Format

Return only valid JSON with no additional commentary or explanation. Every response should be parseable JSON matching the schema above."""

        job_id = self.infer(
            data,
            model,
            name,
            description,
            system_prompt=system_prompt,
            output_schema=output_schema,
            column=column,
            output_column=output_column,
            job_priority=job_priority,
            truncate_rows=truncate_rows,
            stay_attached=False,
        )

        return self.await_job_completion(job_id)
