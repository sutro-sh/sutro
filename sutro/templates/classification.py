from typing import Union, List
import polars as pl
import pandas as pd
from openai import BaseModel

from ..common import ModelOptions
from ..interfaces import BaseSutroClient


class ClassificationTemplates(BaseSutroClient):
    def classify(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        classes: list[str],
        class_descriptions: list[str] | None = None,
        model: ModelOptions = "gemma-14b-it",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        output_column: str = "inference_result",
        column: Union[str, List[str]] = None,
        truncate_rows: bool = True,
    ):
        """
        A simple template style function to perform classification on the provided data with Sutro. The intention is that the implemented code should be very easy to extend further, while showing a basic structure for large-scale classification with Sutro.

        It uses structured outputs with a scratchpad field, enabling the model to reason step-by-step before providing the final classification.

        The method supports various input formats including lists, DataFrames (Polars or Pandas), file paths, and datasets.

        The method will wait for the classification job to complete before returning the results.

        Args:
            data (Union[List, pd.DataFrame, pl.DataFrame, str]): The data to classify. Each row should contain some text to classifiy that fits into one of the passed in labels.
            model (ModelOptions, optional): The LLM to use. Defaults to "gemma-14b-it"; a model chosen for its balance of performance and efficiency, that also retains competency across a broad numver of different domains.
            job_priority (int, optional): The priority of the job. Defaults to 0. See https://docs.sutro.sh/concepts/job-priority for more info.
            classes (list[str] | None, optional): The list of category labels to classify into. The model will classify each input into exactly one of these labels.
            class_descriptions (list[str] | None, optional): Optional descriptions for each class to provide additional context to the model. If provided, must be the same length as classes. Descriptions can improve classification accuracy, especially for ambiguous or domain-specific categories. Defaults to None.
            name (Union[str, List[str]], optional): A job name for experiment/metadata tracking purposes. Defaults to None.
            description (Union[str, List[str]], optional): A job description for experiment/metadata tracking purposes. Defaults to None.
            output_column (str, optional): The column name to store the classification results in if the input is a DataFrame. Defaults to "inference_result".
            column (Union[str, List[str]], optional): The column name to use for classification. Required if data is a DataFrame, file path, or dataset. If a list is supplied, it will concatenate the columns of the list into a single column, accepting separator strings.
            truncate_rows (bool, optional): If True, any rows that have a token count exceeding the context window length of the selected model will be truncated to the max length that will fit within the context window. Defaults to True.

        Returns:
            The completed classification results for the provided data, including both the scratchpad reasoning and final classification for each input.

        """
        if class_descriptions is not None:
            if len(classes) != len(class_descriptions):
                raise ValueError(
                    "classes and class_descriptions must have the same length"
                )
            categories = "\n".join(
                [f"- {cls}: {desc}" for cls, desc in zip(classes, class_descriptions)]
            )
        else:
            categories = "\n".join([f"- {cls}" for cls in classes])

        system_prompt = f"""You are an expert classifier. Your task is to accurately categorize the input into one of the provided classes.

## Categories

{categories}

## Instructions

1. **Analyze the input carefully**: Read and understand the full context - identify key elements, themes, and characteristics

2. **Consider each class**: For each possible category, evaluate how similar the input is to its typical characteristics

3. **Provide your reasoning in the scratchpad**: Think through which category fits best and why

4. **Provide output**: Give your final classification

If needed, use the scratchpad field to work through steps 1-3, then provide your final answer in the classification field.

## Guidelines

- Select exactly ONE category, even if multiple seem applicable (choose the best match)
- If the input is ambiguous, choose the closest fit and explain your reasoning
- Base your decision on the actual content, not assumptions or implications
- Similar inputs should receive the same classification
- When outputting the classification, solely included the class label, no descriptive text should be included here

Respond using the structured format with scratchpad and classification fields."""

        class ClassificationOutput(BaseModel):
            # Since we're using structured outputs, we want to give the model some
            # space to reason and think as needed
            scratchpad: str
            classification: str

        job_id = self.infer(
            data,
            model,
            name,
            description,
            system_prompt=system_prompt,
            output_schema=ClassificationOutput,
            column=column,
            output_column=output_column,
            job_priority=job_priority,
            truncate_rows=truncate_rows,
            stay_attached=False,
        )

        return self.await_job_completion(job_id)
