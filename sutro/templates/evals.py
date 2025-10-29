from typing import Union, List, Tuple
import polars as pl
import pandas as pd
from ..common import EmbeddingModelOptions
from ..interfaces import BaseSutroClient


class EvalsTemplates(BaseSutroClient):
    def score(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: EmbeddingModelOptions = "gemma-3-12b-it",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        column: Union[str, List[str]] = None,
        # function-specific parameters
        criteria: Union[str, List[str]] = None,
        score_column_name: str = "score",
        range: Tuple[int, int] = (0, 10),
    ):
        """
        A simple invocation of an LLM-as-a-judge numerical scoring function, with a default 0-10 range.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        system_prompt = """
        You are a judge. Your job is to score the data presented to you according to the following criteria:
        {criteria}
        Return a score between {range[0]} and {range[1]}, and nothing else.
        """.format(criteria=', '.join(criteria), range=range)

        json_schema = {
            "type": "object",
            "properties": {
                f"{score_column_name}": {"type": "integer", "minimum": range[0], "maximum": range[1]},
            },
            "required": [score_column_name],
        }

        job_id = self.infer(
            data=data,
            model=model,
            name=name,
            description=description,
            column=column,
            system_prompt=system_prompt,
            output_schema=json_schema,
            job_priority=job_priority,
        )

        if isinstance(data, pd.DataFrame) or isinstance(data, pl.DataFrame):
            return self.await_job_completion(job_id, with_original_df=data)
        else:
            return self.await_job_completion(job_id)