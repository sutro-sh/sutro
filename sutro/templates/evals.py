from typing import Union, List, Tuple
import polars as pl
import pandas as pd
from ..common import ModelOptions
from ..interfaces import BaseSutroClient


class EvalTemplates(BaseSutroClient):
    def score(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: ModelOptions = "gemma-3-12b-it",
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

    def rank(
        self,
        model: ModelOptions = "gemma-3-12b-it",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        # function-specific parameters
        data: Union[List[List], pd.DataFrame, pl.DataFrame, str] = None, # data is always required, but this method accepts a list of lists as well
        option_labels: List[str] = None,
        criteria: Union[str, List[str]] = None,
        ranking_column_name: str = "ranking",
    ):
        """
        A simple invocation of a LLM-as-a-judge ranking (pairwise comparison) function, accepting multiple options to rank.
        Accepts a list of lists, a pandas dataframe, or a polars dataframe, as well as 
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        system_prompt = """
        You are a judge. Your job is to rank the options presented to you according to the following criteria:
        {criteria}
        The option labels are: {options_list}
        Return a ranking of the options as an ordered list of the labels from best to worst, and nothing else.
        """.format(criteria=', '.join(criteria), option_labels=', '.join(option_labels))

        json_schema = {
            "type": "object",
            "properties": {
                f"{ranking_column_name}": {"type": "array", "items": {"type": "string"}},
            },
            "required": [ranking_column_name],
        }

        if isinstance(data, list):
            # create a polars dataframe from the list of lists
            data = pl.DataFrame(data, schema=option_labels)
        elif isinstance(data, pd.DataFrame):
            # convert to polars dataframe
            data = data.from_pandas(data)

        exprs = [] # because the option literals are the same as the same as the column names we don't use the built-in column concatenation helper function
        for i in range(len(option_labels)):
            exprs.append(pl.lit(option_labels[i]))
            exprs.append(pl.col(option_labels[i]))

        data = data.with_columns(pl.concat(exprs)).alias('options_with_labels')

        job_id = self.infer(
            data=data,
            column='options_with_labels',
            model=model,
            name=name,
            description=description,
            system_prompt=system_prompt,
            output_schema=json_schema,
            job_priority=job_priority,
        )

        if isinstance(data, pd.DataFrame) or isinstance(data, pl.DataFrame):
            return self.await_job_completion(job_id, with_original_df=data)
        else:
            return self.await_job_completion(job_id)