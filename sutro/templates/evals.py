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

        Accepts a normal Sutro input data type, as well as a string or list of strings to use as criteria, 
        a column name to use for the scoring, and a range to use for the scoring (default 0-10).

        Returns a pandas or polars dataframe with the scores as a column, corresponding to the column name provided.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        system_prompt = f"""
        You are a judge. Your job is to score the data presented to you according to the following criteria:
        {', '.join(criteria)}
        Return a score between {range[0]} and {range[1]}, and nothing else.
        """

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
            stay_attached=False,
        )

        res = self.await_job_completion(job_id)
        if isinstance(data, pl.DataFrame):
            return data.with_columns(pl.Series(score_column_name, res[score_column_name]))
        elif isinstance(data, pd.DataFrame):
            return data.assign(**{score_column_name: res[score_column_name]})
        else:
            return res

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
        Accepts a list of lists, a pandas or polars dataframe, as well as option labels to use for the ranking.

        If using a lists of lists, the option labels should correspond to the labels you would like to use for the ranking, in the same order as the lists.
        If using a pandas or polars dataframe, the option labels should correspond to the column names of the dataframe to use for the ranking.

        Returns a list of lists of rankings ordered from best to worst, corresponding to the option labels. 
        If using a pandas or polars dataframe, the rankings will be returned as a column in the original dataframe.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        system_prompt = f"""
        You are a judge. Your job is to rank the options presented to you according to the following criteria:
        {', '.join(criteria)}
        The option labels are: {', '.join(option_labels)}
        Return a ranking of the options as an ordered list of the labels from best to worst, and nothing else.
        """

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
        for _, label in enumerate(option_labels):
            exprs.append(pl.lit(label + ':'))
            exprs.append(pl.col(label))

        data = data.select(pl.concat_str(exprs, separator=" ", ignore_nulls=False).alias('options_with_labels'))

        job_id = self.infer(
            data=data,
            column='options_with_labels',
            model=model,
            name=name,
            description=description,
            system_prompt=system_prompt,
            output_schema=json_schema,
            job_priority=job_priority,
            stay_attached=False,
        )

        res = self.await_job_completion(job_id)
        if isinstance(data, pl.DataFrame):
            return data.with_columns(pl.Series(ranking_column_name, res[ranking_column_name]))
        elif isinstance(data, pd.DataFrame):
            return data.assign(**{ranking_column_name: res[ranking_column_name]})
        else:
            return res