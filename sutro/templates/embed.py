from typing import Union, List, TYPE_CHECKING, Optional
import polars as pl
import pandas as pd

from ..common import EmbeddingModelOptions

if TYPE_CHECKING:
    from ..sdk import Sutro


def _embed_simple(
    client: "Sutro",
    data: Union[List, pd.DataFrame, pl.DataFrame, str],
    model: EmbeddingModelOptions,
    name: Union[str, List[str], None],
    description: Union[str, List[str], None],
    output_column: Optional[str],
    column: Union[str, List[str], None],
    job_priority: int,
    truncate_rows: bool,
):
    job_id = client.infer(
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

    return client.await_job_completion(job_id)
