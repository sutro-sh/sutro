import requests
import pandas as pd
import polars as pl
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from sutro.common import ModelOptions


class BaseSutroClient:
    """
    Base class declaring attributes and method interfaces for mixins.

    This provides type hints for all methods that will be available
    on the Sutro client through mixins.
    """

    # Shared state
    _api_key: str
    _base_url: str
    _http: requests.Session

    # Core inference method interface
    def infer(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: Union[ModelOptions, List[ModelOptions]] = "gemma-3-12b-it",
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        column: Union[str, List[str]] = None,
        output_column: str = "inference_result",
        job_priority: int = 0,
        output_schema: Union[Dict[str, Any], BaseModel] = None,
        sampling_params: dict = None,
        system_prompt: str = None,
        dry_run: bool = False,
        stay_attached: Optional[bool] = None,
        random_seed_per_input: bool = False,
        truncate_rows: bool = True,
    ) -> Any:
        """
        Run inference on a dataset.

        Args:
            data: Input data (list, DataFrame, or dataset ID)
            model: Model(s) to use for inference
            name: Job name(s)
            description: Job description(s)
            column: Column(s) to process
            output_column: Name for output column
            job_priority: Job priority (0-10, higher = more priority)
            output_schema: Pydantic model or JSON schema for structured output
            sampling_params: Model sampling parameters
            system_prompt: System prompt for the model
            dry_run: If True, validate without running
            stay_attached: Wait for job completion
            random_seed_per_input: Use random seed per input
            truncate_rows: Truncate long inputs

        Returns:
            Job result or job ID
        """
        ...

    def await_job_completion(
        self,
        job_id: str,
        timeout: Optional[int] = 7200,
        obtain_results: bool = True,
        is_cost_estimate: bool = False,
    ) -> list | None: ...
