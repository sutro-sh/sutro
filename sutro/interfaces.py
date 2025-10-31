from enum import Enum

import pandas as pd
import polars as pl
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel

from sutro.common import ModelOptions


class BaseSutroClient:
    """
    Base class declaring attributes and method interfaces for template function mixins
    to use.
    """

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
        output_schema: Union[Dict[str, Any], Type[BaseModel]] = None,
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
        output_column: str = "inference_result",
        is_cost_estimate: bool = False,
    ) -> pl.DataFrame | None: ...


class JobStatus(str, Enum):
    """Job statuses that will be returned by the API & SDK"""

    UNKNOWN = "UNKNOWN"
    QUEUED = "QUEUED"  # Job is waiting to start
    STARTING = "STARTING"  # Job is in the process of starting up
    RUNNING = "RUNNING"  # Job is actively running
    SUCCEEDED = "SUCCEEDED"  # Job completed successfully
    CANCELLING = "CANCELLING"  # Job is in the process of being canceled
    CANCELLED = "CANCELLED"  # Job was canceled by the user
    FAILED = "FAILED"  # Job failed

    @classmethod
    def terminal_statuses(cls) -> list["JobStatus"]:
        return [
            cls.SUCCEEDED,
            cls.FAILED,
            cls.CANCELLING,
            cls.CANCELLED,
        ]

    def is_terminal(self) -> bool:
        return self in self.terminal_statuses()
