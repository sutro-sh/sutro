import os
from typing import TYPE_CHECKING, Any, Dict, Union, List, Optional
import pandas as pd
import polars as pl
import requests
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..sdk import Sutro, handle_data_helper, ModelOptions


def do_dataframe_column_concatenation(
    data: Union[pd.DataFrame, pl.DataFrame], column: Union[str, List[str]]
):
    """
    If the user has supplied a dataframe and a list of columns, this will intelligenly concatenate the columns into a single column, accepting separator strings.
    """
    try:
        if isinstance(data, pd.DataFrame):
            series_parts = []
            for p in column:
                if p in data.columns:
                    s = data[p].astype("string").fillna("")
                else:
                    # Treat as a literal separator
                    s = pd.Series([p] * len(data), index=data.index, dtype="string")
                series_parts.append(s)

            out = series_parts[0]
            for s in series_parts[1:]:
                out = out.str.cat(s, na_rep="")

            return out.tolist()
        elif isinstance(data, pl.DataFrame):
            exprs = []
            for p in column:
                if p in data.columns:
                    exprs.append(pl.col(p).cast(pl.Utf8).fill_null(""))
                else:
                    exprs.append(pl.lit(p))

            result = data.select(
                pl.concat_str(exprs, separator="", ignore_nulls=False).alias("concat")
            )
            return result["concat"].to_list()
        return None
    except Exception as e:
        raise ValueError(f"Error handling column concatentation: {e}")


def handle_data_helper(
    data: Union[List, pd.DataFrame, pl.DataFrame, str], column: str = None
):
    if isinstance(data, list):
        input_data = data
    elif isinstance(data, (pd.DataFrame, pl.DataFrame)):
        if column is None:
            raise ValueError("Column name must be specified for DataFrame input")
        if isinstance(column, list):
            input_data = do_dataframe_column_concatenation(data, column)
        elif isinstance(column, str):
            input_data = data[column].to_list()
    elif isinstance(data, str):
        if data.startswith("dataset-"):
            input_data = data + ":" + column
        else:
            file_ext = os.path.splitext(data)[1].lower()
            if file_ext == ".csv":
                df = pl.read_csv(data)
            elif file_ext == ".parquet":
                df = pl.read_parquet(data)
            elif file_ext in [".txt", ""]:
                with open(data, "r") as file:
                    input_data = [line.strip() for line in file]
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            if file_ext in [".csv", ".parquet"]:
                if column is None:
                    raise ValueError(
                        "Column name must be specified for CSV/Parquet input"
                    )
                input_data = df[column].to_list()
    else:
        raise ValueError(
            "Unsupported data type. Please provide a list, DataFrame, or file path."
        )

    return input_data


def make_auth_headers(client: "Sutro") -> Dict[str, str]:
    """Helper to create auth headers."""
    return {"Authorization": f"Bearer {client.api_key}"}


def do_request(client: "Sutro", method: str, endpoint: str, **kwargs: Any) -> dict:
    """
    Helper to make authenticated requests.

    Args:
        client: Sutro client instance
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        endpoint: API endpoint (e.g., "/datasets" or "datasets")
        **kwargs: Additional arguments passed to requests (json, data, files, etc.)

    Returns:
        JSON response from the API
    """
    headers = make_auth_headers(client)
    # Merge with any headers passed in kwargs
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))

    url = f"{client.base_url}/{endpoint.lstrip('/')}"

    # Explicit method dispatch
    method = method.upper()
    if method == "GET":
        response = requests.get(url, headers=headers, **kwargs)
    elif method == "POST":
        response = requests.post(url, headers=headers, **kwargs)
    elif method == "PUT":
        response = requests.put(url, headers=headers, **kwargs)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers, **kwargs)
    elif method == "PATCH":
        response = requests.patch(url, headers=headers, **kwargs)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    response.raise_for_status()
    return response.json()


class BatchInferenceRequest(BaseModel):
    """Request model for batch inference."""

    model: ModelOptions
    data: Union[List, pd.DataFrame, pl.DataFrame, str]
    column: Union[str, List[str]]
    job_priority: int
    json_schema: Optional[Dict[str, Any]] = None
    sampling_params: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    cost_estimate: bool = False
    random_seed_per_input: bool = False
    truncate_rows: bool = False
    name: Optional[str] = None
    description: Optional[str] = None


def _do_batch_inference_request(
    client: "Sutro",
    request: BatchInferenceRequest,
):
    return do_request(
        client,
        "POST",
        "/batch-inference",
        json=request.model_dump(mode="json", exclude_none=True),
    )
