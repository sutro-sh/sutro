import os
import sys
from typing import Union, List, Literal, Dict, Any, Type, Optional

import pandas as pd
import polars as pl
from colorama import Fore, Style
from pydantic import BaseModel
from tqdm import tqdm

EmbeddingModelOptions = Literal[
    "qwen-3-embedding-0.6b",
    "qwen-3-embedding-6b",
    "qwen-3-embedding-8b",
]

# Models available for inference.  Keep in sync with the backend configuration
# so users get helpful autocompletion when selecting a model.
ModelOptions = Literal[
    "llama-3.2-3b",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "llama-3.3-70b",
    "qwen-3-4b",
    "qwen-3-14b",
    "qwen-3-32b",
    "qwen-3-30b-a3b",
    "qwen-3-235b-a22b",
    "qwen-3-4b-thinking",
    "qwen-3-14b-thinking",
    "qwen-3-32b-thinking",
    "qwen-3-235b-a22b-thinking",
    "qwen-3-30b-a3b-thinking",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "qwen-3-embedding-0.6b",
    "qwen-3-embedding-6b",
    "qwen-3-embedding-8b",
]


# This is how yaspin defines is_jupyter logic
def is_jupyter() -> bool:
    return not sys.stdout.isatty()


def make_clickable_link(url, text=None):
    """
    Create a clickable link for terminals that support OSC 8 hyperlinks.
    Falls back to plain text for terminals that don't support it.
    """
    # Don't need to add the special chars for jupyter notebook
    if is_jupyter():
        return url

    if text is None:
        text = url
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


# Adding color to text is not supported in Jupyter notebooks and breaks
# things
BASE_OUTPUT_COLOR = None if is_jupyter() else "blue"


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
                pl.concat_str(exprs, separator="", ignore_nulls=True).alias("concat")
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


def normalize_output_schema(
    output_schema: Union[Dict[str, Any], Type[BaseModel], None],
):
    """Consolidate any varied types for output_schema to dict format."""
    if hasattr(output_schema, "model_json_schema"):
        return output_schema.model_json_schema()
    elif isinstance(output_schema, dict):
        return output_schema
    else:
        raise ValueError(
            "Invalid output schema type. Must be a dictionary or a pydantic Model."
        )


def to_colored_text(
    text: str, state: Optional[Literal["success", "fail", "callout"]] = None
) -> str:
    """
    Apply color to text based on state.

    Args:
        text (str): The text to color
        state (Optional[Literal['success', 'fail']]): The state that determines the color.
            Options: 'success', 'fail', or None (default blue)

    Returns:
        str: Text with appropriate color applied
    """
    # If we add color styles to the text like below, it breaks outputting in Jupyter
    if is_jupyter():
        return text

    match state:
        case "success":
            return f"{Fore.GREEN}{text}{Style.RESET_ALL}"
        case "fail":
            return f"{Fore.RED}{text}{Style.RESET_ALL}"
        case "callout":
            return f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"
        case _:
            # Default to blue for normal/processing states
            return f"{Fore.BLUE}{text}{Style.RESET_ALL}"


def fancy_tqdm(
    total: int,
    desc: str = "Progress",
    color: str = BASE_OUTPUT_COLOR,
    style=1,
    postfix: str = None,
):
    """
    Creates a customized tqdm progress bar with different styling options.

    Args:
        total (int): Total iterations
        desc (str): Description for the progress bar
        color (str): Color of the progress bar (green, blue, red, yellow, magenta)
        style (int): Style preset (1-4)
        postfix (str): Postfix for the progress bar
    """

    # Style presets
    style_presets = {
        1: {
            "bar_format": "{l_bar}{bar:30}| {n_fmt}/{total_fmt} | {percentage:3.0f}% {postfix}",
            "ascii": "░▒█",
        },
        2: {
            "bar_format": "╢{l_bar}{bar:30}╟ {percentage:3.0f}%",
            "ascii": "▁▂▃▄▅▆▇█",
        },
        3: {
            "bar_format": "{desc}: |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
            "ascii": "◯◔◑◕●",
        },
        4: {
            "bar_format": "⏳ {desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}",
            "ascii": "⬜⬛",
        },
        5: {
            "bar_format": "⏳ {desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}",
            "ascii": "▏▎▍▌▋▊▉█",
        },
    }

    # Get style configuration
    style_config = style_presets.get(style, style_presets[1])

    return tqdm(
        total=total,
        desc=desc,
        colour=color,
        bar_format=style_config["bar_format"],
        ascii=style_config["ascii"],
        ncols=80,
        dynamic_ncols=True,
        smoothing=0.3,
        leave=True,
        postfix=postfix,
    )
