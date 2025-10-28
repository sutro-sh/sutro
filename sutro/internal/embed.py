from typing import Union, List, TYPE_CHECKING
import polars as pl
import pandas as pd
from yaspin import yaspin

if TYPE_CHECKING:
    from ..sdk import Sutro, make_clickable_link
from sutro.internal.common import BatchInferenceRequest, _do_batch_inference_request
from sutro.sdk import to_colored_text, ModelOptions, SPINNER, YASPIN_COLOR


def _embed_simple(
    client: "Sutro",
    data: Union[List, pd.DataFrame, pl.DataFrame, str],
    model: ModelOptions = "qwen-3-embedding-0.6b",
    name: Union[str, List[str]] = None,
    description: Union[str, List[str]] = None,
    column: Union[str, List[str]] = None,
    job_priority: int = 0,
    truncate_rows: bool = True,
):
    req = BatchInferenceRequest(
        model=model,
        data=data,
        column=column,
        job_priority=job_priority,
        truncate_rows=truncate_rows,
        name=name,
        description=description,
    )

    try:
        spinner_text = to_colored_text(f"Creating priority {job_priority} job")
        with yaspin(SPINNER, text=spinner_text, color=YASPIN_COLOR) as spinner:
            # TODO correctly handle spinner text on error
            result = _do_batch_inference_request(client, req)
            job_id = result["results"]

            name_text = f" and name {name}" if name is not None else ""
            spinner.write(
                to_colored_text(
                    f"🛠 Priority {job_priority} Job created with ID: {job_id}{name_text}.",
                    state="success",
                )
            )
            clickable_link = make_clickable_link(f"https://app.sutro.sh/jobs/{job_id}")
            spinner.write(
                to_colored_text(
                    f"Use `so.get_job_status('{job_id}')` to check the status of the job, or monitor progress at {clickable_link}"
                )
            )

            return job_id
    except KeyboardInterrupt:
        pass
    finally:
        if spinner:
            spinner.stop()
