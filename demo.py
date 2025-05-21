import sutro as so
import polars as pl
from pydantic import BaseModel

so.set_base_url("https://staging.api.sutro.sh")

df = pl.read_parquet("demo_data/sample_1000.parquet")

system_prompt = """
You will be shown a hacker news post's title and text (sometimes there is no text). Your job is to classify whether the post is related to aviation. It must explicitly, directly be related to aviation. If it is related to aviation, return True, otherwise return False. Provide a justification for your answer.

An example might be Title: "The Future of Aviation" Text: "The future of aviation is electric. We will see more electric airplanes in the future." This post is related to aviation, so the answer is True.

Another example might be Title: "Haskell is the best programming language" Text: None. This post is not related to aviation, so the answer is False.
"""

# class AviationClassification(BaseModel):
#     justification: str
#     is_aviation_related: bool

json_schema = {
    "type": "object",
    "properties": {"is_aviation_related": {"type": "boolean"}},
    "required": ["is_aviation_related"],
}

results = so.infer(
    df,
    column="TITLE",
    system_prompt=system_prompt,
    model="llama-3.2-3b",
    job_priority=0,
    output_schema=json_schema,
)

# results = so.get_job_results('job-cb6cdc5a-c018-4666-9d8f-fcbc27e482a5')
print(results)