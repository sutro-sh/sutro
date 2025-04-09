import materialized_intelligence as mi
import polars as pl


df = pl.read_parquet("hackernews_stories_chunk_0.parquet")

mi = mi.MaterializedIntelligence()
mi.set_base_url("https://cooper-test.api.materialized.dev")

# stage_id = mi.create_stage()

stage_id = 'stage-c7787aa9-8a63-4092-99bd-f38e76ffb377'
# mi.upload_to_stage(stage_id, "hackernews_stories_chunk_0.parquet")

system_prompt = """
You will be shown a hacker news post's title and text (sometimes there is no text). Your job is to classify whether the post is related to aviation. It must explicitly, directly be related to aviation. If it is related to aviation, return True, otherwise return False. Provide a justification for your answer.

An example might be Title: "The Future of Aviation" Text: "The future of aviation is electric. We will see more electric airplanes in the future." This post is related to aviation, so the answer is True.

Another example might be Title: "Haskell is the best programming language" Text: None. This post is not related to aviation, so the answer is False.
"""

json_schema = {
    "type": "object",
    "properties": {
        "justification": {"type": "string"},
        "is_aviation_related": {"type": "boolean"},
    },
    "required": ["justification", "is_aviation_related"]
}

results = mi.infer(
    df.slice(0, 200),
    column="input",
    system_prompt=system_prompt,
    model="qwen-qwq-32b-8k",
    json_schema=json_schema,
    job_priority=0,
)

print(results)

# jobs = mi.list_jobs()
# for job in jobs:
#     print(job)
