import materialized_intelligence as mi

mi.set_base_url("https://staging.api.materialized.dev")
import polars as pl

df = pl.read_csv("demo_data/sample_1000.csv")

system_prompt = """
Extract out the name of the company from the following title.
If there is no company name, return null. Do not return a specific product if it is not a company name.
"""

json_schema = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
        }
    },
    "required": ["company_name"]
}


results = mi.infer(df, column="TITLE", system_prompt=system_prompt, model="llama-3.1-8b", job_priority=0, json_schema=json_schema)

print(results)

# bad data example
# df = pl.read_parquet("examples/bad_data/example1.snappy.parquet")
# df = df[677971:]
# # take only the first 100000 rows of this data
# df = df[:100000]

# system_prompt = """
# You will be shown a hacker news post or comment. It will have a type, title, and/or text. Your job is to classify the sentiment of the post or comment. If can be one of the following: positive, negative, or neutral.

# An example might be: "I hate this new feature. It's not useful and it's annoying." This post is negative, so the answer is negative.

# Another example might be: "I love this company. They're doing great work." This post is positive, so the answer is positive.

# Another might be: "The new Nokia phone comes in a variety of colors." This post is neither positive nor negative, so the answer is neutral.
# """

# mi.infer(df, column="SKYSIGHT_PROMPTS", system_prompt=system_prompt, model="llama-3.1-8b", job_priority=0)
