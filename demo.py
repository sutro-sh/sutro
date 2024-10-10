import materialized_intelligence as mi 
import polars as pl
import json

mi.set_base_url("https://staging.api.materialized.dev/")

texts = [
    "Hi, my name is John Doe. I am a software engineer at Google.",
    "Hi, my name is Jane Doe. I am a software engineer at Meta.",
    "Hi, my name is John Smith. I am a software engineer at Apple.",
    "Hi, my name is Jane Smith. I am a software engineer at Microsoft.",
    "Hi, my name is John Doe. I am a software engineer at Google.",
    "Hi, my name is Jane Doe. I am a software engineer at Meta.",
    "Hi, my name is John Smith. I am a software engineer at Apple.",
    "Hi, my name is Jane Smith. I am a software engineer at Microsoft.",
]

results = mi.infer(texts, system_prompt="Extract out the name of the company from the following title. If there is no company name, return null. Do you not return a specific product if it is not a company name.")

# json_schema = {
#     "$schema": "http://json-schema.org/draft-07/schema#",
#     "type": "object",
#     "properties": {
#         "company_name": {
#             "type": "string",
#             "description": "The name of the company"
#         }
#     },
#     "required": ["company_name"]
# }

# prompt_prefix = "Extract out the name of the company from the following title. If there is no company name, return null. Do you not return a specific product if it is not a company name."

# df = pl.read_csv("demo_data/sample_1000_company_name_golden.csv")

# # results = mi.infer(df, model="llama-3.1-70b", column="TITLE", prompt_prefix=prompt_prefix, json_schema=json_schema)

# jobs = {
#     '1b': 'job-ef839342-7a01-4e89-830c-14912a3ab0cf',
#     '3b': 'job-b30fc3b9-b32e-45f3-9b39-7c9797e9b1c1',
#     '8b': 'job-1f239859-722f-4e72-8ff4-177858c256c8',
#     '70b': 'job-9549050b-d62d-4a39-8728-fb0baa3af5a9'
# }

# for params, job_id in jobs.items():
#     results = mi.get_job_results(job_id)
#     company_names = [json.loads(result)['company_name'] if result is not None else None for result in results]

#     df = df.with_columns(pl.Series(company_names).alias("company_name_pred"))
#     df = df.with_columns(pl.col("Ground Truth").eq(pl.col("company_name_pred")).alias("is_correct"))
#     df.write_csv(f"demo_data/results/results_{params}.csv")

#     pct_correct = df["is_correct"].sum() / df["is_correct"].len()
#     print(f"Accuracy for {params} params: {pct_correct:.2f}")


