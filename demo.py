import materialized_intelligence as mi

mi.set_base_url("https://staging.api.materialized.dev")
import polars as pl

df = pl.read_csv("demo_data/sample_10000.csv")

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

results = mi.infer(df, column="TITLE", system_prompt=system_prompt, model="llama-3.1-8b", job_priority=0, stay_attached=True)

# jobs = mi.list_jobs()
# for job in jobs:
#     print(job)