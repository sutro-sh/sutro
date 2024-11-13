import materialized_intelligence as mi
mi.set_base_url("https://staging.api.materialized.dev")
import polars as pl
import os

# create a stage
stage_id = mi.create_stage()

df = pl.read_csv("demo_data/sample_1000.csv")
df.write_parquet("demo_data/sample_1000.parquet")

# upload a file to the stage
mi.upload_file_to_stage(stage_id, "demo_data/sample_1000.parquet")

system_prompt = """
Extract out the name of the company from the following title.
If there is no company name, return null. Do not return a specific product if it is not a company name.
"""

results = mi.infer(stage_id, column="TITLE", system_prompt=system_prompt, model="llama-3.1-8b", job_priority=1)
os.remove("demo_data/sample_1000.parquet")
