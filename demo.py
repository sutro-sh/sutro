import materialized_intelligence as mi
mi.set_base_url("https://staging.api.materialized.dev")
# import polars as pl

system_prompt = """
Extract out the name of the company from the following title.
If there is no company name, return null. Do not return a specific product if it is not a company name.
"""

results = mi.infer("demo_data/sample_1000.csv", column="TITLE", system_prompt=system_prompt, model="llama-3.2-3b")

# poll for job status with a 1 second delay and a 2 hour timeout
# start_time = time.time()
# while True and (time.time() - start_time) < 2 * 60 * 60:
#     status = mi.get_job_status(job_id)['job_status'][job_id]
#     print(status)
#     if status == "SUCCEEDED":
#         results = mi.get_job_results(job_id)
#         print(results)
#         break
#     if status == "FAILED":
#         raise Exception("Job failed")
#     time.sleep(10)
    
    

# get the results
# results = mi.get_job_results("job-c7f010c8-6cf1-404b-908a-a826ce023348", include_inputs=True)
# results = pl.DataFrame(results)

# for row in results.iter_rows(named=True):
#     print(row['inputs'])
#     print(row['outputs'])
#     print()
