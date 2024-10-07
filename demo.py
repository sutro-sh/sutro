import materialized_intelligence as mi 

results = mi.infer("demo_data/sample_1000.csv", model="llama-3.2-3b", column="TITLE", prompt_prefix="Extract out the name of the company from the following title:")

print(results)