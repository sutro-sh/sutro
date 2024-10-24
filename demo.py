import materialized_intelligence as mi 

system_prompt = """
Extract out the name of the company from the following title.
If there is no company name, return null. Do not return a specific product if it is not a company name.
"""

json_schema = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "minLength": 1
        }
    },
    "required": ["company_name"]
}

results = mi.infer("demo_data/sample_1000_company_name_golden.csv", column="TITLE", system_prompt=system_prompt, json_schema=json_schema)

print(results)
