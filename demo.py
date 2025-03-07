from materialized_intelligence import MaterializedIntelligence

mi = MaterializedIntelligence()
mi.set_base_url("https://cooper-test.aaterialized.dev")
import polars as pl

sid = mi.list_jobs()