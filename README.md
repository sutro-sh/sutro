![Sutro Logo](./assets/sutro-logo-dark.png)
![PyPI - Version](https://img.shields.io/pypi/v/sutro)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sutro)

Use Sutro to process or generate unstructured data using AI **quickly, inexpensively, and near limitless scale**.

Whether you're generating synthetic data, running model evals, structuring unstructured data, classifying data, or generating embeddings - batch inference is better with Sutro.

Visit [sutro.sh](https://sutro.sh) to learn more and request access to the cloud beta.

## 🚀 Quickstart

Install:

```bash
[uv] pip install sutro
```

Authenticate:

```bash
sutro login
```

### Run your first job:

```python
import sutro
import pandas as pd
from pydantic import BaseModel

# Example dataset
df = pd.DataFrame({
    "review": [
        "The battery life is terrible.",
        "Great camera and build quality!",
        "Too expensive for what it offers."
    ]
})

system_prompt = """
Classify the sentiment of the review as positive, neutral, or negative.
"""

class Sentiment(BaseModel):
    sentiment: str

# Run sentiment classification at scale
df = sutro.run(
    df,
    column="review",
    model="qwen-3-32b"
    output_schema=Sentiment
)
```

### Scaling up:

```python
df = pd.read_csv("reviews_1m.csv")

df = sutro.run(
    df,
    column="review",
    model="qwen-3-32b",
    output_schema=Sentiment,
    job_priority=1 # <-- a single line of code for near-limitless scale
)
```


## What is Sutro?

Sutro is a **high-throughput batch inference service for LLM workloads**. With just a few lines of Python, you can quickly run fast inference jobs across open-source foundation models—at scale, with strong cost/time guarantees, and without worrying about infrastructure.

Think of Sutro as **online analytical processing (OLAP) for AI**: you submit queries over unstructured data (documents, emails, product reviews, etc.), and Sutro handles the heavy lifting of job execution from distributed execution to optimized LLM inference. 


## 📚 Documentation & Examples

- [Documentation](https://docs.sutro.sh/)  
- Example Guides: 
    - [Synthetic Data Zero to Hero](https://docs.sutro.sh/examples/synthetic-data-zero-to-hero)
    - [Synthetic Data for Privacy Preservation](https://docs.sutro.sh/examples/synthetic-data-privacy)
    - [Large Scale Embedding Generation with Qwen3 0.6B](https://docs.sutro.sh/examples/large-scale-embeddings)
    - More coming soon...

## ✨ Features

- **⚡ Run experiments faster**
  Small scale jobs complete in minutes, large scale jobs run within 1 hour - more than 20x faster than competing cloud services.

- **📈 Seamless scaling**  
  Use the same interface to run jobs with a few tokens, or billions at a time.

- **💰 Decreased Costs and Transparent Pricing**  
  Up to 10x cheaper than alternative inference services. Use dry run mode to estimate costs before running large jobs.

- **🐍 Pythonic DataFrame and file integrations**  
  Submit and receive results directly as Pandas/Polars DataFrames, or upload CSV/Parquet files.

- **🏗️ Zero infrastructure setup**  
  No need to manage GPUs, tune inference frameworks, or orchestrate parallelization. Just data in, results out.

- **📊 Real-time observability dashboard**
  Use the Sutro web app to monitor your jobs in real-time and see results as they are generated, tag jobs for easier tracking, and share results with your team.

- **🔒 Built with security in mind**  
  Custom data retention options, and bring-your-own s3-compatible storage options available.


## 🧑‍💻 Typical Use Cases

- **Synthetic data generation**: Create millions of product reviews, conversations, or paraphrases for pre-training or distillation.
- **Model evals**: Easily run LLM benchmarks at scale, without managing GPUs or data.
- **Unstructured data analytics**: Run analytics over unstructured data (e.g. customer reviews, product descriptions, emails, etc.).
- **Semantic tagging**: Add boolean/numeric/closed-set tags to messy data (e.g. LinkedIn bios, company descriptions).  
- **Structured Extraction**: Pull structured fields out of unstructured docs at scale.  
- **Classification**: Apply consistent labels across large datasets (spam, sentiment, topic, compliance risk).  
- **Embedding generation**: Generate and store embeddings for downstream search/analytics.  

---

## ⚙️ How It Works

1. **Prototype quickly**: Use a sample of your data, test multiple models and/or configurations, and use the Sutro web app to compare results.
2. **Scale seamlessly**: Once you're satisfied with your configuration, change a single line of code to run your job at near-limitless scale. Await job completion or trigger a callback when your job is done.
3. **Move to production**: Easily schedule your job using your favorite orchestrator, and easily build pipelines with Pythonic data integrations.  

Behind the scenes, Sutro optimizes for throughput, cost, and reliability:
- Smart batching  
- GPU scheduling, parallelization, and fault-tolerance 
- Inference framework and hardware optimizations
- Flat usage-based pricing per million tokens (input + output)

---

## 🔌 Integrations

- **DataFrames**: Pandas, Polars  
- **Files**: CSV, Parquet  
- **Storage**: S3-Compatible Object Stores (e.g. R2, S3, GCS, etc.)

---

## 🛡️ When to Use Sutro

- You want to process or generate unstructured data using foundation models **quickly, cheaply, and at scale**.
- You want a clean, **easy-to-use Python SDK and stateful job management**.
- You want **deterministic SLAs** instead of ad-hoc scripts.  
- You’d like a **query-like UX** over unstructured data.  
- You’d prefer **not** to manage GPUs, quotas, or container orchestration.  

---

## 🤝 Contributing

We welcome contributions! Please reach out to us at [team@sutro.sh](mailto:team@sutro.sh) to get involved.

---

## 📄 License

Apache 2.0