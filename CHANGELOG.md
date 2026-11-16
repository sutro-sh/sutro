# Changelog

## 0.3.0

Breaking changes:

- Remove the `embed()`, `classify()`, `score()`, `rank()` and `elo()` templates.
  Use `infer()` with an embedding model or a prompt instead.
- Remove `gpt-oss-20b`, `gpt-oss-120b`, their `-no-thinking` variants, the
  Nemotron 3 models, `qwen-3-4b-base` and `qwen-3-30b-a3b`. Requests for them
  now fail with `Model not found`.
- Drop `qwen-3-embedding-4b`, which the service never served, and the hidden
  `qwen-3.5-2b` test models from `ModelOptions`.
- `infer()` defaults to `gemma-4-26b-a4b-it` instead of `gpt-oss-20b`.

## 0.2.0

- Give result URL preparation its own 180-second total deadline, configurable
  downward with `preparation_timeout` on `results_download_url` and
  `download_job_results`. Slow headers and trickling metadata share that budget.
- Send one authenticated cached results GET. Do not retry export preparation
  after HTTP errors or timeouts, follow deployment redirects, or recreate jobs.
- Preserve signed-file transfer inactivity limits and ETag-validated partial
  resume. Return a clear preparation failure while retaining existing work.
- Include the release's batch-disabled deployment policy and recovery behavior.

This release supersedes the unpublished 0.1.71 release candidate. Services keep
the existing results URL protocol; older SDKs retain their 45-second wait and
need an upgrade for large cold downloads with inputs.
