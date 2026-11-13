# Changelog

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
