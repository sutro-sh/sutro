#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall -q sutro tests

# Batch orchestration must execute in CI; compilation cannot verify receipts,
# retry fencing or output integrity. Keep the environment independent of uv.lock.
test_env="$(mktemp -d)"
trap 'rm -rf "$test_env"' EXIT
python3 -m venv "$test_env/venv"
"$test_env/venv/bin/python" -m pip install --disable-pip-version-check -q -e . pytest pytest-subtests
"$test_env/venv/bin/python" -m pytest -q tests/test_batch_cli.py tests/test_batch_output.py tests/test_job_results_json.py tests/test_sdk.py::TestPresignedResultsDownload
