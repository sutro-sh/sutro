#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall -q sutro tests

# Batch orchestration must execute in CI; compilation cannot verify receipts,
# retry fencing or output integrity. Keep the environment independent of uv.lock.
test_env="$(mktemp -d)"
trap 'rm -rf "$test_env"' EXIT
python3 -m venv "$test_env/venv"
"$test_env/venv/bin/python" -m pip install --disable-pip-version-check --no-deps -q \
  "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
"$test_env/venv/bin/python" -m pip install --disable-pip-version-check -q -e . pytest pytest-subtests
"$test_env/venv/bin/python" -m pytest -q \
  tests/test_batch_cli.py tests/test_batch_output.py tests/test_job_results_json.py \
  tests/test_sdk.py::TestPresignedResultsDownload \
  tests/test_function_run_polling.py tests/test_batch_image_assets.py \
  tests/test_client_headers.py tests/test_results_preparation.py \
  tests/test_trusted_publisher.py \
  tests/test_attached_requests.py::TestAttachedRequests::test_authorization_and_deployment_url_are_preserved \
  tests/test_batch_submission_errors.py \
  tests/test_sdk.py::TestRunFunction tests/test_sdk.py::TestAssetHelpers
