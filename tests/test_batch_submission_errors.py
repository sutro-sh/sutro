"""Batch submission preserves HTTP failures without replay or secret diagnostics."""

import json
import traceback
from unittest.mock import Mock

import pytest
import requests

from sutro.sdk import Sutro
from sutro import _batch
import time


SECRET_BODY = "private-response-body-sentinel"
SECRET_URL = "private-signed-url-sentinel"
SECRET_AUTH = "private-authorization-sentinel"


@pytest.fixture
def client(monkeypatch):
    # Construct only the explicit configuration used by do_request. Avoid the
    # constructor's unrelated package-version request and ambient config reads.
    result = Sutro.__new__(Sutro)
    result._api_url = "https://deployment.example.test/v1"
    result._api_url_error = None
    result.api_key = SECRET_AUTH
    monkeypatch.setattr(
        "sutro.sdk.time.sleep", Mock(side_effect=AssertionError("No retry"))
    )
    monkeypatch.setattr("requests.get", Mock(side_effect=AssertionError("No poll")))
    monkeypatch.setattr(
        "requests.delete", Mock(side_effect=AssertionError("No cancel"))
    )
    monkeypatch.setattr("sutro.sdk._create_batch_traces", Mock(return_value=False))
    return result


def response(status, body, content_type):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result.headers["Content-Type"] = content_type
    result.url = (
        f"https://deployment.example.test/v1/batch-inference?signature={SECRET_URL}"
    )
    result.request = requests.Request(
        "POST", result.url, headers={"Authorization": f"Key {SECRET_AUTH}"}
    ).prepare()
    return result


def submit(client):
    return client.batch_run_function(
        "image-function", [{"question": "Describe"}], stay_attached=False
    )


@pytest.mark.parametrize(
    ("status", "body", "content_type"),
    [
        (500, SECRET_BODY.encode(), "text/plain"),
        (524, f"<html>{SECRET_BODY}</html>".encode(), "text/html"),
    ],
)
def test_non_json_http_error_preserves_status_without_replaying_or_leaking(
    client, monkeypatch, capsys, status, body, content_type
):
    failed = response(status, body, content_type)
    post = Mock(return_value=failed)
    monkeypatch.setattr("requests.post", post)

    with pytest.raises(requests.HTTPError) as raised:
        submit(client)

    error = raised.value
    assert type(error) is requests.HTTPError
    assert error.response is failed
    assert error.response.status_code == status
    assert str(status) in str(error)
    assert error.__suppress_context__ is True
    diagnostics = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    captured = capsys.readouterr()
    diagnostics += captured.out + captured.err + repr(error)
    for secret in (SECRET_BODY, SECRET_URL, SECRET_AUTH):
        assert secret not in diagnostics
    assert "JSONDecodeError" not in diagnostics
    post.assert_called_once()
    assert (
        post.call_args.args[0] == "https://deployment.example.test/v1/batch-inference"
    )


@pytest.mark.parametrize("status", [500, 524])
def test_json_http_error_keeps_existing_error_details_and_no_retry(
    client, monkeypatch, capsys, status
):
    failed = response(
        status,
        json.dumps({"error": "synthetic rejection"}).encode(),
        "application/json",
    )
    post = Mock(return_value=failed)
    monkeypatch.setattr("requests.post", post)

    assert submit(client) is None

    assert "synthetic rejection" in capsys.readouterr().out
    post.assert_called_once()


def test_successful_json_submission_returns_the_original_job(client, monkeypatch):
    accepted = response(200, b'{"results":"job-synthetic"}', "application/json")
    post = Mock(return_value=accepted)
    monkeypatch.setattr("requests.post", post)

    assert submit(client) == "job-synthetic"
    post.assert_called_once()


def test_http_error_without_response_is_not_replaced_by_an_attribute_error(
    client, monkeypatch
):
    error = requests.HTTPError("Synthetic response unavailable")
    post = Mock(side_effect=error)
    monkeypatch.setattr("requests.post", post)

    with pytest.raises(requests.HTTPError) as raised:
        submit(client)

    assert raised.value is error
    post.assert_called_once()


@pytest.mark.parametrize("priority", [0, 1])
def test_disabled_batch_policy_is_explained_without_retry_or_cancellation(
    client, monkeypatch, capsys, priority
):
    message = (
        "New batch inference is disabled for this deployment. "
        "Existing jobs and results remain available."
    )
    failed = response(403, json.dumps({"detail": message}).encode(), "application/json")
    failed.headers["X-Sutro-Error-Code"] = "batch_inference_disabled"
    post = Mock(return_value=failed)
    monkeypatch.setattr("requests.post", post)
    assert (
        client.batch_run_function(
            "function",
            [{"question": "row"}],
            stay_attached=False,
            job_priority=priority,
        )
        is None
    )
    assert message in capsys.readouterr().out
    post.assert_called_once()


@pytest.mark.parametrize("priority", [0, 1])
def test_resumable_batch_explains_policy_and_preserves_submission_key(
    client, monkeypatch, priority
):
    failed = response(403, b'{"detail":"Batch unavailable"}', "application/json")
    failed.headers["X-Sutro-Error-Code"] = "batch_inference_disabled"
    post = Mock(return_value=failed)
    monkeypatch.setattr("requests.post", post)
    key = "00000000-0000-4000-8000-000000000002"
    body = {"idempotency_key": key, "job_priority": priority}
    with pytest.raises(
        _batch.ApiError, match="New batch inference is disabled"
    ) as error:
        _batch._submit_with_key(
            client,
            key,
            lambda: body,
            time.monotonic() + 60,
            lambda message: None,
            submit_first=True,
        )
    assert error.value.code == "batch_inference_disabled"
    assert "Check deployment access permissions" not in str(error.value)
    assert body["idempotency_key"] == key
    post.assert_called_once()
