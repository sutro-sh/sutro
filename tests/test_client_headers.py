"""SDK identification follows every authenticated API transport."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import runpy
from unittest.mock import Mock, patch

import pytest
import requests

from sutro.sdk import Sutro


@pytest.fixture
def client():
    result = Sutro.__new__(Sutro)
    result._api_url = "https://deployment.example.test/v1"
    result._api_url_error = None
    result.api_key = "test-key"
    return result


def response(status=200):
    result = requests.Response()
    result.status_code = status
    result._content = b"{}"
    return result


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
def test_api_requests_identify_the_installed_sdk(client, monkeypatch, method):
    send = Mock(return_value=response())
    monkeypatch.setattr(requests, method.lower(), send)

    client.do_request(method, "test-route")

    headers = send.call_args.kwargs["headers"]
    assert headers["X-Sutro-SDK-Version"] == version("sutro")
    assert headers["User-Agent"] == f"sutro-python/{version('sutro')}"
    assert headers["Authorization"] == "Key test-key"


def test_custom_headers_preserve_user_agent_and_cannot_replace_sdk_version(
    client, monkeypatch,
):
    send = Mock(return_value=response())
    monkeypatch.setattr(requests, "get", send)
    custom_headers = {
        "user-agent": "SutroBatch/custom",
        "x-sutro-sdk-version": "incorrect-version",
        "X-Request-ID": "request-1",
    }

    client.do_request("GET", "test-route", headers=custom_headers)

    headers = requests.structures.CaseInsensitiveDict(send.call_args.kwargs["headers"])
    assert headers["User-Agent"] == "SutroBatch/custom"
    assert headers["X-Sutro-SDK-Version"] == version("sutro")
    assert headers["X-Request-ID"] == "request-1"
    assert custom_headers["x-sutro-sdk-version"] == "incorrect-version"


def test_retries_and_progress_streams_keep_sdk_headers(client, monkeypatch):
    send = Mock(side_effect=[response(524), response()])
    monkeypatch.setattr(requests, "get", send)
    monkeypatch.setattr("sutro.sdk.time.sleep", lambda _: None)

    client.do_request("GET", "stream-job-progress/job-1", stream=True, max_retries=1)

    assert send.call_count == 2
    for call in send.call_args_list:
        assert call.kwargs["headers"]["X-Sutro-SDK-Version"] == version("sutro")
        assert call.kwargs["stream"] is True


@pytest.mark.parametrize("installed", [True, False])
def test_sdk_headers_use_package_metadata_with_an_uninstalled_fallback(installed):
    metadata_version = Mock(return_value="1!2.0rc1+build.4")
    if not installed:
        metadata_version.side_effect = PackageNotFoundError("sutro")
    with patch("importlib.metadata.version", metadata_version):
        module = runpy.run_path(
            str(Path(__file__).parents[1] / "sutro" / "_client_headers.py")
        )

    headers = module["sdk_request_headers"]({})

    expected = "1!2.0rc1+build.4" if installed else "unknown"
    assert headers["X-Sutro-SDK-Version"] == expected
    assert headers["User-Agent"] == f"sutro-python/{expected}"
    metadata_version.assert_called_once_with("sutro")
