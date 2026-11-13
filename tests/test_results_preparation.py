"""Result URL preparation has its own total deadline and never retries exports."""

import json
from contextlib import contextmanager
import threading
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock

import anyio
import httpx
import pytest
import requests

from sutro import _attached_requests
from sutro.sdk import DOWNLOAD_REQUEST_TIMEOUT, Sutro


PAYLOAD = {
    "artifact": {"filename": "results.parquet", "size_bytes": 3},
    "urls": {"head": "https://objects.test/head", "get": "https://objects.test/get"},
}


@pytest.fixture
def client():
    result = Sutro.__new__(Sutro)
    result._api_url = "https://deployment.example.test/v1"
    result._api_url_error = None
    result.api_key = "private-api-key"
    return result


class SlowBody(httpx.AsyncByteStream):
    def __init__(self, chunks, delay):
        self.chunks, self.delay, self.closed = chunks, delay, False

    async def __aiter__(self):
        for chunk in self.chunks:
            await anyio.sleep(self.delay)
            yield chunk

    async def aclose(self):
        self.closed = True


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    requests_seen, clients = [], []

    async def send(request):
        requests_seen.append(request)
        response = handler(request)
        if response.is_stream_consumed:
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                stream=httpx.ByteStream(response.content),
            )
        return response

    def create(**kwargs):
        result = original(transport=httpx.MockTransport(send), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(_attached_requests.httpx, "AsyncClient", create)
    return requests_seen, clients


def test_delayed_preparation_keeps_owner_options_and_sdk_metadata(client, monkeypatch):
    body = SlowBody([json.dumps(PAYLOAD).encode()], 0.12)
    seen, clients = transport(
        monkeypatch, lambda request: httpx.Response(200, stream=body)
    )
    result = client.results_download_url(
        "job-original",
        include_inputs=True,
        include_cumulative_logprobs=True,
        expires_in_seconds=600,
        preparation_timeout=0.5,
    )
    assert result == PAYLOAD and len(seen) == 1
    assert seen[0].url.path == "/v1/jobs/job-original/results-url"
    assert dict(seen[0].url.params) == {
        "format": "parquet",
        "include_inputs": "true",
        "include_cumulative_logprobs": "true",
        "expires_in_seconds": "600",
    }
    assert seen[0].headers["Authorization"] == "Key private-api-key"
    assert seen[0].headers["X-Sutro-SDK-Version"] == version("sutro")
    assert seen[0].headers["Accept-Encoding"] == "identity"
    assert body.closed and all(item.is_closed for item in clients)


def test_trickling_response_obeys_total_budget_and_keeps_partial_file(
    client, monkeypatch, tmp_path, capsys
):
    body = SlowBody([b" "] * 100, 0.02)
    seen, clients = transport(
        monkeypatch, lambda request: httpx.Response(200, stream=body)
    )
    partial = tmp_path / "results.parquet.part"
    partial.write_bytes(b"existing partial transfer")
    start = time.monotonic()
    assert (
        client.download_job_results(
            "job-original",
            output_path=str(tmp_path / "results.parquet"),
            preparation_timeout=0.12,
        )
        is None
    )
    assert 0.08 <= time.monotonic() - start < 1
    assert len(seen) == 1 and body.closed and all(item.is_closed for item in clients)
    assert partial.read_bytes() == b"existing partial transfer"
    assert not [t for t in threading.enumerate() if t.name == "sutro-attached-lookup"]
    output = capsys.readouterr().out
    assert "no request was retried" in output and "private-api-key" not in output


@pytest.mark.parametrize("status", [429, 500, 502, 503, 524])
def test_transient_errors_never_retry_an_expensive_export(
    client, monkeypatch, status, capsys
):
    seen, _ = transport(
        monkeypatch,
        lambda request: httpx.Response(status, json={"private": "signed-secret"}),
    )
    assert client.results_download_url("job-original") is None
    assert len(seen) == 1
    output = capsys.readouterr().out
    assert str(status) in output and "signed-secret" not in output


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_does_not_forward_credentials(client, monkeypatch, status):
    seen, _ = transport(
        monkeypatch,
        lambda request: httpx.Response(
            status, headers={"Location": "https://other.test/private"}
        ),
    )
    assert client.results_download_url("job-original") is None
    assert len(seen) == 1 and seen[0].url.host == "deployment.example.test"


@pytest.mark.parametrize(
    "timeout", [0, -1, True, None, float("nan"), float("inf"), 181]
)
def test_invalid_preparation_budget_sends_nothing(client, monkeypatch, timeout):
    send = Mock()
    monkeypatch.setattr("sutro.sdk.get_results_response", send)
    with pytest.raises(ValueError, match="preparation_timeout"):
        client.results_download_url("job-original", preparation_timeout=timeout)
    send.assert_not_called()


@pytest.mark.parametrize(
    "body", [b"invalid JSON", b"[" * 5000, b'{"value":"' + b"x" * (1024**2) + b'"}']
)
def test_bad_or_excessive_preparation_response_fails_closed(client, monkeypatch, body):
    seen, clients = transport(
        monkeypatch, lambda request: httpx.Response(200, content=body)
    )
    assert client.results_download_url("job-original", preparation_timeout=1) is None
    assert len(seen) == 1 and all(item.is_closed for item in clients)


def test_signed_body_uses_separate_unchanged_inactivity_budget(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(client, "results_download_url", Mock(return_value=PAYLOAD))
    head = Mock(headers={"Content-Length": "3", "ETag": "one"})
    head.raise_for_status.return_value = None
    response = Mock(status_code=200, headers={"Content-Length": "3", "ETag": "one"})
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.iter_content.return_value = [b"abc"]
    send_head, send_get = Mock(return_value=head), Mock(return_value=response)
    monkeypatch.setattr(requests, "head", send_head)
    monkeypatch.setattr(requests, "get", send_get)
    destination = str(tmp_path / "results.parquet")
    assert (
        client.download_job_results(
            "job-original", output_path=destination, preparation_timeout=75
        )
        == destination
    )
    assert Path(destination).read_bytes() == b"abc"
    assert client.results_download_url.call_args.kwargs["preparation_timeout"] == 75
    assert (
        send_head.call_args.kwargs["timeout"]
        == send_get.call_args.kwargs["timeout"]
        == DOWNLOAD_REQUEST_TIMEOUT
        == (10, 120)
    )
    assert "headers" not in send_head.call_args.kwargs
    assert send_get.call_args.kwargs["headers"] is None


def test_interrupt_cancels_preparation_and_joins_its_worker(client, monkeypatch):
    body = SlowBody([b" "] * 1000, 0.01)
    seen, clients = transport(
        monkeypatch, lambda request: httpx.Response(200, stream=body)
    )
    entered = threading.Event()
    original_iter = body.__class__.__aiter__

    async def started(self):
        entered.set()
        async for chunk in original_iter(self):
            yield chunk

    monkeypatch.setattr(SlowBody, "__aiter__", started)
    original_portal = _attached_requests._start_attached_portal

    @contextmanager
    def interrupted_portal():
        with original_portal() as portal:

            def call(function, *args):
                portal.start_task_soon(function, *args)
                assert entered.wait(2)
                raise KeyboardInterrupt

            portal.call = call
            yield portal

    monkeypatch.setattr(
        _attached_requests, "_start_attached_portal", interrupted_portal
    )
    with pytest.raises(KeyboardInterrupt):
        client.results_download_url("job-original")
    assert len(seen) == 1 and body.closed
    assert all(item.is_closed for item in clients)
    assert not [t for t in threading.enumerate() if t.name == "sutro-attached-lookup"]
