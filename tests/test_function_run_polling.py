"""Native Function polling retains one submission and cancels owned transport."""

import json
from importlib.metadata import version
from contextlib import contextmanager
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import anyio
import httpx
import pytest
import requests

from sutro.sdk import (
    FunctionRunResult,
    Sutro,
    SutroRateLimitError,
    SutroValidationError,
)


REQUEST_ID = "rt_" + "a" * 32
POLL_PATH = f"/v1/functions/image-check/runs/{REQUEST_ID}"


def pending(**changes):
    return {
        "request_id": REQUEST_ID,
        "poll_path": POLL_PATH,
        "deadline_seconds": 1800,
        "retry_after_seconds": 0.001,
        **changes,
    }


def response(status, payload, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(payload).encode()
    result.url = "https://deployment.example.test/v1/functions/image-check/run"
    return result


@pytest.fixture
def client():
    with patch("sutro.sdk.check_version"):
        return Sutro(
            api_key="private-test-key", api_url="https://deployment.example.test"
        )


@pytest.fixture
def transport():
    original = httpx.AsyncClient
    clients = []

    def install(handler):
        def create(**kwargs):
            def send(request):
                result = handler(request)
                if result.is_stream_consumed:
                    return httpx.Response(
                        result.status_code,
                        headers=result.headers,
                        stream=httpx.ByteStream(result.content),
                    )
                return result

            item = original(transport=httpx.MockTransport(send), **kwargs)
            clients.append(item)
            return item

        return patch("sutro.sdk.httpx.AsyncClient", side_effect=create)

    yield install
    assert all(item.is_closed for item in clients)
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name == "sutro-attached-lookup"
    ]


def test_async_success_polls_one_owned_run_and_preserves_result(client, transport):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Key private-test-key"
        assert request.headers["X-Sutro-SDK-Version"] == version("sutro")
        assert request.headers["User-Agent"] == f"sutro-python/{version('sutro')}"
        assert str(request.url) == "https://deployment.example.test" + POLL_PATH
        if len(calls) == 1:
            return httpx.Response(202, json=pending(state="running"))
        return httpx.Response(
            200,
            json={
                "request_id": REQUEST_ID,
                "output": {"label": "yes"},
                "confidence": 0.9,
                "usage": {"cost_usd": 0.01},
                "function": {"name": "image-check"},
            },
        )

    with (
        patch("requests.post", return_value=response(202, pending())) as post,
        transport(handle),
    ):
        result = client.run_function("image-check", {"question": "color?"})
    assert isinstance(result, FunctionRunResult)
    assert result.output == {"label": "yes"}
    assert result.confidence == 0.9
    assert result.usage == {"cost_usd": 0.01}
    assert result.request_id == REQUEST_ID
    assert [request.method for request in calls] == ["GET", "GET"]
    post.assert_called_once()
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "status,exception",
    [
        (429, SutroRateLimitError),
        (422, SutroValidationError),
        (503, requests.HTTPError),
    ],
)
def test_async_poll_errors_keep_existing_error_contract(
    client, transport, status, exception
):
    def handle(request):
        return httpx.Response(
            status,
            json={
                "request_id": REQUEST_ID,
                "code": "test_failure",
                "detail": "Cannot run",
            },
            headers={"Retry-After": "3"},
        )

    with (
        patch("requests.post", return_value=response(202, pending())) as post,
        transport(handle),
    ):
        with pytest.raises(exception) as caught:
            client.run_function("image-check", "x")
    assert caught.value.request_id == REQUEST_ID
    assert caught.value.code == "test_failure"
    assert caught.value.detail == "Cannot run"
    post.assert_called_once()
    if status == 429:
        assert caught.value.retry_after == 3


class Trickle(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        while True:
            await anyio.sleep(0.005)
            yield b" "

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("explicit", [True, False])
def test_timeout_closes_trickling_poll_and_requests_cancellation(
    client, transport, explicit
):
    stream, calls = Trickle(), []

    def handle(request):
        calls.append(request.method)
        assert request.headers["X-Sutro-SDK-Version"] == version("sutro")
        if request.method == "POST":
            return httpx.Response(202, json=pending(deadline_seconds=0.2))
        if request.method == "DELETE":
            return httpx.Response(202, json={"state": "cancelling"})
        return httpx.Response(202, stream=stream)

    start = time.monotonic()
    with (
        patch(
            "requests.post", return_value=response(202, pending(deadline_seconds=0.2))
        ),
        transport(handle),
    ):
        with pytest.raises(requests.Timeout) as caught:
            client.run_function(
                "image-check", "x", **({"timeout_seconds": 0.2} if explicit else {})
            )
    assert time.monotonic() - start < 1
    assert calls == (["POST"] if explicit else []) + ["GET", "DELETE"]
    assert stream.closed
    assert caught.value.request_id == REQUEST_ID
    assert caught.value.cancellation_requested is True


def test_post_time_consumes_shared_deadline_and_only_cancels_once(client, transport):
    calls, stream = [], Trickle()

    class DelayedResponse(httpx.AsyncByteStream):
        async def __aiter__(self):
            await anyio.sleep(0.15)
            yield json.dumps(pending()).encode()

    def handle(request):
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, stream=DelayedResponse())
        if request.method == "GET":
            return httpx.Response(202, stream=stream)
        return httpx.Response(200, json={"state": "cancelled"})

    started = time.monotonic()
    with patch("requests.post") as submit, transport(handle):
        with pytest.raises(requests.Timeout) as caught:
            client.run_function("image-check", "x", timeout_seconds=0.3)
    submit.assert_not_called()
    assert time.monotonic() - started < 0.5
    assert calls == ["POST", "GET", "DELETE"]
    assert stream.closed
    assert caught.value.cancellation_requested is True


def test_timeout_reports_failed_cancellation(client, transport):
    calls = []

    def handle(request):
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json=pending(retry_after_seconds=5))
        return httpx.Response(503, json={"detail": "unavailable"})

    with (
        patch("requests.post") as submit,
        transport(handle),
    ):
        with pytest.raises(requests.Timeout) as caught:
            client.run_function("image-check", "x", timeout_seconds=0.2)
    submit.assert_not_called()
    assert calls == ["POST", "DELETE"]
    assert caught.value.cancellation_requested is False
    assert "could not be confirmed" in str(caught.value)


def test_explicit_deadline_closes_trickled_initial_post_without_resubmitting(
    client, transport
):
    stream, calls = Trickle(), []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(202, stream=stream)

    started = time.monotonic()
    with patch("requests.post") as submit, transport(handle):
        with pytest.raises(requests.Timeout):
            client.run_function("image-check", "x", timeout_seconds=0.2)
    assert time.monotonic() - started < 1
    assert calls == ["POST"]
    assert stream.closed
    submit.assert_not_called()


def test_each_poll_has_a_short_total_deadline(client, transport, monkeypatch):
    monkeypatch.setattr("sutro.sdk.FUNCTION_RUN_POLL_REQUEST_TIMEOUT", 0.15)
    stream, calls = Trickle(), []

    def handle(request):
        calls.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(202, json={"state": "cancelling"})
        return httpx.Response(202, stream=stream)

    started = time.monotonic()
    with (
        patch(
            "requests.post", return_value=response(202, pending(deadline_seconds=10))
        ),
        transport(handle),
    ):
        with pytest.raises(requests.Timeout):
            client.run_function("image-check", "x")
    assert time.monotonic() - started < 1
    assert calls == ["GET", "DELETE"]
    assert stream.closed


@pytest.mark.parametrize(
    "poll_path",
    [
        "https://evil.example/runs/" + REQUEST_ID,
        "https://user:password@evil.example/runs/" + REQUEST_ID,
        "//evil.example/runs/" + REQUEST_ID,
        POLL_PATH + "?redirect=https://evil.example",
        POLL_PATH + "#fragment",
        POLL_PATH.replace("image-check", "other-function"),
        POLL_PATH.replace(REQUEST_ID, "rt_" + "b" * 32),
        "/v1/functions/image-check/../runs/" + REQUEST_ID,
    ],
)
def test_poll_paths_cannot_send_credentials_to_another_route(client, poll_path):
    with (
        patch(
            "requests.post", return_value=response(202, pending(poll_path=poll_path))
        ) as post,
        patch("sutro.sdk.httpx.AsyncClient") as network,
    ):
        with pytest.raises(ValueError, match="poll_path"):
            client.run_function("image-check", "x")
    post.assert_called_once()
    network.assert_not_called()


def test_redirects_are_not_followed_or_resubmitted(client, transport):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://evil.example/steal"})

    with (
        patch("requests.post", return_value=response(202, pending())) as post,
        transport(handle),
    ):
        with pytest.raises(requests.HTTPError, match="must not redirect"):
            client.run_function("image-check", "x")
    assert calls == ["https://deployment.example.test" + POLL_PATH] * 2
    post.assert_called_once()


def test_poll_connection_error_cancels_once_and_preserves_error(client, transport):
    calls = []

    def handle(request):
        calls.append(request.method)
        if request.method == "GET":
            raise httpx.ConnectError("poll connection lost", request=request)
        return httpx.Response(202, json={"state": "cancelling"})

    with (
        patch("requests.post", return_value=response(202, pending())) as post,
        transport(handle),
    ):
        with pytest.raises(requests.ConnectionError, match="poll connection lost"):
            client.run_function("image-check", "x")
    post.assert_called_once()
    assert calls == ["GET", "DELETE"]


def test_keyboard_interrupt_cancels_once_and_preserves_original(client, transport):
    interrupted = KeyboardInterrupt("original Ctrl-C")
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(202, json={"state": "cancelling"})

    with (
        patch("requests.post", return_value=response(202, pending())) as post,
        transport(handle),
        patch("sutro.sdk.time.sleep", side_effect=interrupted),
    ):
        with pytest.raises(KeyboardInterrupt) as caught:
            client.run_function("image-check", "x")
    assert caught.value is interrupted
    assert calls == ["DELETE"]
    post.assert_called_once()


def test_interrupt_during_poll_portal_startup_still_cancels(client, transport):
    from sutro._attached_requests import _start_attached_portal

    interrupted = KeyboardInterrupt("startup Ctrl-C")
    calls = []

    @contextmanager
    def interrupted_portal():
        raise interrupted
        yield

    def handle(request):
        calls.append(request.method)
        return httpx.Response(202, json={"state": "cancelling"})

    with (
        patch("requests.post", return_value=response(202, pending())) as post,
        transport(handle),
        patch(
            "sutro.sdk._start_attached_portal",
            side_effect=[interrupted_portal(), _start_attached_portal()],
        ),
    ):
        with pytest.raises(KeyboardInterrupt) as caught:
            client.run_function("image-check", "x")
    assert caught.value is interrupted
    assert calls == ["DELETE"]
    post.assert_called_once()


def test_function_name_is_quoted_and_poll_uses_frozen_deployment_key(client, transport):
    path = POLL_PATH.replace("image-check", "image%20check")
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "https://deployment.example.test" + path
        assert request.headers["Authorization"] == "Key private-test-key"
        if len(calls) == 1:
            client.set_api_url("https://new-deployment.example.test")
            return httpx.Response(202, json=pending(poll_path=path))
        return httpx.Response(200, json={"request_id": REQUEST_ID, "output": "ok"})

    with (
        patch(
            "requests.post", return_value=response(202, pending(poll_path=path))
        ) as post,
        transport(handle),
    ):
        assert client.run_function("image check", "x").output == "ok"
    assert post.call_args.args[0].endswith("/v1/functions/image%20check/run")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_never_submits(client, timeout):
    with patch("requests.post") as post:
        with pytest.raises(ValueError, match="timeout_seconds"):
            client.run_function("image-check", "x", timeout_seconds=timeout)
    post.assert_not_called()


def test_hosted_result_keeps_single_synchronous_request(client):
    payload = {"request_id": "rt_legacy", "output": "ok", "usage": {"cost_usd": 0.02}}
    with (
        patch("requests.post", return_value=response(200, payload)) as post,
        patch("sutro.sdk.httpx.AsyncClient") as network,
    ):
        assert client.run_function("image-check", "x") == payload
    post.assert_called_once()
    network.assert_not_called()


def test_native_timeout_disconnects_real_http_stream_and_cancels(client):
    connected, disconnected, cancelled = (threading.Event() for _ in range(3))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            connected.set()
            self.send_response(202)
            self.send_header("Content-Length", "10000")
            self.end_headers()
            try:
                for _ in range(1000):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()

        def do_DELETE(self):
            cancelled.set()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"state":"cancelled"}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        client.set_api_url(f"http://127.0.0.1:{server.server_port}")
        client.set_api_key("private-test-key")
        with patch(
            "requests.post", return_value=response(202, pending(deadline_seconds=0.6))
        ):
            with pytest.raises(requests.Timeout) as caught:
                client.run_function("image-check", "x")
        assert connected.is_set()
        assert disconnected.wait(1)
        assert cancelled.is_set()
        assert caught.value.cancellation_requested is True
        assert not [
            thread
            for thread in threading.enumerate()
            if thread.name == "sutro-attached-lookup"
        ]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(1)


def test_real_sigint_cancels_native_run_and_closes_poll_socket():
    # Send a real signal only to a disposable child, never the pytest process.
    script = r"""
import json, os, signal, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import requests
os.environ['LANGSMITH_TRACING'] = 'false'
with patch('requests.get', side_effect=requests.ConnectionError('Network disabled')):
    from sutro.sdk import Sutro

connected, disconnected = threading.Event(), threading.Event()
deletes = []
request_id = 'rt_' + 'a' * 32
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        connected.set()
        self.send_response(202)
        self.send_header('Content-Length', '10000')
        self.end_headers()
        try:
            for _ in range(1000):
                self.wfile.write(b' '); self.wfile.flush(); time.sleep(.005)
        except (BrokenPipeError, ConnectionResetError): disconnected.set()
    def do_DELETE(self):
        deletes.append(self.path)
        self.send_response(202); self.end_headers()
        self.wfile.write(b'{"state":"cancelling"}')
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
server.daemon_threads = True
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
def interrupt():
    if connected.wait(3): os.kill(os.getpid(), signal.SIGINT)
threading.Thread(target=interrupt, daemon=True).start()
try:
    with patch('sutro.sdk.check_version'):
        client = Sutro(api_key='fake-test-key', api_url=f'http://127.0.0.1:{server.server_port}')
    accepted = requests.Response(); accepted.status_code = 202
    accepted._content = json.dumps({
        'request_id':request_id,
        'poll_path':f'/v1/functions/image-check/runs/{request_id}',
        'deadline_seconds':5,'retry_after_seconds':.001,
    }).encode()
    with patch('requests.post', return_value=accepted) as post:
        try: client.run_function('image-check', 'x')
        except KeyboardInterrupt: pass
        else: raise AssertionError('Expected real KeyboardInterrupt')
    assert connected.is_set()
    assert disconnected.wait(1)
    assert deletes == [f'/v1/functions/image-check/runs/{request_id}']
    assert post.call_count == 1
    assert not [t for t in threading.enumerate() if t.name == 'sutro-attached-lookup']
    print(json.dumps({'interrupted':True,'cancel_requests':len(deletes),'socket_closed':True,'post_requests':post.call_count}))
finally:
    server.shutdown(); server.server_close(); worker.join(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence == {
        "interrupted": True,
        "cancel_requests": 1,
        "socket_closed": True,
        "post_requests": 1,
    }
