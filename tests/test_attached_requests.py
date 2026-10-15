"""Transport regressions for cancellable attached-inference auxiliary lookups."""

import asyncio
import contextlib
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import anyio
import httpx
import requests

from sutro import _attached_requests
from sutro.sdk import Sutro, SutroConfigurationError
from sutro.validation import DIRECT_TENSOR_FACTORY_API_ERROR


class RecordingStream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.started = threading.Event()

    async def __aiter__(self):
        self.started.set()
        for chunk in self.chunks:
            await anyio.sleep(self.delay)
            yield chunk

    async def aclose(self):
        self.closed = True


class TestAttachedRequests(unittest.TestCase):
    def setUp(self):
        self.client = object.__new__(Sutro)
        self.client.api_url = "https://deployment.example.test"
        self.client.api_key = "private-key"
        self.client._api_url_error = None
        self.async_client = httpx.AsyncClient
        self.clients = []

    def transport(self, handler):
        def streaming_response(request):
            response = handler(request)
            if response.is_stream_consumed:
                return httpx.Response(
                    response.status_code, headers=response.headers,
                    stream=httpx.ByteStream(response.content),
                )
            return response

        def create_client(**kwargs):
            client = self.async_client(transport=httpx.MockTransport(streaming_response), **kwargs)
            self.clients.append(client)
            return client

        return patch("sutro._attached_requests.httpx.AsyncClient", side_effect=create_client)

    def portal_threads(self):
        return [t for t in threading.enumerate() if t.name == "sutro-attached-lookup"]

    def run_lifecycle_regression(self, mode):
        # Isolate startup/join races so a regression cannot hang the pytest run.
        script = r'''
import threading
import sys
from concurrent.futures import Future
from unittest.mock import patch
import anyio
from anyio.from_thread import BlockingPortal
import requests
with patch('requests.get', side_effect=requests.ConnectionError('Network disabled in regression')):
    from sutro._attached_requests import _start_attached_portal

mode = sys.argv[1]
entered = threading.Event()
release = threading.Event()
original_enter = BlockingPortal.__aenter__
original_result = Future.result
original_publish = Future.set_result
original_start = threading.Thread.start
original_join = threading.Thread.join
original_event = threading.Event
original_sleep = anyio.sleep
interrupted = False
join_interrupted = False
wait_interrupted = False
publication_interrupts = 0
event_count = 0
entered_block = False

async def held_enter(self):
    if mode == 'entry_failure':
        raise RuntimeError('portal startup failed')
    await original_enter(self)
    if mode == 'before_publication':
        entered.set()
        while not release.is_set():
            await anyio.sleep(0.001)
    return self

def held_publish(future, value):
    original_publish(future, value)
    if mode == 'handoff' and isinstance(value, BlockingPortal):
        entered.set()
        assert release.wait(1)

def interrupt_result(future, *args, **kwargs):
    global interrupted
    if mode in ('before_publication', 'handoff', 'second_interrupt'):
        if threading.current_thread() is threading.main_thread() and not interrupted:
            if mode != 'second_interrupt':
                assert entered.wait(1)
            interrupted = True
            release.set()
            raise KeyboardInterrupt
    return original_result(future, *args, **kwargs)

async def held_sleep(seconds):
    if mode in ('completion_wait', 'second_wait_interrupt') and seconds == 0.01:
        entered.set()
        while not release.is_set():
            await original_sleep(0.001)
    else:
        await original_sleep(seconds)

def guarded_event():
    global event_count
    event = original_event()
    event_count += 1
    if event_count == 1 and mode in (
        'stop_publication', 'repeated_stop_publication', 'initial_and_stop_publication'
    ):
        original_set = event.set
        def interrupted_set():
            global publication_interrupts
            required_interrupts = 1 if mode == 'stop_publication' else 2
            if publication_interrupts < required_interrupts:
                publication_interrupts += 1
                raise KeyboardInterrupt
            return original_set()
        event.set = interrupted_set
    if event_count == 2 and mode in ('completion_wait', 'second_wait_interrupt'):
        original_wait = event.wait
        def interrupted_wait(*args, **kwargs):
            global wait_interrupted
            if not wait_interrupted:
                wait_interrupted = True
                release.set()
                raise KeyboardInterrupt
            return original_wait(*args, **kwargs)
        event.wait = interrupted_wait
    return event

def interrupted_start(worker):
    if worker.name == 'sutro-attached-lookup' and mode == 'before_native_start':
        raise KeyboardInterrupt
    original_start(worker)
    if worker.name == 'sutro-attached-lookup' and mode == 'after_native_start':
        raise KeyboardInterrupt

def interrupted_join(worker, *args, **kwargs):
    global join_interrupted
    if worker.name == 'sutro-attached-lookup' and mode in ('teardown', 'second_interrupt'):
        if not join_interrupted:
            join_interrupted = True
            raise KeyboardInterrupt
    return original_join(worker, *args, **kwargs)

with patch.object(BlockingPortal, '__aenter__', held_enter), \
     patch.object(Future, 'set_result', held_publish), \
     patch.object(Future, 'result', interrupt_result), \
     patch.object(threading.Thread, 'start', interrupted_start), \
     patch.object(threading.Thread, 'join', interrupted_join), \
     patch.object(threading, 'Event', guarded_event), \
     patch.object(anyio, 'sleep', held_sleep):
    try:
        with _start_attached_portal():
            entered_block = True
            if mode == 'completion_wait':
                assert entered.wait(1)
            elif mode == 'second_wait_interrupt':
                assert entered.wait(1)
                raise KeyboardInterrupt
            elif mode == 'initial_and_stop_publication':
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        assert mode != 'entry_failure'
    except RuntimeError as error:
        assert mode == 'entry_failure' and str(error) == 'portal startup failed'
    else:
        raise AssertionError('Expected cancellation or startup error')

if mode not in (
    'teardown', 'completion_wait', 'second_wait_interrupt',
    'stop_publication', 'repeated_stop_publication', 'initial_and_stop_publication'
):
    assert not entered_block, 'Started auxiliary work after startup cancellation'
assert not [t for t in threading.enumerate() if t.name == 'sutro-attached-lookup']
if mode in ('stop_publication', 'repeated_stop_publication', 'initial_and_stop_publication'):
    assert publication_interrupts == (1 if mode == 'stop_publication' else 2)
print('cleanly stopped')
'''
        result = subprocess.run(
            [sys.executable, "-c", script, mode],
            timeout=5, check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip(), "cleanly stopped")

    def test_interrupt_before_publication_and_during_handoff_stops_worker(self):
        for mode in ("before_publication", "handoff"):
            with self.subTest(mode=mode):
                self.run_lifecycle_regression(mode)

    def test_interrupt_before_and_after_native_start_does_not_leak_or_wait_forever(self):
        for mode in ("before_native_start", "after_native_start"):
            with self.subTest(mode=mode):
                self.run_lifecycle_regression(mode)

    def test_portal_startup_failure_propagates_and_joins_worker(self):
        self.run_lifecycle_regression("entry_failure")

    def test_teardown_and_second_interrupt_finish_joining_worker(self):
        for mode in ("teardown", "second_interrupt"):
            with self.subTest(mode=mode):
                self.run_lifecycle_regression(mode)

    def test_interrupts_during_completion_event_wait_finish_cleanup(self):
        for mode in ("completion_wait", "second_wait_interrupt"):
            with self.subTest(mode=mode):
                self.run_lifecycle_regression(mode)

    def test_stop_publication_survives_initial_and_repeated_interrupts(self):
        for mode in (
            "stop_publication", "repeated_stop_publication", "initial_and_stop_publication"
        ):
            with self.subTest(mode=mode):
                self.run_lifecycle_regression(mode)

    def test_interrupted_native_start_before_identity_self_cleans_without_download(self):
        script = r'''
import threading
from unittest.mock import AsyncMock, patch
import requests
with patch('requests.get', side_effect=requests.ConnectionError('Network disabled in regression')):
    from sutro import _attached_requests

bootstrap_entered = threading.Event()
release_bootstrap = threading.Event()
original_bootstrap = threading.Thread._bootstrap_inner
original_start = threading.Thread.start
workers = []
started_waiters = []

def held_bootstrap(worker):
    if worker.name == 'sutro-attached-lookup':
        bootstrap_entered.set()
        assert release_bootstrap.wait(2)
    return original_bootstrap(worker)

def interrupted_start(worker):
    if worker.name == 'sutro-attached-lookup':
        workers.append(worker)
        started_waiters.append(worker._started.wait)
        def interrupted_started_wait(*args, **kwargs):
            assert bootstrap_entered.wait(1)
            assert worker.ident is None
            raise KeyboardInterrupt
        worker._started.wait = interrupted_started_wait
    return original_start(worker)

download = AsyncMock()
with patch.object(threading.Thread, '_bootstrap_inner', held_bootstrap), \
     patch.object(threading.Thread, 'start', interrupted_start), \
     patch.object(_attached_requests, '_download_json', download):
    try:
        _attached_requests.get_auxiliary_json('https://unused.invalid', {}, (1, 1))
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('Expected native-start interruption')
    assert workers[0].ident is None
    assert not release_bootstrap.is_set()
    download.assert_not_awaited()
    download.assert_not_called()
    release_bootstrap.set()
    assert started_waiters[0](2)
    workers[0].join(timeout=2)
    assert not workers[0].is_alive(), 'Worker did not self-clean after bootstrap'
    download.assert_not_awaited()
    download.assert_not_called()

assert not [t for t in threading.enumerate() if t.name == 'sutro-attached-lookup']
print('pending bootstrap self-cleaned; no download started')
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            timeout=5, check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip(), "pending bootstrap self-cleaned; no download started")

    def test_trickling_mock_stream_is_cancelled_and_closed(self):
        stream = RecordingStream([b" "] * 100, delay=0.01)
        with self.transport(lambda request: httpx.Response(200, stream=stream)):
            started = time.monotonic()
            with self.assertRaises(requests.Timeout):
                _attached_requests.get_auxiliary_json("https://example.test", {}, (0.05, 0.05))

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(stream.closed)
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertEqual(self.portal_threads(), [])

    def test_status_trickle_uses_remaining_completion_budget(self):
        stream = RecordingStream([b" "] * 100, delay=0.01)
        with self.transport(lambda request: httpx.Response(200, stream=stream)):
            started = time.monotonic()
            result = self.client._fetch_stream_job_status("job-123", timeout_budget=0.1)

        self.assertIsNone(result)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(stream.closed)
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertEqual(self.portal_threads(), [])

    def test_failure_detail_trickle_falls_back_and_closes(self):
        stream = RecordingStream([b" "] * 100, delay=0.01)
        spinner = MagicMock()
        with self.transport(lambda request: httpx.Response(200, stream=stream)), patch(
            "sutro.sdk.ATTACHED_JOB_LOOKUP_REQUEST_TIMEOUT", (0.05, 0.05)
        ):
            self.client._report_attached_job_failure("job-123", "FAILED", spinner)

        self.assertIn("No failure details available", spinner.write.call_args.args[0])
        self.assertTrue(stream.closed)
        self.assertEqual(self.portal_threads(), [])

    def test_body_size_is_bounded_before_json_parsing_and_response_closes(self):
        stream = RecordingStream([b"x" * 9])
        with self.transport(lambda request: httpx.Response(200, stream=stream)), patch(
            "sutro._attached_requests.MAX_AUXILIARY_RESPONSE_BYTES", 8
        ), patch("sutro._attached_requests.json.loads") as loads:
            self.assertIsNone(self.client._fetch_stream_job_status("job-123"))

        loads.assert_not_called()
        self.assertTrue(stream.closed)
        self.assertEqual(self.portal_threads(), [])

    def test_bounded_parse_that_crosses_deadline_is_rejected(self):
        real_loads = json.loads

        def slow_parse(*args, **kwargs):
            time.sleep(0.12)
            return real_loads(*args, **kwargs)

        with self.transport(lambda request: httpx.Response(200, json={"ok": True})), patch(
            "sutro._attached_requests.json.loads", side_effect=slow_parse
        ):
            with self.assertRaises(requests.Timeout):
                _attached_requests.get_auxiliary_json("https://example.test", {}, (0.05, 0.05))

        self.assertEqual(self.portal_threads(), [])

    def test_lookup_works_for_caller_already_in_event_loop(self):
        async def caller():
            return self.client._fetch_stream_job_status("job-123")

        with self.transport(
            lambda request: httpx.Response(200, json={"job_status": {"job-123": "SUCCEEDED"}})
        ):
            self.assertEqual(asyncio.run(caller()), "SUCCEEDED")

        self.assertEqual(self.portal_threads(), [])

    def test_keyboard_interrupt_cancels_request_and_joins_portal(self):
        stream = RecordingStream([b" "] * 100, delay=0.01)
        real_result = Future.result

        def interrupted_result(future, *args, **kwargs):
            if stream.started.is_set() and not future.done():
                raise KeyboardInterrupt
            return real_result(future, *args, **kwargs)

        # Interrupt the blocking bridge after its download task has started.
        with self.transport(lambda request: httpx.Response(200, stream=stream)), patch(
            "concurrent.futures.Future.result", interrupted_result
        ):
            # Synchronize once before injecting the interrupt into task waiting.
            original_call = _attached_requests._start_attached_portal

            @contextlib.contextmanager
            def interrupting_portal(*args, **kwargs):
                with original_call(*args, **kwargs) as portal:
                    original_portal_call = portal.call

                    def call(function, *call_args):
                        if function is _attached_requests._download_json:
                            future = portal.start_task_soon(function, *call_args)
                            self.assertTrue(stream.started.wait(1))
                            return future.result()
                        return original_portal_call(function, *call_args)

                    portal.call = call
                    yield portal

            with patch("sutro._attached_requests._start_attached_portal", interrupting_portal):
                with self.assertRaises(KeyboardInterrupt):
                    self.client._fetch_stream_job_status("job-123")

        self.assertTrue(stream.closed)
        self.assertEqual(self.portal_threads(), [])

    def test_repeated_cancelled_requests_leave_no_portal_threads(self):
        for _ in range(3):
            stream = RecordingStream([b" "] * 100, delay=0.005)
            with self.transport(lambda request: httpx.Response(200, stream=stream)):
                self.assertIsNone(self.client._fetch_stream_job_status("job-123", timeout_budget=0.05))
            self.assertTrue(stream.closed)
            self.assertEqual(self.portal_threads(), [])

    def test_authorization_and_deployment_url_are_preserved(self):
        seen = []

        def response(request):
            seen.append(request)
            return httpx.Response(200, json={"job_status": {"job-123": "SUCCEEDED"}})

        with self.transport(response):
            self.assertEqual(self.client._fetch_stream_job_status("job-123"), "SUCCEEDED")

        self.assertEqual(str(seen[0].url), "https://deployment.example.test/v1/job-status/job-123")
        self.assertEqual(seen[0].headers["Authorization"], "Key private-key")
        self.assertEqual(seen[0].headers["Accept-Encoding"], "identity")

    def test_unexpected_compression_is_rejected_before_decoding(self):
        stream = RecordingStream([b"compressed data"])
        with self.transport(
            lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)
        ), patch("sutro._attached_requests.json.loads") as loads:
            self.assertIsNone(self.client._fetch_stream_job_status("job-123"))
        loads.assert_not_called()
        self.assertFalse(stream.started.is_set())
        self.assertTrue(stream.closed)

    def test_redirect_bodies_are_bounded_and_cross_origin_key_is_stripped(self):
        seen = []

        def redirected_response(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(302, headers={"Location": "https://other.example.test/status"})
            return httpx.Response(200, json={"job_status": {"job-123": "SUCCEEDED"}})

        with self.transport(redirected_response):
            self.assertEqual(self.client._fetch_stream_job_status("job-123"), "SUCCEEDED")
        self.assertEqual(len(seen), 2)
        self.assertNotIn("Authorization", seen[1].headers)

    def test_large_redirect_body_is_rejected_before_following(self):
        stream = RecordingStream([b"x" * 9])
        seen = []

        def redirected_response(request):
            seen.append(request)
            return httpx.Response(302, headers={"Location": "/next"}, stream=stream)

        with self.transport(redirected_response), patch(
            "sutro._attached_requests.MAX_AUXILIARY_RESPONSE_BYTES", 8
        ):
            self.assertIsNone(self.client._fetch_stream_job_status("job-123"))
        self.assertEqual(len(seen), 1)
        self.assertTrue(stream.closed)

    def test_legacy_direct_api_error_preserves_configuration_exception(self):
        with self.transport(
            lambda request: httpx.Response(410, json={"detail": DIRECT_TENSOR_FACTORY_API_ERROR})
        ):
            with self.assertRaisesRegex(SutroConfigurationError, "Direct Sutro Batch API"):
                self.client._fetch_stream_job_status("job-123")

    def test_legacy_error_diagnosis_uses_bounded_integer_parser(self):
        body = (
            '{"detail": ' + json.dumps(DIRECT_TENSOR_FACTORY_API_ERROR)
            + ', "huge": ' + "1" * 101 + '}'
        ).encode()
        with self.transport(lambda request: httpx.Response(410, content=body)):
            with self.assertRaises(requests.HTTPError) as raised:
                self.client._fetch_stream_job_status("job-123")
        self.assertEqual(raised.exception.response.status_code, 410)

    def test_legacy_error_diagnosis_stays_within_parse_deadline(self):
        real_loads = json.loads

        def slow_parse(*args, **kwargs):
            time.sleep(0.12)
            return real_loads(*args, **kwargs)

        with self.transport(
            lambda request: httpx.Response(410, json={"detail": DIRECT_TENSOR_FACTORY_API_ERROR})
        ), patch("sutro._attached_requests.json.loads", side_effect=slow_parse):
            with self.assertRaises(requests.Timeout):
                _attached_requests.get_auxiliary_json("https://example.test", {}, (0.05, 0.05))
        self.assertEqual(self.portal_threads(), [])

    def test_deep_failure_detail_json_uses_missing_details_fallback(self):
        spinner = MagicMock()
        body = b"[" * 2000 + b"0" + b"]" * 2000
        with self.transport(lambda request: httpx.Response(200, content=body)):
            self.client._report_attached_job_failure("job-123", "FAILED", spinner)
        self.assertIn("No failure details available", spinner.write.call_args.args[0])

    def test_certificate_errors_keep_their_fatal_error_type(self):
        def invalid_certificate(request):
            try:
                raise ssl.SSLCertVerificationError("Certificate verification failed")
            except ssl.SSLError as error:
                raise httpx.ConnectError("TLS failed") from error

        with self.transport(invalid_certificate):
            with self.assertRaises(requests.exceptions.SSLError):
                self.client._fetch_stream_job_status("job-123")

    def test_fatal_http_authentication_errors_propagate(self):
        with self.transport(lambda request: httpx.Response(401)):
            with self.assertRaises(requests.HTTPError) as raised:
                self.client._fetch_stream_job_status("job-123")
        self.assertEqual(raised.exception.response.status_code, 401)

    def test_requests_and_curl_custom_ca_bundles_have_expected_precedence(self):
        real_context = ssl.create_default_context
        for environment, expected in (
            ({"REQUESTS_CA_BUNDLE": "requests.pem", "CURL_CA_BUNDLE": "curl.pem"}, "requests.pem"),
            ({"CURL_CA_BUNDLE": "curl.pem"}, "curl.pem"),
        ):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), patch(
                "sutro._attached_requests.ssl.create_default_context", return_value=real_context()
            ) as factory:
                _attached_requests._verification_context()
                factory.assert_called_once_with(cafile=expected)

    def test_custom_ca_directory_is_preserved(self):
        with patch.dict(os.environ, {"REQUESTS_CA_BUNDLE": "/custom/certs"}, clear=True), patch(
            "sutro._attached_requests.os.path.isdir", return_value=True
        ), patch("sutro._attached_requests.ssl.create_default_context") as factory:
            _attached_requests._verification_context()
        factory.assert_called_once_with(capath="/custom/certs")

    def test_default_ca_preserves_requests_semantics_with_ssl_cert_environment(self):
        with patch.dict(
            os.environ,
            {"SSL_CERT_FILE": "/missing.pem", "SSL_CERT_DIR": "/missing-certs"},
            clear=True,
        ), self.transport(
            lambda request: httpx.Response(200, json={"job_status": {"job-123": "SUCCEEDED"}})
        ):
            self.assertEqual(self.client._fetch_stream_job_status("job-123"), "SUCCEEDED")

        self.assertEqual(self.portal_threads(), [])

    def test_real_local_trickling_http_body_is_cancelled(self):
        disconnected = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    disconnected.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1/job-status/job-123"
            started = time.monotonic()
            with self.assertRaises(requests.Timeout):
                _attached_requests.get_auxiliary_json(url, {}, (0.05, 0.05))
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(disconnected.wait(1))
            self.assertEqual(self.portal_threads(), [])
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(1)


if __name__ == "__main__":
    unittest.main()
