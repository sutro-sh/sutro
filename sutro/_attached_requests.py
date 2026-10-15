"""Cancellable HTTP lookups used only by attached inference."""

import json
import os
import ssl
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager

import anyio
from anyio.from_thread import BlockingPortal
import httpx
import requests


MAX_AUXILIARY_RESPONSE_BYTES = 1024 * 1024


@contextmanager
def _start_attached_portal():
    """Own startup cancellation before publication as well as after handoff.

    If native startup is interrupted before the worker publishes its identity,
    cancellation may return before that worker self-cleans. Stop intent is
    already set, and no download can start before the caller receives a portal.
    """
    ready = Future()
    stop_requested = threading.Event()
    finished = threading.Event()

    async def run_portal():
        async with BlockingPortal() as portal:
            ready.set_result(portal)
            # Cancellation stays observable before/after publication and during
            # teardown; cleanup never depends on scheduling a cross-thread task.
            while not stop_requested.is_set():
                await anyio.sleep(0.01)
            await portal.stop(cancel_remaining=True)

    def run_worker():
        try:
            anyio.run(run_portal)
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
        finally:
            finished.set()

    worker = threading.Thread(target=run_worker, name="sutro-attached-lookup", daemon=True)
    interrupted = None
    try:
        worker.start()
        yield ready.result()
    finally:
        # This also covers interruption before receiving the startup Future.
        while True:
            try:
                stop_requested.set()
                break
            except KeyboardInterrupt as error:
                interrupted = error
        # Thread.start() can itself raise before launch or just after launch.
        if worker.ident is not None:
            # Wait on our own event while work is active: interruption of a
            # native Thread.join() can incorrectly mark a running thread stopped.
            while not finished.is_set():
                try:
                    finished.wait()
                except KeyboardInterrupt as error:
                    interrupted = error
            while True:
                try:
                    worker.join()
                    break
                except KeyboardInterrupt as error:
                    # Finish cleanup even if the user interrupts a second time.
                    interrupted = error
        if interrupted is not None:
            raise interrupted


def _verification_context():
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if not bundle:
        # Requests ignores SSL_CERT_FILE / SSL_CERT_DIR and defaults to certifi.
        # Keep trust_env enabled for proxies without changing certificate sources.
        return ssl.create_default_context(cafile=requests.certs.where())
    if os.path.isdir(bundle):
        return ssl.create_default_context(capath=bundle)
    return ssl.create_default_context(cafile=bundle)


def _parse_integer(value):
    # Avoid the expensive conversion of enormous integers on Python 3.10.
    if len(value) > 100:
        raise ValueError("Auxiliary JSON integer is too long")
    return int(value)


def parse_auxiliary_json(body):
    if len(body) > MAX_AUXILIARY_RESPONSE_BYTES:
        raise ValueError("Auxiliary response exceeds the size limit")
    return json.loads(body, parse_int=_parse_integer)


def _requests_response(response, body=b""):
    result = requests.Response()
    result.status_code = response.status_code
    result.headers.update(response.headers)
    result.url = str(response.url)
    result._content = body
    return result


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Auxiliary lookup deadline exceeded")


async def _download_json(url, headers, request_timeout, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Auxiliary lookup deadline exceeded")
    with anyio.fail_after(remaining):
        timeout = httpx.Timeout(
            connect=request_timeout[0],
            read=request_timeout[1],
            write=request_timeout[1],
            pool=request_timeout[0],
        )
        async with httpx.AsyncClient(timeout=timeout, verify=_verification_context()) as client:
            headers = {**headers, "Accept-Encoding": "identity"}
            request = client.build_request("GET", url, headers=headers)
            for _ in range(31):
                response = await client.send(request, stream=True, follow_redirects=False)
                try:
                    # Only legacy-configuration errors need their body for diagnosis.
                    if response.is_error and response.status_code != 410:
                        _requests_response(response).raise_for_status()
                    encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
                    if encoding not in ("", "identity"):
                        raise ValueError("Unexpected auxiliary response encoding")
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(body) + len(chunk) > MAX_AUXILIARY_RESPONSE_BYTES:
                            raise ValueError("Auxiliary response exceeds the size limit")
                        body.extend(chunk)
                    if response.next_request is not None:
                        request = response.next_request
                        continue
                    bounded_response = _requests_response(response, bytes(body))
                    _check_deadline(deadline)
                    if response.status_code == 410:
                        try:
                            diagnostic = parse_auxiliary_json(body)
                        except (ValueError, RecursionError):
                            diagnostic = None
                        _check_deadline(deadline)
                        bounded_response._auxiliary_json = diagnostic
                    bounded_response.raise_for_status()
                    result = parse_auxiliary_json(body)
                    _check_deadline(deadline)
                    return result
                finally:
                    await response.aclose()
            raise requests.TooManyRedirects("Exceeded 30 auxiliary redirects")


def get_auxiliary_json(url, headers, request_timeout):
    """Cancel network I/O at a total deadline and clean up the worker portal.

    The response is capped before synchronous JSON parsing. Parsing is bounded
    by that input limit and checked against the deadline afterward, but cannot
    be preempted and may briefly exceed the deadline on a slow CPU. OS resolver
    work and scheduler delays are likewise outside cooperative cancellation.
    Generic SDK requests do not use this transport.
    """
    deadline = time.monotonic() + sum(request_timeout)
    try:
        # A separate, joined loop also supports callers already in an event loop.
        # On KeyboardInterrupt the portal cancels its task before joining.
        with _start_attached_portal() as portal:
            return portal.call(_download_json, url, headers, request_timeout, deadline)
    except (TimeoutError, httpx.TimeoutException) as error:
        raise requests.Timeout("Auxiliary lookup deadline exceeded") from error
    except httpx.UnsupportedProtocol as error:
        raise requests.exceptions.InvalidSchema(str(error)) from error
    except httpx.LocalProtocolError as error:
        raise requests.RequestException(str(error)) from error
    except httpx.TransportError as error:
        cause = error
        while cause is not None:
            if isinstance(cause, ssl.SSLError):
                raise requests.exceptions.SSLError(str(error)) from error
            cause = cause.__cause__
        raise requests.ConnectionError(str(error)) from error
