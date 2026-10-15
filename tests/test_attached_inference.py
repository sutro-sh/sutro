"""Regression tests for batch jobs that finish before progress can be streamed."""

import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import requests
import httpx

from sutro.sdk import Sutro, SutroConfigurationError


class FakeClock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def malformed_status_responses():
    return (
        ("invalid_json", requests.exceptions.JSONDecodeError("Invalid JSON", "oops", 0)),
        ("empty_root", {}),
        ("null_root", None),
        ("list_root", []),
        ("string_root", "SUCCEEDED"),
        ("number_root", 1),
        ("null_mapping", {"job_status": None}),
        ("list_mapping", {"job_status": []}),
        ("string_mapping", {"job_status": "SUCCEEDED"}),
        ("number_mapping", {"job_status": 1}),
        ("empty_mapping", {"job_status": {}}),
        ("absent_job", {"job_status": {"other-job": "SUCCEEDED"}}),
        ("null_status", {"job_status": {"job-123": None}}),
        ("list_status", {"job_status": {"job-123": ["SUCCEEDED"]}}),
        ("mapping_status", {"job_status": {"job-123": {"status": "SUCCEEDED"}}}),
        ("number_status", {"job_status": {"job-123": 1}}),
        ("boolean_status", {"job_status": {"job-123": True}}),
        ("empty_status", {"job_status": {"job-123": ""}}),
        ("unrecognized_status", {"job_status": {"job-123": "COMPLETED"}}),
    )


class TestAttachedInference(unittest.TestCase):
    def setUp(self):
        self.client = object.__new__(Sutro)
        self.client.api_url = "https://harmonize.example.test/v1"
        self.client.api_key = "test-key"
        self.client._api_url_error = None
        self.clock = FakeClock()
        self.stdout = io.StringIO()
        self.spinner = MagicMock()
        self.spinner.__enter__.return_value = self.spinner
        self.progress = MagicMock(n=0)
        self.progress.update.side_effect = self._update_progress
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(patch("sutro.sdk.yaspin", return_value=self.spinner))
        self.stack.enter_context(
            patch("sutro.sdk.fancy_tqdm", return_value=self.progress)
        )
        self.stack.enter_context(patch(
            "sutro.sdk.time",
            SimpleNamespace(
                time=lambda: self.clock.now,
                monotonic=lambda: self.clock.now,
                sleep=self.clock.sleep,
            ),
        ))
        real_async_client = httpx.AsyncClient

        def auxiliary_transport(request):
            endpoint = request.url.path.removeprefix("/v1/")
            timeouts = request.extensions["timeout"]
            response = self.requests(
                "GET", endpoint,
                timeout=(timeouts["connect"], timeouts["read"]), max_retries=0,
            )
            try:
                response_data = response.json()
            except ValueError:
                return httpx.Response(200, stream=httpx.ByteStream(b"invalid json"))
            return httpx.Response(
                response.status_code, stream=httpx.ByteStream(json.dumps(response_data).encode())
            )

        self.stack.enter_context(
            patch(
                "sutro._attached_requests.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=httpx.MockTransport(auxiliary_transport), **kwargs
                ),
            )
        )

    def _update_progress(self, amount):
        self.progress.n += amount

    def configure_job(self, statuses, events=(), failure_reason=None):
        self.spinner.write.reset_mock()
        self.statuses = list(statuses)
        self.stream = MagicMock(status_code=200)
        self.stream.__enter__.return_value = self.stream
        self.stream.iter_lines.return_value = iter(
            json.dumps(event).encode() for event in events
        )

        def request(method, endpoint, **kwargs):
            response = MagicMock(status_code=200)
            if (method, endpoint) == ("POST", "batch-inference"):
                response.json.return_value = {"results": "job-123"}
            elif (method, endpoint) == ("GET", "job-status/job-123"):
                status = self.statuses[0]
                if len(self.statuses) > 1:
                    self.statuses.pop(0)
                if isinstance(status, BaseException):
                    raise status
                response.json.return_value = {"job_status": {"job-123": status}}
            elif (method, endpoint) == ("GET", "/stream-job-progress/job-123"):
                return self.stream
            elif (method, endpoint) == ("GET", "jobs/job-123"):
                response.json.return_value = {
                    "job": {"failure_reason": failure_reason}
                }
            elif (method, endpoint) == ("POST", "job-results"):
                response.json.return_value = {
                    "results": {"outputs": ["answer-1", "answer-2"]}
                }
            else:
                self.fail(f"Unexpected request: {method} {endpoint}")
            return response

        self.requests = self.stack.enter_context(
            patch.object(self.client, "do_request", side_effect=request)
        )

    def request_count(self, endpoint):
        return sum(call.args[1] == endpoint for call in self.requests.call_args_list)

    def replace_auxiliary_status_response(self, response_data, repeat=False):
        normal_request = self.requests.side_effect
        replaced = False

        def malformed_response(method, endpoint, **kwargs):
            nonlocal replaced
            response = normal_request(method, endpoint, **kwargs)
            if endpoint == "job-status/job-123" and "timeout" in kwargs:
                if repeat or not replaced:
                    replaced = True
                    if isinstance(response_data, Exception):
                        response.json.side_effect = response_data
                    else:
                        response.json.return_value = response_data
            return response

        self.requests.side_effect = malformed_response

    def messages(self):
        return "\n".join(call.args[0] for call in self.spinner.write.call_args_list)

    def assert_success(self, data=None, **kwargs):
        if data is None:
            data = ["prompt-1", "prompt-2"]
        self.assertEqual(self.client.infer(data, **kwargs), "job-123")
        self.assertIn("answer-1", self.stdout.getvalue())
        self.assertIn("answer-2", self.stdout.getvalue())
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertEqual(self.request_count("job-results"), 1)
        submission = self.requests.call_args_list[0]
        self.assertEqual(submission.kwargs["max_retries"], 0)
        self.assertEqual(submission.kwargs["json"]["job_priority"], 0)

    def test_already_succeeded_returns_job_id_and_preview_without_streaming(self):
        self.configure_job(["SUCCEEDED"])

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 1)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 0)
        self.assertEqual(self.clock.sleeps, [])

    def test_pending_can_transition_directly_to_succeeded(self):
        self.configure_job(["PENDING", "SUCCEEDED"])

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 0)
        self.assertEqual(self.clock.sleeps, [5])

    def test_normal_running_stream_preserves_preview_and_cleans_up(self):
        self.configure_job(
            ["RUNNING", "SUCCEEDED"],
            events=[
                {"update_type": "progress", "result": 1},
                {"update_type": "tokens", "result": {"input_tokens": 12}},
                {"update_type": "progress", "result": 2},
            ],
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.progress.n, 2)
        self.progress.close.assert_called_once()
        self.stream.__exit__.assert_called_once()
        self.assertEqual(self.clock.sleeps, [])

    def test_completion_between_poll_and_stream_is_recovered(self):
        self.configure_job(["RUNNING", "SUCCEEDED"])

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.clock.sleeps, [])

    def stream_until_completion(self, events):
        for event in events:
            yield json.dumps(event).encode()
        self.fail("SDK kept reading the stream after completion")

    def test_full_progress_waits_for_persisted_success_when_stream_stays_open(self):
        self.configure_job(["RUNNING", "SUCCEEDED"])
        self.stream.iter_lines.return_value = self.stream_until_completion(
            [
                {"update_type": "progress", "result": 2},
                {"update_type": "heartbeat"},
            ]
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.stream.__exit__.assert_called_once()

    def test_full_row_progress_does_not_override_later_persisted_failure(self):
        self.configure_job(["RUNNING", "RUNNING", "FAILED"])
        self.stream.iter_lines.return_value = self.stream_until_completion(
            [
                {"update_type": "progress", "result": 2},
                {"update_type": "heartbeat"},
                {"update_type": "heartbeat"},
            ]
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 3)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.stream.__exit__.assert_called_once()

    def test_full_progress_at_eof_requires_persisted_success(self):
        self.configure_job(
            ["RUNNING", "SUCCEEDED"],
            events=[{"update_type": "progress", "result": 2}],
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.progress.n, 2)
        self.assertEqual(self.clock.sleeps, [])

    def test_full_progress_at_eof_does_not_override_persisted_failure(self):
        self.configure_job(
            ["RUNNING", "FAILED"],
            events=[{"update_type": "progress", "result": 2}],
            failure_reason={"message": "Finalization failed"},
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertIn("Failure reason: Finalization failed", self.messages())
        self.assertNotIn("Job succeeded", self.messages())

    def test_full_progress_at_eof_does_not_override_persisted_cancellation(self):
        self.configure_job(
            ["RUNNING", "CANCELLED"],
            events=[{"update_type": "progress", "result": 2}],
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertIn("Job has been cancelled", self.messages())
        self.assertNotIn("Job succeeded", self.messages())

    def test_full_progress_at_eof_does_not_fetch_results_while_still_running(self):
        self.configure_job(
            ["RUNNING"], events=[{"update_type": "progress", "result": 2}]
        )

        with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 12):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 4)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertNotIn("Job succeeded", self.messages())
        self.assertEqual(self.clock.sleeps, [5, 5, 2])
        self.assertIn("Stopped waiting for job completion", self.messages())

    def test_full_progress_at_eof_retries_transient_status_error_before_success(self):
        error = requests.HTTPError(response=MagicMock(status_code=500))
        self.configure_job(
            ["RUNNING", error, "SUCCEEDED"],
            events=[{"update_type": "progress", "result": 2}],
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 3)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.clock.sleeps, [5])

    def test_eof_waits_through_running_statuses_until_persisted_success(self):
        self.configure_job(["RUNNING", "RUNNING", "RUNNING", "SUCCEEDED"])

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 4)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.clock.sleeps, [5, 5])

    def test_full_progress_waits_longer_than_three_polls_for_finalization(self):
        self.configure_job(
            ["RUNNING"] * 6 + ["SUCCEEDED"],
            events=[{"update_type": "progress", "result": 2}],
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 7)
        self.assertEqual(self.clock.sleeps, [5] * 5)
        self.progress.close.assert_called_once()

    def test_eof_waits_through_nonterminal_states_and_transient_errors(self):
        error = requests.HTTPError(response=MagicMock(status_code=502))
        self.configure_job(
            ["RUNNING", "RUNNING", error, "STARTING", requests.Timeout(), "SUCCEEDED"]
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 6)
        self.assertEqual(self.clock.sleeps, [5] * 4)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)

    def test_eof_reports_delayed_failure_or_cancellation_after_full_progress(self):
        for terminal_status in ["FAILED", "CANCELLED"]:
            with self.subTest(status=terminal_status):
                self.configure_job(
                    ["RUNNING", "RUNNING", "RUNNING", terminal_status],
                    events=[{"update_type": "progress", "result": 2}],
                    failure_reason={"message": "Finalization failed"},
                )
                sleeps_before = len(self.clock.sleeps)

                self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                self.assertEqual(self.request_count("job-status/job-123"), 4)
                self.assertEqual(self.request_count("job-results"), 0)
                self.assertEqual(self.request_count("batch-inference"), 1)
                self.assertEqual(self.clock.sleeps[sleeps_before:], [5, 5])
                if terminal_status == "FAILED":
                    self.assertIn("Failure reason: Finalization failed", self.messages())
                else:
                    self.assertIn("Job has been cancelled", self.messages())

    def test_eof_interrupt_during_status_poll_detaches_and_cleans_up(self):
        self.configure_job(
            ["RUNNING", "RUNNING", KeyboardInterrupt(), "SUCCEEDED"],
            events=[{"update_type": "progress", "result": 2}],
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 3)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertEqual(self.clock.sleeps, [5])
        self.progress.close.assert_called_once()
        self.stream.__exit__.assert_called_once()
        self.spinner.stop.assert_called()

    def test_eof_interrupt_during_poll_delay_detaches_and_cleans_up(self):
        self.configure_job(
            ["RUNNING", "RUNNING", "SUCCEEDED"],
            events=[{"update_type": "progress", "result": 2}],
        )

        with patch("sutro.sdk.time.sleep", side_effect=KeyboardInterrupt):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.progress.close.assert_called_once()
        self.stream.__exit__.assert_called_once()
        self.spinner.stop.assert_called()

    def test_heartbeat_recovers_completion_when_final_notification_is_missed(self):
        self.configure_job(["RUNNING", "RUNNING", "SUCCEEDED"])
        self.stream.iter_lines.return_value = self.stream_until_completion(
            [{"update_type": "heartbeat"}, {"update_type": "heartbeat"}]
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 3)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.clock.sleeps, [])

    def test_heartbeat_stops_stream_after_failure_or_cancellation(self):
        for terminal_status in ["FAILED", "CANCELLED"]:
            with self.subTest(status=terminal_status):
                self.configure_job(["RUNNING", terminal_status])
                self.stream.iter_lines.return_value = self.stream_until_completion(
                    [{"update_type": "heartbeat"}]
                )

                self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                self.assertEqual(self.request_count("job-status/job-123"), 2)
                self.assertEqual(self.request_count("job-results"), 0)
                self.assertEqual(self.request_count("batch-inference"), 1)
                self.stream.__exit__.assert_called_once()
                if terminal_status == "FAILED":
                    self.assertIn("No failure details available", self.messages())
                    self.assertEqual(self.request_count("jobs/job-123"), 1)
                else:
                    self.assertIn("Job has been cancelled", self.messages())
                    self.assertEqual(self.request_count("jobs/job-123"), 0)

    def test_completed_stream_without_final_progress_uses_status(self):
        self.configure_job(
            ["STARTING", "SUCCEEDED"],
            events=[
                {"update_type": "init"},
                {"update_type": "progress", "result": 1},
            ],
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.progress.close.assert_called_once()

    def test_empty_stream_for_unfinished_job_stops_at_completion_deadline(self):
        self.configure_job(["RUNNING"])

        with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 10):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 3)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.clock.sleeps, [5, 5])
        self.assertIn("Stopped waiting for job completion", self.messages())
        self.assertIn("so.await_job_completion('job-123')", self.messages())

    def test_heartbeat_status_request_has_timeouts_and_no_internal_retries(self):
        self.configure_job(
            ["RUNNING", "SUCCEEDED"], events=[{"update_type": "heartbeat"}]
        )

        self.assert_success()

        status_calls = [
            call for call in self.requests.call_args_list
            if call.args[1] == "job-status/job-123"
        ]
        self.assertEqual(status_calls[0].kwargs, {})
        self.assertEqual(status_calls[1].kwargs["timeout"], (10, 30))
        self.assertEqual(status_calls[1].kwargs["max_retries"], 0)

    def test_eof_request_timeouts_shrink_with_remaining_completion_budget(self):
        self.configure_job(["RUNNING"])

        with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 12):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        recovery_calls = [
            call for call in self.requests.call_args_list
            if call.args[1] == "job-status/job-123"
        ][1:]
        self.assertEqual(
            [call.kwargs["timeout"] for call in recovery_calls],
            [(6, 6), (3.5, 3.5), (1, 1)],
        )
        self.assertTrue(all(call.kwargs["max_retries"] == 0 for call in recovery_calls))
        self.assertEqual(self.clock.now, 12)
        self.assertEqual(self.clock.sleeps, [5, 5, 2])

    def test_eof_uses_normal_status_timeouts_when_budget_is_large(self):
        self.configure_job(["RUNNING", "SUCCEEDED"])

        self.assert_success()

        recovery_call = next(
            call for call in self.requests.call_args_list
            if call.args[1] == "job-status/job-123" and "timeout" in call.kwargs
        )
        self.assertEqual(recovery_call.kwargs["timeout"], (10, 30))
        self.assertEqual(recovery_call.kwargs["max_retries"], 0)

    def test_exhausted_status_request_budget_does_not_send_request(self):
        self.configure_job(["SUCCEEDED"])

        self.assertIsNone(self.client._fetch_stream_job_status("job-123", timeout_budget=0))
        self.assertIsNone(self.client._fetch_stream_job_status("job-123", timeout_budget=-1))

        self.requests.assert_not_called()

    def test_request_timeout_consuming_deadline_stops_without_sleep_or_retry(self):
        self.configure_job(["RUNNING"])
        normal_request = self.requests.side_effect

        def timed_out_request(method, endpoint, **kwargs):
            if "timeout" in kwargs:
                self.assertEqual(kwargs["timeout"], (5, 5))
                self.assertEqual(kwargs["max_retries"], 0)
                self.clock.now += sum(kwargs["timeout"])
                raise requests.ReadTimeout("Status lookup timed out")
            return normal_request(method, endpoint, **kwargs)

        self.requests.side_effect = timed_out_request

        with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 10):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertEqual(self.clock.now, 10)
        self.assertEqual(self.clock.sleeps, [])
        self.assertIn("Stopped waiting for job completion", self.messages())

    def test_524_status_errors_are_retried_only_by_outer_completion_polling(self):
        error = requests.HTTPError(response=MagicMock(status_code=524))
        self.configure_job(["RUNNING", error, "SUCCEEDED"])

        self.assert_success()

        recovery_calls = [
            call for call in self.requests.call_args_list
            if call.args[1] == "job-status/job-123"
        ][1:]
        self.assertEqual(len(recovery_calls), 2)
        self.assertTrue(all(call.kwargs["max_retries"] == 0 for call in recovery_calls))
        self.assertEqual(self.clock.sleeps, [5])

    def test_request_duration_reduces_budget_for_next_status_lookup(self):
        self.configure_job(["RUNNING"])
        normal_request = self.requests.side_effect
        recovery_timeouts = []

        def delayed_request(method, endpoint, **kwargs):
            if "timeout" in kwargs:
                recovery_timeouts.append(kwargs["timeout"])
                if len(recovery_timeouts) == 1:
                    self.clock.now += 4
                else:
                    self.clock.now += sum(kwargs["timeout"])
                    raise requests.ReadTimeout("Remaining budget exhausted")
            return normal_request(method, endpoint, **kwargs)

        self.requests.side_effect = delayed_request

        with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 12):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(recovery_timeouts, [(6, 6), (1.5, 1.5)])
        self.assertEqual(self.clock.now, 12)
        self.assertEqual(self.clock.sleeps, [5])
        self.assertEqual(self.request_count("job-status/job-123"), 3)
        self.assertEqual(self.request_count("job-results"), 0)

    def test_malformed_heartbeat_status_response_recovers_on_next_heartbeat(self):
        for label, response_data in malformed_status_responses():
            with self.subTest(response=label):
                self.configure_job(["RUNNING", "RUNNING", "SUCCEEDED"])
                self.stream.iter_lines.return_value = self.stream_until_completion(
                    [{"update_type": "heartbeat"}, {"update_type": "heartbeat"}]
                )
                self.replace_auxiliary_status_response(response_data)
                sleeps_before = len(self.clock.sleeps)

                self.assert_success()

                self.assertEqual(self.request_count("job-status/job-123"), 3)
                self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
                self.assertEqual(len(self.clock.sleeps), sleeps_before)

    def test_malformed_eof_status_response_recovers_within_completion_budget(self):
        for label, response_data in malformed_status_responses():
            with self.subTest(response=label):
                self.configure_job(["RUNNING", "RUNNING", "SUCCEEDED"])
                self.replace_auxiliary_status_response(response_data)
                sleeps_before = len(self.clock.sleeps)

                self.assert_success()

                self.assertEqual(self.request_count("job-status/job-123"), 3)
                self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
                self.assertEqual(self.clock.sleeps[sleeps_before:], [5])

    def test_malformed_status_then_failure_or_cancellation_is_reported(self):
        for terminal_status in ["FAILED", "CANCELLED"]:
            for events in ([], [{"update_type": "heartbeat"}] * 2):
                with self.subTest(status=terminal_status, events=events):
                    self.configure_job(
                        ["RUNNING", "RUNNING", terminal_status],
                        events=events,
                        failure_reason={"message": "Worker failed"},
                    )
                    self.replace_auxiliary_status_response({})

                    self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                    self.assertEqual(self.request_count("job-status/job-123"), 3)
                    self.assertEqual(self.request_count("batch-inference"), 1)
                    self.assertEqual(self.request_count("job-results"), 0)
                    if terminal_status == "FAILED":
                        self.assertIn("Failure reason: Worker failed", self.messages())
                    else:
                        self.assertIn("Job has been cancelled", self.messages())

    def test_malformed_only_eof_status_responses_stop_at_completion_deadline(self):
        for label, response_data in malformed_status_responses():
            with self.subTest(response=label):
                self.configure_job(["RUNNING"])
                self.replace_auxiliary_status_response(response_data, repeat=True)
                start_time = self.clock.now
                sleeps_before = len(self.clock.sleeps)

                with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 12):
                    self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                self.assertEqual(self.clock.now - start_time, 12)
                self.assertEqual(self.clock.sleeps[sleeps_before:], [5, 5, 2])
                self.assertEqual(self.request_count("job-status/job-123"), 4)
                self.assertEqual(self.request_count("batch-inference"), 1)
                self.assertEqual(self.request_count("job-results"), 0)
                self.assertIn("Stopped waiting for job completion", self.messages())

    def test_failed_without_details_returns_none_without_crashing(self):
        self.configure_job(["FAILED"])

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("No failure details available", self.messages())
        self.assertEqual(self.request_count("job-status/job-123"), 1)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 0)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.clock.sleeps, [])

    def test_failure_details_without_message_are_safe(self):
        self.configure_job(["FAILED"], failure_reason={"code": "quota_exceeded"})

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("No failure details available", self.messages())

    def test_failure_message_is_preserved(self):
        self.configure_job(["FAILED"], failure_reason={"message": "Quota exceeded"})

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("Failure reason: Quota exceeded", self.messages())

    def test_cancelled_without_failure_details_does_not_fetch_them(self):
        self.configure_job(["CANCELLED"])

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("Job has been cancelled", self.messages())
        self.assertEqual(self.request_count("job-status/job-123"), 1)
        self.assertEqual(self.request_count("jobs/job-123"), 0)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 0)
        self.assertEqual(self.clock.sleeps, [])

    def test_wait_timeout_without_failure_reason_returns_none(self):
        self.configure_job(["PENDING"])

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.clock.now, 7200)
        self.assertEqual(self.request_count("job-status/job-123"), 1440)
        self.assertEqual(self.request_count("jobs/job-123"), 0)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 0)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertIn("Stopped waiting for job start", self.messages())

    def test_completed_job_preserves_pandas_preview_column(self):
        self.configure_job(["SUCCEEDED"])
        data = pd.DataFrame({"prompt": ["prompt-1", "prompt-2"]})

        self.assert_success(data, column="prompt", output_column="answer")

        self.assertEqual(data["answer"].tolist(), ["answer-1", "answer-2"])
        self.assertIn("Displaying result preview", self.messages())

    def test_completed_job_preserves_polars_preview_column(self):
        self.configure_job(["SUCCEEDED"])
        data = pl.DataFrame({"prompt": ["prompt-1", "prompt-2"]})

        self.assert_success(data, column="prompt", output_column="answer")

        self.assertIn("answer", self.stdout.getvalue())
        self.assertIn("Displaying result preview", self.messages())

    def test_failed_job_after_stream_does_not_fetch_results(self):
        self.configure_job(["RUNNING", "FAILED"])

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertIn("No failure details available", self.messages())
        self.assertEqual(self.request_count("jobs/job-123"), 1)

    def test_cancelled_job_after_stream_does_not_fetch_results(self):
        self.configure_job(["RUNNING", "CANCELLED"])

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertIn("Job has been cancelled", self.messages())
        self.assertEqual(self.request_count("jobs/job-123"), 0)

    def test_heartbeat_failure_reports_available_details(self):
        self.configure_job(
            ["RUNNING", "FAILED"],
            events=[{"update_type": "heartbeat"}],
            failure_reason={"message": "Worker failed during finalization"},
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("Failure reason: Worker failed during finalization", self.messages())
        self.assertEqual(self.request_count("jobs/job-123"), 1)

    def test_stream_eof_failure_reports_available_details(self):
        self.configure_job(
            ["RUNNING", "FAILED"], failure_reason={"message": "Out of memory"}
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("Failure reason: Out of memory", self.messages())
        self.assertEqual(self.request_count("jobs/job-123"), 1)

    def test_missing_message_after_stream_uses_safe_fallback(self):
        self.configure_job(["RUNNING", "FAILED"], failure_reason={"code": "worker_error"})

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("No failure details available", self.messages())

    def test_failed_job_reports_safe_fallback_when_details_request_times_out(self):
        self.configure_job(["RUNNING", "FAILED"], events=[{"update_type": "heartbeat"}])
        normal_request = self.requests.side_effect

        def timed_out_details(method, endpoint, **kwargs):
            if endpoint == "jobs/job-123":
                raise requests.Timeout("timed out")
            return normal_request(method, endpoint, **kwargs)

        self.requests.side_effect = timed_out_details

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertIn("No failure details available", self.messages())
        self.assertEqual(self.request_count("job-results"), 0)

    def test_failure_details_are_bounded_without_retries_at_every_attached_stage(self):
        stages = (
            (["FAILED"], []),
            (["RUNNING", "FAILED"], [{"update_type": "heartbeat"}]),
            (["RUNNING", "FAILED"], []),
        )
        for statuses, events in stages:
            with self.subTest(statuses=statuses, events=events):
                self.configure_job(
                    statuses, events=events, failure_reason={"message": "Worker failed"}
                )

                self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                details_call = next(
                    call for call in self.requests.call_args_list
                    if call.args[1] == "jobs/job-123"
                )
                self.assertEqual(details_call.kwargs["timeout"], (10, 30))
                self.assertEqual(details_call.kwargs["max_retries"], 0)
                self.assertEqual(self.request_count("jobs/job-123"), 1)
                self.assertIn("Failure reason: Worker failed", self.messages())

    def test_failure_detail_request_errors_fall_back_at_every_attached_stage(self):
        stages = (
            (["FAILED"], []),
            (["RUNNING", "FAILED"], [{"update_type": "heartbeat"}]),
            (["RUNNING", "FAILED"], []),
        )
        errors = (
            requests.ReadTimeout("Status details stalled"),
            requests.ConnectionError("Connection reset"),
            requests.HTTPError(response=MagicMock(status_code=500)),
            requests.HTTPError(response=MagicMock(status_code=524)),
        )
        for statuses, events in stages:
            for error in errors:
                with self.subTest(statuses=statuses, events=events, error=error):
                    self.configure_job(statuses, events=events)
                    self.spinner.write.reset_mock()
                    normal_request = self.requests.side_effect

                    def failed_details(method, endpoint, **kwargs):
                        if endpoint == "jobs/job-123":
                            self.assertEqual(kwargs["timeout"], (10, 30))
                            self.assertEqual(kwargs["max_retries"], 0)
                            raise error
                        return normal_request(method, endpoint, **kwargs)

                    self.requests.side_effect = failed_details
                    sleeps_before = len(self.clock.sleeps)

                    self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                    self.assertIn("No failure details available", self.messages())
                    self.assertEqual(self.request_count("jobs/job-123"), 1)
                    self.assertEqual(self.request_count("job-results"), 0)
                    self.assertEqual(self.request_count("batch-inference"), 1)
                    self.assertEqual(len(self.clock.sleeps), sleeps_before)

    def test_malformed_failure_detail_response_uses_safe_fallback(self):
        for response_data in (None, [], {"job": None}, {"job": []}):
            with self.subTest(response_data=response_data):
                self.configure_job(["FAILED"])
                normal_request = self.requests.side_effect

                def malformed_details(method, endpoint, **kwargs):
                    response = normal_request(method, endpoint, **kwargs)
                    if endpoint == "jobs/job-123":
                        response.json.return_value = response_data
                    return response

                self.requests.side_effect = malformed_details

                self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

                self.assertIn("No failure details available", self.messages())

    def test_generic_job_fetch_retains_its_existing_request_behavior(self):
        self.configure_job(["FAILED"], failure_reason={"message": "Worker failed"})

        self.assertEqual(self.client._get_failure_reason("job-123"), {"message": "Worker failed"})

        self.requests.assert_called_once_with("GET", "jobs/job-123")

    def test_transient_heartbeat_http_error_keeps_stream_until_later_success(self):
        for status_code in (500, 502):
            with self.subTest(status_code=status_code):
                error = requests.HTTPError(response=MagicMock(status_code=status_code))
                self.configure_job(["RUNNING", error, "SUCCEEDED"])
                self.stream.iter_lines.return_value = self.stream_until_completion(
                    [{"update_type": "heartbeat"}, {"update_type": "heartbeat"}]
                )

                self.assert_success()

                self.assertEqual(self.request_count("job-status/job-123"), 3)
                self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
                self.assertEqual(self.clock.sleeps, [])

    def test_transient_heartbeat_network_errors_keep_stream_until_later_success(self):
        for error in (requests.ConnectionError("reset"), requests.Timeout("timed out")):
            with self.subTest(error=type(error).__name__):
                self.configure_job(["RUNNING", error, "SUCCEEDED"])
                self.stream.iter_lines.return_value = self.stream_until_completion(
                    [{"update_type": "heartbeat"}, {"update_type": "heartbeat"}]
                )

                self.assert_success()

                self.assertEqual(self.request_count("job-status/job-123"), 3)
                self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
                self.assertEqual(self.clock.sleeps, [])

    def test_transient_heartbeat_error_does_not_discard_later_progress(self):
        error = requests.HTTPError(response=MagicMock(status_code=500))
        self.configure_job(
            ["RUNNING", error, "SUCCEEDED"],
            events=[
                {"update_type": "heartbeat"},
                {"update_type": "progress", "result": 2},
            ],
        )

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 3)

    def test_stream_eof_retries_transient_status_errors_with_bounded_delay(self):
        error = requests.HTTPError(response=MagicMock(status_code=502))
        self.configure_job(["RUNNING", error, requests.ConnectionError("reset"), "SUCCEEDED"])

        self.assert_success()

        self.assertEqual(self.request_count("job-status/job-123"), 4)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.clock.sleeps, [5, 5])

    def test_stream_eof_exhausted_transient_errors_stop_and_report(self):
        self.configure_job(
            ["RUNNING", requests.Timeout("timed out")],
            events=[{"update_type": "progress", "result": 2}],
        )

        with patch("sutro.sdk.ATTACHED_JOB_COMPLETION_TIMEOUT", 15):
            self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 4)
        self.assertEqual(self.request_count("/stream-job-progress/job-123"), 1)
        self.assertEqual(self.request_count("batch-inference"), 1)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.clock.sleeps, [5, 5, 5])
        self.assertIn("Stopped waiting for job completion", self.messages())

    def test_fatal_status_errors_still_propagate_from_heartbeat_and_eof(self):
        errors = [
            requests.HTTPError(response=MagicMock(status_code=401)),
            requests.HTTPError(response=MagicMock(status_code=403)),
            requests.HTTPError(response=MagicMock(status_code=404)),
            requests.exceptions.SSLError("Certificate verification failed"),
            SutroConfigurationError("Missing deployment"),
        ]
        for events in ([], [{"update_type": "heartbeat"}]):
            for error in errors:
                with self.subTest(events=events, error=error):
                    self.configure_job(["RUNNING", error], events=events)

                    with self.assertRaises(type(error)) as raised:
                        self.client.infer(["prompt-1", "prompt-2"])

                    self.assertIs(raised.exception, error)
                    self.assertEqual(self.request_count("job-status/job-123"), 2)
                    self.assertEqual(self.request_count("job-results"), 0)
                    self.stream.__exit__.assert_called_once()

    def test_heartbeat_interrupt_detaches_without_status_retries(self):
        self.configure_job(["RUNNING", KeyboardInterrupt(), "SUCCEEDED"])
        self.stream.iter_lines.return_value = self.stream_until_completion(
            [{"update_type": "heartbeat"}]
        )

        self.assertIsNone(self.client.infer(["prompt-1", "prompt-2"]))

        self.assertEqual(self.request_count("job-status/job-123"), 2)
        self.assertEqual(self.request_count("job-results"), 0)
        self.assertEqual(self.clock.sleeps, [])
        self.stream.__exit__.assert_called_once()
