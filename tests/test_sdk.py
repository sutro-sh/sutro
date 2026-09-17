import unittest
import time
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import shutil
import sys
import io
import tempfile

import pandas as pd
import requests

from colorama import Fore, Style

from pydantic import BaseModel

from sutro.assets import Asset, Image
from sutro.sdk import (
    FUNCTION_RUN_REQUEST_TIMEOUT,
    Sutro,
    SutroRateLimitError,
    SutroValidationError,
)
from sutro.common import to_colored_text, prepare_input_data


class TestSutro(unittest.TestCase):
    def setUp(self):
        # Create an instance of Sutro with a dummy API key
        self.so = Sutro(
            api_key="test_api_key",
            api_url="https://harmonize.example.test",
        )

        # Setup capture of stdout for testing console output
        self.stdout_capture = io.StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout_capture

    def tearDown(self):
        # Reset stdout
        sys.stdout = self.old_stdout

    @patch("requests.post")
    def test_infer_success(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": ["result1", "result2"],
            "metadata": {"job_id": "test_job_id"},
        }
        mock_post.return_value = mock_response

        # Call the method
        result = self.so.infer(["input1", "input2"])

        # Check that the API was called correctly
        mock_post.assert_called_once()

        print(result, flush=True)

        # Verify results
        self.assertEqual(result, ["result1", "result2"])

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Materialized results received", output)
        self.assertIn(
            f"You can re-obtain the results with `so.get_job_results('test_job_id')`",
            output,
        )

    @patch("requests.post")
    def test_infer_failure(self, mock_post):
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_post.return_value = mock_response

        # Call the method
        result = self.so.infer(["input1", "input2"])

        # Verify the result is None for failed request
        self.assertIsNone(result)

        # Check output for error message
        output = self.stdout_capture.getvalue()
        self.assertIn(f"Error: 400", output)
        self.assertTrue(
            "{'error': 'Bad request'}" in output or '{"error": "Bad request"}' in output
        )

    @patch("requests.get")
    def test_list_jobs_success(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": ["job1", "job2"]}
        mock_get.return_value = mock_response

        # Call the method
        result = self.so.list_jobs()

        # Verify results
        self.assertEqual(result, ["job1", "job2"])

    @patch("requests.get")
    def test_list_jobs_failure(self, mock_get):
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}
        mock_get.return_value = mock_response

        # Call the method
        result = self.so.list_jobs()

        # Verify the result is None for failed request
        self.assertIsNone(result)

        # Check output for error message
        output = self.stdout_capture.getvalue()
        self.assertIn(f"Bad status code: 401", output)

    @patch("requests.get")
    def test_get_job_status_success(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"job_status": {"test_job_id": "COMPLETED"}}
        mock_get.return_value = mock_response

        # Call the method
        result = self.so.get_job_status("test_job_id")

        # Verify results
        self.assertEqual(result, "COMPLETED")

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Job status retrieved!", output)

    @patch("requests.post")
    def test_get_job_results_success(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": ["result1", "result2"]}
        mock_post.return_value = mock_response

        # Call the method
        result = self.so.get_job_results("test_job_id")

        # Verify results
        self.assertEqual(result, ["result1", "result2"])

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Job results retrieved", output)

    @patch("requests.get")
    def test_cancel_job_success(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "CANCELLED"}
        mock_get.return_value = mock_response

        # Call the method
        result = self.so.cancel_job("test_job_id")

        # Verify results
        self.assertEqual(result, {"status": "CANCELLED"})

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Job cancelled", output)

    # Test for color output formatting
    def test_to_colored_text(self):
        # Test success state
        success_text = to_colored_text("Success message", state="success")
        self.assertEqual(success_text, f"{Fore.GREEN}Success message{Style.RESET_ALL}")

        # Test fail state
        fail_text = to_colored_text("Fail message", state="fail")
        self.assertEqual(fail_text, f"{Fore.RED}Fail message{Style.RESET_ALL}")

        # Test default (blue) state
        default_text = to_colored_text("Default message")
        self.assertEqual(default_text, f"{Fore.BLUE}Default message{Style.RESET_ALL}")


class TestRequestRetries(unittest.TestCase):
    def setUp(self):
        self.so = Sutro(
            api_key="test_api_key",
            api_url="https://harmonize.example.test",
        )

    @patch("requests.post")
    def test_batch_submission_is_not_retried_after_524(self, mock_post):
        timeout_response = MagicMock()
        timeout_response.status_code = 524
        timeout_response.json.return_value = {"error": "Cloudflare timeout"}
        timeout_response.raise_for_status.side_effect = requests.HTTPError(
            response=timeout_response
        )
        mock_post.return_value = timeout_response

        result = self.so.infer(["input"], stay_attached=False)

        self.assertIsNone(result)
        mock_post.assert_called_once()

    @patch("requests.get")
    def test_zero_retry_budget_reraises_initial_524(self, mock_get):
        timeout_response = MagicMock()
        timeout_response.status_code = 524
        timeout_error = requests.HTTPError(response=timeout_response)
        timeout_response.raise_for_status.side_effect = timeout_error
        mock_get.return_value = timeout_response

        with self.assertRaises(requests.HTTPError) as context:
            self.so.do_request("GET", "job-status/test-job", max_retries=0)

        self.assertIs(context.exception, timeout_error)
        mock_get.assert_called_once()

    @patch("sutro.sdk.time.sleep")
    @patch("requests.get")
    def test_get_request_still_retries_after_524(self, mock_get, mock_sleep):
        timeout_response = MagicMock()
        timeout_response.status_code = 524
        timeout_response.raise_for_status.side_effect = requests.HTTPError(
            response=timeout_response
        )

        success_response = MagicMock()
        success_response.status_code = 200
        mock_get.side_effect = [timeout_response, success_response]

        result = self.so.do_request(
            "GET", "job-status/test-job", max_retries=1
        )

        self.assertIs(result, success_response)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(1)


class TestUserExperience(unittest.TestCase):
    """Tests focused on user-facing behavior and experience"""

    def setUp(self):
        # Create an instance of Sutro with a dummy API key
        self.so = Sutro(
            api_key="test_api_key",
            api_url="https://harmonize.example.test",
        )

        # Setup capture of stdout for testing console output
        self.stdout_capture = io.StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout_capture

    def tearDown(self):
        # Reset stdout
        sys.stdout = self.old_stdout

    def get_captured_output(self):
        """Helper to get and clear the captured output"""
        output = self.stdout_capture.getvalue()
        self.stdout_capture = io.StringIO()  # Reset capture
        sys.stdout = self.stdout_capture
        return output

    @patch("requests.post")
    def test_progress_indicators_during_inference(self, mock_post):
        """Test that the user sees appropriate progress indicators during inference"""

        # Setup a delayed response to simulate processing time
        def delayed_response(*args, **kwargs):
            time.sleep(0.5)  # Small delay to ensure spinner is visible
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": ["result1", "result2"],
                "metadata": {"job_id": "test_job_id"},
            }
            return mock_response

        mock_post.side_effect = delayed_response

        # Call the method
        self.so.infer(["input1", "input2"])

        # Check output for progress indicators
        output = self.get_captured_output()
        self.assertIn("Materializing results", output)
        self.assertIn("✔ Materialized results received", output)
        self.assertIn("You can re-obtain the results with", output)

    @patch("requests.post")
    def test_helpful_error_messages(self, mock_post):
        """Test that error messages are helpful and user-friendly"""
        # Mock failed response with a typical API error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Invalid inputs",
            "details": "Input data format is not supported",
        }
        mock_post.return_value = mock_response

        # Call the method
        self.so.infer(["input1", "input2"])

        # Check output for detailed error information
        output = self.get_captured_output()
        self.assertIn("Error: 400", output)
        # Verify the full error details are shown to the user
        self.assertTrue("Invalid inputs" in output)
        self.assertTrue(
            "Input data format is not supported" in output or "details" in output
        )

    def test_input_validation_feedback(self):
        """Test that users get clear feedback for invalid inputs"""
        # Test with DataFrame but no column specified
        df = pd.DataFrame({"data": ["a", "b", "c"]})

        with self.assertRaises(ValueError) as context:
            self.so.infer(df)

        self.assertIn("Column name must be specified", str(context.exception))

    @patch("requests.post")
    def test_priority_job_feedback(self, mock_post):
        """Test feedback for priority jobs"""
        # Mock response for priority job
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": "priority_job_123"}
        mock_post.return_value = mock_response

        # Call with priority
        self.so.infer(["input1"], job_priority=5)

        # Check output contains helpful next steps
        output = self.get_captured_output()
        self.assertIn("Priority 5 Job created with ID", output)
        self.assertIn(
            "Use `so.get_job_status('priority_job_123')` to check the status", output
        )

    @patch("requests.post")
    def test_dry_run_feedback(self, mock_post):
        """Test the user feedback for dry run cost estimates"""
        # Mock response for dry run
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {"total_tokens": 1000, "estimated_cost": "$0.05"}
        }
        mock_post.return_value = mock_response

        # Call with dry run
        result = self.so.infer(["input1", "input2"], dry_run=True)

        # Check output
        output = self.get_captured_output()
        self.assertIn("Retrieving cost estimates", output)
        self.assertIn("✔ Cost estimates retrieved", output)
        self.assertEqual(result, {"total_tokens": 1000, "estimated_cost": "$0.05"})

    @patch("requests.get")
    def test_retryable_error_guidance(self, mock_get):
        """Test guidance provided for retryable errors like rate limits"""
        # Mock a rate limit response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": "Rate limit exceeded",
            "retry_after": 5,
        }
        mock_get.return_value = mock_response

        # Call method
        self.so.get_job_status("test_job_id")

        # Check output for retry guidance
        output = self.get_captured_output()
        self.assertIn("Rate limit exceeded", output)

    @patch("requests.post")
    def test_schema_validation_feedback(self, mock_post):
        """Test feedback when using JSON schema validation"""
        # Set up a schema
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                }
            },
        }

        # Mock response with validation error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Schema validation failed",
            "details": "Response didn't match schema: Expected 'sentiment' to be one of ['positive', 'negative', 'neutral']",
        }
        mock_post.return_value = mock_response

        # Call with schema
        self.so.infer(["This product is great!"], json_schema=schema)

        # Check output for helpful schema validation feedback
        output = self.get_captured_output()
        self.assertIn("Schema validation failed", output)

    @patch("requests.post")
    def test_dataframe_result_feedback(self, mock_post):
        """Test feedback when returning results into a dataframe"""
        # Create a test dataframe
        df = pd.DataFrame({"text": ["sample 1", "sample 2", "sample 3"]})

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": ["result1", "result2", "result3"],
            "metadata": {"job_id": "test_job_id"},
        }
        mock_post.return_value = mock_response

        # Call with dataframe
        result_df = self.so.infer(df, column="text")

        # Check that results were added to dataframe and user gets feedback
        output = self.get_captured_output()
        self.assertIn("✔ Materialized results received", output)
        self.assertTrue("inference_result" in result_df.columns)
        self.assertEqual(
            list(result_df["inference_result"]), ["result1", "result2", "result3"]
        )


class TestColorFormatting(unittest.TestCase):
    """Tests specifically for color formatting in user-facing messages"""

    def setUp(self):
        # Create an instance of Sutro with a dummy API key
        self.so = Sutro(
            api_key="test_api_key",
            api_url="https://harmonize.example.test",
        )

        # Setup capture of stdout for testing console output
        self.stdout_capture = io.StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout_capture

    def tearDown(self):
        # Reset stdout
        sys.stdout = self.old_stdout

    def get_captured_output(self):
        """Helper to get the captured output"""
        return self.stdout_capture.getvalue()

    def assert_colored_text_in_output(self, text, color, output):
        """Helper to check if colored text appears in output"""
        colored_text = f"{color}{text}{Style.RESET_ALL}"
        self.assertIn(colored_text, output)

    @patch("requests.post")
    def test_success_message_colors(self, mock_post):
        """Test that success messages use green color"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": ["result1", "result2"],
            "metadata": {"job_id": "test_job_id"},
        }
        mock_post.return_value = mock_response

        # Call the method
        self.so.infer(["input1", "input2"])

        # Get output and check color formatting
        output = self.get_captured_output()

        # Success indicators should be green
        self.assert_colored_text_in_output(
            "✔ Materialized results received", Fore.GREEN, output
        )

        # Job ID reference should be blue (informational)
        self.assert_colored_text_in_output(
            f"You can re-obtain the results with `so.get_job_results('test_job_id')`",
            Fore.BLUE,
            output,
        )

    @patch("requests.post")
    def test_error_message_colors(self, mock_post):
        """Test that error messages use red color"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_post.return_value = mock_response

        # Call the method
        self.so.infer(["input1", "input2"])

        # Get output and check color formatting
        output = self.get_captured_output()

        # Error indicators should be red
        self.assert_colored_text_in_output(f"Error: 400", Fore.RED, output)
        self.assert_colored_text_in_output(
            '{"error": "Bad request"}'
            if '{"error": "Bad request"}' in output
            else "{'error': 'Bad request'}",
            Fore.RED,
            output,
        )

    @patch("requests.get")
    def test_in_progress_message_colors(self, mock_get):
        """Test that in-progress messages use blue color"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"job_status": {"test_job_id": "COMPLETED"}}
        mock_get.return_value = mock_response

        # Call the method
        self.so.get_job_status("test_job_id")

        # Get output and check color formatting
        output = self.get_captured_output()

        # In-progress/processing messages should be blue
        self.assert_colored_text_in_output(
            f"Checking job status with ID: test_job_id", Fore.BLUE, output
        )

        # Success indicators should be green
        self.assert_colored_text_in_output(
            "✔ Job status retrieved!", Fore.GREEN, output
        )

    @patch("requests.post")
    def test_job_priority_message_colors(self, mock_post):
        """Test color formatting for priority job messages"""
        # Mock response for priority job
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": "job_priority_id"}
        mock_post.return_value = mock_response

        # Call with priority
        self.so.infer(["input1"], job_priority=2)

        # Get output and check color formatting
        output = self.get_captured_output()

        # Priority job creation message should be blue (processing)
        self.assert_colored_text_in_output("Creating priority 2 job", Fore.BLUE, output)

        # Success message should be green
        self.assert_colored_text_in_output(
            "Priority 2 Job created with ID: job_priority_id", Fore.GREEN, output
        )

    @patch("requests.get")
    def test_auth_failure_colors(self, mock_get):
        """Test color formatting for authentication failure messages"""
        # Mock auth failure response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_get.return_value = mock_response

        # Call method
        self.so.try_authentication("invalid_key")

        # Get output and check color formatting
        output = self.get_captured_output()

        # Auth failure should be red
        self.assert_colored_text_in_output(
            "API key failed to authenticate: 401", Fore.RED, output
        )

    @patch("requests.post")
    def test_dry_run_colors(self, mock_post):
        """Test color formatting for dry run messages"""
        # Mock dry run response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": {"tokens": 100, "cost": "$0.01"}}
        mock_post.return_value = mock_response

        # Call with dry run
        self.so.infer(["test"], dry_run=True)

        # Get output and check color formatting
        output = self.get_captured_output()

        # Dry run message should be blue (processing)
        self.assert_colored_text_in_output(
            "Retrieving cost estimates...", Fore.BLUE, output
        )

        # Success message should be green
        self.assert_colored_text_in_output(
            "✔ Cost estimates retrieved", Fore.GREEN, output
        )

    @patch("requests.get")
    def test_cancel_job_colors(self, mock_get):
        """Test color formatting for job cancellation messages"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "CANCELLED"}
        mock_get.return_value = mock_response

        # Call the method
        self.so.cancel_job("test_job_id")

        # Get output and check color formatting
        output = self.get_captured_output()

        # Cancellation in progress should be blue
        self.assert_colored_text_in_output(
            "Cancelling job: test_job_id", Fore.BLUE, output
        )

        # Success message should be green
        self.assert_colored_text_in_output("✔ Job cancelled", Fore.GREEN, output)

    @patch("requests.get")
    def test_cancel_job_failure_colors(self, mock_get):
        """Test color formatting for job cancellation failure messages"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Job not found"}
        mock_get.return_value = mock_response

        # Call the method
        self.so.cancel_job("nonexistent_job")

        # Get output and check color formatting
        output = self.get_captured_output()

        # Failure message should be red
        self.assert_colored_text_in_output("Failed to cancel job", Fore.RED, output)
        self.assert_colored_text_in_output(
            '{"error": "Job not found"}'
            if '{"error": "Job not found"}' in output
            else "{'error': 'Job not found'}",
            Fore.RED,
            output,
        )

    def test_color_consistency_across_methods(self):
        """Test that color formatting is consistent across all methods"""
        # Define states and their expected colors
        states = {
            "success": Fore.GREEN,
            "fail": Fore.RED,
            None: Fore.BLUE,  # Default is blue
        }

        # Test each state with same message
        message = "Test message"
        for state, expected_color in states.items():
            colored_message = to_colored_text(message, state=state)
            expected_message = f"{expected_color}{message}{Style.RESET_ALL}"
            self.assertEqual(colored_message, expected_message)


class TestPrepareInputData(unittest.TestCase):
    def test_download_url_passes_through_with_column(self):
        input_data, column_name = prepare_input_data(
            "https://example.com/data.parquet", "text"
        )
        self.assertEqual(input_data, "https://example.com/data.parquet")
        self.assertEqual(column_name, "text")

    def test_legacy_dataset_id_rejected_with_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            prepare_input_data(
                "dataset-123e4567-e89b-12d3-a456-426614174000", "text"
            )
        self.assertIn("datasets have been removed", str(ctx.exception))


class TestPresignedResultsDownload(unittest.TestCase):
    def setUp(self):
        self.so = Sutro(
            api_key="test_api_key",
            api_url="https://harmonize.example.test",
        )

        self.stdout_capture = io.StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout_capture

        self.tmpdir = tempfile.mkdtemp()

        self.payload = {
            "job_id": "test-job",
            "format": "parquet",
            "artifact": {
                "bucket": "test-bucket",
                "key": "results/test-job.parquet",
                "filename": "test-job.parquet",
                "size_bytes": 10,
            },
            "urls": {
                "get": "https://r2.example.com/get",
                "head": "https://r2.example.com/head",
            },
        }

    def tearDown(self):
        sys.stdout = self.old_stdout
        shutil.rmtree(self.tmpdir)

    @staticmethod
    def make_head_response(etag='"abc123"', content_length=10):
        response = MagicMock()
        response.headers = {"Content-Length": str(content_length)}
        if etag is not None:
            response.headers["ETag"] = etag
        return response

    @staticmethod
    def make_get_response(body, status_code=200, headers=None):
        response = MagicMock()
        response.status_code = status_code
        response.headers = headers or {}
        response.iter_content.return_value = [body]
        response.__enter__.return_value = response
        return response

    @patch("requests.get")
    def test_results_download_url_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.payload
        mock_get.return_value = mock_response

        result = self.so.results_download_url(
            "test-job", include_inputs=True, expires_in_seconds=600
        )

        self.assertEqual(result, self.payload)
        url = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        self.assertIn("jobs/test-job/results-url", url)
        self.assertEqual(params["format"], "parquet")
        self.assertTrue(params["include_inputs"])
        self.assertFalse(params["include_cumulative_logprobs"])
        self.assertEqual(params["expires_in_seconds"], 600)

    @patch("requests.get")
    def test_results_download_url_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Job not found"}
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        result = self.so.results_download_url("missing-job")

        self.assertIsNone(result)
        output = self.stdout_capture.getvalue()
        self.assertIn("Bad status code: 404", output)

    @patch("requests.get")
    @patch("requests.head")
    def test_download_job_results_fresh_download(self, mock_head, mock_get):
        mock_head.return_value = self.make_head_response()
        mock_get.return_value = self.make_get_response(b"0123456789")

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        expected_path = os.path.join(self.tmpdir, "test-job.parquet")
        self.assertEqual(result, expected_path)
        with open(expected_path, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")
        self.assertFalse(os.path.exists(expected_path + ".part"))
        self.assertFalse(os.path.exists(expected_path + ".part.etag"))
        # A fresh download must not send a Range header.
        self.assertIsNone(mock_get.call_args.kwargs["headers"])
        # Both direct requests must bound connect/read time so a stalled peer
        # can't hang the download forever.
        self.assertIsNotNone(mock_head.call_args.kwargs.get("timeout"))
        self.assertIsNotNone(mock_get.call_args.kwargs.get("timeout"))

    @patch("requests.get")
    @patch("requests.head")
    def test_download_job_results_resumes_partial_download(self, mock_head, mock_get):
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "wb") as f:
            f.write(b"0123")
        with open(destination + ".part.etag", "w") as f:
            f.write('"abc123"')

        mock_head.return_value = self.make_head_response()
        mock_get.return_value = self.make_get_response(b"456789", status_code=206)

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertEqual(result, destination)
        with open(destination, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")
        self.assertEqual(
            mock_get.call_args.kwargs["headers"],
            {"Range": "bytes=4-", "If-Range": '"abc123"'},
        )

    @patch("requests.get")
    @patch("requests.head")
    def test_download_job_results_restarts_on_etag_mismatch(self, mock_head, mock_get):
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "wb") as f:
            f.write(b"stale")
        with open(destination + ".part.etag", "w") as f:
            f.write('"old-etag"')

        mock_head.return_value = self.make_head_response(etag='"new-etag"')
        mock_get.return_value = self.make_get_response(b"0123456789")

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertEqual(result, destination)
        with open(destination, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")
        self.assertIsNone(mock_get.call_args.kwargs["headers"])

    @patch("requests.get")
    @patch("requests.head")
    def test_failed_restart_does_not_resume_onto_stale_partial(
        self, mock_head, mock_get
    ):
        # An artifact rebuilt server-side (new ETag) plus a GET failure on the
        # first attempt must not leave a stale .part that the retry
        # Range-appends onto, which would corrupt the file.
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "wb") as f:
            f.write(b"stale")
        with open(destination + ".part.etag", "w") as f:
            f.write('"old-etag"')

        mock_head.return_value = self.make_head_response(etag='"new-etag"')

        expired_response = MagicMock()
        expired_response.status_code = 403
        failing_get = MagicMock()
        failing_get.__enter__.return_value = failing_get
        failing_get.raise_for_status.side_effect = requests.HTTPError(
            response=expired_response
        )
        mock_get.return_value = failing_get

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)
        self.assertIsNone(result)

        mock_get.return_value = self.make_get_response(b"0123456789")
        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertEqual(result, destination)
        with open(destination, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")
        self.assertIsNone(mock_get.call_args.kwargs["headers"])

    @patch("requests.get")
    @patch("requests.head")
    def test_download_job_results_trailing_slash_creates_directory(
        self, mock_head, mock_get
    ):
        mock_head.return_value = self.make_head_response()
        mock_get.return_value = self.make_get_response(b"0123456789")

        output_dir = os.path.join(self.tmpdir, "downloads") + os.sep
        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=output_dir)

        expected_path = os.path.join(self.tmpdir, "downloads", "test-job.parquet")
        self.assertEqual(result, expected_path)
        with open(expected_path, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")

    @patch("requests.get")
    def test_results_download_url_non_json_error_body(self, mock_get):
        # Cloudflare 5xx pages are HTML; printing the error must not raise
        # JSONDecodeError from inside the handler.
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "<html>502 Bad Gateway</html>"
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "<html>", 0
        )
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        result = self.so.results_download_url("test-job")

        self.assertIsNone(result)
        output = self.stdout_capture.getvalue()
        self.assertIn("Bad status code: 502", output)
        self.assertIn("502 Bad Gateway", output)

    @patch("requests.get")
    def test_results_download_url_transport_error_returns_none(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("connection reset by peer")

        result = self.so.results_download_url("test-job")

        self.assertIsNone(result)
        output = self.stdout_capture.getvalue()
        self.assertIn("Request failed", output)

    @patch("requests.get")
    @patch("requests.head")
    def test_download_without_etag_skips_resume(self, mock_head, mock_get):
        # Some S3-compatible gateways omit ETag; resume can't be validated so
        # the download must restart fresh instead of crashing or appending.
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "wb") as f:
            f.write(b"0123")
        with open(destination + ".part.etag", "w") as f:
            f.write('"abc123"')

        mock_head.return_value = self.make_head_response(etag=None)
        mock_get.return_value = self.make_get_response(b"0123456789")

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertEqual(result, destination)
        with open(destination, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")
        self.assertIsNone(mock_get.call_args.kwargs["headers"])
        self.assertFalse(os.path.exists(destination + ".part.etag"))

    @patch("requests.get")
    @patch("requests.head")
    def test_resume_restarts_when_artifact_changes_between_head_and_get(
        self, mock_head, mock_get
    ):
        # If the cached artifact is replaced between the HEAD check and the
        # Range GET, a 206 for the new object must not be appended onto the
        # old partial.
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "wb") as f:
            f.write(b"0123")
        with open(destination + ".part.etag", "w") as f:
            f.write('"abc123"')

        mock_head.return_value = self.make_head_response()
        mock_get.side_effect = [
            self.make_get_response(
                b"456789", status_code=206, headers={"ETag": '"rebuilt"'}
            ),
            self.make_get_response(b"0123456789"),
        ]

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertEqual(result, destination)
        with open(destination, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")
        first_headers = mock_get.call_args_list[0].kwargs["headers"]
        self.assertEqual(first_headers["Range"], "bytes=4-")
        self.assertEqual(first_headers["If-Range"], '"abc123"')
        self.assertIsNone(mock_get.call_args_list[1].kwargs["headers"])

    @patch("requests.head")
    def test_download_job_results_presigned_failure_returns_none(self, mock_head):
        error_response = MagicMock()
        error_response.status_code = 403
        mock_head.return_value.raise_for_status.side_effect = requests.HTTPError(
            response=error_response
        )

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertIsNone(result)
        output = self.stdout_capture.getvalue()
        self.assertIn("Download failed with status code: 403", output)

    @patch("requests.get")
    @patch("requests.head")
    def test_streaming_failure_returns_none_and_keeps_partial(
        self, mock_head, mock_get
    ):
        # Connection resets mid-stream raise RequestException subclasses that
        # aren't HTTPErrors; they must follow the print-and-return-None error
        # style and leave the partial file in place for a resumed retry.
        mock_head.return_value = self.make_head_response()

        def interrupted_chunks():
            yield b"0123"
            raise requests.exceptions.ChunkedEncodingError("connection reset")

        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.iter_content.return_value = interrupted_chunks()
        response.__enter__.return_value = response
        mock_get.return_value = response

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertIsNone(result)
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "rb") as f:
            self.assertEqual(f.read(), b"0123")
        with open(destination + ".part.etag") as f:
            self.assertEqual(f.read(), '"abc123"')
        output = self.stdout_capture.getvalue()
        self.assertIn("Download interrupted", output)

    @patch("requests.get")
    @patch("requests.head")
    def test_transport_error_output_redacts_presigned_credentials(
        self, mock_head, mock_get
    ):
        # str(ConnectionError) includes the full request URL; presigned query
        # params grant access to results and must never reach logs.
        mock_head.return_value = self.make_head_response()
        mock_get.side_effect = requests.ConnectionError(
            "HTTPSConnectionPool(host='r2.example.com'): Max retries exceeded "
            "with url: /results.parquet?X-Amz-Credential=AKIASECRET"
            "&X-Amz-Signature=deadbeefcafe"
        )

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertIsNone(result)
        output = self.stdout_capture.getvalue()
        self.assertIn("Download interrupted", output)
        self.assertIn("ConnectionError", output)
        self.assertNotIn("X-Amz-Signature", output)
        self.assertNotIn("deadbeefcafe", output)
        self.assertNotIn("AKIASECRET", output)

    @patch("requests.get")
    @patch("requests.head")
    def test_restart_validates_against_replacement_metadata(
        self, mock_head, mock_get
    ):
        # HEAD sees the old 10-byte artifact, but by GET time it was replaced
        # by a 12-byte one: If-Range downgrades to a 200 full body. The
        # download must validate against the replacement's size/ETag, not the
        # stale HEAD metadata, or the complete download gets rejected.
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        with open(destination + ".part", "wb") as f:
            f.write(b"0123")
        with open(destination + ".part.etag", "w") as f:
            f.write('"abc123"')

        mock_head.return_value = self.make_head_response()
        replacement = b"0123456789AB"
        mock_get.return_value = self.make_get_response(
            replacement,
            status_code=200,
            headers={"ETag": '"replaced"', "Content-Length": "12"},
        )

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertEqual(result, destination)
        with open(destination, "rb") as f:
            self.assertEqual(f.read(), replacement)
        self.assertFalse(os.path.exists(destination + ".part"))
        self.assertFalse(os.path.exists(destination + ".part.etag"))

    @patch("requests.get")
    @patch("requests.head")
    def test_short_download_is_not_published(self, mock_head, mock_get):
        # A GET that ends cleanly with fewer bytes than HEAD advertised (e.g.
        # an ETag-less gateway serving a replaced object) must not be renamed
        # into a "successful" truncated Parquet file.
        mock_head.return_value = self.make_head_response(content_length=10)
        mock_get.return_value = self.make_get_response(b"0123")

        with patch.object(Sutro, "results_download_url", return_value=self.payload):
            result = self.so.download_job_results("test-job", output_path=self.tmpdir)

        self.assertIsNone(result)
        destination = os.path.join(self.tmpdir, "test-job.parquet")
        self.assertFalse(os.path.exists(destination))
        self.assertTrue(os.path.exists(destination + ".part"))
        output = self.stdout_capture.getvalue()
        self.assertIn("Download incomplete", output)

    def test_download_job_results_propagates_url_failure(self):
        with patch.object(Sutro, "results_download_url", return_value=None):
            result = self.so.download_job_results("test-job")

        self.assertIsNone(result)


class TestRunFunction(unittest.TestCase):
    """Real-time Function execution: POST /v1/functions/{name}/run."""

    def setUp(self):
        self.so = Sutro(
            api_key="test_api_key",
            api_url="https://harmonize.example.test",
        )

    def _response(self, status_code, payload, headers=None):
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.headers = headers or {}
        response.json.return_value = payload
        if status_code >= 400:
            response.raise_for_status.side_effect = requests.HTTPError(
                f"{status_code} Error", response=response
            )
        else:
            response.raise_for_status.return_value = None
        return response

    @patch("requests.post")
    def test_run_function_success(self, mock_post):
        mock_post.return_value = self._response(
            200,
            {
                "request_id": "rt_abc",
                "function": {
                    "name": "pcr-checker",
                    "model": "claude-sonnet-4-5",
                    "model_source": "model-sweep",
                },
                "output": {"label": "yes", "reasoning": "because"},
                "confidence": 0.8,
                "usage": {
                    "input_tokens": 4060,
                    "output_tokens": 205,
                    "cost_usd": 0.0155,
                },
            },
        )

        result = self.so.run_function("pcr-checker", {"title": "a", "body": "b"})

        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://harmonize.example.test/v1/functions/pcr-checker/run",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["Authorization"],
            "Key test_api_key",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {"input": {"title": "a", "body": "b"}},
        )

        self.assertEqual(result.output, {"label": "yes", "reasoning": "because"})
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.usage["input_tokens"], 4060)
        self.assertEqual(result.request_id, "rt_abc")
        self.assertEqual(result.function["model"], "claude-sonnet-4-5")
        # The payload itself stays available as a plain dict.
        self.assertEqual(result["confidence"], 0.8)

    @patch("requests.post")
    def test_run_function_accepts_a_bare_string_input(self, mock_post):
        mock_post.return_value = self._response(
            200, {"request_id": "rt_1", "output": "yes", "confidence": 1.0}
        )

        result = self.so.run_function("pcr-checker", "one text field")

        self.assertEqual(
            mock_post.call_args.kwargs["json"], {"input": "one text field"}
        )
        self.assertEqual(result.output, "yes")
        self.assertEqual(result.usage, {})

    @patch("requests.post")
    def test_run_function_serializes_pydantic_input(self, mock_post):
        class Input(BaseModel):
            title: str

        mock_post.return_value = self._response(200, {"output": "ok"})

        self.so.run_function("pcr-checker", Input(title="a"))

        self.assertEqual(
            mock_post.call_args.kwargs["json"], {"input": {"title": "a"}}
        )

    @patch("requests.post")
    def test_run_function_does_not_retry(self, mock_post):
        mock_post.return_value = self._response(
            524, {"detail": "Timeout", "code": "timeout"}
        )

        with self.assertRaises(requests.HTTPError):
            self.so.run_function("pcr-checker", {"title": "a"})

        self.assertEqual(mock_post.call_count, 1)
        # The caller is never left waiting on a stalled connection.
        self.assertEqual(
            mock_post.call_args.kwargs["timeout"], FUNCTION_RUN_REQUEST_TIMEOUT
        )

    @patch("requests.post")
    def test_run_function_waits_as_long_as_the_caller_asks(self, mock_post):
        mock_post.return_value = self._response(
            524, {"detail": "Timeout", "code": "timeout"}
        )

        with self.assertRaises(requests.HTTPError):
            self.so.run_function("pcr-checker", {"title": "a"}, timeout_seconds=300)

        # A deployment whose deadline was raised needs a longer read timeout.
        self.assertEqual(mock_post.call_args.kwargs["timeout"], (10, 300))

    @patch("requests.post")
    def test_run_function_rate_limited(self, mock_post):
        mock_post.return_value = self._response(
            429,
            {
                "detail": "Rate limit exceeded.",
                "code": "rate_limited",
                "request_id": "rt_429",
            },
            headers={"Retry-After": "3"},
        )

        with self.assertRaises(SutroRateLimitError) as caught:
            self.so.run_function("pcr-checker", {"title": "a"})

        error = caught.exception
        self.assertEqual(error.retry_after, 3.0)
        self.assertEqual(error.detail, "Rate limit exceeded.")
        self.assertEqual(error.code, "rate_limited")
        self.assertEqual(error.request_id, "rt_429")
        self.assertIsInstance(error, requests.HTTPError)

    @patch("requests.post")
    def test_run_function_rate_limited_without_retry_after(self, mock_post):
        mock_post.return_value = self._response(
            429, {"detail": "Slow down.", "code": "rate_limited"}
        )

        with self.assertRaises(SutroRateLimitError) as caught:
            self.so.run_function("pcr-checker", {"title": "a"})

        self.assertIsNone(caught.exception.retry_after)

    @patch("requests.post")
    def test_run_function_validation_error(self, mock_post):
        mock_post.return_value = self._response(
            422,
            {
                "detail": "Missing required input field(s): body.",
                "code": "invalid_input",
                "request_id": "rt_422",
            },
        )

        with self.assertRaises(SutroValidationError) as caught:
            self.so.run_function("pcr-checker", {"title": "a"})

        error = caught.exception
        self.assertEqual(error.detail, "Missing required input field(s): body.")
        self.assertEqual(error.code, "invalid_input")
        self.assertEqual(error.request_id, "rt_422")
        self.assertIn("body", str(error))

    @patch("requests.post")
    def test_run_function_other_api_errors_keep_the_server_detail(self, mock_post):
        for status_code, code in (
            (404, "function_not_found"),
            (409, "no_runnable_prompt"),
            (502, "provider_error"),
            (503, "provider_not_configured"),
            (504, "timeout"),
        ):
            with self.subTest(status_code=status_code):
                mock_post.return_value = self._response(
                    status_code,
                    {
                        "detail": f"{code} happened",
                        "code": code,
                        "request_id": "rt_err",
                    },
                )

                with self.assertRaises(requests.HTTPError) as caught:
                    self.so.run_function("pcr-checker", {"title": "a"})

                error = caught.exception
                self.assertNotIsInstance(error, SutroRateLimitError)
                self.assertNotIsInstance(error, SutroValidationError)
                self.assertEqual(error.detail, f"{code} happened")
                self.assertEqual(error.code, code)
                self.assertEqual(error.request_id, "rt_err")

    @patch("requests.post")
    def test_run_function_error_without_a_json_body(self, mock_post):
        response = self._response(500, {})
        response.json.side_effect = ValueError("no json")
        mock_post.return_value = response

        with self.assertRaises(requests.HTTPError) as caught:
            self.so.run_function("pcr-checker", {"title": "a"})

        self.assertIsNone(caught.exception.code)
        self.assertIn("500", caught.exception.detail)

    @patch("requests.post")
    def test_run_function_sends_asset_values(self, mock_post):
        mock_post.return_value = self._response(200, {"output": "ok"})

        self.so.run_function(
            "invoice-reader",
            {
                "note": "check page 2",
                "scan": Asset.from_url("https://files.example.test/a.pdf"),
                "photo": Image.from_bytes(b"\x89PNG\r\n\x1a\n", "image/png"),
            },
        )

        sent = mock_post.call_args.kwargs["json"]["input"]
        self.assertEqual(sent["scan"], {"url": "https://files.example.test/a.pdf"})
        self.assertEqual(
            sent["photo"],
            {
                "type": "image",
                "base64": "iVBORw0KGgo=",
                "mime_type": "image/png",
            },
        )
        # The payload must survive JSON serialization unchanged.
        self.assertEqual(json.loads(json.dumps(sent))["photo"], sent["photo"])

    def test_run_function_rejects_unsupported_input_types(self):
        with self.assertRaises(TypeError):
            self.so.run_function("pcr-checker", ["a", "b"])


class TestAssetHelpers(unittest.TestCase):
    """The asset value shapes a deployment accepts on a Function input field."""

    PNG = b"\x89PNG\r\n\x1a\n"

    def test_image_from_bytes(self):
        self.assertEqual(
            Image.from_bytes(self.PNG, "image/png"),
            {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
        )

    def test_image_from_bytes_normalizes_image_jpg(self):
        self.assertEqual(
            Image.from_bytes(b"\xff\xd8\xff", "image/jpg")["mime_type"],
            "image/jpeg",
        )

    def test_image_from_bytes_rejects_a_pdf(self):
        with self.assertRaises(ValueError):
            Image.from_bytes(b"%PDF-1.4", "application/pdf")

    def test_image_from_bytes_rejects_empty_bytes(self):
        with self.assertRaises(ValueError):
            Image.from_bytes(b"", "image/png")

    def test_image_from_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "photo.png")
            with open(path, "wb") as handle:
                handle.write(self.PNG)

            self.assertEqual(
                Image.from_path(path),
                {
                    "type": "image",
                    "base64": "iVBORw0KGgo=",
                    "filename": "photo.png",
                    "mime_type": "image/png",
                },
            )

    def test_asset_from_path_reads_a_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "scan.pdf")
            with open(path, "wb") as handle:
                handle.write(b"%PDF-1.4")

            asset = Asset.from_path(path)

        self.assertEqual(asset["type"], "pdf")
        self.assertEqual(asset["mime_type"], "application/pdf")
        self.assertEqual(asset["filename"], "scan.pdf")

    def test_asset_from_path_rejects_an_unknown_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "notes.txt")
            with open(path, "wb") as handle:
                handle.write(b"hello")

            with self.assertRaises(ValueError):
                Asset.from_path(path)

    def test_asset_from_url(self):
        self.assertEqual(
            Asset.from_url("https://files.example.test/a.pdf"),
            {"url": "https://files.example.test/a.pdf"},
        )

    def test_image_from_url_declares_its_type(self):
        self.assertEqual(
            Image.from_url("https://files.example.test/a.png"),
            {"type": "image", "url": "https://files.example.test/a.png"},
        )

    def test_asset_from_url_requires_https(self):
        for url in ("http://files.example.test/a.pdf", "file:///etc/passwd", ""):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Asset.from_url(url)

    def test_asset_from_name(self):
        self.assertEqual(Asset.from_name(" invoice.pdf "), {"name": "invoice.pdf"})

    def test_asset_from_name_requires_a_name(self):
        with self.assertRaises(ValueError):
            Asset.from_name("  ")


if __name__ == "__main__":
    unittest.main()
