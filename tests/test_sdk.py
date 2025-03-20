import unittest
import time
from unittest.mock import patch, MagicMock, mock_open
import sys
import io
import json
import os

import pandas as pd

# Add these imports from your MaterializedIntelligence class
from colorama import Fore, Style

from materialized_intelligence import MaterializedIntelligence
from materialized_intelligence.sdk import to_colored_text


class TestMaterializedIntelligence(unittest.TestCase):
    def setUp(self):
        # Create an instance of MaterializedIntelligence with a dummy API key
        self.mi = MaterializedIntelligence(api_key="test_api_key")

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
        result = self.mi.infer(["input1", "input2"])

        # Check that the API was called correctly
        mock_post.assert_called_once()

        # Verify results
        self.assertEqual(result, ["result1", "result2"])

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Materialized results received", output)
        self.assertIn(
            f"You can re-obtain the results with `mi.get_job_results('test_job_id')`",
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
        result = self.mi.infer(["input1", "input2"])

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
        result = self.mi.list_jobs()

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
        result = self.mi.list_jobs()

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
        result = self.mi.get_job_status("test_job_id")

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
        result = self.mi.get_job_results("test_job_id")

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
        result = self.mi.cancel_job("test_job_id")

        # Verify results
        self.assertEqual(result, {"status": "CANCELLED"})

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Job cancelled", output)

    @patch("requests.get")
    def test_create_stage_success(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stage_id": "test_stage_id"}
        mock_get.return_value = mock_response

        # Call the method
        result = self.mi.create_stage()

        # Verify results
        self.assertEqual(result, "test_stage_id")

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Stage created with ID: test_stage_id", output)

    @patch("requests.post")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("os.path.basename")
    @patch("builtins.open")
    @patch("materialized_intelligence.MaterializedIntelligence.create_stage")
    def test_upload_to_stage_success(
        self,
        mock_create_stage,
        mock_open,
        mock_basename,
        mock_listdir,
        mock_isdir,
        mock_post,
    ):
        # Mock necessary functions
        mock_create_stage.return_value = "new_stage_id"
        mock_isdir.return_value = False
        mock_basename.return_value = "test_file.csv"
        mock_open.return_value = MagicMock()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call the method with single file
        result = self.mi.upload_to_stage("test_stage_id", "test_file.csv")

        # Verify results
        self.assertEqual(result, "test_stage_id")

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ 1 files successfully uploaded to stage", output)

    @patch("requests.post")
    def test_list_stages_success(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stages": ["stage1", "stage2"]}
        mock_post.return_value = mock_response

        # Call the method
        result = self.mi.list_stages()

        # Verify results
        self.assertEqual(result, ["stage1", "stage2"])

        # Check output for success message
        output = self.stdout_capture.getvalue()
        self.assertIn("✔ Stages retrieved", output)

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


class TestUserExperience(unittest.TestCase):
    """Tests focused on user-facing behavior and experience"""

    def setUp(self):
        # Create an instance of MaterializedIntelligence with a dummy API key
        self.mi = MaterializedIntelligence(api_key="test_api_key")

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
        self.mi.infer(["input1", "input2"])

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
        self.mi.infer(["input1", "input2"])

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
            self.mi.infer(df)

        self.assertIn("Column name must be specified", str(context.exception))

    @patch("requests.post")
    @patch.object(MaterializedIntelligence, "create_stage")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("builtins.open", new_callable=mock_open, read_data="test data")
    def test_multi_file_upload_progress(
        self, mock_file, mock_listdir, mock_isdir, mock_create_stage, mock_post
    ):
        """Test that users see progress during multi-file uploads"""
        # Setup for multiple files
        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.txt", "file2.txt", "file3.txt"]
        mock_create_stage.return_value = "test_stage_id"

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call the method
        self.mi.upload_to_stage("/fake/directory")

        # Check output for file upload progress indicators
        output = self.get_captured_output()
        self.assertIn("Uploading file 1/3", output)
        self.assertIn("Uploading file 2/3", output)
        self.assertIn("Uploading file 3/3", output)
        self.assertIn("✔ 3 files successfully uploaded", output)

    @patch("requests.post")
    def test_priority_job_feedback(self, mock_post):
        """Test feedback for priority jobs"""
        # Mock response for priority job
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": "priority_job_123"}
        mock_post.return_value = mock_response

        # Call with priority
        self.mi.infer(["input1"], job_priority=5)

        # Check output contains helpful next steps
        output = self.get_captured_output()
        self.assertIn("Priority 5 Job created with ID", output)
        self.assertIn(
            "Use `mi.get_job_status('priority_job_123')` to check the status", output
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
        result = self.mi.infer(["input1", "input2"], dry_run=True)

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
        self.mi.get_job_status("test_job_id")

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
        self.mi.infer(["This product is great!"], json_schema=schema)

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
        result_df = self.mi.infer(df, column="text")

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
        # Create an instance of MaterializedIntelligence with a dummy API key
        self.mi = MaterializedIntelligence(api_key="test_api_key")

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
        self.mi.infer(["input1", "input2"])

        # Get output and check color formatting
        output = self.get_captured_output()

        # Success indicators should be green
        self.assert_colored_text_in_output(
            "✔ Materialized results received", Fore.GREEN, output
        )

        # Job ID reference should be blue (informational)
        self.assert_colored_text_in_output(
            f"You can re-obtain the results with `mi.get_job_results('test_job_id')`",
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
        self.mi.infer(["input1", "input2"])

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
        self.mi.get_job_status("test_job_id")

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
    def test_stage_upload_progress_colors(self, mock_post):
        """Test color formatting during file upload progress"""
        # Setup mocks
        with patch("os.path.isdir") as mock_isdir, patch(
            "os.listdir"
        ) as mock_listdir, patch("os.path.basename") as mock_basename, patch(
            "builtins.open"
        ) as mock_open, patch.object(
            MaterializedIntelligence, "create_stage"
        ) as mock_create_stage:
            # Configure mocks
            mock_create_stage.return_value = "test_stage_id"
            mock_isdir.return_value = True
            mock_listdir.return_value = ["file1.txt", "file2.txt"]
            mock_basename.side_effect = lambda x: x
            mock_open.return_value = MagicMock()

            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            # Call the method
            self.mi.upload_to_stage("/fake/dir/")

            # Get output and check color formatting
            output = self.get_captured_output()

            # Progress messages should be blue
            self.assert_colored_text_in_output(
                "Uploading files to stage: test_stage_id", Fore.BLUE, output
            )
            self.assert_colored_text_in_output(
                "Uploading file 1/2 to stage: test_stage_id", Fore.BLUE, output
            )
            self.assert_colored_text_in_output(
                "Uploading file 2/2 to stage: test_stage_id", Fore.BLUE, output
            )

            # Success message should be green
            self.assert_colored_text_in_output(
                "✔ 2 files successfully uploaded to stage", Fore.GREEN, output
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
        self.mi.infer(["input1"], job_priority=2)

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
        self.mi.try_authentication("invalid_key")

        # Get output and check color formatting
        output = self.get_captured_output()

        # Auth failure should be red
        self.assert_colored_text_in_output(
            "API key failed to authenticate: 401", Fore.RED, output
        )

    @patch("requests.post")
    def test_network_error_colors(self, mock_post):
        """Test color formatting for network error messages"""
        # Mock network error
        mock_post.side_effect = Exception("Connection error")

        # Patch upload_to_stage to handle the exception
        with patch.object(
            MaterializedIntelligence,
            "upload_to_stage",
            side_effect=lambda *a, **kw: print(
                to_colored_text("Upload failed: Connection error", state="fail")
            ),
        ):
            self.mi.upload_to_stage("test_stage", "test_file.txt")

        # Get output and check color formatting
        output = self.get_captured_output()

        # Network error should be red
        self.assert_colored_text_in_output(
            "Upload failed: Connection error", Fore.RED, output
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
        self.mi.infer(["test"], dry_run=True)

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
        self.mi.cancel_job("test_job_id")

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
        self.mi.cancel_job("nonexistent_job")

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


if __name__ == "__main__":
    unittest.main()
