import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sutro.cli import check_auth, cli, get_sdk
from sutro.sdk import FunctionRunResult, SutroValidationError


class TestCliConfiguration(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        config_dir = Path(self.temp_directory.name) / ".sutro"
        config_file = config_dir / "config.json"
        self.config_dir_patch = patch("sutro.validation.CONFIG_DIR", str(config_dir))
        self.config_file_patch = patch("sutro.validation.CONFIG_FILE", str(config_file))
        self.config_dir_patch.start()
        self.config_file_patch.start()
        self.addCleanup(self.config_dir_patch.stop)
        self.addCleanup(self.config_file_patch.stop)
        self.config_file = config_file
        self.runner = CliRunner()

    def test_environment_only_configuration_is_authenticated(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "https://harmonize.example.test",
            }
        )

        self.assertTrue(check_auth())

        sdk = MagicMock()
        sdk.get_quotas.return_value = []
        with patch("sutro.cli.get_sdk", return_value=sdk):
            result = self.runner.invoke(cli, ["quotas"])

        self.assertEqual(result.exit_code, 0, result.output)
        sdk.get_quotas.assert_called_once_with()

    @patch("sutro.sdk.check_version")
    def test_get_sdk_uses_environment_configuration(self, _check_version):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "https://harmonize.example.test",
            }
        )

        sdk = get_sdk()

        self.assertEqual(sdk.api_key, "environment-key")
        self.assertEqual(sdk.api_url, "https://harmonize.example.test/v1")

    def test_login_prompts_validates_and_persists_deployment_config(self):
        sdk = MagicMock()
        sdk.try_authentication.return_value = {"authenticated": True}
        with patch("sutro.cli.Sutro", return_value=sdk) as sutro_class:
            result = self.runner.invoke(
                cli,
                ["login"],
                input="https://harmonize.example.test\nsk_new_key\n",
            )

        self.assertEqual(result.exit_code, 0, result.output)
        sutro_class.assert_called_once_with(
            api_key="sk_new_key",
            api_url="https://harmonize.example.test/v1",
        )
        sdk.try_authentication.assert_called_once_with("sk_new_key")
        config = json.loads(self.config_file.read_text())
        self.assertEqual(config["api_key"], "sk_new_key")
        self.assertEqual(config["api_url"], "https://harmonize.example.test/v1")
        self.assertNotIn("base_url", config)

    def test_login_does_not_reuse_key_after_deployment_changes(self):
        self.config_file.parent.mkdir(parents=True)
        self.config_file.write_text(
            json.dumps(
                {
                    "api_key": "old-deployment-key",
                    "api_url": "https://old.example.test/v1",
                }
            )
        )
        sdk = MagicMock()
        sdk.try_authentication.return_value = {"authenticated": True}

        with patch("sutro.cli.Sutro", return_value=sdk) as sutro_class:
            result = self.runner.invoke(
                cli,
                ["login"],
                input="https://new.example.test\nnew-deployment-key\n",
            )

        self.assertEqual(result.exit_code, 0, result.output)
        sutro_class.assert_called_once_with(
            api_key="new-deployment-key",
            api_url="https://new.example.test/v1",
        )
        sdk.try_authentication.assert_called_once_with("new-deployment-key")
        self.assertNotIn("API key is already set", result.output)

    def test_login_warns_when_environment_will_shadow_saved_credentials(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": (
                    "https://skysight-inc--tensor-factory-fastapi-app.modal.run"
                ),
            }
        )
        sdk = MagicMock()
        sdk.try_authentication.return_value = {"authenticated": True}

        with patch("sutro.cli.Sutro", return_value=sdk):
            result = self.runner.invoke(
                cli,
                ["login"],
                input="https://new.example.test\nnew-deployment-key\n",
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("environment variables take precedence", result.output)
        self.assertIn("Unset them", result.output)
        config = json.loads(self.config_file.read_text())
        self.assertEqual(config["api_key"], "new-deployment-key")
        self.assertEqual(config["api_url"], "https://new.example.test/v1")

    def test_malformed_environment_does_not_block_set_api_url_recovery(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "not-an-absolute-url",
            }
        )

        result = self.runner.invoke(
            cli,
            ["set-api-url", "https://new.example.test"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        config = json.loads(self.config_file.read_text())
        self.assertEqual(config["api_url"], "https://new.example.test/v1")

    def test_malformed_environment_explains_failure_for_non_recovery_command(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "not-an-absolute-url",
            }
        )

        result = self.runner.invoke(cli, ["quotas"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("absolute HTTP(S) Sutro deployment URL", result.output)

    def test_set_api_url_normalizes_and_clears_key_for_changed_deployment(self):
        self.config_file.parent.mkdir(parents=True)
        self.config_file.write_text(
            json.dumps({"api_key": "existing-key", "other": "value"})
        )

        result = self.runner.invoke(
            cli,
            ["set-api-url", "http://localhost:8000"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        config = json.loads(self.config_file.read_text())
        self.assertNotIn("api_key", config)
        self.assertEqual(config["other"], "value")
        self.assertEqual(config["api_url"], "http://localhost:8000/v1")
        self.assertIn("API key was cleared", result.output)

    def test_set_api_url_preserves_key_for_same_deployment(self):
        self.config_file.parent.mkdir(parents=True)
        self.config_file.write_text(
            json.dumps(
                {
                    "api_key": "existing-key",
                    "api_url": "https://harmonize.example.test",
                }
            )
        )

        result = self.runner.invoke(
            cli,
            ["set-api-url", "https://harmonize.example.test/v1"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        config = json.loads(self.config_file.read_text())
        self.assertEqual(config["api_key"], "existing-key")
        self.assertNotIn("API key was cleared", result.output)

    def test_set_api_url_explains_direct_tensor_factory_migration(self):
        result = self.runner.invoke(
            cli,
            [
                "set-api-url",
                "https://skysight-inc--tensor-factory-fastapi-app.modal.run",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "Direct Sutro Batch API access is no longer supported",
            result.output,
        )
        self.assertIn("SUTRO_API_URL", result.output)
        self.assertFalse(self.config_file.exists())

    def test_hidden_set_base_url_migrates_legacy_config(self):
        self.config_file.parent.mkdir(parents=True)
        self.config_file.write_text(
            json.dumps(
                {
                    "api_key": "existing-key",
                    "base_url": "https://old.example.test",
                }
            )
        )

        result = self.runner.invoke(
            cli,
            ["set-base-url", "https://new.example.test/v1"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("deprecated", result.output)
        config = json.loads(self.config_file.read_text())
        self.assertEqual(config["api_url"], "https://new.example.test/v1")
        self.assertNotIn("api_key", config)
        self.assertNotIn("base_url", config)


class TestFunctionsRunCommand(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "https://harmonize.example.test",
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.runner = CliRunner()
        self.payload = {
            "request_id": "rt_abc",
            "function": {"name": "pcr-checker", "revision": 7},
            "output": {"label": "yes"},
            "confidence": 0.8,
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }

    def test_run_prints_the_json_response(self):
        sdk = MagicMock()
        sdk.run_function.return_value = FunctionRunResult(self.payload)

        with patch("sutro.cli.get_sdk", return_value=sdk):
            result = self.runner.invoke(
                cli,
                ["functions", "run", "pcr-checker", "--input", '{"title": "a"}'],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        sdk.run_function.assert_called_once_with("pcr-checker", {"title": "a"})
        self.assertEqual(json.loads(result.output), self.payload)

    def test_run_reads_input_from_a_file(self):
        sdk = MagicMock()
        sdk.run_function.return_value = FunctionRunResult(self.payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text('{"title": "from file"}')

            with patch("sutro.cli.get_sdk", return_value=sdk):
                result = self.runner.invoke(
                    cli,
                    ["functions", "run", "pcr-checker", "--input", f"@{path}"],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        sdk.run_function.assert_called_once_with(
            "pcr-checker", {"title": "from file"}
        )

    def test_run_reports_the_server_detail_and_exits_non_zero(self):
        sdk = MagicMock()
        sdk.run_function.side_effect = SutroValidationError(
            "Missing required input field(s): body.",
            detail="Missing required input field(s): body.",
            code="invalid_input",
        )

        with patch("sutro.cli.get_sdk", return_value=sdk):
            result = self.runner.invoke(
                cli,
                ["functions", "run", "pcr-checker", "--input", '{"title": "a"}'],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing required input field(s): body.", result.output)

    def test_run_rejects_input_that_is_not_json(self):
        sdk = MagicMock()

        with patch("sutro.cli.get_sdk", return_value=sdk):
            result = self.runner.invoke(
                cli,
                ["functions", "run", "pcr-checker", "--input", "not json"],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("valid JSON", result.output)
        sdk.run_function.assert_not_called()


if __name__ == "__main__":
    unittest.main()
