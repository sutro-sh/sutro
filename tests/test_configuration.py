import importlib
import os
import stat
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import polars as pl
import requests
import sutro as so

from sutro.sdk import Sutro, SutroConfigurationError
from sutro.validation import (
    DIRECT_TENSOR_FACTORY_API_ERROR,
    load_config,
    normalize_api_url,
    save_config,
)


class TestApiConfiguration(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

        self.config = patch("sutro.validation.load_config", return_value={})
        self.config.start()
        self.addCleanup(self.config.stop)

        self.version_check = patch("sutro.sdk.check_version")
        self.version_check.start()
        self.addCleanup(self.version_check.stop)

    def test_normalizes_harmonize_origin_and_v1_prefix(self):
        self.assertEqual(
            normalize_api_url("https://harmonize.example.test"),
            "https://harmonize.example.test/v1",
        )
        self.assertEqual(
            normalize_api_url("https://harmonize.example.test/v1/"),
            "https://harmonize.example.test/v1",
        )

    def test_rejects_non_deployment_paths_and_url_metadata(self):
        invalid_urls = [
            "harmonize.example.test",
            "https://harmonize.example.test/api",
            "https://harmonize.example.test?token=secret",
            "https://user:password@harmonize.example.test",
        ]
        for api_url in invalid_urls:
            with self.subTest(api_url=api_url), self.assertRaises(ValueError):
                normalize_api_url(api_url)

    def test_rejects_known_direct_tensor_factory_hosts(self):
        direct_tensor_factory_urls = [
            "https://api.sutro.sh",
            "https://api.sutro.sh/v1",
            "https://staging.api.sutro.sh",
            "https://serve.sutro.sh",
            "https://cooper-test.api.materialized.dev",
            "https://skysight-inc--tensor-factory-fastapi-app.modal.run",
            "https://skysight-inc-ci-123--tensor-factory-fastapi-app.modal.run/v1",
        ]
        for api_url in direct_tensor_factory_urls:
            with (
                self.subTest(api_url=api_url),
                self.assertRaisesRegex(
                    ValueError,
                    "Direct Sutro Batch API access is no longer supported",
                ),
            ):
                normalize_api_url(api_url)

    def test_plaintext_http_is_limited_to_loopback_development(self):
        self.assertEqual(
            normalize_api_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(
            normalize_api_url("http://[::1]:8000/v1"),
            "http://[::1]:8000/v1",
        )
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            normalize_api_url("http://harmonize.example.test")

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_persisted_config_is_owner_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = os.path.join(temporary_directory, ".sutro")
            config_file = os.path.join(config_directory, "config.json")
            with (
                patch("sutro.validation.CONFIG_DIR", config_directory),
                patch("sutro.validation.CONFIG_FILE", config_file),
            ):
                save_config({"api_key": "sk_secret"})

            self.assertEqual(stat.S_IMODE(os.stat(config_directory).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(config_file).st_mode), 0o600)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_loading_existing_config_tightens_permissions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = os.path.join(temporary_directory, ".sutro")
            config_file = os.path.join(config_directory, "config.json")
            os.mkdir(config_directory, mode=0o755)
            with open(config_file, "w") as persisted_config:
                persisted_config.write('{"api_key": "sk_secret"}')
            os.chmod(config_directory, 0o755)
            os.chmod(config_file, 0o644)

            with (
                patch("sutro.validation.CONFIG_DIR", config_directory),
                patch("sutro.validation.CONFIG_FILE", config_file),
            ):
                self.assertEqual(load_config()["api_key"], "sk_secret")

            self.assertEqual(stat.S_IMODE(os.stat(config_directory).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(config_file).st_mode), 0o600)

    def test_explicit_configuration_wins_over_environment_and_config(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "https://environment.example.test",
            }
        )
        with patch(
            "sutro.validation.load_config",
            return_value={
                "api_key": "config-key",
                "api_url": "https://config.example.test",
            },
        ):
            client = Sutro(
                api_key="explicit-key",
                api_url="https://explicit.example.test",
            )

        self.assertEqual(client.api_key, "explicit-key")
        self.assertEqual(client.api_url, "https://explicit.example.test/v1")

    def test_environment_wins_over_config(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "https://environment.example.test/v1",
            }
        )
        with patch(
            "sutro.validation.load_config",
            return_value={
                "api_key": "config-key",
                "api_url": "https://config.example.test",
            },
        ):
            client = Sutro()

        self.assertEqual(client.api_key, "environment-key")
        self.assertEqual(client.api_url, "https://environment.example.test/v1")

    def test_configuration_pair_is_loaded_from_one_snapshot(self):
        with patch(
            "sutro.validation.load_config",
            side_effect=[
                {
                    "api_key": "deployment-a-key",
                    "api_url": "https://deployment-a.example.test",
                },
                {
                    "api_key": "deployment-b-key",
                    "api_url": "https://deployment-b.example.test",
                },
            ],
        ) as load:
            client = Sutro()

        self.assertEqual(load.call_count, 1)
        self.assertEqual(client.api_key, "deployment-a-key")
        self.assertEqual(client.api_url, "https://deployment-a.example.test/v1")

    def test_environment_pair_is_loaded_from_one_snapshot(self):
        first_snapshot = {
            "SUTRO_API_KEY": "deployment-a-key",
            "SUTRO_API_URL": "https://deployment-a.example.test",
        }
        second_snapshot = {
            "SUTRO_API_KEY": "deployment-b-key",
            "SUTRO_API_URL": "https://deployment-b.example.test",
        }

        with patch(
            "sutro.validation.os.environ.copy",
            side_effect=[first_snapshot, second_snapshot],
        ) as copy_environment:
            client = Sutro()

        copy_environment.assert_called_once_with()
        self.assertEqual(client.api_key, "deployment-a-key")
        self.assertEqual(client.api_url, "https://deployment-a.example.test/v1")

    @patch("sutro.sdk.requests.get")
    def test_direct_environment_url_is_deferred_until_first_request(self, request):
        os.environ.update(
            {
                "SUTRO_API_KEY": "deployment-key",
                "SUTRO_API_URL": "https://api.sutro.sh/v1",
            }
        )

        client = Sutro()

        self.assertIsNone(client.api_url)
        with self.assertRaises(SutroConfigurationError) as raised:
            client.do_request("GET", "list-jobs")

        self.assertEqual(str(raised.exception), DIRECT_TENSOR_FACTORY_API_ERROR)
        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_malformed_environment_url_is_deferred_until_first_request(self, request):
        os.environ.update(
            {
                "SUTRO_API_KEY": "deployment-key",
                "SUTRO_API_URL": "not-an-absolute-url",
            }
        )

        client = Sutro()

        self.assertIsNone(client.api_url)
        with self.assertRaisesRegex(
            SutroConfigurationError,
            "absolute HTTP",
        ):
            client.do_request("GET", "list-jobs")
        request.assert_not_called()

    @patch("sutro.sdk.check_version")
    def test_top_level_package_reloads_with_invalid_environment_urls(
        self, _check_version
    ):
        try:
            cases = (
                ("https://api.sutro.sh/v1", DIRECT_TENSOR_FACTORY_API_ERROR),
                (
                    "not-an-absolute-url",
                    "Sutro API URL must be an absolute HTTP(S) Sutro deployment URL.",
                ),
            )
            for api_url, expected_error in cases:
                with self.subTest(api_url=api_url):
                    os.environ.update(
                        {
                            "SUTRO_API_KEY": "deployment-key",
                            "SUTRO_API_URL": api_url,
                        }
                    )
                    reloaded = importlib.reload(so)
                    self.assertIsNone(reloaded._instance.api_url)
                    with self.assertRaises(SutroConfigurationError) as raised:
                        reloaded._instance.do_request("GET", "list-jobs")
                    self.assertEqual(str(raised.exception), expected_error)
        finally:
            os.environ.clear()
            importlib.reload(so)

    def test_partial_environment_configuration_does_not_mix_with_config(self):
        persisted = {
            "api_key": "config-key",
            "api_url": "https://config.example.test",
        }
        with patch("sutro.validation.load_config", return_value=persisted):
            os.environ["SUTRO_API_URL"] = "https://environment.example.test"
            url_only_client = Sutro()

        self.assertIsNone(url_only_client.api_key)
        self.assertEqual(
            url_only_client.api_url,
            "https://environment.example.test/v1",
        )

        os.environ.clear()
        with patch("sutro.validation.load_config", return_value=persisted):
            os.environ["SUTRO_API_KEY"] = "environment-key"
            key_only_client = Sutro()

        self.assertEqual(key_only_client.api_key, "environment-key")
        self.assertIsNone(key_only_client.api_url)

    def test_partial_explicit_configuration_only_reuses_proven_pairings(self):
        persisted = {
            "api_key": "config-key",
            "api_url": "https://config.example.test",
        }
        with patch("sutro.validation.load_config", return_value=persisted):
            url_only_client = Sutro(api_url="https://explicit.example.test")
            key_only_client = Sutro(api_key="explicit-key")
            matching_url_client = Sutro(api_url="https://config.example.test/v1")

        self.assertIsNone(url_only_client.api_key)
        self.assertEqual(
            url_only_client.api_url,
            "https://explicit.example.test/v1",
        )
        self.assertEqual(key_only_client.api_key, "explicit-key")
        self.assertIsNone(key_only_client.api_url)
        self.assertEqual(matching_url_client.api_key, "config-key")
        self.assertEqual(
            matching_url_client.api_url,
            "https://config.example.test/v1",
        )

    def test_explicit_key_can_use_environment_url(self):
        os.environ["SUTRO_API_URL"] = "https://environment.example.test"

        client = Sutro(api_key="explicit-key")

        self.assertEqual(client.api_key, "explicit-key")
        self.assertEqual(client.api_url, "https://environment.example.test/v1")

    def test_explicit_url_only_reuses_environment_key_for_same_url(self):
        os.environ.update(
            {
                "SUTRO_API_KEY": "environment-key",
                "SUTRO_API_URL": "https://environment.example.test",
            }
        )

        matching = Sutro(api_url="https://environment.example.test/v1")
        different = Sutro(api_url="https://different.example.test")

        self.assertEqual(matching.api_key, "environment-key")
        self.assertIsNone(different.api_key)

    def test_config_api_url_precedes_legacy_base_url(self):
        with patch(
            "sutro.validation.load_config",
            return_value={
                "api_key": "config-key",
                "api_url": "https://current.example.test",
                "base_url": "https://legacy.example.test",
            },
        ):
            client = Sutro()

        self.assertEqual(client.api_key, "config-key")
        self.assertEqual(client.api_url, "https://current.example.test/v1")

    @patch("sutro.sdk.requests.get")
    def test_known_direct_persisted_urls_give_migration_guidance(self, request):
        configs = [
            {
                "api_key": "legacy-key",
                "base_url": "https://api.sutro.sh/",
            },
            {
                "api_key": "legacy-key",
                "api_url": "https://staging.api.sutro.sh/v1",
            },
            {
                "api_key": "legacy-key",
                "base_url": "https://cooper-test.api.materialized.dev",
            },
            {
                "api_key": "legacy-key",
                "api_url": (
                    "https://skysight-inc-staging--tensor-factory-fastapi-app.modal.run"
                ),
            },
        ]
        for config in configs:
            with (
                self.subTest(config=config),
                patch(
                    "sutro.validation.load_config",
                    return_value=config,
                ),
            ):
                client = Sutro()

            self.assertIsNone(client.api_url)
            with self.assertRaisesRegex(
                SutroConfigurationError,
                "Direct Sutro Batch API access is no longer supported",
            ):
                client.do_request("GET", "list-jobs")
        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_malformed_persisted_url_is_deferred_until_first_request(self, request):
        with patch(
            "sutro.validation.load_config",
            return_value={
                "api_key": "persisted-key",
                "api_url": "https://deployment.example.test/not-v1",
            },
        ):
            client = Sutro()

        self.assertIsNone(client.api_url)
        with self.assertRaisesRegex(
            SutroConfigurationError,
            "deployment origin or end at its /v1 API prefix",
        ):
            client.do_request("GET", "list-jobs")
        request.assert_not_called()

    def test_custom_legacy_base_url_remains_supported(self):
        with patch(
            "sutro.validation.load_config",
            return_value={
                "api_key": "legacy-key",
                "base_url": "https://harmonize.example.test",
            },
        ):
            client = Sutro()

        self.assertEqual(client.api_url, "https://harmonize.example.test/v1")

    def test_legacy_positional_base_urls_remain_source_compatible(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = Sutro(
                "legacy-key",
                "https://legacy.example.test",
                "https://legacy-serving.example.test",
            )

        self.assertEqual(client.api_url, "https://legacy.example.test/v1")
        self.assertEqual(client.base_url, client.api_url)
        self.assertEqual(client.serving_base_url, "https://legacy-serving.example.test")
        self.assertGreaterEqual(len(caught), 2)

    def test_setters_normalize_url_and_preserve_deprecated_alias(self):
        unpaired_client = Sutro(api_key="unpaired-key")
        unpaired_client.set_api_url("https://one.example.test")
        self.assertIsNone(unpaired_client.api_key)

        client = Sutro(api_key="key", api_url="https://one.example.test")
        client.set_api_url("http://localhost:8000")
        self.assertEqual(client.api_url, "http://localhost:8000/v1")
        self.assertIsNone(client.api_key)

        client.set_api_key("local-key")
        client.set_api_url("http://localhost:8000/v1")
        self.assertEqual(client.api_key, "local-key")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.set_base_url("https://two.example.test/v1")
        self.assertEqual(client.api_url, "https://two.example.test/v1")
        self.assertEqual(len(caught), 1)

    @patch("sutro.sdk.requests.get")
    def test_missing_url_fails_on_first_request_without_network(self, request):
        client = Sutro(api_key="configured-key")

        with self.assertRaisesRegex(SutroConfigurationError, "SUTRO_API_URL"):
            client.do_request("GET", "list-jobs")

        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_missing_key_fails_on_first_request_without_network(self, request):
        client = Sutro(api_url="https://harmonize.example.test")

        with self.assertRaisesRegex(SutroConfigurationError, "SUTRO_API_KEY"):
            client.do_request("GET", "list-jobs")

        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_explicit_empty_overrides_do_not_fall_back(self, request):
        client = Sutro(
            api_key="configured-key",
            api_url="https://harmonize.example.test",
        )

        with self.assertRaisesRegex(SutroConfigurationError, "SUTRO_API_KEY"):
            client.do_request("GET", "list-jobs", api_key_override="")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            client.do_request("GET", "list-jobs", base_url_override="")

        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_url_override_cannot_reuse_another_deployments_key(self, request):
        client = Sutro(
            api_key="configured-key",
            api_url="https://one.example.test",
        )

        with self.assertRaisesRegex(
            SutroConfigurationError,
            "requires api_key_override",
        ):
            client.do_request(
                "GET",
                "list-jobs",
                base_url_override="https://two.example.test",
            )

        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_url_override_rejects_direct_tensor_factory_hosts(self, request):
        client = Sutro(
            api_key="configured-key",
            api_url="https://harmonize.example.test",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Direct Sutro Batch API access is no longer supported",
        ):
            client.do_request(
                "GET",
                "list-jobs",
                api_key_override="other-key",
                base_url_override="https://api.sutro.sh/v1",
            )

        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_direct_url_guidance_precedes_missing_key_error(self, request):
        client = Sutro(api_url="https://harmonize.example.test")

        with self.assertRaisesRegex(
            ValueError,
            "Direct Sutro Batch API access is no longer supported",
        ):
            client.do_request(
                "GET",
                "list-jobs",
                base_url_override=(
                    "https://skysight-inc--tensor-factory-fastapi-app.modal.run"
                ),
            )

        request.assert_not_called()

    @patch("sutro.sdk.requests.get")
    def test_tensor_factory_410_becomes_configuration_error(self, request):
        response = MagicMock(status_code=410)
        response.json.return_value = {"detail": DIRECT_TENSOR_FACTORY_API_ERROR}
        http_error = requests.HTTPError(response=response)
        response.raise_for_status.side_effect = http_error
        request.return_value = response
        client = Sutro(
            api_key="configured-key",
            api_url="https://unrecognized-backend-alias.example.test",
        )

        with self.assertRaises(SutroConfigurationError) as raised:
            client.do_request("GET", "list-jobs")

        self.assertEqual(str(raised.exception), DIRECT_TENSOR_FACTORY_API_ERROR)
        self.assertIs(raised.exception.__cause__, http_error)
        request.assert_called_once()

    @patch("sutro.sdk.requests.get")
    def test_unrelated_410_remains_http_error(self, request):
        response = MagicMock(status_code=410)
        response.json.return_value = {"detail": "This job is no longer available."}
        http_error = requests.HTTPError(response=response)
        response.raise_for_status.side_effect = http_error
        request.return_value = response
        client = Sutro(
            api_key="configured-key",
            api_url="https://harmonize.example.test",
        )

        with self.assertRaises(requests.HTTPError) as raised:
            client.do_request("GET", "jobs/deleted-job")

        self.assertIs(raised.exception, http_error)
        request.assert_called_once()


class TestDeploymentRequestRouting(unittest.TestCase):
    def setUp(self):
        self.version_check = patch("sutro.sdk.check_version")
        self.version_check.start()
        self.addCleanup(self.version_check.stop)

    @patch("sutro.sdk.requests.post")
    def test_polars_batch_function_posts_to_harmonize_v1(self, post):
        response = MagicMock(status_code=200)
        response.json.return_value = {"results": "job-123"}
        post.return_value = response
        client = Sutro(
            api_key="sk_deployment_key",
            api_url="https://harmonize.example.test",
        )
        df = pl.DataFrame(
            {
                "company_name": [
                    "Example company_name",
                    "Example company_name 2",
                    "Example company_name 3",
                ],
                "company_domain": [
                    "Example company_domain",
                    "Example company_domain 2",
                    "Example company_domain 3",
                ],
                "company_website_url": [
                    "Example company_website_url",
                    "Example company_website_url 2",
                    "Example company_website_url 3",
                ],
            }
        )

        job_id = client.batch_run_function(
            name="sutro-lead-strength-from-jds-v2",
            data=df,
        )

        self.assertEqual(job_id, "job-123")
        post.assert_called_once()
        request_url = post.call_args.args[0]
        request_headers = post.call_args.kwargs["headers"]
        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            request_url,
            "https://harmonize.example.test/v1/batch-inference",
        )
        self.assertEqual(request_headers["Authorization"], "Key sk_deployment_key")
        self.assertEqual(payload["model"], "sutro-lead-strength-from-jds-v2")
        self.assertEqual(payload["inputs"], df.to_dicts())
        self.assertEqual(payload["job_priority"], 0)
        self.assertFalse(payload["truncate_rows"])

    @patch("sutro.sdk.requests.post")
    def test_module_level_interface_posts_to_harmonize_v1(self, post):
        response = MagicMock(status_code=200)
        response.json.return_value = {"results": "job-456"}
        post.return_value = response
        previous_api_key = so._instance.api_key
        previous_api_url = so._instance.api_url
        previous_api_url_error = so._instance._api_url_error
        try:
            so.set_api_url("https://module-level.example.test")
            so.set_api_key("module-level-key")
            job_id = so.batch_run_function(
                name="module-level-function",
                data=pl.DataFrame({"text": ["hello"]}),
            )
        finally:
            so._instance.api_key = previous_api_key
            so._instance._api_url = previous_api_url
            so._instance._api_url_error = previous_api_url_error

        self.assertEqual(job_id, "job-456")
        self.assertEqual(
            post.call_args.args[0],
            "https://module-level.example.test/v1/batch-inference",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Key module-level-key",
        )

    def test_job_result_cache_is_scoped_to_deployment_and_key(self):
        deployment_a = Sutro(
            api_key="deployment-a-key",
            api_url="https://deployment-a.example.test",
        )
        deployment_a_other_key = Sutro(
            api_key="deployment-a-other-key",
            api_url="https://deployment-a.example.test",
        )
        deployment_b = Sutro(
            api_key="deployment-b-key",
            api_url="https://deployment-b.example.test",
        )
        response = MagicMock()
        response.json.return_value = {
            "results": {"outputs": ["deployment-b-result"]},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:

            def expanduser(path):
                return path.replace("~", temporary_directory, 1)

            with patch("sutro.sdk.os.path.expanduser", side_effect=expanduser):
                deployment_a_cache_path = deployment_a._job_results_cache_file_path(
                    "job-shared"
                )
                deployment_b_cache_path = deployment_b._job_results_cache_file_path(
                    "job-shared"
                )
                deployment_a_other_key_cache_path = (
                    deployment_a_other_key._job_results_cache_file_path("job-shared")
                )
                self.assertNotEqual(
                    deployment_a_cache_path,
                    deployment_b_cache_path,
                )
                self.assertNotEqual(
                    deployment_a_cache_path,
                    deployment_a_other_key_cache_path,
                )
                os.makedirs(os.path.dirname(deployment_a_cache_path), exist_ok=True)
                pl.DataFrame(
                    {"inference_result": ["deployment-a-result"]}
                ).write_parquet(deployment_a_cache_path)

                with (
                    patch("sutro.sdk._has_open_batch_traces", return_value=False),
                    patch.object(
                        deployment_b,
                        "do_request",
                        return_value=response,
                    ) as do_request,
                ):
                    results = deployment_b.get_job_results(
                        "job-shared",
                        unpack_json=False,
                    )

        do_request.assert_called_once_with(
            "POST",
            "job-results",
            json={
                "job_id": "job-shared",
                "include_inputs": False,
                "include_cumulative_logprobs": False,
            },
        )
        self.assertEqual(
            results["inference_result"].to_list(),
            ["deployment-b-result"],
        )

    def test_legacy_unscoped_job_cache_is_deleted_instead_of_reused(self):
        client = Sutro(
            api_key="deployment-key",
            api_url="https://deployment.example.test",
        )
        response = MagicMock()
        response.json.return_value = {
            "results": {"outputs": ["fresh-deployment-result"]},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:

            def expanduser(path):
                return path.replace("~", temporary_directory, 1)

            with patch("sutro.sdk.os.path.expanduser", side_effect=expanduser):
                scoped_cache_path = client._job_results_cache_file_path("job-legacy")
                legacy_cache_path = os.path.join(
                    os.path.dirname(scoped_cache_path),
                    "job-legacy.snappy.parquet",
                )
                os.makedirs(os.path.dirname(legacy_cache_path), exist_ok=True)
                pl.DataFrame(
                    {"inference_result": ["unscoped-stale-result"]}
                ).write_parquet(legacy_cache_path)

                with (
                    patch("sutro.sdk._has_open_batch_traces", return_value=False),
                    patch.object(
                        client,
                        "do_request",
                        return_value=response,
                    ) as do_request,
                ):
                    results = client.get_job_results(
                        "job-legacy",
                        unpack_json=False,
                    )

                self.assertFalse(os.path.lexists(legacy_cache_path))
                self.assertTrue(os.path.exists(scoped_cache_path))

        do_request.assert_called_once()
        self.assertEqual(
            results["inference_result"].to_list(),
            ["fresh-deployment-result"],
        )

    def test_legacy_cache_cleanup_does_not_interpret_job_id_as_path(self):
        client = Sutro(
            api_key="deployment-key",
            api_url="https://deployment.example.test",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:

            def expanduser(path):
                return path.replace("~", temporary_directory, 1)

            outside_path = os.path.join(
                temporary_directory,
                ".sutro",
                "outside.snappy.parquet",
            )
            os.makedirs(os.path.dirname(outside_path), exist_ok=True)
            os.makedirs(
                os.path.join(temporary_directory, ".sutro", "job-results"),
                exist_ok=True,
            )
            with open(outside_path, "w") as outside_file:
                outside_file.write("keep")

            with patch("sutro.sdk.os.path.expanduser", side_effect=expanduser):
                client._remove_legacy_job_results_cache_file("../outside")

            self.assertTrue(os.path.exists(outside_path))

    @patch("sutro.sdk.requests.get")
    def test_authentication_uses_native_harmonize_check(self, get):
        response = MagicMock(status_code=200)
        response.json.return_value = {"authenticated": True}
        get.return_value = response
        client = Sutro(
            api_key="configured-key",
            api_url="https://harmonize.example.test/v1",
        )

        result = client.try_authentication("candidate-key")

        self.assertEqual(result, {"authenticated": True})
        self.assertEqual(
            get.call_args.args[0],
            "https://harmonize.example.test/v1/auth/check",
        )
        self.assertEqual(
            get.call_args.kwargs["headers"]["Authorization"],
            "Key candidate-key",
        )

    @patch("sutro.sdk.requests.post")
    def test_run_function_never_contacts_legacy_serving_host(self, post):
        client = Sutro(
            api_key="configured-key",
            api_url="https://harmonize.example.test",
        )

        with self.assertRaisesRegex(NotImplementedError, "batch_run_function"):
            client.run_function("function-name", {"text": "hello"})

        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
