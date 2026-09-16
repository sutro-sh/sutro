import importlib.metadata
import ipaddress
import json
import os
import tempfile
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from sutro.common import to_colored_text


CONFIG_DIR = os.path.expanduser("~/.sutro")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
API_KEY_ENV = "SUTRO_API_KEY"
API_URL_ENV = "SUTRO_API_URL"
DIRECT_TENSOR_FACTORY_API_ERROR = (
    "Direct Sutro Batch API access is no longer supported by the Sutro SDK. "
    "Upgrade the sutro package, then set SUTRO_API_URL to your Sutro "
    "single-tenant deployment URL and use an API key from that deployment's "
    "API Keys panel."
)
CENTRALIZED_API_HOST_SUFFIXES = (
    "api.sutro.sh",
    "serve.sutro.sh",
    "api.materialized.dev",
)
TENSOR_FACTORY_MODAL_HOST_SUFFIX = "--tensor-factory-fastapi-app.modal.run"


def _tighten_existing_config_permissions() -> None:
    """Make an existing persisted credential owner-only on POSIX systems."""
    if os.name != "posix":
        return
    if os.path.isdir(CONFIG_DIR):
        os.chmod(CONFIG_DIR, 0o700)
    if os.path.isfile(CONFIG_FILE):
        os.chmod(CONFIG_FILE, 0o600)


def check_version(package_name: str):
    try:
        # Local version
        local_version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        print(f"{package_name} is not installed.")
        return

    try:
        # Latest release from PyPI
        resp = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=2)
        resp.raise_for_status()
        latest_version = resp.json()["info"]["version"]

        if local_version != latest_version:
            msg = (
                f"⚠️  You are using {package_name} {local_version}, "
                f"but the latest release is {latest_version}. "
                f"Run `[uv] pip install -U {package_name}` to upgrade."
            )
            print(to_colored_text(msg, state="callout"))
    except Exception:
        # Fail silently or log, you don’t want this blocking usage
        pass


def load_config() -> dict[str, Any]:
    """Load the user's persisted Sutro configuration."""
    if os.path.exists(CONFIG_FILE):
        _tighten_existing_config_permissions()
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config if isinstance(config, dict) else {}
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Atomically persist Sutro configuration with owner-only permissions."""
    os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    if os.name == "posix":
        os.chmod(CONFIG_DIR, 0o700)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".config-", dir=CONFIG_DIR, text=True
    )
    try:
        if os.name == "posix" and hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        else:  # pragma: no cover - Windows uses path-based chmod.
            os.chmod(temporary_path, 0o600)
        opened_file = os.fdopen(descriptor, "w")
        descriptor = -1
        with opened_file as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, CONFIG_FILE)
        if os.name == "posix":
            os.chmod(CONFIG_FILE, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _nonempty_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _is_known_direct_tensor_factory_api_url(value: str) -> bool:
    """Return whether a URL targets a known direct Tensor Factory host."""
    # Requests decodes unreserved hostname escapes and normalizes IDNA before
    # sending a request. Inspect that same destination so an encoded hostname
    # cannot bypass the deployment-only routing rule. Preparing sends no I/O.
    try:
        prepared_url = requests.Request("GET", value).prepare().url
    except requests.RequestException as exc:
        raise ValueError(
            "Sutro API URL must be a valid absolute HTTP(S) deployment URL."
        ) from exc
    parsed = urlsplit(prepared_url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return hostname.endswith(TENSOR_FACTORY_MODAL_HOST_SUFFIX) or any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in CENTRALIZED_API_HOST_SUFFIXES
    )


def _api_url_resolution_from_config(
    config: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve a persisted URL without letting invalid config break imports."""
    configured_url = _nonempty_string(config.get("api_url"))
    if configured_url is not None:
        return _normalize_resolved_api_url(configured_url)

    legacy_base_url = _nonempty_string(config.get("base_url"))
    return _normalize_resolved_api_url(legacy_base_url)


def _normalize_resolved_api_url(
    api_url: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Normalize inherited config while preserving a deferred error message."""
    if api_url is None:
        return None, None
    try:
        return normalize_api_url(api_url), None
    except ValueError as exc:
        # Environment and persisted configuration are evaluated during
        # ``import sutro``. Preserve the reason for rejection, but defer it to
        # the first request so login and set-api-url remain available.
        return None, str(exc)


def resolve_api_configuration_with_context() -> tuple[
    Optional[str], Optional[str], Optional[str]
]:
    """Resolve a credential pair plus any deferred URL validation error."""
    environment = os.environ.copy()
    if API_KEY_ENV in environment or API_URL_ENV in environment:
        # URL and key form one credential-routing pair. If either environment
        # variable is present, never borrow the missing half from persisted
        # configuration for a potentially different deployment.
        return _environment_api_configuration_with_context(environment)

    config = load_config()
    api_url, api_url_error = _api_url_resolution_from_config(config)
    return (
        _nonempty_string(config.get("api_key")),
        api_url,
        api_url_error,
    )


def resolve_environment_api_configuration_with_context() -> tuple[
    Optional[str], Optional[str], Optional[str]
]:
    """Resolve only environment credentials and deferred URL errors."""
    return _environment_api_configuration_with_context(os.environ.copy())


def _environment_api_configuration_with_context(
    environment: dict[str, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a key and URL from one immutable environment snapshot."""
    api_url, api_url_error = _normalize_resolved_api_url(
        _nonempty_string(environment.get(API_URL_ENV))
    )
    return _nonempty_string(environment.get(API_KEY_ENV)), api_url, api_url_error


def resolve_api_configuration() -> tuple[Optional[str], Optional[str]]:
    """Resolve the API key and URL together from one configuration snapshot."""
    api_key, api_url, _ = resolve_api_configuration_with_context()
    return api_key, api_url


def check_for_api_key() -> Optional[str]:
    """Resolve an API key from the environment, then the user config file."""
    api_key, _ = resolve_api_configuration()
    return api_key


def check_for_api_url() -> Optional[str]:
    """Resolve an API URL from the environment, then current/legacy config."""
    _, api_url = resolve_api_configuration()
    return api_url


def normalize_api_url(api_url: str) -> str:
    """Return the canonical ``<Sutro deployment origin>/v1`` API prefix.

    Users may provide either a deployment origin or the deployment's ``/v1``
    API prefix. Other paths are rejected so credentials cannot accidentally be
    sent to an unrelated endpoint.
    """
    value = _nonempty_string(api_url)
    if value is None:
        raise ValueError(
            "Sutro API URL cannot be empty. Set SUTRO_API_URL to your "
            "Sutro deployment URL."
        )

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Sutro API URL must be an absolute HTTP(S) Sutro deployment URL."
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Sutro API URL must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Sutro API URL must not contain a query string or fragment.")
    if _is_known_direct_tensor_factory_api_url(value):
        raise ValueError(DIRECT_TENSOR_FACTORY_API_ERROR)
    if parsed.scheme == "http":
        hostname = parsed.hostname
        is_loopback = hostname == "localhost"
        if hostname is not None and not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError(
                "Sutro API URL must use HTTPS unless the deployment is on "
                "localhost or a loopback address."
            )

    path = parsed.path.rstrip("/")
    if path in {"", "/v1"}:
        path = "/v1"
    else:
        raise ValueError(
            "Sutro API URL must be a Sutro deployment origin or end at its /v1 "
            "API prefix."
        )

    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
