"""Identify the installed SDK on requests to Sutro deployments."""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version

from requests.structures import CaseInsensitiveDict


try:
    SDK_VERSION = version("sutro")
except PackageNotFoundError:
    SDK_VERSION = "unknown"


def sdk_request_headers(headers: Mapping[str, str]) -> CaseInsensitiveDict:
    result = CaseInsensitiveDict({"User-Agent": f"sutro-python/{SDK_VERSION}"})
    result.update(headers)
    result["X-Sutro-SDK-Version"] = SDK_VERSION
    return result
