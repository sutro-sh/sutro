import importlib.metadata
import json
import os

import requests

from sutro.common import to_colored_text


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


def check_for_api_key():
    """
    Check for an API key in the user's home directory.

    This method looks for a configuration file named 'config.json' in the
    '.sutro' directory within the user's home directory.
    If the file exists, it attempts to read the API key from it.

    Returns:
        str or None: The API key if found in the configuration file, or None if not found.

    Note:
        The expected structure of the config.json file is:
        {
            "api_key": "your_api_key_here"
        }
    """
    CONFIG_DIR = os.path.expanduser("~/.sutro")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config.get("api_key")
    else:
        return None
