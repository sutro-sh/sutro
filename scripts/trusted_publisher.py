"""SDK-only publication guards; uploads remain owned by the official PyPA action.

Public PyPI metadata reads never receive credentials. Trusted-publisher minting
is performed only by uv's verify-only command or the official publishing action.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tomllib
from urllib import error, request
import zipfile


REPOSITORY = "sutro-sh/foundation"
WORKFLOW = ".github/workflows/harmonize-release.yml"
UV_VERSION = "0.13.6"
SHA = re.compile(r"[a-f0-9]{40}\Z")
DIGEST = re.compile(r"[a-f0-9]{64}\Z")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Selection:
    mode: str
    source_sha: str
    sdk_tree: str
    version: str
    wheel_sha256: str
    sdist_sha256: str

    @classmethod
    def from_env(cls, environment: dict[str, str]) -> Selection:
        selected = cls(
            *(
                environment[name]
                for name in (
                    "SDK_MODE",
                    "SDK_SOURCE_SHA",
                    "SDK_TREE",
                    "SDK_VERSION",
                    "SDK_WHEEL_SHA256",
                    "SDK_SDIST_SHA256",
                )
            )
        )
        require(selected.mode in {"verify-only", "publish"}, "Unknown SDK mode")
        require(bool(SHA.fullmatch(selected.source_sha)), "Use a full source SHA")
        require(bool(SHA.fullmatch(selected.sdk_tree)), "Use a full SDK tree SHA")
        require(
            bool(VERSION.fullmatch(selected.version)), "Use a canonical SDK version"
        )
        require(
            all(
                DIGEST.fullmatch(value)
                for value in (selected.wheel_sha256, selected.sdist_sha256)
            ),
            "Use exact reviewed SHA256 distributions",
        )
        return selected

    @property
    def files(self) -> dict[str, str]:
        return {
            f"sutro-{self.version}-py3-none-any.whl": self.wheel_sha256,
            f"sutro-{self.version}.tar.gz": self.sdist_sha256,
        }


def workflow_guard(selected: Selection, environment: dict[str, str]) -> dict | None:
    require(
        environment.get("GITHUB_REPOSITORY") == REPOSITORY,
        "Publisher requires the foundation repository",
    )
    require(
        environment.get("GITHUB_REF") == "refs/heads/main",
        "Publisher requires protected main",
    )
    require(
        environment.get("GITHUB_EVENT_NAME") == "workflow_dispatch",
        "Publisher is manual only",
    )
    require(
        environment.get("GITHUB_WORKFLOW_REF")
        == f"{REPOSITORY}/{WORKFLOW}@refs/heads/main",
        "Use the reviewed publisher workflow identity",
    )
    require(
        environment.get("SUTRO_PYPI_PUBLISHING_ENABLED") == "true",
        "Protected SDK publishing is disabled",
    )
    if selected.mode == "verify-only":
        return None
    require(
        environment.get("GITHUB_RUN_ATTEMPT") == "1",
        "Do not replay a publishing attempt; reconcile public filenames "
        "and the original receipt",
    )
    expected = environment.get("SDK_SERVICE_ACCEPTANCE_SHA256", "")
    require(
        bool(DIGEST.fullmatch(expected)),
        "Publishing requires the post-H service-acceptance digest",
    )
    encoded = environment.get("SDK_SERVICE_ACCEPTANCE_B64", "")
    require(len(encoded) <= 32768, "Acceptance receipt exceeds its metadata bound")
    raw = base64.b64decode(encoded, validate=True)
    require(
        hashlib.sha256(raw).hexdigest() == expected,
        "Service-acceptance receipt digest differs",
    )
    receipt = json.loads(raw)
    required = {
        "schema_version": 1,
        "phase": "H_complete",
        "accepted": True,
        "source_sha": selected.source_sha,
        "sdk_tree": selected.sdk_tree,
        "version": selected.version,
        "wheel_sha256": selected.wheel_sha256,
        "sdist_sha256": selected.sdist_sha256,
    }
    require(
        type(receipt) is dict and set(receipt) == set(required),
        "Use only the nonsecret service-acceptance schema",
    )
    require(
        all(
            type(receipt[key]) is type(value) and receipt[key] == value
            for key, value in required.items()
        ),
        "Services/H or SDK artifact identity has not been accepted",
    )
    confirmation = (
        f"PUBLISH sutro {selected.version} FROM {selected.source_sha} AFTER {expected}"
    )
    require(
        environment.get("SDK_CONFIRMATION") == confirmation,
        "Explicit publication confirmation does not match this accepted release",
    )
    return {"sha256": expected, "receipt": receipt}


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def source_guard(
    tools: Path, source: Path, selected: Selection, environment: dict[str, str]
) -> None:
    workflow_sha = environment["GITHUB_SHA"]
    require(bool(SHA.fullmatch(workflow_sha)), "Use the workflow's full main SHA")
    require(
        git(tools, "rev-parse", "HEAD") == workflow_sha,
        "Publisher verifier differs from the workflow main commit",
    )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(tools),
            "merge-base",
            "--is-ancestor",
            selected.source_sha,
            workflow_sha,
        ],
        capture_output=True,
    )
    require(
        ancestor.returncode == 0,
        "SDK source is not an ancestor of the protected-main workflow commit",
    )
    require(
        git(source, "rev-parse", "HEAD") == selected.source_sha,
        "SDK source checkout changed",
    )
    require(not git(source, "status", "--porcelain"), "SDK source must be clean")
    require(
        git(source, "rev-parse", "HEAD:sutro") == selected.sdk_tree,
        "SDK source tree differs from review",
    )
    project = tomllib.loads((source / "sutro/pyproject.toml").read_text())["project"]
    require(
        project["name"] == "sutro" and project["version"] == selected.version,
        "SDK project/version differs from review",
    )


def validate_distributions(directory: Path, selected: Selection) -> list[dict]:
    require(
        {path.name for path in directory.iterdir()} == set(selected.files),
        "Build must contain exactly the reviewed wheel and sdist",
    )
    result = []
    for name, expected in selected.files.items():
        path = directory / name
        require(
            path.is_file() and not path.is_symlink(),
            "SDK distribution must be an ordinary file",
        )
        require(
            digest(path) == expected,
            "SDK distribution differs from its reviewed SHA256",
        )
        if name.endswith(".whl"):
            with zipfile.ZipFile(path) as archive:
                metadata = BytesParser().parsebytes(
                    archive.read(f"sutro-{selected.version}.dist-info/METADATA")
                )
                require(
                    metadata["Name"] == "sutro"
                    and metadata["Version"] == selected.version,
                    "Wheel package identity differs",
                )
        else:
            with tarfile.open(path, "r:gz") as archive:
                metadata = archive.extractfile(
                    f"sutro-{selected.version}/pyproject.toml"
                )
                require(metadata is not None, "Sdist project metadata missing")
                project = tomllib.loads(metadata.read().decode())["project"]
                require(
                    project["name"] == "sutro"
                    and project["version"] == selected.version,
                    "Sdist package identity differs",
                )
        result.append(
            {"filename": name, "sha256": expected, "bytes": path.stat().st_size}
        )
    return result


def build(
    tools: Path,
    source: Path,
    output: Path,
    selected: Selection,
    environment: dict[str, str],
) -> dict:
    workflow_guard(selected, environment)
    source_guard(tools, source, selected, environment)
    version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
    require(version == UV_VERSION, "Use the reviewed uv 0.13.6 builder")
    output.mkdir(parents=True, exist_ok=False)
    dist = output / "dist"
    # uv build uses isolated build requirements; it does not sync SDK runtime
    # dependencies or rewrite the intentionally retained SDK uv.lock.
    subprocess.run(
        ["uv", "build", "--python", "3.13", "--no-sources", "--out-dir", str(dist)],
        cwd=source / "sutro",
        check=True,
        timeout=300,
    )
    # uv 0.13.6 creates this exact cache-exclusion marker beside its output.
    # Remove only that known marker in our fresh owned directory; artifact
    # validation still requires exactly the two reviewed distributions.
    marker = dist / ".gitignore"
    if marker.exists() or marker.is_symlink():
        require(
            marker.is_file()
            and not marker.is_symlink()
            and marker.read_bytes() == b"*",
            "Unexpected uv output marker; retain the build for inspection",
        )
        marker.unlink()
    source_guard(tools, source, selected, environment)
    files = validate_distributions(dist, selected)
    receipt = {
        "schema_version": 1,
        "source_sha": selected.source_sha,
        "sdk_tree": selected.sdk_tree,
        "version": selected.version,
        "uv_version": version,
        "files": files,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_artifacts(artifacts: Path, selected: Selection) -> dict:
    require(
        {path.name for path in artifacts.iterdir()} == {"dist", "receipt.json"},
        "Unexpected SDK build artifact entries",
    )
    receipt = json.loads((artifacts / "receipt.json").read_text())
    require(
        receipt["schema_version"] == 1
        and receipt["source_sha"] == selected.source_sha
        and receipt["sdk_tree"] == selected.sdk_tree
        and receipt["version"] == selected.version
        and receipt["uv_version"] == UV_VERSION,
        "Original SDK build receipt differs from this request",
    )
    require(
        receipt["files"] == validate_distributions(artifacts / "dist", selected),
        "Original distribution receipt changed",
    )
    return receipt


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def public_files(version: str) -> dict[str, str]:
    opener = request.build_opener(NoRedirect())
    try:
        with opener.open(
            f"https://pypi.org/pypi/sutro/{version}/json", timeout=15
        ) as response:
            data = response.read(2 * 1024**2 + 1)
    except error.HTTPError as failure:
        if failure.code == 404:
            return {}
        raise
    require(len(data) <= 2 * 1024**2, "PyPI metadata exceeds its bound")
    payload = json.loads(data)
    require(
        payload["info"]["name"] == "sutro" and payload["info"]["version"] == version,
        "PyPI metadata package/version differs",
    )
    files = {item["filename"]: item["digests"]["sha256"] for item in payload["urls"]}
    require(len(files) == len(payload["urls"]), "Duplicate PyPI distribution names")
    return files


def missing_files(selected: Selection, published: dict[str, str]) -> list[str]:
    require(
        set(published) <= set(selected.files),
        "Unexpected public distributions for the selected SDK version",
    )
    require(
        all(selected.files[name] == checksum for name, checksum in published.items()),
        "Existing PyPI filename has a different SHA256; do not overwrite or skip it",
    )
    return [name for name in selected.files if name not in published]


def select_upload(
    artifacts: Path, output: Path, selected: Selection, environment: dict[str, str]
) -> dict:
    acceptance = workflow_guard(selected, environment)
    receipt = load_artifacts(artifacts, selected)
    missing = missing_files(selected, public_files(selected.version))
    output.mkdir(parents=True, exist_ok=False)
    for name in missing:
        shutil.copyfile(artifacts / "dist" / name, output / name)
        require(
            digest(output / name) == selected.files[name],
            "Selected upload bytes changed",
        )
    intent = {
        "mode": selected.mode,
        "source_sha": selected.source_sha,
        "sdk_tree": selected.sdk_tree,
        "version": selected.version,
        "files": receipt["files"],
        "missing": missing,
        "run_id": environment["GITHUB_RUN_ID"],
        "run_attempt": environment["GITHUB_RUN_ATTEMPT"],
        "service_acceptance": acceptance,
        "phase": "verification_prepared"
        if selected.mode == "verify-only"
        else "publish_dispatch_intent",
        "recovery": "No automatic upload replay. Inspect this original run and "
        "public filenames/SHA256; use a fresh verify-only dispatch for reconciliation.",
    }
    (output.parent / "publication-intent.json").write_text(
        json.dumps(intent, indent=2) + "\n"
    )
    with Path(environment["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"upload_required={'true' if missing else 'false'}\n")
    return intent


def verify_published(
    artifacts: Path, output: Path, selected: Selection, environment: dict[str, str]
) -> dict:
    workflow_guard(selected, environment)
    receipt = load_artifacts(artifacts, selected)
    missing = missing_files(selected, public_files(selected.version))
    require(
        not missing,
        "Public publication is incomplete or unconfirmed; inspect the original "
        "operation, never auto-retry its upload",
    )
    result = {
        "phase": "published_exact_files_verified",
        "source_sha": selected.source_sha,
        "sdk_tree": selected.sdk_tree,
        "version": selected.version,
        "files": receipt["files"],
        "run_id": environment["GITHUB_RUN_ID"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("build", "select", "verify", "record-dry-run")
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = Selection.from_env(os.environ)
    if args.action == "build":
        result = build(
            Path(__file__).resolve().parents[2],
            args.source,
            args.artifacts,
            selected,
            os.environ,
        )
    elif args.action == "select":
        result = select_upload(args.artifacts, args.output, selected, os.environ)
    elif args.action == "verify":
        result = verify_published(args.artifacts, args.output, selected, os.environ)
    else:
        workflow_guard(selected, os.environ)
        require(selected.mode == "verify-only", "Dry-run receipt is verify-only")
        receipt = load_artifacts(args.artifacts, selected)
        result = {
            "phase": "uv_trusted_dry_run_completed_no_upload",
            "source_sha": selected.source_sha,
            "sdk_tree": selected.sdk_tree,
            "version": selected.version,
            "files": receipt["files"],
            "run_id": os.environ["GITHUB_RUN_ID"],
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
