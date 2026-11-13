"""Offline publication safety tests: no OIDC exchange or upload is performed."""

import base64
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from urllib.error import HTTPError
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sdk_trusted_publisher", ROOT / "sutro/scripts/trusted_publisher.py"
)
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)


@pytest.fixture
def selected():
    return publisher.Selection(
        "verify-only", "a" * 40, "b" * 40, "0.2.0", "c" * 64, "d" * 64
    )


@pytest.fixture
def environment(tmp_path, selected):
    return {
        "GITHUB_REPOSITORY": "sutro-sh/foundation",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_WORKFLOW_REF": "sutro-sh/foundation/.github/workflows/"
        "harmonize-release.yml@refs/heads/main",
        "GITHUB_SHA": "e" * 40,
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123",
        "GITHUB_OUTPUT": str(tmp_path / "step-output"),
        "SUTRO_PYPI_PUBLISHING_ENABLED": "true",
        "SDK_MODE": selected.mode,
        "SDK_SOURCE_SHA": selected.source_sha,
        "SDK_TREE": selected.sdk_tree,
        "SDK_VERSION": selected.version,
        "SDK_WHEEL_SHA256": selected.wheel_sha256,
        "SDK_SDIST_SHA256": selected.sdist_sha256,
    }


def accept_after_H(environment, selected, **changes):
    receipt = {
        "schema_version": 1,
        "phase": "H_complete",
        "accepted": True,
        "source_sha": selected.source_sha,
        "sdk_tree": selected.sdk_tree,
        "version": selected.version,
        "wheel_sha256": selected.wheel_sha256,
        "sdist_sha256": selected.sdist_sha256,
        **changes,
    }
    raw = json.dumps(receipt).encode()
    checksum = hashlib.sha256(raw).hexdigest()
    environment.update(
        SDK_MODE="publish",
        SDK_SERVICE_ACCEPTANCE_B64=base64.b64encode(raw).decode(),
        SDK_SERVICE_ACCEPTANCE_SHA256=checksum,
        SDK_CONFIRMATION=f"PUBLISH sutro {selected.version} "
        f"FROM {selected.source_sha} AFTER {checksum}",
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_REPOSITORY", "someone/foundation"),
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_EVENT_NAME", "push"),
        (
            "GITHUB_WORKFLOW_REF",
            "sutro-sh/foundation/.github/workflows/other.yml@refs/heads/main",
        ),
        ("SUTRO_PYPI_PUBLISHING_ENABLED", "false"),
    ],
)
def test_both_modes_reject_wrong_repository_branch_event_or_publisher_identity(
    environment, selected, key, value
):
    for mode in ("verify-only", "publish"):
        environment[key] = value
        with pytest.raises(ValueError):
            publisher.workflow_guard(replace(selected, mode=mode), environment)


def test_verify_only_needs_no_service_receipt_or_upload_permission(
    environment, selected
):
    environment["GITHUB_RUN_ATTEMPT"] = "3"
    assert publisher.workflow_guard(selected, environment) is None


@pytest.mark.parametrize(
    "change",
    [
        {"accepted": False},
        {"accepted": "true"},
        {"phase": "G_complete"},
        {"source_sha": "f" * 40},
        {"sdk_tree": "f" * 40},
        {"wheel_sha256": "f" * 64},
        {"sdist_sha256": "f" * 64},
        {"version": "0.2.1"},
        {"credential": "must-not-be-accepted"},
    ],
)
def test_publish_rejects_unaccepted_H_or_different_source_artifacts(
    environment, selected, change
):
    accept_after_H(environment, selected, **change)
    with pytest.raises(ValueError):
        publisher.workflow_guard(replace(selected, mode="publish"), environment)


@pytest.mark.parametrize(
    "key,value",
    [
        ("SDK_CONFIRMATION", "yes"),
        ("SDK_SERVICE_ACCEPTANCE_SHA256", "0" * 64),
        ("GITHUB_RUN_ATTEMPT", "2"),
    ],
)
def test_publish_requires_exact_confirmation_and_never_replays_attempt(
    environment, selected, key, value
):
    accept_after_H(environment, selected)
    environment[key] = value
    with pytest.raises(ValueError):
        publisher.workflow_guard(replace(selected, mode="publish"), environment)


def test_publish_acceptance_is_exact_digest_and_metadata_only(environment, selected):
    accept_after_H(environment, selected)
    proof = publisher.workflow_guard(replace(selected, mode="publish"), environment)
    assert proof["sha256"] == environment["SDK_SERVICE_ACCEPTANCE_SHA256"]
    assert proof["receipt"]["phase"] == "H_complete"


def distributions(directory, selected, *, wheel_name="sutro", sdist_version="0.2.0"):
    directory.mkdir(parents=True)
    wheel = directory / f"sutro-{selected.version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"sutro-{selected.version}.dist-info/METADATA",
            f"Name: {wheel_name}\nVersion: {selected.version}\n",
        )
    sdist = directory / f"sutro-{selected.version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        body = f'[project]\nname="sutro"\nversion="{sdist_version}"\n'.encode()
        info = tarfile.TarInfo(f"sutro-{selected.version}/pyproject.toml")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    return replace(
        selected,
        wheel_sha256=publisher.digest(wheel),
        sdist_sha256=publisher.digest(sdist),
    )


@pytest.fixture
def artifacts(tmp_path, selected):
    root = tmp_path / "original-build"
    selected = distributions(root / "dist", selected)
    receipt = {
        "schema_version": 1,
        "source_sha": selected.source_sha,
        "sdk_tree": selected.sdk_tree,
        "version": selected.version,
        "uv_version": "0.13.6",
        "files": publisher.validate_distributions(root / "dist", selected),
    }
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root, selected


@pytest.mark.parametrize("output_kind", ["uv-marker", "wrong-marker", "unknown-file"])
def test_fresh_build_removes_only_uv_known_output_marker(
    tmp_path, selected, environment, monkeypatch, output_kind
):
    original = tmp_path / "prepared"
    selected = distributions(original, selected)
    monkeypatch.setattr(publisher, "source_guard", lambda *args: None)
    monkeypatch.setattr(
        publisher.subprocess, "check_output", lambda *args, **kwargs: "uv 0.13.6\n"
    )

    def fake_build(command, **kwargs):
        destination = Path(command[-1])
        publisher.shutil.copytree(original, destination)
        if output_kind == "uv-marker":
            (destination / ".gitignore").write_bytes(b"*")
        elif output_kind == "wrong-marker":
            (destination / ".gitignore").write_bytes(b"other")
        else:
            (destination / "unreviewed.whl").write_bytes(b"unknown")

    monkeypatch.setattr(publisher.subprocess, "run", fake_build)
    output = tmp_path / "build"
    if output_kind == "uv-marker":
        receipt = publisher.build(ROOT, ROOT, output, selected, environment)
        assert {path.name for path in (output / "dist").iterdir()} == set(
            selected.files
        )
        assert receipt == publisher.load_artifacts(output, selected)
    else:
        with pytest.raises(ValueError):
            publisher.build(ROOT, ROOT, output, selected, environment)
        assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("change", ["extra", "bytes", "symlink", "receipt"])
def test_original_build_artifacts_cannot_be_substituted(artifacts, change):
    root, selected = artifacts
    wheel = root / "dist" / next(iter(selected.files))
    if change == "extra":
        (root / "dist/other.whl").write_bytes(b"unreviewed")
    elif change == "bytes":
        wheel.write_bytes(wheel.read_bytes() + b"changed")
    elif change == "symlink":
        outside = root / "external"
        wheel.rename(outside)
        wheel.symlink_to(outside)
    else:
        receipt = json.loads((root / "receipt.json").read_text())
        receipt["source_sha"] = "f" * 40
        (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        publisher.load_artifacts(root, selected)


@pytest.mark.parametrize(
    "wheel_name,sdist_version", [("other", "0.2.0"), ("sutro", "0.2.1")]
)
def test_even_reviewed_hashes_must_have_the_expected_package_metadata(
    tmp_path, selected, wheel_name, sdist_version
):
    selected = distributions(
        tmp_path / "dist", selected, wheel_name=wheel_name, sdist_version=sdist_version
    )
    with pytest.raises(ValueError):
        publisher.validate_distributions(tmp_path / "dist", selected)


@pytest.mark.parametrize("existing", [0, 1, 2])
def test_only_missing_exact_original_distributions_are_selected(
    artifacts, environment, tmp_path, monkeypatch, existing
):
    root, selected = artifacts
    published = dict(list(selected.files.items())[:existing])
    monkeypatch.setattr(publisher, "public_files", lambda version: published)
    output = tmp_path / "sdk-upload"
    intent = publisher.select_upload(root, output, selected, environment)
    assert {path.name for path in output.iterdir()} == set(selected.files) - set(
        published
    )
    assert intent["missing"] == list(
        name for name in selected.files if name not in published
    )
    assert intent["phase"] == "verification_prepared"
    assert json.loads((tmp_path / "publication-intent.json").read_text()) == intent


def test_collision_or_unknown_public_result_never_selects_an_upload(
    artifacts, environment, tmp_path, monkeypatch
):
    root, selected = artifacts
    for published in (
        {next(iter(selected.files)): "0" * 64},
        {"unexpected.whl": "0" * 64},
    ):
        monkeypatch.setattr(publisher, "public_files", lambda _: published)
        with pytest.raises(ValueError):
            publisher.select_upload(
                root, tmp_path / "not-created", selected, environment
            )
        assert not (tmp_path / "not-created").exists()
    monkeypatch.setattr(
        publisher,
        "public_files",
        lambda _: (_ for _ in ()).throw(
            HTTPError("https://pypi.org", 429, "busy", {}, None)
        ),
    )
    with pytest.raises(HTTPError):
        publisher.select_upload(root, tmp_path / "not-created", selected, environment)
    assert not (tmp_path / "not-created").exists()


def test_incomplete_or_unknown_publication_is_not_reported_success(
    artifacts, environment, tmp_path, monkeypatch
):
    root, selected = artifacts
    monkeypatch.setattr(
        publisher, "public_files", lambda _: dict(list(selected.files.items())[:1])
    )
    with pytest.raises(ValueError, match="never auto-retry"):
        publisher.verify_published(
            root, tmp_path / "published.json", selected, environment
        )
    assert not (tmp_path / "published.json").exists()
    monkeypatch.setattr(publisher, "public_files", lambda _: selected.files)
    assert (
        publisher.verify_published(
            root, tmp_path / "published.json", selected, environment
        )["phase"]
        == "published_exact_files_verified"
    )


def test_public_metadata_redirects_are_refused():
    assert (
        publisher.NoRedirect().redirect_request(None, None, None, None, None, None)
        is None
    )


def command(*arguments, cwd):
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def test_real_git_ancestry_rejects_a_clean_unmerged_sdk_source(
    tmp_path, selected, environment
):
    tools = tmp_path / "tools"
    tools.mkdir()
    command("init", "-b", "main", cwd=tools)
    command("config", "user.email", "fixture@example.invalid", cwd=tools)
    command("config", "user.name", "Fixture", cwd=tools)
    (tools / "sutro").mkdir()
    (tools / "sutro/pyproject.toml").write_text(
        '[project]\nname="sutro"\nversion="0.2.0"\n'
    )
    command("add", ".", cwd=tools)
    command("commit", "-m", "protected source", cwd=tools)
    accepted = command("rev-parse", "HEAD", cwd=tools)
    tree = command("rev-parse", "HEAD:sutro", cwd=tools)
    command("switch", "-c", "unmerged", cwd=tools)
    (tools / "sutro/unreviewed.py").write_text("unmerged=True\n")
    command("add", ".", cwd=tools)
    command("commit", "-m", "unmerged source", cwd=tools)
    unmerged = command("rev-parse", "HEAD", cwd=tools)
    unmerged_tree = command("rev-parse", "HEAD:sutro", cwd=tools)
    command("switch", "main", cwd=tools)
    source = tmp_path / "source"
    command("clone", "--shared", str(tools), str(source), cwd=tmp_path)
    environment["GITHUB_SHA"] = accepted
    publisher.source_guard(
        tools,
        source,
        replace(selected, source_sha=accepted, sdk_tree=tree),
        environment,
    )
    command("switch", "--detach", unmerged, cwd=source)
    with pytest.raises(ValueError, match="not an ancestor"):
        publisher.source_guard(
            tools,
            source,
            replace(selected, source_sha=unmerged, sdk_tree=unmerged_tree),
            environment,
        )


def test_restored_workflow_is_manual_sdk_only_and_keeps_oidc_identity():
    text = (ROOT / publisher.WORKFLOW).read_text()
    triggers, jobs = text.split("on:\n", 1)[1].split("jobs:\n", 1)
    assert re.findall(r"^  ([a-z_]+):$", triggers, re.M) == ["workflow_dispatch"]
    assert "default: verify-only" in triggers
    assert re.findall(r"^  ([a-z_]+):$", jobs, re.M) == ["sdk_build", "sdk_publish"]
    build, publish = jobs.split("  sdk_publish:\n")
    assert "id-token" not in build
    assert "      contents: read\n      id-token: write" in publish
    assert "name: sutro-pypi" in publish
    action = publish.split("uses: pypa/", 1)[1].split("      - name:", 1)[0]
    assert action.startswith(
        "gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e"
    )
    assert "attestations: true" in action and "skip-existing: false" in action
    assert (
        "if: inputs.mode == 'publish' && "
        "steps.selection.outputs.upload_required == 'true'" in publish
    )
    assert "uv publish --dry-run --trusted-publishing always" in text
    assert not any(
        word in text
        for word in (
            "aws-actions/",
            "secrets.",
            "git push",
            "make -C harmonize",
            "make -C admin",
        )
    )
