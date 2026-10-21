"""Internal batch submission protocol and CLI receipt orchestration.

Receipts contain object identities and job IDs, never AWS credentials or signed
URLs. They are saved before submission; only an explicit new run gets a new key.
"""

from contextlib import contextmanager
import ctypes
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
import sys

from sutro._batch_output import ResultShapeError, normalize_results


JOB_ID = re.compile(r"job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
FORMATS = {"csv", "parquet", "jsonl"}
POLL_SECONDS = 15
try:
    _VERSION = importlib.metadata.version("sutro")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"
USER_AGENT = f"SutroBatch/{_VERSION}"


class BatchError(Exception):
    """A safe user-facing failure, without input content or credentials."""


class NetworkError(BatchError):
    """A transient transport failure; read-only polling may retry it."""


class ApiError(BatchError):
    def __init__(self, status, retryable=False, code=None):
        self.status = status
        self.retryable = retryable
        self.code = code
        hints = {
            401: "Check the deployment API key.",
            403: "Check deployment access permissions.",
            410: "This job or its results have expired; no replacement was submitted.",
            409: "The saved submission conflicts with this request; "
            "inspect it before starting a new run.",
            422: "Check the function, dataset format, and request fields. "
            "After correcting a terminal failure, use a new submission key "
            "(--new-run in the CLI).",
        }
        code_text = f" [{code}]" if isinstance(code, str) and code else ""
        super().__init__(
            f"Deployment returned HTTP {status}{code_text}. "
            f"{hints.get(status, 'The receipt is preserved.')}"
        )


def _json_request(client, method, endpoint, deadline, body=None, *, params=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BatchError("Client time limit reached; the remote job was not cancelled.")
    try:
        with client.do_request(
            method,
            endpoint,
            max_retries=0,
            allow_redirects=False,
            headers={"User-Agent": USER_AGENT},
            timeout=(min(5, remaining), min(45, remaining)),
            **({"json": body} if body is not None else {}),
            **({"params": params} if params is not None else {}),
        ) as response:
            if 300 <= response.status_code < 400:
                raise BatchError("The deployment redirected the API request; no credentials were forwarded.")
            if response.status_code not in (200, 202):
                raise ApiError(response.status_code)
            data = response.json()
            if not isinstance(data, dict):
                raise BatchError("The deployment returned an invalid API response.")
            return response.status_code, data
    except requests.HTTPError as error:
        try:
            details = error.response.json()
            retryable = isinstance(details, dict) and details.get("retryable") is True
            code = details.get("code") if isinstance(details, dict) else None
        except ValueError:
            retryable = False
            code = None
        raise ApiError(
            error.response.status_code, retryable=retryable, code=code
        ) from None
    except requests.RequestException as error:
        # requests exception messages can contain presigned URLs.
        raise NetworkError(
            f"Network request failed ({type(error).__name__})."
        ) from None
    except (ValueError, UnicodeError):
        raise BatchError("The deployment did not return JSON.") from None


def _save_receipt(path, receipt):
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _output_lock(output):
    # flock is released by the OS on crash; an empty lock file is harmless.
    try:
        import fcntl
    except ImportError:
        raise BatchError("Resumable batch runs currently require macOS or Linux.") from None
    lock_path = Path(str(output) + ".sutro.lock")
    if lock_path.is_symlink():
        raise BatchError("The output lock must not be a symbolic link.")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BatchError(
                "Another process is already using this output path."
            ) from None
        yield
    finally:
        os.close(descriptor)


def _s3_location(dataset):
    try:
        uri = urlsplit(dataset)
        invalid_authority = uri.username or uri.port
    except (ValueError, TypeError):
        raise BatchError("Invalid S3 object URI.") from None
    key = uri.path.removeprefix("/")
    if (
        uri.scheme != "s3"
        or not uri.netloc
        or not key
        or uri.query
        or uri.fragment
        or invalid_authority
        or any(c in dataset for c in "*[]")
    ):
        raise BatchError(
            "--dataset must identify one S3 object: "
            "s3://bucket/path/file.csv, .parquet, or .jsonl."
        )
    extension = Path(key).suffix.lower().removeprefix(".")
    if extension not in FORMATS:
        raise BatchError(
            "The S3 object must be CSV, Parquet, or JSONL "
            "(not a directory or compressed CSV/JSONL)."
        )
    return uri.netloc, key, extension


def _s3_client(region=None):
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def _source_identity(dataset):
    bucket, key, _ = _s3_location(dataset)
    try:
        client = _s3_client()
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        raise BatchError(
            f"Cannot inspect the S3 object ({type(error).__name__}); "
            "check local AWS credentials and S3/KMS permissions."
        ) from None
    if (
        not head.get("ETag")
        or not isinstance(head.get("ContentLength"), int)
        or head["ContentLength"] <= 0
    ):
        raise BatchError(
            "The S3 object is empty or lacks a verifiable object identity."
        )
    source = {
        "bucket": bucket,
        "key": key,
        "etag": head["ETag"],
        "size_bytes": head["ContentLength"],
    }
    if head.get("VersionId") not in (None, "null"):
        source["version_id"] = head["VersionId"]
    region = (
        head.get("ResponseMetadata", {})
        .get("HTTPHeaders", {})
        .get("x-amz-bucket-region")
    )
    if not region:
        # HeadObject redirects update botocore's private cache, not
        # client.meta.region_name. HeadBucket exposes the actual region even
        # when the caller has GetObject access but no ListBucket permission.
        try:
            bucket_metadata = client.head_bucket(Bucket=bucket)
        except ClientError as error:
            bucket_metadata = error.response
        except Exception as error:
            raise BatchError(
                f"Cannot determine the S3 bucket region ({type(error).__name__})."
            ) from None
        region = (
            bucket_metadata.get("ResponseMetadata", {})
            .get("HTTPHeaders", {})
            .get("x-amz-bucket-region")
        )
        if not region:
            raise BatchError(
                "S3 did not return the bucket region; no job was submitted."
            )
    return source, region


def _signed_input(source, region=None):
    params = {"Bucket": source["bucket"], "Key": source["key"]}
    if source.get("version_id"):
        params["VersionId"] = source["version_id"]
    try:
        return _s3_client(region).generate_presigned_url(
            "get_object", Params=params, ExpiresIn=3600
        )
    except Exception as error:
        raise BatchError(
            f"Cannot sign the saved S3 object ({type(error).__name__}); "
            "check local AWS credentials."
        ) from None


def _check_capabilities(client, extension, deadline):
    try:
        _, capabilities = _json_request(client, "GET", "batch-capabilities", deadline)
    except ApiError as error:
        if error.status in (404, 405, 501):
            raise BatchError(
                "This deployment does not support resumable batch submission. "
                "Upgrade it first; no job was submitted."
            ) from None
        raise
    if (
        capabilities.get("idempotent_submission") != 1
        or extension not in capabilities.get("input_formats", [])
        or "parquet" not in capabilities.get("result_formats", [])
        or not isinstance(capabilities.get("submission_owner_id"), str)
        or not capabilities["submission_owner_id"]
    ):
        raise BatchError(
            "This deployment does not advertise the required batch capabilities; "
            "no job was submitted."
        )
    return capabilities["submission_owner_id"]


def _pause(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BatchError("Client time limit reached; remote work was not cancelled.")
    time.sleep(min(POLL_SECONDS, remaining))


def _submit_with_key(
    client, key, make_body, deadline, emit,
    *, on_lookup=None, before_submit=None, on_ingesting=None, submit_first=False,
):
    """Submit/recover one key without prescribing receipts, files or results."""
    attempts = 0
    while time.monotonic() < deadline:
        should_submit = submit_first
        if not should_submit:
            try:
                code, data = _json_request(
                    client, "GET", f"batch-submissions/{key}", deadline
                )
                state = data.get("status")
                if data.get("idempotency_key") != key:
                    raise BatchError("The deployment returned a different submission key.")
                if on_lookup is not None:
                    on_lookup()
                should_submit = (
                    state in {"INGESTING", "RETRYABLE"}
                    and data.get("retryable") is True
                )
                if state == "FAILED":
                    raise ApiError(422, retryable=False, code=data.get("code"))
            except ApiError as error:
                if error.status == 404 or (error.status == 503 and error.retryable):
                    should_submit = True
                elif error.status >= 500:
                    emit(
                        "Submission lookup temporarily unavailable; "
                        "checking again without creating a new key."
                    )
                    _pause(deadline)
                    continue
                else:
                    raise
            except NetworkError:
                emit("Submission lookup disconnected; retrying the read-only check.")
                _pause(deadline)
                continue
        if should_submit:
            if before_submit is not None:
                before_submit()
            if attempts >= 3:
                raise BatchError(
                    "Submission is still unconfirmed. Retry the same submission key; "
                    "do not start a new run."
                ) from None
            body = make_body()
            attempts += 1
            try:
                code, data = _json_request(
                    client, "POST", "batch-inference", deadline, body
                )
                # SDK callers supply the contract anew. Until a POST is
                # acknowledged, GET alone cannot prove it matched this key.
                submit_first = False
            except ApiError as post_error:
                if post_error.status < 500:
                    raise
                emit(
                    "Submission response uncertain; checking the saved submission key."
                )
                _pause(deadline)
                continue
            except BatchError:
                emit(
                    "Submission response uncertain; checking the saved submission key."
                )
                _pause(deadline)
                continue
        if code in (200, 202) and data.get("status") == "INGESTING":
            if data.get("idempotency_key") != key:
                raise BatchError(
                    "The deployment returned a different submission key; "
                    "inspect the saved submission key."
                )
            if on_ingesting is not None:
                on_ingesting()
            emit("INGESTING | retaining and validating the input dataset")
            _pause(deadline)
            continue
        job_id = data.get("results")
        if code != 200 or not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise BatchError(
                "Invalid submission response. Retry the same submission key to check it."
            )
        return job_id
    raise BatchError(
        "Submission wait timed out. Retry the same submission key; "
        "no cancellation was sent."
    )


def _submit_or_resume(client, receipt_path, receipt, deadline, emit, owner_id=None):
    if receipt.get("job_id"):
        return receipt["job_id"]
    key = receipt["idempotency_key"]

    def bind_owner():
        if owner_id is not None:
            receipt.update(version=3, identity={**receipt["identity"], "owner_id": owner_id})
            _save_receipt(receipt_path, receipt)

    def save_status(status):
        receipt["status"] = status
        _save_receipt(receipt_path, receipt)

    def before_submit():
        if not receipt["identity"].get("owner_id"):
            raise BatchError(
                "This older receipt has no owner identity and its submission is not visible. "
                "Use the original owner's API key or resume a known job ID. "
                "No new job was submitted."
            )
        save_status("SUBMITTING")

    def make_body():
        body = {
            "model": receipt["function"],
            "inputs": _signed_input(receipt["input_source"], receipt.get("s3_region")),
            "input_source": receipt["input_source"],
            "idempotency_key": key,
            "job_priority": 1,
            "truncate_rows": False,
            "name": "cli-" + key,
        }
        if receipt.get("id_column"):
            body["id_column_name"] = receipt["id_column"]
        return body

    job_id = _submit_with_key(
        client, key, make_body, deadline, emit,
        on_lookup=bind_owner, before_submit=before_submit,
        on_ingesting=lambda: save_status("INGESTING"),
    )
    receipt.update(job_id=job_id, status="ACCEPTED")
    _save_receipt(receipt_path, receipt)
    return job_id


def _progress(client, job_id, deadline, stop, emit):
    reconnects = 0
    while (
        reconnects < 2
        and not stop.is_set()
        and time.monotonic() < deadline
    ):
        try:
            with client.do_request(
                "GET",
                f"stream-job-progress/{job_id}",
                stream=True,
                max_retries=0,
                allow_redirects=False,
                headers={"User-Agent": USER_AGENT},
                timeout=(5, None),
            ) as response:
                response.raw.decode_content = True
                while not stop.is_set() and time.monotonic() < deadline:
                    line = response.raw.readline(1024 * 1024 + 1)
                    if not line or len(line) > 1024 * 1024:
                        break
                    line = line.strip()
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    if not line.startswith(b"{"):
                        continue
                    event = json.loads(line)
                    kind, result = event.get("update_type"), event.get("result")
                    values = (
                        {"rows_completed": result}
                        if kind == "progress"
                        else result
                        if kind == "tokens"
                        else {}
                    )
                    if not isinstance(values, dict):
                        continue
                    allowed = {
                        "rows_completed",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens_processed_per_second",
                    }
                    safe = {
                        k: v
                        for k, v in values.items()
                        if k in allowed
                        and type(v) in (int, float)
                        and math.isfinite(v)
                        and v >= 0
                    }
                    if safe:
                        emit("Progress | " + json.dumps(safe))
        except Exception:
            pass
        reconnects += 1
        if not stop.is_set() and time.monotonic() < deadline:
            stop.wait(min(2, max(0, deadline - time.monotonic())))
    if not stop.is_set():
        emit("Live progress unavailable; status polling continues.")


def wait_for_job(fetch_status, job_id, deadline, emit, *, on_status=None, poll_seconds=None):
    """Poll one job for both the existing SDK helper and durable batch runs."""
    started = time.monotonic()
    while time.monotonic() < deadline:
        try:
            status = fetch_status()
            if not isinstance(status, str):
                raise BatchError(
                    "Invalid job status response; no replacement job was submitted."
                )
            if on_status is not None:
                on_status(status)
            emit(f"{int(time.monotonic() - started)}s | {status} | {job_id}")
            if status == "SUCCEEDED":
                return status
            if status in ("FAILED", "CANCELLED", "EXPIRED", "PARTIAL", "PARTIALLY_SUCCEEDED") or status.startswith("FAILED_"):
                return status
        except (ApiError, requests.HTTPError) as error:
            code = error.status if isinstance(error, ApiError) else error.response.status_code
            if code < 500 and code != 429:
                raise
            emit("Status temporarily unavailable; continuing to watch the same job.")
        except (NetworkError, requests.RequestException):
            emit("Status connection interrupted; continuing to watch the same job.")
        if poll_seconds is None:
            _pause(deadline)
        else:
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    raise BatchError("Client time limit reached; the job may still be running.")


def _watch(client, receipt_path, receipt, deadline, emit):
    job_id = receipt["job_id"]
    stop = threading.Event()
    threading.Thread(
        target=_progress, args=(client, job_id, deadline, stop, emit), daemon=True
    ).start()

    def fetch_status():
        _, data = _json_request(client, "GET", f"job-status/{job_id}", deadline)
        return data.get("job_status", {}).get(job_id)

    def save_status(status):
        receipt["status"] = status
        _save_receipt(receipt_path, receipt)

    try:
        status = wait_for_job(fetch_status, job_id, deadline, emit, on_status=save_status)
        if status != "SUCCEEDED":
            raise BatchError(
                f"Job {status}. Inspect {job_id}; rerunning will not submit a replacement."
            )
    finally:
        stop.set()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_no_replace(source, destination):
    """Publish atomically without clobbering, with a fallback for missing hard links."""
    try:
        os.link(source, destination)
        return
    except FileExistsError:
        raise
    except OSError:
        pass
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        try:
            rename = libc.renameat2
        except AttributeError:
            raise BatchError("Atomic publication is unavailable; use overwrite=True.") from None
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), os.fspath(destination))


def _complete_download(receipt_path, receipt, digest, options):
    receipt.update(status="DOWNLOADED", output_sha256=digest, result_options=options)
    receipt.pop("pending_output_sha256", None)
    receipt.pop("pending_result_options", None)
    _save_receipt(receipt_path, receipt)


def _download(
    client, receipt_path, receipt, output, unpack_json, include_inputs, overwrite, emit
):
    work = output.parent / ("." + output.name + ".sutro-" + receipt["idempotency_key"])
    if work.is_symlink():
        raise BatchError("The download work directory must not be a symbolic link.")
    work.mkdir(mode=0o700, exist_ok=True)
    normalized = work / "normalized.parquet"
    # Distinct options get distinct raw artifacts; never reuse an inputs-free
    # artifact when the caller asks to include inputs on a resumed download.
    raw = work / ("raw-inputs.parquet" if include_inputs else "raw.parquet")
    # Refresh metadata on every attempt: the server can rebuild an artifact
    # after staged sources change. Only the SDK's ETag-validated .part transfer
    # may be reused, never an unvalidated complete local raw-result cache.
    emit("Downloading completed results (resumable Parquet transfer).")
    if client.download_job_results(
        receipt["job_id"], output_path=str(raw), include_inputs=include_inputs
    ) is None:
        raise BatchError(
            "Result download incomplete. Rerun the same command to resume; "
            "no new job is needed."
        )
    if normalized.exists():
        normalized.unlink()
    try:
        normalize_results(
            raw,
            normalized,
            unpack_json=unpack_json,
            include_inputs=include_inputs,
            source_output_column=receipt["job_id"],
        )
    except ResultShapeError as error:
        raise BatchError(
            str(error).replace("unpack_json=False", "unpack_json=False (--no-unpack-json in the CLI)")
        ) from None
    normalized.chmod(0o600)
    with normalized.open("rb") as stream:
        os.fsync(stream.fileno())
    digest = _file_hash(normalized)
    options = {"unpack_json": unpack_json, "include_inputs": include_inputs}
    receipt.update(
        status="PUBLISHING",
        pending_output_sha256=digest,
        pending_result_options=options,
    )
    _save_receipt(receipt_path, receipt)
    if overwrite:
        os.replace(normalized, output)
    else:
        try:
            _publish_no_replace(normalized, output)
        except FileExistsError:
            raise BatchError(
                "Output already exists. Use --overwrite to replace it "
                "with this job's results."
            ) from None
        normalized.unlink(missing_ok=True)
    _complete_download(receipt_path, receipt, digest, options)
    emit(f"Results saved to {output}")
    # Leave only partial downloads on interruption, not a duplicate full result.
    for name in ("raw.parquet", "raw-inputs.parquet"):
        path = work / name
        if path.exists():
            path.unlink()
    try:
        work.rmdir()
    except OSError:
        pass


def run_batch(
    client,
    *,
    output_file_name,
    function=None,
    dataset=None,
    id_column=None,
    unpack_json=True,
    include_inputs=False,
    new_run=False,
    overwrite=False,
    resume_job_id=None,
    timeout=86400,
    emit=print,
):
    """Submit/continue one exact S3 input, or perform a read-only job resume."""
    output = Path(output_file_name).expanduser().absolute()
    if output.suffix.lower() != ".parquet" or output.is_dir() or output.is_symlink():
        raise BatchError(
            "--output-file-name must be a local .parquet file, "
            "not a directory or symbolic link."
        )
    if resume_job_id and not JOB_ID.fullmatch(resume_job_id):
        raise BatchError("Resume requires a complete job-UUID.")
    if not resume_job_id:
        if (
            not isinstance(function, str)
            or not function.strip()
            or function != function.strip()
        ):
            raise BatchError(
                "--function must be a nonempty published function name "
                "without surrounding whitespace."
            )
        if id_column is not None and not id_column.strip():
            raise BatchError("--id-column must name a dataset column.")
        _, _, extension = _s3_location(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(str(output) + ".sutro.json")
    identity = {"api_url": client.api_url}
    deadline = time.monotonic() + timeout
    owner_id = None
    with _output_lock(output):
        if receipt_path.is_symlink():
            raise BatchError("The run receipt must not be a symbolic link.")
        receipt = None
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text())
                uuid.UUID(receipt["idempotency_key"])
                saved_identity = receipt.get("identity")
                same_deployment = (
                    isinstance(saved_identity, dict)
                    and saved_identity.get("api_url") == identity["api_url"]
                )
                if receipt.get("version") not in (1, 2, 3) or not same_deployment:
                    raise BatchError(
                        "This receipt belongs to a different deployment or "
                        "version; choose another output path."
                    )
                if receipt.get("version") == 1:
                    receipt.update(version=2, identity=identity)
                    _save_receipt(receipt_path, receipt)
            except (ValueError, KeyError, TypeError):
                raise BatchError(
                    "The run receipt is invalid; do not delete it or resubmit blindly."
                ) from None
        if receipt and new_run:
            archive = Path(str(receipt_path) + "." + receipt["idempotency_key"])
            if not archive.exists():
                os.link(receipt_path, archive)
            emit(
                "Starting an explicit new run; the previous receipt is archived "
                "and its job is not cancelled."
            )
            receipt = None
        if receipt:
            matches = (
                receipt.get("job_id") == resume_job_id
                if resume_job_id
                else (
                    receipt.get("function") == function
                    and receipt.get("dataset") == dataset
                    and receipt.get("id_column") == id_column
                )
            )
            if not matches:
                raise BatchError(
                    "The receipt describes a different run. Use another output "
                    "path or --new-run for an intentional new submission."
                )
            options = {"unpack_json": unpack_json, "include_inputs": include_inputs}
            pending = receipt.get("status") == "PUBLISHING"
            saved_options = receipt.get(
                "pending_result_options" if pending else "result_options"
            )
            saved_digest = receipt.get(
                "pending_output_sha256" if pending else "output_sha256"
            )
            if (
                output.exists()
                and receipt.get("status") in ("DOWNLOADED", "PUBLISHING")
                and saved_options == options
                and not overwrite
                and saved_digest == _file_hash(output)
            ):
                if pending:
                    _complete_download(receipt_path, receipt, saved_digest, saved_options)
                emit(f"Already downloaded | {receipt['job_id']} | {output}")
                return receipt["job_id"]
        if output.exists() and not overwrite:
            raise BatchError(
                "Output already exists; choose another path or use --overwrite. "
                "No job was submitted."
            )
        if not receipt:
            if not resume_job_id:
                owner_id = _check_capabilities(client, extension, deadline)
            receipt = {
                "version": 3,
                "identity": {**identity, **({"owner_id": owner_id} if owner_id else {})},
                "idempotency_key": str(uuid.uuid4()),
                "function": function,
                "dataset": dataset,
                "id_column": id_column,
                "job_id": resume_job_id,
                "status": "ACCEPTED" if resume_job_id else "PREPARED",
            }
            if not resume_job_id:
                receipt["input_source"], receipt["s3_region"] = _source_identity(
                    dataset
                )
            _save_receipt(receipt_path, receipt)
        elif not receipt.get("job_id"):
            owner_id = _check_capabilities(client, extension, deadline)
            saved_owner = receipt["identity"].get("owner_id")
            if saved_owner is not None and saved_owner != owner_id:
                raise BatchError(
                    "This receipt belongs to a different owner. Use the original owner's API key; "
                    "no new job was submitted."
                )
        emit(f"Receipt: {receipt_path}")
        try:
            job_id = _submit_or_resume(client, receipt_path, receipt, deadline, emit, owner_id)
            emit(f"Job: {job_id}")
            _watch(client, receipt_path, receipt, deadline, emit)
            _download(
                client,
                receipt_path,
                receipt,
                output,
                unpack_json,
                include_inputs,
                overwrite,
                emit,
            )
            return job_id
        except KeyboardInterrupt:
            emit(
                "Stopped watching. No cancellation was sent; "
                "rerun the same command to resume."
            )
            raise
        except Exception:
            emit(
                "Receipt preserved. Rerun the same command to resume, "
                "not --new-run. No cancellation was sent."
            )
            raise
