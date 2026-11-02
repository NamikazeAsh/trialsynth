"""Run extraction on Bedrock (batch or sync) or an OpenAI-compatible server."""
import os
import re
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import boto3
import click
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MAX_JOBS = 10
DEFAULT_TIMEOUT_HOURS = 24
SYNC_READ_TIMEOUT = 300
MIN_BATCH_RECORDS = 100
TERMINAL_JOB_STATUSES = ("Completed", "Failed", "Stopped", "PartiallyCompleted")
INPUT_FILE_RE = re.compile(r"_input_(\d+)\.jsonl$", re.IGNORECASE)
S3_URI_RE = re.compile(r"s3://[^/\s]+(?:/[^\s]*)?")
DUPLICATE_JOB_ERROR_CODES = (
    "ConflictException",
    "ResourceConflictException",
    "Conflict",
)
QUOTA_ERROR_CODES = (
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
    "LimitExceededException",
)


def _is_s3_uri(value: str) -> bool:
    return bool(S3_URI_RE.fullmatch(value))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not _is_s3_uri(uri):
        raise ValueError(f"Not a valid S3 URI: {uri}")
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _require_s3_uri(value: str, label: str) -> None:
    if not _is_s3_uri(value):
        raise click.UsageError(f"{label} must be an S3 URI (s3://bucket/key)")


def _is_jsonl_object_uri(uri: str) -> bool:
    _, key = _parse_s3_uri(uri)
    return key.lower().endswith(".jsonl")


def _count_jsonl_records(s3_uri: str) -> int:
    """Count non-empty lines in an S3 JSONL object."""
    bucket, key = _parse_s3_uri(s3_uri)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return sum(1 for line in body.splitlines() if line.strip())


def _batch_fallback_to_sync(
    s3_input_jsonl_path: str, s3_output_path: str, model_id: str
) -> None:
    """Run sync mode and upload the result where a batch job would have."""
    _, input_key = _parse_s3_uri(s3_input_jsonl_path)
    output_bucket, output_prefix = _parse_s3_uri(s3_output_path)
    output_key = f"{output_prefix.rstrip('/')}/{Path(input_key).stem}.jsonl.out"
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_output = Path(tmp_dir) / Path(output_key).name
        extract_trial_data_bedrock_sync(
            s3_input_jsonl_path=s3_input_jsonl_path,
            output_jsonl_path=str(local_output),
            model_id=model_id,
        )
        boto3.client("s3").upload_file(str(local_output), output_bucket, output_key)


def _client_error_code_message(exc: ClientError) -> tuple[str, str]:
    error = exc.response.get("Error", {})
    return error.get("Code", ""), error.get("Message", "")


def _is_duplicate_job_error(exc: ClientError) -> bool:
    code, msg = _client_error_code_message(exc)
    if code in DUPLICATE_JOB_ERROR_CODES:
        return True
    combined = f"{code} {msg}".lower()
    return (
        "already exist" in combined
        or "already in use" in combined
        or ("duplicate" in combined and "name" in combined)
        or ("job name" in combined and "unique" in combined)
    )


def _is_quota_error(exc: ClientError) -> bool:
    code, msg = _client_error_code_message(exc)
    if code in QUOTA_ERROR_CODES:
        return True
    combined = msg.lower()
    return (
        "quota" in combined
        or "too many" in combined
        or "limit exceeded" in combined
        or "concurrent" in combined
    )


def _wait_for_jobs(
    bedrock_client,
    job_arns: list[str],
    poll_interval: int,
) -> None:
    pending = list(job_arns)
    while pending:
        still_pending = []
        for job_arn in pending:
            status = bedrock_client.get_model_invocation_job(
                jobIdentifier=job_arn
            )["status"]
            click.echo(f"[{datetime.now().isoformat()}] {job_arn}: {status}")
            if status in TERMINAL_JOB_STATUSES:
                if status != "Completed":
                    click.echo(
                        f"Job {job_arn} ended with status {status}", err=True
                    )
            else:
                still_pending.append(job_arn)
        pending = still_pending
        if pending:
            time.sleep(poll_interval)


def _list_input_files(s3_prefix: str) -> list[tuple[int, str]]:
    bucket, prefix = _parse_s3_uri(s3_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    input_files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            match = INPUT_FILE_RE.search(name)
            if match is None:
                continue
            input_files.append((int(match.group(1)), f"s3://{bucket}/{key}"))

    input_files.sort(key=lambda item: item[0])
    return input_files


def output_uri_for_input(input_uri: str, output_prefix: str) -> str:
    """Derive the output prefix URI for an input JSONL S3 URI.

    ``s3://some_path/batch_run_20260827_input_4000.jsonl`` under output prefix
    ``s3://bucket/run/`` becomes
    ``s3://bucket/run/batch_run_20260827_output_4000/``.

    Parameters
    ----------
    input_uri :
        S3 URI of the input JSONL file.
    output_prefix :
        S3 URI prefix for outputs.

    Returns
    -------
    :
        Output S3 URI with a trailing slash.
    """
    _, key = _parse_s3_uri(input_uri)
    filename = key.rsplit("/", 1)[-1]
    if not filename.lower().endswith(".jsonl"):
        raise ValueError(f"Input URI is not a JSONL object: {input_uri}")
    stem = filename[: -len(".jsonl")]
    if "_input_" not in stem:
        raise ValueError(
            f"Input filename does not contain '_input_': {filename}"
        )
    out_name = stem.replace("_input_", "_output_", 1)
    return f"{output_prefix.rstrip('/')}/{out_name}/"


def extract_trial_data_bedrock_batch(
    job_name: str,
    s3_input_jsonl_path: str,
    s3_output_path: str,
    role_arn: str | None = None,
    model_id: str = DEFAULT_MODEL,
    poll_interval: int = 60,
    wait: bool = True,
    timeout_hours: int = DEFAULT_TIMEOUT_HOURS,
) -> str:
    """Submit a Bedrock batch inference job.

    Parameters
    ----------
    job_name :
        Name of the Bedrock model invocation job.
    s3_input_jsonl_path :
        S3 URI of the input JSONL file.
    s3_output_path :
        S3 URI of the output prefix. Must be an S3 URI.
    role_arn :
        IAM role ARN for the batch job. Defaults to the
        ``BEDROCK_JOB_ROLE_ARN`` environment variable.
    model_id :
        Bedrock model ID. Default is Claude Haiku 4.5.
    poll_interval :
        Seconds between job-status polls. Default is 60.
    wait :
        If True, poll until the job reaches a terminal status. Default is
        True.
    timeout_hours :
        Hours Bedrock lets the job run before it expires (24-168). Default
        is 24.

    Returns
    -------
    :
        The submitted job ARN.
    """
    if role_arn is None:
        role_arn = os.environ.get("BEDROCK_JOB_ROLE_ARN")
    if not role_arn:
        raise ValueError(
            "A role ARN is required for batch mode. Pass --role-arn or set "
            "the BEDROCK_JOB_ROLE_ARN environment variable."
        )
    if not _is_s3_uri(s3_input_jsonl_path):
        raise ValueError(
            f"s3_input_jsonl_path must be an S3 URI, got {s3_input_jsonl_path!r}"
        )
    if not _is_s3_uri(s3_output_path):
        raise ValueError(
            f"s3_output_path must be an S3 URI, got {s3_output_path!r}"
        )

    bedrock_client = boto3.client("bedrock")
    response = bedrock_client.create_model_invocation_job(
        jobName=job_name,
        roleArn=role_arn,
        modelId=model_id,
        inputDataConfig={
            "s3InputDataConfig": {
                "s3InputFormat": "JSONL",
                "s3Uri": s3_input_jsonl_path,
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": s3_output_path,
            }
        },
        timeoutDurationInHours=timeout_hours,
    )
    job_arn = response["jobArn"]
    click.echo(f"Submitted batch job {job_name}: {job_arn}")

    if wait:
        _wait_for_jobs(bedrock_client, [job_arn], poll_interval)

    return job_arn


def extract_trial_data_bedrock_batch_many(
    job_name_prefix: str,
    s3_input_prefix: str,
    s3_output_prefix: str,
    role_arn: str | None = None,
    model_id: str = DEFAULT_MODEL,
    poll_interval: int = 60,
    wait: bool = False,
    max_jobs: int = DEFAULT_MAX_JOBS,
    timeout_hours: int = DEFAULT_TIMEOUT_HOURS,
) -> list[str]:
    """Submit Bedrock batch jobs for ``*_input_{N}.jsonl`` files under a prefix.

    Job names are ``{job_name_prefix}-{N}``. Output URIs are
    ``{s3_output_prefix}/{stem with _input_ replaced by _output_}/``.
    Files whose job name already exists are skipped. At most ``max_jobs``
    new jobs are created; remaining files are printed as not submitted.

    Parameters
    ----------
    job_name_prefix :
        Prefix for Bedrock job names, e.g. ``trial-extract-20260827``.
    s3_input_prefix :
        S3 URI prefix containing ``*_input_{N}.jsonl`` input objects.
    s3_output_prefix :
        S3 URI prefix under which per-input output prefixes are created.
    role_arn :
        IAM role ARN for the batch jobs. Defaults to the
        ``BEDROCK_JOB_ROLE_ARN`` environment variable.
    model_id :
        Bedrock model ID. Default is Claude Haiku 4.5.
    poll_interval :
        Seconds between job-status polls. Default is 60.
    wait :
        If True, poll jobs created in this call until they finish. Default is
        False.
    max_jobs :
        Maximum number of new jobs to create. Default is 10.
    timeout_hours :
        Hours Bedrock lets each job run before it expires (24-168). Default
        is 24.

    Returns
    -------
    :
        ARNs of jobs created in this call.
    """
    # See https://aws.amazon.com/blogs/machine-learning/automate-amazon-bedrock-batch-inference-building-a-scalable-and-efficient-pipeline/
    # and https://docs.aws.amazon.com/bedrock/latest/userguide/capacity-limits-cost-optimization.html#limits-quotas
    # for quota and limits:
    # - Job size: Up to 10,000 records per batch
    # - File size: Maximum 200 MB input file
    # - Processing time: 24-hour completion window
    # - Concurrent jobs: Region-specific quotas (typically 10)
    if role_arn is None:
        role_arn = os.environ.get("BEDROCK_JOB_ROLE_ARN")
    if not role_arn:
        raise ValueError(
            "A role ARN is required for batch mode. Pass --role-arn or set "
            "the BEDROCK_JOB_ROLE_ARN environment variable."
        )
    if not _is_s3_uri(s3_input_prefix):
        raise ValueError(
            f"s3_input_prefix must be an S3 URI, got {s3_input_prefix!r}"
        )
    if not _is_s3_uri(s3_output_prefix):
        raise ValueError(
            f"s3_output_prefix must be an S3 URI, got {s3_output_prefix!r}"
        )

    input_files = _list_input_files(s3_input_prefix)
    if not input_files:
        raise ValueError(
            f"No *_input_{{N}}.jsonl objects found under {s3_input_prefix}"
        )

    submitted: list[tuple[int, str, str, str]] = []
    skipped: list[tuple[int, str, str]] = []
    not_submitted: list[tuple[int, str, str]] = []

    for index, (file_n, input_uri) in enumerate(input_files):
        if len(submitted) >= max_jobs:
            not_submitted.extend(
                (n, uri, f"{job_name_prefix}-{n}")
                for n, uri in input_files[index:]
            )
            break

        job_name = f"{job_name_prefix}-{file_n}"
        output_uri = output_uri_for_input(input_uri, s3_output_prefix)
        try:
            job_arn = extract_trial_data_bedrock_batch(
                job_name=job_name,
                s3_input_jsonl_path=input_uri,
                s3_output_path=output_uri,
                role_arn=role_arn,
                model_id=model_id,
                poll_interval=poll_interval,
                wait=False,
                timeout_hours=timeout_hours,
            )
        except ClientError as exc:
            if _is_duplicate_job_error(exc):
                skipped.append((file_n, input_uri, job_name))
                click.echo(f"Skipping {job_name}: already submitted")
                continue
            if _is_quota_error(exc):
                click.echo(
                    f"Quota or throttling while submitting {job_name}: {exc}",
                    err=True,
                )
                not_submitted.append((file_n, input_uri, job_name))
                not_submitted.extend(
                    (n, uri, f"{job_name_prefix}-{n}")
                    for n, uri in input_files[index + 1 :]
                )
                break
            raise
        submitted.append((file_n, input_uri, job_name, job_arn))

    def _print_group(title: str, rows) -> None:
        click.echo(f"{title} ({len(rows)}):")
        if not rows:
            click.echo("  (none)")
            return
        for row in rows:
            input_uri, job_name = row[1], row[2]
            extra = f"  {row[3]}" if len(row) > 3 else ""
            click.echo(f"  {job_name}  {input_uri}{extra}")

    click.echo("")
    _print_group("Submitted", submitted)
    _print_group("Skipped (already submitted)", skipped)
    _print_group("Not submitted", not_submitted)

    job_arns = [row[3] for row in submitted]
    if wait and job_arns:
        bedrock_client = boto3.client("bedrock")
        _wait_for_jobs(bedrock_client, job_arns, poll_interval)
    return job_arns


def extract_trial_data_bedrock_sync(
    s3_input_jsonl_path: str,
    output_jsonl_path: str,
    model_id: str = DEFAULT_MODEL,
) -> list[dict]:
    """Invoke Bedrock synchronously for each record in an S3 JSONL file.

    Parameters
    ----------
    s3_input_jsonl_path :
        S3 URI of the input JSONL file.
    output_jsonl_path :
        Local path to write output JSONL records.
    model_id :
        Bedrock model ID. Default is Claude Haiku 4.5.

    Returns
    -------
    :
        List of ``{"recordId", "modelInput", "modelOutput"}`` result dicts,
        matching the record shape Bedrock batch writes.
    """
    if not output_jsonl_path:
        raise ValueError("output_jsonl_path is required")
    if not _is_s3_uri(s3_input_jsonl_path):
        raise ValueError(
            f"s3_input_jsonl_path must be an S3 URI, got {s3_input_jsonl_path!r}"
        )

    s3 = boto3.client("s3")
    bedrock_runtime = boto3.client(
        "bedrock-runtime", config=Config(read_timeout=SYNC_READ_TIMEOUT)
    )

    bucket, s3_key = _parse_s3_uri(s3_input_jsonl_path)
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    lines = obj["Body"].read().decode("utf-8").splitlines()

    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(lines, desc="Invoking Bedrock"):
            record = json.loads(line)
            record_id = record["recordId"]
            model_input = record["modelInput"]

            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(model_input),
                contentType="application/json",
                accept="application/json",
            )
            model_output = json.loads(response["body"].read())

            result = {
                "recordId": record_id,
                "modelInput": model_input,
                "modelOutput": model_output,
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    return results


def _read_jsonl_lines(path: str) -> list[str]:
    """Return the non-empty lines of a JSONL file given as an S3 URI or local path."""
    if _is_s3_uri(path):
        bucket, key = _parse_s3_uri(path)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
        text = body.read().decode("utf-8")
    else:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def to_openai_request(model_input: dict, model: str) -> dict:
    """Translate a Bedrock Anthropic ``modelInput`` into an OpenAI chat request.

    The Anthropic ``output_config`` JSON schema becomes an OpenAI
    ``response_format``, which vLLM enforces with guided decoding, so the
    reply parses exactly as a Bedrock structured-output reply does.
    """
    messages = []
    if model_input.get("system"):
        messages.append({"role": "system", "content": model_input["system"]})
    messages.extend(model_input["messages"])

    request = {
        "model": model,
        "messages": messages,
        "max_tokens": model_input["max_tokens"],
    }
    for key in ("temperature", "top_p"):
        if key in model_input:
            request[key] = model_input[key]
    fmt = (model_input.get("output_config") or {}).get("format")
    if fmt and fmt.get("type") == "json_schema":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "schema": fmt["schema"],
                "strict": True,
            },
        }
    return request


def to_bedrock_output(response: dict) -> dict:
    """Wrap an OpenAI chat completion in the Anthropic shape Bedrock returns.

    ``process`` reads ``content[0].text``; matching the shape keeps it
    backend-agnostic.
    """
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    return {
        "content": [{"type": "text", "text": choice["message"]["content"]}],
        "stop_reason": (
            "max_tokens" if choice.get("finish_reason") == "length" else "end_turn"
        ),
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
    }


def extract_trial_data_openai(
    input_jsonl_path: str,
    output_jsonl_path: str,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    model: str | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Run Bedrock input records against an OpenAI-compatible server (e.g. vLLM).

    Parameters
    ----------
    input_jsonl_path :
        S3 URI or local path of a ``*_input_{N}.jsonl`` file from ``prepare``.
    output_jsonl_path :
        Local path to write output JSONL records.
    base_url :
        Server base URL, including the ``/v1`` suffix.
    model :
        Served model name. Defaults to the first model the server lists.
    max_workers :
        Concurrent requests. vLLM batches them on the GPU.

    Returns
    -------
    :
        List of ``{"recordId", "modelInput", "modelOutput"}`` result dicts,
        matching the record shape Bedrock batch writes.
    """
    base_url = base_url.rstrip("/")
    session = requests.Session()
    # Cluster http_proxy settings would otherwise intercept local calls.
    session.trust_env = False
    if model is None:
        resp = session.get(f"{base_url}/models", timeout=30)
        resp.raise_for_status()
        model = resp.json()["data"][0]["id"]
    click.echo(f"Using model {model} at {base_url}")

    records = [json.loads(line) for line in _read_jsonl_lines(input_jsonl_path)]

    def invoke(record: dict) -> dict:
        resp = session.post(
            f"{base_url}/chat/completions",
            json=to_openai_request(record["modelInput"], model),
            timeout=1800,
        )
        if not resp.ok:
            raise click.ClickException(
                f"{record['recordId']}: HTTP {resp.status_code} {resp.text}"
            )
        return {
            "recordId": record["recordId"],
            "modelInput": record["modelInput"],
            "modelOutput": to_bedrock_output(resp.json()),
        }

    output_path = Path(output_jsonl_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(output_path, "w", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in tqdm(
            pool.map(invoke, records), total=len(records), desc="Invoking model"
        ):
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
    return results


@click.command()
@click.argument(
    "mode", type=click.Choice(["batch", "sync", "openai"], case_sensitive=False)
)
@click.argument("s3_input_jsonl_path")
@click.argument("output_jsonl_path")
@click.option(
    "--job-name",
    help=(
        "Bedrock batch job name. Required when MODE is batch. Full name for "
        "a single JSONL input; name prefix (e.g. trial-extract-20260827) "
        "when the input is an S3 prefix containing input JSONL files."
    ),
)
@click.option(
    "--role-arn",
    envvar="BEDROCK_JOB_ROLE_ARN",
    help=(
        "IAM role ARN for the batch job. Defaults to the "
        "BEDROCK_JOB_ROLE_ARN environment variable."
    ),
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help=(
        f"Model ID. Bedrock modes default to {DEFAULT_MODEL}; openai mode "
        "defaults to the first model the server lists."
    ),
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    default=DEFAULT_OPENAI_BASE_URL,
    show_default=True,
    help="openai mode: server base URL including /v1. Reads OPENAI_BASE_URL.",
)
@click.option(
    "--max-workers",
    type=int,
    default=8,
    show_default=True,
    help="openai mode: concurrent requests to the server.",
)
@click.option(
    "--poll-interval",
    type=int,
    default=60,
    show_default=True,
    help="Seconds between batch job status polls.",
)
@click.option(
    "--wait/--no-wait",
    is_flag=True,
    default=None,
    help=(
        "Poll until submitted job(s) finish. Default is wait for a single "
        "JSONL input and no-wait for multiple inputs."
    ),
)
@click.option(
    "--max-jobs",
    type=int,
    default=DEFAULT_MAX_JOBS,
    show_default=True,
    help=(
        "Maximum number of new batch jobs to create for multi batch input. "
        "Already-submitted names do not count toward the limit."
    ),
)
@click.option(
    "--timeout-hours",
    type=click.IntRange(24, 168),
    default=DEFAULT_TIMEOUT_HOURS,
    show_default=True,
    help="batch mode: hours before Bedrock expires an unfinished job.",
)
def main(
    mode: str,
    s3_input_jsonl_path: str,
    output_jsonl_path: str,
    job_name: str | None,
    role_arn: str | None,
    model_id: str | None,
    base_url: str,
    max_workers: int,
    poll_interval: int,
    wait: bool | None,
    max_jobs: int,
    timeout_hours: int,
) -> None:
    """Extract trial data with Amazon Bedrock or an OpenAI-compatible server.

    MODE is ``batch``, ``sync`` or ``openai``. For ``batch`` there are two modes:
    1. A single JSONL input file (``*_input_{N}.jsonl``) is processed in one
    Bedrock batch job.
    2. A prefix containing multiple ``*_input_{N}.jsonl`` files is processed in
    multiple Bedrock batch jobs submitted simultaneously (up to ``--max-jobs``).

    S3_INPUT_JSONL_PATH must be an S3 URI. For batch it may be a single JSONL
    object or a prefix of ``*_input_{N}.jsonl`` files. OUTPUT_JSONL_PATH
    must be an S3 URI when MODE is batch (output S3 prefix for multiple jobs, or
    the output URI for a single file); for sync it is a local file path.

    ``openai`` sends each record to an OpenAI-compatible server such as vLLM
    (``--base-url``) and writes Bedrock-shaped output, so ``process`` reads it
    unchanged. Its input may be an S3 URI or a local path; output is local.
    """
    mode = mode.lower()
    if mode == "openai":
        extract_trial_data_openai(
            input_jsonl_path=s3_input_jsonl_path,
            output_jsonl_path=output_jsonl_path,
            base_url=base_url,
            model=model_id,
            max_workers=max_workers,
        )
        return

    _require_s3_uri(s3_input_jsonl_path, "input JSONL path")
    model_id = model_id or DEFAULT_MODEL

    if mode == "batch":
        _require_s3_uri(output_jsonl_path, "output JSONL path")
        if not job_name:
            raise click.UsageError("--job-name is required when MODE is batch")
        if not role_arn:
            raise click.UsageError(
                "A role ARN is required when MODE is batch. Pass --role-arn "
                "or set the BEDROCK_JOB_ROLE_ARN environment variable."
            )
        if _is_jsonl_object_uri(s3_input_jsonl_path):
            if _count_jsonl_records(s3_input_jsonl_path) < MIN_BATCH_RECORDS:
                click.echo(
                    f"Input has fewer than {MIN_BATCH_RECORDS} records, "
                    "Bedrock batch jobs require at least that many. "
                    "Using sync mode instead."
                )
                _batch_fallback_to_sync(s3_input_jsonl_path, output_jsonl_path, model_id)
            else:
                extract_trial_data_bedrock_batch(
                    job_name=job_name,
                    s3_input_jsonl_path=s3_input_jsonl_path,
                    s3_output_path=output_jsonl_path,
                    role_arn=role_arn,
                    model_id=model_id,
                    poll_interval=poll_interval,
                    wait=True if wait is None else wait,
                    timeout_hours=timeout_hours,
                )
        else:
            extract_trial_data_bedrock_batch_many(
                job_name_prefix=job_name,
                s3_input_prefix=s3_input_jsonl_path,
                s3_output_prefix=output_jsonl_path,
                role_arn=role_arn,
                model_id=model_id,
                poll_interval=poll_interval,
                wait=False if wait is None else wait,
                max_jobs=max_jobs,
                timeout_hours=timeout_hours,
            )
    else:
        if not _is_jsonl_object_uri(s3_input_jsonl_path):
            raise click.UsageError(
                "sync mode requires an S3 URI of a JSONL object, not a prefix"
            )
        extract_trial_data_bedrock_sync(
            s3_input_jsonl_path=s3_input_jsonl_path,
            output_jsonl_path=output_jsonl_path,
            model_id=model_id,
        )


if __name__ == "__main__":
    main()
