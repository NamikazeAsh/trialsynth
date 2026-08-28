"""Run Bedrock extraction in batch or synchronous mode."""
import os
import json
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import boto3
from tqdm import tqdm
import click

DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "us-east-1"
TERMINAL_JOB_STATUSES = ("Completed", "Failed", "Stopped", "PartiallyCompleted")


def _is_s3_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "s3" and bool(parsed.netloc)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not _is_s3_uri(uri):
        raise ValueError(f"Not a valid S3 URI: {uri}")
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _require_s3_uri(value: str, label: str) -> None:
    if not _is_s3_uri(value):
        raise click.UsageError(f"{label} must be an S3 URI (s3://bucket/key)")


def extract_trial_data_bedrock_batch(
    job_name: str,
    s3_input_jsonl_path: str,
    s3_output_path: str,
    role_arn: str | None = None,
    model_id: str = DEFAULT_MODEL,
    region: str = DEFAULT_REGION,
    poll_interval: int = 60,
) -> str:
    """Submit a Bedrock batch inference job and wait until it finishes.

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
    region :
        AWS region. Default is ``us-east-1``.
    poll_interval :
        Seconds between job-status polls. Default is 60.

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

    bedrock_client = boto3.client("bedrock", region_name=region)
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
    )
    job_arn = response["jobArn"]
    click.echo(f"Submitted batch job: {job_arn}")

    while True:
        status = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)[
            "status"
        ]
        click.echo(f"[{datetime.now().isoformat()}] status: {status}")
        if status in TERMINAL_JOB_STATUSES:
            if status != "Completed":
                click.echo(f"Job {job_arn} ended with status {status}", err=True)
            break
        time.sleep(poll_interval)

    return job_arn


def extract_trial_data_bedrock_sync(
    s3_input_jsonl_path: str,
    output_jsonl_path: str,
    model_id: str = DEFAULT_MODEL,
    region: str = DEFAULT_REGION,
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
    region :
        AWS region. Default is ``us-east-1``.

    Returns
    -------
    :
        List of ``{"recordId", "modelOutput"}`` result dicts.
    """
    if not output_jsonl_path:
        raise ValueError("output_jsonl_path is required")
    if not _is_s3_uri(s3_input_jsonl_path):
        raise ValueError(
            f"s3_input_jsonl_path must be an S3 URI, got {s3_input_jsonl_path!r}"
        )

    s3 = boto3.client("s3", region_name=region)
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

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

            result = {"recordId": record_id, "modelOutput": model_output}
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    return results


@click.command()
@click.argument("mode", type=click.Choice(["batch", "sync"], case_sensitive=False))
@click.argument("s3_input_jsonl_path")
@click.argument("output_jsonl_path")
@click.option(
    "--job-name",
    help="Bedrock batch job name. Required when MODE is batch.",
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
    default=DEFAULT_MODEL,
    show_default=True,
    help="Bedrock model ID.",
)
@click.option(
    "--region",
    default=DEFAULT_REGION,
    show_default=True,
    help="AWS region.",
)
@click.option(
    "--poll-interval",
    type=int,
    default=60,
    show_default=True,
    help="Seconds between batch job status polls.",
)
def main(
    mode: str,
    s3_input_jsonl_path: str,
    output_jsonl_path: str,
    job_name: str | None,
    role_arn: str | None,
    model_id: str,
    region: str,
    poll_interval: int,
) -> None:
    """Extract trial data with Amazon Bedrock.

    MODE is either ``batch`` (Bedrock model invocation job) or ``sync``
    (one ``invoke_model`` call per input record).

    S3_INPUT_JSONL_PATH must be an S3 URI. OUTPUT_JSONL_PATH must also be
    an S3 URI when MODE is batch; for sync it is a local file path.
    """
    _require_s3_uri(s3_input_jsonl_path, "input JSONL path")
    mode = mode.lower()

    if mode == "batch":
        _require_s3_uri(output_jsonl_path, "output JSONL path")
        if not job_name:
            raise click.UsageError("--job-name is required when MODE is batch")
        if not role_arn:
            raise click.UsageError(
                "A role ARN is required when MODE is batch. Pass --role-arn "
                "or set the BEDROCK_JOB_ROLE_ARN environment variable."
            )
        extract_trial_data_bedrock_batch(
            job_name=job_name,
            s3_input_jsonl_path=s3_input_jsonl_path,
            s3_output_path=output_jsonl_path,
            role_arn=role_arn,
            model_id=model_id,
            region=region,
            poll_interval=poll_interval,
        )
    else:
        extract_trial_data_bedrock_sync(
            s3_input_jsonl_path=s3_input_jsonl_path,
            output_jsonl_path=output_jsonl_path,
            model_id=model_id,
            region=region,
        )


if __name__ == "__main__":
    main()
