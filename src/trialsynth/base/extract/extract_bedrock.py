import json
import os
import sys
import time

import boto3
from tqdm import tqdm


BEDROCK_JOB_ROLE_ARN = os.environ["BEDROCK_JOB_ROLE_ARN"]
DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def extract_trial_data_bedrock_batch(
    job_name: str, s3_input_jsonl_path: str, s3_output_path: str
):
    bedrock_client = boto3.client("bedrock", region_name='us-east-1')
    response = bedrock_client.create_model_invocation_job(
        jobName=job_name,
        roleArn=BEDROCK_JOB_ROLE_ARN,
        modelId=DEFAULT_MODEL,
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
    print(f"Submitted batch job: {job_arn}")

    # Poll the job status until it's done
    while True:
        status = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)[
            "status"
        ]
        print(f"  status: {status}")
        if status in ("Completed", "Failed", "Stopped", "PartiallyCompleted"):
            break
        time.sleep(60)

    return job_arn


def extract_trial_data_bedrock_sync(
    s3_input_jsonl_path: str, output_jsonl_path: str
) -> list[dict]:
    if not output_jsonl_path:
        raise ValueError("output_jsonl_path is required")

    s3 = boto3.client("s3", region_name="us-east-1")
    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Pull down the same JSONL file used as batch input
    bucket, s3_key = s3_input_jsonl_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    lines = obj["Body"].read().decode("utf-8").splitlines()

    results = []
    with open(output_jsonl_path, "w") as out_f:
        for line in tqdm(lines, desc="Invoking Bedrock"):
            record = json.loads(line)
            record_id = record["recordId"]
            model_input = record["modelInput"]

            response = bedrock_runtime.invoke_model(
                modelId=DEFAULT_MODEL,
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <s3_input_jsonl_path> <output_jsonl_path>")
        sys.exit(1)
    _ = extract_trial_data_bedrock_sync(sys.argv[1], sys.argv[2])

