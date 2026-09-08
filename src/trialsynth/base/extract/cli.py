"""Click CLI for the TrialSynth Bedrock extract pipeline."""
import logging
from pathlib import Path

import boto3
import click

from trialsynth.base.extract.build_batch_input_json_schema import (
    MAX_RECORDS_PER_FILE,
    main as build_batch_input,
)
from trialsynth.base.extract.extract_bedrock import (
    DEFAULT_REGION,
    _is_jsonl_object_uri,
    _is_s3_uri,
    _parse_s3_uri,
    main as extract_main,
)
from trialsynth.base.extract.extract_util import (
    download_texts_bulk,
    get_trial_pmids,
)
from trialsynth.base.extract.process_bedrock import main as process_main


def _resolve_pmids(
    pmids: tuple[str, ...],
    pmid_file: Path | None,
) -> list[str]:
    pmid_list: list[str] = []
    for value in pmids:
        pmid_list.extend(part for part in value.replace(",", " ").split() if part)

    if pmid_list and pmid_file is not None:
        raise click.UsageError("--pmids and --pmid-file are mutually exclusive")
    if pmid_file is not None:
        return [
            line.strip()
            for line in pmid_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if pmid_list:
        return pmid_list
    return get_trial_pmids()


def _s3_upload_targets(
    local_paths: list[Path], s3_uri: str
) -> list[tuple[Path, str, str]]:
    if _is_jsonl_object_uri(s3_uri):
        if len(local_paths) != 1:
            raise click.UsageError(
                "Multiple JSONL input files cannot be uploaded to a single "
                ".jsonl object URI. Pass an S3 prefix "
                "(e.g. s3://bucket/prefix/) instead."
            )
        bucket, key = _parse_s3_uri(s3_uri)
        return [(local_paths[0], bucket, key)]

    bucket, key = _parse_s3_uri(s3_uri)
    prefix = key.rstrip("/")
    targets = []
    for path in local_paths:
        dest_key = f"{prefix}/{path.name}" if prefix else path.name
        targets.append((path, bucket, dest_key))
    return targets


@click.group()
def cli():
    """TrialSynth Bedrock extract pipeline.

    Subcommands follow the pipeline order: prepare, extract, process.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@cli.command("prepare")
@click.option(
    "--pmids",
    multiple=True,
    metavar="PMID",
    help=(
        "PMID(s) to include. Repeat the option or pass a comma-separated "
        "list. Mutually exclusive with --pmid-file."
    ),
)
@click.option(
    "--pmid-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="File of PMIDs, one per line. Mutually exclusive with --pmids.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Use only the first N PMIDs after the list is resolved.",
)
@click.option(
    "-o",
    "--output",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help=(
        "Output JSONL path for JSONL input files. Files are written as "
        "<stem>_input_<end>.jsonl (e.g. run.jsonl -> run_input_500.jsonl)."
    ),
)
@click.option(
    "--s3-uri",
    default=None,
    help=(
        "Optional S3 destination. A .jsonl object URI uploads a single "
        "file; a prefix uploads each file by filename. Omit to skip upload."
    ),
)
@click.option(
    "--max-records",
    type=int,
    default=MAX_RECORDS_PER_FILE,
    show_default=True,
    help="Maximum records per input JSONL file.",
)
@click.option(
    "--max-workers",
    type=int,
    default=8,
    show_default=True,
    help="Worker threads for download_texts_bulk.",
)
@click.option(
    "--region",
    default=DEFAULT_REGION,
    show_default=True,
    help="AWS region for optional S3 upload.",
)
def prepare(
    pmids: tuple[str, ...],
    pmid_file: Path | None,
    limit: int | None,
    out_path: Path,
    s3_uri: str | None,
    max_records: int,
    max_workers: int,
    region: str,
) -> None:
    """Download texts and write Bedrock batch input JSONL.

    If neither --pmids nor --pmid-file is given, PMIDs come from
    trial-publication edges (requires the ctgov pipeline).
    """
    try:
        resolved = _resolve_pmids(pmids, pmid_file)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if limit is not None:
        if limit < 0:
            raise click.UsageError("--limit must be >= 0")
        resolved = resolved[:limit]
    if not resolved:
        raise click.UsageError("No PMIDs to prepare")

    download_texts_bulk(resolved, max_workers=max_workers)

    try:
        written = build_batch_input(
            resolved, out_path=out_path, max_records=max_records
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if not written:
        raise click.ClickException("No JSONL files were written")

    click.echo("Wrote:")
    for path in written:
        click.echo(f"  {path}")

    if s3_uri:
        if not _is_s3_uri(s3_uri):
            raise click.UsageError(
                f"--s3-uri must be an S3 URI (s3://bucket/key), got {s3_uri!r}"
            )
        if not s3_uri.endswith('.jsonl') and not s3_uri.endswith('/'):
            raise click.UsageError(
                f"--s3-uri must be a .jsonl object URI or a prefix ending with '/', got {s3_uri!r}"
            )
        s3 = boto3.client("s3", region_name=region)
        for path, bucket, key in _s3_upload_targets(written, s3_uri):
            s3.upload_file(str(path), bucket, key)
            click.echo(f"Uploaded {path} -> s3://{bucket}/{key}")


cli.add_command(extract_main, name="extract")
cli.add_command(process_main, name="process")
