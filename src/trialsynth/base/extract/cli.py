"""Click CLI for the TrialSynth Bedrock extract pipeline."""
import logging
from pathlib import Path

import boto3
import click

from trialsynth.base.extract.build_batch_input_json_schema import (
    DEFAULT_PREFIX,
    MAX_RECORDS_PER_FILE,
    main as build_batch_input,
)
from trialsynth.base.extract.corpus import (
    CORPORA,
    DEFAULT_CORPUS,
    Corpus,
    get_corpus,
)
from trialsynth.base.extract.extract_bedrock import (
    _is_jsonl_object_uri,
    _is_s3_uri,
    _parse_s3_uri,
    main as extract_main,
)
from trialsynth.base.extract.process_bedrock import main as process_main


def _resolve_ids(
    ids: tuple[str, ...],
    id_file: Path | None,
    corpus: Corpus,
) -> list[str]:
    id_list: list[str] = []
    for value in ids:
        id_list.extend(part for part in value.replace(",", " ").split() if part)

    if id_list and id_file is not None:
        raise click.UsageError("--ids and --id-file are mutually exclusive")
    if id_file is not None:
        return [
            line.strip()
            for line in id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if id_list:
        return id_list
    return corpus.resolve_ids()


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
    "--corpus",
    "corpus_name",
    type=click.Choice(sorted(CORPORA)),
    default=DEFAULT_CORPUS,
    show_default=True,
    help="Record collection the IDs belong to.",
)
@click.option(
    "--ids",
    "--pmids",
    "ids",
    multiple=True,
    metavar="ID",
    help=(
        "Record ID(s) to include. Repeat the option or pass a comma-separated "
        "list. Mutually exclusive with --id-file."
    ),
)
@click.option(
    "--id-file",
    "--pmid-file",
    "id_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="File of record IDs, one per line. Mutually exclusive with --ids.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Use only the first N record IDs after the list is resolved.",
)
@click.option(
    "-o",
    "--output",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write *_input_{N}.jsonl files into.",
)
@click.option(
    "--prefix",
    default=DEFAULT_PREFIX,
    show_default=True,
    help="Filename prefix. Files are written as <prefix>_input_<end>.jsonl.",
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
    help="Worker threads for the corpus text download.",
)
def prepare(
    corpus_name: str,
    ids: tuple[str, ...],
    id_file: Path | None,
    limit: int | None,
    out_dir: Path,
    prefix: str,
    s3_uri: str | None,
    max_records: int,
    max_workers: int,
) -> None:
    """Download texts and write Bedrock batch input JSONL.

    If neither --ids nor --id-file is given, the IDs come from the corpus
    itself. For the pubmed corpus that is the PMIDs on the trial-publication
    edges, which requires the ctgov pipeline to have run.
    """
    corpus = get_corpus(corpus_name)

    try:
        resolved = _resolve_ids(ids, id_file, corpus)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if limit is not None:
        if limit < 0:
            raise click.UsageError("--limit must be >= 0")
        resolved = resolved[:limit]
    if not resolved:
        raise click.UsageError(f"No {corpus.name} record IDs to prepare")

    texts = corpus.load_texts(resolved, max_workers=max_workers)

    try:
        written = build_batch_input(
            resolved,
            out_dir=out_dir,
            corpus=corpus,
            texts=texts,
            prefix=prefix,
            max_records=max_records,
        )
    except ValueError as exc:
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
        s3 = boto3.client("s3")
        for path, bucket, key in _s3_upload_targets(written, s3_uri):
            s3.upload_file(str(path), bucket, key)
            click.echo(f"Uploaded {path} -> s3://{bucket}/{key}")


# Simply re-add the extract and process commands from their respective modules
# to this CLI group.
cli.add_command(extract_main, name="extract")
cli.add_command(process_main, name="process")
