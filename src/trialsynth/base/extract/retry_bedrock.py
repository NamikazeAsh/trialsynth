"""Build a retry batch for Bedrock records that failed with a given stop_reason."""

import json
import logging
from pathlib import Path

import click

from trialsynth.base.extract.build_batch_input_json_schema import (
    DEFAULT_PREFIX,
    MAX_RECORDS_PER_FILE,
    build_batch_input_jsonl,
)
from trialsynth.base.extract.corpus import CORPORA, DEFAULT_CORPUS, Corpus, get_corpus
from trialsynth.base.extract.process_bedrock import _iter_lines, _jsonl_paths, _source_text

logger = logging.getLogger(__name__)

RETRYABLE_STOP_REASONS = {"max_tokens", "content_filter"}
RETRY_TOKEN_LADDER = [32000, 64000]


def _next_tier(current_max_tokens: int) -> int | None:
    """Next higher rung on ``RETRY_TOKEN_LADDER``, or ``None`` at the ceiling."""
    for tier in RETRY_TOKEN_LADDER:
        if tier > current_max_tokens:
            return tier
    return None


def _read_failures(path: Path) -> dict[str, str]:
    """Read a ``<record_id>\\t<stop_reason>`` failures file."""
    failures = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record_id, _, stop_reason = line.partition("\t")
        failures[record_id] = stop_reason or "unknown"
    return failures


def _recover_records(
    jsonl_paths: list[Path], record_ids: set[str], corpus: Corpus
) -> dict[str, tuple[str, int]]:
    """Recover ``(source_text, current_max_tokens)`` for ``record_ids``."""
    recovered = {}
    remaining = set(record_ids)
    for _, _, line in _iter_lines(jsonl_paths):
        if not remaining:
            break
        rec = json.loads(line)
        record_id = str(rec.get("recordId") or "")
        if record_id not in remaining:
            continue
        try:
            text = _source_text(rec, corpus)
        except ValueError:
            continue
        max_tokens = (rec.get("modelInput") or {}).get("max_tokens")
        if not isinstance(max_tokens, int):
            continue
        recovered[record_id] = (text, max_tokens)
        remaining.discard(record_id)
    return recovered


def main(
    failures_path: Path,
    input_path: Path,
    out_dir: Path,
    corpus_name: str = DEFAULT_CORPUS,
    retry_max_tokens: int | None = None,
    prefix: str = DEFAULT_PREFIX,
    max_records: int = MAX_RECORDS_PER_FILE,
    manual_review_out: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """Build a retry batch input from a failures file."""
    corpus = get_corpus(corpus_name)
    failures = _read_failures(failures_path)
    if not failures:
        raise ValueError(f"No failures found in {failures_path}")

    retry_ids = {
        rid for rid, reason in failures.items() if reason in RETRYABLE_STOP_REASONS
    }
    manual_ids = set(failures) - retry_ids

    if not retry_ids:
        _write_manual_review(manual_review_out, manual_ids)
        logger.info("No retryable failures (stop_reason in %s)", RETRYABLE_STOP_REASONS)
        return [], sorted(manual_ids)

    jsonl_paths = _jsonl_paths(input_path)
    recovered = _recover_records(jsonl_paths, retry_ids, corpus)
    missing = retry_ids - set(recovered)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise ValueError(
            f"Could not recover source text for {len(missing)} record(s) "
            f"from {input_path}: {preview}"
        )

    # Group records by their target max_tokens, since an escalation ladder
    # means different records can be due for different tiers. content_filter
    # isn't token-related, so it's resubmitted unchanged rather than escalated.
    by_tier: dict[int, list[tuple[str, str]]] = {}
    for rid in sorted(retry_ids):
        text, current_max_tokens = recovered[rid]
        if retry_max_tokens is not None:
            tier = retry_max_tokens
        elif failures[rid] == "content_filter":
            tier = current_max_tokens
        else:
            tier = _next_tier(current_max_tokens)
        if tier is None:
            manual_ids.add(rid)
            continue
        by_tier.setdefault(tier, []).append((rid, text))

    _write_manual_review(manual_review_out, manual_ids)

    written = []
    for tier, records in sorted(by_tier.items()):
        written.extend(
            build_batch_input_jsonl(
                records,
                out_dir=out_dir,
                corpus=corpus,
                prefix=f"{prefix}_{tier}",
                max_records=max_records,
                max_tokens=tier,
            )
        )
    return written, sorted(manual_ids)


def _write_manual_review(path: Path | None, manual_ids: set[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(manual_ids)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


@click.command()
@click.argument(
    "failures_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--corpus", "corpus_name", type=click.Choice(sorted(CORPORA)), default=DEFAULT_CORPUS)
@click.option("--retry-max-tokens", type=int, default=None)
@click.option("--prefix", default=DEFAULT_PREFIX)
@click.option("--max-records", type=int, default=MAX_RECORDS_PER_FILE)
@click.option("--manual-review-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def cli(
    failures_path: Path,
    input_path: Path,
    out_dir: Path,
    corpus_name: str,
    retry_max_tokens: int,
    prefix: str,
    max_records: int,
    manual_review_out: Path | None,
) -> None:
    """Build a retry batch input from a failures file."""
    written, manual_ids = main(
        failures_path=failures_path,
        input_path=input_path,
        out_dir=out_dir,
        corpus_name=corpus_name,
        retry_max_tokens=retry_max_tokens,
        prefix=prefix,
        max_records=max_records,
        manual_review_out=manual_review_out,
    )
    for path in written:
        click.echo(f"Wrote: {path}")
    if manual_ids:
        click.echo(f"{len(manual_ids)} record(s) need manual review, not retried.")
