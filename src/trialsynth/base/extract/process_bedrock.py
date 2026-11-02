"""Turn Bedrock JSONL extraction output into grounded per-record JSON.

Reads Bedrock batch ``*.jsonl.out`` records, resolves ``evidence_anchor``
fields to full sentences, grounds the result in memory, and writes grounded
files to the corpus results directory.
"""

import json
import logging
from pathlib import Path

import click
from tqdm import tqdm

from trialsynth.base.extract.corpus import (
    CORPORA,
    DEFAULT_CORPUS,
    Corpus,
    get_corpus,
)
from trialsynth.base.extract.extract_util import resolve_anchors, split_sentences
from trialsynth.base.extract.ground_results import (
    AE_SHORT_TOKEN_MIN_LEN_DEFAULT,
    ground_extraction,
)

logger = logging.getLogger(__name__)


def _jsonl_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl.out"))
    raise FileNotFoundError(path)


class RecordFailure(Exception):
    """A Bedrock record failed, classified by its stop_reason."""

    def __init__(self, record_id: str, stop_reason: str | None, message: str):
        super().__init__(message)
        self.record_id = record_id
        self.stop_reason = stop_reason


def _source_text(rec: dict, corpus: Corpus) -> str:
    """Recover source text from a record's echoed ``modelInput``."""
    messages = (rec.get("modelInput") or {}).get("messages") or []
    if not messages or not isinstance(messages[0], dict):
        raise ValueError("missing modelInput.messages")
    user_content = messages[0].get("content")
    if not isinstance(user_content, str) or not user_content:
        raise ValueError("modelInput.messages[0].content is empty")
    source_text = corpus.unframe(user_content)
    if not source_text:
        raise ValueError("empty source text after un-framing the model input")
    return source_text


def _parse_record(line: str, corpus: Corpus) -> tuple[str, dict, str]:
    """Parse one Bedrock JSONL record.

    Parameters
    ----------
    line :
        A single JSONL line.
    corpus :
        Corpus whose framing was used to build the input, used to recover the
        source text from the echoed model input.

    Returns
    -------
    :
        ``(record_id, extraction, source_text)``.
    """
    rec = json.loads(line)
    record_id = rec.get("recordId")
    if not record_id:
        raise ValueError("missing recordId")
    record_id = str(record_id)

    model_output = rec.get("modelOutput")
    if not model_output:
        raise ValueError("missing modelOutput")

    stop_reason = model_output.get("stop_reason")
    if stop_reason not in (None, "end_turn"):
        raise RecordFailure(
            record_id, stop_reason, f"stop_reason={stop_reason}"
        )

    content = model_output.get("content") or []
    if not content or not isinstance(content[0], dict):
        raise ValueError("modelOutput.content is empty or malformed")
    text = content[0].get("text")
    if not text:
        raise ValueError("modelOutput.content[0] has no text")
    extraction = json.loads(text)
    if not isinstance(extraction, dict):
        raise ValueError("extraction JSON is not an object")

    return record_id, extraction, _source_text(rec, corpus)


def _process_record(
    record_id: str,
    extraction: dict,
    source_text: str,
    output_path: Path,
    ae_min_len: int,
    corpus: Corpus,
) -> None:
    """Resolve anchors and ground one extraction to ``output_path``.

    Parameters
    ----------
    record_id :
        Record ID from the Bedrock recordId; set on the extraction under the
        corpus's ``id_field`` if missing.
    extraction :
        Parsed LLM JSON (mutated in place by ``resolve_anchors``).
    source_text :
        Article text the model saw, used to resolve anchors.
    output_path :
        Destination grounded JSON path.
    ae_min_len :
        Passed through to ``ground_extraction``.
    corpus :
        Corpus the record came from, supplying the ID field to stamp.
    """
    resolved = resolve_anchors(extraction, split_sentences(source_text))
    if not resolved.get(corpus.id_field):
        resolved[corpus.id_field] = record_id
    grounded = ground_extraction(resolved, ae_min_len=ae_min_len)
    output_path.write_text(json.dumps(grounded, indent=2), encoding="utf-8")


def _count_records(jsonl_paths: list[Path]) -> int:
    n = 0
    for jsonl_path in jsonl_paths:
        with jsonl_path.open(encoding="utf-8") as fh:
            n += sum(1 for line in fh if line.strip())
    return n


def _iter_lines(jsonl_paths: list[Path]):
    """Yield non-empty JSONL lines from ``jsonl_paths``.

    Parameters
    ----------
    jsonl_paths :
        JSONL files to read in order.

    Yields
    ------
    :
        ``(path, line_no, line)`` triples.
    """
    for jsonl_path in jsonl_paths:
        with jsonl_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if line.strip():
                    yield jsonl_path, line_no, line


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--corpus",
    "corpus_name",
    type=click.Choice(sorted(CORPORA)),
    default=DEFAULT_CORPUS,
    show_default=True,
    help="Corpus the Bedrock input was built from.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory for grounded <record_id>.json files. Defaults to the "
        "corpus results directory."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Rewrite grounded files that already exist.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after this many newly grounded records (skips do not count).",
)
@click.option(
    "--ae-min-len",
    type=int,
    default=AE_SHORT_TOKEN_MIN_LEN_DEFAULT,
    show_default=True,
    help="Minimum adverse-event name length for grounding.",
)
@click.option(
    "--failures-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Write a <record_id>\\t<stop_reason> line for each classified "
        "failure here, for 'retry' to pick up."
    ),
)
def main(
    input_path: Path,
    corpus_name: str,
    output_dir: Path | None,
    overwrite: bool,
    limit: int | None,
    ae_min_len: int,
    failures_out: Path | None,
) -> None:
    """Ground Bedrock JSONL extraction output.

    INPUT_PATH is a JSONL file or a directory of ``*.jsonl.out`` files.
    """
    corpus = get_corpus(corpus_name)
    jsonl_paths = _jsonl_paths(input_path)
    if not jsonl_paths:
        raise click.ClickException(f"No JSONL files found under {input_path}")

    if output_dir is None:
        output_dir = corpus.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    classified_failures: list[tuple[str, str]] = []

    for jsonl_path, line_no, line in tqdm(
        _iter_lines(jsonl_paths),
        desc="Records",
        unit="rec",
        total=_count_records(jsonl_paths),
    ):
        if limit is not None and n_ok >= limit:
            break
        loc = f"{jsonl_path.name}:{line_no}"
        try:
            record_id, extraction, source_text = _parse_record(line, corpus)
        except RecordFailure as exc:
            n_fail += 1
            classified_failures.append(
                (exc.record_id, exc.stop_reason or "unknown")
            )
            logger.warning("Failed to parse %s: %s", loc, exc)
            continue
        except Exception as exc:
            n_fail += 1
            logger.warning("Failed to parse %s: %s", loc, exc)
            continue

        out_path = output_dir / f"{record_id}.json"
        if out_path.exists() and not overwrite:
            n_skip += 1
            continue

        try:
            _process_record(
                record_id,
                extraction,
                source_text,
                out_path,
                ae_min_len,
                corpus,
            )
        except Exception as exc:
            n_fail += 1
            logger.warning(
                "Failed to process %s record %s (%s): %s",
                corpus.name,
                record_id,
                loc,
                exc,
            )
            continue
        n_ok += 1

    if failures_out is not None and classified_failures:
        failures_out.parent.mkdir(parents=True, exist_ok=True)
        with failures_out.open("w", encoding="utf-8") as fh:
            for record_id, stop_reason in classified_failures:
                fh.write(f"{record_id}\t{stop_reason}\n")

    click.echo(
        f"Processed: {n_ok}  skipped: {n_skip}  failed: {n_fail}\n"
        f"Wrote to: {output_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
