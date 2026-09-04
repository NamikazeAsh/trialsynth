"""Turn Bedrock JSONL extraction output into grounded per-PMID JSON.

Reads Bedrock batch ``*.jsonl.out`` records, resolves ``evidence_anchor``
fields to full sentences, then grounds the result. Grounded files are written
to ``RESULTS_GROUNDED_DIR``; resolved JSON is kept only in a temporary
directory for ``ground_json``.
"""

import json
import logging
import re
import tempfile
from pathlib import Path

import click
from tqdm import tqdm

from trialsynth.base.extract.extract_util import resolve_anchors, split_sentences
from trialsynth.base.extract.ground_results import (
    AE_SHORT_TOKEN_MIN_LEN_DEFAULT,
    ground_json,
)
from trialsynth.base.extract.paths import RESULTS_GROUNDED_DIR

logger = logging.getLogger(__name__)

_TEXT_PREFIX = re.compile(r"^Text \(PMID \d+\):\s*")


def _jsonl_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl.out"))
    raise FileNotFoundError(path)


def _parse_record(line: str) -> tuple[str, dict, str]:
    """Parse one Bedrock JSONL record.

    Parameters
    ----------
    line :
        A single JSONL line.

    Returns
    -------
    :
        ``(pmid, extraction, source_text)``.
    """
    rec = json.loads(line)
    pmid = rec.get("recordId")
    if not pmid:
        raise ValueError("missing recordId")
    pmid = str(pmid)

    model_output = rec.get("modelOutput")
    if not model_output:
        raise ValueError("missing modelOutput")
    content = model_output.get("content") or []
    if not content or not isinstance(content[0], dict):
        raise ValueError("modelOutput.content is empty or malformed")
    text = content[0].get("text")
    if not text:
        raise ValueError("modelOutput.content[0] has no text")
    extraction = json.loads(text)
    if not isinstance(extraction, dict):
        raise ValueError("extraction JSON is not an object")

    messages = (rec.get("modelInput") or {}).get("messages") or []
    if not messages or not isinstance(messages[0], dict):
        raise ValueError("missing modelInput.messages")
    user_content = messages[0].get("content")
    if not isinstance(user_content, str) or not user_content:
        raise ValueError("modelInput.messages[0].content is empty")
    source_text = _TEXT_PREFIX.sub("", user_content, count=1).strip()
    if not source_text:
        raise ValueError("empty source text after stripping PMID prefix")
    return pmid, extraction, source_text


def _process_record(
    pmid: str,
    extraction: dict,
    source_text: str,
    temp_dir: Path,
    output_path: Path,
    ae_min_len: int,
) -> None:
    """Resolve anchors and ground one extraction to ``output_path``.

    Parameters
    ----------
    pmid :
        PubMed ID used as the temporary filename stem.
    extraction :
        Parsed LLM JSON (mutated in place by ``resolve_anchors``).
    source_text :
        Article text the model saw, used to resolve anchors.
    temp_dir :
        Directory for the resolved JSON consumed by ``ground_json``.
    output_path :
        Destination grounded JSON path.
    ae_min_len :
        Passed through to ``ground_json``.
    """
    resolved = resolve_anchors(extraction, split_sentences(source_text))
    raw_path = temp_dir / f"{pmid}.json"
    raw_path.write_text(json.dumps(resolved), encoding="utf-8")
    _ = ground_json(raw_path, output_path, ae_min_len=ae_min_len)


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
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=RESULTS_GROUNDED_DIR.base,
    show_default=True,
    help="Directory for grounded <pmid>.json files.",
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
def main(
    input_path: Path,
    output_dir: Path,
    overwrite: bool,
    limit: int | None,
    ae_min_len: int,
) -> None:
    """Ground Bedrock JSONL extraction output.

    INPUT_PATH is a JSONL file or a directory of ``*.jsonl.out`` shards.
    """
    jsonl_paths = _jsonl_paths(input_path)
    if not jsonl_paths:
        raise click.ClickException(f"No JSONL files found under {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
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
                pmid, extraction, source_text = _parse_record(line)
            except Exception as exc:
                n_fail += 1
                logger.warning("Failed to parse %s: %s", loc, exc)
                continue

            out_path = output_dir / f"{pmid}.json"
            if out_path.exists() and not overwrite:
                n_skip += 1
                continue

            try:
                _process_record(
                    pmid,
                    extraction,
                    source_text,
                    temp_dir,
                    out_path,
                    ae_min_len,
                )
            except Exception as exc:
                n_fail += 1
                logger.warning("Failed to process PMID %s (%s): %s", pmid, loc, exc)
                continue
            n_ok += 1

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
