import argparse
import gzip
import json
from pathlib import Path

from trialsynth.base.extract.paths import PMID_TEXTS_CACHE
from trialsynth.base.extract.resources import (
    PROMPT,
    TRIAL_RESULT_SCHEMA_ANCHOR,
)

MAX_RECORDS_PER_FILE = 10000
ANTHROPIC_VERSION = "bedrock-2023-05-31"
MAX_TOKENS = 16000
DEFAULT_PREFIX = "run"


def build_batch_input_jsonl(
    articles: list[tuple[str, str, str]],
    out_dir: Path | str,
    prefix: str = DEFAULT_PREFIX,
    prompt: str = PROMPT,
    schema=None,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    """Write Bedrock batch input JSONL for clinical trial article extraction.

    Output files are ``<prefix>_input_<end_index>.jsonl`` in ``out_dir`` so they
    match the ``*_input_{N}.jsonl`` pattern used by extract multi-job
    submission. A single chunk still uses this pattern: ``-o /my/input/dir``
    with prefix ``run`` and 500 records writes
    ``/my/input/dir/run_input_500.jsonl``. For 28741 articles and a max of
    10000, that is ``run_input_10000.jsonl``, ``run_input_20000.jsonl``,
    ``run_input_28741.jsonl``.

    Parameters
    ----------
    articles :
        List of (articleId, pmid, text) tuples.
    out_dir :
        Directory to write JSONL input files into.
    prefix :
        Filename prefix. Default is ``run``. Files are named
        ``<prefix>_input_<end_index>.jsonl``.
    prompt :
        Prompt text to use for the model input. Default is the prompt defined in
        resources/prompt.txt.
    schema :
        JSON schema to use for the model output. Default is the schema defined
        in resources/trial_result_schema_anchor.json.
    max_records :
        Maximum number of records per output file. If more articles are
        provided, the output is split. Default is 10000.

    Returns
    -------
    :
        Local input file paths that were written, in order.
    """
    if schema is None:
        schema = TRIAL_RESULT_SCHEMA_ANCHOR
    prefix = prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        raise ValueError(
            f"prefix must be a non-empty filename, got {prefix!r}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(articles)
    written: list[Path] = []
    for start in range(0, n, max_records):
        end = min(start + max_records, n)
        chunk_path = out_dir / f"{prefix}_input_{end}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as out:
            for article_id, pmid, text in articles[start:end]:
                model_input = {
                    "anthropic_version": ANTHROPIC_VERSION,
                    "max_tokens": MAX_TOKENS,
                    "system": prompt,
                    "messages": [
                        {"role": "user", "content": f"Text (PMID {pmid}): {text}"}
                    ],
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": schema,
                        }
                    },
                }
                record = {"recordId": article_id, "modelInput": model_input}
                out.write(json.dumps(record) + "\n")
        written.append(chunk_path)
    return written


def main(
    pmids: list[str],
    out_dir: Path | str,
    texts: dict | None = None,
    prefix: str = DEFAULT_PREFIX,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    """Build batch input JSONL from PMIDs.

    Uses each PMID as the Bedrock recordId. Article text comes from the
    PMID text cache mapping (title plus preferred body text).

    Parameters
    ----------
    pmids :
        PMIDs to include in the batch input.
    out_dir :
        Directory to write ``<prefix>_input_<end_index>.jsonl`` files into.
    texts :
        Mapping from PMID to a dict with ``title`` and optionally
        ``abstract`` or ``fulltext``. Defaults to the on-disk cache.
    prefix :
        Filename prefix. Default is ``run``.
    max_records :
        Maximum number of records per output file. Default is 10000.

    Returns
    -------
    :
        Local input file paths that were written, in order.
    """
    if texts is None:
        cache_file = Path(PMID_TEXTS_CACHE)
        texts = {}
        if cache_file.exists():
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                texts = {str(pmid): rec for pmid, rec in json.load(f).items()}

    articles = []
    missing = []
    for pmid in pmids:
        pmid = str(pmid)
        rec = texts.get(pmid)
        if rec is None:
            missing.append(pmid)
            continue
        title = (rec.get("title") or "").strip()
        body = (rec.get("fulltext") or rec.get("abstract") or "").strip()
        if title and body:
            text = body if body.startswith(title) else f"{title}\n\n{body}"
        else:
            text = title or body
        articles.append((pmid, pmid, text))

    if missing:
        preview = ", ".join(missing[:10])
        extra = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"No cached text for {len(missing)} PMID(s): {preview}{extra}"
        )

    return build_batch_input_jsonl(
        articles,
        out_dir=Path(out_dir),
        prefix=prefix,
        max_records=max_records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Bedrock batch input JSONL from article texts."
    )
    pmid_group = parser.add_mutually_exclusive_group(required=True)
    pmid_group.add_argument(
        "--pmid-file",
        type=Path,
        help="Path to a file containing PMIDs, one per line.",
    )
    pmid_group.add_argument(
        "--pmids",
        nargs="+",
        metavar="PMID",
        help="Explicit list of PMIDs to include.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory to write *_input_{N}.jsonl files into.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=(
            "Filename prefix for output files named "
            "<prefix>_input_<end_index>.jsonl. Default: run."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS_PER_FILE,
        help=(
            "Maximum records per output file. Larger inputs are split into "
            "files named <prefix>_input_<end_index>.jsonl. Default: 10000."
        ),
    )
    args = parser.parse_args()
    if args.pmid_file:
        pmids = [
            line.strip()
            for line in args.pmid_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        pmids = args.pmids
    _ = main(
        pmids, args.output, prefix=args.prefix, max_records=args.max_records
    )
