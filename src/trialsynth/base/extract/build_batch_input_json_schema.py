import argparse
import json
from pathlib import Path

from trialsynth.base.extract.paths import CONTENT_TXT_DIR
from trialsynth.base.extract.resources import (
    PROMPT,
    TRIAL_RESULT_SCHEMA_ANCHOR,
)

MAX_RECORDS_PER_FILE = 10000
ANTHROPIC_VERSION = "bedrock-2023-05-31"
MAX_TOKENS = 16000


def build_batch_input_jsonl(
    articles: list[tuple[str, str, Path | str]],
    out_path: Path | str,
    prompt: str = PROMPT,
    schema=None,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    """Write Bedrock batch input JSONL for clinical trial article extraction.

    Output is always written as ``<stem>_input_<end_index>.jsonl`` so input files
    match the ``*_input_{N}.jsonl`` pattern used by extract multi-job submission.
    A single chunk still uses this pattern: ``-o /my/input/dir/run.jsonl`` with 500
    records writes ``/my/input/dir/run_input_500.jsonl``. For 28741 articles and a
    max of 10000, that is ``/my/input/dir/run_input_10000.jsonl``,
    ``/my/input/dir/run_input_20000.jsonl``, ``/my/input/dir/run_input_28741.jsonl``.

    Parameters
    ----------
    articles :
        List of (articleId, pmid, article_path) tuples.
    out_path :
        Output JSONL path whose stem is used as the prefix.
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(articles)
    written: list[Path] = []
    for start in range(0, n, max_records):
        end = min(start + max_records, n)
        chunk_path = out_path.with_name(f"{out_path.stem}_input_{end}.jsonl")
        with open(chunk_path, "w", encoding="utf-8") as out:
            for article_id, pmid, article_path in articles[start:end]:
                text = Path(article_path).read_text(encoding="utf-8")
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
    out_path: Path | str,
    content_dir: Path | None = None,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    """Build batch input JSONL from PMIDs.

    Uses each PMID as the Bedrock recordId. Article text is read from
    ``<content_dir>/<pmid>.txt``.

    Parameters
    ----------
    pmids :
        PMIDs to include in the batch input.
    out_path :
        Output JSONL path whose stem is used for
        ``<stem>_input_<end_index>.jsonl`` input files.
    content_dir :
        Directory containing ``<pmid>.txt`` files. Defaults to the
        trialsynth content/txt store.
    max_records :
        Maximum number of records per output file. Default is 10000.

    Returns
    -------
    :
        Local input file paths that were written, in order.
    """
    if content_dir is None:
        content_dir = CONTENT_TXT_DIR.base

    articles = []
    missing = []
    for pmid in pmids:
        article_path = Path(content_dir) / f"{pmid}.txt"
        if not article_path.exists():
            missing.append(pmid)
            continue
        articles.append((pmid, pmid, article_path))

    if missing:
        preview = ", ".join(missing[:10])
        extra = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"No text file found for {len(missing)} PMID(s) in {content_dir}: "
            f"{preview}{extra}"
        )

    return build_batch_input_jsonl(
        articles, out_path=Path(out_path), max_records=max_records
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
        help="Output path for the batch input JSONL file.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS_PER_FILE,
        help=(
            "Maximum records per output file. Larger inputs are split into "
            "files named <stem>_input_<end_index>.jsonl. Default: 10000."
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
    _ = main(pmids, args.output, max_records=args.max_records)
