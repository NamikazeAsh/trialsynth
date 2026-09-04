import argparse
import json
from pathlib import Path

from trialsynth.base.extract.paths import CONTENT_TXT_DIR
from trialsynth.base.extract.resources import (
    PROMPT,
    TRIAL_RESULT_SCHEMA_ANCHOR,
)

MIN_NUM_RECORDS = 100
MAX_RECORDS_PER_FILE = 10000
ANTHROPIC_VERSION = "bedrock-2023-05-31"
MAX_TOKENS = 16000
OUT_PATH = Path(__file__).parent.parent / "outputs" / "batch_input_json_schema_100.jsonl"


def build_batch_input_jsonl(
    articles: list[tuple[str, str, Path | str]],
    out_path: Path = OUT_PATH,
    prompt: str = PROMPT,
    schema=None,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> None:
    """Write Bedrock batch input JSONL for clinical trial article extraction.

    If the number of articles exceeds ``max_records``, the output is split
    into multiple files named ``<stem>_<end_index><suffix>``. For 28741
    articles and a max of 10000, that is ``outfile_10000.jsonl``,
    ``outfile_20000.jsonl``, ``outfile_28741.jsonl``.

    Parameters
    ----------
    articles :
        List of (articleId, pmid, article_path) tuples.
    out_path :
        Output JSONL file path. Used as-is when no split is needed; otherwise
        the stem is used as a prefix for the chunk files.
    prompt :
        Prompt text to use for the model input. Default is the prompt defined in
        resources/prompt.txt.
    schema :
        JSON schema to use for the model output. Default is the schema defined
        in resources/trial_result_schema_anchor.json.
    max_records :
        Maximum number of records per output file. If more articles are
        provided, the output is split. Default is 10000.
    """
    if schema is None:
        schema = TRIAL_RESULT_SCHEMA_ANCHOR
    if len(articles) < MIN_NUM_RECORDS:
        raise RuntimeError(
            f"Only {len(articles)} articles provided, need at least {MIN_NUM_RECORDS} "
            f"for batch processing."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(articles)
    split = n > max_records
    for start in range(0, n, max_records):
        end = min(start + max_records, n)
        chunk_path = (
            out_path.with_name(f"{out_path.stem}_{end}{out_path.suffix}")
            if split
            else out_path
        )
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


def main(
    pmids: list[str],
    out_path: Path | str,
    content_dir: Path | None = None,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> None:
    """Build batch input JSONL from PMIDs.

    Uses each PMID as the Bedrock recordId. Article text is read from
    ``<content_dir>/<pmid>.txt``.

    Parameters
    ----------
    pmids :
        PMIDs to include in the batch input.
    out_path :
        Output JSONL file path. Split into ``<stem>_<end_index><suffix>``
        files if the number of PMIDs exceeds ``max_records``.
    content_dir :
        Directory containing ``<pmid>.txt`` files. Defaults to the
        trialsynth content/txt store.
    max_records :
        Maximum number of records per output file. Default is 10000.
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

    build_batch_input_jsonl(
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
            "files named <stem>_<end_index><suffix>. Default: 10000."
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
    main(pmids, args.output, max_records=args.max_records)
