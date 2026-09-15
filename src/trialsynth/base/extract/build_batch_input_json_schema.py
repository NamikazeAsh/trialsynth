import json
from pathlib import Path

from trialsynth.base.extract.corpus import Corpus

MAX_RECORDS_PER_FILE = 10000
ANTHROPIC_VERSION = "bedrock-2023-05-31"
MAX_TOKENS = 16000
DEFAULT_PREFIX = "run"


def build_batch_input_jsonl(
    records: list[tuple[str, str]],
    out_dir: Path | str,
    corpus: Corpus,
    prefix: str = DEFAULT_PREFIX,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    """Write Bedrock batch input JSONL for a corpus.

    Output files are ``<prefix>_input_<end_index>.jsonl`` in ``out_dir`` so they
    match the ``*_input_{N}.jsonl`` pattern used by extract multi-job
    submission. A single chunk still uses this pattern: ``-o /my/input/dir``
    with prefix ``run`` and 500 records writes
    ``/my/input/dir/run_input_500.jsonl``. For 28741 records and a max of
    10000, that is ``run_input_10000.jsonl``, ``run_input_20000.jsonl``,
    ``run_input_28741.jsonl``.

    Parameters
    ----------
    records :
        List of (record_id, text) tuples.
    out_dir :
        Directory to write JSONL input files into.
    corpus :
        Corpus supplying the prompt, result schema, and text framing.
    prefix :
        Filename prefix. Default is ``run``. Files are named
        ``<prefix>_input_<end_index>.jsonl``.
    max_records :
        Maximum number of records per output file. If more records are
        provided, the output is split. Default is 10000.

    Returns
    -------
    :
        Local input file paths that were written, in order.
    """
    prefix = prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        raise ValueError(
            f"prefix must be a non-empty filename, got {prefix!r}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    written: list[Path] = []
    for start in range(0, n, max_records):
        end = min(start + max_records, n)
        chunk_path = out_dir / f"{prefix}_input_{end}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as out:
            for record_id, text in records[start:end]:
                model_input = {
                    "anthropic_version": ANTHROPIC_VERSION,
                    "max_tokens": MAX_TOKENS,
                    "system": corpus.prompt,
                    "messages": [
                        {"role": "user", "content": corpus.frame(record_id, text)}
                    ],
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": corpus.schema,
                        }
                    },
                }
                record = {"recordId": record_id, "modelInput": model_input}
                out.write(json.dumps(record) + "\n")
        written.append(chunk_path)
    return written


def main(
    record_ids: list[str],
    out_dir: Path | str,
    corpus: Corpus,
    texts: dict[str, str],
    prefix: str = DEFAULT_PREFIX,
    max_records: int = MAX_RECORDS_PER_FILE,
) -> list[Path]:
    """Build batch input JSONL from record IDs.

    Uses each record ID as the Bedrock recordId.

    Parameters
    ----------
    record_ids :
        Record IDs to include in the batch input.
    out_dir :
        Directory to write ``<prefix>_input_<end_index>.jsonl`` files into.
    corpus :
        Corpus supplying the prompt, schema, and framing.
    texts :
        Mapping from record ID to source text, as returned by
        ``corpus.load_texts``.
    prefix :
        Filename prefix. Default is ``run``.
    max_records :
        Maximum number of records per output file. Default is 10000.

    Returns
    -------
    :
        Local input file paths that were written, in order.

    Raises
    ------
    ValueError
        If any record ID has no text.
    """
    records = []
    missing = []
    for record_id in record_ids:
        record_id = str(record_id)
        text = texts.get(record_id)
        if not text:
            missing.append(record_id)
            continue
        records.append((record_id, text))

    if missing:
        preview = ", ".join(missing[:10])
        extra = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"No text for {len(missing)} {corpus.name} record(s): "
            f"{preview}{extra}"
        )

    return build_batch_input_jsonl(
        records,
        out_dir=Path(out_dir),
        corpus=corpus,
        prefix=prefix,
        max_records=max_records,
    )
