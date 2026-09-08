"""Tests for Bedrock batch input JSONL file naming."""
import json
import pytest

from trialsynth.base.extract.build_batch_input_json_schema import (
    build_batch_input_jsonl,
    main as build_batch_input,
)
from trialsynth.base.extract.extract_bedrock import INPUT_FILE_RE


def _write_articles(tmp_path, n: int):
    # Create n fake <pmid>.txt files and matching (id, pmid, path) tuples.
    articles = []
    for i in range(1, n + 1):
        pmid = str(i)
        path = tmp_path / f"{pmid}.txt"
        path.write_text(f"article {pmid}", encoding="utf-8")
        articles.append((pmid, pmid, path))
    return articles


def test_single_chunk_uses_input_n_name(tmp_path):
    # Two articles fit in one file when max_records is large.
    articles = _write_articles(tmp_path, 2)
    written = build_batch_input_jsonl(
        articles,
        tmp_path / "run.jsonl",
        prompt="test prompt",
        schema={"type": "object"},
        max_records=10000,
    )
    # Output is <stem>_input_<n>.jsonl
    assert [path.name for path in written] == ["run_input_2.jsonl"]
    assert written[0].exists()
    assert not (tmp_path / "run.jsonl").exists()
    # Name should match the extract multi-job *_input_{N}.jsonl file name regex.
    assert INPUT_FILE_RE.search(written[0].name)
    # Check that the JSONL file has the expected record content.
    lines = written[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["recordId"] == "1"
    assert rec["modelInput"]["system"] == "test prompt"


def test_multi_chunk_uses_input_end_index(tmp_path):
    # Five articles with max_records=2 should split into three files.
    articles = _write_articles(tmp_path, 5)
    written = build_batch_input_jsonl(
        articles,
        tmp_path / "run.jsonl",
        prompt="test prompt",
        schema={"type": "object"},
        max_records=2,
    )
    # File names use the cumulative end index: 2, 4, then 5 (remainder).
    assert [path.name for path in written] == [
        "run_input_2.jsonl",
        "run_input_4.jsonl",
        "run_input_5.jsonl",
    ]
    for path in written:
        assert path.exists()
        assert INPUT_FILE_RE.search(path.name)
    # Record counts are 2, 2, and a remainder of 1.
    counts = [
        len(path.read_text(encoding="utf-8").splitlines()) for path in written
    ]
    assert counts == [2, 2, 1]


def test_builder_main_missing_pmid_lists_ids(tmp_path):
    # Only PMID 1 has a text file; 2 and 3 are missing.
    (tmp_path / "1.txt").write_text("ok", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"2\.txt|2") as exc_info:
        build_batch_input(
            ["1", "2", "3"],
            tmp_path / "run.jsonl",
            content_dir=tmp_path,
        )
    # Error message names the missing PMIDs.
    message = str(exc_info.value)
    assert "2" in message
    assert "3" in message


def test_builder_main_writes_input_named_file(tmp_path):
    # PMID-based builder should write the same *_input_{N}.jsonl names.
    (tmp_path / "123.txt").write_text("hello", encoding="utf-8")
    written = build_batch_input(
        ["123"],
        tmp_path / "run.jsonl",
        content_dir=tmp_path,
        max_records=10000,
    )
    assert written == [tmp_path / "run_input_1.jsonl"]
    assert written[0].exists()
    assert INPUT_FILE_RE.search(written[0].name)
