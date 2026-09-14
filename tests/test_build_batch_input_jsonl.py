"""Tests for Bedrock batch input JSONL file naming."""
import json
import pytest

from trialsynth.base.extract.build_batch_input_json_schema import (
    build_batch_input_jsonl,
    main as build_batch_input,
)
from trialsynth.base.extract.extract_bedrock import INPUT_FILE_RE


def _articles(n: int):
    # In-memory (articleId, pmid, text) tuples.
    return [(str(i), str(i), f"article {i}") for i in range(1, n + 1)]


def test_single_chunk_uses_input_n_name(tmp_path):
    # Two articles fit in one file when max_records is large.
    articles = _articles(2)
    written = build_batch_input_jsonl(
        articles,
        tmp_path,
        prompt="test prompt",
        schema={"type": "object"},
        max_records=10,
    )
    # Output is <prefix>_input_<n>.jsonl
    assert [path.name for path in written] == ["run_input_2.jsonl"]
    assert written[0] == tmp_path / "run_input_2.jsonl"
    assert written[0].exists()
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
    articles = _articles(5)
    written = build_batch_input_jsonl(
        articles,
        tmp_path,
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
    # Only PMID 111 is in the cache mapping; 112 and 113 are missing.
    with pytest.raises(ValueError, match=r"2") as exc_info:
        build_batch_input(
            ["111", "112", "113"],
            tmp_path,
            texts={"111": {"title": "ok"}},
        )
    # Error message names the missing PMIDs.
    message = str(exc_info.value)
    assert "112" in message
    assert "113" in message


def test_builder_main_writes_input_named_file(tmp_path):
    # PMID-based builder should write the same *_input_{N}.jsonl names.
    written = build_batch_input(
        ["123"],
        tmp_path,
        texts={"123": {"title": "hello"}},
    )
    assert written == [tmp_path / "run_input_1.jsonl"]
    assert written[0].exists()
    assert INPUT_FILE_RE.search(written[0].name)


def test_builder_prefers_fulltext_over_abstract(tmp_path):
    # Prompt body uses fulltext when both fulltext and abstract are present.
    written = build_batch_input(
        ["111"],
        tmp_path,
        texts={
            "111": {
                "title": "Title",
                "abstract": "Abstract body",
                "fulltext": "Full text body",
            }
        },
    )
    rec = json.loads(written[0].read_text(encoding="utf-8").splitlines()[0])
    content = rec["modelInput"]["messages"][0]["content"]
    assert "Full text body" in content
    assert "Abstract body" not in content


def test_builder_falls_back_to_abstract(tmp_path):
    # Prompt body uses abstract when fulltext is absent.
    written = build_batch_input(
        ["111"],
        tmp_path,
        texts={"111": {"title": "Title", "abstract": "Abstract body"}},
    )
    rec = json.loads(written[0].read_text(encoding="utf-8").splitlines()[0])
    content = rec["modelInput"]["messages"][0]["content"]
    assert "Abstract body" in content


def test_custom_prefix_names_output_files(tmp_path):
    # --prefix becomes the filename prefix for *_input_{N}.jsonl files.
    articles = _articles(1)
    written = build_batch_input_jsonl(
        articles,
        tmp_path,
        prefix="bedrock_test",
        prompt="test prompt",
        schema={"type": "object"},
    )
    assert written == [tmp_path / "bedrock_test_input_1.jsonl"]
    assert written[0].exists()
