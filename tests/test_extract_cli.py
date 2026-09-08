"""Tests for the TrialSynth extract Click CLI."""
from trialsynth.base.extract.cli import _s3_upload_targets


def test_s3_upload_targets_single_object(tmp_path):
    # One local file uploaded to an explicit .jsonl object URI.
    local = tmp_path / "run_input_1.jsonl"
    local.write_text("{}\n", encoding="utf-8")
    assert _s3_upload_targets([local], "s3://bucket/path/in.jsonl") == [
        (local, "bucket", "path/in.jsonl")
    ]


def test_s3_upload_targets_prefix_keeps_filename(tmp_path):
    # Two files under a prefix keep their filenames in the S3 keys.
    a = tmp_path / "run_input_2.jsonl"
    b = tmp_path / "run_input_4.jsonl"
    a.write_text("{}\n", encoding="utf-8")
    b.write_text("{}\n", encoding="utf-8")
    targets = _s3_upload_targets([a, b], "s3://bucket/prefix/")
    assert targets == [
        (a, "bucket", "prefix/run_input_2.jsonl"),
        (b, "bucket", "prefix/run_input_4.jsonl"),
    ]
