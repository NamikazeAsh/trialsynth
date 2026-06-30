"""
Anchor phrase extractor for clinical trial result papers.

Replaces the v1 schema (full text spans) with a two-stage approach:
the LLM returns a short verbatim anchor phrase (5-15 words) per data point;
post-processing resolves it to the full containing sentence via fuzzy match.

Output is consumed by ground_results.py and generate_html.py.
"""

import argparse
import csv
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from openai import OpenAI

from trialsynth.base.extract.resources import TRIAL_RESULT_SCHEMA_ANCHOR
from trialsynth.base.extract.paths import CONTENT_TXT_DIR, RESULTS_RAW_DIR, \
    RESULTS_GROUNDED_DIR, RESULTS_DIR

logger = logging.getLogger('trialsynth.base.extract.extract')

LLM_MODEL = "gpt-5.4-mini"


# Abbreviations whose trailing period must not be treated as a sentence
# boundary, e.g. "nausea (57.1% vs. 8.6%)" should stay one sentence.
_SENTENCE_ABBREVIATIONS = {
    "vs", "e.g", "i.e", "etc", "al", "fig", "figs", "no", "nos",
    "cf", "approx", "ca", "vol", "ref", "eq", "pp", "incl",
}


def _ends_with_abbreviation(sentence: str) -> bool:
    match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", sentence.rstrip())
    return bool(match) and match.group(1).lower().rstrip(".") in _SENTENCE_ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    blob = re.sub(r"\s+", " ", text.strip())
    if not blob:
        return []
    parts = re.split(r"(?<=[.!?])\s+", blob)
    sentences: list[str] = []
    for part in parts:
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


def normalize(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9%.\- ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def best_sentence_for_anchor(anchor: str, sentences: list[str]) -> str:
    """Returns the full sentence containing the anchor, or the anchor itself if no match."""
    if not anchor or not sentences:
        return anchor

    anchor_lower = anchor.lower()
    for sentence in sentences:
        if anchor_lower in sentence.lower():
            return sentence

    query_norm = normalize(anchor)
    query_tokens = set(query_norm.split())
    query_nums = set(extract_numbers(anchor))

    best_sentence = anchor
    best_score = 0.0

    for sentence in sentences:
        sent_norm = normalize(sentence)
        if not sent_norm:
            continue
        ratio = SequenceMatcher(None, query_norm, sent_norm).ratio()
        sent_tokens = set(sent_norm.split())
        overlap = len(query_tokens & sent_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
        sent_nums = set(extract_numbers(sentence))
        num_overlap = len(query_nums & sent_nums) / max(1, len(query_nums)) if query_nums else 0.0

        score = (0.55 * ratio) + (0.25 * overlap) + (0.20 * num_overlap)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score < 0.20:
        logger.warning(f"Low confidence match (score={best_score:.2f}) for anchor: '{anchor[:60]}'")

    return best_sentence


def resolve_anchors(raw: dict, sentences: list[str]) -> dict:
    """
    Walk the raw LLM output and replace every evidence_anchor string with
    the full containing sentence, using canonical field names
    (source_sentence / evidence_text) expected by ground_results.py and generate_html.py.
    """
    for arm in raw.get("arms", []):
        anchor = arm.pop("evidence_anchor", "")
        arm["source_sentence"] = best_sentence_for_anchor(anchor, sentences)
        for m in arm.get("metrics", []):
            a = m.pop("evidence_anchor", "")
            m["source_sentence"] = best_sentence_for_anchor(a, sentences)
        for ae in arm.get("adverse_events", []):
            a = ae.pop("evidence_anchor", "")
            ae["source_sentence"] = best_sentence_for_anchor(a, sentences)

    for item in raw.get("results", []):
        a = item.pop("evidence_anchor", "")
        item["evidence_text"] = best_sentence_for_anchor(a, sentences)

    for item in raw.get("inclusion_criteria", []):
        a = item.pop("evidence_anchor", "")
        item["evidence_text"] = best_sentence_for_anchor(a, sentences)

    for item in raw.get("exclusion_criteria", []):
        a = item.pop("evidence_anchor", "")
        item["evidence_text"] = best_sentence_for_anchor(a, sentences)

    for comp in raw.get("statistical_comparisons", []):
        for m in comp.get("metrics", []):
            a = m.pop("evidence_anchor", "")
            m["source_sentence"] = best_sentence_for_anchor(a, sentences)

    genetic = raw.get("genetic", {})
    for item in genetic.get("genetic_inclusion", []):
        a = item.pop("evidence_anchor", "")
        item["evidence_text"] = best_sentence_for_anchor(a, sentences)
    for item in genetic.get("genetic_exclusion", []):
        a = item.pop("evidence_anchor", "")
        item["evidence_text"] = best_sentence_for_anchor(a, sentences)

    return raw


def extract_trial_data(client: OpenAI, text: str, pmid: str) -> dict:
    sentences = split_sentences(text)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical data scientist. Transform medical text into a queryable database structure. "
                    "1. For EVERY single extracted data point (including Arm definitions, Dosages, Inclusion/Exclusion Criteria, "
                    "Genetic Markers, Metrics, and Adverse Events), you MUST return a short verbatim anchor phrase "
                    "(5 to 15 words maximum) copied directly from the source text that uniquely identifies the sentence "
                    "containing the evidence. Use the field 'evidence_anchor' for this. "
                    "Do NOT copy the full sentence — just a unique short phrase from it. "
                    "This anchor is used programmatically to locate the full sentence, so it must be verbatim. "
                    "2. Extract qualitative results and metrics. "
                    "3. Assign safety events to arms. "
                    "4. Embed structured metrics into Arms and Comparisons. When multiple subgroups "
                    "exist for one outcome, isolate the subgroup name (e.g., 'Women <50') into the 'name' field "
                    "and the clean numeric/CI result into 'value_text'. Do NOT repeat the baseline comparison. "
                    "5. Isolate genetic biomarkers into the 'genetic' block using official HGNC Gene Symbols and HGVS nomenclature. "
                    "Always use the official HGNC gene symbol in the text field (e.g. ERBB2 not HER2, MYC not cMYC, "
                    "ABL1 not ABL, MS4A1 not CD20) so downstream grounding tools can resolve them correctly."
                )
            },
            {"role": "user", "content": f"Text (PMID {pmid}): {text}"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_trial_extraction_anchor",
                "strict": True,
                "schema": TRIAL_RESULT_SCHEMA_ANCHOR
            }
        }
    )

    raw = json.loads(response.choices[0].message.content)
    resolved = resolve_anchors(raw, sentences)
    return resolved, response.usage


def process_pmid(pmid: str, client: OpenAI, content_dir: Path, output_dir: Path) -> dict:
    out_file = output_dir / f"{pmid}.json"
    if out_file.exists():
        return {"pmid": pmid, "status": "skipped"}

    txt_path = content_dir / f"{pmid}.txt"
    if not txt_path.exists():
        return {"pmid": pmid, "status": "missing_text"}

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        result, usage = extract_trial_data(client, text, pmid)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  PMID {pmid} | input: {usage.prompt_tokens} | output: {usage.completion_tokens}")
        return {
            "pmid": pmid,
            "status": "ok",
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
    except Exception as e:
        logger.error(f"Failed {pmid}: {e}")
        return {"pmid": pmid, "status": "error"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmids", nargs="+", default=None, help="Explicit PMID list to process")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_RAW_DIR.base
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pmids:
        target_pmids = args.pmids
    else:
        grounded_dir = RESULTS_GROUNDED_DIR.base
        target_pmids = [f.stem for f in sorted(grounded_dir.glob("*.json"))][:10]

    logger.info(f"Running anchor extraction on {len(target_pmids)} PMIDs...")

    client = OpenAI()
    stats = []

    content_dir = CONTENT_TXT_DIR.base
    for pmid in target_pmids:
        row = process_pmid(pmid, client, content_dir, output_dir)
        stats.append(row)

    completed = [r for r in stats if r["status"] == "ok"]
    total_in = sum(r["input_tokens"] for r in completed)
    total_out = sum(r["output_tokens"] for r in completed)
    logger.info("=" * 60)
    logger.info(f"{'PMID':<15} {'Input Tokens':>15} {'Output Tokens':>15}")
    logger.info("-" * 60)
    for r in completed:
        logger.info(f"{r['pmid']:<15} {r['input_tokens']:>15} {r['output_tokens']:>15}")
    logger.info("-" * 60)
    logger.info(f"{'TOTAL':<15} {total_in:>15} {total_out:>15}")
    logger.info(f"{'AVG/paper':<15} {total_in // max(1, len(completed)):>15} {total_out // max(1, len(completed)):>15}")
    logger.info("=" * 60)

    csv_path = RESULTS_DIR.join(name="token_comparison_anchor.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pmid", "status", "input_tokens", "output_tokens"])
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
