"""
Functions for extracting and resolving evidence anchors to full sentences in
clinical trial text.
"""
import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger('trialsynth.base.extract.extract')


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
    """Replace every evidence_anchor with the full containing sentence.

    Uses canonical field names (source_sentence / evidence_text) expected by
    ground_results.py and generate_html.py. Genetic markers keep their ``role``
    field; only ``evidence_anchor`` is replaced with ``evidence_text``.

    Parameters
    ----------
    raw :
        Parsed LLM extraction JSON.
    sentences :
        Source-text sentences used to resolve each evidence_anchor.

    Returns
    -------
    :
        The same dict, mutated in place.
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
    for item in genetic.get("markers", []):
        a = item.pop("evidence_anchor", "")
        item["evidence_text"] = best_sentence_for_anchor(a, sentences)

    return raw
