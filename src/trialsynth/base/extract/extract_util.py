"""
Functions for extracting and resolving evidence anchors to full sentences in
clinical trial text.
"""
import re
import csv
import gzip
import tqdm
import logging
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

from indra.literature.pmc_client import id_lookup, get_text_s3
from indra.literature.pubmed_client import get_abstract, get_metadata_for_all_ids

from trialsynth.ctgov.config import CTConfig
from trialsynth.base.extract.paths import CONTENT_TXT_DIR


logger = logging.getLogger(__name__)


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


def get_trial_pmids() -> list[str]:
    """Return PMIDs linked to trials from either or both of ctgov or pubmed

    Returns
    -------
    :
        List of PMIDs that are from either the registry result links and the
        PubMed XML links.
    """

    ct_config = CTConfig()
    if not ct_config.trial_publication_edges_path.exists():
        raise FileNotFoundError(
            f"Trial-publication edges file not found: "
            f"{ct_config.trial_publication_edges_path}. Must run clinicaltrials "
            f"pipeline before running this script."
        )
    with gzip.open(ct_config.trial_publication_edges_path, "rt") as f:
        reader = csv.reader(f)
        _ = next(reader)
        # Headers are:
        # trial_id, pmid, rel_type, source, ref_type
        intersection = {
            row[1] for row in reader if row[1]
        }

    return sorted(intersection)


def download_texts(pmids: list[str]):
    """Download texts for PMIDs sequentially

    Parameters
    ----------
    pmids :
        List of PMIDs to download text for.
    """
    logger.info(f"Downloading text for {len(pmids)} PMIDs...")

    for pmid in tqdm.tqdm(pmids):
        _download_one_text(pmid)


def _download_one_text(pmid: str) -> None:
    # Tries PMC full text from S3 first, then falls back to the PubMed abstract.
    # Writes ``<pmid>.txt`` to CONTENT_TXT_DIR on success.
    if CONTENT_TXT_DIR.join(name=f"{pmid}.txt").exists():
        return

    try:
        text = None

        pmcid = id_lookup(pmid, idtype="pmid").get("pmcid")
        if pmcid:
            text = get_text_s3(pmcid)

        if not text:
            text = get_abstract(pmid, prepend_title=True)

        if text:
            CONTENT_TXT_DIR.join(name=f"{pmid}.txt").write_text(text, encoding="utf-8")
        else:
            tqdm.tqdm.write(f"{pmid} - NO CONTENT")

    except Exception as e:
        tqdm.tqdm.write(f"{pmid} - FAILED: {e}")


def download_texts_parallel(pmids: list[str], max_workers: int = 8):
    """Download texts for PMIDs concurrently.

    Same per-PMID behavior as :func:`download_texts`, using a thread pool.

    Parameters
    ----------
    pmids :
        List of PMIDs to download text for.
    max_workers :
        Maximum number of worker threads. Default: 8.
    """
    logger.info(
        f"Downloading text for {len(pmids)} PMIDs with {max_workers} workers..."
    )

    pending = [
        pmid for pmid in pmids
        if not CONTENT_TXT_DIR.join(name=f"{pmid}.txt").exists()
    ]
    skipped = len(pmids) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} PMIDs with existing text files")

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [
            executor.submit(_download_one_text, pmid) for pmid in pending
        ]
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
            fut.result()


def _attempt_fulltext(pmid: str, pmcid: str, abstract) -> str:
    try:
        text = get_text_s3(pmcid)
        if text:
            CONTENT_TXT_DIR.join(name=f"{pmid}.txt").write_text(
                text, encoding="utf-8"
            )
            return "s3"
    except Exception as e:
        tqdm.tqdm.write(f"{pmid} - S3 FAILED: {e}")

    if abstract:
        CONTENT_TXT_DIR.join(name=f"{pmid}.txt").write_text(
            abstract, encoding="utf-8"
        )
        return "abs"

    tqdm.tqdm.write(f"{pmid} - NO CONTENT")
    return "none"


def download_texts_bulk(pmids: list[str], max_workers: int = 8):
    """Download texts via a bulk PubMed metadata fetch and S3 PMC

    Parameters
    ----------
    pmids :
        List of PMIDs to download text for.
    max_workers :
        Maximum number of worker threads for download. Default: 8.
    """
    logger.info(f"Bulk-downloading text for {len(pmids)} PMIDs...")

    pending = [
        pmid for pmid in pmids
        if not CONTENT_TXT_DIR.join(name=f"{pmid}.txt").exists()
    ]
    skipped = len(pmids) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} PMIDs with existing text files")
    if not pending:
        return

    metadata = get_metadata_for_all_ids(
        pending, get_abstracts=True, prepend_title=True
    ) or {}

    n_abs = 0
    n_no_content = 0
    s3_jobs = []
    missing = []
    for pmid in tqdm.tqdm(pending, desc="Bulk metadata"):
        rec = metadata.get(pmid)
        if rec is None:
            missing.append(pmid)
            continue
        abstract = rec.get("abstract") or None
        pmcid = rec.get("pmcid")
        if pmcid:
            s3_jobs.append((pmid, pmcid, abstract))
        elif abstract:
            CONTENT_TXT_DIR.join(name=f"{pmid}.txt").write_text(
                abstract, encoding="utf-8"
            )
            n_abs += 1
        else:
            tqdm.tqdm.write(f"{pmid} - NO CONTENT")
            n_no_content += 1

    logger.info(
        f"Bulk metadata: {n_abs} abstracts written, {len(s3_jobs)} with "
        f"PMCID, {len(missing)} missing from response"
    )

    n_s3 = 0
    if s3_jobs:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [
                executor.submit(_attempt_fulltext, pmid, pmcid, abstract)
                for pmid, pmcid, abstract in s3_jobs
            ]
            for fut in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="S3 full text"
            ):
                status = fut.result()
                if status == "s3":
                    n_s3 += 1
                elif status == "abs":
                    n_abs += 1
                else:
                    n_no_content += 1

    if missing:
        logger.info(
            f"Falling back to per-PMID download for {len(missing)} PMIDs"
        )
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [
                executor.submit(_download_one_text, pmid) for pmid in missing
            ]
            for fut in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="PMID fallback"
            ):
                fut.result()

    logger.info(
        f"Bulk download complete: {n_abs} abstracts, {n_s3} S3 full texts, "
        f"{n_no_content} with no content"
    )
