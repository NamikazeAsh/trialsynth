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


def get_pmid_texts(pmids: list[str]) -> dict:
    """Return title, abstract, and full text for each PMID.

    Tries the SQLite lite DB first when configured, then the Postgres
    INDRA DB, then PubMed metadata and PMC S3. Full text is preferred
    over abstract. Title is always included.

    Parameters
    ----------
    pmids :
        PubMed IDs.

    Returns
    -------
    :
        Mapping from PMID to a dict with ``title`` and optionally
        ``abstract`` or ``fulltext``.
    """
    pmid_strs = set(str(pmid) for pmid in pmids)
    out = {pmid: {"title": ""} for pmid in pmid_strs}
    filled = set()
    pbar = tqdm.tqdm(
        total=len(pmid_strs), desc="Fetching PMID texts", unit="pmid"
    )

    def _note_filled(pmid: str) -> None:
        if pmid in filled or pmid not in out:
            return
        rec = out[pmid]
        if rec.get("title") or rec.get("abstract") or rec.get("fulltext"):
            filled.add(pmid)
            pbar.update(1)

    try:
        from indra.config import has_config
        if has_config("INDRA_DB_LITE_LOCATION"):
            from indra_db_lite import (
                get_paragraphs_for_text_ref_ids,
                get_text_ref_ids_for_pmids,
            )
            pmid_to_trid = get_text_ref_ids_for_pmids(
                [int(pmid) for pmid in pmid_strs]
            )
            trid_to_pmid = {
                trid: str(pmid) for pmid, trid in pmid_to_trid.items()
            }
            if pmid_to_trid:
                content = get_paragraphs_for_text_ref_ids(pmid_to_trid.values())
                for trid, paragraphs in content.fulltexts.items():
                    pmid = trid_to_pmid[trid]
                    text = "\n".join(p for p in paragraphs if p)
                    if text:
                        out[pmid]["fulltext"] = text
                        _note_filled(pmid)
                for trid, paragraphs in content.abstracts.items():
                    pmid = trid_to_pmid[trid]
                    if paragraphs:
                        out[pmid]["title"] = paragraphs[0] or ""
                    abstract = "\n".join(p for p in paragraphs[1:] if p)
                    if abstract:
                        out[pmid]["abstract"] = abstract
                    _note_filled(pmid)
                for trid, paragraphs in content.titles.items():
                    pmid = trid_to_pmid[trid]
                    if paragraphs and paragraphs[0]:
                        out[pmid]["title"] = paragraphs[0]
                        _note_filled(pmid)
        else:
            print("DEBUG: indra_db_lite is not available for text retrieval")
            logger.info("INDRA_DB_LITE_LOCATION is not set in the environment, falling back to INDRA DB")
    except Exception as e:
        print("DEBUG: indra_db_lite is not available for text retrieval: %s", e)
        logger.info("indra_db_lite is not available for text retrieval: %s", e)

    need_content = [
        pmid for pmid in pmid_strs
        if "fulltext" not in out[pmid]
        and "abstract" not in out[pmid]
        and not out[pmid]["title"]
    ]
    need_title = [pmid for pmid in pmid_strs if not out[pmid]["title"]]
    if need_content or need_title:
        try:
            from indra.literature.adeft_tools import universal_extract_text
            from indra_db.client.principal.content import get_text
            from indra_db.util import get_db
            from indra_db.util.content_scripts import get_text_content_from_pmids

            db = get_db("primary")
            if db is None:
                raise ValueError("Primary database is not available")
            if need_title:
                for pmid, title in get_text(db, need_title, "title").items():
                    pmid = str(pmid)
                    if pmid in out and title and not out[pmid]["title"]:
                        out[pmid]["title"] = title
                        _note_filled(pmid)
            if need_content:
                identifiers, content = get_text_content_from_pmids(
                    need_content, db=db
                )
                for pmid, ident in identifiers.items():
                    pmid = str(pmid)
                    if pmid not in out or "fulltext" in out[pmid]:
                        continue
                    raw = content.get(ident)
                    if not raw:
                        continue
                    text = universal_extract_text(raw)
                    if not text:
                        continue
                    text_type = ident[3]
                    if text_type == "fulltext":
                        out[pmid]["fulltext"] = text
                        out[pmid].pop("abstract", None)
                    elif text_type in ["abstract", "elsevier_abstract"]:
                        out[pmid]["abstract"] = text
                    elif text_type == "title" and not out[pmid]["title"]:
                        out[pmid]["title"] = text
                    else:
                        continue
                    _note_filled(pmid)
        except Exception as e:
            print(f"DEBUG: get_text_content_from_pmids failed: {e}")
            logger.info("INDRA DB is not available for text retrieval: %s", e)

    need_live = [
        pmid for pmid in pmid_strs
        if not out[pmid]["title"]
        or ("fulltext" not in out[pmid] and "abstract" not in out[pmid])
    ]
    try:
        if need_live:
            metadata = get_metadata_for_all_ids(
                need_live, get_abstracts=True, prepend_title=False
            ) or {}
            missing = []
            s3_jobs = []
            for pmid in need_live:
                rec = metadata.get(pmid)
                if rec is None:
                    missing.append(pmid)
                    continue
                title = rec.get("title") or ""
                if title and not out[pmid]["title"]:
                    out[pmid]["title"] = title
                    _note_filled(pmid)
                if "fulltext" in out[pmid]:
                    continue
                abstract = rec.get("abstract") or None
                pmcid = rec.get("pmcid")
                if pmcid:
                    s3_jobs.append((pmid, pmcid, abstract))
                elif abstract:
                    out[pmid]["abstract"] = abstract
                    _note_filled(pmid)

            if s3_jobs:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {
                        executor.submit(get_text_s3, pmcid): (pmid, abstract)
                        for pmid, pmcid, abstract in s3_jobs
                    }
                    for fut in as_completed(futures):
                        pmid, abstract = futures[fut]
                        try:
                            text = fut.result()
                        except Exception as e:
                            logger.info("%s - S3 FAILED: %s", pmid, e)
                            text = None
                        if text:
                            out[pmid]["fulltext"] = text
                            out[pmid].pop("abstract", None)
                        elif abstract:
                            out[pmid]["abstract"] = abstract
                        _note_filled(pmid)

            for pmid in missing:
                try:
                    pmcid = id_lookup(pmid, idtype="pmid").get("pmcid")
                    text = get_text_s3(pmcid) if pmcid else None
                    if text:
                        out[pmid]["fulltext"] = text
                        out[pmid].pop("abstract", None)
                    elif "fulltext" not in out[pmid]:
                        abstract = get_abstract(pmid, prepend_title=False)
                        if abstract:
                            out[pmid]["abstract"] = abstract
                    _note_filled(pmid)
                except Exception as e:
                    print("DEBUG: FAILED: %s", e)
                    logger.info("%s - FAILED: %s", pmid, e)
    finally:
        pbar.close()
    return out


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
