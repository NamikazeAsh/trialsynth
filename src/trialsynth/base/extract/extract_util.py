"""
Functions for extracting and resolving evidence anchors to full sentences in
clinical trial text.
"""
import re
import csv
import gzip
import json
import os
import tqdm
import logging
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from indra.literature.pmc_client import id_lookup, get_text_s3
from indra.literature.pubmed_client import get_abstract, get_metadata_for_all_ids

from trialsynth.ctgov.config import CTConfig
from trialsynth.base.extract.paths import PMID_TEXTS_CACHE


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


def get_pmid_texts(pmids: list[str], max_workers: int = 8) -> dict:
    """Return title, abstract, and full text for each PMID.

    Tries the SQLite lite DB first when configured, then the Postgres
    INDRA DB, then PubMed metadata and PMC S3. Full text is preferred
    over abstract. Title is always included.

    Parameters
    ----------
    pmids :
        PubMed IDs.
    max_workers :
        Maximum number of worker threads for PMC S3 full-text fetch.
        Default: 8.

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
                with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
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


def download_texts_bulk(pmids: list[str], max_workers: int = 8, cache_path=None) -> dict:
    """Fill the local PMID text cache for the given PMIDs.

    Loads the gzipped JSON cache, fetches only missing PMIDs via
    ``get_pmid_texts``, and merges records that have a title, abstract,
    or full text. Fully empty results are not stored so later runs retry
    them.

    Parameters
    ----------
    pmids :
        List of PMIDs to download text for.
    max_workers :
        Maximum number of worker threads for PMC S3 full-text fetch.
        Default: 8.
    cache_path : pathlib.Path or str, optional
        Cache file to read and write. Defaults to ``PMID_TEXTS_CACHE``.

    Returns
    -------
    :
        The full cache mapping after any merge.
    """
    logger.info(f"Bulk-downloading text for {len(pmids)} PMIDs...")

    cache_file = Path(cache_path) if cache_path is not None else Path(PMID_TEXTS_CACHE)
    cache = {}
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            cache = {str(pmid): rec for pmid, rec in json.load(f).items()}

    pending = [str(pmid) for pmid in pmids if str(pmid) not in cache]
    skipped = len(pmids) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} PMIDs already in the text cache")
    if not pending:
        return cache

    fetched = get_pmid_texts(pending, max_workers=max_workers)
    n_added = 0
    n_empty = 0
    for pmid in pending:
        fetched_rec = fetched.get(pmid) or {}
        title = (fetched_rec.get("title") or "").strip()
        abstract = (fetched_rec.get("abstract") or "").strip()
        fulltext = (fetched_rec.get("fulltext") or "").strip()
        if not title and not abstract and not fulltext:
            n_empty += 1
            continue
        rec = {"title": title}
        if abstract:
            rec["abstract"] = abstract
        if fulltext:
            rec["fulltext"] = fulltext
        cache[pmid] = rec
        n_added += 1

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_file.with_name(cache_file.name + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))
    os.replace(tmp_path, cache_file)

    logger.info(
        f"Bulk download complete: {n_added} cached, {n_empty} with no "
        f"content, {skipped} already present"
    )
    return cache


def _join_title_body(record: dict) -> str:
    """Join one cache record's title and preferred body into source text.

    Full text is preferred over the abstract. The title is skipped when the
    body already opens with it, which is how the PubMed metadata fetch returns
    abstracts.
    """
    title = (record.get("title") or "").strip()
    body = (record.get("fulltext") or record.get("abstract") or "").strip()
    if title and body:
        return body if body.startswith(title) else f"{title}\n\n{body}"
    return title or body


def load_texts(pmids: Sequence[str], max_workers: int = 8) -> dict[str, str]:
    """Return ``{pmid: source text}``, downloading anything not yet cached.

    Parameters
    ----------
    pmids :
        PubMed IDs to load.
    max_workers :
        Maximum number of worker threads for PMC S3 full-text fetch.
        Default: 8.

    Returns
    -------
    :
        Source text per PMID. PMIDs with no text are left out.
    """
    pmid_strs = [str(pmid) for pmid in pmids]
    cache = download_texts_bulk(pmid_strs, max_workers=max_workers)
    texts = {}
    for pmid in pmid_strs:
        text = _join_title_body(cache.get(pmid) or {})
        if text:
            texts[pmid] = text
    return texts
