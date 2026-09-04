"""
Todo: write file docstring
"""

import csv
import gzip
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
from indra.literature.pmc_client import id_lookup, get_text_s3
from indra.literature.pubmed_client import get_abstract, get_metadata_for_all_ids

from trialsynth.ctgov.config import CTConfig
from trialsynth.base.extract.paths import CONTENT_TXT_DIR


logger = logging.getLogger(__name__)


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
