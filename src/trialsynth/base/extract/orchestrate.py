"""
Orchestrator: intersection PMIDs -> text download -> anchor extraction ->
grounded JSONs.

Usage:
    python orchestrate.py           # default: 1000 PMIDs
    python orchestrate.py --limit 500

Two-stage checkpointing:
- Text download: skipped if <pmid>.txt already exists in txt_archive
- Extraction:    skipped if <pmid>.json already exists in output_dir

Re-running safely resumes from wherever it left off.
"""

import csv
import gzip
import logging
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
from openai import OpenAI
from indra.literature.pmc_client import id_lookup, get_text_s3
from indra.literature.pubmed_client import get_abstract, get_metadata_for_all_ids

from trialsynth.ctgov.config import CTConfig
from trialsynth.base.extract.extract import process_pmid
from trialsynth.base.extract.paths import RESULTS_RAW_DIR, RESULTS_DIR, \
    CONTENT_TXT_DIR


logger = logging.getLogger('trialsynth.base.extract.orchestrate')


def get_intersection_pmids(limit: int = None) -> list[str]:
    """Return PMIDs defined from the output of

    Parameters
    ----------
    limit :
        Optional limit on the number of PMIDs to return. If None, return all.

    Returns
    -------
    :
        List of PMIDs that are in both the registry result links and the PubMed
        scan links.
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
        intersection = {
            row[1] for row in reader if row[1]
        }

    return sorted(intersection)[:limit] if limit else sorted(intersection)


def download_texts(pmids: list[str]):
    # Download texts for PMIDs sequentially. Useful if the host environment has
    # limited concurrency capabilities.
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


def _overlay_fulltext_s3(pmid: str, pmcid: str, abstract) -> str:
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
        # _write_pmid_text(pmid, abstract)
        CONTENT_TXT_DIR.join(name=f"{pmid}.txt").write_text(
            abstract, encoding="utf-8"
        )
        return "abs"

    tqdm.tqdm.write(f"{pmid} - NO CONTENT")
    return "none"


def download_texts_bulk(pmids: list[str], max_workers: int = 8):
    """Download texts via a bulk PubMed metadata fetch plus optional S3 overlay.

    Fetches PubMed XML in batches of 200 so abstracts and PMCIDs come back in
    one pass. Abstracts are written immediately for PMIDs that have no PMCID.
    PMIDs with a PMCID are then fetched from the PMC OA S3 bucket in parallel:
    an S3 hit writes full text; a miss writes the abstract already returned by
    the bulk fetch. PMIDs missing from the bulk response fall back to
    per-PMID download.

    Parameters
    ----------
    pmids :
        List of PMIDs to download text for.
    max_workers :
        Maximum number of worker threads for S3 full-text overlay and
        per-PMID fallback. Default: 8.
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
                executor.submit(_overlay_fulltext_s3, pmid, pmcid, abstract)
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


def run_extraction(pmids: list[str]):
    logger.info(f"Running anchor extraction on {len(pmids)} PMIDs...")
    client = OpenAI()
    stats = []

    for pmid in pmids:
        row = process_pmid(pmid, client, CONTENT_TXT_DIR.base, RESULTS_RAW_DIR.base)
        stats.append(row)
        if row["status"] == "ok":
            logger.info(f"  {pmid} extracted ({row['output_tokens']} output tokens)")
        elif row["status"] == "skipped":
            logger.info(f"  {pmid} skipped (already exists)")
        else:
            logger.warning(f"  {pmid} -> {row['status']}")

    statuses = Counter(r["status"] for r in stats)

    n_errors = (
        len(statuses) - statuses['ok'] - statuses['skipped'] - statuses['missing_text']
    )
    logger.info(
        f"Extraction complete: {statuses['ok']} new, {statuses['skipped']} skipped, "
        f"{statuses['missing_text']} missing text, {n_errors} errors"
    )
    if statuses.get('completed'):
        total_out = sum(r["output_tokens"] for r in stats if r["status"] == "completed")
        logger.info(f"Avg output tokens/paper: {total_out // statuses['completed']}")

    csv_path = RESULTS_DIR.join(name="extraction_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pmid", "status", "input_tokens", "output_tokens"]
        )
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"Stats saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max PMIDs to process (default: 1000)")
    args = parser.parse_args()

    pmids_path = RESULTS_DIR.join(name="intersection_pmids.csv")

    if pmids_path.exists():
        with open(pmids_path, "r") as f:
            saved = [line.strip() for line in f if line.strip()]
    else:
        saved = []

    if len(saved) >= args.limit:
        pmids = saved[:args.limit]
        logger.info(f"Loaded {len(pmids)} PMIDs from {pmids_path}")
    else:
        pmids = get_intersection_pmids(limit=args.limit)
        with open(pmids_path, "w") as f:
            for pmid in pmids:
                f.write(pmid + "\n")
        logger.info(f"Saved {len(pmids)} PMIDs to {pmids_path}")

    download_texts(pmids)
    run_extraction(pmids)


if __name__ == "__main__":
    main()
