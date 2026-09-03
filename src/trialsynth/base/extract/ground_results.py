"""
Ground genetic markers and clinical criteria in extracted JSON files using local Gilda.

Reads raw anchor JSONs and writes grounded versions with HGNC groundings for
genetic markers, HP/DOID/MESH/EFO groundings for inclusion/exclusion criteria, and
HP/DOID/MESH/EFO/OAE groundings for adverse events.

Usage:
    python ground_results.py --input-dir <raw_dir> --output-dir <grounded_dir>
"""

import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import gilda
import gilda.ner
from gilda import make_grounder
from gilda.process import normalize as _gilda_normalize
from gilda.term import Term

from trialsynth.base.extract.paths import RESULTS_RAW_DIR, RESULTS_GROUNDED_DIR, RESOURCES_DIR

logger = logging.getLogger('trialsynth.base.extract.ground_results')
AE_NAMESPACES = ["HP", "DOID", "MESH", "EFO", "OAE"]
AE_SHORT_TOKEN_MIN_LEN_DEFAULT = 4

_OAE_OWL_URL = "http://purl.obolibrary.org/obo/oae.owl"

_OWL_NS = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


def _oae_term(text: str, status: str, oae_id: str, entry_name: str) -> Term:
    return Term(
        norm_text=_gilda_normalize(text),
        text=text,
        db="OAE",
        id=oae_id,
        entry_name=entry_name,
        status=status,
        source="OAE",
    )


def _oae_terms() -> List[Term]:
    """Parse oae.owl (downloading via pystow if not cached) into OAE Gilda Terms."""
    oae_owl = RESOURCES_DIR.ensure(url=_OAE_OWL_URL, name="oae.owl")
    root = ET.parse(oae_owl).getroot()
    terms = []
    for cls in root.findall("owl:Class", _OWL_NS):
        # rdf:about holds the term IRI; ElementTree needs the full namespace, not the "rdf:" prefix
        iri = cls.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
        m = re.search(r"OAE_(\d+)", iri)
        if not m:
            continue
        oae_id = m.group(1)
        label_el = cls.find("rdfs:label", _OWL_NS)
        if label_el is None or not label_el.text:
            continue
        label = label_el.text.strip()
        entry_name = re.sub(r"\s+ae$", "", label, flags=re.IGNORECASE).strip()
        terms.append(_oae_term(label, "name", oae_id, entry_name))
        # OAE labels end in " AE" (e.g. "diarrhea AE"); also register the stripped form so plain queries match
        if label.lower().endswith(" ae"):
            terms.append(_oae_term(entry_name, "synonym", oae_id, entry_name))
    return terms


_grounder = None


def _get_grounder():
    """Build once and return a trialsynth grounder: Gilda's default terms plus OAE terms."""
    global _grounder
    if _grounder is None:
        entries = {norm: list(terms) for norm, terms in gilda.get_grounder().entries.items()}
        for term in _oae_terms():
            entries.setdefault(term.norm_text, []).append(term)
        _grounder = make_grounder(entries)
    return _grounder


def get_gilda_grounding(text: str, sources: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Ground text using local Gilda, returning the top hit or None."""
    if not text:
        return None
    results = _get_grounder().ground(text, namespaces=sources)
    if results:
        top = results[0].term
        return {
            "entry_name": top.entry_name,
            "db": top.db,
            "id": top.id,
            "score": results[0].score
        }
    return None


CRITERIA_NAMESPACES = {"HP", "DOID", "MESH", "EFO"}

ANNOTATE_STOPLIST = {
    'FISH', 'IV', 'WT', 'HR', 'CI', 'OR', 'RR', 'OS', 'PFS', 'CR', 'PR',
    'SD', 'PD', 'CT', 'MRI', 'PCR', 'IHC', 'AE', 'SAE', 'PS', 'ECOG',
    'HET', 'SET', 'CAT', 'ACT', 'ALL', 'CML', 'AML', 'NHL', 'DNA', 'RNA',
    'ATP', 'GTP', 'II', 'III', 'BL', 'large', 'real'
}


def _annotate_fallback(evidence_text: str) -> List[Dict[str, Any]]:
    """Fallback: annotate the full sentence with the trialsynth grounder, return HGNC hits above min length not in stoplist."""
    if not evidence_text:
        return []
    hits = []
    for r in gilda.ner.annotate(evidence_text, grounder=_get_grounder()):
        if not r.matches:
            continue
        top = r.matches[0]
        if top.term.db != 'HGNC':
            continue
        if len(r.text) < 4:
            continue
        if r.text.upper() in ANNOTATE_STOPLIST or r.text in ANNOTATE_STOPLIST:
            continue
        hits.append({"symbol": r.text, "info": {
            "entry_name": top.term.entry_name,
            "db": top.term.db,
            "id": top.term.id,
            "score": top.score,
            "source": "annotate_fallback"
        }})
    return hits


def ground_marker(item: Any) -> Dict[str, Any]:
    """Ground a genetic marker item, returning HGNC groundings and variant if found."""
    if isinstance(item, str):
        text = item
        evidence_text = ""
    else:
        text = item.get("text") or ""
        evidence_text = item.get("evidence_text") or ""

    raw = text.strip()
    parts = re.split(r'[:/-]', raw)
    symbols = [re.findall(r'[A-Z0-9]+', p, re.IGNORECASE) for p in parts]
    symbols = [s[0] for s in symbols if s]

    groundings = []
    for s in symbols:
        if len(s) < 2:
            continue
        g = get_gilda_grounding(s, sources=["HGNC", "UP", "MESH"])
        if g:
            groundings.append({"symbol": s, "info": g})

    if not groundings and evidence_text:
        fallback_hits = _annotate_fallback(evidence_text)
        groundings.extend(fallback_hits)

    variant = None
    var_match = re.search(r"(p\.[A-Za-z0-9]+|[A-Z][0-9]{2,4}[A-Z])", raw, re.IGNORECASE)
    if var_match:
        variant = var_match.group(1)

    return {
        "text": text,
        "evidence_text": evidence_text,
        "groundings": groundings,
        "variant": variant
    }


def _normalize_ae_text(text: str) -> str:
    """Normalize whitespace in AE event name text."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _annotate_fallback_ae(text: str) -> Optional[Dict[str, Any]]:
    """Fallback: annotate AE text with the trialsynth grounder, return top hit filtered to AE_NAMESPACES."""
    if not text:
        return None
    for r in gilda.ner.annotate(text, grounder=_get_grounder(), namespaces=AE_NAMESPACES):
        if not r.matches:
            continue
        top = r.matches[0]
        if top.term.db not in AE_NAMESPACES:
            continue
        if len(r.text) < 4:
            continue
        return {
            "db": top.term.db,
            "id": top.term.id,
            "name": top.term.entry_name,
            "score": top.score,
            "source": "annotate",
        }
    return None


def ground_adverse_event(
    event_name: str,
    *,
    min_len: int,
) -> Optional[Dict[str, Any]]:
    """Ground an adverse event name with the trialsynth grounder, falling back to annotate."""
    clean = _normalize_ae_text(event_name)
    if len(clean) < min_len:
        return None
    top = get_gilda_grounding(clean, sources=AE_NAMESPACES)
    if top:
        return {
            "db": top["db"],
            "id": top["id"],
            "name": top["entry_name"],
            "score": top["score"],
            "source": "ground",
        }
    return _annotate_fallback_ae(clean)


def ground_json(
    input_path: Path,
    output_path: Path,
    *,
    ae_min_len: int,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Ground all genetic markers, criteria, and AEs in a single JSON file and write output.

    Genetic markers are read from ``genetic.markers``. Each grounded object keeps
    its ``role`` (defaulting to ``"other"`` if missing). Inclusion-role markers
    are also written to ``genetic.grounded_inclusion`` for CoGEx compatibility;
    the full list is written to ``genetic.grounded_markers``.

    Parameters
    ----------
    input_path :
        Path to the resolved (anchor) JSON file.
    output_path :
        Path to write the grounded JSON file.
    ae_min_len :
        Minimum AE event-name length for grounding.

    Returns
    -------
    :
        AE total count, AE grounded count, and AE review rows.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "results" in data:
        data["results"] = [
            {"text": r, "evidence_text": ""} if isinstance(r, str) else r
            for r in data["results"]
        ]

    grounded_markers = []
    for item in data.get('genetic', {}).get('markers', []):
        grounded = ground_marker(item)
        role = (item.get('role') if isinstance(item, dict) else None) or 'other'
        grounded['role'] = role
        grounded_markers.append(grounded)
    genetic = data.setdefault('genetic', {})
    genetic['grounded_markers'] = grounded_markers
    genetic['grounded_inclusion'] = [
        m for m in grounded_markers if m['role'] == 'inclusion'
    ]

    grounded_inclusion = []
    for item in data.get('inclusion_criteria', []):
        if isinstance(item, str):
            text = item
            ev = ""
        else:
            text = item.get("text", "")
            ev = item.get("evidence_text", "")
        match = get_gilda_grounding(text, sources=["HP", "DOID", "MESH", "EFO"])
        if match and match.get("db") not in CRITERIA_NAMESPACES:
            match = None
        grounded_inclusion.append({"text": text, "evidence_text": ev, "grounding": match})
    data['grounded_inclusion_criteria'] = grounded_inclusion

    grounded_exclusion = []
    for item in data.get('exclusion_criteria', []):
        if isinstance(item, str):
            text = item
            ev = ""
        else:
            text = item.get("text", "")
            ev = item.get("evidence_text", "")
        match = get_gilda_grounding(text, sources=["HP", "DOID", "MESH", "EFO"])
        if match and match.get("db") not in CRITERIA_NAMESPACES:
            match = None
        grounded_exclusion.append({"text": text, "evidence_text": ev, "grounding": match})
    data['grounded_exclusion_criteria'] = grounded_exclusion
    ae_total = 0
    ae_grounded = 0
    ae_review_rows: List[Dict[str, Any]] = []
    for arm in data.get("arms", []):
        arm_name = arm.get("arm_name", "")
        for ae in arm.get("adverse_events", []):
            ae_total += 1
            event_name = ae.get("event_name", "")
            grounding = ground_adverse_event(
                event_name,
                min_len=ae_min_len,
            )
            ae["grounding"] = grounding
            if grounding:
                ae_grounded += 1
            ae_review_rows.append({
                "pmid": str(data.get("pmid", input_path.stem)),
                "arm_name": arm_name,
                "event_name": event_name,
                "grounded_db": (grounding or {}).get("db", ""),
                "grounded_id": (grounding or {}).get("id", ""),
                "grounded_name": (grounding or {}).get("name", ""),
                "score": (grounding or {}).get("score", ""),
                "evidence_text": ae.get("source_sentence", ""),
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return ae_total, ae_grounded, ae_review_rows


def main():
    """CLI entry point: ground all JSON files in input-dir and write to output-dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=RESULTS_RAW_DIR.base)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_GROUNDED_DIR.base)
    parser.add_argument("--pmid-list", type=Path, default=None,
                        help="Optional file with one PMID per line for pilot/smoke subsets.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional max number of JSON files to process.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used when sampling --max-files without --pmid-list.")
    parser.add_argument("--ae-min-len", type=int, default=AE_SHORT_TOKEN_MIN_LEN_DEFAULT)
    parser.add_argument("--ae-review-csv", type=Path, default=None,
                        help="Optional CSV path for AE grounding review table.")
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Optional JSON path for pilot/smoke AE grounding metrics.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(args.input_dir.glob("*.json"))
    if args.pmid_list:
        pmids = {
            line.strip() for line in args.pmid_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        json_files = [p for p in json_files if p.stem in pmids]
    elif args.max_files and len(json_files) > args.max_files:
        random.seed(args.seed)
        json_files = random.sample(json_files, args.max_files)

    logger.info(f"Grounding {len(json_files)} files...")
    ae_total = 0
    ae_grounded = 0
    ae_rows: List[Dict[str, Any]] = []
    for i, jf in enumerate(json_files):
        out = args.output_dir / jf.name
        if out.exists():
            continue
        file_total, file_grounded, file_rows = ground_json(
            jf,
            out,
            ae_min_len=args.ae_min_len,
        )
        ae_total += file_total
        ae_grounded += file_grounded
        ae_rows.extend(file_rows)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(json_files)} done")
    if args.ae_review_csv:
        args.ae_review_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.ae_review_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "pmid", "arm_name", "event_name", "grounded_db", "grounded_id",
                "grounded_name", "score", "evidence_text",
            ])
            writer.writeheader()
            writer.writerows(ae_rows)
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        grounded_rows = [r for r in ae_rows if r["grounded_id"]]
        namespace_counts: Dict[str, int] = {}
        empty_event_name = 0
        duplicate_groundings = 0
        seen = set()
        for row in ae_rows:
            if not _normalize_ae_text(row["event_name"]):
                empty_event_name += 1
            key = (row["pmid"], row["arm_name"], row["event_name"], row["grounded_db"], row["grounded_id"])
            if row["grounded_id"] and key in seen:
                duplicate_groundings += 1
            seen.add(key)
        for row in grounded_rows:
            namespace_counts[row["grounded_db"]] = namespace_counts.get(row["grounded_db"], 0) + 1
        metrics = {
            "files_processed": len(json_files),
            "ae_total": ae_total,
            "ae_grounded": ae_grounded,
            "ae_coverage": (ae_grounded / ae_total) if ae_total else 0.0,
            "namespace_distribution": namespace_counts,
            "empty_event_name_rate": (empty_event_name / ae_total) if ae_total else 0.0,
            "duplicate_grounding_rate": (duplicate_groundings / ae_total) if ae_total else 0.0,
            "ae_min_len": args.ae_min_len,
            "ae_namespaces": AE_NAMESPACES,
        }
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
