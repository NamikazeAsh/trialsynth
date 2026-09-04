"""Ground genetic markers, criteria, and adverse events with Gilda.

Attaches HGNC groundings for genetic markers, HP/DOID/MESH/EFO groundings for
inclusion/exclusion criteria, and HP/DOID/MESH/EFO/OAE groundings for adverse
events.
"""

import re
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import gilda
import gilda.ner
from gilda import make_grounder
from gilda.process import normalize as _gilda_normalize
from gilda.term import Term

from trialsynth.base.extract.paths import RESOURCES_DIR

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


def ground_extraction(data: dict, *, ae_min_len: int) -> dict:
    """Ground genetic markers, criteria, and adverse events in an extraction.

    Parameters
    ----------
    data :
        Resolved extraction dict.
    ae_min_len :
        Minimum AE event-name length for grounding.

    Returns
    -------
    :
        The same dict, mutated in place.
    """
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

    for arm in data.get("arms", []):
        for ae in arm.get("adverse_events", []):
            ae["grounding"] = ground_adverse_event(
                ae.get("event_name", ""),
                min_len=ae_min_len,
            )

    return data
