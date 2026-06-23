"""
Unit tests for trialsynth.base.extract.ground_results.
"""

from trialsynth.base.extract.ground_results import (
    _annotate_fallback,
    _normalize_ae_text,
    get_gilda_grounding,
    ground_adverse_event,
    ground_marker,
)


# ---------------------------------------------------------------------------
# _normalize_ae_text  (pure string, no Gilda)
# ---------------------------------------------------------------------------

def test_normalize_ae_text_collapses_spaces():
    """Multiple internal spaces are collapsed to a single space."""
    assert _normalize_ae_text("nausea   and   vomiting") == "nausea and vomiting"


def test_normalize_ae_text_strips_whitespace():
    """Leading and trailing whitespace is stripped."""
    assert _normalize_ae_text("  fatigue  ") == "fatigue"


def test_normalize_ae_text_clean_input_unchanged():
    """Already clean input is returned as-is."""
    assert _normalize_ae_text("nausea") == "nausea"


def test_normalize_ae_text_empty_string():
    """Empty string returns empty string without error."""
    assert _normalize_ae_text("") == ""


# ---------------------------------------------------------------------------
# get_gilda_grounding
# ---------------------------------------------------------------------------


def test_get_gilda_grounding_brca1():
    """BRCA1 grounds to HGNC:1100 with correct structure."""
    result = get_gilda_grounding("BRCA1", sources=["HGNC"])
    assert result is not None
    assert result["db"] == "HGNC"
    assert result["id"] == "1100"
    assert result["entry_name"] == "BRCA1"
    assert isinstance(result["score"], float)


# ---------------------------------------------------------------------------
# ground_marker
# ---------------------------------------------------------------------------

def test_ground_marker_single_char_filtered():
    """Single-character symbols are skipped before any Gilda call."""
    result = ground_marker("t")
    assert result["groundings"] == []


def test_ground_marker_returns_grounding_on_hit():
    """Valid gene symbol returns a correctly structured grounding entry."""
    result = ground_marker("BRCA1")
    assert len(result["groundings"]) == 1
    assert result["groundings"][0]["info"]["db"] == "HGNC"
    assert result["groundings"][0]["info"]["id"] == "1100"


def test_ground_marker_annotate_fallback_fires_when_direct_fails():
    """Symbol that cannot be grounded triggers annotate fallback on evidence_text."""
    result = ground_marker({"text": "ZZZZZ12345", "evidence_text": "BRCA1 is expressed."})
    hgnc_hits = [g for g in result["groundings"] if g["info"]["db"] == "HGNC"]
    assert len(hgnc_hits) >= 1
    assert hgnc_hits[0]["info"]["source"] == "annotate_fallback"



def test_ground_marker_variant_extracted():
    """HGVS-style variant notation is parsed from marker text."""
    result = ground_marker("BRAF V600E")
    assert result["variant"] == "V600E"


def test_ground_marker_no_variant_returns_none():
    """Marker with no variant notation has variant field set to None."""
    result = ground_marker("BRCA1")
    assert result["variant"] is None


# ---------------------------------------------------------------------------
# _annotate_fallback
# ---------------------------------------------------------------------------

def test_annotate_fallback_returns_hgnc_hits():
    """BRCA1 embedded in a sentence is found and returned as an HGNC hit."""
    hits = _annotate_fallback("Patients positive for BRCA1 were excluded.")
    hgnc_hits = [h for h in hits if h["info"]["db"] == "HGNC"]
    assert len(hgnc_hits) >= 1
    assert hgnc_hits[0]["info"]["id"] == "1100"


def test_annotate_fallback_non_hgnc_filtered():
    """MESH/HP annotations from Gilda are dropped by our HGNC-only filter."""
    hits = _annotate_fallback("Patient reported nausea.")
    nausea_hits = [h for h in hits if h["symbol"].lower() == "nausea"]
    assert nausea_hits == []


def test_annotate_fallback_stoplist_filtered():
    """'FISH' is 4 chars so length filter passes, but it is in ANNOTATE_STOPLIST so it is filtered out."""
    hits = _annotate_fallback("The FISH result was positive.")
    fish_hits = [h for h in hits if h["symbol"].upper() == "FISH"]
    assert fish_hits == []


def test_annotate_fallback_short_match_filtered():
    """'KIT' annotates as HGNC with span length 3, which our length filter drops."""
    hits = _annotate_fallback("The KIT expression was elevated.")
    kit_hits = [h for h in hits if h["symbol"].upper() == "KIT"]
    assert kit_hits == []



# ---------------------------------------------------------------------------
# ground_adverse_event
# ---------------------------------------------------------------------------

def test_ground_adverse_event_below_min_len_returns_none():
    """Input shorter than min_len is rejected before any Gilda call."""
    assert ground_adverse_event("AE", min_len=4) is None


def test_ground_adverse_event_returns_ground_source():
    """Known AE term grounds directly and returns source set to 'ground'."""
    result = ground_adverse_event("nausea", min_len=4)
    assert result is not None
    assert result["source"] == "ground"
    assert result["db"] == "MESH"


def test_ground_adverse_event_annotate_fallback_fires():
    """Multi-word phrase does not ground directly; annotate finds 'leukopenia' as an AE term."""
    result = ground_adverse_event("Grade 3 leukopenia", min_len=4)
    assert result is not None
    assert result["source"] == "annotate"


def test_ground_adverse_event_non_ae_namespace_not_returned():
    """Gene symbol only grounds to HGNC which is not in AE_NAMESPACES, so result is None."""
    result = ground_adverse_event("BRCA1", min_len=4)
    assert result is None


