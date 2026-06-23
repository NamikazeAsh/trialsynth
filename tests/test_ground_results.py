"""
Unit tests for trialsynth.base.extract.ground_results.

Real Gilda calls are used for grounding tests. Simple patches (return_value=[])
are used only for testing our own logic paths (filters, fallbacks) where we
need to control whether Gilda finds anything, without building fake objects.
"""

from unittest.mock import patch

from trialsynth.base.extract.ground_results import (
    _annotate_fallback,
    _normalize_ae_text,
    get_gilda_grounding,
    ground_adverse_event,
    ground_marker,
    ANNOTATE_STOPLIST,
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

def test_get_gilda_grounding_empty_input_returns_none():
    """Empty string is rejected before calling Gilda."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground") as mock_ground:
        result = get_gilda_grounding("")
    mock_ground.assert_not_called()
    assert result is None


def test_get_gilda_grounding_no_results_returns_none():
    """Returns None when Gilda finds no matches."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground", return_value=[]):
        result = get_gilda_grounding("unknownterm")
    assert result is None


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
    with patch("trialsynth.base.extract.ground_results.gilda.ground") as mock_ground:
        result = ground_marker("t")
    mock_ground.assert_not_called()
    assert result["groundings"] == []


def test_ground_marker_returns_grounding_on_hit():
    """Valid symbol produces a grounding entry with correct structure."""
    result = ground_marker("BRCA1")
    assert len(result["groundings"]) == 1
    assert result["groundings"][0]["info"]["db"] == "HGNC"
    assert result["groundings"][0]["info"]["id"] == "1100"


def test_ground_marker_annotate_fallback_fires_when_direct_fails():
    """When direct grounding returns nothing and evidence_text exists, annotate fallback is used."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground", return_value=[]):
        result = ground_marker({"text": "BRCA1", "evidence_text": "Patients with BRCA1 mutations."})
    assert len(result["groundings"]) == 1
    assert result["groundings"][0]["info"]["source"] == "annotate_fallback"


def test_ground_marker_no_fallback_without_evidence_text():
    """Annotate fallback is not attempted when evidence_text is empty."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground", return_value=[]), \
         patch("trialsynth.base.extract.ground_results.gilda.annotate") as mock_ann:
        result = ground_marker({"text": "BRCA1", "evidence_text": ""})
    mock_ann.assert_not_called()
    assert result["groundings"] == []


def test_ground_marker_variant_extracted():
    """HGVS-style variant notation is parsed from marker text."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground", return_value=[]):
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
    """Non-HGNC hits from annotate are excluded."""
    with patch("trialsynth.base.extract.ground_results.gilda.annotate", return_value=[]):
        hits = _annotate_fallback("Patient reported nausea.")
    assert hits == []


def test_annotate_fallback_stoplist_filtered():
    """Symbols in ANNOTATE_STOPLIST are excluded even if Gilda annotates them."""
    stopword = next(iter(ANNOTATE_STOPLIST))
    with patch("trialsynth.base.extract.ground_results.gilda.annotate", return_value=[]):
        hits = _annotate_fallback(f"The {stopword} value was high.")
    assert hits == []


def test_annotate_fallback_short_match_filtered():
    """Matches shorter than 4 characters are excluded."""
    with patch("trialsynth.base.extract.ground_results.gilda.annotate", return_value=[]):
        hits = _annotate_fallback("The AB level was elevated.")
    assert hits == []


def test_annotate_fallback_empty_input_returns_empty():
    """Empty input returns empty list without calling Gilda."""
    with patch("trialsynth.base.extract.ground_results.gilda.annotate") as mock_ann:
        hits = _annotate_fallback("")
    mock_ann.assert_not_called()
    assert hits == []


# ---------------------------------------------------------------------------
# ground_adverse_event
# ---------------------------------------------------------------------------

def test_ground_adverse_event_below_min_len_returns_none():
    """Input shorter than min_len is rejected before any Gilda call."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground") as mock_ground:
        result = ground_adverse_event("AE", min_len=4)
    mock_ground.assert_not_called()
    assert result is None


def test_ground_adverse_event_returns_ground_source():
    """When gilda.ground succeeds, source is set to 'ground'."""
    result = ground_adverse_event("nausea", min_len=4)
    assert result is not None
    assert result["source"] == "ground"
    assert result["db"] == "MESH"


def test_ground_adverse_event_annotate_fallback_fires():
    """When gilda.ground fails, annotate fallback is tried and source is 'annotate'."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground", return_value=[]):
        result = ground_adverse_event("Grade 3 leukopenia", min_len=4)
    assert result is not None
    assert result["source"] == "annotate"


def test_ground_adverse_event_non_ae_namespace_not_returned():
    """Hits outside AE_NAMESPACES from annotate are not returned."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground", return_value=[]), \
         patch("trialsynth.base.extract.ground_results.gilda.annotate", return_value=[]):
        result = ground_adverse_event("BRCA1 mutation", min_len=4)
    assert result is None


def test_ground_adverse_event_empty_string_returns_none():
    """Empty string is rejected by min_len check."""
    with patch("trialsynth.base.extract.ground_results.gilda.ground") as mock_ground:
        result = ground_adverse_event("", min_len=4)
    mock_ground.assert_not_called()
    assert result is None


# ---------------------------------------------------------------------------
# CRITERIA_NAMESPACES filter (via get_gilda_grounding)
# ---------------------------------------------------------------------------

def test_criteria_chebi_hit_dropped():
    """A CHEBI grounding (e.g. aspirin) is dropped by the CRITERIA_NAMESPACES filter in ground_json."""
    result = get_gilda_grounding("aspirin", sources=["HP", "DOID", "MESH", "EFO", "CHEBI"])
    assert result is not None
    assert result["db"] == "CHEBI"
    from trialsynth.base.extract.ground_results import CRITERIA_NAMESPACES
    assert result["db"] not in CRITERIA_NAMESPACES
