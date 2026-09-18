"""
Unit tests for trialsynth.base.ground's DrugBank-priority drug grounder.
"""

import gilda

from trialsynth.base.ground import get_drugbank_grounder, drugbank_ground


def test_lepirudin_and_dornase_alfa_ground_correctly():
    """Both drugs ground to their correct DrugBank ID (each also outscores its only CHEBI/MESH match)."""
    cases = {
        "Lepirudin": "DB00001",
        "Dornase alfa": "DB00003",
    }
    for name, expected_id in cases.items():
        result = drugbank_ground(name)
        assert result is not None, f"{name} did not ground at all"
        assert result["db"] == "DRUGBANK", f"{name} grounded to {result['db']}, not DRUGBANK"
        assert result["id"] == expected_id, f"{name} grounded to {result['id']}, expected {expected_id}"


def test_drugbank_priority_overrides_existing_grounding():
    """DrugBank should win even when CHEBI already has a curated match for the same drug."""
    result = drugbank_ground("Tamoxifen")
    assert result is not None
    assert result["db"] == "DRUGBANK", f"Expected DRUGBANK to win under priority ordering, got {result['db']}"
    assert result["id"] == "DB00675"


def test_conflicting_crossref_id_still_grounds():
    """Grounding by name still works even if the DrugBank ID has a messy CHEBI cross-reference."""
    result = drugbank_ground("Leuprolide")
    assert result is not None
    assert result["db"] == "DRUGBANK"
    assert result["id"] == "DB00007"


def test_global_grounder_not_mutated():
    """Building this grounder shouldn't touch Gilda's actual global grounder."""
    default_entries_before = gilda.get_grounder().entries
    get_drugbank_grounder()
    default_entries_after = gilda.get_grounder().entries
    assert default_entries_before is default_entries_after
    default_results = gilda.get_grounder().ground("Ticagrelor")
    drugbank_hit = any(r.term.db == "DRUGBANK" for r in default_results)
    assert not drugbank_hit, "Global default grounder was mutated with DrugBank terms"


def test_non_drug_terms_unaffected():
    """Placebo and Usual care shouldn't ever match a DrugBank entry."""
    for term in ["Placebo", "Usual care"]:
        result = drugbank_ground(term)
        if result is not None:
            assert result["db"] != "DRUGBANK", f"{term!r} incorrectly grounded to DrugBank"
