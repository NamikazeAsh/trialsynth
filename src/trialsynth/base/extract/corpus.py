"""What varies between the record collections the extract pipeline runs over.

A corpus is one collection of source records -- PubMed articles or
ClinicalTrials.gov registry records -- together with everything the
pipeline needs to know that differs between them: how to resolve the record
IDs, how to load each record's text, which prompt and result schema to send to
Bedrock, how a record's text is framed into the model input and recovered from
it afterwards, and where grounded results live.

Those decisions travel together. The prompt is written against a result schema,
and the framing written at input-build time is only readable by the matching
un-framing at process time, so a corpus carries all of them as one value rather
than leaving them to be re-paired correctly at each call site.

The stages in between are corpus-agnostic: ``extract_bedrock`` moves JSONL
through Bedrock without knowing what a record is, and grounding works on the
extraction dict alone.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from trialsynth.base.extract.ctgov_text import (
    load_texts as load_registry_texts,
    resolve_nct_ids,
)
from trialsynth.base.extract.extract_util import (
    get_trial_pmids,
    load_texts as load_pmid_texts,
)
from trialsynth.base.extract.paths import (
    RESULTS_GROUNDED_CTGOV_DIR,
    RESULTS_GROUNDED_DIR,
)
from trialsynth.base.extract.resources import (
    CTGOV_PROMPT,
    CTGOV_RESULT_SCHEMA_ANCHOR,
    PROMPT,
    TRIAL_RESULT_SCHEMA_ANCHOR,
)


@dataclass(frozen=True)
class Corpus:
    """One collection of source records and how the pipeline handles it.

    Attributes
    ----------
    name :
        Value accepted by the ``--corpus`` CLI option.
    id_field :
        Key the record ID is written to on the grounded extraction. Corpora use
        different ID namespaces, so this is not shared: a PMID and an NCT ID
        must not land in the same field.
    results_dir :
        Default directory for grounded ``<record_id>.json`` output.
    prompt :
        System prompt sent to the model.
    schema :
        JSON schema constraining the model's output.
    resolve_ids :
        Returns every record ID in the corpus, used when the CLI is given no
        explicit IDs.
    load_texts :
        Returns ``{record_id: text}`` for the given IDs, fetching or rendering
        any text not already cached. IDs with no text are left out.
    frame :
        Builds the user message content from a record ID and its text.
    unframe :
        Recovers the text from the framed content. Inverse of ``frame``.
    """

    name: str
    id_field: str
    results_dir: Path
    prompt: str
    schema: dict
    resolve_ids: Callable[[], list[str]]
    load_texts: Callable[[Sequence[str], int], dict[str, str]]
    frame: Callable[[str, str], str]
    unframe: Callable[[str], str]


# Kept adjacent to the matching un-framing below: the format string and the
# pattern that strips it back off are only correct as a pair.
_PUBMED_PREFIX_RE = re.compile(r"^Text \(PMID \d+\):\s*")


def _pubmed_frame(record_id: str, text: str) -> str:
    return f"Text (PMID {record_id}): {text}"


def _pubmed_unframe(content: str) -> str:
    return _PUBMED_PREFIX_RE.sub("", content, count=1).strip()


#: PubMed articles, keyed by PMID.
PUBMED = Corpus(
    name="pubmed",
    id_field="pmid",
    results_dir=RESULTS_GROUNDED_DIR.base,
    prompt=PROMPT,
    schema=TRIAL_RESULT_SCHEMA_ANCHOR,
    resolve_ids=get_trial_pmids,
    load_texts=load_pmid_texts,
    frame=_pubmed_frame,
    unframe=_pubmed_unframe,
)

_CTGOV_PREFIX_RE = re.compile(r"^Registry record \(NCT\d+\):\s*")


def _ctgov_frame(record_id: str, text: str) -> str:
    return f"Registry record ({record_id}): {text}"


def _ctgov_unframe(content: str) -> str:
    return _CTGOV_PREFIX_RE.sub("", content, count=1).strip()


#: ClinicalTrials.gov registry records, keyed by NCT ID.
CTGOV = Corpus(
    name="ctgov",
    id_field="nct_id",
    results_dir=RESULTS_GROUNDED_CTGOV_DIR.base,
    prompt=CTGOV_PROMPT,
    schema=CTGOV_RESULT_SCHEMA_ANCHOR,
    resolve_ids=resolve_nct_ids,
    load_texts=load_registry_texts,
    frame=_ctgov_frame,
    unframe=_ctgov_unframe,
)

CORPORA: dict[str, Corpus] = {PUBMED.name: PUBMED, CTGOV.name: CTGOV}

DEFAULT_CORPUS = PUBMED.name


def get_corpus(name: str) -> Corpus:
    """Look up a corpus by name.

    Parameters
    ----------
    name :
        Corpus name, as accepted by ``--corpus``.

    Returns
    -------
    :
        The named corpus.

    Raises
    ------
    KeyError
        If no corpus goes by that name.
    """
    try:
        return CORPORA[name]
    except KeyError:
        known = ", ".join(sorted(CORPORA))
        raise KeyError(f"Unknown corpus {name!r}. Known corpora: {known}") from None
