"""Render ClinicalTrials.gov registry records as extraction source text.

The text comes from the pickled :class:`~trialsynth.base.models.Trial` dump the
ctgov pipeline already writes, so registry fields are mapped in exactly one
place -- ``CTFetcher`` -- rather than re-walked from raw API JSON here. That
also means no per-record network fetch: a trial missing from the dump is
treated as having no record, and re-running the ctgov fetch widens coverage.

What a registry record can support is protocol-as-planned. ``API_FIELDS`` in
the config carries no results or outcome-measures module, only outcome measure
*names*, so there are no metric values and no adverse events to extract --
which is why the ctgov result schema asks for neither.

A field is rendered only if the result schema can draw on it -- criteria
source, or somewhere a biomarker gets named. Everything else (phase, design,
enrollment, status, locations, the age/sex/healthy-volunteer fields) costs
tokens and hands a hallucinated criterion a real line to anchor to.

ponytail: protocol-only ceiling. To extract reported results and adverse
events, add the results modules to ``API_FIELDS`` and the matching fields to
``Trial``; this module and the corpus stay as they are.
"""

import functools
import gzip
import logging
import pickle
from collections.abc import Sequence

from tqdm import tqdm

from trialsynth.base.models import Outcome, Trial

logger = logging.getLogger(__name__)


@functools.cache
def trials_by_nct() -> dict[str, Trial]:
    """Load the pickled ctgov trial dump, keyed by NCT ID.

    Returns
    -------
    :
        Every :class:`~trialsynth.base.models.Trial` in the dump.

    Raises
    ------
    FileNotFoundError
        If the ctgov pipeline has not been run.
    """
    from trialsynth.ctgov.config import CTConfig

    path = CTConfig().raw_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Trial dump not found: {path}. Must run the clinicaltrials "
            f"pipeline before preparing the ctgov corpus."
        )
    logger.info("Loading trial dump from %s", path)
    with gzip.open(path, "rb") as fh:
        return {trial.ns_id: trial for trial in pickle.load(fh)}



def resolve_nct_ids() -> list[str]:
    """Return every NCT ID in the ctgov trial dump."""
    return sorted(trials_by_nct())


def _criteria_text(criteria: str) -> str:
    """Normalize the registry's criteria blob to one criterion per line.

    Unescapes the blob and strips bullet markers. Terminal punctuation is
    added later, by :func:`_terminate` over every rendered line.
    """
    lines = []
    for line in criteria.replace("\\n", "\n").replace("\\", "").splitlines():
        line = line.strip().lstrip("*-• \t").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _terminate(line: str) -> str:
    """End ``line`` with sentence punctuation unless it already has some.

    ``resolve_anchors`` locates evidence by splitting the source text on
    sentence punctuation (``extract_util.split_sentences``) after collapsing
    whitespace, so a line with no terminator merges into the next one. Without
    this every anchor in a registry record resolves to the same enormous
    sentence. Lines already ending in ``:`` are left alone so a heading stays
    attached to what it introduces.
    """
    line = line.rstrip()
    if not line or line[-1] in ".!?:":
        return line
    return f"{line}."


def render_trial(trial: Trial) -> str:
    """Render one registry record as the text sent to the model.

    Parameters
    ----------
    trial :
        A trial from the ctgov dump.

    Returns
    -------
    :
        Plain-text rendering of the record's protocol fields.
    """
    # No "record NCT01234567" header line: the NCT ID is already the Bedrock
    # recordId and already in the framing the corpus wraps this text in.
    sections: list[str] = []

    def add(label: str, value) -> None:
        if not value:
            return
        sections.append(f"{label}: {value}")

    add("Brief title", trial.title)
    add("Official title", trial.official_title)
    add("Conditions", ", ".join(c.text for c in trial.conditions if c.text))

    # The registry type ("drug", "device") goes with the other structured
    # fields; only the name and description can name a biomarker.
    interventions = [
        f"- {i.text}: {i.description}" if i.description else f"- {i.text}"
        for i in trial.interventions
        if i.text
    ]
    if interventions:
        sections.append("Interventions:")
        sections.extend(interventions)

    add("Brief summary", trial.brief_summary)
    add("Detailed description", trial.detailed_description)

    # Names only, primary and secondary in one list: no schema field takes a
    # time frame or cares which list a measure came from.
    measures = [
        outcome.measure if isinstance(outcome, Outcome) else outcome
        for outcome in (*trial.primary_outcomes, *trial.secondary_outcomes)
    ]
    measures = [f"- {measure}" for measure in measures if measure]
    if measures:
        sections.append("Outcome measures:")
        sections.extend(measures)

    # Age, sex and healthy-volunteer eligibility are deliberately not rendered
    # from their own registry fields: the schema only wants them as criteria,
    # and only when the criteria block itself states them.
    if trial.eligibility.criteria:
        sections.append("Eligibility criteria:")
        sections.append(_criteria_text(trial.eligibility.criteria))

    return "\n".join(
        _terminate(line) for section in sections for line in section.splitlines()
    )


def load_texts(nct_ids: Sequence[str], max_workers: int = 8) -> dict[str, str]:
    """Return ``{nct_id: rendered text}`` for the given records.

    Parameters
    ----------
    nct_ids :
        NCT IDs to render.
    max_workers :
        Ignored; rendering reads the local dump. Kept for corpus parity.

    Returns
    -------
    :
        Rendered text per NCT ID. IDs missing from the trial dump, and records
        with no renderable field, are left out.
    """
    trials = trials_by_nct()
    texts = {}
    missing = []
    for nct_id in tqdm(nct_ids, desc="Rendering registry records"):
        trial = trials.get(nct_id)
        text = render_trial(trial) if trial else ""
        if not text:
            missing.append(nct_id)
            continue
        texts[nct_id] = text

    logger.info("Rendered %d registry records", len(texts))
    if missing:
        logger.warning(
            "%d NCT ID(s) are not in the trial dump, or render empty (%s...). "
            "Re-run the clinicaltrials fetch to widen coverage.",
            len(missing),
            ", ".join(missing[:10]),
        )
    return texts
