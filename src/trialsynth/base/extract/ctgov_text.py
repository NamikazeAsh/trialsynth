"""Render ClinicalTrials.gov registry records as extraction source text.

The text comes from the pickled :class:`~trialsynth.base.models.Trial` dump the
ctgov pipeline already writes, so registry fields are mapped in exactly one
place -- ``CTFetcher`` -- rather than re-walked from raw API JSON here. That
also means no per-record network fetch: a trial missing from the dump is
treated as having no record, and re-running the ctgov fetch widens coverage.

What a registry record can support is protocol-as-planned. ``API_FIELDS`` in
the config carries no results or outcome-measures module, only outcome measure
*names* and time frames plus a ``HasResults`` flag, so there are no metric
values and no adverse events to extract -- which is why the ctgov result schema
asks for neither.

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


def _phase(phases: list[str]) -> str | None:
    # The ctgov fetcher lowercases these, so "PHASE3" arrives as "phase3".
    parts = [
        phase.removeprefix("phase") for phase in phases if phase and phase != "na"
    ]
    return "/".join(parts) or None


def _label(entity) -> str | None:
    # BioEntity labels are ["intervention", <registry type>] and the like.
    kind = entity.labels[0] if entity.labels else None
    return next((label for label in entity.labels if label != kind), None)


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


def _outcome_lines(outcomes: list, heading: str) -> list[str]:
    rendered = []
    for outcome in outcomes:
        if isinstance(outcome, Outcome):
            measure, time_frame = outcome.measure, outcome.time_frame
        else:
            measure, time_frame = outcome, None
        if not measure:
            continue
        suffix = f" (time frame: {time_frame})" if time_frame else ""
        rendered.append(f"- {measure}{suffix}")
    return [heading, *rendered] if rendered else []


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
    eligibility = trial.eligibility
    design = trial.design
    sections: list[str] = [f"ClinicalTrials.gov registry record {trial.ns_id}"]

    def add(label: str, value) -> None:
        # Only absent and empty values are skipped: False and 0 are facts about
        # the record ("Accepts healthy volunteers: False") and must survive.
        if value is None or value == "":
            return
        sections.append(f"{label}: {value}")

    add("Brief title", trial.title)
    add("Official title", trial.official_title)
    add("Overall status", trial.overall_status)
    add("Why stopped", trial.why_stopped)
    add(
        "Study type",
        next((label for label in trial.labels if label != "clinical_trial"), None),
    )
    add("Phase", _phase(trial.phases))
    add(
        "Design",
        ", ".join(
            part
            for part in (
                design.purpose,
                design.allocation,
                design.masking,
                design.assignment,
            )
            if part
        ),
    )
    if trial.enrollment is not None:
        kind = f" ({trial.enrollment_type})" if trial.enrollment_type else ""
        add("Enrollment", f"{trial.enrollment}{kind}")
    add("Conditions", ", ".join(c.text for c in trial.conditions if c.text))

    interventions = [i for i in trial.interventions if i.text]
    if interventions:
        sections.append("Interventions:")
        for intervention in interventions:
            kind = _label(intervention)
            head = f"- {intervention.text}" + (f" ({kind})" if kind else "")
            if intervention.description:
                head = f"{head}: {intervention.description}"
            sections.append(head)

    add("Brief summary", trial.brief_summary)
    add("Detailed description", trial.detailed_description)

    sections.extend(_outcome_lines(trial.primary_outcomes, "Primary outcome measures:"))
    sections.extend(
        _outcome_lines(trial.secondary_outcomes, "Secondary outcome measures:")
    )

    ages = " to ".join(
        part for part in (eligibility.minimum_age, eligibility.maximum_age) if part
    )
    add("Eligible sex", eligibility.sex)
    add("Eligible ages", ages)
    add("Standard age groups", ", ".join(eligibility.std_ages))
    if eligibility.healthy_volunteers is not None:
        add("Accepts healthy volunteers", eligibility.healthy_volunteers)
    if eligibility.criteria:
        sections.append("Eligibility criteria:")
        sections.append(_criteria_text(eligibility.criteria))

    countries = sorted({loc.country for loc in trial.locations if loc.country})
    if countries:
        add(
            "Locations",
            f"{len(trial.locations)} site(s) in {', '.join(countries)}",
        )
    if trial.has_results is not None:
        add("Registry carries a results section", trial.has_results)

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
        Accepted for parity with the other corpora and ignored -- rendering
        reads the local trial dump, so there is nothing to parallelize.

    Returns
    -------
    :
        Rendered text per NCT ID. IDs missing from the trial dump are left out.
    """
    trials = trials_by_nct()
    texts = {}
    missing = []
    for nct_id in tqdm(nct_ids, desc="Rendering registry records"):
        trial = trials.get(nct_id)
        if trial is None:
            missing.append(nct_id)
            continue
        texts[nct_id] = render_trial(trial)

    logger.info("Rendered %d registry records", len(texts))
    if missing:
        preview = ", ".join(missing[:10])
        extra = "..." if len(missing) > 10 else ""
        logger.warning(
            "%d NCT ID(s) are not in the trial dump: %s%s. Re-run the "
            "clinicaltrials fetch to widen coverage.",
            len(missing),
            preview,
            extra,
        )
    return texts
