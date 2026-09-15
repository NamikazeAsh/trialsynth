"""Gets Clinicaltrials.gov data from REST API or saved file"""
import datetime
from time import sleep

import requests
from overrides import overrides
from tqdm import tqdm
import logging

from ..base.fetch import Fetcher
from ..base.models import (
    Condition,
    DesignInfo,
    Eligibility,
    Intervention,
    Location,
    Outcome,
    SecondaryId,
    Trial,
)
from .rest_api_response_models import UnflattenedTrial
from .config import CTConfig
from ..base.extract.build_pubmed_nct_links import generate_pubmed_trial_links, PMID_NCT_LINKS

logger = logging.getLogger(__name__)


class CTFetcher(Fetcher):
    """Fetches data from the Clinicaltrials.gov REST API and transforms it into a list of :class:`Trial` objects

    Attributes
    ----------
    raw_data : list[Trial]
        Raw data from the API
    url : str
        URL of the API endpoint
    api_parameters : dict
        Parameters to send with the API request
    config : Config
        User-mutable properties of registry data processing

    Parameters
    ----------
    config : Config
        User-mutable properties of registry data processing
    """

    def __init__(self, config: CTConfig):
        super().__init__(config)
        self.api_parameters = {
            "fields": self.config.api_fields,  # actually column names, not fields
            "pageSize": 1000,
            "countTotal": "true",
        }
        self.total_pages = 0

    @overrides
    def get_api_data(self, reload: bool = False, max_pages=None, *kwargs) -> None:
        trial_path = self.config.raw_data_path
        if trial_path.is_file() and not reload:
            self.load_saved_data()
        else:
            # Data does not exist or reload is True, so fetch from API
            logger.info(f"Fetching Clinicaltrials.gov data from {self.url}")

            try:
                self._read_next_page()

                pages = self.total_pages if max_pages is None else max_pages
                page_size = self.api_parameters.get("pageSize")
                with tqdm(
                    desc="Downloading ClinicalTrials.gov trials",
                    total=int(pages * page_size) if max_pages is None else max_pages * page_size,
                    unit="trial",
                    unit_scale=True,
                ) as pbar:
                    pbar.update(page_size)
                    for _ in range(int(pages)):
                        self._read_next_page()
                        pbar.update(page_size)

            except Exception:
                logger.exception(f"Could not fetch data from {self.url}")
                raise

            self.save_raw_data()

        # Run the PubMed link generation
        if reload or not PMID_NCT_LINKS.exists():
            generate_pubmed_trial_links(
                download_missing=True, max_files=max_pages
            )

    def _read_next_page(self, retries: int = 3) -> None:

        # TODO: timeout should be a config var
        timeout = 300
        try:
            response = requests.get(self.url, self.api_parameters, timeout=timeout)
        except requests.exceptions.Timeout:
            if retries > 0:
                logger.warning(
                    f'Retrying request to {self.url} with params {self.api_parameters} '
                    f'due to timeout. Retries left: {retries - 1}'
                )
                sleep(5)  # Wait a bit before retrying
                self._read_next_page(retries - 1)
                return
            logger.info(f'Connection timed-out after {timeout}s. To avoid this, '
                        f'either set the timeout max higher, or establish a '
                        f'better internet connection.')
            raise
        response.raise_for_status()
        json_data = response.json()

        studies = json_data.get("studies", [])
        trials = self._json_to_trials(studies)
        self.raw_data.extend(trials)
        self.api_parameters["pageToken"] = json_data.get("nextPageToken")

        if not self.total_pages:
            self.total_pages = json_data.get(
                "totalCount"
            ) / self.api_parameters.get("pageSize")

    def _json_to_trials(self, data: dict) -> list[Trial]:
        trials = []

        for study in data:
            rest_trial = UnflattenedTrial(**study)

            trial = Trial(
                ns="clinicaltrials",
                id=rest_trial.protocol_section.id_module.nct_id,
            )

            # Brief Title, summary and detailed description
            trial.title = rest_trial.protocol_section.id_module.brief_title
            trial.official_title = rest_trial.protocol_section.id_module.official_title
            trial.brief_summary = rest_trial.protocol_section.description_module.brief_summary
            trial.detailed_description = (
                rest_trial.protocol_section.description_module.detailed_description
            )

            # Study Type e.g. "Interventional", "Observational"
            study_type = rest_trial.protocol_section.design_module.study_type

            if study_type:
                trial.labels.append(study_type.strip().lower())

            # Phases e.g. "PHASE1", "PHASE2|PHASE3", "EARLY_PHASE_1", "NA"
            phases = rest_trial.protocol_section.design_module.phases

            if phases:
                trial.phases.extend([phase.strip().lower() for phase in phases])

            # Start date, completion date, primary completion date, last update date
            start_date_str = (
                rest_trial.protocol_section.status_module.start_date_struct.date
            )
            start_date_type = rest_trial.protocol_section.status_module.start_date_struct.date_type
            if start_date_str is not None:
                trial.start_date = _parse_date(start_date_str)
                trial.start_date_type = start_date_type.strip().lower() if start_date_type else None
            completion_date_str = (
                rest_trial.protocol_section.status_module.completion_date_struct.date
            )
            completion_date_type = (
                rest_trial.protocol_section.status_module.completion_date_struct.date_type
            )
            if completion_date_str is not None:
                trial.completion_date = _parse_date(completion_date_str)
                trial.completion_date_type = (
                    completion_date_type.strip().lower() if completion_date_type else None
                )
            primary_completion_date_str = (
                rest_trial.protocol_section.status_module.primary_completion_date_struct.date
            )
            primary_completion_date_type = (
                rest_trial.protocol_section.status_module.primary_completion_date_struct.date_type
            )
            if primary_completion_date_str is not None:
                trial.primary_completion_date = _parse_date(
                    primary_completion_date_str
                )
                trial.primary_completion_date_type = (
                    primary_completion_date_type.strip().lower()
                    if primary_completion_date_type
                    else None
                )
            last_update_date_str = (
                rest_trial.protocol_section.status_module.last_update_submit_date
            )
            if last_update_date_str is not None:
                trial.last_update_submit_date = _parse_date(last_update_date_str)

            # Overall status e.g. "COMPLETED", "RECRUITING", "TERMINATED"
            overall_status = rest_trial.protocol_section.status_module.overall_status
            trial.overall_status = overall_status.strip().lower()

            # Why stopped e.g. "REASON_FOR_TERMINATION"
            why_stopped = rest_trial.protocol_section.status_module.why_stopped
            if why_stopped:
                trial.why_stopped = why_stopped.strip().lower()

            # Enrollment, either the planned target or the number actually run
            enrollment_info = rest_trial.protocol_section.design_module.enrollment_info
            trial.enrollment = enrollment_info.count
            if enrollment_info.enrollment_type:
                trial.enrollment_type = enrollment_info.enrollment_type.strip().lower()

            # Whether the record carries a results section
            trial.has_results = rest_trial.has_results

            # Who the trial will and will not enrol. The criteria are a single
            # free-text blob, not separate inclusion/exclusion fields.
            eligibility = rest_trial.protocol_section.eligibility_module
            trial.eligibility = Eligibility(
                criteria=eligibility.eligibility_criteria,
                sex=eligibility.sex.strip().lower() if eligibility.sex else None,
                minimum_age=eligibility.minimum_age,
                maximum_age=eligibility.maximum_age,
                std_ages=[age.strip().lower() for age in eligibility.std_ages],
                healthy_volunteers=eligibility.healthy_volunteers,
            )

            # Sites the trial is or was run at
            trial.locations = [
                Location(
                    facility=location.facility,
                    city=location.city,
                    state=location.state,
                    zip_code=location.zip_code,
                    country=location.country,
                    latitude=location.geo_point.lat,
                    longitude=location.geo_point.lon,
                )
                for location in (
                    rest_trial.protocol_section.contacts_locations_module.locations
                )
            ]

            # Design information
            design_info = rest_trial.protocol_section.design_module.design_info
            trial.design = DesignInfo(
                purpose=design_info.purpose,  # E.g. "TREATMENT"
                allocation=design_info.allocation,  # E.g. "RANDOMIZED"
                masking=design_info.masking_info.masking,  # E.g. "NONE"
                assignment=(
                    design_info.intervention_assignment  # E.g. "CROSSOVER"
                    if design_info.intervention_assignment
                    else design_info.observation_assignment
                ),
            )

            # Assigned conditions Mesh terms
            condition_meshes = (
                rest_trial.derived_section.condition_browse_module.condition_meshes
            )
            # Conditions
            conditions = (
                rest_trial.protocol_section.conditions_module.conditions
            )
            trial.entities = [
                Condition(
                    text=condition,
                    origin=trial.curie,
                    source=self.config.registry,
                )
                for condition in conditions
            ]
            trial.entities.extend(
                [
                    Condition(
                        ns="MESH",
                        id=mesh.mesh_id,
                        text=mesh.term,
                        origin=trial.curie,
                        source=self.config.registry,
                        grounding_source="mesh"
                    )
                    for mesh in condition_meshes
                ]
            )

            # Assigned intervention text
            intervention_arms = (
                rest_trial.protocol_section.arms_interventions_module.arms_interventions
            )
            # Assigned intervention Mesh terms
            intervention_meshes = (
                rest_trial.derived_section.intervention_browse_module.intervention_meshes
            )

            trial.entities.extend([
                Intervention(
                    text=i.name,
                    description=i.description,
                    labels=[i.intervention_type],
                    origin=trial.curie,
                    source=self.config.registry,
                )
                for i in intervention_arms
                if i.name
            ])
            trial.entities.extend(
                [
                    Intervention(
                        ns="MESH",
                        id=mesh.mesh_id,
                        text=mesh.term,
                        origin=trial.curie,
                        source=self.config.registry,
                        grounding_source="mesh"
                    )
                    for mesh in intervention_meshes
                ]
            )

            primary_outcomes = (
                rest_trial.protocol_section.outcomes_module.primary_outcome
            )
            trial.primary_outcomes = [
                Outcome(o.measure, o.time_frame) for o in primary_outcomes
            ]

            secondary_outcomes = (
                rest_trial.protocol_section.outcomes_module.secondary_outcome
            )
            trial.secondary_outcomes = [
                Outcome(o.measure, o.time_frame) for o in secondary_outcomes
            ]

            secondary_info = (
                rest_trial.protocol_section.id_module.secondary_ids
            )
            trial.secondary_ids = [
                SecondaryId(ns=s.id_type, id=s.secondary_id)
                for s in secondary_info
            ]

            # References
            references = rest_trial.protocol_section.references_module.references

            trial.references += [
                (ref.pmid, ref.type) for ref in references if ref.pmid is not None
            ]

            trial.source = self.config.registry

            trials.append(trial)

        return trials


def _parse_date(date_str: str) -> datetime.datetime:
    """Parse a date string into a datetime object."""
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.datetime.strptime(date_str, "%Y-%m")
