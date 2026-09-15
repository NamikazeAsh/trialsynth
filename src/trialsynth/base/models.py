import logging
from datetime import datetime
from typing import Optional, Union, Literal

import indra.statements.agent as agent
from bioregistry import curie_to_str
from indra.ontology.standardize import standardize_name_db_refs

logger = logging.getLogger(__name__)


class SecondaryId:
    """Secondary ID for a trial

    Attributes
    ----------
    ns : str
        The secondary ID's namespace
    id : str
        The ID of the secondary ID
    """

    def __init__(self, ns: str = None, id: str = None):
        self.ns = ns
        self.id = id

    @property
    def curie(self) -> str:
        """Creates a CURIE from the namespace and ID

        Returns
        -------
        str
            The CURIE
        """
        std_name, db_ref = standardize_name_db_refs({self.ns: self.id})
        ns, id = agent.get_grounding(db_ref)
        if ns and id:
            self.ns = ns
            self.id = id

        return curie_to_str(self.ns, self.id)


class DesignInfo:
    """Design information for a trial

    Attributes
    ----------
    purpose : str
        The purpose of the design
    allocation : str
        The allocation of the design
    masking : str
        The masking of the design
    assignment : str
        The assignment of the design
    fallback : Optional[str]
        The fallback design information, if the design information is not in the expected format

    Parameters
    ----------
    purpose : str
        The purpose of the design
    allocation : str
        The allocation of the design
    masking : str
        The masking of the design
    assignment : str
        The assignment of the design
    fallback : Optional[str]
        The fallback design information, if the design information is not in the expected format
    """

    def __init__(
        self,
        purpose=None,
        allocation=None,
        masking=None,
        assignment=None,
        fallback: Optional[str] = None,
    ):
        self.purpose: str = purpose
        self.allocation: str = allocation
        self.masking: str = masking
        self.assignment: str = assignment
        self.fallback: str = fallback


class Outcome:
    """Outcome for a trial

    Attributes
    ----------
    measure : str
        The measure of the outcome
    time_frame : str
        The time frame of the outcome

    Parameters
    ----------
    measure : str
        The measure of the outcome
    time_frame : str
        The time frame of the outcome
    """

    def __init__(self, measure: str = None, time_frame: str = None):
        self.measure = measure
        self.time_frame = time_frame


class Eligibility:
    """Who a trial will and will not enrol

    ``criteria`` is the registry's free-text blob, listing inclusion and
    exclusion criteria under their own headings rather than as separate fields.

    Attributes
    ----------
    criteria : Optional[str]
        Free-text inclusion and exclusion criteria
    sex : Optional[str]
        The sex eligible for the trial, e.g. "FEMALE", "ALL"
    minimum_age : Optional[str]
        The minimum age, as reported, e.g. "18 Years"
    maximum_age : Optional[str]
        The maximum age, as reported, e.g. "85 Years"
    std_ages : list[str]
        Standardized age groups, e.g. ["ADULT", "OLDER_ADULT"]
    healthy_volunteers : Optional[bool]
        Whether the trial accepts healthy volunteers

    Parameters
    ----------
    criteria : Optional[str]
        Free-text inclusion and exclusion criteria
    sex : Optional[str]
        The sex eligible for the trial
    minimum_age : Optional[str]
        The minimum age, as reported
    maximum_age : Optional[str]
        The maximum age, as reported
    std_ages : Optional[list[str]]
        Standardized age groups
    healthy_volunteers : Optional[bool]
        Whether the trial accepts healthy volunteers
    """

    def __init__(
        self,
        criteria: Optional[str] = None,
        sex: Optional[str] = None,
        minimum_age: Optional[str] = None,
        maximum_age: Optional[str] = None,
        std_ages: Optional[list[str]] = None,
        healthy_volunteers: Optional[bool] = None,
    ):
        self.criteria = criteria
        self.sex = sex
        self.minimum_age = minimum_age
        self.maximum_age = maximum_age
        self.std_ages: list[str] = std_ages or []
        self.healthy_volunteers = healthy_volunteers


class Location:
    """A site where a trial is or was run

    Attributes
    ----------
    facility : Optional[str]
        The name of the facility
    city : Optional[str]
        The city the facility is in
    state : Optional[str]
        The state or region the facility is in
    zip_code : Optional[str]
        The postal code of the facility
    country : Optional[str]
        The country the facility is in
    latitude : Optional[float]
        The latitude of the facility, as geocoded by the registry
    longitude : Optional[float]
        The longitude of the facility, as geocoded by the registry

    Parameters
    ----------
    facility : Optional[str]
        The name of the facility
    city : Optional[str]
        The city the facility is in
    state : Optional[str]
        The state or region the facility is in
    zip_code : Optional[str]
        The postal code of the facility
    country : Optional[str]
        The country the facility is in
    latitude : Optional[float]
        The latitude of the facility
    longitude : Optional[float]
        The longitude of the facility
    """

    def __init__(
        self,
        facility: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        zip_code: Optional[str] = None,
        country: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ):
        self.facility = facility
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.country = country
        self.latitude = latitude
        self.longitude = longitude


# types of all nodes should be standardized to a class holding enumerations in the future.


class Node:
    """Node for a trial or bioentity

    Attributes
    ----------
    ns : str
        The namespace of the node
    ns_id : str
        The ID of the node
    labels : list[str]
        The labels of the node (default: []).
    source : Optional[str]
        The source registry of the node

    Parameters
    ----------
    ns : str
        The namespace of the node (default: None).
    ns_id : str
        The ID of the node (default: None).
    """

    def __init__(
        self,
        source: str,
        ns: str = None,
        ns_id: str = None,
    ):
        self.ns: str = ns
        self.ns_id: str = ns_id
        self.labels: list[str] = []
        self.source: str = source

    @property
    def curie(self) -> str:
        if not self.ns or not self.ns_id:
            logger.warning(
                f"{self} does not have a namespace or ID to produce a CURIE with."
            )
            return ""
        return curie_to_str(self.ns.lower(), self.ns_id)

    @curie.setter
    def curie(self, curie: str):
        self.ns, self.ns_id = curie.split(":")


class BioEntity(Node):
    """Holds information about a biological entity

    Attributes
    ----------
    ns: str
        The namespace of the bioentity
    id: str
        The ID of the bioentity
    source: Optional[str]
        The source registry of the bioentity
    text: str
        The free-text of the bioentity
    description: Optional[str]
        The description of the bioentity
    labels: list[str]
        The labels of the bioentity
    grounded_term: Optional[str]
        The entry-term for the grounded bioentity from the given namespace
    origin: Optional[str]
        The trial CURIE that the bioentity is associated with

    Parameters
    ----------
    text: str
        The text term of the bioentity from the given namespace
    labels: list[str]
        The labels of the bioentity
    origin: str
        The trial CURIE that the bioentity is associated with
    source: Optional[str]
        The source registry of the bioentity.
    ns: Optional[str]
        The namespace of the bioentity (default: None).
    id: Optional[str]
        The ID of the bioentity (default: None).
    """

    def __init__(
        self,
        text: str,
        labels: list[str],
        origin: str,
        source: str,
        grounding_source: Optional[Literal["gilda", "mesh"]] = None,
        description: Optional[str] = None,
        ns: Optional[str] = None,
        id: Optional[str] = None,
        grounded_term: Optional[str] = None,
    ):
        super().__init__(ns=ns, ns_id=id, source=source)
        self.labels = labels
        self.text: str = text
        self.description: str = description
        self.origin: str = origin
        self.grounded_term: str = grounded_term
        self.grounding_source: str = grounding_source


class Condition(BioEntity):
    """Represents a condition in a clinical trial

    Parameters
    ----------
    text: str
        The text term of the bioentity from the given namespace
    description: Optional[str]
        The description of the bioentity (default: None).
    labels: list[str]
        The labels of the bioentity
    origin: str
        The trial CURIE that the bioentity is associated with
    source: Optional[str]
        The source registry of the bioentity.
    ns: Optional[str]
        The namespace of the bioentity (default: None).
    id: Optional[str]
        The ID of the bioentity (default: None).
    """

    def __init__(
        self,
        text: str,
        origin: str,
        source: str,
        grounding_source: Optional[Literal["gilda", "mesh"]] = None,
        description: Optional[str] = None,
        labels: Optional[list[str]] = None,
        ns: Optional[str] = None,
        id: Optional[str] = None,
    ):
        super().__init__(
            text=text,
            labels=['condition'],
            origin=origin,
            source=source,
            grounding_source=grounding_source,
            ns=ns,
            id=id,
            description=description,
        )
        if labels:
            self.labels.extend(labels)


class Intervention(BioEntity):
    """
    Represents an intervention in a clinical trial.

    Attributes
    ----------
    text : str
        The text term of the intervention.
    origin : str
        The trial CURIE that the intervention is associated with.
    source : str
        The source registry of the intervention.
    description : Optional[str]
        The description of the intervention (default: None).
    labels : list[str], optional
        Additional labels for the intervention (default: ['intervention']).
    ns : str, optional
        The namespace of the intervention (default: None).
    id : str, optional
        The ID of the intervention (default: None).

    Parameters
    ----------
    text : str
        The text term of the intervention.
    origin : str
        The trial CURIE that the intervention is associated with.
    source : str
        The source registry of the intervention.
    labels : list[str], optional
        Additional labels for the intervention (default: None).
    ns : str, optional
        The namespace of the intervention (default: None).
    id : str, optional
        The ID of the intervention (default: None).
    """
    def __init__(
        self,
        text: str,
        origin: str,
        source: str,
        grounding_source: Optional[Literal["gilda", "mesh"]] = None,
        description: Optional[str] = None,
        labels: Optional[list[str]] = None,
        ns: Optional[str] = None,
        id: Optional[str] = None,
    ):
        super().__init__(
            text=text,
            description=description,
            labels=['intervention'],
            origin=origin,
            grounding_source=grounding_source,
            source=source,
            ns=ns,
            id=id
        )
        if labels:
            self.labels.extend(labels)


class Trial(Node):
    """Holds information about a clinical trial

    Attributes
    ----------
    ns: str
        The namespace of the trial
    id: str
        The ID of the trial
    labels: list[str]
        The labels of the trial (default: ['clinicaltrial']).
    source: Optional[str]
        The source registry of the trial (default: None).
    title: str
        The title of the trial
    official_title: Optional[str]
        The official title of the trial (default: None).
    design: Union[DesignInfo, str]
        The design information of the trial
    conditions: list
        The conditions targeted in the trial
    interventions: list
        The interventions used in the trial
    primary_outcomes: Union[Outcome, str]
        The primary outcome of the trial
    secondary_outcomes: Union[Outcome, str]
        The secondary outcome of the trial
    secondary_ids: Union[list[SecondaryId], list[str]]
        The secondary IDs of the trial
    eligibility: Eligibility
        Who the trial will and will not enrol
    locations: list[Location]
        The sites where the trial is or was run
    enrollment: Optional[int]
        The number of participants, planned or actual
    enrollment_type: Optional[str]
        Whether ``enrollment`` is "estimated" (planned) or "actual" (as run)
    has_results: Optional[bool]
        Whether the registry record carries a results section

    Parameters
    ----------
    ns : str
        The namespace of the trial
    id : str
        The ID of the trial
    labels : Optional[list[str]]
        The labels of the trial (default: None).
    source : Optional[str]
        The source registry of the trial (default: None).
    """

    def __init__(
        self,
        ns: str,
        id: str,
        labels: Optional[list[str]] = None,
        source: Optional[str] = None,
    ):
        super().__init__(source=source, ns=ns, ns_id=id)
        self.labels: list[str] = ["clinical_trial"]
        self.phases: list[str] = []
        self.start_date: Optional[datetime] = None
        self.start_date_type: Optional[str] = None
        self.completion_date: Optional[datetime] = None
        self.completion_date_type: Optional[str] = None
        self.primary_completion_date: Optional[datetime] = None
        self.primary_completion_date_type: Optional[str] = None
        self.last_update_submit_date: Optional[datetime] = None
        self.overall_status: Optional[str] = None
        self.why_stopped: Optional[str] = None

        if labels:
            self.labels.extend(labels)

        self.title: Optional[str] = None
        self.official_title: Optional[str] = None
        self.brief_summary: Optional[str] = None
        self.detailed_description: Optional[str] = None
        self.design: DesignInfo = DesignInfo()
        self.entities: list[BioEntity] = []
        self.primary_outcomes: list[Union[Outcome, str]] = []
        self.secondary_outcomes: list[Union[Outcome, str]] = []
        self.secondary_ids: list[SecondaryId] = []
        self.references: list[tuple[str, str]] = []
        self.eligibility: Eligibility = Eligibility()
        self.locations: list[Location] = []
        self.enrollment: Optional[int] = None
        self.enrollment_type: Optional[str] = None
        self.has_results: Optional[bool] = None

    @property
    def conditions(self) -> list[Condition]:
        return [entity for entity in self.entities if isinstance(entity, Condition)]

    @property
    def interventions(self) -> list[Intervention]:
        return [entity for entity in self.entities if isinstance(entity, Intervention)]


class Edge:
    """Edge between a trial and a bioentity

    Attributes
    ----------
    trial :
        The trial that has a relation to an entity
    entity :
        The bioentity that is related to the trial.
    rel_type :
        The type of relation.
    grounding_sources :
        The sources of grounding for the bioentity.
    """

    def __init__(
        self,
        trial: Trial,
        entity: BioEntity,
        source: str,
        grounding_sources: list[Literal["gilda", "mesh"]],
    ):
        self.trial = trial
        self.entity = entity
        self.source = source

        self.rel_type = f"has_{type(entity).__name__.lower()}"
        self.grounding_sources: list[Literal["gilda", "mesh"]] = grounding_sources


class PublicationEdge:
    """Edge between a trial and a publication

    Attributes
    ----------
    trial :
        The ID of the trial that has a relation to the publication
    publication :
        The publication that is referenced by the trial
    rel_type :
        The type of relation.
    """

    def __init__(
        self,
        trial: str,
        publication: str,
        source: str,
        ref_type: Optional[str] = None,
    ):
        self.trial = trial
        self.publication = publication
        self.rel_type = "has_publication"
        self.source = source
        self.ref_type = ref_type
