import copy
import logging
import warnings
from typing import Iterator, Optional, Callable, Literal

nmslib_logger = logging.getLogger('nmslib')
nmslib_logger.setLevel(logging.ERROR)
warnings.simplefilter('ignore')

import gilda
import gilda.ner
from gilda import make_grounder
from gilda.grounder import Annotation, ScoredMatch
from gilda.process import normalize
from gilda.term import Term
from indra.databases import drugbank_client as db
from indra.databases import mesh_client

from .models import BioEntity
from .util import (
    CONDITION_NS,
    DRUG_NAMESPACES,
    INTERVENTION_NS,
    must_override
)

logger = logging.getLogger(__name__)

ANNOTATE_MIN_LEN = 4
ANNOTATE_MIN_SCORE = 0.7
ANNOTATE_STOPLIST = {
    "NT", "CON", "GCA", "TAB", "CPAP", "COPE", "CHCPE", "TPF", "PF",
    "JIA", "OR", "IV", "WT", "HR", "CI", "RR", "OS", "PFS", "CR", "PR",
    "SD", "PD", "CT", "MRI", "PCR", "IHC", "AE", "SAE", "PS", "ECOG",
}

GrounderSignature = Callable[
    [str, Optional[str], Optional[list[str]], Optional[list[str]]],
    list[ScoredMatch]
]
AnnotatorSignature = Optional[Callable[[str, Optional[str]], list[Annotation]]]


class Annotator:
    def __init__(
        self,
        *,
        namespaces: Optional[list[str]] = None,
        mesh_prefix: Optional[Literal["mesh", "MESH"]] = "MESH"
    ):
        """Base class for annotators that annotate text with named entities.

        Parameters
        ----------
        namespaces : Optional[list[str]]
            A list of namespaces to consider for annotation. If None, defaults to ["MESH"].
        mesh_prefix : Optional[Literal["mesh", "MESH"]]
            The prefix to use for MESH grounding. If None, defaults to "MESH". Use
            this to specify the capitalization of the MESH prefix for grounding.
        """
        if namespaces is None:
            namespaces = [mesh_prefix]
        self.namespaces = namespaces

    def __call__(self, text: str, *, context: str = None) -> list[Annotation]:
        return self.annotate(text, context=context)

    @must_override
    def annotate(self, text: str, *, context: str = None) -> list[Annotation]:
        pass


class GildaAnnotator(Annotator):
    def __init__(
        self,
        *,
        namespaces: Optional[list[str]] = None,
        mesh_prefix: Optional[Literal["mesh", "MESH"]] = "MESH"
    ):
        if namespaces is None:
            logger.info("No namespaces provided, using default Gilda namespaces.")
            namespaces = gilda.grounder.DEFAULT_NAMESPACE_PRIORITY
        super().__init__(namespaces=namespaces, mesh_prefix=mesh_prefix)
    def annotate(self, text: str, *, context: str = None):
        return gilda.annotate(
            text=text, context_text=context, namespaces=self.namespaces
        )




class Grounder:
    """A callable class that grounds a BioEntity to a database identifier.

    Parameters
    ----------
    namespaces : Optional[list[str]]
        A list of namespaces to consider for grounding (default: None).
    restrict_mesh_prefix : list[str], optional
        A list of MESH tree prefixes to restrict grounding to (default: None).
    annotator : Callable[[str, Optional[str]], list[Tuple[Annotation]]], optional
        A callable that takes a string and returns a list of annotations
        (default: GildaAnnotator).
    grounder_func : Optional[GrounderSignature], optional
        A callable that takes a string and returns a list of ScoredMatches.
        If None, defaults to `gilda.ground` (default: None).
    mesh_prefix : Optional[Literal["mesh", "MESH"]]
        The prefix to use for MESH grounding. If None, defaults to "MESH". Use
        this to specify the capitalization of the MESH prefix for grounding.

    Attributes
    ----------
    namespaces : Optional[list[str]]
        A list of namespaces to consider for grounding.
    restrict_mesh_prefix : list[str]
        A list of MESH tree prefixes to restrict grounding to.
    annotator : Callable[[str, Optional[str]], list[Annotation]]
        A callable that annotates text with named entities.
    grounder_func : GrounderSignature
        A callable that grounds text to named entities.
    mesh_prefix : Optional[Literal["mesh", "MESH"]]
        The prefix to use for MESH grounding. Should match the capitalization
        returned by the grounding function.
    """

    def __init__(
        self,
        *,
        namespaces: Optional[list[str]] = None,
        restrict_mesh_prefix: list[str] = None,
        annotator: AnnotatorSignature = None,
        grounder_func: Optional[GrounderSignature] = None,
        mesh_prefix: Optional[Literal["mesh", "MESH"]] = "MESH"
    ):
        self.namespaces: Optional[list[str]] = namespaces
        self.restrict_mesh_prefix = restrict_mesh_prefix
        if annotator is None:
            annotator = GildaAnnotator(namespaces=namespaces, mesh_prefix=mesh_prefix)
        self.annotator = annotator
        if grounder_func is None:
            grounder_func = gilda.ground
        self.grounder_func = grounder_func
        self.mesh_prefix = mesh_prefix

    @must_override
    def preprocess(self, entity: BioEntity) -> BioEntity:
        """Preprocess the BioEntity before grounding.

        This method can be overridden by subclasses to perform any necessary preprocessing
        steps on the BioEntity before grounding it.

        Parameters
        ----------
        entity : BioEntity
            The BioEntity to preprocess.

        Returns
        -------
        BioEntity
            The preprocessed BioEntity.
        """

    def __call__(
        self, entity: BioEntity, context: Optional[str] = None
    ) -> Iterator[BioEntity]:
        return self.ground(entity, context)

    def _create_grounded_entity(
        self, entity: BioEntity, *, db_ns, db_id: str, norm_text: str
    ) -> BioEntity:
        grounded_entity = copy.deepcopy(entity)
        grounded_entity.grounding_source = "gilda"
        grounded_entity.ns = db_ns
        grounded_entity.ns_id = db_id
        grounded_entity.grounded_term = norm_text
        return grounded_entity

    def _yield_entity(
        self, entity: BioEntity, match: ScoredMatch
    ) -> Iterator[BioEntity]:
        # Do special handling for MESH entities
        if match.term.db == self.mesh_prefix:
            mesh_id = match.term.id
            if self.restrict_mesh_prefix and any(mesh_client.has_tree_prefix(mesh_id, prefix) for prefix in self.restrict_mesh_prefix):
                yield self._create_grounded_entity(
                    entity, db_ns=self.mesh_prefix, db_id=mesh_id, norm_text=match.term.entry_name
                )
            if not self.restrict_mesh_prefix:
                yield self._create_grounded_entity(
                    entity, db_ns=self.mesh_prefix, db_id=mesh_id, norm_text=match.term.entry_name
                )
        else:
            # If no MESH ID is found, we yield the original entity with the match
            yield self._create_grounded_entity(
                entity,
                db_ns=match.term.db,
                db_id=match.term.id,
                norm_text=match.term.entry_name,
            )

    def ground(
        self,
        entity: BioEntity,
        context: Optional[str] = None
    ) -> Iterator[BioEntity]:
        """Ground a BioEntity to a CURIE."""
        entity = self.preprocess(entity)
        # Do special handling for MESH entities
        if entity.ns and entity.ns == self.mesh_prefix and entity.ns_id:
            mesh_name = mesh_client.get_mesh_name(entity.ns_id, offline=True)
            if mesh_name:
                entity.grounded_term = mesh_name
                yield entity
            else:
                matches = self.grounder_func(entity.text, namespaces=[self.mesh_prefix])
                if matches:
                    yield from self._yield_entity(entity, matches[0])
        # If the entity already has a namespace and ID, we assume it's already grounded
        elif entity.ns and entity.ns_id:
            assert entity.grounding_source, "Grounding source must be provided if entity has namespace and ID"
            yield entity
        # Otherwise, we ground the entity using Gilda
        else:
            matches = self.grounder_func(
                entity.text, namespaces=self.namespaces, context=context
            )
            if matches:
                yield from self._yield_entity(entity, matches[0])
            else:
                annotations = self.annotator(entity.text, context=context)
                for annotation in annotations:
                    yield from self._yield_entity(entity, annotation.matches[0])
                # If no matches are found, we try to annotate the description if it exists
                if entity.description:
                    annotations = self.annotator(
                        entity.description, context=context
                    )
                    for annotation in annotations:
                        yield from self._yield_entity(entity, annotation.matches[0])


class ConditionGrounder(Grounder):
    def __init__(
        self,
        namespaces: Optional[list[str]] = None,
        annotator: Optional[AnnotatorSignature] = None,
        grounder_func: Optional[GrounderSignature] = None,
        mesh_prefix: Literal["mesh", "MESH"] = "MESH"
    ):
        if namespaces is None:
            namespaces = CONDITION_NS
        if annotator is None:
            annotator = GildaAnnotator(namespaces=namespaces)
        super().__init__(
            namespaces=namespaces,
            restrict_mesh_prefix=['C', 'F'],
            annotator=annotator,
            grounder_func=grounder_func,
            mesh_prefix=mesh_prefix
        )


class InterventionGrounder(Grounder):
    def __init__(
        self,
        namespaces: Optional[list[str]] = None,
        annotator: Optional[AnnotatorSignature] = None,
        grounder_func: Optional[GrounderSignature] = None,
        mesh_prefix: Literal["MESH", "mesh"] = "MESH"
    ):
        if namespaces is None:
            namespaces = INTERVENTION_NS
        if annotator is None:
            annotator = GildaAnnotator(namespaces=namespaces)
        super().__init__(
            namespaces=namespaces,
            restrict_mesh_prefix=['D', 'E'],
            annotator=annotator,
            grounder_func=grounder_func,
            mesh_prefix=mesh_prefix
        )

def build_drugbank_terms():
    """Parse INDRA's DrugBank ID to name mappings into Gilda Terms."""
    terms = []
    for drugbank_id, name in db.drugbank_names.items():
        terms.append(Term(
            norm_text=normalize(name),
            text=name,
            db="DRUGBANK",
            id=drugbank_id,
            entry_name=name,
            status="name",
            source="drugbank",
        ))
    return terms

_drugbank_grounder = None
def get_drugbank_grounder():
    """Build once and return a DrugBank-only grounder, without touching Gilda's global grounder."""
    global _drugbank_grounder
    if _drugbank_grounder is None:
        _drugbank_grounder = make_grounder(build_drugbank_terms())
    return _drugbank_grounder

def _term_result(scored_match):
    """Return db, id, name, and score for a scored match."""
    top = scored_match.term
    return {"db": top.db, "id": top.id, "entry_name": top.entry_name, "score": scored_match.score}

def _first_valid_annotation(text, grounder, namespaces):
    """Fallback: annotate text with the given grounder, return the first hit above the length, stoplist, and score filters."""
    annotations = gilda.ner.annotate(text, grounder=grounder, namespaces=namespaces)
    for annotation in annotations:
        matched_text = annotation.text.strip()
        if len(matched_text) < ANNOTATE_MIN_LEN:
            continue
        if matched_text.upper() in ANNOTATE_STOPLIST:
            continue
        if annotation.matches[0].score < ANNOTATE_MIN_SCORE:
            continue
        return annotation.matches[0]
    return None

def drugbank_ground(text):
    """Ground text with DrugBank first, falling back to the default Gilda grounder."""
    drugbank_results = get_drugbank_grounder().ground(text)
    if drugbank_results:
        return _term_result(drugbank_results[0])
    drugbank_match = _first_valid_annotation(text, get_drugbank_grounder(), namespaces=["DRUGBANK"])
    if drugbank_match:
        return _term_result(drugbank_match)
    fallback_results = gilda.get_grounder().ground(text, namespaces=DRUG_NAMESPACES)
    if fallback_results:
        return _term_result(fallback_results[0])
    fallback_match = _first_valid_annotation(text, gilda.get_grounder(), namespaces=DRUG_NAMESPACES)
    if fallback_match:
        return _term_result(fallback_match)
    return None
