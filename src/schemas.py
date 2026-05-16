"""Typed schemas for a LICS-style distributed games research harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


EntryType = Literal[
    "paper_summary",
    "concept",
    "model",
    "theorem",
    "reduction",
    "open_problem",
    "conjecture_seed",
]
PendingStatus = Literal["pending", "approved", "rejected", "flagged"]
ClusterStatus = Literal["active", "watchlist", "archived"]
ConceptType = Literal[
    "model",
    "objective",
    "strategy",
    "architecture",
    "complexity_class",
    "logic",
    "proof_technique",
    "reduction_type",
]
ConceptRelationType = Literal[
    "generalizes",
    "specializes",
    "equivalent_to",
    "reduces_to",
    "uses",
    "conflicts_with",
    "related_to",
]
TheoremType = Literal[
    "decidability",
    "undecidability",
    "complexity_upper",
    "complexity_lower",
    "completeness",
    "memory_upper",
    "memory_lower",
    "equivalence",
    "characterization",
    "algorithm",
]
Confidence = Literal["pending", "verified", "rejected", "needs_review"]
ReviewStatus = Literal["draft", "active", "closed", "refuted", "proved", "paused", "abandoned"]
ConjectureExpectedStatus = Literal["true", "false", "unknown"]
ModelType = Literal[
    "ATS",
    "CDM",
    "2DM",
    "control",
    "distributed_synthesis",
    "asynchronous_game",
    "petri_game",
    "graph_game",
    "partial_information_game",
    "automata",
    "trace",
    "other",
]


class StrictBase(BaseModel):
    """Base schema for typed SQLite boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)


class ResearchCluster(StrictBase):
    """A literature and problem cluster within the broader research map."""

    cluster_id: int | None = None
    name: str = Field(min_length=1)
    description: str | None = None
    status: ClusterStatus = "active"
    priority: int = 0
    notes: str | None = None


class Concept(StrictBase):
    """A reusable ontology concept such as a model, objective, or technique."""

    concept_id: int | None = None
    name: str = Field(min_length=1)
    concept_type: ConceptType
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, aliases: list[str]) -> list[str]:
        return [alias.strip() for alias in aliases if alias.strip()]


class ConceptLink(StrictBase):
    """A typed relation between two ontology concepts."""

    source_concept_id: int
    target_concept_id: int
    relation_type: ConceptRelationType
    notes: str | None = None


class Paper(StrictBase):
    """Bibliographic metadata for a paper, thesis, note, or manuscript."""

    id: int | None = None
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1800, le=3000)
    venue: str | None = None
    pdf_path: str | None = None
    notes: str | None = None
    cluster_id: int | None = None

    @field_validator("authors")
    @classmethod
    def authors_must_not_be_blank(cls, authors: list[str]) -> list[str]:
        cleaned = [author.strip() for author in authors if author.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty author is required")
        return cleaned


class Model(StrictBase):
    """A formal model family or generated instance."""

    id: int | None = None
    name: str = Field(min_length=1)
    model_type: ModelType = "other"
    description: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    source_paper_id: int | None = None
    cluster_id: int | None = None


class Theorem(StrictBase):
    """A broad research result about synthesis, games, automata, or verification."""

    theorem_id: int | None = Field(default=None, validation_alias=AliasChoices("theorem_id", "id"))
    title: str | None = None
    statement: str = Field(min_length=1)
    theorem_type: TheoremType = "characterization"
    model_family: str | None = None
    objective_family: str | None = None
    architecture_assumptions: list[str] = Field(default_factory=list)
    information_assumptions: list[str] = Field(default_factory=list)
    strategy_assumptions: list[str] = Field(default_factory=list)
    process_bound: str | None = None
    complexity_upper: str | None = None
    complexity_lower: str | None = None
    memory_upper: str | None = None
    memory_lower: str | None = None
    source_paper_id: int | None = None
    source_location: str | None = None
    proof_technique: str | None = None
    confidence: Confidence = "pending"
    cluster_id: int | None = None
    notes: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    conclusion: str | None = None
    paper_id: int | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_compatibility_fields(self) -> "Theorem":
        if self.source_paper_id is None and self.paper_id is not None:
            self.source_paper_id = self.paper_id
        if not self.architecture_assumptions and self.assumptions:
            self.architecture_assumptions = list(self.assumptions)
        if not self.title:
            self.title = _title_from_statement(self.statement)
        return self

    @property
    def id(self) -> int | None:
        return self.theorem_id


class Reduction(StrictBase):
    """A reduction between formal problems, models, or decision questions."""

    id: int | None = None
    title: str = Field(min_length=1)
    source_problem: str = Field(min_length=1)
    target_problem: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    paper_id: int | None = None
    source_paper_id: int | None = None
    source_location: str | None = None
    proof_technique: str | None = None
    cluster_id: int | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def fill_source_paper(self) -> "Reduction":
        if self.source_paper_id is None and self.paper_id is not None:
            self.source_paper_id = self.paper_id
        return self


class OpenProblem(StrictBase):
    """A stored open problem, gap, or frontier question."""

    id: int | None = None
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    context: str | None = None
    status: ReviewStatus = "active"
    paper_id: int | None = None
    source_paper_id: int | None = None
    source_location: str | None = None
    cluster_id: int | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def fill_source_paper(self) -> "OpenProblem":
        if self.source_paper_id is None and self.paper_id is not None:
            self.source_paper_id = self.paper_id
        return self


class PendingEntry(StrictBase):
    """An extracted candidate awaiting human curation."""

    id: int | None = None
    entry_type: EntryType
    payload: dict[str, Any]
    source_text: str | None = None
    status: PendingStatus = "pending"
    duplicate_of: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DerivedResult(StrictBase):
    """A locally derived result or synthesis of known results."""

    id: int | None = None
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    proof_sketch: str | None = None
    status: ReviewStatus = "draft"
    cluster_id: int | None = None
    notes: str | None = None


class Conjecture(StrictBase):
    """A conjecture tracked for attack, examples, and proof attempts."""

    conjecture_id: int | None = Field(default=None, validation_alias=AliasChoices("conjecture_id", "id"))
    title: str | None = None
    statement: str = Field(min_length=1)
    cluster_id: int | None = None
    motivation: str | None = None
    related_theorems: list[int] = Field(default_factory=list)
    expected_status: ConjectureExpectedStatus = "unknown"
    confidence: Confidence = "needs_review"
    attack_plan: str | None = None
    possible_counterexamples: list[str] = Field(default_factory=list)
    status: ReviewStatus = "active"
    notes: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def fill_compatibility_fields(self) -> "Conjecture":
        if self.motivation is None and self.rationale is not None:
            self.motivation = self.rationale
        if not self.title:
            self.title = _title_from_statement(self.statement)
        return self

    @property
    def id(self) -> int | None:
        return self.conjecture_id


class ProofAttempt(StrictBase):
    """A proof attempt for a theorem, conjecture, or derived result."""

    id: int | None = None
    target_type: Literal["theorem", "conjecture", "derived_result"]
    target_id: int
    strategy: str = Field(min_length=1)
    notes: str | None = None
    status: ReviewStatus = "draft"
    cluster_id: int | None = None


class EvidenceSpan(StrictBase):
    """A precise evidence pointer for a research claim."""

    evidence_id: int | None = None
    paper_id: int
    entry_type: Literal["theorem", "model", "reduction", "open_problem", "derived_result", "concept"]
    entry_id: int
    page_start: int | None = None
    page_end: int | None = None
    quote_or_summary: str = Field(min_length=1)
    confidence: Confidence = "needs_review"
    notes: str | None = None


class ExperimentRun(StrictBase):
    """A stored run from a small checker, generator, or solver experiment."""

    run_id: int | None = None
    cluster_id: int | None = None
    experiment_type: str = Field(min_length=1)
    input_json: dict[str, Any] = Field(default_factory=dict)
    output_json: dict[str, Any] = Field(default_factory=dict)
    result_summary: str | None = None
    notes: str | None = None


SCHEMA_BY_ENTRY_TYPE: dict[EntryType, type[StrictBase]] = {
    "concept": Concept,
    "model": Model,
    "theorem": Theorem,
    "reduction": Reduction,
    "open_problem": OpenProblem,
    "conjecture_seed": Conjecture,
}


def _title_from_statement(statement: str) -> str:
    sentence = statement.replace("\n", " ").strip()
    if not sentence:
        return "Untitled result"
    return sentence[:80].rstrip(" .")
