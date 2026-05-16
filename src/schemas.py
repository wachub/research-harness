"""Typed Pydantic schemas for research artifacts stored by the harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EntryType = Literal["model", "theorem", "reduction", "open_problem"]
PendingStatus = Literal["pending", "approved", "rejected"]
ReviewStatus = Literal["draft", "active", "closed", "refuted", "proved"]
ModelType = Literal["ATS", "CDM", "2DM", "control", "distributed_synthesis", "other"]


class StrictBase(BaseModel):
    """Base schema with strict defaults useful for database boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Paper(StrictBase):
    """Bibliographic metadata for a paper or manuscript."""

    id: int | None = None
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1800, le=3000)
    venue: str | None = None
    pdf_path: str | None = None
    notes: str | None = None

    @field_validator("authors")
    @classmethod
    def authors_must_not_be_blank(cls, authors: list[str]) -> list[str]:
        cleaned = [author.strip() for author in authors if author.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty author is required")
        return cleaned


class Model(StrictBase):
    """Stored description of a formal model or generated game instance."""

    id: int | None = None
    name: str = Field(min_length=1)
    model_type: ModelType = "other"
    description: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    source_paper_id: int | None = None


class Theorem(StrictBase):
    """A theorem, lemma, proposition, or formal claim."""

    id: int | None = None
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    conclusion: str | None = None
    paper_id: int | None = None
    tags: list[str] = Field(default_factory=list)


class Reduction(StrictBase):
    """A reduction between formal problems."""

    id: int | None = None
    title: str = Field(min_length=1)
    source_problem: str = Field(min_length=1)
    target_problem: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    paper_id: int | None = None
    tags: list[str] = Field(default_factory=list)


class OpenProblem(StrictBase):
    """A stored open problem or research question."""

    id: int | None = None
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    context: str | None = None
    status: ReviewStatus = "active"
    paper_id: int | None = None
    tags: list[str] = Field(default_factory=list)


class PendingEntry(StrictBase):
    """An extracted candidate awaiting manual curation."""

    id: int | None = None
    entry_type: EntryType
    payload: dict[str, Any]
    source_text: str | None = None
    status: PendingStatus = "pending"
    duplicate_of: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DerivedResult(StrictBase):
    """A result derived inside the local research workflow."""

    id: int | None = None
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    proof_sketch: str | None = None
    status: ReviewStatus = "draft"


class Conjecture(StrictBase):
    """A conjecture tracked by the harness."""

    id: int | None = None
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    rationale: str | None = None
    status: ReviewStatus = "active"


class ProofAttempt(StrictBase):
    """A proof attempt for a theorem, conjecture, or derived result."""

    id: int | None = None
    target_type: Literal["theorem", "conjecture", "derived_result"]
    target_id: int
    strategy: str = Field(min_length=1)
    notes: str | None = None
    status: ReviewStatus = "draft"


SCHEMA_BY_ENTRY_TYPE: dict[EntryType, type[StrictBase]] = {
    "model": Model,
    "theorem": Theorem,
    "reduction": Reduction,
    "open_problem": OpenProblem,
}

