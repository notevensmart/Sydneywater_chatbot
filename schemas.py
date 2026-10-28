"""Small shared contracts for the local retrieval server and final answer."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Topic = Literal["all", "billing", "payments", "assistance", "account", "contact"]
SectionID = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchInput(Contract):
    query: str = Field(min_length=2, max_length=300)
    topic: Topic = "all"
    limit: int = Field(default=3, ge=1, le=4)

    @model_validator(mode="after")
    def valid_query(self):
        if len(self.query.strip()) < 2 or not any(c.isalnum() for c in self.query) or any(ord(c) < 32 for c in self.query):
            raise ValueError("Use a nonempty plain-text search query")
        self.query = self.query.strip()
        return self


class ReadInput(Contract):
    section_id: SectionID


class Passage(Contract):
    evidence_id: str = Field(min_length=1, max_length=110)
    document_id: str = Field(min_length=1, max_length=60)
    section_id: SectionID
    title: str = Field(min_length=1, max_length=250)
    heading: str = Field(min_length=1, max_length=300)
    canonical_url: str = Field(min_length=1, max_length=500)
    fetched_at: str = Field(min_length=1, max_length=40)
    reviewed_at: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=8000)
    is_excerpt: bool
    related_section_ids: list[SectionID] = Field(default_factory=list, max_length=3)


class Evidence(Contract):
    schema_version: Literal["1"] = "1"
    status: Literal["ok", "no_results", "not_found"]
    passages: list[Passage] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "ok") != bool(self.passages):
            raise ValueError("Evidence status and passages disagree")
        if sum(len(p.text) for p in self.passages) > 8000:
            raise ValueError("Evidence exceeds the tool text budget")
        return self


class AnswerBlock(Contract):
    text: str = Field(min_length=1, max_length=500, description="One plain-text, source-supported point; no URLs or markup")
    evidence_ids: list[str] = Field(min_length=1, max_length=3, description="Exact retrieved evidence_id values including :excerpt or :full suffix; never section_id alone")

    @model_validator(mode="after")
    def unique_citations(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Duplicate citation")
        return self


class Answer(Contract):
    """Finish the user turn. Call this as soon as retrieved evidence answers the question,
    or when clarification/unsupported/insufficient evidence is appropriate. Do not keep
    searching after the required facts and conditions have been retrieved.
    """

    kind: Literal["answer", "clarify", "unsupported", "insufficient_evidence"]
    blocks: list[AnswerBlock] = Field(default_factory=list, max_length=6)
    clarification: str = Field(default="", max_length=250)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "answer":
            if not self.blocks or self.clarification:
                raise ValueError("Answers require cited blocks and no clarification")
        elif self.kind == "clarify":
            if self.blocks or not self.clarification.strip():
                raise ValueError("Clarify requires only a question")
        elif self.blocks or self.clarification:
            raise ValueError("Fallback text is owned by the application")
        return self
