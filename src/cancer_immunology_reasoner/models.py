from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class Domain(str, Enum):
    IMMUNOLOGY = "immunology"
    CANCER_PATHOGENESIS = "cancer_pathogenesis"
    INTERSECTION = "intersection"


class HierarchyLevel(str, Enum):
    L0_AXIOM = "L0_axiom"
    L1_MECHANISTIC_PATHWAY = "L1_mechanistic_pathway"
    L2_CONTEXT_MODIFIER = "L2_context_modifier"
    L3_KNOWN_EXCEPTION = "L3_known_exception"

    @classmethod
    def parse(cls, value: str) -> "HierarchyLevel":
        if not value:
            return cls.L1_MECHANISTIC_PATHWAY
        v = value.lower().replace(" ", "_")
        for member in cls:
            if member.value.lower() in v or member.name.lower() in v:
                return member
        if "l3" in v or "exception" in v or "resistance" in v:
            return cls.L3_KNOWN_EXCEPTION
        if "l2" in v or "context" in v or "modifier" in v:
            return cls.L2_CONTEXT_MODIFIER
        if "l0" in v or "axiom" in v:
            return cls.L0_AXIOM
        return cls.L1_MECHANISTIC_PATHWAY


class QueryType(str, Enum):
    MECHANISTIC = "mechanistic"
    THERAPEUTIC_HYPOTHESIS = "therapeutic_hypothesis"
    EDGE_CASE_EXPLORATION = "edge_case_exploration"
    COMPARATIVE = "comparative"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"

    @classmethod
    def parse(cls, value: str) -> "ConfidenceLevel":
        """Robustly parse a confidence string that may be verbose."""
        if not value:
            return cls.MEDIUM
        v = value.lower()
        if "high" in v:
            return cls.HIGH
        if "medium" in v or "moderate" in v:
            return cls.MEDIUM
        if "speculative" in v or "beyond direct" in v or "extrapolat" in v:
            return cls.SPECULATIVE
        if "low" in v:
            return cls.LOW
        try:
            return cls(value.lower())
        except ValueError:
            return cls.MEDIUM


class Principle(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    domain: Domain
    hierarchy_level: HierarchyLevel
    entities: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)  # IDs of principles this depends on
    source_citation: str  # document + location
    embedding: list[float] | None = None


class QueryUnderstanding(BaseModel):
    query_type: QueryType
    entities: dict[str, list[str]] = Field(default_factory=dict)  # category -> list of entities
    causal_question: str
    context: dict[str, str] = Field(default_factory=dict)
    original_query: str


class CausalStep(BaseModel):
    step_number: int
    principle_id: str
    principle_content: str
    citation: str
    mechanistic_consequence: str
    confidence: ConfidenceLevel
    assumptions: list[str] = Field(default_factory=list)


class ThoughtExperiment(BaseModel):
    initial_conditions: str
    intervention: str
    held_constant: list[str] = Field(default_factory=list)
    causal_chain: list[CausalStep]
    predicted_outcome: str
    outcome_confidence: ConfidenceLevel
    flagged_assumptions: list[str] = Field(default_factory=list)


class EdgeCase(BaseModel):
    condition: str
    mechanism_of_deviation: str
    citation: str
    principle_id: str
    hierarchy_level: HierarchyLevel
    severity: str  # "reverses", "weakens", "qualifies"


class ReasoningReport(BaseModel):
    query_restatement: str
    principles_by_level: dict[str, list[Principle]]  # hierarchy_level -> principles
    thought_experiment: ThoughtExperiment
    edge_cases: list[EdgeCase]
    flagged_assumptions: list[str]
    suggested_followups: list[str]


class ConversationState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    understanding: QueryUnderstanding
    retrieval_set: list[Principle]
    report: ReasoningReport
    history: list[dict] = Field(default_factory=list)  # for feedback loop