from __future__ import annotations
import json
import re
from typing import Optional

from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_fast_model, supports_json_mode
from cancer_immunology_reasoner.models import (
    ReasoningReport, ThoughtExperiment, EdgeCase, QueryUnderstanding, Principle,
    HierarchyLevel, ConfidenceLevel
)


SUGGESTED_FOLLOWUPS_PROMPT = """Given the query, predicted outcome, edge cases, and assumptions flagged during reasoning, suggest specific follow-up questions or experiments that would resolve remaining uncertainty.

Focus on experiments or analyses that would actually distinguish between the predicted outcome and the edge cases surfaced.

Return JSON:
{
  "suggested_followups": [
    "string - specific question or experiment",
    ...
  ]
}"""


class ReportGenerator:
    def __init__(self, client = None):
        self.client = client or get_client()
        self.model = get_fast_model()
        self.json_mode = supports_json_mode()
    
    def generate(self, understanding: QueryUnderstanding,
                 principles_by_level: dict[HierarchyLevel, list[Principle]],
                 thought_experiment: ThoughtExperiment,
                 edge_cases: list[EdgeCase]) -> ReasoningReport:
        
        # Collect flagged assumptions from thought experiment
        assumptions = list(thought_experiment.flagged_assumptions)
        for step in thought_experiment.causal_chain:
            for a in step.assumptions:
                if a not in assumptions:
                    assumptions.append(a)
        
        # Generate follow-up suggestions
        followups = self._generate_followups(
            understanding, thought_experiment, edge_cases, assumptions
        )
        
        # Group principles by level for report
        principles_by_level_dict = {
            level.value: principles
            for level, principles in principles_by_level.items()
        }
        
        return ReasoningReport(
            query_restatement=understanding.original_query,
            principles_by_level=principles_by_level_dict,
            thought_experiment=thought_experiment,
            edge_cases=edge_cases,
            flagged_assumptions=assumptions,
            suggested_followups=followups
        )
    
    def _generate_followups(self, understanding, te, edge_cases, assumptions) -> list[str]:
        prompt = f"""Query: {understanding.original_query}
Predicted outcome: {te.predicted_outcome} (confidence: {te.outcome_confidence.value})
Edge cases: {[ec.condition for ec in edge_cases]}
Flagged assumptions: {assumptions}"""
        
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": SUGGESTED_FOLLOWUPS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if self.json_mode:
            data = json.loads(content)
        else:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            data = json.loads(match.group() if match else content)
        return data.get("suggested_followups", [])


def format_report(report: ReasoningReport) -> str:
    """Format the report as a beautiful structured text output."""
    lines = []
    sep = "=" * 78
    
    lines.append(sep)
    lines.append("CANCER IMMUNOLOGY CAUSAL REASONING REPORT")
    lines.append(sep)
    
    # Query
    lines.append("\n1. QUERY RESTATEMENT")
    lines.append("-" * 50)
    lines.append(f"   {report.query_restatement}")
    
    # Principles
    lines.append(f"\n2. PRINCIPLES INVOKED")
    lines.append("-" * 50)
    
    level_labels = {
        "L0_axiom": "L0 - Axioms (Foundational Principles)",
        "L1_mechanistic_pathway": "L1 - Mechanistic Pathways",
        "L2_context_modifier": "L2 - Context Modifiers",
        "L3_known_exception": "L3 - Known Exceptions / Resistance Mechanisms"
    }
    
    for level_name, label in level_labels.items():
        principles = report.principles_by_level.get(level_name, [])
        if not principles:
            continue
        lines.append(f"\n   [{label}]")
        for p in principles:
            lines.append(f"      * {p.content}")
            lines.append(f"        Source: {p.source_citation}")
    
    # Thought experiment
    te = report.thought_experiment
    lines.append(f"\n3. THOUGHT EXPERIMENT")
    lines.append("-" * 50)
    lines.append(f"   Initial Conditions: {te.initial_conditions}")
    lines.append(f"   Intervention: {te.intervention}")
    lines.append(f"   Held Constant: {', '.join(te.held_constant)}")
    
    lines.append(f"\n   Causal Chain:")
    for step in te.causal_chain:
        lines.append(f"\n   Step {step.step_number}:")
        lines.append(f"      Principle: {step.principle_content}")
        lines.append(f"      Citation: {step.citation}")
        lines.append(f"      Consequence: {step.mechanistic_consequence}")
        lines.append(f"      Confidence: {step.confidence.value.upper()}")
        if step.assumptions:
            lines.append(f"      Assumptions: {', '.join(step.assumptions)}")
    
    # Predicted outcome
    lines.append(f"\n   PREDICTED OUTCOME:")
    lines.append(f"      {te.predicted_outcome}")
    lines.append(f"      Confidence: {te.outcome_confidence.value.upper()}")
    
    # Edge cases
    lines.append(f"\n4. EDGE CASES")
    lines.append("-" * 50)
    if report.edge_cases:
        for i, ec in enumerate(report.edge_cases, 1):
            severity_label = {"reverses": "REVERSES", "weakens": "WEAKENS", "qualifies": "QUALIFIES"}
            lines.append(f"\n   Edge Case #{i} [{severity_label.get(ec.severity, ec.severity)}]")
            lines.append(f"      Condition: {ec.condition}")
            lines.append(f"      Mechanism: {ec.mechanism_of_deviation}")
            lines.append(f"      Source: {ec.citation}")
    else:
        lines.append("   (None identified - prediction is robust across known conditions)")
    
    # Flagged assumptions
    lines.append(f"\n5. FLAGGED ASSUMPTIONS / UNDERSPECIFIED VARIABLES")
    lines.append("-" * 50)
    if report.flagged_assumptions:
        for a in report.flagged_assumptions:
            lines.append(f"   * {a}")
    else:
        lines.append("   (None flagged)")
    
    # Suggested followups
    lines.append(f"\n6. SUGGESTED FOLLOW-UP QUESTIONS / EXPERIMENTS")
    lines.append("-" * 50)
    for i, f in enumerate(report.suggested_followups, 1):
        lines.append(f"   {i}. {f}")
    
    lines.append(f"\n{sep}")
    
    return "\n".join(lines)