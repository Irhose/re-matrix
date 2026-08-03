from __future__ import annotations
import json
from typing import Optional
from loguru import logger


from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_reasoning_model, supports_json_mode
from cancer_immunology_reasoner.models import (
    EdgeCase, ThoughtExperiment, QueryUnderstanding, Principle,
    HierarchyLevel, ConfidenceLevel
)


EDGE_CASE_PROMPT = """You are a deliberately adversarial cancer immunologist whose job is to stress-test predictions.

Your ONLY goal: find conditions under which the predicted outcome would fail, weaken, reverse, or behave unpredictably.

You have:
1. A query and its causal question
2. A predicted thought experiment's causal chain and outcome
3. L2 context modifiers (conditions that alter how mechanisms behave)
4. L3 known exceptions (documented resistance/evasion mechanisms)
5. All principles with citations

INSTRUCTIONS:
For each L2 and L3 principle, actively ask: "Does this change, weaken, or reverse the predicted outcome?"
Only keep ones that MATERIALLY DO.
Be aggressive — standard immunology reasoning is already baked into the prediction; your job is to find the cracks.

Consider specifically:
- Known resistance/evasion mechanisms (antigen loss, MHC-I downregulation, alternative pathway compensation)
- Tumor heterogeneity (only a subclone responds, others don't)
- Context variants (hypoxia, stromal density, immunocompromised states)
- Prior treatment history effects
- Combination therapy interactions
- Timing issues (too early, too late in disease progression)
- Population variability (age, microbiome, genetic background)
- "Cold tumor" scenarios where T cell infiltration is limited

For EACH edge case you identify, return:
{
  "condition": "the specific condition under which prediction breaks",
  "mechanism_of_deviation": "detailed biological mechanism of why it changes",
  "citation": "specific source citation",
  "principle_id": "ID of the principle that reveals this edge case",
  "hierarchy_level": "L2_context_modifier or L3_known_exception",
  "severity": "reverses | weakens | qualifies"
}

Return JSON:
{
  "edge_cases": [...]
}

If you find zero real edge cases, return {"edge_cases": []}. But in practice there are almost always edge cases."""


class EdgeCaseGenerator:
    def __init__(self, client = None):
        self.client = client or get_client()
        self.model = get_reasoning_model()
        self.json_mode = supports_json_mode()
    
    @staticmethod
    def _extract_json(text: str) -> str:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            return match.group(1)
        return text
    
    def generate(self, understanding: QueryUnderstanding,
                 thought_experiment: ThoughtExperiment,
                 principles_by_level: dict[HierarchyLevel, list[Principle]]) -> list[EdgeCase]:
        
        principles_text = self._format_principles(principles_by_level)
        
        prompt = f"""QUERY: {understanding.original_query}
CAUSAL QUESTION: {understanding.causal_question}

PREDICTED THOUGHT EXPERIMENT:
Initial Conditions: {thought_experiment.initial_conditions}
Intervention: {thought_experiment.intervention}
Predicted Outcome: {thought_experiment.predicted_outcome}
Outcome Confidence: {thought_experiment.outcome_confidence.value}

Causal Chain:
{self._format_causal_chain(thought_experiment)}

PRINCIPLES (with emphasis on L2 context modifiers and L3 known exceptions):
{principles_text}"""
        
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": EDGE_CASE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if self.json_mode:
            data = json.loads(content)
        else:
            data = json.loads(self._extract_json(content))
        
        edge_cases = []
        for ec in data.get("edge_cases", []):
            try:
                hl = HierarchyLevel.parse(ec.get("hierarchy_level", "L3_known_exception"))
            except (ValueError, AttributeError):
                hl = HierarchyLevel.L3_KNOWN_EXCEPTION
            
            edge_cases.append(EdgeCase(
                condition=ec["condition"],
                mechanism_of_deviation=ec["mechanism_of_deviation"],
                citation=ec.get("citation", "unknown"),
                principle_id=ec.get("principle_id", ""),
                hierarchy_level=hl,
                severity=ec.get("severity", "qualifies")
            ))
        
        logger.info(f"Generated {len(edge_cases)} edge cases")
        return edge_cases
    
    def _format_principles(self, principles_by_level: dict[HierarchyLevel, list[Principle]]) -> str:
        lines = []
        for level in [HierarchyLevel.L2_CONTEXT_MODIFIER, HierarchyLevel.L3_KNOWN_EXCEPTION,
                      HierarchyLevel.L1_MECHANISTIC_PATHWAY, HierarchyLevel.L0_AXIOM]:
            principles = principles_by_level.get(level, [])
            if not principles:
                continue
            lines.append(f"\n=== {level.value} ({len(principles)}) ===")
            for p in principles:
                lines.append(f"ID: {p.id}")
                lines.append(f"Content: {p.content}")
                lines.append(f"Entities: {', '.join(p.entities)}")
                lines.append(f"Citation: {p.source_citation}")
                lines.append("")
        return "\n".join(lines)
    
    def _format_causal_chain(self, te: ThoughtExperiment) -> str:
        lines = []
        for step in te.causal_chain:
            lines.append(f"  Step {step.step_number}:")
            lines.append(f"    Principle: {step.principle_content[:100]}...")
            lines.append(f"    Consequence: {step.mechanistic_consequence}")
            lines.append(f"    Confidence: {step.confidence.value}")
            lines.append(f"    Citation: {step.citation}")
            lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from cancer_immunology_reasoner.retrieval import PrincipleIndex, HierarchyAwareRetriever
    from cancer_immunology_reasoner.query_understanding import QueryUnderstander
    from cancer_immunology_reasoner.thought_experiment import ThoughtExperimentGenerator
    
    query = sys.argv[1] if len(sys.argv) > 1 else "What happens if we block PD-1 in a cold tumor with low TMB?"
    
    index = PrincipleIndex.load(Path("data/index/principles.json"))
    retriever = HierarchyAwareRetriever(index)
    u = QueryUnderstander()
    understanding = u.understand(query)
    principles_by_level = retriever.retrieve(understanding)
    
    gen = ThoughtExperimentGenerator()
    experiment = gen.generate(understanding, principles_by_level)
    
    ec_gen = EdgeCaseGenerator()
    edge_cases = ec_gen.generate(understanding, experiment, principles_by_level)
    
    for ec in edge_cases:
        print(f"  [{ec.severity}] {ec.condition} -> {ec.mechanism_of_deviation[:80]}... [{ec.citation}]")