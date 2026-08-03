from __future__ import annotations
import json
from typing import Optional
from loguru import logger


from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_reasoning_model, supports_json_mode
from cancer_immunology_reasoner.models import (
    ThoughtExperiment, CausalStep, QueryUnderstanding, Principle, 
    HierarchyLevel, ConfidenceLevel
)


THOUGHT_EXPERIMENT_PROMPT = """You are an expert cancer immunologist conducting a rigorous causal simulation.

Given:
- A research query with initial conditions and intervention
- A set of principles organized by hierarchy level (L0 axioms → L1 pathways → L2 context modifiers → L3 exceptions)
- Each principle has a citation to source material

Your task: Construct a step-by-step causal chain simulating what happens.

RULES:
1. State initial conditions explicitly: what is present, what intervention/perturbation is introduced, what is held constant
2. For EACH step in the causal chain:
   - Name the principle invoked (with its citation)
   - State the mechanistic consequence
   - Carry that consequence forward as input to the next step
   - Make every inferential link explicit — NO jumping to conclusions
3. Predict the most likely outcome with an explicit confidence qualifier grounded in the evidence:
   - "high confidence — supported by L0/L1 principles"
   - "medium confidence — supported by L1/L2 principles"
   - "low confidence — extrapolated from limited direct evidence"
   - "speculative — beyond direct evidence, theoretical extrapolation"
4. Explicitly flag ALL assumptions made because the query underspecified them (e.g., tumor mutational burden, timing, patient immune status, tumor type specifics)

Return JSON:
{
  "initial_conditions": "string",
  "intervention": "string",
  "held_constant": ["string", ...],
  "causal_chain": [
    {
      "step_number": 1,
      "principle_id": "principle UUID",
      "principle_content": "the principle text",
      "citation": "source citation",
      "mechanistic_consequence": "what happens at this step",
      "confidence": "high|medium|low|speculative",
      "assumptions": ["assumption1", ...]
    },
    ...
  ],
  "predicted_outcome": "string",
  "outcome_confidence": "high|medium|low|speculative",
  "flagged_assumptions": ["string", ...]
}"""


class ThoughtExperimentGenerator:
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
                 principles_by_level: dict[HierarchyLevel, list[Principle]]) -> ThoughtExperiment:
        
        principles_text = self._format_principles(principles_by_level)
        
        prompt = f"""Query: {understanding.original_query}
Causal Question: {understanding.causal_question}
Context: {json.dumps(understanding.context, indent=2)}
Entities: {json.dumps(understanding.entities, indent=2)}

Principles:
{principles_text}"""
        
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": THOUGHT_EXPERIMENT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if self.json_mode:
            data = json.loads(content)
        else:
            data = json.loads(self._extract_json(content))
        
        causal_chain = []
        for step in data["causal_chain"]:
            causal_chain.append(CausalStep(
                step_number=step["step_number"],
                principle_id=step["principle_id"],
                principle_content=step["principle_content"],
                citation=step["citation"],
                mechanistic_consequence=step["mechanistic_consequence"],
                confidence=ConfidenceLevel.parse(step["confidence"]),
                assumptions=step.get("assumptions", [])
            ))
        
        return ThoughtExperiment(
            initial_conditions=data["initial_conditions"],
            intervention=data["intervention"],
            held_constant=data.get("held_constant", []),
            causal_chain=causal_chain,
            predicted_outcome=data["predicted_outcome"],
            outcome_confidence=ConfidenceLevel.parse(data["outcome_confidence"]),
            flagged_assumptions=data.get("flagged_assumptions", [])
        )
    
    def _format_principles(self, principles_by_level: dict[HierarchyLevel, list[Principle]]) -> str:
        lines = []
        for level in HierarchyLevel:
            principles = principles_by_level.get(level, [])
            if not principles:
                continue
            lines.append(f"\n=== {level.value} ({len(principles)}) ===")
            for p in principles:
                lines.append(f"ID: {p.id}")
                lines.append(f"Content: {p.content}")
                lines.append(f"Entities: {', '.join(p.entities)}")
                lines.append(f"Citation: {p.source_citation}")
                lines.append(f"Depends on: {', '.join(p.depends_on) if p.depends_on else 'none'}")
                lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from cancer_immunology_reasoner.retrieval import PrincipleIndex, HierarchyAwareRetriever
    from cancer_immunology_reasoner.query_understanding import QueryUnderstander
    
    query = sys.argv[1] if len(sys.argv) > 1 else "What happens if we block PD-1 in a cold tumor with low TMB?"
    
    index = PrincipleIndex.load(Path("data/index/principles.json"))
    retriever = HierarchyAwareRetriever(index)
    u = QueryUnderstander()
    understanding = u.understand(query)
    principles_by_level = retriever.retrieve(understanding)
    
    gen = ThoughtExperimentGenerator()
    experiment = gen.generate(understanding, principles_by_level)
    
    print(experiment.model_dump_json(indent=2))