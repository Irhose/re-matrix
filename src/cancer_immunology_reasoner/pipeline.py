from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
from loguru import logger
from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_reasoning_model, supports_json_mode
from cancer_immunology_reasoner.models import (
    ConversationState, ReasoningReport, QueryUnderstanding, Principle,
    HierarchyLevel, QueryType, ThoughtExperiment, CausalStep, ConfidenceLevel,
    EdgeCase
)
from cancer_immunology_reasoner.query_understanding import QueryUnderstander
from cancer_immunology_reasoner.retrieval import HierarchyAwareRetriever, PrincipleIndex
from cancer_immunology_reasoner.thought_experiment import ThoughtExperimentGenerator
from cancer_immunology_reasoner.edge_cases import EdgeCaseGenerator
from cancer_immunology_reasoner.report import ReportGenerator, format_report


class Position(TypedDict):
    line: int


class FeedbackTarget(TypedDict, total=False):
    causal_step: int
    causal_link: str
    additional_context: str
    disputed_claim: str


class ReasoningPipeline:
    """Main pipeline orchestrating Stages 2-6.
    
    Stage 1 (ingestion) is handled separately as a batch job.
    """
    
    def __init__(self, index_path: Path):
        self.index = PrincipleIndex.load(index_path)
        self.retriever = HierarchyAwareRetriever(self.index)
        self.query_understander = QueryUnderstander()
        self.thought_generator = ThoughtExperimentGenerator()
        self.edge_generator = EdgeCaseGenerator()
        self.report_generator = ReportGenerator()
        self.client = get_client()
        self.reasoning_model = get_reasoning_model()
        self.json_mode = supports_json_mode()
    
    def run(self, query: str) -> tuple[ReasoningReport, ConversationState]:
        """Run full pipeline: Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6."""
        logger.info(f"Processing query: {query}")
        
        # Stage 2: Query understanding
        understanding = self.query_understander.understand(query)
        logger.info(f"Query type: {understanding.query_type.value}")
        
        # Stage 3: Hierarchy-aware retrieval
        principles_by_level = self.retriever.retrieve(understanding)
        
        # Collect all principles into flat list
        all_principles = []
        for principles in principles_by_level.values():
            all_principles.extend(principles)
        
        # Stage 4: Thought experiment generation
        thought_experiment = self.thought_generator.generate(
            understanding, principles_by_level
        )
        
        # Stage 5: Edge case generation (separate, adversarial call)
        edge_cases = self.edge_generator.generate(
            understanding, thought_experiment, principles_by_level
        )
        
        # Stage 6: Report generation
        report = self.report_generator.generate(
            understanding, dict(principles_by_level),
            thought_experiment, edge_cases
        )
        
        # Package conversation state
        state = ConversationState(
            query=query,
            understanding=understanding,
            retrieval_set=all_principles,
            report=report
        )
        
        return report, state
    
    def refine(self, state: ConversationState, feedback: str,
               feedback_target: Optional[FeedbackTarget] = None) -> tuple[ReasoningReport, ConversationState]:
        """Stage 7: Refine based on researcher feedback."""
        
        # Record feedback
        state.history.append({"role": "researcher", "content": feedback})
        
        if feedback_target and feedback_target.get("additional_context"):
            # Re-run retrieval with added context
            new_context = feedback_target["additional_context"]
            state.understanding.context["user_specified"] = new_context
            
            # Update causal question with context
            state.understanding.causal_question = (
                f"{state.understanding.causal_question} (with added context: {new_context})"
            )
            
            # Re-run Stage 3-6
            principles_by_level = self.retriever.retrieve(state.understanding)
            
            all_principles = []
            for principles in principles_by_level.values():
                all_principles.extend(principles)
            
            thought_experiment = self.thought_generator.generate(
                state.understanding, principles_by_level
            )
            
            edge_cases = self.edge_generator.generate(
                state.understanding, thought_experiment, principles_by_level
            )
            
            report = self.report_generator.generate(
                state.understanding, dict(principles_by_level),
                thought_experiment, edge_cases
            )
            
            # Update state
            state.retrieval_set = all_principles
            state.report = report
            state.history.append({"role": "system", "content": "Regenerated with context"})
            
        elif feedback_target and feedback_target.get("disputed_claim"):
            # Focused re-run: challenge a specific claim
            # For now, re-run edge case generation pointing at the disputed step
            disputed_step_num = feedback_target.get("causal_step")
            
            # Re-run edge case generation with explicit focus on disputed claim
            edge_cases = self._challenge_step(state, disputed_step_num, feedback)
            state.report.edge_cases = edge_cases
            state.history.append({
                "role": "system",
                "content": f"Edge cases regenerated focusing on step {disputed_step_num}"
            })
        else:
            # General re-run of thought experiment with feedback as additional context
            state.understanding.context["feedback"] = feedback
            principles_by_level = self.retriever.retrieve(state.understanding)
            
            thought_experiment = self.thought_generator.generate(
                state.understanding, principles_by_level
            )
            
            state.report.thought_experiment = thought_experiment
            state.history.append({"role": "system", "content": "Thought experiment regenerated"})
        
        return state.report, state
    
    def _challenge_step(self, state: ConversationState, step_num: int,
                        challenge: str) -> list[EdgeCase]:
        """Re-run edge case generation focused on a disputed step."""
        from cancer_immunology_reasoner.edge_cases import EDGE_CASE_PROMPT
        
        disputed_step = None
        for step in state.report.thought_experiment.causal_chain:
            if step.step_number == step_num:
                disputed_step = step
                break
        
        if not disputed_step:
            logger.warning(f"Step {step_num} not found in causal chain")
            return state.report.edge_cases
        
        prompt = f"""The researcher has challenged Step {step_num} of the reasoning:

Step {step_num} details:
- Principle: {disputed_step.principle_content}
- Citation: {disputed_step.citation}
- Consequence: {disputed_step.mechanistic_consequence}

Challenge: {challenge}

Your task: Re-examine the evidence. Find ANY edge cases, counterexamples, or conditions under which this specific causal link would not hold. Be aggressive."""

        import json
        kwargs = dict(
            model=self.reasoning_model,
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
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            data = json.loads(match.group() if match else content)
        edge_cases = []
        for ec in data.get("edge_cases", []):
            try:
                hl = HierarchyLevel(ec.get("hierarchy_level", "L3_known_exception"))
            except ValueError:
                hl = HierarchyLevel.L3_KNOWN_EXCEPTION
            
            edge_cases.append(EdgeCase(
                condition=ec["condition"],
                mechanism_of_deviation=ec["mechanism_of_deviation"],
                citation=ec.get("citation", "unknown"),
                principle_id=ec.get("principle_id", ""),
                hierarchy_level=hl,
                severity=ec.get("severity", "qualifies")
            ))
        
        # Append to existing edge cases (keep originals too)
        existing_ids = {id(ec): ec for ec in state.report.edge_cases}
        new_edge_cases = list(state.report.edge_cases)
        for ec in edge_cases:
            if ec.condition not in {x.condition for x in new_edge_cases}:
                new_edge_cases.append(ec)
        
        return new_edge_cases


def save_conversation(state: ConversationState, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "conversation_id": state.conversation_id,
            "query": state.query,
            "history": state.history,
            "report": state.report.model_dump()
        }, f, indent=2)


def load_conversation(path: Path) -> ConversationState:
    with open(path, "r") as f:
        data = json.load(f)
    
    state = ConversationState(
        conversation_id=data["conversation_id"],
        query=data["query"],
        understanding=QueryUnderstanding(
            query_type=QueryType.MECHANISTIC,
            entities={},
            causal_question=data["query"],
            original_query=data["query"]
        ),
        retrieval_set=[],
        report=ReasoningReport(**data["report"]),
        history=data.get("history", [])
    )
    return state