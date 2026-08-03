from __future__ import annotations
import json
import re
from typing import Optional
from loguru import logger

from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_fast_model, supports_json_mode
from cancer_immunology_reasoner.models import (
    QueryUnderstanding, QueryType
)


QUERY_UNDERSTANDING_PROMPT = """You are an expert cancer immunologist analyzing a researcher's query.

Classify the query type:
- mechanistic: "what happens if X binds Y?" / "how does X lead to Y?"
- therapeutic_hypothesis: "would blocking X improve outcomes in condition Y?"
- edge_case_exploration: "under what conditions would X fail?" / "when does Y not work?"
- comparative: "how does mechanism A differ from B in context C?"

Extract entities by category:
- cell_types: ["CD8+ T cell", "NK cell", "dendritic cell", ...]
- molecules: ["PD-1", "PD-L1", "CTLA-4", "IFN-γ", "MHC-I", ...]
- pathways: ["TCR signaling", "PD-1/PD-L1", "JAK/STAT", "antigen presentation", ...]
- interventions: ["anti-PD-1", "CTLA-4 blockade", "CAR-T", "knockout", ...]
- contexts: ["melanoma", "hypoxic TME", "post-chemotherapy", "cold tumor", ...]
- tumor_types: ["NSCLC", "melanoma", "pancreatic", ...]

Extract the implied causal question: What is the researcher actually asking to predict?
(e.g., "if we knock out gene X in cell type Y, what happens to tumor growth and immune infiltration over time?")

Return JSON:
{
  "query_type": "mechanistic | therapeutic_hypothesis | edge_case_exploration | comparative",
  "entities": {
    "cell_types": [...],
    "molecules": [...],
    "pathways": [...],
    "interventions": [...],
    "contexts": [...],
    "tumor_types": [...]
  },
  "causal_question": "string",
  "context": {"key": "value", ...}
}"""


class QueryUnderstander:
    def __init__(self, client = None):
        self.client = client or get_client()
        self.model = get_fast_model()
        self.json_mode = supports_json_mode()

    def understand(self, query: str) -> QueryUnderstanding:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": QUERY_UNDERSTANDING_PROMPT},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.1,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if self.json_mode:
            data = json.loads(content)
        else:
            data = json.loads(self._extract_json(content))
        
        return QueryUnderstanding(
            query_type=QueryType(data["query_type"]),
            entities=data.get("entities", {}),
            causal_question=data.get("causal_question", query),
            context=data.get("context", {}),
            original_query=query
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            return match.group(1)
        return text


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What happens if we block PD-1 in a cold tumor with low TMB?"
    u = QueryUnderstander()
    result = u.understand(query)
    print(result.model_dump_json(indent=2))