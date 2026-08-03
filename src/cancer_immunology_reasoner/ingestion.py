from __future__ import annotations
import json
import pdfplumber
from pathlib import Path
from typing import Optional
from loguru import logger


from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_fast_model, supports_json_mode
from cancer_immunology_reasoner.models import (
    Principle, Domain, HierarchyLevel
)


EXTRACTION_PROMPT = """You are an expert cancer immunologist extracting structured principles from scientific text.

Given a text chunk, extract ONE principle per call. If the chunk contains multiple distinct principles, extract the most central one and note others exist.

Return JSON with exactly these fields:
{
  "content": "The principle stated as a single, self-contained causal/mechanistic claim",
  "domain": "immunology | cancer_pathogenesis | intersection",
  "hierarchy_level": "L0_axiom | L1_mechanistic_pathway | L2_context_modifier | L3_known_exception",
  "entities": ["cell_type1", "molecule1", "pathway1", "receptor1", ...],
  "depends_on": ["description of principle this builds on", ...],
  "source_citation": "document_name:page_or_section"
}

Hierarchy level guide:
- L0_axiom: Near-universal, rarely context-dependent (e.g., "T-cell activation requires TCR engagement + costimulation")
- L1_mechanistic_pathway: Specific causal pathway built on axioms (e.g., "CTLA-4 outcompetes CD28 for B7 ligands, dampening costimulation")
- L2_context_modifier: Conditions altering L0/L1 behavior (e.g., "In hypoxic TME, HIF-1α upregulates PD-L1")
- L3_known_exception: Documented cases where expected behavior fails/reverses (e.g., "MHC-I loss variants evade CD8+ T cells entirely, making checkpoint blockade ineffective")

Extract entities as specific named cells, molecules, pathways, receptors (e.g., "CD8+ T cell", "PD-1", "TCR signaling", "IFN-γ", "JAK/STAT").

depends_on should reference OTHER principles by their conceptual description, not IDs (we'll link later).

Only return the JSON object. No extra text."""


class PrincipleExtractor:
    def __init__(self, client = None):
        self.client = client or get_client()
        self.model = get_fast_model()
        self.json_mode = supports_json_mode()

    def extract_from_chunk(self, chunk_text: str, source_citation: str) -> Optional[Principle]:
        """Extract a single principle from a text chunk."""
        try:
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Source: {source_citation}\n\nText:\n{chunk_text}"}
                ],
                temperature=0.1,
            )
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            # Extract JSON from content if not in json mode
            if self.json_mode:
                data = json.loads(content)
            else:
                data = json.loads(self._extract_json(content))

            if not isinstance(data, dict) or "content" not in data:
                logger.warning(f"Invalid extraction result: {str(data)[:100]}")
                return None

            # Validate hierarchy level
            try:
                hierarchy_level = HierarchyLevel(data["hierarchy_level"])
            except ValueError:
                logger.warning(f"Invalid hierarchy level: {data['hierarchy_level']}, defaulting to L1")
                hierarchy_level = HierarchyLevel.L1_MECHANISTIC_PATHWAY

            # Validate domain
            try:
                raw_domain = data["domain"]
                # Handle cases where LLM returns "immunology | cancer_pathogenesis" etc.
                if "|" in raw_domain:
                    # Pick the first valid domain
                    for part in raw_domain.split("|"):
                        part = part.strip()
                        try:
                            domain = Domain(part)
                            break
                        except ValueError:
                            continue
                    else:
                        domain = Domain.INTERSECTION
                elif raw_domain in ("both", "all"):
                    domain = Domain.INTERSECTION
                else:
                    domain = Domain(raw_domain)
            except ValueError:
                logger.warning(f"Invalid domain: {data['domain']}, defaulting to intersection")
                domain = Domain.INTERSECTION

            principle = Principle(
                content=data["content"].strip(),
                domain=domain,
                hierarchy_level=hierarchy_level,
                entities=data.get("entities", []),
                depends_on=data.get("depends_on", []),
                source_citation=source_citation
            )
            return principle

        except Exception as e:
            logger.error(f"Extraction failed for {source_citation}: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object from text that may contain markdown or surrounding text."""
        import re
        # Try to find a JSON object in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group()
        # Fallback: try to find json block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            return match.group(1)
        return text


def chunk_by_conceptual_unit(text: str, source_name: str, page_num: int) -> list[tuple[str, str]]:
    """
    Split text into conceptual units rather than fixed windows.
    For now, use paragraphs as a proxy; could be enhanced with LLM-based chunking.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > settings.chunk_size and current_chunk:
            citation = f"{source_name}:page_{page_num}"
            chunks.append((current_chunk.strip(), citation))
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk:
        citation = f"{source_name}:page_{page_num}"
        chunks.append((current_chunk.strip(), citation))
    
    return chunks


def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from PDF, returning list of (page_num, text)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i + 1, text))
    return pages


def ingest_corpus(corpus_dir: Path, output_path: Path) -> list[Principle]:
    """
    Ingest all PDFs in corpus_dir, extract principles, save to output_path.
    """
    extractor = PrincipleExtractor()
    all_principles = []
    
    pdf_files = list(corpus_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {corpus_dir}")
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}...")
        pages = extract_text_from_pdf(pdf_path)
        
        for page_num, page_text in pages:
            chunks = chunk_by_conceptual_unit(page_text, pdf_path.stem, page_num)
            
            for chunk_text, citation in chunks:
                principle = extractor.extract_from_chunk(chunk_text, citation)
                if principle:
                    all_principles.append(principle)
                    logger.debug(f"Extracted: {principle.content[:80]}... [{principle.hierarchy_level.value}]")
    
    # Save principles
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([p.model_dump() for p in all_principles], f, indent=2)
    
    logger.info(f"Extracted {len(all_principles)} principles, saved to {output_path}")
    return all_principles


if __name__ == "__main__":
    import sys
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/corpus")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/index/principles.json")
    ingest_corpus(corpus_dir, output_path)