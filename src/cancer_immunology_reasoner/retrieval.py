from __future__ import annotations
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from collections import defaultdict
from loguru import logger

from cancer_immunology_reasoner.config import settings
from cancer_immunology_reasoner.llm_client import get_client, get_embedding_model, try_ollama_embedding
from cancer_immunology_reasoner.models import Principle, HierarchyLevel, QueryUnderstanding


class PrincipleIndex:
    """Vector index + dependency graph for principles."""
    
    def __init__(self, principles: list[Principle], client = None):
        self.principles = {p.id: p for p in principles}
        self.client = client or get_client()
        self.emb_model, self.emb_dim = get_embedding_model()
        self._build_index()
        self._build_dependency_graph()
    
    def _build_index(self):
        """Build FAISS index from principle embeddings."""
        texts = []
        ids = []
        for p in self.principles.values():
            if p.embedding is None:
                texts.append(p.content)
                ids.append(p.id)
        
        if texts:
            logger.info(f"Generating embeddings for {len(texts)} principles...")
            embeddings = self._generate_embeddings(texts)
            if embeddings:
                for pid, emb in zip(ids, embeddings):
                    self.principles[pid].embedding = emb
        
        embeddings = []
        valid_ids = []
        for pid, p in self.principles.items():
            if p.embedding is not None:
                embeddings.append(p.embedding)
                valid_ids.append(pid)
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
            self.index.add(embeddings_array)
            self.index_ids = valid_ids
        else:
            self.index = None
            self.index_ids = []

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings using available backend."""
        # Try Ollama embedding endpoint first
        ollama_embs = try_ollama_embedding(self.emb_model, texts, self.client)
        if ollama_embs:
            return ollama_embs
        # Fallback: use OpenAI-compatible embeddings endpoint
        try:
            response = self.client.embeddings.create(
                model=self.emb_model,
                input=texts
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Last resort: use dummy embeddings (not ideal but allows testing)
            logger.warning("Using zero embeddings as fallback — retrieval will be degraded")
            dim = self.emb_dim
            return [[0.0] * dim for _ in texts]
    
    def _build_dependency_graph(self):
        """Build adjacency list from depends_on references.
        
        Multi-strategy: exact content match, entity overlap, token similarity,
        then embedding-based semantic match as the robust fallback.
        """
        import re
        principles_list = list(self.principles.values())
        
        def tokens(text: str) -> set[str]:
            stop = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to",
                    "with", "by", "that", "this", "which", "is", "are", "via",
                    "from", "as", "at", "be", "it", "its"}
            return {w for w in re.findall(r'[a-z0-9+\-]+', text.lower()) if w not in stop}
        
        def jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)
        
        self.depends_on = defaultdict(list)
        self.dependents = defaultdict(list)
        
        # Embedding index for semantic fallback
        embeddings_array = None
        if self.index is not None and self.index_ids:
            embeddings_array = np.array(
                [self.principles[pid].embedding for pid in self.index_ids],
                dtype=np.float32
            )
        
        # Collect all depends_on descriptions that need matching
        dep_descs = []
        for p in principles_list:
            for dep_desc in p.depends_on:
                dep_descs.append(dep_desc)
        
        dep_embeddings = None
        if dep_descs and embeddings_array is not None:
            try:
                ollama_embs = try_ollama_embedding(self.emb_model, dep_descs, self.client)
                if ollama_embs:
                    dep_embeddings = np.array(ollama_embs, dtype=np.float32)
                else:
                    resp = self.client.embeddings.create(model=self.emb_model, input=dep_descs)
                    dep_embeddings = np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Dependency embedding failed: {e}")
        
        dep_idx = 0
        for p in principles_list:
            for dep_desc in p.depends_on:
                dep_lower = dep_desc.lower().strip()
                dep_tokens = tokens(dep_desc)
                best_id = None
                best_score = 0.0
                best_emb_score = 0.0
                
                for candidate in principles_list:
                    if candidate.id == p.id:
                        continue
                    cand_lower = candidate.content.lower().strip()
                    
                    # Strategy 1: exact/substring content match
                    if dep_lower == cand_lower or dep_lower in cand_lower or cand_lower in dep_lower:
                        best_id = candidate.id
                        best_score = 1.0
                        break
                    
                    # Strategy 2: token Jaccard + entity overlap
                    cand_tokens = tokens(candidate.content)
                    score = jaccard(dep_tokens, cand_tokens)
                    entity_overlap = set(e.lower() for e in candidate.entities) & set(e.lower() for e in p.entities)
                    if entity_overlap:
                        score += 0.1 * min(len(entity_overlap), 3)
                    
                    if score > best_score:
                        best_score = score
                        best_id = candidate.id
                
                # Strategy 3: embedding similarity (most reliable)
                if dep_embeddings is not None and embeddings_array is not None:
                    query_emb = dep_embeddings[dep_idx].reshape(1, -1)
                    faiss.normalize_L2(query_emb)
                    scores, idxs = self.index.search(query_emb, 3)
                    for cand_idx, cand_score in zip(idxs[0], scores[0]):
                        if cand_idx >= len(self.index_ids):
                            continue
                        cand_id = self.index_ids[cand_idx]
                        if cand_id == p.id:
                            continue
                        if cand_score > best_emb_score:
                            best_emb_score = cand_score
                            best_emb_id = cand_id
                
                # Decide: prefer embedding match if strong, else lexical match
                final_id = None
                if best_emb_score >= 0.55:
                    final_id = best_emb_id
                elif best_score >= 0.20:
                    final_id = best_id
                elif best_emb_score >= 0.40:
                    final_id = best_emb_id
                
                if final_id:
                    if final_id not in self.depends_on[p.id]:
                        self.depends_on[p.id].append(final_id)
                        self.dependents[final_id].append(p.id)
                else:
                    logger.debug(f"Could not resolve dependency: {dep_desc} for {p.content[:60]}")
                
                dep_idx += 1
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "principles": {pid: p.model_dump() for pid, p in self.principles.items()},
            "depends_on": dict(self.depends_on),
            "dependents": dict(self.dependents)
        }
        with open(path, "w") as f:
            json.dump(data, f)
        
        if self.index:
            faiss.write_index(self.index, str(path.with_suffix(".faiss")))
    
    @classmethod
    def load(cls, path: Path, client=None) -> "PrincipleIndex":
        with open(path, "r") as f:
            data = json.load(f)
        
        principles = [Principle(**p) for p in data["principles"].values()]
        idx = cls(principles, client)
        idx.depends_on = defaultdict(list, data.get("depends_on", {}))
        idx.dependents = defaultdict(list, data.get("dependents", {}))
        
        faiss_path = path.with_suffix(".faiss")
        if faiss_path.exists():
            idx.index = faiss.read_index(str(faiss_path))
        
        return idx


class HierarchyAwareRetriever:
    """Retrieve principles using hierarchy-aware graph traversal."""
    
    def __init__(self, index: PrincipleIndex):
        self.index = index
    
    def retrieve(self, understanding: QueryUnderstanding, top_k: int = None) -> dict[HierarchyLevel, list[Principle]]:
        """
        Retrieve principles organized by hierarchy level.
        Strategy:
        1. Semantic search for seed principles matching query entities
        2. Walk up to L0 axioms
        3. Walk sideways/down to L2 context modifiers for query context
        4. Walk down to L3 exceptions for entities in play
        """
        top_k = top_k or settings.top_k_retrieval
        seed_principles = self._semantic_search(understanding, top_k)
        
        # Organize seeds by level
        seeds_by_level = defaultdict(list)
        for p in seed_principles:
            seeds_by_level[p.hierarchy_level].append(p)
        
        # Walk graph
        retrieved = defaultdict(list)
        seen_ids = set()
        
        # Add seeds
        for level, principles in seeds_by_level.items():
            for p in principles:
                if p.id not in seen_ids:
                    retrieved[level].append(p)
                    seen_ids.add(p.id)
        
        # Walk UP to L0 axioms
        for p in seed_principles:
            self._walk_up(p.id, retrieved, seen_ids)
        
        # Walk DOWN/SIDEWAYS to L2 context modifiers relevant to query context
        context_entities = set()
        for cat in ["contexts", "tumor_types", "cell_types", "molecules"]:
            context_entities.update(e.lower() for e in understanding.entities.get(cat, []))
        
        for p in seed_principles:
            self._walk_to_context_modifiers(p.id, context_entities, retrieved, seen_ids)
        
        # Walk DOWN to L3 exceptions for entities in play
        query_entities = set()
        for cat in ["molecules", "pathways", "cell_types", "interventions"]:
            query_entities.update(e.lower() for e in understanding.entities.get(cat, []))
        
        for p in seed_principles:
            self._walk_to_exceptions(p.id, query_entities, retrieved, seen_ids)
        
        # Convert to regular dict with HierarchyLevel keys
        result = {}
        for level in HierarchyLevel:
            result[level] = retrieved.get(level, [])
        
        logger.info(f"Retrieved: L0={len(result[HierarchyLevel.L0_AXIOM])}, "
                    f"L1={len(result[HierarchyLevel.L1_MECHANISTIC_PATHWAY])}, "
                    f"L2={len(result[HierarchyLevel.L2_CONTEXT_MODIFIER])}, "
                    f"L3={len(result[HierarchyLevel.L3_KNOWN_EXCEPTION])}")
        
        return result
    
    def _embed_query(self, query_text: str) -> np.ndarray | None:
        try:
            # Try Ollama embedding endpoint
            ollama_embs = try_ollama_embedding(self.index.emb_model, [query_text], self.index.client)
            if ollama_embs:
                return np.array(ollama_embs, dtype=np.float32)
            response = self.index.client.embeddings.create(
                model=self.index.emb_model,
                input=[query_text]
            )
            return np.array([response.data[0].embedding], dtype=np.float32)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def _semantic_search(self, understanding: QueryUnderstanding, top_k: int) -> list[Principle]:
        if not self.index.index:
            return []
        
        query_parts = [understanding.causal_question]
        for cat, entities in understanding.entities.items():
            query_parts.extend(entities)
        query_text = " ".join(query_parts)
        
        query_emb_array = self._embed_query(query_text)
        if query_emb_array is None:
            return []
        faiss.normalize_L2(query_emb_array)
        
        scores, indices = self.index.index.search(query_emb_array, min(top_k * 2, len(self.index.index_ids)))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.index.index_ids):
                pid = self.index.index_ids[idx]
                p = self.index.principles[pid]
                results.append(p)
        
        return results[:top_k]
    
    def _walk_up(self, pid: str, retrieved: dict, seen: set, depth: int = 0):
        """Walk up dependency graph to L0 axioms."""
        if depth > settings.max_graph_walk_depth:
            return
        for parent_id in self.index.depends_on.get(pid, []):
            if parent_id not in seen:
                parent = self.index.principles.get(parent_id)
                if parent:
                    retrieved[parent.hierarchy_level].append(parent)
                    seen.add(parent_id)
                    self._walk_up(parent_id, retrieved, seen, depth + 1)
    
    def _walk_to_context_modifiers(self, pid: str, context_entities: set, 
                                    retrieved: dict, seen: set, depth: int = 0):
        """Walk to L2 context modifiers matching query context."""
        if depth > settings.max_graph_walk_depth:
            return
        for child_id in self.index.dependents.get(pid, []):
            if child_id not in seen:
                child = self.index.principles.get(child_id)
                if child and child.hierarchy_level == HierarchyLevel.L2_CONTEXT_MODIFIER:
                    # Check if this context modifier is relevant
                    child_entities = {e.lower() for e in child.entities}
                    if child_entities & context_entities:
                        retrieved[child.hierarchy_level].append(child)
                        seen.add(child_id)
                self._walk_to_context_modifiers(child_id, context_entities, retrieved, seen, depth + 1)
    
    def _walk_to_exceptions(self, pid: str, query_entities: set,
                           retrieved: dict, seen: set, depth: int = 0):
        """Walk to L3 exceptions relevant to query entities."""
        if depth > settings.max_graph_walk_depth:
            return
        for child_id in self.index.dependents.get(pid, []):
            if child_id not in seen:
                child = self.index.principles.get(child_id)
                if child and child.hierarchy_level == HierarchyLevel.L3_KNOWN_EXCEPTION:
                    child_entities = {e.lower() for e in child.entities}
                    if child_entities & query_entities:
                        retrieved[child.hierarchy_level].append(child)
                        seen.add(child_id)
                self._walk_to_exceptions(child_id, query_entities, retrieved, seen, depth + 1)


if __name__ == "__main__":
    import sys
    
    
    # Load index
    index = PrincipleIndex.load(Path("data/index/principles.json"))
    retriever = HierarchyAwareRetriever(index)
    
    # Test query
    query = sys.argv[1] if len(sys.argv) > 1 else "What happens if we block PD-1 in a cold tumor?"
    
    # Quick test without full query understanding
    from cancer_immunology_reasoner.query_understanding import QueryUnderstander
    u = QueryUnderstander()
    understanding = u.understand(query)
    
    results = retriever.retrieve(understanding)
    for level, principles in results.items():
        print(f"\n=== {level.value} ({len(principles)}) ===")
        for p in principles:
            print(f"  - {p.content[:100]}... [{p.source_citation}]")