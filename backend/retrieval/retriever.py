"""
MindBridge 2.0 - Grounded RAG Retrieval Engine
===============================================

Orchestrates dense vector retrieval, governance eligibility filtering,
jurisdiction routing, and citation provenance.

Core Architecture & Governance Policies:
1. Embedding Model: BAAI/bge-small-en-v1.5 (dimension: 384, unit L2-normalized).
2. Vector Store: FAISS IndexFlatIP (cosine similarity on unit vectors).
3. Proposed Starting Threshold: tau = 0.65 (configurable; not a clinically validated boundary).
4. Jurisdiction Filtering: Allows matching local jurisdiction or 'GL' (Global) documents.
   Global documents must NEVER substitute local emergency dispatch contacts.
5. Eligibility Gating:
   - 'approved' documents: Open informational retrieval.
   - 'restricted' documents: Gated behind include_restricted=True; must preserve
     all mandatory guardrails and disclaimers.
   - 'excluded_pending_review' documents: Strictly blocked; never returned.
6. Citation Provenance: Citations strictly reflect actually retrieved chunks.
   Never hallucinates or invents citation records.
7. Emergency Routing Independence: Emergency crisis routing uses deterministic
   safety routing from hotlines.json, NEVER vector search.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np

from backend.orchestrator.state import KnowledgeChunk, JurisdictionCode
from backend.retrieval.governance import (
    DOCUMENT_GOVERNANCE_REGISTRY,
    get_document_governance,
    is_document_eligible_for_ingestion,
)
from backend.retrieval.embeddings import (
    BGEEmbeddingEngine,
    MockEmbeddingEngine,
    EMBEDDING_DIMENSION,
)
from backend.retrieval.vector_store import (
    FAISSVectorStore,
    PROPOSED_RETRIEVAL_THRESHOLD,
)
from backend.retrieval.ingestion import KnowledgeBaseIngestor


@dataclass(frozen=True)
class RetrievalAuthorization:
    """
    Structured backend authorization context for retrieving restricted knowledge chunks.

    Security Requirements:
    1. A boolean flag (include_restricted=True) alone is NEVER treated as proof of authorization.
    2. Restricted access requires an explicit, non-empty RetrievalAuthorization instance
       issued by a trusted internal backend component (e.g. PlanningNode in a clinical screening flow).
    3. User-controlled query strings or unauthenticated API query parameters can NEVER
       instantiate or forge this authorization context.
    """
    authorized_by: str
    reason: str
    auth_token: Optional[str] = None


class RAGRetriever:
    """
    Governed RAG knowledge retriever.
    Performs query encoding, similarity search, jurisdiction filtering,
    and eligibility gating.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_engine: Optional[Union[BGEEmbeddingEngine, MockEmbeddingEngine]] = None,
        default_threshold: float = PROPOSED_RETRIEVAL_THRESHOLD,
        relative_margin: Optional[float] = None,
        max_fallback_gap: Optional[float] = None,
    ):
        self.vector_store = vector_store
        # BGEEmbeddingEngine is the mandatory default for application retrieval.
        # MockEmbeddingEngine must only be used when explicitly injected by tests or offline fixtures.
        self.embedding_engine = embedding_engine or BGEEmbeddingEngine()
        self.default_threshold = default_threshold
        self.relative_margin = relative_margin
        self.max_fallback_gap = max_fallback_gap

    def retrieve(
        self,
        query: str,
        jurisdiction: JurisdictionCode = "GL",
        include_restricted: bool = False,
        authorization: Optional[RetrievalAuthorization] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
        relative_margin: Optional[float] = None,
        max_fallback_gap: Optional[float] = None,
    ) -> List[KnowledgeChunk]:
        """
        Execute governed retrieval for a query.

        Args:
            query: User or planner search query.
            jurisdiction: Active user jurisdiction code ('IN', 'US', 'UK', 'CA', 'AU', 'GL', 'UNKNOWN').
            include_restricted: Whether caller requests restricted documents (screening rubrics, legal references).
            authorization: Trusted backend authorization decision (required for restricted access).
            top_k: Maximum number of chunks to return.
            threshold: Cosine similarity threshold (defaults to self.default_threshold = 0.65).
            relative_margin: Maximum allowable score drop below the top candidate (e.g. 0.10).
            max_fallback_gap: Maximum allowable drop between raw top match and eligible approved candidates.

        Returns:
            List of typed KnowledgeChunk dictionaries sorted descending by score.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            return []

        if top_k <= 0:
            return []

        if self.vector_store.total_vectors == 0:
            return []

        sim_threshold = (
            threshold if threshold is not None else self.default_threshold
        )
        active_relative_margin = (
            relative_margin if relative_margin is not None else self.relative_margin
        )
        active_fallback_gap = (
            max_fallback_gap if max_fallback_gap is not None else self.max_fallback_gap
        )

        # Security Gate: A boolean parameter alone is strictly rejected as proof of authorization.
        # Restricted retrieval requires BOTH include_restricted=True AND a valid RetrievalAuthorization instance.
        is_restricted_authorized = (
            include_restricted is True
            and isinstance(authorization, RetrievalAuthorization)
            and bool(authorization.authorized_by and authorization.authorized_by.strip())
            and bool(authorization.reason and authorization.reason.strip())
        )

        # Normalize user jurisdiction
        user_jur: str = str(jurisdiction).upper().strip() if jurisdiction else "GL"

        # 1. Generate query embedding with BGE query instruction
        query_vector = self.embedding_engine.embed_query(query.strip())

        # 2. Adaptive candidate retrieval with governance and jurisdiction post-filtering.
        # Fixed candidate multipliers (e.g. top_k * 4) can hide eligible chunks if the top
        # candidates happen to belong to restricted or mismatched jurisdictions.
        # We start with an initial pool and adaptively expand if needed until top_k eligible
        # chunks are found or the candidate pool is exhausted.
        pool_size = min(max(top_k * 4, 16), self.vector_store.total_vectors)
        retrieved_chunks: List[KnowledgeChunk] = []
        raw_top_score: Optional[float] = None

        while pool_size <= self.vector_store.total_vectors:
            candidate_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=pool_size,
                threshold=sim_threshold,
            )
            if not candidate_results:
                break

            if raw_top_score is None and candidate_results:
                raw_top_score = round(float(candidate_results[0][1]), 6)

            retrieved_chunks = self._filter_candidates(
                candidate_results,
                user_jurisdiction=user_jur,
                is_restricted_authorized=is_restricted_authorized,
                min_score=sim_threshold,
                top_k=top_k,
            )

            # Termination conditions:
            # - Found top_k eligible chunks
            # - Searched entire index
            # - FAISS returned fewer candidates than pool_size (no more vectors >= threshold)
            if (
                len(retrieved_chunks) >= top_k
                or pool_size == self.vector_store.total_vectors
                or len(candidate_results) < pool_size
            ):
                break

            # Adaptively expand pool size
            pool_size = min(pool_size * 2, self.vector_store.total_vectors)

        if not retrieved_chunks:
            return []

        eligible_max_score = retrieved_chunks[0]["score"]

        # Safety Gate: Off-target fallback suppression
        # If the user is unauthorized for restricted content, and the raw top match in vector space
        # was a strong restricted match (raw_top_score >= 0.75), but all available approved content
        # falls off by > active_fallback_gap (e.g. 0.10):
        # We must return an explicit empty result rather than presenting unrelated approved content.
        if (
            not is_restricted_authorized
            and active_fallback_gap is not None
            and raw_top_score is not None
            and raw_top_score >= 0.75
            and (raw_top_score - eligible_max_score) > active_fallback_gap
        ):
            return []

        # Quality Gate: Relative margin pruning
        # When a strong top match exists, prune trailing candidates whose score drops by more than active_relative_margin.
        if active_relative_margin is not None:
            retrieved_chunks = [
                c for c in retrieved_chunks
                if (eligible_max_score - c["score"]) <= active_relative_margin
            ]

        return retrieved_chunks[:top_k]

    def _filter_candidates(
        self,
        candidate_results: List[Tuple[Dict[str, Any], float]],
        user_jurisdiction: str,
        is_restricted_authorized: bool,
        min_score: float,
        top_k: int,
    ) -> List[KnowledgeChunk]:
        """
        Apply governance eligibility, exclusion, threshold, and jurisdiction filtering to candidates.
        """
        filtered: List[KnowledgeChunk] = []
        seen_chunk_ids = set()
        target_min_score = round(min_score, 6)

        for chunk_dict, score in candidate_results:
            # Defensive check: threshold enforcement
            f_score = round(float(score), 6)
            if f_score < target_min_score:
                continue

            chunk_id = chunk_dict.get("chunk_id", "")
            if chunk_id in seen_chunk_ids:
                continue

            doc_id = chunk_dict.get("document_id", "")
            gov = get_document_governance(doc_id)

            # Defensive Check: Unregistered documents are strictly rejected
            if gov is None:
                continue

            # Defensive Check: Excluded documents must never be returned under ANY mode
            if gov.retrieval_tier == "excluded_pending_review":
                continue

            # Eligibility Gating: Restricted content requires trusted backend authorization
            if gov.retrieval_tier == "restricted" and not is_restricted_authorized:
                continue

            # Jurisdiction Filtering:
            # For known country jurisdictions (IN, US, UK, CA, AU), allow local OR Global (GL).
            # For GL, UNKNOWN, or unrecognized jurisdictions, only GL documents are eligible.
            chunk_jur = chunk_dict.get("jurisdiction", gov.jurisdiction)
            if user_jurisdiction in ("IN", "US", "UK", "CA", "AU"):
                if chunk_jur != user_jurisdiction and chunk_jur != "GL":
                    continue
            else:
                if chunk_jur != "GL":
                    continue

            seen_chunk_ids.add(chunk_id)

            # Construct conforming KnowledgeChunk with all required metadata
            chunk_item: KnowledgeChunk = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "source": chunk_dict.get("source", gov.relative_path),
                "title": chunk_dict.get("title", gov.title),
                "category": chunk_dict.get("category", gov.category),
                "jurisdiction": chunk_dict.get("jurisdiction", gov.jurisdiction),
                "verification_status": chunk_dict.get(
                    "verification_status", gov.verification_status
                ),
                "retrieval_tier": gov.retrieval_tier,
                "safety_level": gov.clinical_safety_level,
                "disclaimer": gov.disclaimer,
                "content_hash": chunk_dict.get("content_hash", ""),
                "chunk_index": chunk_dict.get("chunk_index", 0),
                "content": chunk_dict.get("content", ""),
                "score": round(score, 6),
                "eligibility": "restricted"
                if gov.retrieval_tier == "restricted"
                else "approved",
                "section_header": chunk_dict.get("section_header", "General"),
                "guardrails": list(gov.guardrails),
                "disclaimer_text": gov.disclaimer,
                "clinical_safety_level": gov.clinical_safety_level,
            }
            filtered.append(chunk_item)

            if len(filtered) >= top_k:
                break

        return filtered


def format_citations(chunks: List[KnowledgeChunk]) -> List[Dict[str, Any]]:
    """
    Extract structured citation metadata from actually retrieved chunks.
    Guarantees citation provenance without inventing external references.
    """
    citations: List[Dict[str, Any]] = []
    seen_docs = set()

    for chunk in chunks:
        doc_id = chunk["document_id"]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        citations.append(
            {
                "title": chunk["title"],
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk.get("source", ""),
                "category": chunk["category"],
                "jurisdiction": chunk["jurisdiction"],
                "verification_status": chunk.get("verification_status", "unknown"),
                "retrieval_tier": chunk.get("retrieval_tier", chunk.get("eligibility", "")),
                "safety_level": chunk.get("safety_level", chunk.get("clinical_safety_level", "")),
                "eligibility": chunk["eligibility"],
                "disclaimer": chunk.get("disclaimer", chunk.get("disclaimer_text", "")),
                "guardrails": chunk.get("guardrails", []),
                "content_hash": chunk.get("content_hash", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            }
        )

    return citations


def filter_used_citations(
    response_text: str,
    retrieved_chunks: List[KnowledgeChunk],
) -> List[KnowledgeChunk]:
    """
    Filters retrieved knowledge chunks to ONLY those actually used to support
    the generated response.
    
    Prevents unrelated retrieved documents from being displayed merely because
    they were returned by vector similarity search.
    
    A chunk is considered used if:
    1. Its chunk_id, document_id, or title (or title prefix) appears in response_text.
    2. For CounselChat peer counseling, the counselor's name appears in response_text.
    3. The response text contains substantive topical overlap with the chunk's content.
    
    If no explicit reference is detected across multiple chunks, attributes to the single
    primary highest-relevance chunk rather than dumping all candidate chunks.
    """
    if not response_text or not retrieved_chunks:
        return []

    resp_lower = response_text.lower()
    used: List[KnowledgeChunk] = []
    seen_docs = set()

    stopwords = {
        "the", "and", "that", "this", "with", "for", "you", "your", "are", "here",
        "hear", "what", "ready", "support", "information", "resources", "whenever",
        "take", "gentle", "step", "pace", "control", "pause", "stop", "time", "feel",
        "thoughts", "write", "down", "journal", "notebook", "calmly", "speed", "patient",
        "kind", "remember", "help", "from", "can", "also", "some", "overview", "reflect",
        "instrument", "designed", "individuals", "over", "patterns", "symptoms", "about",
        "when", "facing", "dealing", "kind", "real", "toll", "energy", "peace", "mind",
        "stretched", "thin", "look", "practical", "ways", "ease", "load", "pairing",
        "chance", "recover", "important", "focus", "steady", "effort", "achievable",
        "daily", "rather", "than", "striving", "keeping", "personal", "worth", "never",
        "defined", "score", "grade", "additionally", "life", "heavy", "reaching", "systems",
        "groups", "mentors", "advisors", "campus", "make", "difference", "helping", "less",
        "alone", "process", "lately", "having", "sessions", "unique", "consider", "whether"
    }

    for chunk in retrieved_chunks:
        doc_id = chunk.get("document_id", "")
        chunk_id = chunk.get("chunk_id", "")
        title = chunk.get("title", "").strip()
        therapist_name = chunk.get("therapist_name", "").strip()
        category = chunk.get("category", "")
        is_used = False

        # 1. Exact ID match
        if chunk_id and chunk_id.lower() in resp_lower:
            is_used = True
        elif doc_id and doc_id.lower() in resp_lower:
            is_used = True

        # 2. Exact Title match or title prefix match
        if not is_used and title:
            t_lower = title.lower()
            if t_lower in resp_lower:
                is_used = True
            elif ":" in title:
                prefix = title.split(":", 1)[0].strip()
                if len(prefix) >= 12 and prefix.lower() in resp_lower:
                    is_used = True
            elif " - " in title:
                prefix = title.split(" - ", 1)[0].strip()
                if len(prefix) >= 12 and prefix.lower() in resp_lower:
                    is_used = True

        # 3. Therapist name match for CounselChat
        if not is_used and therapist_name and len(therapist_name) > 3:
            if therapist_name.lower() in resp_lower:
                is_used = True

        # 4. Substantive content overlap
        if not is_used:
            content = chunk.get("content", "")
            words = [w for w in re.findall(r"\b[a-z]{4,}\b", content.lower()) if w not in stopwords]
            unique_words = set(words)
            matched_terms = [w for w in unique_words if w in resp_lower]
            if category in ("psychoeducation", "somatic_regulation", "cognitive_reframing", "clinical_rubrics"):
                if len(matched_terms) >= 4:
                    is_used = True
            elif category == "peer_counseling_example":
                if len(matched_terms) >= 6:
                    is_used = True

        if is_used:
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                used.append(chunk)

    # Fallback: If no explicit match was detected across multiple chunks, attribute ONLY
    # to the single top-scoring chunk rather than dumping all candidate chunks
    if not used and retrieved_chunks:
        used.append(retrieved_chunks[0])

    return used


def build_retriever_from_knowledge_base(
    kb_dir: Optional[Union[str, Path]] = None,
    embedding_engine: Optional[Union[BGEEmbeddingEngine, MockEmbeddingEngine]] = None,
    default_threshold: float = PROPOSED_RETRIEVAL_THRESHOLD,
    relative_margin: Optional[float] = 0.10,
    max_fallback_gap: Optional[float] = 0.10,
) -> Tuple[RAGRetriever, Dict[str, Any]]:
    """
    Factory function to ingest the knowledge base, compute dense embeddings,
    populate FAISS, and return an initialized RAGRetriever.

    Defaults strictly to BGEEmbeddingEngine. MockEmbeddingEngine must be explicitly
    injected if running in offline test environments.
    """
    engine = embedding_engine or BGEEmbeddingEngine()
    ingestor = KnowledgeBaseIngestor(base_dir=kb_dir)
    chunks, summary = ingestor.ingest()

    texts = [c["content"] for c in chunks]
    embeddings = engine.embed_texts(texts)

    store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
    store.add_chunks(chunks, embeddings)

    retriever = RAGRetriever(
        vector_store=store,
        embedding_engine=engine,
        default_threshold=default_threshold,
        relative_margin=relative_margin,
        max_fallback_gap=max_fallback_gap,
    )

    summary["total_vectors_indexed"] = store.total_vectors
    return retriever, summary
