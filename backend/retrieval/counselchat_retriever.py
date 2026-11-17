"""
MindBridge 2.0 - CounselChat RAG Retriever & Multi-Collection Architecture
==========================================================================

Implements the CounselChat RAG retrieval engine, separate FAISS vector indexing,
and multi-collection arbitration between canonical clinical guidance and
illustrative community peer counseling perspectives.

Strict Governance & Safety Invariants:
1. Separate Index Isolation:
   CounselChat embeddings reside in a dedicated vector store directory
   (`data/knowledge_base/counselchat_faiss_index` or `data/processed/counselchat/experimental_faiss_index`).
   The 31-vector canonical clinical FAISS index is strictly read-only and immutable.
2. Clinical KB Absolute Precedence:
   Canonical clinical knowledge chunks ALWAYS take precedence over CounselChat peer examples.
   CounselChat examples are secondary and illustrative; they NEVER substitute for clinical guidance.
3. Universal Emergency Crisis Bypass:
   Under emergency_bypass or acute crisis, ALL vector retrieval across both collections
   is halted immediately. Zero queries are issued to any FAISS collection.
4. Non-Clinical Framing & Attribution:
   CounselChat chunks are explicitly tagged with non-clinical disclaimers and
   therapist attributions. Clinical approval remains strictly "not_assessed".
5. Unapproved Record Isolation:
   Unapproved records are strictly prevented from entering the production retrieval index.
   The experimental dev subset is clearly isolated in `data/experimental/counselchat/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from backend.orchestrator.state import (
    JurisdictionCode,
    KnowledgeChunk,
    MindBridgeState,
    RetrievalStatus,
)
from backend.retrieval.counselchat_rag_design import (
    COUNSELCHAT_COLLECTION_NAME,
    COUNSELCHAT_DEFAULT_JURISDICTION,
    COUNSELCHAT_DISCLAIMER,
    COUNSELCHAT_GUARDRAILS,
    COUNSELCHAT_RETRIEVAL_TIER,
    COUNSELCHAT_SAFETY_LEVEL,
    REVIEWER_PLACEHOLDER_MARKER,
    arbitrate_collection_retrieval,
    format_counselchat_citation,
    is_retrieval_bypassed_by_safety,
    normalize_topics,
    split_answer_into_sentence_slices,
    validate_counselchat_record_for_ingestion,
)
from backend.retrieval.embeddings import (
    BGEEmbeddingEngine,
    MockEmbeddingEngine,
    EMBEDDING_DIMENSION,
)
from backend.retrieval.retriever import RAGRetriever, RetrievalAuthorization
from backend.retrieval.vector_store import (
    FAISSVectorStore,
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
)

logger = logging.getLogger("mindbridge.retrieval.counselchat")

DEFAULT_CANONICAL_JSONL = Path("data/processed/counselchat/counselchat_canonical.jsonl")
DEFAULT_APPROVED_JSONL = Path("data/processed/counselchat/counselchat_approved.jsonl")
DEFAULT_EXPERIMENTAL_SUBSET_PATH = Path("data/experimental/counselchat/counselchat_experimental_dev_subset.jsonl")
DEFAULT_EXPERIMENTAL_FAISS_DIR = Path("data/processed/counselchat/experimental_faiss_index")
DEFAULT_PRODUCTION_FAISS_DIR = Path("data/processed/counselchat/faiss_index")
DEFAULT_CONSOLIDATED_JSONL = Path("data/processed/counselchat/counselchat_merged_cleaned.jsonl")
DEFAULT_CONSOLIDATED_FAISS_DIR = Path("data/knowledge_base/counselchat_faiss_index")

# Selected 15 diverse suitable candidate records spanning core non-crisis counseling domains
EXPERIMENTAL_DEV_RECORD_IDS = [
    "counselchat_0000",  # Family Conflict (mother and wife disagreements)
    "counselchat_0003",  # Behavioral Change / Social Relationships (causes of possessiveness)
    "counselchat_0004",  # Anxiety (non-pharmacological anxiety coping)
    "counselchat_0006",  # Anger Management (explosive anger management)
    "counselchat_0007",  # Sleep Improvement (insomnia and sleep disruption)
    "counselchat_0010",  # Relationships / Marriage (trust and infidelity coping)
    "counselchat_0012",  # Marriage / Intimacy (relational intimacy)
    "counselchat_0018",  # Anxiety / Depression / Stress (feeling heard)
    "counselchat_0028",  # Grief and Loss (grieving a mother's passing)
    "counselchat_0036",  # Parenting / Anger Management (step-child behavioral challenges)
    "counselchat_0038",  # Workplace Relationships (workplace conflict and accusation)
    "counselchat_0040",  # Self-esteem / Parenting (family conflict and validation)
    "counselchat_0080",  # Anxiety / Career Counseling (life changes and career stress)
    "counselchat_0168",  # Anxiety / Family / Stress (communication dynamics)
    "counselchat_0701",  # Workplace Relationships / Career Counseling (coping after crying at work)
]


def build_counselchat_experimental_dev_subset(
    canonical_path: Union[str, Path] = DEFAULT_CANONICAL_JSONL,
    output_path: Union[str, Path] = DEFAULT_EXPERIMENTAL_SUBSET_PATH,
    record_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts a balanced, clearly labeled experimental development subset of CounselChat
    records for RAG prototyping.
    """
    canonical_file = Path(canonical_path)
    output_file = Path(output_path)
    target_ids = set(record_ids or EXPERIMENTAL_DEV_RECORD_IDS)

    if not canonical_file.exists():
        raise FileNotFoundError(f"Canonical CounselChat dataset not found at: {canonical_file}")

    extracted_records: List[Dict[str, Any]] = []

    with open(canonical_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            rec_id = str(rec.get("stable_record_id") or rec.get("record_id", ""))
            if rec_id in target_ids:
                exp_rec: Dict[str, Any] = {
                    "record_id": rec_id,
                    "stable_record_id": rec_id,
                    "question_id": str(rec.get("questionID") or rec.get("question_id", "")),
                    "question_title": str(rec.get("questionTitle") or rec.get("question_title", "")),
                    "question_text": str(rec.get("questionText") or rec.get("question_text", "")),
                    "question_url": str(rec.get("questionUrl") or rec.get("question_url", "")),
                    "answer_text": str(rec.get("clean_answer_text") or rec.get("answerText") or rec.get("answer_text", "")),
                    "topics": normalize_topics(rec.get("topics", [])),
                    "therapist_name": str(rec.get("therapistName") or rec.get("therapist_name", "")),
                    "therapist_url": str(rec.get("therapistUrl") or rec.get("therapist_url", "")),
                    "upvotes": int(rec.get("upvotes", 0)) if rec.get("upvotes") is not None else 0,
                    "source_dataset": str(rec.get("source_dataset", "weiting016/counselchat-data")),
                    "source_row_index": int(rec.get("source_row_index", 0)),
                    "is_experimental_dev": True,
                    "dataset_tier": "experimental_dev_subset",
                    "review_status": "pending_review",
                    "retrieval_permission": "experimental_only",
                    "clinical_approval": "not_assessed",
                    "quality_flags": list(rec.get("quality_flags", [])),
                    "notes": "Experimental development subset for RAG prototyping only. NOT clinically verified or approved for production.",
                }
                extracted_records.append(exp_rec)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for item in extracted_records:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Created experimental dev subset: %d records written to %s",
        len(extracted_records),
        output_file,
    )
    return extracted_records


def chunk_counselchat_experimental_record(
    record: Dict[str, Any],
    max_answer_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """
    Chunks an experimental CounselChat record into sentence-bounded slices.
    Accepts both canonical and experimental field naming variants.
    """
    rid = str(record.get("record_id") or record.get("stable_record_id", "")).strip()
    title = str(record.get("question_title") or record.get("questionTitle", "")).strip() or f"CounselChat Q&A {rid}"
    question = str(record.get("question_text") or record.get("questionText", "")).strip()
    answer = str(record.get("answer_text") or record.get("clean_answer_text") or record.get("answerText", "")).strip()
    topics = normalize_topics(record.get("topics", []))

    if not question:
        # Fallback to title if question body is empty
        question = title

    if not question or not answer:
        raise ValueError(f"Record {rid}: question and answer must not be empty.")

    slices = split_answer_into_sentence_slices(answer, max_chars=max_answer_chars)
    if not slices:
        raise ValueError(f"Record {rid}: could not produce sentence slices.")

    chunks: List[Dict[str, Any]] = []
    total_slices = len(slices)

    for idx, slice_text in enumerate(slices):
        chunk_id = f"counselchat_chunk_{rid}_{idx:03d}"

        if total_slices == 1:
            content = f"Question: {question}\n\nCounselor Response:\n{slice_text}"
        else:
            content = (
                f"Question: {question}\n\n"
                f"Counselor Response (Part {idx + 1} of {total_slices}):\n{slice_text}"
            )

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        chunk: Dict[str, Any] = {
            "record_id": rid,
            "document_id": rid,
            "source_dataset": str(record.get("source_dataset", "weiting016/counselchat-data")),
            "source_row_index": int(record.get("source_row_index", 0)),
            "question": question,
            "answer": slice_text,
            "topics": topics,
            "is_experimental_dev": True,
            "dataset_tier": "experimental_dev_subset",
            "review_status": "pending_review",
            "retrieval_permission": "experimental_only",
            "clinical_approval": "not_assessed",
            "content_hash": content_hash,
            "question_id": str(record.get("question_id") or record.get("questionID", "")),
            "therapist_name": str(record.get("therapist_name") or record.get("therapistName", "")),
            "therapist_url": str(record.get("therapist_url") or record.get("therapistUrl", "")),
            "question_url": str(record.get("question_url") or record.get("questionUrl", "")),
            "upvotes": int(record.get("upvotes", 0)),
            "collection": COUNSELCHAT_COLLECTION_NAME,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "total_chunks": total_slices,
            "title": title,
            "content": content,
            "jurisdiction": "unspecified",
            "category": "peer_counseling_example",
            "retrieval_tier": "experimental_dev_subset",
            "clinical_safety_level": COUNSELCHAT_SAFETY_LEVEL,
            "disclaimer": COUNSELCHAT_DISCLAIMER,
            "guardrails": list(COUNSELCHAT_GUARDRAILS),
            "is_clinical_guidance": False,
            "score": 0.0,
            "eligibility": "approved",
        }
        chunks.append(chunk)

    return chunks


def chunk_consolidated_counselchat_dataset(
    merged_jsonl_path: Union[str, Path] = DEFAULT_CONSOLIDATED_JSONL,
    max_answer_chars: int = 1200,
    include_categories: Optional[Sequence[str]] = ("SUITABLE_CANDIDATE", "NEEDS_HUMAN_REVIEW"),
) -> List[Dict[str, Any]]:
    """
    Extracts and chunks retrieval-ready records from the consolidated cleaned dataset.
    Preserves multiple therapist perspectives per question.
    Conservative safety filtering: excludes crisis/self-harm restricted records
    and hard-excluded malformed/prescribing records.
    """
    merged_file = Path(merged_jsonl_path)
    if not merged_file.exists():
        raise FileNotFoundError(f"Consolidated dataset not found at: {merged_file}")

    target_cats = set(include_categories or ["SUITABLE_CANDIDATE", "NEEDS_HUMAN_REVIEW"])
    chunks: List[Dict[str, Any]] = []

    with open(merged_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            q_record = json.loads(line_str)
            if not q_record.get("is_retrieval_ready"):
                continue

            cons_id = q_record["consolidated_id"]
            question = q_record["question"]
            title = q_record["question_title"]
            topics = q_record.get("topics", [])
            all_therapists = q_record.get("therapist_attributions", [])
            total_answers = q_record.get("answers_count", 1)

            for ans in q_record.get("answers", []):
                cat = ans.get("automated_review_category", "")
                # Exclude crisis/self-harm restricted records and hard-excluded prescribing/malformed records
                if cat in ("RESTRICTED_SAFETY_EVALUATION", "EXCLUDE_FROM_RETRIEVAL"):
                    continue
                if cat not in target_cats:
                    continue

                answer_text = ans.get("answer_text", "").strip()
                if not answer_text or not question:
                    continue

                t_name = ans.get("therapist_name", "Counselor")
                slices = split_answer_into_sentence_slices(answer_text, max_chars=max_answer_chars)
                total_slices = len(slices)

                for idx, slice_text in enumerate(slices):
                    chunk_id = f"counselchat_chunk_{ans['source_record_id']}_{idx:03d}"
                    if total_slices == 1:
                        content = f"Question: {question}\n\nCounselor Response ({t_name}):\n{slice_text}"
                    else:
                        content = (
                            f"Question: {question}\n\n"
                            f"Counselor Response (Part {idx + 1} of {total_slices}) ({t_name}):\n{slice_text}"
                        )

                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    chunk: Dict[str, Any] = {
                        "chunk_id": chunk_id,
                        "document_id": ans["source_record_id"],
                        "consolidated_id": cons_id,
                        "title": title,
                        "question": question,
                        "topics": topics,
                        "answer": slice_text,
                        "content": content,
                        "content_hash": content_hash,
                        "therapist_name": t_name,
                        "therapist_url": ans.get("therapist_url", ""),
                        "source_record_id": ans["source_record_id"],
                        "source_row_index": ans.get("source_row_index", 0),
                        "source_url": ans.get("source_url", ""),
                        "upvotes": ans.get("upvotes", 0),
                        "automated_review_category": cat,
                        "quality_flags": ans.get("quality_flags", []),
                        "review_status": ans.get("review_status", "pending_review"),
                        "category": "peer_counseling_example",
                        "collection": COUNSELCHAT_COLLECTION_NAME,
                        "retrieval_tier": "counselchat_consolidated",
                        "clinical_safety_level": COUNSELCHAT_SAFETY_LEVEL,
                        "clinical_approval": "not_assessed",
                        "disclaimer": COUNSELCHAT_DISCLAIMER,
                        "guardrails": list(COUNSELCHAT_GUARDRAILS),
                        "is_clinical_guidance": False,
                        "score": 0.0,
                        "eligibility": "approved",
                        "jurisdiction": "unspecified",
                        "total_answers_for_question": total_answers,
                        "all_therapists_for_question": all_therapists,
                    }
                    chunks.append(chunk)

    return chunks


def build_consolidated_counselchat_faiss_index(
    merged_jsonl_path: Union[str, Path] = DEFAULT_CONSOLIDATED_JSONL,
    output_dir: Union[str, Path] = DEFAULT_CONSOLIDATED_FAISS_DIR,
    embedding_engine: Optional[Union[BGEEmbeddingEngine, MockEmbeddingEngine]] = None,
    include_categories: Optional[Sequence[str]] = ("SUITABLE_CANDIDATE", "NEEDS_HUMAN_REVIEW"),
) -> Tuple[FAISSVectorStore, Dict[str, Any]]:
    """
    Builds the consolidated CounselChat FAISS index from retrieval-ready consolidated questions.
    Saves index.faiss and metadata.json atomically to output_dir.
    """
    engine = embedding_engine or BGEEmbeddingEngine(batch_size=64)
    chunks = chunk_consolidated_counselchat_dataset(
        merged_jsonl_path=merged_jsonl_path,
        include_categories=include_categories,
    )

    out_path = Path(output_dir)
    store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

    summary: Dict[str, Any] = {
        "merged_jsonl_path": str(merged_jsonl_path),
        "output_dir": str(out_path),
        "total_chunks_indexed": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if chunks:
        texts = [c["content"] for c in chunks]
        embeddings = engine.embed_texts(texts)
        store.add_chunks(chunks, embeddings)

    store.save(out_path)
    summary["total_chunks_indexed"] = store.total_vectors
    logger.info("Saved consolidated CounselChat FAISS index to %s (%d vectors).", out_path, store.total_vectors)
    return store, summary


def build_counselchat_faiss_index(
    input_path: Union[str, Path] = DEFAULT_EXPERIMENTAL_SUBSET_PATH,
    output_dir: Union[str, Path] = DEFAULT_EXPERIMENTAL_FAISS_DIR,
    embedding_engine: Optional[Union[BGEEmbeddingEngine, MockEmbeddingEngine]] = None,
    is_experimental: bool = True,
) -> Tuple[FAISSVectorStore, Dict[str, Any]]:
    """
    Builds a dedicated CounselChat FAISS index and saves it atomically to disk.
    """
    input_file = Path(input_path)
    out_dir = Path(output_dir)
    engine = embedding_engine or BGEEmbeddingEngine()

    store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
    summary: Dict[str, Any] = {
        "input_path": str(input_file),
        "output_dir": str(out_dir),
        "is_experimental": is_experimental,
        "records_read": 0,
        "chunks_indexed": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not input_file.exists() or input_file.stat().st_size == 0:
        logger.info("Empty or missing input file %s; initializing empty FAISS index (0 vectors).", input_file)
        store.save(out_dir)
        summary["chunks_indexed"] = 0
        return store, summary

    records: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    summary["records_read"] = len(records)
    all_chunks: List[Dict[str, Any]] = []

    for r in records:
        if is_experimental:
            chunks = chunk_counselchat_experimental_record(r)
            all_chunks.extend(chunks)
        else:
            validation_errors = validate_counselchat_record_for_ingestion(r)
            if validation_errors:
                raise ValueError(f"Record {r.get('record_id')} failed governance validation: {validation_errors}")
            from backend.retrieval.counselchat_rag_design import chunk_counselchat_record
            chunks = chunk_counselchat_record(r)
            all_chunks.extend(chunks)

    if all_chunks:
        texts = [c["content"] for c in all_chunks]
        embeddings = engine.embed_texts(texts)
        store.add_chunks(all_chunks, embeddings)

    store.save(out_dir)
    summary["chunks_indexed"] = store.total_vectors
    logger.info(
        "Built CounselChat FAISS index at %s with %d chunks (is_experimental=%s).",
        out_dir,
        store.total_vectors,
        is_experimental,
    )
    return store, summary


class CounselChatRetriever:
    """
    Dense vector similarity retriever for CounselChat community counseling examples.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedding_engine: Optional[Union[BGEEmbeddingEngine, MockEmbeddingEngine]] = None,
        default_threshold: float = 0.60,
        is_experimental: bool = True,
        index_dir: Optional[Union[str, Path]] = None,
    ):
        self.embedding_engine = embedding_engine or BGEEmbeddingEngine()
        self.default_threshold = default_threshold
        self.is_experimental = is_experimental

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            target_dir = Path(index_dir) if index_dir else None
            if target_dir is None:
                # Prefer consolidated index if present, else experimental, else empty store
                if (DEFAULT_CONSOLIDATED_FAISS_DIR / "index.faiss").exists():
                    target_dir = DEFAULT_CONSOLIDATED_FAISS_DIR
                elif (DEFAULT_EXPERIMENTAL_FAISS_DIR / "index.faiss").exists():
                    target_dir = DEFAULT_EXPERIMENTAL_FAISS_DIR

            if target_dir and (target_dir / "index.faiss").exists():
                self.vector_store = FAISSVectorStore.load(target_dir)
            else:
                self.vector_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
        safety_route: Optional[str] = None,
        **kwargs: Any,
    ) -> List[KnowledgeChunk]:
        """
        Retrieves top-k CounselChat examples matching the query.
        """
        if safety_route and is_retrieval_bypassed_by_safety(safety_route):
            logger.info("CounselChat retrieval bypassed due to safety route: %s", safety_route)
            return []

        if not query or not query.strip() or self.vector_store.total_vectors == 0:
            return []

        active_threshold = threshold if threshold is not None else self.default_threshold
        query_vector = self.embedding_engine.embed_query(query.strip())

        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=max(top_k * 3, 6),
            threshold=active_threshold,
        )

        retrieved: List[KnowledgeChunk] = []
        seen_answers = set()

        for chunk_meta, score in raw_results:
            rid = str(chunk_meta.get("record_id") or chunk_meta.get("document_id", ""))
            src_rid = str(chunk_meta.get("source_record_id") or rid)
            t_name = str(chunk_meta.get("therapist_name", ""))
            cons_id = str(chunk_meta.get("consolidated_id") or rid)

            # Deduplicate by distinct therapist answer (prevents duplicate slices from the same answer,
            # while preserving distinct therapist answers for the same question)
            answer_key = src_rid if src_rid else f"{cons_id}_{t_name}"
            if answer_key in seen_answers:
                continue

            # Governance check for production approved-only mode
            if not self.is_experimental:
                if (
                    chunk_meta.get("review_status") != "approved"
                    or chunk_meta.get("retrieval_permission") != "yes"
                ):
                    continue

            seen_answers.add(answer_key)

            jur = chunk_meta.get("jurisdiction", "unspecified")
            norm_jur: JurisdictionCode = "GL" if jur == "unspecified" else jur  # type: ignore

            k_chunk: KnowledgeChunk = {
                "chunk_id": chunk_meta.get("chunk_id", f"counselchat_chunk_{rid}"),
                "document_id": rid,
                "title": chunk_meta.get("title", f"CounselChat Q&A {rid}"),
                "category": chunk_meta.get("category", "peer_counseling_example"),
                "topics": chunk_meta.get("topics", []),  # type: ignore
                "jurisdiction": norm_jur,
                "content": chunk_meta.get("content", ""),
                "score": round(float(score), 6),
                "eligibility": "approved",
                "source": f"counselchat/{rid}",
                "source_dataset": chunk_meta.get("source_dataset", "weiting016/counselchat-data"),
                "disclaimer": chunk_meta.get("disclaimer", COUNSELCHAT_DISCLAIMER),
                "disclaimer_text": chunk_meta.get("disclaimer", COUNSELCHAT_DISCLAIMER),
                "guardrails": chunk_meta.get("guardrails", list(COUNSELCHAT_GUARDRAILS)),
                "content_hash": chunk_meta.get("content_hash", ""),
                "chunk_index": chunk_meta.get("chunk_index", 0),
                "clinical_safety_level": chunk_meta.get("clinical_safety_level", COUNSELCHAT_SAFETY_LEVEL),
                "retrieval_tier": chunk_meta.get("retrieval_tier", "counselchat_consolidated"),
                "safety_level": COUNSELCHAT_SAFETY_LEVEL,
                "therapist_name": chunk_meta.get("therapist_name", ""),  # type: ignore
                "clinical_approval": "not_assessed",  # type: ignore
                "collection": COUNSELCHAT_COLLECTION_NAME,  # type: ignore
                "source_url": chunk_meta.get("source_url", ""),  # type: ignore
                "total_answers_for_question": chunk_meta.get("total_answers_for_question", 1),  # type: ignore
                "all_therapists_for_question": chunk_meta.get("all_therapists_for_question", []),  # type: ignore
            }
            retrieved.append(k_chunk)

            if len(retrieved) >= top_k:
                break

        return retrieved


class MultiCollectionRetriever:
    """
    Composite retriever coordinating retrieval across:
    1. Canonical Clinical Knowledge Base (RAGRetriever) - Primary authoritative source.
    2. CounselChat Peer Perspectives (CounselChatRetriever) - Secondary illustrative source.

    MindBridge Invariants Enforced:
    - Canonical clinical KB ALWAYS takes priority over CounselChat.
    - CounselChat never substitutes for missing clinical guidance.
    - Emergency crisis bypass halts all retrieval across both collections.
    - Total chunks capped at `max_total_chunks`.
    """

    def __init__(
        self,
        canonical_retriever: RAGRetriever,
        counselchat_retriever: Optional[CounselChatRetriever] = None,
        max_total_chunks: int = 5,
        enable_counselchat: bool = True,
    ):
        self.canonical_retriever = canonical_retriever
        self.counselchat_retriever = counselchat_retriever
        self.max_total_chunks = max_total_chunks
        self.enable_counselchat = enable_counselchat

    @property
    def vector_store(self) -> FAISSVectorStore:
        """Exposes canonical vector store for backward-compatible readiness inspection."""
        return self.canonical_retriever.vector_store

    @property
    def embedding_engine(self) -> Any:
        """Exposes primary embedding engine."""
        return self.canonical_retriever.embedding_engine

    def retrieve(
        self,
        query: str,
        jurisdiction: JurisdictionCode = "GL",
        include_restricted: bool = False,
        authorization: Optional[RetrievalAuthorization] = None,
        top_k: int = 5,
        safety_route: Optional[str] = None,
        **kwargs: Any,
    ) -> List[KnowledgeChunk]:
        """
        Executes dual-collection retrieval according to MindBridge governance rules.
        """
        # Universal Emergency Bypass
        if safety_route and is_retrieval_bypassed_by_safety(safety_route):
            logger.info("Multi-collection retrieval halted due to safety route: %s", safety_route)
            return []

        effective_top_k = min(top_k, self.max_total_chunks)

        # 1. Primary: Retrieve authoritative clinical knowledge base chunks
        canonical_chunks = self.canonical_retriever.retrieve(
            query=query,
            jurisdiction=jurisdiction,
            include_restricted=include_restricted,
            authorization=authorization,
            top_k=effective_top_k,
        )

        # 2. Secondary: If slots remain and CounselChat is enabled, retrieve peer examples
        remaining_budget = effective_top_k - len(canonical_chunks)
        counselchat_chunks: List[KnowledgeChunk] = []

        if self.enable_counselchat and self.counselchat_retriever is not None and remaining_budget > 0:
            # Evidence Relevance Optimization:
            # If canonical retrieval already found high-confidence psychoeducational or clinical
            # guidance (score >= 0.70), prioritize it. Only admit CounselChat peer examples if they
            # have strong topical relevance (score >= 0.73) to avoid diluting clinical guidance with
            # off-topic community queries, and cap peer examples to at most 1 supplementary chunk.
            has_strong_canonical = any(c.get("score", 0.0) >= 0.70 for c in canonical_chunks)
            if has_strong_canonical:
                cc_budget = min(remaining_budget, 1)
                cc_threshold = max(getattr(self.counselchat_retriever, "default_threshold", 0.60), 0.73)
            else:
                cc_budget = remaining_budget
                cc_threshold = kwargs.get("threshold")

            counselchat_chunks = self.counselchat_retriever.retrieve(
                query=query,
                top_k=cc_budget,
                threshold=cc_threshold,
                safety_route=safety_route,
            )

        # 3. Governance arbitration: Clinical KB chunks first, then CounselChat examples
        combined = arbitrate_collection_retrieval(
            canonical_chunks=canonical_chunks,  # type: ignore
            counselchat_chunks=counselchat_chunks,  # type: ignore
            max_total_chunks=effective_top_k,
        )

        return combined  # type: ignore
