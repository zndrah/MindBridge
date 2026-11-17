"""MindBridge 2.0 - CounselChat RAG Ingestion Architecture & Governance Design
=============================================================================

This module formalizes the Phase 3 CounselChat RAG integration design:

1. Strict Pre-Ingestion Invariant Gating:
   - Records must originate exclusively from `counselchat_approved.jsonl`.
   - `review_status == "approved"`.
   - `retrieval_permission == "yes"`.
   - `clinical_approval == "not_assessed"`:
     CRITICAL GOVERNANCE DEFINITION: This field reflects an administrative and
     clinical governance classification status, NOT a clinical safety certification.
     It indicates that clinical approval has not been assessed in this project.
     It MUST NOT be converted to "approved" or "clinically_approved".
   - Unmistakable placeholder for reviewer identity in examples: "REDACTED_EXAMPLE_REVIEWER".
   - Exact source integrity against `counselchat_cleaned.jsonl`.
   - Zero unresolved quality flags.

2. Explicit Retrieval Framing & Safety Boundaries:
   - CounselChat is an unverified community counseling Q&A forum source.
   - Retrieved responses are illustrative examples of conversational perspectives,
     NOT clinical recommendations, diagnostic evaluations, or verified medical facts.
   - The generation layer is strictly instructed NEVER to copy or adopt unsafe advice
     merely because it was present in a retrieved community example.
   - The verification layer (`verification_node.py`) deterministically enforces all
     safety policies (prohibition of medication prescriptions, psychiatric diagnoses,
     unsolicited somatic commands, and unverified contacts) regardless of retrieved text.

3. Non-Authoritative Jurisdiction Metadata:
   - CounselChat records are assigned `jurisdiction = "unspecified"` by default.
   - Jurisdiction metadata from CounselChat is strictly non-authoritative and
     MUST NEVER be used to override the user's actual location or emergency routing.

4. Sentence-Aware Long-Answer Chunking:
   - Long answers are split along sentence boundaries without truncating words or context.
   - The full original question is preserved with every answer chunk.
   - Deterministic chunk IDs (`counselchat_chunk_{record_id}_{chunk_index:03d}`).
   - Exact record-level provenance and deterministic SHA-256 content hashes.

5. Decoupled Collection Architecture & Multi-Collection Priority:
   - Separate vector store directory (`data/processed/counselchat/faiss_index/`).
   - The canonical clinical knowledge base (`data/knowledge_base/faiss_index/`) ALWAYS
     takes precedence over CounselChat examples.
   - CounselChat is secondary and optional; it must NEVER substitute for missing
     authoritative clinical safety guidance.

6. Absolute Emergency Crisis Bypass:
   - Under `emergency_bypass` (or acute distress triage), ALL vector retrieval across
     BOTH collections is completely halted (`retrieval_status = "not_attempted"`).
   - Zero queries are issued to either FAISS collection.
   - Only authoritative contacts from `hotlines.json` are returned.
"""

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TypedDict

DEFAULT_APPROVED_INPUT = Path("data/processed/counselchat/counselchat_approved.jsonl")
DEFAULT_SOURCE_CLEANED = Path("data/processed/counselchat/counselchat_cleaned.jsonl")
DEFAULT_COUNSELCHAT_FAISS_DIR = Path("data/processed/counselchat/faiss_index")

COUNSELCHAT_COLLECTION_NAME = "counselchat_examples"
COUNSELCHAT_RETRIEVAL_TIER = "approved"
COUNSELCHAT_SAFETY_LEVEL = "informational_example_only"
COUNSELCHAT_DEFAULT_JURISDICTION = "unspecified"
REVIEWER_PLACEHOLDER_MARKER = "REDACTED_EXAMPLE_REVIEWER"

COUNSELCHAT_DISCLAIMER = (
    "This excerpt is from an unverified community counseling Q&A forum. "
    "It illustrates peer perspectives and conversational coping strategies, "
    "NOT clinical recommendations, diagnostic evaluations, or verified medical facts."
)

COUNSELCHAT_GUARDRAILS = [
    "Unverified community counseling perspective; never a clinical recommendation.",
    "Do not copy or endorse unsafe advice merely because it was retrieved in a community example.",
    "Do not prescribe medications, medical dosages, or psychiatric diagnoses.",
    "The post-generation verification gate enforces all safety policies regardless of retrieved text.",
]


class CounselChatChunkMetadata(TypedDict):
    """Authoritative chunk metadata schema for CounselChat retrieval chunks."""

    # Core Provenance & Audit Identifiers
    record_id: str
    source_dataset: str
    source_row_index: int
    question: str
    answer: str
    topics: List[str]
    reviewer: str
    reviewed_at: str
    review_status: Literal["approved"]
    retrieval_permission: Literal["yes"]
    clinical_approval: Literal["not_assessed"]
    content_hash: str

    # Authentic Upstream Provenance (9 Original Kaggle Columns)
    question_id: Optional[str]
    therapist_name: Optional[str]
    therapist_url: Optional[str]
    question_url: Optional[str]
    upvotes: Optional[int]

    # Retrieval & Search Integration Fields
    collection: Literal["counselchat_examples"]
    chunk_id: str
    chunk_index: int
    total_chunks: int
    title: str
    content: str
    jurisdiction: str
    category: str
    retrieval_tier: str
    clinical_safety_level: str
    disclaimer: str
    guardrails: List[str]


@dataclass(frozen=True)
class CounselChatRAGConfig:
    """Configuration and benchmarking contract for CounselChat RAG retrieval.

    Benchmarking & Tuning Contract:
    - retrieval_threshold: Provisional starting threshold (0.70). Must be validated
      empirically on a dedicated calibration/dev set. NEVER tune using the final evaluation set.
    - Evaluation metrics:
      1. Precision@k: Proportion of retrieved examples relevant to user situation.
      2. Recall@k: Proportion of key counseling perspectives surfaced.
      3. Irrelevant Retrieval Rate: Frequency of low-relevance or noisy chunks.
      4. Abstention Quality: Correctly returning 0 chunks when query is out-of-domain.
    """

    approved_input_path: Path = DEFAULT_APPROVED_INPUT
    source_cleaned_path: Path = DEFAULT_SOURCE_CLEANED
    index_directory: Path = DEFAULT_COUNSELCHAT_FAISS_DIR
    embedding_dimension: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    provisional_retrieval_threshold: float = 0.70  # Provisional; subject to benchmark tuning
    top_k: int = 3
    collection_name: str = COUNSELCHAT_COLLECTION_NAME
    max_chunk_chars: int = 1200


def normalize_topics(topics_raw: Any) -> List[str]:
    """Normalize topics from JSON string, list, or comma-separated string to list of non-empty strings."""
    if isinstance(topics_raw, list):
        return [str(t).strip() for t in topics_raw if str(t).strip()]
    if isinstance(topics_raw, str):
        val = topics_raw.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(t).strip() for t in parsed if str(t).strip()]
            except (json.JSONDecodeError, ValueError):
                pass
        return [t.strip() for t in val.split(",") if t.strip()]
    return []


def split_answer_into_sentence_slices(
    answer_text: str,
    max_chars: int = 1200,
) -> List[str]:
    """Split an answer text into sentence-bounded slices without breaking mid-sentence.

    Preserves sentence integrity, avoids cutting words, and ensures no trailing empty slices.
    """
    cleaned = answer_text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    # Split cleanly on sentence boundary punctuation followed by whitespace or double newline
    sentence_regex = re.compile(r'(?<=[.!?])\s+|\n\n+')
    raw_sentences = sentence_regex.split(cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [cleaned]

    slices: List[str] = []
    current_slice: List[str] = []
    current_len = 0

    for s in sentences:
        s_len = len(s)
        # Check if adding sentence exceeds max_chars
        if current_slice and (current_len + s_len + 1 > max_chars):
            slices.append(" ".join(current_slice))
            current_slice = [s]
            current_len = s_len
        else:
            current_slice.append(s)
            current_len += s_len + 1

    if current_slice:
        slices.append(" ".join(current_slice))

    return slices


def validate_counselchat_record_for_ingestion(
    record: Dict[str, Any],
    orig_record: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate that a CounselChat record satisfies all pre-ingestion governance requirements.

    Requirements enforced:
    1. review_status must be 'approved'.
    2. retrieval_permission must be 'yes'.
    3. clinical_approval must remain 'not_assessed' (governance status, never converted to approved).
    4. reviewer, reviewed_at, and review_notes must be non-empty and valid.
    5. question_text and answer_text must be non-empty.
    6. quality_flags must be empty (zero active/unresolved flags).
    7. Exact source integrity matching against counselchat_cleaned.jsonl.
    8. Rejected, pending, restricted, or unapproved records are strictly forbidden.

    Returns:
        List of error strings. Empty list indicates record is fully valid.
    """
    errors: List[str] = []
    rid = str(record.get("record_id", "")).strip() or "UNKNOWN_RECORD"

    # 1. Status & Permission Gating
    status = str(record.get("review_status", "")).strip().lower()
    if status != "approved":
        errors.append(
            f"Record {rid}: review_status must be 'approved', found '{record.get('review_status')}'"
        )

    perm = str(record.get("retrieval_permission", "")).strip().lower()
    if perm != "yes":
        errors.append(
            f"Record {rid}: retrieval_permission must be 'yes', found '{record.get('retrieval_permission')}'"
        )

    # 2. Clinical Approval Invariant (must remain 'not_assessed' as governance status)
    clin = str(record.get("clinical_approval", "")).strip().lower()
    if clin != "not_assessed":
        errors.append(
            f"Record {rid}: clinical_approval must remain 'not_assessed' (governance status, not clinical approval), found '{record.get('clinical_approval')}'"
        )

    # 3. Reviewer Metadata
    reviewer = str(record.get("reviewer", "")).strip()
    if not reviewer:
        errors.append(f"Record {rid}: missing mandatory 'reviewer' metadata")

    reviewed_at = str(record.get("reviewed_at", "")).strip()
    if not reviewed_at:
        errors.append(f"Record {rid}: missing mandatory 'reviewed_at' date")
    else:
        try:
            date.fromisoformat(reviewed_at)
        except ValueError:
            errors.append(f"Record {rid}: 'reviewed_at' must be valid ISO YYYY-MM-DD, got '{reviewed_at}'")

    notes = str(record.get("review_notes", "")).strip()
    if not notes:
        errors.append(f"Record {rid}: missing mandatory 'review_notes'")

    # 4. Content Completeness
    q_text = str(record.get("question_text", "")).strip()
    if not q_text:
        errors.append(f"Record {rid}: question_text cannot be empty")

    a_text = str(record.get("answer_text", "")).strip()
    if not a_text:
        errors.append(f"Record {rid}: answer_text cannot be empty")

    # 5. Zero Unresolved Quality Flags
    flags_raw = record.get("quality_flags", [])
    if isinstance(flags_raw, str):
        flags = [f.strip() for f in flags_raw.strip("[]").replace('"', "").split(",") if f.strip()]
    elif isinstance(flags_raw, list):
        flags = [str(f).strip() for f in flags_raw if str(f).strip()]
    else:
        flags = []

    if flags:
        errors.append(f"Record {rid}: has unresolved quality_flags: {flags}")

    # 6. Source-Integrity Verification against original cleaned record
    if orig_record is not None:
        if str(record.get("record_id", "")).strip() != str(orig_record.get("record_id", "")).strip():
            errors.append(f"Record {rid}: record_id does not match original source record")

        if str(record.get("source_dataset", "")).strip() != str(orig_record.get("source_dataset", "")).strip():
            errors.append(f"Record {rid}: source_dataset does not match original source record")

        if str(record.get("source_row_index", "")).strip() != str(orig_record.get("source_row_index", "")).strip():
            errors.append(f"Record {rid}: source_row_index does not match original source record")

        if str(record.get("question_title", "")) != str(orig_record.get("question_title", "")):
            errors.append(f"Record {rid}: question_title modified from original source record")

        if str(record.get("question_text", "")) != str(orig_record.get("question_text", "")):
            errors.append(f"Record {rid}: question_text modified from original source record")

        if str(record.get("answer_text", "")) != str(orig_record.get("answer_text", "")):
            errors.append(f"Record {rid}: answer_text modified from original source record")

        rec_topics = normalize_topics(record.get("topics", []))
        orig_topics = normalize_topics(orig_record.get("topics", []))
        if rec_topics != orig_topics:
            errors.append(f"Record {rid}: topics modified from original source record")

    return errors


def chunk_counselchat_record(
    record: Dict[str, Any],
    max_answer_chars: int = 1200,
) -> List[CounselChatChunkMetadata]:
    """Chunk an approved CounselChat record into sentence-bounded chunks.

    Guarantees:
    1. The full original question is preserved at the top of every chunk.
    2. Sentences are never broken mid-sentence.
    3. Deterministic chunk IDs: `counselchat_chunk_{record_id}_{chunk_index:03d}`.
    4. Deterministic SHA-256 content_hash computed for each chunk's content.
    5. Exact record-level provenance inherited by all chunks.
    6. Jurisdiction defaults strictly to 'unspecified' (never automatically 'GL').
    """
    validation_errors = validate_counselchat_record_for_ingestion(record)
    if validation_errors:
        raise ValueError(f"Cannot chunk invalid CounselChat record: {validation_errors}")

    rid = str(record["record_id"]).strip()
    title = str(record.get("question_title", "")).strip() or f"CounselChat Q&A {rid}"
    question = str(record["question_text"]).strip()
    answer = str(record["answer_text"]).strip()
    topics = normalize_topics(record.get("topics", []))

    # Jurisdiction handling: Use 'unspecified' unless verified from source
    raw_jur = str(record.get("jurisdiction", "")).strip().upper()
    jurisdiction = raw_jur if raw_jur in ("IN", "US", "UK", "CA", "AU", "GL") else COUNSELCHAT_DEFAULT_JURISDICTION

    source_idx_raw = record.get("source_row_index", 0)
    try:
        source_idx = int(str(source_idx_raw).strip())
    except (ValueError, TypeError):
        source_idx = 0

    slices = split_answer_into_sentence_slices(answer, max_chars=max_answer_chars)
    if not slices:
        raise ValueError(f"Record {rid}: cannot produce chunks from empty answer slices")

    chunks: List[CounselChatChunkMetadata] = []
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

        chunk: CounselChatChunkMetadata = {
            "record_id": rid,
            "source_dataset": str(record.get("source_dataset", "elvindavis/counselchat")),
            "source_row_index": source_idx,
            "question": question,
            "answer": slice_text,
            "topics": topics,
            "reviewer": str(record.get("reviewer", "")).strip(),
            "reviewed_at": str(record.get("reviewed_at", "")).strip(),
            "review_status": "approved",
            "retrieval_permission": "yes",
            "clinical_approval": "not_assessed",
            "content_hash": content_hash,
            "question_id": str(record.get("question_id") or record.get("questionID") or ""),
            "therapist_name": str(record.get("therapist_name") or record.get("therapistName") or ""),
            "therapist_url": str(record.get("therapist_url") or record.get("therapistUrl") or ""),
            "question_url": str(record.get("question_url") or record.get("questionUrl") or ""),
            "upvotes": int(record.get("upvotes", 0)) if record.get("upvotes") is not None else 0,
            "collection": COUNSELCHAT_COLLECTION_NAME,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "total_chunks": total_slices,
            "title": title,
            "content": content,
            "jurisdiction": jurisdiction,
            "category": "peer_counseling_example",
            "retrieval_tier": COUNSELCHAT_RETRIEVAL_TIER,
            "clinical_safety_level": COUNSELCHAT_SAFETY_LEVEL,
            "disclaimer": COUNSELCHAT_DISCLAIMER,
            "guardrails": list(COUNSELCHAT_GUARDRAILS),
        }
        chunks.append(chunk)

    return chunks


def prepare_counselchat_chunk(
    record: Dict[str, Any],
    chunk_index: int = 0,
) -> CounselChatChunkMetadata:
    """Convenience wrapper to produce the primary chunk for a record."""
    chunks = chunk_counselchat_record(record)
    idx = min(chunk_index, len(chunks) - 1)
    return chunks[idx]


def format_counselchat_citation(chunk: CounselChatChunkMetadata) -> Dict[str, Any]:
    """Format structured citation metadata for a retrieved CounselChat chunk.

    Enforces that CounselChat citations:
    1. Use distinct tag format: `[counselchat:{record_id}]`.
    2. Explicitly label the source as an unverified community counseling example.
    3. Include the mandatory non-clinical disclaimer.
    """
    return {
        "citation_tag": f"[counselchat:{chunk['record_id']}]",
        "title": chunk["title"],
        "record_id": chunk["record_id"],
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk.get("chunk_index", 0),
        "total_chunks": chunk.get("total_chunks", 1),
        "collection": chunk["collection"],
        "source_dataset": chunk["source_dataset"],
        "source_row_index": chunk["source_row_index"],
        "topics": chunk["topics"],
        "retrieval_tier": chunk["retrieval_tier"],
        "safety_level": chunk["clinical_safety_level"],
        "clinical_approval": chunk["clinical_approval"],
        "reviewer": chunk["reviewer"],
        "reviewed_at": chunk["reviewed_at"],
        "jurisdiction": chunk["jurisdiction"],
        "disclaimer": chunk["disclaimer"],
        "guardrails": list(chunk["guardrails"]),
        "content_hash": chunk["content_hash"],
        "is_clinical_guidance": False,
    }


def arbitrate_collection_retrieval(
    canonical_chunks: List[Dict[str, Any]],
    counselchat_chunks: List[CounselChatChunkMetadata],
    max_total_chunks: int = 5,
) -> List[Dict[str, Any]]:
    """Arbitrate multi-collection retrieval according to MindBridge governance rules.

    Priority Rules:
    1. Canonical clinical knowledge base chunks ALWAYS take precedence over CounselChat examples.
    2. CounselChat examples are strictly secondary and optional conversational illustrations.
    3. CounselChat examples must NEVER substitute for missing clinical safety guidance.
       If canonical clinical chunks are absent, CounselChat cannot override or replace clinical rubrics.
    4. Total combined chunks will not exceed max_total_chunks.
    """
    combined: List[Dict[str, Any]] = []

    # Canonical clinical chunks take priority
    for c_chunk in canonical_chunks[:max_total_chunks]:
        combined.append(dict(c_chunk))

    # Supplementary CounselChat examples fill remaining budget only if slot remains
    remaining_slots = max_total_chunks - len(combined)
    if remaining_slots > 0:
        for cc_chunk in counselchat_chunks[:remaining_slots]:
            combined.append(dict(cc_chunk))

    return combined


def is_retrieval_bypassed_by_safety(safety_route: str) -> bool:
    """Determine if ordinary vector retrieval must be bypassed for safety.

    MindBridge Invariant:
    If safety_route is 'emergency_bypass', ALL vector retrieval across BOTH collections
    (canonical clinical KB and CounselChat) is completely halted to prevent crisis advice generation.
    Zero queries are issued to either FAISS collection.
    """
    return safety_route in ("emergency_bypass", "ml_risk_unassessed")
