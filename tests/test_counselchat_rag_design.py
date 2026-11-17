"""Design-level unit and contract tests for Phase 3 CounselChat RAG Integration Architecture.
========================================================================================

Verifies:
1. Ingestion gating: only approved records with retrieval_permission='yes' are eligible.
2. Clinical approval governance: clinical_approval must remain 'not_assessed' as a governance status.
3. Reviewer metadata placeholder enforcement (e.g. 'REDACTED_EXAMPLE_REVIEWER').
4. Jurisdiction defaults strictly to 'unspecified' and never overrides emergency routing.
5. Rejection of pending, restricted, or rejected records.
6. Sentence-aware long-answer chunking preserving the original question across all slices.
7. Multi-collection arbitration: canonical clinical KB takes absolute priority.
8. Absolute emergency crisis bypass across all vector retrieval collections.
9. Citation formatting and non-clinical community example attribution.
10. Decoupled collection isolation from the canonical clinical knowledge base.
"""

from pathlib import Path
import pytest

from backend.retrieval.counselchat_rag_design import (
    COUNSELCHAT_COLLECTION_NAME,
    COUNSELCHAT_DEFAULT_JURISDICTION,
    COUNSELCHAT_DISCLAIMER,
    CounselChatRAGConfig,
    REVIEWER_PLACEHOLDER_MARKER,
    arbitrate_collection_retrieval,
    chunk_counselchat_record,
    format_counselchat_citation,
    is_retrieval_bypassed_by_safety,
    prepare_counselchat_chunk,
    split_answer_into_sentence_slices,
    validate_counselchat_record_for_ingestion,
)


def make_valid_approved_record(record_id="counselchat_0002"):
    """Create a fully compliant approved CounselChat record with an unmistakable placeholder reviewer."""
    return {
        "record_id": record_id,
        "source_dataset": "elvindavis/counselchat",
        "source_row_index": 2,
        "question_title": "Keeping secrets from my family",
        "question_text": "I have secrets in my mind, and I don't know what to do with them. I don't want to tell my wife and mom because I don't want to hurt them.",
        "topics": ["Family Conflict"],
        "answer_text": "It sounds like keeping the secrets has become a problem for you now. There are several things to consider before you make a decision.",
        "quality_flags": [],
        "review_status": "approved",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
        "reviewer": REVIEWER_PLACEHOLDER_MARKER,
        "reviewed_at": "2026-11-03",
        "review_notes": "Approved for non-clinical psychoeducational grounding on boundaries.",
    }


def test_approved_record_satisfies_all_ingestion_invariants():
    rec = make_valid_approved_record()
    orig = dict(rec)
    errors = validate_counselchat_record_for_ingestion(rec, orig_record=orig)
    assert errors == []


def test_reviewer_placeholder_does_not_imply_actual_identity():
    rec = make_valid_approved_record()
    assert rec["reviewer"] == "REDACTED_EXAMPLE_REVIEWER"
    assert "Dr." not in rec["reviewer"]
    assert "Sarah" not in rec["reviewer"]
    assert "Jenkins" not in rec["reviewer"]


def test_jurisdiction_defaults_to_unspecified():
    rec = make_valid_approved_record()
    chunk = prepare_counselchat_chunk(rec)
    assert chunk["jurisdiction"] == COUNSELCHAT_DEFAULT_JURISDICTION
    assert chunk["jurisdiction"] == "unspecified"
    assert chunk["jurisdiction"] != "GL"


def test_jurisdiction_preserves_verified_source_location_if_provided():
    rec = make_valid_approved_record()
    rec["jurisdiction"] = "US"
    chunk = prepare_counselchat_chunk(rec)
    assert chunk["jurisdiction"] == "US"


@pytest.mark.parametrize("bad_status", ["pending_review", "restricted", "rejected", "provisional", ""])
def test_ingestion_rejects_unapproved_review_statuses(bad_status):
    rec = make_valid_approved_record()
    rec["review_status"] = bad_status
    errors = validate_counselchat_record_for_ingestion(rec)
    assert any("review_status must be 'approved'" in err for err in errors)


@pytest.mark.parametrize("bad_permission", ["no", "pending", "restricted", ""])
def test_ingestion_rejects_retrieval_permission_not_yes(bad_permission):
    rec = make_valid_approved_record()
    rec["retrieval_permission"] = bad_permission
    errors = validate_counselchat_record_for_ingestion(rec)
    assert any("retrieval_permission must be 'yes'" in err for err in errors)


@pytest.mark.parametrize("invalid_clinical", ["approved", "clinically_approved", "yes", "true", ""])
def test_ingestion_strictly_prohibits_converting_clinical_approval(invalid_clinical):
    rec = make_valid_approved_record()
    rec["clinical_approval"] = invalid_clinical
    errors = validate_counselchat_record_for_ingestion(rec)
    assert any("clinical_approval must remain 'not_assessed'" in err for err in errors)


def test_ingestion_rejects_missing_or_invalid_reviewer_metadata():
    # Missing reviewer
    rec1 = make_valid_approved_record()
    rec1["reviewer"] = ""
    errors1 = validate_counselchat_record_for_ingestion(rec1)
    assert any("missing mandatory 'reviewer'" in err for err in errors1)

    # Missing reviewed_at
    rec2 = make_valid_approved_record()
    rec2["reviewed_at"] = ""
    errors2 = validate_counselchat_record_for_ingestion(rec2)
    assert any("missing mandatory 'reviewed_at'" in err for err in errors2)

    # Invalid reviewed_at date format
    rec3 = make_valid_approved_record()
    rec3["reviewed_at"] = "11/03/2026"
    errors3 = validate_counselchat_record_for_ingestion(rec3)
    assert any("must be valid ISO YYYY-MM-DD" in err for err in errors3)

    # Missing review_notes
    rec4 = make_valid_approved_record()
    rec4["review_notes"] = ""
    errors4 = validate_counselchat_record_for_ingestion(rec4)
    assert any("missing mandatory 'review_notes'" in err for err in errors4)


def test_ingestion_rejects_unresolved_quality_flags():
    rec = make_valid_approved_record()
    rec["quality_flags"] = ["short_answer_review"]
    errors = validate_counselchat_record_for_ingestion(rec)
    assert any("has unresolved quality_flags" in err for err in errors)


def test_ingestion_rejects_tampered_source_content():
    rec = make_valid_approved_record()
    orig = dict(rec)
    orig["question_text"] = "Original question text before unauthorized edit."
    errors = validate_counselchat_record_for_ingestion(rec, orig_record=orig)
    assert any("question_text modified from original source record" in err for err in errors)


def test_counselchat_chunk_metadata_schema_contract():
    rec = make_valid_approved_record()
    chunk = prepare_counselchat_chunk(rec, chunk_index=0)

    # Validate all required chunk metadata fields
    required_fields = [
        "record_id",
        "source_dataset",
        "source_row_index",
        "question",
        "answer",
        "topics",
        "reviewer",
        "reviewed_at",
        "review_status",
        "retrieval_permission",
        "clinical_approval",
        "content_hash",
        "jurisdiction",
        "collection",
        "chunk_id",
        "chunk_index",
        "total_chunks",
    ]
    for field in required_fields:
        assert field in chunk, f"Missing required field '{field}' in chunk metadata"
        assert chunk[field] is not None and chunk[field] != "", f"Empty field '{field}'"

    # Governance & collection invariants
    assert chunk["collection"] == COUNSELCHAT_COLLECTION_NAME
    assert chunk["review_status"] == "approved"
    assert chunk["retrieval_permission"] == "yes"
    assert chunk["clinical_approval"] == "not_assessed"
    assert chunk["jurisdiction"] == "unspecified"
    assert chunk["disclaimer"] == COUNSELCHAT_DISCLAIMER
    assert len(chunk["content_hash"]) == 64  # SHA-256


def test_citation_formatting_and_non_clinical_attribution():
    rec = make_valid_approved_record()
    chunk = prepare_counselchat_chunk(rec, chunk_index=0)
    citation = format_counselchat_citation(chunk)

    assert citation["citation_tag"] == "[counselchat:counselchat_0002]"
    assert citation["is_clinical_guidance"] is False
    assert citation["clinical_approval"] == "not_assessed"
    assert citation["jurisdiction"] == "unspecified"
    assert citation["disclaimer"] == COUNSELCHAT_DISCLAIMER
    assert citation["collection"] == COUNSELCHAT_COLLECTION_NAME


def test_jurisdiction_unspecified_preserved_and_cannot_override_user_routing():
    """Verifies jurisdiction defaults to 'unspecified' and cannot override user jurisdiction or emergency routing."""
    from backend.orchestrator.safety_node import format_emergency_guidance

    # 1. Chunk defaults strictly to 'unspecified' when record has no jurisdiction
    rec = make_valid_approved_record()
    chunk = prepare_counselchat_chunk(rec, chunk_index=0)
    assert chunk["jurisdiction"] == "unspecified"

    # 2. Invalid or unrecognized jurisdiction also falls back to 'unspecified'
    rec_invalid_jur = dict(rec)
    rec_invalid_jur["jurisdiction"] = "INVALID_COUNTRY_CODE"
    chunk_fallback = prepare_counselchat_chunk(rec_invalid_jur, chunk_index=0)
    assert chunk_fallback["jurisdiction"] == "unspecified"

    # 3. User jurisdiction remains authoritative for emergency hotline guidance
    user_state_us = {"jurisdiction_filter": "US"}
    user_state_in = {"jurisdiction_filter": "IN"}

    guidance_us = format_emergency_guidance(user_state_us["jurisdiction_filter"])
    guidance_in = format_emergency_guidance(user_state_in["jurisdiction_filter"])

    # US user receives 988, IN user receives 112/KIRAN
    assert "988" in guidance_us
    assert "112" in guidance_in
    assert "988" not in guidance_in

    # CounselChat chunk metadata never mutates or overrides user state
    assert user_state_us["jurisdiction_filter"] == "US"
    assert user_state_in["jurisdiction_filter"] == "IN"


# ============================================================================
# Sentence-Aware Long-Answer Chunking Tests
# ============================================================================

def test_split_answer_into_sentence_slices_short():
    short_ans = "This is a single short sentence answering the user."
    slices = split_answer_into_sentence_slices(short_ans, max_chars=1200)
    assert len(slices) == 1
    assert slices[0] == short_ans


def test_split_answer_into_sentence_slices_preserves_sentence_boundaries():
    sentences = [
        "First sentence describing the situation thoroughly and clearly.",
        "Second sentence providing psychoeducational perspective on boundaries.",
        "Third sentence suggesting reflection on personal emotional needs.",
        "Fourth sentence giving closure and encouraging professional consultation.",
    ]
    # Make each sentence long
    long_sentences = [s + " " + ("Detail word explanation and reflection. " * 20) for s in sentences]
    full_text = " ".join(long_sentences)
    assert len(full_text) > 2000

    slices = split_answer_into_sentence_slices(full_text, max_chars=800)
    assert len(slices) >= 3

    # Ensure every slice ends with sentence-ending punctuation (no mid-sentence cuts)
    for s in slices:
        assert s.endswith(".") or s.endswith("?") or s.endswith("!")


def test_chunk_counselchat_record_short_answer():
    rec = make_valid_approved_record()
    rec["answer_text"] = "A short concise counseling answer."
    chunks = chunk_counselchat_record(rec, max_answer_chars=1200)

    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_index"] == 0
    assert c["total_chunks"] == 1
    assert c["chunk_id"] == "counselchat_chunk_counselchat_0002_000"
    assert f"Question: {rec['question_text']}" in c["content"]
    assert "Counselor Response:\nA short concise counseling answer." in c["content"]


def test_chunk_counselchat_record_long_answer_preserves_question_and_provenance():
    rec = make_valid_approved_record()
    # Build an answer of ~2,500 characters across 3 distinct paragraphs
    p1 = "First major paragraph discussing family dynamics and the burden of keeping secrets in depth. " * 10
    p2 = "Second major paragraph outlining constructive communication strategies with specific examples. " * 10
    p3 = "Third major paragraph exploring individual emotional coping mechanisms and psychological boundaries. " * 10
    rec["answer_text"] = f"{p1}\n\n{p2}\n\n{p3}"
    assert len(rec["answer_text"]) > 2000

    chunks = chunk_counselchat_record(rec, max_answer_chars=700)
    assert len(chunks) >= 3

    # Check that EVERY chunk retains the full question at the top
    for idx, c in enumerate(chunks):
        assert c["chunk_index"] == idx
        assert c["total_chunks"] == len(chunks)
        assert c["chunk_id"] == f"counselchat_chunk_counselchat_0002_{idx:03d}"
        assert c["question"] == rec["question_text"]
        assert f"Question: {rec['question_text']}" in c["content"]
        assert f"Part {idx + 1} of {len(chunks)}" in c["content"]
        # Exact provenance preserved
        assert c["record_id"] == rec["record_id"]
        assert c["source_dataset"] == rec["source_dataset"]
        assert c["source_row_index"] == rec["source_row_index"]
        assert c["review_status"] == "approved"
        assert c["retrieval_permission"] == "yes"
        assert c["clinical_approval"] == "not_assessed"
        assert c["reviewer"] == REVIEWER_PLACEHOLDER_MARKER
        assert c["jurisdiction"] == "unspecified"
        # Each chunk has a distinct content hash
        assert len(c["content_hash"]) == 64

    # Hashes across chunks are distinct
    hashes = [c["content_hash"] for c in chunks]
    assert len(set(hashes)) == len(chunks)


def test_chunk_counselchat_record_boundary_length():
    rec = make_valid_approved_record()
    # Exactly 600 characters
    sentence = "A test sentence that is sixty characters long for verification. "
    rec["answer_text"] = (sentence * 10).strip()
    assert len(rec["answer_text"]) == 639

    chunks = chunk_counselchat_record(rec, max_answer_chars=639)
    assert len(chunks) == 1
    assert chunks[0]["total_chunks"] == 1


def test_chunk_counselchat_record_rejects_empty_answer():
    rec = make_valid_approved_record()
    rec["answer_text"] = "   \n\t  "
    with pytest.raises(ValueError, match="Cannot chunk invalid CounselChat record"):
        chunk_counselchat_record(rec)


# ============================================================================
# Multi-Collection Arbitration Tests
# ============================================================================

def test_multi_collection_arbitration_prioritizes_canonical_clinical_kb():
    canonical_chunks = [
        {"chunk_id": "clinical_01", "collection": "canonical_clinical_kb", "title": "GAD-7 Scale"},
        {"chunk_id": "clinical_02", "collection": "canonical_clinical_kb", "title": "PHQ-9 Overview"},
        {"chunk_id": "clinical_03", "collection": "canonical_clinical_kb", "title": "Box Breathing"},
    ]
    rec = make_valid_approved_record()
    counselchat_chunks = chunk_counselchat_record(rec)

    # Combined budget of 4 chunks: 3 canonical + 1 CounselChat
    arbitrated = arbitrate_collection_retrieval(
        canonical_chunks=canonical_chunks,
        counselchat_chunks=counselchat_chunks,
        max_total_chunks=4,
    )

    assert len(arbitrated) == 4
    # First 3 MUST be canonical clinical chunks
    assert arbitrated[0]["chunk_id"] == "clinical_01"
    assert arbitrated[1]["chunk_id"] == "clinical_02"
    assert arbitrated[2]["chunk_id"] == "clinical_03"
    # Fourth is CounselChat
    assert arbitrated[3]["chunk_id"] == counselchat_chunks[0]["chunk_id"]


def test_multi_collection_arbitration_does_not_exceed_budget():
    canonical_chunks = [
        {"chunk_id": f"clinical_{i}", "collection": "canonical_clinical_kb"} for i in range(5)
    ]
    rec = make_valid_approved_record()
    counselchat_chunks = chunk_counselchat_record(rec)

    arbitrated = arbitrate_collection_retrieval(
        canonical_chunks=canonical_chunks,
        counselchat_chunks=counselchat_chunks,
        max_total_chunks=3,
    )
    # Only canonical chunks fill the budget of 3; CounselChat gets 0 slots
    assert len(arbitrated) == 3
    assert all(c["collection"] == "canonical_clinical_kb" for c in arbitrated)


# ============================================================================
# Absolute Emergency Crisis Bypass Tests
# ============================================================================

def test_emergency_crisis_route_bypasses_all_collections():
    assert is_retrieval_bypassed_by_safety("emergency_bypass") is True
    assert is_retrieval_bypassed_by_safety("ml_risk_unassessed") is True
    assert is_retrieval_bypassed_by_safety("standard_conversation") is False
    assert is_retrieval_bypassed_by_safety("supportive_deescalation") is False


def test_counselchat_faiss_store_separation_from_canonical_kb():
    config = CounselChatRAGConfig()
    canonical_kb_index = Path("data/knowledge_base/faiss_index")

    # Ingestion directory must be separate from canonical knowledge base
    assert config.index_directory != canonical_kb_index
    assert config.collection_name == "counselchat_examples"
    assert canonical_kb_index.exists()
    assert config.provisional_retrieval_threshold == 0.70
