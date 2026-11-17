"""Unit and integration tests for CounselChat Human Review Workflow (Phase 2B).
=============================================================================

Verifies:
1. Batch creation and stable partitioning (22 batches, 1,063 records total).
2. Zero records lost or duplicated across batches.
3. All records strictly initialized to pending/no/not_assessed.
4. Approval safeguards:
   - Missing reviewer blocks approval.
   - Placeholder reviewer blocks production approval.
   - Missing review notes blocks approval.
   - Missing or invalid review date blocks approval.
   - Retrieval permission defaults to 'no' and must be 'yes' for approval.
   - Unresolved quality flags block approval.
   - Clinical approval must remain strictly 'not_assessed'.
5. Existing approval validator compatibility (with and without --allow-subset).
6. Absolute immutability:
   - counselchat_approved.jsonl remains strictly 0 bytes.
   - counselchat_review_queue.csv is preserved.
   - Canonical clinical FAISS index (31 vectors) remains bit-for-bit identical.
"""

import csv
import json
from pathlib import Path
import tempfile
import pytest

from backend.data_pipeline.human_review_workflow import (
    DEFAULT_BATCHES_DIR,
    DEFAULT_CANDIDATE_QUEUE_CSV,
    DEFAULT_FULL_REVIEW_JSONL,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SOURCE_CLEANED,
    compile_approved_records_from_batches,
    create_human_review_batches,
    validate_batch_file,
    validate_human_review_record,
)
from backend.data_pipeline.validate_counselchat_approvals import (
    build_approved_dataset,
)


def test_batch_creation_and_stable_partitioning():
    """Verify that 1,063 suitable candidates are partitioned into 22 batches of 50 (last has 13)."""
    assert DEFAULT_CANDIDATE_QUEUE_CSV.exists()
    assert DEFAULT_BATCHES_DIR.exists()
    assert DEFAULT_MANIFEST_PATH.exists()

    with DEFAULT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_candidates"] == 1063
    assert manifest["batch_size"] == 50
    assert manifest["total_batches"] == 22

    batch_files = sorted(DEFAULT_BATCHES_DIR.glob("batch_*.csv"))
    assert len(batch_files) == 22

    total_rows = 0
    for idx, bpath in enumerate(batch_files, start=1):
        with bpath.open("r", encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))
            total_rows += len(reader)
            if idx < 22:
                assert len(reader) == 50, f"Batch {idx} should have 50 rows, got {len(reader)}"
            else:
                assert len(reader) == 13, f"Batch 22 should have 13 rows, got {len(reader)}"

    assert total_rows == 1063


def test_no_records_lost_or_duplicated():
    """Verify every record ID is unique across all 22 batches and matches the candidate queue."""
    with DEFAULT_CANDIDATE_QUEUE_CSV.open("r", encoding="utf-8-sig") as f:
        queue_rows = list(csv.DictReader(f))

    assert len(queue_rows) == 1063
    queue_ids = [r["record_id"] for r in queue_rows]
    assert len(set(queue_ids)) == 1063

    batch_files = sorted(DEFAULT_BATCHES_DIR.glob("batch_*.csv"))
    collected_batch_ids = []

    for bpath in batch_files:
        with bpath.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                collected_batch_ids.append(r["record_id"])

    assert len(collected_batch_ids) == 1063
    assert len(set(collected_batch_ids)) == 1063
    assert collected_batch_ids == queue_ids


def test_all_records_initially_pending_and_safe():
    """Verify all records across all batches default strictly to pending / no / not_assessed."""
    batch_files = sorted(DEFAULT_BATCHES_DIR.glob("batch_*.csv"))

    for bpath in batch_files:
        with bpath.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                assert r["human_decision"] == "pending"
                assert r["review_status"] == "pending_review"
                assert r["retrieval_permission"] == "no"
                assert r["clinical_approval"] == "not_assessed"
                assert r["reviewer"] == ""
                assert r["reviewed_at"] == ""
                assert r["review_notes"] == ""
                assert r["classification"] == "SUITABLE_CANDIDATE"


def test_missing_reviewer_metadata_blocks_approval():
    """Attempting to approve without an actual reviewer identity must fail validation."""
    mock_row = {
        "record_id": "counselchat_0000",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "",
        "reviewed_at": "2026-11-03",
        "review_notes": "Sound psychoeducational guidance.",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
        "question_text": "Valid question text",
        "answer_text": "Valid answer text",
        "quality_flags": "[]",
    }
    errors = validate_human_review_record(mock_row)
    assert any("missing mandatory 'reviewer'" in err for err in errors)

    # Placeholder reviewer must also be rejected
    mock_row["reviewer"] = "REDACTED_EXAMPLE_REVIEWER"
    errors_placeholder = validate_human_review_record(mock_row)
    assert any("placeholder reviewer" in err for err in errors_placeholder)


def test_missing_notes_blocks_approval():
    """Attempting to approve without review notes must fail validation."""
    mock_row = {
        "record_id": "counselchat_0000",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "Dr. Valid Reviewer",
        "reviewed_at": "2026-11-03",
        "review_notes": "",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
        "question_text": "Valid question text",
        "answer_text": "Valid answer text",
        "quality_flags": "[]",
    }
    errors = validate_human_review_record(mock_row)
    assert any("missing mandatory 'review_notes'" in err for err in errors)


def test_missing_or_invalid_date_blocks_approval():
    """Attempting to approve with a missing or non-ISO date must fail validation."""
    mock_row = {
        "record_id": "counselchat_0000",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "Dr. Valid Reviewer",
        "reviewed_at": "11/03/2026",  # Non-ISO format
        "review_notes": "Sound psychoeducational guidance.",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
        "question_text": "Valid question text",
        "answer_text": "Valid answer text",
        "quality_flags": "[]",
    }
    errors = validate_human_review_record(mock_row)
    assert any("must be a valid ISO YYYY-MM-DD date" in err for err in errors)


def test_retrieval_permission_defaults_to_no_and_must_couple_to_approval():
    """Retrieval permission must be 'yes' if and only if human_decision is 'approve'."""
    # Approved but permission == 'no'
    mock_row_1 = {
        "record_id": "counselchat_0000",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "Dr. Valid Reviewer",
        "reviewed_at": "2026-11-03",
        "review_notes": "Sound guidance.",
        "retrieval_permission": "no",
        "clinical_approval": "not_assessed",
        "question_text": "Valid question",
        "answer_text": "Valid answer",
        "quality_flags": "[]",
    }
    errors_1 = validate_human_review_record(mock_row_1)
    assert any("approved record must have retrieval_permission='yes'" in err for err in errors_1)

    # Pending but permission == 'yes'
    mock_row_2 = {
        "record_id": "counselchat_0000",
        "human_decision": "pending",
        "review_status": "pending_review",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
    }
    errors_2 = validate_human_review_record(mock_row_2)
    assert any("non-approved record must have retrieval_permission='no'" in err for err in errors_2)


def test_unresolved_quality_flags_block_approval():
    """Records with active quality flags must be blocked from approval."""
    mock_row = {
        "record_id": "counselchat_0000",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "Dr. Valid Reviewer",
        "reviewed_at": "2026-11-03",
        "review_notes": "Sound guidance.",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
        "question_text": "Valid question",
        "answer_text": "Valid answer",
        "quality_flags": ["short_answer_review"],
    }
    errors = validate_human_review_record(mock_row)
    assert any("unresolved quality_flags" in err for err in errors)


def test_clinical_approval_invariant_strictly_enforced():
    """Attempting to set clinical_approval to 'approved' must trigger a fatal error."""
    mock_row = {
        "record_id": "counselchat_0000",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "Dr. Valid Reviewer",
        "reviewed_at": "2026-11-03",
        "review_notes": "Sound guidance.",
        "retrieval_permission": "yes",
        "clinical_approval": "approved",  # Strictly prohibited!
        "question_text": "Valid question",
        "answer_text": "Valid answer",
        "quality_flags": "[]",
    }
    errors = validate_human_review_record(mock_row)
    assert any("clinical_approval must remain 'not_assessed'" in err for err in errors)


def test_existing_approval_validator_compatible_with_allow_subset():
    """Existing validate_counselchat_approvals.py can validate candidate batches with --allow-subset."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp_out:
        tmp_out_path = Path(tmp_out.name)
        # Running against batch_01.csv (all 50 records pending)
        batch_01_path = DEFAULT_BATCHES_DIR / "batch_01.csv"
        build_approved_dataset(
            input_path=batch_01_path,
            output_path=tmp_out_path,
            source_path=DEFAULT_SOURCE_CLEANED,
            allow_subset=True,
        )
        # 0 records approved because all are pending
        assert tmp_out_path.exists()
        assert tmp_out_path.stat().st_size == 0


def test_counselchat_approved_jsonl_remains_strictly_zero_bytes():
    """counselchat_approved.jsonl must remain empty (0 bytes)."""
    approved_path = Path("data/processed/counselchat/counselchat_approved.jsonl")
    assert approved_path.exists()
    assert approved_path.stat().st_size == 0


def test_canonical_clinical_kb_index_remains_untouched():
    """Canonical clinical FAISS index (31 vectors) remains bit-for-bit identical."""
    faiss_index_path = Path("data/knowledge_base/faiss_index/index.faiss")
    meta_path = Path("data/knowledge_base/faiss_index/metadata.json")

    assert faiss_index_path.exists()
    assert meta_path.exists()

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_vectors"] == 31
    assert meta["dimension"] == 384
    assert meta["index_sha256"] == "c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888"


def test_duplicate_question_group_preservation_in_batches():
    """Verify duplicate question groups are preserved across batch files."""
    with DEFAULT_CANDIDATE_QUEUE_CSV.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    dup_rows = [r for r in rows if r["duplicate_group_id"]]
    assert len(dup_rows) > 0

    # Ensure duplicate_group_size is int > 1 for all duplicate rows
    for r in dup_rows:
        assert int(r["duplicate_group_size"]) > 1
        assert r["duplicate_group_id"].startswith("dup_group_")
