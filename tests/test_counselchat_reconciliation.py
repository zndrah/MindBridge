"""Unit and integration tests for CounselChat Dataset Reconciliation (Phase 2B).
=============================================================================

Validates:
1. Original 9-column schema and record count preservation (1,482 records).
2. Stable questionID and duplicate question handling.
3. Accurate handling of missing fields (nulls) and therapist attribution.
4. Clean 1-to-1 reconciliation between original Kaggle CSV and project pipeline.
5. Safe migration of previous review decisions with zero fabricated reviewers.
6. Preservation of all four review categories (1,063, 358, 49, 12).
7. Approval safeguards: restricted and excluded records cannot bypass safeguards.
8. Immutability of canonical clinical FAISS index (31 vectors) and zero CounselChat vectors.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys
import pytest

# Expand field limit for raw Kaggle dataset fields (contains large HTML/Base64 strings)
csv.field_size_limit(sys.maxsize)

from backend.data_pipeline.reconcile_counselchat import (
    DEFAULT_CANONICAL_CSV,
    DEFAULT_CANONICAL_JSONL,
    DEFAULT_MASTER_QUEUE_CSV,
    DEFAULT_MIGRATION_REPORT,
    DEFAULT_ORIGINAL_CSV,
    DEFAULT_RECONCILIATION_REPORT,
    ORIGINAL_NINE_COLUMNS,
    locate_original_source_csv,
    reconcile_datasets,
)
from backend.data_pipeline.human_review_workflow import (
    DEFAULT_BATCHES_DIR,
    DEFAULT_CANDIDATE_QUEUE_CSV,
    DEFAULT_MANIFEST_PATH,
    validate_human_review_record,
)
from backend.data_pipeline.validate_counselchat_approvals import (
    validate_record,
)


def test_original_dataset_presence_and_nine_columns():
    """Verify original CSV exists and contains exactly the 9 Kaggle columns and 1,482 records."""
    csv_path = locate_original_source_csv()
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in ORIGINAL_NINE_COLUMNS:
            assert col in fields, f"Missing original column: {col}"

        rows = list(reader)
        assert len(rows) == 1482, f"Expected 1482 records, got {len(rows)}"


def test_reconciliation_exact_one_to_one():
    """Verify reconciliation report shows 100% exact 1-to-1 alignment with zero ambiguities."""
    assert DEFAULT_RECONCILIATION_REPORT.exists()

    with DEFAULT_RECONCILIATION_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["original_record_count"] == 1482
    assert report["existing_pipeline_record_count"] == 1482
    assert report["exact_matches"] == 1482
    assert report["ambiguous_matches"] == 0
    assert report["unmatched_records"] == 0
    assert report["reconciliation_status"] == "SUCCESS_EXACT_ONE_TO_ONE"
    assert report["migration_safe"] is True

    # Shared fields must have 0 mismatches
    for col, mismatches in report["shared_fields_mismatches_with_raw_4col"].items():
        assert mismatches == 0, f"Field {col} had {mismatches} mismatches"


def test_canonical_dataset_metadata_and_provenance():
    """Verify canonical dataset preserves all 9 original columns plus project metadata."""
    assert DEFAULT_CANONICAL_JSONL.exists()
    assert DEFAULT_CANONICAL_CSV.exists()
    assert DEFAULT_MASTER_QUEUE_CSV.exists()

    records = []
    with DEFAULT_CANONICAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 1482

    # Check first record structure
    r0 = records[0]
    for col in ORIGINAL_NINE_COLUMNS:
        assert col in r0, f"Canonical record missing original column: {col}"

    assert "stable_record_id" in r0
    assert "source_dataset" in r0
    assert "source_row_index" in r0
    assert "duplicate_group_id" in r0
    assert "duplicate_group_size" in r0
    assert "quality_flags" in r0
    assert "automated_review_category" in r0
    assert "automated_review_reasons" in r0
    assert "human_decision" in r0
    assert "reviewer" in r0
    assert "reviewed_at" in r0
    assert "review_notes" in r0
    assert "retrieval_permission" in r0
    assert "clinical_approval" in r0
    assert "faiss_indexed" in r0
    assert "clean_answer_text" in r0

    # Invariants for newly created records
    assert all(r["human_decision"] == "pending" for r in records)
    assert all(r["reviewer"] is None for r in records)
    assert all(r["reviewed_at"] is None for r in records)
    assert all(r["review_notes"] is None for r in records)
    assert all(r["retrieval_permission"] == "no" for r in records)
    assert all(r["clinical_approval"] == "not_assessed" for r in records)
    assert all(not r["faiss_indexed"] for r in records)


def test_therapist_attribution_and_provenance_preservation():
    """Verify therapist attribution is preserved across canonical records."""
    with DEFAULT_CANONICAL_JSONL.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    therapists = {r["therapistName"] for r in records if r["therapistName"]}
    assert len(therapists) == 159, f"Expected 159 unique therapists, got {len(therapists)}"

    therapist_urls = {r["therapistUrl"] for r in records if r["therapistUrl"]}
    assert len(therapist_urls) >= 150

    question_urls = {r["questionUrl"] for r in records if r["questionUrl"]}
    assert len(question_urls) == 780


def test_migration_audit_and_zero_exceptions():
    """Verify migration report confirms zero exceptions and preservation of sample recommendations."""
    assert DEFAULT_MIGRATION_REPORT.exists()

    with DEFAULT_MIGRATION_REPORT.open("r", encoding="utf-8") as f:
        mig = json.load(f)

    assert mig["total_canonical_records"] == 1482
    assert mig["unresolved_records_count"] == 0
    assert len(mig["migration_exceptions"]) == 0
    assert mig["previous_sample_recommendations_migrated"] == 40

    cats = mig["automated_review_categories"]
    assert cats["SUITABLE_CANDIDATE"] == 1063
    assert cats["NEEDS_HUMAN_REVIEW"] == 358
    assert cats["RESTRICTED_SAFETY_EVALUATION"] == 49
    assert cats["EXCLUDE_FROM_RETRIEVAL"] == 12

    assert mig["governance_safeguards"]["all_records_initially_pending"] is True
    assert mig["governance_safeguards"]["retrieval_permission_strictly_no"] is True
    assert mig["governance_safeguards"]["clinical_approval_strictly_not_assessed"] is True
    assert mig["governance_safeguards"]["faiss_indexed_strictly_false"] is True


def test_restricted_and_excluded_safeguard_enforcement():
    """Verify that records marked RESTRICTED_SAFETY_EVALUATION or EXCLUDE_FROM_RETRIEVAL cannot be approved."""
    mock_restricted = {
        "record_id": "counselchat_0100",
        "human_decision": "approve",
        "review_status": "approved",
        "reviewer": "Dr. Human Reviewer",
        "reviewed_at": "2026-11-04",
        "review_notes": "Attempting to approve a restricted record.",
        "retrieval_permission": "yes",
        "clinical_approval": "not_assessed",
        "question_text": "Sample question text",
        "answer_text": "Sample answer text",
        "quality_flags": "[]",
        "classification": "RESTRICTED_SAFETY_EVALUATION",
    }
    errors = validate_human_review_record(mock_restricted)
    assert any("RESTRICTED_SAFETY_EVALUATION" in err and "cannot be approved" in err for err in errors)

    mock_excluded = dict(mock_restricted)
    mock_excluded["classification"] = "EXCLUDE_FROM_RETRIEVAL"
    errors_exc = validate_human_review_record(mock_excluded)
    assert any("EXCLUDE_FROM_RETRIEVAL" in err and "cannot be approved" in err for err in errors_exc)

    # In validator as well
    val_errors = validate_record(mock_restricted, row_number=2, orig_record=mock_restricted)
    assert any("RESTRICTED_SAFETY_EVALUATION" in err and "cannot be approved" in err for err in val_errors)


def test_master_review_queue_covers_all_1482_records():
    """Verify master review queue retains all 1,482 records across all 4 categories."""
    assert DEFAULT_MASTER_QUEUE_CSV.exists()

    with DEFAULT_MASTER_QUEUE_CSV.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1482
    ids = [r["stable_record_id"] for r in rows]
    assert len(set(ids)) == 1482
    assert ids[0] == "counselchat_0000"
    assert ids[-1] == "counselchat_1481"


def test_batches_display_all_nine_fields():
    """Verify human review batches display all 9 original fields plus duplicate and classification info."""
    batch_01_path = DEFAULT_BATCHES_DIR / "batch_01.csv"
    assert batch_01_path.exists()

    with batch_01_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        # Required fields in prompt
        assert "record_id" in fields
        assert "question_id" in fields
        assert "question_title" in fields
        assert "question_text" in fields
        assert "topics" in fields
        assert "therapist_name" in fields
        assert "therapist_url" in fields
        assert "answer_text" in fields
        assert "question_url" in fields
        assert "upvotes" in fields
        assert "duplicate_group_id" in fields
        assert "duplicate_group_size" in fields
        assert "quality_flags" in fields
        assert "classification" in fields
        assert "classification_reason" in fields
        assert "human_decision" in fields
        assert "reviewer" in fields
        assert "reviewed_at" in fields
        assert "review_notes" in fields
        assert "retrieval_permission" in fields
        assert "clinical_approval" in fields

        rows = list(reader)
        assert len(rows) == 50

        # Verify values are non-empty
        assert rows[0]["question_id"] != ""
        assert rows[0]["therapist_name"] != ""
        assert rows[0]["question_url"] != ""
        assert rows[0]["human_decision"] == "pending"


def test_canonical_faiss_immutability():
    """Verify canonical clinical FAISS index remains untouched with 31 vectors and exact SHA-256."""
    faiss_index_path = Path("data/knowledge_base/faiss_index/index.faiss")
    meta_path = Path("data/knowledge_base/faiss_index/metadata.json")
    approved_jsonl = Path("data/processed/counselchat/counselchat_approved.jsonl")

    assert faiss_index_path.exists()
    assert meta_path.exists()
    assert approved_jsonl.exists()

    # counselchat_approved.jsonl must remain 0 bytes
    assert approved_jsonl.stat().st_size == 0

    # Index sha256
    expected_sha256 = "c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888"
    actual_sha256 = hashlib.sha256(faiss_index_path.read_bytes()).hexdigest()
    assert actual_sha256 == expected_sha256

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_vectors"] == 31
    assert meta["dimension"] == 384
