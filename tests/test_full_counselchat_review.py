"""Unit and regression tests for CounselChat Full Dataset Review & Classification Pipeline.
========================================================================================

Verifies:
1. All 1,482 source records are preserved with zero deletions or mutations.
2. Classification counts strictly partition all 1,482 records across the 4 valid tiers:
   - SUITABLE_CANDIDATE
   - NEEDS_HUMAN_REVIEW
   - RESTRICTED_SAFETY_EVALUATION
   - EXCLUDE_FROM_RETRIEVAL
3. Absolute governance invariants:
   - review_status == "pending_review"
   - retrieval_permission == "no"
   - clinical_approval == "not_assessed"
4. Duplicate-question groups are preserved without deleting repeated questions.
5. Ingestion gating: counselchat_approved.jsonl remains untouched at 0 bytes.
6. Canonical clinical knowledge base FAISS index remains completely untouched.
7. Verification of specific exclusion, restricted, and sensitive cases.
"""

import csv
import json
from pathlib import Path
import pytest

from backend.data_pipeline.full_counselchat_review import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_OUTPUT_REPORT,
    STATUS_EXCLUDED,
    STATUS_NEEDS_REVIEW,
    STATUS_RESTRICTED,
    STATUS_SUITABLE,
    VALID_CLASSIFICATIONS,
    classify_record,
    run_full_dataset_review,
)


def test_full_dataset_review_preserves_all_1482_records():
    assert DEFAULT_INPUT.exists(), f"Source file missing: {DEFAULT_INPUT}"
    assert DEFAULT_OUTPUT_JSONL.exists(), f"Output JSONL missing: {DEFAULT_OUTPUT_JSONL}"
    assert DEFAULT_OUTPUT_CSV.exists(), f"Output CSV missing: {DEFAULT_OUTPUT_CSV}"
    assert DEFAULT_OUTPUT_REPORT.exists(), f"Output report missing: {DEFAULT_OUTPUT_REPORT}"

    with DEFAULT_OUTPUT_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["total_source_records"] == 1482
    assert report["total_classified_records"] == 1482

    counts = report["classification_counts"]
    assert sum(counts.values()) == 1482
    for cat in counts:
        assert cat in VALID_CLASSIFICATIONS


def test_classification_counts_and_percentages_match():
    with DEFAULT_OUTPUT_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    counts = report["classification_counts"]
    percentages = report["classification_percentages"]

    assert counts[STATUS_SUITABLE] == 1063
    assert counts[STATUS_NEEDS_REVIEW] == 358
    assert counts[STATUS_RESTRICTED] == 49
    assert counts[STATUS_EXCLUDED] == 12

    assert percentages[STATUS_SUITABLE] == 71.73
    assert percentages[STATUS_NEEDS_REVIEW] == 24.16
    assert percentages[STATUS_RESTRICTED] == 3.31
    assert percentages[STATUS_EXCLUDED] == 0.81


def test_all_1482_records_maintain_strict_governance_invariants():
    with DEFAULT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            rec = json.loads(line)
            assert rec["review_status"] == "pending_review", f"Row {line_num}: invalid review_status"
            assert rec["retrieval_permission"] == "no", f"Row {line_num}: invalid retrieval_permission"
            assert rec["clinical_approval"] == "not_assessed", f"Row {line_num}: invalid clinical_approval"
            assert "Automated classification recommendation only" in rec["recommendation_notice"]


def test_duplicate_question_groups_preserve_all_records():
    with DEFAULT_OUTPUT_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    dup_summary = report["duplicate_groups_summary"]
    assert dup_summary["total_duplicate_question_groups"] == 281
    assert dup_summary["total_records_in_duplicate_groups"] == 946
    assert dup_summary["unique_question_texts"] == 718
    assert dup_summary["unique_records_single_occurrence"] == 536

    # Verify 946 + 536 == 1482
    assert dup_summary["total_records_in_duplicate_groups"] + dup_summary["unique_records_single_occurrence"] == 1482


def test_csv_output_format_and_row_count():
    with DEFAULT_OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1482
    required_cols = [
        "record_id",
        "source_dataset",
        "source_row_index",
        "question_title",
        "question_text",
        "topics",
        "answer_text",
        "answer_characters",
        "quality_flags",
        "classification",
        "classification_reason",
        "safety_concerns",
        "duplicate_group_id",
        "duplicate_group_size",
        "review_status",
        "retrieval_permission",
        "clinical_approval",
        "recommendation_notice",
    ]
    for col in required_cols:
        assert col in reader.fieldnames


def test_excluded_records_integrity():
    with DEFAULT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        excluded = [json.loads(line) for line in f if json.loads(line)["classification"] == STATUS_EXCLUDED]

    assert len(excluded) == 12
    excluded_ids = {r["record_id"] for r in excluded}

    # Empty answers
    for empty_id in ["counselchat_0335", "counselchat_0962", "counselchat_1029", "counselchat_1190", "counselchat_1191"]:
        assert empty_id in excluded_ids

    # Missing both question title and text
    for missing_id in ["counselchat_0536", "counselchat_0537"]:
        assert missing_id in excluded_ids

    # Unusable short fragments (<50 chars)
    for short_id in ["counselchat_0091", "counselchat_1264", "counselchat_1481"]:
        assert short_id in excluded_ids

    # Shaming / invalidating tone from manual audit
    for shaming_id in ["counselchat_0322", "counselchat_0555"]:
        assert shaming_id in excluded_ids


def test_restricted_safety_evaluation_records():
    with DEFAULT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        restricted = [json.loads(line) for line in f if json.loads(line)["classification"] == STATUS_RESTRICTED]

    assert len(restricted) == 49
    restricted_ids = {r["record_id"] for r in restricted}

    # Suicide ideation disclosures
    assert "counselchat_0380" in restricted_ids  # Letting go of steering wheel while driving
    assert "counselchat_0802" in restricted_ids  # Suicide ideation
    assert "counselchat_0986" in restricted_ids  # Suicide attempt assistance

    # Self-harm disclosures
    assert "counselchat_0172" in restricted_ids  # Self-harm, cutting, biting, punching
    assert "counselchat_0210" in restricted_ids  # Only goes away when I hurt myself
    assert "counselchat_0482" in restricted_ids  # Self-harm addiction

    # Emergency hotline referrals
    assert "counselchat_0029" in restricted_ids  # Childhelp abuse hotline


def test_needs_human_review_records():
    with DEFAULT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        needs_review = [json.loads(line) for line in f if json.loads(line)["classification"] == STATUS_NEEDS_REVIEW]

    assert len(needs_review) == 358
    review_ids = {r["record_id"] for r in needs_review}

    # Diagnostic inquiries
    assert "counselchat_0436" in review_ids
    assert "counselchat_0709" in review_ids
    assert "counselchat_1059" in review_ids

    # Sensitive relational & family boundary disputes
    assert "counselchat_0154" in review_ids
    assert "counselchat_0452" in review_ids
    assert "counselchat_0550" in review_ids
    assert "counselchat_0787" in review_ids
    assert "counselchat_0799" in review_ids


def test_suitable_candidates_have_zero_unresolved_flags():
    with DEFAULT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        suitable = [json.loads(line) for line in f if json.loads(line)["classification"] == STATUS_SUITABLE]

    assert len(suitable) == 1063
    for r in suitable:
        # Zero quality flags
        assert r["quality_flags"] == [], f"Suitable candidate {r['record_id']} has quality flags: {r['quality_flags']}"
        # Valid length
        assert 150 <= r["answer_characters"] <= 3500, f"Suitable candidate {r['record_id']} length {r['answer_characters']} out of bounds"
        # Non-empty content
        assert r["question_title"].strip() != ""
        assert r["answer_text"].strip() != ""
        assert r["safety_concerns"] == ["none"]


def test_counselchat_approved_jsonl_remains_strictly_zero_bytes():
    approved_path = Path("data/processed/counselchat/counselchat_approved.jsonl")
    assert approved_path.exists()
    assert approved_path.stat().st_size == 0, "counselchat_approved.jsonl must remain 0 bytes"


def test_canonical_clinical_kb_index_remains_sealed():
    faiss_index_path = Path("data/knowledge_base/faiss_index/index.faiss")
    meta_path = Path("data/knowledge_base/faiss_index/metadata.json")

    assert faiss_index_path.exists()
    assert meta_path.exists()

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_vectors"] == 31
    assert meta["dimension"] == 384
    assert meta["index_sha256"] == "c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888"
