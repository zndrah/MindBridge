"""Unit and regression tests for the reproducible CounselChat human-audit sampling workflow."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from backend.data_pipeline.sample_counselchat_audit import (
    ALLOWED_MANUAL_DECISIONS,
    AUTHORITATIVE_DECISION_RECOMMENDATIONS,
    CSV_COLUMNS,
    DEFAULT_DECISION_REPORT,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_OUTPUT_REPORT,
    backup_audit_sample,
    generate_audit_decision_report,
    load_provisional_candidates,
    run_sampling_workflow,
    sample_provisional_candidates,
    update_audit_sample_with_decisions,
    validate_audit_sample,
)


def make_mock_curated_record(
    record_id: str,
    status: str = "PROVISIONAL_CANDIDATE",
    primary_topic: str = "Relationships",
    answer_length: int = 500,
    source_row_index: int = 0,
):
    """Create a mock record mimicking counselchat_auto_curated.jsonl."""
    topics = [primary_topic]
    if primary_topic in ("Relationships", "Family Conflict"):
        topics.append("Communication")

    answer = "A" * answer_length

    return {
        "record_id": record_id,
        "source_dataset": "elvindavis/counselchat",
        "source_row_index": source_row_index,
        "question_title": f"Question Title for {record_id}",
        "question_text": f"Question text with specific details for {record_id}?",
        "topics": topics,
        "answer_text": answer,
        "answer_characters": answer_length,
        "review_status": "pending_review",
        "quality_flags": [],
        "curation_status": status,
        "curation_reasons": ["clean_candidate"] if status == "PROVISIONAL_CANDIDATE" else ["other_reason"],
        "automated_quality_score": 0.90 if status == "PROVISIONAL_CANDIDATE" else 0.40,
        "retrieval_permission": "no",
        "clinical_approval": "not_assessed",
    }


def write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# 1. Exactly 40 records are selected when at least 40 candidates exist
def test_sampling_selects_exact_40_when_available(tmp_path):
    records = []
    topics_cycle = ["Relationships", "Depression", "Anxiety", "Family Conflict", "Marriage"]
    for i in range(100):
        records.append(
            make_mock_curated_record(
                record_id=f"rec_{i:04d}",
                primary_topic=topics_cycle[i % len(topics_cycle)],
                answer_length=200 + (i * 20),
                source_row_index=i,
            )
        )

    jsonl_path = tmp_path / "curated.jsonl"
    csv_path = tmp_path / "sample.csv"
    report_path = tmp_path / "report.json"
    write_jsonl(jsonl_path, records)

    report = run_sampling_workflow(
        input_path=jsonl_path,
        output_csv=csv_path,
        output_report=report_path,
        sample_size=40,
        seed=42,
    )

    assert report["number_sampled"] == 40
    assert report["total_provisional_candidates"] == 100

    with csv_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40


def test_sampling_handles_fewer_than_40_candidates(tmp_path):
    records = [
        make_mock_curated_record(f"rec_{i:04d}", source_row_index=i)
        for i in range(15)
    ]
    jsonl_path = tmp_path / "curated.jsonl"
    csv_path = tmp_path / "sample.csv"
    report_path = tmp_path / "report.json"
    write_jsonl(jsonl_path, records)

    report = run_sampling_workflow(
        input_path=jsonl_path,
        output_csv=csv_path,
        output_report=report_path,
        sample_size=40,
        seed=42,
    )

    assert report["number_sampled"] == 15
    assert report["total_provisional_candidates"] == 15


# 2. Sampling is deterministic with seed 42
def test_sampling_is_deterministic_with_seed_42(tmp_path):
    records = []
    topics = ["Relationships", "Depression", "Anxiety", "Stress", "Grief"]
    for i in range(80):
        records.append(
            make_mock_curated_record(
                record_id=f"rec_{i:04d}",
                primary_topic=topics[i % len(topics)],
                answer_length=300 + (i * 15),
                source_row_index=i,
            )
        )

    # First run
    run1 = sample_provisional_candidates(records, sample_size=40, seed=42)
    ids1 = [r["record_id"] for r in run1]

    # Second run
    run2 = sample_provisional_candidates(records, sample_size=40, seed=42)
    ids2 = [r["record_id"] for r in run2]

    # Third run
    run3 = sample_provisional_candidates(records, sample_size=40, seed=42)
    ids3 = [r["record_id"] for r in run3]

    assert ids1 == ids2 == ids3
    assert len(ids1) == 40


# 3. Only provisional candidates are sampled
def test_only_provisional_candidates_are_sampled(tmp_path):
    records = [
        make_mock_curated_record("prov_01", status="PROVISIONAL_CANDIDATE"),
        make_mock_curated_record("prov_02", status="PROVISIONAL_CANDIDATE"),
        make_mock_curated_record("rev_01", status="NEEDS_REVIEW"),
        make_mock_curated_record("rev_02", status="NEEDS_REVIEW"),
        make_mock_curated_record("ex_01", status="EXCLUDED"),
        make_mock_curated_record("ex_02", status="EXCLUDED"),
    ]
    jsonl_path = tmp_path / "curated.jsonl"
    write_jsonl(jsonl_path, records)

    candidates, total_scanned = load_provisional_candidates(jsonl_path)
    assert total_scanned == 6
    assert len(candidates) == 2
    assert all(c["curation_status"] == "PROVISIONAL_CANDIDATE" for c in candidates)

    sampled = sample_provisional_candidates(candidates, sample_size=40, seed=42)
    assert len(sampled) == 2
    assert {s["record_id"] for s in sampled} == {"prov_01", "prov_02"}


# 4. No duplicate record IDs appear in the sample
def test_no_duplicate_record_ids_in_sample(tmp_path):
    records = [
        make_mock_curated_record(f"rec_{i:04d}", source_row_index=i)
        for i in range(60)
    ]
    sampled = sample_provisional_candidates(records, sample_size=40, seed=42)
    record_ids = [r["record_id"] for r in sampled]

    assert len(record_ids) == 40
    assert len(set(record_ids)) == 40


# 5. All manual decisions initialize to pending
def test_all_manual_decisions_initialize_to_pending(tmp_path):
    records = [
        make_mock_curated_record(f"rec_{i:04d}", source_row_index=i)
        for i in range(45)
    ]
    jsonl_path = tmp_path / "curated.jsonl"
    csv_path = tmp_path / "sample.csv"
    report_path = tmp_path / "report.json"
    write_jsonl(jsonl_path, records)

    run_sampling_workflow(
        input_path=jsonl_path,
        output_csv=csv_path,
        output_report=report_path,
        sample_size=40,
        seed=42,
    )

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == 40
    for row in rows:
        assert row["manual_decision"] == "pending"
        assert row["review_notes"] == ""
        assert row["reviewer"] == ""
        assert row["reviewed_at"] == ""
        assert float(row["automated_score"]) > 0.0


# 6. No source or permission fields are modified
def test_no_source_or_permission_fields_modified(tmp_path):
    records = [
        make_mock_curated_record("rec_0001", primary_topic="Parenting", answer_length=450),
    ]
    jsonl_path = tmp_path / "curated.jsonl"
    csv_path = tmp_path / "sample.csv"
    report_path = tmp_path / "report.json"
    write_jsonl(jsonl_path, records)

    run_sampling_workflow(
        input_path=jsonl_path,
        output_csv=csv_path,
        output_report=report_path,
        sample_size=1,
        seed=42,
    )

    with csv_path.open("r", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))

    assert row["record_id"] == "rec_0001"
    assert row["question_title"] == records[0]["question_title"]
    assert row["question_text"] == records[0]["question_text"]
    assert json.loads(row["topics"]) == records[0]["topics"]
    assert row["answer_text"] == records[0]["answer_text"]
    assert json.loads(row["quality_flags"]) == records[0]["quality_flags"]
    assert float(row["automated_score"]) == records[0]["automated_quality_score"]


# 7. Original cleaned dataset checksum remains unchanged
def test_original_cleaned_dataset_checksum_remains_unchanged(tmp_path):
    cleaned_path = Path("data/processed/counselchat/counselchat_cleaned.jsonl")
    if not cleaned_path.exists():
        pytest.skip("Real cleaned dataset not present in local workspace")

    initial_sha = hashlib.sha256(cleaned_path.read_bytes()).hexdigest()

    # Run sampling on auto_curated dataset
    curated_path = Path("data/processed/counselchat/counselchat_auto_curated.jsonl")
    if curated_path.exists():
        out_csv = tmp_path / "test_sample.csv"
        out_rep = tmp_path / "test_report.json"
        run_sampling_workflow(
            input_path=curated_path,
            output_csv=out_csv,
            output_report=out_rep,
            sample_size=40,
            seed=42,
        )

    final_sha = hashlib.sha256(cleaned_path.read_bytes()).hexdigest()
    assert initial_sha == final_sha


# 8. Stratified sampling covers multiple topics and answer lengths
def test_stratified_sampling_covers_multiple_topics_and_lengths(tmp_path):
    records = []
    topics = [
        "Relationships", "Depression", "Family Conflict", "Anxiety",
        "Marriage", "Behavioral Change", "Self-esteem", "Parenting",
        "Anger Management", "Intimacy",
    ]
    for i in range(100):
        records.append(
            make_mock_curated_record(
                record_id=f"rec_{i:04d}",
                primary_topic=topics[i % len(topics)],
                answer_length=200 + ((i % 5) * 500),
                source_row_index=i,
            )
        )

    sampled = sample_provisional_candidates(records, sample_size=40, seed=42)
    assert len(sampled) == 40

    sample_topics = {r["topics"][0] for r in sampled}
    assert len(sample_topics) >= 8

    lengths = [len(r["answer_text"]) for r in sampled]
    assert min(lengths) <= 300
    assert max(lengths) >= 2000


# 9. Audit report contains all mandatory documentation fields
def test_audit_report_documentation_fields(tmp_path):
    records = [
        make_mock_curated_record(f"rec_{i:04d}", source_row_index=i)
        for i in range(50)
    ]
    jsonl_path = tmp_path / "curated.jsonl"
    csv_path = tmp_path / "sample.csv"
    report_path = tmp_path / "report.json"
    write_jsonl(jsonl_path, records)

    report = run_sampling_workflow(
        input_path=jsonl_path,
        output_csv=csv_path,
        output_report=report_path,
        sample_size=40,
        seed=42,
    )

    assert report_path.exists()
    with report_path.open("r", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert saved_report["total_provisional_candidates"] == 50
    assert saved_report["number_sampled"] == 40
    assert saved_report["sampling_seed"] == 42
    assert "topic_distribution" in saved_report
    assert "answer_length_distribution" in saved_report
    assert "duplicate_question_handling" in saved_report
    assert "output_paths" in saved_report
    assert saved_report["retrieval_permission"] == "no"
    assert saved_report["clinical_approval"] == "not_assessed"
    assert saved_report["faiss_indexed"] is False


# 10. Backup creates an exact byte-for-byte copy
def test_backup_audit_sample_creates_exact_copy(tmp_path):
    src_csv = tmp_path / "sample.csv"
    src_csv.write_text("record_id,manual_decision\nrec_1,pending\n", encoding="utf-8-sig")

    bak_path = backup_audit_sample(src_csv)
    assert bak_path.exists()
    assert bak_path.name == "sample.csv.bak"
    assert bak_path.read_bytes() == src_csv.read_bytes()


# 11. Update sample applies all authoritative decisions and notes
def test_update_audit_sample_with_decisions_success(tmp_path):
    rows = []
    for rid, info in AUTHORITATIVE_DECISION_RECOMMENDATIONS.items():
        rows.append({
            "record_id": rid,
            "question_title": f"Title {rid}",
            "question_text": f"Question {rid}",
            "topics": "[\"Relationships\"]",
            "answer_text": f"Answer text for {rid}",
            "quality_flags": "[]",
            "automated_score": "0.95",
            "manual_decision": "pending",
            "review_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        })

    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    updated_rows, bak_file = update_audit_sample_with_decisions(
        csv_path=csv_path,
        output_path=csv_path,
        backup=True,
    )

    assert bak_file.exists()
    assert len(updated_rows) == 40

    dec_counts = {}
    for r in updated_rows:
        d = r["manual_decision"]
        dec_counts[d] = dec_counts.get(d, 0) + 1
        assert r["reviewer"] == ""
        assert r["reviewed_at"] == ""
        assert r["review_notes"] != ""
        assert r["question_text"].startswith("Question counselchat_")
        assert r["answer_text"].startswith("Answer text for counselchat_")

    assert dec_counts == {"approved": 8, "rejected": 10, "pending": 22}


# 12. Update fails if a record ID is not found in decisions map
def test_update_audit_sample_fails_on_missing_id(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "record_id": "counselchat_9999_unknown",
            "question_title": "Q",
            "question_text": "Text",
            "topics": "[]",
            "answer_text": "Ans",
            "quality_flags": "[]",
            "automated_score": "0.9",
            "manual_decision": "pending",
            "review_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        })

    with pytest.raises(ValueError, match="Missing decision mapping for record ID"):
        update_audit_sample_with_decisions(csv_path=csv_path, backup=False)


# 13. Update fails if decision is invalid
def test_update_audit_sample_fails_on_invalid_decision(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "record_id": "counselchat_0002",
            "question_title": "Q",
            "question_text": "Text",
            "topics": "[]",
            "answer_text": "Ans",
            "quality_flags": "[]",
            "automated_score": "0.9",
            "manual_decision": "pending",
            "review_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        })

    bad_map = {"counselchat_0002": {"decision": "maybe", "notes": "Some notes"}}
    with pytest.raises(ValueError, match="Invalid manual decision"):
        update_audit_sample_with_decisions(csv_path=csv_path, decisions_map=bad_map, backup=False)


# 14. Validation passes on valid sample
def test_validate_audit_sample_success(tmp_path):
    rows = []
    for rid, info in AUTHORITATIVE_DECISION_RECOMMENDATIONS.items():
        rows.append({
            "record_id": rid,
            "question_title": f"Title {rid}",
            "question_text": f"Question {rid}",
            "topics": "[\"Relationships\"]",
            "answer_text": f"Answer text for {rid}",
            "quality_flags": "[]",
            "automated_score": "0.95",
            "manual_decision": info["decision"],
            "review_notes": info["notes"],
            "reviewer": "",
            "reviewed_at": "",
        })

    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    val = validate_audit_sample(csv_path, expected_count=40)
    assert val["valid"] is True
    assert val["total_records"] == 40
    assert val["counts_by_decision"] == {"approved": 8, "rejected": 10, "pending": 22}


# 15. Validation rejects non-empty reviewer or reviewed_at
def test_validate_audit_sample_rejects_unauthorized_reviewer(tmp_path):
    rows = []
    for rid, info in AUTHORITATIVE_DECISION_RECOMMENDATIONS.items():
        rows.append({
            "record_id": rid,
            "question_title": f"Title {rid}",
            "question_text": f"Question {rid}",
            "topics": "[\"Relationships\"]",
            "answer_text": f"Answer text for {rid}",
            "quality_flags": "[]",
            "automated_score": "0.95",
            "manual_decision": info["decision"],
            "review_notes": info["notes"],
            "reviewer": "Dr. Unauthorized" if rid == "counselchat_0002" else "",
            "reviewed_at": "",
        })

    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="populated reviewer"):
        validate_audit_sample(csv_path, expected_count=40)


# 16. Validation rejects row count mismatch or duplicate IDs
def test_validate_audit_sample_rejects_wrong_count_or_duplicate(tmp_path):
    # Too few rows
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "record_id": "counselchat_0002",
            "question_title": "Q",
            "question_text": "Text",
            "topics": "[]",
            "answer_text": "Ans",
            "quality_flags": "[]",
            "automated_score": "0.9",
            "manual_decision": "approved",
            "review_notes": "Approved note",
            "reviewer": "",
            "reviewed_at": "",
        })

    with pytest.raises(ValueError, match="Expected exactly 40 rows"):
        validate_audit_sample(csv_path, expected_count=40)


# 17. Decision report generation generates correct payload
def test_generate_audit_decision_report(tmp_path):
    rows = []
    for rid, info in AUTHORITATIVE_DECISION_RECOMMENDATIONS.items():
        rows.append({
            "record_id": rid,
            "question_title": f"Title {rid}",
            "question_text": f"Question {rid}",
            "topics": "[\"Relationships\"]",
            "answer_text": f"Answer text for {rid}",
            "quality_flags": "[]",
            "automated_score": "0.95",
            "manual_decision": info["decision"],
            "review_notes": info["notes"],
            "reviewer": "",
            "reviewed_at": "",
        })

    csv_path = tmp_path / "sample.csv"
    dec_report_path = tmp_path / "dec_report.json"
    main_report_path = tmp_path / "main_report.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with main_report_path.open("w", encoding="utf-8") as f:
        json.dump({"dummy": "value"}, f)

    rep = generate_audit_decision_report(
        csv_path=csv_path,
        report_path=dec_report_path,
        update_main_report=True,
        main_report_path=main_report_path,
    )

    assert dec_report_path.exists()
    assert rep["decision_counts"] == {"approved": 8, "rejected": 10, "pending": 22}
    assert rep["retrieval_permission"] == "no"
    assert rep["clinical_approval"] == "not_assessed"
    assert rep["faiss_indexed"] is False

    with main_report_path.open("r", encoding="utf-8") as f:
        main_data = json.load(f)
    assert "audit_decision_summary" in main_data
    assert main_data["audit_decision_summary"]["decision_counts"] == {"approved": 8, "rejected": 10, "pending": 22}


# 18. Live sample file conforms to all authoritative decision requirements
def test_live_audit_sample_and_reports_conform_to_requirements():
    sample_csv = DEFAULT_OUTPUT_CSV
    if not sample_csv.exists():
        pytest.skip("Sample CSV does not exist in workspace")

    val = validate_audit_sample(sample_csv, expected_count=40)
    assert val["valid"] is True
    assert val["total_records"] == 40
    assert val["counts_by_decision"] == {"approved": 8, "rejected": 10, "pending": 22}

    expected_approved = {
        "counselchat_0002", "counselchat_0204", "counselchat_0208", "counselchat_0399",
        "counselchat_0596", "counselchat_0704", "counselchat_0907", "counselchat_1067",
    }
    expected_rejected = {
        "counselchat_0154", "counselchat_0322", "counselchat_0436", "counselchat_0452",
        "counselchat_0550", "counselchat_0555", "counselchat_0709", "counselchat_0787",
        "counselchat_0799", "counselchat_1059",
    }
    expected_pending = {
        "counselchat_0016", "counselchat_0043", "counselchat_0073", "counselchat_0083",
        "counselchat_0146", "counselchat_0155", "counselchat_0256", "counselchat_0384",
        "counselchat_0385", "counselchat_0408", "counselchat_0430", "counselchat_0484",
        "counselchat_0501", "counselchat_0504", "counselchat_0553", "counselchat_0577",
        "counselchat_0590", "counselchat_0593", "counselchat_0751", "counselchat_0782",
        "counselchat_0786", "counselchat_0857",
    }

    assert set(val["record_ids_by_decision"]["approved"]) == expected_approved
    assert set(val["record_ids_by_decision"]["rejected"]) == expected_rejected
    assert set(val["record_ids_by_decision"]["pending"]) == expected_pending

    # Confirm backup file exists
    bak_file = sample_csv.with_name(f"{sample_csv.name}.bak")
    assert bak_file.exists()

    # Confirm decision report exists and matches
    assert DEFAULT_DECISION_REPORT.exists()
    with DEFAULT_DECISION_REPORT.open("r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["decision_counts"] == {"approved": 8, "rejected": 10, "pending": 22}
    assert rep["retrieval_permission"] == "no"
    assert rep["clinical_approval"] == "not_assessed"
    assert rep["faiss_indexed"] is False
