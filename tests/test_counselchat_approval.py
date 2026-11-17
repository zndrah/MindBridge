import csv
import json
from pathlib import Path

import pytest

from backend.data_pipeline.prepare_counselchat_review import prepare_review_queue
from backend.data_pipeline.validate_counselchat_approvals import (
    build_approved_dataset,
    cleanup_stale_output,
    REQUIRED_CSV_COLUMNS,
)


FIELDS = list(REQUIRED_CSV_COLUMNS)


def make_record(
    status="pending_review",
    record_id="counselchat_0001",
    retrieval_permission=None,
    quality_flags="",
    clinical_approval="not_assessed",
    reviewer=None,
    reviewed_at=None,
    review_notes=None,
    question_text="A sufficiently detailed test question?",
    answer_text="A sufficiently detailed test answer for testing.",
    topics="Anxiety",
    source_dataset="elvindavis/counselchat",
    source_row_index="1",
    question_title="Test question",
):
    if retrieval_permission is None:
        retrieval_permission = "yes" if status == "approved" else "no"
    if reviewer is None:
        reviewer = "reviewer_1" if status == "approved" else ""
    if reviewed_at is None:
        reviewed_at = "2026-11-03" if status == "approved" else ""
    if review_notes is None:
        review_notes = "Reviewed for retrieval suitability." if status == "approved" else ""

    return {
        "record_id": record_id,
        "source_dataset": source_dataset,
        "source_row_index": str(source_row_index),
        "question_title": question_title,
        "question_text": question_text,
        "topics": topics,
        "answer_text": answer_text,
        "quality_flags": quality_flags,
        "review_status": status,
        "retrieval_permission": retrieval_permission,
        "review_notes": review_notes,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "clinical_approval": clinical_approval,
    }


def make_source_record(
    record_id="counselchat_0001",
    source_dataset="elvindavis/counselchat",
    source_row_index=1,
    question_title="Test question",
    question_text="A sufficiently detailed test question?",
    topics=None,
    answer_text="A sufficiently detailed test answer for testing.",
    quality_flags=None,
    answer_characters=48,
    review_status="pending_review",
):
    if topics is None:
        topics = ["Anxiety"]
    if quality_flags is None:
        quality_flags = []
    return {
        "record_id": record_id,
        "source_dataset": source_dataset,
        "source_row_index": int(source_row_index),
        "question_title": question_title,
        "question_text": question_text,
        "topics": topics,
        "answer_text": answer_text,
        "answer_characters": answer_characters,
        "review_status": review_status,
        "quality_flags": quality_flags,
    }


def write_csv(path, records, fieldnames=None):
    fieldnames = fieldnames or FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 1. Pending records are not exported
def test_pending_records_are_not_exported(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record("pending_review")])

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    assert output_jsonl.read_text(encoding="utf-8") == ""


# 2. Valid approved records with explicit permission are exported
def test_valid_approved_record_is_exported(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record("approved", retrieval_permission="yes")])

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    records = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["record_id"] == "counselchat_0001"
    assert records[0]["review_status"] == "approved"
    assert records[0]["retrieval_permission"] == "yes"
    assert records[0]["clinical_approval"] == "not_assessed"
    assert records[0]["reviewer"] == "reviewer_1"
    assert records[0]["reviewed_at"] == "2026-11-03"
    assert records[0]["review_notes"] == "Reviewed for retrieval suitability."
    assert records[0]["question_text"] == "A sufficiently detailed test question?"
    assert records[0]["answer_text"] == "A sufficiently detailed test answer for testing."
    assert records[0]["topics"] == ["Anxiety"]


# 3. Approved records without retrieval permission are rejected
@pytest.mark.parametrize("bad_permission", ["no", "", "invalid"])
def test_approved_records_without_retrieval_permission_are_rejected(tmp_path, bad_permission):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("approved", retrieval_permission=bad_permission)
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 4. Approved records with unresolved quality flags are rejected
def test_approved_records_with_unresolved_quality_flags_are_rejected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record(quality_flags=["short_answer_review"])])
    record = make_record("approved", retrieval_permission="yes", quality_flags="short_answer_review")
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 5. Modified question text is detected
def test_modified_question_text_is_detected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record(question_text="Original question text?")])
    record = make_record("approved", retrieval_permission="yes", question_text="Modified question text?")
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 6. Modified answer text is detected
def test_modified_answer_text_is_detected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record(answer_text="Original answer text.")])
    record = make_record("approved", retrieval_permission="yes", answer_text="Altered answer text.")
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 7. Modified topics or source metadata are detected
@pytest.mark.parametrize(
    "override_key,override_val",
    [
        ("topics", "Depression"),
        ("source_dataset", "unauthorized/dataset"),
        ("source_row_index", "999"),
        ("question_title", "Altered Title"),
    ],
)
def test_modified_topics_or_source_metadata_is_detected(tmp_path, override_key, override_val):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("approved", retrieval_permission="yes")
    record[override_key] = override_val
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 8. Missing source records are detected
def test_missing_source_records_are_detected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(
        source_jsonl,
        [
            make_source_record("counselchat_0001"),
            make_source_record("counselchat_0002"),
        ],
    )
    write_csv(review_csv, [make_record("pending_review", record_id="counselchat_0001")])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 9. Unexpected records are detected
def test_unexpected_records_are_detected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record("counselchat_0001")])
    write_csv(
        review_csv,
        [
            make_record("pending_review", record_id="counselchat_0001"),
            make_record("pending_review", record_id="counselchat_9999"),
        ],
    )

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 10. Duplicate record IDs are detected
def test_duplicate_record_ids_in_review_queue_are_detected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record("counselchat_0001")])
    write_csv(
        review_csv,
        [
            make_record("pending_review", record_id="counselchat_0001"),
            make_record("pending_review", record_id="counselchat_0001"),
        ],
    )

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


def test_duplicate_record_ids_in_source_are_detected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(
        source_jsonl,
        [
            make_source_record("counselchat_0001"),
            make_source_record("counselchat_0001"),
        ],
    )
    write_csv(review_csv, [make_record("pending_review", record_id="counselchat_0001")])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 11. Restricted and rejected records cannot be exported
@pytest.mark.parametrize("status", ["restricted", "rejected"])
def test_restricted_and_rejected_records_are_not_exported(tmp_path, status):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record(status, retrieval_permission="no")])

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)
    assert output_jsonl.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("status", ["restricted", "rejected"])
def test_restricted_and_rejected_records_with_permission_yes_are_rejected(tmp_path, status):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record(status, retrieval_permission="yes")])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 12. Clinical approval cannot be changed from not_assessed through this workflow
@pytest.mark.parametrize("bad_clinical", ["approved", "clinically_verified", "true", "yes"])
def test_clinical_approval_cannot_be_changed(tmp_path, bad_clinical):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("approved", retrieval_permission="yes", clinical_approval=bad_clinical)
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# 13. Invalid review status and invalid permission are rejected
def test_invalid_review_status_is_rejected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record("automatically_approved", retrieval_permission="no")])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


def test_invalid_retrieval_permission_is_rejected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record("pending_review", retrieval_permission="maybe")])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# Additional safeguard tests
def test_approved_record_without_reviewer_is_rejected(tmp_path):
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("approved", retrieval_permission="yes")
    record["reviewer"] = ""
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


def test_failed_validation_removes_stale_approved_output(tmp_path):
    """If validation fails, a previously generated approved output file is removed."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    # Pre-seed stale output
    output_jsonl.write_text('{"stale": "data"}\n', encoding="utf-8")
    assert output_jsonl.exists()

    write_jsonl(source_jsonl, [make_source_record()])
    # CSV has modified question text which causes validation failure
    record = make_record("approved", retrieval_permission="yes", question_text="Modified?")
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    # Stale output must be removed
    assert not output_jsonl.exists()


def test_missing_required_csv_columns_is_rejected(tmp_path):
    """Missing required review queue columns (e.g. retrieval_permission) fails validation."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("pending_review")
    # Write CSV missing the 'retrieval_permission' column in header
    partial_fields = [f for f in FIELDS if f != "retrieval_permission"]
    partial_record = {k: v for k, v in record.items() if k != "retrieval_permission"}
    write_csv(review_csv, [partial_record], fieldnames=partial_fields)

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


def test_malformed_jsonl_is_rejected(tmp_path):
    """Malformed JSON syntax in source JSONL fails validation safely."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    source_jsonl.write_text('{"record_id": "counselchat_0001", "unclosed_json\n', encoding="utf-8")
    write_csv(review_csv, [make_record("pending_review")])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


@pytest.mark.parametrize("bad_idx", ["abc", "-1", ""])
def test_invalid_source_row_index_is_rejected(tmp_path, bad_idx):
    """Non-integer or negative source_row_index in review queue is rejected."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("pending_review", source_row_index=bad_idx)
    write_csv(review_csv, [record])

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


def test_exact_source_topics_and_quality_flags_preserved_in_output(tmp_path):
    """Exported record preserves exact original topics list and types without CSV corruption."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    original_topics = ["Family Conflict", "Marriage Counseling"]
    write_jsonl(source_jsonl, [make_source_record(topics=original_topics)])
    # In CSV, comma-separated representation
    write_csv(review_csv, [make_record("approved", retrieval_permission="yes", topics="Family Conflict, Marriage Counseling")])

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["topics"] == original_topics
    assert isinstance(records[0]["topics"], list)
    assert records[0]["quality_flags"] == []
    assert isinstance(records[0]["quality_flags"], list)


def test_json_array_topics_in_csv_cell_are_supported_without_comma_ambiguity(tmp_path):
    """Topics formatted as JSON array in CSV cell with internal commas are preserved unambiguously."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    original_topics = ["Trauma, Acute", "PTSD"]
    write_jsonl(source_jsonl, [make_source_record(topics=original_topics)])
    # Reviewer CSV uses JSON array format to handle internal comma cleanly
    write_csv(
        review_csv,
        [
            make_record(
                "approved",
                retrieval_permission="yes",
                topics=json.dumps(original_topics),
            )
        ],
    )

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["topics"] == ["Trauma, Acute", "PTSD"]


def test_failed_write_cleans_up_temp_file_and_never_publishes_partial_output(tmp_path, monkeypatch):
    """If an I/O error occurs during write, temporary files are removed and no output is published."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record("approved", retrieval_permission="yes")])

    def broken_dumps(*args, **kwargs):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(json, "dumps", broken_dumps)

    with pytest.raises(OSError, match="Simulated disk write failure"):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    # Output file must not have been published
    assert not output_jsonl.exists()
    # No temporary files should be left behind
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0


def test_malformed_csv_excess_columns_rejected(tmp_path):
    """CSV row containing excess/unmapped values is rejected."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])

    # Manually append an extra column value to a row
    with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(FIELDS) + "\n")
        f.write(
            "counselchat_0001,elvindavis/counselchat,1,Test,Question?,Anxiety,Answer.,,pending_review,no,,,not_assessed,EXTRA_VALUE\n"
        )

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


def test_malformed_csv_missing_column_values_rejected(tmp_path):
    """CSV row truncated with missing column values is rejected."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])

    # Manually write truncated row
    with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(FIELDS) + "\n")
        f.write("counselchat_0001,elvindavis/counselchat,1\n")

    with pytest.raises(SystemExit):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)


# Tests for FIX 1: Stale approved output protection
def test_stale_output_cleanup_when_source_file_missing(tmp_path):
    """Missing source file removes stale approved output before raising FileNotFoundError."""
    source_jsonl = tmp_path / "missing_source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    output_jsonl.write_text('{"stale": "data"}\n', encoding="utf-8")
    assert output_jsonl.exists()

    write_csv(review_csv, [make_record("pending_review")])

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    assert not output_jsonl.exists()


def test_stale_output_cleanup_when_input_file_missing(tmp_path):
    """Missing review queue CSV removes stale approved output before raising FileNotFoundError."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "missing_review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    write_jsonl(source_jsonl, [make_source_record()])
    output_jsonl.write_text('{"stale": "data"}\n', encoding="utf-8")
    assert output_jsonl.exists()

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    assert not output_jsonl.exists()


def test_stale_output_deletion_failure_raises_fatal_error(tmp_path, monkeypatch):
    """If stale approved output exists but cannot be deleted, raise a fatal RuntimeError."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    output_jsonl.write_text('{"stale": "data"}\n', encoding="utf-8")
    assert output_jsonl.exists()

    write_jsonl(source_jsonl, [make_source_record()])
    record = make_record("approved", retrieval_permission="yes", question_text="Modified text?")
    write_csv(review_csv, [record])

    orig_unlink = Path.unlink

    def mock_unlink(self, *args, **kwargs):
        if self.resolve() == output_jsonl.resolve():
            raise PermissionError("Simulated write-protect access denied")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", mock_unlink)

    with pytest.raises(RuntimeError, match="Fatal error: Unable to remove stale approved output file"):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    assert output_jsonl.exists()


def test_stale_output_deletion_failure_on_missing_source_raises_fatal_error(tmp_path, monkeypatch):
    """If stale approved output exists and source file is missing, deletion failure raises RuntimeError."""
    source_jsonl = tmp_path / "missing_source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    output_jsonl.write_text('{"stale": "data"}\n', encoding="utf-8")
    assert output_jsonl.exists()
    write_csv(review_csv, [make_record("pending_review")])

    orig_unlink = Path.unlink

    def mock_unlink(self, *args, **kwargs):
        if self.resolve() == output_jsonl.resolve():
            raise OSError("Simulated hardware error")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", mock_unlink)

    with pytest.raises(RuntimeError, match="Fatal error: Unable to remove stale approved output file"):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    assert output_jsonl.exists()


# Tests for FIX 2: Consistent JSON-array serialization
def test_prepare_review_queue_exports_json_arrays(tmp_path):
    """prepare_review_queue exports topics and quality_flags as valid JSON arrays."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"

    source_records = [
        make_source_record(
            "counselchat_0001",
            topics=["Anxiety", "Family Conflict"],
            quality_flags=[],
        ),
        make_source_record(
            "counselchat_0002",
            topics=["Trauma, Acute", "PTSD"],
            quality_flags=["short_answer_review"],
        ),
        make_source_record(
            "counselchat_0003",
            topics=[],
            quality_flags=[],
        ),
    ]
    write_jsonl(source_jsonl, source_records)

    prepare_review_queue(source_jsonl, review_csv)

    assert review_csv.exists()
    with review_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3

    # Row 1: multiple topics, empty flags
    assert json.loads(rows[0]["topics"]) == ["Anxiety", "Family Conflict"]
    assert json.loads(rows[0]["quality_flags"]) == []
    assert rows[0]["retrieval_permission"] == "no"
    assert rows[0]["review_status"] == "pending_review"
    assert rows[0]["clinical_approval"] == "not_assessed"

    # Row 2: topic with internal comma, quality flag
    assert json.loads(rows[1]["topics"]) == ["Trauma, Acute", "PTSD"]
    assert json.loads(rows[1]["quality_flags"]) == ["short_answer_review"]
    assert rows[1]["retrieval_permission"] == "no"
    assert rows[1]["review_status"] == "pending_review"

    # Row 3: empty topics and quality flags
    assert json.loads(rows[2]["topics"]) == []
    assert json.loads(rows[2]["quality_flags"]) == []


def test_prepare_and_validate_roundtrip_with_internal_commas(tmp_path):
    """Full roundtrip through exporter and validator with internal commas in topics."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    complex_topics = ["Trauma, Acute", "Family Conflict, High-Stakes"]
    write_jsonl(
        source_jsonl,
        [make_source_record("counselchat_0001", topics=complex_topics, quality_flags=[])],
    )

    prepare_review_queue(source_jsonl, review_csv)

    # Simulate reviewer approving the record
    with review_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    rows[0]["review_status"] = "approved"
    rows[0]["retrieval_permission"] = "yes"
    rows[0]["reviewer"] = "dr_alex"
    rows[0]["reviewed_at"] = "2026-11-03"
    rows[0]["review_notes"] = "Approved after clinical triage."

    write_csv(review_csv, rows)

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    approved = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(approved) == 1
    assert approved[0]["topics"] == complex_topics
    assert approved[0]["review_status"] == "approved"
    assert approved[0]["retrieval_permission"] == "yes"


def test_validator_backward_compatibility_comma_separated_topics(tmp_path):
    """Validator continues to accept legacy comma-separated topics in CSV."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    original_topics = ["Anxiety", "Family Conflict"]
    write_jsonl(source_jsonl, [make_source_record(topics=original_topics)])
    # Legacy CSV row with comma-separated topics and empty string quality_flags
    record = make_record(
        "approved",
        retrieval_permission="yes",
        topics="Anxiety, Family Conflict",
        quality_flags="",
    )
    write_csv(review_csv, [record])

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["topics"] == original_topics


def test_failed_write_does_not_accidentally_delete_previously_valid_output(tmp_path, monkeypatch):
    """If write or replace fails after validation succeeds, existing valid output is preserved."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    valid_content = '{"record_id": "prev_001", "review_status": "approved"}\n'
    output_jsonl.write_text(valid_content, encoding="utf-8")
    assert output_jsonl.exists()

    write_jsonl(source_jsonl, [make_source_record()])
    write_csv(review_csv, [make_record("approved", retrieval_permission="yes")])

    def broken_dumps(*args, **kwargs):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(json, "dumps", broken_dumps)

    with pytest.raises(OSError, match="Simulated disk write failure"):
        build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    # Existing valid output must NOT have been destroyed
    assert output_jsonl.exists()
    assert output_jsonl.read_text(encoding="utf-8") == valid_content
    # No temporary files left behind
    assert len(list(tmp_path.glob("*.tmp_*"))) == 0


def test_unicode_values_survive_export_and_approval(tmp_path):
    """Unicode characters in questions, answers, and topics survive CSV round trip."""
    source_jsonl = tmp_path / "source.jsonl"
    review_csv = tmp_path / "review.csv"
    output_jsonl = tmp_path / "approved.jsonl"

    unicode_topics = ["Santé Mentale", "焦虑与抑郁", "Terapia de Pareja"]
    unicode_q = "J'ai besoin d'aide avec mon anxiété: ¿Qué puedo hacer? 🌸"
    unicode_a = "Voici une réponse bienveillante avec des conseils: respira profundamente. 🌟"

    source_record = make_source_record(
        "counselchat_0001",
        question_title="Titre en français",
        question_text=unicode_q,
        answer_text=unicode_a,
        topics=unicode_topics,
        quality_flags=[],
    )
    write_jsonl(source_jsonl, [source_record])

    prepare_review_queue(source_jsonl, review_csv)

    with review_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    rows[0]["review_status"] = "approved"
    rows[0]["retrieval_permission"] = "yes"
    rows[0]["reviewer"] = "dr_rené"
    rows[0]["reviewed_at"] = "2026-11-03"
    rows[0]["review_notes"] = "Validation clinique réussie."

    write_csv(review_csv, rows)

    build_approved_dataset(review_csv, output_jsonl, source_path=source_jsonl)

    approved = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(approved) == 1
    assert approved[0]["question_text"] == unicode_q
    assert approved[0]["answer_text"] == unicode_a
    assert approved[0]["topics"] == unicode_topics
    assert approved[0]["reviewer"] == "dr_rené"
