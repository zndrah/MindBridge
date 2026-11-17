import argparse
import csv
import json
from datetime import date
import os
from pathlib import Path
import re

DEFAULT_INPUT = Path(
    "data/processed/counselchat/counselchat_review_queue.csv"
)
DEFAULT_SOURCE = Path(
    "data/processed/counselchat/counselchat_cleaned.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/processed/counselchat/counselchat_approved.jsonl"
)

ALLOWED_STATUSES = {
    "pending_review",
    "approved",
    "restricted",
    "rejected",
}

ALLOWED_PERMISSIONS = {
    "yes",
    "no",
}

REQUIRED_CSV_COLUMNS = [
    "record_id",
    "source_dataset",
    "source_row_index",
    "question_title",
    "question_text",
    "topics",
    "answer_text",
    "quality_flags",
    "review_status",
    "retrieval_permission",
    "review_notes",
    "reviewer",
    "reviewed_at",
    "clinical_approval",
]


def cleanup_stale_output(output_path):
    """Safely remove stale approved output file on validation or execution failure.

    Raises RuntimeError if the stale file exists but cannot be removed.
    """
    output_path = Path(output_path)
    if output_path.exists():
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RuntimeError(
                f"Fatal error: Unable to remove stale approved output file '{output_path}': {exc}"
            ) from exc


def normalize_list_field(val):
    """Normalize list, JSON-array string, or comma-separated string to a list of stripped non-empty strings."""
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (json.JSONDecodeError, ValueError):
                pass
        return [item.strip() for item in val.split(",") if item.strip()]
    return []


def safe_str(val):
    """Safely convert value to string, converting None to empty string."""
    return "" if val is None else str(val)


def validate_record(record, row_number, orig_record=None):
    errors = []
    record_id = safe_str(record.get("record_id")).strip() or f"Row_{row_number}"

    # 1. Validate source_row_index numeric format
    raw_idx = safe_str(record.get("source_row_index")).strip()
    try:
        csv_idx = int(raw_idx)
        if csv_idx < 0:
            errors.append(
                f"Row {row_number} (Record {record_id}): source_row_index must be non-negative, got '{raw_idx}'"
            )
    except (ValueError, TypeError):
        errors.append(
            f"Row {row_number} (Record {record_id}): invalid source_row_index '{raw_idx}' (must be integer)"
        )

    # 2. Source-integrity checks against original source record
    if orig_record is not None:
        if safe_str(record.get("record_id")).strip() != safe_str(orig_record.get("record_id")).strip():
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'record_id' modified"
            )

        csv_ds = safe_str(record.get("source_dataset")).strip()
        orig_ds = safe_str(orig_record.get("source_dataset")).strip()
        if csv_ds != orig_ds and not ({csv_ds, orig_ds} <= {"elvindavis/counselchat", "weiting016/counselchat-data"}):
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'source_dataset' modified"
            )

        if safe_str(record.get("source_row_index")).strip() != safe_str(orig_record.get("source_row_index")).strip():
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'source_row_index' modified"
            )

        # Content fields: must match source record (allowing equivalent whitespace formatting)
        csv_qtitle = re.sub(r"\s+", " ", safe_str(record.get("question_title"))).strip()
        orig_qtitle = re.sub(r"\s+", " ", safe_str(orig_record.get("question_title"))).strip()
        if csv_qtitle != orig_qtitle:
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'question_title' modified"
            )

        csv_qtext = re.sub(r"\s+", " ", safe_str(record.get("question_text"))).strip()
        orig_qtext = re.sub(r"\s+", " ", safe_str(orig_record.get("question_text"))).strip()
        if csv_qtext != orig_qtext:
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'question_text' modified"
            )

        csv_atext = re.sub(r"\s+", " ", safe_str(record.get("answer_text"))).strip()
        orig_atext = re.sub(r"\s+", " ", safe_str(orig_record.get("answer_text"))).strip()
        if csv_atext != orig_atext:
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'answer_text' modified"
            )

        # List fields: normalize CSV and original representations before comparison
        csv_topics = normalize_list_field(record.get("topics", ""))
        orig_topics = normalize_list_field(orig_record.get("topics", []))
        if csv_topics != orig_topics:
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'topics' modified"
            )

        csv_flags = normalize_list_field(record.get("quality_flags", ""))
        orig_flags = normalize_list_field(orig_record.get("quality_flags", []))
        if csv_flags != orig_flags:
            errors.append(
                f"Row {row_number} (Record {record_id}): source field 'quality_flags' modified"
            )

    # 3. Review status validation
    status = safe_str(record.get("review_status")).strip().lower()
    if status not in ALLOWED_STATUSES:
        errors.append(
            f"Row {row_number} (Record {record_id}): invalid review_status '{record.get('review_status')}'"
        )

    # 4. Retrieval permission validation
    raw_perm = record.get("retrieval_permission")
    permission = safe_str(raw_perm).strip().lower()
    if permission not in ALLOWED_PERMISSIONS:
        errors.append(
            f"Row {row_number} (Record {record_id}): invalid retrieval_permission '{record.get('retrieval_permission')}' (must be 'yes' or 'no')"
        )
    else:
        if status == "approved" and permission != "yes":
            errors.append(
                f"Row {row_number} (Record {record_id}): approved record must have retrieval_permission='yes'"
            )
        elif status in {"pending_review", "restricted", "rejected"} and permission != "no":
            errors.append(
                f"Row {row_number} (Record {record_id}): {status} record must have retrieval_permission='no', found '{permission}'"
            )

    # 5. Clinical approval validation (must remain 'not_assessed')
    clinical = safe_str(record.get("clinical_approval")).strip().lower()
    if clinical != "not_assessed":
        errors.append(
            f"Row {row_number} (Record {record_id}): clinical_approval must remain 'not_assessed', found '{record.get('clinical_approval')}'"
        )

    # 6. Approved record mandatory fields
    if status == "approved":
        for field in ("reviewer", "reviewed_at", "review_notes"):
            if not safe_str(record.get(field)).strip():
                errors.append(
                    f"Row {row_number} (Record {record_id}): approved record missing {field}"
                )

        reviewer = safe_str(record.get("reviewer")).strip()
        if reviewer.upper() in {"REDACTED_EXAMPLE_REVIEWER", "NONE", "N/A", "UNKNOWN", "TEST", "ADMIN", "USER"}:
            errors.append(
                f"Row {row_number} (Record {record_id}): approved record has placeholder reviewer '{reviewer}' "
                "(actual human reviewer identity required for production approval)"
            )

        notes = safe_str(record.get("review_notes")).strip()
        if notes and len(notes) < 10:
            errors.append(
                f"Row {row_number} (Record {record_id}): review_notes must contain substantive justification (at least 10 characters)"
            )

        reviewed_at = safe_str(record.get("reviewed_at")).strip()
        if reviewed_at:
            try:
                date.fromisoformat(reviewed_at)
            except ValueError:
                errors.append(
                    f"Row {row_number} (Record {record_id}): reviewed_at must use YYYY-MM-DD"
                )

        if not safe_str(record.get("question_text")).strip():
            errors.append(
                f"Row {row_number} (Record {record_id}): approved record has no question text"
            )

        if not safe_str(record.get("answer_text")).strip():
            errors.append(
                f"Row {row_number} (Record {record_id}): approved record has no answer text"
            )

        # Unresolved quality flags check
        active_flags = (
            normalize_list_field(orig_record.get("quality_flags", []))
            if orig_record is not None
            else normalize_list_field(record.get("quality_flags", ""))
        )
        if active_flags:
            errors.append(
                f"Row {row_number} (Record {record_id}): approved record has unresolved quality_flags: {', '.join(active_flags)}"
            )

        # Classification safeguard: Restricted and excluded records cannot be approved
        cls_val = safe_str(
            record.get("classification")
            or record.get("automated_review_category")
            or (orig_record.get("classification") if orig_record else "")
            or (orig_record.get("automated_review_category") if orig_record else "")
        ).strip().upper()
        if cls_val in {"RESTRICTED_SAFETY_EVALUATION", "EXCLUDE_FROM_RETRIEVAL"}:
            errors.append(
                f"Row {row_number} (Record {record_id}): record with classification '{cls_val}' cannot be approved into retrieval"
            )

    return errors


def build_approved_dataset(input_path, output_path, source_path=None, allow_subset=False):
    input_path = Path(input_path)
    output_path = Path(output_path)
    source_path = Path(source_path) if source_path is not None else DEFAULT_SOURCE

    if not input_path.exists():
        cleanup_stale_output(output_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not source_path.exists():
        cleanup_stale_output(output_path)
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # 1. Load original source dataset from JSONL with strict decoding
    orig_records_by_id = {}
    orig_id_order = []
    errors = []

    with source_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                orig = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"Source JSONL line {line_number}: malformed JSON ({type(exc).__name__}: {exc})"
                )
                continue

            if not isinstance(orig, dict):
                errors.append(
                    f"Source JSONL line {line_number}: line is not a JSON object"
                )
                continue

            rec_id = str(orig.get("record_id", "")).strip()
            if not rec_id:
                errors.append(f"Source JSONL line {line_number}: missing record_id")
                continue
            if rec_id in orig_records_by_id:
                errors.append(
                    f"Source JSONL line {line_number}: duplicate record_id '{rec_id}' in original dataset"
                )
            else:
                orig_records_by_id[rec_id] = orig
                orig_id_order.append(rec_id)

            # Check source_row_index validity in original record
            orig_idx_raw = orig.get("source_row_index")
            try:
                orig_idx = int(str(orig_idx_raw).strip())
                if orig_idx < 0:
                    errors.append(
                        f"Source JSONL line {line_number} (Record {rec_id}): source_row_index must be non-negative, got '{orig_idx_raw}'"
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"Source JSONL line {line_number} (Record {rec_id}): invalid source_row_index '{orig_idx_raw}' (must be integer)"
                )

    if not orig_records_by_id and not errors:
        cleanup_stale_output(output_path)
        raise ValueError("Source dataset is empty or has no records.")

    # 2. Load review queue from CSV with column validation
    fieldnames = None
    records = []
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            if not fieldnames:
                cleanup_stale_output(output_path)
                raise ValueError("Review CSV is empty or has no header.")
            missing_cols = [col for col in REQUIRED_CSV_COLUMNS if col not in fieldnames]
            if missing_cols:
                errors.append(
                    f"Review CSV is missing required columns: {', '.join(missing_cols)}"
                )
            records = list(reader)
    except (csv.Error, UnicodeDecodeError) as exc:
        errors.append(f"Review CSV parsing error: {type(exc).__name__}: {exc}")

    # Check for malformed rows in review CSV
    for row_number, record in enumerate(records, start=2):
        if None in record:
            errors.append(f"Row {row_number}: malformed CSV row (excess fields detected)")
        if any(record.get(col) is None for col in REQUIRED_CSV_COLUMNS if col in (fieldnames or [])):
            errors.append(f"Row {row_number}: malformed CSV row (missing column values)")

    # 3. Check for duplicates in review queue and cross-dataset integrity
    seen_csv_ids = set()
    for row_number, record in enumerate(records, start=2):
        rec_id = record.get("record_id", "").strip() if record.get("record_id") else ""
        if not rec_id:
            errors.append(f"Row {row_number}: missing record_id")
        elif rec_id in seen_csv_ids:
            errors.append(
                f"Row {row_number}: duplicate record_id '{rec_id}' in review queue"
            )
        else:
            seen_csv_ids.add(rec_id)

    if not allow_subset:
        missing_ids = [rid for rid in orig_id_order if rid not in seen_csv_ids]
        if missing_ids:
            errors.append(
                f"Missing original records from review queue ({len(missing_ids)} missing): "
                f"{', '.join(missing_ids[:10])}{'...' if len(missing_ids) > 10 else ''}"
            )

    unexpected_ids = [rid for rid in seen_csv_ids if rid not in orig_records_by_id]
    if unexpected_ids:
        errors.append(
            f"Unexpected records found in review queue ({len(unexpected_ids)} unexpected): "
            f"{', '.join(unexpected_ids[:10])}{'...' if len(unexpected_ids) > 10 else ''}"
        )

    # 4. Validate each record in review queue
    approved_records = []
    for row_number, record in enumerate(records, start=2):
        rec_id = record.get("record_id", "").strip() if record.get("record_id") else ""
        orig_record = orig_records_by_id.get(rec_id)
        row_errors = validate_record(record, row_number, orig_record=orig_record)
        errors.extend(row_errors)

        if not row_errors and record.get("review_status", "").strip().lower() == "approved":
            approved_records.append(record)

    # Stale-output protection: if validation fails, ensure previous output cannot linger
    if errors:
        cleanup_stale_output(output_path)
        print("Approval validation failed:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    # 5. Output approved JSONL with atomic replacement
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(f"{output_path.name}.tmp_{os.getpid()}")

    try:
        with tmp_output.open("w", encoding="utf-8") as file:
            for record in approved_records:
                orig = orig_records_by_id[record["record_id"]]
                output_record = dict(orig)
                output_record["review_status"] = "approved"
                output_record["retrieval_permission"] = "yes"
                output_record["reviewer"] = record["reviewer"].strip()
                output_record["reviewed_at"] = record["reviewed_at"].strip()
                output_record["review_notes"] = record["review_notes"].strip()
                output_record["clinical_approval"] = "not_assessed"

                # Ensure proper typed representation
                if isinstance(output_record.get("topics"), str):
                    output_record["topics"] = normalize_list_field(output_record["topics"])
                if isinstance(output_record.get("quality_flags"), str):
                    output_record["quality_flags"] = normalize_list_field(output_record["quality_flags"])
                if "source_row_index" in output_record:
                    try:
                        output_record["source_row_index"] = int(output_record["source_row_index"])
                    except (ValueError, TypeError):
                        pass

                file.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        tmp_output.replace(output_path)
    except Exception:
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except OSError:
                pass
        raise

    print("\n=== COUNSELCHAT APPROVAL VALIDATION ===")
    print(f"Total records checked: {len(records)}")
    print(f"Approved for retrieval: {len(approved_records)}")
    print("Excluded records:", len(records) - len(approved_records))
    print(f"Output: {output_path}")
    print("Clinical approval: not assessed")
    print("FAISS indexing: not performed")


def main():
    parser = argparse.ArgumentParser(
        description="Validate human-reviewed CounselChat records."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow validating a review batch or subset without requiring all 1,482 records.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        cleanup_stale_output(args.output)
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.source.exists():
        cleanup_stale_output(args.output)
        raise FileNotFoundError(f"Source file not found: {args.source}")

    build_approved_dataset(
        args.input,
        args.output,
        source_path=args.source,
        allow_subset=args.allow_subset,
    )


if __name__ == "__main__":
    main()
