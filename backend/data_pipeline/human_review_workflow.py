"""MindBridge 2.0 - CounselChat Human Review Workflow Manager
===========================================================

This module establishes a structured, safe, auditable human-review workflow for the
1,063 `SUITABLE_CANDIDATE` records identified in Phase 2A and reconciled with the 9-column
canonical CounselChat dataset.

Key Capabilities:
1. Batch Partitioning:
   - Partitions the 1,063 suitable candidates into manageable batches of 50 records.
   - 21 batches of 50 records + 1 batch of 13 records = 22 batches.
   - Preserves all 9 original columns (questionID, questionTitle, questionText, topics,
     therapistName, therapistUrl, answerText, questionUrl, upvotes) alongside stable record IDs,
     duplicate-group tracking, and automated classification metadata.
2. Invariant Initialization:
   - All records are initialized to `human_decision = "pending"`.
   - All records are initialized to `review_status = "pending_review"`.
   - All records are initialized to `retrieval_permission = "no"`.
   - All records maintain `clinical_approval = "not_assessed"`.
   - Reviewer, reviewed date, and review notes are initialized to empty strings / null.
   - Zero automated approvals, zero simulated reviewer names, zero synthetic dates.
3. Reviewer Safeguards & Decision Validation:
   - `validate_human_review_record()`: Enforces that any approved record has a valid reviewer,
     a valid ISO date, non-empty review notes, retrieval permission explicitly 'yes',
     and clinical approval strictly 'not_assessed'.
   - Records with active quality flags or classified as RESTRICTED_SAFETY_EVALUATION or
     EXCLUDE_FROM_RETRIEVAL cannot be approved into user-facing retrieval.
4. Manifest & Progress Tracking:
   - Exports a machine-readable manifest tracking batch statuses, progress, and review criteria.
"""

from collections import Counter
import csv
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_CANONICAL_JSONL = Path("data/processed/counselchat/counselchat_canonical.jsonl")
DEFAULT_FULL_REVIEW_JSONL = Path("data/processed/counselchat/counselchat_full_review.jsonl")
DEFAULT_SOURCE_CLEANED = Path("data/processed/counselchat/counselchat_cleaned.jsonl")
DEFAULT_CANDIDATE_QUEUE_CSV = Path("data/processed/counselchat/human_review_queue.csv")
DEFAULT_BATCHES_DIR = Path("data/processed/counselchat/human_review_batches")
DEFAULT_MANIFEST_PATH = Path("data/processed/counselchat/human_review_manifest.json")
DEFAULT_APPROVED_OUTPUT = Path("data/processed/counselchat/counselchat_approved.jsonl")

DEFAULT_BATCH_SIZE = 50

ALLOWED_HUMAN_DECISIONS = {
    "pending",
    "approve",
    "reject",
    "needs_further_review",
}

ALLOWED_REVIEW_STATUSES = {
    "pending_review",
    "approved",
    "rejected",
    "restricted",
}

ALLOWED_PERMISSIONS = {
    "yes",
    "no",
}

BATCH_CSV_COLUMNS = [
    "record_id",
    "question_id",
    "source_dataset",
    "source_row_index",
    "batch_id",
    "batch_index",
    "question_title",
    "question_text",
    "topics",
    "therapist_name",
    "therapist_url",
    "answer_text",
    "question_url",
    "upvotes",
    "answer_characters",
    "duplicate_group_id",
    "duplicate_group_size",
    "quality_flags",
    "classification",
    "classification_reason",
    "human_decision",
    "review_status",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "retrieval_permission",
    "clinical_approval",
]

DECISION_TO_STATUS_MAP = {
    "pending": "pending_review",
    "approve": "approved",
    "reject": "rejected",
    "needs_further_review": "restricted",
}

STATUS_TO_DECISION_MAP = {
    "pending_review": "pending",
    "approved": "approve",
    "rejected": "reject",
    "restricted": "needs_further_review",
}


def create_human_review_batches(
    input_jsonl_path: Optional[Path] = None,
    output_queue_csv_path: Path = DEFAULT_CANDIDATE_QUEUE_CSV,
    batches_dir: Path = DEFAULT_BATCHES_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Extract all SUITABLE_CANDIDATE records and partition them into auditable review batches."""
    if input_jsonl_path is None:
        input_jsonl_path = DEFAULT_CANONICAL_JSONL if DEFAULT_CANONICAL_JSONL.exists() else DEFAULT_FULL_REVIEW_JSONL

    input_jsonl_path = Path(input_jsonl_path)
    output_queue_csv_path = Path(output_queue_csv_path)
    batches_dir = Path(batches_dir)
    manifest_path = Path(manifest_path)

    if not input_jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    # 1. Read classified records from canonical or full review dataset
    suitable_candidates: List[Dict[str, Any]] = []
    with input_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            classification = record.get("automated_review_category", record.get("classification", ""))
            if classification == "SUITABLE_CANDIDATE":
                suitable_candidates.append(record)

    total_candidates = len(suitable_candidates)
    if total_candidates == 0:
        raise ValueError("No SUITABLE_CANDIDATE records found in input dataset.")

    # 2. Partition into batches of batch_size
    batches_dir.mkdir(parents=True, exist_ok=True)
    output_queue_csv_path.parent.mkdir(parents=True, exist_ok=True)

    batch_manifests: List[Dict[str, Any]] = []
    all_queue_rows: List[Dict[str, Any]] = []

    total_batches = (total_candidates + batch_size - 1) // batch_size

    for b_idx in range(total_batches):
        b_num = b_idx + 1
        b_id = f"batch_{b_num:02d}"
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, total_candidates)
        batch_slice = suitable_candidates[start_idx:end_idx]

        batch_rows: List[Dict[str, Any]] = []
        batch_record_ids: List[str] = []

        for in_batch_idx, cand in enumerate(batch_slice, start=1):
            rid = cand.get("stable_record_id", cand.get("record_id", ""))
            batch_record_ids.append(rid)

            topics_val = cand.get("topics", [])
            topics_field = json.dumps(topics_val, ensure_ascii=False) if isinstance(topics_val, list) else str(topics_val)

            flags_val = cand.get("quality_flags", [])
            flags_field = json.dumps(flags_val, ensure_ascii=False) if isinstance(flags_val, list) else str(flags_val)

            q_text = re.sub(r"\s+", " ", str(cand.get("question_text") or cand.get("questionText") or "")).strip()
            q_title = re.sub(r"\s+", " ", str(cand.get("question_title") or cand.get("questionTitle") or "")).strip()
            ans_clean = str(cand.get("clean_answer_text") or cand.get("answer_text") or "").strip()
            ans_chars = len(ans_clean)

            row = {
                "record_id": rid,
                "question_id": str(cand.get("questionID", cand.get("question_id", ""))),
                "source_dataset": str(cand.get("source_dataset", "weiting016/counselchat-data")),
                "source_row_index": cand.get("source_row_index", 0),
                "batch_id": b_id,
                "batch_index": in_batch_idx,
                "question_title": q_title,
                "question_text": q_text,
                "topics": topics_field,
                "therapist_name": str(cand.get("therapistName", cand.get("therapist_name", ""))),
                "therapist_url": str(cand.get("therapistUrl", cand.get("therapist_url", ""))),
                "answer_text": ans_clean,
                "question_url": str(cand.get("questionUrl", cand.get("question_url", ""))),
                "upvotes": cand.get("upvotes", 0),
                "answer_characters": ans_chars,
                "duplicate_group_id": str(cand.get("duplicate_group_id", "")),
                "duplicate_group_size": cand.get("duplicate_group_size", 1),
                "quality_flags": flags_field,
                "classification": "SUITABLE_CANDIDATE",
                "classification_reason": str(cand.get("automated_review_reasons", cand.get("classification_reason", ""))),
                "human_decision": "pending",
                "review_status": "pending_review",
                "reviewer": "",
                "reviewed_at": "",
                "review_notes": "",
                "retrieval_permission": "no",
                "clinical_approval": "not_assessed",
            }
            batch_rows.append(row)
            all_queue_rows.append(row)

        # Write individual batch CSV
        batch_file_path = batches_dir / f"{b_id}.csv"
        with batch_file_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BATCH_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(batch_rows)

        batch_manifests.append({
            "batch_id": b_id,
            "batch_number": b_num,
            "file_name": batch_file_path.name,
            "relative_path": str(batch_file_path),
            "record_count": len(batch_rows),
            "start_record_id": batch_record_ids[0],
            "end_record_id": batch_record_ids[-1],
            "status": "pending_review",
            "reviewed_count": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "needs_further_review_count": 0,
        })

    # 3. Write unified candidate review queue CSV
    with output_queue_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_queue_rows)

    # 4. Write manifest JSON
    manifest = {
        "workflow": "MindBridge Phase 2B Human Review Batch Workflow",
        "dataset": "weiting016/counselchat-data",
        "source_candidates_file": str(input_jsonl_path),
        "unified_queue_csv": str(output_queue_csv_path),
        "batches_directory": str(batches_dir),
        "total_candidates": total_candidates,
        "batch_size": batch_size,
        "total_batches": len(batch_manifests),
        "governance_invariants": {
            "all_records_initially_pending": True,
            "retrieval_permission_strictly_no": True,
            "clinical_approval_strictly_not_assessed": True,
            "automated_approvals_prohibited": True,
            "simulated_reviewers_prohibited": True,
        },
        "review_decision_options": {
            "approve": "Constructive, respectful, psychoeducational Q&A safe for secondary RAG grounding.",
            "reject": "Harmful, misleading, shaming, invalidating, diagnostic, or inappropriate advice.",
            "needs_further_review": "Ambiguous, sensitive, incomplete, or requires senior clinician assessment.",
            "pending": "Not yet evaluated by a human reviewer.",
        },
        "batch_summary": batch_manifests,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def validate_human_review_record(
    record: Dict[str, Any],
    row_num: int = 1,
    orig_record: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate that a human-reviewed candidate record satisfies all approval safeguards.

    Returns:
        List of error messages. Empty list indicates record is valid.
    """
    errors: List[str] = []
    rid = str(record.get("record_id", "")).strip() or f"Row_{row_num}"

    # 1. Decision validation
    raw_dec = str(record.get("human_decision", "")).strip().lower()
    if raw_dec not in ALLOWED_HUMAN_DECISIONS:
        errors.append(
            f"Row {row_num} ({rid}): invalid human_decision '{record.get('human_decision')}' "
            f"(must be one of: {', '.join(sorted(ALLOWED_HUMAN_DECISIONS))})"
        )

    # 2. Review status validation (must match decision)
    raw_status = str(record.get("review_status", "")).strip().lower()
    expected_status = DECISION_TO_STATUS_MAP.get(raw_dec, "pending_review")
    if raw_status and raw_status != expected_status:
        errors.append(
            f"Row {row_num} ({rid}): review_status '{raw_status}' does not match human_decision '{raw_dec}' "
            f"(expected '{expected_status}')"
        )

    # 3. Retrieval permission validation
    perm = str(record.get("retrieval_permission", "")).strip().lower()
    if perm not in ALLOWED_PERMISSIONS:
        errors.append(
            f"Row {row_num} ({rid}): invalid retrieval_permission '{record.get('retrieval_permission')}' (must be 'yes' or 'no')"
        )
    else:
        if raw_dec == "approve" and perm != "yes":
            errors.append(f"Row {row_num} ({rid}): approved record must have retrieval_permission='yes'")
        elif raw_dec in {"pending", "reject", "needs_further_review"} and perm != "no":
            errors.append(f"Row {row_num} ({rid}): non-approved record must have retrieval_permission='no', found '{perm}'")

    # 4. Clinical approval separation invariant
    clin = str(record.get("clinical_approval", "")).strip().lower()
    if clin != "not_assessed":
        errors.append(
            f"Row {row_num} ({rid}): clinical_approval must remain 'not_assessed' (governance status), found '{record.get('clinical_approval')}'"
        )

    # 5. Approved record mandatory fields & safeguards
    if raw_dec == "approve":
        reviewer = str(record.get("reviewer", "")).strip()
        if not reviewer:
            errors.append(f"Row {row_num} ({rid}): approved record missing mandatory 'reviewer'")
        elif reviewer.upper() in {"REDACTED_EXAMPLE_REVIEWER", "NONE", "N/A", "UNKNOWN", "TEST", "ADMIN", "USER"}:
            errors.append(
                f"Row {row_num} ({rid}): approved record has placeholder reviewer '{reviewer}' "
                "(actual human reviewer identity required for production approval)"
            )

        reviewed_at = str(record.get("reviewed_at", "")).strip()
        if not reviewed_at:
            errors.append(f"Row {row_num} ({rid}): approved record missing mandatory 'reviewed_at'")
        else:
            try:
                date.fromisoformat(reviewed_at)
            except ValueError:
                errors.append(
                    f"Row {row_num} ({rid}): 'reviewed_at' must be a valid ISO YYYY-MM-DD date, got '{reviewed_at}'"
                )

        notes = str(record.get("review_notes", "")).strip()
        if not notes:
            errors.append(f"Row {row_num} ({rid}): approved record missing mandatory 'review_notes'")

        q_text = str(record.get("question_text", "")).strip()
        if not q_text:
            errors.append(f"Row {row_num} ({rid}): approved record missing question_text")

        a_text = str(record.get("answer_text", "")).strip()
        if not a_text:
            errors.append(f"Row {row_num} ({rid}): approved record missing answer_text")

        # Zero unresolved quality flags
        q_flags = record.get("quality_flags", [])
        if isinstance(q_flags, str):
            q_flags_clean = [f.strip() for f in q_flags.strip("[]").replace('"', "").split(",") if f.strip()]
        elif isinstance(q_flags, list):
            q_flags_clean = [str(f).strip() for f in q_flags if str(f).strip()]
        else:
            q_flags_clean = []

        if q_flags_clean:
            errors.append(f"Row {row_num} ({rid}): approved record has unresolved quality_flags: {q_flags_clean}")

        # Classification safeguard: Restricted and excluded records cannot be approved
        cls_val = str(record.get("classification", "")).strip().upper()
        if cls_val in {"RESTRICTED_SAFETY_EVALUATION", "EXCLUDE_FROM_RETRIEVAL"}:
            errors.append(
                f"Row {row_num} ({rid}): record with classification '{cls_val}' cannot be approved into retrieval"
            )

    # 6. Source-integrity checks against original source record
    if orig_record is not None:
        orig_rid = str(orig_record.get("record_id", orig_record.get("stable_record_id", ""))).strip()
        if str(record.get("record_id", "")).strip() != orig_rid:
            errors.append(f"Row {row_num} ({rid}): record_id modified from original source record")
        if str(record.get("source_row_index", "")).strip() != str(orig_record.get("source_row_index", "")).strip():
            errors.append(f"Row {row_num} ({rid}): source_row_index modified from original source record")

    return errors


def validate_batch_file(
    batch_file_path: Path,
    source_cleaned_path: Path = DEFAULT_SOURCE_CLEANED,
) -> Tuple[List[str], Dict[str, int]]:
    """Validate all rows in a batch CSV file against approval safeguards.

    Returns:
        (errors_list, decision_counts)
    """
    batch_file_path = Path(batch_file_path)
    source_cleaned_path = Path(source_cleaned_path)

    if not batch_file_path.exists():
        return [f"Batch file not found: {batch_file_path}"], {}

    # Load source records for integrity checking
    orig_records: Dict[str, Dict[str, Any]] = {}
    if source_cleaned_path.exists():
        with source_cleaned_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rec = json.loads(line_str)
                    orig_records[rec.get("record_id", rec.get("stable_record_id", ""))] = rec

    errors: List[str] = []
    decision_counts = Counter()

    with batch_file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            rid = row.get("record_id", "")
            orig = orig_records.get(rid)
            row_errors = validate_human_review_record(row, row_num=row_num, orig_record=orig)
            errors.extend(row_errors)

            dec = str(row.get("human_decision", "pending")).strip().lower()
            decision_counts[dec] += 1

    return errors, dict(decision_counts)


def compile_approved_records_from_batches(
    batch_file_paths: List[Path],
    output_approved_path: Path = DEFAULT_APPROVED_OUTPUT,
    source_cleaned_path: Path = DEFAULT_SOURCE_CLEANED,
) -> int:
    """Compile verified approved records from human-reviewed batches into counselchat_approved.jsonl.

    Enforces that:
    1. Only records with human_decision == 'approve' and retrieval_permission == 'yes' are compiled.
    2. Zero validation errors exist across the compiled records.
    3. If zero records are approved, counselchat_approved.jsonl is left empty (0 bytes).
    4. Clinical approval remains strictly 'not_assessed'.

    Returns:
        Number of approved records written.
    """
    output_approved_path = Path(output_approved_path)
    source_cleaned_path = Path(source_cleaned_path)

    orig_records: Dict[str, Dict[str, Any]] = {}
    if source_cleaned_path.exists():
        with source_cleaned_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rec = json.loads(line_str)
                    orig_records[rec.get("record_id", rec.get("stable_record_id", ""))] = rec

    all_errors: List[str] = []
    approved_records: List[Dict[str, Any]] = []

    for bpath in batch_file_paths:
        bpath = Path(bpath)
        if not bpath.exists():
            all_errors.append(f"Batch file not found: {bpath}")
            continue

        with bpath.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                rid = row.get("record_id", "")
                orig = orig_records.get(rid)
                dec = str(row.get("human_decision", "")).strip().lower()

                if dec == "approve":
                    row_errors = validate_human_review_record(row, row_num=row_num, orig_record=orig)
                    all_errors.extend(row_errors)
                    if not row_errors:
                        output_rec = dict(orig) if orig else dict(row)
                        output_rec["review_status"] = "approved"
                        output_rec["retrieval_permission"] = "yes"
                        output_rec["reviewer"] = row["reviewer"].strip()
                        output_rec["reviewed_at"] = row["reviewed_at"].strip()
                        output_rec["review_notes"] = row["review_notes"].strip()
                        output_rec["clinical_approval"] = "not_assessed"
                        approved_records.append(output_rec)

    if all_errors:
        raise ValueError(f"Approval validation failed with {len(all_errors)} errors: {all_errors[:5]}")

    # Atomic write to approved output
    output_approved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_approved_path.with_name(f"{output_approved_path.name}.tmp_{os.getpid()}")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for rec in approved_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.replace(output_approved_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return len(approved_records)


if __name__ == "__main__":
    man = create_human_review_batches()
    print("=== COUNSELCHAT HUMAN REVIEW BATCH WORKFLOW CREATED ===")
    print(f"Total Candidates: {man['total_candidates']}")
    print(f"Total Batches: {man['total_batches']} (Batch size: {man['batch_size']})")
    print(f"Unified Queue: {man['unified_queue_csv']}")
    print(f"Batches Directory: {man['batches_directory']}")
    print(f"Manifest: {DEFAULT_MANIFEST_PATH}")
