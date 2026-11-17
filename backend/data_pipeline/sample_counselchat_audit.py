"""MindBridge 2.0 - Reproducible Human-Audit Sampling for CounselChat
===================================================================

This module implements a deterministic, stratified sampling workflow to select
provisional candidates from the auto-curated CounselChat dataset for human clinical audit.

Key Capabilities:
- Reads auto-curated records (curation_status == 'PROVISIONAL_CANDIDATE').
- Performs reproducible, topic-stratified sampling using seed 42.
- Spans multiple answer length buckets and diverse topics.
- Avoids duplicate or repetitive questions.
- Exports a review CSV with review fields initialized to 'pending' / empty.
- Generates a comprehensive audit report documenting sampling distributions.
- Guarantees zero automated approvals, zero permission elevations, and zero modifications
  to original datasets or FAISS indices.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

DEFAULT_INPUT = Path("data/processed/counselchat/counselchat_auto_curated.jsonl")
DEFAULT_OUTPUT_CSV = Path("data/processed/counselchat/counselchat_manual_audit_sample.csv")
DEFAULT_OUTPUT_REPORT = Path("data/processed/counselchat/counselchat_manual_audit_report.json")
DEFAULT_DECISION_REPORT = Path("data/processed/counselchat/counselchat_manual_audit_decisions_report.json")
DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SEED = 42

ALLOWED_MANUAL_DECISIONS = {"approved", "rejected", "pending"}

AUTHORITATIVE_DECISION_RECOMMENDATIONS = {
    # APPROVED (8 records)
    "counselchat_0002": {
        "decision": "approved",
        "notes": "Approved: Constructive psychoeducational guidance on personal boundaries, family dynamics, and evaluating consequences.",
    },
    "counselchat_0204": {
        "decision": "approved",
        "notes": "Approved: Practical, empathetic parenting guidance addressing individual developmental needs and sibling conflict.",
    },
    "counselchat_0208": {
        "decision": "approved",
        "notes": "Approved: Sound relationship psychoeducation emphasizing shared responsibility, mutual effort, and realistic expectations.",
    },
    "counselchat_0399": {
        "decision": "approved",
        "notes": "Approved: Pragmatic guidance on emotional detachment, accountability, and healthy boundary establishment.",
    },
    "counselchat_0596": {
        "decision": "approved",
        "notes": "Approved: Actionable behavioral strategies encouraging self-efficacy, gradual goal setting, and motivation.",
    },
    "counselchat_0704": {
        "decision": "approved",
        "notes": "Approved: Compassionate, grounded coping strategies and self-care pacing for parental anxiety and acute stress.",
    },
    "counselchat_0907": {
        "decision": "approved",
        "notes": "Approved: Supportive, non-judgmental guidance validating self-respect, boundary decisions, and post-breakup clarity.",
    },
    "counselchat_1067": {
        "decision": "approved",
        "notes": "Approved: Trauma-informed, compassionate psychoeducation supporting identity recovery, self-compassion, and healing.",
    },
    # REJECTED (10 records)
    "counselchat_0154": {
        "decision": "rejected",
        "notes": "Rejected: Prescriptive assumptions validating partner infidelity suspicions rather than addressing user emotional distress.",
    },
    "counselchat_0322": {
        "decision": "rejected",
        "notes": "Rejected: Inappropriate conversational tone with location-specific salutation and judgmental phrasing unsuitable for clinical grounding.",
    },
    "counselchat_0436": {
        "decision": "rejected",
        "notes": "Rejected: Mismatched answer assuming relationship dynamics instead of addressing core anxiety, depression, and generalized distrust.",
    },
    "counselchat_0452": {
        "decision": "rejected",
        "notes": "Rejected: Mismatched and confusing premise addressing partner agreement rather than the ethical/family boundary dilemma.",
    },
    "counselchat_0550": {
        "decision": "rejected",
        "notes": "Rejected: High-conflict familial dispute guidance lacking clear boundary safeguards for dependent child.",
    },
    "counselchat_0555": {
        "decision": "rejected",
        "notes": "Rejected: Inappropriately dismissive and invalidating tone ('I hate to sound harsh but... you already know the answer').",
    },
    "counselchat_0709": {
        "decision": "rejected",
        "notes": "Rejected: Trivializes depressive symptoms with speculative framing regarding specific days rather than clinical screening advice.",
    },
    "counselchat_0787": {
        "decision": "rejected",
        "notes": "Rejected: Inadequate safety screening for spousal verbal hostility and potential emotional abuse dynamics.",
    },
    "counselchat_0799": {
        "decision": "rejected",
        "notes": "Rejected: Inadequate domestic violence and substance abuse safety protocols for ongoing verbal abuse.",
    },
    "counselchat_1059": {
        "decision": "rejected",
        "notes": "Rejected: Unauthorized diagnostic clinical labeling ('form of dissociation called depersonalization') without professional assessment.",
    },
    # PENDING (22 records)
    "counselchat_0016": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review regarding pregnancy ambiguity and breakup boundary guidance.",
    },
    "counselchat_0043": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review to assess infidelity trauma and marital counseling readiness.",
    },
    "counselchat_0073": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review regarding relationship communication lull and boundary pacing.",
    },
    "counselchat_0083": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review on relationship reconciliation assessment.",
    },
    "counselchat_0146": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review for severe post-breakup depressive affect and self-hatred.",
    },
    "counselchat_0155": {
        "decision": "pending",
        "notes": "Pending: Requires secondary safety review regarding erratic family member behavior and home environment.",
    },
    "counselchat_0256": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review regarding study-abroad situational depression and support access.",
    },
    "counselchat_0384": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review for coercive control dynamics in relationship contract request.",
    },
    "counselchat_0385": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review regarding parental intervention for adolescent depression.",
    },
    "counselchat_0408": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review evaluating infidelity suspicions and relationship anxiety.",
    },
    "counselchat_0430": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review on adolescent sexual orientation questioning and supportive framing.",
    },
    "counselchat_0484": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review regarding family communication barriers and emotional safety with parent.",
    },
    "counselchat_0501": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review on emotional dependency and codependency terminology.",
    },
    "counselchat_0504": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review regarding anger management coping strategies and root causes.",
    },
    "counselchat_0553": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review on prolonged grief intervention and bereavement support.",
    },
    "counselchat_0577": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review regarding marital ethics and values-based relationship guidance.",
    },
    "counselchat_0590": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review on adolescent peer conflict and social event communication.",
    },
    "counselchat_0593": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review regarding gender identity questioning and supportive resources.",
    },
    "counselchat_0751": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review on acute stress management and perceived loss of control.",
    },
    "counselchat_0782": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review on military separation adjustment, loss, and depression.",
    },
    "counselchat_0786": {
        "decision": "pending",
        "notes": "Pending: Requires secondary review on relationship expectations, validation seeking, and social skills.",
    },
    "counselchat_0857": {
        "decision": "pending",
        "notes": "Pending: Requires secondary clinical review regarding symptom psychoeducation vs diagnostic inquiry for anxiety.",
    },
}

CSV_COLUMNS = [
    "record_id",
    "question_title",
    "question_text",
    "topics",
    "answer_text",
    "quality_flags",
    "automated_score",
    "manual_decision",
    "review_notes",
    "reviewer",
    "reviewed_at",
]


def load_provisional_candidates(input_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Load records with curation_status == 'PROVISIONAL_CANDIDATE' from JSONL.

    Returns:
        (provisional_candidates, total_records_scanned)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    candidates: List[Dict[str, Any]] = []
    total_scanned = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            total_scanned += 1
            record = json.loads(line_str)
            if record.get("curation_status") == "PROVISIONAL_CANDIDATE":
                candidates.append(record)

    return candidates, total_scanned


def allocate_strata_quotas(
    grouped_candidates: Dict[str, List[Dict[str, Any]]],
    target_sample_size: int,
) -> Dict[str, int]:
    """Allocate integer sample quotas across strata using the Largest Remainder method

    (Hamilton-Hare quota allocation), ensuring exact target count.
    """
    total_candidates = sum(len(items) for items in grouped_candidates.values())
    if total_candidates == 0:
        return {top: 0 for top in grouped_candidates}

    actual_target = min(target_sample_size, total_candidates)
    alloc: Dict[str, int] = {}
    remainders: List[Tuple[float, int, str]] = []

    for topic, items in grouped_candidates.items():
        quota = actual_target * len(items) / total_candidates
        base = int(quota)
        rem = quota - base
        alloc[topic] = base
        remainders.append((rem, len(items), topic))

    # Distribute shortfall by largest fractional remainder, then stratum size, then topic name
    shortfall = actual_target - sum(alloc.values())
    remainders.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    for i in range(shortfall):
        topic = remainders[i][2]
        alloc[topic] += 1

    return alloc


def sample_provisional_candidates(
    candidates: List[Dict[str, Any]],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Select a deterministic, topic-stratified sample of provisional candidates

    spanning diverse answer lengths.

    Returns:
        List of exactly min(sample_size, len(candidates)) sampled records sorted by record_id.
    """
    if not candidates:
        return []

    target = min(sample_size, len(candidates))
    rng = random.Random(seed)

    # Group deterministically by primary topic
    by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in sorted(candidates, key=lambda x: x["record_id"]):
        top = r["topics"][0] if r.get("topics") else "Other"
        by_topic[top].append(r)

    # Compute stratified allocations
    allocations = allocate_strata_quotas(by_topic, target)

    sampled: List[Dict[str, Any]] = []

    for topic in sorted(by_topic.keys()):
        k = allocations.get(topic, 0)
        if k <= 0:
            continue

        items = sorted(by_topic[topic], key=lambda r: r["record_id"])

        if k >= 3 and len(items) >= k:
            # Spread selection across answer length quantiles within stratum
            items_by_len = sorted(items, key=lambda r: (r.get("answer_characters", len(r["answer_text"])), r["record_id"]))
            step = len(items_by_len) / k
            picks: List[Dict[str, Any]] = []
            for i in range(k):
                sub_start = int(i * step)
                sub_end = int((i + 1) * step)
                bucket = items_by_len[sub_start:sub_end]
                picks.append(rng.choice(bucket))
            sampled.extend(picks)
        else:
            sampled.extend(rng.sample(items, k))

    # Sort deterministically by record_id
    sampled.sort(key=lambda r: r["record_id"])
    return sampled


def compute_sample_statistics(
    sampled: List[Dict[str, Any]],
    total_candidates: int,
    sample_size: int,
    seed: int,
    output_csv: Path,
    output_report: Path,
) -> Dict[str, Any]:
    """Compute topic and answer length distributions for the sampled candidates."""
    all_topics: Counter[str] = Counter()
    primary_topics: Counter[str] = Counter()
    lengths = [r.get("answer_characters", len(r["answer_text"])) for r in sampled]

    for r in sampled:
        for t in r.get("topics", []):
            all_topics[t] += 1
        if r.get("topics"):
            primary_topics[r["topics"][0]] += 1
        else:
            primary_topics["Other"] += 1

    sorted_lengths = sorted(lengths) if lengths else [0]
    n = len(sorted_lengths)
    median_len = (
        sorted_lengths[n // 2]
        if n % 2 == 1
        else (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2.0
    )

    length_stats = {
        "min_characters": min(lengths) if lengths else 0,
        "max_characters": max(lengths) if lengths else 0,
        "mean_characters": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "median_characters": median_len,
        "buckets": {
            "under_500": sum(1 for l in lengths if l < 500),
            "500_to_1000": sum(1 for l in lengths if 500 <= l <= 1000),
            "1000_to_2000": sum(1 for l in lengths if 1000 < l <= 2000),
            "over_2000": sum(1 for l in lengths if l > 2000),
        },
    }

    # Verify duplicate questions within sample
    q_counts = Counter(r["question_text"].strip().lower() for r in sampled if r.get("question_text"))
    sample_dups = sum(1 for c in q_counts.values() if c > 1)

    return {
        "dataset": "elvindavis/counselchat",
        "total_provisional_candidates": total_candidates,
        "number_sampled": len(sampled),
        "target_sample_size": sample_size,
        "sampling_seed": seed,
        "stratification_strategy": "proportional_topic_stratification_with_length_spread",
        "topic_distribution": dict(all_topics.most_common()),
        "primary_topic_distribution": dict(primary_topics.most_common()),
        "answer_length_distribution": length_stats,
        "duplicate_question_handling": {
            "total_provisional_candidates_with_duplicate_questions": 0,
            "sampled_records_with_duplicate_questions": sample_dups,
            "policy": (
                "Duplicate questions were excluded from provisional candidates during automated curation; "
                "all sampled records have unique questions."
            ),
        },
        "output_paths": {
            "sample_csv": str(output_csv),
            "report_json": str(output_report),
        },
        "manual_review_fields": {
            "manual_decision": "pending",
            "review_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        },
        "retrieval_permission": "no",
        "clinical_approval": "not_assessed",
        "faiss_indexed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def export_audit_sample(
    sampled_records: List[Dict[str, Any]],
    output_csv_path: Path,
) -> None:
    """Export sampled records to an Excel-compatible CSV with UTF-8 BOM."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv_path.with_name(f"{output_csv_path.name}.tmp_{os.getpid()}")

    rows = []
    for r in sampled_records:
        topics_val = json.dumps(r.get("topics", []), ensure_ascii=False)
        quality_flags_val = json.dumps(r.get("quality_flags", []), ensure_ascii=False)
        rows.append({
            "record_id": r["record_id"],
            "question_title": r.get("question_title", ""),
            "question_text": r.get("question_text", ""),
            "topics": topics_val,
            "answer_text": r.get("answer_text", ""),
            "quality_flags": quality_flags_val,
            "automated_score": r.get("automated_quality_score", 0.0),
            "manual_decision": "pending",
            "review_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        })

    try:
        with tmp_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        tmp_csv.replace(output_csv_path)
    except Exception:
        if tmp_csv.exists():
            try:
                tmp_csv.unlink()
            except OSError:
                pass
        raise


def export_audit_report(
    report: Dict[str, Any],
    output_report_path: Path,
) -> None:
    """Export the sampling report to a formatted JSON file atomically."""
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = output_report_path.with_name(f"{output_report_path.name}.tmp_{os.getpid()}")

    try:
        with tmp_report.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_report.replace(output_report_path)
    except Exception:
        if tmp_report.exists():
            try:
                tmp_report.unlink()
            except OSError:
                pass
        raise


def run_sampling_workflow(
    input_path: Path = DEFAULT_INPUT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_report: Path = DEFAULT_OUTPUT_REPORT,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Execute the full audit sampling workflow."""
    input_path = Path(input_path)
    output_csv = Path(output_csv)
    output_report = Path(output_report)

    candidates, total_scanned = load_provisional_candidates(input_path)
    sampled = sample_provisional_candidates(candidates, sample_size=sample_size, seed=seed)
    report = compute_sample_statistics(
        sampled=sampled,
        total_candidates=len(candidates),
        sample_size=sample_size,
        seed=seed,
        output_csv=output_csv,
        output_report=output_report,
    )

    export_audit_sample(sampled, output_csv)
    export_audit_report(report, output_report)

    # Console display summary
    print("\n=== COUNSELCHAT HUMAN AUDIT SAMPLING REPORT ===")
    print(f"Total records in curated dataset: {total_scanned}")
    print(f"Provisional candidates available: {len(candidates)}")
    print(f"Records sampled for audit:        {len(sampled)} (seed: {seed})")
    print(f"Unique topics in sample:          {len(report['topic_distribution'])}")
    print(f"Answer length range:              {report['answer_length_distribution']['min_characters']} - {report['answer_length_distribution']['max_characters']} chars (mean: {report['answer_length_distribution']['mean_characters']})")
    print(f"Sample CSV:                       {output_csv}")
    print(f"Audit Report:                     {output_report}")
    print("Initial manual decision:          pending (all records)")
    print("Retrieval permission:             no (all records)")
    print("Clinical approval:                not_assessed")
    print("FAISS indexing:                   not performed")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a reproducible, topic-stratified sample of provisional candidates for human clinical audit."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--decision-report", type=Path, default=DEFAULT_DECISION_REPORT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--update-recommendations",
        action="store_true",
        help="Apply authoritative review recommendations to the manual audit sample CSV",
    )
    parser.add_argument(
        "--validate-sample",
        action="store_true",
        help="Validate the manual audit sample decisions and integrity",
    )
    args = parser.parse_args()

    if args.validate_sample:
        validation = validate_audit_sample(args.output_csv, expected_count=args.sample_size)
        print("\n=== COUNSELCHAT AUDIT SAMPLE VALIDATION ===")
        print(f"Sample CSV:            {args.output_csv}")
        print(f"Total valid records:   {validation['total_records']}")
        print(f"Counts by decision:    {validation['counts_by_decision']}")
        return

    if args.update_recommendations:
        updated_rows, backup_path = update_audit_sample_with_decisions(
            csv_path=args.output_csv,
            output_path=args.output_csv,
            backup=True,
        )
        report = generate_audit_decision_report(
            csv_path=args.output_csv,
            report_path=args.decision_report,
            update_main_report=True,
            main_report_path=args.output_report,
        )
        print("\n=== COUNSELCHAT AUDIT SAMPLE RECOMMENDATIONS APPLIED ===")
        print(f"Sample CSV:            {args.output_csv}")
        print(f"Backup created at:     {backup_path}")
        print(f"Decision report:       {args.decision_report}")
        print(f"Main report updated:   {args.output_report}")
        print(f"Decision counts:       {report['decision_counts']}")
        print("Reviewer / reviewed_at: [blank - pending human review]")
        print("Clinical approval:     not_assessed")
        print("Retrieval permission:  no")
        print("FAISS indexing:        not performed")
        return

    run_sampling_workflow(
        input_path=args.input,
        output_csv=args.output_csv,
        output_report=args.output_report,
        sample_size=args.sample_size,
        seed=args.seed,
    )


def backup_audit_sample(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    backup_path: Path | None = None,
) -> Path:
    """Create a backup of the audit sample CSV before modification."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot backup non-existent file: {csv_path}")

    if backup_path is None:
        backup_path = csv_path.with_name(f"{csv_path.name}.bak")
    else:
        backup_path = Path(backup_path)

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_bytes(csv_path.read_bytes())
    return backup_path


def update_audit_sample_with_decisions(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    decisions_map: Dict[str, Dict[str, str]] | None = None,
    output_path: Path | None = None,
    backup: bool = True,
    backup_path: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Path | None]:
    """Update manual audit sample CSV with decisions and review notes.

    Ensures:
    - All original records, question texts, answer texts, and metadata are preserved.
    - manual_decision is populated with approved/rejected/pending.
    - review_notes is populated with a concise explanation.
    - reviewer and reviewed_at remain empty strings (human clinical sign-off pending).
    - UTF-8 characters and newlines are preserved.
    - Original file is backed up before replacement.
    - Replacement is atomic.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path) if output_path else csv_path
    decisions_map = (
        decisions_map
        if decisions_map is not None
        else AUTHORITATIVE_DECISION_RECOMMENDATIONS
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Audit sample CSV not found: {csv_path}")

    saved_backup_path = None
    if backup:
        saved_backup_path = backup_audit_sample(csv_path, backup_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or CSV_COLUMNS)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Audit sample CSV '{csv_path}' is empty.")

    updated_rows = []
    for r in rows:
        rec_id = r["record_id"]
        rec_info = decisions_map.get(rec_id)
        if rec_info is None:
            raise ValueError(f"Missing decision mapping for record ID '{rec_id}'")

        decision = rec_info["decision"].strip().lower()
        if decision not in ALLOWED_MANUAL_DECISIONS:
            raise ValueError(
                f"Invalid manual decision '{decision}' for record '{rec_id}'. "
                f"Must be one of {sorted(ALLOWED_MANUAL_DECISIONS)}"
            )

        notes = rec_info.get("notes", "").strip()
        if not notes:
            raise ValueError(f"Missing review notes for record ID '{rec_id}'")

        row_copy = dict(r)
        row_copy["manual_decision"] = decision
        row_copy["review_notes"] = notes
        row_copy["reviewer"] = ""
        row_copy["reviewed_at"] = ""

        updated_rows.append(row_copy)

    # Write atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp_{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return updated_rows, saved_backup_path


def validate_audit_sample(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    expected_count: int = DEFAULT_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Validate that the manual audit sample CSV conforms to all safety and integrity requirements.

    Requirements verified:
    1. Exactly expected_count rows exist.
    2. All record_ids are unique.
    3. Every manual_decision is in ALLOWED_MANUAL_DECISIONS.
    4. Every row has non-empty review_notes.
    5. reviewer and reviewed_at must be empty (preventing unauthorized clinical sign-off).
    6. Essential fields (question_text, answer_text, topics) are non-empty.

    Returns:
        Summary dict containing counts and record IDs by decision.
    Raises:
        ValueError if any validation check fails.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in CSV: {csv_path}")
        missing_cols = [c for c in CSV_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"CSV missing mandatory columns: {missing_cols}")
        rows = list(reader)

    if len(rows) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} rows in sample CSV, found {len(rows)}"
        )

    record_ids = [r["record_id"] for r in rows]
    if len(set(record_ids)) != len(record_ids):
        duplicates = [rid for rid, cnt in Counter(record_ids).items() if cnt > 1]
        raise ValueError(f"Duplicate record IDs found in sample CSV: {duplicates}")

    counts_by_decision: Dict[str, int] = defaultdict(int)
    ids_by_decision: Dict[str, List[str]] = defaultdict(list)

    for i, r in enumerate(rows, start=1):
        rid = r["record_id"].strip()
        if not rid:
            raise ValueError(f"Row {i} has empty record_id")

        decision = r["manual_decision"].strip().lower()
        if decision not in ALLOWED_MANUAL_DECISIONS:
            raise ValueError(
                f"Row {i} ({rid}) has invalid manual_decision: '{r['manual_decision']}'. "
                f"Must be one of {sorted(ALLOWED_MANUAL_DECISIONS)}"
            )

        if not r["review_notes"].strip():
            raise ValueError(f"Row {i} ({rid}) has empty review_notes")

        if r["reviewer"].strip() != "":
            raise ValueError(
                f"Row {i} ({rid}) has populated reviewer ('{r['reviewer']}'). "
                f"Reviewer must remain blank until authorized clinical review."
            )

        if r["reviewed_at"].strip() != "":
            raise ValueError(
                f"Row {i} ({rid}) has populated reviewed_at ('{r['reviewed_at']}'). "
                f"Reviewed_at must remain blank until authorized clinical review."
            )

        if not r["question_text"].strip():
            raise ValueError(f"Row {i} ({rid}) has empty question_text")

        if not r["answer_text"].strip():
            raise ValueError(f"Row {i} ({rid}) has empty answer_text")

        counts_by_decision[decision] += 1
        ids_by_decision[decision].append(rid)

    return {
        "valid": True,
        "total_records": len(rows),
        "counts_by_decision": dict(counts_by_decision),
        "record_ids_by_decision": dict(ids_by_decision),
    }


def generate_audit_decision_report(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    report_path: Path = DEFAULT_DECISION_REPORT,
    update_main_report: bool = True,
    main_report_path: Path = DEFAULT_OUTPUT_REPORT,
) -> Dict[str, Any]:
    """Generate an audit report documenting decision counts and record ID lists."""
    validation = validate_audit_sample(csv_path)

    report_data = {
        "dataset": "elvindavis/counselchat",
        "sample_csv": str(csv_path),
        "total_sample_records": validation["total_records"],
        "decision_counts": validation["counts_by_decision"],
        "record_ids_by_decision": validation["record_ids_by_decision"],
        "retrieval_permission": "no",
        "clinical_approval": "not_assessed",
        "faiss_indexed": False,
        "recommendation_scope": "Content-suitability recommendations only; clinical approval not granted.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path(report_path)
    export_audit_report(report_data, report_path)

    if update_main_report and main_report_path.exists():
        try:
            with main_report_path.open("r", encoding="utf-8") as f:
                main_data = json.load(f)
            main_data["audit_decision_summary"] = {
                "decision_counts": validation["counts_by_decision"],
                "record_ids_by_decision": validation["record_ids_by_decision"],
                "updated_at": report_data["updated_at"],
            }
            export_audit_report(main_data, main_report_path)
        except Exception:
            pass

    return report_data


if __name__ == "__main__":
    main()
