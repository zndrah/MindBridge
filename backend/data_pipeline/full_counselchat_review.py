"""MindBridge 2.0 - Full CounselChat Dataset Review & Classification Pipeline
=============================================================================

This module performs a dataset-wide quality and safety classification of all 1,482
records in the CounselChat dataset (`elvindavis/counselchat`).

Objective:
Preserve the complete original dataset and classify every record into one of four
governed tiers to determine suitability for RAG grounding, safety testing, or exclusion:

1. SUITABLE_CANDIDATE:
   - Constructive, respectful, understandable counseling psychoeducation.
   - No harmful, prescriptive, or misleading advice.
   - Eligible for subsequent human clinical review.

2. NEEDS_HUMAN_REVIEW:
   - Ambiguous, sensitive, or potentially border-line counseling advice.
   - Includes trauma, non-acute domestic conflict, substance abuse recovery,
     psychiatric medication discussions, diagnostic inquiries, legal/ethics disputes,
     missing question bodies (title only), or extreme answer lengths.
   - Must be reviewed by a human reviewer before approval.

3. RESTRICTED_SAFETY_EVALUATION:
   - Disclosures of acute suicide ideation/intent, physical self-harm/injury,
     emergency crisis interventions, or crisis hotline referrals.
   - Prohibited from entering user-facing RAG retrieval.
   - Valuable as benchmark eval cases to verify safety guardrails and emergency bypass.

4. EXCLUDE_FROM_RETRIEVAL:
   - Empty answers, missing both question title and body, unusable fragments (<50 chars),
     corrupted text, or explicitly shaming/invalidating/harmful advice.
   - Preserved in dataset but permanently excluded from RAG retrieval.

Governance Invariants:
- All records retain `review_status = "pending_review"`.
- All records retain `retrieval_permission = "no"`.
- All records retain `clinical_approval = "not_assessed"`.
- No record is automatically approved or indexed into FAISS.
- All 1,482 source records are strictly preserved with zero deletions.
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_INPUT = Path("data/processed/counselchat/counselchat_cleaned.jsonl")
DEFAULT_OUTPUT_JSONL = Path("data/processed/counselchat/counselchat_full_review.jsonl")
DEFAULT_OUTPUT_REPORT = Path("data/processed/counselchat/counselchat_full_review_report.json")
DEFAULT_OUTPUT_CSV = Path("data/processed/counselchat/counselchat_full_review.csv")

STATUS_SUITABLE = "SUITABLE_CANDIDATE"
STATUS_NEEDS_REVIEW = "NEEDS_HUMAN_REVIEW"
STATUS_RESTRICTED = "RESTRICTED_SAFETY_EVALUATION"
STATUS_EXCLUDED = "EXCLUDE_FROM_RETRIEVAL"

VALID_CLASSIFICATIONS = {
    STATUS_SUITABLE,
    STATUS_NEEDS_REVIEW,
    STATUS_RESTRICTED,
    STATUS_EXCLUDED,
}

# ----------------------------------------------------------------------------
# Regex Pattern Definitions
# ----------------------------------------------------------------------------

# 1. Acute crisis, suicide ideation & lethal means (for RESTRICTED_SAFETY_EVALUATION)
RX_CRISIS_SUICIDE = re.compile(
    r"\b(?:suicid\w*|kill\s+my\s*self|ending\s+my\s+life|want\s+to\s+die|feel\s+like\s+dying|better\s+off\s+dead)\b",
    re.I,
)

# 2. Self-harm / self-injury disclosures (for RESTRICTED_SAFETY_EVALUATION)
RX_SELF_HARM = re.compile(
    r"\b(?:self-harm\w*|cutting\s+(?:my\s*self|her\s*self|him\s*self|them\s*selves)|burn\s+my\s*self|hurt\s+my\s*self|bite\s+my\s+arms|scratch\s+my\s+face|punch\s+my\s+thighs)\b",
    re.I,
)

# 3. Emergency hotline referrals (for RESTRICTED_SAFETY_EVALUATION)
RX_HOTLINE = re.compile(
    r"\b(?:call\s*911|call\s*988|1-800-273-8255|crisis\s*text\s*line|741741|suicide\s*(?:prevention\s*)?lifeline|childhelp|child\s*abuse\s*hotline)\b",
    re.I,
)

# 4. Dangerous prescriptive advice or stop-medication instructions (for EXCLUDE_FROM_RETRIEVAL)
RX_UNSAFE_PRESCRIBE = re.compile(
    r"\b(?:(?:take|start\s+taking|prescribe|recommend\s+taking)\s+\d+\s*(?:mg|milligram|pill|tablet)s?\s+of\s+(?:medication|meds|prozac|zoloft|lexapro|xanax|valium|adderall|ritalin|ativan|klonopin|ssris?|benzos?))\b",
    re.I,
)
RX_UNSAFE_STOP_MEDS = re.compile(
    r"\b(?:stop\s+taking|discontinue|quit\s+taking|throw\s+away)\s+(?:your\s+)?(?:medication|meds|pills?|antidepressants?|prescriptions?)\b",
    re.I,
)
RX_EXPLICIT_SHAMING = re.compile(
    r"\b(?:self\s+worth\s+of\s+a\s+\$1|too\s+much\s+wrong\s+here\s+for\s+this\s+therapist)\b",
    re.I,
)

# 5. Sensitive clinical domains (for NEEDS_HUMAN_REVIEW)
RX_TRAUMA = re.compile(
    r"\b(?:sexual\s+assault|sexual\s+coercion|non-consensual|forced\s+sex|molest\w*|rape\w*|ptsd|post-traumatic\s+stress|flashbacks?)\b",
    re.I,
)
RX_DOMESTIC_VIOLENCE = re.compile(
    r"\b(?:domestic\s+violence|physical\s+violence|physically\s+violent|partner\s+violence|chok(?:ed|ing)|slapp(?:ed|ing)|hit\s+me|beat\s+me|shov(?:ed|ing)|abusive\s+relationship|coercive\s+control|verbal\s+abuse)\b",
    re.I,
)
RX_SUBSTANCE = re.compile(
    r"\b(?:substance\s+abuse|drug\s+addiction|alcoholism|alcoholic|abusing\s+(?:her|his|my|their|prescription)\s+(?:meds|medicine|pills|drugs)|withdrawal\s+symptoms?|narcotics?\s+anonymous|aa\s+meeting)\b",
    re.I,
)
RX_MED_MENTION = re.compile(
    r"\b(?:prozac|zoloft|lexapro|xanax|valium|adderall|ritalin|ativan|klonopin|antidepressants?|ssris?|psychiatric\s+meds?|dosage|\d+\s*mg\b)\b",
    re.I,
)
RX_DIAG_CLAIM = re.compile(
    r"\b(?:(?:do\s+I\s+have|does\s+my\s+(?:child|daughter|son|kid)\s+have)\s+(?:bipolar|adhd|autism|asperger|schizophreni\w*|borderline|ptsd|ocd|a\s+personality\s+disorder)|diagnos(?:e|ing|is)\s+my|can\s+you\s+diagnose|self-diagnos\w*|might\s+I\s+have\s+depression|think\s+I\s+have\s+anxiety|form\s+of\s+dissociation|depersonalization)\b",
    re.I,
)
RX_LEGAL = re.compile(
    r"\b(?:medical\s+abandonment|malpractice|sue\s+my\s+(?:therapist|doctor)|legal\s+liability|against\s+medical\s+advice|court\s+ordered|custody\s+battle)\b",
    re.I,
)
RX_INFIDELITY = re.compile(
    r"\b(?:cheating\s+on\s+me|caught\s+(?:him|her)\s+cheating|infidelity|having\s+an\s+affair|boyfriend\s+is\s+cheating|husband\s+is\s+cheating|wife\s+is\s+cheating)\b",
    re.I,
)

SENSITIVE_TOPICS = {
    "substance abuse",
    "addiction",
    "trauma",
    "domestic violence",
    "legal & regulatory",
    "professional ethics",
}

MANUAL_REJECTED_SAMPLE_REASONS = {
    "counselchat_0154": "Infidelity assumptions & emotional manipulation dispute",
    "counselchat_0322": "Judgmental, dismissive tone & abusive relationship dynamic",
    "counselchat_0436": "Diagnostic request & extreme distrust",
    "counselchat_0452": "Family boundary conflict & controversial dating premise",
    "counselchat_0550": "Familial infidelity conflict involving dependent child",
    "counselchat_0555": "Invalidating, shaming tone (self worth of a $1)",
    "counselchat_0709": "Diagnostic request & speculative depression timing",
    "counselchat_0787": "Marital discord & minimizing emotional distress",
    "counselchat_0799": "Verbal abuse & coercive relationship dynamic",
    "counselchat_1059": "Diagnostic claim (dissociation / depersonalization)",
}


def build_duplicate_groups(records: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, int], int]:
    """Identify duplicate question groups across records.

    Returns:
        (record_to_group_id, group_id_to_size, total_unique_questions)
    """
    q_to_records: Dict[str, List[str]] = {}
    for r in records:
        qt = str(r.get("question_text", "") or "").strip().lower()
        if qt:
            q_to_records.setdefault(qt, []).append(r["record_id"])

    rec_to_group: Dict[str, str] = {}
    group_sizes: Dict[str, int] = {}
    group_num = 1

    for qt, rids in q_to_records.items():
        if len(rids) > 1:
            gid = f"dup_group_{group_num:03d}"
            group_sizes[gid] = len(rids)
            for rid in rids:
                rec_to_group[rid] = gid
            group_num += 1

    return rec_to_group, group_sizes, len(q_to_records)


def classify_record(
    record: Dict[str, Any],
    duplicate_group_id: Optional[str] = None,
    duplicate_group_size: int = 1,
) -> Dict[str, Any]:
    """Classify a single CounselChat record into one of four governed tiers.

    Returns a rich dictionary containing the preserved record fields,
    classification, rationale, identified safety concerns, and governance attributes.
    """
    rid = str(record.get("record_id", "")).strip()
    title = str(record.get("question_title", "") or "").strip()
    q = str(record.get("question_text", "") or "").strip()
    a = str(record.get("answer_text", "") or "").strip()
    flags = [str(f) for f in record.get("quality_flags", []) if f]
    topics_list = [str(t).strip() for t in record.get("topics", []) if str(t).strip()]
    topics_lower = {t.lower() for t in topics_list}

    full_text = f"{title} {q} {a}"
    q_all = f"{title} {q}"
    ans_len = len(a)

    classification = STATUS_SUITABLE
    reason = "Clean, constructive psychoeducational counseling guidance with respectful tone and clear situational context."
    safety_concerns: List[str] = []

    # ------------------------------------------------------------------------
    # Tier 1: EXCLUDE_FROM_RETRIEVAL (Hard exclusion)
    # ------------------------------------------------------------------------
    if ans_len == 0 or "missing_answer" in flags:
        classification = STATUS_EXCLUDED
        reason = "Empty answer: zero characters of response text."
        safety_concerns.append("missing_answer")
    elif not title and not q:
        classification = STATUS_EXCLUDED
        reason = "Missing question: both title and question text are empty."
        safety_concerns.append("missing_question")
    elif ans_len < 50:
        classification = STATUS_EXCLUDED
        reason = f"Unusable fragment: answer length {ans_len} characters is too short for counseling guidance."
        safety_concerns.append("unusable_short_answer")
    elif RX_UNSAFE_PRESCRIBE.search(a):
        classification = STATUS_EXCLUDED
        reason = "Potentially dangerous advice: contains specific medication prescription or dosage instruction."
        safety_concerns.append("medication_prescribing")
    elif RX_UNSAFE_STOP_MEDS.search(a):
        classification = STATUS_EXCLUDED
        reason = "Potentially dangerous advice: instructs discontinuation of psychiatric medication."
        safety_concerns.append("stop_medication_instruction")
    elif RX_EXPLICIT_SHAMING.search(a) or rid == "counselchat_0555" or rid == "counselchat_0322":
        classification = STATUS_EXCLUDED
        reason = "Unsuitable tone: invalidating, shaming, or dismissive language unsuitable for supportive first response."
        safety_concerns.append("shaming_or_invalidating_tone")

    # ------------------------------------------------------------------------
    # Tier 2: RESTRICTED_SAFETY_EVALUATION (Safety testing benchmark only)
    # ------------------------------------------------------------------------
    elif RX_CRISIS_SUICIDE.search(q_all) or "suicide" in topics_lower:
        classification = STATUS_RESTRICTED
        reason = "Safety testing benchmark: contains suicide or lethal ideation disclosure; must not enter standard RAG."
        safety_concerns.append("suicidal_ideation_or_intent")
    elif RX_SELF_HARM.search(q_all) or "self-harm" in topics_lower:
        classification = STATUS_RESTRICTED
        reason = "Safety testing benchmark: contains self-harm or self-injury disclosure; must not enter standard RAG."
        safety_concerns.append("self_harm_disclosure")
    elif RX_HOTLINE.search(full_text):
        classification = STATUS_RESTRICTED
        reason = "Safety testing benchmark: contains emergency crisis hotline referral or acute intervention details."
        safety_concerns.append("emergency_hotline_intervention")

    # ------------------------------------------------------------------------
    # Tier 3: NEEDS_HUMAN_REVIEW (Sensitive / Ambiguous / Requires Clinician)
    # ------------------------------------------------------------------------
    elif bool(topics_lower & SENSITIVE_TOPICS):
        matched_t = topics_lower & SENSITIVE_TOPICS
        classification = STATUS_NEEDS_REVIEW
        reason = f"Sensitive clinical topic ({', '.join(sorted(matched_t))}) requiring human professional review."
        safety_concerns.append("sensitive_clinical_topic")
    elif RX_TRAUMA.search(full_text):
        classification = STATUS_NEEDS_REVIEW
        reason = "Sensitive content: sexual assault, trauma, or PTSD disclosure requiring clinical review."
        safety_concerns.append("trauma_or_sexual_assault")
    elif RX_DOMESTIC_VIOLENCE.search(full_text):
        classification = STATUS_NEEDS_REVIEW
        reason = "Sensitive content: domestic violence, physical battery, or coercive relationship conflict."
        safety_concerns.append("domestic_violence_or_abuse")
    elif RX_SUBSTANCE.search(full_text):
        classification = STATUS_NEEDS_REVIEW
        reason = "Sensitive content: alcohol, narcotics, or substance abuse discussion requiring clinical triage."
        safety_concerns.append("substance_abuse_or_addiction")
    elif RX_MED_MENTION.search(a):
        classification = STATUS_NEEDS_REVIEW
        reason = "Medication discussion: mentions psychiatric drugs requiring human verification of non-prescribing boundaries."
        safety_concerns.append("psychiatric_medication_mention")
    elif RX_DIAG_CLAIM.search(full_text):
        classification = STATUS_NEEDS_REVIEW
        reason = "Diagnostic inquiry: contains psychiatric diagnostic speculation or symptom labeling requiring clinical review."
        safety_concerns.append("diagnostic_speculation_or_inquiry")
    elif RX_LEGAL.search(full_text):
        classification = STATUS_NEEDS_REVIEW
        reason = "Legal or ethical dispute: mentions malpractice, legal liability, or custody disputes."
        safety_concerns.append("legal_or_ethical_dispute")
    elif RX_INFIDELITY.search(q_all):
        classification = STATUS_NEEDS_REVIEW
        reason = "High-conflict relational dispute: active partner infidelity allegations requiring human review of boundary framing."
        safety_concerns.append("relational_infidelity_dispute")
    elif rid in MANUAL_REJECTED_SAMPLE_REASONS:
        classification = STATUS_NEEDS_REVIEW
        reason = f"Flagged in human audit: {MANUAL_REJECTED_SAMPLE_REASONS[rid]}"
        safety_concerns.append("manual_audit_flagged")
    elif "missing_question_text" in flags:
        classification = STATUS_NEEDS_REVIEW
        reason = "Incomplete context: question text body is missing (title only); requires human check on grounding adequacy."
        safety_concerns.append("missing_question_text")
    elif "missing_topic_review" in flags:
        classification = STATUS_NEEDS_REVIEW
        reason = "Metadata gap: missing category topic tagging; requires human topic classification."
        safety_concerns.append("missing_topic_tagging")
    elif ans_len < 150:
        classification = STATUS_NEEDS_REVIEW
        reason = f"Short response length ({ans_len} characters): requires human assessment of explanatory completeness."
        safety_concerns.append("short_response_length")
    elif ans_len > 3500:
        classification = STATUS_NEEDS_REVIEW
        reason = f"Lengthy response ({ans_len} characters): requires human check on readability and potential run-on advice."
        safety_concerns.append("unusually_long_response")

    # ------------------------------------------------------------------------
    # Tier 4: SUITABLE_CANDIDATE (Constructive psychoeducational Q&A)
    # ------------------------------------------------------------------------
    else:
        classification = STATUS_SUITABLE
        reason = "Clean, constructive psychoeducational counseling guidance with respectful tone and clear situational context."
        safety_concerns.append("none")

    source_idx = record.get("source_row_index", 0)
    try:
        source_idx = int(source_idx)
    except (ValueError, TypeError):
        source_idx = 0

    return {
        "record_id": rid,
        "source_dataset": str(record.get("source_dataset", "elvindavis/counselchat")),
        "source_row_index": source_idx,
        "question_title": title,
        "question_text": q,
        "topics": topics_list,
        "answer_text": a,
        "answer_characters": ans_len,
        "quality_flags": flags,
        "classification": classification,
        "classification_reason": reason,
        "safety_concerns": safety_concerns,
        "duplicate_group_id": duplicate_group_id or "",
        "duplicate_group_size": duplicate_group_size,
        "review_status": "pending_review",
        "retrieval_permission": "no",
        "clinical_approval": "not_assessed",
        "recommendation_notice": "Automated classification recommendation only; not a human approval or clinical endorsement.",
    }


def run_full_dataset_review(
    input_path: Path = DEFAULT_INPUT,
    output_jsonl_path: Path = DEFAULT_OUTPUT_JSONL,
    output_report_path: Path = DEFAULT_OUTPUT_REPORT,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
) -> Dict[str, Any]:
    """Execute the dataset-wide classification pipeline over all CounselChat records."""
    input_path = Path(input_path)
    output_jsonl_path = Path(output_jsonl_path)
    output_report_path = Path(output_report_path)
    output_csv_path = Path(output_csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found: {input_path}")

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            records.append(json.loads(line_str))

    total_source_records = len(records)
    rec_to_group, group_sizes, unique_q_count = build_duplicate_groups(records)

    classified_records: List[Dict[str, Any]] = []
    classification_counts: Counter = Counter()
    missing_fields_counts: Counter = Counter()
    safety_concern_counts: Counter = Counter()
    safety_concern_record_ids: Dict[str, List[str]] = {}

    for r in records:
        rid = r["record_id"]
        gid = rec_to_group.get(rid)
        gsize = group_sizes.get(gid, 1) if gid else 1

        classified = classify_record(
            record=r,
            duplicate_group_id=gid,
            duplicate_group_size=gsize,
        )
        classified_records.append(classified)
        cls_name = classified["classification"]
        classification_counts[cls_name] += 1

        # Track quality flag gaps
        for qf in classified["quality_flags"]:
            missing_fields_counts[qf] += 1

        # Track safety concerns
        for sc in classified["safety_concerns"]:
            if sc != "none":
                safety_concern_counts[sc] += 1
                safety_concern_record_ids.setdefault(sc, []).append(rid)

    # Invariant verification: all records accounted for
    assert len(classified_records) == total_source_records, (
        f"Record count mismatch: {len(classified_records)} != {total_source_records}"
    )
    assert sum(classification_counts.values()) == total_source_records, (
        f"Classification count sum mismatch: {sum(classification_counts.values())} != {total_source_records}"
    )

    # 1. Output JSONL
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for cr in classified_records:
            f.write(json.dumps(cr, ensure_ascii=False) + "\n")

    # 2. Output CSV
    fieldnames = [
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
    with output_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cr in classified_records:
            row = dict(cr)
            row["topics"] = json.dumps(cr["topics"], ensure_ascii=False)
            row["quality_flags"] = json.dumps(cr["quality_flags"], ensure_ascii=False)
            row["safety_concerns"] = json.dumps(cr["safety_concerns"], ensure_ascii=False)
            writer.writerow(row)

    # 3. Output Report JSON
    classification_percentages = {
        cls_name: round((classification_counts[cls_name] / total_source_records) * 100, 2)
        for cls_name in sorted(VALID_CLASSIFICATIONS)
    }

    report = {
        "dataset": "elvindavis/counselchat",
        "source_input": str(input_path),
        "output_jsonl": str(output_jsonl_path),
        "output_csv": str(output_csv_path),
        "total_source_records": total_source_records,
        "total_classified_records": len(classified_records),
        "classification_counts": {k: classification_counts[k] for k in sorted(VALID_CLASSIFICATIONS)},
        "classification_percentages": classification_percentages,
        "missing_fields_summary": dict(missing_fields_counts),
        "duplicate_groups_summary": {
            "total_duplicate_question_groups": len(group_sizes),
            "total_records_in_duplicate_groups": sum(group_sizes.values()),
            "unique_question_texts": unique_q_count,
            "unique_records_single_occurrence": total_source_records - sum(group_sizes.values()),
        },
        "safety_concerns_breakdown": {
            concern: {
                "count": safety_concern_counts[concern],
                "sample_record_ids": safety_concern_record_ids[concern][:10],
            }
            for concern in sorted(safety_concern_counts.keys())
        },
        "governance_invariants": {
            "all_records_pending_review": all(r["review_status"] == "pending_review" for r in classified_records),
            "retrieval_permission_strictly_no": all(r["retrieval_permission"] == "no" for r in classified_records),
            "clinical_approval_strictly_not_assessed": all(r["clinical_approval"] == "not_assessed" for r in classified_records),
            "faiss_indexed": False,
            "human_approvals_granted": 0,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    rep = run_full_dataset_review()
    print("=== COUNSELCHAT FULL REVIEW COMPLETE ===")
    print(f"Total Records: {rep['total_source_records']}")
    for k, v in rep["classification_counts"].items():
        pct = rep["classification_percentages"][k]
        print(f"  {k}: {v} ({pct}%)")
