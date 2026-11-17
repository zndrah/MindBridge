"""MindBridge 2.0 - Automated CounselChat Curation Pipeline
========================================================

Deterministic automated screening pipeline that screens CounselChat records and
classifies them into three categories:

1. PROVISIONAL_CANDIDATE: Clean, complete Q&A pairs with reasonable answer lengths,
   valid topics, no data-quality flags, and no sensitive/unsafe content.
2. NEEDS_REVIEW: Incomplete records, duplicate questions, short or unusually long answers,
   or content with sensitive medication mentions or crisis advice requiring clinical triage.
3. EXCLUDED: Empty answers, unusable short fragments, corrupted text, or clearly
   inappropriate/unsafe advice (prescribing, stop meds, self-harm encouragement, violence).

Safety & Engineering Invariants:
- All records initially receive retrieval_permission="no".
- All records receive clinical_approval="not_assessed".
- No record is automatically marked as clinically approved or indexed into FAISS.
- Crisis handling is independent of CounselChat; crisis-related guidance is flagged for review.
- Source records and metadata are preserved without deletions or alterations.
- Deterministic rule-based execution with zero external API or model dependencies.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

DEFAULT_INPUT = Path("data/processed/counselchat/counselchat_cleaned.jsonl")
DEFAULT_OUTPUT = Path("data/processed/counselchat/counselchat_auto_curated.jsonl")
DEFAULT_REPORT = Path("data/processed/counselchat/counselchat_curation_report.json")

STATUS_PROVISIONAL = "PROVISIONAL_CANDIDATE"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"
STATUS_EXCLUDED = "EXCLUDED"

# Conservative regex patterns for obvious unsafe advice
UNSAFE_PRESCRIBE_PATTERN = re.compile(
    r"\b(?:(?:start\s+taking|you\s+should\s+take|recommend\s+taking|take\s+a\s+dose\s+of)"
    r"\s+(?:(?:\d+\s*(?:mg|milligram)s?\s+of\s+)?(?:medication|meds|pills?|antidepressants?|prozac|zoloft|lexapro|xanax|valium|adderall|ritalin|ativan|klonopin|ssris?|benzos?|antibiotics?))"
    r"|take\s+\d+\s*(?:mg|milligram|pill|tablet)s?\s+of\s+(?:medication|meds|pills?|antidepressants?|prozac|zoloft|lexapro|xanax|valium|adderall|ritalin|ativan|klonopin|ssris?|benzos?)"
    r"|I\s+prescribe\b"
    r"|prescribe\s+(?:you|to\s+you)\s+(?:some\s+)?(?:medication|meds|pills?|antidepressants?)"
    r"|dosage\s+of\s+\d+\s*(?:mg|milligram)s?)\b",
    re.I,
)

UNSAFE_STOP_MEDS_PATTERN = re.compile(
    r"\b(?:stop\s+taking|discontinue|quit\s+taking|throw\s+away|stop\s+your)\s+"
    r"(?:your\s+)?(?:medication|meds|pills?|antidepressants?|prescriptions?|prozac|zoloft|lexapro|xanax)\b",
    re.I,
)

UNSAFE_SELF_HARM_PATTERN = re.compile(
    r"\b(?:you\s+should\s+(?:kill|end)\s+your\s*self|"
    r"go\s+ahead\s+and\s+(?:kill|end)\s+your\s*self|"
    r"better\s+off\s+dead|"
    r"nobody\s+(?:would|will)\s+miss\s+you|"
    r"why\s+don['\u2019]?t\s+you\s+(?:just\s+)?(?:kill|end)\s+your\s*self|"
    r"kill\s+your\s*self\s+already|"
    r"cut\s+your\s*self\s+to\s+feel\s+better)\b",
    re.I,
)

UNSAFE_DANGEROUS_PATTERN = re.compile(
    r"\b(?:you\s+should\s+hit\b|"
    r"physically\s+abuse\b|"
    r"physically\s+assault\b|"
    r"go\s+ahead\s+and\s+hit\b|"
    r"punch\s+(?:him|her|them)\b|"
    r"poison\s+(?:him|her|them)\b|"
    r"overdose\s+on\s+(?:pills|drugs|medication)\b)\b",
    re.I,
)

# Content patterns requiring clinician triage rather than automated provisional candidate
DOMESTIC_VIOLENCE_TOPICS = {"domestic violence"}
DOMESTIC_VIOLENCE_PATTERN = re.compile(
    r"\b(?:domestic\s+violence|physical\s+violence|physically\s+violent|partner\s+violence|"
    r"intimate\s+partner\s+violence|chok(?:ed|ing)|slapp(?:ed|ing)|hit\s+me|beat\s+me|"
    r"shov(?:ed|ing)|throw\s+me\s+out|closed\s+.*window\s+on|abusive\s+relationship|coercive\s+control)\b",
    re.I,
)

SUICIDE_SELF_HARM_TOPICS = {"self-harm", "suicide"}
SUICIDE_SELF_HARM_PATTERN = re.compile(
    r"\b(?:suicid\w*|self-harm\w*|kill\s+my\s*self|ending\s+my\s+life|want\s+to\s+die|"
    r"feel\s+like\s+dying|better\s+off\s+dead|hurt\s+my\s*self|cutting\s+my\s*self)\b",
    re.I,
)
CRISIS_URL_HOTLINE_PATTERN = re.compile(
    r"\b(?:call\s*911|emergency\s+room|suicide\s*(?:prevention\s*)?lifeline|"
    r"crisis\s*text\s*line|1-800-273-8255|call\s*988|741741)\b|"
    r"(?:suicidepreventionlifeline\.org|crisistextline\.org|988lifeline\.org)",
    re.I,
)

TRAUMA_COERCION_PATTERN = re.compile(
    r"\b(?:sexual\s+assault|sexual\s+coercion|non-consensual|forced\s+sex|molest\w*|rape\w*|"
    r"didn['\u2019]?t\s+say\s+no\s+to.*advances|coerced\s+into\s+sex|unwanted\s+touching|"
    r"ptsd|post-traumatic\s+stress|flashbacks?)\b",
    re.I,
)

DIAGNOSIS_REQUEST_PATTERN = re.compile(
    r"\b(?:(?:do\s+I\s+have|does\s+my\s+(?:child|daughter|son|kid)\s+have|how\s+do\s+I\s+know\s+if\s+my\s+(?:child|daughter|son|kid)\s+has)\s+"
    r"(?:bipolar|adhd|autism|asperger|schizophreni\w*|borderline|ptsd|depression|ocd|a\s+personality\s+disorder|dissociative)|"
    r"diagnos(?:e|ing|is)\s+my\s+(?:child|daughter|son|kid)|can\s+you\s+diagnose|self-diagnos\w*)\b",
    re.I,
)

LEGAL_MALPRACTICE_TOPICS = {"legal & regulatory"}
LEGAL_MALPRACTICE_PATTERN = re.compile(
    r"\b(?:medical\s+abandonment|malpractice|sue\s+my\s+(?:therapist|doctor)|legal\s+liability|abandonment\s+charges?|against\s+medical\s+advice)\b",
    re.I,
)

SUBSTANCE_TOPICS = {"substance abuse", "addiction"}
SUBSTANCE_PATTERN = re.compile(
    r"\b(?:substance\s+abuse|drug\s+addiction|alcoholism|abusing\s+(?:her|his|my|their|prescription)\s+(?:meds|medicine|pills|drugs)|"
    r"withdrawal\s+symptoms?|narcotics?\s+anonymous|aa\s+meeting)\b",
    re.I,
)

MEDICATION_MENTION_PATTERN = re.compile(
    r"\b(?:prozac|zoloft|lexapro|xanax|valium|adderall|ritalin|ativan|klonopin|dosage|\d+\s*mg\b)\b",
    re.I,
)

HTML_OR_ENTITY_PATTERN = re.compile(
    r"<[a-zA-Z\/][^>]*>|&(?:[a-zA-Z]+|#\d+|#[xX][0-9a-fA-F]+);",
)


def screen_record(
    record: Dict[str, Any],
    question_counts: Counter,
) -> Tuple[str, List[str], float, bool]:
    """Screen a single CounselChat record and determine its curation status, reasons,

    automated quality score, and whether an unsafe pattern was detected.

    Returns:
        (curation_status, curation_reasons, automated_quality_score, is_unsafe_pattern)
    """
    title = str(record.get("question_title", "") or "").strip()
    question = str(record.get("question_text", "") or "").strip()
    answer = str(record.get("answer_text", "") or "").strip()
    flags = [str(f) for f in record.get("quality_flags", []) if f]
    topics = [str(t) for t in record.get("topics", []) if t]

    ans_len = len(answer)
    q_lower = question.lower()

    excluded_reasons: List[str] = []
    is_unsafe_pattern = False

    # 1. EXCLUDED checks
    if ans_len == 0 or "missing_answer" in flags:
        excluded_reasons.append("empty_answer")
    elif ans_len < 20:
        excluded_reasons.append("unusable_short_answer")

    if (not title and not question) or "missing_question_title_and_text" in flags:
        excluded_reasons.append("missing_question_title_and_text")

    # Unsafe patterns (medication prescribing, stop medication, self-harm, dangerous advice)
    if UNSAFE_PRESCRIBE_PATTERN.search(answer):
        excluded_reasons.append("unsafe_medication_prescribing")
        is_unsafe_pattern = True
    if UNSAFE_STOP_MEDS_PATTERN.search(answer):
        excluded_reasons.append("unsafe_stop_medication_instruction")
        is_unsafe_pattern = True
    if UNSAFE_SELF_HARM_PATTERN.search(answer):
        excluded_reasons.append("unsafe_self_harm_encouragement")
        is_unsafe_pattern = True
    if UNSAFE_DANGEROUS_PATTERN.search(answer):
        excluded_reasons.append("dangerous_advice")
        is_unsafe_pattern = True

    # Encoding corruption / binary artifacts
    if "\ufffd" in answer or "\x00" in answer or "<script" in answer.lower():
        excluded_reasons.append("corrupted_text_encoding")

    if excluded_reasons:
        return STATUS_EXCLUDED, excluded_reasons, 0.0, is_unsafe_pattern

    # 2. NEEDS_REVIEW checks
    review_reasons: List[str] = []

    if not question or "missing_question_text" in flags:
        review_reasons.append("missing_question_text")
    elif question_counts.get(q_lower, 0) > 1:
        review_reasons.append("duplicate_question")

    if ans_len < 100 or "short_answer_review" in flags:
        review_reasons.append("short_answer")
    elif ans_len > 5000 or "unusually_long_cleaned_answer_review" in flags:
        review_reasons.append("unusually_long_answer")

    if not topics or "missing_topic_review" in flags:
        review_reasons.append("missing_topics")

    if HTML_OR_ENTITY_PATTERN.search(answer):
        review_reasons.append("html_or_encoded_content")

    # Safety refinements based on targeted clinical audit
    top_lower = {t.lower() for t in topics}
    full_text = f"{title} {question} {answer}"
    q_all = f"{title} {question}"

    # 1. Domestic violence: topic or explicit ongoing physical violence / coercive abuse
    if top_lower & DOMESTIC_VIOLENCE_TOPICS or DOMESTIC_VIOLENCE_PATTERN.search(full_text):
        review_reasons.append("domestic_violence_or_abuse")

    # 2. Suicide & self-harm: topic, ideation, attempts, or crisis hotline URLs (including space-less URLs)
    if (
        top_lower & SUICIDE_SELF_HARM_TOPICS
        or SUICIDE_SELF_HARM_PATTERN.search(full_text)
        or CRISIS_URL_HOTLINE_PATTERN.search(full_text)
    ):
        review_reasons.append("crisis_or_self_harm_content")

    # 3. Trauma & sexual coercion: assault, non-consensual coercion, PTSD flashbacks
    if TRAUMA_COERCION_PATTERN.search(full_text):
        review_reasons.append("trauma_or_sexual_coercion")

    # 4. Diagnostic requests & medical/legal malpractice
    if DIAGNOSIS_REQUEST_PATTERN.search(q_all):
        review_reasons.append("diagnostic_request")

    if top_lower & LEGAL_MALPRACTICE_TOPICS or LEGAL_MALPRACTICE_PATTERN.search(full_text):
        review_reasons.append("legal_or_malpractice_content")

    # 5. Substance abuse & addiction
    if top_lower & SUBSTANCE_TOPICS or SUBSTANCE_PATTERN.search(full_text):
        review_reasons.append("substance_abuse_or_addiction")

    # 6. Specific medication / dosage mentions requiring clinical triage
    if MEDICATION_MENTION_PATTERN.search(answer):
        review_reasons.append("medication_or_dosage_mention")

    for f in flags:
        if f not in (
            "missing_answer",
            "missing_question_title_and_text",
            "missing_question_text",
            "short_answer_review",
            "missing_topic_review",
            "unusually_long_cleaned_answer_review",
        ) and f not in review_reasons:
            review_reasons.append(f)

    if review_reasons:
        # Compute quality score for NEEDS_REVIEW (0.15 to 0.65)
        score = 0.55
        if "duplicate_question" in review_reasons:
            score -= 0.15
        if "missing_question_text" in review_reasons:
            score -= 0.10
        if "short_answer" in review_reasons:
            score -= 0.10
        if "missing_topics" in review_reasons:
            score -= 0.10
        if (
            "domestic_violence_or_abuse" in review_reasons
            or "crisis_or_self_harm_content" in review_reasons
            or "trauma_or_sexual_coercion" in review_reasons
        ):
            score -= 0.10
        if "diagnostic_request" in review_reasons or "legal_or_malpractice_content" in review_reasons:
            score -= 0.05
        if "substance_abuse_or_addiction" in review_reasons or "medication_or_dosage_mention" in review_reasons:
            score -= 0.05
        score = max(0.15, min(0.65, round(score, 2)))
        return STATUS_NEEDS_REVIEW, review_reasons, score, False

    # 3. PROVISIONAL_CANDIDATE
    # Clean candidate score (0.80 to 1.0)
    score = 0.85
    if 300 <= ans_len <= 2500:
        score += 0.05
    if len(question) > 50:
        score += 0.05
    if len(topics) >= 2:
        score += 0.05
    score = min(1.0, round(score, 2))

    return STATUS_PROVISIONAL, ["clean_candidate"], score, False


def run_auto_curation(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Execute the automated CounselChat curation pipeline."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found: {input_path}")

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            records.append(json.loads(line_str))

    # Pre-compute question frequencies for duplicate detection
    question_counts = Counter(
        str(r.get("question_text", "") or "").strip().lower()
        for r in records
        if str(r.get("question_text", "") or "").strip()
    )

    curated_records: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    unsafe_pattern_count = 0
    empty_answer_count = 0
    duplicate_records_count = 0

    for record in records:
        # Track original quality flags
        for flag in record.get("quality_flags", []):
            quality_flag_counts[flag] += 1

        ans_text = str(record.get("answer_text", "") or "").strip()
        if not ans_text:
            empty_answer_count += 1

        status, reasons, score, is_unsafe = screen_record(record, question_counts)

        if is_unsafe:
            unsafe_pattern_count += 1

        if "duplicate_question" in reasons:
            duplicate_records_count += 1

        status_counts[status] += 1
        for reason in reasons:
            reason_counts[reason] += 1

        # Build output record: strictly preserve original fields and append curation metadata
        curated_record = dict(record)
        curated_record["curation_status"] = status
        curated_record["curation_reasons"] = reasons
        curated_record["automated_quality_score"] = score
        # Requirement 6: retrieval_permission MUST be "no" for all initially
        curated_record["retrieval_permission"] = "no"
        # Requirement 4: clinical_approval MUST be "not_assessed"
        curated_record["clinical_approval"] = "not_assessed"

        curated_records.append(curated_record)

    # Calculate repeated question groups and duplicate counts
    repeated_question_groups = sum(1 for q, count in question_counts.items() if count > 1)
    records_in_repeated_groups = sum(count for q, count in question_counts.items() if count > 1)
    excess_duplicate_records = sum(count - 1 for q, count in question_counts.items() if count > 1)

    report = {
        "dataset": "elvindavis/counselchat",
        "source_file": str(input_path),
        "output_file": str(output_path),
        "total_records": len(curated_records),
        "provisional_candidates": status_counts[STATUS_PROVISIONAL],
        "needs_review": status_counts[STATUS_NEEDS_REVIEW],
        "excluded": status_counts[STATUS_EXCLUDED],
        "classification_counts": {
            STATUS_PROVISIONAL: status_counts[STATUS_PROVISIONAL],
            STATUS_NEEDS_REVIEW: status_counts[STATUS_NEEDS_REVIEW],
            STATUS_EXCLUDED: status_counts[STATUS_EXCLUDED],
        },
        "quality_flag_counts": dict(quality_flag_counts),
        "repeated_question_groups": repeated_question_groups,
        "records_in_repeated_question_groups": records_in_repeated_groups,
        "excess_duplicate_records": excess_duplicate_records,
        "duplicate_question_count": records_in_repeated_groups,
        "duplicate_question_records_in_review": duplicate_records_count,
        "empty_answer_count": empty_answer_count,
        "unsafe_pattern_flag_count": unsafe_pattern_count,
        "curation_reason_counts": dict(reason_counts),
        "retrieval_permission": "no",
        "clinical_approval": "not_assessed",
        "faiss_indexed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write output JSONL atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(f"{output_path.name}.tmp_{os.getpid()}")
    try:
        with tmp_output.open("w", encoding="utf-8") as f:
            for rec in curated_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_output.replace(output_path)
    except Exception:
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except OSError:
                pass
        raise

    # Write report JSON atomically
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = report_path.with_name(f"{report_path.name}.tmp_{os.getpid()}")
    try:
        with tmp_report.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_report.replace(report_path)
    except Exception:
        if tmp_report.exists():
            try:
                tmp_report.unlink()
            except OSError:
                pass
        raise

    # Console output summary
    print("\n=== COUNSELCHAT AUTOMATED CURATION REPORT ===")
    print(f"Total records screened:              {report['total_records']}")
    print(f"Provisional candidates:              {report['provisional_candidates']}")
    print(f"Needs review:                        {report['needs_review']}")
    print(f"Excluded:                            {report['excluded']}")
    print(f"Repeated question groups:            {report['repeated_question_groups']}")
    print(f"Records in repeated question groups: {report['records_in_repeated_question_groups']}")
    print(f"Excess duplicate records:            {report['excess_duplicate_records']}")
    print(f"Duplicate records in review queue:   {report['duplicate_question_records_in_review']}")
    print(f"Empty answers:                       {report['empty_answer_count']}")
    print(f"Unsafe patterns flagged:             {report['unsafe_pattern_flag_count']}")
    print(f"Original quality flags:              {report['quality_flag_counts']}")
    print(f"Output JSONL:                        {output_path}")
    print(f"Report JSON:                         {report_path}")
    print("Retrieval permission:                no (all records)")
    print("Clinical approval:                   not_assessed")
    print("FAISS indexing:                      not performed")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run automated deterministic curation on the cleaned CounselChat dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    run_auto_curation(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
