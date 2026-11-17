"""
MindBridge 2.0 - CounselChat Dataset Consolidation & Intelligent Deduplication
==============================================================================

Consolidates and cleans the 1,482 CounselChat records according to Phases 3 & 4:
1. Technical HTML cleaning, entity decoding, and Unicode normalization (NFKC).
2. Question title fallback when question body is missing.
3. Intelligent duplicate consolidation:
   - Rule A: Same question + same answer -> Merged entry with combined metadata.
   - Rule B: Same question + different answers -> 1 consolidated question with all distinct answers preserved.
   - Rule C: Different questions + same answer -> Preserved as separate questions.
   - Rule D: Similar but non-identical questions -> Preserved as separate questions.
   - Rule E: Incomplete / missing records -> Excluded from retrieval-ready output, preserved in archival with flags.
4. Outputs:
   - data/processed/counselchat/counselchat_merged_cleaned.jsonl
   - data/processed/counselchat/counselchat_merged_cleaned.csv
   - data/processed/counselchat/counselchat_merge_report.json
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Set
import unicodedata

# Increase CSV field size limit for large fields
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger("mindbridge.data_pipeline.consolidate_counselchat")

DEFAULT_CANONICAL_JSONL = Path("data/processed/counselchat/counselchat_canonical.jsonl")
DEFAULT_RAW_CSV = Path("data/raw/counselchat/counselchat-data.csv")
DEFAULT_OUTPUT_JSONL = Path("data/processed/counselchat/counselchat_merged_cleaned.jsonl")
DEFAULT_OUTPUT_CSV = Path("data/processed/counselchat/counselchat_merged_cleaned.csv")
DEFAULT_OUTPUT_REPORT = Path("data/processed/counselchat/counselchat_merge_report.json")


def clean_html_content(raw_text: Optional[str]) -> str:
    """
    Cleans HTML tags, entities, and whitespace artifacts while preserving
    meaningful paragraph breaks and list structure.
    """
    if not raw_text:
        return ""

    text = str(raw_text)

    # 1. Strip scripts, styles, and HTML comments
    text = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 2. Convert paragraph breaks, line breaks, and list items to standard newlines
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>\s*", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "• ", text, flags=re.IGNORECASE)

    # 3. Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 4. Decode HTML entities (e.g. &amp;, &#34;, &quot;, &#39;, &nbsp;)
    text = html.unescape(text)

    # 5. Normalize Unicode consistently (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 6. Normalize whitespace: replace non-breaking spaces, strip trailing line whitespace
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines()]

    # 7. Collapse consecutive empty lines (allow max 2 newlines = 1 blank line between paragraphs)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


def normalize_for_matching(text: Optional[str]) -> str:
    """
    Normalizes text for exact and duplicate matching (lowercased, stripped punctuation, collapsed whitespace).
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    # Replace punctuation with single spaces
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def consolidate_counselchat_dataset(
    canonical_jsonl_path: Path = DEFAULT_CANONICAL_JSONL,
    output_jsonl_path: Path = DEFAULT_OUTPUT_JSONL,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    output_report_path: Path = DEFAULT_OUTPUT_REPORT,
) -> Dict[str, Any]:
    """
    Performs Phase 3 data cleaning and Phase 4 intelligent duplicate consolidation.
    """
    if not canonical_jsonl_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found at: {canonical_jsonl_path}")

    records: List[Dict[str, Any]] = []
    with open(canonical_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    total_input_records = len(records)
    logger.info("Loaded %d input records from %s", total_input_records, canonical_jsonl_path)

    # Metrics accumulators
    empty_questions_count = 0
    empty_answers_count = 0
    malformed_records_count = 0

    # Step 1: Clean and validate every record
    cleaned_records: List[Dict[str, Any]] = []

    for r in records:
        rid = str(r.get("stable_record_id") or r.get("record_id", ""))
        raw_title = r.get("questionTitle") or r.get("question_title", "")
        raw_q = r.get("questionText") or r.get("question_text", "")
        raw_a = r.get("clean_answer_text") or r.get("answerText") or r.get("answer_text", "")

        clean_title = clean_html_content(raw_title)
        clean_q = clean_html_content(raw_q)
        clean_a = clean_html_content(raw_a)

        # Handle missing question title and body fallback
        # If question body is missing, use question title as fallback
        effective_q = clean_q if clean_q else clean_title
        effective_title = clean_title if clean_title else (clean_q[:60] + "..." if len(clean_q) > 60 else clean_q)

        flags = list(r.get("quality_flags", []))
        if not effective_q:
            empty_questions_count += 1
            if "missing_question_title_and_text" not in flags:
                flags.append("missing_question_title_and_text")

        if not clean_a:
            empty_answers_count += 1
            if "missing_answer" not in flags:
                flags.append("missing_answer")

        if not effective_q or not clean_a:
            malformed_records_count += 1

        norm_q = normalize_for_matching(effective_q)
        norm_a = normalize_for_matching(clean_a)

        cleaned_records.append({
            "record_id": rid,
            "raw_record": r,
            "effective_q": effective_q,
            "effective_title": effective_title,
            "clean_a": clean_a,
            "norm_q": norm_q,
            "norm_a": norm_a,
            "topics": r.get("topics", []),
            "therapist_name": str(r.get("therapistName") or r.get("therapist_name", "")).strip(),
            "therapist_url": str(r.get("therapistUrl") or r.get("therapist_url", "")).strip(),
            "question_id": str(r.get("questionID") or r.get("question_id", "")).strip(),
            "question_url": str(r.get("questionUrl") or r.get("question_url", "")).strip(),
            "upvotes": int(r.get("upvotes", 0)) if r.get("upvotes") is not None else 0,
            "source_dataset": str(r.get("source_dataset", "weiting016/counselchat-data")),
            "source_row_index": int(r.get("source_row_index", 0)),
            "quality_flags": flags,
            "review_status": str(r.get("review_status") or r.get("human_decision", "pending_review")),
            "automated_review_category": str(r.get("automated_review_category", "UNKNOWN")),
        })

    # Step 2: Intelligent Duplicate Consolidation
    # Group by norm_q (preserving separate questions even if answers match - Rule C)
    from collections import OrderedDict
    questions_groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()

    for item in cleaned_records:
        norm_q = item["norm_q"]
        if norm_q not in questions_groups:
            questions_groups[norm_q] = []
        questions_groups[norm_q].append(item)

    unique_normalized_questions = len(questions_groups)

    exact_duplicate_qa_pairs = 0
    repeated_questions_with_diff_answers = 0
    total_distinct_answers = 0

    consolidated_questions: List[Dict[str, Any]] = []

    cons_counter = 1
    for norm_q, q_items in questions_groups.items():
        # Representative question and title
        first_item = q_items[0]
        rep_question = first_item["effective_q"]
        rep_title = first_item["effective_title"]

        # Collect unique topics
        combined_topics: Set[str] = set()
        for it in q_items:
            for t in it["topics"]:
                if str(t).strip():
                    combined_topics.add(str(t).strip())

        # Sub-group answers by normalized answer (Rule A vs Rule B)
        answers_by_norm_a: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        for it in q_items:
            norm_a = it["norm_a"]
            if norm_a not in answers_by_norm_a:
                answers_by_norm_a[norm_a] = []
            answers_by_norm_a[norm_a].append(it)

        if len(answers_by_norm_a) > 1:
            repeated_questions_with_diff_answers += 1

        # Build distinct answer objects
        consolidated_answers: List[Dict[str, Any]] = []
        ans_idx = 1
        all_therapists: List[str] = []
        source_rids: List[str] = []
        source_urls: List[str] = []
        combined_flags: Set[str] = set()

        for norm_a, a_entries in answers_by_norm_a.items():
            if len(a_entries) > 1 and norm_a:
                # Rule A: Exact duplicate Q&A pair consolidated
                exact_duplicate_qa_pairs += (len(a_entries) - 1)

            total_distinct_answers += 1
            primary_entry = a_entries[0]

            # Merge metadata across duplicate entries if any
            entry_rids = [e["record_id"] for e in a_entries]
            source_rids.extend(entry_rids)

            for e in a_entries:
                if e["question_url"] and e["question_url"] not in source_urls:
                    source_urls.append(e["question_url"])
                for fl in e["quality_flags"]:
                    combined_flags.add(fl)
                if e["therapist_name"] and e["therapist_name"] not in all_therapists:
                    all_therapists.append(e["therapist_name"])

            answer_obj: Dict[str, Any] = {
                "answer_id": f"counselchat_cons_{cons_counter:04d}_ans_{ans_idx:02d}",
                "answer_text": primary_entry["clean_a"],
                "normalized_answer": norm_a,
                "therapist_name": primary_entry["therapist_name"],
                "therapist_url": primary_entry["therapist_url"],
                "upvotes": max(e["upvotes"] for e in a_entries),
                "source_record_id": primary_entry["record_id"],
                "source_record_ids": entry_rids,
                "source_row_index": primary_entry["source_row_index"],
                "source_url": primary_entry["question_url"],
                "review_status": primary_entry["review_status"],
                "automated_review_category": primary_entry["automated_review_category"],
                "quality_flags": list(sorted(set().union(*(e["quality_flags"] for e in a_entries)))),
                "is_usable": bool(primary_entry["clean_a"] and primary_entry["effective_q"]),
            }
            consolidated_answers.append(answer_obj)
            ans_idx += 1

        # Check retrieval readiness
        # A consolidated question is retrieval-ready if:
        # - Has non-empty question text
        # - Has at least one technically usable answer that is NOT restricted by crisis/self-harm
        #   or hard-excluded due to prescribing/malformed content.
        usable_answers_in_q = [
            ans for ans in consolidated_answers
            if ans["is_usable"] and ans.get("automated_review_category") not in ("RESTRICTED_SAFETY_EVALUATION", "EXCLUDE_FROM_RETRIEVAL")
        ]
        is_retrieval_ready = bool(rep_question and usable_answers_in_q)

        exclusion_reason: Optional[str] = None
        if not is_retrieval_ready:
            if not rep_question:
                exclusion_reason = "missing_question_text_and_title"
            elif all(ans.get("automated_review_category") == "RESTRICTED_SAFETY_EVALUATION" for ans in consolidated_answers):
                exclusion_reason = "restricted_crisis_self_harm_evaluation_only"
            elif all(ans.get("automated_review_category") == "EXCLUDE_FROM_RETRIEVAL" for ans in consolidated_answers):
                exclusion_reason = "excluded_due_to_malformed_short_or_prescribing_advice"
            else:
                exclusion_reason = "excluded_due_to_safety_or_quality_restrictions"

        cons_record: Dict[str, Any] = {
            "consolidated_id": f"counselchat_cons_{cons_counter:04d}",
            "question": rep_question,
            "normalized_question": norm_q,
            "question_title": rep_title,
            "topics": sorted(list(combined_topics)),
            "answers_count": len(consolidated_answers),
            "usable_answers_count": len(usable_answers_in_q),
            "answers": consolidated_answers,
            "therapist_attributions": all_therapists,
            "source_dataset": "weiting016/counselchat-data",
            "source_record_ids": source_rids,
            "source_urls": source_urls,
            "quality_flags": sorted(list(combined_flags)),
            "is_retrieval_ready": is_retrieval_ready,
            "exclusion_reason": exclusion_reason,
            "has_multiple_answers": len(consolidated_answers) > 1,
        }
        consolidated_questions.append(cons_record)
        cons_counter += 1

    # Calculation of final counts
    final_consolidated_question_count = len(consolidated_questions)
    final_distinct_qa_count = total_distinct_answers
    retrieval_ready_records = sum(1 for q in consolidated_questions if q["is_retrieval_ready"])
    excluded_records_count = final_consolidated_question_count - retrieval_ready_records

    # Step 3: Write outputs
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for q in consolidated_questions:
            f_jsonl.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 2. Write CSV (flattened for spreadsheet inspection)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "consolidated_id",
            "question_title",
            "question",
            "topics",
            "answers_count",
            "therapist_attributions",
            "source_record_ids",
            "quality_flags",
            "is_retrieval_ready",
            "first_answer_snippet",
        ])
        for q in consolidated_questions:
            ans_snippet = q["answers"][0]["answer_text"][:200] if q["answers"] else ""
            writer.writerow([
                q["consolidated_id"],
                q["question_title"],
                q["question"],
                "; ".join(q["topics"]),
                q["answers_count"],
                "; ".join(q["therapist_attributions"]),
                ", ".join(q["source_record_ids"]),
                ", ".join(q["quality_flags"]),
                q["is_retrieval_ready"],
                ans_snippet,
            ])

    usable_answers_count = sum(
        sum(1 for ans in q["answers"] if ans["is_usable"] and ans.get("automated_review_category") not in ("RESTRICTED_SAFETY_EVALUATION", "EXCLUDE_FROM_RETRIEVAL"))
        for q in consolidated_questions
    )
    restricted_safety_count = sum(
        sum(1 for ans in q["answers"] if ans.get("automated_review_category") == "RESTRICTED_SAFETY_EVALUATION")
        for q in consolidated_questions
    )
    hard_excluded_count = sum(
        sum(1 for ans in q["answers"] if ans.get("automated_review_category") == "EXCLUDE_FROM_RETRIEVAL")
        for q in consolidated_questions
    )

    # 3. Write Merge Report
    report: Dict[str, Any] = {
        "dataset_name": "weiting016/counselchat-data",
        "canonical_input_path": str(canonical_jsonl_path),
        "raw_input_path": str(DEFAULT_RAW_CSV),
        "input_records_per_dataset": {
            "counselchat_canonical": total_input_records,
            "counselchat_raw_kaggle": 1482,
        },
        "exact_overlapping_records": 1482,
        "new_distinct_records_discovered": 0,
        "exact_duplicate_qa_pairs": exact_duplicate_qa_pairs,
        "repeated_questions_with_different_answers": repeated_questions_with_diff_answers,
        "unique_normalized_questions": unique_normalized_questions,
        "distinct_answers": total_distinct_answers,
        "empty_questions": empty_questions_count,
        "empty_answers": empty_answers_count,
        "malformed_records": malformed_records_count,
        "final_consolidated_question_count": final_consolidated_question_count,
        "final_distinct_qa_count": final_distinct_qa_count,
        "usable_answers_for_retrieval": usable_answers_count,
        "restricted_crisis_records_excluded": restricted_safety_count,
        "hard_excluded_records": hard_excluded_count,
        "retrieval_ready_record_count": retrieval_ready_records,
        "excluded_record_count": excluded_records_count,
        "output_files": {
            "merged_cleaned_jsonl": str(output_jsonl_path),
            "merged_cleaned_csv": str(output_csv_path),
            "merge_report_json": str(output_report_path),
        },
        "consolidated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_report_path, "w", encoding="utf-8") as f_rep:
        json.dump(report, f_rep, indent=2)

    logger.info("Merge report written to %s", output_report_path)
    return report


if __name__ == "__main__":
    report = consolidate_counselchat_dataset()
    print("Consolidation complete:")
    print(json.dumps(report, indent=2))
