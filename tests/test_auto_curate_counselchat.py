"""Unit and integration tests for the automated CounselChat curation pipeline."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from backend.data_pipeline.auto_curate_counselchat import (
    run_auto_curation,
    screen_record,
    STATUS_EXCLUDED,
    STATUS_NEEDS_REVIEW,
    STATUS_PROVISIONAL,
)


def make_clean_record(
    record_id="counselchat_0001",
    question_title="Managing daily stress",
    question_text="What are some practical and evidence-based ways to manage daily workplace stress?",
    topics=None,
    answer_text=None,
    quality_flags=None,
    source_row_index=1,
):
    if topics is None:
        topics = ["Stress Management", "Workplace Mental Health"]
    if quality_flags is None:
        quality_flags = []
    if answer_text is None:
        answer_text = (
            "Managing workplace stress effectively begins with identifying specific stressors "
            "and setting healthy boundaries. Techniques such as structured time blocking, "
            "taking short sensory breaks throughout the day, and practicing diaphragmatic "
            "breathing can significantly reduce physiological tension. It is also beneficial "
            "to communicate clearly with colleagues regarding workload expectations."
        )

    return {
        "record_id": record_id,
        "source_dataset": "elvindavis/counselchat",
        "source_row_index": source_row_index,
        "question_title": question_title,
        "question_text": question_text,
        "topics": topics,
        "answer_text": answer_text,
        "answer_characters": len(answer_text),
        "review_status": "pending_review",
        "quality_flags": quality_flags,
    }


def write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# 1. Clean provisional candidate
def test_provisional_candidate_classification():
    record = make_clean_record()
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_PROVISIONAL
    assert reasons == ["clean_candidate"]
    assert 0.80 <= score <= 1.0
    assert not is_unsafe


# 2. Empty answers are excluded
@pytest.mark.parametrize("empty_ans", ["", "   ", "\n\t"])
def test_empty_answer_is_excluded(empty_ans):
    record = make_clean_record(answer_text=empty_ans)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "empty_answer" in reasons
    assert score == 0.0


def test_missing_answer_quality_flag_is_excluded():
    record = make_clean_record(answer_text="", quality_flags=["missing_answer"])
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "empty_answer" in reasons
    assert score == 0.0


# 3. Unusable short answers (< 20 chars) are excluded
def test_unusable_short_answer_is_excluded():
    record = make_clean_record(answer_text="Certainly.")
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "unusable_short_answer" in reasons
    assert score == 0.0


# 4. Missing both question title and text is excluded
def test_missing_question_title_and_text_is_excluded():
    record = make_clean_record(
        question_title="",
        question_text="",
        quality_flags=["missing_question_title_and_text"],
    )
    q_counts = Counter()
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "missing_question_title_and_text" in reasons
    assert score == 0.0


# 5. Unsafe medication prescribing is excluded
@pytest.mark.parametrize(
    "unsafe_ans",
    [
        "I prescribe you 50mg of Zoloft to take every morning.",
        "You should take 20mg of Lexapro for your severe anxiety.",
        "Start taking Prozac right away to feel better.",
        "Take a dose of 10mg of Xanax when you feel panicked.",
    ],
)
def test_unsafe_medication_prescribing_is_excluded(unsafe_ans):
    record = make_clean_record(answer_text=unsafe_ans)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "unsafe_medication_prescribing" in reasons
    assert is_unsafe is True
    assert score == 0.0


# 6. Unsafe stop medication instruction is excluded
@pytest.mark.parametrize(
    "stop_med_ans",
    [
        "Stop taking your medication immediately if you feel strange.",
        "Discontinue your antidepressants right now without asking.",
        "Throw away your pills, therapy is all you need.",
        "Stop your medication today.",
    ],
)
def test_unsafe_stop_medication_is_excluded(stop_med_ans):
    record = make_clean_record(answer_text=stop_med_ans)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "unsafe_stop_medication_instruction" in reasons
    assert is_unsafe is True
    assert score == 0.0


# 7. Unsafe self-harm encouragement is excluded
@pytest.mark.parametrize(
    "self_harm_ans",
    [
        "You should kill yourself, nobody will miss you anyway.",
        "Go ahead and end yourself if nothing matters.",
        "You would be better off dead.",
        "Nobody would miss you if you were gone.",
    ],
)
def test_unsafe_self_harm_encouragement_is_excluded(self_harm_ans):
    record = make_clean_record(answer_text=self_harm_ans)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "unsafe_self_harm_encouragement" in reasons
    assert is_unsafe is True
    assert score == 0.0


# 8. Dangerous advice is excluded
@pytest.mark.parametrize(
    "dangerous_ans",
    [
        "You should hit him if he refuses to listen to you.",
        "Physically abuse your child to establish firm discipline.",
        "Go ahead and hit your partner when angry.",
    ],
)
def test_dangerous_advice_is_excluded(dangerous_ans):
    record = make_clean_record(answer_text=dangerous_ans)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "dangerous_advice" in reasons
    assert is_unsafe is True
    assert score == 0.0


# 9. Corrupted text or binary artifacts are excluded
def test_corrupted_text_is_excluded():
    record = make_clean_record(answer_text="This answer has corrupted \ufffd characters.")
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_EXCLUDED
    assert "corrupted_text_encoding" in reasons


# 10. Duplicate questions are flagged for review without record deletion
def test_duplicate_questions_flagged_for_review_without_deletion(tmp_path):
    q_text = "How do I cope with overwhelming exam anxiety?"
    r1 = make_clean_record(record_id="rec_1", question_text=q_text, source_row_index=0)
    r2 = make_clean_record(record_id="rec_2", question_text=q_text, source_row_index=1)

    source_path = tmp_path / "source.jsonl"
    out_path = tmp_path / "curated.jsonl"
    rep_path = tmp_path / "report.json"

    write_jsonl(source_path, [r1, r2])

    report = run_auto_curation(source_path, out_path, rep_path)

    # Both records are preserved in output
    assert report["total_records"] == 2
    assert report["needs_review"] == 2
    assert report["provisional_candidates"] == 0
    assert report["duplicate_question_count"] == 2
    assert report["repeated_question_groups"] == 1
    assert report["records_in_repeated_question_groups"] == 2
    assert report["excess_duplicate_records"] == 1

    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    for line in lines:
        assert line["curation_status"] == STATUS_NEEDS_REVIEW
        assert "duplicate_question" in line["curation_reasons"]
        assert line["retrieval_permission"] == "no"
        assert line["clinical_approval"] == "not_assessed"


# 11. Missing question text is flagged for review
def test_missing_question_text_flagged_for_review():
    record = make_clean_record(question_text="", quality_flags=["missing_question_text"])
    q_counts = Counter()
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "missing_question_text" in reasons
    assert 0.15 <= score <= 0.65


# 12. Short answers (20 to 100 chars) flagged for review
def test_short_answer_flagged_for_review():
    record = make_clean_record(answer_text="Try deep breathing and taking a short walk outside.")
    assert 20 <= len(record["answer_text"]) < 100
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "short_answer" in reasons


# 13. Unusually long answers (> 5000 chars) flagged for review
def test_unusually_long_answer_flagged_for_review():
    long_ans = "Detailed advice paragraph. " * 250  # ~7000 chars
    record = make_clean_record(answer_text=long_ans)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "unusually_long_answer" in reasons


# 14. Missing topics flagged for review
def test_missing_topics_flagged_for_review():
    record = make_clean_record(topics=[])
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "missing_topics" in reasons


# 15. HTML tags or entities flagged for review
@pytest.mark.parametrize(
    "html_sample",
    [
        "This answer contains <p>HTML tags</p> that should be reviewed.",
        "Here is some text with &amp; entity encoding.",
    ],
)
def test_html_content_flagged_for_review(html_sample):
    record = make_clean_record(answer_text=html_sample)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "html_or_encoded_content" in reasons


# 16. Crisis hotline content flagged for review (keeping crisis handling independent)
@pytest.mark.parametrize(
    "crisis_sample",
    [
        "Please call 911 immediately if you are in danger.",
        "Contact the Suicide Prevention Lifeline or call 988 right away.",
        "You can reach out to the Crisis Text Line by texting HOME to 741741.",
        "Visit http://www.suicidepreventionlifeline.org/ for 24/7 help.",
        "Check out https://crisistextline.org or 988lifeline.org.",
        "Anytime sadness causes thoughts of suicide or self-harm, seek help.",
    ],
)
def test_crisis_and_self_harm_content_flagged_for_review(crisis_sample):
    record = make_clean_record(answer_text=crisis_sample)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "crisis_or_self_harm_content" in reasons


# 17. Medication discussion flagged for review
def test_medication_discussion_flagged_for_review():
    record = make_clean_record(
        answer_text=(
            "Consulting a psychiatrist about your symptoms is advisable. Some patients "
            "find that 0.5mg of Xanax or starting an SSRI can help alleviate panic attacks "
            "when combined with cognitive behavioral therapy."
        )
    )
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "medication_or_dosage_mention" in reasons


# 18. Full pipeline execution integration test
def test_full_pipeline_run_generates_jsonl_and_report(tmp_path):
    records = [
        # Provisional candidate
        make_clean_record(record_id="c_001", source_row_index=0),
        # Needs review (short answer)
        make_clean_record(
            record_id="c_002",
            question_text="A unique question for c_002?",
            answer_text="Take a brisk walk and drink water daily.",
            source_row_index=1,
        ),
        # Excluded (empty answer)
        make_clean_record(
            record_id="c_003",
            question_text="A unique question for c_003?",
            answer_text="",
            quality_flags=["missing_answer"],
            source_row_index=2,
        ),
        # Excluded (unsafe prescribing)
        make_clean_record(
            record_id="c_004",
            question_text="A unique question for c_004?",
            answer_text="I prescribe you 50mg of Zoloft.",
            source_row_index=3,
        ),
    ]

    source_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "output.jsonl"
    rep_path = tmp_path / "report.json"

    write_jsonl(source_path, records)

    report = run_auto_curation(source_path, out_path, rep_path)

    assert report["total_records"] == 4
    assert report["provisional_candidates"] == 1
    assert report["needs_review"] == 1
    assert report["excluded"] == 2
    assert report["empty_answer_count"] == 1
    assert report["unsafe_pattern_flag_count"] == 1
    assert report["retrieval_permission"] == "no"
    assert report["clinical_approval"] == "not_assessed"

    assert out_path.exists()
    assert rep_path.exists()

    output_lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(output_lines) == 4

    # Confirm all original fields and required metadata exist
    for line in output_lines:
        assert "record_id" in line
        assert "source_dataset" in line
        assert "source_row_index" in line
        assert "curation_status" in line
        assert "curation_reasons" in line
        assert "automated_quality_score" in line
        assert line["retrieval_permission"] == "no"
        assert line["clinical_approval"] == "not_assessed"


# 19. Source dataset remains strictly unmodified during pipeline run
def test_source_dataset_strictly_unmodified(tmp_path):
    records = [make_clean_record()]
    source_path = tmp_path / "source.jsonl"
    out_path = tmp_path / "output.jsonl"
    rep_path = tmp_path / "report.json"

    write_jsonl(source_path, records)
    initial_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

    run_auto_curation(source_path, out_path, rep_path)

    final_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert initial_sha == final_sha


# 20. Existing quality flags prevent provisional candidate classification
def test_existing_quality_flags_prevent_provisional_classification():
    record = make_clean_record(quality_flags=["exact_duplicate_record_review"])
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "exact_duplicate_record_review" in reasons


# 21. All output records have retrieval_permission="no" and clinical_approval="not_assessed"
def test_retrieval_permission_always_no_and_clinical_not_assessed(tmp_path):
    records = [
        make_clean_record(record_id="c_01", source_row_index=0),
        make_clean_record(record_id="c_02", answer_text="", quality_flags=["missing_answer"], source_row_index=1),
        make_clean_record(record_id="c_03", question_text="", quality_flags=["missing_question_text"], source_row_index=2),
    ]
    source_path = tmp_path / "source.jsonl"
    out_path = tmp_path / "output.jsonl"
    rep_path = tmp_path / "report.json"

    write_jsonl(source_path, records)
    run_auto_curation(source_path, out_path, rep_path)

    curated = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(curated) == 3
    for rec in curated:
        assert rec["retrieval_permission"] == "no"
        assert rec["clinical_approval"] == "not_assessed"


# 22. Domestic violence topic and text patterns are flagged for NEEDS_REVIEW
def test_domestic_violence_topic_flagged_for_review():
    record = make_clean_record(topics=["Domestic Violence", "Relationships"])
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "domestic_violence_or_abuse" in reasons


@pytest.mark.parametrize(
    "abuse_text",
    [
        "He became physically violent and choked me during an argument.",
        "She slapped me and threatened to throw me out of the house.",
        "I am dealing with coercive control and an abusive relationship.",
    ],
)
def test_domestic_violence_text_patterns_flagged_for_review(abuse_text):
    record = make_clean_record(answer_text=f"A supportive introduction. {abuse_text} Detailed guidance follows.")
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "domestic_violence_or_abuse" in reasons


# 23. Trauma and sexual coercion are flagged for NEEDS_REVIEW
@pytest.mark.parametrize(
    "trauma_text",
    [
        "I was the victim of sexual assault in college and still struggle.",
        "I feel guilty because I was coerced into sex and did not want it.",
        "I suffer from PTSD and experience severe flashbacks at night.",
    ],
)
def test_trauma_and_coercion_flagged_for_review(trauma_text):
    record = make_clean_record(answer_text=f"Helpful therapeutic context. {trauma_text} Further advice.")
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "trauma_or_sexual_coercion" in reasons


# 24. Diagnostic requests (especially pediatric) are flagged for NEEDS_REVIEW
@pytest.mark.parametrize(
    "diag_q",
    [
        "How do I know if my daughter has bipolar disorder?",
        "Does my child have ADHD or autism?",
        "Do I have borderline personality disorder?",
    ],
)
def test_diagnostic_requests_flagged_for_review(diag_q):
    record = make_clean_record(question_text=diag_q)
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "diagnostic_request" in reasons


# 25. Substance abuse and addiction are flagged for NEEDS_REVIEW
def test_substance_abuse_topics_flagged_for_review():
    for topic in ["Substance Abuse", "Addiction"]:
        record = make_clean_record(topics=[topic, "Mental Health"])
        q_counts = Counter([record["question_text"].lower()])
        status, reasons, score, is_unsafe = screen_record(record, q_counts)

        assert status == STATUS_NEEDS_REVIEW
        assert "substance_abuse_or_addiction" in reasons


def test_substance_abuse_text_flagged_for_review():
    record = make_clean_record(
        answer_text="My friend is abusing her prescription medicine and struggling with alcoholism and drug addiction."
    )
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "substance_abuse_or_addiction" in reasons


# 26. Medical malpractice and legal liability flagged for NEEDS_REVIEW
def test_legal_malpractice_flagged_for_review():
    record = make_clean_record(
        topics=["Legal & Regulatory"],
        answer_text="The general rule against medical abandonment says that healthcare providers have legal liability.",
    )
    q_counts = Counter([record["question_text"].lower()])
    status, reasons, score, is_unsafe = screen_record(record, q_counts)

    assert status == STATUS_NEEDS_REVIEW
    assert "legal_or_malpractice_content" in reasons


# 27. Regression tests for the audit's 5 suspicious sample records
def test_regression_audit_suspicious_samples():
    # counselchat_0024: Domestic violence / physical violence
    r_0024 = make_clean_record(
        record_id="counselchat_0024",
        question_title="When should I give up on a relationship?",
        question_text="He closed his car window on my finger and tried to throw me out of his car.",
        topics=["Relationships", "Domestic Violence"],
        answer_text="Physical violence is of most concern. Violence is never okay.",
    )
    s, reasons, _, _ = screen_record(r_0024, Counter([r_0024["question_text"].lower()]))
    assert s == STATUS_NEEDS_REVIEW
    assert "domestic_violence_or_abuse" in reasons

    # counselchat_0400: Crisis hotline URL without spaces
    r_0400 = make_clean_record(
        record_id="counselchat_0400",
        answer_text="You can call the free 24/7 hotline at http://www.suicidepreventionlifeline.org/ for help.",
    )
    s, reasons, _, _ = screen_record(r_0400, Counter([r_0400["question_text"].lower()]))
    assert s == STATUS_NEEDS_REVIEW
    assert "crisis_or_self_harm_content" in reasons

    # counselchat_0476: Sexual coercion / boundary violation
    r_0476 = make_clean_record(
        record_id="counselchat_0476",
        question_text="I feel guilty because I didn't say no to a guy's advances.",
        answer_text="There is nothing wrong with you. Many people struggle after non-consensual advances.",
    )
    s, reasons, _, _ = screen_record(r_0476, Counter([r_0476["question_text"].lower()]))
    assert s == STATUS_NEEDS_REVIEW
    assert "trauma_or_sexual_coercion" in reasons

    # counselchat_0758: Pediatric psychiatric diagnosis request
    r_0758 = make_clean_record(
        record_id="counselchat_0758",
        question_title="How do I know if my daughter has bipolar disorder?",
        question_text="How do I know if my daughter has bipolar disorder? She has rapid mood swings.",
    )
    s, reasons, _, _ = screen_record(r_0758, Counter([r_0758["question_text"].lower()]))
    assert s == STATUS_NEEDS_REVIEW
    assert "diagnostic_request" in reasons

    # counselchat_0997: Medical abandonment legal liability
    r_0997 = make_clean_record(
        record_id="counselchat_0997",
        topics=["Legal & Regulatory"],
        answer_text="The general rule against medical abandonment says that healthcare providers face liability.",
    )
    s, reasons, _, _ = screen_record(r_0997, Counter([r_0997["question_text"].lower()]))
    assert s == STATUS_NEEDS_REVIEW
    assert "legal_or_malpractice_content" in reasons


