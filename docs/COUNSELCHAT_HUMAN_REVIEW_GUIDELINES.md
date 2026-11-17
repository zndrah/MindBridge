# MindBridge 2.0 — CounselChat Human Review Guidelines & Protocol
**Document Version: 1.0.0 | Date: 2026-11-03 | Status: Authoritative Reviewer Guide**

---

## 1. Objective & Purpose

This document provides operational instructions for reviewers conducting human content-suitability evaluations on the **1,063 `SUITABLE_CANDIDATE`** records from the CounselChat dataset (`elvindavis/counselchat`).

In MindBridge, CounselChat records are utilized exclusively as **secondary, unverified peer counseling examples and illustrative psychoeducational perspectives**. They are completely decoupled from the canonical clinical knowledge base (`data/knowledge_base/faiss_index/`).

---

## 2. Review Decision Taxonomy

For every candidate record, the reviewer must assign exactly one `human_decision`:

| Decision | Operational Definition | Retrieval Permission | Clinical Approval | Action in Workflow |
| :--- | :--- | :---: | :---: | :--- |
| **`approve`** | High-quality, constructive, empathetic, and respectful psychoeducational guidance. Safe for secondary illustrative RAG retrieval. | `yes` | `not_assessed` | Eligible for compilation into `counselchat_approved.jsonl`. |
| **`reject`** | Harmful, misleading, victim-blaming, shaming, invalidating, diagnostic, or unprofessional counseling advice. | `no` | `not_assessed` | Permanently blocked from RAG retrieval. Preserved in dataset for auditability. |
| **`needs_further_review`** | Ambiguous, sensitive, context-dependent, or boundary-testing advice requiring senior clinician triage or project lead discussion. | `no` | `not_assessed` | Held in quarantine; blocked from retrieval until resolved. |
| **`pending`** | Initial default state. Not yet reviewed by a human evaluator. | `no` | `not_assessed` | Blocked from retrieval. |

---

## 3. Strict Approval Safeguards & Mandatory Fields

A record is **strictly rejected** by the approval validator and cannot enter `counselchat_approved.jsonl` unless all of the following requirements are met:

1. **Explicit Human Decision**: `human_decision` must be explicitly set to `"approve"` (`review_status` to `"approved"`).
2. **Reviewer Attribution**: `reviewer` must contain a non-empty, genuine reviewer identity (e.g., `Adarsh (MindBridge Student Reviewer)`). Generic placeholders (such as `REDACTED_EXAMPLE_REVIEWER`, `NONE`, `TEST`) are strictly prohibited in production reviews.
3. **Review Timestamp**: `reviewed_at` must contain a valid ISO calendar date in `YYYY-MM-DD` format (e.g., `2026-11-03`). Future dates or malformed date strings trigger validation failure.
4. **Mandatory Review Notes**: `review_notes` must contain a concise explanation summarizing why the answer is psychoeducationally sound (e.g., *"Approved: Sound psychoeducational advice on boundary setting and emotional self-reflection."*).
5. **Coupled Retrieval Permission**: `retrieval_permission` must be explicitly set to `"yes"`.
6. **Separation of Clinical Approval**: `clinical_approval` must remain strictly `"not_assessed"`.
   > [!IMPORTANT]
   > Human suitability review assesses conversational tone, safety, and constructive psychoeducation. It **does NOT constitute medical board certification or clinical consensus**. Reviewers are strictly forbidden from altering `clinical_approval` to `"approved"` or `"clinically_approved"`.
7. **Zero Unresolved Quality Flags**: `quality_flags` must be empty (`[]`). Any record with flags (e.g. missing question body, extreme length) cannot be approved without remediation.
8. **Category Exclusion**: Records classified under `RESTRICTED_SAFETY_EVALUATION` or `EXCLUDE_FROM_RETRIEVAL` are unconditionally barred from entering retrieval.
9. **Single Stable Mapping**: The `record_id` must map to exactly one canonical source record without alteration of source index or text.

---

## 4. Content Red Lines & Category Definitions

### APPROVE
- Constructive, empathetic, relevant, non-harmful psychoeducational counseling advice.
- Clear and contextually appropriate responses providing emotional validation and coping mechanisms.
- Answers that do not make unsupported medical or clinical claims.

### REJECT (Content Red Lines)
Reviewers must **immediately reject** (`human_decision = "reject"`) any answer exhibiting any of the following:
1. **Medication Prescribing or Dosing**: Recommending specific psychiatric medications, specific milligram dosages (e.g., *"Take 20mg Prozac"*), or instructing the user to stop/alter medications prescribed by their doctor.
2. **Diagnostic Labeling**: Asserting formal clinical diagnoses (e.g., *"You are suffering from Borderline Personality Disorder"*, *"Your husband is clearly a narcissist"*).
3. **Invalidating, Shaming or Dismissive Tone**: Blaming the user for their distress, expressing impatience, or using condescending language (e.g. *"I hate to sound harsh but... have self worth of a $1"*).
4. **Pseudo-Scientific Claims**: Asserting unverified medical claims regarding brain chemicals, hormone imbalances, or unproven holistic cures without grounding.
5. **Breach of Confidentiality or Legal Boundaries**: Encouraging illegal acts, custody violations, or deceptive behavior.

### NEEDS FURTHER REVIEW
Assign `human_decision = "needs_further_review"` for:
- Ambiguous or incomplete counseling answers.
- Complex trauma disclosures, non-acute domestic conflict, or domestic violence nuances.
- Substance abuse recovery, dependency, and withdrawal discussions.
- Complex ethical, custody, or legal disputes.
- Any situation where the reviewer is uncertain whether the advice is clinically appropriate.

---

## 5. Provenance, Therapist Attribution & Clinical Truth

> [!IMPORTANT]
> **Therapist Attribution is Provenance, Not Verification**  
> Every batch record displays `therapist_name` and `therapist_url` from the Kaggle dataset. This attribution represents historical forum metadata (who authored the post). It **does NOT certify** that the individual response is clinically validated or medically safe.
>
> **Human Review is Content-Suitability, Not Clinical Certification**  
> Project reviewers evaluate conversational tone, safety, empathy, and general psychoeducational utility. This **does NOT constitute medical board certification or clinical consensus**. Therefore, `clinical_approval` must remain strictly `"not_assessed"`.
>
> **CounselChat is Secondary to Canonical Protocols**  
> MindBridge's verified clinical knowledge base (`data/knowledge_base/faiss_index/`, 31 vectors) is the primary authoritative source. Approved CounselChat records provide secondary peer perspectives only. In acute crisis situations, emergency bypass takes precedence over all RAG retrieval.

---

## 5. Duplicate Question Protocol

In CounselChat, multiple counselors often answered the exact same user question (281 repeated question groups comprising 946 records).

- **Preserve Separate Answers**: Do **not** delete or merge duplicate questions.
- **Independent Evaluation**: Reviewers must evaluate each counselor's answer independently on its own merits:
  - Counselor A's answer may be empathetic and constructive (`approve`).
  - Counselor B's answer to the exact same question may be dismissive or prescriptive (`reject`).
- **Duplicate Tracking**: Every row records `duplicate_group_id` (e.g., `dup_group_001`) and `duplicate_group_size`, enabling the downstream retriever to avoid presenting repetitive answers to the user.

---

## 6. Batch Review Workflow Instructions

1. **Locate Your Batch**: Open the assigned batch CSV in [`data/processed/counselchat/human_review_batches/`](file:///home/adarsh/projects/MindBridge/data/processed/counselchat/human_review_batches/) (e.g. `batch_01.csv`, 50 records).
2. **Read Full Context**: Inspect `question_title`, `question_text`, and `answer_text`.
3. **Enter Decisions**:
   - For suitable records: Set `human_decision = "approve"`, `review_status = "approved"`, `retrieval_permission = "yes"`, `reviewer = "<Your Name>"`, `reviewed_at = "YYYY-MM-DD"`, and `review_notes = "<Your Rationale>"`.
   - For unsuitable records: Set `human_decision = "reject"`, `review_status = "rejected"`, `retrieval_permission = "no"`, `reviewer = "<Your Name>"`, `reviewed_at = "YYYY-MM-DD"`, and `review_notes = "<Rejection Reason>"`.
   - For ambiguous records: Set `human_decision = "needs_further_review"`, `review_status = "restricted"`, `retrieval_permission = "no"`.
4. **Save File**: Save as CSV with UTF-8 encoding.
5. **Run Validation**: Run `backend/data_pipeline/human_review_workflow.py` to validate your batch before compilation into `counselchat_approved.jsonl`.
