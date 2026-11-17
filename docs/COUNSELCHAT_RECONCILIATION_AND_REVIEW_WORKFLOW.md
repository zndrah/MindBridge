# MindBridge 2.0 — CounselChat Dataset Reconciliation & Human Review Workflow
**Authoritative Architectural and Operational Specification**  
*Document Version: 2.0.0 | Date: 2026-11-03 | Scope: Phase 2B Dataset Reconciliation & Human Review*

---

## 1. Executive Summary & Objective

In accordance with project requirements, the MindBridge engineering team has completed the exhaustive reconciliation of the original **9-column CounselChat dataset** (`weiting016/counselchat-data`) with the existing 1,482-record processing and review pipeline.

This implementation achieves:
1. **100% Exact 1-to-1 Row-by-Row Reconciliation**: Verified bit-for-bit alignment across all 1,482 rows between the Kaggle dataset source and the existing data pipeline without any shifted, dropped, or reordered records.
2. **Canonical Data Model**: Preserved all nine original fields alongside full project governance metadata in `data/processed/counselchat/counselchat_canonical.jsonl` and `counselchat_canonical.csv`.
3. **Safe Review Migration**: Migrated all previous 40-record sample review recommendations and dataset-wide classifications with **zero migration exceptions** and **zero fabricated reviewer identities or timestamps**.
4. **Governed Human Review Workflow**: Partitioned the 1,063 `SUITABLE_CANDIDATE` records into 22 auditable review batches of 50 records displaying full therapist attribution, URLs, duplicate groups, and quality flags.
5. **Absolute Invariant Gating**: Enforced strict validation safeguards ensuring `counselchat_approved.jsonl` remains strictly **0 bytes** and CounselChat FAISS vectors remain strictly **0** until authentic human reviewers approve records.

---

## 2. Dataset Provenance, Source & Nine-Column Schema

### Source Details
- **Primary Upstream Source**: Kaggle CounselChat dataset (`weiting016/counselchat-data`)
- **Original Archive Path**: `C:\Users\adars\Downloads\archive (8)\counselchat-data.csv`
- **WSL Project Path**: `data/raw/counselchat/counselchat-data.csv`
- **File Size**: 3,019,672 bytes
- **SHA-256 Checksum**: `00d7b44d1bb3c300b6077d1442e32db7ac41b6cc337122d238e0ca78b75c7825`
- **Total Record Count**: Exactly 1,482 records

### Original Nine Columns

| # | Column Name | Type | Unique Count | Null Count | Description |
| :-: | :--- | :--- | :-: | :-: | :--- |
| 1 | `questionID` | `str` | 781 | 0 | Unique MongoDB ObjectId representing the user's inquiry. |
| 2 | `questionTitle` | `str` | 774 | 2 | Subject title summarizing the user's presenting dilemma. |
| 3 | `questionText` | `str` | 718 | 99 | Detailed narrative text submitted by the user (99 records title-only). |
| 4 | `questionUrl` | `str` | 780 | 0 | Canonical web URL of the CounselChat forum thread. |
| 5 | `topics` | `str` | 31 | 10 | Primary clinical or situational tags (e.g. *Anxiety*, *Family Conflict*). |
| 6 | `therapistName` | `str` | 159 | 0 | Professional name of the licensed counselor responding. |
| 7 | `therapistUrl` | `str` | 155 | 0 | Profile URL of the therapist on the CounselChat platform. |
| 8 | `answerText` | `str` | 1,478 | 0 | Counselor's response text containing raw HTML formatting. |
| 9 | `upvotes` | `int` | 1 (all 0) | 0 | Community peer upvote tally (all 0 in this dataset export). |

---

## 3. Data Reconciliation Results

A rigorous comparison script was executed across `counselchat-data.csv`, `cleaned_counselchat_data.csv`, `counselchat_cleaned.jsonl`, and `counselchat_full_review.jsonl`:

```text
Reconciliation Metrics:
- Original CSV Records:                1,482
- Existing Pipeline Records:           1,482
- Exact Row-by-Row Matches:            1,482 (100.00%)
- Content-Only Matches:                0 (no fallback matching required)
- Ambiguous Matches:                   0
- Unmatched Records:                   0
- Row Order Deviations:                0
- Shared Column Mismatches:            0 (questionTitle: 0, questionText: 0, topics: 0, answerText: 0)
- Reconciliation Status:               SUCCESS_EXACT_ONE_TO_ONE
- Safe for Decision Migration:         YES
```

### Uniqueness & Duplicate Distribution
- **Unique Questions**: 781 unique `questionID` values.
  - Single-answer questions: 480 (61.46%)
  - Multi-answer questions: 301 (38.54%)
  - Maximum answers per single question: 22 answers (Question ID `579e400d188cd61c1684e835`)
- **Duplicate Answers**: Exactly 5 records shared identical answer text. All 5 consisted solely of the empty HTML string `<p><br></p>` (Rows 335, 962, 1029, 1190, 1191), which are flagged and permanently excluded.
- **Duplicate (questionID, therapistName) Pairs**: Exactly 1 occurrence (Rows 1147 and 1329: therapist Marissa Talarico provided two distinct answers to the same question).
- **Duplicate (questionID, answerText) Pairs**: 0 occurrences. Every single question-answer pairing is unique.

---

## 4. Canonical CounselChat Data Model

The canonical dataset is stored in:
- JSONL: `data/processed/counselchat/counselchat_canonical.jsonl`
- CSV: `data/processed/counselchat/counselchat_canonical.csv`
- Master Queue: `data/processed/counselchat/counselchat_master_review_queue.csv`

Every canonical record combines all nine original fields with MindBridge governance metadata:

```json
{
  "questionID": "5566fab2a64752d71ec3ca69",
  "questionTitle": "Escalating disagreements between mother and wife",
  "questionText": "My wife and mother are having tense disagreements...",
  "questionUrl": "https://counselchat.com/questions/escalating-disagreements-between-mother-and-wife",
  "topics": ["Family Conflict"],
  "topics_raw": "Family Conflict",
  "therapistName": "Kristi King-Morgan, LMSW",
  "therapistUrl": "https://counselchat.com/therapists/kristi-king-morgan-lmsw",
  "answerText": "<p>What you are describing is something psychologists have termed &#34;triangulation&#34;...</p>",
  "clean_answer_text": "What you are describing is something psychologists have termed \"triangulation\"...",
  "upvotes": 0,
  "stable_record_id": "counselchat_0000",
  "source_dataset": "weiting016/counselchat-data",
  "source_row_index": 0,
  "duplicate_group_id": "",
  "duplicate_group_size": 1,
  "quality_flags": [],
  "automated_review_category": "SUITABLE_CANDIDATE",
  "automated_review_reasons": "Clean, constructive psychoeducational counseling guidance with respectful tone and clear situational context.",
  "previous_sample_decision": null,
  "previous_sample_notes": null,
  "human_decision": "pending",
  "reviewer": null,
  "reviewed_at": null,
  "review_notes": null,
  "retrieval_permission": "no",
  "clinical_approval": "not_assessed",
  "faiss_indexed": false
}
```

---

## 5. Review Migration and Four-Tier Classification

The 1,482 records are preserved across four governed tiers:

| Tier | Record Count | % of Dataset | Retrieval Eligibility | Purpose & Role |
| :--- | :---: | :---: | :---: | :--- |
| **`SUITABLE_CANDIDATE`** | 1,063 | 71.73% | Eligible for Human Review | High-quality psychoeducational guidance partitioned into batches for review. |
| **`NEEDS_HUMAN_REVIEW`** | 358 | 24.16% | Quarantined | Sensitive clinical subjects (medication discussions, trauma, divorce conflict) held in separate review category. |
| **`RESTRICTED_SAFETY_EVALUATION`** | 49 | 3.31% | Permanently Blocked from User RAG | Suicide ideation, self-injury disclosures, crisis hotline referrals. Used exclusively as test cases for crisis guardrails. |
| **`EXCLUDE_FROM_RETRIEVAL`** | 12 | 0.81% | Permanently Blocked | Empty answers (`<p><br></p>`), fragments under 50 chars, and shaming/invalidating counseling advice. |
| **Total** | **1,482** | **100.00%** | — | **Zero records deleted or discarded.** |

### Prior Sample Decision Migration
- 40 records previously audited in the preliminary sample audit are linked to their canonical records.
- All 8 preliminary recommendations remain classified with `human_decision = "pending"`, `reviewer = null`, `reviewed_at = null`, and `retrieval_permission = "no"`.
- Zero automated approvals occurred; zero synthetic reviewer identities were generated.

---

## 6. Human Review Batch Organization

The 1,063 `SUITABLE_CANDIDATE` records are organized into 22 batch CSV files in `data/processed/counselchat/human_review_batches/`:
- **Batches 01 to 21**: 50 records each (1,050 records)
- **Batch 22**: 13 records (total 1,063 records)

### Batch Schema
Each row in the review batch CSV provides reviewers with full context:
1. `record_id`: Stable identifier (`counselchat_XXXX`)
2. `question_id`: Upstream MongoDB question ID
3. `source_dataset`: `weiting016/counselchat-data`
4. `source_row_index`: Source line index
5. `batch_id`: e.g. `batch_01`
6. `batch_index`: Record 1..50 within batch
7. `question_title`: Question subject
8. `question_text`: Complete user narrative
9. `topics`: Clinical topic tags
10. `therapist_name`: Answering counselor
11. `therapist_url`: Answering counselor profile URL
12. `answer_text`: Clean, readable counseling response (HTML stripped, no Base64 clutter)
13. `question_url`: Web forum link
14. `upvotes`: Community upvotes
15. `answer_characters`: Response length
16. `duplicate_group_id`: Duplicate question group ID
17. `duplicate_group_size`: Total counselor answers for this question
18. `quality_flags`: Automated pipeline quality flags
19. `classification`: `SUITABLE_CANDIDATE`
20. `classification_reason`: Pipeline curation notes
21. `human_decision`: `pending` / `approve` / `reject` / `needs_further_review`
22. `review_status`: `pending_review` / `approved` / `rejected` / `restricted`
23. `reviewer`: Human reviewer initials or academic identity
24. `reviewed_at`: ISO review date (`YYYY-MM-DD`)
25. `review_notes`: Detailed justification for decision
26. `retrieval_permission`: `no` / `yes`
27. `clinical_approval`: Strictly `not_assessed`

---

## 7. Approval Validation Safeguards (Safeguards 1–8)

Before any record can be compiled from a review batch into `counselchat_approved.jsonl`, it must pass all eight validation safeguards in `validate_counselchat_approvals.py` and `human_review_workflow.py`:

1. **Explicit Human Decision**: `human_decision` must be `"approve"` (`review_status == "approved"`).
2. **Reviewer Identity Present**: `reviewer` must contain a non-empty, genuine human reviewer identity. Generic placeholders (`REDACTED_EXAMPLE_REVIEWER`, `NONE`, `UNKNOWN`, `TEST`, `ADMIN`, `USER`) are strictly rejected.
3. **Valid ISO Timestamp**: `reviewed_at` must contain a valid ISO calendar date (`YYYY-MM-DD`).
4. **Substantive Review Notes**: `review_notes` must contain at least 10 characters justifying why the answer is safe and constructive.
5. **Coupled Retrieval Permission**: `retrieval_permission` must be explicitly set to `"yes"`. Non-approved records strictly require `"no"`.
6. **Zero Quality Flags**: `quality_flags` must be completely empty (`[]`).
7. **Category Exclusion**: Records classified as `RESTRICTED_SAFETY_EVALUATION` or `EXCLUDE_FROM_RETRIEVAL` are unconditionally barred from approval.
8. **Stable Source Mapping**: The `record_id` must map to exactly one canonical source record without modification of question title, text, or source index.

---

## 8. Clinical vs. Content-Suitability Distinction & RAG Hierarchy

> [!IMPORTANT]
> **Therapist Attribution is Provenance, NOT Clinical Verification**  
> While CounselChat answers were written by individuals identifying as licensed therapists, their presence in a web scrape does not independently verify medical or clinical accuracy.
>
> **Content-Suitability Approval $\neq$ Clinical Approval**  
> Human student or project reviewer evaluation assesses tone, empathy, absence of harm, and general psychoeducational utility. It **does NOT constitute clinical certification**. `clinical_approval` must remain strictly `"not_assessed"`.
>
> **RAG Priority Hierarchy**  
> 1. **Primary**: Canonical verified clinical knowledge base (`data/knowledge_base/faiss_index/`, 31 vectors).
> 2. **Secondary**: CounselChat Q&A examples (illustrative only, strictly unverified, empty until human review).
> 3. **Emergency Bypass**: In acute crisis situations, vector retrieval across ALL collections is bypassed entirely in favor of deterministic hotlines.

---

## 9. Verification & Checksums

| File | Status | Checksum / Count |
| :--- | :---: | :--- |
| `counselchat-data.csv` (Original) | Verified | SHA-256: `00d7b44d1bb3c300b6077d1442e32db7ac41b6cc337122d238e0ca78b75c7825` (1,482 rows) |
| `cleaned_counselchat_data.csv` (Raw) | Preserved | SHA-256: `316c6f7fbb2dda02ebbc6b0548eccd8bf1b012c485dee3922df8b095757ff40a` (1,482 rows) |
| `counselchat_approved.jsonl` | Empty | **0 bytes** (0 approved records) |
| `index.faiss` (Canonical Clinical KB) | Unchanged | SHA-256: `c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888` (31 vectors) |
| CounselChat FAISS Vectors | Sealed | **0 vectors** |
