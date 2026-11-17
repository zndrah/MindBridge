# MindBridge 2.0 — CounselChat RAG Pipeline Implementation Report

**Status:** IMPLEMENTED & VERIFIED  
**Date:** 2026-11-03  
**Authors:** MindBridge Engineering Team  
**Scope:** Phase 3 — End-to-End CounselChat RAG Retrieval, FAISS Indexing & Multi-Collection Arbitration

---

## 1. Architectural Overview & Governance Boundaries

MindBridge integrates the professor-assigned **CounselChat** (`elvindavis/counselchat`) dataset into its conversational first-response system as an illustrative, non-clinical peer perspective resource. 

To maintain strict non-clinical safety boundaries and prevent harmful advice propagation, this implementation adheres to six authoritative invariants:

1. **Strict Index Decoupling:**
   - **Canonical Clinical Knowledge Base:** `data/knowledge_base/faiss_index/` (31 vectors, SHA-256 `c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888`). Strictly immutable and untouched.
   - **Production CounselChat Index:** `data/processed/counselchat/faiss_index/`. Populated strictly from `counselchat_approved.jsonl`. Since 0 records have undergone human review approval, the production index holds **0 vectors**.
   - **Experimental Development Index:** `data/processed/counselchat/experimental_faiss_index/`. Dedicated isolated index for RAG prototyping built strictly from the labeled experimental development subset.

2. **Absolute Clinical KB Precedence:**
   In `MultiCollectionRetriever`, canonical clinical knowledge chunks (psychoeducation, somatic exercises like box breathing, and cognitive reframing worksheets) **always take priority**. CounselChat chunks are secondary and fill remaining top-k budget slots only if slots remain.

3. **No Clinical Guidance Substitution:**
   CounselChat examples illustrate community peer perspectives and coping strategies; they **never** substitute for missing authoritative clinical guidance.

4. **Universal Emergency Crisis Bypass:**
   Under acute suicidal intent, self-harm disclosure, or emergency safety routing, all vector retrieval across **both collections** is immediately halted (`retrieval_status = "not_attempted"`). The system issues zero queries to FAISS and dispatches deterministic hotline resources from `hotlines.json`.

5. **Mandatory Non-Clinical Framing & Attribution:**
   All CounselChat chunks surface with therapist attribution (`therapist_name`), source ID, stable Kaggle question URL, and the mandatory non-clinical disclaimer. `clinical_approval` remains strictly `"not_assessed"`.

6. **Unapproved Record Isolation:**
   Unreviewed records are strictly excluded from any production/approved retrieval index.

---

## 2. Experimental Development Subset Specification

To enable immediate end-to-end RAG prototyping without waiting for all 1,063 candidates to complete human review (and without automatically approving unreviewed records), a balanced experimental development subset was extracted:

- **File Location:** `data/experimental/counselchat/counselchat_experimental_dev_subset.jsonl`
- **Total Records:** 15 distinct CounselChat records
- **Total Sentence-Aware Chunks Indexed:** 27 chunks
- **Topics Covered:** Family Conflict, Behavioral Change, Social Relationships, Anxiety, Anger Management, Sleep Improvement, Relationships, Marriage, Depression, Grief and Loss, Parenting, Workplace Relationships, Self-Esteem, Career Counseling.
- **Provenance:** Preserves all 9 original Kaggle columns (`questionID`, `questionTitle`, `questionText`, `questionUrl`, `topics`, `therapistName`, `therapistUrl`, `answerText`, `clean_answer_text`, `upvotes`).
- **Governance Stamp:**
  - `is_experimental_dev`: `true`
  - `dataset_tier`: `"experimental_dev_subset"`
  - `review_status`: `"pending_review"` (preserves unapproved status)
  - `retrieval_permission`: `"experimental_only"`
  - `clinical_approval`: `"not_assessed"`
  - `quality_flags`: `[]`

---

## 3. End-to-End Pipeline Implementation

### 3.1 Components Reused
- **Dense Embedding Engine:** [`BGEEmbeddingEngine`](file:///home/adarsh/projects/MindBridge/backend/retrieval/embeddings.py) (`BAAI/bge-small-en-v1.5`, 384 dimensions) for dense query and chunk embeddings. [`MockEmbeddingEngine`](file:///home/adarsh/projects/MindBridge/backend/retrieval/embeddings.py) for hermetic offline testing.
- **Vector Store:** [`FAISSVectorStore`](file:///home/adarsh/projects/MindBridge/backend/retrieval/vector_store.py) (`faiss.IndexFlatIP` on normalized unit L2 vectors with atomic save, SHA-256 verification, and metadata store).
- **Chunking & Governance:** [`chunk_counselchat_experimental_record`](file:///home/adarsh/projects/MindBridge/backend/retrieval/counselchat_retriever.py) and [`split_answer_into_sentence_slices`](file:///home/adarsh/projects/MindBridge/backend/retrieval/counselchat_rag_design.py) preserving full question context with each answer slice.
- **Orchestrator Graph:** [`build_mindbridge_graph`](file:///home/adarsh/projects/MindBridge/backend/orchestrator/graph.py), [`planning_node`](file:///home/adarsh/projects/MindBridge/backend/orchestrator/planning_node.py), [`response_node`](file:///home/adarsh/projects/MindBridge/backend/orchestrator/response_node.py), and [`deterministic_safety_node`](file:///home/adarsh/projects/MindBridge/backend/orchestrator/safety_node.py).

### 3.2 Key Modules Implemented & Modified
1. **`backend/retrieval/counselchat_retriever.py`:**
   - `build_counselchat_experimental_dev_subset()`: Extracts the 15-record dev subset.
   - `chunk_counselchat_experimental_record()`: Sentence-aware chunker producing metadata-stamped chunks.
   - `build_counselchat_faiss_index()`: Atomically builds and saves FAISS index and metadata.
   - `CounselChatRetriever`: Vector search with cosine similarity thresholding (default 0.60–0.70), record deduplication, and production governance gating.
   - `MultiCollectionRetriever`: Dual-collection arbitration prioritizing canonical clinical KB first, then CounselChat peer examples.
2. **`backend/orchestrator/response_node.py`:**
   - Dedicated synthesis branch for `category == "peer_counseling_example"` providing supportive peer framing and therapist attribution.
   - Automatic attachment of the mandatory non-clinical CounselChat disclaimer.
   - Neutral supportive fallback (`NEUTRAL_SUPPORTIVE_FALLBACK`) whenever retrieval returns empty or out-of-domain.
   - LLM generation failure containment falling back gracefully to deterministic templates.
3. **`backend/orchestrator/planning_node.py`:**
   - Generalized retriever injection accepting `MultiCollectionRetriever` and `CounselChatRetriever`.
4. **`backend/retrieval/__init__.py`:**
   - Exported `CounselChatRetriever`, `MultiCollectionRetriever`, `build_counselchat_faiss_index`, and `build_counselchat_experimental_dev_subset`.

---

## 4. Evaluation Benchmark Results

The evaluation harness in [`scripts/evaluate_counselchat_retrieval.py`](file:///home/adarsh/projects/MindBridge/scripts/evaluate_counselchat_retrieval.py) was run across 13 test queries (8 in-domain counseling disclosures, 3 out-of-domain queries, and 2 acute crisis disclosures):

```
================================================================================
MINDBRIDGE 2.0 - COUNSELCHAT RETRIEVAL EVALUATION REPORT
================================================================================
Timestamp: 2026-11-03T22:56:22.444150+00:00
Threshold: 0.6 | Top-k: 3
--------------------------------------------------------------------------------
In-Domain Recall@k (k=3):        100.0% (8 queries)
In-Domain Precision@1:                 100.0%
Out-of-Domain Abstention Rate:         100.0% (3 queries)
Crisis Emergency Bypass Rate:          100.0% (2 queries)
Attribution & Governance Compliance:   100.0%
Average Retrieval Latency:             303.13 ms
--------------------------------------------------------------------------------
```

---

## 5. Test Suite Verification

A dedicated test suite in [`tests/test_counselchat_rag_pipeline.py`](file:///home/adarsh/projects/MindBridge/tests/test_counselchat_rag_pipeline.py) verifies all 10 core safety and functional requirements:

1. `test_canonical_clinical_faiss_index_remains_strictly_unchanged` — PASSED
2. `test_counselchat_approved_remains_empty_and_production_index_zero_vectors` — PASSED
3. `test_experimental_dev_subset_governance_integrity` — PASSED
4. `test_relevant_document_retrieval` — PASSED
5. `test_irrelevant_query_fallback` — PASSED
6. `test_source_attribution_and_non_clinical_framing` — PASSED
7. `test_unapproved_record_isolation_in_production_mode` — PASSED
8. `test_crisis_bypass_halts_all_retrieval` — PASSED
9. `test_generator_failure_fallback` — PASSED
10. `test_end_to_end_orchestrator_pipeline_flow` — PASSED

---

## 6. Current Limitations & Governance Notice

- **Production Status:** `data/processed/counselchat/counselchat_approved.jsonl` has 0 approved records (0 bytes). The production CounselChat index in `data/processed/counselchat/faiss_index` contains 0 vectors.
- **No Automatic Approval:** Zero CounselChat records have been automatically approved. All 1,063 candidates in `data/processed/counselchat/human_review_batches/` remain in `pending_review` status awaiting explicit human review.
- **Non-Clinical Classification:** The `clinical_approval` status for all CounselChat records is strictly `not_assessed`. They must never be represented to users as medically or clinically verified.
