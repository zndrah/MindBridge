"""
MindBridge Phase 5 Step 5 - Retrieval Quality Audit Runner
Evaluates real local BGE model + FAISS index against 50 realistic queries.
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np

from backend.retrieval import (
    build_retriever_from_knowledge_base,
    RetrievalAuthorization,
)
from experiments.retrieval_eval_dataset import EVALUATION_QUERIES


def run_retrieval_audit(threshold: float = 0.65, top_k: int = 5):
    print(f"=== INITIALIZING RETRIEVAL AUDIT (tau={threshold}, top_k={top_k}) ===")
    t0 = time.perf_counter()
    retriever, summary = build_retriever_from_knowledge_base(default_threshold=threshold)
    init_time = time.perf_counter() - t0
    print(f"Built retriever in {init_time:.2f}s with {retriever.vector_store.total_vectors} vectors.")

    p1_list, p3_list, p5_list = [], [], []
    r1_list, r3_list, r5_list = [], [], []
    mrr_list = []
    
    # Specific tracking for queries expected to return NO results
    # (OOD, unauthorized restricted, blocked cross-jurisdiction, excluded doc)
    empty_expected_count = 0
    empty_correct_count = 0

    # Safety tracking
    restricted_leakage_count = 0
    excluded_leakage_count = 0
    jurisdiction_violations = 0

    # Score tracking
    relevant_scores = []
    irrelevant_scores = []

    per_query_details = []

    for item in EVALUATION_QUERIES:
        qid = item["id"]
        cat = item["category"]
        query = item["query"]
        jur = item["jurisdiction"]
        inc_rest = item["include_restricted"]
        auth_raw = item["auth_context"]
        expected = item["expected_doc_ids"]

        auth = (
            RetrievalAuthorization(
                authorized_by=auth_raw["authorized_by"],
                reason=auth_raw["reason"],
            )
            if auth_raw
            else None
        )

        t_start = time.perf_counter()
        results = retriever.retrieve(
            query=query,
            jurisdiction=jur,
            include_restricted=inc_rest,
            authorization=auth,
            top_k=top_k,
            threshold=threshold,
        )
        latency = (time.perf_counter() - t_start) * 1000

        retrieved_doc_ids = [r["document_id"] for r in results]
        retrieved_scores = [r["score"] for r in results]
        retrieved_tiers = [r["retrieval_tier"] for r in results]
        retrieved_jurs = [r["jurisdiction"] for r in results]

        # 1. Jurisdiction check
        for r_jur in retrieved_jurs:
            if jur in ("IN", "US", "UK", "CA", "AU"):
                if r_jur != jur and r_jur != "GL":
                    jurisdiction_violations += 1
            else:
                if r_jur != "GL":
                    jurisdiction_violations += 1

        # 2. Safety leak checks
        for r_tier, r_doc in zip(retrieved_tiers, retrieved_doc_ids):
            if r_tier == "restricted" and not (inc_rest and auth):
                restricted_leakage_count += 1
            if r_tier == "excluded_pending_review" or "06_vagus_nerve" in r_doc:
                excluded_leakage_count += 1

        # 3. Score collection
        for r_doc, sc in zip(retrieved_doc_ids, retrieved_scores):
            if expected and r_doc in expected:
                relevant_scores.append(sc)
            else:
                irrelevant_scores.append(sc)

        # 4. Evaluation metrics (Precision, Recall, MRR)
        if not expected:
            # Queries expected to return 0 results
            empty_expected_count += 1
            is_empty = (len(results) == 0)
            if is_empty:
                empty_correct_count += 1
                p1, p3, p5 = 1.0, 1.0, 1.0
                r1, r3, r5 = 1.0, 1.0, 1.0
                mrr = 1.0
            else:
                # Returned results when 0 expected -> false positives!
                p1, p3, p5 = 0.0, 0.0, 0.0
                r1, r3, r5 = 0.0, 0.0, 0.0
                mrr = 0.0
        else:
            # Queries expected to return specific docs
            # Precision@K = (relevant in top K) / K
            # Recall@K = (relevant in top K) / (total relevant)
            def compute_pr(k):
                top_k_docs = retrieved_doc_ids[:k]
                rel_ret = [d for d in top_k_docs if d in expected]
                p = len(rel_ret) / k if k > 0 else 0.0
                r = len(set(rel_ret)) / len(expected) if expected else 1.0
                return p, r

            p1, r1 = compute_pr(1)
            p3, r3 = compute_pr(3)
            p5, r5 = compute_pr(5)

            # MRR: reciprocal rank of first relevant doc
            rr = 0.0
            for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id in expected:
                    rr = 1.0 / rank_idx
                    break
            mrr = rr

        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        r1_list.append(r1)
        r3_list.append(r3)
        r5_list.append(r5)
        mrr_list.append(mrr)

        per_query_details.append({
            "id": qid,
            "category": cat,
            "query": query,
            "expected": expected,
            "retrieved": list(zip(retrieved_doc_ids, retrieved_scores)),
            "mrr": mrr,
            "p1": p1,
            "latency": latency,
        })

    print(f"\n=== EVALUATION RESULTS (N={len(EVALUATION_QUERIES)}) ===")
    print(f"Mean Reciprocal Rank (MRR): {np.mean(mrr_list):.4f}")
    print(f"Precision@1: {np.mean(p1_list):.4f} | Recall@1: {np.mean(r1_list):.4f}")
    print(f"Precision@3: {np.mean(p3_list):.4f} | Recall@3: {np.mean(r3_list):.4f}")
    print(f"Precision@5: {np.mean(p5_list):.4f} | Recall@5: {np.mean(r5_list):.4f}")
    print(f"No-Result Expected Queries: {empty_expected_count}")
    print(f"No-Result Correct (True Negatives): {empty_correct_count}/{empty_expected_count} ({empty_correct_count/empty_expected_count*100:.1f}%)")
    print(f"Restricted Leakage Count: {restricted_leakage_count}")
    print(f"Excluded Leakage Count: {excluded_leakage_count}")
    print(f"Jurisdiction Violations: {jurisdiction_violations}")

    if relevant_scores:
        print(f"\nRelevant Match Scores (N={len(relevant_scores)}):")
        print(f"  Min: {np.min(relevant_scores):.4f} | Mean: {np.mean(relevant_scores):.4f} | Median: {np.median(relevant_scores):.4f} | Max: {np.max(relevant_scores):.4f}")
    if irrelevant_scores:
        print(f"Irrelevant Match Scores (N={len(irrelevant_scores)}):")
        print(f"  Min: {np.min(irrelevant_scores):.4f} | Mean: {np.mean(irrelevant_scores):.4f} | Median: {np.median(irrelevant_scores):.4f} | Max: {np.max(irrelevant_scores):.4f}")

    print("\n=== PROBLEMATIC / WEAK MATCH QUERIES (MRR < 1.0 or False Positives) ===")
    for d in per_query_details:
        if d["mrr"] < 1.0 or (not d["expected"] and len(d["retrieved"]) > 0):
            print(f"\n[{d['id']}] ({d['category']}) Query: \"{d['query']}\"")
            print(f"  Expected: {d['expected']}")
            print(f"  Retrieved: {d['retrieved']}")
            print(f"  MRR: {d['mrr']:.4f} | P@1: {d['p1']:.4f}")


if __name__ == "__main__":
    run_retrieval_audit(threshold=0.65)
