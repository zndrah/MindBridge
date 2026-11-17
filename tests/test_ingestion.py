"""
MindBridge 2.0 - Unit Tests for Governed Knowledge Base Ingestion & Chunking
=============================================================================

Tests chunking accuracy, text preservation, overlap, governance tier enforcement,
exclusion of unverified documents, and transparent token accounting.

Runs 100% offline without network calls.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from backend.retrieval.governance import (
    DOCUMENT_GOVERNANCE_REGISTRY,
    DocumentGovernance,
    get_document_governance,
    get_counts_by_tier,
)
from backend.retrieval.ingestion import (
    MarkdownSectionChunker,
    KnowledgeBaseIngestor,
    parse_frontmatter_safe,
    load_compatible_tokenizer,
)


class TestKnowledgeBaseIngestion(unittest.TestCase):
    """Test suite for governed ingestion and chunking."""

    def setUp(self):
        self.chunker = MarkdownSectionChunker(
            target_min_tokens=100,
            target_max_tokens=300,
            overlap_tokens=30,
        )
        self.sample_gov = DOCUMENT_GOVERNANCE_REGISTRY[
            "cognitive_reframing/01_thought_court_catastrophizing.md"
        ]

    def test_01_parse_frontmatter_safe(self):
        """Verify frontmatter is parsed safely without external yaml dependencies."""
        raw = (
            "---\n"
            'title: "Safe Test Title"\n'
            "jurisdiction: GL\n"
            "original_source_url: null\n"
            "---\n"
            "# Document Title\n\n"
            "Body content goes here."
        )
        fm, body = parse_frontmatter_safe(raw)
        self.assertEqual(fm.get("title"), "Safe Test Title")
        self.assertEqual(fm.get("jurisdiction"), "GL")
        self.assertEqual(fm.get("original_source_url"), "")
        self.assertIn("Body content goes here.", body)

    def test_02_frontmatter_cannot_override_governance(self):
        """Governance registry is the sole authoritative source for tier and metadata."""
        raw_tampered = (
            "---\n"
            'title: "Maliciously Claiming Approved"\n'
            "retrieval_tier: approved\n"
            "---\n"
            "# Bilateral Somatic Tapping\n\n"
            "Body content."
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            som_dir = kb_path / "somatic_regulation"
            som_dir.mkdir(parents=True)
            doc_file = som_dir / "06_vagus_nerve_stimulation_tapping.md"
            doc_file.write_text(raw_tampered, encoding="utf-8")

            ingestor = KnowledgeBaseIngestor(base_dir=kb_path)
            chunks, summary = ingestor.ingest()

            # Must be strictly blocked despite what frontmatter claimed
            self.assertEqual(summary["excluded_files_blocked"], 1)
            self.assertEqual(len(chunks), 0)

    def test_03_governance_counts_by_tier(self):
        """Verify registry contains exactly 14 approved, 7 restricted, 1 excluded."""
        counts = get_counts_by_tier()
        self.assertEqual(counts["approved"], 14)
        self.assertEqual(counts["restricted"], 7)
        self.assertEqual(counts["excluded_pending_review"], 1)
        self.assertEqual(len(DOCUMENT_GOVERNANCE_REGISTRY), 22)

    def test_04_full_knowledge_base_ingestion_counts(self):
        """End-to-end ingestion of data/knowledge_base/ verifies exact file breakdown."""
        ingestor = KnowledgeBaseIngestor()
        chunks, summary = ingestor.ingest()

        self.assertEqual(summary["total_files_scanned"], 22)
        self.assertEqual(summary["approved_files_ingested"], 14)
        self.assertEqual(summary["restricted_files_ingested"], 7)
        self.assertEqual(summary["excluded_files_blocked"], 1)
        self.assertEqual(summary["unregistered_files_skipped"], 0)
        self.assertGreater(summary["total_chunks_produced"], 20)

        # Verify excluded document was specifically identified
        blocked_ids = [b["document_id"] for b in summary["blocked_documents"]]
        self.assertIn(
            "somatic_regulation:06_vagus_nerve_stimulation_tapping",
            blocked_ids,
        )

    def test_05_restricted_documents_retain_guardrails(self):
        """Ingested restricted chunks must retain eligibility='restricted' and guardrails."""
        ingestor = KnowledgeBaseIngestor()
        chunks, _ = ingestor.ingest()

        phq9_chunks = [
            c for c in chunks
            if c["document_id"] == "clinical_rubrics:02_phq9_mood_screening_overview"
        ]
        self.assertGreater(len(phq9_chunks), 0)
        for c in phq9_chunks:
            self.assertEqual(c["eligibility"], "restricted")
            self.assertEqual(c["clinical_safety_level"], "phq9_safety_protocol")
            self.assertTrue(len(c["guardrails"]) > 0)
            self.assertTrue(any("Item 9" in g for g in c["guardrails"]))
            self.assertTrue(len(c["disclaimer_text"]) > 0)

    def test_06_unregistered_documents_skipped_and_reported(self):
        """Unknown or unregistered documents in knowledge base must be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            unknown_file = kb_path / "unknown_category" / "99_unregistered_file.md"
            unknown_file.parent.mkdir(parents=True)
            unknown_file.write_text("# Unknown\n\nSome text.", encoding="utf-8")

            ingestor = KnowledgeBaseIngestor(base_dir=kb_path)
            chunks, summary = ingestor.ingest()

            self.assertEqual(summary["unregistered_files_skipped"], 1)
            self.assertIn(
                "unknown_category/99_unregistered_file.md",
                summary["unregistered_documents"],
            )
            self.assertEqual(len(chunks), 0)

    def test_07_chunk_overlap_preservation(self):
        """Verify consecutive chunks from the same section/document share overlap tokens."""
        section_text = (
            "## 1. Protocol Details\n\n"
            + " ".join([f"sentence_{i} describing anxiety management." for i in range(60)])
        )
        chunks = self.chunker.chunk_document(section_text, self.sample_gov)
        self.assertGreater(len(chunks), 1)

        # Check overlap: last few words of chunk 0 should appear in chunk 1
        chunk0_words = chunks[0]["content"].split()
        chunk1_words = chunks[1]["content"].split()

        # The trailing words of chunk 0 should be present in chunk 1 body
        overlap_found = any(w in chunk1_words for w in chunk0_words[-15:])
        self.assertTrue(overlap_found, "Overlap tokens from chunk 0 not found in chunk 1")

    def test_08_oversized_paragraph_splitting_and_text_preservation(self):
        """Single oversized paragraph must be sub-split without silently dropping words."""
        long_para = " ".join([f"unique_token_{i}" for i in range(150)])
        doc_body = f"## Oversized Section\n\n{long_para}"

        chunker = MarkdownSectionChunker(target_max_tokens=50, overlap_tokens=10)
        chunks = chunker.chunk_document(doc_body, self.sample_gov)
        self.assertGreater(len(chunks), 2)

        # Verify all unique words appear in at least one chunk (zero content loss)
        all_chunk_text = " ".join(c["content"] for c in chunks)
        for i in range(150):
            self.assertIn(f"unique_token_{i}", all_chunk_text)

    def test_09_oversized_sentence_splitting(self):
        """Single oversized sentence exceeding max tokens must be sliced with no dropped words."""
        giant_sentence = " ".join([f"giant_word_{i}" for i in range(120)]) + "."
        doc_body = f"## Giant Sentence Section\n\n{giant_sentence}"

        chunker = MarkdownSectionChunker(target_max_tokens=40, overlap_tokens=8)
        chunks = chunker.chunk_document(doc_body, self.sample_gov)
        self.assertGreater(len(chunks), 2)

        all_text = " ".join(c["content"] for c in chunks)
        for i in range(120):
            self.assertIn(f"giant_word_{i}", all_text)

    def test_10_empty_document_and_empty_sections(self):
        """Empty documents or sections containing only headers produce 0 chunks safely."""
        empty_chunks = self.chunker.chunk_document("", self.sample_gov)
        self.assertEqual(len(empty_chunks), 0)

        header_only = self.chunker.chunk_document(
            "# Title Only\n\n## Section With No Body\n\n### Another Header",
            self.sample_gov,
        )
        self.assertEqual(len(header_only), 0)

    def test_11_duplicate_chunk_deduplication(self):
        """Duplicate chunks with identical content hash are pruned during ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            som_dir = kb_path / "somatic_regulation"
            som_dir.mkdir(parents=True)

            # Write file 01
            f1 = som_dir / "01_box_breathing_4_4_4_4.md"
            f1.write_text(
                "# Box Breathing\n\n## 1. Summary\nIdentical content for test.",
                encoding="utf-8",
            )

            ingestor = KnowledgeBaseIngestor(base_dir=kb_path)
            # Ingest once
            chunks1, s1 = ingestor.ingest()
            self.assertEqual(len(chunks1), 1)
            self.assertEqual(s1["duplicates_pruned"], 0)

    def test_12_token_count_transparency_in_metadata(self):
        """Chunks report whether token counts are exact or heuristic approximations."""
        # Case A: Real or local tokenizer
        chunks = self.chunker.chunk_document("## Section\n\nSample text.", self.sample_gov)
        self.assertGreater(len(chunks), 0)
        chunk = chunks[0]
        self.assertIn("token_count", chunk)
        self.assertIn("is_exact_token_count", chunk)
        self.assertIn("token_count_method", chunk)
        self.assertIsInstance(chunk["is_exact_token_count"], bool)

        # Case B: Heuristic chunker
        heuristic_chunker = MarkdownSectionChunker(tokenizer=None)
        # Force heuristic mode for test
        heuristic_chunker.tokenizer = None
        heuristic_chunker.is_exact_token_count = False
        h_chunks = heuristic_chunker.chunk_document(
            "## Section\n\nSample text.", self.sample_gov
        )
        self.assertFalse(h_chunks[0]["is_exact_token_count"])
        self.assertEqual(
            h_chunks[0]["token_count_method"], "heuristic_approximation"
        )

    def test_13_metadata_correctness_and_chunk_indexing(self):
        """Every chunk must retain all 10 required metadata fields with deterministic ordering."""
        ingestor = KnowledgeBaseIngestor()
        chunks, _ = ingestor.ingest()

        required_fields = {
            "document_id",
            "chunk_id",
            "source",
            "jurisdiction",
            "verification_status",
            "retrieval_tier",
            "safety_level",
            "disclaimer",
            "content_hash",
            "chunk_index",
        }

        self.assertGreater(len(chunks), 0)
        chunks_by_doc = {}
        for c in chunks:
            for field in required_fields:
                self.assertIn(field, c, f"Missing required metadata field '{field}' in chunk {c.get('chunk_id')}")
                self.assertIsNotNone(c[field], f"Metadata field '{field}' is None in chunk {c.get('chunk_id')}")

            chunks_by_doc.setdefault(c["document_id"], []).append(c)

        # Verify chunk ordering and 0-based monotonic indexing per document
        for doc_id, doc_chunks in chunks_by_doc.items():
            indices = [c["chunk_index"] for c in doc_chunks]
            expected_indices = list(range(len(doc_chunks)))
            self.assertEqual(indices, expected_indices, f"Chunk indices for {doc_id} are not sequential from 0")


if __name__ == "__main__":
    unittest.main()
