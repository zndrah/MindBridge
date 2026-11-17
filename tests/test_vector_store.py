"""
MindBridge 2.0 - Unit Tests for FAISS Dense Vector Store
=========================================================

Tests FAISS IndexFlatIP index construction, 1-to-1 vector-metadata alignment,
strict tamper/integrity verification, duplicate rejection, and empty handling.

Runs 100% offline without network calls.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from backend.retrieval.vector_store import (
    FAISSVectorStore,
    IndexIntegrityError,
    PROPOSED_RETRIEVAL_THRESHOLD,
    DEFAULT_VECTOR_DIMENSION,
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
)


class TestFAISSVectorStore(unittest.TestCase):
    """Test suite for FAISS vector store operations and integrity."""

    def setUp(self):
        self.store = FAISSVectorStore(dimension=DEFAULT_VECTOR_DIMENSION)

    def _generate_unit_vectors(self, n: int, dim: int = DEFAULT_VECTOR_DIMENSION) -> np.ndarray:
        vecs = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_01_index_initialization(self):
        """Vector store initializes with empty FAISS index and 384 dimensions."""
        self.assertEqual(self.store.dimension, 384)
        self.assertEqual(self.store.total_vectors, 0)
        self.assertEqual(len(self.store.metadata_store), 0)

    def test_02_add_chunks_strict_1to1_alignment(self):
        """Verifies 1-to-1 positional correspondence between FAISS vectors and metadata list."""
        chunks = [
            {"chunk_id": f"chunk_{i}", "title": f"Doc {i}", "content": f"Text {i}", "content_hash": f"hash_{i}"}
            for i in range(5)
        ]
        embeddings = self._generate_unit_vectors(5)

        added = self.store.add_chunks(chunks, embeddings)
        self.assertEqual(added, 5)
        self.assertEqual(self.store.total_vectors, 5)
        self.assertEqual(len(self.store.metadata_store), 5)

        for i in range(5):
            self.assertEqual(self.store.metadata_store[i]["chunk_id"], f"chunk_{i}")

    def test_03_dimension_and_count_mismatch_rejection(self):
        """Rejects embeddings with mismatched dimensions or count mismatch."""
        chunks = [{"chunk_id": "c1", "content": "text"}]

        # Dimension mismatch (128 != 384)
        wrong_dim_vecs = np.random.randn(1, 128).astype(np.float32)
        with self.assertRaises(ValueError):
            self.store.add_chunks(chunks, wrong_dim_vecs)

        # Count mismatch (1 chunk vs 3 vectors)
        vecs_3 = self._generate_unit_vectors(3)
        with self.assertRaises(ValueError):
            self.store.add_chunks(chunks, vecs_3)

    def test_04_duplicate_chunk_deduplication(self):
        """Duplicate chunks with identical content_hash are not inserted into FAISS."""
        chunks = [
            {"chunk_id": "c1", "content": "text", "content_hash": "hash_dup"},
            {"chunk_id": "c2", "content": "text", "content_hash": "hash_dup"},
        ]
        vecs = self._generate_unit_vectors(2)

        added = self.store.add_chunks(chunks, vecs)
        self.assertEqual(added, 1)
        self.assertEqual(self.store.total_vectors, 1)

        # Re-adding identical chunk
        re_added = self.store.add_chunks([chunks[0]], self._generate_unit_vectors(1))
        self.assertEqual(re_added, 0)
        self.assertEqual(self.store.total_vectors, 1)

    def test_05_similarity_search_and_threshold_filtering(self):
        """Verifies cosine similarity ranking and threshold filtering."""
        vecs = self._generate_unit_vectors(3)
        chunks = [
            {"chunk_id": "c0", "title": "Zero", "content_hash": "h0"},
            {"chunk_id": "c1", "title": "One", "content_hash": "h1"},
            {"chunk_id": "c2", "title": "Two", "content_hash": "h2"},
        ]
        self.store.add_chunks(chunks, vecs)

        # Query using vector identical to chunk 0; threshold=-1.0 ensures all are returned
        results = self.store.search(vecs[0], top_k=3, threshold=-1.0)
        self.assertEqual(len(results), 3)
        # Top result must be c0 with score ~1.0
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk["chunk_id"], "c0")
        self.assertAlmostEqual(top_score, 1.0, places=4)

        # Strict threshold test: score >= 0.99 must return only c0
        strict_results = self.store.search(vecs[0], top_k=3, threshold=0.99)
        self.assertEqual(len(strict_results), 1)
        self.assertEqual(strict_results[0][0]["chunk_id"], "c0")

    def test_06_save_and_load_persistence_roundtrip(self):
        """Verifies full roundtrip persistence preserves index, metadata, and checksums."""
        vecs = self._generate_unit_vectors(4)
        chunks = [
            {"chunk_id": f"chunk_{i}", "title": f"Doc {i}", "content_hash": f"hash_{i}"}
            for i in range(4)
        ]
        self.store.add_chunks(chunks, vecs)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path, meta_path = self.store.save(tmpdir)
            self.assertTrue(idx_path.exists())
            self.assertTrue(meta_path.exists())

            # Load into fresh store
            loaded_store = FAISSVectorStore.load(tmpdir)
            self.assertEqual(loaded_store.total_vectors, 4)
            self.assertEqual(loaded_store.dimension, 384)
            self.assertEqual(len(loaded_store.metadata_store), 4)

            # Metadata ordering preserved
            for i in range(4):
                self.assertEqual(loaded_store.metadata_store[i]["chunk_id"], f"chunk_{i}")

            # Verify search produces identical top score
            res = loaded_store.search(vecs[0], top_k=1, threshold=0.5)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0][0]["chunk_id"], "chunk_0")

    def test_07_tampered_index_checksum_detection(self):
        """Corrupting the FAISS index binary raises IndexIntegrityError."""
        vecs = self._generate_unit_vectors(2)
        chunks = [{"chunk_id": f"c_{i}", "content_hash": f"h_{i}"} for i in range(2)]
        self.store.add_chunks(chunks, vecs)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            idx_file = Path(tmpdir) / INDEX_FILE_NAME

            # Tamper with index binary by appending corrupt bytes
            with open(idx_file, "ab") as f:
                f.write(b"CORRUPT_BYTES_INJECTION")

            with self.assertRaises(IndexIntegrityError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("Index checksum mismatch", str(ctx.exception))

    def test_08_tampered_metadata_count_mismatch_detection(self):
        """Tampering with vector counts in metadata JSON raises IndexIntegrityError."""
        vecs = self._generate_unit_vectors(2)
        chunks = [{"chunk_id": f"c_{i}", "content_hash": f"h_{i}"} for i in range(2)]
        self.store.add_chunks(chunks, vecs)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            meta_file = Path(tmpdir) / METADATA_FILE_NAME

            with open(meta_file, "r") as f:
                meta_json = json.load(f)
            meta_json["total_vectors"] = 999  # Tamper count
            with open(meta_file, "w") as f:
                json.dump(meta_json, f)

            with self.assertRaises(IndexIntegrityError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("vector count", str(ctx.exception).lower())

    def test_09_missing_persistence_files_raise_filenotfound(self):
        """Missing index or metadata files raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                FAISSVectorStore.load(tmpdir)

    def test_10_empty_index_and_empty_query_safe_handling(self):
        """Empty vector store returns empty list cleanly without crashing."""
        dummy_query = np.zeros(384, dtype=np.float32)
        results = self.store.search(dummy_query, top_k=5, threshold=0.5)
        self.assertEqual(results, [])

    def test_11_unnormalized_vectors_are_normalized_before_adding_and_searching(self):
        """Unnormalized vectors passed to add_chunks or search are automatically normalized."""
        # Create unnormalized vector with norm = 100.0
        raw_vec = np.ones((1, 384), dtype=np.float32) * 5.0
        chunk = [{"chunk_id": "c_unnorm", "content_hash": "h_unnorm", "content": "text"}]

        added = self.store.add_chunks(chunk, raw_vec)
        self.assertEqual(added, 1)

        # Search with unnormalized query of same direction (norm = 50.0)
        query = np.ones((1, 384), dtype=np.float32) * 2.5
        results = self.store.search(query, top_k=1, threshold=0.99)
        self.assertEqual(len(results), 1)
        # Cosine similarity between identical directions is 1.0
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

    def test_12_empty_file_corruption_detection(self):
        """0-byte index or metadata files raise IndexIntegrityError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_file = Path(tmpdir) / INDEX_FILE_NAME
            meta_file = Path(tmpdir) / METADATA_FILE_NAME

            # 0-byte index file
            idx_file.touch()
            meta_file.write_text('{"dimension": 384, "total_vectors": 0, "index_sha256": "abc", "metadata": []}')

            with self.assertRaises(IndexIntegrityError):
                FAISSVectorStore.load(tmpdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_file = Path(tmpdir) / INDEX_FILE_NAME
            meta_file = Path(tmpdir) / METADATA_FILE_NAME

            idx_file.write_bytes(b"some faiss header")
            # 0-byte metadata file
            meta_file.touch()

            with self.assertRaises(IndexIntegrityError):
                FAISSVectorStore.load(tmpdir)

    def test_13_corrupted_metadata_json_detection(self):
        """Corrupt non-JSON or invalid schema in metadata file raises IndexIntegrityError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_file = Path(tmpdir) / INDEX_FILE_NAME
            meta_file = Path(tmpdir) / METADATA_FILE_NAME

            idx_file.write_bytes(b"dummy")
            meta_file.write_text("NOT_VALID_JSON{:::}")

            with self.assertRaises(IndexIntegrityError):
                FAISSVectorStore.load(tmpdir)

    def test_14_deterministic_threshold_boundaries_080_065_064_018(self):
        """
        Deterministic verification of threshold 0.65 boundary:
        - Score 0.80: accepted.
        - Score 0.65: accepted (inclusive comparison).
        - Score 0.64: rejected.
        - Score 0.18: rejected.
        """
        store = FAISSVectorStore(dimension=DEFAULT_VECTOR_DIMENSION)

        e0 = np.zeros(DEFAULT_VECTOR_DIMENSION, dtype=np.float32)
        e0[0] = 1.0
        e1 = np.zeros(DEFAULT_VECTOR_DIMENSION, dtype=np.float32)
        e1[1] = 1.0

        target_scores = [0.80, 0.65, 0.64, 0.18]
        chunks = []
        vectors = []

        for s in target_scores:
            v = np.float32(s) * e0 + np.float32(np.sqrt(1.0 - s**2)) * e1
            v = v / np.linalg.norm(v)
            cid = f"chunk_{s:.2f}"
            chunks.append({
                "chunk_id": cid,
                "document_id": f"doc_{s:.2f}",
                "title": f"Doc {s:.2f}",
                "content_hash": f"hash_{s:.2f}",
                "content": f"Content {s:.2f}",
            })
            vectors.append(v)

        store.add_chunks(chunks, np.vstack(vectors))

        # Query vector is exactly e0
        query_vector = e0.copy()

        # Search with exact proposed retrieval threshold = 0.65
        results = store.search(query_vector, top_k=10, threshold=PROPOSED_RETRIEVAL_THRESHOLD)
        retrieved_ids = [c["chunk_id"] for c, _ in results]

        # 0.80 must be accepted
        self.assertIn("chunk_0.80", retrieved_ids, "Score 0.80 must be accepted")

        # 0.65 must be accepted (inclusive comparison)
        self.assertIn("chunk_0.65", retrieved_ids, "Score 0.65 must be accepted (inclusive)")

        # 0.64 must be rejected
        self.assertNotIn("chunk_0.64", retrieved_ids, "Score 0.64 must be rejected")

        # 0.18 must be rejected
        self.assertNotIn("chunk_0.18", retrieved_ids, "Score 0.18 must be rejected")


if __name__ == "__main__":
    unittest.main()
