"""
MindBridge 2.0 - Unit Tests for Dense Vector Embeddings Engine
==============================================================

Tests BGE-small-en-v1.5 engine configuration, lazy loading, L2 normalization,
dimension verification (384), batching, query prefix conventions, error handling,
and deterministic mock engine behavior.

Runs 100% offline without downloading model weights.
"""

import unittest
from typing import List
import numpy as np
import torch
import torch.nn as nn

from backend.retrieval.embeddings import (
    BGEEmbeddingEngine,
    MockEmbeddingEngine,
    EmbeddingError,
    EMBEDDING_DIMENSION,
    BGE_QUERY_PREFIX,
    get_default_embedding_engine,
)


class DummyHFModel(nn.Module):
    """Offline dummy PyTorch model outputting 384-dim unnormalized vectors for testing."""

    def __init__(self, dim: int = EMBEDDING_DIMENSION):
        super().__init__()
        self.dim = dim

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 8

        hidden = torch.zeros((batch_size, seq_len, self.dim), dtype=torch.float32)
        for b in range(batch_size):
            base_val = float(input_ids[b].sum().item()) if input_ids is not None else 1.0
            for d in range(self.dim):
                hidden[b, :, d] = base_val + (d + 1) * 0.1

        # HF models return a tuple/ModelOutput where index 0 is last_hidden_state
        return (hidden,)


class DummyTokenizer:
    """Offline dummy tokenizer simulating HuggingFace tokenizer interface."""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        # Generate dummy input_ids
        max_len = 16
        input_ids = []
        attention_mask = []
        for t in texts:
            # Deterministic token IDs from characters
            ids = [ord(c) % 1000 + 1 for c in t[:max_len]]
            pad_len = max_len - len(ids)
            ids = ids + [0] * pad_len
            mask = [1] * min(len(t), max_len) + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TestEmbeddingsEngine(unittest.TestCase):
    """Test suite for BGE and Mock embedding engines."""

    def test_01_lazy_loading_prevents_unprompted_downloads(self):
        """Instantiation of BGEEmbeddingEngine must NOT trigger download or load."""
        engine = BGEEmbeddingEngine(model_name_or_path="non_existent_bge_model_path")
        self.assertFalse(engine.is_loaded)
        self.assertEqual(engine.dimension, 384)

    def test_02_load_failure_raises_actionable_error(self):
        """Engine raises EmbeddingError if weights cannot be loaded; never silently switches to mock."""
        engine = BGEEmbeddingEngine(model_name_or_path="/invalid/non_existent_path")
        with self.assertRaises(EmbeddingError) as ctx:
            engine.load()
        self.assertIn("Failed to load BGE embedding model", str(ctx.exception))
        self.assertFalse(engine.is_loaded)

    def test_03_injected_model_produces_384_dim_l2_normalized_vectors(self):
        """Verifies 384-dimensional output and strict L2 normalization using injected fake model."""
        dummy_model = DummyHFModel(dim=384)
        dummy_tokenizer = DummyTokenizer()

        engine = BGEEmbeddingEngine(
            model=dummy_model,
            tokenizer=dummy_tokenizer,
            batch_size=2,
        )
        self.assertTrue(engine.is_loaded)

        passages = [
            "Box breathing parasympathetic nervous system protocol.",
            "Cognitive restructuring and thought challenging.",
            "Stimulus control therapy for insomnia.",
        ]

        embeddings = engine.embed_texts(passages)

        # 1. Shape check: (3, 384)
        self.assertEqual(embeddings.shape, (3, 384))
        self.assertEqual(embeddings.dtype, np.float32)

        # 2. Strict L2 normalization: norm must equal 1.0 within 1e-5
        for i in range(len(passages)):
            norm = np.linalg.norm(embeddings[i])
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_04_batch_inference_consistency(self):
        """Batch embedding produces identical results regardless of batch_size."""
        dummy_model = DummyHFModel(dim=384)
        dummy_tokenizer = DummyTokenizer()

        texts = [f"Psychoeducational concept turn {i}" for i in range(5)]

        engine_batch1 = BGEEmbeddingEngine(
            model=dummy_model, tokenizer=dummy_tokenizer, batch_size=1
        )
        engine_batch4 = BGEEmbeddingEngine(
            model=dummy_model, tokenizer=dummy_tokenizer, batch_size=4
        )

        res1 = engine_batch1.embed_texts(texts)
        res4 = engine_batch4.embed_texts(texts)

        np.testing.assert_allclose(res1, res4, rtol=1e-5, atol=1e-5)

    def test_05_bge_query_prefix_convention(self):
        """Query embedding prepends BGE query instruction; passage embedding does not."""
        dummy_model = DummyHFModel(dim=384)

        captured_inputs = []

        class RecordingTokenizer:
            def __call__(self, texts, **kwargs):
                captured_inputs.extend(texts)
                return DummyTokenizer()(texts, **kwargs)

        rec_tok = RecordingTokenizer()
        engine = BGEEmbeddingEngine(model=dummy_model, tokenizer=rec_tok)

        # Query embedding
        engine.embed_query("How to handle panic?")
        self.assertTrue(any(t.startswith(BGE_QUERY_PREFIX) for t in captured_inputs))

        # Passage embedding
        captured_inputs.clear()
        engine.embed_texts(["Direct passage without prefix."])
        self.assertFalse(any(t.startswith(BGE_QUERY_PREFIX) for t in captured_inputs))
        self.assertEqual(captured_inputs[0], "Direct passage without prefix.")

    def test_06_mock_embedding_engine_determinism_and_normalization(self):
        """MockEmbeddingEngine produces reproducible unit L2 vectors offline."""
        mock = MockEmbeddingEngine(dimension=384)
        self.assertTrue(mock.is_loaded)

        v1 = mock.embed_texts(["Stress reduction exercise"])
        v2 = mock.embed_texts(["Stress reduction exercise"])
        v3 = mock.embed_texts(["Completely different topic"])

        # Deterministic reproducibility
        np.testing.assert_array_equal(v1, v2)

        # Distinct text produces distinct vector
        self.assertFalse(np.allclose(v1, v3))

        # L2 norm check
        self.assertAlmostEqual(np.linalg.norm(v1[0]), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(v3[0]), 1.0, places=5)

        # Query embedding norm
        qv = mock.embed_query("How do I breathe?")
        self.assertEqual(qv.shape, (384,))
        self.assertAlmostEqual(np.linalg.norm(qv), 1.0, places=5)

    def test_07_empty_inputs_handling(self):
        """Empty texts or empty query strings are handled safely without exceptions."""
        mock = MockEmbeddingEngine(dimension=384)

        empty_passages = mock.embed_texts([])
        self.assertEqual(empty_passages.shape, (0, 384))

        empty_query = mock.embed_query("")
        self.assertEqual(empty_query.shape, (384,))
        np.testing.assert_array_equal(empty_query, np.zeros(384, dtype=np.float32))

        whitespace_query = mock.embed_query("   ")
        self.assertEqual(whitespace_query.shape, (384,))
        np.testing.assert_array_equal(whitespace_query, np.zeros(384, dtype=np.float32))

    def test_08_invalid_input_handling(self):
        """Invalid types for embed_texts and embed_query raise TypeError."""
        mock = MockEmbeddingEngine(dimension=384)

        with self.assertRaises(TypeError):
            mock.embed_texts(None)

        with self.assertRaises(TypeError):
            mock.embed_texts("not a list")

        with self.assertRaises(TypeError):
            mock.embed_texts(["valid text", 12345])

        with self.assertRaises(TypeError):
            mock.embed_query(12345)

        # Same for BGE engine with dummy model
        dummy_model = DummyHFModel(dim=384)
        dummy_tokenizer = DummyTokenizer()
        bge_engine = BGEEmbeddingEngine(model=dummy_model, tokenizer=dummy_tokenizer)

        with self.assertRaises(TypeError):
            bge_engine.embed_texts(None)

        with self.assertRaises(TypeError):
            bge_engine.embed_texts("not a list")

        with self.assertRaises(TypeError):
            bge_engine.embed_texts(["valid", None])

        with self.assertRaises(TypeError):
            bge_engine.embed_query(12345)

    def test_09_default_engine_is_bge_not_mock(self):
        """Default application engine is strictly BGEEmbeddingEngine; Mock requires explicit injection."""
        default_engine = get_default_embedding_engine()
        self.assertIsInstance(default_engine, BGEEmbeddingEngine)
        self.assertNotIsInstance(default_engine, MockEmbeddingEngine)
        self.assertFalse(default_engine.is_loaded)

    def test_10_bge_failure_never_falls_back_to_mock(self):
        """When BGE fails to load, it raises EmbeddingError and never silently swaps in Mock."""
        engine = BGEEmbeddingEngine(model_name_or_path="/completely/fake/path/bge")
        self.assertFalse(engine.is_loaded)

        with self.assertRaises(EmbeddingError) as ctx:
            engine.embed_texts(["Test text requiring inference"])
        self.assertIn("Failed to load BGE embedding model", str(ctx.exception))
        # Ensure it didn't secretly replace itself with mock
        self.assertNotIsInstance(engine, MockEmbeddingEngine)
        self.assertFalse(engine.is_loaded)


if __name__ == "__main__":
    unittest.main()
