"""
MindBridge 2.0 - Dense Vector Embeddings Engine (BGE-small-en-v1.5)
===================================================================

CPU-compatible, deterministic dense embedding generation based on BAAI/bge-small-en-v1.5.

Architecture & Conventions:
- Embedding dimension: 384
- Inference device: CPU (torch.no_grad)
- Normalization: Unit L2 normalization (cosine similarity via inner product)
- Query instruction prefix: "Represent this sentence for searching relevant passages: "
- Passage / document encoding: Direct text without prefix
- Lazy loading: Importing this module does NOT load or download any model
- Offline Mock Engine: Dedicated MockEmbeddingEngine for deterministic offline testing
- Privacy: All computation is strictly local; user mental-health messages are NEVER sent externally
"""

import re
import hashlib
from pathlib import Path
from typing import List, Optional, Any, Union
import numpy as np

EMBEDDING_DIMENSION = 384
DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LOCAL_BGE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "embeddings"
    / "bge-small-en-v1.5"
)


class EmbeddingError(Exception):
    """Raised when embedding model loading or inference fails."""
    pass


class BGEEmbeddingEngine:
    """
    Local BGE-small-en-v1.5 embedding engine.
    Adheres to lazy model loading and CPU execution.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_BGE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self.model_name_or_path = model_name_or_path
        self._model = model
        self._tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.dimension = EMBEDDING_DIMENSION
        self._is_loaded = model is not None and tokenizer is not None

    @property
    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are loaded in memory."""
        return self._is_loaded

    @property
    def tokenizer(self) -> Optional[Any]:
        """Return the loaded tokenizer, or None if not loaded."""
        return self._tokenizer

    def load(self) -> None:
        """
        Explicitly load tokenizer and model onto CPU.
        Prefers local models/embeddings/bge-small-en-v1.5 if present.
        Raises EmbeddingError if loading fails.
        """
        if self._is_loaded:
            return

        target_path = self.model_name_or_path
        local_files_only = False
        if target_path == DEFAULT_BGE_MODEL and LOCAL_BGE_DIR.exists():
            target_path = str(LOCAL_BGE_DIR)
            local_files_only = True
        elif Path(target_path).exists():
            local_files_only = True

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            kwargs = {"local_files_only": True} if local_files_only else {}
            self._tokenizer = AutoTokenizer.from_pretrained(
                target_path, **kwargs
            )
            self._model = AutoModel.from_pretrained(
                target_path, **kwargs
            )
            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load BGE embedding model '{self.model_name_or_path}': {type(e).__name__}: {str(e)}. "
                "Ensure local model weights are present, provide a local filesystem path, "
                "or explicitly inject a model/tokenizer or MockEmbeddingEngine for offline testing."
            ) from e

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate L2-normalized 384-dimensional embeddings for a list of document texts.
        Passages are encoded WITHOUT any query prefix.
        """
        if texts is None:
            raise TypeError("texts cannot be None; must be a list of strings")
        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"texts must be a list or tuple of strings, got {type(texts).__name__}")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be strings")

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        self._ensure_loaded()
        import torch

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    model_output = self._model(**encoded)
                    # CLS token pooling (standard for BGE models)
                    cls_representation = model_output[0][:, 0]
                    # Unit L2 normalization
                    norm_embeddings = torch.nn.functional.normalize(
                        cls_representation, p=2, dim=1
                    )
                    batch_vecs = norm_embeddings.cpu().numpy().astype(np.float32)
                    all_embeddings.append(batch_vecs)
            except Exception as e:
                raise EmbeddingError(
                    f"Inference failure in batch {i // self.batch_size}: {type(e).__name__}: {str(e)}"
                ) from e

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized 384-dimensional embedding for a query.
        Applies BGE query instruction prefix.
        """
        if query is None:
            return np.zeros((self.dimension,), dtype=np.float32)
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        if not query.strip():
            # Return zero vector for empty query
            return np.zeros((self.dimension,), dtype=np.float32)

        prefixed_query = f"{self.query_prefix}{query.strip()}"
        vectors = self.embed_texts([prefixed_query])
        return vectors[0]

    # Alias for compatibility with LangChain / standard embedding interfaces
    embed_documents = embed_texts


class MockEmbeddingEngine:
    """
    Deterministic, offline mock embedding engine for tests and hermetic environments.
    Guarantees 384-dimensional, unit L2-normalized vectors without downloading weights.
    """

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self.dimension = dimension
        self.query_prefix = query_prefix
        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def _word_vector(self, word: str) -> np.ndarray:
        """Map individual word to a deterministic pseudo-random unit vector."""
        digest = hashlib.sha256(word.lower().encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        raw_vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(raw_vec)
        if norm < 1e-12:
            raw_vec[0] = 1.0
            norm = 1.0
        return raw_vec / norm

    def _hash_text_to_vector(self, text: str) -> np.ndarray:
        """
        Deterministically map text to a reproducible 384-dim unit vector.
        Uses normalized sum of word vectors so that lexical/semantic overlap
        produces realistic cosine similarity in offline hermetic tests.
        """
        # Strip query prefix if present so queries align with passages
        clean_text = text
        if clean_text.startswith(self.query_prefix):
            clean_text = clean_text[len(self.query_prefix):]

        words = re.findall(r"\w+", clean_text.lower())
        if not words:
            # Deterministic vector from raw string if no word tokens found
            digest = hashlib.sha256(clean_text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], byteorder="big")
            rng = np.random.RandomState(seed)
            raw_vec = rng.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(raw_vec)
            if norm < 1e-12:
                raw_vec[0] = 1.0
                norm = 1.0
            return (raw_vec / norm).astype(np.float32)

        composite = sum(self._word_vector(w) for w in words)
        norm = np.linalg.norm(composite)
        if norm < 1e-12:
            composite[0] = 1.0
            norm = 1.0
        return (composite / norm).astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic unit L2 vectors for passages."""
        if texts is None:
            raise TypeError("texts cannot be None; must be a list of strings")
        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"texts must be a list or tuple of strings, got {type(texts).__name__}")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be strings")

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = [self._hash_text_to_vector(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate deterministic unit L2 vector for a query with prefix."""
        if query is None:
            return np.zeros((self.dimension,), dtype=np.float32)
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        if not query.strip():
            return np.zeros((self.dimension,), dtype=np.float32)

        prefixed_query = f"{self.query_prefix}{query.strip()}"
        return self._hash_text_to_vector(prefixed_query)

    # Alias for compatibility with LangChain / standard embedding interfaces
    embed_documents = embed_texts


def get_default_embedding_engine() -> BGEEmbeddingEngine:
    """
    Factory returning the system-default embedding engine (BGEEmbeddingEngine).
    MockEmbeddingEngine is NEVER returned here and can only be used by explicit injection.
    """
    return BGEEmbeddingEngine()
