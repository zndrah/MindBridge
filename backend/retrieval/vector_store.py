"""
MindBridge 2.0 - FAISS Dense Vector Store
==========================================

Grounded vector store implementation leveraging FAISS IndexFlatIP for
exact inner product (cosine similarity on unit L2-normalized vectors).

Key Guarantees:
1. 384-dimensional dense vectors.
2. 1-to-1 strict positional alignment between FAISS internal vector IDs (0..N-1)
   and metadata store.
3. Persistent storage via native `faiss.write_index` and JSON metadata.
   Untrusted pickle deserialization is strictly prohibited.
4. Rigorous integrity validation upon loading: checks vector counts, dimensions,
   and SHA-256 checksums to reject corrupt or mismatched stores.
5. Deduplication: Rejects duplicate chunks by SHA-256 content hash.
6. Safe empty handling: Returns empty result lists cleanly without crashing.
7. Proposed starting threshold: tau = 0.65 (configurable; not a clinically validated boundary).
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, Union

import faiss
import numpy as np

DEFAULT_VECTOR_DIMENSION = 384
PROPOSED_RETRIEVAL_THRESHOLD = 0.65
INDEX_FILE_NAME = "index.faiss"
METADATA_FILE_NAME = "metadata.json"
METADATA_SCHEMA_VERSION = "1.0.0"


class IndexIntegrityError(Exception):
    """Raised when FAISS index or companion metadata fails integrity or consistency checks."""
    pass


class FAISSVectorStore:
    """
    In-memory and persistent FAISS dense vector store.
    """

    def __init__(self, dimension: int = DEFAULT_VECTOR_DIMENSION):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []
        self._seen_content_hashes: Set[str] = set()

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors stored in the FAISS index."""
        return self.index.ntotal

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add chunks and their corresponding embeddings to the vector store.
        Maintains 1-to-1 correspondence and filters duplicate chunks by content_hash.

        Args:
            chunks: List of chunk metadata dictionaries.
            embeddings: Numpy array of shape (N, dimension) with unit L2 normalized vectors.

        Returns:
            Number of newly added chunks.
        """
        if len(chunks) == 0:
            return 0

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings shape {embeddings.shape} does not match expected (N, {self.dimension})"
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided, but embeddings has {embeddings.shape[0]} rows"
            )

        valid_chunks: List[Dict[str, Any]] = []
        valid_indices: List[int] = []

        for idx, chunk in enumerate(chunks):
            c_hash = chunk.get("content_hash")
            if not c_hash:
                # Compute SHA-256 if not provided
                content = chunk.get("content", "")
                c_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                chunk["content_hash"] = c_hash

            if c_hash in self._seen_content_hashes:
                continue

            self._seen_content_hashes.add(c_hash)
            valid_chunks.append(chunk)
            valid_indices.append(idx)

        if not valid_chunks:
            return 0

        valid_embeddings = np.ascontiguousarray(
            embeddings[valid_indices], dtype=np.float32
        )
        # Unit L2 normalize vectors before adding to index
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        valid_embeddings = valid_embeddings / norms

        # Add to FAISS index
        self.index.add(valid_embeddings)

        # Add to metadata store
        self.metadata_store.extend(valid_chunks)

        # Post-condition verification
        if self.index.ntotal != len(self.metadata_store):
            raise IndexIntegrityError(
                f"Internal alignment desynchronization: index.ntotal={self.index.ntotal} != len(metadata_store)={len(self.metadata_store)}"
            )

        return len(valid_chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = PROPOSED_RETRIEVAL_THRESHOLD,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search vector store for top-k nearest chunks above the similarity threshold.

        Args:
            query_vector: 1D or 2D array of shape (dimension,) or (1, dimension).
            top_k: Maximum number of chunks to return.
            threshold: Minimum cosine similarity score (faiss inner product).

        Returns:
            List of (chunk_dict, score) tuples sorted descending by score.
        """
        if self.total_vectors == 0:
            return []

        # Ensure correct shape and dtype
        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)

        if q.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {q.shape[1]} does not match index dimension {self.dimension}"
            )

        # Normalize query vector before searching
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        if np.all(q_norm == 0):
            return []
        q_norm = np.where(q_norm == 0, 1e-12, q_norm)
        q = q / q_norm

        # Clamp top_k to total available vectors
        search_k = min(top_k, self.total_vectors)
        if search_k <= 0:
            return []

        scores, indices = self.index.search(q, search_k)

        results: List[Tuple[Dict[str, Any], float]] = []
        target_thresh = round(threshold, 6)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            # Round score to 6 decimal places to eliminate IEEE 754 float32 precision
            # representation anomalies (e.g. np.float32(0.65) == 0.649999976... in float64)
            f_score = round(float(score), 6)
            if f_score >= target_thresh:
                chunk = dict(self.metadata_store[idx])
                results.append((chunk, f_score))

        return results

    def save(self, directory: Union[str, Path]) -> Tuple[Path, Path]:
        """
        Persist FAISS index binary and companion metadata JSON to disk using atomic replacement.
        Computes SHA-256 of the FAISS binary to guarantee tamper detection on load.

        Returns:
            Tuple of (index_path, metadata_path)
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_path = dir_path / INDEX_FILE_NAME
        metadata_path = dir_path / METADATA_FILE_NAME

        tmp_index_path = dir_path / f"{INDEX_FILE_NAME}.tmp_{os.getpid()}"
        tmp_metadata_path = dir_path / f"{METADATA_FILE_NAME}.tmp_{os.getpid()}"

        try:
            # 1. Write FAISS binary to temporary file
            faiss.write_index(self.index, str(tmp_index_path))

            # 2. Compute SHA-256 checksum of saved index
            with open(tmp_index_path, "rb") as f:
                index_sha256 = hashlib.sha256(f.read()).hexdigest()

            # 3. Write metadata JSON to temporary file
            content_hashes = [
                chunk.get("content_hash", "") for chunk in self.metadata_store
            ]
            meta_payload = {
                "schema_version": METADATA_SCHEMA_VERSION,
                "dimension": self.dimension,
                "total_vectors": self.total_vectors,
                "index_sha256": index_sha256,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content_hashes": content_hashes,
                "metadata": self.metadata_store,
            }

            with open(tmp_metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2, ensure_ascii=False)

            # 4. Atomic file replacement
            os.replace(tmp_index_path, index_path)
            os.replace(tmp_metadata_path, metadata_path)
        except Exception:
            if tmp_index_path.exists():
                try:
                    tmp_index_path.unlink()
                except OSError:
                    pass
            if tmp_metadata_path.exists():
                try:
                    tmp_metadata_path.unlink()
                except OSError:
                    pass
            raise

        return index_path, metadata_path

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        expected_dimension: int = DEFAULT_VECTOR_DIMENSION,
    ) -> "FAISSVectorStore":
        """
        Load FAISS index and companion metadata from disk with strict integrity verification.

        Validations:
        1. Both index.faiss and metadata.json must exist and be non-empty.
        2. metadata.json must be valid JSON with matching schema.
        3. Index dimension must match expected_dimension.
        4. index.ntotal must exactly equal metadata count.
        5. index.faiss SHA-256 must match stored index_sha256.
        """
        dir_path = Path(directory)
        index_path = dir_path / INDEX_FILE_NAME
        metadata_path = dir_path / METADATA_FILE_NAME

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {index_path}"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        if index_path.stat().st_size == 0:
            raise IndexIntegrityError(f"FAISS index file is empty (0 bytes): {index_path}")
        if metadata_path.stat().st_size == 0:
            raise IndexIntegrityError(f"Metadata file is empty (0 bytes): {metadata_path}")

        # 1. Read metadata JSON safely
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
        except Exception as e:
            raise IndexIntegrityError(
                f"Corrupt or unreadable metadata JSON: {type(e).__name__}: {str(e)}"
            ) from e

        # Validate metadata schema fields
        for field in ("dimension", "total_vectors", "index_sha256", "metadata"):
            if field not in meta_payload:
                raise IndexIntegrityError(
                    f"Metadata JSON missing required field: '{field}'"
                )

        metadata_list = meta_payload["metadata"]
        if not isinstance(metadata_list, list):
            raise IndexIntegrityError("Metadata field must be a list of chunk records")

        stored_dim = meta_payload["dimension"]
        if stored_dim != expected_dimension:
            raise IndexIntegrityError(
                f"Dimension mismatch in metadata: stored={stored_dim}, expected={expected_dimension}"
            )

        # 2. Verify index file SHA-256 checksum
        with open(index_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()

        if actual_sha256 != meta_payload["index_sha256"]:
            raise IndexIntegrityError(
                f"Index checksum mismatch: index file may be corrupted or tampered with. Stored: {meta_payload['index_sha256'][:16]}..., Actual: {actual_sha256[:16]}..."
            )

        # 3. Read FAISS index binary
        try:
            index = faiss.read_index(str(index_path))
        except Exception as e:
            raise IndexIntegrityError(
                f"Failed to read FAISS index binary: {type(e).__name__}: {str(e)}"
            ) from e

        # 4. Validate FAISS internal properties
        if index.d != expected_dimension:
            raise IndexIntegrityError(
                f"FAISS index internal dimension {index.d} does not match expected {expected_dimension}"
            )

        if index.ntotal != len(metadata_list):
            raise IndexIntegrityError(
                f"Vector count mismatch: index.ntotal={index.ntotal} != len(metadata)={len(metadata_list)}"
            )

        if meta_payload["total_vectors"] != index.ntotal:
            raise IndexIntegrityError(
                f"Metadata recorded vector count {meta_payload['total_vectors']} != index.ntotal {index.ntotal}"
            )

        # 5. Reconstruct store instance
        store = cls(dimension=expected_dimension)
        store.index = index
        store.metadata_store = metadata_list
        store._seen_content_hashes = {
            chunk.get("content_hash", "")
            for chunk in metadata_list
            if chunk.get("content_hash")
        }

        return store
