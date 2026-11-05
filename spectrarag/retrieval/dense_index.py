"""Dense vector index with cosine similarity and fast matrix operations."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from spectrarag.schema import Chunk, RetrievalResult
from spectrarag.config import default_config
import hashlib
import json
import os


class DenseVectorIndex:
    """In-memory & disk-persistable dense vector index.
    
    Uses L2-normalized high-dimensional representations for exact dot-product
    cosine similarity search with matrix broadcasting.
    """

    def __init__(self, embedding_dim: int = default_config.embedding_dim):
        self.embedding_dim = embedding_dim
        self.chunks: List[Chunk] = []
        self.chunk_map: Dict[str, Chunk] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None  # Shape: (N, D)

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Deterministic, rich feature embedding for local reproducible benchmarking.
        
        Maps n-grams, word stems, and positional semantic hashes into a
        reproducible D-dimensional unit hypersphere.
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        # Term and bi-gram hashing with positional decay
        for idx, word in enumerate(words):
            # Positional factor
            pos_weight = 1.0 / (1.0 + 0.05 * idx)

            # Word hash
            h1 = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            slot1 = h1 % self.embedding_dim
            sign1 = 1.0 if (h1 >> 1) % 2 == 0 else -1.0
            vec[slot1] += sign1 * 1.5 * pos_weight

            # Bi-gram hash
            if idx > 0:
                bigram = f"{words[idx-1]}_{word}"
                h2 = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                slot2 = h2 % self.embedding_dim
                sign2 = 1.0 if (h2 >> 1) % 2 == 0 else -1.0
                vec[slot2] += sign2 * 2.2 * pos_weight

        # L2 Normalization to unit length
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Indexes chunks and builds the embedding matrix."""
        if not chunks:
            return

        new_embeddings = []
        for c in chunks:
            if c.embedding is None or len(c.embedding) != self.embedding_dim:
                emb = self._compute_embedding(c.text)
                c.embedding = emb.tolist()
            else:
                emb = np.array(c.embedding, dtype=np.float32)
            new_embeddings.append(emb)
            self.chunks.append(c)
            self.chunk_map[c.id] = c

        new_mat = np.vstack(new_embeddings)
        if self.embeddings_matrix is None:
            self.embeddings_matrix = new_mat
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_mat])

    def search(
        self,
        query: str,
        top_k: int = default_config.dense_top_k,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float, int]]:
        """Searches index for nearest neighbors by cosine similarity.
        
        Returns:
            List of (Chunk, similarity_score, rank_1_indexed)
        """
        if self.embeddings_matrix is None or len(self.chunks) == 0:
            return []

        q_vec = self._compute_embedding(query)  # Shape: (D,)
        # Compute cosine similarity via matrix-vector multiplication (N, D) @ (D,) -> (N,)
        scores = np.dot(self.embeddings_matrix, q_vec)

        # Sort indices descending
        sorted_indices = np.argsort(-scores)

        results: List[Tuple[Chunk, float, int]] = []
        rank = 1
        for idx in sorted_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])

            # Apply metadata filters if specified
            if filters:
                match = all(chunk.metadata.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            results.append((chunk, score, rank))
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: str) -> None:
        """Persists index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "chunks": [c.model_dump() for c in self.chunks],
            "embedding_dim": self.embedding_dim,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str) -> None:
        """Loads index from disk."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.embedding_dim = data["embedding_dim"]
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.chunk_map = {c.id: c for c in self.chunks}
        if self.chunks:
            embs = [np.array(c.embedding, dtype=np.float32) for c in self.chunks]
            self.embeddings_matrix = np.vstack(embs)

