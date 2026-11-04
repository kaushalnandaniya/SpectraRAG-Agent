"""Sparse lexical retriever using Okapi BM25 with inverse document frequency caching."""

import math
import re
from typing import List, Dict, Tuple, Optional, Any
from spectrarag.schema import Chunk
from spectrarag.config import default_config


class SparseBM25Retriever:
    """Okapi BM25 sparse keyword retriever.
    
    Parameters:
        k1: Controls term frequency saturation (default 1.5)
        b: Controls document length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.chunk_map: Dict[str, Chunk] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(chunk_idx, tf)]
        self.idf_cache: Dict[str, float] = {}
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple, fast lowercased alpha-numeric tokenizer."""
        return [w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Builds BM25 inverted index over provided chunks."""
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        for c in chunks:
            self.chunk_map[c.id] = c

        total_tokens = sum(self.doc_lengths)
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            tokens = self._tokenize(chunk.text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            # Compute term frequencies in this chunk
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, count in tf_map.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, count))

        self.num_docs = len(self.chunks)
        self.avg_doc_len = total_tokens / max(1, self.num_docs)

        # Precompute and cache IDFs
        for term, df in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF formula
            self.idf_cache[term] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = default_config.sparse_top_k,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float, int]]:
        """Scores chunks against query using Okapi BM25 formula.
        
        Returns:
            List of (Chunk, bm25_score, rank_1_indexed)
        """
        if self.num_docs == 0:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores: Dict[int, float] = {}

        for term in q_tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf_cache.get(term, 0.0)
            postings = self.inverted_index[term]

            for chunk_idx, tf in postings:
                d_len = self.doc_lengths[chunk_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(1e-5, self.avg_doc_len)))
                term_score = idf * (numerator / denominator)
                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + term_score

        # Sort descending by score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: List[Tuple[Chunk, float, int]] = []
        rank = 1
        for idx, score in sorted_chunks:
            chunk = self.chunks[idx]
            if filters:
                match = all(chunk.metadata.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            results.append((chunk, float(score), rank))
            rank += 1
            if len(results) >= top_k:
                break

        return results
