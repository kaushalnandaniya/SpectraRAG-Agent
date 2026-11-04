"""Hybrid retriever fusing Dense Embeddings and Sparse BM25 via Reciprocal Rank Fusion (RRF)."""

from typing import List, Dict, Any, Optional
from spectrarag.schema import Chunk, RetrievalResult
from spectrarag.retrieval.dense_index import DenseVectorIndex
from spectrarag.retrieval.sparse_bm25 import SparseBM25Retriever
from spectrarag.config import default_config


class HybridRetriever:
    """Production hybrid retrieval system.
    
    Combines dense neural semantic search with sparse lexical BM25 matching.
    Balances precision on rare acronyms/keywords with semantic recall across paraphrases.
    """

    def __init__(
        self,
        dense_index: Optional[DenseVectorIndex] = None,
        sparse_index: Optional[SparseBM25Retriever] = None,
        rrf_k: int = default_config.rrf_k,
    ):
        self.dense_index = dense_index or DenseVectorIndex()
        self.sparse_index = sparse_index or SparseBM25Retriever()
        self.rrf_k = rrf_k
        self.chunk_registry: Dict[str, Chunk] = {}

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Indexes chunks across both dense and sparse backends."""
        for c in chunks:
            self.chunk_registry[c.id] = c
        self.dense_index.index_chunks(chunks)
        self.sparse_index.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        dense_top_k: int = default_config.dense_top_k,
        sparse_top_k: int = default_config.sparse_top_k,
        fused_top_k: int = 10,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Executes dual retrieval and fuses candidate pools via Reciprocal Rank Fusion.
        
        Formula:
            RRF_Score(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
        """
        dense_hits = self.dense_index.search(query, top_k=dense_top_k, filters=filters)
        sparse_hits = self.sparse_index.search(query, top_k=sparse_top_k, filters=filters)

        rrf_scores: Dict[str, float] = {}
        dense_meta: Dict[str, Tuple[float, int]] = {}   # chunk_id -> (score, rank)
        sparse_meta: Dict[str, Tuple[float, int]] = {}  # chunk_id -> (score, rank)

        for chunk, score, rank in dense_hits:
            dense_meta[chunk.id] = (score, rank)
            rrf_contrib = dense_weight / (self.rrf_k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + rrf_contrib

        for chunk, score, rank in sparse_hits:
            sparse_meta[chunk.id] = (score, rank)
            rrf_contrib = sparse_weight / (self.rrf_k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + rrf_contrib

        # Rank all combined chunks by RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for chunk_id, rrf_score in sorted_candidates[:fused_top_k]:
            chunk = self.chunk_registry[chunk_id]
            d_score, d_rank = dense_meta.get(chunk_id, (None, None))
            s_score, s_rank = sparse_meta.get(chunk_id, (None, None))

            result = RetrievalResult(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=round(rrf_score, 6),
                dense_rank=d_rank,
                sparse_rank=s_rank,
                dense_score=round(d_score, 4) if d_score is not None else None,
                sparse_score=round(s_score, 4) if s_score is not None else None,
                rrf_score=round(rrf_score, 6),
                metadata=chunk.metadata,
            )
            results.append(result)

        return results
