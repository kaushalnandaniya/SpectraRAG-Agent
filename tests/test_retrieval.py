"""Unit tests for dense, sparse, and hybrid retrieval backends."""

import pytest
from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.retrieval.dense_index import DenseVectorIndex
from spectrarag.retrieval.sparse_bm25 import SparseBM25Retriever
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.schema import Document


@pytest.fixture
def indexed_retrievers():
    docs = [
        Document(id="d_defxv", text="DefXV AR glasses provide real-time speech-to-text on micro-OLED displays.", metadata={"type": "ar"}),
        Document(id="d_deskavr", text="DeskAVR is an interview simulator powered by photorealistic metahumans.", metadata={"type": "simulation"}),
        Document(id="d_llm", text="Speculative decoding accelerates language models via draft-model verification.", metadata={"type": "ai"}),
    ]
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=20, min_chunk_len=10)
    chunks = chunker.chunk_documents(docs)

    dense = DenseVectorIndex(embedding_dim=128)
    sparse = SparseBM25Retriever()
    hybrid = HybridRetriever(dense_index=dense, sparse_index=sparse, rrf_k=60)
    hybrid.index_chunks(chunks)

    return dense, sparse, hybrid


def test_dense_search(indexed_retrievers):
    dense, _, _ = indexed_retrievers
    hits = dense.search("AR glasses micro-OLED", top_k=2)
    assert len(hits) > 0
    top_chunk, score, rank = hits[0]
    assert rank == 1
    assert "glasses" in top_chunk.text or "ar" in top_chunk.text.lower()


def test_sparse_bm25_search(indexed_retrievers):
    _, sparse, _ = indexed_retrievers
    hits = sparse.search("metahumans interview simulator", top_k=2)
    assert len(hits) > 0
    top_chunk, score, rank = hits[0]
    assert "deskavr" in top_chunk.doc_id or "metahumans" in top_chunk.text.lower()


def test_hybrid_rrf_fusion(indexed_retrievers):
    _, _, hybrid = indexed_retrievers
    results = hybrid.retrieve("speculative decoding language models", fused_top_k=2)
    assert len(results) > 0
    top_res = results[0]
    assert top_res.rrf_score is not None
    assert top_res.rrf_score > 0.0
    assert "speculative" in top_res.text.lower()
