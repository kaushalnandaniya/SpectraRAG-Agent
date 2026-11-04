"""Unit tests for LangGraph multi-agent state graph transitions."""

import pytest
from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.retrieval.dense_index import DenseVectorIndex
from spectrarag.retrieval.sparse_bm25 import SparseBM25Retriever
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.agent.graph import SpectraAgentEngine
from spectrarag.schema import Document


@pytest.fixture
def agent_engine():
    docs = [
        Document(
            id="d_defxv",
            text="DefXV uses micro-OLED displays with sub-18ms latency and quad-microphone beamforming arrays.",
            metadata={"category": "hardware"},
        ),
        Document(
            id="d_h2o",
            text="H2O KV-cache eviction drops 40% memory by retaining heavy-hitter tokens.",
            metadata={"category": "llm"},
        ),
    ]
    chunker = SemanticChunker(chunk_size=150, chunk_overlap=20, min_chunk_len=10)
    chunks = chunker.chunk_documents(docs)

    dense = DenseVectorIndex(embedding_dim=128)
    sparse = SparseBM25Retriever()
    hybrid = HybridRetriever(dense_index=dense, sparse_index=sparse)
    hybrid.index_chunks(chunks)

    reranker = CrossEncoderReranker(top_k=3, confidence_threshold=0.40)
    return SpectraAgentEngine(retriever=hybrid, reranker=reranker)


def test_agent_execution_trace(agent_engine):
    resp = agent_engine.execute("What display and latency does DefXV use?")

    assert resp.answer != ""
    assert len(resp.sources) > 0
    assert resp.faithfulness_score >= 0.0
    assert resp.latency_ms > 0.0
    assert len(resp.agent_steps) >= 5

    # Check node presence in traces
    trace_text = " ".join(resp.agent_steps)
    assert "[Planner]" in trace_text
    assert "[Retriever]" in trace_text
    assert "[ReRanker]" in trace_text
    assert "[Synthesizer]" in trace_text
    assert "[Auditor]" in trace_text
