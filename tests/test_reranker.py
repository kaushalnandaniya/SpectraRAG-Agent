"""Unit tests for Cross-Encoder re-ranker and pruning."""

import pytest
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.schema import RetrievalResult


def test_reranker_scoring_and_pruning():
    reranker = CrossEncoderReranker(top_k=2, confidence_threshold=0.50)

    candidates = [
        RetrievalResult(chunk_id="c1", doc_id="d1", text="Unrelated recipe for baking chocolate sourdough bread.", score=0.01),
        RetrievalResult(chunk_id="c2", doc_id="d2", text="DefXV AR glasses feature micro-OLED transparent displays.", score=0.02),
        RetrievalResult(chunk_id="c3", doc_id="d3", text="DefXV has 18ms latency speech-to-text transcription.", score=0.03),
    ]

    reranked = reranker.rerank(
        query="DefXV AR glasses display latency",
        candidates=candidates,
        top_k=2,
        prune_below_threshold=True,
    )

    assert len(reranked) <= 2
    # The unrelated cooking recipe should be pruned or ranked lowest
    top_ids = [r.chunk_id for r in reranked]
    assert "c1" not in top_ids or reranked[0].chunk_id != "c1"
    assert reranked[0].rerank_score is not None
