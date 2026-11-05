"""Unit tests for quantitative evaluation metrics."""

import pytest
from spectrarag.evaluation.metrics import (
    calculate_faithfulness,
    calculate_mrr,
    calculate_context_recall,
    calculate_ndcg,
    calculate_answer_relevance,
)


def test_faithfulness_metric():
    sources = ["DefXV uses micro-OLED displays with 18ms latency."]
    ans_grounded = "DefXV utilizes micro-OLED displays with 18ms latency."
    score_grounded = calculate_faithfulness(ans_grounded, sources)
    assert score_grounded >= 0.80

    ans_hallucinated = "DefXV utilizes quantum laser holograms and operates on Mars."
    score_hallucinated = calculate_faithfulness(ans_hallucinated, sources)
    assert score_hallucinated < 0.50


def test_mrr_metric():
    retrieved = ["doc_c", "doc_a", "doc_b"]
    gt = {"doc_a"}
    mrr = calculate_mrr(retrieved, gt)
    assert mrr == 0.5  # 1 / rank 2


def test_recall_metric():
    retrieved = ["doc_a", "doc_b"]
    gt = {"doc_a", "doc_b", "doc_c"}
    recall = calculate_context_recall(retrieved, gt)
    assert recall == round(2 / 3, 3)


def test_answer_relevance():
    query = "What is the latency of DefXV?"
    ans = "The latency of DefXV is sub-18 milliseconds."
    rel = calculate_answer_relevance(query, ans)
    assert rel >= 0.70

