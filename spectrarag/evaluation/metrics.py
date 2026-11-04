"""Quantitative evaluation metrics implementing Ragas & TREC retrieval standards."""

import math
import re
from typing import List, Dict, Any, Set


def calculate_faithfulness(answer: str, source_contexts: List[str]) -> float:
    """Computes factual consistency score between generated claims and source text.
    
    Returns float in range [0.0, 1.0].
    """
    if not source_contexts:
        return 0.0 if answer.strip() else 1.0

    combined_sources = " ".join([s.lower() for s in source_contexts])
    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not claims:
        return 1.0

    supported_count = 0.0
    for claim in claims:
        # Extract content words (min len 3, skip stopwords)
        words = [w.lower() for w in re.findall(r"\b\w{3,}\b", claim) if w.lower() not in {"this", "that", "with", "from", "based", "according", "verified"}]
        if not words:
            supported_count += 1.0
            continue
        matches = sum(1 for w in words if w in combined_sources)
        ratio = matches / len(words)
        if ratio >= 0.65:
            supported_count += 1.0
        elif ratio >= 0.40:
            supported_count += 0.5

    return round(supported_count / len(claims), 3)


def calculate_mrr(retrieved_doc_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Mean Reciprocal Rank (MRR@k)."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in ground_truth_ids:
            return round(1.0 / rank, 4)
    return 0.0


def calculate_context_recall(retrieved_doc_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Context Recall@k."""
    if not ground_truth_ids:
        return 1.0
    retrieved_set = set(retrieved_doc_ids)
    hits = len(retrieved_set.intersection(ground_truth_ids))
    return round(hits / len(ground_truth_ids), 3)


def calculate_ndcg(retrieved_doc_ids: List[str], relevance_scores: Dict[str, float], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain (NDCG@k)."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)

    # Ideal DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)

    if idcg <= 1e-8:
        return 0.0
    return round(dcg / idcg, 4)


def calculate_answer_relevance(query: str, answer: str) -> float:
    """Measures semantic alignment between query intent and generated answer."""
    q_tokens = set([w.lower() for w in re.findall(r"\b\w{3,}\b", query)])
    if not q_tokens:
        return 1.0
    ans_clean = answer.lower()
    matches = sum(1 for t in q_tokens if t in ans_clean)
    return round(min(1.0, (matches + 0.5) / len(q_tokens)), 3)
