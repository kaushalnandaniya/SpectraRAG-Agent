"""Cross-Encoder re-ranker with confidence thresholding and context pruning."""

import re
import math
from typing import List, Tuple, Optional
from spectrarag.schema import RetrievalResult
from spectrarag.config import default_config


class CrossEncoderReranker:
    """Simulates a neural cross-encoder (e.g. bge-reranker-large / ms-marco-MiniLM)
    
    Computes fine-grained query-document cross-attention interaction scores:
    1. Lexical phrase match density and contiguous spans
    2. Exact named entity and acronym preservation
    3. Inverse length penalty to prevent bloated, uninformative passages
    4. Confidence threshold filtering to prune noisy distractors
    """

    def __init__(
        self,
        top_k: int = default_config.rerank_top_k,
        confidence_threshold: float = default_config.rerank_threshold,
    ):
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    def _score_interaction(self, query: str, passage: str) -> float:
        """Computes cross-attention interaction score in range [0.0, 1.0]."""
        q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", query)]
        p_tokens = [w.lower() for w in re.findall(r"\b\w+\b", passage)]

        if not q_tokens or not p_tokens:
            return 0.0

        p_set = set(p_tokens)
        matched_tokens = [t for t in q_tokens if t in p_set]
        token_coverage = len(matched_tokens) / len(q_tokens)

        # Contiguous phrase / n-gram match bonus
        q_clean = query.lower()
        p_clean = passage.lower()
        phrase_bonus = 0.0
        if len(q_clean) > 8 and q_clean in p_clean:
            phrase_bonus = 0.35
        else:
            # Check for 3-gram matches
            trigrams = [
                " ".join(q_tokens[i : i + 3])
                for i in range(len(q_tokens) - 2)
            ]
            if trigrams:
                matched_trigrams = sum(1 for tg in trigrams if tg in p_clean)
                phrase_bonus = 0.25 * (matched_trigrams / len(trigrams))

        # Passage density / informativeness factor
        density = len(matched_tokens) / max(10, len(p_tokens))
        density_score = min(1.0, density * 5.0)

        # Nonlinear sigmoid-like scaling
        raw_score = 0.45 * token_coverage + 0.35 * phrase_bonus + 0.20 * density_score
        normalized_score = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.35)))

        return float(min(1.0, max(0.0, normalized_score)))

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
        prune_below_threshold: bool = True,
    ) -> List[RetrievalResult]:
        """Reranks candidates and prunes chunks failing the confidence threshold."""
        target_k = top_k or self.top_k
        scored_candidates: List[Tuple[RetrievalResult, float]] = []

        for item in candidates:
            cross_score = self._score_interaction(query, item.text)
            item.rerank_score = round(cross_score, 4)
            scored_candidates.append((item, cross_score))

        # Sort descending by re-rank score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for item, score in scored_candidates:
            if prune_below_threshold and score < self.confidence_threshold:
                continue
            item.score = score
            results.append(item)
            if len(results) >= target_k:
                break

        # Fallback: if all pruned, return top 1 candidate to avoid total dropout
        if not results and scored_candidates:
            fallback_item = scored_candidates[0][0]
            fallback_item.score = scored_candidates[0][1]
            results.append(fallback_item)

        return results
