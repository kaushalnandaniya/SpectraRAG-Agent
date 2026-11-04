"""Automated benchmark runner executing multi-tier evaluation and ablation studies."""

import json
import time
import os
from typing import Dict, Any, List, Optional
import numpy as np

from spectrarag.schema import Document, BenchmarkSummary
from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.retrieval.dense_index import DenseVectorIndex
from spectrarag.retrieval.sparse_bm25 import SparseBM25Retriever
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.agent.graph import SpectraAgentEngine
from spectrarag.evaluation.metrics import (
    calculate_faithfulness,
    calculate_mrr,
    calculate_context_recall,
    calculate_answer_relevance,
)
from spectrarag.evaluation.latency_profiler import LatencyProfiler
from spectrarag.config import default_config


class BenchmarkRunner:
    """Executes full evaluation suite comparing SpectraRAG against baseline architectures."""

    def __init__(
        self,
        docs_path: Optional[str] = None,
        queries_path: Optional[str] = None,
    ):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.docs_path = docs_path or os.path.join(base_dir, "data", "sample_docs.json")
        self.queries_path = queries_path or os.path.join(base_dir, "data", "benchmark_queries.json")

        self.chunker = SemanticChunker()
        self.dense_index = DenseVectorIndex()
        self.sparse_index = SparseBM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            dense_index=self.dense_index,
            sparse_index=self.sparse_index,
        )
        self.reranker = CrossEncoderReranker()
        self.agent_engine = SpectraAgentEngine(
            retriever=self.hybrid_retriever,
            reranker=self.reranker,
        )
        self.profiler = LatencyProfiler()
        self._is_indexed = False

    def load_and_index(self) -> int:
        """Loads corpus and indexes chunks across vector and BM25 tables."""
        with open(self.docs_path, "r") as f:
            raw_docs = json.load(f)
        docs = [Document(**d) for d in raw_docs]
        chunks = self.chunker.chunk_documents(docs)
        self.hybrid_retriever.index_chunks(chunks)
        self._is_indexed = True
        return len(chunks)

    def run_benchmark(self, max_queries: Optional[int] = None) -> BenchmarkSummary:
        """Runs evaluation over the benchmark queries and calculates system metrics."""
        if not self._is_indexed:
            self.load_and_index()

        with open(self.queries_path, "r") as f:
            queries = json.load(f)

        if max_queries is not None:
            queries = queries[:max_queries]

        faithfulness_scores: List[float] = []
        mrr_scores: List[float] = []
        recall_scores: List[float] = []
        relevance_scores: List[float] = []
        hallucination_flags: List[int] = []
        total_costs: List[float] = []

        # Ablation tracking
        dense_mrr: List[float] = []
        sparse_mrr: List[float] = []
        hybrid_mrr: List[float] = []

        baseline_token_count = 0
        pruned_token_count = 0

        for item in queries:
            q_text = item["query"]
            gt_doc_ids = set(item.get("ground_truth_doc_ids", []))

            # Run ablations
            d_hits = self.dense_index.search(q_text, top_k=5)
            s_hits = self.sparse_index.search(q_text, top_k=5)
            h_hits = self.hybrid_retriever.retrieve(q_text, fused_top_k=5)

            dense_mrr.append(calculate_mrr([c.doc_id for c, _, _ in d_hits], gt_doc_ids))
            sparse_mrr.append(calculate_mrr([c.doc_id for c, _, _ in s_hits], gt_doc_ids))
            hybrid_mrr.append(calculate_mrr([r.doc_id for r in h_hits], gt_doc_ids))

            # Track token counts for pruning savings
            baseline_tokens = sum(len(r.text.split()) for r in h_hits)
            baseline_token_count += baseline_tokens

            # Execute full SpectraRAG Agent
            t0 = time.perf_counter()
            resp = self.agent_engine.execute(q_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.profiler.record("e2e", elapsed_ms)

            retrieved_doc_ids = [s.doc_id for s in resp.sources]
            pruned_tokens = sum(len(s.text.split()) for s in resp.sources)
            pruned_token_count += pruned_tokens

            # Calculate individual scores
            f_score = calculate_faithfulness(resp.answer, [s.text for s in resp.sources])
            m_score = calculate_mrr(retrieved_doc_ids, gt_doc_ids)
            r_score = calculate_context_recall(retrieved_doc_ids, gt_doc_ids)
            rel_score = calculate_answer_relevance(q_text, resp.answer)

            faithfulness_scores.append(f_score)
            mrr_scores.append(m_score)
            recall_scores.append(r_score)
            relevance_scores.append(rel_score)
            hallucination_flags.append(1 if f_score < default_config.faithfulness_threshold else 0)
            total_costs.append(resp.cost_usd)

        lat_summary = self.profiler.summary("e2e")
        hallucination_rate = round(float(np.mean(hallucination_flags)) * 100.0, 1)

        token_savings_pct = 0.0
        if baseline_token_count > 0:
            token_savings_pct = round(((baseline_token_count - pruned_token_count) / baseline_token_count) * 100.0, 1)

        ablation_summary = {
            "Dense_Only_MRR": round(float(np.mean(dense_mrr)), 4),
            "Sparse_BM25_Only_MRR": round(float(np.mean(sparse_mrr)), 4),
            "Hybrid_RRF_MRR": round(float(np.mean(hybrid_mrr)), 4),
            "SpectraRAG_Final_MRR": round(float(np.mean(mrr_scores)), 4),
            "MRR_Improvement_vs_Dense": f"+{round(((np.mean(mrr_scores) - np.mean(dense_mrr)) / max(1e-5, np.mean(dense_mrr))) * 100.0, 1)}%",
            "MRR_Improvement_vs_BM25": f"+{round(((np.mean(mrr_scores) - np.mean(sparse_mrr)) / max(1e-5, np.mean(sparse_mrr))) * 100.0, 1)}%",
        }

        return BenchmarkSummary(
            total_queries=len(queries),
            faithfulness=round(float(np.mean(faithfulness_scores)), 3),
            context_precision_mrr=round(float(np.mean(mrr_scores)), 3),
            context_recall=round(float(np.mean(recall_scores)), 3),
            answer_relevance=round(float(np.mean(relevance_scores)), 3),
            hallucination_rate_pct=hallucination_rate,
            p50_latency_ms=lat_summary["p50"],
            p95_latency_ms=lat_summary["p95"],
            p99_latency_ms=lat_summary["p99"],
            avg_cost_per_query_usd=round(float(np.mean(total_costs)), 6),
            token_savings_pct=token_savings_pct,
            ablation_results=ablation_summary,
        )
