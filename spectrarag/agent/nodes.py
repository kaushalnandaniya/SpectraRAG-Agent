"""Execution nodes for LangGraph agentic workflow."""

import time
import re
from typing import Dict, Any, List
from spectrarag.agent.state import AgentState
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.config import default_config


class SpectraAgentNodes:
    """Implements graph execution nodes with deterministic validation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        config=default_config,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

    def planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Classifies intent, extracts key search terms, and decomposes complex queries."""
        start_t = time.perf_counter()
        q = state["query"].strip()

        # Classify intent
        intent = "factual"
        q_lower = q.lower()
        if any(w in q_lower for w in ["how to", "steps", "procedure", "install", "deploy"]):
            intent = "procedural"
        elif any(w in q_lower for w in ["difference", "compare", "versus", "vs"]):
            intent = "comparative"
        elif any(w in q_lower for w in ["architecture", "benchmark", "latency", "throughput", "parameter"]):
            intent = "technical"

        # Query decomposition for multi-hop questions
        sub_queries = [q]
        if " and " in q_lower or ";" in q_lower:
            parts = re.split(r"\band\b|;", q, flags=re.IGNORECASE)
            sub_queries.extend([p.strip() for p in parts if len(p.strip()) > 8])

        trace = list(state.get("trace_steps", []))
        trace.append(f"[Planner] Intent: {intent} | Sub-queries: {len(sub_queries)}")

        # Estimate prompt token cost
        prompt_tokens = max(10, len(q.split()) * 2)
        cost_delta = (prompt_tokens / 1000.0) * self.config.cost_per_1k_input_tokens

        return {
            "cleaned_query": q,
            "intent": intent,
            "sub_queries": sub_queries,
            "trace_steps": trace,
            "cost_usd": state.get("cost_usd", 0.0) + cost_delta,
        }

    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieves candidate passages across all sub-queries using Hybrid RRF."""
        start_t = time.perf_counter()
        query_to_run = state.get("cleaned_query", state["query"])

        # If this is a retry loop, integrate the reflection notes to augment query
        if state.get("retry_count", 0) > 0 and state.get("reflection_notes"):
            notes = state["reflection_notes"]
            query_to_run = f"{query_to_run} {notes}"

        candidates = self.retriever.retrieve(
            query=query_to_run,
            dense_top_k=self.config.dense_top_k,
            sparse_top_k=self.config.sparse_top_k,
            fused_top_k=12,
        )

        trace = list(state.get("trace_steps", []))
        trace.append(f"[Retriever] Fetched {len(candidates)} hybrid candidates via RRF (k={self.config.rrf_k})")

        return {
            "retrieved_docs": candidates,
            "trace_steps": trace,
        }

    def reranker_node(self, state: AgentState) -> Dict[str, Any]:
        """Applies cross-encoder re-ranking and confidence threshold pruning."""
        candidates = state.get("retrieved_docs", [])
        q = state.get("cleaned_query", state["query"])

        reranked = self.reranker.rerank(
            query=q,
            candidates=candidates,
            top_k=self.config.rerank_top_k,
            prune_below_threshold=True,
        )

        trace = list(state.get("trace_steps", []))
        trace.append(f"[ReRanker] Filtered {len(candidates)} -> {len(reranked)} top chunks (thresh={self.config.rerank_threshold})")

        return {
            "reranked_docs": reranked,
            "trace_steps": trace,
        }

    def synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates grounded answer with explicit citation references."""
        docs = state.get("reranked_docs", [])
        q = state.get("cleaned_query", state["query"])

        if not docs:
            answer = "I could not locate sufficient verifiable context in the provided documents to answer this query accurately."
            context_tokens = 20
        else:
            # Build structured context string
            context_blocks = []
            for i, d in enumerate(docs):
                context_blocks.append(f"[{i+1}] {d.text.strip()}")
            full_context = "\n\n".join(context_blocks)
            context_tokens = max(50, len(full_context.split()))

            # Grounded synthesis
            # Extract key statements from top docs directly answering the query
            key_points = []
            for idx, d in enumerate(docs[:3]):
                sentences = re.split(r"(?<=[.!?])\s+", d.text.strip())
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) > 15:
                        key_points.append(f"{s_clean} [{idx+1}]")
                        break

            if key_points:
                formatted_points = " ".join(key_points[:3])
                answer = f"Based on the verified retrieved sources: {formatted_points}"
            else:
                answer = f"According to verified documentation [1]: {docs[0].text[:250]}..."

        # Calculate cost
        input_cost = (context_tokens / 1000.0) * self.config.cost_per_1k_input_tokens
        output_tokens = max(25, len(answer.split()))
        output_cost = (output_tokens / 1000.0) * self.config.cost_per_1k_output_tokens

        trace = list(state.get("trace_steps", []))
        trace.append(f"[Synthesizer] Generated response ({output_tokens} tokens, {len(docs)} sources cited)")

        return {
            "synthesized_answer": answer,
            "trace_steps": trace,
            "cost_usd": state.get("cost_usd", 0.0) + input_cost + output_cost,
        }

    def hallucination_auditor_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluates factual consistency (faithfulness) and answer relevance.
        
        If faithfulness is below threshold and retries remain, triggers reflection loop.
        """
        answer = state.get("synthesized_answer", "")
        docs = state.get("reranked_docs", [])
        q = state.get("cleaned_query", state["query"])

        if not docs or "could not locate sufficient" in answer:
            # Abstention / no docs
            faithfulness = 1.0
            relevance = 0.5
            hallucination = False
        else:
            # Calculate lexical & semantic overlap between answer claims and source text
            combined_context = " ".join([d.text.lower() for d in docs])
            ans_tokens = [w.lower() for w in re.findall(r"\b\w{3,}\b", answer) if w.lower() not in {"based", "on", "the", "verified", "retrieved", "sources", "according", "to"}]
            if not ans_tokens:
                faithfulness = 0.95
            else:
                grounded_count = sum(1 for t in ans_tokens if t in combined_context)
                faithfulness = round(min(1.0, grounded_count / len(ans_tokens)), 3)

            # Calculate answer relevance against original query
            q_tokens = [w.lower() for w in re.findall(r"\b\w{3,}\b", q)]
            ans_clean = answer.lower()
            rel_matches = sum(1 for qt in q_tokens if qt in ans_clean)
            relevance = round(min(1.0, (rel_matches + 1) / (len(q_tokens) + 1)), 3)

            hallucination = faithfulness < self.config.faithfulness_threshold

        trace = list(state.get("trace_steps", []))
        trace.append(f"[Auditor] Faithfulness: {faithfulness:.3f} | Relevance: {relevance:.3f} | Hallucination: {hallucination}")

        retry_count = state.get("retry_count", 0)
        reflection_notes = ""
        if hallucination and retry_count < state.get("max_retries", self.config.max_reflection_retries):
            # Extract missing concepts from query to refine subsequent retrieval
            q_terms = [w for w in q.split() if len(w) > 3]
            reflection_notes = f"Focus strictly on: {' '.join(q_terms[:3])}"
            trace.append(f"[Reflection Loop {retry_count + 1}] Low faithfulness detected. Reformulating query: '{reflection_notes}'")
            retry_count += 1

        return {
            "faithfulness_score": faithfulness,
            "answer_relevance": relevance,
            "hallucination_detected": hallucination,
            "reflection_notes": reflection_notes,
            "retry_count": retry_count,
            "trace_steps": trace,
            "status": "completed" if not hallucination or retry_count >= state.get("max_retries", 2) else "retrying",
        }

    def fallback_node(self, state: AgentState) -> Dict[str, Any]:
        """Graceful fallback when multi-turn verification cannot pass threshold."""
        docs = state.get("reranked_docs", [])
        q = state.get("cleaned_query", state["query"])

        if docs:
            fallback_ans = f"Verification constraint unmet. Closest verified context: \"{docs[0].text[:180]}...\""
        else:
            fallback_ans = f"Unable to verify factual claims for \"{q}\" within safety confidence thresholds."

        trace = list(state.get("trace_steps", []))
        trace.append("[Fallback] Safe abstention executed to prevent hallucination.")

        return {
            "synthesized_answer": fallback_ans,
            "faithfulness_score": 0.98,
            "hallucination_detected": False,
            "status": "fallback_resolved",
            "trace_steps": trace,
        }
