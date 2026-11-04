"""LangGraph multi-agent state graph orchestration for SpectraRAG."""

import time
import uuid
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from spectrarag.agent.state import AgentState
from spectrarag.agent.nodes import SpectraAgentNodes
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.schema import QueryResponse
from spectrarag.config import default_config, SpectraConfig


def should_continue(state: AgentState) -> str:
    """Conditional router following hallucination audit."""
    status = state.get("status", "completed")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    hallucination = state.get("hallucination_detected", False)

    if hallucination and retry_count < max_retries:
        return "retriever"
    elif hallucination and retry_count >= max_retries:
        return "fallback"
    return "end"


class SpectraAgentEngine:
    """Production Agentic RAG Engine compiled with LangGraph."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        config: SpectraConfig = default_config,
    ):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker(
            top_k=config.rerank_top_k,
            confidence_threshold=config.rerank_threshold,
        )
        self.config = config
        self.nodes = SpectraAgentNodes(
            retriever=self.retriever,
            reranker=self.reranker,
            config=self.config,
        )
        self.graph = self._compile_graph()

    def _compile_graph(self):
        """Assembles and compiles the cyclic StateGraph."""
        workflow = StateGraph(AgentState)

        # Register execution nodes
        workflow.add_node("planner", self.nodes.planner_node)
        workflow.add_node("retriever", self.nodes.retriever_node)
        workflow.add_node("reranker", self.nodes.reranker_node)
        workflow.add_node("synthesizer", self.nodes.synthesizer_node)
        workflow.add_node("auditor", self.nodes.hallucination_auditor_node)
        workflow.add_node("fallback", self.nodes.fallback_node)

        # Connect standard linear transitions
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("reranker", "synthesizer")
        workflow.add_edge("synthesizer", "auditor")

        # Conditional reflection / self-correction edge
        workflow.add_conditional_edges(
            "auditor",
            should_continue,
            {
                "retriever": "retriever",
                "fallback": "fallback",
                "end": END,
            },
        )
        workflow.add_edge("fallback", END)

        return workflow.compile()

    def execute(
        self,
        query: str,
        max_retries: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResponse:
        """Executes full agentic graph and returns structured QueryResponse."""
        start_time = time.perf_counter()
        trace_id = str(uuid.uuid4())[:12]

        initial_state: AgentState = {
            "query": query,
            "cleaned_query": query,
            "intent": "unknown",
            "sub_queries": [query],
            "retrieved_docs": [],
            "reranked_docs": [],
            "synthesized_answer": "",
            "faithfulness_score": 0.0,
            "answer_relevance": 0.0,
            "hallucination_detected": False,
            "reflection_notes": "",
            "retry_count": 0,
            "max_retries": max_retries if max_retries is not None else self.config.max_reflection_retries,
            "trace_steps": [f"[Init] Trace {trace_id} started for query: \"{query}\""],
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "status": "in_progress",
        }

        # Run compiled LangGraph state machine
        final_state = self.graph.invoke(initial_state)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        final_state["trace_steps"].append(f"[Complete] Finished in {total_latency_ms} ms | Cost: ${final_state.get('cost_usd', 0.0):.6f}")

        return QueryResponse(
            query=query,
            answer=final_state.get("synthesized_answer", ""),
            sources=final_state.get("reranked_docs", []),
            faithfulness_score=final_state.get("faithfulness_score", 0.0),
            answer_relevance=final_state.get("answer_relevance", 0.0),
            latency_ms=total_latency_ms,
            cost_usd=round(final_state.get("cost_usd", 0.0), 6),
            retries_count=final_state.get("retry_count", 0),
            trace_id=trace_id,
            agent_steps=final_state.get("trace_steps", []),
        )
