"""State definition for LangGraph multi-agent orchestration."""

from typing import TypedDict, List, Dict, Any, Optional
from spectrarag.schema import RetrievalResult


class AgentState(TypedDict, total=False):
    """Execution state passed through the LangGraph agentic pipeline."""
    query: str
    cleaned_query: str
    intent: str
    sub_queries: List[str]
    expanded_queries: List[str]
    retrieved_docs: List[RetrievalResult]
    reranked_docs: List[RetrievalResult]
    synthesized_answer: str
    faithfulness_score: float
    answer_relevance: float
    hallucination_detected: bool
    reflection_notes: str
    retry_count: int
    max_retries: int
    tool_calls: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    human_guidance: Optional[str]
    awaiting_human_approval: bool
    trace_steps: List[str]
    cost_usd: float
    latency_ms: float
    status: str
