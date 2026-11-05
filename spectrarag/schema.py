"""Data models and schemas for SpectraRAG-Agent."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    doc_id: str
    text: str
    chunk_index: int
    char_count: int
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_reranking: bool = True
    enable_reflection: bool = True
    human_guidance: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[RetrievalResult]
    faithfulness_score: float
    answer_relevance: float
    latency_ms: float
    cost_usd: float
    retries_count: int
    trace_id: str
    agent_steps: List[str] = Field(default_factory=list)


class IngestionRequest(BaseModel):
    documents: List[Document]


class IngestionResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    ingestion_time_ms: float


class BenchmarkMetric(BaseModel):
    metric_name: str
    score: float
    sample_size: int
    std_dev: float = 0.0
    description: str


class BenchmarkSummary(BaseModel):
    total_queries: int
    faithfulness: float
    context_precision_mrr: float
    context_recall: float
    answer_relevance: float
    hallucination_rate_pct: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_cost_per_query_usd: float
    token_savings_pct: float
    ablation_results: Dict[str, Any] = Field(default_factory=dict)

