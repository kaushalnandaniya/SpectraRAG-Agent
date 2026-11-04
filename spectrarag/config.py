"""Configuration and hyperparameters for SpectraRAG-Agent."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class SpectraConfig:
    # Chunking Hyperparameters
    chunk_size: int = 400
    chunk_overlap: int = 60
    min_chunk_len: int = 50

    # Retrieval Hyperparameters
    dense_top_k: int = 15
    sparse_top_k: int = 15
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    rerank_top_k: int = 5
    rerank_threshold: float = 0.55

    # Self-Reflection & Hallucination Guardrails
    faithfulness_threshold: float = 0.72
    max_reflection_retries: int = 2
    confidence_decay_penalty: float = 0.15

    # Evaluation & Token Cost Modeling (based on gpt-4o-mini / claude-3-haiku tier)
    cost_per_1k_input_tokens: float = 0.00015
    cost_per_1k_output_tokens: float = 0.00060

    # System & Paths
    embedding_dim: int = 384
    storage_dir: str = os.getenv("SPECTRARAG_STORAGE_DIR", "./data_store")
    enable_cache: bool = True
    cache_ttl_sec: int = 3600


default_config = SpectraConfig()
