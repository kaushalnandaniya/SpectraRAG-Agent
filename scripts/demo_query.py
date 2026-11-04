#!/usr/bin/env python3
"""Interactive demo script executing a sample query through the LangGraph agent."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.agent.graph import SpectraAgentEngine
from spectrarag.schema import Document


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What display technology and latency does DefXV AR glasses use for speech-to-text?"

    print("=" * 80)
    print("   SpectraRAG-Agent: Live Multi-Agent Execution Demo")
    print("=" * 80)
    print(f"Query: \"{query}\"\n")

    # Load corpus
    docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "spectrarag", "data", "sample_docs.json")
    with open(docs_path) as f:
        docs = [Document(**d) for d in json.load(f)]

    chunker = SemanticChunker()
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    chunks = chunker.chunk_documents(docs)
    retriever.index_chunks(chunks)

    engine = SpectraAgentEngine(retriever=retriever, reranker=reranker)

    print("[*] Running LangGraph multi-agent execution...")
    resp = engine.execute(query)

    print("\n--- AGENT TRACE STEPS ---")
    for step in resp.agent_steps:
        print(f"  {step}")

    print("\n--- VERIFIED RESPONSE ---")
    print(resp.answer)

    print("\n--- PERFORMANCE & SAFETY METRICS ---")
    print(f"  Faithfulness Score:  {resp.faithfulness_score:.3f} / 1.000")
    print(f"  Answer Relevance:    {resp.answer_relevance:.3f} / 1.000")
    print(f"  Latency:             {resp.latency_ms:.2f} ms")
    print(f"  Estimated Cost:      ${resp.cost_usd:.6f}")
    print(f"  Sources Cited:       {len(resp.sources)} chunks")
    print("=" * 80)


if __name__ == "__main__":
    main()
