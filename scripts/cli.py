#!/usr/bin/env python3
"""Interactive Terminal CLI and REPL for SpectraRAG-Agent.

Provides colorful terminal traces, retrieval inspection, tool execution breakdown,
and human course-correction capabilities.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.agent.graph import SpectraAgentEngine
from spectrarag.schema import Document

# ANSI Color codes for rich terminal styling without heavy dependencies
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def load_engine() -> SpectraAgentEngine:
    docs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "spectrarag",
        "data",
        "sample_docs.json",
    )
    with open(docs_path) as f:
        docs = [Document(**d) for d in json.load(f)]

    chunker = SemanticChunker()
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    chunks = chunker.chunk_documents(docs)
    retriever.index_chunks(chunks)

    return SpectraAgentEngine(retriever=retriever, reranker=reranker)


def print_banner():
    banner = f"""
{CYAN}{BOLD}================================================================================
   SpectraRAG-Agent: Autonomous Multi-Agent Retrieval & Reasoning CLI
================================================================================{RESET}
{DIM}Cyclic LangGraph StateGraph | Hybrid RRF (Dense + BM25) | Tool-Calling & Course-Correction{RESET}
"""
    print(banner)


def run_query(engine: SpectraAgentEngine, query: str, guidance: str = None):
    print(f"\n{BOLD}{CYAN}Query:{RESET} {query}")
    if guidance:
        print(f"{YELLOW}{BOLD}Course-Correction Guidance:{RESET} {guidance}")

    print(f"\n{MAGENTA}[*] Executing LangGraph state graph...{RESET}")
    t0 = time.perf_counter()
    resp = engine.execute(query=query, human_guidance=guidance)
    elapsed = (time.perf_counter() - t0) * 1000.0

    print(f"\n{BOLD}--- AGENT EXECUTION TRACE ---{RESET}")
    for step in resp.agent_steps:
        if "[Planner]" in step:
            print(f"  {CYAN}{step}{RESET}")
        elif "[Retriever]" in step or "[ReRanker]" in step:
            print(f"  {YELLOW}{step}{RESET}")
        elif "[ToolExecutor]" in step:
            print(f"  {MAGENTA}{step}{RESET}")
        elif "[Auditor]" in step:
            print(f"  {GREEN}{step}{RESET}")
        else:
            print(f"  {DIM}{step}{RESET}")

    print(f"\n{BOLD}{GREEN}--- VERIFIED GROUNDED ANSWER ---{RESET}")
    print(f"{resp.answer}\n")

    print(f"{BOLD}--- RETRIEVED SOURCES ({len(resp.sources)}) ---{RESET}")
    for i, s in enumerate(resp.sources):
        print(
            f"  [{i+1}] {DIM}(Score: {s.score:.4f} | Rerank: {s.rerank_score or 0.0:.3f}){RESET} {s.text[:120]}..."
        )

    print(f"\n{BOLD}--- METRICS & RESOURCE USAGE ---{RESET}")
    print(f"  Faithfulness Score:  {GREEN}{resp.faithfulness_score:.3f} / 1.000{RESET}")
    print(f"  Answer Relevance:    {GREEN}{resp.answer_relevance:.3f} / 1.000{RESET}")
    print(f"  End-to-End Latency:  {CYAN}{resp.latency_ms:.2f} ms{RESET}")
    print(f"  Estimated Cost:      {YELLOW}${resp.cost_usd:.6f}{RESET}")
    print(f"  Reflection Retries:  {resp.retries_count}")
    print(f"{DIM}{'-' * 80}{RESET}")


def interactive_repl(engine: SpectraAgentEngine):
    print(f"\n{YELLOW}Type your question, or 'exit' / 'quit' to quit.{RESET}")
    print(f"{DIM}Tip: Suffix with ' --guide <your guidance>' to test Human Course-Correction.{RESET}\n")

    while True:
        try:
            raw = input(f"{BOLD}SpectraRAG>{RESET} ").strip()
            if not raw:
                continue
            if raw.lower() in ["exit", "quit", "q"]:
                print("Exiting SpectraRAG CLI. Goodbye!")
                break

            guidance = None
            if " --guide " in raw:
                parts = raw.split(" --guide ", 1)
                query = parts[0].strip()
                guidance = parts[1].strip()
            else:
                query = raw

            run_query(engine, query, guidance)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def main():
    print_banner()
    print("[*] Initializing hybrid tables and compiling LangGraph engine...")
    engine = load_engine()
    print(f"{GREEN}[+] Engine loaded and ready.{RESET}")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(engine, query)
    else:
        interactive_repl(engine)


if __name__ == "__main__":
    main()
