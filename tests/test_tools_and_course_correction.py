"""Unit and integration tests for agent tools and human course-correction."""

import pytest
from spectrarag.tools.registry import CalculatorTool, AcronymLookupTool, ToolRegistry
from spectrarag.agent.graph import SpectraAgentEngine
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.schema import Document


def test_calculator_tool():
    tool = CalculatorTool()
    res = tool.run(expression="1000 / 1.25")
    assert res.get("result") == 800.0
    assert "800.0" in res.get("formatted")

    # Safety checks
    bad = tool.run(expression="__import__('os').system('ls')")
    assert "error" in bad


def test_acronym_lookup_tool():
    tool = AcronymLookupTool()
    res = tool.run(acronym="RRF")
    assert res.get("found") is True
    assert "Reciprocal Rank Fusion" in res.get("definition")

    unknown = tool.run(acronym="XYZ123")
    assert unknown.get("found") is False


def test_tool_registry():
    registry = ToolRegistry()
    assert len(registry.list_tools()) >= 2
    calc_res = registry.execute("calculator", expression="45 * 2")
    assert calc_res.get("result") == 90.0


def test_agent_with_tool_invocation_and_course_correction():
    chunker = SemanticChunker()
    retriever = HybridRetriever()

    doc = Document(
        id="doc_tools_test",
        text="The system baseline latency is 120 ms. The optimized agentic path achieves 1.2 ms.",
        metadata={"category": "benchmark"}
    )
    chunks = chunker.chunk_documents([doc])
    retriever.index_chunks(chunks)

    engine = SpectraAgentEngine(retriever=retriever)

    # 1. Test calculation tool triggering through natural query
    q = "Please calculate 120 / 1.2 to find the speedup factor"
    resp = engine.execute(q)
    assert any("[ToolExecutor]" in step for step in resp.agent_steps)
    assert "100.0" in resp.answer

    # 2. Test Human Course Correction guidance
    resp_guided = engine.execute(
        "What is the latency?",
        human_guidance="Focus specifically on the optimized agentic path"
    )
    assert "Course Correction Applied" in resp_guided.answer
    assert "1.2 ms" in resp_guided.answer
