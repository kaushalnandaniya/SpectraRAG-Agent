"""Tool execution framework for SpectraRAG agents."""

from typing import Dict, Any, Callable, List, Optional
import math
import re


class BaseTool:
    """Abstract base class for agent tools."""
    name: str = "base_tool"
    description: str = ""

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    """Executes safe mathematical and statistical calculations."""
    name: str = "calculator"
    description: str = "Calculates numerical expressions, speedups, percentage differences, and latency metrics."

    def run(self, expression: str, **kwargs) -> Dict[str, Any]:
        # Clean expression
        expr = expression.strip()
        # Only allow safe mathematical operations
        safe_chars = set("0123456789+-*/().,%^ eE")
        if not all(c in safe_chars for c in expr):
            return {"error": "Invalid or unsafe characters in calculation expression."}

        try:
            # Replace ^ with ** for power
            sanitized = expr.replace("^", "**")
            # Evaluate using restricted environment
            allowed_globals = {"__builtins__": None, "math": math}
            result = eval(sanitized, allowed_globals, {})
            return {
                "expression": expr,
                "result": round(float(result), 4),
                "formatted": f"{expr} = {round(float(result), 4)}",
            }
        except Exception as e:
            return {"error": f"Calculation failed: {str(e)}"}


class AcronymLookupTool(BaseTool):
    """Resolves assistive technology and AI engineering acronyms."""
    name: str = "acronym_lookup"
    description: str = "Looks up definitions for AI, RAG, and assistive technology abbreviations."

    GLOSSARY = {
        "rrf": "Reciprocal Rank Fusion - An unsupervised rank aggregation algorithm that merges rankings from dense and sparse retrieval models using 1 / (k + rank).",
        "mrr": "Mean Reciprocal Rank - An information retrieval metric assessing where the first relevant document occurs in a candidate list (1/rank).",
        "rag": "Retrieval-Augmented Generation - Augmenting LLM generation with external knowledge retrieval.",
        "asr": "Automated Speech Recognition - Transcribing spoken human audio into written text.",
        "hitl": "Human-in-the-Loop - Integrating human verification checkpoints into autonomous agent workflows.",
        "ttft": "Time-To-First-Token - The latency between sending a request and receiving the initial streaming output token.",
        "plddt": "Predicted Local Distance Difference Test - AlphaFold confidence metric for per-residue accuracy (0-100).",
        "mcp": "Model Context Protocol - Open standard for connecting AI agents to external tools and data contexts.",
    }

    def run(self, acronym: str, **kwargs) -> Dict[str, Any]:
        key = acronym.strip().lower()
        if key in self.GLOSSARY:
            return {"acronym": acronym.upper(), "definition": self.GLOSSARY[key], "found": True}
        return {"acronym": acronym.upper(), "definition": "Term not found in built-in glossary.", "found": False}


class ToolRegistry:
    """Registry maintaining available agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.register(CalculatorTool())
        self.register(AcronymLookupTool())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found."}
        return tool.run(**kwargs)

