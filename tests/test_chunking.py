"""Unit tests for semantic chunking module."""

import pytest
from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.schema import Document


def test_chunking_basic():
    chunker = SemanticChunker(chunk_size=120, chunk_overlap=20, min_chunk_len=10)
    text = (
        "DefXV is an augmented reality glasses system. "
        "It empowers deaf and mute individuals with speech-to-text. "
        "The micro-OLED displays render text with sub-18ms latency. "
        "It also features bone conduction transducers for audio output."
    )
    doc = Document(id="doc_test_1", text=text, metadata={"category": "hardware"})
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc_test_1"
        assert c.char_count >= 10
        assert "chunk_index" in c.metadata
        assert c.metadata["category"] == "hardware"


def test_chunking_empty_doc():
    chunker = SemanticChunker()
    doc = Document(id="empty", text="", metadata={})
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunking_batch():
    chunker = SemanticChunker(chunk_size=150, chunk_overlap=20, min_chunk_len=10)
    docs = [
        Document(id="d1", text="First test document discussing LLM serving and speculative decoding algorithms."),
        Document(id="d2", text="Second test document covering LangGraph multi-agent cyclic workflows and checkpointing."),
    ]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= 2
    doc_ids = set(c.doc_id for c in chunks)
    assert "d1" in doc_ids and "d2" in doc_ids

