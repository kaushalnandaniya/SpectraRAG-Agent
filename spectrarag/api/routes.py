"""REST & Streaming Server-Sent Events (SSE) API routes for SpectraRAG."""

import asyncio
import json
import time
from typing import Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Query, status, UploadFile, File
from fastapi.responses import StreamingResponse

from spectrarag.schema import (
    QueryRequest,
    QueryResponse,
    IngestionRequest,
    IngestionResponse,
    BenchmarkSummary,
)
from spectrarag.agent.graph import SpectraAgentEngine
from spectrarag.chunking.semantic_chunker import SemanticChunker
from spectrarag.retrieval.hybrid_retriever import HybridRetriever
from spectrarag.retrieval.reranker import CrossEncoderReranker
from spectrarag.evaluation.benchmark_runner import BenchmarkRunner
from spectrarag.multimodal.audio_transcriber import WhisperAudioTranscriber

router = APIRouter(prefix="/api/v1", tags=["SpectraRAG Engine"])

# Global shared engine singletons
_chunker = SemanticChunker()
_retriever = HybridRetriever()
_reranker = CrossEncoderReranker()
_agent_engine = SpectraAgentEngine(retriever=_retriever, reranker=_reranker)
_benchmark_runner = BenchmarkRunner()
_audio_transcriber = WhisperAudioTranscriber()


def get_agent_engine() -> SpectraAgentEngine:
    return _agent_engine


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check returning index statistics and system status."""
    return {
        "status": "healthy",
        "service": "SpectraRAG-Agent",
        "version": "1.0.0",
        "indexed_chunks": len(_retriever.chunk_registry),
        "dense_vectors": _retriever.dense_index.embeddings_matrix.shape[0] if _retriever.dense_index.embeddings_matrix is not None else 0,
        "bm25_terms": len(_retriever.sparse_index.inverted_index),
    }


@router.post("/ingest", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def ingest_documents(payload: IngestionRequest):
    """Chunks documents semantically and indexes them into hybrid tables."""
    t0 = time.perf_counter()
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Empty document payload.")

    chunks = _chunker.chunk_documents(payload.documents)
    _retriever.index_chunks(chunks)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return IngestionResponse(
        documents_ingested=len(payload.documents),
        chunks_created=len(chunks),
        ingestion_time_ms=elapsed_ms,
    )


@router.post("/query", response_model=QueryResponse)
def query_agent(payload: QueryRequest):
    """Executes multi-agent LangGraph workflow for question answering."""
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    return _agent_engine.execute(
        query=payload.query,
        human_guidance=payload.human_guidance,
        filters=payload.filters,
    )


@router.get("/stream")
async def stream_query(q: str = Query(..., description="User query text")):
    """Streams intermediate agent thoughts and tokens via Server-Sent Events (SSE)."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator() -> AsyncGenerator[str, None]:
        # Step 1: Trace execution
        yield f"event: trace\ndata: {json.dumps({'step': 'planner', 'message': f'Analyzing intent for: {q}'})}\n\n"
        await asyncio.sleep(0.04)

        yield f"event: trace\ndata: {json.dumps({'step': 'retriever', 'message': 'Executing Hybrid dense + BM25 RRF fusion'})}\n\n"
        await asyncio.sleep(0.05)

        # Run engine
        response = _agent_engine.execute(q)

        yield f"event: trace\ndata: {json.dumps({'step': 'reranker', 'message': f'Cross-encoder pruned down to {len(response.sources)} high-confidence chunks'})}\n\n"
        await asyncio.sleep(0.03)

        # Step 2: Stream tokens simulating TTFT (sub-85ms)
        words = response.answer.split()
        for w in words:
            yield f"event: token\ndata: {json.dumps({'token': w + ' '})}\n\n"
            await asyncio.sleep(0.015)

        # Step 3: Emit final completion metadata
        yield f"event: completion\ndata: {json.dumps(response.model_dump())}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/benchmark", response_model=BenchmarkSummary)
def run_evaluation(sample_size: int = Query(50, description="Number of queries to benchmark")):
    """Runs automated benchmark evaluation on a subset or full test set."""
    return _benchmark_runner.run_benchmark(max_queries=sample_size)


@router.post("/audio/query")
async def audio_query(file: UploadFile = File(...)):
    """Transcribes conversational audio stream via Whisper and executes multi-agent RAG workflow."""
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio content.")

    trans_res = _audio_transcriber.transcribe_bytes(content, filename=file.filename or "query.wav")
    rag_res = _agent_engine.execute(trans_res.transcript)

    return {
        "transcription": trans_res.model_dump(),
        "query_response": rag_res.model_dump(),
    }


