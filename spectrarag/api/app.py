"""FastAPI application factory and middleware configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from spectrarag.api.routes import router


def create_app() -> FastAPI:
    """Builds and configures the FastAPI instance."""
    app = FastAPI(
        title="SpectraRAG-Agent API",
        description="Autonomous Agentic RAG & Evaluation Engine with LangGraph, Hybrid Retrieval & Cross-Encoders",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
