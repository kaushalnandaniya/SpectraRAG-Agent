"""FastAPI application factory and middleware configuration."""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>SpectraRAG-Agent API Running</h1><p>Visit <a href='/docs'>/docs</a></p>")

    return app


app = create_app()

