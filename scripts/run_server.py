#!/usr/bin/env python3
"""CLI script to start the FastAPI production server."""

import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"[*] Launching SpectraRAG-Agent API server at http://{host}:{port}")
    uvicorn.run("spectrarag.api.app:app", host=host, port=port, reload=False)
