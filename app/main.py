"""
SurfaceScope -- RGB-D Surface Profiler & Analyzer -- FastAPI Application
=========================================================================

Main entry point for the FastAPI web server.  Serves the single-page
application (static files) and mounts the REST API router.

Server Configuration
--------------------
- Port: 8009
- Static files: ``app/static/``
- API prefix: ``/api/``
- CORS: enabled for local development

Usage::

    uvicorn app.main:app --host 0.0.0.0 --port 8009 --reload
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SurfaceScope — RGB-D Surface Profiler & Analyzer",
    description=(
        "RGB-D depth profiling application for synthetic and real depth data. "
        "Generates depth maps, applies processing pipelines (bilateral filter, "
        "hole filling), reconstructs 3D surfaces, and computes roughness metrics."
    ),
    version="2.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Resolve static directory relative to this file
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Mount static files
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def serve_index():
    """Serve the main single-page application."""
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/health")
async def health():
    """Health-check endpoint."""
    return {"status": "ok", "version": "2.0.0"}
