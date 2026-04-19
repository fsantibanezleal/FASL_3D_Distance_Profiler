"""
Application state for SurfaceScope.
==========================================

Holds the current RGB-D scene (RGB image, depth map, normals, processed
depth, and scene metadata) for a single user.  Extracted from
``app.api.routes`` so the state can be unit-tested in isolation and
injected into FastAPI routes via ``Depends(get_app_state)``.

This module intentionally has **no** FastAPI or HTTP imports so that the
domain state stays framework-agnostic — only the provider
``get_app_state`` is coupled to the request lifecycle.

Usage
-----
Inside a route:

    from fastapi import Depends
    from app.core import AppState, get_app_state

    @router.get("/state")
    async def get_state(state: AppState = Depends(get_app_state)):
        ...

For tests:

    from app.core import AppState
    state = AppState()
    state.depth = np.zeros((8, 8), dtype=np.float32)
    assert state.has_data()
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class AppState:
    """Holds the current scene data in memory.

    Attributes
    ----------
    rgb : ndarray or None
        (H, W, 3) uint8 RGB image.
    depth : ndarray or None
        (H, W) float32 depth map in millimetres (raw input).
    normals : ndarray or None
        (H, W, 3) float32 surface normals, unit length.
    processed_depth : ndarray or None
        (H, W) float32 depth after the filtering pipeline.  Falls back
        to ``depth`` when the pipeline has not run yet.
    scene_type : str
        Label describing the loaded scene (synthetic type or
        ``"uploaded"``).
    width, height : int
        Pixel dimensions of the current scene.
    """

    def __init__(self) -> None:
        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        self.processed_depth: Optional[np.ndarray] = None
        self.scene_type: str = ""
        self.width: int = 256
        self.height: int = 256

    def has_data(self) -> bool:
        """Return True once a depth map has been loaded or generated."""
        return self.depth is not None


# ---------------------------------------------------------------------------
# Dependency provider
# ---------------------------------------------------------------------------

# Single process-wide instance.  Kept module-level so FastAPI's
# ``Depends(get_app_state)`` returns the same object across requests, which
# matches the single-user demo-application semantics this project ships
# with.  A future multi-user rework would swap this for a per-session
# factory without touching route signatures.
_APP_STATE = AppState()


def get_app_state() -> AppState:
    """FastAPI dependency that returns the process-wide ``AppState``.

    Unit tests can monkey-patch ``app.core.state._APP_STATE`` or
    override this dependency via ``app.dependency_overrides`` to inject
    a fresh ``AppState`` per test.
    """
    return _APP_STATE
