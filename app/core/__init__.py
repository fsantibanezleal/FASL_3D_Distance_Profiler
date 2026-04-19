"""
Core application primitives.

This package holds cross-cutting concerns (application state, dependency
providers) that are consumed by the API layer (``app.api``).  Keeping
these out of the route module makes them unit-testable in isolation and
lets FastAPI inject the state via ``Depends()``.
"""

from app.core.state import AppState, get_app_state

__all__ = ["AppState", "get_app_state"]
