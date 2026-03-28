"""
PyInstaller Entry Point for FASL 3D Distance Profiler
======================================================

Launches the Uvicorn ASGI server hosting the FastAPI application
on port 8009.  This file is the target for PyInstaller packaging.

Usage (development)::

    python run_app.py

Usage (packaged)::

    FASL_3D_Distance_Profiler.exe
"""

import sys
import os
import webbrowser
import threading


def main():
    """Start the FASL 3D Distance Profiler server."""
    # Ensure the project root is on sys.path so that `app` is importable
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import uvicorn

    host = "127.0.0.1"
    port = 8009

    # Open browser after a short delay
    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"Starting FASL 3D Distance Profiler on http://{host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
