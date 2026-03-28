# Architecture

## System Overview

FASL 3D Distance Profiler is a web-based RGB-D depth profiling application built with a Python/FastAPI backend and a Three.js-powered frontend.

## Technology Stack

| Layer      | Technology                     |
|------------|--------------------------------|
| Backend    | Python 3.10+, FastAPI, Uvicorn |
| Compute    | NumPy, SciPy                   |
| Frontend   | HTML5, CSS3, JavaScript ES6    |
| 3D Render  | Three.js r128 + OrbitControls  |
| Packaging  | PyInstaller                    |

## Component Diagram

```
Browser (localhost:8009)
  +---------------------------+
  | index.html                |
  |  +-------+ +----------+  |
  |  |2D     | |3D Three.js|  |
  |  |Canvas | |Surface    |  |
  |  +-------+ +----------+  |
  |  |Controls Panel       |  |
  +-----|---------------------+
        | REST API (JSON)
        v
  +---------------------------+
  | FastAPI (app/main.py)     |
  |  /api/generate            |
  |  /api/upload              |
  |  /api/process             |
  |  /api/profile             |
  |  /api/metrics             |
  +-----|---------------------+
        |
        v
  +---------------------------+
  | Simulation Engine         |
  |  depth_generator.py       |
  |  depth_processing.py      |
  |  surface_reconstruction.py|
  |  profile_analysis.py      |
  |  colormap.py              |
  +---------------------------+
```

## Data Flow

1. **Scene Generation**: User selects scene type and parameters. Backend generates RGB (H,W,3), depth (H,W), and normal (H,W,3) arrays.

2. **Processing Pipeline**: Bilateral filter (edge-preserving smoothing) followed by hole filling (nearest-neighbour interpolation).

3. **Mesh Conversion**: Depth map is converted to a triangle mesh by treating each valid pixel as a 3D vertex and connecting adjacent pixels into triangles.

4. **Frontend Rendering**: Images are sent as base64 PNGs. Mesh data (vertices, faces, colours) is sent as JSON arrays for Three.js BufferGeometry.

5. **Profile Analysis**: User clicks two points on the depth canvas. Backend extracts a 1D cross-section via bilinear interpolation and computes roughness metrics.

## API Contract

All endpoints accept/return JSON. Images are embedded as data URIs. Mesh data uses flat arrays compatible with Three.js BufferGeometry.

## Port Assignment

The application runs on **port 8009** to avoid conflicts with other FASL projects.
