# Architecture

## System Overview

SurfaceScope -- RGB-D Surface Profiler & Analyzer is a web-based depth profiling application built with a Python/FastAPI backend and a Three.js-powered frontend. The system follows a single-page application (SPA) architecture with a REST API separating the frontend presentation from the backend computation engine.

![Architecture Diagram](svg/architecture.svg)

![Processing Pipeline](svg/pipeline.svg)

## Technology Stack

| Layer      | Technology                               | Purpose                              |
|------------|------------------------------------------|--------------------------------------|
| Backend    | Python 3.10+, FastAPI 0.135+, Uvicorn    | REST API server, static file serving |
| Compute    | NumPy 2.4+, SciPy 1.17+                 | Array operations, filtering, labelling |
| Imaging    | Pillow 12+                               | Image I/O, format conversion         |
| Frontend   | HTML5, CSS3, JavaScript ES6              | Single-page application UI           |
| 3D Render  | Three.js r128 + OrbitControls            | WebGL 3D surface visualisation       |
| Packaging  | PyInstaller                              | Standalone Windows executable        |

## Directory Structure

```
FASL_3D_Distance_Profiler/
|-- app/
|   |-- __init__.py
|   |-- main.py                          # FastAPI app factory, CORS, static mount, health check
|   |-- api/
|   |   |-- __init__.py
|   |   |-- routes.py                    # All REST endpoints + in-memory AppState
|   |-- simulation/
|   |   |-- __init__.py
|   |   |-- colormap.py                  # hot/viridis/jet/greyscale colourmap functions
|   |   |-- depth_generator.py           # 5 synthetic scene generators + normal/texture helpers
|   |   |-- depth_processing.py          # Bilateral filter, hole fill, normals, point cloud
|   |   |-- export.py                    # PLY, PCD, OBJ serialisation
|   |   |-- profile_analysis.py          # Roughness (Ra,Rq,Rz,Rsk,Rku), histogram, object detection
|   |   |-- surface_reconstruction.py    # Mesh generation (loop + vectorised), curvature, cross-section
|   |-- static/
|       |-- index.html                   # SPA entry point with 3-panel layout + help modal
|       |-- css/style.css                # Dark-theme responsive CSS
|       |-- js/app.js                    # Main controller: DOM wiring, API calls, state dispatch
|       |-- js/renderer2d.js             # 2D canvas: base64 PNG drawing, cross-section tool
|       |-- js/renderer3d.js             # Three.js: mesh/wireframe/points, camera, lighting
|-- docs/                                # Technical documentation
|-- tests/                               # Unit test suite (unittest)
|-- build.spec                           # PyInstaller configuration
|-- Build_PyInstaller.ps1                # Windows build script
|-- requirements.txt                     # Pinned Python dependencies
|-- run_app.py                           # Uvicorn launcher with auto-browser
```

## API Endpoint Table

| Method | Path               | Request Body / Params                                     | Response                                  |
|--------|--------------------|-----------------------------------------------------------|-------------------------------------------|
| POST   | `/api/generate`    | `{scene_type, width, height, seed}`                       | Full state: images (b64), mesh (JSON)     |
| POST   | `/api/upload`      | Multipart: `depth_file`, optional `rgb_file`              | Full state                                |
| POST   | `/api/process`     | `{bilateral_sigma_spatial, bilateral_sigma_range, fill_hole_size}` | Full state (reprocessed)          |
| GET    | `/api/state`       | --                                                        | Full state or `{loaded: false}`           |
| POST   | `/api/profile`     | `{start_x, start_y, end_x, end_y, num_samples}`          | `{distances, heights, metrics, start, end}` |
| GET    | `/api/metrics`     | --                                                        | `{histogram, objects, curvature, depth_stats}` |
| POST   | `/api/measure`     | `{measurement_type, point_a, point_b, normal_a, normal_b, pixel_size}` | `{measurement_type, value, unit}` |
| GET    | `/api/export/ply`  | --                                                        | PLY file (streaming download)             |
| GET    | `/api/export/pcd`  | --                                                        | PCD file (streaming download)             |
| GET    | `/api/export/obj`  | --                                                        | OBJ file (streaming download)             |
| GET    | `/api/scene_types` | --                                                        | `{scene_types: [...]}`                    |
| GET    | `/api/colormaps`   | --                                                        | `{colormaps: [...]}`                      |
| GET    | `/health`          | --                                                        | `{status: "ok", version: "2.0.0"}`       |

## Data Flow

1. **Scene Generation / Upload**: User selects a scene type and parameters (or uploads images). The backend generates or loads RGB (H,W,3 uint8), depth (H,W float32), and normal (H,W,3 float32) arrays. These are stored in the in-memory `AppState` singleton.

2. **Processing Pipeline**: On user request, the bilateral filter (edge-preserving smoothing) is applied to the depth map, followed by hole filling (nearest-neighbour interpolation). Normals are recomputed from the processed depth.

3. **Mesh Conversion**: The processed depth map is converted to a triangle mesh by the vectorised `depth_to_mesh_fast()`. Each valid pixel becomes a 3D vertex; adjacent valid quads produce two triangles each. Per-vertex colours come from the RGB image.

4. **State Response**: The full state response bundles three base64-encoded PNG images (RGB, depth colourmap, normal map) and the mesh data (vertices, faces, colours as flat JSON arrays) for Three.js `BufferGeometry`.

5. **Frontend Rendering**: `renderer2d.js` draws the PNG images onto HTML5 canvases. `renderer3d.js` constructs a Three.js indexed `BufferGeometry` from the mesh arrays and renders with Phong shading, optional wireframe overlay, and point cloud mode.

6. **Profile Analysis**: The user clicks two points on the depth canvas. The backend extracts a 1D cross-section via bilinear interpolation and computes ISO 4287 roughness metrics (Ra, Rq, Rz, Rsk, Rku) on the profile. The chart and metrics are rendered on the profile canvas.

7. **Metrics & Detection**: The metrics endpoint computes the depth histogram, detects raised objects via thresholding + connected-component labelling, and reports Gaussian and mean curvature statistics.

8. **Export**: PLY, PCD, and OBJ endpoints generate the file content server-side and stream it as a downloadable attachment.

## Performance Notes

- **Vectorised mesh generation**: `depth_to_mesh_fast()` uses NumPy array operations instead of Python loops, achieving ~10x speedup for 256x256 depth maps.
- **Bilateral filter**: The explicit O(N * k^2) implementation with kernel half-size k = ceil(2 * sigma_spatial). For sigma_spatial=5 on a 256x256 map, this processes ~2.7M weight evaluations. Acceptable for interactive use at typical resolutions.
- **Subsample factor**: Mesh resolution is controlled by a subsample parameter (default=2), which reduces vertex count by 4x while preserving visual quality.
- **In-memory state**: Single-user design with all data held in RAM. No database or filesystem I/O during normal operation.
- **Base64 encoding**: Images are encoded as data URIs to avoid filesystem writes. This adds ~33% overhead to transfer size but simplifies the API contract.

## Port Assignment

The application runs on **port 8009** to avoid conflicts with other FASL/SCIAN projects.
