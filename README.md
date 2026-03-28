# FASL 3D Distance Profiler

RGB-D depth profiling application for synthetic and real depth data. Generates depth maps, applies processing pipelines, reconstructs 3D surfaces, and computes roughness metrics.

## Features

- **5 synthetic scene types**: Gaussian hills, terrain, object on table, conveyor belt, wave surface
- **Bilateral filter**: Edge-preserving depth smoothing
- **Hole filling**: Nearest-neighbour interpolation for invalid pixels
- **Surface normals**: Per-pixel normals via finite-difference gradients
- **3D mesh generation**: Depth-to-triangle-mesh for Three.js rendering
- **Surface curvature**: Gaussian (K) and mean (H) curvature computation
- **Cross-section profiles**: Bilinear-interpolated 1D height profiles
- **Roughness metrics**: ISO 4287 parameters (Ra, Rq, Rz, Rsk, Rku)
- **Object detection**: Depth-thresholding with connected-component labelling
- **Interactive 3D viewer**: Three.js with OrbitControls, wireframe, point cloud modes

## Quick Start

```bash
cd "d:/_Repos/_SCIENCE/FASL_3D_Distance_Profiler"
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python run_app.py
```

Open http://localhost:8009 in your browser.

## API Endpoints

| Method | Endpoint         | Description                          |
|--------|------------------|--------------------------------------|
| POST   | /api/generate    | Generate synthetic scene             |
| POST   | /api/upload      | Upload depth + RGB images            |
| POST   | /api/process     | Apply filtering pipeline             |
| GET    | /api/state       | Get current state (images + mesh)    |
| POST   | /api/profile     | Extract cross-section profile        |
| GET    | /api/metrics     | Surface roughness and curvature      |

## Running Tests

```bash
python tests/test_depth_processing.py
python tests/test_surface.py
python tests/test_profile.py
```

## Building with PyInstaller

```powershell
.\Build_PyInstaller.ps1
```

## Port

8009
