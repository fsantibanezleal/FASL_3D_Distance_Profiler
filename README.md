# SurfaceScope -- RGB-D Surface Profiler & Analyzer

A web-based RGB-D depth profiling application for synthetic and real depth data. SurfaceScope generates depth maps from five configurable scene types, applies edge-preserving processing pipelines (bilateral filter, hole filling), reconstructs interactive 3D surface meshes, computes differential geometry quantities (Gaussian and mean curvature), extracts cross-section profiles, and reports ISO 4287 surface roughness metrics (Ra, Rq, Rz, Rsk, Rku). Built with a Python/FastAPI backend and a Three.js-powered frontend for real-time 3D visualization.

## Frontend

![Frontend](docs/png/frontend.png)

## Architecture

![Architecture](docs/svg/architecture.svg)

## Processing Pipeline

![Pipeline](docs/svg/pipeline.svg)

## Quick Start

```bash
cd "d:/_Repos/_SCIENCE/FASL_3D_Distance_Profiler"
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python run_app.py
```

Open **http://localhost:8009** in your browser.

## Features

- **5 synthetic scene types**: Gaussian hills, terrain, object on table, conveyor belt, wave surface
- **Bilateral filter**: Edge-preserving depth smoothing with configurable spatial and range sigma
- **Hole filling**: Nearest-neighbour interpolation for invalid pixels with configurable maximum hole size
- **Surface normals**: Per-pixel unit normals via finite-difference gradients (numpy.gradient)
- **3D mesh generation**: Vectorised depth-to-triangle-mesh conversion for Three.js BufferGeometry
- **Surface curvature**: Gaussian curvature K and mean curvature H from second-order depth derivatives
- **Cross-section profiles**: Bilinear-interpolated 1D height profiles along arbitrary user-drawn lines
- **Roughness metrics**: ISO 4287 parameters -- Ra, Rq, Rz, Rsk, Rku
- **Object detection**: Depth-thresholding with connected-component labelling for raised objects
- **Interactive 3D viewer**: Three.js r128 with OrbitControls, wireframe overlay, point cloud mode
- **Surface area measurement**: Vectorised 3D triangle-area computation over the depth map
- **Image upload**: Support for user-provided depth + RGB image pairs (8-bit and 16-bit PNG)
- **Export**: PLY (ASCII), PCD (ASCII), and Wavefront OBJ mesh export
- **Colourmap rendering**: Hot, Viridis, Jet, and Greyscale depth visualisation
- **Help modal**: In-app documentation with keyboard/mouse controls and pipeline explanation

## Project Structure

```
FASL_3D_Distance_Profiler/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # REST API endpoints (generate, upload, process, profile, metrics, export, measure)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── colormap.py             # Depth-to-colour mapping (hot, viridis, jet, greyscale)
│   │   ├── depth_generator.py      # Synthetic RGB-D scene generator (5 scene types)
│   │   ├── depth_processing.py     # Bilateral filter, hole filling, normals, point cloud projection
│   │   ├── export.py               # PLY, PCD, OBJ file export
│   │   ├── profile_analysis.py     # Roughness metrics (Ra, Rq, Rz, Rsk, Rku), histogram, object detection
│   │   └── surface_reconstruction.py # Depth-to-mesh, curvature, cross-section extraction
│   └── static/
│       ├── index.html              # Single-page application frontend
│       ├── css/
│       │   └── style.css           # Dark-theme 3-panel layout stylesheet
│       └── js/
│           ├── app.js              # Main controller: wires UI to API
│           ├── renderer2d.js       # 2D canvas rendering + cross-section tool
│           └── renderer3d.js       # Three.js 3D surface renderer
├── tests/
│   ├── __init__.py
│   ├── test_depth_processing.py    # Bilateral filter, hole filling, normals, point cloud tests
│   ├── test_export.py              # PLY, PCD, OBJ export tests
│   ├── test_measurement.py         # Distance, angle, area measurement tests
│   ├── test_profile.py             # Roughness, histogram, object detection tests
│   └── test_surface.py             # Mesh generation, curvature, cross-section tests
├── docs/
│   ├── architecture.md             # System architecture and component diagram
│   ├── depth_theory.md             # RGB-D depth processing theory with equations
│   ├── development_history.md      # Changelog with mathematical foundations
│   ├── references.md               # Academic papers, standards, and library references
│   └── svg/
│       ├── architecture.svg        # Architecture diagram
│       └── pipeline.svg            # Processing pipeline diagram
├── build.spec                      # PyInstaller spec file
├── Build_PyInstaller.ps1           # PowerShell build script
├── run_app.py                      # Uvicorn launcher with auto-browser
├── requirements.txt                # Python dependencies
└── __init__.py
```

## API Documentation

| Method | Path               | Description                                         |
|--------|--------------------|-----------------------------------------------------|
| POST   | `/api/generate`    | Generate a synthetic RGB-D scene                    |
| POST   | `/api/upload`      | Upload depth map + optional RGB image               |
| POST   | `/api/process`     | Apply bilateral filter + hole filling pipeline      |
| GET    | `/api/state`       | Get current state (base64 images + mesh JSON)       |
| POST   | `/api/profile`     | Extract cross-section profile between two points    |
| GET    | `/api/metrics`     | Compute depth histogram, curvature, object detection|
| POST   | `/api/measure`     | Measure distance, angle, or surface area            |
| GET    | `/api/export/ply`  | Download point cloud / mesh as PLY (ASCII)          |
| GET    | `/api/export/pcd`  | Download point cloud as PCD (ASCII)                 |
| GET    | `/api/export/obj`  | Download mesh as Wavefront OBJ                      |
| GET    | `/api/scene_types` | List available synthetic scene types                |
| GET    | `/api/colormaps`   | List available colourmap names                      |
| GET    | `/health`          | Health check endpoint                               |

## Port

**8009**

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

## Documentation

- [Architecture](docs/architecture.md) -- System overview, technology stack, data flow
- [Depth Theory](docs/depth_theory.md) -- Mathematical foundations: pinhole model, bilateral filter, curvature, roughness
- [Development History](docs/development_history.md) -- Changelog with equations
- [References](docs/references.md) -- Academic papers, standards, hardware, and software references

## Mathematical Model

### Pinhole Camera Back-Projection

3D world coordinates are recovered from pixel (u, v) and depth Z:

```
X = (u - cx) * Z / fx,    Y = (v - cy) * Z / fy
```

where `(cx, cy)` is the principal point and `(fx, fy)` are the focal lengths in pixels.

### Gaussian Curvature

The intrinsic surface curvature is computed from second-order depth derivatives:

```
K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2
```

where `f_x, f_y` are first derivatives and `f_xx, f_yy, f_xy` are second derivatives of the depth map.

### Arithmetic Mean Roughness (ISO 4287)

The average absolute deviation of the height profile from the mean line:

```
Ra = (1/N) * Sum_i |z_i - z_bar|
```

where `z_i` are the sampled heights along the cross-section and `z_bar` is their mean.

### Root-Mean-Square Roughness

```
Rq = sqrt((1/N) * Sum_i (z_i - z_bar)^2)
```

### Bilateral Filter

Edge-preserving smoothing combining spatial and range Gaussians:

```
BF[I](p) = (1/W) * Sum_q  G_sigma_s(||p - q||) * G_sigma_r(|I(p) - I(q)|) * I(q)
```

where `W` is the normalizing partition function, `G_sigma_s` is the spatial kernel, and `G_sigma_r` is the range (intensity) kernel.

### Surface Normals

Per-pixel unit normals computed from depth gradients:

```
n_hat = normalize(-dz/dx, -dz/dy, 1)
```

---

## References

- Tomasi, C. & Manduchi, R. (1998). Bilateral Filtering for Gray and Color Images. ICCV.
- ISO 4287:1997. GPS -- Surface texture: Profile method.
- do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces.
- Hartley, R. & Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
- Three.js -- JavaScript 3D library: https://threejs.org/
