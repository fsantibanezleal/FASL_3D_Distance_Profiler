# Development History

## v2.0.0 (2026-03-28)

Complete rewrite of the 3D Distance Profiler application.

### Features
- **Synthetic RGB-D scene generation**: 5 scene types (Gaussian hills, terrain, object on table, conveyor belt, wave surface) with configurable resolution and random seeds
- **Bilateral filter**: Edge-preserving depth smoothing with configurable spatial and range sigma parameters
- **Hole filling**: Nearest-neighbour interpolation for small invalid regions with configurable maximum hole size
- **Surface normal computation**: Per-pixel normals via finite-difference gradient estimation
- **Depth-to-point-cloud projection**: Pinhole camera model inverse projection (X = (u-cx)*Z/fx)
- **Triangle mesh generation**: Vectorised depth-to-mesh conversion for Three.js rendering with per-vertex RGB colouring
- **Surface curvature**: Gaussian curvature K and mean curvature H from second-order depth derivatives
- **Cross-section profiles**: Bilinear-interpolated 1D height profiles along arbitrary lines
- **Roughness metrics**: ISO 4287 metrics (Ra, Rq, Rz, Rsk, Rku) for surface characterisation
- **Object detection**: Depth-thresholding-based detection of raised objects with connected-component labelling
- **Colourmap rendering**: Hot, Viridis, Jet, and Greyscale depth visualisation with alpha-blended RGB overlays
- **Three.js 3D viewer**: Interactive surface mesh with OrbitControls, wireframe, normals, and point cloud modes
- **Image upload**: Support for user-provided depth + RGB image pairs
- **FastAPI REST API**: 6 endpoints for generation, upload, processing, state, profile, and metrics

### Technical Details
- Backend: Python 3.10+, FastAPI, NumPy, SciPy, Pillow
- Frontend: Three.js r128, vanilla JavaScript ES6
- Port: 8009
- Architecture: Single-page application with 3-panel layout (2D views, 3D viewer, controls)

### Mathematical Foundations
- Bilateral filter: BF[I](p) = (1/Wp) SUM G_s(||p-q||) G_r(|I(p)-I(q)|) I(q)
- Surface normals: n = (-dz/dx, -dz/dy, 1) / ||...||
- Gaussian curvature: K = (fxx*fyy - fxy^2) / (1+fx^2+fy^2)^2
- Mean curvature: H = ((1+fy^2)fxx - 2*fx*fy*fxy + (1+fx^2)fyy) / (2*(1+fx^2+fy^2)^(3/2))
- Roughness Ra: (1/N) SUM |zi - z_mean|
- Point cloud: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy
