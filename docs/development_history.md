# Development History

---

## v2.0.0 (2026-03-28) -- Complete Python Rewrite

Complete rewrite of the 3D Distance Profiler as a web application with Python/FastAPI backend and Three.js frontend. Renamed to **SurfaceScope -- RGB-D Surface Profiler & Analyzer**.

### Surface Reconstruction

Depth-to-mesh via quad triangulation. For each valid 2x2 pixel quad, two triangles are created:

```
T1 = [idx(u,v), idx(u+1,v), idx(u,v+1)]
T2 = [idx(u+1,v), idx(u+1,v+1), idx(u,v+1)]
```

Vertex position:

```
V(u, v) = (u * pixel_size, v * pixel_size, depth[v, u])
```

Both a loop-based reference implementation and a vectorised (NumPy) fast path are provided.

### Bilateral Filter

Edge-preserving smoothing applied to depth maps:

```
BF[I](p) = (1/Wp) * SUM_{q in Omega} G_sigma_s(||p-q||) * G_sigma_r(|I(p)-I(q)|) * I(q)
```

where:
- `G_sigma_s(d) = exp(-d^2 / (2 * sigma_s^2))` -- spatial Gaussian
- `G_sigma_r(delta) = exp(-delta^2 / (2 * sigma_r^2))` -- range Gaussian
- `Wp = SUM_q G_sigma_s(||p-q||) * G_sigma_r(|I(p)-I(q)|)` -- normaliser

### Curvature Analysis

Gaussian curvature (intrinsic surface bending):

```
K = (fxx * fyy - fxy^2) / (1 + fx^2 + fy^2)^2
```

Mean curvature (average principal curvature):

```
H = ((1 + fy^2)*fxx - 2*fx*fy*fxy + (1 + fx^2)*fyy) / (2*(1 + fx^2 + fy^2)^(3/2))
```

### Surface Normals

Per-pixel unit normals via finite-difference gradients:

```
n = normalize(-dz/dx, -dz/dy, 1)
```

where derivatives are estimated by central differences using `numpy.gradient`.

### ISO 4287 Roughness

Arithmetic average roughness:

```
Ra = (1/N) * SUM |zi - z_mean|
```

Root mean square roughness:

```
Rq = sqrt((1/N) * SUM (zi - z_mean)^2)
```

Peak-to-valley height:

```
Rz = max(z) - min(z)
```

Skewness:

```
Rsk = (1/(N*Rq^3)) * SUM (zi - z_mean)^3
```

Kurtosis:

```
Rku = (1/(N*Rq^4)) * SUM (zi - z_mean)^4
```

### Point Cloud Projection

Pinhole camera model inverse projection:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth[v, u]
```

### Features Added

- **5 synthetic scene types**: Gaussian hills, Perlin-like terrain, object on table, conveyor belt, wave surface
- **Bilateral filter**: Configurable sigma_spatial and sigma_range
- **Hole filling**: Nearest-neighbour interpolation with max hole size control
- **Surface normal computation**: Central finite differences via `numpy.gradient`
- **Triangle mesh generation**: Vectorised depth-to-mesh conversion
- **Surface curvature**: Gaussian K and mean H from second-order derivatives
- **Cross-section profiles**: Bilinear-interpolated 1D height profiles
- **Roughness metrics**: Ra, Rq, Rz, Rsk, Rku (ISO 4287)
- **Object detection**: Depth-thresholding + connected-component labelling
- **3D surface area**: Vectorised triangle area summation
- **Export**: PLY, PCD, OBJ file formats
- **Colourmap rendering**: Hot, Viridis, Jet, Greyscale
- **Interactive 3D viewer**: Three.js r128 + OrbitControls with wireframe, normals, point cloud modes
- **Image upload**: Support for 8-bit and 16-bit depth images
- **Help modal**: In-app documentation with keyboard controls
- **FastAPI REST API**: 13 endpoints

### Technical Stack

- Backend: Python 3.10+, FastAPI, Uvicorn, NumPy, SciPy, Pillow
- Frontend: Three.js r128, vanilla JavaScript ES6, HTML5 Canvas
- Port: 8009
- Architecture: Single-page application with 3-panel layout
- Packaging: PyInstaller (one-dir mode)

---

## v1.x (2019) [Legacy -- MATLAB]

Original implementation of the 3D Distance Profiler using MATLAB.

### Features

- **SICK Ranger3D data reader**: Parsed raw line-scan profiles from SICK Ranger3D industrial 3D cameras
- **Depth map assembly**: Stitched individual line profiles into full 2D depth maps
- **Basic surface visualisation**: MATLAB `surf()` and `mesh()` plotting
- **Profile extraction**: Manual line selection for cross-section analysis
- **Simple roughness**: Ra computation from profiles

### Limitations

- MATLAB licence required
- No web interface
- No real-time 3D interaction
- Limited export capabilities
- No automated object detection
- No curvature analysis
