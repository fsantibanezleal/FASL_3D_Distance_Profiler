# User Guide — SurfaceScope (RGB-D Surface Profiler & Analyzer)

A task-oriented guide for operators. For the theory behind each step, see [depth_theory.md](depth_theory.md); for the system layout, see [architecture.md](architecture.md).

---

## 1. Launching the App

```bash
cd "d:/_Repos/_SCIENCE/FASL_3D_Distance_Profiler"
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python run_app.py
```

The app opens automatically at [http://localhost:8009](http://localhost:8009). The server is single-user and holds its state in RAM — restart clears everything.

---

## 2. Panel Overview

SurfaceScope uses a 3-panel layout:

| Panel           | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| Left (controls) | Scene selection, upload, processing parameters, export buttons          |
| Centre (2D)     | RGB / depth colourmap / normal map canvases + cross-section tool        |
| Right (3D)      | Three.js interactive surface viewer with mesh / wireframe / point modes |

Press the `?` button in the header for the in-app keyboard/mouse reference.

---

## 3. Typical Workflow

### 3.1 Load data

Pick one of the two sources:

- **Generate synthetic scene** — choose from `gaussian_hills`, `terrain`, `object_on_table`, `conveyor_belt`, `wave_surface`. Adjust `width`, `height`, and `seed` for reproducibility. Good for demos and regression.
- **Upload depth + RGB** — supports 8-bit and 16-bit PNG. The depth image is treated as raw depth in millimetres (16-bit) or normalised (8-bit). The RGB image is optional — a neutral grey texture is applied if omitted.

Once loaded, the 2D panel shows RGB, depth (with the selected colourmap), and per-pixel surface normals.

### 3.2 Process the depth map

Under **Processing**, tune the pipeline:

- `bilateral_sigma_spatial` (σ_s) — spatial kernel radius in pixels. Larger = more smoothing. Typical: 3–7.
- `bilateral_sigma_range` (σ_r) — depth-similarity kernel in millimetres. Larger = more smoothing across steps. Typical: 5–30.
- `fill_hole_size` — maximum contiguous invalid-pixel region to fill via nearest-neighbour interpolation. Use 0 to disable.

Click **Process** — the depth map, normals, and 3D mesh update in place.

### 3.3 Inspect in 3D

Drag on the 3D canvas to orbit (left button), pan (right button), and wheel to zoom. Use the toggles to switch between:

- **Mesh** — shaded Phong surface with per-vertex RGB colours.
- **Wireframe** — triangle edges only. Useful to inspect sampling density.
- **Points** — vertex cloud. Good for visual noise assessment.

### 3.4 Extract a cross-section profile

Click two points on the depth canvas. The backend extracts a 1D height profile via bilinear interpolation along the line and returns:

- distances / heights arrays (rendered as a profile chart)
- ISO 4287 roughness metrics: **Ra** (mean deviation), **Rq** (RMS), **Rz** (peak-to-valley), **Rsk** (skewness), **Rku** (kurtosis)

Ra and Rq characterise average roughness; Rz captures the worst feature; Rsk and Rku describe asymmetry and peakedness of the height distribution.

### 3.5 Metrics & object detection

The **Metrics** panel shows the full-frame depth histogram, Gaussian and mean curvature statistics, and the list of detected raised objects (connected-component labelling after depth thresholding). Each object entry includes bounding box, centroid, pixel count, and mean height.

### 3.6 Export

From the export section:

| Format | Use case                                     | Tool                     |
|--------|----------------------------------------------|--------------------------|
| PLY    | Point cloud + mesh + per-vertex colour       | MeshLab, CloudCompare    |
| PCD    | Point cloud only (no topology)               | PCL, Open3D              |
| OBJ    | Mesh with triangular faces                   | Blender, MeshLab         |

Each file is streamed as an ASCII download — no temporary files are written to disk.

---

## 4. Troubleshooting

| Symptom                              | Likely cause                                    | Fix                                                             |
|--------------------------------------|-------------------------------------------------|-----------------------------------------------------------------|
| Port 8009 already in use             | Another SCIAN/FASL app is running               | Stop the other app or change the port in `run_app.py`           |
| 3D panel empty after load            | Depth map all-zero (bad upload)                 | Verify depth PNG encodes non-zero pixels                        |
| Bilateral filter feels too slow      | Large σ_s (kernel half-size ∝ σ_s)              | Reduce σ_s or subsample resolution                              |
| Cross-section jagged                 | Hole filling disabled + invalid pixels on path  | Increase `fill_hole_size`                                       |
| Object detection misses raised parts | Threshold too close to background               | Adjust depth threshold under Metrics                            |
| Exported PLY opens empty in MeshLab  | File saved with ASCII header but binary payload | Re-export — SurfaceScope always writes ASCII; retry the download|

---

## 5. Programmatic Use (REST API)

Every UI action corresponds to a REST endpoint. Example using `curl`:

```bash
# Generate a synthetic scene
curl -X POST http://localhost:8009/api/generate \
     -H "Content-Type: application/json" \
     -d '{"scene_type": "gaussian_hills", "width": 256, "height": 256, "seed": 42}'

# Apply the processing pipeline
curl -X POST http://localhost:8009/api/process \
     -H "Content-Type: application/json" \
     -d '{"bilateral_sigma_spatial": 5, "bilateral_sigma_range": 20, "fill_hole_size": 3}'

# Extract a profile and compute roughness
curl -X POST http://localhost:8009/api/profile \
     -H "Content-Type: application/json" \
     -d '{"start_x": 32, "start_y": 64, "end_x": 224, "end_y": 192, "num_samples": 200}'

# Download the surface as PLY
curl -O -J http://localhost:8009/api/export/ply
```

See [architecture.md](architecture.md) for the full endpoint table and request/response shapes.

---

## 6. Where to go next

- Theory reference: [depth_theory.md](depth_theory.md)
- System architecture: [architecture.md](architecture.md)
- Version log: [development_history.md](development_history.md)
- Papers & standards: [references.md](references.md)
