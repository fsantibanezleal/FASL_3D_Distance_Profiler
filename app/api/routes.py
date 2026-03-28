"""
API Routes for FASL 3D Distance Profiler
==========================================

RESTful endpoints for generating, uploading, processing, and analysing
RGB-D depth data.  All heavy computation runs in the backend; the
frontend receives pre-computed meshes, images, and metrics as JSON.

Endpoints
---------
POST /api/generate   - Generate a synthetic RGB-D scene
POST /api/upload     - Upload depth + RGB images
POST /api/process    - Apply filtering pipeline to current scene
GET  /api/state      - Get current state (images + mesh data)
POST /api/profile    - Extract cross-section profile between two points
GET  /api/metrics    - Compute surface roughness metrics
"""

import io
import base64
import numpy as np
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from app.simulation.depth_generator import generate_scene, list_scene_types
from app.simulation.depth_processing import (
    bilateral_filter_depth,
    fill_holes,
    compute_normals,
    depth_to_point_cloud,
)
from app.simulation.surface_reconstruction import (
    depth_to_mesh_fast,
    compute_surface_curvature,
    extract_cross_section,
)
from app.simulation.profile_analysis import (
    compute_roughness,
    compute_histogram,
    detect_objects,
)
from app.simulation.colormap import apply_colormap, COLOURMAP_NAMES

router = APIRouter(prefix="/api", tags=["api"])


# ---------------------------------------------------------------------------
# In-memory state (single-user demo application)
# ---------------------------------------------------------------------------

class AppState:
    """Holds the current scene data in memory."""
    def __init__(self):
        self.rgb: Optional[np.ndarray] = None          # (H, W, 3) uint8
        self.depth: Optional[np.ndarray] = None         # (H, W) float32
        self.normals: Optional[np.ndarray] = None       # (H, W, 3) float32
        self.processed_depth: Optional[np.ndarray] = None
        self.scene_type: str = ""
        self.width: int = 256
        self.height: int = 256

    def has_data(self) -> bool:
        return self.depth is not None


state = AppState()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    scene_type: str = Field(default="gaussian_hills", description="Scene type identifier")
    width: int = Field(default=256, ge=32, le=512, description="Image width")
    height: int = Field(default=256, ge=32, le=512, description="Image height")
    seed: Optional[int] = Field(default=None, description="Random seed")

class ProcessRequest(BaseModel):
    bilateral_sigma_spatial: float = Field(default=5.0, ge=0.1, le=50.0)
    bilateral_sigma_range: float = Field(default=10.0, ge=0.1, le=200.0)
    fill_hole_size: int = Field(default=10, ge=0, le=100)

class ProfileRequest(BaseModel):
    start_x: float = Field(..., description="Start X pixel coordinate")
    start_y: float = Field(..., description="Start Y pixel coordinate")
    end_x: float = Field(..., description="End X pixel coordinate")
    end_y: float = Field(..., description="End Y pixel coordinate")
    num_samples: int = Field(default=256, ge=10, le=1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    """Convert a uint8 ndarray to a base64-encoded PNG string.

    Parameters
    ----------
    arr : ndarray
        (H, W, 3) uint8 for RGB or (H, W) uint8 for greyscale.

    Returns
    -------
    b64 : str
        Data-URI-ready base64 PNG string.
    """
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _normals_to_image(normals: np.ndarray) -> np.ndarray:
    """Convert normal vectors to a visualisation image.

    Maps normals from [-1, 1] range to [0, 255] uint8:
        colour = (normal + 1) / 2 * 255
    """
    vis = ((normals + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return vis


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/scene_types")
async def get_scene_types():
    """Return available synthetic scene types."""
    return {"scene_types": list_scene_types()}


@router.get("/colormaps")
async def get_colormaps():
    """Return available colourmap names."""
    return {"colormaps": COLOURMAP_NAMES}


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a synthetic RGB-D scene and store in application state.

    Returns base64-encoded PNG images for RGB, depth colourmap, and
    normal map, plus the triangle mesh for Three.js rendering.
    """
    rgb, depth, normals = generate_scene(
        scene_type=req.scene_type,
        width=req.width,
        height=req.height,
        seed=req.seed,
    )

    state.rgb = rgb
    state.depth = depth
    state.normals = normals
    state.processed_depth = depth.copy()
    state.scene_type = req.scene_type
    state.width = req.width
    state.height = req.height

    return _build_state_response()


@router.post("/upload")
async def upload(
    depth_file: UploadFile = File(..., description="Depth map image (16-bit PNG or 8-bit)"),
    rgb_file: Optional[UploadFile] = File(None, description="Optional RGB image"),
):
    """Upload depth and optional RGB images.

    Depth images are loaded as float32 arrays.  16-bit PNGs are scaled
    by dividing by 65535 * 1000 to get millimetre values.  8-bit images
    are scaled by 1000/255.
    """
    # Load depth
    depth_bytes = await depth_file.read()
    depth_img = Image.open(io.BytesIO(depth_bytes))
    depth_arr = np.array(depth_img).astype(np.float64)

    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]  # take first channel

    # Scale to mm
    if depth_arr.max() > 255:
        # 16-bit
        depth_arr = depth_arr / 65535.0 * 1000.0
    else:
        depth_arr = depth_arr / 255.0 * 1000.0

    depth_arr = depth_arr.astype(np.float32)

    # Load RGB
    if rgb_file is not None:
        rgb_bytes = await rgb_file.read()
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb_arr = np.array(rgb_img).astype(np.uint8)
    else:
        # Generate greyscale RGB from depth
        d_vis = apply_colormap(depth_arr, colormap="viridis")
        rgb_arr = d_vis

    normals = compute_normals(depth_arr)

    state.rgb = rgb_arr
    state.depth = depth_arr
    state.normals = normals
    state.processed_depth = depth_arr.copy()
    state.scene_type = "uploaded"
    state.height, state.width = depth_arr.shape

    return _build_state_response()


@router.post("/process")
async def process(req: ProcessRequest):
    """Apply the depth processing pipeline to the current scene.

    Pipeline steps:
        1. Bilateral filter (edge-preserving noise reduction)
        2. Hole filling (nearest-neighbour interpolation)
        3. Recompute normals from processed depth
    """
    if not state.has_data():
        raise HTTPException(status_code=400, detail="No scene loaded. Generate or upload first.")

    processed = state.depth.copy()

    # Step 1: bilateral filter
    if req.bilateral_sigma_spatial > 0:
        processed = bilateral_filter_depth(
            processed,
            sigma_spatial=req.bilateral_sigma_spatial,
            sigma_range=req.bilateral_sigma_range,
        )

    # Step 2: hole filling
    if req.fill_hole_size > 0:
        processed = fill_holes(processed, max_hole_size=req.fill_hole_size)

    state.processed_depth = processed
    state.normals = compute_normals(processed)

    return _build_state_response()


@router.get("/state")
async def get_state():
    """Return the full current state (images + mesh)."""
    if not state.has_data():
        return {"loaded": False}
    return _build_state_response()


@router.post("/profile")
async def profile(req: ProfileRequest):
    """Extract a cross-section profile between two points.

    Returns the 1D height profile, cumulative distances, and
    roughness metrics along the profile.
    """
    if not state.has_data():
        raise HTTPException(status_code=400, detail="No scene loaded.")

    depth = state.processed_depth if state.processed_depth is not None else state.depth

    distances, heights = extract_cross_section(
        depth,
        start_point=(req.start_x, req.start_y),
        end_point=(req.end_x, req.end_y),
        num_samples=req.num_samples,
    )

    metrics = compute_roughness(heights)

    return {
        "distances": distances.tolist(),
        "heights": [float(h) if not np.isnan(h) else None for h in heights],
        "metrics": metrics,
        "start": [req.start_x, req.start_y],
        "end": [req.end_x, req.end_y],
    }


@router.get("/metrics")
async def metrics():
    """Compute global surface metrics for the current scene.

    Returns depth histogram, object detection results, and curvature
    statistics.
    """
    if not state.has_data():
        raise HTTPException(status_code=400, detail="No scene loaded.")

    depth = state.processed_depth if state.processed_depth is not None else state.depth

    histogram = compute_histogram(depth, bins=80)
    objects = detect_objects(depth)
    K, H = compute_surface_curvature(depth)

    return {
        "histogram": histogram,
        "objects": objects,
        "curvature": {
            "gaussian_mean": float(np.nanmean(K)),
            "gaussian_std": float(np.nanstd(K)),
            "mean_curvature_mean": float(np.nanmean(H)),
            "mean_curvature_std": float(np.nanstd(H)),
        },
        "depth_stats": {
            "min": float(depth[depth > 0].min()) if (depth > 0).any() else 0,
            "max": float(depth.max()),
            "mean": float(depth[depth > 0].mean()) if (depth > 0).any() else 0,
        },
    }


# ---------------------------------------------------------------------------
# Internal state builder
# ---------------------------------------------------------------------------

def _build_state_response(colormap: str = "viridis", subsample: int = 2) -> dict:
    """Build the full JSON state response including images and mesh."""
    depth = state.processed_depth if state.processed_depth is not None else state.depth

    # Images as base64 PNGs
    rgb_b64 = _ndarray_to_base64_png(state.rgb)
    depth_vis = apply_colormap(depth, colormap=colormap)
    depth_b64 = _ndarray_to_base64_png(depth_vis)
    normals_vis = _normals_to_image(state.normals)
    normals_b64 = _ndarray_to_base64_png(normals_vis)

    # Mesh for Three.js
    mesh = depth_to_mesh_fast(depth, rgb=state.rgb, subsample=subsample)

    return {
        "loaded": True,
        "scene_type": state.scene_type,
        "width": state.width,
        "height": state.height,
        "images": {
            "rgb": rgb_b64,
            "depth": depth_b64,
            "normals": normals_b64,
        },
        "mesh": mesh,
    }
