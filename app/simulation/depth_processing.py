"""
Depth Map Processing Pipeline
==============================

Provides filtering, hole-filling, alignment, and conversion utilities
for RGB-D depth maps.  All operations are implemented in pure NumPy/SciPy
without external point-cloud libraries.

Mathematical Background
-----------------------

**Bilateral Filter** (Tomasi & Manduchi, 1998)

The bilateral filter smooths an image while preserving edges by weighting
each neighbour by *both* spatial proximity and intensity similarity:

    BF[I](p) = (1 / W_p) * sum_{q in Omega}
               G_sigma_s(||p - q||) * G_sigma_r(|I(p) - I(q)|) * I(q)

where:
    G_sigma_s(x) = exp(-x^2 / (2 * sigma_s^2))   -- spatial Gaussian
    G_sigma_r(x) = exp(-x^2 / (2 * sigma_r^2))   -- range Gaussian
    W_p = sum_{q} G_sigma_s(||p-q||) * G_sigma_r(|I(p)-I(q)|)  -- normaliser

**Surface Normals from Depth**

Given depth z(u, v) and pixel spacing h:
    dz/du ~ (z[u+1,v] - z[u-1,v]) / (2h)      (central differences)
    dz/dv ~ (z[u,v+1] - z[u,v-1]) / (2h)

    n = (-dz/du, -dz/dv, 1) / ||(-dz/du, -dz/dv, 1)||

**Depth-to-Point-Cloud Projection** (pinhole camera model)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth[v, u]

where (fx, fy) are focal lengths and (cx, cy) is the principal point.

References
----------
- Tomasi, C., & Manduchi, R. (1998). Bilateral Filtering for Gray and
  Color Images. ICCV.
- Hartley, R., & Zisserman, A. (2003). Multiple View Geometry, Chapter 6.
- Paris, S., et al. (2009). Bilateral Filtering: Theory and Applications.
  Foundations and Trends in Computer Graphics and Vision.
"""

import numpy as np
from scipy.ndimage import (
    uniform_filter,
    binary_dilation,
    distance_transform_edt,
    label,
)
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Bilateral filter
# ---------------------------------------------------------------------------

def bilateral_filter_depth(
    depth: np.ndarray,
    sigma_spatial: float = 5.0,
    sigma_range: float = 10.0,
    kernel_size: int = 0,
) -> np.ndarray:
    """Edge-preserving bilateral filter for a depth map.

    Preserves sharp depth discontinuities (object edges) while smoothing
    measurement noise on planar or slowly-varying surfaces.

    The output at pixel p is:

        out(p) = sum_{q in window} w_s(p,q) * w_r(p,q) * depth(q)
                 / sum_{q} w_s(p,q) * w_r(p,q)

    with:
        w_s(p,q) = exp(-||p-q||^2 / (2 * sigma_spatial^2))
        w_r(p,q) = exp(-(depth(p) - depth(q))^2 / (2 * sigma_range^2))

    Implementation note: A full bilateral filter is O(N * k^2); we use
    a separable approximation by iterating over the spatial kernel
    explicitly for acceptable quality at moderate kernel sizes.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Input depth map in mm.  Zero values are treated as invalid.
    sigma_spatial : float
        Standard deviation of the spatial Gaussian (pixels).
    sigma_range : float
        Standard deviation of the range Gaussian (mm).
    kernel_size : int
        Window half-size.  If 0, defaults to ``ceil(2 * sigma_spatial)``.

    Returns
    -------
    filtered : ndarray (H, W), float32
        Filtered depth map.  Invalid (zero) pixels remain zero.
    """
    if kernel_size <= 0:
        kernel_size = int(np.ceil(2 * sigma_spatial))

    depth = depth.astype(np.float64)
    h, w = depth.shape
    valid = depth > 0
    output = np.zeros_like(depth)
    weight_sum = np.zeros_like(depth)

    # Pre-compute spatial weights for one quadrant (symmetric)
    offsets = np.arange(-kernel_size, kernel_size + 1)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    spatial_w = np.exp(-(ox ** 2 + oy ** 2) / (2 * sigma_spatial ** 2))

    for dy in range(-kernel_size, kernel_size + 1):
        for dx in range(-kernel_size, kernel_size + 1):
            # Shifted depth
            shifted = np.zeros_like(depth)
            sy = slice(max(0, dy), min(h, h + dy))
            sy_src = slice(max(0, -dy), min(h, h - dy))
            sx = slice(max(0, dx), min(w, w + dx))
            sx_src = slice(max(0, -dx), min(w, w - dx))
            shifted[sy, sx] = depth[sy_src, sx_src]

            shifted_valid = shifted > 0
            both_valid = valid & shifted_valid

            range_diff = depth - shifted
            w_r = np.exp(-(range_diff ** 2) / (2 * sigma_range ** 2))
            w_s = spatial_w[dy + kernel_size, dx + kernel_size]

            w_total = w_s * w_r * both_valid.astype(np.float64)
            output += w_total * shifted
            weight_sum += w_total

    # ``np.where`` evaluates both branches before selecting, so the division
    # runs on zero-weight pixels too and would emit a spurious
    # ``RuntimeWarning: invalid value encountered in divide``.  We silence the
    # warning scoped to just this division; the mask still picks ``0.0`` for
    # those pixels, so the output is unchanged.
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight_sum > 0, output / weight_sum, 0.0)
    result[~valid] = 0.0
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------

def fill_holes(
    depth: np.ndarray,
    max_hole_size: int = 10,
) -> np.ndarray:
    """Fill small holes (zero-depth pixels) using nearest-neighbour interpolation.

    Algorithm:
        1. Label connected components of invalid (zero) pixels.
        2. Discard components larger than *max_hole_size* x *max_hole_size*
           pixels in area (these are likely real occlusions).
        3. For remaining small holes, assign each pixel the depth value
           of its nearest valid neighbour (using Euclidean distance transform).

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map.  Zero = invalid / no data.
    max_hole_size : int
        Maximum area (in pixels) of a hole to fill.  Larger holes are
        left as-is.

    Returns
    -------
    filled : ndarray (H, W), float32
        Depth map with small holes filled.
    """
    depth = depth.astype(np.float64).copy()
    invalid = depth <= 0
    if not invalid.any():
        return depth.astype(np.float32)

    # Label connected components of holes
    labelled, n_labels = label(invalid)

    # Identify small holes
    small_hole_mask = np.zeros_like(invalid)
    for lbl in range(1, n_labels + 1):
        component = labelled == lbl
        if component.sum() <= max_hole_size ** 2:
            small_hole_mask |= component

    if not small_hole_mask.any():
        return depth.astype(np.float32)

    # Nearest-neighbour interpolation via distance transform
    valid = depth > 0
    _, nearest_indices = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    filled_values = depth[nearest_indices[0], nearest_indices[1]]

    depth[small_hole_mask] = filled_values[small_hole_mask]
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Surface normals
# ---------------------------------------------------------------------------

def compute_normals(
    depth: np.ndarray,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Compute per-pixel surface normals from a depth map.

    Uses numpy.gradient (central differences for interior, forward/backward
    at boundaries) to estimate partial derivatives dz/dx and dz/dy, then
    forms the normal vector:

        n = (-dz/dx, -dz/dy, 1)
        n_hat = n / ||n||

    Convention: z-axis points *towards* the camera (positive depth direction),
    so a flat surface facing the camera has normal (0, 0, 1).

    Parameters
    ----------
    depth : ndarray (H, W)
        Depth map in mm.
    pixel_size : float
        Physical spacing between adjacent pixels (mm/pixel).
        Affects the relative magnitude of x/y gradients vs z.

    Returns
    -------
    normals : ndarray (H, W, 3), float32
        Unit normal vectors.
    """
    dz_dy, dz_dx = np.gradient(depth.astype(np.float64), pixel_size)
    normals = np.zeros((*depth.shape, 3), dtype=np.float64)
    normals[..., 0] = -dz_dx
    normals[..., 1] = -dz_dy
    normals[..., 2] = 1.0

    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(mag, 1e-8)
    return normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth ↔ RGB alignment
# ---------------------------------------------------------------------------

def align_depth_to_rgb(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Optional[dict] = None,
) -> np.ndarray:
    """Align a depth map to an RGB image coordinate frame.

    In a real RGB-D system the depth and colour sensors have different
    positions and fields of view.  Alignment requires:
        1. Projecting depth pixels to 3D using depth intrinsics.
        2. Transforming 3D points via the extrinsic rotation+translation
           from the depth frame to the colour frame.
        3. Re-projecting into the colour image using colour intrinsics.

    This implementation performs a *simulated* alignment by resampling the
    depth map to match the RGB image resolution using bilinear interpolation,
    which is sufficient for synthetic data where both modalities share the
    same virtual camera.

    Parameters
    ----------
    depth : ndarray (Hd, Wd)
        Depth map (possibly different resolution to RGB).
    rgb : ndarray (Hc, Wc, 3)
        Colour image defining the target resolution.
    intrinsics : dict or None
        Camera intrinsic parameters (currently unused for synthetic data).

    Returns
    -------
    aligned : ndarray (Hc, Wc), float32
        Depth map resampled to RGB resolution.
    """
    from scipy.ndimage import zoom

    h_target, w_target = rgb.shape[:2]
    h_src, w_src = depth.shape

    if h_src == h_target and w_src == w_target:
        return depth.astype(np.float32)

    zoom_y = h_target / h_src
    zoom_x = w_target / w_src
    aligned = zoom(depth.astype(np.float64), (zoom_y, zoom_x), order=1)
    return aligned.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth → Point Cloud
# ---------------------------------------------------------------------------

def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert a depth map to a coloured 3D point cloud.

    Uses the standard pinhole camera model to back-project each pixel
    with valid depth into 3D space:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth[v, u]

    where:
        (fx, fy) -- focal lengths in pixels
        (cx, cy) -- principal point (defaults to image centre)
        (u, v)   -- pixel column and row indices

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map in mm.  Zero values are excluded.
    rgb : ndarray (H, W, 3), uint8 or None
        Optional colour image for point colouring.
    fx, fy : float
        Focal lengths in pixel units.
    cx, cy : float or None
        Principal point.  Defaults to (W/2, H/2).

    Returns
    -------
    points : ndarray (N, 3), float32
        3D coordinates [X, Y, Z] of valid pixels.
    colors : ndarray (N, 3), float32 or None
        Normalised [0, 1] RGB colours for each point, or None if no
        RGB image was provided.
    """
    h, w = depth.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    v_coords, u_coords = np.where(depth > 0)
    z = depth[v_coords, u_coords].astype(np.float64)
    x = (u_coords.astype(np.float64) - cx) * z / fx
    y = (v_coords.astype(np.float64) - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    colors = None
    if rgb is not None:
        colors = rgb[v_coords, u_coords].astype(np.float32) / 255.0

    return points, colors
