"""
Surface Reconstruction from Depth Maps
========================================

Converts dense depth maps into triangle meshes and computes differential
geometry quantities (curvature, cross-sections) for surface analysis.

Mathematical Background
-----------------------

**Depth-to-Mesh Conversion**

Given a depth image z(u, v) on a regular grid, each pixel with valid
depth defines a 3D vertex:

    V(u, v) = (u * pixel_size, v * pixel_size, z(u, v))

Adjacent valid pixels are connected into triangles.  For each 2x2 quad
of valid pixels at (u, v), (u+1, v), (u, v+1), (u+1, v+1), two
triangles are created:

    T1 = [idx(u,v), idx(u+1,v), idx(u,v+1)]
    T2 = [idx(u+1,v), idx(u+1,v+1), idx(u,v+1)]

**Gaussian Curvature K** (Gauss, 1827)

    K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2

where f_x, f_y are first partial derivatives, and f_xx, f_yy, f_xy are
second partial derivatives of the depth function z(x, y).

**Mean Curvature H** (differential geometry)

    H = ((1 + f_y^2)*f_xx - 2*f_x*f_y*f_xy + (1 + f_x^2)*f_yy)
        / (2 * (1 + f_x^2 + f_y^2)^(3/2))

**Cross-Section Extraction**

A 1D height profile is extracted along a line from point A to point B
using bilinear interpolation of the depth map at equally-spaced sample
points along the line segment.

References
----------
- do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces.
- Botsch, M., et al. (2010). Polygon Mesh Processing. A K Peters/CRC Press.
- Open3D documentation: Surface Reconstruction.
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def depth_to_mesh(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    subsample: int = 2,
    pixel_size: float = 1.0,
) -> dict:
    """Convert a depth map to a triangle mesh suitable for Three.js rendering.

    Algorithm:
        1. Subsample the depth map by the given factor.
        2. Create a 3D vertex at each valid pixel: (u, v, z).
        3. For each 2x2 quad of valid vertices, emit two triangles.
        4. Optionally assign per-vertex colour from the RGB image.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map in mm.  Zero = invalid.
    rgb : ndarray (H, W, 3), uint8 or None
        Colour image for per-vertex colouring.
    subsample : int
        Down-sampling factor (1 = full resolution, 2 = half, etc.).
    pixel_size : float
        Physical size of one pixel for x/y coordinates.

    Returns
    -------
    mesh : dict
        Keys:
            vertices : list of [x, y, z] (flat list for Three.js BufferGeometry)
            faces    : list of [i, j, k] triangle index triplets
            colors   : list of [r, g, b] normalised per-vertex colours
            n_vertices : int
            n_faces    : int
    """
    # Subsample
    d = depth[::subsample, ::subsample].copy()
    h, w = d.shape

    if rgb is not None:
        c = rgb[::subsample, ::subsample].astype(np.float64) / 255.0
    else:
        c = None

    # Build vertex array and index map
    valid = d > 0
    idx_map = np.full((h, w), -1, dtype=np.int32)
    vertices = []
    colors = []
    count = 0

    for v in range(h):
        for u in range(w):
            if valid[v, u]:
                x = u * pixel_size * subsample
                y = v * pixel_size * subsample
                z = float(d[v, u])
                vertices.append([x, y, z])
                if c is not None:
                    colors.append([float(c[v, u, 0]), float(c[v, u, 1]), float(c[v, u, 2])])
                else:
                    # Default grey
                    colors.append([0.7, 0.7, 0.7])
                idx_map[v, u] = count
                count += 1

    # Build triangles from 2x2 quads
    faces = []
    for v in range(h - 1):
        for u in range(w - 1):
            i00 = idx_map[v, u]
            i10 = idx_map[v, u + 1]
            i01 = idx_map[v + 1, u]
            i11 = idx_map[v + 1, u + 1]

            if i00 >= 0 and i10 >= 0 and i01 >= 0:
                faces.append([int(i00), int(i10), int(i01)])
            if i10 >= 0 and i11 >= 0 and i01 >= 0:
                faces.append([int(i10), int(i11), int(i01)])

    return {
        "vertices": vertices,
        "faces": faces,
        "colors": colors,
        "n_vertices": len(vertices),
        "n_faces": len(faces),
    }


def depth_to_mesh_fast(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    subsample: int = 2,
    pixel_size: float = 1.0,
) -> dict:
    """Vectorised (fast) depth-to-mesh conversion.

    Uses NumPy vectorisation instead of Python loops for better performance
    on large depth maps.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map in mm.
    rgb : ndarray (H, W, 3), uint8 or None
        Colour image.
    subsample : int
        Downsampling factor.
    pixel_size : float
        Pixel physical size.

    Returns
    -------
    mesh : dict
        Same structure as ``depth_to_mesh``.
    """
    d = depth[::subsample, ::subsample].astype(np.float64)
    h, w = d.shape
    valid = d > 0

    # Grid coordinates
    vs, us = np.mgrid[0:h, 0:w]
    xs = us.astype(np.float64) * pixel_size * subsample
    ys = vs.astype(np.float64) * pixel_size * subsample

    # Flatten valid vertices
    mask_flat = valid.ravel()
    all_xyz = np.stack([xs.ravel(), ys.ravel(), d.ravel()], axis=1)
    vertices = all_xyz[mask_flat]

    # Colours
    if rgb is not None:
        c = rgb[::subsample, ::subsample].astype(np.float64) / 255.0
        all_colors = c.reshape(-1, 3)
        colors = all_colors[mask_flat]
    else:
        colors = np.full((vertices.shape[0], 3), 0.7)

    # Index map
    idx_map = np.full(h * w, -1, dtype=np.int32)
    idx_map[mask_flat] = np.arange(vertices.shape[0], dtype=np.int32)
    idx_map = idx_map.reshape(h, w)

    # Faces from 2x2 quads (vectorised)
    i00 = idx_map[:-1, :-1].ravel()
    i10 = idx_map[:-1, 1:].ravel()
    i01 = idx_map[1:, :-1].ravel()
    i11 = idx_map[1:, 1:].ravel()

    # Triangle 1: i00, i10, i01
    t1_valid = (i00 >= 0) & (i10 >= 0) & (i01 >= 0)
    t1 = np.stack([i00[t1_valid], i10[t1_valid], i01[t1_valid]], axis=1)

    # Triangle 2: i10, i11, i01
    t2_valid = (i10 >= 0) & (i11 >= 0) & (i01 >= 0)
    t2 = np.stack([i10[t2_valid], i11[t2_valid], i01[t2_valid]], axis=1)

    faces = np.concatenate([t1, t2], axis=0) if t1.size and t2.size else (
        t1 if t1.size else t2
    )

    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "colors": colors.tolist(),
        "n_vertices": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
    }


# ---------------------------------------------------------------------------
# Curvature computation
# ---------------------------------------------------------------------------

def compute_surface_curvature(
    depth: np.ndarray,
    pixel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian and mean curvature from a depth map.

    Uses finite-difference approximations of the first and second partial
    derivatives of the surface z(x, y).

    **Gaussian curvature:**

        K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2

    where K > 0 indicates an elliptic point (bowl), K < 0 a saddle,
    and K = 0 a parabolic or flat point.

    **Mean curvature:**

        H = ((1 + f_y^2) * f_xx - 2 * f_x * f_y * f_xy + (1 + f_x^2) * f_yy)
            / (2 * (1 + f_x^2 + f_y^2)^(3/2))

    H > 0 indicates a concave surface (from above), H < 0 convex.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map in mm.
    pixel_size : float
        Physical pixel spacing in mm.

    Returns
    -------
    gaussian_curvature : ndarray (H, W), float32
        Gaussian curvature K at each pixel.
    mean_curvature : ndarray (H, W), float32
        Mean curvature H at each pixel.
    """
    z = depth.astype(np.float64)

    # First derivatives (central differences via np.gradient)
    fy, fx = np.gradient(z, pixel_size)

    # Second derivatives
    fyy, fxy_from_y = np.gradient(fy, pixel_size)
    fxy_from_x, fxx = np.gradient(fx, pixel_size)
    fxy = 0.5 * (fxy_from_y + fxy_from_x)  # average for symmetry

    denom_sq = 1.0 + fx ** 2 + fy ** 2

    # Gaussian curvature
    K = (fxx * fyy - fxy ** 2) / (denom_sq ** 2 + 1e-12)

    # Mean curvature
    H = (
        (1 + fy ** 2) * fxx
        - 2 * fx * fy * fxy
        + (1 + fx ** 2) * fyy
    ) / (2 * denom_sq ** 1.5 + 1e-12)

    return K.astype(np.float32), H.astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-section extraction
# ---------------------------------------------------------------------------

def extract_cross_section(
    depth: np.ndarray,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    num_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a 1D height profile along a line between two points.

    Samples the depth map at equally-spaced points along the line from
    *start_point* to *end_point* using bilinear interpolation.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map.
    start_point : (x, y)
        Start pixel coordinates (column, row).
    end_point : (x, y)
        End pixel coordinates.
    num_samples : int
        Number of samples along the profile.

    Returns
    -------
    distances : ndarray (num_samples,), float32
        Cumulative distance along the profile (in pixels).
    heights : ndarray (num_samples,), float32
        Depth values along the profile.  Invalid regions are set to NaN.
    """
    h, w = depth.shape
    x0, y0 = start_point
    x1, y1 = end_point

    t = np.linspace(0, 1, num_samples)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    total_dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    distances = (t * total_dist).astype(np.float32)

    # Bilinear interpolation
    heights = np.full(num_samples, np.nan, dtype=np.float32)
    for i in range(num_samples):
        x, y = xs[i], ys[i]
        if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            continue
        x0i, y0i = int(np.floor(x)), int(np.floor(y))
        x1i, y1i = x0i + 1, y0i + 1
        dx, dy = x - x0i, y - y0i

        v00 = depth[y0i, x0i]
        v10 = depth[y0i, x1i]
        v01 = depth[y1i, x0i]
        v11 = depth[y1i, x1i]

        # Skip if any corner is invalid
        if v00 <= 0 or v10 <= 0 or v01 <= 0 or v11 <= 0:
            continue

        val = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        heights[i] = val

    return distances, heights
