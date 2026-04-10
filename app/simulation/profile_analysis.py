"""
Surface Profile Analysis
=========================

Computes surface roughness metrics, depth histograms, and performs
object detection on depth maps.  These tools are commonly used in
industrial surface inspection (e.g., conveyor-belt profiling with
SICK Ranger3D cameras).

Mathematical Background
-----------------------

**Arithmetic Average Roughness (Ra)** -- ISO 4287

    Ra = (1/N) * sum_{i=1}^{N} |z_i - z_mean|

Ra is the most widely used roughness parameter.  It gives the mean
absolute deviation of the profile from its mean line.

**Root Mean Square Roughness (Rq)** -- ISO 4287

    Rq = sqrt( (1/N) * sum_{i=1}^{N} (z_i - z_mean)^2 )

Rq (also called RMS roughness) is more sensitive to peaks and valleys
than Ra because it squares the deviations before averaging.

**Peak-to-Valley Height (Rz)** -- ISO 4287

    Rz = max(z) - min(z)

Rz gives the total range of the profile.  It is sensitive to outliers
and is often used alongside Ra for a more complete description.

**Skewness (Rsk)** -- ISO 4287

    Rsk = (1/(N * Rq^3)) * sum (z_i - z_mean)^3

Positive skewness indicates more peaks than valleys; negative skewness
indicates more deep scratches/valleys.

**Kurtosis (Rku)** -- ISO 4287

    Rku = (1/(N * Rq^4)) * sum (z_i - z_mean)^4

Rku = 3 for a Gaussian surface.  Rku > 3 indicates sharp peaks/valleys;
Rku < 3 indicates a bumpy surface.

**Object Detection via Depth Thresholding**

Objects raised above a reference plane are detected by:
    1. Estimating the background (reference) depth via robust statistics
       (median of the depth map).
    2. Thresholding: pixels significantly above the background are
       labelled as object pixels.
    3. Connected-component labelling groups adjacent object pixels.
    4. Small components below *min_area* are discarded as noise.

References
----------
- ISO 4287:1997 -- Geometrical Product Specifications (GPS) --
  Surface texture: Profile method -- Terms, definitions and surface
  texture parameters.
- Whitehouse, D. J. (2002). Surfaces and their Measurement. Kogan Page.
"""

import numpy as np
from scipy.ndimage import label
from typing import Dict, Any, List, Tuple, Optional, Union


# ---------------------------------------------------------------------------
# Roughness metrics
# ---------------------------------------------------------------------------

def compute_roughness(
    profile: np.ndarray,
    method: str = "all",
) -> Dict[str, float]:
    """Compute surface roughness metrics for a 1D height profile.

    Parameters
    ----------
    profile : ndarray (N,)
        Height values along a cross-section.  NaN values are excluded.
    method : str
        Which metric(s) to compute:
            'Ra'  -- arithmetic average roughness only
            'Rq'  -- RMS roughness only
            'Rz'  -- peak-to-valley only
            'all' -- all metrics (Ra, Rq, Rz, Rsk, Rku)

    Returns
    -------
    metrics : dict
        Keys are metric names, values are floats.
        Returns empty dict if profile has no valid points.

    Notes
    -----
    Formulae (ISO 4287):

        Ra  = mean(|z - z_mean|)
        Rq  = sqrt(mean((z - z_mean)^2))
        Rz  = max(z) - min(z)
        Rsk = mean((z - z_mean)^3) / Rq^3
        Rku = mean((z - z_mean)^4) / Rq^4
    """
    valid = profile[~np.isnan(profile)]
    if len(valid) < 2:
        return {}

    z_mean = np.mean(valid)
    deviations = valid - z_mean

    ra = float(np.mean(np.abs(deviations)))
    rq = float(np.sqrt(np.mean(deviations ** 2)))
    rz = float(np.max(valid) - np.min(valid))

    if method == "Ra":
        return {"Ra": ra}
    elif method == "Rq":
        return {"Rq": rq}
    elif method == "Rz":
        return {"Rz": rz}

    # Full set
    rsk = float(np.mean(deviations ** 3) / (rq ** 3 + 1e-12))
    rku = float(np.mean(deviations ** 4) / (rq ** 4 + 1e-12))

    return {
        "Ra": ra,
        "Rq": rq,
        "Rz": rz,
        "Rsk": rsk,
        "Rku": rku,
    }


# ---------------------------------------------------------------------------
# Depth histogram
# ---------------------------------------------------------------------------

def compute_histogram(
    depth: np.ndarray,
    bins: int = 100,
) -> Dict[str, Any]:
    """Compute the depth-value distribution histogram.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map.  Zero values are excluded.
    bins : int
        Number of histogram bins.

    Returns
    -------
    result : dict
        'counts'    : list of int   -- bin counts
        'bin_edges' : list of float -- bin edge values (length = bins + 1)
        'bin_centers': list of float -- bin centre values (length = bins)
        'mean'      : float         -- mean depth
        'std'       : float         -- standard deviation
        'median'    : float         -- median depth
    """
    valid = depth[depth > 0].astype(np.float64)
    if len(valid) == 0:
        return {
            "counts": [],
            "bin_edges": [],
            "bin_centers": [],
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
        }

    counts, bin_edges = np.histogram(valid, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
    }


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def detect_objects(
    depth: np.ndarray,
    threshold: Optional[float] = None,
    min_area: int = 100,
) -> List[Dict[str, Any]]:
    """Detect raised objects on a flat reference surface.

    Algorithm:
        1. Compute background depth as the median of all valid pixels.
        2. If *threshold* is None, set it to 2 * MAD (median absolute
           deviation) of the depth values -- a robust noise estimate.
        3. Mark pixels whose depth is at least *threshold* mm less than
           the background (i.e., closer to the camera = raised).
        4. Label connected components; discard those smaller than *min_area*.
        5. For each remaining component, compute bounding box, centroid,
           area, and mean height above the reference plane.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map in mm.
    threshold : float or None
        Minimum height above background to count as object (mm).
        If None, estimated automatically from the data.
    min_area : int
        Minimum number of pixels for a valid object.

    Returns
    -------
    objects : list of dict
        Each dict contains:
            'id'       : int        -- object label
            'centroid' : (y, x)     -- centroid in pixel coords
            'bbox'     : (y0, x0, y1, x1) -- bounding box
            'area'     : int        -- area in pixels
            'mean_height' : float   -- mean height above background (mm)
            'max_height'  : float   -- max height above background (mm)
    """
    valid = depth > 0
    if not valid.any():
        return []

    valid_values = depth[valid].astype(np.float64)
    background = float(np.median(valid_values))

    if threshold is None:
        mad = float(np.median(np.abs(valid_values - background)))
        threshold = max(2.0 * mad, 5.0)  # at least 5 mm

    # Objects are *closer* to camera => lower depth value
    object_mask = valid & (depth < background - threshold)

    if not object_mask.any():
        return []

    labelled, n_labels = label(object_mask)

    objects = []
    for lbl in range(1, n_labels + 1):
        component = labelled == lbl
        area = int(component.sum())
        if area < min_area:
            continue

        ys, xs = np.where(component)
        centroid = (float(np.mean(ys)), float(np.mean(xs)))
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        heights = background - depth[component]
        mean_height = float(np.mean(heights))
        max_height = float(np.max(heights))

        objects.append({
            "id": lbl,
            "centroid": centroid,
            "bbox": bbox,
            "area": area,
            "mean_height": round(mean_height, 2),
            "max_height": round(max_height, 2),
        })

    return objects


# ---------------------------------------------------------------------------
# Measurement tools
# ---------------------------------------------------------------------------

def measure_distance_3d(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Euclidean distance between two 3D points.

    Parameters
    ----------
    point_a : ndarray (3,)
        First point [x, y, z].
    point_b : ndarray (3,)
        Second point [x, y, z].

    Returns
    -------
    distance : float
        Euclidean distance ||point_b - point_a||.
    """
    return float(np.linalg.norm(np.asarray(point_b) - np.asarray(point_a)))


def measure_angle_between_normals(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    """Angle between two surface normals in degrees.

    Uses the formula:
        theta = arccos( dot(n1, n2) / (|n1| * |n2|) )

    Parameters
    ----------
    normal_a : ndarray (3,)
        First surface normal vector.
    normal_b : ndarray (3,)
        Second surface normal vector.

    Returns
    -------
    angle : float
        Angle in degrees in [0, 180].
    """
    na = np.asarray(normal_a, dtype=np.float64)
    nb = np.asarray(normal_b, dtype=np.float64)
    cos_angle = np.dot(na, nb) / (np.linalg.norm(na) * np.linalg.norm(nb) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def measure_area(
    depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pixel_size: float = 1.0,
) -> float:
    """Vectorised surface area computation over a depth map region.

    Computes the 3D surface area by projecting each pixel quad into 3D
    and summing the areas of two triangles per quad.  Uses NumPy
    vectorisation for performance.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map in mm.
    mask : ndarray (H, W), bool or None
        Region of interest.  If None, all pixels with depth > 0 are used.
    pixel_size : float
        Physical size of one pixel in mm.

    Returns
    -------
    area : float
        Total 3D surface area in mm^2.
    """
    H, W = depth.shape
    if mask is None:
        mask = depth > 0

    # Valid quads: all 4 corners must be valid
    valid = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]

    # 3D positions of quad corners
    y_grid, x_grid = np.mgrid[:H, :W]
    x3d = x_grid.astype(np.float64) * pixel_size
    y3d = y_grid.astype(np.float64) * pixel_size
    z = depth.astype(np.float64)

    # Triangle 1: (i,j), (i,j+1), (i+1,j)
    v1x = x3d[:-1, 1:] - x3d[:-1, :-1]
    v1y = y3d[:-1, 1:] - y3d[:-1, :-1]
    v1z = z[:-1, 1:] - z[:-1, :-1]
    v2x = x3d[1:, :-1] - x3d[:-1, :-1]
    v2y = y3d[1:, :-1] - y3d[:-1, :-1]
    v2z = z[1:, :-1] - z[:-1, :-1]

    cross1 = np.sqrt(
        (v1y * v2z - v1z * v2y) ** 2
        + (v1z * v2x - v1x * v2z) ** 2
        + (v1x * v2y - v1y * v2x) ** 2
    )

    # Triangle 2: (i,j+1), (i+1,j+1), (i+1,j)
    v3x = x3d[1:, 1:] - x3d[:-1, 1:]
    v3y = y3d[1:, 1:] - y3d[:-1, 1:]
    v3z = z[1:, 1:] - z[:-1, 1:]
    v4x = x3d[1:, :-1] - x3d[:-1, 1:]
    v4y = y3d[1:, :-1] - y3d[:-1, 1:]
    v4z = z[1:, :-1] - z[:-1, 1:]

    cross2 = np.sqrt(
        (v3y * v4z - v3z * v4y) ** 2
        + (v3z * v4x - v3x * v4z) ** 2
        + (v3x * v4y - v3y * v4x) ** 2
    )

    total = 0.5 * np.sum((cross1 + cross2) * valid)
    return float(total)
