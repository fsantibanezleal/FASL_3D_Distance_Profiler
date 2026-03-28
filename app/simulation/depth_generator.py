"""
Synthetic RGB-D Scene Generator
================================

Generates synthetic test data simulating output from depth cameras such as
SICK Ranger3D line-scan profilers or Intel RealSense D400 stereo cameras.

Each scene produces a triplet:
    - RGB image  (H, W, 3) uint8   -- color texture
    - Depth map  (H, W)   float32  -- distance in mm (0 = no data, >0 = valid)
    - Normal map (H, W, 3) float32 -- unit surface normals computed from depth

Supported scene types:
    1. gaussian_hills   -- Multiple Gaussian bumps on a flat plane
    2. terrain          -- Perlin-like noise terrain via octave sums of sinusoids
    3. object_on_table  -- Sphere or cube sitting on a flat surface
    4. conveyor_belt    -- Industrial profiling: objects on a moving belt
    5. wave_surface     -- Sinusoidal ripple patterns

Mathematical Background
-----------------------
Gaussian hill at centre (x0, y0) with amplitude A and spread sigma:

    z(x, y) = A * exp(-((x - x0)^2 + (y - y0)^2) / (2 * sigma^2))

Perlin-like terrain via octave summation of sinusoidal basis functions:

    z(x, y) = sum_{k=0}^{N-1} (persistence^k) * sin(frequency * 2^k * x + phase_k)
                                                * sin(frequency * 2^k * y + phase_k')

Surface normals from depth gradient (finite differences):

    nx = -dz/dx,  ny = -dz/dy,  nz = 1
    n  = (nx, ny, nz) / ||(nx, ny, nz)||

References
----------
- Perlin, K. (1985). An Image Synthesizer. SIGGRAPH.
- Intel RealSense D400 Series datasheet.
- SICK Ranger3D industrial 3D camera specifications.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SCENE_TYPES = [
    "gaussian_hills",
    "terrain",
    "object_on_table",
    "conveyor_belt",
    "wave_surface",
]


def generate_scene(
    scene_type: str = "gaussian_hills",
    width: int = 256,
    height: int = 256,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic RGB-D scene.

    Parameters
    ----------
    scene_type : str
        One of ``SCENE_TYPES``.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    rgb : ndarray, shape (H, W, 3), dtype uint8
        Colour texture image.
    depth : ndarray, shape (H, W), dtype float32
        Depth map in millimetres (0 = invalid/no data).
    normals : ndarray, shape (H, W, 3), dtype float32
        Unit surface normals at each pixel.

    Raises
    ------
    ValueError
        If *scene_type* is not recognised.
    """
    if scene_type not in SCENE_TYPES:
        raise ValueError(
            f"Unknown scene_type '{scene_type}'. Choose from {SCENE_TYPES}"
        )

    rng = np.random.default_rng(seed)

    generator = {
        "gaussian_hills": _generate_gaussian_hills,
        "terrain": _generate_terrain,
        "object_on_table": _generate_object_on_table,
        "conveyor_belt": _generate_conveyor_belt,
        "wave_surface": _generate_wave_surface,
    }[scene_type]

    depth = generator(width, height, rng)
    normals = _compute_normals_from_depth(depth)
    rgb = _depth_to_rgb_texture(depth, rng)

    return rgb, depth.astype(np.float32), normals.astype(np.float32)


def list_scene_types() -> list:
    """Return available scene type identifiers."""
    return list(SCENE_TYPES)


# ---------------------------------------------------------------------------
# Scene generators (private)
# ---------------------------------------------------------------------------

def _generate_gaussian_hills(
    w: int, h: int, rng: np.random.Generator
) -> np.ndarray:
    """Multiple Gaussian bumps on a flat plane.

    z(x, y) = base + sum_i A_i * exp(-((x-x0_i)^2 + (y-y0_i)^2) / (2*sigma_i^2))

    Parameters
    ----------
    w, h : int
        Width and height.
    rng : Generator
        Numpy random generator.

    Returns
    -------
    depth : ndarray (h, w) float64
        Depth values in mm.
    """
    base_depth = 500.0  # mm
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    depth = np.full((h, w), base_depth, dtype=np.float64)

    n_hills = rng.integers(3, 8)
    for _ in range(n_hills):
        x0 = rng.uniform(0.1, 0.9)
        y0 = rng.uniform(0.1, 0.9)
        amplitude = rng.uniform(30.0, 120.0)
        sigma = rng.uniform(0.05, 0.2)
        depth -= amplitude * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2)
        )

    return depth


def _generate_terrain(
    w: int, h: int, rng: np.random.Generator
) -> np.ndarray:
    """Perlin-like noise terrain using octave summation of sinusoidal bases.

    z(x, y) = base + sum_{k=0}^{octaves-1} persistence^k
              * sin(freq * 2^k * x + phi_k) * sin(freq * 2^k * y + psi_k)

    Parameters
    ----------
    w, h : int
        Width and height.
    rng : Generator
        Numpy random generator.

    Returns
    -------
    depth : ndarray (h, w) float64
    """
    base_depth = 500.0
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)

    terrain = np.zeros((h, w), dtype=np.float64)
    octaves = 5
    persistence = 0.5
    base_freq = 1.0

    for k in range(octaves):
        freq = base_freq * (2 ** k)
        amp = persistence ** k
        phi = rng.uniform(0, 2 * np.pi)
        psi = rng.uniform(0, 2 * np.pi)
        terrain += amp * np.sin(freq * xx + phi) * np.sin(freq * yy + psi)

    # Normalise to depth range
    terrain -= terrain.min()
    terrain = terrain / (terrain.max() + 1e-8) * 150.0  # 0..150 mm relief

    depth = base_depth - terrain
    return depth


def _generate_object_on_table(
    w: int, h: int, rng: np.random.Generator
) -> np.ndarray:
    """A sphere sitting on a flat table surface.

    The sphere is modelled as:
        z_sphere(x, y) = table_z - sqrt(R^2 - (x-cx)^2 - (y-cy)^2)
    where R is the sphere radius and (cx, cy) its centre.

    Parameters
    ----------
    w, h : int
        Width and height.
    rng : Generator
        Numpy random generator.

    Returns
    -------
    depth : ndarray (h, w) float64
    """
    table_depth = 500.0  # mm
    depth = np.full((h, w), table_depth, dtype=np.float64)

    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    # Sphere parameters
    cx = rng.uniform(0.3, 0.7)
    cy = rng.uniform(0.3, 0.7)
    radius_norm = rng.uniform(0.1, 0.25)  # normalised radius
    radius_mm = rng.uniform(40, 80)       # mm height above table

    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist_sq < radius_norm ** 2

    sphere_height = np.sqrt(
        np.maximum(radius_norm ** 2 - dist_sq, 0)
    ) / radius_norm * radius_mm

    depth[mask] = table_depth - sphere_height[mask]

    return depth


def _generate_conveyor_belt(
    w: int, h: int, rng: np.random.Generator
) -> np.ndarray:
    """Industrial conveyor belt scene with multiple objects.

    Simulates a SICK Ranger3D line-scan scenario: flat belt surface
    with rectangular and circular objects of varying heights.

    Parameters
    ----------
    w, h : int
        Width and height.
    rng : Generator
        Numpy random generator.

    Returns
    -------
    depth : ndarray (h, w) float64
    """
    belt_depth = 600.0  # mm
    depth = np.full((h, w), belt_depth, dtype=np.float64)

    # Add slight belt texture (noise)
    depth += rng.normal(0, 0.5, (h, w))

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    n_objects = rng.integers(3, 7)
    for _ in range(n_objects):
        obj_type = rng.choice(["box", "cylinder"])
        cx = rng.integers(w // 6, 5 * w // 6)
        cy = rng.integers(h // 6, 5 * h // 6)
        obj_height = rng.uniform(20, 100)  # mm above belt

        if obj_type == "box":
            half_w = rng.integers(w // 16, w // 6)
            half_h = rng.integers(h // 16, h // 6)
            mask = (
                (xx >= cx - half_w)
                & (xx <= cx + half_w)
                & (yy >= cy - half_h)
                & (yy <= cy + half_h)
            )
            depth[mask] = belt_depth - obj_height
        else:
            radius = rng.integers(min(w, h) // 16, min(w, h) // 6)
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = dist_sq < radius ** 2
            # Dome shape
            dome = np.sqrt(np.maximum(radius ** 2 - dist_sq, 0)) / radius * obj_height
            depth[mask] = belt_depth - dome[mask]

    return depth


def _generate_wave_surface(
    w: int, h: int, rng: np.random.Generator
) -> np.ndarray:
    """Sinusoidal ripple surface.

    z(x, y) = base + A1*sin(k1*x + phi1) + A2*sin(k2*y + phi2)
              + A3*sin(k3*(x+y) + phi3)

    Parameters
    ----------
    w, h : int
        Width and height.
    rng : Generator
        Numpy random generator.

    Returns
    -------
    depth : ndarray (h, w) float64
    """
    base_depth = 500.0
    x = np.linspace(0, 6 * np.pi, w)
    y = np.linspace(0, 6 * np.pi, h)
    xx, yy = np.meshgrid(x, y)

    waves = np.zeros((h, w), dtype=np.float64)
    for _ in range(3):
        amp = rng.uniform(15, 50)
        kx = rng.uniform(0.5, 3.0)
        ky = rng.uniform(0.5, 3.0)
        phi = rng.uniform(0, 2 * np.pi)
        waves += amp * np.sin(kx * xx + ky * yy + phi)

    depth = base_depth - waves
    return depth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_normals_from_depth(
    depth: np.ndarray, pixel_size: float = 1.0
) -> np.ndarray:
    """Compute surface normals from a depth map via finite differences.

    Given a depth map z(u, v), the surface normal at each pixel is:

        nx = -dz/dx
        ny = -dz/dy
        nz = 1.0

    The vector (nx, ny, nz) is then normalised to unit length:

        n = (nx, ny, nz) / sqrt(nx^2 + ny^2 + nz^2)

    Central differences are used for interior pixels. At boundaries,
    forward/backward differences are used via ``np.gradient``.

    Parameters
    ----------
    depth : ndarray (H, W)
        Depth map in mm.
    pixel_size : float
        Physical size of one pixel (for correct gradient scaling).

    Returns
    -------
    normals : ndarray (H, W, 3) float64
        Unit surface normals.  Convention: z-axis points towards camera.
    """
    dz_dy, dz_dx = np.gradient(depth, pixel_size)
    normals = np.zeros((*depth.shape, 3), dtype=np.float64)
    normals[..., 0] = -dz_dx
    normals[..., 1] = -dz_dy
    normals[..., 2] = 1.0

    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    mag = np.maximum(mag, 1e-8)
    normals /= mag

    return normals


def _depth_to_rgb_texture(
    depth: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Generate a plausible RGB texture from a depth map.

    Creates a colour image by mapping depth to a base colour via a simple
    gradient, then adding per-pixel noise to simulate surface texture.

    Parameters
    ----------
    depth : ndarray (H, W)
        Depth map.
    rng : Generator
        Numpy random generator.

    Returns
    -------
    rgb : ndarray (H, W, 3) uint8
    """
    h, w = depth.shape
    d_min = depth[depth > 0].min() if (depth > 0).any() else 0
    d_max = depth.max()
    d_range = d_max - d_min if d_max > d_min else 1.0

    normalised = np.clip((depth - d_min) / d_range, 0, 1)

    rgb = np.zeros((h, w, 3), dtype=np.float64)
    # Earthy colour gradient: brown (close) -> green (far)
    rgb[..., 0] = 0.4 + 0.4 * normalised  # R
    rgb[..., 1] = 0.5 + 0.3 * (1.0 - normalised)  # G
    rgb[..., 2] = 0.3 + 0.2 * normalised  # B

    # Add texture noise
    noise = rng.normal(0, 0.03, (h, w, 3))
    rgb = np.clip(rgb + noise, 0, 1)

    return (rgb * 255).astype(np.uint8)
