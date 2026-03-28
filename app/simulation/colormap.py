"""
Depth-to-Colour Mapping and Overlay Utilities
===============================================

Provides colourmap application for depth map visualisation and compositing
of depth overlays onto RGB images.

Supported Colourmap Schemes
----------------------------

1. **Hot** (black -> red -> yellow -> white)
   Linear interpolation through control points mapping normalised depth
   [0, 1] to RGB:
       0.00 -> (0, 0, 0)
       0.33 -> (1, 0, 0)
       0.66 -> (1, 1, 0)
       1.00 -> (1, 1, 1)

2. **Viridis** (dark purple -> teal -> yellow)
   Perceptually uniform colourmap (Matplotlib default).  Approximated
   here via cubic polynomial fits of the RGB channels.

3. **Jet** (blue -> cyan -> green -> yellow -> red)
   Classic rainbow colourmap with control points:
       0.00 -> (0, 0, 0.5)
       0.11 -> (0, 0, 1)
       0.35 -> (0, 1, 1)
       0.50 -> (0, 1, 0)
       0.65 -> (1, 1, 0)
       0.89 -> (1, 0, 0)
       1.00 -> (0.5, 0, 0)

4. **Greyscale** (black -> white)

References
----------
- Moreland, K. (2009). Diverging Color Maps for Scientific Visualization.
- van der Walt, S. & Smith, N. (2015). A Better Default Colormap for
  Matplotlib. SciPy Conference.
"""

import numpy as np
from typing import Tuple, Optional


COLOURMAP_NAMES = ["hot", "viridis", "jet", "greyscale"]


def apply_colormap(
    depth: np.ndarray,
    colormap: str = "viridis",
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
) -> np.ndarray:
    """Map a depth image to an RGB colour image using the specified colourmap.

    Depth values are first normalised to [0, 1] using:

        t = (depth - d_min) / (d_max - d_min)

    then mapped through the selected colourmap lookup.  Invalid (zero)
    depth pixels are rendered as black.

    Parameters
    ----------
    depth : ndarray (H, W), float
        Depth map.
    colormap : str
        One of 'hot', 'viridis', 'jet', 'greyscale'.
    d_min, d_max : float or None
        Depth range for normalisation.  If None, auto-detected from
        valid (non-zero) pixels.

    Returns
    -------
    rgb : ndarray (H, W, 3), uint8
        Colourised depth image.
    """
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if d_min is None:
        d_min = float(depth[valid].min())
    if d_max is None:
        d_max = float(depth[valid].max())

    d_range = d_max - d_min if d_max > d_min else 1.0
    t = np.clip((depth.astype(np.float64) - d_min) / d_range, 0, 1)

    cmap_func = {
        "hot": _cmap_hot,
        "viridis": _cmap_viridis,
        "jet": _cmap_jet,
        "greyscale": _cmap_greyscale,
    }.get(colormap, _cmap_viridis)

    rgb_float = cmap_func(t)
    rgb_float[~valid] = 0.0

    return (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)


def overlay_depth_on_rgb(
    rgb: np.ndarray,
    depth: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Composite a colourised depth overlay onto an RGB image.

    Blending formula (alpha compositing):

        out = alpha * depth_colour + (1 - alpha) * rgb

    Only valid (non-zero) depth pixels are blended; where depth is zero,
    the original RGB pixel is preserved.

    Parameters
    ----------
    rgb : ndarray (H, W, 3), uint8
        Original colour image.
    depth : ndarray (H, W), float
        Depth map.
    colormap : str
        Colourmap for depth rendering.
    alpha : float
        Blending weight for the depth overlay (0 = invisible, 1 = opaque).

    Returns
    -------
    blended : ndarray (H, W, 3), uint8
    """
    depth_colour = apply_colormap(depth, colormap=colormap)
    valid = depth > 0

    blended = rgb.astype(np.float64).copy()
    dc = depth_colour.astype(np.float64)

    mask3 = np.stack([valid] * 3, axis=-1)
    blended[mask3] = alpha * dc[mask3] + (1 - alpha) * blended[mask3]

    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal colourmap functions
# ---------------------------------------------------------------------------

def _cmap_hot(t: np.ndarray) -> np.ndarray:
    """Hot colourmap: black -> red -> yellow -> white.

    Piecewise linear with breakpoints at t = 0, 1/3, 2/3, 1.
    """
    rgb = np.zeros((*t.shape, 3), dtype=np.float64)
    rgb[..., 0] = np.clip(t * 3, 0, 1)
    rgb[..., 1] = np.clip(t * 3 - 1, 0, 1)
    rgb[..., 2] = np.clip(t * 3 - 2, 0, 1)
    return rgb


def _cmap_viridis(t: np.ndarray) -> np.ndarray:
    """Viridis-like colourmap approximated via cubic polynomials.

    Perceptually uniform, colourblind-friendly.  Polynomial coefficients
    are fitted to the Matplotlib viridis LUT.
    """
    rgb = np.zeros((*t.shape, 3), dtype=np.float64)
    # Approximate cubic fits
    rgb[..., 0] = 0.267 + 0.004 * t + 1.256 * t ** 2 - 0.527 * t ** 3
    rgb[..., 1] = 0.004 + 1.517 * t - 1.428 * t ** 2 + 0.908 * t ** 3
    rgb[..., 2] = 0.329 + 1.442 * t - 3.456 * t ** 2 + 2.165 * t ** 3
    return np.clip(rgb, 0, 1)


def _cmap_jet(t: np.ndarray) -> np.ndarray:
    """Jet colourmap: blue -> cyan -> green -> yellow -> red.

    Piecewise linear mapping with five control segments.
    """
    rgb = np.zeros((*t.shape, 3), dtype=np.float64)

    # Red channel
    rgb[..., 0] = np.where(
        t < 0.35, 0.0,
        np.where(t < 0.65, (t - 0.35) / 0.3,
                 np.where(t < 0.89, 1.0, 1.0 - (t - 0.89) / 0.11 * 0.5))
    )
    # Green channel
    rgb[..., 1] = np.where(
        t < 0.11, 0.0,
        np.where(t < 0.35, (t - 0.11) / 0.24,
                 np.where(t < 0.65, 1.0,
                          np.where(t < 0.89, 1.0 - (t - 0.65) / 0.24, 0.0)))
    )
    # Blue channel
    rgb[..., 2] = np.where(
        t < 0.11, 0.5 + t / 0.11 * 0.5,
        np.where(t < 0.35, 1.0,
                 np.where(t < 0.65, 1.0 - (t - 0.35) / 0.3, 0.0))
    )
    return np.clip(rgb, 0, 1)


def _cmap_greyscale(t: np.ndarray) -> np.ndarray:
    """Simple linear greyscale mapping."""
    rgb = np.zeros((*t.shape, 3), dtype=np.float64)
    rgb[..., 0] = t
    rgb[..., 1] = t
    rgb[..., 2] = t
    return rgb
