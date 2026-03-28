"""
Tests for surface_reconstruction module
=========================================

Covers:
    - depth_to_mesh / depth_to_mesh_fast: vertex and face generation
    - compute_surface_curvature: Gaussian and mean curvature
    - extract_cross_section: bilinear interpolation along a line
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.surface_reconstruction import (
    depth_to_mesh,
    depth_to_mesh_fast,
    compute_surface_curvature,
    extract_cross_section,
)


class TestDepthToMesh(unittest.TestCase):
    """Test triangle mesh generation from depth maps."""

    def test_simple_mesh(self):
        """A 4x4 flat depth map should produce a valid mesh."""
        depth = np.full((4, 4), 500.0, dtype=np.float32)
        mesh = depth_to_mesh(depth, subsample=1)
        self.assertGreater(mesh["n_vertices"], 0)
        self.assertGreater(mesh["n_faces"], 0)
        # 4x4 grid = 16 vertices, (3x3)*2 = 18 faces
        self.assertEqual(mesh["n_vertices"], 16)
        self.assertEqual(mesh["n_faces"], 18)

    def test_zero_depth_excluded(self):
        """Vertices with zero depth should be excluded from the mesh."""
        depth = np.full((4, 4), 500.0, dtype=np.float32)
        depth[0, 0] = 0.0
        mesh = depth_to_mesh(depth, subsample=1)
        self.assertEqual(mesh["n_vertices"], 15)

    def test_colors_present(self):
        """Per-vertex colours should match the number of vertices."""
        depth = np.full((8, 8), 500.0, dtype=np.float32)
        rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
        mesh = depth_to_mesh(depth, rgb=rgb, subsample=1)
        self.assertEqual(len(mesh["colors"]), mesh["n_vertices"])

    def test_subsample_reduces_vertices(self):
        """Subsampling should reduce the vertex count."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        mesh1 = depth_to_mesh(depth, subsample=1)
        mesh2 = depth_to_mesh(depth, subsample=2)
        self.assertGreater(mesh1["n_vertices"], mesh2["n_vertices"])


class TestDepthToMeshFast(unittest.TestCase):
    """Test the vectorised mesh generation path."""

    def test_matches_slow_path(self):
        """Fast and slow paths should produce the same vertex count."""
        depth = np.full((16, 16), 500.0, dtype=np.float32)
        mesh_slow = depth_to_mesh(depth, subsample=1)
        mesh_fast = depth_to_mesh_fast(depth, subsample=1)
        self.assertEqual(mesh_slow["n_vertices"], mesh_fast["n_vertices"])
        self.assertEqual(mesh_slow["n_faces"], mesh_fast["n_faces"])

    def test_fast_with_rgb(self):
        """Fast path should handle RGB colouring."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        mesh = depth_to_mesh_fast(depth, rgb=rgb, subsample=2)
        self.assertEqual(len(mesh["colors"]), mesh["n_vertices"])


class TestSurfaceCurvature(unittest.TestCase):
    """Test Gaussian and mean curvature computation."""

    def test_flat_surface_zero_curvature(self):
        """A flat surface should have zero Gaussian and mean curvature."""
        depth = np.full((64, 64), 500.0, dtype=np.float64)
        K, H = compute_surface_curvature(depth)
        np.testing.assert_allclose(K, 0.0, atol=1e-10)
        np.testing.assert_allclose(H, 0.0, atol=1e-10)

    def test_sphere_positive_gaussian(self):
        """A sphere (bowl shape) should have positive Gaussian curvature."""
        x = np.linspace(-1, 1, 64)
        y = np.linspace(-1, 1, 64)
        xx, yy = np.meshgrid(x, y)
        r2 = xx ** 2 + yy ** 2
        depth = 500.0 + 50.0 * r2  # paraboloid (bowl)
        K, H = compute_surface_curvature(depth, pixel_size=2.0 / 63)
        # Interior should have positive K
        K_interior = K[16:48, 16:48]
        self.assertGreater(np.mean(K_interior), 0,
                           "Bowl surface should have positive Gaussian curvature")

    def test_output_shapes(self):
        """Output arrays should match input shape."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        K, H = compute_surface_curvature(depth)
        self.assertEqual(K.shape, (32, 32))
        self.assertEqual(H.shape, (32, 32))

    def test_output_dtype(self):
        """Output should be float32."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        K, H = compute_surface_curvature(depth)
        self.assertEqual(K.dtype, np.float32)
        self.assertEqual(H.dtype, np.float32)


class TestExtractCrossSection(unittest.TestCase):
    """Test cross-section profile extraction."""

    def test_horizontal_profile(self):
        """A horizontal profile across a flat surface should be constant."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        distances, heights = extract_cross_section(
            depth, start_point=(5, 32), end_point=(58, 32), num_samples=50
        )
        self.assertEqual(len(distances), 50)
        self.assertEqual(len(heights), 50)
        valid = heights[~np.isnan(heights)]
        np.testing.assert_allclose(valid, 500.0, atol=0.1)

    def test_sloped_profile(self):
        """A profile across a tilted surface should show a linear slope."""
        x = np.arange(64)
        depth = np.tile(500.0 + x * 2.0, (64, 1)).astype(np.float32)
        distances, heights = extract_cross_section(
            depth, start_point=(10, 32), end_point=(50, 32), num_samples=100
        )
        valid = heights[~np.isnan(heights)]
        # Should increase monotonically
        diffs = np.diff(valid)
        self.assertTrue(np.all(diffs >= -0.5),
                        "Sloped profile should be monotonically increasing")

    def test_out_of_bounds_returns_nan(self):
        """Points outside the image should be NaN."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        _, heights = extract_cross_section(
            depth, start_point=(-10, 16), end_point=(40, 16), num_samples=50
        )
        # Some points should be NaN (out of bounds)
        self.assertTrue(np.any(np.isnan(heights)))


if __name__ == "__main__":
    unittest.main()
