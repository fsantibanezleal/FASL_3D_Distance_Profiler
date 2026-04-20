"""
Tests for depth_processing module
===================================

Covers:
    - bilateral_filter_depth: edge preservation, noise reduction
    - fill_holes: small hole interpolation
    - compute_normals: unit normal vectors, flat surface test
    - depth_to_point_cloud: pinhole projection correctness
"""

import sys
import os
import unittest
import warnings
import numpy as np

# Ensure app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.depth_processing import (
    bilateral_filter_depth,
    fill_holes,
    compute_normals,
    depth_to_point_cloud,
    align_depth_to_rgb,
)


class TestBilateralFilter(unittest.TestCase):
    """Test bilateral filter on depth maps."""

    def test_flat_surface_unchanged(self):
        """A perfectly flat depth map should remain unchanged after filtering."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        filtered = bilateral_filter_depth(depth, sigma_spatial=3, sigma_range=10)
        np.testing.assert_allclose(filtered, depth, atol=0.1)

    def test_noise_reduction(self):
        """Filter should reduce additive Gaussian noise."""
        rng = np.random.default_rng(42)
        base = np.full((64, 64), 500.0, dtype=np.float32)
        noisy = base + rng.normal(0, 5, (64, 64)).astype(np.float32)
        filtered = bilateral_filter_depth(noisy, sigma_spatial=3, sigma_range=10)
        noise_before = np.std(noisy - base)
        noise_after = np.std(filtered - base)
        self.assertLess(noise_after, noise_before,
                        "Bilateral filter should reduce noise")

    def test_edge_preservation(self):
        """A sharp step edge should be preserved (not blurred away)."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        depth[:, 32:] = 400.0  # sharp step at column 32
        filtered = bilateral_filter_depth(depth, sigma_spatial=3, sigma_range=5)
        # Left side should stay near 500, right near 400
        self.assertGreater(filtered[32, 5], 490.0)
        self.assertLess(filtered[32, 60], 410.0)

    def test_zero_pixels_remain_zero(self):
        """Invalid (zero) pixels should remain zero after filtering."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        depth[10:15, 10:15] = 0.0
        filtered = bilateral_filter_depth(depth, sigma_spatial=3, sigma_range=10)
        self.assertTrue(np.all(filtered[10:15, 10:15] == 0.0))

    def test_output_dtype(self):
        """Output should be float32."""
        depth = np.full((32, 32), 500.0, dtype=np.float64)
        filtered = bilateral_filter_depth(depth)
        self.assertEqual(filtered.dtype, np.float32)

    def test_no_runtime_warning_on_zero_border(self):
        """Zero-border pixels must not raise a divide-by-zero RuntimeWarning.

        Regression test for issue #18: ``np.where(weight_sum > 0, ...)``
        still evaluates the division on zero-weight pixels, so an
        ``np.errstate`` scope must silence the warning.
        """
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        # Invalid border — every pixel within 5 px of the edge is zero,
        # which creates zero-weight rows once the kernel reaches them.
        depth[:5, :] = 0.0
        depth[-5:, :] = 0.0
        depth[:, :5] = 0.0
        depth[:, -5:] = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Should not raise
            filtered = bilateral_filter_depth(
                depth, sigma_spatial=3, sigma_range=10
            )
        # Interior still valid
        self.assertGreater(filtered[16, 16], 490.0)
        # Border still zero
        self.assertTrue(np.all(filtered[:5, :] == 0.0))


class TestFillHoles(unittest.TestCase):
    """Test hole-filling via nearest-neighbour interpolation."""

    def test_no_holes(self):
        """If there are no holes, output should equal input."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        filled = fill_holes(depth)
        np.testing.assert_array_equal(filled, depth)

    def test_small_hole_filled(self):
        """A small hole (single pixel) should be filled."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        depth[16, 16] = 0.0
        filled = fill_holes(depth, max_hole_size=5)
        self.assertGreater(filled[16, 16], 0.0)
        self.assertAlmostEqual(float(filled[16, 16]), 500.0, places=0)

    def test_large_hole_not_filled(self):
        """A hole larger than max_hole_size should NOT be filled."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        depth[10:40, 10:40] = 0.0  # 30x30 = 900 pixels
        filled = fill_holes(depth, max_hole_size=5)  # max = 25 pixels
        # Large hole should remain zero
        self.assertTrue(np.all(filled[15:35, 15:35] == 0.0))

    def test_output_dtype(self):
        """Output should be float32."""
        depth = np.full((16, 16), 500.0, dtype=np.float64)
        filled = fill_holes(depth)
        self.assertEqual(filled.dtype, np.float32)


class TestComputeNormals(unittest.TestCase):
    """Test surface normal computation from depth."""

    def test_flat_surface_normals(self):
        """A flat surface should have normals pointing straight up: (0, 0, 1)."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        normals = compute_normals(depth)
        self.assertEqual(normals.shape, (64, 64, 3))
        # Interior normals should be approximately (0, 0, 1)
        interior = normals[10:54, 10:54]
        np.testing.assert_allclose(interior[..., 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(interior[..., 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(interior[..., 2], 1.0, atol=1e-5)

    def test_normals_are_unit_vectors(self):
        """All normals should have magnitude 1."""
        rng = np.random.default_rng(0)
        depth = 500.0 + rng.normal(0, 10, (64, 64)).astype(np.float32)
        depth[depth <= 0] = 1.0
        normals = compute_normals(depth)
        mags = np.linalg.norm(normals, axis=-1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-5)

    def test_tilted_surface(self):
        """A surface tilted in x should have normals with negative x component."""
        x = np.linspace(0, 100, 64)
        y = np.linspace(0, 0, 64)
        xx, _ = np.meshgrid(x, y)
        depth = 500.0 - xx  # tilts: deeper on left, shallower on right
        normals = compute_normals(depth.astype(np.float32), pixel_size=1.0)
        # dz/dx ~ 100/64 > 0, so nx = -dz/dx < 0... wait, depth decreases
        # Actually depth = 500 - x, so dz/dx = -100/63, nx = -(-100/63) > 0
        interior_nx = normals[10:54, 10:54, 0]
        self.assertTrue(np.mean(interior_nx) > 0,
                        "Normal x-component should be positive for this tilt")

    def test_output_dtype(self):
        """Output should be float32."""
        depth = np.full((16, 16), 500.0, dtype=np.float32)
        normals = compute_normals(depth)
        self.assertEqual(normals.dtype, np.float32)


class TestDepthToPointCloud(unittest.TestCase):
    """Test depth-to-point-cloud conversion using pinhole model."""

    def test_centre_pixel(self):
        """The centre pixel should project to (0, 0, Z)."""
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[50, 50] = 1000.0  # 1000 mm at centre
        points, colors = depth_to_point_cloud(depth, fx=500, fy=500)
        self.assertEqual(points.shape[0], 1)
        np.testing.assert_allclose(points[0, 0], 0.0, atol=5.0)
        np.testing.assert_allclose(points[0, 1], 0.0, atol=5.0)
        np.testing.assert_allclose(points[0, 2], 1000.0, atol=0.1)

    def test_zero_depth_excluded(self):
        """Zero-depth pixels should not appear in the point cloud."""
        depth = np.zeros((32, 32), dtype=np.float32)
        depth[5, 5] = 100.0
        depth[10, 10] = 200.0
        points, _ = depth_to_point_cloud(depth)
        self.assertEqual(points.shape[0], 2)

    def test_color_assignment(self):
        """Colours should be correctly assigned from the RGB image."""
        depth = np.zeros((32, 32), dtype=np.float32)
        depth[16, 16] = 500.0
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[16, 16] = [255, 128, 0]
        points, colors = depth_to_point_cloud(depth, rgb=rgb)
        self.assertEqual(colors.shape[0], 1)
        np.testing.assert_allclose(colors[0], [1.0, 128/255.0, 0.0], atol=0.01)

    def test_no_rgb_returns_none(self):
        """If no RGB is provided, colours should be None."""
        depth = np.full((16, 16), 500.0, dtype=np.float32)
        _, colors = depth_to_point_cloud(depth)
        self.assertIsNone(colors)


class TestAlignDepthToRgb(unittest.TestCase):
    """Test depth-to-RGB alignment (resize)."""

    def test_same_size(self):
        """Same-size inputs should return unchanged depth."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        aligned = align_depth_to_rgb(depth, rgb)
        np.testing.assert_array_equal(aligned, depth)

    def test_upscale(self):
        """Depth should be upscaled to match RGB resolution."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        aligned = align_depth_to_rgb(depth, rgb)
        self.assertEqual(aligned.shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
