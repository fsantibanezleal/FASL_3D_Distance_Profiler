"""
Tests for measurement tools in profile_analysis module
========================================================

Covers:
    - measure_distance_3d: Euclidean distance between 3D points
    - measure_angle_between_normals: angle between surface normal vectors
    - measure_area: vectorised 3D surface area computation
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.profile_analysis import (
    measure_distance_3d,
    measure_angle_between_normals,
    measure_area,
)


class TestMeasureDistance3D(unittest.TestCase):
    """Test 3D Euclidean distance measurement."""

    def test_same_point_zero(self):
        """Distance from a point to itself should be zero."""
        p = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(measure_distance_3d(p, p), 0.0)

    def test_unit_distance(self):
        """Distance along one axis should equal axis displacement."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(measure_distance_3d(a, b), 1.0)

    def test_diagonal(self):
        """Distance from origin to (1,1,1) should be sqrt(3)."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(measure_distance_3d(a, b), np.sqrt(3), places=10)

    def test_known_distance(self):
        """Test a known 3-4-5 triangle in 3D."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        self.assertAlmostEqual(measure_distance_3d(a, b), 5.0)

    def test_negative_coordinates(self):
        """Distance should work with negative coordinates."""
        a = np.array([-1.0, -2.0, -3.0])
        b = np.array([1.0, 2.0, 3.0])
        expected = np.sqrt(4 + 16 + 36)
        self.assertAlmostEqual(measure_distance_3d(a, b), expected)

    def test_symmetry(self):
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 7.0])
        self.assertAlmostEqual(measure_distance_3d(a, b), measure_distance_3d(b, a))

    def test_returns_float(self):
        """Return type should be float."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        self.assertIsInstance(measure_distance_3d(a, b), float)


class TestMeasureAngleBetweenNormals(unittest.TestCase):
    """Test angle measurement between surface normals."""

    def test_parallel_normals_zero_angle(self):
        """Parallel normals should have near-zero angle."""
        n = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(measure_angle_between_normals(n, n), 0.0, places=2)

    def test_perpendicular_normals_90(self):
        """Perpendicular normals should have 90 degree angle."""
        na = np.array([1.0, 0.0, 0.0])
        nb = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(measure_angle_between_normals(na, nb), 90.0, places=5)

    def test_opposite_normals_180(self):
        """Opposite normals should have near-180 degree angle."""
        na = np.array([0.0, 0.0, 1.0])
        nb = np.array([0.0, 0.0, -1.0])
        self.assertAlmostEqual(measure_angle_between_normals(na, nb), 180.0, places=2)

    def test_45_degree_angle(self):
        """Test a known 45 degree angle."""
        na = np.array([1.0, 0.0, 0.0])
        nb = np.array([1.0, 1.0, 0.0])  # 45 degrees from x-axis in xy-plane
        self.assertAlmostEqual(measure_angle_between_normals(na, nb), 45.0, places=4)

    def test_non_unit_normals(self):
        """Should work with non-unit-length normals."""
        na = np.array([2.0, 0.0, 0.0])
        nb = np.array([0.0, 3.0, 0.0])
        self.assertAlmostEqual(measure_angle_between_normals(na, nb), 90.0, places=4)

    def test_symmetry(self):
        """Angle should be symmetric: angle(a,b) == angle(b,a)."""
        na = np.array([1.0, 1.0, 0.0])
        nb = np.array([0.0, 1.0, 1.0])
        self.assertAlmostEqual(
            measure_angle_between_normals(na, nb),
            measure_angle_between_normals(nb, na),
        )

    def test_returns_float(self):
        """Return type should be float."""
        na = np.array([1, 0, 0])
        nb = np.array([0, 1, 0])
        self.assertIsInstance(measure_angle_between_normals(na, nb), float)


class TestMeasureArea(unittest.TestCase):
    """Test vectorised surface area computation."""

    def test_flat_surface_area(self):
        """A flat surface area should equal (H-1)*(W-1)*pixel_size^2.

        For a 10x10 grid with pixel_size=1, the flat area is 9*9 = 81.
        Each quad contributes 2 triangles of area 0.5 each = 1.0 per quad.
        """
        depth = np.full((10, 10), 100.0, dtype=np.float64)
        area = measure_area(depth, pixel_size=1.0)
        expected = 9 * 9 * 1.0  # 81 mm^2
        self.assertAlmostEqual(area, expected, places=5)

    def test_flat_surface_with_pixel_size(self):
        """Pixel size should scale the area quadratically."""
        depth = np.full((10, 10), 100.0, dtype=np.float64)
        area = measure_area(depth, pixel_size=2.0)
        expected = 9 * 9 * 4.0  # 324 mm^2
        self.assertAlmostEqual(area, expected, places=5)

    def test_tilted_surface_larger_than_flat(self):
        """A tilted surface should have larger area than its flat projection."""
        H, W = 20, 20
        depth_flat = np.full((H, W), 100.0, dtype=np.float64)
        # Create a sloped surface
        x = np.arange(W, dtype=np.float64)
        depth_slope = np.tile(100.0 + x * 5.0, (H, 1))

        area_flat = measure_area(depth_flat, pixel_size=1.0)
        area_slope = measure_area(depth_slope, pixel_size=1.0)
        self.assertGreater(area_slope, area_flat)

    def test_mask_restricts_area(self):
        """A mask should restrict the area computation."""
        depth = np.full((20, 20), 100.0, dtype=np.float64)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True  # 10x10 region
        area = measure_area(depth, mask=mask, pixel_size=1.0)
        expected = 9 * 9 * 1.0  # 9x9 quads in 10x10 region
        self.assertAlmostEqual(area, expected, places=5)

    def test_zero_depth_mask(self):
        """Pixels with zero depth should be excluded by default mask."""
        depth = np.full((10, 10), 100.0, dtype=np.float64)
        depth[0:5, :] = 0.0  # top half zero
        area = measure_area(depth, pixel_size=1.0)
        # Only bottom 5 rows valid: 4 rows of quads * 9 cols = 36
        expected = 4 * 9 * 1.0
        self.assertAlmostEqual(area, expected, places=5)

    def test_single_pixel_zero_area(self):
        """A single valid pixel should have zero area (no quads)."""
        depth = np.zeros((10, 10), dtype=np.float64)
        depth[5, 5] = 100.0
        area = measure_area(depth, pixel_size=1.0)
        self.assertAlmostEqual(area, 0.0, places=10)

    def test_returns_float(self):
        """Return type should be float."""
        depth = np.full((5, 5), 100.0, dtype=np.float64)
        self.assertIsInstance(measure_area(depth), float)

    def test_all_zero_depth(self):
        """All-zero depth should give zero area."""
        depth = np.zeros((10, 10), dtype=np.float64)
        area = measure_area(depth, pixel_size=1.0)
        self.assertAlmostEqual(area, 0.0)


if __name__ == "__main__":
    unittest.main()
