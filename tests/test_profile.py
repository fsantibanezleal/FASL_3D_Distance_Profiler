"""
Tests for profile_analysis module
====================================

Covers:
    - compute_roughness: Ra, Rq, Rz, Rsk, Rku
    - compute_histogram: depth distribution
    - detect_objects: object detection on flat backgrounds
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.profile_analysis import (
    compute_roughness,
    compute_histogram,
    detect_objects,
)


class TestComputeRoughness(unittest.TestCase):
    """Test surface roughness metric computation."""

    def test_flat_profile_zero_roughness(self):
        """A constant profile should have Ra = Rq = 0 and Rz = 0."""
        profile = np.full(100, 500.0)
        metrics = compute_roughness(profile)
        self.assertAlmostEqual(metrics["Ra"], 0.0, places=10)
        self.assertAlmostEqual(metrics["Rq"], 0.0, places=10)
        self.assertAlmostEqual(metrics["Rz"], 0.0, places=10)

    def test_known_values(self):
        """Test against hand-computed values for a simple profile.

        Profile: [0, 1, 0, -1, 0]
        Mean = 0
        |deviations| = [0, 1, 0, 1, 0]
        Ra = (0+1+0+1+0)/5 = 0.4
        Rq = sqrt((0+1+0+1+0)/5) = sqrt(0.4) ~ 0.6325
        Rz = 1 - (-1) = 2.0
        """
        profile = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
        metrics = compute_roughness(profile)
        self.assertAlmostEqual(metrics["Ra"], 0.4, places=5)
        self.assertAlmostEqual(metrics["Rq"], np.sqrt(0.4), places=5)
        self.assertAlmostEqual(metrics["Rz"], 2.0, places=5)

    def test_single_method_ra(self):
        """Requesting only Ra should return just Ra."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_roughness(profile, method="Ra")
        self.assertIn("Ra", metrics)
        self.assertNotIn("Rq", metrics)

    def test_single_method_rq(self):
        """Requesting only Rq should return just Rq."""
        profile = np.array([1.0, 2.0, 3.0])
        metrics = compute_roughness(profile, method="Rq")
        self.assertIn("Rq", metrics)
        self.assertNotIn("Ra", metrics)

    def test_single_method_rz(self):
        """Requesting only Rz should return just Rz."""
        profile = np.array([10.0, 20.0, 30.0])
        metrics = compute_roughness(profile, method="Rz")
        self.assertEqual(metrics["Rz"], 20.0)

    def test_nan_handling(self):
        """NaN values in the profile should be excluded."""
        profile = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        metrics = compute_roughness(profile)
        self.assertGreater(metrics["Ra"], 0)

    def test_empty_profile(self):
        """An empty profile should return an empty dict."""
        profile = np.array([])
        metrics = compute_roughness(profile)
        self.assertEqual(metrics, {})

    def test_all_nan_profile(self):
        """An all-NaN profile should return empty dict."""
        profile = np.array([np.nan, np.nan, np.nan])
        metrics = compute_roughness(profile)
        self.assertEqual(metrics, {})

    def test_skewness_symmetric(self):
        """A symmetric profile should have near-zero skewness."""
        profile = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        metrics = compute_roughness(profile)
        self.assertAlmostEqual(metrics["Rsk"], 0.0, places=5)

    def test_kurtosis_gaussian_approx(self):
        """A Gaussian-distributed profile should have Rku near 3."""
        rng = np.random.default_rng(42)
        profile = rng.normal(0, 10, 10000)
        metrics = compute_roughness(profile)
        self.assertAlmostEqual(metrics["Rku"], 3.0, delta=0.3)


class TestComputeHistogram(unittest.TestCase):
    """Test depth histogram computation."""

    def test_uniform_depth(self):
        """A uniform depth map should produce a single-bin histogram."""
        depth = np.full((32, 32), 500.0, dtype=np.float32)
        result = compute_histogram(depth, bins=10)
        self.assertEqual(len(result["counts"]), 10)
        self.assertAlmostEqual(result["mean"], 500.0, places=1)
        self.assertAlmostEqual(result["median"], 500.0, places=1)
        self.assertAlmostEqual(result["std"], 0.0, places=1)

    def test_zero_excluded(self):
        """Zero-depth pixels should not appear in the histogram."""
        depth = np.zeros((32, 32), dtype=np.float32)
        depth[0:16, :] = 500.0
        result = compute_histogram(depth, bins=10)
        self.assertAlmostEqual(result["mean"], 500.0, places=1)

    def test_empty_depth(self):
        """An all-zero depth map should produce empty histogram."""
        depth = np.zeros((16, 16), dtype=np.float32)
        result = compute_histogram(depth)
        self.assertEqual(result["counts"], [])

    def test_bin_count(self):
        """Number of bins should match the requested count."""
        rng = np.random.default_rng(0)
        depth = 400 + rng.uniform(0, 200, (64, 64)).astype(np.float32)
        result = compute_histogram(depth, bins=50)
        self.assertEqual(len(result["counts"]), 50)
        self.assertEqual(len(result["bin_centers"]), 50)
        self.assertEqual(len(result["bin_edges"]), 51)


class TestDetectObjects(unittest.TestCase):
    """Test depth-based object detection."""

    def test_no_objects_on_flat(self):
        """A flat surface should have no detected objects."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        objects = detect_objects(depth, threshold=10)
        self.assertEqual(len(objects), 0)

    def test_detect_single_object(self):
        """A raised block on a flat surface should be detected."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        depth[20:40, 20:40] = 400.0  # 100mm raised block
        objects = detect_objects(depth, threshold=20, min_area=50)
        self.assertGreaterEqual(len(objects), 1)
        # Check the largest object
        obj = max(objects, key=lambda o: o["area"])
        self.assertGreater(obj["area"], 100)
        self.assertGreater(obj["mean_height"], 50)

    def test_small_objects_filtered(self):
        """Objects smaller than min_area should be excluded."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        depth[30, 30] = 400.0  # single pixel
        objects = detect_objects(depth, threshold=20, min_area=10)
        self.assertEqual(len(objects), 0)

    def test_multiple_objects(self):
        """Multiple separated objects should be detected individually."""
        depth = np.full((128, 128), 500.0, dtype=np.float32)
        depth[10:25, 10:25] = 400.0   # object 1
        depth[60:80, 60:80] = 380.0   # object 2
        depth[100:115, 100:115] = 420.0  # object 3
        objects = detect_objects(depth, threshold=20, min_area=50)
        self.assertGreaterEqual(len(objects), 2)

    def test_auto_threshold(self):
        """With threshold=None, automatic threshold should be computed."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        depth[20:40, 20:40] = 350.0  # large height difference
        objects = detect_objects(depth, threshold=None, min_area=50)
        self.assertGreaterEqual(len(objects), 1)

    def test_object_properties(self):
        """Detected objects should have correct property keys."""
        depth = np.full((64, 64), 500.0, dtype=np.float32)
        depth[10:30, 10:30] = 400.0
        objects = detect_objects(depth, threshold=20, min_area=10)
        if len(objects) > 0:
            obj = objects[0]
            self.assertIn("id", obj)
            self.assertIn("centroid", obj)
            self.assertIn("bbox", obj)
            self.assertIn("area", obj)
            self.assertIn("mean_height", obj)
            self.assertIn("max_height", obj)


if __name__ == "__main__":
    unittest.main()
