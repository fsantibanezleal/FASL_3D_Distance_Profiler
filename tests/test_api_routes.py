"""
Route-level tests for the SurfaceScope FastAPI application.
================================================================

Exercises every public endpoint in ``app/api/routes.py`` via
``fastapi.testclient.TestClient``.  Each test starts from a fresh
``AppState`` through ``app.dependency_overrides[get_app_state]`` so the
tests are order-independent and do not leak state between cases.

Coverage:
    - GET  /api/scene_types
    - GET  /api/colormaps
    - POST /api/generate    (tiny 32x32 scene)
    - GET  /api/state       (before and after generate)
    - POST /api/process
    - POST /api/profile
    - GET  /api/metrics
    - GET  /api/export/ply
    - GET  /api/export/pcd
    - GET  /api/export/obj
    - POST /api/measure     (distance / angle / area / bad type)
    - Negative: /api/process with no scene loaded returns 400
"""

import os
import sys
import unittest

# Ensure the project root is importable when running from ``tests/``
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.main import app
from app.core import AppState, get_app_state


def _fresh_state_provider():
    """Factory returning a new :class:`AppState` per TestClient lifetime.

    The returned callable is bound as the ``get_app_state`` dependency
    override so all routes within one ``with TestClient(app) as client``
    block share a single fresh state.
    """
    state = AppState()

    def _provider() -> AppState:
        return state

    return _provider, state


class _RouteTestBase(unittest.TestCase):
    """Shared harness that wires a fresh AppState for each test."""

    def setUp(self) -> None:
        self._provider, self.state = _fresh_state_provider()
        app.dependency_overrides[get_app_state] = self._provider
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_app_state, None)
        self.client.close()

    def _generate_tiny_scene(self) -> dict:
        """Helper that populates state via POST /api/generate."""
        r = self.client.post(
            "/api/generate",
            json={
                "scene_type": "gaussian_hills",
                "width": 32,
                "height": 32,
                "seed": 7,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


class TestMetaEndpoints(_RouteTestBase):
    """Routes that do not require a loaded scene."""

    def test_scene_types_returns_list(self):
        r = self.client.get("/api/scene_types")
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertIn("scene_types", payload)
        self.assertIsInstance(payload["scene_types"], list)
        self.assertGreater(len(payload["scene_types"]), 0)

    def test_colormaps_returns_list(self):
        r = self.client.get("/api/colormaps")
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertIn("colormaps", payload)
        self.assertIsInstance(payload["colormaps"], list)
        self.assertGreater(len(payload["colormaps"]), 0)

    def test_state_before_generate_is_unloaded(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"loaded": False})


class TestGenerateAndState(_RouteTestBase):
    """POST /api/generate populates state and GET /api/state echoes it."""

    def test_generate_returns_loaded_state(self):
        payload = self._generate_tiny_scene()
        self.assertTrue(payload["loaded"])
        self.assertEqual(payload["width"], 32)
        self.assertEqual(payload["height"], 32)
        self.assertEqual(payload["scene_type"], "gaussian_hills")
        self.assertIn("images", payload)
        for key in ("rgb", "depth", "normals"):
            self.assertIn(key, payload["images"])
            self.assertTrue(payload["images"][key].startswith("data:image/png;base64,"))
        self.assertIn("mesh", payload)
        self.assertIn("vertices", payload["mesh"])

    def test_state_after_generate_matches_dims(self):
        self._generate_tiny_scene()
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["loaded"])
        self.assertEqual(body["width"], 32)
        self.assertEqual(body["height"], 32)

    def test_generate_state_persists_across_requests(self):
        """The overridden AppState is shared across requests in one client."""
        self._generate_tiny_scene()
        r1 = self.client.get("/api/state")
        r2 = self.client.get("/api/state")
        self.assertTrue(r1.json()["loaded"])
        self.assertTrue(r2.json()["loaded"])


class TestProcess(_RouteTestBase):
    """POST /api/process runs the filtering pipeline."""

    def test_process_preserves_dims(self):
        self._generate_tiny_scene()
        r = self.client.post(
            "/api/process",
            json={
                "bilateral_sigma_spatial": 2.0,
                "bilateral_sigma_range": 10.0,
                "fill_hole_size": 2,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["loaded"])
        self.assertEqual(body["width"], 32)
        self.assertEqual(body["height"], 32)

    def test_process_without_scene_returns_400(self):
        r = self.client.post(
            "/api/process",
            json={
                "bilateral_sigma_spatial": 2.0,
                "bilateral_sigma_range": 10.0,
                "fill_hole_size": 2,
            },
        )
        self.assertEqual(r.status_code, 400)


class TestProfile(_RouteTestBase):
    """POST /api/profile returns 1D cross section and roughness."""

    def test_profile_returns_expected_shape(self):
        self._generate_tiny_scene()
        r = self.client.post(
            "/api/profile",
            json={
                "start_x": 2.0,
                "start_y": 16.0,
                "end_x": 30.0,
                "end_y": 16.0,
                "num_samples": 64,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(len(body["distances"]), 64)
        self.assertEqual(len(body["heights"]), 64)
        self.assertIn("metrics", body)

    def test_profile_without_scene_returns_400(self):
        r = self.client.post(
            "/api/profile",
            json={
                "start_x": 0.0,
                "start_y": 0.0,
                "end_x": 10.0,
                "end_y": 10.0,
            },
        )
        self.assertEqual(r.status_code, 400)


class TestMetrics(_RouteTestBase):
    """GET /api/metrics returns histogram / objects / curvature / stats."""

    def test_metrics_payload_keys(self):
        self._generate_tiny_scene()
        r = self.client.get("/api/metrics")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("histogram", "objects", "curvature", "depth_stats"):
            self.assertIn(key, body)
        for key in (
            "gaussian_mean",
            "gaussian_std",
            "mean_curvature_mean",
            "mean_curvature_std",
        ):
            self.assertIn(key, body["curvature"])


class TestExports(_RouteTestBase):
    """Export endpoints return streaming file bodies."""

    def test_ply_export(self):
        self._generate_tiny_scene()
        r = self.client.get("/api/export/ply")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "application/x-ply")
        # PLY ASCII header begins with the magic word "ply".
        self.assertTrue(r.text.startswith("ply"))

    def test_pcd_export(self):
        self._generate_tiny_scene()
        r = self.client.get("/api/export/pcd")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "application/x-pcd")
        # PCD ASCII header begins with a version/comment line.
        self.assertIn("VERSION", r.text[:200])

    def test_obj_export(self):
        self._generate_tiny_scene()
        r = self.client.get("/api/export/obj")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "application/x-wavefront-obj")
        # OBJ bodies typically contain at least one vertex line.
        self.assertIn("v ", r.text[:2048])


class TestMeasure(_RouteTestBase):
    """POST /api/measure covers distance, angle, area, and bad types."""

    def test_distance_measurement(self):
        r = self.client.post(
            "/api/measure",
            json={
                "measurement_type": "distance",
                "point_a": [0.0, 0.0, 0.0],
                "point_b": [3.0, 4.0, 0.0],
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["measurement_type"], "distance")
        self.assertAlmostEqual(body["value"], 5.0, places=4)
        self.assertEqual(body["unit"], "mm")

    def test_angle_measurement(self):
        r = self.client.post(
            "/api/measure",
            json={
                "measurement_type": "angle",
                "normal_a": [0.0, 0.0, 1.0],
                "normal_b": [1.0, 0.0, 0.0],
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertAlmostEqual(body["value"], 90.0, places=2)
        self.assertEqual(body["unit"], "degrees")

    def test_area_measurement_requires_scene(self):
        r = self.client.post(
            "/api/measure",
            json={"measurement_type": "area", "pixel_size": 1.0},
        )
        self.assertEqual(r.status_code, 400)

    def test_area_measurement_after_generate(self):
        self._generate_tiny_scene()
        r = self.client.post(
            "/api/measure",
            json={"measurement_type": "area", "pixel_size": 1.0},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertGreater(body["value"], 0.0)
        self.assertEqual(body["unit"], "mm^2")

    def test_unknown_measurement_type_returns_400(self):
        r = self.client.post(
            "/api/measure",
            json={"measurement_type": "bogus"},
        )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
