"""
Tests for export module
========================

Covers:
    - export_ply: PLY format generation (vertices, colors, faces)
    - export_pcd: PCD format generation (vertices, colors)
    - export_obj_mesh: OBJ format generation (vertices, faces)
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.export import export_ply, export_pcd, export_obj_mesh


class TestExportPLY(unittest.TestCase):
    """Test PLY export."""

    def test_header_format(self):
        """PLY output should start with 'ply' and contain correct header."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = export_ply(points)
        lines = result.strip().split('\n')
        self.assertEqual(lines[0], 'ply')
        self.assertEqual(lines[1], 'format ascii 1.0')
        self.assertIn('element vertex 3', result)
        self.assertIn('end_header', result)

    def test_vertex_count(self):
        """Number of data lines after header should match vertex count."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = export_ply(points)
        lines = result.strip().split('\n')
        header_end = lines.index('end_header')
        data_lines = lines[header_end + 1:]
        self.assertEqual(len(data_lines), 2)

    def test_with_colors(self):
        """PLY with colors should include RGB properties in header and data."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        colors = np.array([[255, 128, 0]], dtype=np.uint8)
        result = export_ply(points, colors=colors)
        self.assertIn('property uchar red', result)
        self.assertIn('property uchar green', result)
        self.assertIn('property uchar blue', result)
        # Data line should have 6 values (x y z r g b)
        lines = result.strip().split('\n')
        data_line = lines[-1]
        parts = data_line.split()
        self.assertEqual(len(parts), 6)

    def test_with_faces(self):
        """PLY with faces should include face element and data."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        result = export_ply(points, faces=faces)
        self.assertIn('element face 1', result)
        self.assertIn('property list uchar int vertex_indices', result)
        # Last line should be the face
        lines = result.strip().split('\n')
        self.assertEqual(lines[-1], '3 0 1 2')

    def test_empty_points(self):
        """PLY with 0 vertices should still produce valid header."""
        points = np.zeros((0, 3), dtype=np.float64)
        result = export_ply(points)
        self.assertIn('element vertex 0', result)

    def test_no_faces_no_face_element(self):
        """PLY without faces should not contain face element."""
        points = np.array([[1, 2, 3]], dtype=np.float64)
        result = export_ply(points, faces=None)
        self.assertNotIn('element face', result)

    def test_color_mismatch_ignored(self):
        """If color count does not match point count, colors are ignored."""
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        colors = np.array([[255, 0, 0]], dtype=np.uint8)  # only 1 color for 2 points
        result = export_ply(points, colors=colors)
        self.assertNotIn('property uchar red', result)


class TestExportPCD(unittest.TestCase):
    """Test PCD export."""

    def test_header_format(self):
        """PCD output should contain required header fields."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = export_pcd(points)
        self.assertIn('VERSION 0.7', result)
        self.assertIn('FIELDS x y z', result)
        self.assertIn('WIDTH 2', result)
        self.assertIn('POINTS 2', result)
        self.assertIn('DATA ascii', result)

    def test_data_line_count(self):
        """Number of data lines should match point count."""
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        result = export_pcd(points)
        lines = result.strip().split('\n')
        # 11 header lines + 3 data lines
        data_start = lines.index('DATA ascii') + 1
        data_lines = lines[data_start:]
        self.assertEqual(len(data_lines), 3)

    def test_with_colors(self):
        """PCD with colors should have packed RGB field."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        colors = np.array([[255, 0, 0]], dtype=np.uint8)
        result = export_pcd(points, colors=colors)
        self.assertIn('FIELDS x y z rgb', result)
        # Packed RGB for (255, 0, 0) = 255 << 16 = 16711680
        lines = result.strip().split('\n')
        data_line = lines[-1]
        parts = data_line.split()
        self.assertEqual(len(parts), 4)
        self.assertEqual(int(parts[3]), 16711680)

    def test_without_colors(self):
        """PCD without colors should only have xyz fields."""
        points = np.array([[1, 2, 3]], dtype=np.float64)
        result = export_pcd(points)
        self.assertNotIn('rgb', result.split('\n')[2])  # FIELDS line

    def test_empty_cloud(self):
        """PCD with 0 points should still have valid header."""
        points = np.zeros((0, 3), dtype=np.float64)
        result = export_pcd(points)
        self.assertIn('POINTS 0', result)


class TestExportOBJ(unittest.TestCase):
    """Test OBJ mesh export."""

    def test_vertex_lines(self):
        """OBJ should have correct 'v' lines."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        result = export_obj_mesh(points, faces)
        lines = result.strip().split('\n')
        v_lines = [l for l in lines if l.startswith('v ')]
        self.assertEqual(len(v_lines), 2)

    def test_face_lines_one_indexed(self):
        """OBJ faces should be 1-indexed."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        result = export_obj_mesh(points, faces)
        lines = result.strip().split('\n')
        f_lines = [l for l in lines if l.startswith('f ')]
        self.assertEqual(len(f_lines), 1)
        self.assertEqual(f_lines[0], 'f 1 2 3')

    def test_multiple_faces(self):
        """OBJ should handle multiple faces correctly."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        result = export_obj_mesh(points, faces)
        f_lines = [l for l in result.strip().split('\n') if l.startswith('f ')]
        self.assertEqual(len(f_lines), 2)
        self.assertEqual(f_lines[0], 'f 1 2 3')
        self.assertEqual(f_lines[1], 'f 2 4 3')

    def test_header_comment(self):
        """OBJ should start with a comment line."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        faces = np.array([[0, 0, 0]], dtype=np.int32)
        result = export_obj_mesh(points, faces)
        self.assertTrue(result.startswith('#'))


if __name__ == "__main__":
    unittest.main()
