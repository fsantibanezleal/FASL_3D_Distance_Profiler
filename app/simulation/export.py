"""
Point cloud and mesh export to standard 3D file formats.

Supports:
- PLY (Polygon File Format): ASCII and binary, widely supported
- PCD (Point Cloud Data): ASCII, used by PCL library
- OBJ (Wavefront): Triangle mesh format

These formats enable post-processing in MeshLab, CloudCompare, Blender, etc.
"""
import numpy as np
from typing import Optional


def export_ply(points: np.ndarray, colors: Optional[np.ndarray] = None,
                faces: Optional[np.ndarray] = None, binary: bool = False) -> str:
    """Export point cloud or mesh to PLY format (ASCII).

    PLY header specifies vertex count, properties (x,y,z,r,g,b),
    and optionally face list.

    Args:
        points: (N, 3) vertex positions.
        colors: (N, 3) RGB colors in [0, 255] uint8.
        faces: (M, 3) triangle face indices (optional).
        binary: If True, return binary PLY (not implemented, ASCII only).

    Returns:
        PLY file content as string.
    """
    N = len(points)
    has_color = colors is not None and len(colors) == N
    has_faces = faces is not None and len(faces) > 0

    lines = ['ply', 'format ascii 1.0']
    lines.append(f'element vertex {N}')
    lines.append('property float x')
    lines.append('property float y')
    lines.append('property float z')
    if has_color:
        lines.append('property uchar red')
        lines.append('property uchar green')
        lines.append('property uchar blue')
    if has_faces:
        lines.append(f'element face {len(faces)}')
        lines.append('property list uchar int vertex_indices')
    lines.append('end_header')

    for i in range(N):
        x, y, z = points[i]
        if has_color:
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            lines.append(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}')
        else:
            lines.append(f'{x:.6f} {y:.6f} {z:.6f}')

    if has_faces:
        for face in faces:
            lines.append(f'3 {face[0]} {face[1]} {face[2]}')

    return '\n'.join(lines) + '\n'


def export_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None) -> str:
    """Export point cloud to PCD format (ASCII).

    Args:
        points: (N, 3) vertex positions.
        colors: (N, 3) RGB colors in [0, 255] uint8 (optional).

    Returns:
        PCD file content as string.
    """
    N = len(points)
    has_color = colors is not None and len(colors) == N

    fields = 'x y z'
    sizes = '4 4 4'
    types = 'F F F'
    count = '1 1 1'

    if has_color:
        fields += ' rgb'
        sizes += ' 4'
        types += ' U'
        count += ' 1'

    lines = [
        '# .PCD v0.7 - Point Cloud Data file',
        'VERSION 0.7',
        f'FIELDS {fields}',
        f'SIZE {sizes}',
        f'TYPE {types}',
        f'COUNT {count}',
        f'WIDTH {N}',
        'HEIGHT 1',
        'VIEWPOINT 0 0 0 1 0 0 0',
        f'POINTS {N}',
        'DATA ascii',
    ]

    for i in range(N):
        x, y, z = points[i]
        if has_color:
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            rgb_packed = (r << 16) | (g << 8) | b
            lines.append(f'{x:.6f} {y:.6f} {z:.6f} {rgb_packed}')
        else:
            lines.append(f'{x:.6f} {y:.6f} {z:.6f}')

    return '\n'.join(lines) + '\n'


def export_obj_mesh(points: np.ndarray, faces: np.ndarray) -> str:
    """Export triangle mesh to OBJ format.

    Args:
        points: (N, 3) vertex positions.
        faces: (M, 3) triangle face indices (0-indexed, converted to 1-indexed).

    Returns:
        OBJ file content as string.
    """
    lines = ['# Exported from SurfaceScope']
    for p in points:
        lines.append(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}')
    for f in faces:
        lines.append(f'f {f[0]+1} {f[1]+1} {f[2]+1}')  # 1-indexed
    return '\n'.join(lines) + '\n'
