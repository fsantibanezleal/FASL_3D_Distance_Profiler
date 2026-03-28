# RGB-D Depth Processing Theory

## 1. RGB-D Cameras

RGB-D cameras produce two co-registered images per frame:
- **RGB image** (H, W, 3): standard colour photograph
- **Depth map** (H, W): per-pixel distance from camera to scene surface

### Sensor Technologies

| Technology      | Example Cameras        | Range    | Resolution |
|-----------------|------------------------|----------|------------|
| Structured light| Intel RealSense D400   | 0.2-10 m | 1280x720  |
| Time-of-flight  | Microsoft Kinect Azure | 0.5-5 m  | 640x576   |
| Laser profiling | SICK Ranger3D          | 0.1-2 m  | 2560x1    |

## 2. Pinhole Camera Model

The relationship between a 3D world point (X, Y, Z) and its pixel coordinates (u, v) is governed by the pinhole camera model:

```
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

The inverse projection (depth to 3D point cloud) is:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth[v, u]
```

where:
- **(fx, fy)**: focal lengths in pixel units
- **(cx, cy)**: principal point (image centre)

## 3. Bilateral Filter

The bilateral filter is an edge-preserving smoothing filter (Tomasi & Manduchi, 1998):

```
BF[I](p) = (1/Wp) * SUM_q G_sigma_s(||p-q||) * G_sigma_r(|I(p)-I(q)|) * I(q)
```

where:
- **G_sigma_s**: spatial Gaussian with standard deviation sigma_s
- **G_sigma_r**: range (intensity) Gaussian with standard deviation sigma_r
- **Wp**: normalisation factor

### Properties
- sigma_spatial controls the spatial extent of smoothing
- sigma_range controls edge sensitivity: small values preserve more edges
- Computational complexity: O(N * k^2) for kernel size k

## 4. Surface Normals from Depth

Given a depth map z(u, v), the surface normal at each pixel is computed from the gradient:

```
nx = -dz/dx
ny = -dz/dy
nz = 1.0
n_hat = (nx, ny, nz) / ||(nx, ny, nz)||
```

Derivatives are estimated using central finite differences via `numpy.gradient`.

### Normal Map Visualisation
Normals are mapped to colours: `colour = (normal + 1) / 2 * 255`, yielding a "normal map" where:
- Red encodes the x-component
- Green encodes the y-component
- Blue encodes the z-component (mostly blue for surfaces facing the camera)

## 5. Surface Curvature

### Gaussian Curvature (K)

```
K = (fxx * fyy - fxy^2) / (1 + fx^2 + fy^2)^2
```

- K > 0: elliptic point (bowl/dome)
- K < 0: hyperbolic point (saddle)
- K = 0: parabolic or flat

### Mean Curvature (H)

```
H = ((1 + fy^2)*fxx - 2*fx*fy*fxy + (1 + fx^2)*fyy) / (2*(1 + fx^2 + fy^2)^(3/2))
```

- H > 0: concave (valley)
- H < 0: convex (ridge)

## 6. Surface Roughness (ISO 4287)

### Ra -- Arithmetic Average Roughness

```
Ra = (1/N) * SUM |zi - z_mean|
```

### Rq -- Root Mean Square Roughness

```
Rq = sqrt( (1/N) * SUM (zi - z_mean)^2 )
```

### Rz -- Peak-to-Valley Height

```
Rz = max(z) - min(z)
```

### Rsk -- Skewness

```
Rsk = (1/(N * Rq^3)) * SUM (zi - z_mean)^3
```

### Rku -- Kurtosis

```
Rku = (1/(N * Rq^4)) * SUM (zi - z_mean)^4
```

A Gaussian surface has Rku = 3. Values above 3 indicate sharp peaks/valleys.

## 7. Depth-to-Mesh Conversion

A depth map on a regular pixel grid is converted to a triangle mesh:

1. Each valid pixel (depth > 0) becomes a vertex: V(u, v) = (u, v, z(u,v))
2. For each 2x2 quad of valid adjacent pixels, two triangles are formed
3. Per-vertex colours are assigned from the co-registered RGB image

This produces a mesh directly renderable in Three.js using BufferGeometry with indexed faces.

## References

- Tomasi, C. & Manduchi, R. (1998). Bilateral Filtering for Gray and Color Images. ICCV.
- Hartley, R. & Zisserman, A. (2003). Multiple View Geometry in Computer Vision. Cambridge University Press.
- Paris, S. et al. (2009). Bilateral Filtering: Theory and Applications. FTCGV 4(1).
- ISO 4287:1997. GPS -- Surface texture: Profile method.
- do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces.
