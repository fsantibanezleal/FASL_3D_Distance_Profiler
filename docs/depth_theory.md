# RGB-D Depth Processing Theory

Comprehensive mathematical reference for the depth processing, surface reconstruction, and analysis algorithms implemented in SurfaceScope.

---

## 1. RGB-D Cameras and Sensor Technologies

RGB-D cameras produce two co-registered images per frame:
- **RGB image** (H, W, 3): standard colour photograph
- **Depth map** (H, W): per-pixel distance from camera to scene surface (typically in millimetres)

### Sensor Technologies

| Technology       | Principle                            | Example Cameras        | Range      | Resolution  |
|------------------|--------------------------------------|------------------------|------------|-------------|
| Structured light | Project IR pattern, triangulate      | Intel RealSense D400   | 0.2--10 m  | 1280x720    |
| Time-of-flight   | Measure IR round-trip time           | Microsoft Kinect Azure | 0.5--5.5 m | 640x576     |
| Active stereo    | Stereo matching with IR texture      | Intel RealSense D415   | 0.3--10 m  | 1280x720    |
| Laser profiling  | Laser triangulation (line-scan)      | SICK Ranger3D          | 0.1--2 m   | 2560x1      |

### Comparison with Other 3D Sensing Methods

| Method           | Principle                     | Range       | Accuracy    | Speed     | Cost    |
|------------------|-------------------------------|-------------|-------------|-----------|---------|
| RGB-D camera     | Active depth + colour         | 0.2--10 m   | 1--10 mm    | 30--90 Hz | Low     |
| LiDAR            | Laser time-of-flight scanning | 1--300 m    | 1--5 mm     | 10--20 Hz | High    |
| Photogrammetry   | Multi-view stereo from photos | Arbitrary   | 0.1--5 mm   | Minutes   | Low     |
| Structured light scanner | Phase-shift projection | 0.1--2 m   | 0.01--0.1 mm | Seconds  | Medium  |

---

## 2. Pinhole Camera Model

The relationship between a 3D world point (X, Y, Z) and its pixel coordinates (u, v) is governed by the pinhole camera model.

### Forward Projection (3D to 2D)

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

In matrix form:

```
s * [u]   [fx  0  cx] [X]
    [v] = [ 0 fy  cy] [Y]
    [1]   [ 0  0   1] [Z]
```

where:
- **(fx, fy)** -- focal lengths in pixel units (fx = F / pixel_width, fy = F / pixel_height)
- **(cx, cy)** -- principal point, typically the image centre (W/2, H/2)
- **s** -- scale factor (equal to Z)

### Inverse Projection (Depth to 3D Point Cloud)

Given a depth map z(v, u), each pixel with valid depth is back-projected into 3D:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth[v, u]
```

This produces a 3D point cloud P = {(Xi, Yi, Zi)} for all pixels where depth > 0.

### Intrinsic Parameter Matrix

```
K = [fx   0  cx]
    [ 0  fy  cy]
    [ 0   0   1]
```

For SurfaceScope's synthetic scenes, default intrinsics are: fx = fy = 500 px, cx = W/2, cy = H/2.

---

## 3. Bilateral Filter

The bilateral filter (Tomasi & Manduchi, 1998) is an edge-preserving smoothing filter that combines spatial proximity weighting with intensity similarity weighting.

### Full Equation

```
BF[I](p) = (1 / Wp) * SUM_{q in Omega} G_sigma_s(||p - q||) * G_sigma_r(|I(p) - I(q)|) * I(q)
```

where the normalisation factor is:

```
Wp = SUM_{q in Omega} G_sigma_s(||p - q||) * G_sigma_r(|I(p) - I(q)|)
```

### Spatial and Range Gaussians (Decomposition)

The spatial weight:
```
G_sigma_s(d) = exp(-d^2 / (2 * sigma_s^2))
```
where d = ||p - q|| is the Euclidean pixel distance.

The range (intensity) weight:
```
G_sigma_r(delta) = exp(-delta^2 / (2 * sigma_r^2))
```
where delta = |I(p) - I(q)| is the absolute intensity (depth) difference.

The total weight for neighbour q contributing to pixel p:
```
w(p, q) = G_sigma_s(||p - q||) * G_sigma_r(|I(p) - I(q)|)
```

### Properties

- **sigma_s** (sigma spatial) controls the spatial extent of smoothing in pixels. Larger values average over a wider neighbourhood.
- **sigma_r** (sigma range) controls edge sensitivity in mm. Smaller values preserve more edges by down-weighting neighbours with large depth differences.
- **Computational complexity**: O(N * k^2) for kernel half-size k, where N is the number of pixels.
- **Edge preservation**: When two neighbouring pixels lie across a depth discontinuity (|I(p) - I(q)| >> sigma_r), the range weight approaches zero, preventing blurring across the edge.

### Implementation

SurfaceScope implements the bilateral filter as an explicit double loop over the spatial kernel, with pre-computed spatial weights and per-pixel range weights. Zero-depth pixels are treated as invalid and excluded from both input and output.

---

## 4. Surface Normals from Depth

Given a depth map z(u, v), the surface normal at each pixel is computed from the gradient of the depth function.

### Derivation

Model the surface as the graph z = f(x, y). The two tangent vectors along the grid axes are:

```
t_x = (1, 0, dz/dx)
t_y = (0, 1, dz/dy)
```

The surface normal is their cross product:

```
n = t_x x t_y = (-dz/dx, -dz/dy, 1)
```

Normalised to unit length:

```
n_hat = (-dz/dx, -dz/dy, 1) / sqrt((dz/dx)^2 + (dz/dy)^2 + 1)
```

Or equivalently:

```
n = normalize(-dz/dx, -dz/dy, 1)
```

### Finite Difference Approximation

Partial derivatives are estimated via central differences using `numpy.gradient`:

```
dz/dx ~ (z[v, u+1] - z[v, u-1]) / (2 * pixel_size)
dz/dy ~ (z[v+1, u] - z[v-1, u]) / (2 * pixel_size)
```

At image boundaries, forward or backward differences are used automatically.

### Normal Map Visualisation

Normal vectors are mapped to colours for display:

```
colour = (n_hat + 1) / 2 * 255
```

This yields a "normal map" where:
- **Red** channel encodes the x-component (left-right tilt)
- **Green** channel encodes the y-component (up-down tilt)
- **Blue** channel encodes the z-component (mostly blue for surfaces facing the camera, since nz is close to 1)

---

## 5. Surface Curvature

### Gaussian Curvature (K)

Gaussian curvature measures the intrinsic curvature of a surface. For a surface defined as z = f(x, y):

```
K = (fxx * fyy - fxy^2) / (1 + fx^2 + fy^2)^2
```

where:
- **fx = dz/dx**, **fy = dz/dy** are first partial derivatives
- **fxx = d^2z/dx^2**, **fyy = d^2z/dy^2** are second partial derivatives
- **fxy = d^2z/dxdy** is the mixed partial derivative

**Interpretation**:
- K > 0: elliptic point (bowl or dome shape -- both principal curvatures have the same sign)
- K < 0: hyperbolic point (saddle shape -- principal curvatures have opposite signs)
- K = 0: parabolic or flat point (at least one principal curvature is zero)

### Mean Curvature (H)

Mean curvature is the average of the two principal curvatures:

```
H = ((1 + fy^2) * fxx - 2 * fx * fy * fxy + (1 + fx^2) * fyy) / (2 * (1 + fx^2 + fy^2)^(3/2))
```

**Interpretation**:
- H > 0: concave surface (valley, when viewed from above)
- H < 0: convex surface (ridge)
- H = 0: minimal surface (locally area-minimising, like a soap film)

### Principal Curvatures

The two principal curvatures kappa_1 and kappa_2 relate to K and H by:

```
K = kappa_1 * kappa_2
H = (kappa_1 + kappa_2) / 2
```

They can be recovered as:

```
kappa_1 = H + sqrt(H^2 - K)
kappa_2 = H - sqrt(H^2 - K)
```

### Implementation Notes

Second derivatives are computed by applying `numpy.gradient` twice. The mixed derivative fxy is computed from both orderings and averaged for symmetry: `fxy = 0.5 * (d(fx)/dy + d(fy)/dx)`.

---

## 6. Surface Roughness (ISO 4287)

ISO 4287:1997 defines standard parameters for characterising surface texture from 1D height profiles.

### Ra -- Arithmetic Average Roughness

```
Ra = (1/N) * SUM_{i=1}^{N} |zi - z_mean|
```

Ra is the most widely used roughness parameter. It gives the mean absolute deviation of the profile from its mean line. Simple, robust, but insensitive to the spatial distribution of peaks and valleys.

### Rq -- Root Mean Square Roughness

```
Rq = sqrt( (1/N) * SUM_{i=1}^{N} (zi - z_mean)^2 )
```

Rq (also called RMS roughness) is more sensitive to large peaks and deep valleys than Ra because it squares the deviations before averaging. For a Gaussian distribution: Rq = Ra * sqrt(pi/2) ~ 1.25 * Ra.

### Rz -- Peak-to-Valley Height

```
Rz = max(z) - min(z)
```

Rz gives the total range of the profile. It is highly sensitive to outliers and single extreme events. Often used alongside Ra for a more complete description.

### Rsk -- Skewness

```
Rsk = (1 / (N * Rq^3)) * SUM_{i=1}^{N} (zi - z_mean)^3
```

**Interpretation**:
- Rsk > 0: profile has more peaks than valleys (plateau with scratches)
- Rsk < 0: profile has more valleys than peaks (peaks worn down)
- Rsk = 0: symmetric height distribution

### Rku -- Kurtosis

```
Rku = (1 / (N * Rq^4)) * SUM_{i=1}^{N} (zi - z_mean)^4
```

**Interpretation**:
- Rku = 3: Gaussian (normal) height distribution
- Rku > 3: sharp peaks and/or deep valleys (leptokurtic)
- Rku < 3: bumpy, rounded surface (platykurtic)

---

## 7. Point Cloud Projection

### Full Derivation

Starting from the pinhole camera model, each pixel (u, v) with depth Z = depth[v, u] > 0 is projected to 3D:

1. Subtract the principal point to get normalised pixel coordinates:
   ```
   u' = u - cx
   v' = v - cy
   ```

2. Divide by focal length to get normalised camera coordinates:
   ```
   x_cam = u' / fx
   y_cam = v' / fy
   ```

3. Scale by depth to get metric 3D coordinates:
   ```
   X = x_cam * Z = (u - cx) * Z / fx
   Y = y_cam * Z = (v - cy) * Z / fy
   Z = depth[v, u]
   ```

4. Optionally transform to world coordinates via extrinsic matrix:
   ```
   P_world = R * P_cam + t
   ```
   (For synthetic data, identity transformation is used.)

### Colour Assignment

If an RGB image is co-registered with the depth map, each 3D point inherits the colour of its source pixel:

```
colour[i] = rgb[v_i, u_i] / 255.0
```

producing normalised [0, 1] floating-point RGB values for rendering.

---

## 8. Mesh Triangulation

### Quad Decomposition

A depth map on a regular grid naturally forms a lattice of quads. Each 2x2 pixel neighbourhood defines a quad:

```
(u, v)      (u+1, v)
   +----------+
   |          |
   |   quad   |
   |          |
   +----------+
(u, v+1)    (u+1, v+1)
```

### Triangle Generation

Each quad is split into two triangles:

```
Triangle 1: [idx(u,v), idx(u+1,v), idx(u,v+1)]
Triangle 2: [idx(u+1,v), idx(u+1,v+1), idx(u,v+1)]
```

Only quads where all four corner pixels have valid depth (> 0) produce triangles. This naturally handles occlusion boundaries and sensor shadows.

### Face Winding

Counter-clockwise winding (when viewed from the camera) ensures correct face normals for lighting. Three.js `DoubleSide` material is used as a fallback to render both face orientations.

### Vertex Position

```
V(u, v) = (u * pixel_size * subsample, v * pixel_size * subsample, depth[v, u])
```

where `subsample` controls the mesh resolution (1 = full, 2 = every other pixel, etc.).

---

## 9. 3D File Format Specifications

### PLY (Polygon File Format)

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face M
property list uchar int vertex_indices
end_header
<N lines: x y z r g b>
<M lines: 3 i j k>
```

Originally developed at Stanford (Turk & Levoy, 1994). Supports both ASCII and binary encodings. Widely supported by MeshLab, CloudCompare, Blender, Open3D.

### PCD (Point Cloud Data)

```
# .PCD v0.7 - Point Cloud Data file
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH N
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS N
DATA ascii
<N lines: x y z rgb_packed>
```

Native format of the Point Cloud Library (PCL). RGB is packed as a single uint32: `rgb = (r << 16) | (g << 8) | b`.

### OBJ (Wavefront)

```
# Exported from SurfaceScope
v x1 y1 z1
v x2 y2 z2
...
f i1 j1 k1    (1-indexed face indices)
f i2 j2 k2
...
```

Simple, universal triangle mesh format. Note: OBJ uses 1-based indexing for faces.

---

## 10. RGB-D Sensor Types -- Detailed Principles

### Structured Light

Projects a known infrared (IR) pattern onto the scene. A camera observes the deformed pattern and triangulates depth from the distortion. Advantages: dense depth maps, works indoors. Limitations: fails in sunlight, limited range.

### Time-of-Flight (ToF)

Emits modulated IR light and measures the phase shift of the reflected signal to compute round-trip time, hence distance: `d = c * delta_t / 2`. Advantages: compact, works in dark. Limitations: lower resolution, multi-path interference.

### Stereo Vision

Two cameras separated by a known baseline observe the scene. Depth is computed from disparity (pixel offset between corresponding points): `Z = f * B / d`, where f is focal length, B is baseline, and d is disparity in pixels. Active stereo projects IR texture to improve matching on textureless surfaces.

### Laser Line Profiling (SICK Ranger3D)

A laser line is projected onto the surface and a camera captures the line's position from an oblique angle. Triangulation yields a single row of depth values per scan. The object moves on a conveyor belt to build up a full 2D depth map line by line. Advantages: very high accuracy (micrometres), high speed. Used in industrial inspection.

---

## 11. Comparison: RGB-D vs LiDAR vs Photogrammetry

| Aspect            | RGB-D Camera              | LiDAR                       | Photogrammetry             |
|-------------------|---------------------------|-----------------------------|-----------------------------|
| Range             | 0.2--10 m (indoor)        | 1--300 m (outdoor)          | Arbitrary                   |
| Accuracy          | 1--10 mm                  | 1--5 mm at 100 m            | 0.1--5 mm (close range)     |
| Frame rate        | 30--90 Hz                 | 10--20 Hz                   | Offline (minutes to hours)  |
| Dense output      | Yes (per pixel)           | Sparse (scanning pattern)   | Dense (after reconstruction)|
| Colour            | Co-registered RGB         | Separate camera required    | Inherent from photos        |
| Outdoor use       | Limited (IR interference) | Excellent                   | Excellent                   |
| Cost              | $200--$2,000              | $5,000--$100,000            | Camera + software           |
| Typical use       | Robotics, AR, inspection  | Autonomous driving, survey  | Cultural heritage, mapping  |

---

## References

- Tomasi, C. & Manduchi, R. (1998). Bilateral Filtering for Gray and Color Images. ICCV.
- Hartley, R. & Zisserman, A. (2003). Multiple View Geometry in Computer Vision. Cambridge University Press.
- Paris, S. et al. (2009). Bilateral Filtering: Theory and Applications. FTCGV 4(1).
- ISO 4287:1997. GPS -- Surface texture: Profile method.
- do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces. Prentice-Hall.
- Botsch, M. et al. (2010). Polygon Mesh Processing. A K Peters/CRC Press.
- Turk, G. & Levoy, M. (1994). Zippered Polygon Meshes from Range Images. SIGGRAPH.
- Intel RealSense D400 Series datasheet. https://www.intelrealsense.com/
- SICK Ranger3D specifications. https://www.sick.com/
- Perlin, K. (1985). An Image Synthesizer. SIGGRAPH.
