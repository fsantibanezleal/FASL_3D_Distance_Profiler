/**
 * renderer3d.js -- Three.js 3D Surface Renderer
 * ===============================================
 *
 * Renders depth-map-derived triangle meshes using Three.js with:
 *   - OrbitControls for mouse-driven rotation, pan, zoom
 *   - Per-vertex colouring from RGB texture or depth colourmap
 *   - Toggle: wireframe overlay, surface normals, point cloud mode
 *   - Ambient + directional lighting for surface shading
 *
 * Mesh data format (from backend):
 *   vertices: [[x, y, z], ...]   -- 3D positions
 *   faces:    [[i, j, k], ...]   -- triangle indices
 *   colors:   [[r, g, b], ...]   -- normalised [0,1] per-vertex colours
 */

const Renderer3D = (() => {
    "use strict";

    let container;
    let scene, camera, renderer, controls;
    let surfaceMesh = null;
    let wireframeMesh = null;
    let pointsMesh = null;
    let normalHelpers = null;

    let showWireframe = false;
    let showNormals = false;
    let showPoints = false;

    /**
     * Initialise the Three.js scene, camera, renderer, and controls.
     */
    function init() {
        container = document.getElementById("three-container");
        const w = container.clientWidth;
        const h = container.clientHeight;

        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a1a);

        // Camera
        camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 10000);
        camera.position.set(200, -200, 400);
        camera.up.set(0, 0, -1);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.rotateSpeed = 0.8;

        // Lighting
        const ambient = new THREE.AmbientLight(0x404060, 0.6);
        scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(200, -200, -300);
        scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0x8888cc, 0.4);
        dirLight2.position.set(-100, 100, -200);
        scene.add(dirLight2);

        // Grid helper
        const grid = new THREE.GridHelper(500, 20, 0x222244, 0x222244);
        grid.rotation.x = Math.PI / 2;
        scene.add(grid);

        // Resize handler
        window.addEventListener("resize", _onResize);

        // Animation loop
        _animate();
    }

    /**
     * Load mesh data from the backend and render it.
     * @param {Object} meshData - {vertices, faces, colors, n_vertices, n_faces}
     */
    function loadMesh(meshData) {
        // Clear previous
        _clearMeshes();

        if (!meshData || !meshData.vertices || meshData.vertices.length === 0) return;

        const vertices = meshData.vertices;
        const faces = meshData.faces;
        const colors = meshData.colors;

        // Compute centre for camera target
        let cx = 0, cy = 0, cz = 0;
        for (const v of vertices) {
            cx += v[0]; cy += v[1]; cz += v[2];
        }
        cx /= vertices.length;
        cy /= vertices.length;
        cz /= vertices.length;

        // BufferGeometry
        const geometry = new THREE.BufferGeometry();

        const posArr = new Float32Array(vertices.length * 3);
        const colArr = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
            posArr[i * 3] = vertices[i][0] - cx;
            posArr[i * 3 + 1] = vertices[i][1] - cy;
            posArr[i * 3 + 2] = -(vertices[i][2] - cz);  // flip Z for camera
            colArr[i * 3] = colors[i][0];
            colArr[i * 3 + 1] = colors[i][1];
            colArr[i * 3 + 2] = colors[i][2];
        }
        geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colArr, 3));

        // Index buffer
        const indexArr = new Uint32Array(faces.length * 3);
        for (let i = 0; i < faces.length; i++) {
            indexArr[i * 3] = faces[i][0];
            indexArr[i * 3 + 1] = faces[i][1];
            indexArr[i * 3 + 2] = faces[i][2];
        }
        geometry.setIndex(new THREE.BufferAttribute(indexArr, 1));
        geometry.computeVertexNormals();

        // Surface mesh
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 30,
            flatShading: false,
        });
        surfaceMesh = new THREE.Mesh(geometry, material);
        scene.add(surfaceMesh);

        // Wireframe overlay
        const wireMat = new THREE.MeshBasicMaterial({
            color: 0x4ecdc4,
            wireframe: true,
            transparent: true,
            opacity: 0.3,
        });
        wireframeMesh = new THREE.Mesh(geometry, wireMat);
        wireframeMesh.visible = showWireframe;
        scene.add(wireframeMesh);

        // Point cloud
        const pointsMat = new THREE.PointsMaterial({
            vertexColors: true,
            size: 1.5,
            sizeAttenuation: true,
        });
        pointsMesh = new THREE.Points(geometry, pointsMat);
        pointsMesh.visible = showPoints;
        scene.add(pointsMesh);

        // Camera framing
        const bbox = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute("position"));
        const size = bbox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        camera.position.set(maxDim * 0.8, -maxDim * 0.8, maxDim * 1.2);
        controls.target.set(0, 0, 0);
        controls.update();

        // Update info
        document.getElementById("info-vertices").textContent = `Vertices: ${meshData.n_vertices}`;
        document.getElementById("info-faces").textContent = `Faces: ${meshData.n_faces}`;
    }

    /**
     * Toggle wireframe visibility.
     */
    function setWireframe(enabled) {
        showWireframe = enabled;
        if (wireframeMesh) wireframeMesh.visible = enabled;
    }

    /**
     * Toggle point cloud mode.
     */
    function setPointCloud(enabled) {
        showPoints = enabled;
        if (pointsMesh) pointsMesh.visible = enabled;
        if (surfaceMesh) surfaceMesh.visible = !enabled;
        if (wireframeMesh && !enabled) wireframeMesh.visible = showWireframe;
        if (wireframeMesh && enabled) wireframeMesh.visible = false;
    }

    /**
     * Toggle normal visualisation (via flat shading toggle).
     */
    function setNormals(enabled) {
        showNormals = enabled;
        if (surfaceMesh) {
            surfaceMesh.material.flatShading = enabled;
            surfaceMesh.material.needsUpdate = true;
        }
    }

    // ----- Private -----

    function _clearMeshes() {
        [surfaceMesh, wireframeMesh, pointsMesh].forEach(m => {
            if (m) {
                scene.remove(m);
                if (m.geometry) m.geometry.dispose();
                if (m.material) m.material.dispose();
            }
        });
        surfaceMesh = null;
        wireframeMesh = null;
        pointsMesh = null;
    }

    function _onResize() {
        if (!container) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }

    function _animate() {
        requestAnimationFrame(_animate);
        controls.update();
        renderer.render(scene, camera);
    }

    return { init, loadMesh, setWireframe, setPointCloud, setNormals };
})();
