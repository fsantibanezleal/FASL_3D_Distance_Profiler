/**
 * app.js -- Main Application Controller
 * =======================================
 *
 * Orchestrates the UI: wires DOM controls to API calls, dispatches
 * results to the 2D and 3D renderers, and manages application state.
 */

(() => {
    "use strict";

    const API = "/api";

    // ----- Status bar -----
    function setStatus(msg, type) {
        const bar = document.getElementById("status-bar");
        bar.textContent = msg;
        bar.className = "status-bar" + (type ? ` ${type}` : "");
    }

    // ----- API helpers -----
    async function apiPost(endpoint, body) {
        setStatus(`Working... ${endpoint}`, "");
        try {
            const res = await fetch(`${API}${endpoint}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || res.statusText);
            }
            const data = await res.json();
            setStatus("Ready", "success");
            return data;
        } catch (e) {
            setStatus(`Error: ${e.message}`, "error");
            throw e;
        }
    }

    async function apiGet(endpoint) {
        setStatus(`Fetching... ${endpoint}`, "");
        try {
            const res = await fetch(`${API}${endpoint}`);
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || res.statusText);
            }
            const data = await res.json();
            setStatus("Ready", "success");
            return data;
        } catch (e) {
            setStatus(`Error: ${e.message}`, "error");
            throw e;
        }
    }

    // ----- Handle scene data -----
    function handleSceneData(data) {
        if (!data || !data.loaded) return;
        Renderer2D.updateImages(data.images, data.width, data.height);
        Renderer3D.loadMesh(data.mesh);
    }

    // ----- Profile callback -----
    async function onProfileLine(startX, startY, endX, endY) {
        try {
            const data = await apiPost("/profile", {
                start_x: startX,
                start_y: startY,
                end_x: endX,
                end_y: endY,
                num_samples: 256,
            });
            Renderer2D.drawProfile(data.distances, data.heights, data.metrics);
        } catch (e) {
            console.error("Profile error:", e);
        }
    }

    // ----- Initialisation -----
    document.addEventListener("DOMContentLoaded", () => {
        // Init renderers
        Renderer2D.init(onProfileLine);
        Renderer3D.init();

        // ----- Generate button -----
        document.getElementById("btn-generate").addEventListener("click", async () => {
            const sceneType = document.getElementById("select-scene").value;
            const width = parseInt(document.getElementById("input-width").value) || 256;
            const height = parseInt(document.getElementById("input-height").value) || 256;
            const seedStr = document.getElementById("input-seed").value;
            const seed = seedStr ? parseInt(seedStr) : null;

            const data = await apiPost("/generate", {
                scene_type: sceneType,
                width: width,
                height: height,
                seed: seed,
            });
            handleSceneData(data);
        });

        // ----- Upload button -----
        document.getElementById("btn-upload").addEventListener("click", async () => {
            const depthFile = document.getElementById("upload-depth").files[0];
            if (!depthFile) {
                setStatus("Please select a depth map file.", "error");
                return;
            }
            const rgbFile = document.getElementById("upload-rgb").files[0];

            const formData = new FormData();
            formData.append("depth_file", depthFile);
            if (rgbFile) formData.append("rgb_file", rgbFile);

            setStatus("Uploading...", "");
            try {
                const res = await fetch(`${API}/upload`, {
                    method: "POST",
                    body: formData,
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({ detail: res.statusText }));
                    throw new Error(err.detail || res.statusText);
                }
                const data = await res.json();
                setStatus("Upload complete", "success");
                handleSceneData(data);
            } catch (e) {
                setStatus(`Upload error: ${e.message}`, "error");
            }
        });

        // ----- Process button -----
        document.getElementById("btn-process").addEventListener("click", async () => {
            const sigmaS = parseFloat(document.getElementById("slider-sigma-s").value);
            const sigmaR = parseFloat(document.getElementById("slider-sigma-r").value);
            const holeSize = parseInt(document.getElementById("slider-hole").value);

            const data = await apiPost("/process", {
                bilateral_sigma_spatial: sigmaS,
                bilateral_sigma_range: sigmaR,
                fill_hole_size: holeSize,
            });
            handleSceneData(data);
        });

        // ----- Slider value displays -----
        const sliders = [
            { id: "slider-sigma-s", display: "val-sigma-s" },
            { id: "slider-sigma-r", display: "val-sigma-r" },
            { id: "slider-hole", display: "val-hole" },
        ];
        sliders.forEach(({ id, display }) => {
            const slider = document.getElementById(id);
            const disp = document.getElementById(display);
            slider.addEventListener("input", () => {
                disp.textContent = slider.value;
            });
        });

        // ----- Display toggles -----
        document.getElementById("chk-wireframe").addEventListener("change", (e) => {
            Renderer3D.setWireframe(e.target.checked);
        });
        document.getElementById("chk-normals").addEventListener("change", (e) => {
            Renderer3D.setNormals(e.target.checked);
        });
        document.getElementById("chk-points").addEventListener("change", (e) => {
            Renderer3D.setPointCloud(e.target.checked);
        });

        // ----- Metrics button -----
        document.getElementById("btn-metrics").addEventListener("click", async () => {
            try {
                const data = await apiGet("/metrics");
                const display = document.getElementById("metrics-display");
                let text = "";
                text += `Depth: min=${data.depth_stats.min.toFixed(1)} max=${data.depth_stats.max.toFixed(1)} mean=${data.depth_stats.mean.toFixed(1)} mm\n`;
                text += `Gauss K: mean=${data.curvature.gaussian_mean.toExponential(3)} std=${data.curvature.gaussian_std.toExponential(3)}\n`;
                text += `Mean H:  mean=${data.curvature.mean_curvature_mean.toExponential(3)} std=${data.curvature.mean_curvature_std.toExponential(3)}\n`;
                if (data.objects && data.objects.length > 0) {
                    text += `\nDetected ${data.objects.length} object(s):\n`;
                    data.objects.forEach(obj => {
                        text += `  #${obj.id}: area=${obj.area}px height=${obj.mean_height}mm\n`;
                    });
                }
                text += `\nHistogram: mean=${data.histogram.mean.toFixed(2)} std=${data.histogram.std.toFixed(2)} median=${data.histogram.median.toFixed(2)}`;
                display.textContent = text;
            } catch (e) {
                console.error("Metrics error:", e);
            }
        });

        // ----- Auto-generate initial scene -----
        setTimeout(async () => {
            try {
                const data = await apiPost("/generate", {
                    scene_type: "gaussian_hills",
                    width: 256,
                    height: 256,
                    seed: 42,
                });
                handleSceneData(data);
            } catch (e) {
                setStatus("Server not reachable. Start the backend on port 8009.", "error");
            }
        }, 500);
    });
})();
