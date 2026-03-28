/**
 * renderer2d.js -- 2D Canvas Rendering for RGB, Depth, and Normal maps
 * =====================================================================
 *
 * Handles drawing base64-encoded PNG images onto HTML5 canvases and
 * provides the cross-section line drawing tool for profile extraction.
 *
 * Cross-Section Tool:
 *   1. First click on the depth canvas sets the start point.
 *   2. Second click sets the end point.
 *   3. A line overlay is drawn between the two points.
 *   4. The profile request is dispatched via a callback.
 */

const Renderer2D = (() => {
    "use strict";

    // Canvas references
    let canvasRgb, ctxRgb;
    let canvasDepth, ctxDepth;
    let canvasNormals, ctxNormals;
    let canvasProfile, ctxProfile;

    // Cross-section state
    let profileStart = null;
    let profileEnd = null;
    let profileCallback = null;

    // Stored images for redraw
    let currentImages = {};

    /**
     * Initialise the 2D renderer.
     * @param {Function} onProfileLine - Callback (startX, startY, endX, endY)
     */
    function init(onProfileLine) {
        canvasRgb = document.getElementById("canvas-rgb");
        ctxRgb = canvasRgb.getContext("2d");

        canvasDepth = document.getElementById("canvas-depth");
        ctxDepth = canvasDepth.getContext("2d");

        canvasNormals = document.getElementById("canvas-normals");
        ctxNormals = canvasNormals.getContext("2d");

        canvasProfile = document.getElementById("canvas-profile");
        ctxProfile = canvasProfile.getContext("2d");

        profileCallback = onProfileLine;

        // Cross-section click handler on depth canvas
        canvasDepth.addEventListener("click", _onDepthClick);
    }

    /**
     * Update all 2D canvases with new image data.
     * @param {Object} images - {rgb, depth, normals} base64 data URIs
     * @param {number} width
     * @param {number} height
     */
    function updateImages(images, width, height) {
        currentImages = images;

        // Resize canvases
        [canvasRgb, canvasDepth, canvasNormals].forEach(c => {
            c.width = width;
            c.height = height;
        });

        _drawBase64(ctxRgb, images.rgb, width, height);
        _drawBase64(ctxDepth, images.depth, width, height);
        _drawBase64(ctxNormals, images.normals, width, height);

        // Reset profile state
        profileStart = null;
        profileEnd = null;
    }

    /**
     * Draw a cross-section profile chart.
     * @param {number[]} distances - x-axis values
     * @param {(number|null)[]} heights - y-axis values (null = invalid)
     * @param {Object} metrics - roughness metrics {Ra, Rq, Rz, ...}
     */
    function drawProfile(distances, heights, metrics) {
        const w = canvasProfile.width;
        const h = canvasProfile.height;
        ctxProfile.clearRect(0, 0, w, h);

        // Filter valid points
        const valid = [];
        for (let i = 0; i < distances.length; i++) {
            if (heights[i] !== null && heights[i] !== undefined) {
                valid.push({ d: distances[i], h: heights[i] });
            }
        }
        if (valid.length < 2) return;

        const dMin = valid[0].d;
        const dMax = valid[valid.length - 1].d;
        const hMin = Math.min(...valid.map(v => v.h));
        const hMax = Math.max(...valid.map(v => v.h));
        const dRange = dMax - dMin || 1;
        const hRange = hMax - hMin || 1;

        const pad = 30;
        const plotW = w - 2 * pad;
        const plotH = h - 2 * pad;

        // Axes
        ctxProfile.strokeStyle = "#555";
        ctxProfile.lineWidth = 1;
        ctxProfile.beginPath();
        ctxProfile.moveTo(pad, pad);
        ctxProfile.lineTo(pad, h - pad);
        ctxProfile.lineTo(w - pad, h - pad);
        ctxProfile.stroke();

        // Labels
        ctxProfile.fillStyle = "#aab";
        ctxProfile.font = "10px monospace";
        ctxProfile.fillText(hMax.toFixed(1), 2, pad + 4);
        ctxProfile.fillText(hMin.toFixed(1), 2, h - pad);
        ctxProfile.fillText(dMin.toFixed(0), pad, h - pad + 12);
        ctxProfile.fillText(dMax.toFixed(0), w - pad - 20, h - pad + 12);

        // Profile line
        ctxProfile.strokeStyle = "#4ecdc4";
        ctxProfile.lineWidth = 1.5;
        ctxProfile.beginPath();
        let started = false;
        for (const pt of valid) {
            const px = pad + ((pt.d - dMin) / dRange) * plotW;
            const py = h - pad - ((pt.h - hMin) / hRange) * plotH;
            if (!started) {
                ctxProfile.moveTo(px, py);
                started = true;
            } else {
                ctxProfile.lineTo(px, py);
            }
        }
        ctxProfile.stroke();

        // Mean line
        const mean = valid.reduce((s, v) => s + v.h, 0) / valid.length;
        const meanY = h - pad - ((mean - hMin) / hRange) * plotH;
        ctxProfile.strokeStyle = "#e94560";
        ctxProfile.lineWidth = 0.8;
        ctxProfile.setLineDash([4, 4]);
        ctxProfile.beginPath();
        ctxProfile.moveTo(pad, meanY);
        ctxProfile.lineTo(w - pad, meanY);
        ctxProfile.stroke();
        ctxProfile.setLineDash([]);

        // Display metrics
        const metricsDiv = document.getElementById("profile-metrics");
        if (metrics && Object.keys(metrics).length > 0) {
            let text = "";
            for (const [k, v] of Object.entries(metrics)) {
                text += `${k}: ${v.toFixed(4)} mm\n`;
            }
            metricsDiv.textContent = text;
        }
    }

    // ----- Private -----

    function _drawBase64(ctx, dataUri, w, h) {
        const img = new Image();
        img.onload = () => {
            ctx.clearRect(0, 0, w, h);
            ctx.drawImage(img, 0, 0, w, h);
        };
        img.src = dataUri;
    }

    function _onDepthClick(e) {
        const rect = canvasDepth.getBoundingClientRect();
        const scaleX = canvasDepth.width / rect.width;
        const scaleY = canvasDepth.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        if (!profileStart) {
            profileStart = { x, y };
            // Draw start marker
            _drawMarker(ctxDepth, x, y, "#e94560");
        } else {
            profileEnd = { x, y };
            // Draw end marker and line
            _drawMarker(ctxDepth, x, y, "#4ecdc4");
            _drawLine(ctxDepth, profileStart.x, profileStart.y, x, y);

            if (profileCallback) {
                profileCallback(profileStart.x, profileStart.y, x, y);
            }
            // Reset for next profile
            profileStart = null;
            profileEnd = null;
        }
    }

    function _drawMarker(ctx, x, y, color) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    function _drawLine(ctx, x0, y0, x1, y1) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    return { init, updateImages, drawProfile };
})();
