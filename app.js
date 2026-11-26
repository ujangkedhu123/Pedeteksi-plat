import { YOLO, annotate } from "@ultralytics/yolo";

/**
 * YOLO26 browser inference template
 * ---------------------------------
 * Keep the default model in ./models/ so the app works from a GitHub Pages
 * project site (including repositories served below /<repository-name>/).
 */
const CONFIG = Object.freeze({
  defaultModelPath: "models/best.onnx",
  defaultConfidence: 0.25,
  annotation: {
    lineWidth: 3,
    font: "600 14px Manrope, sans-serif",
    labels: true,
    keypoints: true,
  },
});

const elements = {
  video: document.querySelector("#webcam"),
  image: document.querySelector("#image-preview"),
  canvas: document.querySelector("#overlay"),
  stage: document.querySelector("#preview-stage"),
  emptyState: document.querySelector("#empty-state"),
  statusIndicator: document.querySelector("#status-indicator"),
  statusText: document.querySelector("#status-text"),
  modelState: document.querySelector("#model-state"),
  deviceState: document.querySelector("#device-state"),
  previewMode: document.querySelector("#preview-mode"),
  previewHint: document.querySelector("#preview-hint-text"),
  modelPath: document.querySelector("#model-path"),
  modelFile: document.querySelector("#model-file"),
  modelFileLabel: document.querySelector("#model-file-label"),
  loadModel: document.querySelector("#load-model"),
  startCamera: document.querySelector("#start-camera"),
  stopCamera: document.querySelector("#stop-camera"),
  cameraFacing: document.querySelector("#camera-facing"),
  chooseImage: document.querySelector("#choose-image"),
  imageFile: document.querySelector("#image-file"),
  clearPreview: document.querySelector("#clear-preview"),
  confidence: document.querySelector("#confidence"),
  confidenceValue: document.querySelector("#confidence-value"),
  detectionCount: document.querySelector("#detection-count"),
  detectionBadge: document.querySelector("#detection-badge"),
  inferenceTime: document.querySelector("#inference-time"),
  fpsValue: document.querySelector("#fps-value"),
  detectionList: document.querySelector("#detection-list"),
};

const state = {
  model: null,
  stream: null,
  running: false,
  frameRequest: 0,
  imageUrl: null,
  activeSource: null,
  lastFrameAt: 0,
  isLoading: false,
};

function setStatus(message, type = "idle") {
  elements.statusText.textContent = message;
  elements.statusIndicator.className = "status-indicator";
  if (type === "ready" || type === "running") elements.statusIndicator.classList.add("is-ready");
  if (type === "loading") elements.statusIndicator.classList.add("is-loading");
  if (type === "error") elements.statusIndicator.classList.add("is-error");
}

function setPreviewMode(label, live = false) {
  elements.previewMode.innerHTML = "";
  const dot = document.createElement("span");
  elements.previewMode.append(dot, document.createTextNode(` ${label}`));
  elements.previewMode.classList.toggle("is-live", live);
}

function setStageAspect(width, height) {
  if (width > 0 && height > 0) {
    elements.stage.style.aspectRatio = `${width} / ${height}`;
  }
}

function showSource(sourceType) {
  state.activeSource = sourceType;
  elements.video.classList.toggle("is-visible", sourceType === "camera");
  elements.image.classList.toggle("is-visible", sourceType === "image");
  elements.canvas.classList.toggle("is-visible", sourceType !== null);
  elements.emptyState.hidden = sourceType !== null;
  if (sourceType === null) {
    elements.canvas.width = 0;
    elements.canvas.height = 0;
    elements.stage.style.aspectRatio = "4 / 3";
  }
}

function clearDetections(message = "Predictions will appear here.") {
  elements.detectionCount.textContent = "—";
  elements.detectionBadge.textContent = "0";
  elements.inferenceTime.textContent = "—";
  elements.fpsValue.textContent = "—";
  elements.detectionList.replaceChildren();

  const empty = document.createElement("div");
  empty.className = "list-empty";
  const icon = document.createElement("span");
  icon.className = "list-empty-icon";
  icon.textContent = "⌁";
  const text = document.createElement("span");
  text.textContent = message;
  empty.append(icon, text);
  elements.detectionList.append(empty);
}

function updateDetections(results, frameTime = null) {
  const boxes = Array.isArray(results?.boxes) ? results.boxes : [];
  const classification = results?.probs;
  const count = boxes.length || (classification?.name ? 1 : 0);

  elements.detectionCount.textContent = String(count);
  elements.detectionBadge.textContent = String(count);

  const inference = results?.speed?.inference;
  elements.inferenceTime.textContent = Number.isFinite(inference) ? inference.toFixed(1) : "—";

  if (frameTime !== null && state.lastFrameAt > 0) {
    const elapsed = frameTime - state.lastFrameAt;
    if (elapsed > 0) elements.fpsValue.textContent = (1000 / elapsed).toFixed(1);
  }
  if (frameTime !== null) state.lastFrameAt = frameTime;

  elements.detectionList.replaceChildren();

  if (boxes.length) {
    const sortedBoxes = [...boxes].sort((a, b) => (b.conf ?? 0) - (a.conf ?? 0));
    for (const box of sortedBoxes) {
      const row = document.createElement("div");
      row.className = "detection-item";

      const swatch = document.createElement("span");
      swatch.className = "detection-swatch";
      if (typeof box.color === "string") swatch.style.setProperty("--swatch", box.color);

      const info = document.createElement("div");
      info.className = "detection-info";
      const name = document.createElement("span");
      name.className = "detection-name";
      name.textContent = box.name || `Class ${box.cls ?? "?"}`;
      const meta = document.createElement("span");
      meta.className = "detection-meta";
      meta.textContent = `CLASS ${box.cls ?? "—"}${box.id !== undefined ? ` · TRACK ${box.id}` : ""}`;
      info.append(name, meta);

      const score = document.createElement("span");
      score.className = "detection-score";
      score.textContent = Number.isFinite(box.conf) ? `${(box.conf * 100).toFixed(1)}%` : "—";
      row.append(swatch, info, score);
      elements.detectionList.append(row);
    }
    return;
  }

  if (classification?.name) {
    const row = document.createElement("div");
    row.className = "detection-item";
    const swatch = document.createElement("span");
    swatch.className = "detection-swatch";
    const info = document.createElement("div");
    info.className = "detection-info";
    const name = document.createElement("span");
    name.className = "detection-name";
    name.textContent = classification.name;
    const meta = document.createElement("span");
    meta.className = "detection-meta";
    meta.textContent = `TOP CLASS ${classification.top1 ?? "—"}`;
    info.append(name, meta);
    const score = document.createElement("span");
    score.className = "detection-score";
    score.textContent = Number.isFinite(classification.top1conf) ? `${(classification.top1conf * 100).toFixed(1)}%` : "—";
    row.append(swatch, info, score);
    elements.detectionList.append(row);
    return;
  }

  const empty = document.createElement("div");
  empty.className = "list-empty";
  empty.textContent = "No predictions above the current confidence threshold.";
  elements.detectionList.append(empty);
}

function currentPredictionOptions() {
  return { conf: Number(elements.confidence.value) };
}

async function loadModel() {
  if (state.isLoading) return;

  state.isLoading = true;
  elements.loadModel.disabled = true;
  elements.loadModel.textContent = "Loading model…";
  setStatus("Loading model and initializing browser runtime…", "loading");
  elements.modelState.textContent = "LOADING MODEL";

  try {
    const selectedFile = elements.modelFile.files?.[0];
    const typedPath = elements.modelPath.value.trim() || CONFIG.defaultModelPath;
    const modelSource = selectedFile || new URL(typedPath.replace(/^\.\//, ""), import.meta.url).href;

    // Automatic device selection prefers WebGPU when available and falls back to CPU/WASM.
    const loadedModel = await YOLO.load(modelSource, { device: "auto" });
    state.model = loadedModel;

    const actualDevice = loadedModel.device || "auto";
    elements.modelState.textContent = "MODEL READY";
    elements.deviceState.textContent = `DEVICE ${String(actualDevice).toUpperCase()}`;
    elements.startCamera.disabled = false;
    elements.chooseImage.disabled = false;
    elements.loadModel.innerHTML = "<svg viewBox=\"0 0 20 20\" fill=\"none\" aria-hidden=\"true\"><path d=\"m4 10 4 4 8-8\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/></svg> Reload model";
    setStatus("Model ready. Start a camera session or test an image.", "ready");
    elements.previewHint.textContent = "Model loaded. Choose a camera or an image to begin.";
  } catch (error) {
    console.error("YOLO26 model load failed:", error);
    elements.modelState.textContent = "MODEL ERROR";
    setStatus(error?.message ? `Model load failed: ${error.message}` : "Model load failed. Check the model file and try again.", "error");
  } finally {
    state.isLoading = false;
    elements.loadModel.disabled = false;
    if (elements.loadModel.textContent === "Loading model…") elements.loadModel.textContent = "Load model";
  }
}

function stopCamera({ keepPreview = false } = {}) {
  state.running = false;
  if (state.frameRequest) cancelAnimationFrame(state.frameRequest);
  state.frameRequest = 0;

  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
  }
  elements.video.pause();
  elements.video.srcObject = null;
  elements.stopCamera.disabled = true;
  elements.startCamera.disabled = !state.model;

  if (!keepPreview && state.activeSource === "camera") {
    showSource(null);
    setPreviewMode("STANDBY");
  }
}

async function startCamera() {
  if (!state.model) {
    setStatus("Load a model before starting the camera.", "error");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access is unavailable. Use HTTPS or localhost, or test an image instead.", "error");
    return;
  }

  stopCamera({ keepPreview: true });
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }

  try {
    setStatus("Requesting camera permission…", "loading");
    const facingMode = elements.cameraFacing.value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    elements.video.srcObject = state.stream;
    await elements.video.play();
    setStageAspect(elements.video.videoWidth, elements.video.videoHeight);
    showSource("camera");
    state.running = true;
    state.lastFrameAt = 0;
    elements.stopCamera.disabled = false;
    elements.startCamera.disabled = true;
    elements.previewHint.textContent = "Live frames are processed locally in this browser.";
    setPreviewMode("LIVE CAMERA", true);
    setStatus("Camera active. Running inference…", "running");
    clearDetections("Waiting for the first prediction…");
    runCameraLoop();
  } catch (error) {
    console.error("Camera initialization failed:", error);
    stopCamera({ keepPreview: true });
    setStatus(error?.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access or test an image."
      : `Could not start the camera: ${error?.message || "unknown error"}`, "error");
    setPreviewMode("CAMERA ERROR");
  }
}

async function runCameraLoop() {
  if (!state.running || !state.model) return;

  try {
    const results = await state.model.predict(elements.video, currentPredictionOptions());
    await annotate(elements.canvas, elements.video, results, CONFIG.annotation);
    updateDetections(results, performance.now());
  } catch (error) {
    console.error("Frame inference failed:", error);
    stopCamera();
    setStatus(`Inference stopped: ${error?.message || "unknown error"}`, "error");
    return;
  }

  if (state.running) state.frameRequest = requestAnimationFrame(runCameraLoop);
}

function clearPreview() {
  stopCamera({ keepPreview: true });
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }
  elements.image.removeAttribute("src");
  showSource(null);
  setPreviewMode("STANDBY");
  elements.previewHint.textContent = state.model
    ? "Model loaded. Choose a camera or an image to begin."
    : "Your media stays in this browser session.";
  clearDetections();
  setStatus(state.model ? "Model ready. Waiting for input." : "Waiting for model initialization", state.model ? "ready" : "idle");
}

async function analyzeImage(file) {
  if (!file || !state.model) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Please select a supported image file.", "error");
    return;
  }

  stopCamera({ keepPreview: true });
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(file);
  clearDetections("Analyzing image…");
  setStatus("Loading image…", "loading");
  setPreviewMode("IMAGE TEST");
  elements.previewHint.textContent = `Local image: ${file.name}`;

  const image = new Image();
  image.decoding = "async";
  image.onload = async () => {
    try {
      elements.image.src = state.imageUrl;
      setStageAspect(image.naturalWidth, image.naturalHeight);
      showSource("image");
      const results = await state.model.predict(image, currentPredictionOptions());
      await annotate(elements.canvas, image, results, CONFIG.annotation);
      updateDetections(results);
      setStatus("Image analysis complete.", "ready");
    } catch (error) {
      console.error("Image inference failed:", error);
      setStatus(`Image inference failed: ${error?.message || "unknown error"}`, "error");
      clearDetections("Image inference failed.");
    }
  };
  image.onerror = () => {
    setStatus("The selected image could not be decoded by this browser.", "error");
  };
  image.src = state.imageUrl;
}

// UI events

elements.loadModel.addEventListener("click", loadModel);
elements.startCamera.addEventListener("click", startCamera);
elements.stopCamera.addEventListener("click", () => {
  stopCamera();
  setStatus("Camera stopped. Model remains loaded.", "ready");
  elements.previewHint.textContent = "Camera is off. Your model is ready for another input.";
});
elements.clearPreview.addEventListener("click", clearPreview);
elements.chooseImage.addEventListener("click", () => elements.imageFile.click());
elements.imageFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) analyzeImage(file);
  event.target.value = "";
});
elements.modelFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  elements.modelFileLabel.textContent = file ? file.name : "Choose an ONNX file";
});
elements.confidence.addEventListener("input", () => {
  elements.confidenceValue.value = Number(elements.confidence.value).toFixed(2);
  elements.confidenceValue.textContent = Number(elements.confidence.value).toFixed(2);
});
elements.cameraFacing.addEventListener("change", () => {
  if (state.running) startCamera();
});

// Initialize the confidence readout and neutral empty state.
elements.confidence.value = String(CONFIG.defaultConfidence);
elements.confidenceValue.value = CONFIG.defaultConfidence.toFixed(2);
elements.confidenceValue.textContent = CONFIG.defaultConfidence.toFixed(2);
clearDetections();
