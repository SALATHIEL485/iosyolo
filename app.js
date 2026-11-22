// ====== EDIT THESE ======
const MODEL_URL = "./my_model.onnx";
const INPUT_SIZE = 640;                       // must match the imgsz used for export
const CLASSES = ["duck"];         // same order as classes.txt
// ========================

const IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 100;
const COLORS = ["#ff9500", "#34c759", "#007aff", "#ff2d55", "#af52de", "#5ac8fa", "#ffcc00", "#ff6482"];

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const video = $("video");
const camBtn = $("camBtn");
const statusEl = $("status");
const badge = $("badge");

const work = document.createElement("canvas");
work.width = work.height = INPUT_SIZE;
const wctx = work.getContext("2d", { willReadFrequently: true });

let session = null;
let stream = null;
let running = false;
let busy = false;
let confThreshold = 0.35;

/* ---------- Model ---------- */
async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "./";
    ort.env.wasm.numThreads = 1;   // GitHub Pages can't send the headers multi-threading needs
    ort.env.wasm.proxy = false;
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    statusEl.textContent = "Model ready.";
    camBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = "Couldn’t load " + MODEL_URL + ". Check the file name and that it’s in the repo. (" + err.message + ")";
  }
}

/* ---------- Pre-processing (letterbox to square) ---------- */
function preprocess(source, sw, sh) {
  const r = Math.min(INPUT_SIZE / sw, INPUT_SIZE / sh);
  const nw = Math.round(sw * r), nh = Math.round(sh * r);
  const dx = (INPUT_SIZE - nw) / 2, dy = (INPUT_SIZE - nh) / 2;

  wctx.fillStyle = "rgb(114,114,114)";
  wctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  wctx.drawImage(source, dx, dy, nw, nh);

  const px = wctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const area = INPUT_SIZE * INPUT_SIZE;
  const data = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    data[i] = px[i * 4] / 255;
    data[area + i] = px[i * 4 + 1] / 255;
    data[2 * area + i] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]), r, dx, dy };
}

/* ---------- Post-processing for YOLO OBB ----------
   Output shape: [1, 4 + nc + 1, N]
   rows: cx, cy, w, h, class scores..., angle (radians) */
function decode(output, meta, sw, sh) {
  const [, C, N] = output.dims;
  const nc = C - 5;
  const d = output.data;
  const found = [];

  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) {
      const s = d[(4 + c) * N + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < confThreshold) continue;

    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    const ang = d[(4 + nc) * N + i];
    const cos = Math.cos(ang), sin = Math.sin(ang);
    const hw = w / 2, hh = h / 2;

    // corners in model space, then mapped back to the original image
    const pts = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => {
      const mx = cx + x * cos - y * sin;
      const my = cy + x * sin + y * cos;
      return [(mx - meta.dx) / meta.r, (my - meta.dy) / meta.r];
    });
    found.push({ cls, score: best, pts, cx: (cx - meta.dx) / meta.r, cy: (cy - meta.dy) / meta.r, rad: Math.hypot(w, h) / 2 / meta.r });
  }

  found.sort((a, b) => b.score - a.score);
  return nms(found.slice(0, 300)).slice(0, MAX_DETECTIONS);
}

/* ---------- Rotated NMS ---------- */
function area(p) {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
}
function ccw(p) { return area(p) < 0 ? p.slice().reverse() : p; }

function clip(subject, clipper) {
  let out = subject;
  for (let i = 0; i < clipper.length && out.length; i++) {
    const a = clipper[i], b = clipper[(i + 1) % clipper.length];
    const inside = (p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0;
    const inter = (p, q) => {
      const x1 = p[0], y1 = p[1], x2 = q[0], y2 = q[1];
      const x3 = a[0], y3 = a[1], x4 = b[0], y4 = b[1];
      const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) || 1e-9;
      const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
      return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
    };
    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const cur = input[j], prev = input[(j + input.length - 1) % input.length];
      if (inside(cur)) {
        if (!inside(prev)) out.push(inter(prev, cur));
        out.push(cur);
      } else if (inside(prev)) {
        out.push(inter(prev, cur));
      }
    }
  }
  return out;
}

function rotatedIoU(a, b) {
  if (Math.hypot(a.cx - b.cx, a.cy - b.cy) > a.rad + b.rad) return 0;   // quick reject
  const pa = ccw(a.pts), pb = ccw(b.pts);
  const inter = clip(pa, pb);
  if (inter.length < 3) return 0;
  const ia = Math.abs(area(inter));
  const ua = Math.abs(area(pa)) + Math.abs(area(pb)) - ia;
  return ua > 0 ? ia / ua : 0;
}

function nms(boxes) {
  const keep = [];
  for (const box of boxes) {
    let ok = true;
    for (const k of keep) {
      if (k.cls === box.cls && rotatedIoU(k, box) > IOU_THRESHOLD) { ok = false; break; }
    }
    if (ok) keep.push(box);
  }
  return keep;
}

/* ---------- Drawing ---------- */
function labelOf(c) { return CLASSES[c] || "class " + c; }

function draw(source, sw, sh, dets) {
  canvas.width = sw;
  canvas.height = sh;
  ctx.drawImage(source, 0, 0, sw, sh);

  const lw = Math.max(2, Math.round(sw / 300));
  ctx.lineWidth = lw;
  ctx.font = "600 " + Math.max(14, Math.round(sw / 45)) + "px -apple-system, Helvetica, Arial, sans-serif";
  ctx.textBaseline = "top";

  for (const d of dets) {
    const color = COLORS[d.cls % COLORS.length];
    ctx.strokeStyle = color;
    ctx.beginPath();
    d.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    const text = labelOf(d.cls) + " " + Math.round(d.score * 100) + "%";
    const tw = ctx.measureText(text).width + 10;
    const th = parseInt(ctx.font.match(/(\d+)px/)[1], 10) + 6;
    const top = d.pts.reduce((m, p) => (p[1] < m[1] ? p : m));
    const lx = Math.min(Math.max(0, top[0]), sw - tw);
    const ly = Math.max(0, top[1] - th);
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, lx + 5, ly + 3);
  }
}

function renderList(dets) {
  const list = $("list");
  if (!dets.length) { list.innerHTML = '<div class="empty">Nothing detected.</div>'; return; }
  list.innerHTML = "";
  dets.slice(0, 20).forEach((d) => {
    const row = document.createElement("div");
    row.className = "row";
    const left = document.createElement("span");
    left.className = "name";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = COLORS[d.cls % COLORS.length];
    left.append(dot, labelOf(d.cls));
    const right = document.createElement("small");
    right.textContent = Math.round(d.score * 100) + "%";
    row.append(left, right);
    list.appendChild(row);
  });
}

/* ---------- Run one frame ---------- */
async function detect(source, sw, sh) {
  const meta = preprocess(source, sw, sh);
  const feeds = {};
  feeds[session.inputNames[0]] = meta.tensor;
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  const dets = decode(output, meta, sw, sh);
  draw(source, sw, sh, dets);
  renderList(dets);
  return dets;
}

/* ---------- Camera ---------- */
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    statusEl.textContent = "Camera blocked. Allow camera access in Settings > Safari, then try again.";
    return;
  }
  video.srcObject = stream;
  await video.play();
  $("placeholder").hidden = true;
  running = true;
  camBtn.textContent = "Stop camera";
  camBtn.classList.add("stop");
  statusEl.textContent = "Detecting…";
  loop();
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  badge.hidden = true;
  camBtn.textContent = "Start camera";
  camBtn.classList.remove("stop");
  statusEl.textContent = "Camera stopped.";
}

async function loop() {
  if (!running) return;
  if (!busy && video.videoWidth) {
    busy = true;
    const t0 = performance.now();
    try {
      await detect(video, video.videoWidth, video.videoHeight);
      const ms = performance.now() - t0;
      badge.hidden = false;
      badge.textContent = Math.round(ms) + " ms · " + (1000 / ms).toFixed(1) + " FPS";
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
      stopCamera();
    }
    busy = false;
  }
  requestAnimationFrame(loop);
}

/* ---------- Photo ---------- */
$("photo").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file || !session) return;
  if (running) stopCamera();
  $("placeholder").hidden = true;
  const bmp = await createImageBitmap(file);
  statusEl.textContent = "Detecting…";
  await detect(bmp, bmp.width, bmp.height);
  statusEl.textContent = "Done.";
  e.target.value = "";
});

/* ---------- UI ---------- */
camBtn.addEventListener("click", () => (running ? stopCamera() : startCamera()));
$("conf").addEventListener("input", (e) => {
  confThreshold = parseFloat(e.target.value);
  $("confVal").textContent = confThreshold.toFixed(2);
});

loadModel();
