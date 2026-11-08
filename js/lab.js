import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { Landscape, makeRun, stepRun, OPT_LABEL, lrAt } from "./sim.js";
import { fit, font, C, withAlpha, onThemeChange } from "./canvas.js";

const ALL = ["sgd", "adamw", "muon", "amuse"];
const DEFAULT_CFG = { rotation: 1, noise: 0, lrMult: 1, schedule: "wsd", nsSteps: 5, steps: 400, speed: 2 };
const cfg = { ...DEFAULT_CFG };
const visible = { sgd: true, adamw: true, muon: true, amuse: false };

const TRADEOFFS = [
  {
    id: "river", tag: "pro", title: "Big, consistent steps along the river",
    body: "Orthogonalization gives the flat river directions as big a step as the steep wall directions.",
    quote: "orthogonalized momentum accelerates progress along the river directions", src: "Kim et al., AMUSE (2026)",
    watch: "Follow the purple ball. Muon runs down the river toward the target ring. SGD crawls, because its step size is capped by the steep walls, and AdamW is stuck in a rotated basis. Compare the <b>river</b> % in the readout below.",
    preset: { rotation: 1, noise: 0, lrMult: 1, schedule: "wsd", nsSteps: 5 }, show: ["sgd", "adamw", "muon"],
  },
  {
    id: "basis", tag: "pro", title: "Doesn't care how the weights are rotated",
    body: "AdamW normalizes each entry, so its behaviour depends on which basis the matrix happens to be written in. Muon works on singular directions, so rotating the basis changes nothing.",
    watch: "This starts <b>axis-aligned</b> (rotation 0), where AdamW ≈ Muon. Now drag <b>Basis rotation</b> toward 1: AdamW stalls in the valley while Muon's path stays exactly the same.",
    preset: { rotation: 0, noise: 0, lrMult: 1, schedule: "wsd", nsSteps: 5 }, show: ["adamw", "muon"],
  },
  {
    id: "noise", tag: "con", title: "Not selective: it amplifies noise too",
    body: "Near the valley floor, the steep-direction part of the gradient is small and mostly noise. Orthogonalization still turns it into a full-size step.",
    quote: "Because orthogonalization is not selective, it can amplify noisy components as well as useful ones.", src: "Kim et al., AMUSE (2026)",
    watch: "Muon zig-zags across the valley walls; see its <b>wobble</b>. The green <b>AMUSE-style</b> run applies the paper's fix: it evaluates gradients at a point that drifts toward an averaged iterate. Its wobble is much smaller, at a small cost in early river speed.",
    preset: { rotation: 1, noise: 0.15, lrMult: 1.5, schedule: "constant", nsSteps: 5 }, show: ["adamw", "muon", "amuse"],
  },
  {
    id: "decay", tag: "con", title: "Bounces until the learning rate decays",
    body: "Muon's step has a fixed size, so it never shrinks on its own as the gradient shrinks. With a constant LR it keeps bouncing between the walls.",
    quote: "the sustained high learning rate phase and fast decaying phase are responsible for progress in the river and the mountain directions respectively", src: "Wen et al. (2025)",
    watch: "Constant LR: Muon reaches the river's end but keeps rattling, and loss plateaus. Switch <b>Schedule</b> to <b>Warmup-stable-decay</b>. In the final 25% the bouncing collapses and loss drops sharply.",
    preset: { rotation: 1, noise: 0.08, lrMult: 1.3, schedule: "constant", nsSteps: 5 }, show: ["sgd", "muon"],
  },
  {
    id: "ns", tag: "mix", title: "Newton–Schulz steps: compute vs. quality",
    body: "Each NS iteration costs 3 matmuls. With too few, the small singular values never get lifted, and Muon slides back toward SGD-like behaviour.",
    watch: "With <b>1 NS step</b>, Muon starts out nearly as slow as SGD. Drag <b>NS steps</b> to 5 (Muon's default) and the early river speed comes back.",
    preset: { rotation: 1, noise: 0, lrMult: 1, schedule: "wsd", nsSteps: 1 }, show: ["sgd", "muon"],
  },
];

// ---------- landscape -> world mapping ----------
const XR = [-1.1, 1.1], YR = [-0.3, 3.5];
const SX = 2.6, SZ = 1.55, YMID = (YR[0] + YR[1]) / 2;
const hOf = (L) => Math.max(0, (Math.log10(L + 0.01) + 2) * 0.6);
const wx = (x) => Math.max(XR[0], Math.min(XR[1], x)) * SX;
const wz = (y) => -(Math.max(YR[0], Math.min(YR[1], y)) - YMID) * SZ;

let land = new Landscape({ rotation: cfg.rotation });
let runs = {};
let playing = true;
let T = 0;

// ---------- three.js scene ----------
const canvas = document.getElementById("lab3d");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.set(4.6, 6.2, 6.0);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.3, -0.2);
controls.enableDamping = true;
controls.maxDistance = 22;
controls.minDistance = 3;
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(4, 8, 3);
scene.add(sun);

const surfGroup = new THREE.Group();
scene.add(surfGroup);
const trajGroup = new THREE.Group();
scene.add(trajGroup);
const traj = {};

function textSprite(text, color, size = 0.32) {
  const cv = document.createElement("canvas");
  const ctx = cv.getContext("2d");
  const fs = 64;
  ctx.font = `600 ${fs}px Inter, sans-serif`;
  const w = Math.ceil(ctx.measureText(text).width) + 20;
  cv.width = w; cv.height = fs + 24;
  ctx.font = `600 ${fs}px Inter, sans-serif`;
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, 10, cv.height / 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set((size * w) / cv.height, size, 1);
  return sp;
}

function buildSurface() {
  surfGroup.clear();
  const col = C();
  const nx = 90, ny = 140;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array((nx + 1) * (ny + 1) * 3);
  const colors = new Float32Array((nx + 1) * (ny + 1) * 3);
  const low = new THREE.Color(col.river), mid = new THREE.Color(col.panel2), high = new THREE.Color(col.wall);
  let p = 0;
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
    const x = XR[0] + ((XR[1] - XR[0]) * i) / nx, y = YR[0] + ((YR[1] - YR[0]) * j) / ny;
    const hh = hOf(land.sliceLoss(x, y));
    pos[p] = wx(x); pos[p + 1] = hh; pos[p + 2] = wz(y);
    const t = Math.min(1, hh / 2.2);
    const c = t < 0.4 ? low.clone().lerp(mid, t / 0.4) : mid.clone().lerp(high, (t - 0.4) / 0.6);
    colors[p] = c.r; colors[p + 1] = c.g; colors[p + 2] = c.b;
    p += 3;
  }
  const idx = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  geo.setIndex(idx);
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
  surfGroup.add(new THREE.Mesh(geo, mat));
  // grid lines (cross-sections)
  const lineMat = new THREE.LineBasicMaterial({ color: new THREE.Color(col.ink3), transparent: true, opacity: 0.28 });
  for (let y = 0; y <= 3.4; y += 0.34) {
    const pts = [];
    for (let i = 0; i <= 80; i++) { const x = XR[0] + ((XR[1] - XR[0]) * i) / 80; pts.push(new THREE.Vector3(wx(x), hOf(land.sliceLoss(x, y)) + 0.004, wz(y))); }
    surfGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lineMat));
  }
  for (let x = -1; x <= 1.001; x += 0.25) {
    const pts = [];
    for (let i = 0; i <= 120; i++) { const y = YR[0] + ((YR[1] - YR[0]) * i) / 120; pts.push(new THREE.Vector3(wx(x), hOf(land.sliceLoss(x, y)) + 0.004, wz(y))); }
    surfGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lineMat));
  }
  // river line along the valley floor
  const rpts = [];
  for (let i = 0; i <= 120; i++) { const y = YR[0] + ((YR[1] - YR[0]) * i) / 120; rpts.push(new THREE.Vector3(wx(0), hOf(land.sliceLoss(0, y)) + 0.01, wz(y))); }
  surfGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rpts), new THREE.LineBasicMaterial({ color: new THREE.Color(col.river), transparent: true, opacity: 0.9 })));
  // target ring
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.025, 8, 40), new THREE.MeshBasicMaterial({ color: new THREE.Color(col.good || col.river) }));
  ring.rotation.x = Math.PI / 2;
  ring.position.set(wx(0), hOf(land.sliceLoss(0, 3)) + 0.02, wz(3));
  surfGroup.add(ring);
  const labels = [
    [textSprite("river →", col.river), new THREE.Vector3(wx(0) + 0.9, 0.9, wz(1.5))],
    [textSprite("minimum", col.ink2, 0.24), new THREE.Vector3(wx(0), 0.45, wz(3.0))],
    [textSprite("valley wall", col.wall, 0.28), new THREE.Vector3(wx(-1), 2.6, wz(1.2))],
    [textSprite("start", col.ink2, 0.24), new THREE.Vector3(wx(0.75), hOf(land.sliceLoss(0.75, 0)) + 0.4, wz(0))],
  ];
  for (const [s, v] of labels) { s.position.copy(v); surfGroup.add(s); }
}

function buildTraj() {
  trajGroup.clear();
  const col = C();
  for (const k of ALL) {
    const mat = new LineMaterial({ color: new THREE.Color(col[k]).getHex(), linewidth: 3, depthTest: false, transparent: true, opacity: 0.95 });
    mat.resolution.set(canvas.clientWidth, canvas.clientHeight);
    const line = new Line2(new LineGeometry(), mat);
    line.renderOrder = 10;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 20, 14), new THREE.MeshBasicMaterial({ color: new THREE.Color(col[k]), depthTest: false }));
    head.renderOrder = 11;
    trajGroup.add(line, head);
    traj[k] = { line, head, n: 0, pts: [] };
  }
}

function syncTraj() {
  for (const k of ALL) {
    const t = traj[k], r = runs[k];
    const vis = visible[k] && !!r;
    t.line.visible = t.head.visible = vis;
    if (!vis) continue;
    const h = r.hist;
    if (t.n > h.length) { t.n = 0; t.pts = []; }
    for (let i = t.n; i < h.length; i++) {
      const p = h[i];
      const x = Math.max(XR[0], Math.min(XR[1], p.x)), y = Math.max(YR[0], Math.min(YR[1], p.y));
      t.pts.push(wx(p.x), hOf(land.sliceLoss(x, y)) + 0.03, wz(p.y));
    }
    t.n = h.length;
    if (t.pts.length >= 6) {
      t.line.geometry.dispose();
      t.line.geometry = new LineGeometry();
      t.line.geometry.setPositions(t.pts);
      t.line.computeLineDistances();
    }
    const L = t.pts.length;
    t.head.position.set(t.pts[L - 3], t.pts[L - 2] + 0.06, t.pts[L - 1]);
  }
}

function reset() {
  land = new Landscape({ rotation: cfg.rotation });
  runs = {};
  T = 0;
  for (const k of ALL) runs[k] = makeRun(land, k, cfg);
  for (const k of ALL) if (traj[k]) { traj[k].n = 0; traj[k].pts = []; }
  buildSurface();
  syncTraj();
  drawLoss();
  updateHud();
}

function stepAll(n) {
  for (let s = 0; s < n && T < cfg.steps; s++, T++) for (const k of ALL) if (visible[k]) stepRun(land, runs[k], cfg);
}

// ---------- UI ----------
const hud = document.getElementById("lab-hud");
hud.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
const toggles = document.getElementById("lab-toggles");
for (const k of ALL) {
  const lab = document.createElement("label");
  lab.style.setProperty("--c", `var(--${k === "amuse" ? "good" : k})`);
  lab.innerHTML = `<input type="checkbox" ${visible[k] ? "checked" : ""}> <span style="color:var(--${k === "amuse" ? "good" : k});font-weight:600">${OPT_LABEL[k]}</span>`;
  const cb = lab.querySelector("input");
  cb.dataset.opt = k;
  cb.addEventListener("change", () => { visible[k] = cb.checked; reset(); });
  toggles.append(lab);
}

const controlsEl = document.getElementById("lab-controls");
const ctlRefs = {};
function addSlider(key, label, min, max, step, fmt, rebuild = true) {
  const wrap = document.createElement("div");
  wrap.className = "ctrl";
  wrap.innerHTML = `<label><span>${label}</span><output></output></label><input type="range" min="${min}" max="${max}" step="${step}">`;
  const inp = wrap.querySelector("input"), out = wrap.querySelector("output");
  const set = (v) => { inp.value = v; out.textContent = fmt(+v); };
  set(cfg[key]);
  inp.addEventListener("input", () => { cfg[key] = +inp.value; out.textContent = fmt(+inp.value); if (rebuild) reset(); });
  controlsEl.append(wrap);
  ctlRefs[key] = set;
}
function addSelect(key, label, options) {
  const wrap = document.createElement("div");
  wrap.className = "ctrl";
  wrap.innerHTML = `<label><span>${label}</span></label><select>${Object.entries(options).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>`;
  const s = wrap.querySelector("select");
  s.value = cfg[key];
  s.addEventListener("change", () => { cfg[key] = s.value; reset(); });
  controlsEl.append(wrap);
  ctlRefs[key] = (v) => { s.value = v; };
}
const h3 = document.createElement("h3");
h3.textContent = "Knobs";
controlsEl.append(h3);
addSlider("rotation", "Basis rotation (0 = axis-aligned)", 0, 1, 0.01, (v) => v.toFixed(2));
addSlider("noise", "Gradient noise (steep-direction heavy)", 0, 0.3, 0.01, (v) => v.toFixed(2));
addSlider("lrMult", "Learning rate ×", 0.25, 2, 0.05, (v) => v.toFixed(2) + "×");
addSelect("schedule", "Schedule", { constant: "Constant LR", wsd: "Warmup-stable-decay (decay last 25%)" });
addSlider("nsSteps", "Newton–Schulz steps (Muon)", 1, 8, 1, (v) => String(v));
addSlider("speed", "Animation speed", 1, 8, 1, (v) => v + " steps/frame", false);

const tradeEl = document.getElementById("tradeoffs");
tradeEl.style.cssText = "display:flex;flex-direction:column;gap:10px";
const cards = {};
for (const t of TRADEOFFS) {
  const c = document.createElement("div");
  c.className = "card tradeoff";
  c.innerHTML = `<span class="tag ${t.tag}">${t.tag === "mix" ? "TRADEOFF" : t.tag.toUpperCase()}</span><h3>${t.title}</h3><p>${t.body}</p>` +
    (t.quote ? `<p><q>${t.quote}</q> <span class="src">· ${t.src}</span></p>` : "") +
    `<div class="watch"><b>What to watch:</b> ${t.watch}</div>`;
  c.addEventListener("click", () => applyPreset(t.id));
  tradeEl.append(c);
  cards[t.id] = c;
}

function applyPreset(id) {
  const t = TRADEOFFS.find((x) => x.id === id);
  if (!t) return;
  Object.assign(cfg, t.preset);
  for (const [k, v] of Object.entries(t.preset)) ctlRefs[k]?.(v);
  for (const k of ALL) visible[k] = t.show.includes(k);
  for (const cb of toggles.querySelectorAll("input")) cb.checked = visible[cb.dataset.opt];
  for (const [k, c] of Object.entries(cards)) c.classList.toggle("active", k === id);
  reset();
  playing = true;
  playBtn.textContent = "Pause";
}
window.labPreset = applyPreset;

const playBtn = document.getElementById("lab-play");
playBtn.addEventListener("click", () => {
  if (T >= cfg.steps) { reset(); playing = true; }
  else playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});
document.getElementById("lab-reset").addEventListener("click", () => { reset(); playing = true; playBtn.textContent = "Pause"; });

const hudCells = {};
function updateHud() {
  const active = ALL.filter((k) => visible[k]);
  if (hud.dataset.keys !== active.join()) {
    hud.innerHTML = "";
    for (const k of active) {
      const s = document.createElement("div");
      s.className = "stat";
      s.style.setProperty("--c", `var(--${k === "amuse" ? "good" : k})`);
      s.innerHTML = `<div class="k">${OPT_LABEL[k]}</div><div class="v"></div><div class="d"></div>`;
      hud.append(s);
      hudCells[k] = { v: s.querySelector(".v"), d: s.querySelector(".d") };
    }
    hud.dataset.keys = active.join();
  }
  for (const k of active) {
    const h = runs[k].hist, p = h[h.length - 1];
    const w = h.slice(-30);
    const wob = Math.sqrt(w.reduce((a, q) => a + q.x * q.x, 0) / w.length);
    const prog = Math.max(0, Math.min(1, p.y / 3));
    hudCells[k].v.textContent = runs[k].diverged ? "diverged" : `loss ${p.L.toExponential(1)}`;
    hudCells[k].d.innerHTML = `river <b>${(prog * 100).toFixed(0)}%</b> · wobble <b>${wob.toFixed(3)}</b>`;
  }
  const t = T;
  document.getElementById("lab-step").textContent = `step ${t} / ${cfg.steps} · lr ×${(cfg.lrMult * lrAt(cfg, t)).toFixed(2)}`;
  const active2 = Object.entries(cards).find(([, c]) => c.classList.contains("active"));
  document.getElementById("lab-overlay").innerHTML = active2 ? `<b>${TRADEOFFS.find((x) => x.id === active2[0]).title}</b>` : "";
  if (t >= cfg.steps) playBtn.textContent = "Replay";
}

const lossCv = document.getElementById("lab-loss");
function drawLoss() {
  const col = C();
  const { ctx: c, w, h } = fit(lossCv, 150);
  const L = 44, R = 10, Tp = 10, B = 22;
  const pw = w - L - R, ph = h - Tp - B;
  const lo = -5, hi = 1;
  const X = (t) => L + (t / cfg.steps) * pw;
  const Y = (v) => Tp + ph - ((Math.max(lo, Math.min(hi, Math.log10(v + 1e-12))) - lo) / (hi - lo)) * ph;
  c.strokeStyle = col.line;
  font(c, 10.5, 400, true);
  c.fillStyle = col.ink3;
  for (let e = lo; e <= hi; e++) { const y = Y(10 ** e); c.beginPath(); c.moveTo(L, y); c.lineTo(L + pw, y); c.stroke(); c.textAlign = "right"; c.fillText(`1e${e}`, L - 5, y + 3); }
  if (cfg.schedule === "wsd") {
    const x0 = X(Math.floor(cfg.steps * 0.75));
    c.fillStyle = withAlpha(col.accent, 0.08);
    c.fillRect(x0, Tp, L + pw - x0, ph);
    c.fillStyle = col.ink3; c.textAlign = "left"; c.fillText("LR decay", x0 + 4, Tp + 10);
  }
  c.textAlign = "center"; c.fillStyle = col.ink3;
  c.fillText("training step", L + pw / 2, h - 4);
  for (const k of ALL) {
    if (!visible[k]) continue;
    const hst = runs[k].hist;
    c.strokeStyle = col[k]; c.lineWidth = 1.8; c.beginPath();
    hst.forEach((p, i) => (i ? c.lineTo(X(i), Y(p.L)) : c.moveTo(X(i), Y(p.L))));
    c.stroke(); c.lineWidth = 1;
  }
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const k of ALL) traj[k]?.line.material.resolution.set(w, h);
}
window.addEventListener("resize", () => { resize(); drawLoss(); });
onThemeChange(() => { buildSurface(); buildTraj(); syncTraj(); drawLoss(); });

let frame = 0;
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (playing && T < cfg.steps) {
    stepAll(cfg.speed);
    syncTraj();
    if (frame++ % 2 === 0 || T >= cfg.steps) { drawLoss(); updateHud(); }
  }
  renderer.render(scene, camera);
}

buildTraj();
resize();
reset();
applyPreset("river");
animate();
