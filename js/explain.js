import { matmul, T, diag, givensBasis, newtonSchulz, svd, singularValues, svdEntropy, frob, rng, zeros, phi, NS_PRESETS } from "./linalg.js";
import { fit, font, C, withAlpha, onThemeChange } from "./canvas.js";

// ---------- shared helpers ----------
function el(tag, attrs = {}, html = "") {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") e.style.cssText = v; else e.setAttribute(k, v);
  }
  if (html) e.innerHTML = html;
  return e;
}
function slider(parent, { label, min, max, step, value, fmt = (v) => v, onInput }) {
  const wrap = el("div", { class: "ctrl" });
  const lab = el("label");
  const name = el("span", {}, label);
  const out = el("output");
  const inp = el("input", { type: "range", min, max, step, value });
  out.textContent = fmt(+value);
  inp.addEventListener("input", () => { out.textContent = fmt(+inp.value); onInput(+inp.value); });
  lab.append(name, out);
  wrap.append(lab, inp);
  parent.append(wrap);
  return inp;
}
function selectCtl(parent, { label, options, value, onInput }) {
  const wrap = el("div", { class: "ctrl" });
  wrap.append(el("label", {}, `<span>${label}</span>`));
  const s = el("select");
  for (const [k, v] of Object.entries(options)) {
    const o = el("option", { value: k }, v);
    if (k === value) o.selected = true;
    s.append(o);
  }
  s.addEventListener("change", () => onInput(s.value));
  wrap.append(s);
  parent.append(wrap);
  return s;
}
function hexToRgb(h) {
  h = h.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`;
}
const sign = (G) => G.map((r) => r.map((v) => (v > 0 ? 1 : v < 0 ? -1 : 0)));
const normF = (A) => { const f = frob(A) || 1; return A.map((r) => r.map((v) => v / f)); };

// synthetic momentum matrix: power-law spectrum in a (possibly rotated) basis
function momentum({ n = 16, alpha = 1.2, theta = 1, noise = 0, seed = 5 }) {
  const U = givensBasis(n, theta, seed), V = givensBasis(n, theta, seed + 31);
  const s = Array.from({ length: n }, (_, k) => Math.pow(k + 1, -alpha));
  let G = matmul(matmul(U, diag(s)), T(V));
  if (noise > 0) {
    const R = rng(seed + 77);
    G = G.map((r) => r.map((v) => v + noise * R.gauss() / Math.sqrt(n)));
  }
  return { G, U, V, s };
}
const UPDATES = (G) => ({ sgd: normF(G), adamw: normF(sign(G)), muon: normF(newtonSchulz(G, 5)) });
const NAMES = { sgd: "SGD-momentum", adamw: "AdamW", muon: "Muon" };

function statRow(parent) {
  const row = el("div", { class: "stats" });
  const cells = {};
  for (const k of ["sgd", "adamw", "muon"]) {
    const s = el("div", { class: "stat", style: `--c: var(--${k})` });
    s.innerHTML = `<div class="k">${NAMES[k]}</div><div class="v">–</div><div class="d"></div>`;
    row.append(s);
    cells[k] = { v: s.querySelector(".v"), d: s.querySelector(".d") };
  }
  parent.append(row);
  return cells;
}
function labButton(side, preset, text) {
  const b = el("button", { class: "btn small", style: "margin-top:12px" }, text);
  b.addEventListener("click", () => {
    window.labPreset?.(preset);
    document.getElementById("lab-section").scrollIntoView({ behavior: "smooth" });
  });
  side.append(b);
}

// ---------- 1. River valley / bulk (AMUSE) ----------
const river = {
  id: "river",
  title: "Bigger steps along the river",
  sub: "Kim et al. · AMUSE, 2026",
  quote: "By orthogonalizing the momentum matrix, Muon prevents the update from being dominated by a few large singular components, producing a more balanced matrix-valued step. Empirically, Muon updates contain substantially larger bulk components than SGD or Adam. Moreover, orthogonalization itself explicitly increases this relative bulk component. This provides a geometric explanation for Muon’s efficiency: orthogonalized momentum accelerates progress along the river directions.",
  cite: `Kim, Shin, Yun, Baek, Song &amp; Yun, <a href="https://arxiv.org/abs/2605.22432">AMUSE: Anytime Muon with Stable Gradient Evaluation</a> (2026), §1. The landscape picture comes from Wen et al. (2025) and Song et al. (2025).`,
  build(viz, side) {
    const st = { alpha: 1.2, theta: 1, k: 2 };
    const cv = el("canvas");
    viz.append(cv);
    const cells = statRow(viz);
    side.append(el("p", {}, "Loss landscapes in deep learning look like a <b>river valley</b>. A few high-curvature <b style='color:var(--wall)'>dominant</b> directions form steep walls. A large <b style='color:var(--river)'>bulk</b> of flat directions forms the river, and that is where most of the useful progress happens."));
    side.append(el("p", {}, "The bars show how far a unit-size update moves along each singular direction of the momentum, from the dominant (left) to the bulk (right). SGD's step simply copies the momentum, so the walls swallow it. Muon gives every direction the same step."));
    slider(side, { label: "Spectrum steepness α", min: 0.2, max: 2.5, step: 0.05, value: st.alpha, fmt: (v) => v.toFixed(2), onInput: (v) => { st.alpha = v; draw(); } });
    slider(side, { label: "Basis rotation (0 = axis-aligned)", min: 0, max: 1, step: 0.01, value: st.theta, fmt: (v) => v.toFixed(2), onInput: (v) => { st.theta = v; draw(); } });
    slider(side, { label: "Dominant directions k", min: 1, max: 6, step: 1, value: st.k, onInput: (v) => { st.k = v; draw(); } });
    side.append(el("div", { class: "takeaway" }, "<b>Try:</b> set rotation to 0. AdamW's per-entry normalization then matches Muon, because the singular directions <i>are</i> the coordinates. Rotate the basis and AdamW falls back toward SGD. Muon doesn't care."));
    labButton(side, "river", "Watch it in the 3D lab →");
    function draw() {
      const col = C();
      const n = 16;
      const { G, U, V } = momentum({ n, alpha: st.alpha, theta: st.theta });
      const ups = UPDATES(G);
      const coefs = {};
      for (const [k, D] of Object.entries(ups)) {
        const P = matmul(matmul(T(U), D), V);
        coefs[k] = Array.from({ length: n }, (_, i) => Math.abs(P[i][i]));
        let dom = 0;
        for (let i = 0; i < st.k; i++) dom += P[i][i] * P[i][i];
        const bulk = 1 - dom;
        cells[k].v.textContent = (bulk * 100).toFixed(0) + "%";
        cells[k].d.textContent = "of update energy in the bulk";
      }
      const { ctx, w, h } = fit(cv, 300);
      const L = 40, R = 10, Tp = 26, B = 42;
      const pw = w - L - R, ph = h - Tp - B;
      const max = Math.max(...Object.values(coefs).flat(), 0.05) * 1.08;
      const cw = pw / n;
      ctx.fillStyle = withAlpha(col.wall, 0.1);
      ctx.fillRect(L, Tp, cw * st.k, ph);
      ctx.fillStyle = withAlpha(col.river, 0.08);
      ctx.fillRect(L + cw * st.k, Tp, pw - cw * st.k, ph);
      font(ctx, 12, 600);
      ctx.fillStyle = col.wall;
      ctx.textAlign = "left";
      ctx.fillText("valley walls", L + 4, Tp - 8);
      ctx.fillStyle = col.river;
      ctx.textAlign = "right";
      ctx.fillText("river (bulk) →", L + pw - 4, Tp - 8);
      ctx.strokeStyle = col.line;
      ctx.lineWidth = 1;
      font(ctx, 11, 400, true);
      ctx.fillStyle = col.ink3;
      for (let g = 0; g <= 4; g++) {
        const v = (max * g) / 4, y = Tp + ph - (v / max) * ph;
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
        ctx.textAlign = "right"; ctx.fillText(v.toFixed(2), L - 6, y + 4);
      }
      const keys = ["sgd", "adamw", "muon"];
      const bw = (cw * 0.8) / 3;
      keys.forEach((k, j) => {
        ctx.fillStyle = col[k];
        coefs[k].forEach((v, i) => {
          const x = L + i * cw + cw * 0.1 + j * bw;
          const bh = (v / max) * ph;
          ctx.fillRect(x, Tp + ph - bh, bw - 1, bh);
        });
      });
      ctx.fillStyle = col.ink3;
      ctx.textAlign = "center";
      font(ctx, 11);
      for (let i = 0; i < n; i++) ctx.fillText(i + 1, L + i * cw + cw / 2, Tp + ph + 15);
      font(ctx, 12);
      ctx.fillStyle = col.ink2;
      ctx.fillText("singular direction of the momentum (1 = largest)", L + pw / 2, h - 6);
      ctx.save(); ctx.translate(12, Tp + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText("step along direction", 0, 0); ctx.restore();
    }
    draw();
    return draw;
  },
};

// ---------- 2. Rare directions (Keller Jordan) ----------
const rare = {
  id: "rare",
  title: "Rescue the rare directions",
  sub: "Jordan et al. · Muon blog, 2024",
  quote: "…the updates produced by both SGD-momentum and Adam for the 2D parameters […] typically have very high condition number. […] they are almost low-rank matrices, with the updates for all neurons being dominated by just a few directions. […] orthogonalization effectively increases the scale of other “rare directions” which have small magnitude in the update but are nevertheless important for learning.",
  cite: `Keller Jordan et al., <a href="https://kellerjordan.github.io/posts/muon/">Muon: An optimizer for hidden layers in neural networks</a> (Dec 2024).`,
  build(viz, side) {
    const st = { alpha: 1.6, noise: 0.02 };
    const cv = el("canvas");
    viz.append(cv);
    side.append(el("p", {}, "Each heatmap is a 16×16 update for a layer with 16 neurons (rows). The momentum is nearly low-rank: most rows are copies of the same pattern, so every neuron gets pushed the same way."));
    side.append(el("p", {}, "Below each heatmap are its singular values on a log scale. Muon flattens them to ≈1. The rare directions are now as loud as the dominant one, and the heatmap loses its stripes."));
    slider(side, { label: "How low-rank is the momentum (α)", min: 0.3, max: 3, step: 0.05, value: st.alpha, fmt: (v) => v.toFixed(2), onInput: (v) => { st.alpha = v; draw(); } });
    slider(side, { label: "Gradient noise", min: 0, max: 0.2, step: 0.005, value: st.noise, fmt: (v) => v.toFixed(3), onInput: (v) => { st.noise = v; draw(); } });
    side.append(el("div", { class: "takeaway" }, "<b>Condition number</b> = largest ÷ smallest singular value. <b>Effective rank</b> = exp(entropy of the normalized σ² spectrum), i.e. roughly how many directions the update really uses."));
    function draw() {
      const col = C();
      const n = 16;
      const { G } = momentum({ n, alpha: st.alpha, theta: 1, noise: st.noise, seed: 12 });
      const ups = UPDATES(G);
      const { ctx, w } = fit(cv, 10);
      const gap = 14, pw = (w - 2 * gap) / 3, hm = Math.min(pw, 240);
      const H = 22 + hm + 12 + 70 + 44;
      const { ctx: c2 } = fit(cv, H);
      ["sgd", "adamw", "muon"].forEach((k, idx) => {
        const D = ups[k];
        const x0 = idx * (pw + gap) + (pw - hm) / 2;
        const y0 = 22;
        font(c2, 13, 600);
        c2.fillStyle = col[k];
        c2.textAlign = "center";
        c2.fillText(k === "adamw" ? "AdamW (≈ sign)" : NAMES[k], x0 + hm / 2, 14);
        let m = 0;
        for (const r of D) for (const v of r) m = Math.max(m, Math.abs(v));
        const cs = hm / n;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const v = D[i][j] / (m || 1);
          c2.fillStyle = v >= 0 ? mix(col.panel2, col.pos, Math.min(1, v)) : mix(col.panel2, col.neg, Math.min(1, -v));
          c2.fillRect(x0 + j * cs, y0 + i * cs, cs + 0.5, cs + 0.5);
        }
        const s = singularValues(D);
        const by = y0 + hm + 12, bh = 70;
        const lo = -3, hi = 0.3;
        const smax = s[0];
        const bwid = hm / n;
        c2.fillStyle = withAlpha(col.ink3, 0.12);
        c2.fillRect(x0, by, hm, bh);
        c2.fillStyle = col[k];
        s.forEach((v, i) => {
          const lv = Math.max(lo, Math.log10(v / smax + 1e-12));
          const hh = ((lv - lo) / (hi - lo)) * bh;
          c2.fillRect(x0 + i * bwid + 1, by + bh - hh, bwid - 2, hh);
        });
        const p = s.map((v) => v * v), tot = p.reduce((a, b) => a + b, 0);
        let Hs = 0; for (const q of p) if (q > 0) Hs -= (q / tot) * Math.log(q / tot);
        const effRank = Math.exp(Hs);
        const cond = s[0] / Math.max(s[n - 1], 1e-12);
        font(c2, 12, 400, true);
        c2.fillStyle = col.ink2;
        c2.fillText(`cond ${cond > 1e4 ? cond.toExponential(0) : cond.toFixed(cond < 10 ? 2 : 0)} · eff. rank ${effRank.toFixed(1)}`, x0 + hm / 2, by + bh + 18);
      });
      font(c2, 11);
      c2.fillStyle = col.ink3;
      c2.textAlign = "center";
      c2.fillText("singular values, log scale relative to the largest (1 → 10⁻³)", w / 2, H - 6);
    }
    draw();
    return draw;
  },
};

// ---------- 3. Steepest descent under the spectral norm (Bernstein) ----------
const norm = {
  id: "norm",
  title: "Steepest descent in the right norm",
  sub: "Bernstein · Deriving Muon, 2025",
  quote: "The operator norm tells us how much a matrix can stretch an input vector. […] the operator norm of the weight update allows us to bound how much the outputs can change in response. […] we keep the singular vectors of the gradient but throw away the magnitudes of the singular values.",
  cite: `Jeremy Bernstein, <a href="https://jeremybernste.in/writing/deriving-muon">Deriving Muon</a> (Mar 2025). See also Bernstein &amp; Newhouse, <a href="https://arxiv.org/abs/2409.20325">Old Optimizer, New Norm</a> (2024).`,
  build(viz, side) {
    const st = { ratio: 0.25, phi: 0.5, psi: 0.0, spin: false };
    const cv = el("canvas");
    viz.append(cv);
    const cells = statRow(viz);
    side.append(el("p", {}, "A weight matrix maps inputs to outputs. Here the gradient is a 2×2 matrix. Each panel shows what one update ΔW does to every unit input direction x: the dashed circle is the budget, and the shape is the set of output changes ΔW·x."));
    side.append(el("p", {}, "Every update is scaled to the same operator-norm budget, so no input's output can move more than 1. Within that budget, Muon's ΔW = UVᵀ fills the circle and moves <i>every</i> input's output by the full amount. SGD and AdamW touch the circle in one direction and waste the rest."));
    slider(side, { label: "Gradient's 2nd singular value σ₂/σ₁", min: 0.02, max: 1, step: 0.01, value: st.ratio, fmt: (v) => v.toFixed(2), onInput: (v) => { st.ratio = v; draw(); } });
    const rot = slider(side, { label: "Rotate gradient", min: 0, max: 3.1416, step: 0.01, value: st.phi, fmt: (v) => (v * 180 / Math.PI).toFixed(0) + "°", onInput: (v) => { st.phi = v; draw(); } });
    const spinWrap = el("label", { style: "display:flex;gap:8px;align-items:center;font-size:13px;color:var(--ink-2);cursor:pointer" }, `<input type="checkbox"> Auto-rotate`);
    side.append(spinWrap);
    spinWrap.querySelector("input").addEventListener("change", (e) => { st.spin = e.target.checked; if (st.spin) loop(); });
    side.append(el("div", { class: "takeaway", style: "margin-top:12px" }, "The numbers below give the first-order loss decrease ⟨G, ΔW⟩ under that budget. Muon's is always the largest, because it <i>is</i> the steepest-descent direction for this norm. Turn on auto-rotate: AdamW's shape changes with the coordinate system, while Muon's never does."));
    function loop() {
      if (!st.spin || !cv.isConnected) return;
      st.phi = (st.phi + 0.01) % Math.PI;
      rot.value = st.phi;
      rot.dispatchEvent(new Event("input"));
      requestAnimationFrame(loop);
    }
    function draw() {
      const col = C();
      const R = (a) => [[Math.cos(a), -Math.sin(a)], [Math.sin(a), Math.cos(a)]];
      const G = matmul(matmul(R(st.phi), diag([1, st.ratio])), T(R(st.phi * 0.6 + 0.4)));
      const opn = (A) => singularValues(A)[0] || 1;
      const ups = {
        sgd: G.map((r) => r.map((v) => v / opn(G))),
        adamw: (() => { const S = sign(G); return S.map((r) => r.map((v) => v / opn(S))); })(),
        muon: newtonSchulz(G, 12, NS_PRESETS.safe),
      };
      const nuc = singularValues(G).reduce((a, b) => a + b, 0);
      const { ctx, w } = fit(cv, 10);
      const gap = 12, pw = (w - 2 * gap) / 3, H = Math.min(pw, 260) + 30;
      const { ctx: c } = fit(cv, H);
      ["sgd", "adamw", "muon"].forEach((k, idx) => {
        const D = ups[k];
        const cx = idx * (pw + gap) + pw / 2, cy = 30 + (H - 30) / 2, r = (Math.min(pw, H - 30) / 2) * 0.82;
        font(c, 13, 600);
        c.fillStyle = col[k];
        c.textAlign = "center";
        c.fillText(NAMES[k], cx, 16);
        c.strokeStyle = col.line;
        c.beginPath(); c.moveTo(cx - r * 1.1, cy); c.lineTo(cx + r * 1.1, cy); c.moveTo(cx, cy - r * 1.1); c.lineTo(cx, cy + r * 1.1); c.stroke();
        c.setLineDash([4, 4]);
        c.strokeStyle = col.ink3;
        c.beginPath(); c.arc(cx, cy, r, 0, 2 * Math.PI); c.stroke();
        c.setLineDash([]);
        c.beginPath();
        for (let t = 0; t <= 128; t++) {
          const a = (t / 128) * 2 * Math.PI, x = Math.cos(a), y = Math.sin(a);
          const ox = D[0][0] * x + D[0][1] * y, oy = D[1][0] * x + D[1][1] * y;
          const px = cx + ox * r, py = cy - oy * r;
          t ? c.lineTo(px, py) : c.moveTo(px, py);
        }
        c.closePath();
        c.fillStyle = withAlpha(col[k], 0.22);
        c.fill();
        c.strokeStyle = col[k];
        c.lineWidth = 2;
        c.stroke();
        c.lineWidth = 1;
        const gain = matmul(T(G), D);
        const val = gain[0][0] + gain[1][1];
        cells[k].v.textContent = val.toFixed(3);
        cells[k].d.textContent = `${((val / nuc) * 100).toFixed(0)}% of the best possible (nuclear norm ‖G‖* = ${nuc.toFixed(3)})`;
      });
    }
    draw();
    return draw;
  },
};

// ---------- 4. Newton–Schulz (Keller Jordan) ----------
const ns = {
  id: "ns",
  title: "How it's computed: Newton–Schulz",
  sub: "Jordan et al. · Muon blog, 2024",
  quote: "[The Newton–Schulz iteration] effectively replaces SGD-momentum’s update matrix with the nearest semi-orthogonal matrix to it […] it suffices to run the NS iteration for only 5 steps.",
  cite: `Keller Jordan et al., <a href="https://kellerjordan.github.io/posts/muon/">Muon: An optimizer for hidden layers in neural networks</a> (Dec 2024). Coefficients from the reference implementation.`,
  build(viz, side) {
    const st = { N: 5, coef: "tuned", alpha: 1.2 };
    const cv = el("canvas");
    viz.append(cv);
    const cells = el("div", { class: "stats", style: "grid-template-columns:repeat(2,1fr)" });
    cells.innerHTML = `<div class="stat" style="--c:var(--muon)"><div class="k">Singular values after N steps</div><div class="v" id="ns-range">–</div><div class="d">min … max (target ≈ 1)</div></div><div class="stat" style="--c:var(--ink-3)"><div class="k">Cost</div><div class="v" id="ns-cost">–</div><div class="d">matmuls per update, no SVD needed</div></div>`;
    viz.append(cells);
    side.append(el("p", {}, "Computing UVᵀ with an SVD is slow on GPUs. Instead, Muon normalizes M so that every singular value lies in [0, 1], then repeatedly applies an odd polynomial φ(x) = ax + bx³ + cx⁵. Because φ is odd, applying it to the matrix applies it to each singular value <i>without changing U or V</i>."));
    side.append(el("p", {}, "The dots are the singular values of a real-looking momentum matrix. Watch them climb toward 1 as you add iterations."));
    slider(side, { label: "Iterations N", min: 0, max: 10, step: 1, value: st.N, onInput: (v) => { st.N = v; draw(); } });
    selectCtl(side, { label: "Polynomial", options: Object.fromEntries(Object.entries(NS_PRESETS).map(([k, v]) => [k, v.label])), value: st.coef, onInput: (v) => { st.coef = v; draw(); } });
    side.append(el("div", { class: "takeaway", style: "margin-top:4px" }, "Muon's tuned quintic makes φ′(0) = 3.4 as large as possible, so tiny singular values grow fast. The price is that it never settles at exactly 1: it oscillates in roughly [0.7, 1.2]. Jordan et al. found that close enough to orthogonal is fine. The convergent polynomials are exact but need far more steps."));
    labButton(side, "ns", "Fewer NS steps in the 3D lab →");
    function draw() {
      const col = C();
      const coef = NS_PRESETS[st.coef];
      const { G } = momentum({ n: 16, alpha: st.alpha, theta: 1 });
      const s0 = singularValues(G);
      const f = Math.sqrt(s0.reduce((a, b) => a + b * b, 0));
      const xs = s0.map((v) => v / f);
      const iter = (x) => { for (let i = 0; i < st.N; i++) x = phi(x, coef); return x; };
      const ys = xs.map(iter);
      const { ctx: c, w, h } = fit(cv, 320);
      const L = 58, R = 14, Tp = 14, B = 36;
      const pw = w - L - R, ph = h - Tp - B;
      const ymax = 1.45;
      const X = (x) => L + x * pw, Y = (y) => Tp + ph - (y / ymax) * ph;
      if (st.coef === "tuned") { c.fillStyle = withAlpha(col.muon, 0.08); c.fillRect(L, Y(1.3), pw, Y(0.7) - Y(1.3)); }
      c.strokeStyle = col.line;
      font(c, 11, 400, true);
      c.fillStyle = col.ink3;
      for (let g = 0; g <= 6; g++) { const y = (g * 0.25); if (y > ymax) break; c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + pw, Y(y)); c.stroke(); c.textAlign = "right"; c.fillText(y.toFixed(2), L - 6, Y(y) + 4); }
      for (let g = 0; g <= 4; g++) { c.textAlign = "center"; c.fillText((g / 4).toFixed(2), X(g / 4), Tp + ph + 15); }
      c.setLineDash([4, 4]); c.strokeStyle = col.ink3;
      c.beginPath(); c.moveTo(X(0), Y(0)); c.lineTo(X(1), Y(1)); c.stroke();
      c.beginPath(); c.moveTo(X(0), Y(1)); c.lineTo(X(1), Y(1)); c.stroke();
      c.setLineDash([]);
      c.strokeStyle = col.muon; c.lineWidth = 2.2;
      c.beginPath();
      for (let t = 0; t <= 400; t++) { const x = t / 400, y = Math.min(ymax, Math.max(0, iter(x))); t ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)); }
      c.stroke(); c.lineWidth = 1;
      xs.forEach((x, i) => {
        const y = ys[i];
        c.strokeStyle = withAlpha(col.muon, 0.35);
        c.beginPath(); c.moveTo(X(x), Y(0)); c.lineTo(X(x), Y(Math.min(y, ymax))); c.stroke();
        c.fillStyle = col.sgd; c.beginPath(); c.arc(X(x), Y(0), 3.5, 0, 7); c.fill();
        c.fillStyle = col.muon; c.beginPath(); c.arc(X(x), Y(Math.min(y, ymax)), 4, 0, 7); c.fill();
      });
      font(c, 12); c.fillStyle = col.ink2; c.textAlign = "center";
      c.fillText("input singular value x (after normalizing M by its Frobenius norm)", L + pw / 2, h - 4);
      c.save(); c.translate(13, Tp + ph / 2); c.rotate(-Math.PI / 2); c.fillText(`φ applied ${st.N}× to x`, 0, 0); c.restore();
      document.getElementById("ns-range").textContent = `${Math.min(...ys).toFixed(2)} … ${Math.max(...ys).toFixed(2)}`;
      document.getElementById("ns-cost").textContent = `${st.N * 3}`;
    }
    draw();
    return draw;
  },
};

// ---------- 5. Moonshot: diverse spectrum + RMS matching ----------
const scale = {
  id: "scale",
  title: "A more diverse spectrum, at scale",
  sub: "Liu et al. · Moonshot AI, 2025",
  quote: "…the SVD entropy of Muon is higher than that of AdamW, which verifies the intuition that Muon can provide a more diverse spectrum of updates for the weight matrices.",
  cite: `Liu, Su, Yao et al., <a href="https://arxiv.org/abs/2502.16982">Muon is Scalable for LLM Training</a> (2025), §3.4. The same paper reports that “Muon achieves ∼2× computational efficiency compared to AdamW with compute optimal training”.`,
  build(viz, side) {
    const st = { alpha: 1.2, A: 4096, B: 11008 };
    const cv = el("canvas");
    viz.append(cv);
    const cv2 = el("canvas", { style: "margin-top:18px" });
    viz.append(cv2);
    side.append(el("p", {}, "<b>SVD entropy</b> measures how evenly a matrix spreads its energy across singular directions: 1 means perfectly flat, 0 means a single direction. Top chart: entropy of one update step as the momentum gets more low-rank (left to right)."));
    slider(side, { label: "Momentum spectrum steepness α", min: 0, max: 3, step: 0.05, value: st.alpha, fmt: (v) => v.toFixed(2), onInput: (v) => { st.alpha = v; draw(); } });
    side.append(el("p", { style: "margin-top:14px" }, "<b>Matching AdamW's update size.</b> An orthogonal A×B update has RMS 1/√max(A,B), which shrinks as layers get wider. Moonshot multiplies it by 0.2·√max(A,B), which lands it in AdamW's typical 0.2–0.4 RMS range. AdamW's learning rate and weight decay then transfer as-is."));
    const p2 = (v) => 2 ** v;
    slider(side, { label: "Rows A", min: 6, max: 14, step: 0.25, value: Math.log2(st.A), fmt: (v) => Math.round(p2(v)), onInput: (v) => { st.A = Math.round(p2(v)); draw(); } });
    slider(side, { label: "Columns B", min: 6, max: 14, step: 0.25, value: Math.log2(st.B), fmt: (v) => Math.round(p2(v)), onInput: (v) => { st.B = Math.round(p2(v)); draw(); } });
    side.append(el("div", { class: "takeaway" }, "Moonshot measured this entropy on trained weights of a large MoE model, where Muon was higher for over 90% of matrices. Here we compute it on single updates, to show where that diversity comes from."));
    const alphas = Array.from({ length: 31 }, (_, i) => i * 0.1);
    const curves = { sgd: [], adamw: [], muon: [] };
    for (const a of alphas) {
      const { G } = momentum({ n: 16, alpha: a, theta: 1, noise: 0.01 });
      const ups = UPDATES(G);
      for (const k of Object.keys(curves)) curves[k].push(svdEntropy(singularValues(ups[k])));
    }
    function draw() {
      const col = C();
      // entropy chart
      {
        const { ctx: c, w, h } = fit(cv, 230);
        const L = 40, R = 130, Tp = 12, B = 32;
        const pw = w - L - R, ph = h - Tp - B;
        const X = (a) => L + (a / 3) * pw, Y = (v) => Tp + ph - v * ph;
        c.strokeStyle = col.line; font(c, 11, 400, true); c.fillStyle = col.ink3;
        for (let g = 0; g <= 4; g++) { const v = g / 4; c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(L + pw, Y(v)); c.stroke(); c.textAlign = "right"; c.fillText(v.toFixed(2), L - 6, Y(v) + 4); }
        for (let a = 0; a <= 3; a++) { c.textAlign = "center"; c.fillText(a, X(a), Tp + ph + 15); }
        c.strokeStyle = col.ink3; c.setLineDash([3, 3]);
        c.beginPath(); c.moveTo(X(st.alpha), Tp); c.lineTo(X(st.alpha), Tp + ph); c.stroke(); c.setLineDash([]);
        for (const k of ["sgd", "adamw", "muon"]) {
          c.strokeStyle = col[k]; c.lineWidth = 2.2; c.beginPath();
          curves[k].forEach((v, i) => (i ? c.lineTo(X(alphas[i]), Y(v)) : c.moveTo(X(alphas[i]), Y(v))));
          c.stroke(); c.lineWidth = 1;
          const i = Math.round(st.alpha / 0.1), v = curves[k][i];
          c.fillStyle = col[k]; c.beginPath(); c.arc(X(st.alpha), Y(v), 4.5, 0, 7); c.fill();
          font(c, 12, 600); c.textAlign = "left";
          c.fillText(`${NAMES[k]} ${v.toFixed(2)}`, L + pw + 8, Y(curves[k][30]) + 4 + (k === "adamw" ? 0 : 0));
        }
        font(c, 12); c.fillStyle = col.ink2; c.textAlign = "center";
        c.fillText("momentum spectrum steepness α  (more low-rank →)", L + pw / 2, h - 4);
      }
      // RMS chart
      {
        const { ctx: c, w, h } = fit(cv2, 150);
        const L = 150, R = 20, Tp = 16;
        const pw = w - L - R;
        const maxv = 0.5;
        const X = (v) => L + Math.min(v, maxv) / maxv * pw;
        const raw = 1 / Math.sqrt(Math.max(st.A, st.B));
        const rows = [
          { k: "Muon, raw UVᵀ", v: raw, c: col.muon, alpha: 0.45 },
          { k: "Muon × 0.2√max(A,B)", v: 0.2, c: col.muon, alpha: 1 },
          { k: "AdamW (typical)", v: [0.2, 0.4], c: col.adamw, alpha: 1 },
        ];
        c.strokeStyle = col.line; font(c, 11, 400, true); c.fillStyle = col.ink3;
        for (let g = 0; g <= 5; g++) { const v = g / 10; c.beginPath(); c.moveTo(X(v), Tp); c.lineTo(X(v), h - 22); c.stroke(); c.textAlign = "center"; c.fillText(v.toFixed(1), X(v), h - 6); }
        rows.forEach((r, i) => {
          const y = Tp + 8 + i * 38;
          font(c, 12.5, 500); c.fillStyle = col.ink2; c.textAlign = "right";
          c.fillText(r.k, L - 10, y + 15);
          c.fillStyle = withAlpha(r.c, r.alpha);
          if (Array.isArray(r.v)) {
            c.fillRect(X(r.v[0]), y + 2, X(r.v[1]) - X(r.v[0]), 20);
            c.fillStyle = withAlpha(r.c, 0.25); c.fillRect(X(0), y + 2, X(r.v[0]) - X(0), 20);
          } else c.fillRect(X(0), y + 2, Math.max(2, X(r.v) - X(0)), 20);
          if (!Array.isArray(r.v)) { font(c, 11.5, 600, true); c.fillStyle = col.ink; c.textAlign = "left"; c.fillText(r.v.toFixed(r.v < 0.1 ? 4 : 2), X(r.v) + 6, y + 16); }
        });
        font(c, 11); c.fillStyle = col.ink3; c.textAlign = "left";
        c.fillText(`update RMS for a ${st.A}×${st.B} matrix`, L, 10);
      }
    }
    draw();
    return draw;
  },
};

const EXPLANATIONS = [river, rare, norm, ns, scale];

// ---------- tab wiring ----------
const tabs = document.getElementById("tabs");
const quoteEl = document.getElementById("ex-quote");
const citeEl = document.getElementById("ex-cite");
const vizEl = document.getElementById("ex-viz");
const sideEl = document.getElementById("ex-side");
let redraw = null;

function select(id) {
  const ex = EXPLANATIONS.find((e) => e.id === id) || EXPLANATIONS[0];
  for (const b of tabs.children) b.setAttribute("aria-selected", b.dataset.id === ex.id ? "true" : "false");
  quoteEl.textContent = "“" + ex.quote + "”";
  citeEl.innerHTML = "— " + ex.cite;
  vizEl.innerHTML = "";
  sideEl.innerHTML = "";
  redraw = ex.build(vizEl, sideEl);
  try { history.replaceState(null, "", "#" + ex.id); } catch {}
}
for (const ex of EXPLANATIONS) {
  const b = el("button", { class: "tab", role: "tab", "data-id": ex.id }, `<span class="t">${ex.title}</span><span class="s">${ex.sub}</span>`);
  b.addEventListener("click", () => select(ex.id));
  tabs.append(b);
}
const initial = location.hash.slice(1);
select(EXPLANATIONS.some((e) => e.id === initial) ? initial : "river");
let rt;
window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => redraw?.(), 80); });
onThemeChange(() => redraw?.());
document.fonts?.ready.then(() => redraw?.());
