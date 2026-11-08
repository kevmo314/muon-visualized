// River-valley toy problem over an n x n weight matrix, plus SGD / AdamW / Muon.
import { zeros, matmul, T, frob, givensBasis, newtonSchulz, rng, NS_PRESETS } from "./linalg.js";

export const OPTS = ["sgd", "adamw", "muon", "amuse"];
export const OPT_LABEL = { sgd: "SGD-momentum", adamw: "AdamW", muon: "Muon", amuse: "AMUSE-style" };

// Loss lives in "landscape coordinates" C = U^T W V, where U, V are the Hessian's
// singular bases. Each entry C_ij has its own curvature h_ij. C_00 is the steepest
// valley-wall direction; C_rr (r = n-1) is the flattest "river" direction, and the
// valley floor can meander (wall centre A·sin(ω·C_rr)) and narrow into a gorge
// (wall stiffness multiplied by 1 + g·exp(-(C_rr - y_g)^2 / w^2)).
export const LANDSCAPES = {
  meander: { label: "Meandering river", meander: 0.2, omega: 1.0, gorge: 0 },
  straight: { label: "Straight river", meander: 0, omega: 1, gorge: 0 },
  switchback: { label: "Tight switchbacks", meander: 0.35, omega: 1.6, gorge: 0 },
  gorge: { label: "Bend into a narrow gorge", meander: 0.2, omega: 1.0, gorge: 3 },
};

export class Landscape {
  constructor({ n = 6, rotation = 1, seed = 7, hmax = 40, hmin = 0.01, kDom = 2, meander = 0, omega = 1.6, gorge = 0, gorgeAt = 1.8, gorgeWidth = 0.6 } = {}) {
    this.n = n;
    this.r = n - 1;
    this.kDom = kDom;
    this.U = givensBasis(n, rotation, seed);
    this.V = givensBasis(n, rotation, seed + 101);
    this.h = Array.from({ length: n }, (_, k) => hmax * Math.pow(hmin / hmax, k / (n - 1)));
    this.H = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
      this.H[i][j] = i === j ? this.h[i] : 0.5 * Math.sqrt(this.h[i] * this.h[j]);
    this.hThresh = this.h[kDom - 1] * 0.999;
    this.target = Array.from({ length: n }, (_, k) => (k < kDom ? 0 : 3));
    this.A = meander;
    this.omega = omega;
    this.gorge = gorge;
    this.gorgeAt = gorgeAt;
    this.gorgeWidth = gorgeWidth;
  }
  wallScale(y) { return 1 + this.gorge * Math.exp(-((y - this.gorgeAt) ** 2) / this.gorgeWidth ** 2); }
  wallScaleD(y) { const d = y - this.gorgeAt; return this.gorge * Math.exp(-(d * d) / this.gorgeWidth ** 2) * (-2 * d / this.gorgeWidth ** 2); }
  wallCentreD(y) { return this.A * this.omega * Math.cos(this.omega * y); }
  toC(W) { return matmul(matmul(T(this.U), W), this.V); }
  fromC(C) { return matmul(matmul(this.U, C), T(this.V)); }
  initialW() {
    const C = zeros(this.n, this.n);
    C[0][0] = 0.75;
    for (let k = 1; k < this.kDom; k++) C[k][k] = 0.4;
    return this.fromC(C);
  }
  wallCentre(y) { return this.A * Math.sin(this.omega * y); }
  // loss and gradient in C coordinates
  lossGradC(C) {
    const n = this.n, r = this.r, H = this.H;
    const D = zeros(n, n);
    let L = 0;
    const y = C[r][r], sc = this.wallScale(y);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let e, h = H[i][j];
      if (i === j) e = C[i][i] - (i === 0 ? this.wallCentre(y) : this.target[i]);
      else e = C[i][j];
      if (i === 0 && j === 0) h *= sc;
      L += 0.5 * h * e * e;
      D[i][j] += h * e;
    }
    const e0 = C[0][0] - this.wallCentre(y);
    D[r][r] += -H[0][0] * sc * e0 * this.wallCentreD(y) + 0.5 * H[0][0] * this.wallScaleD(y) * e0 * e0;
    return { L, D };
  }
  // Loss restricted to the (steep, river) plane with every other coordinate optimal.
  sliceLoss(x, y) {
    const e0 = x - this.wallCentre(y), er = y - this.target[this.r];
    return 0.5 * this.h[0] * this.wallScale(y) * e0 * e0 + 0.5 * this.h[this.r] * er * er;
  }
  // fraction of a matrix's energy that lies in the dominant (high-curvature) subspace
  domFraction(M) {
    const C = this.toC(M);
    let dom = 0, tot = 0;
    for (let i = 0; i < this.n; i++) for (let j = 0; j < this.n; j++) {
      const v = C[i][j] * C[i][j];
      tot += v;
      if (this.H[i][j] >= this.hThresh) dom += v;
    }
    return tot > 0 ? dom / tot : 0;
  }
  // energy split of an update: {wall, river, other}
  split(M) {
    const C = this.toC(M);
    let wall = 0, river = 0, other = 0;
    for (let i = 0; i < this.n; i++) for (let j = 0; j < this.n; j++) {
      const v = C[i][j] * C[i][j];
      if (this.H[i][j] >= this.hThresh) wall += v;
      else if (i === j) river += v;
      else other += v;
    }
    const tot = wall + river + other || 1;
    return { wall: wall / tot, river: river / tot, other: other / tot };
  }
}

export const DEFAULT_LR = { sgd: 0.03, adamw: 0.02, muon: 0.02, amuse: 0.02 };

export const AMUSE_T0 = 20, AMUSE_RHO = 0.6, AMUSE_WINDOW = 25;

export function makeRun(land, opt, cfg) {
  const n = land.n;
  const st = {
    opt, W: land.initialW(), M: zeros(n, n), V2: zeros(n, n), t: 0,
    Z: land.initialW(), // AMUSE: fast Muon sequence z; W holds the averaged sequence x
    noise: rng(cfg.seed ?? 1234),
    hist: [], diverged: false,
  };
  const C0 = land.toC(st.W);
  st.hist.push({ x: C0[0][0], y: C0[land.r][land.r], e: C0[0][0] - land.wallCentre(C0[land.r][land.r]), L: land.lossGradC(C0).L, split: null });
  return st;
}

export function lrAt(cfg, t) {
  if (cfg.schedule !== "wsd") return 1;
  const start = Math.floor(cfg.steps * 0.75);
  if (t < start) return 1;
  return Math.max(0, 1 - (t - start) / (cfg.steps - start));
}

export function stepRun(land, st, cfg) {
  if (st.diverged) return;
  const n = land.n;
  let evalW = st.W;
  if (st.opt === "amuse") {
    // gradient point y = w_t z + (1 - w_t) x, w_t = min(1, (T0/t)^rho): starts at the fast
    // Muon iterate, then drifts toward the averaged iterate (less dominated by the walls).
    const w = Math.min(1, Math.pow(AMUSE_T0 / Math.max(st.t, 1), AMUSE_RHO));
    evalW = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) evalW[i][j] = w * st.Z[i][j] + (1 - w) * st.W[i][j];
  }
  const C = land.toC(evalW);
  const { D } = land.lossGradC(C);
  if (cfg.noise > 0) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
      D[i][j] += cfg.noise * Math.sqrt(land.H[i][j]) * st.noise.gauss();
  }
  const G = land.fromC(D);
  const beta = 0.9;
  const lr = (cfg.lr?.[st.opt] ?? DEFAULT_LR[st.opt]) * cfg.lrMult * lrAt(cfg, st.t);
  let upd;
  st.t += 1;
  if (st.opt === "sgd") {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) st.M[i][j] = beta * st.M[i][j] + G[i][j];
    upd = st.M;
  } else if (st.opt === "adamw") {
    const b2 = 0.99, eps = 1e-8;
    upd = zeros(n, n);
    const bc1 = 1 - Math.pow(beta, st.t), bc2 = 1 - Math.pow(b2, st.t);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      st.M[i][j] = beta * st.M[i][j] + (1 - beta) * G[i][j];
      st.V2[i][j] = b2 * st.V2[i][j] + (1 - b2) * G[i][j] * G[i][j];
      upd[i][j] = (st.M[i][j] / bc1) / (Math.sqrt(st.V2[i][j] / bc2) + eps);
    }
  } else { // muon and amuse share the orthogonalized-momentum direction
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) st.M[i][j] = beta * st.M[i][j] + G[i][j];
    const nest = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) nest[i][j] = G[i][j] + beta * st.M[i][j];
    upd = newtonSchulz(nest, cfg.nsSteps ?? 5, NS_PRESETS[cfg.nsCoef ?? "tuned"]);
  }
  if (st.opt === "amuse") {
    // z takes the Muon step; x is a running average of z (anytime / schedule-free style).
    const c = 1 / Math.min(st.t + 1, AMUSE_WINDOW);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      st.Z[i][j] -= lr * upd[i][j];
      st.W[i][j] = (1 - c) * st.W[i][j] + c * st.Z[i][j];
    }
  } else {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) st.W[i][j] -= lr * upd[i][j];
  }
  const C2 = land.toC(st.W);
  const L = land.lossGradC(C2).L;
  const split = frob(upd) > 0 ? land.split(upd) : null;
  if (!isFinite(L) || L > 1e6) st.diverged = true;
  st.hist.push({ x: C2[0][0], y: C2[land.r][land.r], e: C2[0][0] - land.wallCentre(C2[land.r][land.r]), L, split });
}
