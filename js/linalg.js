// Tiny dense linear algebra for small matrices (arrays of Float64Array rows).

export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  next.gauss = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0;
    while (u === 0) u = next();
    v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return next;
}

export function zeros(r, c) {
  const M = new Array(r);
  for (let i = 0; i < r; i++) M[i] = new Float64Array(c);
  return M;
}
export function eye(n) {
  const M = zeros(n, n);
  for (let i = 0; i < n; i++) M[i][i] = 1;
  return M;
}
export function clone(A) { return A.map((row) => Float64Array.from(row)); }
export function rows(A) { return A.length; }
export function cols(A) { return A[0].length; }

export function matmul(A, B) {
  const r = A.length, k = B.length, c = B[0].length;
  const C = zeros(r, c);
  for (let i = 0; i < r; i++) {
    const Ai = A[i], Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < c; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}
export function T(A) {
  const r = A.length, c = A[0].length;
  const B = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) B[j][i] = A[i][j];
  return B;
}
export function map2(A, B, f) {
  return A.map((row, i) => row.map((v, j) => f(v, B[i][j])));
}
export function add(A, B) { return map2(A, B, (a, b) => a + b); }
export function sub(A, B) { return map2(A, B, (a, b) => a - b); }
export function scale(A, s) { return A.map((row) => row.map((v) => v * s)); }
export function axpy(Y, a, X) { // Y += a X in place
  for (let i = 0; i < Y.length; i++) for (let j = 0; j < Y[0].length; j++) Y[i][j] += a * X[i][j];
  return Y;
}
export function frob(A) {
  let s = 0;
  for (const row of A) for (const v of row) s += v * v;
  return Math.sqrt(s);
}
export function inner(A, B) {
  let s = 0;
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) s += A[i][j] * B[i][j];
  return s;
}
export function diag(vals, r = vals.length, c = vals.length) {
  const M = zeros(r, c);
  for (let i = 0; i < Math.min(r, c, vals.length); i++) M[i][i] = vals[i];
  return M;
}

// Orthogonal matrix built from Givens rotations with angles theta * (random in [-pi, pi]).
// theta = 0 -> identity (axis aligned), theta = 1 -> thoroughly mixed basis.
export function givensBasis(n, theta, seed) {
  const R = rng(seed);
  const Q = eye(n);
  for (let sweep = 0; sweep < 2; sweep++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const ang = theta * (R() * 2 - 1) * Math.PI * 0.5;
        const c = Math.cos(ang), s = Math.sin(ang);
        for (let k = 0; k < n; k++) {
          const a = Q[k][i], b = Q[k][j];
          Q[k][i] = c * a - s * b;
          Q[k][j] = s * a + c * b;
        }
      }
    }
  }
  return Q;
}

// Symmetric eigen-decomposition via cyclic Jacobi. Returns {vals, vecs} with vecs columns.
export function symEig(S) {
  const n = S.length;
  const A = clone(S);
  const V = eye(n);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const tau = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = [...Array(n).keys()].sort((i, j) => A[j][j] - A[i][i]);
  return { vals: idx.map((i) => A[i][i]), vecs: V.map((row) => Float64Array.from(idx.map((i) => row[i]))) };
}

export function singularValues(A) {
  const G = A.length <= A[0].length ? matmul(A, T(A)) : matmul(T(A), A);
  return symEig(G).vals.map((v) => Math.sqrt(Math.max(v, 0)));
}

// Full SVD of a square-ish matrix: A = U diag(s) V^T
export function svd(A) {
  const { vals, vecs: V } = symEig(matmul(T(A), A));
  const s = vals.map((v) => Math.sqrt(Math.max(v, 0)));
  const AV = matmul(A, V);
  const r = A.length, k = s.length;
  const U = zeros(r, k);
  for (let j = 0; j < k; j++) {
    const inv = s[j] > 1e-12 ? 1 / s[j] : 0;
    for (let i = 0; i < r; i++) U[i][j] = AV[i][j] * inv;
  }
  return { U, s, V };
}

export const NS_PRESETS = {
  tuned: { label: "Muon quintic (3.4445, −4.7750, 2.0315)", a: 3.4445, b: -4.775, c: 2.0315 },
  safe: { label: "Convergent quintic (2, −1.5, 0.5)", a: 2, b: -1.5, c: 0.5 },
  cubic: { label: "Classic cubic (1.5, −0.5, 0)", a: 1.5, b: -0.5, c: 0 },
};

// Newton–Schulz orthogonalization exactly as in Muon: X <- aX + (bA + cA^2)X, A = X X^T.
export function newtonSchulz(G, steps = 5, coef = NS_PRESETS.tuned) {
  const { a, b, c } = coef;
  let X = G;
  let transposed = false;
  if (X.length > X[0].length) { X = T(X); transposed = true; }
  const nrm = frob(X) + 1e-7;
  X = scale(X, 1 / nrm);
  for (let k = 0; k < steps; k++) {
    const A = matmul(X, T(X));
    const B = add(scale(A, b), scale(matmul(A, A), c));
    X = add(scale(X, a), matmul(B, X));
  }
  return transposed ? T(X) : X;
}

export function phi(x, coef) { return coef.a * x + coef.b * x ** 3 + coef.c * x ** 5; }

export function svdEntropy(s) {
  const tot = s.reduce((acc, v) => acc + v * v, 0) || 1;
  let H = 0;
  for (const v of s) { const p = (v * v) / tot; if (p > 0) H -= p * Math.log(p); }
  return H / Math.log(s.length);
}
