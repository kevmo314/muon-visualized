// Canvas helpers: DPR-aware sizing, theme colors.
export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export const C = () => ({
  sgd: css("--sgd"), adamw: css("--adamw"), muon: css("--muon"), amuse: css("--good"),
  river: css("--river"), wall: css("--wall"), ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"),
  line: css("--line"), panel: css("--panel"), panel2: css("--panel-2"), pos: css("--pos"), neg: css("--neg"),
  accent: css("--accent"),
});

export function fit(canvas, cssHeight) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = cssHeight ?? canvas.clientHeight;
  canvas.style.height = h + "px";
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

export function font(ctx, size = 12, weight = 400, mono = false) {
  ctx.font = `${weight} ${size}px ${mono ? '"JetBrains Mono", monospace' : '"Inter", system-ui, sans-serif'}`;
}

export function withAlpha(color, a) {
  // accepts #rrggbb
  if (color.startsWith("#") && color.length === 7) {
    const v = Math.round(a * 255).toString(16).padStart(2, "0");
    return color + v;
  }
  return color;
}

export function onThemeChange(fn) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn);
}
