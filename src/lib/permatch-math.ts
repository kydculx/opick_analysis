import type { SoccerMatch } from "./queries";

export function fnum(v: unknown, def = 0): number {
  const n = Number(v);
  return v == null || v === "" || Number.isNaN(n) ? def : n;
}

export function netWdl(v: { win?: number | null; draw?: number | null; loss?: number | null } | null | undefined): number {
  if (!v) return 0;
  return fnum(v.win) - fnum(v.loss);
}

export function sameWdl(m: SoccerMatch): { home: number; draw: number; away: number } | null {
  const s = m.same_odds as unknown as {
    win_draw_lose?: { home?: unknown; draw?: unknown; away?: unknown } | null;
    same_odds?: { win_draw_lose?: { home?: unknown; draw?: unknown; away?: unknown } | null } | null;
  } | null | undefined;
  if (!s || typeof s !== "object") return null;
  const w =
    s.win_draw_lose && typeof s.win_draw_lose === "object" ? s.win_draw_lose : s.same_odds?.win_draw_lose;
  if (!w) return null;
  const inv = [w.home, w.draw, w.away].map((v) => 1 / Number(v));
  if (inv.some((v) => !Number.isFinite(v) || v <= 0)) return null;
  const sum = inv[0] + inv[1] + inv[2];
  return { home: inv[0] / sum, draw: inv[1] / sum, away: inv[2] / sum };
}

export const ALL_FEATURES = [
  "rank", "power", "hstr", "cond", "att", "def", "val",
  "form5", "h2h5", "avg_goals", "avg_conceded", "avg_poss", "market",
];

export function selectFeatures(full: number[], names: string[] | null | undefined): number[] {
  if (!names || names.length === 0) return full;
  if (names.length === full.length && names.every((n, i) => n === ALL_FEATURES[i])) return full;
  return names.map((n) => {
    const i = ALL_FEATURES.indexOf(n);
    return i >= 0 ? (full[i] ?? 0) : 0;
  });
}

export function rowFeatures(m: SoccerMatch): number[] {
  const sh = m.strength?.home ?? {};
  const sa = m.strength?.away ?? {};
  const rf = (m.recent_form ?? {}) as Record<string, Record<string, { win?: number | null; loss?: number | null } | null> | undefined>;
  let formNet = 0;
  for (let i = 1; i <= 5; i++) {
    const h = rf.home?.[`recent_${i}`];
    const a = rf.away?.[`recent_${i}`];
    formNet += netWdl(h) - netWdl(a);
  }
  const hh = (m.head_to_head ?? {}) as Record<string, { win?: number | null; loss?: number | null } | null | undefined>;
  let h2hNet = 0;
  for (let i = 1; i <= 5; i++) h2hNet += netWdl(hh[`recent_${i}`]);
  const th = m.team_info?.avg_stats?.home ?? {};
  const ta = m.team_info?.avg_stats?.away ?? {};
  const imp = sameWdl(m);
  const market = imp ? imp.home - imp.away : 0;
  return [
    fnum(m.home_rank) - fnum(m.away_rank),
    fnum(sh.power) - fnum(sa.power),
    fnum(sh.head_to_head) - fnum(sa.head_to_head),
    fnum(sh.condition) - fnum(sa.condition),
    fnum(sh.attack) - fnum(sa.attack),
    fnum(sh.defense) - fnum(sa.defense),
    fnum(sh.value) - fnum(sa.value),
    formNet,
    h2hNet,
    fnum(th.goals) - fnum(ta.goals),
    fnum(th.conceded) - fnum(ta.conceded),
    fnum(th.possession) - fnum(ta.possession),
    market,
  ];
}

export function poissonDraw(xgH: number, xgA: number): number {
  let acc = 0;
  const fh = Math.exp(-xgH);
  const fa = Math.exp(-xgA);
  let ph = fh;
  let pa = fa;
  for (let i = 0; i <= 10; i++) {
    if (i > 0) {
      ph = (ph * xgH) / i;
      pa = (pa * xgA) / i;
    }
    acc += ph * pa;
  }
  return acc;
}

export function drawFeatures(m: SoccerMatch): number[] {
  const th = m.team_info?.avg_stats?.home ?? {};
  const ta = m.team_info?.avg_stats?.away ?? {};
  const xgH = Math.max(0.05, (fnum(th.goals) + fnum(ta.conceded)) / 2);
  const xgA = Math.max(0.05, (fnum(ta.goals) + fnum(th.conceded)) / 2);
  let rg = 0;
  try {
    rg = Math.abs(Number(m.home_rank) - Number(m.away_rank));
    if (Number.isNaN(rg)) rg = 0;
  } catch {
    rg = 0;
  }
  const imp = sameWdl(m);
  const rf = (m.recent_form ?? {}) as Record<string, Record<string, { draw?: unknown; scored?: unknown; conceded?: unknown } | null> | undefined>;
  const hd5 = [1, 2, 3, 4, 5].filter((i) => fnum(rf.home?.[`recent_${i}`]?.draw) > 0).length;
  const ad5 = [1, 2, 3, 4, 5].filter((i) => fnum(rf.away?.[`recent_${i}`]?.draw) > 0).length;
  const hh = (m.head_to_head ?? {}) as Record<string, { draw?: unknown } | null | undefined>;
  const hd = [1, 2, 3, 4, 5].filter((i) => fnum(hh[`recent_${i}`]?.draw) > 0).length;
  let sc5 = 0;
  let co5 = 0;
  for (const side of [rf.home, rf.away]) {
    for (let i = 1; i <= 5; i++) {
      const q = side?.[`recent_${i}`];
      sc5 += fnum(q?.scored);
      co5 += fnum(q?.conceded);
    }
  }
  const pg = Math.abs(fnum(th.possession) - fnum(ta.possession));
  return [poissonDraw(xgH, xgA), rg, xgH + xgA, imp ? imp.draw : 0, hd5, ad5, hd, Math.abs(xgH - xgA), sc5, pg, co5];
}

export function sigmoid(s: number): number {
  const c = Math.max(-30, Math.min(30, s));
  return 1 / (1 + Math.exp(-c));
}

export type LegacyArt = {
  weights?: number[] | null;
  hfa?: number | null;
  draw_prior?: number | null;
  mu: number[];
  sd: number[];
  features?: string[] | null;
  emphasis?: number[] | null;
  draw_weights?: number[] | null;
  draw_bias?: number | null;
  draw_mu?: number[] | null;
  draw_sd?: number[] | null;
  contrib_cap?: number | null;
};

export function stdVec(x: number[], mu: number[], sd: number[]): number[] {
  return x.map((v, i) => (v - mu[i]) / sd[i]);
}

export function legacyParts(
  m: SoccerMatch,
  art: LegacyArt
): { x: number[]; ph: number; dd: number } | null {
  const e = art.emphasis ?? new Array(art.features?.length ?? art.mu.length).fill(1);
  const x = stdVec(selectFeatures(rowFeatures(m), art.features), art.mu, art.sd).map((v, i) => v * e[i]);
  if (!(art.weights && art.hfa != null && art.draw_prior != null)) return null;
  const s = cappedDot(x, art.weights, art.hfa, art.contrib_cap ?? null);
  const ph = sigmoid(s);
  let dd: number;
  if (art.draw_weights && art.draw_mu && art.draw_sd) {
    const df = drawFeatures(m);
    const n = Math.min(df.length, art.draw_weights.length, art.draw_mu.length, art.draw_sd.length);
    const xd = stdVec(df.slice(0, n), art.draw_mu.slice(0, n), art.draw_sd.slice(0, n));
    const ds = xd.reduce((a, v, i) => a + v * (art.draw_weights as number[])[i], 0) + (art.draw_bias ?? 0);
    dd = sigmoid(ds);
  } else {
    dd = art.draw_prior;
  }
  return { x, ph, dd };
}

export function cappedDot(x: number[], w: number[], hfa: number, cap: number | null | undefined): number {
  if (!cap || cap <= 0) return x.reduce((a, v, i) => a + v * w[i], hfa);
  return x.reduce((a, v, i) => a + cap * Math.tanh((v * w[i]) / (cap as number)), hfa);
}

function factorial(n: number): number {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

export function dcProba(lam: number, mu: number, rho: number, maxGoals = 10): number[] {  const grid: number[][] = [];
  for (let x = 0; x <= maxGoals; x++) {
    grid[x] = [];
    const px = (lam ** x * Math.exp(-lam)) / factorial(x);
    for (let y = 0; y <= maxGoals; y++) {
      let tau = 1;
      if (x === 0 && y === 0) tau = 1 - lam * mu * rho;
      else if (x === 0 && y === 1) tau = 1 + lam * rho;
      else if (x === 1 && y === 0) tau = 1 + mu * rho;
      else if (x === 1 && y === 1) tau = 1 - rho;
      grid[x][y] = px * ((mu ** y * Math.exp(-mu)) / factorial(y)) * Math.max(tau, 1e-12);
    }
  }
  const s = grid.flat().reduce((a, b) => a + b, 0) || 1;
  let hw = 0;
  let aw = 0;
  for (let x = 0; x <= maxGoals; x++) {
    for (let y = 0; y <= maxGoals; y++) {
      const p = grid[x][y] / s;
      if (x > y) hw += p;
      else if (y > x) aw += p;
    }
  }
  return [hw, Math.max(0, 1 - hw - aw), aw];
}

export function applyTemp(p: number[], T: number): number[] {
  if (T === 1) return p;
  const L = p.map((v) => Math.log(Math.max(v, 1e-9)) / T);
  const mx = Math.max(...L);
  const E = L.map((v) => Math.exp(v - mx));
  const s = E.reduce((a, b) => a + b, 0);
  return E.map((v) => v / s);
}

export type PatternSet = { home: number[][]; draw: number[][]; away: number[][] };

export function metricWeights(w: number[] | null | undefined, n: number): number[] | null {
  if (!w || w.length !== n) return null;
  const m = w.map((v) => Math.abs(v));
  if (Math.max(...m) < 1e-9) return new Array(n).fill(1);
  const mean = m.reduce((a, b) => a + b, 0) / n || 1;
  return m.map((v) => v / mean);
}

function classTau(cname: string, tau: number, taus?: Record<string, number> | null): number {
  if (taus && Number.isFinite(taus[cname]) && (taus[cname] as number) > 0) return taus[cname] as number;
  if (Number.isFinite(tau) && tau > 0) return tau;
  return 1;
}

export function patternProbs(
  x: number[],
  patterns: PatternSet | undefined,
  tau: number,
  w?: number[] | null,
  taus?: Record<string, number> | null,
  ): number[] | null {
  if (!patterns || !tau || tau <= 0) return null;
  const names = ["home", "draw", "away"] as const;
  const groups = [patterns.home, patterns.draw, patterns.away];
  if (groups.some((g) => !Array.isArray(g) || g.length === 0)) return null;
  const mw = metricWeights(w ?? null, x.length);
  const scores = groups.map((g, gi) => {
    let best = Infinity;
    for (const c of g) {
      let d = 0;
      for (let i = 0; i < x.length; i++) {
        const diff = x[i] - c[i];
        d += mw ? mw[i] * diff * diff : diff * diff;
      }
      if (d < best) best = d;
    }
    return -Math.sqrt(Math.max(best, 0)) / classTau(names[gi], tau, taus ?? null);
  });
  const mx = Math.max(...scores);
  const ex = scores.map((s) => Math.exp(s - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}

export function blendProbs(
  lin: number[],
  x: number[],
  patterns: PatternSet | undefined,
  tau: number,
  alpha = 0.5,
  w?: number[] | null,
  taus?: Record<string, number> | null,
  ): number[] {
  const a = Number.isFinite(alpha) ? Math.min(1, Math.max(0, alpha)) : 0.5;
  if (a <= 0) return lin;
  const pp = patternProbs(x, patterns, tau, w ?? null, taus ?? null);
  if (!pp) return lin;
  if (a >= 1) return pp;
  return lin.map((v, i) => (1 - a) * v + a * pp[i]);
}

export function logregProba(
  x: number[],
  mean: number[],
  scale: number[],
  coef: number[][],
  intercept: number[]
): number[] {
  const z = x.map((v, i) => (v - mean[i]) / (scale[i] || 1));
  const logits = coef.map((row, c) => row.reduce((a, w, i) => a + w * (z[i] ?? 0), intercept[c] ?? 0));
  const mx = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(Math.max(-30, Math.min(30, v - mx))));
  const s = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((v) => v / s);
}
