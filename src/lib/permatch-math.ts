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
  return [poissonDraw(xgH, xgA), rg, xgH + xgA, imp ? imp.draw : 0];
}

export function sigmoid(s: number): number {
  const c = Math.max(-30, Math.min(30, s));
  return 1 / (1 + Math.exp(-c));
}

export function cappedDot(x: number[], w: number[], hfa: number, cap: number | null | undefined): number {
  if (!cap || cap <= 0) return x.reduce((a, v, i) => a + v * w[i], hfa);
  return x.reduce((a, v, i) => a + cap * Math.tanh((v * w[i]) / (cap as number)), hfa);
}

export function softmaxLogits(x: number[], W: number[][], b: number[]): number[] {
  return [0, 1, 2].map((c) => x.reduce((a, v, i) => a + v * (W[i]?.[c] ?? 0), b[c] ?? 0));
}

export function softmaxProbs(logits: number[]): number[] {
  const mx = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(Math.max(-30, Math.min(30, v - mx))));
  const s = ex.reduce((a, v) => a + v, 0);
  return ex.map((v) => v / s);
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

export function patternProbs(x: number[], patterns: PatternSet | undefined, tau: number): number[] | null {
  if (!patterns || !tau || tau <= 0) return null;
  const groups = [patterns.home, patterns.draw, patterns.away];
  if (groups.some((g) => !Array.isArray(g) || g.length === 0)) return null;
  const scores = groups.map((g) => {
    let best = Infinity;
    for (const c of g) {
      let d = 0;
      for (let i = 0; i < x.length; i++) d += (x[i] - c[i]) ** 2;
      if (d < best) best = d;
    }
    return -Math.sqrt(best) / tau;
  });
  const mx = Math.max(...scores);
  const ex = scores.map((s) => Math.exp(s - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}

export function blendProbs(lin: number[], x: number[], patterns: PatternSet | undefined, tau: number): number[] {
  const pp = patternProbs(x, patterns, tau);
  if (!pp) return lin;
  return lin.map((v, i) => 0.5 * v + 0.5 * pp[i]);
}
