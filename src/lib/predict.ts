import { existsSync, readFileSync } from "node:fs";
import type { SoccerMatch } from "@/lib/queries";
import { applyDrawBoost, applyTemp, blendProbs, cappedDot, dcProba, drawFeatures, legacyParts, logregProba, rowFeatures, selectFeatures, sigmoid } from "./permatch-math";
import { slimBooster, xgbProba, type XgbSlim } from "./xgb";

export type PermatchArtifact = {
  model_type?: string;
  weights?: number[];
  hfa?: number;
  T: number;
  draw_prior?: number;
  mu: number[];
  sd: number[];
  features?: string[];
  emphasis?: number[];
  draw_weights?: number[] | null;
  draw_bias?: number;
  draw_mu?: number[] | null;
  draw_sd?: number[] | null;
  patterns?: { home: number[][]; draw: number[][]; away: number[][] };
  pattern_tau?: number;
  pattern_taus?: Record<string, number> | null;
  pattern_alpha?: number;
  draw_boost?: number;
  draw_close_margin?: number;
  draw_total_max?: number;
  grid_params?: {
    grid_step?: unknown; jobs?: unknown; batch?: unknown;
    draw_w?: unknown; recency_w?: unknown; wmin?: unknown; wmax?: unknown;
  } | null;
  contrib_cap?: number | null;
  base_ver?: string;
  blend_w?: number;
  base?: PermatchArtifact | null;
  xgb?: XgbSlim | null;
  train_seasons?: string[];
  teams?: string[];
  att?: number[];
  def?: number[];
  home?: number;
  rho?: number;
  scaler_mean?: number[];
  scaler_scale?: number[];
  coef?: number[][];
  intercept?: number[];
  classes?: number[];
  lstm_file?: string;
  seq_len?: number;
  cumulative?: {
    base_ver?: string;
    walk?: { n: number; acc: number; seasons: Record<string, { n: number; acc: number }> };
  } | null;
};

export type PredEntry = { home: number; draw: number; away: number; ver: string; drawAlert: boolean };

function alertThreshold(art: PermatchArtifact): number {
  const t = (art as { cumulative?: { draw_alert_t?: unknown } }).cumulative?.draw_alert_t;
  return typeof t === "number" && Number.isFinite(t) && t > 0 ? t : 0.3;
}

function coerceArtifact(a: unknown): PermatchArtifact | null {
  try {
    if (!a || typeof a !== "object") return null;
    const o = a as Record<string, unknown>;
    if (o?.model_type === "poisson" || o?.model_type === "dixon") {
      if (!Array.isArray(o.teams) || !Array.isArray(o.att) || !Array.isArray(o.def)) return null;
      if ((o.teams as unknown[]).length !== (o.att as unknown[]).length) return null;
      if ((o.teams as unknown[]).length !== (o.def as unknown[]).length) return null;
      if (typeof o.home !== "number" || typeof o.rho !== "number") return null;
      return o as unknown as PermatchArtifact;
    }
    if (o?.model_type === "logreg") {
      if (!Array.isArray(o.scaler_mean) || !Array.isArray(o.scaler_scale)) return null;
      if (!Array.isArray(o.coef) || !Array.isArray(o.intercept)) return null;
      if ((o.coef as unknown[]).length !== (o.intercept as unknown[]).length) return null;
      return o as unknown as PermatchArtifact;
    }
    if (o?.model_type === "lstm") {
      return o as unknown as PermatchArtifact;
    }
    if (!Array.isArray(o.mu) || !Array.isArray(o.sd)) return null;
    const n = (o.weights as unknown[]).length;
    if (n === 0 || (o.mu as unknown[]).length !== n || (o.sd as unknown[]).length !== n) return null;
    if (Array.isArray(o.features) && (o.features as unknown[]).length !== n) return null;
    return o as unknown as PermatchArtifact;
  } catch {
    return null;
  }
}

export async function loadArtifactAsync(league: string, ver: string): Promise<PermatchArtifact | null> {
  try {
    const { createServerSupabaseClient } = await import("@/lib/supabase/server");
    const supabase = createServerSupabaseClient();
    const { data, error } = await supabase
      .from("models")
      .select("artifact")
      .eq("league_code", league)
      .eq("ver", ver)
      .limit(1);
    if (!error && data && data.length > 0) {
      const raw = (data[0] as { artifact?: unknown }).artifact;
      if (raw && typeof raw === "object") {
        const o = raw as Record<string, unknown>;
        if (o?.model_type === "ensemble") {
          const ens = await readEnsembleAsync(league, o);
          if (ens) return ens;
        } else {
          const c = coerceArtifact(raw);
          if (c) return c;
        }
      }
    }
  } catch {
    // fall through to local
  }
  return loadArtifact(league, ver);
}

async function readEnsembleAsync(league: string, a: Record<string, unknown>): Promise<PermatchArtifact | null> {
  try {
    const baseVer = typeof a.base_ver === "string" ? a.base_ver : "";
    if (!baseVer) return null;
    const baseRaw = await loadArtifactAsync(league, baseVer);
    if (!baseRaw || !Array.isArray(baseRaw.weights)) return null;
    const xgbFile = typeof a.xgb_file === "string" ? a.xgb_file : "";
    if (!xgbFile) return null;
    const xp = `${process.cwd()}/ml/permatch/${xgbFile}`;
    if (!existsSync(xp)) return null;
    const slim = slimBooster(JSON.parse(readFileSync(xp, "utf-8")));
    if (!slim) return null;
    const w = typeof a.blend_w === "number" && Number.isFinite(a.blend_w) ? a.blend_w : 1;
    const feats = Array.isArray(a.features) ? (a.features as string[]) : baseRaw.features;
    return {
      model_type: "ensemble",
      base_ver: baseVer,
      blend_w: Math.max(0, Math.min(1, w)),
      features: feats,
      train_seasons: [],
      base: baseRaw,
      xgb: slim,
    } as unknown as PermatchArtifact;
  } catch {
    return null;
  }
}
export function loadArtifact(league: string, ver: string): PermatchArtifact | null {
  try {
    const p = `${process.cwd()}/ml/permatch/${league}_${ver}.json`;
    if (!existsSync(p)) return null;
    const a = JSON.parse(readFileSync(p, "utf-8"));
    if (a?.model_type === "ensemble") {
      return readEnsemble(league, a as Record<string, unknown>);
    }
    if (a?.model_type === "poisson" || a?.model_type === "dixon") {
      if (!Array.isArray(a.teams) || !Array.isArray(a.att) || !Array.isArray(a.def)) return null;
      if (a.teams.length !== a.att.length || a.teams.length !== a.def.length) return null;
      if (typeof a.home !== "number" || typeof a.rho !== "number") return null;
      return a as PermatchArtifact;
    }
    if (a?.model_type === "logreg") {
      if (!Array.isArray(a.scaler_mean) || !Array.isArray(a.scaler_scale)) return null;
      if (!Array.isArray(a.coef) || !Array.isArray(a.intercept)) return null;
      if (a.coef.length !== a.intercept.length) return null;
      return a as PermatchArtifact;
    }
    if (a?.model_type === "xgb") {
      try {
        const xgbFile = typeof a.xgb_file === "string" ? a.xgb_file : "";
        if (!xgbFile) return null;
        const xp = `${process.cwd()}/ml/permatch/${xgbFile}`;
        if (!existsSync(xp)) return null;
        const slim = slimBooster(JSON.parse(readFileSync(xp, "utf-8")));
        if (!slim) return null;
        const feats = Array.isArray(a.features) ? (a.features as string[]) : undefined;
        return {
          model_type: "xgb",
          features: feats,
          train_seasons: Array.isArray(a.train_seasons) ? (a.train_seasons as string[]) : [],
          T: 1,
          mu: [],
          sd: [],
          xgb: slim,
        } as PermatchArtifact;
      } catch {
        return null;
      }
    }
    if (a?.model_type === "lstm") {
      return a as PermatchArtifact;
    }
    if (!Array.isArray(a.mu) || !Array.isArray(a.sd)) return null;
    const n = (a.weights as unknown[]).length;
    if (n === 0 || a.mu.length !== n || a.sd.length !== n) return null;
    if (Array.isArray(a.features) && a.features.length !== n) return null;
    return a as PermatchArtifact;
  } catch {
    return null;
  }
}

function readEnsemble(league: string, a: Record<string, unknown>): PermatchArtifact | null {
  try {
    const baseVer = typeof a.base_ver === "string" ? a.base_ver : "";
    const xgbFile = typeof a.xgb_file === "string" ? a.xgb_file : "";
    if (!baseVer || !xgbFile) return null;
    const bp = `${process.cwd()}/ml/permatch/${league}_${baseVer}.json`;
    const xp = `${process.cwd()}/ml/permatch/${xgbFile}`;
    if (!existsSync(bp) || !existsSync(xp)) return null;
    const base = JSON.parse(readFileSync(bp, "utf-8")) as PermatchArtifact;
    if (!Array.isArray(base.weights)) return null;
    const slim = slimBooster(JSON.parse(readFileSync(xp, "utf-8")));
    if (!slim) return null;
    const w = typeof a.blend_w === "number" && Number.isFinite(a.blend_w) ? a.blend_w : 1;
    const feats = Array.isArray(a.features) ? (a.features as string[]) : base.features;
    return {
      model_type: "ensemble",
      base_ver: baseVer,
      blend_w: Math.max(0, Math.min(1, w)),
      features: feats,
      train_seasons: [],
      T: base.T,
      mu: base.mu,
      sd: base.sd,
      base,
      xgb: slim,
    };
  } catch {
    return null;
  }
}

function std(x: number[], mu: number[], sd: number[]): number[] {
  return x.map((v, i) => (v - mu[i]) / sd[i]);
}

export function predictForRow(m: SoccerMatch, art: PermatchArtifact, ver: string): PredEntry | null {
  try {
    if (art.model_type === "lstm") return null;
    if (art.model_type === "xgb" && art.xgb) {
      const p = xgbProba(art.xgb, selectFeatures(rowFeatures(m), art.features));
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > alertThreshold(art) };
    }
    if (art.model_type === "logreg" && art.scaler_mean && art.scaler_scale && art.coef && art.intercept) {
      const x = selectFeatures(rowFeatures(m), art.features);
      const p = logregProba(x, art.scaler_mean, art.scaler_scale, art.coef, art.intercept);
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > alertThreshold(art) };
    }
    if ((art.model_type === "poisson" || art.model_type === "dixon") && art.teams && art.att && art.def) {
      const idx = new Map(art.teams.map((t, i) => [t, i]));
      const hi = idx.get(String(m.home_team));
      const ai = idx.get(String(m.away_team));
      const lam = Math.exp((hi != null ? art.att[hi] : 0) + (ai != null ? art.def[ai] : 0) + (art.home ?? 0));
      const mu = Math.exp((ai != null ? art.att[ai] : 0) + (hi != null ? art.def[hi] : 0));
      const rho = art.model_type === "dixon" ? (art.rho ?? 0) : 0;
      const p = dcProba(lam, mu, rho);
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > alertThreshold(art) };
    }
    let lin: number[];
    let x: number[];
    let pats = art.patterns;
    let tau = art.pattern_tau ?? 0;
    let taus = art.pattern_taus ?? null;
    let palpha = art.pattern_alpha ?? 0.5;
    let pw: number[] | null = art.weights ?? null;
    let boost = art.draw_boost ?? 0;
    let boostMargin = art.draw_close_margin ?? 0.12;
    let boostTotal = art.draw_total_max ?? 2.55;
    let T = art.T;
    if (art.model_type === "ensemble" && art.base && art.xgb) {
      const b = legacyParts(m, art.base);
      if (!b) return null;
      const qx = xgbProba(art.xgb, rowFeatures(m));
      const alpha = art.blend_w ?? 1;
      const dd = alpha * b.dd + (1 - alpha) * qx[1];
      lin = [b.ph * (1 - dd), dd, (1 - b.ph) * (1 - dd)];
      x = b.x;
      pats = art.base.patterns;
      tau = art.base.pattern_tau ?? 0;
      taus = art.base.pattern_taus ?? null;
      palpha = art.base.pattern_alpha ?? 0.5;
      pw = art.base.weights ?? null;
      boost = art.base.draw_boost ?? 0;
      boostMargin = art.base.draw_close_margin ?? 0.12;
      boostTotal = art.base.draw_total_max ?? 2.55;
      T = art.base.T;
    } else {
      const b = legacyParts(m, art);
      if (!b) return null;
      lin = [b.ph * (1 - b.dd), b.dd, (1 - b.ph) * (1 - b.dd)];
      x = b.x;
    }
    const total = drawFeatures(m)[2];
    const p = applyDrawBoost(applyTemp(blendProbs(lin, x, pats, tau, palpha, pw, taus), T), total, boost, boostMargin, boostTotal);
    if (p.some((v) => !Number.isFinite(v))) return null;
    return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > alertThreshold(art) };
  } catch {
    return null;
  }
}

export async function buildPredMap(
  rows: SoccerMatch[],
  league: string,
  ver: string
): Promise<Record<string, PredEntry>> {
  const map: Record<string, PredEntry> = {};
  const art = await loadArtifactAsync(league, ver);
  if (!art) return map;
  for (const m of rows) {
    const p = predictForRow(m, art, ver);
    if (p) map[m.source_match_id || String(m.id)] = p;
  }
  return map;
}

function parseScore(v: string | null | undefined): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

export async function computeAccuracy(
  rows: SoccerMatch[],
  league: string,
  ver: string
): Promise<Record<string, { n: number; hit: number; acc: number | null; pending: number }>> {
  const map = await buildPredMap(rows, league, ver);
  const stats: Record<string, { n: number; hit: number; pending: number }> = {};
  for (const m of rows) {
    const p = map[m.source_match_id || String(m.id)];
    if (!p) continue;
    const s = String(m.season);
    stats[s] = stats[s] || { n: 0, hit: 0, pending: 0 };
    const h = parseScore(m.home_score);
    const a = parseScore(m.away_score);
    if (h == null || a == null) {
      stats[s].pending += 1;
      continue;
    }
    const actual = h > a ? 0 : h === a ? 1 : 2;
    const probs = [p.home, p.draw, p.away];
    const pick = probs.indexOf(Math.max(...probs));
    stats[s].n += 1;
    if (pick === actual) stats[s].hit += 1;
  }
  const out: Record<string, { n: number; hit: number; acc: number | null; pending: number }> = {};
  for (const [s, v] of Object.entries(stats)) out[s] = { ...v, acc: v.n > 0 ? v.hit / v.n : null };
  return out;
}
