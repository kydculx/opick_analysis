import { existsSync, readFileSync } from "node:fs";
import type { SoccerMatch } from "@/lib/queries";
import { applyTemp, blendProbs, cappedDot, dcProba, drawFeatures, legacyParts, logregProba, rowFeatures, selectFeatures, sigmoid } from "./permatch-math";
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
};

export type PredEntry = { home: number; draw: number; away: number; ver: string };

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
      return { home: p[0], draw: p[1], away: p[2], ver };
    }
    if (art.model_type === "logreg" && art.scaler_mean && art.scaler_scale && art.coef && art.intercept) {
      const x = selectFeatures(rowFeatures(m), art.features);
      const p = logregProba(x, art.scaler_mean, art.scaler_scale, art.coef, art.intercept);
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver };
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
      return { home: p[0], draw: p[1], away: p[2], ver };
    }
    let lin: number[];
    let x: number[];
    let pats = art.patterns;
    let tau = art.pattern_tau ?? 0;
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
      T = art.base.T;
    } else {
      const b = legacyParts(m, art);
      if (!b) return null;
      lin = [b.ph * (1 - b.dd), b.dd, (1 - b.ph) * (1 - b.dd)];
      x = b.x;
    }
    const p = applyTemp(blendProbs(lin, x, pats, tau), T);
    if (p.some((v) => !Number.isFinite(v))) return null;
    return { home: p[0], draw: p[1], away: p[2], ver };
  } catch {
    return null;
  }
}

export function buildPredMap(
  rows: SoccerMatch[],
  league: string,
  ver: string
): Record<string, PredEntry> {
  const map: Record<string, PredEntry> = {};
  const art = loadArtifact(league, ver);
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

export function computeAccuracy(
  rows: SoccerMatch[],
  league: string,
  ver: string
): Record<string, { n: number; hit: number; acc: number | null }> {
  const map = buildPredMap(rows, league, ver);
  const stats: Record<string, { n: number; hit: number }> = {};
  for (const m of rows) {
    const p = map[m.source_match_id || String(m.id)];
    if (!p) continue;
    const h = parseScore(m.home_score);
    const a = parseScore(m.away_score);
    if (h == null || a == null) continue;
    const actual = h > a ? 0 : h === a ? 1 : 2;
    const probs = [p.home, p.draw, p.away];
    const pick = probs.indexOf(Math.max(...probs));
    const s = String(m.season);
    stats[s] = stats[s] || { n: 0, hit: 0 };
    stats[s].n += 1;
    if (pick === actual) stats[s].hit += 1;
  }
  const out: Record<string, { n: number; hit: number; acc: number | null }> = {};
  for (const [s, v] of Object.entries(stats)) out[s] = { ...v, acc: v.n > 0 ? v.hit / v.n : null };
  return out;
}
