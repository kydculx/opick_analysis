import type { SoccerMatch } from "@/lib/queries";
import { applyDrawBoost, applyTemp, blendProbs, dcProba, drawFeatures, legacyParts, logregProba, rowFeatures, selectFeatures } from "./permatch-math";
import { slimBooster, xgbProba, type XgbSlim } from "./xgb";

export type WalkSnapshot = {
  weights?: number[];
  hfa?: number;
  T?: number;
  draw_prior?: number;
  mu?: number[];
  sd?: number[];
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
  contrib_cap?: number | null;
  train_end?: string;
};

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
  contrib_cap?: number | null;
  walk?: { enabled?: boolean; base_end?: string; snapshots?: Record<string, WalkSnapshot> | null } | null;
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
  seq_len?: number;
};

export type PredEntry = { home: number; draw: number; away: number; ver: string; drawAlert: boolean };

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
    if (o?.model_type === "xgb") {
      const slim = slimBooster(o.xgb);
      if (!slim) return null;
      return {
        model_type: "xgb",
        xgb: slim,
        features: Array.isArray(o.features) ? (o.features as string[]) : undefined,
        train_seasons: Array.isArray(o.train_seasons) ? (o.train_seasons as string[]) : [],
        T: 1,
        mu: [],
        sd: [],
      } as unknown as PermatchArtifact;
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
    if (error || !data || data.length === 0) return null;
    const raw = (data[0] as { artifact?: unknown }).artifact;
    if (!raw || typeof raw !== "object") return null;
    return coerceArtifact(raw);
  } catch {
    return null;
  }
}

export function predictForRow(m: SoccerMatch, art: PermatchArtifact, ver: string): PredEntry | null {
  try {
    if (art.model_type === "lstm") return null;
    if (art.model_type === "xgb" && art.xgb) {
      const p = xgbProba(art.xgb, selectFeatures(rowFeatures(m), art.features));
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > 0.3 };
    }
    if (art.model_type === "logreg" && art.scaler_mean && art.scaler_scale && art.coef && art.intercept) {
      const x = selectFeatures(rowFeatures(m), art.features);
      const p = logregProba(x, art.scaler_mean, art.scaler_scale, art.coef, art.intercept);
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > 0.3 };
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
      return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > 0.3 };
    }
    const b = legacyParts(m, art);
    if (!b) return null;
    const lin = [b.ph * (1 - b.dd), b.dd, (1 - b.ph) * (1 - b.dd)];
    const total = drawFeatures(m)[2];
    const p = applyDrawBoost(
      applyTemp(blendProbs(lin, b.x, art.patterns, art.pattern_tau ?? 0, art.pattern_alpha ?? 0.5, art.weights ?? null, art.pattern_taus ?? null), art.T),
      total,
      art.draw_boost ?? 0,
      art.draw_close_margin ?? 0.12,
      art.draw_total_max ?? 2.55,
    );
    if (p.some((v) => !Number.isFinite(v))) return null;
    return { home: p[0], draw: p[1], away: p[2], ver, drawAlert: p[1] > 0.3 };
  } catch {
    return null;
  }
}

export function artifactForSeason(art: PermatchArtifact, season: unknown): PermatchArtifact {
  const snaps = art.walk?.snapshots;
  if (!snaps) return art;
  const snap = snaps[String(season)];
  if (!snap) return art;
  return { ...art, ...snap } as PermatchArtifact;
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
    const p = predictForRow(m, artifactForSeason(art, m.season), ver);
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
