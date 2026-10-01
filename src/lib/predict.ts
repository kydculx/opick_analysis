import { existsSync, readFileSync } from "node:fs";
import type { SoccerMatch } from "@/lib/queries";
import { applyTemp, blendProbs, cappedDot, drawFeatures, rowFeatures, selectFeatures, sigmoid } from "./permatch-math";

export type PermatchArtifact = {
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
};

export type PredEntry = { home: number; draw: number; away: number; ver: string };

export function loadArtifact(league: string, ver: string): PermatchArtifact | null {
  try {
    const p = `${process.cwd()}/ml/permatch/${league}_${ver}.json`;
    if (!existsSync(p)) return null;
    const a = JSON.parse(readFileSync(p, "utf-8"));
    if (!Array.isArray(a.weights) || (a.weights as unknown[]).length === 0) return null;
    if (!Array.isArray(a.mu) || !Array.isArray(a.sd)) return null;
    const n = (a.weights as unknown[]).length;
    if (n === 0 || a.mu.length !== n || a.sd.length !== n) return null;
    if (Array.isArray(a.features) && a.features.length !== n) return null;
    return a as PermatchArtifact;
  } catch {
    return null;
  }
}

function std(x: number[], mu: number[], sd: number[]): number[] {
  return x.map((v, i) => (v - mu[i]) / sd[i]);
}

export function predictForRow(m: SoccerMatch, art: PermatchArtifact, ver: string): PredEntry | null {
  try {
    const e = art.emphasis ?? new Array(art.features?.length ?? art.mu.length).fill(1);
    const x = std(selectFeatures(rowFeatures(m), art.features), art.mu, art.sd).map((v, i) => v * e[i]);
    let lin: number[];
    if (art.weights && art.hfa != null && art.draw_prior != null) {
      const s = cappedDot(x, art.weights, art.hfa, art.contrib_cap ?? null);
      const ph = sigmoid(s);
      let dd: number;
      if (art.draw_weights && art.draw_mu && art.draw_sd) {
        const xd = std(drawFeatures(m), art.draw_mu, art.draw_sd);
        const ds = xd.reduce((a, v, i) => a + v * (art.draw_weights as number[])[i], 0) + (art.draw_bias ?? 0);
        dd = sigmoid(ds);
      } else {
        dd = art.draw_prior;
      }
      lin = [ph * (1 - dd), dd, (1 - ph) * (1 - dd)];
    } else {
      return null;
    }
    const p = applyTemp(blendProbs(lin, x, art.patterns, art.pattern_tau ?? 0), art.T);
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
