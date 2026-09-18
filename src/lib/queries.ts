import { createServerSupabaseClient } from "@/lib/supabase/server";

export type Wdl = { win?: number | null; draw?: number | null; loss?: number | null } | null;

export type OddsPick = { live?: number | null; current?: number | null; initial?: number | null } | null;

export type CompanyOdds = {
  win_draw_lose?: { home?: OddsPick; draw?: OddsPick; away?: OddsPick } | null;
} | null;

export type MatchOdds = {
  crown?: CompanyOdds;
  bet365?: CompanyOdds;
  one_x_bet?: CompanyOdds;
} | null;

export type OverUnder = { over?: number | null; draw?: number | null; under?: number | null } | null;

export type SameOddsWdl = { home?: number | null; draw?: number | null; away?: number | null } | null;

export type SameOdds = {
  win_draw_lose?: SameOddsWdl;
  same_odds?: { win_draw_lose?: SameOddsWdl } | null;
} | null;

export type H2hRange = {
  win?: number | null;
  draw?: number | null;
  loss?: number | null;
  handicap?: Wdl;
  over_under?: OverUnder;
} | null;

export type TeamStrength = {
  power?: number | null;
  attack?: number | null;
  defense?: number | null;
  condition?: number | null;
  head_to_head?: number | null;
  value?: number | null;
};

export type SoccerMatch = {
  id: number;
  league_code: string;
  season: string;
  round: number | null;
  match_date: string | null;
  match_time: string | null;
  home_team: string;
  home_rank: string | null;
  home_score: string | null;
  away_team: string;
  away_rank: string | null;
  away_score: string | null;
  crawled_at: string | null;
  source_match_id: string | null;
  lineup?: { home?: string | null; away?: string | null } | null;
  strength?: { home?: TeamStrength | null; away?: TeamStrength | null } | null;
  head_to_head?: { recent_5?: H2hRange; recent_10?: H2hRange; recent_20?: H2hRange } | null;
  odds?: MatchOdds;
  same_odds?: SameOdds;
  pred?: { home: number; draw: number; away: number; ver: string } | null;
};

export async function getPredictionsMap(
  league: string,
  seasons: string[]
): Promise<Record<string, { home: number; draw: number; away: number; ver: string }>> {
  try {
    const supabase = createServerSupabaseClient();
    const { data, error } = await supabase
      .from("predictions")
      .select("source_match_id,prob_home,prob_draw,prob_away,model_ver,created_at")
      .eq("league_code", league)
      .in("season", seasons)
      .order("created_at", { ascending: false })
      .limit(20000);
    if (error || !data) return {};
    const map: Record<string, { home: number; draw: number; away: number; ver: string }> = {};
    for (const r of data) {
      if (!(r.source_match_id in map)) {
        map[r.source_match_id] = { home: r.prob_home, draw: r.prob_draw, away: r.prob_away, ver: r.model_ver };
      }
    }
    return map;
  } catch {
    return {};
  }
}

export async function getSeasonAccuracy(
  league: string
): Promise<Record<string, { n: number; hit: number; acc: number | null }>> {
  try {
    const supabase = createServerSupabaseClient();
    const preds = await getPredictionsMapAll(league);
    const stats: Record<string, { n: number; hit: number }> = {};
    const pageSize = 1000;
    let offset = 0;
    for (;;) {
      const { data, error } = await supabase
        .from("soccer_matches")
        .select("season,home_score,away_score,source_match_id")
        .eq("league_code", league)
        .range(offset, offset + pageSize - 1);
      if (error || !data) break;
      for (const r of data) {
        const p = preds[r.source_match_id || ""];
        const h = parseNum(r.home_score);
        const a = parseNum(r.away_score);
        if (!p || h == null || a == null) continue;
        const actual = h > a ? 0 : h === a ? 1 : 2;
        const pick = p.home >= p.draw && p.home >= p.away ? 0 : p.draw >= p.away ? 1 : 2;
        const s = String(r.season);
        stats[s] = stats[s] || { n: 0, hit: 0 };
        stats[s].n += 1;
        if (pick === actual) stats[s].hit += 1;
      }
      if (data.length < pageSize) break;
      offset += pageSize;
      if (offset >= 40000) break;
    }
    const out: Record<string, { n: number; hit: number; acc: number | null }> = {};
    for (const [s, v] of Object.entries(stats)) {
      out[s] = { ...v, acc: v.n > 0 ? v.hit / v.n : null };
    }
    return out;
  } catch {
    return {};
  }
}

function parseNum(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

async function getPredictionsMapAll(
  league: string
): Promise<Record<string, { home: number; draw: number; away: number }>> {
  const supabase = createServerSupabaseClient();
  const map: Record<string, { home: number; draw: number; away: number }> = {};
  const pageSize = 1000;
  let offset = 0;
  for (;;) {
    const { data, error } = await supabase
      .from("predictions")
      .select("source_match_id,prob_home,prob_draw,prob_away,created_at")
      .eq("league_code", league)
      .order("created_at", { ascending: false })
      .range(offset, offset + pageSize - 1);
    if (error || !data) break;
    for (const r of data) {
      if (!(r.source_match_id in map)) {
        map[r.source_match_id] = { home: r.prob_home, draw: r.prob_draw, away: r.prob_away };
      }
    }
    if (data.length < pageSize) break;
    offset += pageSize;
    if (offset >= 40000) break;
  }
  return map;
}

export async function getConnectionStatus(): Promise<{ ok: boolean; message: string }> {
  try {
    const supabase = createServerSupabaseClient();
    const { error, count } = await supabase.from("soccer_matches").select("*", { count: "exact", head: true });
    if (error) return { ok: false, message: error.message };
    return { ok: true, message: `connected (soccer_matches, ${count ?? "?"} rows)` };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : "Unknown error" };
  }
}

const MATCH_COLS =
  "id,league_code,season,round,match_date,match_time,home_team,home_rank,home_score,away_team,away_rank,away_score,crawled_at,source_match_id,lineup,strength,head_to_head,odds,same_odds";

export type League = {
  code: string;
  name_ko: string;
  name_en: string;
};

export async function getLeagues(): Promise<{ data: League[]; error: string | null }> {
  try {
    const supabase = createServerSupabaseClient();
    const { data, error } = await supabase
      .from("leagues")
      .select("code,name_ko,name_en")
      .eq("is_active", true)
      .order("display_order", { ascending: true });
    if (error) return { data: [], error: error.message };
    return { data: (data ?? []) as League[], error: null };
  } catch (e) {
    return { data: [], error: e instanceof Error ? e.message : "Unknown error" };
  }
}

export async function getSeasons(league: string): Promise<{ data: string[]; error: string | null }> {
  try {
    const supabase = createServerSupabaseClient();
    const uniq = new Set<string>();
    const pageSize = 1000;
    let offset = 0;
    for (;;) {
      const { data, error } = await supabase
        .from("soccer_matches")
        .select("season")
        .eq("league_code", league)
        .range(offset, offset + pageSize - 1);
      if (error) return { data: [], error: error.message };
      for (const r of data ?? []) {
        if (r.season) uniq.add(r.season);
      }
      if (!data || data.length < pageSize) break;
      offset += pageSize;
      if (offset > 50000) break;
    }
    return { data: [...uniq].sort().reverse(), error: null };
  } catch (e) {
    return { data: [], error: e instanceof Error ? e.message : "Unknown error" };
  }
}

export async function getMatchesBySeasons(
  league: string,
  seasons: string[]
): Promise<{ data: SoccerMatch[]; error: string | null }> {
  if (!league || seasons.length === 0) return { data: [], error: null };
  try {
    const supabase = createServerSupabaseClient();
    const all: SoccerMatch[] = [];
    const pageSize = 1000;
    let offset = 0;
    for (;;) {
      const { data, error } = await supabase
        .from("soccer_matches")
        .select(MATCH_COLS)
        .eq("league_code", league)
        .in("season", seasons)
        .order("season", { ascending: false })
        .order("round", { ascending: false, nullsFirst: false })
        .order("match_date", { ascending: false })
        .order("match_time", { ascending: false })
        .range(offset, offset + pageSize - 1);
      if (error) return { data: [], error: error.message };
      all.push(...((data ?? []) as SoccerMatch[]));
      if (!data || data.length < pageSize) break;
      offset += pageSize;
      if (offset >= 20000) break;
    }
    const preds = await getPredictionsMap(league, seasons);
    for (const m of all) {
      const key = m.source_match_id || String(m.id);
      m.pred = preds[key] ?? null;
    }
    return { data: all, error: null };
  } catch (e) {
    return { data: [], error: e instanceof Error ? e.message : "Unknown error" };
  }
}
