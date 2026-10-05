import { createServerSupabaseClient } from "@/lib/supabase/server";

export type Wdl = { win?: number | null; draw?: number | null; loss?: number | null } | null;

export type OddsPick = { live?: number | null; current?: number | null; initial?: number | null } | null;

export type CompanyOdds = {
  win_draw_lose?: { home?: OddsPick; draw?: OddsPick; away?: OddsPick } | null;
  handicap?: { home?: OddsPick; away?: OddsPick; line?: OddsPick } | null;
  over_under?: { line?: OddsPick; over?: OddsPick; under?: OddsPick } | null;
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

export type TeamAvgSide = {
  fouls?: number | null;
  goals?: number | null;
  corners?: number | null;
  conceded?: number | null;
  possession?: number | null;
  yellow_cards?: number | null;
  shots_on_target?: number | null;
};

export type TeamGoalSlot = {
  slot?: string | null;
  home_scored?: number | null;
  away_scored?: number | null;
  home_conceded?: number | null;
  away_conceded?: number | null;
};

export type TeamHalfFull = {
  stats?: Record<string, { home?: number | null; away?: number | null }> | null;
  home_matches?: number | null;
  away_matches?: number | null;
};

export type TeamInfo = {
  avg_stats?: { home?: TeamAvgSide | null; away?: TeamAvgSide | null } | null;
  goal_time?: TeamGoalSlot[] | null;
  half_full?: TeamHalfFull | null;
} | null;

export type TeamFormEntry = {
  win?: number | null;
  draw?: number | null;
  loss?: number | null;
  scored?: number | null;
  conceded?: number | null;
  avg_scored?: number | null;
  avg_conceded?: number | null;
  handicap?: Wdl | null;
  over_under?: OverUnder | null;
};

export type RecentFormSide = {
  available?: boolean | null;
  recent_1?: TeamFormEntry | null; recent_2?: TeamFormEntry | null;
  recent_3?: TeamFormEntry | null; recent_4?: TeamFormEntry | null;
  recent_5?: TeamFormEntry | null; recent_6?: TeamFormEntry | null;
  recent_7?: TeamFormEntry | null; recent_8?: TeamFormEntry | null;
  recent_9?: TeamFormEntry | null; recent_10?: TeamFormEntry | null;
  recent_11?: TeamFormEntry | null; recent_12?: TeamFormEntry | null;
  recent_13?: TeamFormEntry | null; recent_14?: TeamFormEntry | null;
  recent_15?: TeamFormEntry | null; recent_16?: TeamFormEntry | null;
  recent_17?: TeamFormEntry | null; recent_18?: TeamFormEntry | null;
  recent_19?: TeamFormEntry | null; recent_20?: TeamFormEntry | null;
};

export type RecentForm = {
  home?: RecentFormSide | null;
  away?: RecentFormSide | null;
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
  head_to_head?: {
    recent_1?: H2hRange; recent_2?: H2hRange; recent_3?: H2hRange; recent_4?: H2hRange;
    recent_5?: H2hRange; recent_6?: H2hRange; recent_7?: H2hRange; recent_8?: H2hRange;
    recent_9?: H2hRange; recent_10?: H2hRange; recent_11?: H2hRange; recent_12?: H2hRange;
    recent_13?: H2hRange; recent_14?: H2hRange; recent_15?: H2hRange; recent_16?: H2hRange;
    recent_17?: H2hRange; recent_18?: H2hRange; recent_19?: H2hRange; recent_20?: H2hRange;
  } | null;
  team_info?: TeamInfo;
  recent_form?: RecentForm;
  goal_time?: TeamGoalSlot[] | null;
  ht_ft?: TeamHalfFull | null;
  odds?: MatchOdds;
  same_odds?: SameOdds;
  pred?: { home: number; draw: number; away: number; ver: string; drawAlert?: boolean } | null;
};

export async function getPredictionsMap(
  league: string,
  seasons: string[],
  modelVer?: string
): Promise<Record<string, { home: number; draw: number; away: number; ver: string }>> {
  if (!modelVer) return {};
  try {
    const { buildPredMap } = await import("@/lib/predict");
    const { data } = await getMatchesBySeasons(league, seasons);
    return buildPredMap(data, league, modelVer);
  } catch {
    return {};
  }
}

export async function getSeasonAccuracy(
  league: string,
  modelVer?: string,
  seasons?: string[]
): Promise<Record<string, { n: number; hit: number; acc: number | null }>> {
  if (!modelVer) return {};
  try {
    const { computeAccuracy } = await import("@/lib/predict");
    let all: SoccerMatch[];
    if (seasons && seasons.length > 0) {
      all = (await fetchMatchesRaw(league, seasons)).map((m) => ({ ...m }));
    } else {
      const supabase = createServerSupabaseClient();
      all = [];
      const pageSize = 1000;
      let offset = 0;
      for (;;) {
        const { data, error } = await supabase
          .from("soccer_matches")
          .select(MATCH_COLS)
          .eq("league_code", league)
          .order("id")
          .range(offset, offset + pageSize - 1);
        if (error || !data) break;
        all.push(...((data ?? []) as SoccerMatch[]));
        if (data.length < pageSize) break;
        offset += pageSize;
        if (offset >= 40000) break;
      }
    }
    return computeAccuracy(all, league, modelVer);
  } catch {
    return {};
  }
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
  "id,league_code,season,round,match_date,match_time,home_team,home_rank,home_score,away_team,away_rank,away_score,crawled_at,source_match_id,lineup,strength,head_to_head,odds,same_odds,team_info,recent_form,goal_time,ht_ft";

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

const MATCH_CACHE_TTL = 5 * 60 * 1000;
const MATCH_CACHE_MAX = 10;
const matchCache = new Map<string, { ts: number; rows: SoccerMatch[] }>();
const matchInflight = new Map<string, Promise<SoccerMatch[]>>();

function matchCacheKey(league: string, seasons: string[]) {
  return `${league}|${[...seasons].sort().join(",")}`;
}

function rememberMatches(key: string, rows: SoccerMatch[]) {
  matchCache.set(key, { ts: Date.now(), rows });
  while (matchCache.size > MATCH_CACHE_MAX) {
    const oldest = matchCache.keys().next().value;
    if (oldest === undefined) break;
    matchCache.delete(oldest);
  }
}

async function fetchMatchesRaw(league: string, seasons: string[]): Promise<SoccerMatch[]> {
  const key = matchCacheKey(league, seasons);
  const hit = matchCache.get(key);
  if (hit && Date.now() - hit.ts < MATCH_CACHE_TTL) {
    matchCache.delete(key);
    matchCache.set(key, hit);
    return hit.rows;
  }
  const ongoing = matchInflight.get(key);
  if (ongoing) return ongoing;
  const p = (async () => {
    const supabase = createServerSupabaseClient();
    const pageSize = 1000;
    const fetchOne = async (s: string) => {
      const out: SoccerMatch[] = [];
      let offset = 0;
      for (;;) {
        const { data, error } = await supabase
          .from("soccer_matches")
          .select(MATCH_COLS)
          .eq("league_code", league)
          .eq("season", s)
          .order("season", { ascending: false })
          .order("round", { ascending: false, nullsFirst: false })
          .order("match_date", { ascending: false })
          .order("match_time", { ascending: false })
          .range(offset, offset + pageSize - 1);
        if (error) throw new Error(error.message);
        out.push(...((data ?? []) as SoccerMatch[]));
        if (!data || data.length < pageSize) break;
        offset += pageSize;
        if (offset >= 20000) break;
      }
      return out;
    };
    const perSeason: SoccerMatch[][] = [];
    for (let i = 0; i < seasons.length; i += 3) {
      const chunk = await Promise.all(seasons.slice(i, i + 3).map(fetchOne));
      perSeason.push(...chunk);
    }
    return perSeason.flat();
  })();
  matchInflight.set(key, p);
  try {
    const rows = await p;
    rememberMatches(key, rows);
    return rows;
  } finally {
    matchInflight.delete(key);
  }
}

export async function getMatchesBySeasons(
  league: string,
  seasons: string[],
  modelVer?: string
): Promise<{ data: SoccerMatch[]; error: string | null }> {
  if (!league || seasons.length === 0) return { data: [], error: null };
  try {
    const all = (await fetchMatchesRaw(league, seasons)).map((m) => ({ ...m }));
    const { buildPredMap } = await import("@/lib/predict");
    const preds = modelVer ? buildPredMap(all, league, modelVer) : {};
    for (const m of all) {
      const key = m.source_match_id || String(m.id);
      m.pred = preds[key] ?? null;
    }
    return { data: all, error: null };
  } catch (e) {
    return { data: [], error: e instanceof Error ? e.message : "Unknown error" };
  }
}
