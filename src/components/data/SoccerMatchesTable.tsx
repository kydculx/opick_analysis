import { Fragment, type ReactNode } from "react";
import type { CompanyOdds, H2hRange, MatchOdds, SameOdds, SoccerMatch, TeamStrength } from "@/lib/queries";

export type ColumnId =
  | "league"
  | "season"
  | "datetime"
  | "home"
  | "score"
  | "away"
  | "result"
  | "pred"
  | "predRatio"
  | "power"
  | "h2h"
  | "condition"
  | "attack"
  | "defense"
  | "value"
  | "recent5"
  | "recent10"
  | "recent20"
  | "crown"
  | "bet365"
  | "onexbet"
  | "sameOdds"
  | "formation";

export const CORE_COLUMNS: ColumnId[] = ["league", "season", "datetime", "home", "score", "away"];

export const FIELD_GROUPS: { id: string; label: string; cols: ColumnId[] }[] = [
  { id: "power", label: "전력대비", cols: ["power", "h2h", "condition", "attack", "defense", "value"] },
  { id: "recent", label: "최근경기", cols: ["recent5", "recent10", "recent20"] },
  { id: "odds", label: "배당률", cols: ["crown", "bet365", "onexbet"] },
];

export const SINGLE_COLUMNS: ColumnId[] = ["pred", "predRatio", "result", "sameOdds", "formation"];

export const COLUMNS: { id: ColumnId; label: string }[] = [
  { id: "league", label: "리그" },
  { id: "season", label: "시즌/R" },
  { id: "datetime", label: "일시" },
  { id: "home", label: "홈" },
  { id: "score", label: "스코어" },
  { id: "away", label: "원정" },
  { id: "result", label: "결과" },
  { id: "pred", label: "예측" },
  { id: "predRatio", label: "예측비율" },
  { id: "power", label: "전력" },
  { id: "h2h", label: "H2H" },
  { id: "condition", label: "컨디션" },
  { id: "attack", label: "공격" },
  { id: "defense", label: "수비" },
  { id: "value", label: "가치" },
  { id: "recent5", label: "최근5" },
  { id: "recent10", label: "최근10" },
  { id: "recent20", label: "최근20" },
  { id: "crown", label: "Crown" },
  { id: "bet365", label: "Bet365" },
  { id: "onexbet", label: "1XBET" },
  { id: "sameOdds", label: "동일배당률" },
  { id: "formation", label: "포메이션" },
];

const CELL = "whitespace-nowrap px-4 py-3 text-center";
const MONO_CELL = "whitespace-nowrap px-4 py-3 font-mono text-xs text-white text-center";

function renderCell(id: ColumnId, m: SoccerMatch, leagueNames?: Record<string, string>): ReactNode {
  switch (id) {
    case "league":
      return <td className={CELL}>{leagueNames?.[m.league_code] ?? m.league_code}</td>;
    case "season":
      return (
        <td className={`${CELL} text-white`}>
          {m.season} {m.round ? `/ ${m.round}R` : ""}
        </td>
      );
    case "datetime":
      return (
        <td className={CELL}>
          {m.match_date ?? "-"} {m.match_time ?? ""}
        </td>
      );
    case "home":
      return (
        <td className={`${CELL} font-medium`}>
          {m.home_team}
          {m.home_rank ? <span className="ml-1 text-xs text-white">({m.home_rank})</span> : null}
        </td>
      );
    case "score":
      return (
        <td className="whitespace-nowrap px-4 py-3 text-center font-mono font-bold">
          {formatScore(m.home_score)} : {formatScore(m.away_score)}
        </td>
      );
    case "away":
      return (
        <td className={`${CELL} font-medium`}>
          {m.away_team}
          {m.away_rank ? <span className="ml-1 text-xs text-white">({m.away_rank})</span> : null}
        </td>
      );
    case "pred":
      return (
        <td className={MONO_CELL}>
          <PredCell m={m} />
        </td>
      );
    case "predRatio":
      return <td className={MONO_CELL}>{formatPredRatio(m)}</td>;
    case "result":
      return <td className={MONO_CELL}>{formatResult(m)}</td>;
    case "power":
      return <td className={MONO_CELL}>{formatStrength(m, "power")}</td>;
    case "h2h":
      return <td className={MONO_CELL}>{formatStrength(m, "head_to_head")}</td>;
    case "condition":
      return <td className={MONO_CELL}>{formatStrength(m, "condition")}</td>;
    case "attack":
      return <td className={MONO_CELL}>{formatStrength(m, "attack")}</td>;
    case "defense":
      return <td className={MONO_CELL}>{formatStrength(m, "defense")}</td>;
    case "value":
      return <td className={MONO_CELL}>{formatStrength(m, "value")}</td>;
    case "recent5":
      return (
        <td className={MONO_CELL}>
          <WdlCell r={m.head_to_head?.recent_5} />
        </td>
      );
    case "recent10":
      return (
        <td className={MONO_CELL}>
          <WdlCell r={m.head_to_head?.recent_10} />
        </td>
      );
    case "recent20":
      return (
        <td className={MONO_CELL}>
          <WdlCell r={m.head_to_head?.recent_20} />
        </td>
      );
    case "crown":
      return <td className={MONO_CELL}>{formatOdds(m.odds, "crown")}</td>;
    case "bet365":
      return <td className={MONO_CELL}>{formatOdds(m.odds, "bet365")}</td>;
    case "onexbet":
      return <td className={MONO_CELL}>{formatOdds(m.odds, "one_x_bet")}</td>;
    case "sameOdds":
      return <td className={MONO_CELL}>{formatSameOdds(m.same_odds)}</td>;
    case "formation":
      return <td className={MONO_CELL}>{formatFormation(m)}</td>;
  }
}

export function SoccerMatchesTable({
  rows,
  leagueNames,
  hidden = [],
}: {
  rows: SoccerMatch[];
  leagueNames?: Record<string, string>;
  hidden?: ColumnId[];
}) {
  if (rows.length === 0) return null;
  const cols = COLUMNS.filter((c) => CORE_COLUMNS.includes(c.id) || !hidden.includes(c.id));
  if (cols.length === 0) return null;
  const groupOf: number[] = [];
  {
    let g = 0;
    let prev: string | null = null;
    for (const m of rows) {
      const key = `${m.season}|${m.round ?? "-"}`;
      if (prev !== null && key !== prev) g += 1;
      prev = key;
      groupOf.push(g);
    }
  }
  return (
    <div className="min-h-0 flex-1 overflow-auto rounded-2xl border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-left text-sm" style={{ minWidth: Math.max(cols.length * 120, 400) }}>
        <thead className="sticky top-0 z-10 bg-zinc-50 text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
          <tr className="divide-x divide-zinc-200 dark:divide-zinc-800">
            {cols.map((c) => (
              <th key={c.id} className="whitespace-nowrap px-4 py-3 text-center font-medium">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((m, i) => (
            <tr
              key={m.id}
              className={`divide-x divide-zinc-200 border-t border-zinc-100 dark:divide-zinc-800 dark:border-zinc-800 ${
                groupOf[i] % 2 === 1 ? "bg-zinc-100/70 dark:bg-zinc-900/70" : ""
              }`}
            >
              {cols.map((c) => (
                <Fragment key={c.id}>{renderCell(c.id, m, leagueNames)}</Fragment>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function parseScore(v: string | null | undefined): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

function formatScore(v: string | null | undefined): string {
  return parseScore(v) == null ? "-" : String(v);
}

function formatResult(m: SoccerMatch): string {
  const h = parseScore(m.home_score);
  const a = parseScore(m.away_score);
  if (h == null || a == null) return "-";
  return h > a ? "승" : h === a ? "무" : "패";
}

function PredCell({ m }: { m: SoccerMatch }) {
  const p = m.pred;
  if (!p) return <>-</>;
  const cands: [string, number][] = [["홈", p.home], ["무", p.draw], ["원정", p.away]];
  cands.sort((a, b) => b[1] - a[1]);
  const pick = cands[0][0];
  const actual = actualPick(m);
  const cls =
    actual == null
      ? "text-white"
      : pick === actual
        ? "font-bold text-red-500 dark:text-red-400"
        : "font-bold text-blue-500 dark:text-blue-400";
  return <span className={cls}>{pick}</span>;
}

function actualPick(m: SoccerMatch): string | null {
  const h = parseScore(m.home_score);
  const a = parseScore(m.away_score);
  if (h == null || a == null) return null;
  return h > a ? "홈" : h === a ? "무" : "원정";
}

function formatPred(m: SoccerMatch): string {
  const p = m.pred;
  if (!p) return "-";
  const cands: [string, number][] = [["홈", p.home], ["무", p.draw], ["원정", p.away]];
  cands.sort((a, b) => b[1] - a[1]);
  return `${cands[0][0]} ${Math.round(cands[0][1] * 100)}%`;
}

function formatPredRatio(m: SoccerMatch): string {
  const p = m.pred;
  if (!p) return "-";
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  return `${pct(p.home)} / ${pct(p.draw)} / ${pct(p.away)}`;
}

function formatFormation(m: SoccerMatch): string {
  const home = m.lineup?.home ?? null;
  const away = m.lineup?.away ?? null;
  if (!home && !away) return "-";
  return `${home ?? "-"} vs ${away ?? "-"}`;
}

function formatStrength(m: SoccerMatch, key: keyof TeamStrength): string {
  const home = m.strength?.home?.[key] ?? null;
  const away = m.strength?.away?.[key] ?? null;
  if (home == null && away == null) return "-";
  return `${home ?? "-"} : ${away ?? "-"}`;
}

function formatOdds(odds: MatchOdds | undefined, company: "crown" | "bet365" | "one_x_bet"): string {
  const wdl: CompanyOdds = odds?.[company] ?? null;
  const pick = (v: { current?: number | null; initial?: number | null } | null | undefined) =>
    v?.current ?? v?.initial ?? null;
  const home = pick(wdl?.win_draw_lose?.home);
  const draw = pick(wdl?.win_draw_lose?.draw);
  const away = pick(wdl?.win_draw_lose?.away);
  if (home == null && draw == null && away == null) return "-";
  return `${home ?? "-"} / ${draw ?? "-"} / ${away ?? "-"}`;
}

function formatSameOdds(same: SameOdds | undefined): string {
  const wdl = same?.win_draw_lose ?? same?.same_odds?.win_draw_lose ?? null;
  if (!wdl || (wdl.home == null && wdl.draw == null && wdl.away == null)) return "-";
  return `${wdl.home ?? "-"} / ${wdl.draw ?? "-"} / ${wdl.away ?? "-"}`;
}

function WdlCell({ r }: { r: H2hRange | undefined }) {
  if (!r || (r.win == null && r.draw == null && r.loss == null)) return <>-</>;
  return (
    <span className="text-white">
      {r.win ?? "-"}승 {r.draw ?? "-"}무 {r.loss ?? "-"}패
    </span>
  );
}
