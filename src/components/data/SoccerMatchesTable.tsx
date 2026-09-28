import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import type { CompanyOdds, H2hRange, MatchOdds, SameOdds, SoccerMatch, TeamFormEntry, TeamStrength, Wdl } from "@/lib/queries";

export type ColumnId =
  | "league"
  | "season"
  | "round"
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
  | "recent1"
  | "recent2"
  | "recent3"
  | "recent4"
  | "recent5"
  | "recent6"
  | "recent7"
  | "recent8"
  | "recent9"
  | "recent10"
  | "recent11"
  | "recent12"
  | "recent13"
  | "recent14"
  | "recent15"
  | "recent16"
  | "recent17"
  | "recent18"
  | "recent19"
  | "recent20"
  | "crown"
  | "bet365"
  | "onexbet"
  | "sameOdds"
  | "formation"
  | "aGoals"
  | "aConceded"
  | "aShots"
  | "aCorners"
  | "aYellow"
  | "aFouls"
  | "aPossession"
  | "g15"
  | "g30"
  | "g45"
  | "g60"
  | "g75"
  | "g90"
  | "hWW"
  | "hDW"
  | "hLW"
  | "hWD"
  | "hDD"
  | "hLD"
  | "hWL"
  | "hDL"
  | "hLL"
  | "rf1"
  | "rf2"
  | "rf3"
  | "rf4"
  | "rf5"
  | "rf6"
  | "rf7"
  | "rf8"
  | "rf9"
  | "rf10"
  | "rf11"
  | "rf12"
  | "rf13"
  | "rf14"
  | "rf15"
  | "rf16"
  | "rf17"
  | "rf18"
  | "rf19"
  | "rf20"
  | "hfMatches";

export const CORE_COLUMNS: ColumnId[] = ["season", "round", "datetime", "home", "score", "away"];

export const FIELD_GROUPS: { id: string; label: string; cols: ColumnId[] }[] = [
  { id: "power", label: "전력대비", cols: ["power", "h2h", "condition", "attack", "defense", "value"] },
  { id: "recent", label: "최근승무패", cols: ["recent1", "recent2", "recent3", "recent4", "recent5", "recent6", "recent7", "recent8", "recent9", "recent10", "recent11", "recent12", "recent13", "recent14", "recent15", "recent16", "recent17", "recent18", "recent19", "recent20"] },
  { id: "recentform", label: "최근폼", cols: ["rf1", "rf2", "rf3", "rf4", "rf5", "rf6", "rf7", "rf8", "rf9", "rf10", "rf11", "rf12", "rf13", "rf14", "rf15", "rf16", "rf17", "rf18", "rf19", "rf20"] },
  { id: "odds", label: "배당률", cols: ["bet365", "crown", "onexbet"] },
  { id: "teaminfo", label: "팀정보", cols: ["aGoals", "aConceded", "aShots", "aCorners", "aYellow", "aFouls", "aPossession"] },
  { id: "goaltime", label: "득실점 시간대", cols: ["g15", "g30", "g45", "g60", "g75", "g90"] },
  { id: "halffull", label: "HT-FT", cols: ["hWW", "hDW", "hLW", "hWD", "hDD", "hLD", "hWL", "hDL", "hLL", "hfMatches"] },
];

export const SINGLE_COLUMNS: ColumnId[] = ["pred", "predRatio", "result", "sameOdds", "formation"];

export const COLUMNS: { id: ColumnId; label: string }[] = [
  { id: "season", label: "시즌" },
  { id: "round", label: "라운드" },
  { id: "datetime", label: "일정" },
  { id: "home", label: "홈" },
  { id: "score", label: "점수" },
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
  { id: "recent1", label: "최근1" },
  { id: "recent2", label: "최근2" },
  { id: "recent3", label: "최근3" },
  { id: "recent4", label: "최근4" },
  { id: "recent5", label: "최근5" },
  { id: "recent6", label: "최근6" },
  { id: "recent7", label: "최근7" },
  { id: "recent8", label: "최근8" },
  { id: "recent9", label: "최근9" },
  { id: "recent10", label: "최근10" },
  { id: "recent11", label: "최근11" },
  { id: "recent12", label: "최근12" },
  { id: "recent13", label: "최근13" },
  { id: "recent14", label: "최근14" },
  { id: "recent15", label: "최근15" },
  { id: "recent16", label: "최근16" },
  { id: "recent17", label: "최근17" },
  { id: "recent18", label: "최근18" },
  { id: "recent19", label: "최근19" },
  { id: "recent20", label: "최근20" },
  { id: "rf1", label: "최근폼1" },
  { id: "rf2", label: "최근폼2" },
  { id: "rf3", label: "최근폼3" },
  { id: "rf4", label: "최근폼4" },
  { id: "rf5", label: "최근폼5" },
  { id: "rf6", label: "최근폼6" },
  { id: "rf7", label: "최근폼7" },
  { id: "rf8", label: "최근폼8" },
  { id: "rf9", label: "최근폼9" },
  { id: "rf10", label: "최근폼10" },
  { id: "rf11", label: "최근폼11" },
  { id: "rf12", label: "최근폼12" },
  { id: "rf13", label: "최근폼13" },
  { id: "rf14", label: "최근폼14" },
  { id: "rf15", label: "최근폼15" },
  { id: "rf16", label: "최근폼16" },
  { id: "rf17", label: "최근폼17" },
  { id: "rf18", label: "최근폼18" },
  { id: "rf19", label: "최근폼19" },
  { id: "rf20", label: "최근폼20" },
  { id: "bet365", label: "Bet365" },
  { id: "crown", label: "Crown" },
  { id: "onexbet", label: "1XBet" },
  { id: "sameOdds", label: "동일배당률" },
  { id: "formation", label: "포메이션" },
  { id: "aGoals", label: "득점" },
  { id: "aConceded", label: "실점" },
  { id: "aShots", label: "유효슈팅" },
  { id: "aCorners", label: "코너" },
  { id: "aYellow", label: "옐로카드" },
  { id: "aFouls", label: "파울" },
  { id: "aPossession", label: "점유율" },
  { id: "g15", label: "1~15분" },
  { id: "g30", label: "16~30분" },
  { id: "g45", label: "31~45분" },
  { id: "g60", label: "46~60분" },
  { id: "g75", label: "61~75분" },
  { id: "g90", label: "76~90분" },
  { id: "hWW", label: "전반-승 / 결과-승" },
  { id: "hDW", label: "전반-무 / 결과-승" },
  { id: "hLW", label: "전반-패 / 결과-승" },
  { id: "hWD", label: "전반-승 / 결과-무" },
  { id: "hDD", label: "전반-무 / 결과-무" },
  { id: "hLD", label: "전반-패 / 결과-무" },
  { id: "hWL", label: "전반-승 / 결과-패" },
  { id: "hDL", label: "전반-무 / 결과-패" },
  { id: "hLL", label: "전반-패 / 결과-패" },
  { id: "hfMatches", label: "HT-FT경기수" },
];

const CELL = "whitespace-nowrap border-t border-zinc-400 px-4 py-1.5 text-center dark:border-zinc-600";const MONO_CELL =
  "whitespace-nowrap border-t border-zinc-400 px-4 py-1.5 font-mono text-xs text-center text-zinc-900 dark:border-zinc-600 dark:text-white";
const PRED_IDS: ReadonlySet<ColumnId> = new Set(["pred", "predRatio", "result"]);

type FilterCol = "season" | "round" | "home" | "away" | "result" | "pred";

function predPickLabel(m: SoccerMatch): string {
  const p = m.pred;
  if (!p) return "-";
  const cands: [string, number][] = [
    ["홈승", p.home],
    ["무승부", p.draw],
    ["원정승", p.away],
  ];
  cands.sort((a, b) => b[1] - a[1]);
  return cands[0][0];
}

function filterColValue(col: FilterCol, m: SoccerMatch): string {
  if (col === "season") return m.season;
  if (col === "round") return m.round != null ? String(m.round) : "-";
  if (col === "home") return m.home_team;
  if (col === "away") return m.away_team;
  if (col === "result") return formatResult(m);
  return predPickLabel(m);
}

function isFilterCol(id: ColumnId): id is FilterCol {
  return id === "season" || id === "round" || id === "home" || id === "away" || id === "result" || id === "pred";
}

type DivColor = "blue" | "gray" | null;

function divAfter(cols: { id: ColumnId }[], index: number): DivColor {
  if (index === cols.length - 1) return null;
  const curPred = PRED_IDS.has(cols[index].id);
  const nextPred = PRED_IDS.has(cols[index + 1].id);
  if ((curPred && !nextPred) || (!curPred && nextPred)) return "blue";
  return "gray";
}

const DIV_GRAY = "border-r border-zinc-400 dark:border-zinc-600";
const FRAME_L_GRAY = "border-l border-zinc-400 dark:border-zinc-600";
const FRAME_T_GRAY = "border-t border-zinc-400 dark:border-zinc-600";
const FRAME_B_GRAY = "border-b border-zinc-400 dark:border-zinc-600";

function divClass(div: DivColor): string {
  return div === "gray" ? DIV_GRAY : "";
}

function predEdgeOf(cols: { id: ColumnId }[], index: number): "single" | "first" | "mid" | "last" | null {
  if (!PRED_IDS.has(cols[index].id)) return null;
  const first = index === 0 || !PRED_IDS.has(cols[index - 1].id);
  const last = index === cols.length - 1 || !PRED_IDS.has(cols[index + 1].id);
  if (first && last) return "single";
  if (first) return "first";
  if (last) return "last";
  return "mid";
}

function predShadow(
  edge: "single" | "first" | "mid" | "last" | null,
  opts?: { top?: boolean; bottom?: boolean }
): string | undefined {
  if (!edge) return undefined;
  const c = "var(--pred-border)";
  const parts: string[] = [];
  if (edge === "single" || edge === "first") parts.push(`inset 1px 0 0 0 ${c}`);
  if (edge === "single" || edge === "last") parts.push(`inset -1px 0 0 0 ${c}`);
  if (opts?.top) parts.push(`inset 0 1px 0 0 ${c}`);
  if (opts?.bottom) parts.push(`inset 0 -1px 0 0 ${c}`);
  return parts.length > 0 ? parts.join(", ") : undefined;
}

function frameClasses(
  cols: { id: ColumnId }[],
  ci: number,
  opts?: { header?: boolean; lastRow?: boolean }
): { cls: string; shadow?: string } {
  const parts = [divClass(ci === cols.length - 1 ? "gray" : divAfter(cols, ci))];
  if (ci === 0) parts.push(FRAME_L_GRAY);
  if (opts?.header) parts.push(FRAME_T_GRAY);
  if (opts?.lastRow) parts.push(FRAME_B_GRAY);
  const shadow = predShadow(predEdgeOf(cols, ci), { top: opts?.header, bottom: opts?.lastRow });
  return { cls: parts.join(" "), shadow };
}

function renderCell(
  id: ColumnId,
  m: SoccerMatch,
  leagueNames?: Record<string, string>,
  cols?: { id: ColumnId }[],
  ci = 0,
  isLastRow = false
): ReactNode {
  const borders = cols ? frameClasses(cols, ci, { lastRow: isLastRow }) : { cls: "", shadow: undefined };
  const borderStyle = borders.shadow ? { boxShadow: borders.shadow } : undefined;
  const borderCls = borders.cls;
  switch (id) {
    case "league":
      return <td style={borderStyle} className={`${CELL} ${borderCls}`}>{leagueNames?.[m.league_code] ?? m.league_code}</td>;
    case "season":
      return (
        <td style={borderStyle} className={`${CELL} ${borderCls}`}>
          {m.season}
        </td>
      );
    case "round":
      return (
        <td style={borderStyle} className={`${CELL} ${borderCls}`}>
          {m.round != null ? m.round : "-"}
        </td>
      );
    case "datetime":
      return (
        <td style={borderStyle} className={`${CELL} ${borderCls}`}>
          {m.match_date ?? "-"} {m.match_time ?? ""}
        </td>
      );
    case "home":
      return (
        <td style={borderStyle} className={`${CELL} ${borderCls} font-medium`}>
          {m.home_team}
          {m.home_rank ? <span className="ml-1 text-xs text-zinc-500 dark:text-zinc-400">({m.home_rank})</span> : null}
        </td>
      );
    case "score":
      return (
        <td style={borderStyle} className={`whitespace-nowrap border-t border-zinc-400 px-4 py-1.5 text-center font-mono font-bold dark:border-zinc-600 ${borderCls}`}>
          {formatScore(m.home_score)} : {formatScore(m.away_score)}
        </td>
      );
    case "away":
      return (
        <td style={borderStyle} className={`${CELL} ${borderCls} font-medium`}>
          {m.away_team}
          {m.away_rank ? <span className="ml-1 text-xs text-zinc-500 dark:text-zinc-400">({m.away_rank})</span> : null}
        </td>
      );
    case "pred":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <PredCell m={m} />
        </td>
      );
    case "predRatio":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          {formatPredRatio(m)}
        </td>
      );
    case "result":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          {formatResult(m)}
        </td>
      );
    case "power":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatStrength(m, "power")}</td>;
    case "h2h":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatStrength(m, "head_to_head")}</td>;
    case "condition":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatStrength(m, "condition")}</td>;
    case "attack":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatStrength(m, "attack")}</td>;
    case "defense":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatStrength(m, "defense")}</td>;
    case "value":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatStrength(m, "value")}</td>;
    case "recent1":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_1} />
        </td>
      );
    case "recent2":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_2} />
        </td>
      );
    case "recent3":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_3} />
        </td>
      );
    case "recent4":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_4} />
        </td>
      );
    case "recent5":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_5} />
        </td>
      );
    case "recent6":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_6} />
        </td>
      );
    case "recent7":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_7} />
        </td>
      );
    case "recent8":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_8} />
        </td>
      );
    case "recent9":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_9} />
        </td>
      );
    case "recent10":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_10} />
        </td>
      );
    case "recent11":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_11} />
        </td>
      );
    case "recent12":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_12} />
        </td>
      );
    case "recent13":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_13} />
        </td>
      );
    case "recent14":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_14} />
        </td>
      );
    case "recent15":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_15} />
        </td>
      );
    case "recent16":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_16} />
        </td>
      );
    case "recent17":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_17} />
        </td>
      );
    case "recent18":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_18} />
        </td>
      );
    case "recent19":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_19} />
        </td>
      );
    case "recent20":
      return (
        <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>
          <WdlCell r={m.head_to_head?.recent_20} />
        </td>
      );
    case "rf1":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 1)}</td>;
    case "rf2":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 2)}</td>;
    case "rf3":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 3)}</td>;
    case "rf4":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 4)}</td>;
    case "rf5":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 5)}</td>;
    case "rf6":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 6)}</td>;
    case "rf7":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 7)}</td>;
    case "rf8":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 8)}</td>;
    case "rf9":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 9)}</td>;
    case "rf10":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 10)}</td>;
    case "rf11":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 11)}</td>;
    case "rf12":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 12)}</td>;
    case "rf13":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 13)}</td>;
    case "rf14":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 14)}</td>;
    case "rf15":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 15)}</td>;
    case "rf16":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 16)}</td>;
    case "rf17":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 17)}</td>;
    case "rf18":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 18)}</td>;
    case "rf19":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 19)}</td>;
    case "rf20":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatRecentForm(m, 20)}</td>;
    case "crown":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatOdds(m.odds, "crown")}</td>;
    case "bet365":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatOdds(m.odds, "bet365")}</td>;
    case "onexbet":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatOdds(m.odds, "one_x_bet")}</td>;
    case "sameOdds":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatSameOdds(m.same_odds)}</td>;
    case "formation":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatFormation(m)}</td>;
    case "aGoals":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "goals")}</td>;
    case "aConceded":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "conceded")}</td>;
    case "aShots":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "shots_on_target")}</td>;
    case "aCorners":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "corners")}</td>;
    case "aYellow":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "yellow_cards")}</td>;
    case "aFouls":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "fouls")}</td>;
    case "aPossession":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatTeamAvg(m, "possession")}</td>;
    case "g15":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatGoalSlot(m, "1~15")}</td>;
    case "g30":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatGoalSlot(m, "16~30")}</td>;
    case "g45":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatGoalSlot(m, "31~45")}</td>;
    case "g60":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatGoalSlot(m, "46~60")}</td>;
    case "g75":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatGoalSlot(m, "61~75")}</td>;
    case "g90":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatGoalSlot(m, "76~90")}</td>;
    case "hWW":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "win_win")}</td>;
    case "hDW":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "draw_win")}</td>;
    case "hLW":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "loss_win")}</td>;
    case "hWD":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "win_draw")}</td>;
    case "hDD":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "draw_draw")}</td>;
    case "hLD":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "loss_draw")}</td>;
    case "hWL":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "win_loss")}</td>;
    case "hDL":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "draw_loss")}</td>;
    case "hLL":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfFull(m, "loss_loss")}</td>;
    case "hfMatches":
      return <td style={borderStyle} className={`${MONO_CELL} ${borderCls}`}>{formatHalfMatches(m)}</td>;
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
  const [openFilter, setOpenFilter] = useState<FilterCol | null>(null);
  const [colFilters, setColFilters] = useState<Partial<Record<FilterCol, string[] | null>>>({});

  const filterOptions = useMemo(
    () => ({
      season: [...new Set(rows.map((r) => r.season))].sort().reverse(),
      round: [...new Set(rows.map((r) => (r.round != null ? String(r.round) : "-")))].sort((a, b) => {
        if (a === "-") return 1;
        if (b === "-") return -1;
        return Number(a) - Number(b);
      }),
      home: [...new Set(rows.map((r) => r.home_team))].sort(),
      away: [...new Set(rows.map((r) => r.away_team))].sort(),
      result: [...new Set(rows.map((r) => formatResult(r)))].sort(),
      pred: [...new Set(rows.map((r) => predPickLabel(r)))].sort(),
    }),
    [rows]
  );
  const leagueKey = useMemo(() => [...new Set(rows.map((r) => r.league_code))].sort().join(","), [rows]);
  useEffect(() => {
    setColFilters({});
    setOpenFilter(null);
  }, [leagueKey]);

  if (rows.length === 0) return null;
  const cols = COLUMNS.filter((c) => CORE_COLUMNS.includes(c.id) || !hidden.includes(c.id));
  if (cols.length === 0) return null;

  function selOf(col: FilterCol): string[] | null {
    return colFilters[col] ?? null;
  }

  function openFilterPanel(col: FilterCol) {
    setOpenFilter(col);
  }

  function setColFilter(col: FilterCol, sel: string[] | null) {
    setColFilters((prev) => ({ ...prev, [col]: sel }));
  }

  function commitDraft(col: FilterCol, draft: string[] | null) {
    const all = filterOptions[col];
    const sel = draft ?? all;
    setColFilter(col, sel.length === all.length ? null : sel);
  }

  const visibleRows = rows.filter((m) =>
    (Object.keys(filterOptions) as FilterCol[]).every((col) => {
      const sel = selOf(col);
      return !sel || sel.includes(filterColValue(col, m));
    })
  );
  const groupOf: number[] = [];
  {
    let g = 0;
    let prev: string | null = null;
    for (const m of visibleRows) {
      const key = `${m.season}|${m.round ?? "-"}`;
      if (prev !== null && key !== prev) g += 1;
      prev = key;
      groupOf.push(g);
    }
  }
  return (
    <div className="max-h-[75vh] min-h-[50vh] w-full flex-none overflow-auto overscroll-none rounded-2xl lg:max-h-full lg:min-h-0 lg:flex-1">
      <table className="w-full border-separate border-spacing-0 text-left text-sm" style={{ minWidth: Math.max(cols.length * 120, 400) }}>
        <thead>
          <tr>
            {cols.map((c, ci) => {
              const frame = frameClasses(cols, ci, { header: true });
              const fid = isFilterCol(c.id) ? c.id : null;
              const sel = fid ? selOf(fid) : null;
              const active = sel !== null;
              return (
                <th
                  key={c.id}
                  style={frame.shadow ? { boxShadow: frame.shadow } : undefined}
                  className={`sticky top-0 z-10 whitespace-nowrap bg-zinc-100 px-4 py-1.5 text-center font-medium text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 ${frame.cls} ${fid ? "relative" : ""}`}
                >
                  {fid ? (
                    <>
                      {c.label}
                      <button
                        onClick={() => openFilterPanel(fid)}
                        aria-label={`${c.label} 필터`}
                        className={`absolute top-1/2 right-1 -translate-y-1/2 rounded p-1 hover:bg-zinc-200 dark:hover:bg-zinc-700 ${
                          active ? "text-blue-600 dark:text-blue-400" : "text-zinc-400 dark:text-zinc-500"
                        }`}
                      >
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                          <path d="M2 3h12l-4.5 5.2V13l-3 1.5V8.2L2 3z" />
                        </svg>
                      </button>
                    </>
                  ) : (
                    c.label
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {visibleRows.length === 0 ? (
            <tr>
              <td colSpan={cols.length} className="px-4 py-8 text-center text-sm text-zinc-500">
                조건에 맞는 경기가 없습니다.
              </td>
            </tr>
          ) : (
            visibleRows.map((m, i) => (
              <tr
                key={m.id}
                className={`${
                  groupOf[i] % 2 === 1 ? "bg-zinc-200/70 dark:bg-zinc-800/60" : ""
                }`}
              >
                {cols.map((c, ci) => (
                  <Fragment key={c.id}>
                    {renderCell(c.id, m, leagueNames, cols, ci, i === visibleRows.length - 1)}
                  </Fragment>
                ))}
              </tr>
            ))
          )}
        </tbody>
        </table>
      {openFilter && (
        <FilterModal
          key={openFilter}
          title={COLUMNS.find((c) => c.id === openFilter)?.label ?? ""}
          values={filterOptions[openFilter]}
          initial={selOf(openFilter) ?? filterOptions[openFilter]}
          onApply={(sel) => {
            commitDraft(openFilter, sel);
            setOpenFilter(null);
          }}
          onCancel={() => setOpenFilter(null)}
        />
      )}
    </div>
  );
}

function FilterModal({
  title,
  values,
  initial,
  onApply,
  onCancel,
}: {
  title: string;
  values: string[];
  initial: string[];
  onApply: (sel: string[]) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState<string[]>(initial);
  const [q, setQ] = useState("");
  const shown = values.filter((v) => v.toLowerCase().includes(q.trim().toLowerCase()));

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  function toggle(v: string) {
    setDraft((cur) => (cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]));
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <div
        className="flex max-h-[80vh] w-full max-w-sm flex-col rounded-2xl border border-zinc-200 bg-white p-4 shadow-xl dark:border-zinc-700 dark:bg-zinc-900"
      >
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold">{title} 필터</span>
          <button
            onClick={onCancel}
            aria-label="필터 닫기"
            className="rounded p-1 text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="검색"
          className="mt-3 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
        />
        <div className="mt-2 flex items-center justify-between text-sm">
          <span className="text-zinc-500">
            {draft.length}/{values.length}
          </span>
          <div className="flex gap-2">
            <button onClick={() => setDraft(values)} className="underline underline-offset-2">
              전체
            </button>
            <button onClick={() => setDraft([])} className="underline underline-offset-2">
              해제
            </button>
          </div>
        </div>
        <div className="mt-2 min-h-0 flex-1 space-y-1 overflow-y-auto">
          {shown.map((v) => (
            <label key={v} className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2 py-2 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800">
              <input
                type="checkbox"
                checked={draft.includes(v)}
                onChange={() => toggle(v)}
                className="h-4 w-4 shrink-0 accent-blue-600 dark:accent-blue-400"
              />
              <span className="whitespace-nowrap">{v}</span>
            </label>
          ))}
          {shown.length === 0 && <p className="px-2 py-1.5 text-sm text-zinc-500">일치하는 항목이 없습니다.</p>}
        </div>
        <button
          onClick={() => onApply(draft)}
          className="mt-3 w-full rounded-xl bg-blue-600 py-2 text-sm font-medium text-white hover:brightness-110 dark:bg-blue-400 dark:text-black"
        >
          적용
        </button>
      </div>
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
  return h > a ? "홈승" : h === a ? "무승부" : "원정승";
}

function PredCell({ m }: { m: SoccerMatch }) {
  const p = m.pred;
  if (!p) return <>-</>;
  const cands: [string, number][] = [["홈승", p.home], ["무승부", p.draw], ["원정승", p.away]];
  cands.sort((a, b) => b[1] - a[1]);
  const pick = cands[0][0];
  const actual = actualPick(m);
  const cls =
    actual == null
      ? "text-zinc-900 dark:text-white"
      : pick === actual
        ? "font-bold text-red-500 dark:text-red-400"
        : "font-bold text-blue-600 dark:text-blue-400";
  return <span className={cls}>{pick}</span>;
}

function actualPick(m: SoccerMatch): string | null {
  const h = parseScore(m.home_score);
  const a = parseScore(m.away_score);
  if (h == null || a == null) return null;
  return h > a ? "홈승" : h === a ? "무승부" : "원정승";
}

function formatPred(m: SoccerMatch): string {
  const p = m.pred;
  if (!p) return "-";
  const cands: [string, number][] = [["홈승", p.home], ["무승부", p.draw], ["원정승", p.away]];
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

function formatTeamAvg(
  m: SoccerMatch,
  key: "goals" | "conceded" | "possession" | "shots_on_target" | "corners" | "yellow_cards" | "fouls"
): string {
  const home = m.team_info?.avg_stats?.home?.[key] ?? null;
  const away = m.team_info?.avg_stats?.away?.[key] ?? null;
  if (home == null && away == null) return "-";
  return `${home ?? "-"} : ${away ?? "-"}`;
}

function formatGoalSlot(m: SoccerMatch, slot: string): string {
  const s = (m.goal_time ?? m.team_info?.goal_time)?.find((x) => x.slot === slot);
  if (!s) return "-";
  const hs = s.home_scored ?? null;
  const hc = s.home_conceded ?? null;
  const as = s.away_scored ?? null;
  const ac = s.away_conceded ?? null;
  if (hs == null && hc == null && as == null && ac == null) return "-";
  return `${hs ?? "-"}:${hc ?? "-"} / ${as ?? "-"}:${ac ?? "-"}`;
}

function formatHalfFull(m: SoccerMatch, key: string): string {
  const v = (m.ht_ft ?? m.team_info?.half_full)?.stats?.[key];
  const home = v?.home ?? null;
  const away = v?.away ?? null;
  if (home == null && away == null) return "-";
  return `${home ?? "-"} : ${away ?? "-"}`;
}

function formatHalfMatches(m: SoccerMatch): string {
  const hf = m.ht_ft ?? m.team_info?.half_full;
  const home = hf?.home_matches ?? null;
  const away = hf?.away_matches ?? null;
  if (home == null && away == null) return "-";
  return `${home ?? "-"} : ${away ?? "-"}`;
}

function formatStrength(m: SoccerMatch, key: keyof TeamStrength): string {
  const home = m.strength?.home?.[key] ?? null;
  const away = m.strength?.away?.[key] ?? null;
  if (home == null && away == null) return "-";
  return `${home ?? "-"} : ${away ?? "-"}`;
}

function formatOdds(odds: MatchOdds | undefined, company: "crown" | "bet365" | "one_x_bet"): string {
  const wdl: CompanyOdds = odds?.[company] ?? null;
  const pick = (v: { live?: number | null; current?: number | null; initial?: number | null } | null | undefined) =>
    v?.initial ?? v?.live ?? null;
  const home = pick(wdl?.win_draw_lose?.home);
  const draw = pick(wdl?.win_draw_lose?.draw);
  const away = pick(wdl?.win_draw_lose?.away);
  if (home == null && draw == null && away == null) return "-";
  return `${home ?? "-"} / ${draw ?? "-"} / ${away ?? "-"}`;
}

function formatRecentForm(m: SoccerMatch, n: number): string {
  const rf = m.recent_form as Record<string, Record<string, TeamFormEntry | undefined> | undefined> | undefined;
  const fmt = (s: TeamFormEntry | undefined) => {
    if (s == null || (s.win == null && s.draw == null && s.loss == null)) return null;
    return `${s.win ?? "-"}승${s.draw ?? "-"}무${s.loss ?? "-"}패`;
  };
  const home = fmt(rf?.home?.[`recent_${n}`]);
  const away = fmt(rf?.away?.[`recent_${n}`]);
  if (home == null && away == null) return "-";
  return `${home ?? "-"} : ${away ?? "-"}`;
}

function formatSameOdds(same: SameOdds | undefined): string {
  const wdl = same?.win_draw_lose ?? same?.same_odds?.win_draw_lose ?? null;
  if (!wdl || (wdl.home == null && wdl.draw == null && wdl.away == null)) return "-";
  return `${wdl.home ?? "-"} / ${wdl.draw ?? "-"} / ${wdl.away ?? "-"}`;
}

function WdlCell({ r }: { r: H2hRange | undefined }) {
  if (!r || (r.win == null && r.draw == null && r.loss == null)) return <>-</>;
  return (
    <span className="text-zinc-900 dark:text-white">
      {r.win ?? "-"}승 {r.draw ?? "-"}무 {r.loss ?? "-"}패
    </span>
  );
}
