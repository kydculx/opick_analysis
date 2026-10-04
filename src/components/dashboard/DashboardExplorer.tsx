"use client";

import { useEffect, useState, useRef } from "react";
import type { SoccerMatch, League } from "@/lib/queries";
import { ALL_FEATURES, applyTemp, blendProbs, cappedDot, dcProba, drawFeatures, legacyParts, logregProba, patternProbs, rowFeatures, selectFeatures, sigmoid } from "@/lib/permatch-math";
import { xgbProba, type XgbSlim } from "@/lib/xgb";
import type { PermatchArtifact } from "@/lib/predict";
import { SoccerMatchesTable, COLUMNS, CORE_COLUMNS, FIELD_GROUPS, SINGLE_COLUMNS, type ColumnId } from "@/components/data/SoccerMatchesTable";
import { EmptyState } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export type SavedModel = {
  ver: string;
  train_seasons: string[];
  model_type: string;
  metrics: { valid_acc?: number; valid_logloss?: number } | null;
  created_at: string;
};

export type PredMapEntry = { home: number; draw: number; away: number; ver: string };

export type AccuracyMap = Record<string, { n: number; hit: number; acc: number | null }>;

// 순차 탐색용 이동폭 사다리: 0.001 단위 전수 스캔(최대 4000회) 대신 12개 대표값만
// 평가하고, 개선된 구간만 조밀하게 재탐색한다.
const TUNE_LADDER = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4];

function parseScoreLocal(v: string | null | undefined): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

function predictLocal(m: SoccerMatch, art: PermatchArtifact, ver: string) {
  try {
    if (art.model_type === "lstm") return null;
    if (art.model_type === "xgb" && art.xgb) {
      const p = xgbProba(art.xgb, selectFeatures(rowFeatures(m), art.features));
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver };
    }
    if (art.model_type === "logreg" && art.scaler_mean && art.scaler_scale && art.coef && art.intercept) {
      const p = logregProba(selectFeatures(rowFeatures(m), art.features), art.scaler_mean, art.scaler_scale, art.coef, art.intercept);
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
    if (art.model_type === "ensemble" && art.base && art.xgb) {
      const b = legacyParts(m, art.base);
      if (!b) return null;
      const qx = xgbProba(art.xgb, rowFeatures(m));
      const alpha = art.blend_w ?? 1;
      const dd = alpha * b.dd + (1 - alpha) * qx[1];
      const lin = [b.ph * (1 - dd), dd, (1 - b.ph) * (1 - dd)];
      const bb = art.base;
      const p = applyTemp(blendProbs(lin, b.x, bb.patterns, bb.pattern_tau ?? 0), bb.T);
      if (p.some((v) => !Number.isFinite(v))) return null;
      return { home: p[0], draw: p[1], away: p[2], ver };
    }
    const e = art.emphasis ?? new Array(art.features?.length ?? art.mu.length).fill(1);
    const rf = selectFeatures(rowFeatures(m), art.features);
    const x = rf.map((v, i) => (v - art.mu[i]) / art.sd[i]).map((v, i) => v * (e[i] ?? 1));
    let lin: number[];
    if (art.weights && art.hfa != null && art.draw_prior != null) {
      const s = cappedDot(x, art.weights, art.hfa, art.contrib_cap ?? null);
      const ph = sigmoid(s);
      let dd: number;
      if (art.draw_weights && art.draw_mu && art.draw_sd) {
        const df = drawFeatures(m);
        const xd = art.draw_mu.map((mu, i) => (df[i] - mu) / (art.draw_sd as number[])[i]);
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

export function DashboardExplorer() {
  const [leagues, setLeagues] = useState<League[]>([]);
  const [league, setLeague] = useState("");
  const [seasons, setSeasons] = useState<string[]>([]);
  const [checked, setChecked] = useState<string[]>([]);
  const [trainCount, setTrainCount] = useState("6");
  const [featSel, setFeatSel] = useState<string[]>([...ALL_FEATURES]);
  const [fullTrain, setFullTrain] = useState(true);
  const [trialsText, setTrialsText] = useState("10000");
  const [jobsText, setJobsText] = useState("4");
  const [matches, setMatches] = useState<SoccerMatch[]>([]);
  const [hiddenCols, setHiddenCols] = useState<ColumnId[]>([]);
  const [loadingLeagues, setLoadingLeagues] = useState(true);
  const [loadingSeasons, setLoadingSeasons] = useState(false);
  const [loadingMatches, setLoadingMatches] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainState, setTrainState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [learnMode, setLearnMode] = useState<"match" | "cumulative">("match");
  const [rawMatches, setRawMatches] = useState<SoccerMatch[]>([]);
  const [predMap, setPredMap] = useState<Record<string, PredMapEntry>>({});
  const rawCacheRef = useRef<Record<string, SoccerMatch[]>>({});
  const tweakedRef = useRef<number[] | null>(null);
  const predCacheRef = useRef<Record<string, Record<string, PredMapEntry>>>({});
  const accCacheRef = useRef<Record<string, AccuracyMap>>({});
  const baseMapRef = useRef<AccuracyMap | null>(null);
  const lastPreviewMapRef = useRef<AccuracyMap | null>(null);
  const [models, setModels] = useState<SavedModel[]>([]);
  const [selVer, setSelVer] = useState("");
  const [accuracy, setAccuracy] = useState<Record<string, { n: number; hit: number; acc: number | null }>>({});
  type ModelDetail = {
    model_type: string;
    detail: {
      model_type?: string;
      features: string[];
      weights?: number[];
      W?: number[][];
      b?: number[];
      emphasis: number[] | null;
      hfa: number | null;
      T: number | null;
      draw_prior: number | null;
      mu: number[] | null;
      sd: number[] | null;
      draw_weights: number[] | null;
      draw_bias: number;
      draw_mu: number[] | null;
      draw_sd: number[] | null;
      patterns: { home: number[][]; draw: number[][]; away: number[][] } | null;
      contrib_cap: number | null;
      train_seasons: string[];
      valid: string | string[] | null;
      trials: number | null;
      pattern_tau: number | null;
      pattern_stats: Record<string, { rank: number; freq: number; acc: number }[]> | null;
      base_ver?: string;
      blend_w?: number;
      base?: PermatchArtifact | null;
      xgb?: XgbSlim | null;
    } | null;
  };
  const [modelDetail, setModelDetail] = useState<ModelDetail | null>(null);
  const [tweaked, setTweaked] = useState<number[] | null>(null);
  const [wOrder, setWOrder] = useState<number[] | null>(null);
  const tuningRef = useRef(false);
  const [tuning, setTuning] = useState(false);
  const [tuneActive, setTuneActive] = useState<number | null>(null);
  const [tuneInfo, setTuneInfo] = useState<{ sweeps: number; base: number; best: number; stale: number } | null>(null);
  const gridRef = useRef(false);
  const gridJobRef = useRef<string | null>(null);
  const gridBestRef = useRef(-1);
  const [gridding, setGridding] = useState(false);
  const [gridInfo, setGridInfo] = useState<{ done: number; total: number; best: number | null; sweep: number | null; curAcc: number | null } | null>(null);
  const [gridLive, setGridLive] = useState<{ weights: number[]; hfa: number; acc: number } | null>(null);
  const [gridStepText, setGridStepText] = useState("0.01");
  const [gridJobsText, setGridJobsText] = useState("4");
  const [gridBatchText, setGridBatchText] = useState("4096");
  const [gridWminText, setGridWminText] = useState("-3.0");
  const [gridWmaxText, setGridWmaxText] = useState("3.0");
  const [gridDrawWText, setGridDrawWText] = useState("0");
  const FEATURE_KO: Record<string, string> = {
    rank: "순위차", power: "전력", hstr: "H2H강도", cond: "컨디션", att: "공격", def: "수비",
    val: "가치", form5: "최근폼", h2h5: "H2H5", avg_goals: "평균득점", avg_conceded: "평균실점",
    avg_poss: "점유율", market: "배당",
  };
  const [applyState, setApplyState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [appliedVer, setAppliedVer] = useState("");
  const appliedKeyRef = useRef("");
  const inflightKeyRef = useRef("");
  const genRef = useRef(0);
  const leagueRef = useRef(league);
  const selVerRef = useRef(selVer);
  const checkedRef = useRef(checked);
  const modelDetailRef = useRef(modelDetail);
  const rawMatchesRef = useRef(rawMatches);
  const previewRef = useRef(previewWith);
  leagueRef.current = league;
  selVerRef.current = selVer;
  checkedRef.current = checked;
  modelDetailRef.current = modelDetail;
  rawMatchesRef.current = rawMatches;
  previewRef.current = previewWith;

  function selectLeague(lg: string) {
    if (!lg || lg === leagueRef.current) return;
    genRef.current += 1;
    inflightKeyRef.current = "";
    appliedKeyRef.current = "";
    setLeague(lg);
    setSeasons([]);
    setChecked([]);
    setTrainState("idle");
    setApplyState("idle");
    setAppliedVer("");
    setModels([]);
    setSelVer("");
    setAccuracy({});
    setRawMatches([]);
    setPredMap({});
    setMatches([]);
    setError(null);
  }

  useEffect(() => {
    fetch("/api/leagues")
      .then((r) => r.json())
      .then((j) => {
        if (j.ok) {
          setLeagues(j.data);
          if (j.data.length > 0) setLeague(j.data[0].code);
        } else {
          setError(j.error ?? "리그 조회 실패");
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "리그 조회 실패"))
      .finally(() => setLoadingLeagues(false));
  }, []);

  useEffect(() => {
    if (!league) return;
    const gen = ++genRef.current;
    setLoadingSeasons(true);
    setChecked([]);
    setTrainState("idle");
    setApplyState("idle");
    setAppliedVer("");
    appliedKeyRef.current = "";
    inflightKeyRef.current = "";
    setModels([]);
    setSelVer("");
    setAccuracy({});
    setRawMatches([]);
    setPredMap({});
    setMatches([]);
    setHiddenCols([...FIELD_GROUPS.flatMap((g) => (g.id === "odds" ? [] : g.cols)), "sameOdds", "formation"]);
    fetchModels(league, gen);
    fetch(`/api/seasons?league=${encodeURIComponent(league)}`)
      .then((r) => r.json())
      .then((j) => {
        if (gen !== genRef.current || leagueRef.current !== league) return;
        if (j.ok) {
          setSeasons(j.data);
          setChecked(j.data);
        } else {
          setError(j.error ?? "시즌 조회 실패");
        }
      })
      .catch((e) => {
        if (gen !== genRef.current) return;
        setError(e instanceof Error ? e.message : "시즌 조회 실패");
      })
      .finally(() => {
        if (gen === genRef.current) setLoadingSeasons(false);
      });
  }, [league]);

  function cacheKey(lg: string, ver: string) {
    return `${lg}|${ver}`;
  }

  function loadPredMap(lg: string, ver: string, seasonList: string[]) {
    const ck = cacheKey(lg, ver);
    const hit = predCacheRef.current[ck];
    if (hit) {
      setPredMap(hit);
      return;
    }
    if (seasonList.length === 0) return;
    const gen = genRef.current;
    setLoadingMatches(true);
    const qs = new URLSearchParams({ league: lg, seasons: seasonList.join(","), model_ver: ver });
    fetch(`/api/predictions?${qs.toString()}`)
      .then((r) => r.json())
      .then((j) => {
        if (gen !== genRef.current) return;
        if (j.ok) {
          predCacheRef.current[ck] = j.data;
          setPredMap(j.data);
        } else {
          setError(j.error ?? "예측 조회 실패");
        }
      })
      .catch((e) => {
        if (gen !== genRef.current) return;
        setError(e instanceof Error ? e.message : "예측 조회 실패");
      })
      .finally(() => {
        if (gen === genRef.current) setLoadingMatches(false);
      });
  }

  function loadAccuracy(lg: string, ver: string, seasonList: string[]) {
    const ck = cacheKey(lg, ver);
    const hit = accCacheRef.current[ck];
    if (hit) {
      setAccuracy(hit);
      baseMapRef.current = hit;
      return;
    }
    const qs = new URLSearchParams({ league: lg, model_ver: ver, seasons: seasonList.join(",") });
    fetch(`/api/accuracy?${qs.toString()}`, { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => {
        if (j.ok) {
          accCacheRef.current[ck] = j.data;
          setAccuracy(j.data);
          baseMapRef.current = j.data;
        }
      })
      .catch(() => setAccuracy({}));
  }

  useEffect(() => {
    if (!league || seasons.length === 0) return;
    const gen = genRef.current;
    const hit = rawCacheRef.current[league];
    if (hit) {
      setRawMatches(hit);
      return;
    }
    const ctrl = new AbortController();
    setLoadingMatches(true);
    const qs = new URLSearchParams({ league, seasons: seasons.join(",") });
    fetch(`/api/matches?${qs.toString()}`, { signal: ctrl.signal })
      .then((r) => r.json())
      .then((j) => {
        if (gen !== genRef.current) return;
        if (j.ok) {
          rawCacheRef.current[league] = j.data;
          setRawMatches(j.data);
        } else {
          setError(j.error ?? "경기 조회 실패");
        }
      })
      .catch((e) => {
        if (ctrl.signal.aborted || gen !== genRef.current) return;
        setError(e instanceof Error ? e.message : "경기 조회 실패");
      })
      .finally(() => {
        if (gen === genRef.current) setLoadingMatches(false);
      });
    return () => ctrl.abort();
  }, [league, seasons]);

  useEffect(() => {
    if (!league || !appliedVer || seasons.length === 0) return;
    loadPredMap(league, appliedVer, seasons);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [league, appliedVer, seasons]);

  useEffect(() => {
    const set = new Set(checked);
    setMatches(
      rawMatches
        .filter((m) => set.has(m.season))
        .map((m) => {
          const key = m.source_match_id || String(m.id);
          const p = predMap[key];
          return { ...m, pred: p ? { home: p.home, draw: p.draw, away: p.away, ver: p.ver } : null };
        })
    );
  }, [rawMatches, checked, predMap]);

  function toggleColumn(id: ColumnId) {
    setHiddenCols((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function pollJob(base: string, jobId: string): Promise<"done" | "error"> {
    for (let i = 0; i < 150; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        const st = await fetch(`${base}?jobId=${encodeURIComponent(jobId)}`).then((r) => r.json());
        if (st.ok && st.job.status !== "running") {
          return st.job.status === "done" ? "done" : "error";
        }
      } catch {
        return "error";
      }
    }
    return "error";
  }

  async function fetchModels(lg: string, gen?: number) {
    if (!lg) return;
    try {
      const j = await fetch(`/api/models?league=${encodeURIComponent(lg)}`).then((r) => r.json());
      if (gen !== undefined && (gen !== genRef.current || leagueRef.current !== lg)) return;
      if (j.ok) {
        setModels(j.data);
        setSelVer((prev) =>
          j.data.some((m: SavedModel) => m.ver === prev) ? prev : (j.data[0]?.ver ?? "")
        );
      }
    } catch {
      if (gen !== undefined && (gen !== genRef.current || leagueRef.current !== lg)) return;
      setModels([]);
    }
  }

  async function handleTrain() {
    if (trainingOrdered.length === 0 || trainState === "running") return;
    if (featSel.length === 0) {
      setError("피처를 1개 이상 선택하세요");
      return;
    }
    setTrainState("running");
    setAppliedVer("");
    setPredMap({});
    appliedKeyRef.current = "";
    try {
      const res = await fetch("/api/match-train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          league,
          seasons: trainingOrdered,
          ...(featSel.length !== ALL_FEATURES.length ? { features: featSel } : {}),
          ...(fullTrain
            ? { mode: "full", trials: trialsText, jobs: jobsText }
            : { mode: "fast" }),
        }),
      }).then((r) => r.json());
      if (!res.ok) {
        setTrainState("error");
        return;
      }
      const done = await pollJob("/api/match-train", res.jobId as string);
      setTrainState(done);
      if (done === "done") {
        await fetchModels(league);
        setSelVer(res.ver as string);
        setApplyState("idle");
      }
    } catch {
      setTrainState("error");
    }
  }

  leagueRef.current = league;
  selVerRef.current = selVer;
  checkedRef.current = checked;

  function buildApplyKey(lg: string, ver: string, seasonList: string[]) {
    return `${lg}|${ver}|${[...seasonList].sort().join(",")}`;
  }

  useEffect(() => {
    if (!league || !selVer || checked.length === 0) {
      if (selVer === "" && inflightKeyRef.current && !inflightKeyRef.current.startsWith(`${league}|`)) {
        inflightKeyRef.current = "";
        setApplyState("idle");
      }
      return;
    }
    if (models.length > 0 && !models.some((m) => m.ver === selVer)) return;
    if (seasons.length > 0 && !checked.every((c) => seasons.includes(c))) return;
    const key = buildApplyKey(league, selVer, checked);
    if (key === appliedKeyRef.current) return;
    appliedKeyRef.current = key;
    setAppliedVer(selVer);
    setApplyState("done");
    delete predCacheRef.current[cacheKey(league, selVer)];
    delete accCacheRef.current[cacheKey(league, selVer)];
    loadAccuracy(league, selVer, checked);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [league, selVer, checked, models, seasons]);

  useEffect(() => {
    tuningRef.current = false;
    setTuning(false);
    setTuneActive(null);
    setTuneInfo(null);
    tweakedRef.current = null;
    setTweaked(null);
    if (!league || !selVer) {
      setModelDetail(null);
      return;
    }
    const gen = genRef.current;
    fetch(`/api/models?league=${encodeURIComponent(league)}&ver=${encodeURIComponent(selVer)}`)
      .then((r) => r.json())
      .then((j) => {
        if (gen !== genRef.current) return;
        if (j.ok) {
          setModelDetail({ model_type: j.model_type, detail: j.detail });
          const feats = j.detail?.features;
          if (Array.isArray(feats) && feats.length > 0) {
            setFeatSel(feats.filter((f: unknown) => typeof f === "string" && (ALL_FEATURES as string[]).includes(f as string)) as string[]);
          } else {
            setFeatSel([...ALL_FEATURES]);
          }
          const ws = j.detail?.weights;
          if (Array.isArray(ws)) {
            setWOrder(ws.map((_, i) => i).sort((a, b) => Math.abs((ws as number[])[b]) - Math.abs((ws as number[])[a])));
          }
        } else {
          setModelDetail(null);
        }
      })
      .catch(() => {
        if (gen === genRef.current) setModelDetail(null);
      });
    return () => {
      tuningRef.current = false;
    };
  }, [league, selVer]);

  useEffect(() => {
    const id = setInterval(async () => {
      if (document.visibilityState !== "visible") return;
      if (tuningRef.current) return;
      const md = modelDetailRef.current;
      const lg = leagueRef.current;
      const ver = selVerRef.current;
      if (!md || md.model_type !== "permatch" || !md.detail || !lg || !ver) return;
      if (tweakedRef.current) return;
      try {
        const j = await fetch(`/api/models?league=${encodeURIComponent(lg)}&ver=${encodeURIComponent(ver)}`, { cache: "no-store" }).then((r) => r.json());
        if (!j.ok || !j.detail || !Array.isArray(j.detail.weights)) return;
        const cur = modelDetailRef.current?.detail;
        if (JSON.stringify(j.detail.weights) === JSON.stringify(cur?.weights) && j.detail.hfa === cur?.hfa) return;
        delete predCacheRef.current[cacheKey(lg, ver)];
        delete accCacheRef.current[cacheKey(lg, ver)];
        const ws = j.detail.weights as number[];
        setModelDetail({ model_type: j.model_type, detail: j.detail });
        setWOrder(ws.map((_, i) => i).sort((a, b) => Math.abs(ws[b]) - Math.abs(ws[a])));
        previewRef.current(ws, typeof j.detail.hfa === "number" ? j.detail.hfa : 0);
        loadAccuracy(lg, ver, checkedRef.current);
      } catch {
      }
    }, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  function overallOf(m: Record<string, { n: number; hit: number }>): number | null {
    let n = 0, hit = 0;
    for (const v of Object.values(m)) { n += v.n; hit += v.hit; }
    return n > 0 ? hit / n : null;
  }

  function previewWith(weights: number[], hfa?: number, full = true): number | null {
    const d = modelDetail?.detail;
    if (!d || !d.mu || !d.sd || rawMatches.length === 0) return null;
    const art: PermatchArtifact = {
      weights,
      hfa: hfa ?? d.hfa ?? 0,
      T: d.T ?? 1,
      draw_prior: d.draw_prior ?? 0,
      mu: d.mu,
      sd: d.sd,
      features: d.features ?? undefined,
      emphasis: d.emphasis ?? undefined,
      draw_weights: d.draw_weights,
      draw_bias: d.draw_bias ?? 0,
      draw_mu: d.draw_mu,
      draw_sd: d.draw_sd,
      patterns: d.patterns ?? undefined,
      pattern_tau: d.pattern_tau ?? 0,
      contrib_cap: d.contrib_cap ?? null,
    };
    const map: Record<string, PredMapEntry> = {};
    for (const m of rawMatches) {
      const p = predictLocal(m, art, selVer);
      if (p) map[m.source_match_id || String(m.id)] = p;
    }
    if (full) setPredMap(map);
    const stats: Record<string, { n: number; hit: number }> = {};
    for (const m of rawMatches) {
      const p = map[m.source_match_id || String(m.id)];
      if (!p) continue;
      const h = parseScoreLocal(m.home_score);
      const a = parseScoreLocal(m.away_score);
      if (h == null || a == null) continue;
      const actual = h > a ? 0 : h === a ? 1 : 2;
      const probs = [p.home, p.draw, p.away];
      const pick = probs.indexOf(Math.max(...probs));
      const s = String(m.season);
      stats[s] = stats[s] || { n: 0, hit: 0 };
      stats[s].n += 1;
      if (pick === actual) stats[s].hit += 1;
    }
    const out: AccuracyMap = {};
    for (const [s, v] of Object.entries(stats)) out[s] = { ...v, acc: v.n > 0 ? v.hit / v.n : null };
    const ov = overallOf(out);
    lastPreviewMapRef.current = out;
    if (weights === d.weights) baseMapRef.current = out;
    setAccuracy(out);
    return ov;
  }

  function nudgeStep(): number {
    const s = Number.parseFloat(gridStepText);
    return Number.isFinite(s) && s > 0 ? s : 0.01;
  }

  function nudge(fi: number, dir: 1 | -1) {
    const base = tweakedRef.current ?? modelDetail?.detail?.weights ?? null;
    if (!base) return;
    const next = [...base];
    next[fi] = Math.round((next[fi] + dir * nudgeStep()) * 10000) / 10000;
    tweakedRef.current = next;
    setTweaked(next);
    previewWith(next);
  }

  function setWeight(fi: number, text: string) {
    const v = Number(text);
    if (!Number.isFinite(v)) return;
    const base = tweakedRef.current ?? modelDetail?.detail?.weights ?? null;
    if (!base) return;
    const next = [...base];
    next[fi] = Math.round(v * 10000) / 10000;
    tweakedRef.current = next;
    setTweaked(next);
    previewWith(next);
  }

  async function autoTune() {
    if (tuningRef.current) {
      tuningRef.current = false;
      setTuning(false);
      return;
    }
    if (gridRef.current) {
      setError("전수탐색 중에는 랜덤을 시작할 수 없습니다");
      return;
    }
    const d = modelDetail?.detail;
    if (!d || !d.mu || !d.sd) return;
    if (!d.weights) {
      setError("가중치 없는 모델은 직접미세조정 미지원");
      return;
    }
    const train = new Set(d.train_seasons);
    const scored = rawMatches.filter((m) => {
      if (!checked.includes(String(m.season)) || train.has(String(m.season))) return false;
      return parseScoreLocal(m.home_score) != null && parseScoreLocal(m.away_score) != null;
    });
    if (scored.length === 0) {
      setError("조절용 미학습 시즌을 체크하세요");
      return;
    }
    const nw = d.weights.length;
    const e = d.emphasis ?? new Array(nw).fill(1);
    const cap = d.contrib_cap ?? null;
    const T = d.T ?? 1;
    const pats = d.patterns ?? undefined;
    const tau = d.pattern_tau ?? 0;
    const Xe: number[][] = [];
    const labels: number[] = [];
    const ddArr: number[] = [];
    for (const m of scored) {
      const h = parseScoreLocal(m.home_score) as number;
      const a = parseScoreLocal(m.away_score) as number;
      labels.push(h > a ? 0 : h === a ? 1 : 2);
      const rf = selectFeatures(rowFeatures(m), d.features);
      Xe.push(rf.map((v, i) => (v - (d.mu as number[])[i]) / (d.sd as number[])[i]).map((v, i) => v * (e[i] ?? 1)));
      if (d.draw_weights && d.draw_mu && d.draw_sd) {
        const df = drawFeatures(m);
        const dw = d.draw_weights as number[];
        const ds = d.draw_mu.map((mu, i) => (df[i] - mu) / (d.draw_sd as number[])[i])
          .reduce((acc, v, i) => acc + v * dw[i], 0) + (d.draw_bias ?? 0);
        ddArr.push(sigmoid(ds));
      } else {
        ddArr.push(d.draw_prior ?? 0);
      }
    }
    const ppArr = Xe.map((x) => patternProbs(x, pats, tau));
    const accLine = (wcur: number[], hfaCur: number) => {
      let hit = 0;
      for (let i = 0; i < Xe.length; i++) {
        const s = cappedDot(Xe[i], wcur, hfaCur, cap);
        const ph = sigmoid(s);
        const dd = ddArr[i];
        const lin = [ph * (1 - dd), dd, (1 - ph) * (1 - dd)];
        const pp = ppArr[i];
        const a = pp ? lin[0] * 0.5 + pp[0] * 0.5 : lin[0];
        const b = pp ? lin[1] * 0.5 + pp[1] * 0.5 : lin[1];
        const c = pp ? lin[2] * 0.5 + pp[2] * 0.5 : lin[2];
        const pick = a >= b ? (a >= c ? 0 : 2) : b >= c ? 1 : 2;
        if (pick === labels[i]) hit++;
      }
      return hit / Xe.length;
    };
    const accOf = (w: number[], hfa: number) => {
      let hit = 0;
      for (let i = 0; i < Xe.length; i++) {
        const s = cappedDot(Xe[i], w, hfa, cap);
        const ph = sigmoid(s);
        const dd = ddArr[i];
        const p = applyTemp(blendProbs([ph * (1 - dd), dd, (1 - ph) * (1 - dd)], Xe[i], pats, tau), T);
        const pick = p.indexOf(Math.max(...p));
        if (pick === labels[i]) hit++;
      }
      return hit / Xe.length;
    };
    tuningRef.current = true;
    setTuning(true);
    let w = [...(tweakedRef.current ?? d.weights)];
    let hfa = d.hfa ?? 0;
    const base = accOf(w, hfa);
    let best = base;
    let bestW = [...w];
    let bestHfa = hfa;
    let sweeps = 0;
    let stale = 0;
    let lastPv = 0;
    let lastDisp = 0;
    let lastYield = 0;
    let tuneMode = "sequential";
    setTuneInfo({ sweeps, base, best, stale });
    previewWith(w, hfa);
    const order = (wOrder && wOrder.length === nw ? [...wOrder] : [...Array(nw).keys()]).concat(-1);
  const WMIN = -3;
  const WMAX = 3;
    const clamp4 = (v: number) => Math.min(WMAX, Math.max(WMIN, Math.round(v * 10000) / 10000));
    while (tuningRef.current) {
      let improved = false;
      if (tuneMode === "sequential") {
        for (const j of order) {
          if (!tuningRef.current) break;
          setTuneActive(j);
          for (const dir of [1, -1] as const) {
            let evals = 0;
            const tryDelta = async (delta: number): Promise<"hit" | "miss" | "stop"> => {
              if (!tuningRef.current) return "stop";
              const cw = [...w];
              let ch = hfa;
              if (j === -1) ch = clamp4(hfa + dir * delta);
              else cw[j] = clamp4(cw[j] + dir * delta);
              if (j === -1 ? ch === hfa : cw[j] === w[j]) return "stop";
              const v = accLine(cw, ch);
              evals++;
              const nowMs = Date.now();
              if (v > best) {
                best = v;
                w = cw;
                hfa = ch;
                improved = true;
                lastDisp = nowMs;
                setTweaked(cw);
                return "hit";
              } else if (nowMs - lastDisp > 120) {
                lastDisp = nowMs;
                setTweaked(cw);
              }
              if (evals % 20 === 0 || nowMs - lastYield > 12) {
                lastYield = nowMs;
                await new Promise((r) => setTimeout(r, 0));
              }
              if (nowMs - lastPv > 2500) {
                lastPv = nowMs;
                previewWith(cw, ch, false);
              }
              return "miss";
            };
            let prev = 0;
            for (const delta of TUNE_LADDER) {
              if (!tuningRef.current) break;
              const before = best;
              const r = await tryDelta(delta);
              if (r === "stop") break;
              if (best > before) {
                const gap = delta - prev;
                if (gap > 0.0015) {
                  const rd = Math.max(0.001, gap / 8);
                  let d = prev + rd;
                  while (d < delta - 1e-9) {
                    if (!tuningRef.current) break;
                    const rr = await tryDelta(Math.round(d * 10000) / 10000);
                    if (rr === "stop") break;
                    d += rd;
                  }
                }
              }
              prev = delta;
            }
            await new Promise((r) => setTimeout(r, 0));
          }
          if (!tuningRef.current) break;
        }
        tuneMode = "random";
        bestW = [...w];
        bestHfa = hfa;
        setTuneActive(null);
      } else {
        setTuneActive(null);
        const kw = [...w].map((v) => clamp4(v + (Math.random() * 2 - 1) * 0.1));
        const kh = clamp4(hfa + (Math.random() * 2 - 1) * 0.05);
        const ka = accLine(kw, kh);
        const nowR = Date.now();
        w = kw;
        hfa = kh;
        if (ka > best) {
          best = ka;
          bestW = [...kw];
          bestHfa = kh;
          improved = true;
        }
        if (nowR - lastDisp > 150) {
          lastDisp = nowR;
          setTweaked(kw);
        }
        if (nowR - lastPv > 2500) {
          lastPv = nowR;
          previewWith(w, hfa, false);
        }
        await new Promise((r) => setTimeout(r, 0));
      }
      if (!improved && tuneMode === "sequential") {
        stale++;
        if (stale % 5 === 0) {
          const kw = bestW.map((v) => clamp4(v + (Math.random() * 2 - 1) * 0.05));
          const kh = clamp4(bestHfa + (Math.random() * 2 - 1) * 0.02);
          const ka = accLine(kw, kh);
          if (ka > best) {
            best = ka;
            w = kw;
            hfa = kh;
            bestW = [...kw];
            bestHfa = kh;
            improved = true;
            stale = 0;
          }
        }
      } else if (!improved && tuneMode === "random") {
        // in random mode, just keep going
      }
      sweeps++;
      tweakedRef.current = w;
      setTweaked(w);
      if (improved) previewWith(w, hfa);
      setTuneInfo({ sweeps, base, best, stale });
      if (improved) {
        let skipSave = false;
        let latestW: number[] | null = null;
        let latestH = 0;
        try {
          const lr = await fetch(`/api/models?league=${encodeURIComponent(league)}&ver=${encodeURIComponent(selVer)}`).then((r) => r.json());
          const lw = lr?.detail?.weights;
          if (lr?.ok && Array.isArray(lw) && lw.length === nw) {
            latestW = [...lw];
            latestH = typeof lr.detail.hfa === "number" ? lr.detail.hfa : 0;
            const lacc = accOf(latestW, latestH);
            if (lacc > best) {
              best = lacc;
              w = [...latestW];
              hfa = latestH;
              bestW = [...latestW];
              bestHfa = latestH;
              tweakedRef.current = w;
              setTweaked(w);
              setModelDetail((prev) =>
                prev?.detail ? { ...prev, detail: { ...prev.detail, weights: [...latestW as number[]], hfa: latestH } } : prev
              );
              const ab = previewWith(w, hfa);
              if (lastPreviewMapRef.current) baseMapRef.current = lastPreviewMapRef.current;
              setTuneInfo({ sweeps, base, best, stale });
              skipSave = true;
            }
          }
        } catch { }
        if (skipSave) {
          setTuneInfo({ sweeps, base, best, stale });
        } else {
          const saveW = tuneMode === "random" ? [...bestW] : w;
          const saveH = tuneMode === "random" ? bestHfa : hfa;
          if (latestW && JSON.stringify(latestW) === JSON.stringify(saveW) && latestH === saveH) {
            setTuneInfo({ sweeps, base, best, stale });
          } else {
            const res = await fetch("/api/models", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ league, ver: selVer, weights: saveW, hfa: saveH }),
        }).then((r) => r.json()).catch(() => ({ ok: false }));
        if (res.ok) {
          setModelDetail((prev) =>
            prev?.detail ? { ...prev, detail: { ...prev.detail, weights: saveW, hfa: saveH } } : prev
          );
          delete accCacheRef.current[cacheKey(league, selVer)];
          previewWith(saveW, saveH, false);
          if (lastPreviewMapRef.current) baseMapRef.current = lastPreviewMapRef.current;
            }
          }
        }
      }
    }
    setTuneActive(null);
    setTuning(false);
  }

  async function gridTune() {
    if (gridRef.current) {
      gridRef.current = false;
      setGridding(false);
      setGridLive(null);
      if (gridJobRef.current) {
        fetch(`/api/grid?jobId=${encodeURIComponent(gridJobRef.current)}`, { method: "DELETE" }).catch(() => {});
      }
      return;
    }
    if (tuningRef.current) {
      setError("랜덤 조절 중에는 전수탐색을 시작할 수 없습니다");
      return;
    }
    const d = modelDetail?.detail;
    if (!d || !league || !selVer) return;
    if (!d.weights) {
      setError("가중치 없는 모델은 직접미세조정 미지원");
      return;
    }
    const train = new Set(d.train_seasons);
    const tune = [...new Set(rawMatches.filter((m) => {
      if (!checked.includes(String(m.season)) || train.has(String(m.season))) return false;
      return parseScoreLocal(m.home_score) != null && parseScoreLocal(m.away_score) != null;
    }).map((m) => String(m.season)))].sort();
    if (tune.length === 0) {
      setError("조절용 미학습 시즌을 체크하세요");
      return;
    }
    const five = selVer.includes("-f5");
    const step = parseFloat(gridStepText);
    if (!Number.isFinite(step) || step < 0.001 || step > 2) {
      setError("축간격은 0.001~2 사이 숫자로 입력하세요");
      return;
    }
    const jobs = Math.floor(Number(gridJobsText));
    if (!Number.isFinite(jobs) || jobs < 1) {
      setError("jobs는 1 이상 정수로 입력하세요");
      return;
    }
    const batch = Math.floor(Number(gridBatchText));
    if (!Number.isFinite(batch) || batch < 1) {
      setError("묶음은 1 이상 정수로 입력하세요");
      return;
    }
    const wmin = Number(gridWminText);
    const wmax = Number(gridWmaxText);
    if (!Number.isFinite(wmin) || !Number.isFinite(wmax) || !(wmin < wmax)) {
      setError("wmin < wmax 숫자로 입력하세요");
      return;
    }
    const drawW = Number(gridDrawWText);
    if (!Number.isFinite(drawW) || drawW < 0) {
      setError("무가중은 0 이상 숫자로 입력하세요");
      return;
    }
    gridRef.current = true;
    setGridding(true);
    setGridInfo(null);
    setGridLive(null);
    gridBestRef.current = -1;
    const nw = d.weights.length;
    const refreshGridView = async (preview: boolean): Promise<void> => {
      try {
        const j = await fetch(`/api/models?league=${encodeURIComponent(league)}&ver=${encodeURIComponent(selVer)}`, { cache: "no-store" }).then((r) => r.json());
        if (!j.ok || !j.detail) return;
        const ws = j.detail?.weights;
        setModelDetail({ model_type: j.model_type, detail: j.detail });
        if (Array.isArray(ws)) {
          setWOrder(ws.map((_: number, i: number) => i).sort((a: number, b: number) => Math.abs((ws as number[])[b]) - Math.abs((ws as number[])[a])));
        }
        if (preview && !tuningRef.current && !tweakedRef.current && Array.isArray(ws)) {
          previewWith(ws as number[], typeof j.detail?.hfa === "number" ? j.detail.hfa : 0);
        }
      } catch {
      }
    };
    try {
      const res = await fetch("/api/grid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          league, ver: selVer, tune, five, jobs, grid_step: step,
          batch, wmin, wmax, draw_w: drawW,
        }),
      }).then((r) => r.json());
      if (!res.ok) {
        gridRef.current = false;
        setGridding(false);
        setError("전수탐색 시작 실패");
        return;
      }
      gridJobRef.current = res.jobId as string;
      const qs = `jobId=${encodeURIComponent(res.jobId as string)}&league=${encodeURIComponent(league)}&ver=${encodeURIComponent(selVer)}${five ? "&five=1" : ""}`;
      for (let i = 0; i < 648000 && gridRef.current; i++) {
        await new Promise((r) => setTimeout(r, 16));
        if (!gridRef.current) break;
        try {
          const st = await fetch(`/api/grid?${qs}`, { cache: "no-store" }).then((r) => r.json());
          if (st.ok && st.progress) {
            const pg = st.progress as {
              done: number; total: number; best: number | null;
              sweep?: number | null; curW?: number[] | null;
              curHfa?: number | null; curAcc?: number | null;
            };
            setGridInfo({
              done: pg.done, total: pg.total, best: pg.best,
              sweep: typeof pg.sweep === "number" ? pg.sweep : null,
              curAcc: typeof pg.curAcc === "number" ? pg.curAcc : null,
            });
            if (Array.isArray(pg.curW) && pg.curW.length === nw
              && pg.curW.every((v) => typeof v === "number")
              && typeof pg.curHfa === "number" && typeof pg.curAcc === "number") {
              setGridLive({ weights: [...pg.curW], hfa: pg.curHfa, acc: pg.curAcc });
            }
            const b = st.progress.best;
            if (typeof b === "number") {
              if (gridBestRef.current < 0) {
                gridBestRef.current = b;
              } else if (b > gridBestRef.current) {
                gridBestRef.current = b;
                delete predCacheRef.current[cacheKey(league, selVer)];
                delete accCacheRef.current[cacheKey(league, selVer)];
                await refreshGridView(true);
                loadAccuracy(league, selVer, checked);
              }
            }
          }
          if (st.ok && st.job.status !== "running") {
            if (st.job.status === "done") {
              await fetchModels(league);
              delete predCacheRef.current[cacheKey(league, selVer)];
              delete accCacheRef.current[cacheKey(league, selVer)];
              await refreshGridView(true);
              loadAccuracy(league, selVer, checked);
            }
            break;
          }
        } catch {
          break;
        }
      }
    } catch {
    }
    gridRef.current = false;
    setGridding(false);
    setGridLive(null);
  }

  async function saveTweaks() {
    if (!tweaked) return;
    const res = await fetch("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ league, ver: selVer, weights: tweaked }),
    }).then((r) => r.json()).catch(() => ({ ok: false, error: "저장 실패" }));
    if (!res.ok) {
      setError(res.error ?? "저장 실패");
      return;
    }
    setModelDetail((prev) =>
      prev?.detail ? { ...prev, detail: { ...prev.detail, weights: tweaked } } : prev
    );
    delete accCacheRef.current[cacheKey(league, selVer)];
    const lb = lastPreviewMapRef.current;
    if (lb) baseMapRef.current = lb;
    tweakedRef.current = null;
    setTweaked(null);
  }

  const ascSeasons = [...seasons].sort();
  const trainN = Math.max(0, Math.min(Number.parseInt(trainCount, 10) || 0, Math.max(0, seasons.length - 1)));
  const trainingOrdered = ascSeasons.slice(0, trainN);
  const viewSeasons = [...ascSeasons].reverse();

  function toggleGroup(cols: ColumnId[]) {
    const allVisible = cols.every((id) => !hiddenCols.includes(id));
    setHiddenCols((prev) =>
      allVisible ? [...prev, ...cols.filter((id) => !prev.includes(id))] : prev.filter((id) => !cols.includes(id))
    );
  }

  const singleCols = COLUMNS.filter((c) => SINGLE_COLUMNS.includes(c.id));
  const predGroupCols: ColumnId[] = ["pred", "predRatio", "result"];
  const detailSingleIds: ColumnId[] = ["sameOdds", "formation"];
  const predSingles = singleCols.filter((c) => predGroupCols.includes(c.id));
  const detailSingles = singleCols.filter((c) => detailSingleIds.includes(c.id));
  const detailGroupCols: ColumnId[] = [...FIELD_GROUPS.flatMap((g) => g.cols), ...detailSingleIds];
  const applying =
    loadingLeagues || loadingSeasons || loadingMatches || applyState === "running";
  const trainSet = new Set(modelDetail?.detail?.train_seasons ?? []);
  const nonTrainOf = (m: Record<string, { n: number; hit: number }>) =>
    Object.fromEntries(Object.entries(m).filter(([s]) => !trainSet.has(s)));
  const curOverall = overallOf(nonTrainOf(accuracy));
  const baseOverall = overallOf(nonTrainOf(baseMapRef.current ?? {}));
  const canSave =
    !!tweaked && curOverall != null && baseOverall != null && curOverall > baseOverall;
  const saveTitle = !tweaked
    ? "가중치를 조절하면 활성화"
    : curOverall == null || baseOverall == null
      ? "적중률 계산 중"
      : `현재 ${(curOverall * 100).toFixed(1)}% vs 저장됨 ${(baseOverall * 100).toFixed(1)}%`;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-visible lg:flex-row lg:overflow-hidden">
      <aside className="w-full shrink-0 space-y-4 rounded-2xl border border-zinc-200 p-4 dark:border-zinc-800 lg:max-h-full lg:w-64 lg:overflow-y-auto">
        <div>
          <label className="text-xs font-medium text-zinc-500">리그 선택</label>
          <select
            value={league}
            onChange={(e) => selectLeague(e.target.value)}
            disabled={loadingLeagues}
            suppressHydrationWarning
            className="mt-1 w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-950"
          >
            {loadingLeagues ? (
              <option>불러오는 중...</option>
            ) : (
              leagues.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name_ko}
                </option>
              ))
            )}
          </select>
        </div>

        <div>
          <div className="flex items-baseline justify-between">
              <span className="text-xs font-medium text-zinc-500">
                시즌 적중률 (미학습 {viewSeasons.filter((s) => !trainSet.has(s)).length}/{viewSeasons.length})
              </span>
          </div>
          <div className="mt-1 space-y-0.5">
            {loadingSeasons ? (
              <p className="text-sm text-zinc-500">시즌 불러오는 중...</p>
            ) : viewSeasons.length === 0 ? (
              <p className="px-2 py-1 text-sm text-zinc-500">조회할 시즌이 없습니다.</p>
            ) : (
              (() => {
                const isTrain = (s: string) => trainSet.has(s);
                const chrono = [...viewSeasons].reverse().filter((s) => !isTrain(s));
                const pts = chrono
                  .map((s, i) => {
                    const a = applyState === "done" ? accuracy[s] : undefined;
                    return a?.acc != null ? { s, i, acc: a.acc, n: a.n, hit: a.hit } : null;
                  })
                  .filter((p): p is { s: string; i: number; acc: number; n: number; hit: number } => p !== null);
                const W = 232, H = 118, PL = 3, PR = 3, PT = 8, PB = 14;
                const n = chrono.length;
                const X = (i: number) => (n <= 1 ? PL + (W - PL - PR) / 2 : PL + (i / (n - 1)) * (W - PL - PR));
                const Y = (acc: number) => PT + (1 - Math.max(0, Math.min(1, acc))) * (H - PT - PB);
                let totN = 0, totHit = 0;
                for (const p of pts) { totN += p.n; totHit += p.hit; }
                const overall = totN > 0 ? totHit / totN : null;
                const line = pts.map((p) => `${X(p.i).toFixed(1)},${Y(p.acc).toFixed(1)}`).join(" ");
                const dotColor = (acc: number) =>
                  acc >= 0.6 ? "#10b981" : acc >= 0.5 ? "#2563eb" : "#f43f5e";
                const best = pts.length > 0 ? pts.reduce((m, p) => (p.acc > m.acc ? p : m), pts[0]) : null;
                const worst = pts.length > 0 ? pts.reduce((m, p) => (p.acc < m.acc ? p : m), pts[0]) : null;
                const short = (s: string) => s.split("-").map((y) => y.slice(2, 4)).join("-");
                const gid = "accArea";
                return (
                  <div>
                    <div className="flex items-end justify-between px-0.5">
                      <div>
                        <div className="font-mono text-[26px] font-bold leading-none text-zinc-900 dark:text-zinc-100">
                          {overall != null ? `${(overall * 100).toFixed(1)}` : "−"}
                          <span className="text-sm font-medium text-zinc-400">%</span>
                        </div>
                        <div className="mt-0.5 font-mono text-[10px] text-zinc-400">
                          {totN > 0 ? `${totHit}/${totN} 적중` : "모델 적용 시 표시"}
                        </div>
                      </div>
                      <div className="pb-0.5 text-right font-mono text-[10px] leading-tight">
                        <div className="text-zinc-500">
                          최고 <span className="font-semibold text-emerald-600 dark:text-emerald-400">{best ? `${(best.acc * 100).toFixed(1)}` : "−"}</span>
                        </div>
                        <div className="text-zinc-500">
                          최저 <span className="font-semibold text-rose-500 dark:text-rose-400">{worst ? `${(worst.acc * 100).toFixed(1)}` : "−"}</span>
                        </div>
                      </div>
                    </div>
                    <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full" role="img" aria-label="시즌 적중률 추이">
                      <defs>
                        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#2563eb" stopOpacity={0.25} />
                          <stop offset="100%" stopColor="#2563eb" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      {[0, 0.5, 1].map((g) => (
                        <line key={g} x1={PL} x2={W - PR} y1={Y(g)} y2={Y(g)} stroke="#52525b" strokeWidth={0.5} opacity={0.3} />
                      ))}
                      {overall != null && (
                        <line x1={PL} x2={W - PR} y1={Y(overall)} y2={Y(overall)} stroke="#a1a1aa" strokeWidth={1} strokeDasharray="3 3" opacity={0.8}>
                          <title>전체 {(overall * 100).toFixed(1)}% ({totHit}/{totN})</title>
                        </line>
                      )}
                      {pts.length > 1 && (
                        <polygon points={`${PL},${H - PB} ${line} ${W - PR},${H - PB}`} fill={`url(#${gid})`} />
                      )}
                      {pts.length > 1 && (
                        <polyline points={line} fill="none" stroke="#2563eb" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
                      )}
                      {pts.map((p) => (
                        <g key={p.s}>
                          <title>{`${p.s} ${(p.acc * 100).toFixed(1)}% (${p.hit}/${p.n})${isTrain(p.s) ? " · 학습시즌" : ""}${best && p.s === best.s ? " · 최고" : ""}${worst && p.s === worst.s ? " · 최저" : ""}`}</title>
                          {best && p.s === best.s && (
                            <circle cx={X(p.i)} cy={Y(p.acc)} r={5.5} fill="none" stroke="#10b981" strokeWidth={1.5} opacity={0.8} />
                          )}
                          <circle cx={X(p.i)} cy={Y(p.acc)} r={3} fill={dotColor(p.acc)} stroke="#fff" strokeWidth={1} />
                        </g>
                      ))}
                      {chrono.map((s, i) => {
                        const parts = short(s).split("-");
                        return (
                          <text key={s} x={X(i)} y={parts.length > 1 ? H - 11 : H - 3} textAnchor="middle" fontSize={7} fill="#71717a" fontFamily="monospace">
                            {parts.map((part, k) => (
                              <tspan key={k} x={X(i)} dy={k === 0 ? 0 : 8}>{part}</tspan>
                            ))}
                          </text>
                        );
                      })}
                    </svg>
                  </div>
                );
              })()
            )}
          </div>
        </div>

        <hr className="border-zinc-200 dark:border-zinc-800" />

        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-500">학습시즌 ({trainingOrdered.length})</span>
            <label className="flex items-center gap-1 text-xs text-zinc-500">
              학습시즌수
              <input
                value={trainCount}
                onChange={(e) => setTrainCount(e.target.value.replace(/[^0-9]/g, ""))}
                disabled={trainState === "running"}
                inputMode="numeric"
                aria-label="학습시즌수"
                className="w-12 rounded-lg border border-zinc-300 bg-white px-2 py-0.5 text-right font-mono text-xs text-zinc-700 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300"
              />
            </label>
          </div>
            <div className="mt-2 flex gap-4 text-sm" role="radiogroup" aria-label="학습 모드">
              <label className="flex cursor-pointer items-center gap-1.5">
              <input
                type="radio"
                name="learn-mode"
                checked={learnMode === "match"}
                onChange={() => setLearnMode("match")}
                className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-400"
              />
              <span>경기당</span>
            </label>
            <label className="flex items-center gap-1.5 opacity-40">
              <input type="radio" name="learn-mode" disabled className="h-3.5 w-3.5" />
              <span>누적(준비중)</span>
            </label>
          </div>
          <details open className="mt-2 rounded-xl border border-zinc-200 dark:border-zinc-800">
            <summary className="cursor-pointer px-2.5 py-1.5 text-xs font-medium text-zinc-500">
              피처 선택 ({featSel.length}/{ALL_FEATURES.length})
            </summary>
            <div className="flex flex-wrap gap-x-3 gap-y-1.5 px-2.5 pb-1">
              {ALL_FEATURES.map((f) => (
                <label key={f} className="flex cursor-pointer items-center gap-1 text-xs hover:opacity-80" title={f}>
                  <input
                    type="checkbox"
                    checked={featSel.includes(f)}
                    onChange={() =>
                      setFeatSel((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]))
                    }
                    disabled={trainState === "running"}
                    className="h-3.5 w-3.5 accent-blue-600 disabled:opacity-40 dark:accent-blue-400"
                  />
                  <span>{FEATURE_KO[f] ?? f}</span>
                </label>
              ))}
            </div>
            <div className="px-2.5 pb-2">
              <button
                onClick={() => setFeatSel([...ALL_FEATURES])}
                disabled={trainState === "running"}
                className="text-[11px] text-blue-600 underline underline-offset-2 disabled:opacity-40 dark:text-blue-400"
              >
                전체 선택
              </button>
            </div>
          </details>
          <details open className="mt-2 rounded-xl border border-zinc-200 dark:border-zinc-800">
            <summary className="cursor-pointer px-2.5 py-1.5 text-xs font-medium text-zinc-500">
              학습 방식 ({fullTrain ? `전체 ${trialsText}회` : "빠른 단발"})
            </summary>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-2.5 pb-2">
              <label className="flex cursor-pointer items-center gap-1 text-xs">
                <input
                  type="checkbox"
                  checked={fullTrain}
                  onChange={() => setFullTrain((v) => !v)}
                  disabled={trainState === "running"}
                  className="h-3.5 w-3.5 accent-blue-600 disabled:opacity-40 dark:accent-blue-400"
                />
                <span>새학습(전체탐색)</span>
              </label>
              {fullTrain && (
                <>
                  <label className="flex items-center gap-1 text-xs text-zinc-500">
                    trials
                    <input
                      value={trialsText}
                      onChange={(e) => setTrialsText(e.target.value.replace(/[^0-9]/g, ""))}
                      disabled={trainState === "running"}
                      inputMode="numeric"
                      aria-label="trials"
                      className="w-16 rounded-lg border border-zinc-300 bg-white px-2 py-0.5 text-right font-mono text-xs text-zinc-700 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300"
                    />
                  </label>
                  <label className="flex items-center gap-1 text-xs text-zinc-500">
                    jobs
                    <input
                      value={jobsText}
                      onChange={(e) => setJobsText(e.target.value.replace(/[^0-9]/g, ""))}
                      disabled={trainState === "running"}
                      inputMode="numeric"
                      aria-label="jobs"
                      className="w-10 rounded-lg border border-zinc-300 bg-white px-2 py-0.5 text-right font-mono text-xs text-zinc-700 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300"
                    />
                  </label>
                </>
              )}
            </div>
          </details>
          <Button
            size="sm"
            disabled={trainingOrdered.length === 0 || trainState === "running" || trainState === "done"}
            onClick={handleTrain}
            className={
              trainingOrdered.length > 0 && (trainState === "idle" || trainState === "error")
                ? "mt-3 w-full bg-blue-600 text-white hover:brightness-110 dark:bg-blue-400 dark:text-black dark:hover:brightness-110"
                : "mt-3 w-full"
            }
          >
            {trainState === "running"
              ? "학습 중..."
              : trainState === "done"
                ? "학습 완료"
                : trainState === "error"
                  ? "실패 · 다시 시도"
                  : `학습하기${trainingOrdered.length > 0 ? ` (${trainingOrdered.length})` : ""}`}
          </Button>
        </div>

        <hr className="border-zinc-200 dark:border-zinc-800" />

        <div>
          <select
            value={selVer}
            onChange={(e) => {
              if (e.target.value === selVer) return;
              setSelVer(e.target.value);
              setApplyState("idle");
              setAppliedVer("");
              setPredMap({});
              appliedKeyRef.current = "";
            }}
            disabled={models.length === 0}
            suppressHydrationWarning
            className="mt-1 w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-950"
          >
            {models.length === 0 ? (
              <option>학습된 모델 없음</option>
            ) : (
              models.map((m) => {
                const labelParts = [
                  m.ver.replace("_", " "),
                  m.train_seasons.join(","),
                  m.metrics?.valid_acc != null ? `acc ${m.metrics.valid_acc.toFixed(3)}` : "",
                ].filter((p) => p !== "");
                return (
                  <option key={m.ver} value={m.ver}>
                    {labelParts.join(" · ")}
                  </option>
                );
              })
            )}
          </select>
          <button
            disabled={!selVer}
            onClick={async () => {
              if (!window.confirm(`모델 ${selVer} 삭제? 예측값도 함께 지워집니다.`)) return;
              const res = await fetch(
                `/api/models?league=${encodeURIComponent(league)}&ver=${encodeURIComponent(selVer)}`,
                { method: "DELETE" }
              ).then((r) => r.json());
              if (res.ok) {
                const delVer = selVer;
                setSelVer("");
                setApplyState("idle");
                setAppliedVer("");
                setPredMap({});
                appliedKeyRef.current = "";
                setAccuracy({});
                delete predCacheRef.current[cacheKey(league, delVer)];
                delete accCacheRef.current[cacheKey(league, delVer)];
                await fetchModels(league);
              } else {
                setError(res.error ?? "모델 삭제 실패");
              }
            }}
            className="mt-2 w-full text-center text-xs text-red-500 underline underline-offset-2 disabled:opacity-40 dark:text-red-400"
          >
            선택 모델 삭제
          </button>
          {modelDetail?.model_type === "permatch" && modelDetail.detail && (() => {
            const d = modelDetail.detail;
            const eff = gridding && gridLive ? gridLive.weights : (tweaked ?? d.weights);
            const orderIdx = wOrder ?? d.features.map((_, i) => i);
            const rows = orderIdx
              .map((i) => ({ name: d.features[i], w: eff?.[i] ?? 0, fi: i }))
              .filter((r) => r.name !== undefined && Number.isFinite(r.w));
            if (rows.length === 0) return null;
            const max = Math.max(...rows.map((r) => Math.abs(r.w)), 1e-9);
            return (
              <div className="mt-2 rounded-xl border border-zinc-200 p-2.5 dark:border-zinc-800">
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="text-xs font-medium text-zinc-500">
                    {gridding && gridLive ? `피처 가중치 · 탐색 중 ${(gridLive.acc * 100).toFixed(1)}%` : "피처 가중치"}
                  </span>
                  <span className="text-[10px] text-zinc-500">± 클릭: 축간격만큼 · 값 직접 입력 가능</span>
                </div>
                <div className="space-y-1">
                  {rows.map((r) => (
                    <div
                      key={r.name}
                      className={`flex items-center gap-1 text-[11px] ${tuning && tuneActive === r.fi ? "rounded bg-green-600/10 dark:bg-green-400/10" : ""}`}
                    >
                      <span className="w-14 shrink-0 truncate text-zinc-600 dark:text-zinc-400" title={r.name}>
                        {FEATURE_KO[r.name] ?? r.name}
                      </span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
                        <div
                          className={`h-full rounded-full ${r.w >= 0 ? "bg-blue-600 dark:bg-blue-400" : "bg-red-500 dark:bg-red-400"}`}
                          style={{ width: `${(Math.abs(r.w) / max) * 100}%` }}
                        />
                      </div>
                      <button
                        onClick={() => nudge(r.fi, -1)}
                        aria-label={`${r.name} 감소`}
                        className="w-5 shrink-0 touch-none rounded border border-zinc-300 text-zinc-600 select-none hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
                      >
                        −
                      </button>
                      <input
                        key={`${r.name}:${r.w.toFixed(4)}`}
                        defaultValue={`${r.w >= 0 ? "+" : ""}${r.w.toFixed(3)}`}
                        onBlur={(e) => setWeight(r.fi, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                        }}
                        aria-label={`${r.name} 직접 입력`}
                        inputMode="decimal"
                        className="w-14 shrink-0 rounded border border-zinc-300 bg-white px-1 py-px text-right font-mono text-zinc-700 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300"
                      />
                      <button
                        onClick={() => nudge(r.fi, 1)}
                        aria-label={`${r.name} 증가`}
                        className="w-5 shrink-0 touch-none rounded border border-zinc-300 text-zinc-600 select-none hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
                      >
                        +
                      </button>
                    </div>
                  ))}
                </div>
                {curOverall != null && (
                  <p className="mt-1 text-[11px] font-medium text-zinc-700 dark:text-zinc-300">
                    예측율 {(curOverall * 100).toFixed(1)}%
                  </p>
                )}
                <div className="mb-1.5 grid grid-cols-2 gap-1">
                  {[
                    { label: "축간격", value: gridStepText, set: setGridStepText, ph: "0.01" },
                    { label: "jobs", value: gridJobsText, set: setGridJobsText, ph: "4" },
                    { label: "묶음", value: gridBatchText, set: setGridBatchText, ph: "4096" },
                    { label: "무가중", value: gridDrawWText, set: setGridDrawWText, ph: "0" },
                    { label: "wmin", value: gridWminText, set: setGridWminText, ph: "-3.0" },
                    { label: "wmax", value: gridWmaxText, set: setGridWmaxText, ph: "3.0" },
                  ].map((f) => (
                    <label key={f.label} className="flex items-center gap-1 text-[11px] text-zinc-500">
                      <span className="shrink-0">{f.label}</span>
                      <input
                        value={f.value}
                        onChange={(e) => f.set(e.target.value)}
                        disabled={gridding}
                        inputMode="decimal"
                        placeholder={f.ph}
                        aria-label={f.label}
                        className="w-full min-w-0 rounded-lg border border-zinc-300 bg-white px-1.5 py-0.5 text-right font-mono text-[11px] text-zinc-700 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300"
                      />
                    </label>
                  ))}
                </div>
                <div className="mt-1.5 flex gap-1.5">
                  <button
                    onClick={autoTune}
                    disabled={gridding}
                    title="미학습 시즌 적중률이 오르면 자동 저장. 다시 누르면 중지"
                    className={`flex-1 rounded-lg px-2 py-1 text-[11px] text-white hover:brightness-110 disabled:opacity-40 dark:text-black ${tuning ? "bg-red-500 dark:bg-red-400" : "bg-green-600 dark:bg-green-500"
                      }`}
                  >
                    {tuning ? "중지" : "랜덤"}
                  </button>
                  <button
                    onClick={gridTune}
                    disabled={tuning}
                    title="전수탐색으로 가중치를 탐색. 다시 누르면 중지"
                    className={`flex-1 rounded-lg px-2 py-1 text-[11px] text-white hover:brightness-110 disabled:opacity-40 dark:text-black ${gridding ? "bg-red-500 dark:bg-red-400" : "bg-purple-600 dark:bg-purple-400"
                      }`}
                  >
                    {gridding ? "중지" : "전수"}
                  </button>
                  <button
                    disabled={!canSave}
                    onClick={saveTweaks}
                    title={saveTitle}
                    className="flex-1 rounded-lg bg-blue-600 px-2 py-1 text-[11px] text-white hover:brightness-110 disabled:opacity-40 dark:bg-blue-400 dark:text-black"
                  >
                    저장
                  </button>
                  <button
                    disabled={!tweaked}
                    onClick={() => {
                      const orig = modelDetail?.detail?.weights;
                      tweakedRef.current = null;
                      setTweaked(null);
                      setWOrder(null);
                      if (orig) previewWith(orig);
                    }}
                    className="flex-1 rounded-lg border border-zinc-300 px-2 py-1 text-[11px] text-zinc-600 hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
                  >
                    원복
                  </button>
                </div>
                {tuneInfo && (
                  <p className="mt-1 text-[11px] text-zinc-500">
                    {tuning ? (tuneInfo.stale >= 5 ? `정체 ${tuneInfo.stale}` : "조절 중") : "조절됨"} {tuneInfo.sweeps}sweep · {tuneInfo.base.toFixed(3)} → {tuneInfo.best.toFixed(3)}
                  </p>
                )}
                {gridInfo && (
                  <div className="mt-1 text-[11px] text-zinc-500">
                    <p>{gridding ? "전수탐색중" : "전수탐색됨"} {gridInfo.total > 0 ? `${Math.min(100, gridInfo.done / gridInfo.total * 100).toFixed(1)}%` : "-"}</p>
                    <p>{gridInfo.done.toLocaleString()}</p>
                    <p>{gridInfo.total.toLocaleString()}</p>
                    <p>best={gridInfo.best != null ? gridInfo.best.toFixed(3) : "-"}</p>
                    <p>탐색중={gridInfo.curAcc != null ? gridInfo.curAcc.toFixed(3) : "-"}</p>
                  </div>
                )}
              </div>
            );
          })()}
          {modelDetail && modelDetail.model_type !== "permatch" && (
            <p className="mt-2 text-[11px] text-zinc-500">가중치 없음 (외부 모델)</p>
          )}
        </div>
      </aside>

      <section className="relative flex min-h-[60vh] min-w-0 flex-1 flex-col gap-3 lg:min-h-0">
        {applying && (
          <div key={league} className="absolute inset-0 z-50 flex items-center justify-center bg-black/40">
            <div className="w-64 rounded-2xl bg-white p-5 shadow-xl dark:bg-zinc-900">
              <div className="mb-3 text-center text-sm font-medium">로딩중...</div>
              <div className="overflow-hidden rounded-full bg-blue-600/15 dark:bg-blue-400/15">
                <div className="loading-bar h-1.5 rounded-full bg-blue-600" />
              </div>
            </div>
          </div>
        )}
        {trainState === "running" && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40">
            <div className="w-64 rounded-2xl bg-white p-5 shadow-xl dark:bg-zinc-900">
              <div className="mb-3 text-center text-sm font-medium">학습중...</div>
              <div className="overflow-hidden rounded-full bg-blue-600/15 dark:bg-blue-400/15">
                <div className="loading-bar h-1.5 rounded-full bg-blue-600" />
              </div>
            </div>
          </div>
        )}
        <div className="grid shrink-0 gap-2 md:grid-cols-2">
          <div className="rounded-xl border border-zinc-200 p-2.5 dark:border-zinc-700">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                상세 그룹 ({detailGroupCols.filter((id) => !hiddenCols.includes(id)).length}/{detailGroupCols.length})
              </span>
              <button
                onClick={() => toggleGroup(detailGroupCols)}
                className="text-[11px] text-zinc-500 underline underline-offset-2"
              >
                {detailGroupCols.every((id) => !hiddenCols.includes(id)) ? "해제" : "전체"}
              </button>
            </div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1.5">
              {FIELD_GROUPS.map((g) => {
                const allVisible = g.cols.every((id) => !hiddenCols.includes(id));
                return (
                  <label
                    key={g.id}
                    className="flex cursor-pointer items-center gap-1.5 text-xs hover:opacity-80"
                  >
                    <input
                      type="checkbox"
                      checked={allVisible}
                      onChange={() => toggleGroup(g.cols)}
                      className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-400"
                    />
                    <span>{g.label}</span>
                  </label>
                );
              })}
              {detailSingles.map((c) => (
                <label
                  key={c.id}
                  className="flex cursor-pointer items-center gap-1.5 text-xs hover:opacity-80"
                >
                  <input
                    type="checkbox"
                    checked={!hiddenCols.includes(c.id)}
                    onChange={() => toggleColumn(c.id)}
                    className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-400"
                  />
                  <span>{c.label}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="rounded-xl border border-blue-600 bg-blue-600/10 p-2.5 dark:border-blue-400 dark:bg-blue-400/10">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-blue-600 dark:text-blue-400">
                예측 그룹 ({predSingles.filter((c) => !hiddenCols.includes(c.id)).length}/{predSingles.length})
              </span>
              <button
                onClick={() => toggleGroup(predGroupCols)}
                className="text-[11px] text-blue-600 underline underline-offset-2 dark:text-blue-400"
              >
                {predGroupCols.every((id) => !hiddenCols.includes(id)) ? "해제" : "전체"}
              </button>
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5">
              {predSingles.map((c) => (
                <label
                  key={c.id}
                  className="flex cursor-pointer items-center gap-1.5 text-xs hover:opacity-80"
                >
                  <input
                    type="checkbox"
                    checked={!hiddenCols.includes(c.id)}
                    onChange={() => toggleColumn(c.id)}
                    className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-400"
                  />
                  <span>{c.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        {error ? (
          <EmptyState title="조회 실패" desc={error} />
        ) : (
          <SoccerMatchesTable
            rows={matches}
            leagueNames={Object.fromEntries(leagues.map((l) => [l.code, l.name_ko]))}
            hidden={hiddenCols}
          />
        )}
      </section>
    </div>
  );
}
