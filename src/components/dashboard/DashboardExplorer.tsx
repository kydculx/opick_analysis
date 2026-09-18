"use client";

import { useEffect, useState } from "react";
import type { SoccerMatch, League } from "@/lib/queries";
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

export function DashboardExplorer() {
  const [leagues, setLeagues] = useState<League[]>([]);
  const [league, setLeague] = useState("");
  const [seasons, setSeasons] = useState<string[]>([]);
  const [checked, setChecked] = useState<string[]>([]);
  const [training, setTraining] = useState<string[]>([]);
  const [matches, setMatches] = useState<SoccerMatch[]>([]);
  const [hiddenCols, setHiddenCols] = useState<ColumnId[]>([]);
  const [loadingLeagues, setLoadingLeagues] = useState(true);
  const [loadingSeasons, setLoadingSeasons] = useState(false);
  const [loadingMatches, setLoadingMatches] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainState, setTrainState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [refreshTick, setRefreshTick] = useState(0);
  const [models, setModels] = useState<SavedModel[]>([]);
  const [selVer, setSelVer] = useState("");
  const [accuracy, setAccuracy] = useState<Record<string, { n: number; hit: number; acc: number | null }>>({});
  const [applyState, setApplyState] = useState<"idle" | "running" | "done" | "error">("idle");

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
    setLoadingSeasons(true);
    setChecked([]);
    setTraining([]);
    setTrainState("idle");
    setApplyState("idle");
    setModels([]);
    setSelVer("");
    setAccuracy({});
    fetchModels(league);
    fetch(`/api/accuracy?league=${encodeURIComponent(league)}`)
      .then((r) => r.json())
      .then((j) => {
        if (j.ok) setAccuracy(j.data);
      })
      .catch(() => setAccuracy({}));
    fetch(`/api/seasons?league=${encodeURIComponent(league)}`)
      .then((r) => r.json())
      .then((j) => {
        if (j.ok) {
          setSeasons(j.data);
          setChecked(j.data);
        } else {
          setError(j.error ?? "시즌 조회 실패");
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "시즌 조회 실패"))
      .finally(() => setLoadingSeasons(false));
  }, [league]);

  useEffect(() => {
    if (!league || checked.length === 0) {
      setMatches([]);
      return;
    }
    setLoadingMatches(true);
    const qs = new URLSearchParams({ league, seasons: checked.join(",") });
    fetch(`/api/matches?${qs.toString()}`)
      .then((r) => r.json())
      .then((j) => {
        if (j.ok) {
          setMatches(j.data);
        } else {
          setError(j.error ?? "경기 조회 실패");
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "경기 조회 실패"))
      .finally(() => setLoadingMatches(false));
  }, [league, checked, refreshTick]);

  function accLabel(s: string): string {
    const a = accuracy[s];
    if (!a || a.acc == null) return "";
    return ` (${Math.round(a.acc * 100)}%)`;
  }

  function toggleSeason(s: string) {
    setChecked((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));
    setApplyState("idle");
  }

  function toggleColumn(id: ColumnId) {
    setHiddenCols((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  function moveToTraining(s: string) {
    setTraining((prev) => (prev.includes(s) ? prev : [...prev, s]));
    setChecked((prev) => prev.filter((x) => x !== s));
    setTrainState("idle");
  }

  function moveToView(s: string) {
    setTraining((prev) => prev.filter((x) => x !== s));
    setTrainState("idle");
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

  async function fetchModels(lg: string) {
    if (!lg) return;
    try {
      const j = await fetch(`/api/models?league=${encodeURIComponent(lg)}`).then((r) => r.json());
      if (j.ok) {
        setModels(j.data);
        setSelVer((prev) => (j.data.some((m: SavedModel) => m.ver === prev) ? prev : (j.data[0]?.ver ?? "")));
      }
    } catch {
      setModels([]);
    }
  }

  async function handleTrain() {
    if (trainingOrdered.length === 0 || trainState === "running") return;
    setTrainState("running");
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ league, seasons: trainingOrdered }),
      }).then((r) => r.json());
      if (!res.ok) {
        setTrainState("error");
        return;
      }
      const done = await pollJob("/api/train", res.jobId as string);
      setTrainState(done);
      if (done === "done") {
        await fetchModels(league);
        setSelVer(res.ver as string);
      }
    } catch {
      setTrainState("error");
    }
  }

  async function handleApply() {
    if (!selVer || checked.length === 0 || applyState === "running") return;
    setApplyState("running");
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ league, ver: selVer, seasons: checked }),
      }).then((r) => r.json());
      if (!res.ok) {
        setApplyState("error");
        return;
      }
      const done = await pollJob("/api/predict", res.jobId as string);
      setApplyState(done);
      if (done === "done") setRefreshTick((t) => t + 1);
    } catch {
      setApplyState("error");
    }
  }

  const viewSeasons = seasons.filter((s) => !training.includes(s));
  const toggleableCols = COLUMNS.filter((c) => !CORE_COLUMNS.includes(c.id));
  const visibleToggleable = toggleableCols.filter((c) => !hiddenCols.includes(c.id)).length;

  function toggleGroup(cols: ColumnId[]) {
    const allVisible = cols.every((id) => !hiddenCols.includes(id));
    setHiddenCols((prev) =>
      allVisible ? [...prev, ...cols.filter((id) => !prev.includes(id))] : prev.filter((id) => !cols.includes(id))
    );
  }

  const singleCols = COLUMNS.filter((c) => SINGLE_COLUMNS.includes(c.id));
  const trainingOrdered = seasons.filter((s) => training.includes(s));

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden lg:flex-row">
      <aside className="w-full shrink-0 space-y-4 rounded-2xl border border-zinc-200 p-4 dark:border-zinc-800 lg:max-h-full lg:w-64 lg:overflow-y-auto">
        <div>
          <label className="text-xs font-medium text-zinc-500">리그 선택</label>
          <select
            value={league}
            onChange={(e) => setLeague(e.target.value)}
            disabled={loadingLeagues}
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
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-500">
              조회시즌 ({checked.length}/{viewSeasons.length})
            </span>
            <div className="flex gap-2 text-xs">
              <button onClick={() => setChecked(viewSeasons)} className="underline underline-offset-2">
                전체
              </button>
              <button onClick={() => setChecked([])} className="underline underline-offset-2">
                해제
              </button>
            </div>
          </div>
          <div className="mt-2 max-h-72 space-y-1 overflow-y-auto">
            {loadingSeasons ? (
              <p className="text-sm text-zinc-500">시즌 불러오는 중...</p>
            ) : viewSeasons.length === 0 ? (
              <p className="px-2 py-1.5 text-sm text-zinc-500">조회할 시즌이 없습니다.</p>
            ) : (
              viewSeasons.map((s) => (
                <div
                  key={s}
                  className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900"
                >
                  <label className="flex flex-1 cursor-pointer items-center gap-2">
                    <input
                      type="checkbox"
                      checked={checked.includes(s)}
                      onChange={() => toggleSeason(s)}
                      className="h-4 w-4 accent-zinc-900 dark:accent-white"
                    />
                    <span className="font-mono">
                      {s}
                      {accLabel(s)}
                    </span>
                  </label>
                  <button
                    onClick={() => moveToTraining(s)}
                    className="rounded-full border border-zinc-200 px-2 py-0.5 text-[11px] text-zinc-500 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    학습
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <hr className="border-zinc-200 dark:border-zinc-800" />

        <div>
          <span className="text-xs font-medium text-zinc-500">학습시즌 ({trainingOrdered.length})</span>
          <div className="mt-2 max-h-72 space-y-1 overflow-y-auto">
            {trainingOrdered.length === 0 ? (
              <p className="px-2 py-1.5 text-sm text-zinc-500">학습 버튼으로 시즌을 옮기세요.</p>
            ) : (
              trainingOrdered.map((s) => (
                <div
                  key={s}
                  className="flex items-center gap-2 rounded-lg bg-zinc-50 px-2 py-1.5 text-sm dark:bg-zinc-900"
                >
                  <span className="flex-1 font-mono">
                    {s}
                    {accLabel(s)}
                  </span>
                  <button
                    onClick={() => moveToView(s)}
                    aria-label={`${s} 조회로 이동`}
                    className="rounded-full border border-zinc-200 px-2 py-0.5 text-[11px] text-zinc-500 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>
          <Button
            size="sm"
            disabled={trainingOrdered.length === 0 || trainState === "running"}
            onClick={handleTrain}
            className="mt-3 w-full"
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
          <span className="text-xs font-medium text-zinc-500">모델 적용 ({models.length})</span>
          <select
            value={selVer}
            onChange={(e) => {
              setSelVer(e.target.value);
              setApplyState("idle");
            }}
            disabled={models.length === 0}
            className="mt-1 w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-950"
          >
            {models.length === 0 ? (
              <option>학습된 모델 없음</option>
            ) : (
              models.map((m) => (
                <option key={m.ver} value={m.ver}>
                  {m.ver.replace("_", " ")} · {m.train_seasons.join(",")}
                  {m.metrics?.valid_acc != null ? ` · acc ${m.metrics.valid_acc.toFixed(3)}` : ""}
                </option>
              ))
            )}
          </select>
          <Button
            size="sm"
            variant="secondary"
            disabled={!selVer || checked.length === 0 || applyState === "running"}
            onClick={handleApply}
            className="mt-3 w-full"
          >
            {applyState === "running"
              ? "적용 중..."
              : applyState === "done"
                ? "적용 완료"
                : applyState === "error"
                  ? "실패 · 다시 시도"
                  : `조회시즌에 적용${checked.length > 0 ? ` (${checked.length})` : ""}`}
          </Button>
          <button
            disabled={!selVer}
            onClick={async () => {
              if (!window.confirm(`모델 ${selVer} 삭제? 예측값도 함께 지워집니다.`)) return;
              const res = await fetch(
                `/api/models?league=${encodeURIComponent(league)}&ver=${encodeURIComponent(selVer)}`,
                { method: "DELETE" }
              ).then((r) => r.json());
              if (res.ok) {
                setSelVer("");
                setApplyState("idle");
                await fetchModels(league);
                setRefreshTick((t) => t + 1);
              }
            }}
            className="mt-2 w-full text-center text-xs text-red-500 underline underline-offset-2 disabled:opacity-40"
          >
            선택 모델 삭제
          </button>
        </div>
      </aside>

      <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
        <div className="shrink-0 rounded-2xl border border-zinc-200 p-3 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-500">
              표시 필드 ({visibleToggleable}/{toggleableCols.length})
            </span>
            <div className="flex gap-2 text-xs">
              <button onClick={() => setHiddenCols([])} className="underline underline-offset-2">
                전체
              </button>
              <button
                onClick={() => setHiddenCols(toggleableCols.map((c) => c.id))}
                className="underline underline-offset-2"
              >
                해제
              </button>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5">
            {FIELD_GROUPS.map((g) => {
              const allVisible = g.cols.every((id) => !hiddenCols.includes(id));
              return (
                <label
                  key={g.id}
                  className="flex cursor-pointer items-center gap-1.5 rounded-full border border-zinc-200 px-2.5 py-1 text-xs font-medium hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
                >
                  <input
                    type="checkbox"
                    checked={allVisible}
                    onChange={() => toggleGroup(g.cols)}
                    className="h-3.5 w-3.5 accent-zinc-900 dark:accent-white"
                  />
                  <span>{g.label}</span>
                </label>
              );
            })}
            {singleCols.map((c) => (
              <label
                key={c.id}
                className="flex cursor-pointer items-center gap-1.5 text-xs hover:opacity-80"
              >
                <input
                  type="checkbox"
                  checked={!hiddenCols.includes(c.id)}
                  onChange={() => toggleColumn(c.id)}
                  className="h-3.5 w-3.5 accent-zinc-900 dark:accent-white"
                />
                <span>{c.label}</span>
              </label>
            ))}
          </div>
        </div>
        {error ? (
          <EmptyState title="조회 실패" desc={error} />
        ) : checked.length === 0 ? (
          <EmptyState title="시즌을 선택하세요" desc="왼쪽에서 시즌을 1개 이상 체크하면 경기가 표시됩니다." />
        ) : loadingMatches && matches.length === 0 ? (
          <EmptyState title="불러오는 중..." />
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
