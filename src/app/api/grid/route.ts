import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { isJobId, killJob, readJob, startJob } from "@/lib/jobs";

function gridVer(ver: string, five: boolean) {
  return five && !ver.endsWith("-f5") ? `${ver}-f5` : ver;
}

function readProgress(league: string, ver: string) {
  try {
    const p = join(process.cwd(), "ml", "permatch", `${league}_${ver}.grid.json`);
    if (!existsSync(p)) return null;
    const c = JSON.parse(readFileSync(p, "utf-8")) as {
      done?: unknown;
      total?: unknown;
      best?: unknown;
      sweep?: unknown;
      curW?: unknown;
      curHfa?: unknown;
      curAcc?: unknown;
    };
    if (typeof c.done !== "number" || typeof c.total !== "number") return null;
    return {
      done: c.done,
      total: c.total,
      best: typeof c.best === "number" ? c.best : null,
      sweep: typeof c.sweep === "number" ? c.sweep : null,
      curW: Array.isArray(c.curW) && c.curW.every((v) => typeof v === "number") ? (c.curW as number[]) : null,
      curHfa: typeof c.curHfa === "number" ? c.curHfa : null,
      curAcc: typeof c.curAcc === "number" ? c.curAcc : null,
    };
  } catch {
    return null;
  }
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as {
    league?: unknown;
    ver?: unknown;
    tune?: unknown;
    grid_step?: unknown;
    five?: unknown;
    jobs?: unknown;
    batch?: unknown;
    wmin?: unknown;
    wmax?: unknown;
    draw_w?: unknown;
    recency_w?: unknown;
    max_combos?: unknown;
    max_minutes?: unknown;
  };
  const league = typeof body.league === "string" ? body.league : "";
  const ver = typeof body.ver === "string" ? body.ver : "";
  const tune = Array.isArray(body.tune) ? body.tune.map((s) => String(s)).filter(Boolean) : [];
  if (!league || !ver || tune.length === 0) {
    return NextResponse.json({ ok: false, error: "league, ver, tune이 필요" }, { status: 400 });
  }
  const five = body.five === true;
  const argv = ["--mode", "grid", "--league", league, "--ver", ver, "--tune", tune.join(",")];
  try {
    const art = JSON.parse(
      readFileSync(join(process.cwd(), "ml", "permatch", `${league}_${gridVer(ver, five)}.json`), "utf-8")
    ) as { features?: unknown };
    if (Array.isArray(art.features) && art.features.length > 0) {
      const names = (art.features as unknown[]).filter(
        (f): f is string => typeof f === "string" && f.length > 0
      );
      if (names.length > 0 && names.length !== 13) argv.push("--features", names.join(","));
    }
  } catch {
  }
  const step = Number(body.grid_step);
  if (Number.isFinite(step) && step > 0) argv.push("--grid-step", String(step));
  if (five) argv.push("--five");
  const jobs = Math.floor(Number(body.jobs));
  argv.push("--jobs", String(Number.isFinite(jobs) && jobs > 0 ? jobs : 4));
  const batch = Math.floor(Number(body.batch));
  if (Number.isFinite(batch) && batch > 0) argv.push("--batch", String(batch));
  const wmin = Number(body.wmin);
  const wmax = Number(body.wmax);
  if (Number.isFinite(wmin) && Number.isFinite(wmax) && wmin < wmax) {
    argv.push("--wmin", String(wmin), "--wmax", String(wmax));
  }
  const drawW = Number(body.draw_w);
  if (Number.isFinite(drawW) && drawW !== 0) argv.push("--draw-w", String(drawW));
  const recW = Number(body.recency_w);
  if (Number.isFinite(recW) && recW !== 0) argv.push("--recency-w", String(recW));
  const maxCombos = Math.floor(Number(body.max_combos));
  if (Number.isFinite(maxCombos) && maxCombos > 0) argv.push("--max-combos", String(maxCombos));
  const maxMinutes = Number(body.max_minutes);
  if (Number.isFinite(maxMinutes) && maxMinutes > 0) argv.push("--max-minutes", String(maxMinutes));
  const job = startJob({
    kind: "train",
    league,
    seasons: tune,
    ver,
    script: "ml/permatch_mode.py",
    argv,
  });
  return NextResponse.json({ ok: true, jobId: job.id });
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get("jobId") ?? "";
  if (!isJobId(id)) {
    return NextResponse.json({ ok: false, error: "jobId가 필요" }, { status: 400 });
  }
  const job = readJob(id);
  if (!job) {
    return NextResponse.json({ ok: false, error: "작업을 찾을 수 없음" }, { status: 404 });
  }
  const league = searchParams.get("league") ?? job.league;
  const ver = searchParams.get("ver") ?? job.ver;
  const five = searchParams.get("five") === "1";
  const progress = readProgress(league, gridVer(ver, five));
  return NextResponse.json({ ok: true, job, progress });
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get("jobId") ?? "";
  if (!isJobId(id)) {
    return NextResponse.json({ ok: false, error: "jobId가 필요" }, { status: 400 });
  }
  const stopped = killJob(id);
  return NextResponse.json({ ok: stopped });
}

export const dynamic = "force-dynamic";
