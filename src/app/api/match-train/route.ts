import { NextResponse } from "next/server";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { JOB_DIR, isJobId, readJob, startJob } from "@/lib/jobs";
import { ALL_FEATURES } from "@/lib/permatch-math";

// GUI ml/grid_console.py make_ver와 동일 규칙: tr{시즌수}-f{피처수}-d{무가중}-x{해시4}-{년월일시분}
function makeVer(trainCsv: string, feats: string[], drawW = "0", ts = "", recencyW = "0"): string {
  const dw = Number.parseFloat(String(drawW).trim() || "0") || 0;
  const rw = Number.parseFloat(String(recencyW).trim() || "0") || 0;
  const key = `${trainCsv}|${feats.join(",")}|${dw === 0 ? "0" : String(dw)}|r${rw}`;
  const h = createHash("sha1").update(key).digest("hex").slice(0, 4);
  const tag = String(dw).replace(".", "_");
  const base = `tr${trainCsv.split(",").filter((s) => s.trim()).length}-f${feats.length}-d${tag}-x${h}${rw !== 0 ? `-r${String(rw).replace(".", "_")}` : ""}`;
  return ts ? `${base}-${ts}` : base;
}

function nowTag(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getFullYear() % 100)}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}`;
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as {
    league?: unknown; seasons?: unknown; features?: unknown;
    mode?: unknown; trials?: unknown; jobs?: unknown; drawW?: unknown; valid?: unknown; recencyW?: unknown;
  };
  const league = typeof body.league === "string" ? body.league : "";
  const seasons = Array.isArray(body.seasons) ? body.seasons.map((s) => String(s)).filter(Boolean) : [];
  const features = Array.isArray(body.features)
    ? body.features.map((f) => String(f)).filter((f) => (ALL_FEATURES as string[]).includes(f))
    : [...ALL_FEATURES];
  if (features.length === 0) {
    return NextResponse.json({ ok: false, error: "피처를 1개 이상 선택하세요" }, { status: 400 });
  }
  const full = typeof body.mode === "string" && body.mode === "full";
  const trials = Math.max(1, Math.min(200000, Number.parseInt(String(body.trials ?? "10000"), 10) || 10000));
  const jobs = Math.max(1, Math.min(16, Number.parseInt(String(body.jobs ?? "4"), 10) || 4));
  const drawW = Number.parseFloat(String(body.drawW ?? "0")) || 0;
  const recencyW = Number.parseFloat(String(body.recencyW ?? "0")) || 0;
  const valid = Array.isArray(body.valid)
    ? body.valid.map((s) => String(s)).filter(Boolean)
    : "auto";
  const sorted = [...new Set(seasons)].sort() as string[];
  const feats = features.length === ALL_FEATURES.length ? [...ALL_FEATURES] : [...features];
  const ver = makeVer(sorted.join(","), feats, String(drawW), nowTag(), String(recencyW));
  const recArg = recencyW !== 0 ? ["--recency-w", String(recencyW)] : [];
  const argv = full
    ? [
        "--league", league,
        "--train", sorted.join(","),
        "--valid", Array.isArray(valid) ? valid.join(",") : valid,
        "--ver", ver,
        "--trials", String(trials),
        "--jobs", String(jobs),
        "--auto-ensemble",
        ...(drawW ? ["--draw-w", String(drawW)] : []),
        ...recArg,
        ...(feats.length !== ALL_FEATURES.length ? ["--features", feats.join(",")] : []),
      ]
    : [
        "--league", league,
        "--train", sorted.join(","),
        "--valid", "auto",
        "--ver", ver,
        "--fast",
        "--auto-ensemble",
        ...recArg,
        ...(feats.length !== ALL_FEATURES.length ? ["--features", feats.join(",")] : []),
      ];
  const job = startJob({
    kind: "train",
    league,
    seasons: sorted,
    ver,
    script: "ml/permatch_mode.py",
    argv,
  });
  return NextResponse.json({ ok: true, jobId: job.id, ver: `${ver}-ens` });
}

export const dynamic = "force-dynamic";

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
  let progress: { done: number; total: number } | null = null;
  let tail: string[] = [];
  try {
    const lines = readFileSync(join(JOB_DIR, `${id}.log`), "utf-8").split("\n").filter((l) => l.trim());
    tail = lines.slice(-3);
    for (let i = lines.length - 1; i >= 0; i--) {
      const m = lines[i].match(/계산\s+([\d,]+)\/([\d,]+)/) ?? lines[i].match(/\(([\d,]+)\/([\d,]+)\)/);
      if (m) {
        progress = {
          done: Number(m[1].replace(/,/g, "")),
          total: Number(m[2].replace(/,/g, "")),
        };
        break;
      }
      if (lines.length - i > 4000) break;
    }
  } catch {
    progress = null;
  }
  return NextResponse.json({ ok: true, job, progress, tail });
}
