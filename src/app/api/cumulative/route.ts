import { NextResponse } from "next/server";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ALL_FEATURES } from "@/lib/permatch-math";
import { JOB_DIR, isJobId, readJob, startJob } from "@/lib/jobs";

export const dynamic = "force-dynamic";

function nowTag(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getFullYear() % 100)}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}`;
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as {
    league?: unknown; seasons?: unknown; base?: unknown; lr?: unknown; ver?: unknown; features?: unknown;
  };
  const league = typeof body.league === "string" ? body.league : "";
  const seasons = Array.isArray(body.seasons) ? body.seasons.map((s) => String(s)).filter(Boolean) : [];
  const base = typeof body.base === "string" ? body.base : "";
  const lr = Math.max(0, Math.min(1, Number.parseFloat(String(body.lr ?? "0.01")) || 0.01));
  const features = Array.isArray(body.features)
    ? body.features.map((f) => String(f)).filter((f) => (ALL_FEATURES as string[]).includes(f))
    : [...ALL_FEATURES];
  if (!/^[A-Za-z0-9_-]+$/.test(league) || seasons.length === 0 || (base && !/^[A-Za-z0-9_.-]+$/.test(base))) {
    return NextResponse.json({ ok: false, error: "league·seasons가 필요" }, { status: 400 });
  }
  if (features.length === 0) {
    return NextResponse.json({ ok: false, error: "피처를 1개 이상 선택하세요" }, { status: 400 });
  }
  const sorted = [...new Set(seasons)].sort() as string[];
  const feats = features.length === ALL_FEATURES.length ? [...ALL_FEATURES] : [...features];
  let resolvedBase = base;
  if (resolvedBase) {
    try {
      const bp = join(process.cwd(), "ml", "permatch", `${league}_${resolvedBase}.json`);
      const ba = JSON.parse(readFileSync(bp, "utf-8"));
      if (ba?.model_type === "ensemble" && typeof ba.base_ver === "string" && ba.base_ver) {
        resolvedBase = ba.base_ver;
      }
    } catch {
      resolvedBase = base;
    }
  }
  const ver =
    typeof body.ver === "string" && /^[A-Za-z0-9_.-]+$/.test(body.ver)
      ? body.ver
      : `${resolvedBase || league}-cumu-${nowTag()}`;
  const argv = [
    "--league", league,
    "--seasons", sorted.join(","),
    "--ver", ver,
    "--lr", String(lr),
  ];
  if (resolvedBase) argv.push("--base", resolvedBase);
  if (feats.length !== ALL_FEATURES.length) argv.push("--features", feats.join(","));
  const job = startJob({
    kind: "train",
    league,
    seasons: sorted,
    ver,
    script: "ml/cumulative.py",
    argv,
  });
  return NextResponse.json({ ok: true, jobId: job.id, ver, base: resolvedBase });
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
  let progress: { done: number; total: number } | null = null;
  let tail: string[] = [];
  try {
    const lines = readFileSync(join(JOB_DIR, `${id}.log`), "utf-8").split("\n").filter((l) => l.trim());
    tail = lines.slice(-3);
    for (let i = lines.length - 1; i >= 0; i--) {
      const m = lines[i].match(/누적\s+([\d,]+)\/([\d,]+)/);
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
