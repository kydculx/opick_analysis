import { NextResponse } from "next/server";
import { createHash } from "node:crypto";
import { isJobId, readJob, startJob } from "@/lib/jobs";
import { ALL_FEATURES } from "@/lib/permatch-math";

// GUI ml/grid_console.py make_ver와 동일 규칙: tr{시즌수}-f{피처수}-d{무가중}-x{해시4}-{년월일시분}
function makeVer(trainCsv: string, feats: string[], drawW = "0", ts = ""): string {
  const dw = Number.parseFloat(String(drawW).trim() || "0") || 0;
  const key = `${trainCsv}|${feats.join(",")}|${dw === 0 ? "0" : String(dw)}`;
  const h = createHash("sha1").update(key).digest("hex").slice(0, 4);
  const tag = String(dw).replace(".", "_");
  const base = `tr${trainCsv.split(",").filter((s) => s.trim()).length}-f${feats.length}-d${tag}-x${h}`;
  return ts ? `${base}-${ts}` : base;
}

function nowTag(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getFullYear() % 100)}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}`;
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { league?: unknown; seasons?: unknown; features?: unknown };
  const league = typeof body.league === "string" ? body.league : "";
  const seasons = Array.isArray(body.seasons) ? body.seasons.map((s) => String(s)).filter(Boolean) : [];
  const features = Array.isArray(body.features)
    ? body.features.map((f) => String(f)).filter((f) => (ALL_FEATURES as string[]).includes(f))
    : [...ALL_FEATURES];
  if (features.length === 0) {
    return NextResponse.json({ ok: false, error: "피처를 1개 이상 선택하세요" }, { status: 400 });
  }
  const sorted = [...new Set(seasons)].sort() as string[];
  const feats = features.length === ALL_FEATURES.length ? [...ALL_FEATURES] : [...features];
  const ver = makeVer(sorted.join(","), feats, "0", nowTag());
  const job = startJob({
    kind: "train",
    league,
    seasons: sorted,
    ver,
    script: "ml/permatch_mode.py",
    argv: [
      "--league", league,
      "--train", sorted.join(","),
      "--valid", "auto",
      "--ver", ver,
      "--fast",
      ...(feats.length !== ALL_FEATURES.length ? ["--features", feats.join(",")] : []),
    ],
  });
  return NextResponse.json({ ok: true, jobId: job.id, ver });
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
  return NextResponse.json({ ok: true, job });
}
