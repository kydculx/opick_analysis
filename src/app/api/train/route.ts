import { NextResponse } from "next/server";
import { isJobId, readJob, startJob } from "@/lib/jobs";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { league?: unknown; seasons?: unknown };
  const league = typeof body.league === "string" ? body.league : "";
  const seasons = Array.isArray(body.seasons) ? body.seasons.map((s) => String(s)).filter(Boolean) : [];
  if (!league || seasons.length === 0) {
    return NextResponse.json({ ok: false, error: "league와 seasons가 필요" }, { status: 400 });
  }
  const now = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const ver = `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}_${p(now.getHours())}-${p(now.getMinutes())}-${p(now.getSeconds())}`;
  const sorted = [...new Set(seasons)].sort() as string[];
  const last = sorted[sorted.length - 1];
  const job = startJob({
    kind: "train",
    league,
    seasons: sorted,
    ver,
    argv: [
      "--mode", "train",
      "--league", league,
      "--train", sorted.join(","),
      "--valid", last,
      "--model", "logreg",
      "--ver", ver,
    ],
  });
  return NextResponse.json({ ok: true, jobId: job.id, ver });
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
  return NextResponse.json({ ok: true, job });
}
