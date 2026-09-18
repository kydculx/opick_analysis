import { NextResponse } from "next/server";
import { isJobId, readJob, startJob } from "@/lib/jobs";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { league?: unknown; ver?: unknown; seasons?: unknown };
  const league = typeof body.league === "string" ? body.league : "";
  const ver = typeof body.ver === "string" ? body.ver : "";
  const seasons = Array.isArray(body.seasons) ? body.seasons.map((s) => String(s)).filter(Boolean) : [];
  if (!league || !ver || seasons.length === 0) {
    return NextResponse.json({ ok: false, error: "league, ver, seasons가 필요" }, { status: 400 });
  }
  const sorted = [...new Set(seasons)].sort() as string[];
  const job = startJob({
    kind: "predict",
    league,
    seasons: sorted,
    ver,
    argv: [
      "--mode", "predict",
      "--league", league,
      "--load-ver", ver,
      "--predict", sorted.join(","),
      "--model", "logreg",
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
