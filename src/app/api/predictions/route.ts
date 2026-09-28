import { NextResponse } from "next/server";
import { getPredictionsMap } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  const seasonsParam = searchParams.get("seasons") ?? "";
  const seasons = seasonsParam.split(",").map((s) => s.trim()).filter(Boolean);
  const modelVer = searchParams.get("model_ver") ?? "";

  if (!league || seasons.length === 0 || !modelVer) {
    return NextResponse.json({ ok: true, league, seasons, data: {} });
  }
  const data = await getPredictionsMap(league, seasons, modelVer);
  return NextResponse.json({ ok: true, league, seasons, data });
}
