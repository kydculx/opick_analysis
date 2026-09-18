import { NextResponse } from "next/server";
import { getMatchesBySeasons } from "@/lib/queries";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  const seasonsParam = searchParams.get("seasons") ?? "";
  const seasons = seasonsParam.split(",").map((s) => s.trim()).filter(Boolean);

  if (!league || seasons.length === 0) {
    return NextResponse.json({ ok: true, league, seasons, data: [] });
  }
  const result = await getMatchesBySeasons(league, seasons);
  if (result.error) {
    return NextResponse.json({ ok: false, error: result.error }, { status: 500 });
  }
  return NextResponse.json({ ok: true, league, seasons, data: result.data });
}
