import { NextResponse } from "next/server";
import { getSeasonAccuracy } from "@/lib/queries";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  if (!league) {
    return NextResponse.json({ ok: false, error: "league가 필요" }, { status: 400 });
  }
  const data = await getSeasonAccuracy(league);
  return NextResponse.json({ ok: true, league, data });
}
