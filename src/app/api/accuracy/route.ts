import { NextResponse } from "next/server";
import { getSeasonAccuracy } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  const modelVer = searchParams.get("model_ver") ?? "";
  const seasons = (searchParams.get("seasons") ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  if (!league) {
    return NextResponse.json({ ok: false, error: "league가 필요" }, { status: 400 });
  }
  const data = await getSeasonAccuracy(league, modelVer, seasons);
  return NextResponse.json({ ok: true, league, modelVer, data });
}
