import { NextResponse } from "next/server";
import { getSeasons } from "@/lib/queries";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  if (!league) {
    return NextResponse.json({ ok: false, error: "league required" }, { status: 400 });
  }
  const result = await getSeasons(league);
  if (result.error) {
    return NextResponse.json({ ok: false, error: result.error }, { status: 500 });
  }
  return NextResponse.json({ ok: true, league, data: result.data });
}
