import { NextResponse } from "next/server";
import { getLeagues } from "@/lib/queries";

export async function GET() {
  const result = await getLeagues();
  if (result.error) {
    return NextResponse.json({ ok: false, error: result.error }, { status: 500 });
  }
  return NextResponse.json({ ok: true, data: result.data });
}
