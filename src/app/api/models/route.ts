import { NextResponse } from "next/server";
import { existsSync, readdirSync, rmSync } from "node:fs";
import { createServerSupabaseClient } from "@/lib/supabase/server";

function diskVersions(league: string): { ver: string; model_type: string }[] {
  const dir = `${process.cwd()}/ml/models`;
  if (!existsSync(dir)) return [];
  const prefix = `${league}_`;
  return readdirSync(dir)
    .filter((f) => f.startsWith(prefix) && (f.endsWith(".pkl") || f.endsWith(".txt")))
    .map((f) => ({
      ver: f.slice(prefix.length, -4),
      model_type: f.endsWith(".pkl") ? "logreg" : "gbm",
    }));
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  if (!league) {
    return NextResponse.json({ ok: false, error: "league가 필요" }, { status: 400 });
  }
  try {
    const supabase = createServerSupabaseClient();
    const { data, error } = await supabase
      .from("models")
      .select("ver,train_seasons,model_type,metrics,created_at")
      .eq("league_code", league)
      .order("created_at", { ascending: false })
      .limit(50);
    const registered = error ? [] : (data ?? []);
    const known = new Set(registered.map((r) => r.ver as string));
    const onDisk = diskVersions(league)
      .filter((d) => !known.has(d.ver))
      .map((d) => ({ ...d, train_seasons: [], metrics: null, created_at: "" }));
    for (const d of onDisk) known.add(d.ver);
    const { data: predVers } = await supabase
      .from("predictions")
      .select("model_ver")
      .eq("league_code", league)
      .limit(20000);
    const fallback = [...new Set((predVers ?? []).map((r) => r.model_ver as string))]
      .filter((ver) => ver && !known.has(ver))
      .sort()
      .reverse()
      .map((ver) => ({ ver, train_seasons: [], model_type: "logreg", metrics: null, created_at: "" }));
    return NextResponse.json({ ok: true, data: [...registered, ...onDisk, ...fallback] });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "Unknown error" },
      { status: 500 }
    );
  }
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  const ver = searchParams.get("ver") ?? "";
  if (!league || !/^[A-Za-z0-9_-]+$/.test(ver)) {
    return NextResponse.json({ ok: false, error: "league와 ver가 필요" }, { status: 400 });
  }
  try {
    const supabase = createServerSupabaseClient();
    const { error: predErr } = await supabase
      .from("predictions")
      .delete()
      .eq("league_code", league)
      .eq("model_ver", ver);
    if (predErr) return NextResponse.json({ ok: false, error: predErr.message }, { status: 500 });
    await supabase.from("models").delete().eq("ver", ver);
    const base = `${process.cwd()}/ml/models/${league}_${ver}`;
    for (const ext of ["pkl", "txt"]) rmSync(`${base}.${ext}`, { force: true });
    return NextResponse.json({ ok: true, ver });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "Unknown error" },
      { status: 500 }
    );
  }
}
