import { NextResponse } from "next/server";
import { existsSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServerSupabaseClient } from "@/lib/supabase/server";

function diskVersions(league: string): { ver: string; model_type: string }[] {
  const dir = `${process.cwd()}/ml/permatch`;
  if (!existsSync(dir)) return [];
  const prefix = `${league}_`;
  const out: { ver: string; model_type: string }[] = [];
  for (const f of readdirSync(dir)) {
    if (!f.startsWith(prefix) || !f.endsWith(".json")) continue;
    if (f.includes(".grid")) continue;
    out.push({ ver: f.slice(prefix.length, -5), model_type: "permatch" });
  }
  return out;
}

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const league = searchParams.get("league") ?? "";
  const ver = searchParams.get("ver") ?? "";
  if (!league) {
    return NextResponse.json({ ok: false, error: "league가 필요" }, { status: 400 });
  }
  if (ver) {
    if (!/^[A-Za-z0-9_-]+$/.test(league) || !/^[A-Za-z0-9_.-]+$/.test(ver)) {
      return NextResponse.json({ ok: false, error: "league와 ver가 필요" }, { status: 400 });
    }
    try {
      const p = `${process.cwd()}/ml/permatch/${league}_${ver}.json`;
      if (!existsSync(p)) {
        return NextResponse.json({ ok: true, league, ver, model_type: "unknown", detail: null });
      }
      const a = JSON.parse(readFileSync(p, "utf-8"));
      const feats: string[] = Array.isArray(a.features) ? a.features : [];
      const weights: number[] = Array.isArray(a.weights) ? a.weights : [];
      const emphasis: number[] | null = Array.isArray(a.emphasis) ? a.emphasis : null;
      return NextResponse.json({
        ok: true,
        league,
        ver,
        model_type: "permatch",
        detail: {
          features: feats,
          weights,
          emphasis,
          hfa: a.hfa ?? null,
          T: a.T ?? null,
          draw_prior: a.draw_prior ?? null,
          mu: Array.isArray(a.mu) ? a.mu : null,
          sd: Array.isArray(a.sd) ? a.sd : null,
          draw_weights: Array.isArray(a.draw_weights) ? a.draw_weights : null,
          draw_bias: a.draw_bias ?? 0,
          draw_mu: Array.isArray(a.draw_mu) ? a.draw_mu : null,
          draw_sd: Array.isArray(a.draw_sd) ? a.draw_sd : null,
          patterns: a.patterns ?? null,
          contrib_cap: a.contrib_cap ?? null,
          train_seasons: a.train_seasons ?? [],
          valid: a.valid ?? null,
          trials: a.trials ?? null,
          pattern_tau: a.pattern_tau ?? null,
          pattern_stats: a.pattern_stats ?? null,
        },
      });
    } catch (e) {
      return NextResponse.json(
        { ok: false, error: e instanceof Error ? e.message : "Unknown error" },
        { status: 500 }
      );
    }
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
    const all = [...registered, ...onDisk];
    all.sort(
      (a, b) =>
        String((b as { created_at?: string }).created_at ?? "").localeCompare(
          String((a as { created_at?: string }).created_at ?? "")
        ) || String((b as { ver?: string }).ver ?? "").localeCompare(String((a as { ver?: string }).ver ?? ""))
    );
    return NextResponse.json({ ok: true, data: all });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "Unknown error" },
      { status: 500 }
    );
  }
}

export async function POST(req: Request) {
  let body: { league?: unknown; ver?: unknown; weights?: unknown; hfa?: unknown };
  try {
    body = (await req.json()) as { league?: unknown; ver?: unknown; weights?: unknown; hfa?: unknown };
  } catch {
    return NextResponse.json({ ok: false, error: "JSON 파싱 실패" }, { status: 400 });
  }
  const league = typeof body.league === "string" ? body.league : "";
  const ver = typeof body.ver === "string" ? body.ver : "";
  const weights = body.weights;
  const hfa = body.hfa;
  if (!/^[A-Za-z0-9_-]+$/.test(league) || !/^[A-Za-z0-9_.-]+$/.test(ver)) {
    return NextResponse.json({ ok: false, error: "league와 ver가 필요" }, { status: 400 });
  }
  if (!Array.isArray(weights) || weights.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
    return NextResponse.json({ ok: false, error: "weights가 필요" }, { status: 400 });
  }
  if (hfa !== undefined && (typeof hfa !== "number" || !Number.isFinite(hfa))) {
    return NextResponse.json({ ok: false, error: "hfa 형식 오류" }, { status: 400 });
  }
  try {
    const p = `${process.cwd()}/ml/permatch/${league}_${ver}.json`;
    if (!existsSync(p)) {
      return NextResponse.json({ ok: false, error: "모델 파일 없음" }, { status: 404 });
    }
    const a = JSON.parse(readFileSync(p, "utf-8"));
    if (!Array.isArray(a.weights) || a.weights.length !== weights.length) {
      return NextResponse.json({ ok: false, error: "가중치 개수 불일치" }, { status: 400 });
    }
    a.weights = weights;
    if (hfa !== undefined) a.hfa = hfa;
    writeFileSync(p, JSON.stringify(a));
    return NextResponse.json({ ok: true, league, ver });
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
    const base = `${process.cwd()}/ml/permatch/${league}_${ver}`;
    for (const ext of ["pkl", "txt", "json"]) rmSync(`${base}.${ext}`, { force: true });
    return NextResponse.json({ ok: true, ver });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "Unknown error" },
      { status: 500 }
    );
  }
}
