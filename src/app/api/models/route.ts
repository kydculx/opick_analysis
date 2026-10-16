import { NextResponse } from "next/server";
import { existsSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServerSupabaseClient } from "@/lib/supabase/server";

const PROG_DIR = join(tmpdir(), "opick_prog");
const LOCAL_DIR = join(process.cwd(), "ml", "permatch");

function cleanupFiles(league: string, ver: string): string[] {
  const prefixes = [`${league}_${ver}.`, `${league}_${ver}-f5.`];
  const deleted: string[] = [];
  for (const dir of [PROG_DIR, LOCAL_DIR]) {
    if (!existsSync(dir)) continue;
    let files: string[];
    try {
      files = readdirSync(dir);
    } catch {
      continue;
    }
    for (const f of files) {
      if (!prefixes.some((p) => f.startsWith(p))) continue;
      const p = join(dir, f);
      try {
        rmSync(p, { force: true });
        if (!existsSync(p)) deleted.push(p);
      } catch {
      }
    }
  }
  return deleted;
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
      const supabase = createServerSupabaseClient();
      const { data, error } = await supabase
        .from("models")
        .select("artifact")
        .eq("league_code", league)
        .eq("ver", ver)
        .limit(1);
      if (error) {
        return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
      }
      const raw = (data?.[0] as { artifact?: unknown } | undefined)?.artifact;
      const a = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : null;
      if (!a) {
        return NextResponse.json({ ok: true, league, ver, model_type: "unknown", detail: null });
      }
      const feats: string[] = Array.isArray(a.features) ? a.features : [];
      const mt = typeof a?.model_type === "string" ? a.model_type : "permatch";
      if (mt === "poisson" || mt === "dixon") {
        return NextResponse.json({
          ok: true, league, ver, model_type: mt,
          detail: {
            model_type: mt, features: feats,
            teams: Array.isArray(a.teams) ? a.teams : [],
            att: Array.isArray(a.att) ? a.att : [],
            def: Array.isArray(a.def) ? a.def : [],
            home: a.home ?? null, rho: a.rho ?? null, xi: a.xi ?? null,
            train_seasons: a.train_seasons ?? [], valid: a.valid ?? null,
            metrics: a.metrics ?? null,
          },
        });
      }
      if (mt === "logreg") {
        return NextResponse.json({
          ok: true, league, ver, model_type: mt,
          detail: {
            model_type: mt, features: feats,
            scaler_mean: Array.isArray(a.scaler_mean) ? a.scaler_mean : null,
            scaler_scale: Array.isArray(a.scaler_scale) ? a.scaler_scale : null,
            coef: Array.isArray(a.coef) ? a.coef : null,
            intercept: Array.isArray(a.intercept) ? a.intercept : null,
            train_seasons: a.train_seasons ?? [], valid: a.valid ?? null,
            metrics: a.metrics ?? null,
          },
        });
      }
      if (mt === "xgb") {
        return NextResponse.json({
          ok: true, league, ver, model_type: mt,
          detail: {
            model_type: mt, features: feats,
            embedded: a.xgb != null,
            train_seasons: a.train_seasons ?? [], valid: a.valid ?? null,
            metrics: a.metrics ?? null,
          },
        });
      }
      if (mt === "lstm") {
        return NextResponse.json({
          ok: true, league, ver, model_type: mt,
          detail: {
            model_type: mt,
            seq_len: a.seq_len ?? null,
            serving: "python-only",
            train_seasons: a.train_seasons ?? [], valid: a.valid ?? null,
            metrics: a.metrics ?? null,
          },
        });
      }
      if (mt !== "permatch") {
        return NextResponse.json({
          ok: true, league, ver, model_type: mt,
          detail: {
            model_type: mt, features: feats,
            train_seasons: a.train_seasons ?? [], valid: a.valid ?? null,
            metrics: a.metrics ?? null,
          },
        });
      }
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
          pattern_taus: a.pattern_taus ?? null,
          pattern_alpha: a.pattern_alpha ?? 0.5,
          draw_boost: a.draw_boost ?? 0,
          draw_close_margin: a.draw_close_margin ?? 0.12,
          draw_total_max: a.draw_total_max ?? 2.55,
          grid_params: a.grid_params ?? null,
          pattern_stats: a.pattern_stats ?? null,
          cumulative: (a.cumulative ?? null) as unknown,
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
    if (error) {
      return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
    }
    return NextResponse.json({ ok: true, data: data ?? [] });
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
    const supabase = createServerSupabaseClient();
    const { data, error } = await supabase
      .from("models")
      .select("artifact")
      .eq("league_code", league)
      .eq("ver", ver)
      .limit(1);
    if (error) {
      return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
    }
    const raw = (data?.[0] as { artifact?: unknown } | undefined)?.artifact;
    const a = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : null;
    if (!a) {
      return NextResponse.json({ ok: false, error: "모델 없음 (DB)" }, { status: 404 });
    }
    if (!Array.isArray(a.weights) || a.weights.length !== weights.length) {
      return NextResponse.json({ ok: false, error: "가중치 개수 불일치" }, { status: 400 });
    }
    a.weights = weights;
    if (hfa !== undefined) a.hfa = hfa;
    const { error: upErr } = await supabase
      .from("models")
      .update({ artifact: a })
      .eq("league_code", league)
      .eq("ver", ver);
    if (upErr) {
      return NextResponse.json({ ok: false, error: upErr.message }, { status: 500 });
    }
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
  if (!/^[A-Za-z0-9_-]+$/.test(league) || !/^[A-Za-z0-9_.-]+$/.test(ver)) {
    return NextResponse.json({ ok: false, error: "league와 ver가 필요" }, { status: 400 });
  }
  try {
    const deleted = cleanupFiles(league, ver);
    try {
      const supabase = createServerSupabaseClient();
      const { error } = await supabase.from("models").delete().eq("league_code", league).eq("ver", ver);
      if (error) throw new Error(error.message);
    } catch (e) {
      return NextResponse.json(
        { ok: false, error: e instanceof Error ? e.message : "DB 삭제 실패" },
        { status: 500 }
      );
    }
    return NextResponse.json({ ok: true, ver, deleted });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "Unknown error" },
      { status: 500 }
    );
  }
}
