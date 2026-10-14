"""Supabase models-table persistence for trained artifacts.

Table `public.models` (see supabase/migrations/003_models_artifact.sql):
  ver PK, league_code, train_seasons, model_type, metrics, artifact, created_at.

Uses SUPABASE_URL / SUPABASE_SECRET_KEY (same env as training).
All functions are best-effort: DB failure only logs, never crashes training.
"""
from __future__ import annotations

import os

def _client():
    try:
        from supabase import create_client
    except ImportError:
        return None
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def save_local_enabled() -> bool:
    # DB-only 정책: 새학습은 무조건 DB에만 저장, 로컬 파일 쓰지 않음.
    return False


def upsert_model(league: str, ver: str, artifact: dict, log=None) -> bool:
    sb = _client()
    if sb is None:
        return False
    row = {
        "ver": ver,
        "league_code": league,
        "train_seasons": [str(s) for s in (artifact.get("train_seasons") or [])],
        "model_type": str(artifact.get("model_type") or "permatch"),
        "metrics": artifact.get("metrics") or {},
        "artifact": artifact,
    }
    try:
        sb.table("models").upsert(row, on_conflict="ver").execute()
        if log:
            log(f"supabase 저장: models/{ver}")
        return True
    except Exception as e:
        if log:
            log(f"supabase 저장 실패 ({type(e).__name__}): {e}")
        return False


def load_model_artifact(league: str, ver: str):
    sb = _client()
    if sb is None:
        return None
    try:
        res = (sb.table("models").select("artifact")
               .eq("league_code", league).eq("ver", ver).limit(1).execute())
    except Exception:
        return None
    rows = res.data or []
    if not rows or not isinstance(rows[0].get("artifact"), dict):
        return None
    return rows[0]["artifact"]
