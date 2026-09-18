"""리그별 승/무/패 예측 학습 (리키지-안전 피처만 사용).

DB의 strength/head_to_head/rank는 크롤링 스냅샷이라 미래 정보가 섞여 있음
(2017 1R recent_5가 과거 기록과 불일치 확인). 그래서 폼/전력은 스코어
히스토리에서 직접 시계열로 재계산한다. 배당은 initial 계열만 사용.

사용법:
  pip install -r ml/requirements.txt
  export SUPABASE_URL=... SUPABASE_SECRET_KEY=...
  python ml/train.py --league k_league_1 --train 2016,2017,2018,2019,2020,2021,2022,2023 --valid 2024 --test 2025
"""

from __future__ import annotations

import argparse
import math
import os

from bins import backoff_keys, combo_key, raw_home_odd

FEATURES = [
    "elo_diff", "elo_home", "elo_away",
    "form_h_pts5", "form_a_pts5", "form_h_n", "form_a_n",
    "h2h_h_w5", "h2h_d5", "h2h_h_l5",
    "odd_ph", "odd_pd", "odd_pa",
    "form_h", "form_a",
    "round_no",
    "att_h", "att_a", "def_h", "def_a",
    "rank_diff",
    "case_ph", "case_pd", "case_pa", "case_n", "case_level",
]

ELO_K = 24
ELO_HOME_ADV = 60


def parse_score(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def implied_probs(wdl: dict | None):
    """동일배당 승무패 -> 내재확률 (overround 제거). 없으면 NaN 3개."""
    if not wdl:
        return (math.nan, math.nan, math.nan)
    try:
        inv = [1.0 / float(wdl[k]) for k in ("home", "draw", "away")]
    except (TypeError, ValueError, ZeroDivisionError, KeyError):
        return (math.nan, math.nan, math.nan)
    s = sum(inv)
    return tuple(v / s for v in inv)


def same_odds_wdl(same) -> dict | None:
    if not isinstance(same, dict):
        return None
    w = same.get("win_draw_lose")
    if isinstance(w, dict) and any(v is not None for v in w.values()):
        return w
    inner = same.get("same_odds")
    if isinstance(inner, dict) and isinstance(inner.get("win_draw_lose"), dict):
        return inner["win_draw_lose"]
    return None


def defense_line(form) -> float:
    try:
        return float(str(form).split("-")[0])
    except (TypeError, ValueError, IndexError, AttributeError):
        return math.nan


def row_combo_key(b) -> tuple:
    f, m = b["feat"], b["match"]
    lin = m.get("lineup") or {}
    return combo_key(f, raw_home_odd(same_odds_wdl, m.get("same_odds")),
                     defense_line(lin.get("home")), defense_line(lin.get("away")))


def build_case_stats(train_rows):
    from collections import Counter

    buckets: dict[tuple, Counter] = {}
    for b in train_rows:
        if b["label"] is None:
            continue
        for bk in backoff_keys(row_combo_key(b)):
            buckets.setdefault(bk, Counter())[b["label"]] += 1
    total = Counter()
    for c in buckets.values():
        total.update(c)
    return buckets, total


def attach_case_features(rows, buckets, total, min_n: int):
    for b in rows:
        key = row_combo_key(b)
        picked, level = None, 99
        for lv, bk in enumerate(backoff_keys(key)):
            c = buckets.get(bk)
            n = sum(c.values()) if c else 0
            if n >= min_n:
                picked, level = c, lv
                break
        if picked is None:
            picked, n = total, sum(total.values())
        else:
            n = sum(picked.values())
        f = b["feat"]
        if n == 0:
            f.update(case_ph=math.nan, case_pd=math.nan, case_pa=math.nan, case_n=0, case_level=99)
            continue
        f["case_ph"], f["case_pd"], f["case_pa"] = picked[0] / n, picked[1] / n, picked[2] / n
        f["case_n"], f["case_level"] = n, level


def season_key(s: str):
    return str(s)


def build_rows(matches, elo_carry: float = 1.0):
    """시간순으로 돌며 피처 생성 + Elo/폼 상태 갱신. 스코어 없는 행은 학습 제외."""
    elos: dict[str, float] = {}
    forms: dict[str, list[int]] = {}
    h2h: dict[tuple[str, str], list[int]] = {}
    att: dict[str, float] = {}
    dfn: dict[str, float] = {}
    pts: dict[str, int] = {}
    gf: dict[str, float] = {}
    ga: dict[str, float] = {}
    out = []
    order = sorted(matches, key=lambda r: (season_key(r["season"]), r.get("round") or 0, r["id"]))
    cur_season = None
    for r in order:
        if r["season"] != cur_season:
            cur_season = r["season"]
            if elo_carry < 1.0:
                for t in elos:
                    elos[t] = 1500.0 + (elos[t] - 1500.0) * elo_carry
            pts.clear()
            gf.clear()
            ga.clear()
        h, a = r["home_team"], r["away_team"]
        eh, ea = elos.get(h, 1500.0), elos.get(a, 1500.0)
        fh = forms.get(h, [])[-5:]
        fa = forms.get(a, [])[-5:]
        key = tuple(sorted((h, a)))
        past = h2h.get(key, [])[-5:]
        hw = sum(1 for t, res in past if (res == 0 and t == h) or (res == 2 and t == a))
        dr = sum(1 for _, res in past if res == 1)
        hl = sum(1 for t, res in past if (res == 2 and t == h) or (res == 0 and t == a))
        ph, pd, pa = implied_probs(same_odds_wdl(r.get("same_odds")))
        lin = r.get("lineup") or {}

        def standing(t: str):
            return (pts.get(t, 0), gf.get(t, 0.0) - ga.get(t, 0.0), gf.get(t, 0.0))

        seen_teams = set(pts) | set(gf) | {h, a}
        rank_h = 1 + sum(1 for o in seen_teams if o != h and standing(o) > standing(h))
        rank_a = 1 + sum(1 for o in seen_teams if o != a and standing(o) > standing(a))

        feat = {
            "elo_diff": eh - ea, "elo_home": eh, "elo_away": ea,
            "form_h_pts5": sum(fh), "form_a_pts5": sum(fa),
            "form_h_n": len(fh), "form_a_n": len(fa),
            "h2h_h_w5": hw, "h2h_d5": dr, "h2h_h_l5": hl,
            "odd_ph": ph, "odd_pd": pd, "odd_pa": pa,
            "form_h": defense_line(lin.get("home")), "form_a": defense_line(lin.get("away")),
            "round_no": r.get("round") or 0,
            "att_h": att.get(h, math.nan), "att_a": att.get(a, math.nan),
            "def_h": dfn.get(h, math.nan), "def_a": dfn.get(a, math.nan),
            "rank_diff": rank_h - rank_a,
        }
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        label = None
        if hs is not None and aws is not None:
            label = 0 if hs > aws else (1 if hs == aws else 2)
            exp_h = 1.0 / (1.0 + 10 ** (-(eh + ELO_HOME_ADV - ea) / 400.0))
            score_h = 1.0 if label == 0 else (0.5 if label == 1 else 0.0)
            k = ELO_K * (1.0 + 0.5 * math.log1p(abs(hs - aws)))
            elos[h] = eh + k * (score_h - exp_h)
            elos[a] = ea + k * ((1 - score_h) - (1 - exp_h))
            pts_h = 3 if label == 0 else (1 if label == 1 else 0)
            pts_a = 3 if label == 2 else (1 if label == 1 else 0)
            forms.setdefault(h, []).append(pts_h)
            forms.setdefault(a, []).append(pts_a)
            h2h.setdefault(key, []).append((h, label))
            pts[h] = pts.get(h, 0) + pts_h
            pts[a] = pts.get(a, 0) + pts_a
            gf[h] = gf.get(h, 0.0) + hs
            ga[h] = ga.get(h, 0.0) + aws
            gf[a] = gf.get(a, 0.0) + aws
            ga[a] = ga.get(a, 0.0) + hs
            alpha = 0.12
            att[h] = hs if h not in att else (1 - alpha) * att[h] + alpha * hs
            att[a] = aws if a not in att else (1 - alpha) * att[a] + alpha * aws
            dfn[h] = aws if h not in dfn else (1 - alpha) * dfn[h] + alpha * aws
            dfn[a] = hs if a not in dfn else (1 - alpha) * dfn[a] + alpha * hs
        out.append({"match": r, "feat": feat, "label": label})
    return out


def fetch_league(url: str, key: str, league: str):
    from supabase import create_client

    sb = create_client(url, key)
    rows, offset, page = [], 0, 1000
    while True:
        res = (
            sb.table("soccer_matches")
            .select("id,league_code,season,round,home_team,away_team,home_score,away_score,lineup,same_odds,source_match_id")
            .eq("league_code", league)
            .order("id")
            .range(offset, offset + page - 1)
            .execute()
        )
        rows += res.data or []
        if not res.data or len(res.data) < page:
            break
        offset += page
    return sb, rows


def evaluate(y_true, proba, tag: str):
    from sklearn.metrics import accuracy_score, log_loss

    import numpy as np

    pred = np.argmax(proba, axis=1)
    base = max(__import__("collections").Counter(y_true).values()) / len(y_true)
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, proba, labels=[0, 1, 2])
    print(f"[{tag}] n={len(y_true)} acc={acc:.3f} logloss={ll:.3f} baseline={base:.3f}")
    return acc, ll


def fit_model(Xtr, ytr, args):
    import lightgbm as lgb

    if args.model == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000))
    else:
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced",
                                   n_estimators=args.estimators, learning_rate=args.lr,
                                   min_child_samples=args.min_child, verbose=-1)
    model.fit(Xtr, ytr)
    return model


def artifact_path(league: str, ver: str, model: str) -> str:
    ext = "txt" if model == "gbm" else "pkl"
    return f"ml/models/{league}_{ver}.{ext}"


def save_artifact(model, league: str, ver: str, model_type: str):
    os.makedirs("ml/models", exist_ok=True)
    path = artifact_path(league, ver, model_type)
    if model_type == "gbm":
        model.booster_.save_model(path)
    else:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return path


def load_artifact(league: str, ver: str, model_type: str):
    path = artifact_path(league, ver, model_type)
    if model_type == "gbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--mode", default="train", choices=["train", "predict"])
    ap.add_argument("--train", default="", help="학습 시즌, 콤마 구분")
    ap.add_argument("--valid", default="")
    ap.add_argument("--test", default="")
    ap.add_argument("--load-ver", default="")
    ap.add_argument("--history", default="", help="추론용 상태 구축 시즌 (콤마)")
    ap.add_argument("--predict", default="", help="추론 대상 시즌 (콤마)")
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--estimators", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-child", type=int, default=20)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--elo-carry", type=float, default=1.0)
    ap.add_argument("--model", default="gbm", choices=["gbm", "logreg"])
    ap.add_argument("--case-min", type=int, default=30)
    ap.add_argument("--use-case", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SECRET_KEY"]
    sb, matches = fetch_league(url, key, args.league)
    print(f"loaded {len(matches)} rows for {args.league}")

    import numpy as np

    def pack(rows):
        rows = [r for r in rows if r["label"] is not None]
        X = np.array([[r["feat"][f] for f in FEATURES] for r in rows], dtype=float)
        y = np.array([r["label"] for r in rows])
        return X, y, rows

    def save_predictions(sb, league: str, ver: str, rows, proba):
        payload = []
        for r, p in zip(rows, proba):
            m = r["match"]
            payload.append({
                "league_code": league,
                "season": str(m["season"]),
                "source_match_id": m.get("source_match_id") or str(m["id"]),
                "model_ver": ver,
                "prob_home": float(p[0]), "prob_draw": float(p[1]), "prob_away": float(p[2]),
            })
        for i in range(0, len(payload), 500):
            sb.table("predictions").upsert(
                payload[i:i + 500], on_conflict="league_code,season,source_match_id,model_ver"
            ).execute()
        print(f"saved {len(payload)} predictions")

    if args.mode == "predict":
        assert args.load_ver and args.predict, "--load-ver와 --predict 필요"
        try:
            model_row = sb.table("models").select("*").eq("ver", args.load_ver).limit(1).execute().data
        except Exception:
            model_row = []
        model_type = (model_row[0]["model_type"] if model_row else args.model)
        if not model_row:
            for cand in ("pkl", "txt"):
                if os.path.exists(artifact_path(args.league, args.load_ver, "logreg" if cand == "pkl" else "gbm")):
                    model_type = "logreg" if cand == "pkl" else "gbm"
                    break
        base_hist = args.history or (",".join(model_row[0]["train_seasons"]) if model_row else "")
        hist_s = set(base_hist.split(",")) if base_hist else set()
        pred_s = set(args.predict.split(","))
        built = build_rows(matches, elo_carry=args.elo_carry)
        targets = [b for b in built if str(b["match"]["season"]) in pred_s]
        used_hist = {str(b["match"]["season"]) for b in built if str(b["match"]["season"]) in hist_s}
        print(f"history seasons used: {sorted(used_hist)} predict rows: {len(targets)}")
        model = load_artifact(args.league, args.load_ver, model_type)
        use_feats = FEATURES
        try:
            if hasattr(model, "num_feature"):
                n_exp = model.num_feature()
            else:
                n_exp = model.steps[-1][1].n_features_in_
            if n_exp == len(FEATURES) - 5:
                use_feats = [f for f in FEATURES if not f.startswith("case_")]
        except Exception:
            pass
        hist_rows = [b for b in built
                     if (str(b["match"]["season"]) in hist_s if hist_s else str(b["match"]["season"]) not in pred_s)
                     and b["label"] is not None]
        if args.use_case and hist_rows:
            buckets, total = build_case_stats(hist_rows)
            attach_case_features(built, buckets, total, args.case_min)
        else:
            for b in built:
                b["feat"].update(case_ph=math.nan, case_pd=math.nan, case_pa=math.nan,
                                 case_n=0, case_level=99)
        Xt = np.array([[t["feat"][f] for f in use_feats] for t in targets], dtype=float)
        proba = model.predict(Xt) if model_type == "gbm" and not hasattr(model, "predict_proba") else model.predict_proba(Xt)
        if not args.no_save:
            save_predictions(sb, args.league, args.load_ver, targets, proba)
        return

    assert args.train and args.valid, "--train과 --valid 필요"
    built = build_rows(matches, elo_carry=args.elo_carry)
    train_s = set(args.train.split(","))
    by_split: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for b in built:
        s = str(b["match"]["season"])
        if s in train_s:
            by_split["train"].append(b)
        if s == args.valid:
            by_split["valid"].append(b)
        if args.test and s == args.test:
            by_split["test"].append(b)

    if args.use_case:
        train_rows = by_split["train"]
        cum: list = []
        for s in sorted({str(b["match"]["season"]) for b in train_rows}):
            grp = [b for b in train_rows if str(b["match"]["season"]) == s]
            buckets, total = build_case_stats(cum)
            attach_case_features(grp, buckets, total, args.case_min)
            cum += grp
        buckets, total = build_case_stats(train_rows)
        for key in ("valid", "test"):
            attach_case_features(by_split[key], buckets, total, args.case_min)
    else:
        for split_rows in by_split.values():
            for b in split_rows:
                b["feat"].update(case_ph=math.nan, case_pd=math.nan, case_pa=math.nan,
                                 case_n=0, case_level=99)

    Xtr, ytr, _ = pack(by_split["train"])
    Xva, yva, _ = pack(by_split["valid"])
    Xte, yte, te_rows = pack(by_split["test"])
    print(f"train={len(ytr)} valid={len(yva)} test={len(yte)}")

    model = fit_model(Xtr, ytr, args)
    vacc, vll = evaluate(yva, model.predict_proba(Xva), "valid")
    tacc, tll = (None, None)
    if len(yte):
        tacc, tll = evaluate(yte, model.predict_proba(Xte), "test")

    if args.model == "gbm":
        gain = sorted(zip(FEATURES, model.booster_.feature_importance(importance_type="gain")),
                      key=lambda x: -x[1])
        print("importance:", ", ".join(f"{f}={g:.0f}" for f, g in gain))

    if args.no_save:
        return

    save_artifact(model, args.league, args.ver, args.model)
    try:
        sb.table("models").upsert({
            "ver": args.ver,
            "league_code": args.league,
            "train_seasons": sorted(train_s),
            "model_type": args.model,
            "metrics": {"valid_acc": vacc, "valid_logloss": vll, "test_acc": tacc, "test_logloss": tll},
        }, on_conflict="ver").execute()
        print(f"registered model {args.ver}")
    except Exception as e:
        print(f"models 테이블 등록 실패 (002_models.sql 실행 필요): {e}")

    if len(yte):
        save_predictions(sb, args.league, args.ver, te_rows, model.predict_proba(Xte))


if __name__ == "__main__":
    main()
