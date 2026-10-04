#!/usr/bin/env python3
"""학습시즌 무승부 분석 (공용).

새학습 시작 시 학습시즌 안에서 무가 나온 경기만 모아
draw 값을 같이 정하기 위한 리포트.
permatch_mode, dc/logreg/xgb/lstm_model에서 재사용한다.

분석 내용:
- 시즌별 n / 무 수 / 무율 + 전체 draw_prior
- 무 vs 무아닌 draw 피처 평균 비교
  (poisson_draw, rank_gap, total_goals, market_draw)
"""
from __future__ import annotations


DRAW_COLS = ["poisson_draw", "rank_gap", "total_goals", "market_draw"]


def _label(r):
    try:
        hs, aws = float(r.get("home_score")), float(r.get("away_score"))
    except (TypeError, ValueError):
        return None
    return 0 if hs > aws else (1 if hs == aws else 2)


def analyze_draws(rows, train_s):
    import permatch_mode as pm
    want = {str(s) for s in train_s}
    per, pool_d, pool_n = {}, [], []
    for r in rows:
        if str(r.get("season")) not in want:
            continue
        y = _label(r)
        if y is None:
            continue
        s = str(r.get("season"))
        d = per.setdefault(s, {"n": 0, "draws": 0})
        d["n"] += 1
        if y == 1:
            d["draws"] += 1
        try:
            xd = pm.draw_row_features(r)
        except Exception:
            xd = None
        if isinstance(xd, list) and len(xd) == 4:
            (pool_d if y == 1 else pool_n).append([float(v) for v in xd])
    seasons = []
    tn = td = 0
    for s in sorted(per):
        n, dr = per[s]["n"], per[s]["draws"]
        tn += n
        td += dr
        seasons.append({"season": s, "n": n, "draws": dr,
                        "rate": round(dr / max(n, 1), 4)})

    def _mean(pool):
        if not pool:
            return [0.0] * 4
        k = len(pool[0])
        return [round(sum(r[j] for r in pool) / len(pool), 4) for j in range(k)]

    return {"seasons": seasons, "n_train": tn, "draws": td,
            "draw_prior": round(td / max(tn, 1), 4),
            "draw_mean": dict(zip(DRAW_COLS, _mean(pool_d))),
            "nondraw_mean": dict(zip(DRAW_COLS, _mean(pool_n))),
            "n_draw_feat": len(pool_d), "n_nondraw_feat": len(pool_n)}


def log_report(rep, log):
    log(f"무분석 train={rep['n_train']} draws={rep['draws']} prior={rep['draw_prior']:.3f}")
    for s in rep["seasons"]:
        log(f"  {s['season']}: n={s['n']} 무={s['draws']} 율={s['rate']:.3f}")
    dm, nm = rep["draw_mean"], rep["nondraw_mean"]
    for c in DRAW_COLS:
        log(f"  {c}: 무={dm[c]:.3f} vs 비무={nm[c]:.3f}")
