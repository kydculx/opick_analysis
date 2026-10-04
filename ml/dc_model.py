#!/usr/bin/env python3
"""Poisson / Dixon-Coles 독립 모델 (학습+저장+예측).

벤치(ml/models_bench.py)의 fit_dc/dc_predict와 동일한 수학을 사용하되,
실전용 아티팩트(ml/permatch/{league}_{ver}.json)로 저장하고 TS 서빙에서 읽는다.

실행:
    ml/.venv/bin/python ml/dc_model.py --league premier_league \\
      --train 2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022 \\
      --valid 2024-2025,2025-2026 --ver dcbase --xi 0.002
    ml/.venv/bin/python ml/dc_model.py --league premier_league \\
      --train 2022-2023,2023-2024 --valid 2024-2025 --ver dcbase-test --no-rho
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import models_bench as B


def parse_seasons(s):
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description="Poisson/Dixon-Coles 학습")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--train", default="", help="학습 시즌 CSV (비면 자동: 오래된순 N개)")
    ap.add_argument("--valid", default="auto", help="검증 시즌 CSV 또는 auto")
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--ver", required=True)
    ap.add_argument("--xi", type=float, default=0.002)
    ap.add_argument("--no-rho", action="store_true", help="지정시 Poisson (rho=0)")
    ap.add_argument("--new", action="store_true", help="기존 아티팩트 덮어쓰기")
    args = ap.parse_args()

    full = B.known_seasons(args.league)
    if args.train:
        train_s = parse_seasons(args.train)
    else:
        train_s, _auto_v = B.split_seasons(full, args.train_n)
    if not args.valid or args.valid == "auto":
        train_set = set(train_s)
        valid_s = [s for s in full if s not in train_set and s != full[-1]] if len(full) > 1 else []
        if not valid_s:
            _, valid_s = B.split_seasons(full, args.train_n)
            valid_s = [s for s in valid_s if s not in set(train_s)]
    else:
        valid_s = parse_seasons(args.valid)

    print(f"league={args.league} train={train_s} valid={valid_s} rho={not args.no_rho}")

    out_p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.json")
    if os.path.exists(out_p) and not args.new:
        try:
            cur = json.load(open(out_p, encoding="utf-8"))
            if cur.get("model_type") in ("poisson", "dixon"):
                print(f"기존 아티팩트 있음: {out_p} (--new으로 덮어쓰기)")
        except (OSError, ValueError):
            pass

    rows = B.load_rows(args.league, train_s + valid_s)
    sidx = {s: i for i, s in enumerate(full)}
    tr = [r for r in rows if str(r.get("season")) in set(train_s)]
    va = [r for r in rows if str(r.get("season")) in set(valid_s)]
    va_scored = [r for r in va if B.parse_label(r) is not None]
    print(f"train rows={len(tr)} valid rows={len(va_scored)}")
    if not tr or not va_scored:
        print("데이터 부족 (ml/.cache 확인 필요)")
        sys.exit(2)
    try:
        import draw_analysis as _da
        draw_rep = _da.analyze_draws(rows, train_s)
        _da.log_report(draw_rep, print)
    except Exception as e:
        print(f"무분석 스킵: {type(e).__name__}")
        draw_rep = {"seasons": [], "n_train": 0, "draws": 0, "draw_prior": 0.0}

    tab_tr = B.build_dc_table(tr, sidx)
    model = B.fit_dc(tab_tr, xi=args.xi, use_rho=not args.no_rho)
    print(f"  home={model['home']:.3f} rho={model['rho']:.3f}")

    probs = [B.dc_predict(model, str(r.get("home_team")), str(r.get("away_team")))
             for r in va_scored]
    labels = [B.parse_label(r) for r in va_scored]
    m = B.metrics_of(probs, labels)
    print(f"{'poisson' if args.no_rho else 'dixon':12} acc={m['acc']:.3f} ll={m['ll']:.4f} rec={m['draw_rec']:.3f} (n={m['n']})")

    art = {
        "model_type": "poisson" if args.no_rho else "dixon",
        "teams": model["teams"],
        "att": model["att"],
        "def": model["def"],
        "home": model["home"],
        "rho": model["rho"],
        "xi": args.xi,
        "train_seasons": sorted(train_s),
        "valid": sorted(valid_s),
        "metrics": {"acc": m["acc"], "ll": m["ll"], "draw_rec": m["draw_rec"]},
        "draw_analysis": draw_rep,
    }
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(art, open(out_p, "w", encoding="utf-8"))
    print(f"saved {out_p}")


if __name__ == "__main__":
    main()
