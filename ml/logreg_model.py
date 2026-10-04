#!/usr/bin/env python3
"""Logistic Regression 독립 모델 (학습+저장).

벤치(ml/models_bench.py run_logreg)와 동일한 설정
(StandardScaler + LogisticRegression C=1.0 max_iter=2000)을
실전용 아티팩트로 저장한다.

실행:
    ml/.venv/bin/python ml/logreg_model.py --league premier_league \\
      --train 2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022 \\
      --valid 2024-2025,2025-2026 --ver logregbase --new
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
    ap = argparse.ArgumentParser(description="LogReg 학습")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--train", default="")
    ap.add_argument("--valid", default="auto")
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--ver", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--new", action="store_true")
    args = ap.parse_args()

    full = B.known_seasons(args.league)
    if args.train:
        train_s = parse_seasons(args.train)
    else:
        train_s, _ = B.split_seasons(full, args.train_n)
    if not args.valid or args.valid == "auto":
        train_set = set(train_s)
        valid_s = [s for s in full if s not in train_set and s != full[-1]] if len(full) > 1 else []
        if not valid_s:
            _, valid_s = B.split_seasons(full, args.train_n)
            valid_s = [s for s in valid_s if s not in set(train_s)]
    else:
        valid_s = parse_seasons(args.valid)

    print(f"league={args.league} train={train_s} valid={valid_s} C={args.C}")
    rows = B.load_rows(args.league, train_s + valid_s)
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

    Xtr, ytr = B.feat_matrix(tr)
    Xva, _ = B.feat_matrix(va_scored)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=args.C)
    clf.fit(sc.transform(Xtr), ytr)
    probs = clf.predict_proba(sc.transform(Xva)).tolist()
    labels = [B.parse_label(r) for r in va_scored]
    m = B.metrics_of(probs, labels)
    print(f"logreg       acc={m['acc']:.3f} ll={m['ll']:.4f} rec={m['draw_rec']:.3f} (n={m['n']})")

    import permatch_mode as pm
    feats = list(pm.FEATURES13)
    art = {
        "model_type": "logreg",
        "features": feats,
        "scaler_mean": [float(v) for v in sc.mean_],
        "scaler_scale": [float(v) for v in sc.scale_],
        "coef": [[float(v) for v in row] for row in clf.coef_],
        "intercept": [float(v) for v in clf.intercept_],
        "classes": [int(v) for v in clf.classes_],
        "C": args.C,
        "train_seasons": sorted(train_s),
        "valid": sorted(valid_s),
        "metrics": {"acc": m["acc"], "ll": m["ll"], "draw_rec": m["draw_rec"]},
        "draw_analysis": draw_rep,
    }
    out_p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(art, open(out_p, "w", encoding="utf-8"))
    print(f"saved {out_p}")


if __name__ == "__main__":
    main()
