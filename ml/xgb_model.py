#!/usr/bin/env python3
"""XGBoost 독립 모델 (학습+저장).

ensemble.py fit_xgb와 동일한 하이퍼파라미터를 사용하되,
legacy 베이스 없이 단독으로 서빙 가능한 아티팩트로 저장한다.

실행:
    ml/.venv/bin/python ml/xgb_model.py --league premier_league \\
      --train 2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022 \\
      --valid 2024-2025,2025-2026 --ver xgbbase --new
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import models_bench as B

PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
              objective="multi:softprob", num_class=3,
              n_jobs=4, random_state=0, tree_method="hist")


def parse_seasons(s):
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description="XGB 독립 학습")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--train", default="")
    ap.add_argument("--valid", default="auto")
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--ver", required=True)
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

    print(f"league={args.league} train={train_s} valid={valid_s}")
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

    import numpy as _np
    from xgboost import XGBClassifier
    import permatch_mode as pm
    Xtr, ytr = B.feat_matrix(tr)
    Xva, _ = B.feat_matrix(va_scored)
    clf = XGBClassifier(**PARAMS)
    clf.fit(_np.asarray(Xtr, dtype=float), _np.asarray(ytr))
    probs = clf.predict_proba(_np.asarray(Xva, dtype=float)).tolist()
    labels = [B.parse_label(r) for r in va_scored]
    m = B.metrics_of(probs, labels)
    print(f"xgb          acc={m['acc']:.3f} ll={m['ll']:.4f} rec={m['draw_rec']:.3f} (n={m['n']})")

    xgb_file = f"{args.league}_{args.ver}.xgb.json"
    xp = os.path.join(ROOT, "ml", "permatch", xgb_file)
    os.makedirs(os.path.dirname(xp), exist_ok=True)
    clf.get_booster().save_model(xp)
    art = {
        "model_type": "xgb",
        "xgb_file": xgb_file,
        "features": list(pm.FEATURES13),
        "params": {k: (float(v) if isinstance(v, float) else v) for k, v in PARAMS.items()},
        "train_seasons": sorted(train_s),
        "valid": sorted(valid_s),
        "metrics": {"acc": m["acc"], "ll": m["ll"], "draw_rec": m["draw_rec"]},
        "draw_analysis": draw_rep,
    }
    out_p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.json")
    json.dump(art, open(out_p, "w", encoding="utf-8"))
    print(f"saved {out_p} + {xgb_file}")


if __name__ == "__main__":
    main()
