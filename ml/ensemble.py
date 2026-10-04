#!/usr/bin/env python3
"""legacy + XGBoost 앙상블 적합.

p = w * legacy + (1 - w) * xgb, w는 검증셋에서 탐색.
XGB 부스터는 JSON으로 저장해서 TS 서빙에도 재사용한다.

실행:
    ml/.venv/bin/python ml/ensemble.py --league premier_league --base <ver>
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(ROOT, "ml"))
import models_bench as B
import permatch_mode as pm


def legacy_parts(art, rows):
    feats = art["features"]
    pm.FEATURES = [n for n in feats if n in pm.FEATURES13]
    pm.SEL = [pm.FEATURES13.index(n) for n in pm.FEATURES]
    mu, sd, e = art["mu"], art["sd"], art.get("emphasis")
    w, hfa = art["weights"], art["hfa"]
    d = art["draw_prior"]
    dw, db = art.get("draw_weights"), art.get("draw_bias", 0.0)
    mu_d, sd_d = art.get("draw_mu"), art.get("draw_sd")
    cap = art.get("contrib_cap")
    out = []
    for r in rows:
        x = pm.apply_emphasis(pm.apply_std(pm.row_features(r), mu, sd), e)
        xd = pm.draw_row_features(r)
        if mu_d and sd_d:
            xd = [(a - b) / s_ for a, b, s_ in zip(xd, mu_d, sd_d)]
        lin = pm.full_proba(x, w, hfa, d, dw, db, xd, cap)
        s = lin[0] + lin[2]
        out.append((x, lin[0] / s if s > 1e-12 else 0.5, lin[1]))
    return out


def legacy_proba_rows(art, rows):
    pats, tau = art.get("patterns"), art.get("pattern_tau", 0.0) or 0.0
    T = art.get("T", 1.0)
    out = []
    for x, ph, dd in legacy_parts(art, rows):
        lin = [ph * (1 - dd), dd, (1 - ph) * (1 - dd)]
        out.append(pm.apply_temp(pm.blend_proba(lin, x, pats, tau), T))
    return out


def fit_xgb(Xtr, ytr, league, ver):
    import numpy as _np
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                        objective="multi:softprob", num_class=3,
                        n_jobs=4, random_state=0, tree_method="hist")
    clf.fit(_np.asarray(Xtr, dtype=float), _np.asarray(ytr))
    path = os.path.join(ROOT, "ml", "permatch", f"{league}_{ver}.xgb.json")
    clf.get_booster().save_model(path)
    return clf, f"{league}_{ver}.xgb.json"


def main():
    ap = argparse.ArgumentParser(description="legacy+XGB 앙상블 적합")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--base", required=True, help="legacy 베이스 버전")
    ap.add_argument("--train-n", type=int, default=0,
                    help="0이면 아티팩트 시즌 사용")
    ap.add_argument("--rec-w", type=float, default=0.5,
                    help="선발식 무승부 가중 (acc + rec_w*rec, 0=순수 적중)")
    args = ap.parse_args()

    base = json.load(open(os.path.join(
        ROOT, "ml", "permatch", f"{args.league}_{args.base}.json"), encoding="utf-8"))
    if not isinstance(base.get("weights"), list):
        print("legacy 아티팩트가 아님")
        sys.exit(2)
    train_s = sorted(base.get("train_seasons", []))
    valid_s = sorted(s for s in (base.get("valid") or []) if s and s != "auto")
    if args.train_n > 0:
        full = B.known_seasons(args.league)
        train_s, valid_s = B.split_seasons(full, args.train_n)
    print(f"base={args.base} train={train_s} valid={valid_s}")

    rows = B.load_rows(args.league, train_s + valid_s)
    tr = [r for r in rows if str(r.get("season")) in set(train_s)]
    va = [r for r in rows if str(r.get("season")) in set(valid_s)]
    va_scored = [r for r in va if B.parse_label(r) is not None]
    yva = [B.parse_label(r) for r in va_scored]
    print(f"train rows={len(tr)} valid rows={len(va_scored)}")

    Xtr, ytr = B.feat_matrix(tr)
    Xva, _ = B.feat_matrix(va_scored)
    ens_ver = f"{args.base}-ens"
    clf, xgb_file = fit_xgb(Xtr, ytr, args.league, ens_ver)
    import numpy as _np
    px = clf.predict_proba(_np.asarray(Xva, dtype=float)).tolist()
    pl = legacy_proba_rows(base, va_scored)

    best = None
    table = []
    pats, tau, T = base.get("patterns"), base.get("pattern_tau", 0.0) or 0.0, base.get("T", 1.0)
    parts = legacy_parts(base, va_scored)
    al = 0.0
    while al <= 1.0001:
        al = round(al, 2)
        pb = []
        for (x, ph, ddl), p2 in zip(parts, px):
            dd = al * ddl + (1 - al) * p2[1]
            lin = [ph * (1 - dd), dd, (1 - ph) * (1 - dd)]
            pb.append(pm.apply_temp(pm.blend_proba(lin, x, pats, tau), T))
        m = B.metrics_of(pb, yva)
        key = (m["acc"] + args.rec_w * m["draw_rec"], -m["ll"])
        table.append((al, m))
        if best is None or key > best[0]:
            best = (key, al, m)
        al += 0.05
    for al, m in table:
        print(f"  a={al:.2f} acc={m['acc']:.3f} ll={m['ll']:.4f} rec={m['draw_rec']:.3f}")
    _, bw, bm = best
    print(f"blend a={bw} acc={bm['acc']:.3f} ll={bm['ll']:.4f} rec={bm['draw_rec']:.3f}")
    m_leg = B.metrics_of(pl, yva)
    m_xgb = B.metrics_of(px, yva)
    print(f"legacy  acc={m_leg['acc']:.3f} ll={m_leg['ll']:.4f} rec={m_leg['draw_rec']:.3f}")
    print(f"xgb     acc={m_xgb['acc']:.3f} ll={m_xgb['ll']:.4f} rec={m_xgb['draw_rec']:.3f}")

    art = {"model_type": "ensemble", "base_ver": args.base, "xgb_file": xgb_file,
           "blend_w": bw, "blend_mode": "draw",
           "features": base["features"],
           "train_seasons": train_s, "valid": valid_s,
           "metrics": {"acc": bm["acc"], "ll": bm["ll"], "draw_rec": bm["draw_rec"]}}
    p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{ens_ver}.json")
    json.dump(art, open(p, "w"))
    print(f"saved {p} + {xgb_file}")


if __name__ == "__main__":
    main()
