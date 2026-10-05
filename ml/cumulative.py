#!/usr/bin/env python3
"""누적(워크포워드) 학습: 가장 오래된 경기부터 시간순으로 예측하고,
실제 결과가 나오면 매번 1스텝씩 가중치를 갱신한 뒤 다음 경기를 예측한다.

갱신식 (3클래스 NLL을 S=w.x+hfa에 대해 미분):
  홈승(y=0): g=-(1-ph) / 원정승(y=2): g=+ph / 무(y=1): g=0
  w -= lr*g*xe, hfa -= lr*g  (무는 S를 안 움직이고 draw 모델에 맡김)

출력:
  ml/permatch/{league}_{ver}.json (legacy 호환: 최종 가중치 + walk 지표)
  ml/permatch/{league}_{ver}.cumu.json (경기별 pick/hit 기록)

실행:
    ml/.venv/bin/python ml/cumulative.py --league premier_league \\
      --base cmp22-24-f13 --ver cmp22-24-f13-cumu --lr 0.01
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))

import models_bench as B
import permatch_mode as pm


def parse_seasons(s):
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description="누적 워크포워드 학습")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--seasons", default="", help="비면 전 시즌 오래된순")
    ap.add_argument("--base", required=True, help="시작 베이스 버전")
    ap.add_argument("--ver", required=True, help="저장 버전")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    full = B.known_seasons(args.league)
    sidx = {s: i for i, s in enumerate(full)}
    seasons = parse_seasons(args.seasons) or list(full)
    seasons = sorted(seasons, key=lambda s: sidx.get(s, 0))
    print(f"league={args.league} seasons={seasons} base={args.base} lr={args.lr} l2={args.l2}")

    base = json.load(open(os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.base}.json"), encoding="utf-8"))
    if not isinstance(base.get("weights"), list):
        print("legacy 베이스만 지원 (현재: %s)" % base.get("model_type"))
        sys.exit(2)
    afeats = base.get("features")
    if isinstance(afeats, list) and afeats:
        pm.FEATURES = [n for n in afeats if n in pm.FEATURES13]
        pm.SEL = [pm.FEATURES13.index(n) for n in pm.FEATURES]
    w = list(base["weights"])
    hfa = float(base["hfa"])
    mu, sd = base["mu"], base["sd"]
    e = base.get("emphasis", [1.0] * len(w))
    d = base["draw_prior"]
    dw, db = base.get("draw_weights"), base.get("draw_bias", 0.0) or 0.0
    mu_d, sd_d = base.get("draw_mu"), base.get("draw_sd")
    pats, tau, T = base.get("patterns"), base.get("pattern_tau", 0.0) or 0.0, base.get("T", 1.0)
    cap = base.get("contrib_cap")

    rows = B.load_rows(args.league, seasons)
    order = sorted(range(len(rows)),
                   key=lambda i: (sidx.get(str(rows[i].get("season")), 0),
                                  int(rows[i].get("round") or 0), rows[i].get("id") or 0))
    scored = []
    for i in order:
        r = rows[i]
        y = B.parse_label(r)
        if y is not None and str(r.get("season")) in set(seasons):
            scored.append(r)
    print(f"scored matches={len(scored)}")
    if not scored:
        print("스코어 있는 경기 없음")
        sys.exit(2)

    lr = max(args.lr, 0.0)
    l2 = max(args.l2, 0.0)
    walk_hit, walk_tot, walk_ll = 0, 0, 0.0
    stat_hit, stat_tot = 0, 0
    per_season = {}
    picks = []
    w0 = list(w)
    for n, r in enumerate(scored, 1):
        y = B.parse_label(r)
        x = pm.apply_emphasis(pm.apply_std(pm.row_features(r), mu, sd), e)
        xd = pm.draw_row_features(r)
        if mu_d and sd_d:
            xd = [(a - b) / s_ for a, b, s_ in zip(xd, mu_d, sd_d)]
        lin = pm.full_proba(x, w, hfa, d, dw, db, xd, cap)
        p = pm.apply_temp(pm.blend_proba(lin, x, pats, tau), T)
        pick = p.index(max(p))
        walk_hit += (pick == y)
        walk_tot += 1
        walk_ll += -math.log(max(p[y], 1e-12))
        s = str(r.get("season"))
        dse = per_season.setdefault(s, {"n": 0, "hit": 0})
        dse["n"] += 1
        dse["hit"] += (pick == y)
        key = r.get("source_match_id", r.get("id"))
        picks.append([key, pick, 1 if pick == y else 0])
        sw, shfa = list(w0), float(base["hfa"])
        slin = pm.full_proba(x, sw, shfa, d, dw, db, xd, cap)
        sp = pm.apply_temp(pm.blend_proba(slin, x, pats, tau), T)
        stat_hit += (sp.index(max(sp)) == y)
        stat_tot += 1
        s_lin = max(-30.0, min(30.0, sum(a * b for a, b in zip(x, w)) + hfa))
        ph = 1.0 / (1.0 + math.exp(-s_lin))
        g = -(1.0 - ph) if y == 0 else (ph if y == 2 else 0.0)
        if g:
            if l2:
                w = [v * (1.0 - lr * l2) for v in w]
            w = [v - lr * g * xv for v, xv in zip(w, x)]
            hfa -= lr * g
        if n % max(args.log_every, 1) == 0 or n == len(scored):
            print(f"누적 {n}/{len(scored)} walk={walk_hit / walk_tot:.3f}", flush=True)

    walk = {"n": walk_tot, "acc": walk_hit / walk_tot, "ll": walk_ll / walk_tot,
            "seasons": {s: {"n": v["n"], "acc": v["hit"] / v["n"]} for s, v in sorted(per_season.items())}}
    print(f"walk acc={walk['acc']:.3f} ll={walk['ll']:.4f} (n={walk_tot})")
    print(f"static base acc={stat_hit / stat_tot:.3f} (n={stat_tot})")

    art = dict(base)
    art.update({"weights": list(w), "hfa": float(hfa),
                "train_seasons": sorted(set(seasons)),
                "metrics": {"acc": walk["acc"], "ll": walk["ll"], "draw_rec": 0.0},
                "cumulative": {"base_ver": args.base, "lr": lr, "l2": l2,
                               "walk": walk, "static_acc": stat_hit / stat_tot}})
    out_p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.json")
    json.dump(art, open(out_p, "w", encoding="utf-8"))
    with open(os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.cumu.json"), "w", encoding="utf-8") as f:
        json.dump({"league": args.league, "ver": args.ver, "walk": walk,
                   "static_acc": stat_hit / stat_tot, "picks": picks}, f)
    print(f"saved {out_p}")


if __name__ == "__main__":
    main()
