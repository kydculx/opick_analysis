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
    ap.add_argument("--test", default="", help="동결 평가 시즌 (갱신なし)")
    ap.add_argument("--base", default="", help="시작 베이스 버전 (비면 콜드스타트)")
    ap.add_argument("--ver", required=True, help="저장 버전")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--lr-decay", type=float, default=0.0, help="매경기 lr/(1+decay*n) 감쇠 (0이면 끔)")
    ap.add_argument("--rollback", action="store_true", help="최근 500경기 최고 가중치로 복원")
    ap.add_argument("--dw-lr", type=float, default=0.01, help="무브랜치 lr")
    ap.add_argument("--draw-up", type=float, default=1.0, help="무 실제시 무브랜치 갱신 배율")
    ap.add_argument("--draw-t", type=float, default=0.3, help="dd가 이 값 넘고 양쪽 확률이 draw-c 미만이면 무")
    ap.add_argument("--draw-c", type=float, default=0.4, help="무 찍을 때 홈/원정 확률 상한")
    ap.add_argument("--alert-t", type=float, default=0.3, help="무복병 플래그 dd 임계값")
    ap.add_argument("--spec-w", type=float, default=0.0, help="무판정기 적합 가중 (0이면 고정 규칙, 켜면 walk 과적합 주의)")
    ap.add_argument("--features", default="", help="콤마 피처 부분집합 (비면 전체 13)")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    full = B.known_seasons(args.league)
    sidx = {s: i for i, s in enumerate(full)}
    seasons = parse_seasons(args.seasons) or list(full)
    seasons = sorted(seasons, key=lambda s: sidx.get(s, 0))
    test_s = [s for s in parse_seasons(args.test) if s not in set(seasons)]
    test_s = sorted(test_s, key=lambda s: sidx.get(s, 0))
    featsel = [s.strip() for s in (args.features or "").split(",") if s.strip()]
    if featsel:
        bad = [n for n in featsel if n not in pm.FEATURES13]
        if bad or not featsel:
            print(f"--features 오류: {bad or featsel}")
            sys.exit(2)
        pm.FEATURES = list(featsel)
        pm.SEL = [pm.FEATURES13.index(n) for n in pm.FEATURES]
    print(f"league={args.league} seasons={seasons} base={args.base} lr={args.lr} l2={args.l2} feats={','.join(pm.FEATURES)}")

    rows = B.load_rows(args.league, seasons + test_s)
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

    base = None
    if args.base:
        base = json.load(open(os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.base}.json"), encoding="utf-8"))
        if not isinstance(base.get("weights"), list):
            print("legacy 베이스만 지원 (현재: %s)" % base.get("model_type"))
            sys.exit(2)
        afeats = base.get("features")
        if not featsel and isinstance(afeats, list) and afeats:
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
        feats = list(pm.FEATURES)
        hfa0 = float(base["hfa"])
    else:
        feats = list(pm.FEATURES)
        Xall = [pm.row_features(r) for r in scored]
        mu, sd = pm.standardize(Xall)
        draws = sum(1 for r in scored if B.parse_label(r) == 1)
        d = draws / max(len(scored), 1)
        w, hfa, hfa0 = [0.0] * len(feats), 0.0, 0.0
        e = [1.0] * len(feats)
        dw, db, mu_d, sd_d = None, 0.0, None, None
        pats, tau, T, cap = None, 0.0, 1.0, None
        print(f"콜드스타트 w=0 draw_prior={d:.3f}")
    if not (dw and mu_d and sd_d):
        Xdall = [pm.draw_row_features(r) for r in scored]
        mu_d, sd_d = pm.standardize(Xdall)
        if not dw:
            dw = [0.0] * len(pm.DRAW_FEATURES)
            db = math.log(max(d, 1e-6) / max(1.0 - d, 1e-6))
        else:
            dw = list(dw)
    else:
        dw = list(dw)

    lr = max(args.lr, 0.0)
    l2 = max(args.l2, 0.0)
    lr_decay = max(args.lr_decay, 0.0)
    use_rollback = bool(args.rollback)
    from collections import deque
    recent = deque(maxlen=500)
    best_roll, best_snap = -1.0, None
    dw_lr = args.dw_lr if args.dw_lr > 0 else lr
    draw_up = max(args.draw_up, 0.0)
    draw_t = args.draw_t
    draw_c = args.draw_c
    walk_hit, walk_tot, walk_ll = 0, 0, 0.0
    stat_hit, stat_tot = 0, 0
    per_season = {}
    picks = []
    wrong = []
    conf_mat = [[0, 0, 0] for _ in range(3)]
    mislead = {}
    conf_hit = 0.0
    w0 = list(w)
    dw0, db0 = list(dw), float(db)
    for n, r in enumerate(scored, 1):
        y = B.parse_label(r)
        x = pm.apply_emphasis(pm.apply_std(pm.row_features(r), mu, sd), e)
        xd = pm.draw_row_features(r)
        if mu_d and sd_d:
            xd = [(a - b) / s_ for a, b, s_ in zip(xd, mu_d, sd_d)]
        lin = pm.full_proba(x, w, hfa, d, dw, db, xd, cap)
        p = pm.apply_temp(pm.blend_proba(lin, x, pats, tau), T)
        dd_now = lin[1]
        if draw_t > 0 and dd_now > draw_t and max(p[0], p[2]) < draw_c:
            pick = 1
        else:
            pick = p.index(max(p))
        walk_hit += (pick == y)
        walk_tot += 1
        walk_ll += -math.log(max(p[y], 1e-12))
        s = str(r.get("season"))
        dse = per_season.setdefault(s, {"n": 0, "hit": 0})
        dse["n"] += 1
        dse["hit"] += (pick == y)
        key = r.get("source_match_id", r.get("id"))
        picks.append([key, pick, 1 if pick == y else 0, round(float(lin[1]), 4), round(float(p[0]), 4)])
        if pick != y:
            cons = sorted(((abs(v * xv), fn, v * xv) for v, xv, fn in zip(w, x, feats)),
                          reverse=True)[:3]
            wrong.append({"key": key, "season": s, "pred": pick, "actual": y,
                          "conf": round(float(max(p)), 4),
                          "draw_miss": y == 1,
                          "top": [[fn, round(float(c), 4)] for _, fn, c in cons]})
            conf_mat[y][pick] += 1
            if cons:
                mislead[cons[0][1]] = mislead.get(cons[0][1], 0) + 1
        else:
            conf_hit += float(max(p))
        sw, shfa = list(w0), float(hfa0)
        slin = pm.full_proba(x, sw, shfa, d, dw0, db0, xd, cap)
        sp = pm.apply_temp(pm.blend_proba(slin, x, pats, tau), T)
        stat_hit += (sp.index(max(sp)) == y)
        stat_tot += 1
        s_lin = max(-30.0, min(30.0, sum(a * b for a, b in zip(x, w)) + hfa))
        ph = 1.0 / (1.0 + math.exp(-s_lin))
        g = -(1.0 - ph) if y == 0 else (ph if y == 2 else 0.0)
        lr_eff = lr / (1.0 + lr_decay * n)
        if g:
            if l2:
                w = [v * (1.0 - lr_eff * l2) for v in w]
            w = [v - lr_eff * g * xv for v, xv in zip(w, x)]
            hfa -= lr_eff * g
        s_dr = max(-30.0, min(30.0, sum(a * b for a, b in zip(xd, dw)) + db))
        dd = 1.0 / (1.0 + math.exp(-s_dr))
        gd = -(1.0 - dd) if y == 1 else dd
        if y == 1 and draw_up != 1.0:
            gd *= draw_up
        dw = [v - dw_lr * gd * xv for v, xv in zip(dw, xd)]
        db -= dw_lr * gd
        if use_rollback:
            recent.append(1 if pick == y else 0)
            if len(recent) == 500:
                roll = sum(recent) / 500
                if roll > best_roll:
                    best_roll = roll
                    best_snap = (list(w), float(hfa), list(dw), float(db), n)
        if n % max(args.log_every, 1) == 0 or n == len(scored):
            print(f"누적 {n}/{len(scored)} walk={walk_hit / walk_tot:.3f}", flush=True)

    walk = {"n": walk_tot, "acc": walk_hit / walk_tot, "ll": walk_ll / walk_tot,
            "seasons": {s: {"n": v["n"], "acc": v["hit"] / v["n"]} for s, v in sorted(per_season.items())}}
    static_acc = stat_hit / stat_tot if stat_tot else 0.0
    print(f"walk acc={walk['acc']:.3f} ll={walk['ll']:.4f} (n={walk_tot})")
    print(f"static base acc={static_acc:.3f} (n={stat_tot})")
    draw_spec = {"t": draw_t, "c": draw_c, "w": args.spec_w}
    if args.spec_w > 0 and picks:
        rec_lookup = {}
        for r in scored:
            rec_lookup[r.get("source_match_id", r.get("id"))] = B.parse_label(r)
        best_key, best_tc = None, (draw_t, draw_c)
        t = 0.15
        while t <= 0.5001:
            for c in (0.35, 0.4, 0.45, 0.5, 0.55):
                h = dr = dy = 0
                for key, _pk, _ht, dd, ph in picks:
                    y = rec_lookup.get(key)
                    if y is None:
                        continue
                    pa = 1.0 - ph - dd
                    base_pk = 0 if ph >= dd and ph >= pa else (1 if dd >= pa else 2)
                    pr = 1 if (dd > round(t, 2) and max(ph, pa) < c) else base_pk
                    h += (pr == y)
                    dy += (y == 1)
                    dr += (y == 1 and pr == 1)
                tot = len(picks)
                key = ((h / tot) + args.spec_w * (dr / dy if dy else 0.0),)
                if best_key is None or key > best_key:
                    best_key, best_tc = key, (round(t, 2), c)
            t = round(t + 0.01, 2)
        draw_spec = {"t": best_tc[0], "c": best_tc[1], "w": args.spec_w}
        print(f"draw_spec t={best_tc[0]} c={best_tc[1]} (walk적합)")
    nw = len(wrong)
    ndm = sum(1 for t in wrong if t["draw_miss"])
    top_mis = sorted(mislead.items(), key=lambda kv: -kv[1])[:5]
    print(f"오답 {nw}건 (무놓침 {ndm}건)")
    print(f"  혼동행렬(실제→예측): 홈{conf_mat[0]} 무{conf_mat[1]} 원{conf_mat[2]}")
    if top_mis:
        print("  최다 오도피처: " + ", ".join(f"{k} {v}건" for k, v in top_mis))
    if nw:
        print(f"  맞힌 자신감 {conf_hit / max(walk_tot - nw, 1):.3f} vs 틀린 자신감 {sum(t['conf'] for t in wrong) / nw:.3f}")
    test = None
    test_recs = []
    _tt, _tc = draw_spec["t"], draw_spec["c"]
    if use_rollback and best_snap is not None:
        w, hfa, dw, db = best_snap[0], best_snap[1], best_snap[2], best_snap[3]
        print(f"롤백: {best_snap[4]}번째 최고(최근500 {best_roll:.3f})로 복원")
    if test_s:
        thit, ttot, tll = 0, 0, 0.0
        tsea = {}
        for r in rows:
            if str(r.get("season")) not in set(test_s):
                continue
            y = B.parse_label(r)
            if y is None:
                continue
            x = pm.apply_emphasis(pm.apply_std(pm.row_features(r), mu, sd), e)
            xd = pm.draw_row_features(r)
            if mu_d and sd_d:
                xd = [(a - b) / s_ for a, b, s_ in zip(xd, mu_d, sd_d)]
            lin = pm.full_proba(x, w, hfa, d, dw, db, xd, cap)
            p = pm.apply_temp(pm.blend_proba(lin, x, pats, tau), T)
            if _tt > 0 and lin[1] > _tt and max(p[0], p[2]) < _tc:
                _pick = 1
            else:
                _pick = p.index(max(p))
            thit += (_pick == y)
            ttot += 1
            tll += -math.log(max(p[y], 1e-12))
            s = str(r.get("season"))
            dse = tsea.setdefault(s, {"n": 0, "hit": 0})
            dse["n"] += 1
            dse["hit"] += (_pick == y)
            test_recs.append([r.get("source_match_id", r.get("id")), _pick, y, round(float(lin[1]), 4)])
        test = {"n": ttot, "acc": thit / ttot if ttot else 0.0, "ll": tll / ttot if ttot else 0.0,
                "seasons": {s: {"n": v["n"], "acc": v["hit"] / v["n"]} for s, v in sorted(tsea.items())},
                "recs": test_recs}
        print(f"test(동결) acc={test['acc']:.3f} ll={test['ll']:.4f} (n={ttot})")

    if base is not None:
        art = dict(base)
    else:
        art = {"T": T, "draw_prior": d, "mu": mu, "sd": sd, "emphasis": e,
               "draw_weights": dw, "draw_bias": db, "draw_mu": mu_d, "draw_sd": sd_d,
               "draw_features": pm.DRAW_FEATURES, "features": feats,
               "patterns": pats, "pattern_tau": tau, "pattern_stats": {},
               "contrib_cap": cap}
    art.update({"weights": list(w), "hfa": float(hfa),
                "train_seasons": sorted(set(seasons)),
                "valid": sorted(set(test_s)),
                "metrics": {"acc": walk["acc"], "ll": walk["ll"], "draw_rec": 0.0},
                "cumulative": {"base_ver": args.base, "lr": lr, "l2": l2,
                               "lr_decay": lr_decay, "rollback": use_rollback,
                               "walk": walk, "static_acc": static_acc, "test": test,
                               "draw_spec": draw_spec, "draw_alert_t": args.alert_t}})
    out_p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.json")
    json.dump(art, open(out_p, "w", encoding="utf-8"))
    with open(os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.cumu.json"), "w", encoding="utf-8") as f:
        json.dump({"league": args.league, "ver": args.ver, "walk": walk,
                   "static_acc": static_acc, "test": test, "picks": picks}, f)
    with open(os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.wrong.json"), "w", encoding="utf-8") as f:
        json.dump({"league": args.league, "ver": args.ver,
                   "summary": {"n_miss": nw, "n_draw_miss": ndm,
                               "confusion": conf_mat,
                               "top_mislead": top_mis},
                   "wrong": wrong}, f)
    print(f"saved {out_p}")


if __name__ == "__main__":
    main()
