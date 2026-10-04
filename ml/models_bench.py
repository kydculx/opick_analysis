#!/usr/bin/env python3
"""모델 벤치마크: baseline / Poisson / Dixon-Coles / LogReg / XGBoost / LSTM.

동일 데이터·동일 분할·동일 지표(acc, logloss, 무승부 리콜)로 비교한다.
DB 없이 ml/.cache만 사용한다.

실행:
    ml/.venv/bin/python ml/models_bench.py --league premier_league [--train-n 8]
    [--xi 0.002] [--seq 5] [--epochs 15] [--only poisson,xgb]

결과: stdout 표 + ml/bench_<league>.json 저장.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")


def known_seasons(league):
    sys.path.insert(0, os.path.join(ROOT, "ml"))
    from grid_console import known_seasons as ks
    return ks(league)


def load_rows(league, seasons):
    import glob as _glob
    want = set(seasons)
    rows, seen = [], set()
    for f in sorted(_glob.glob(os.path.join(ROOT, "ml", ".cache", league + "_*.json"))):
        try:
            if os.path.getsize(f) > 50_000_000:
                continue
            data = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for r in data:
            if str(r.get("season")) in want:
                key = r.get("source_match_id", r.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
    return rows


def parse_label(r):
    try:
        hs, aws = float(r.get("home_score")), float(r.get("away_score"))
    except (TypeError, ValueError):
        return None
    return 0 if hs > aws else (1 if hs == aws else 2)


def split_seasons(full, train_n):
    pool = full[:-1] if len(full) > 1 else list(full)
    n = max(1, min(train_n, len(pool) - 1)) if len(pool) >= 2 else 1
    return pool[:n], [s for s in pool if s not in set(pool[:n])]


def metrics_of(probs_list, labels):
    n = len(labels)
    hit = sum(1 for p, y in zip(probs_list, labels) if p.index(max(p)) == y)
    tot = sum(-math.log(max(p[y], 1e-12)) for p, y in zip(probs_list, labels))
    dm = sum(1 for y in labels if y == 1)
    rec = (sum(1 for p, y in zip(probs_list, labels) if y == 1 and p.index(max(p)) == 1)
           / dm) if dm else 0.0
    return {"n": n, "acc": hit / max(n, 1), "ll": tot / max(n, 1), "draw_rec": rec}


def implied_proba(r):
    try:
        same = r.get("same_odds") or {}
        w = same.get("win_draw_lose") or (same.get("same_odds") or {}).get("win_draw_lose")
        inv = [1.0 / float(w[k]) for k in ("home", "draw", "away")]
    except (TypeError, ValueError, ZeroDivisionError, KeyError, AttributeError):
        return None
    s = sum(inv)
    return [v / s for v in inv]


def build_dc_table(rows, sidx):
    tab = []
    for r in rows:
        try:
            hg, ag = int(r.get("home_score")), int(r.get("away_score"))
        except (TypeError, ValueError):
            continue
        t = sidx.get(str(r.get("season")), 0) * 1000 + int(r.get("round") or 0)
        tab.append((str(r.get("home_team")), str(r.get("away_team")), hg, ag, t))
    return tab


def fit_dc(tab, xi=0.002, use_rho=True):
    import numpy as _np
    from scipy.optimize import minimize
    teams = sorted({h for h, _, _, _, _ in tab} | {a for _, a, _, _, _ in tab})
    tix = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    Tmax = max(t for _, _, _, _, t in tab)
    HG = _np.array([hg for _, _, hg, _, _ in tab], dtype=float)
    AG = _np.array([ag for _, _, _, ag, _ in tab], dtype=float)
    HI = _np.array([tix[h] for h, _, _, _, _ in tab], dtype=int)
    AI = _np.array([tix[a] for _, a, _, _, _ in tab], dtype=int)
    WT = _np.exp(-xi * (Tmax - _np.array([t for _, _, _, _, t in tab], dtype=float)))

    def unpack(v):
        a = _np.zeros(nt)
        d = _np.zeros(nt)
        a[:nt - 1] = v[0:nt - 1]
        d[:nt - 1] = v[nt - 1:2 * (nt - 1)]
        a[nt - 1] = -a[:nt - 1].sum()
        d[nt - 1] = -d[:nt - 1].sum()
        home, rho = v[2 * (nt - 1)], (v[2 * (nt - 1) + 1] if use_rho else 0.0)
        return a, d, home, rho

    def nll(v):
        a, d, home, rho = unpack(v)
        lam = _np.exp(a[HI] + d[AI] + home)
        mu = _np.exp(a[AI] + d[HI])
        ll = (-lam + HG * _np.log(_np.maximum(lam, 1e-12))
              - mu + AG * _np.log(_np.maximum(mu, 1e-12)))
        if use_rho:
            tau = _np.ones_like(lam)
            m00 = (HG == 0) & (AG == 0)
            m01 = (HG == 0) & (AG == 1)
            m10 = (HG == 1) & (AG == 0)
            m11 = (HG == 1) & (AG == 1)
            tau = _np.where(m00, 1.0 - lam * mu * rho,
                    _np.where(m01, 1.0 + lam * rho,
                    _np.where(m10, 1.0 + mu * rho,
                    _np.where(m11, 1.0 - rho, 1.0))))
            tau = _np.maximum(tau, 1e-12)
            ll = ll + _np.log(tau)
        wll = float((WT * ll).sum() / WT.sum())
        return -wll

    x0 = _np.zeros(2 * (nt - 1) + 2)
    bounds = [(-3, 3)] * (2 * (nt - 1)) + [(-1, 1), (-0.3, 0.3)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200})
    a, d, home, rho = unpack(res.x)
    return {"teams": teams, "att": a.tolist(), "def": d.tolist(),
            "home": float(home), "rho": float(rho), "fun": float(res.fun)}


def dc_predict(model, home, away, max_goals=10):
    import numpy as _np
    from math import factorial
    a = dict(zip(model["teams"], model["att"]))
    d = dict(zip(model["teams"], model["def"]))
    lam = math.exp(a.get(home, 0.0) + d.get(away, 0.0) + model["home"])
    mu = math.exp(a.get(away, 0.0) + d.get(home, 0.0))
    rho = model["rho"]
    grid = _np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        px = lam ** x * math.exp(-lam) / factorial(x)
        for y in range(max_goals + 1):
            tau = 1.0
            if x == 0 and y == 0:
                tau = 1.0 - lam * mu * rho
            elif x == 0 and y == 1:
                tau = 1.0 + lam * rho
            elif x == 1 and y == 0:
                tau = 1.0 + mu * rho
            elif x == 1 and y == 1:
                tau = 1.0 - rho
            grid[x, y] = px * (mu ** y * math.exp(-mu) / factorial(y)) * max(tau, 1e-12)
    grid /= grid.sum()
    hw = float(grid[_np.tril_indices(max_goals + 1, -1)].sum())
    aw = float(grid[_np.triu_indices(max_goals + 1, 1)].sum())
    return [hw, max(0.0, 1.0 - hw - aw), aw]


def feat_matrix(rows):
    sys.path.insert(0, os.path.join(ROOT, "ml"))
    import permatch_mode as pm
    X, y = [], []
    for r in rows:
        lab = parse_label(r)
        if lab is None:
            continue
        X.append(pm.row_features(r))
        y.append(lab)
    return X, y


def run_logreg(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xva)).tolist()


def run_xgb(Xtr, ytr, Xva, yva):
    import numpy as _np
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                        objective="multi:softprob", num_class=3,
                        n_jobs=4, random_state=0, tree_method="hist")
    clf.fit(_np.asarray(Xtr, dtype=float), _np.asarray(ytr))
    return clf.predict_proba(_np.asarray(Xva, dtype=float)).tolist()


def build_sequences(rows, sidx, seq_len=5):
    order = sorted(range(len(rows)),
                   key=lambda i: (sidx.get(str(rows[i].get("season")), 0),
                                  int(rows[i].get("round") or 0),
                                  rows[i].get("id") or 0))
    hist = {}
    Xs, Xa, y = [], [], []
    for i in order:
        r = rows[i]
        lab = parse_label(r)
        if lab is None:
            continue
        h, a = str(r.get("home_team")), str(r.get("away_team"))
        try:
            hg, ag = int(r.get("home_score")), int(r.get("away_score"))
        except (TypeError, ValueError):
            continue
        hs = hist.get(h, [])
        aws = hist.get(a, [])
        def enc(seq, is_home):
            pad = [[0.0, 0.0, 0.0, 0.0]] * max(0, seq_len - len(seq))
            return (pad + seq[-seq_len:]) if seq else pad + seq
        Xs.append(enc(hs, True))
        Xa.append(enc(aws, False))
        y.append(lab)
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)
        hist.setdefault(h, []).append([float(hg), float(ag), float(hp), 1.0])
        hist.setdefault(a, []).append([float(ag), float(hg), float(ap), 0.0])
    return Xs, Xa, y


def run_lstm(Xstr, Xsa, ytr, Xsva, Xsav, yva, seq_len=5, epochs=15, hidden=32):
    import numpy as _np
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    Xtr = _np.asarray(Xstr, dtype=_np.float32)
    Xva = _np.asarray(Xsva, dtype=_np.float32)
    mu, sd = Xtr.reshape(-1, 4).mean(0), Xtr.reshape(-1, 4).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (_np.asarray(Xsva, dtype=_np.float32) - mu) / sd
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(4, hidden, batch_first=True)
            self.fc = nn.Linear(hidden * 2, 3)

        def forward(self, h, a):
            _, (hn, _) = self.lstm(h)
            _, (an, _) = self.lstm(a)
            return self.fc(torch.cat([hn[-1], an[-1]], dim=1))

    m = SeqModel()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    Ht = torch.tensor(Xtr)
    At = torch.tensor(((_np.asarray(Xsa, dtype=_np.float32) - mu) / sd).tolist())
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = m(Ht, At)
        loss = lossf(out, ytr_t)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        Hv = torch.tensor(Xva)
        Av = torch.tensor(((_np.asarray(Xsav, dtype=_np.float32) - mu) / sd).tolist())
        logits = m(Hv, Av)
        P = torch.softmax(logits, dim=1).tolist()
    return P


def main():
    ap = argparse.ArgumentParser(description="모델 벤치마크")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--xi", type=float, default=0.002)
    ap.add_argument("--seq", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    full = known_seasons(args.league)
    train_s, valid_s = split_seasons(full, args.train_n)
    print(f"league={args.league} train={train_s} valid={valid_s}")
    rows = load_rows(args.league, train_s + valid_s)
    sidx = {s: i for i, s in enumerate(full)}
    tr = [r for r in rows if str(r.get("season")) in set(train_s)]
    va = [r for r in rows if str(r.get("season")) in set(valid_s)]
    va_scored = [r for r in va if parse_label(r) is not None]
    print(f"train rows={len(tr)} valid rows={len(va_scored)}")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results = {}

    def put(name, probs):
        if only and name not in only:
            return
        m = metrics_of(probs, [parse_label(r) for r in va_scored])
        results[name] = m
        print(f"{name:12} acc={m['acc']:.3f} ll={m['ll']:.4f} rec={m['draw_rec']:.3f} (n={m['n']})")

    put("home", [[1.0, 0.0, 0.0]] * len(va_scored))
    put("market", [implied_proba(r) or [1.0 / 3] * 3 for r in va_scored])

    tab_tr = build_dc_table(tr, sidx)
    if (not only or "poisson" in only or "dixon" in only) and tab_tr:
        m0 = fit_dc(tab_tr, xi=args.xi, use_rho=False)
        put("poisson", [dc_predict(m0, str(r.get("home_team")), str(r.get("away_team")))
                        for r in va_scored])
        m1 = fit_dc(tab_tr, xi=args.xi, use_rho=True)
        print(f"  dc rho={m1['rho']:.3f} home={m1['home']:.3f}")
        put("dixon", [dc_predict(m1, str(r.get("home_team")), str(r.get("away_team")))
                      for r in va_scored])

    Xtr, ytr = feat_matrix(tr)
    Xva, yva2 = feat_matrix(va_scored)
    if not only or "logreg" in only:
        put("logreg", run_logreg(Xtr, ytr, Xva, yva2))
    if not only or "xgb" in only:
        put("xgb", run_xgb(Xtr, ytr, Xva, yva2))
    if not only or "lstm" in only:
        allr = sorted((r for r in tr + va if parse_label(r) is not None),
                      key=lambda r: (sidx.get(str(r.get("season")), 0),
                                     int(r.get("round") or 0),
                                     r.get("id") or 0))
        Xs, Xa, yy = build_sequences(allr, sidx, args.seq)
        cut = sum(1 for r in allr if str(r.get("season")) in set(train_s))
        Pv = run_lstm(Xs[:cut], Xa[:cut], yy[:cut], Xs[cut:], Xa[cut:], yy[cut:],
                      args.seq, args.epochs, args.hidden)
        put("lstm", Pv)

    out = {"league": args.league, "train": train_s, "valid": valid_s, "results": results}
    p = os.path.join(ROOT, "ml", f"bench_{args.league}.json")
    json.dump(out, open(p, "w"))
    print(f"saved {p}")


if __name__ == "__main__":
    main()
