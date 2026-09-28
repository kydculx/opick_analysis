"""Permatch 학습 단일 진입점 (autotune + 단발).

사용법:
  # 이어하기 (기본): ml/permatch/{league}_{ver}.json 있으면 최고점으로 로드 후 계속 탐색
  ml/.venv/bin/python ml/permatch_mode.py --league premier_league \\
    --train 2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022 \\
    --valid 2022-2023 --ver match-2016-2017-2017-2018-2018-2019-2019-2020-2020-2021-2021-2022-tune \\
    --trials 10000 --jobs 4

  # 처음부터 (--new): 기존 아티팩트 무시하고 새로 탐색
  ml/.venv/bin/python ml/permatch_mode.py --league premier_league \\
    --train 2016-2017,2017-2018 --valid 2018-2019 --ver match-v1 --trials 20 --new

  # 단발 학습 (빠른 1회 fit, 기존과 비교해 좋으면 저장)
  ml/.venv/bin/python ml/permatch_mode.py --league k_league_1 \\
    --train 2025 --valid 2024 --ver match-2025-opt --fast
  ml/.venv/bin/python ml/permatch_mode.py --league k_league_1 \\
    --train 2025 --valid 2024 --ver match-2025-opt --fast --new   # 비교 없이 덮어쓰기

  # 예측 (학습 없이 아티팩트로 확률 확인)
  ml/.venv/bin/python ml/permatch_mode.py --league k_league_1 \\
    --predict 2026 --ver match-v1 --mode predict

  # 자동조절 (웹과 동일: 순차 1바퀴 후 랜덤워크, 개선될 때마다 저장, Ctrl+C로 중지)
  ml/.venv/bin/python ml/permatch_mode.py --mode auto --league k_league_1 \\
    --ver match-2016-2017-2018-2019-2020-2021-tune --tune 2022,2023,2024,2025

  # 전수탐색 (주판 순서로 모든 조합, 체크포인트 이어하기, Ctrl+C로 중지)
  ml/.venv/bin/python ml/permatch_mode.py --mode grid --league k_league_1 \\
    --ver match-2016-2017-2018-2019-2020-2021-tune --grid-step 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time


_START = time.time()

_LEAGUE_TAG = ""

def league_tag(code: str) -> str:
    return {"premier_league": "EPL", "k_league_1": "K1", "k_league_2": "K2",
            "j1_league": "J1", "bundesliga": "BL", "laliga": "LL",
            "ligue_1": "L1", "serie_a": "SA", "eredivisie": "ERE",
            "mls": "MLS", "a_league": "ALE"}.get(code or "", "")


def log(msg):
    tag = f"[{_LEAGUE_TAG}] " if _LEAGUE_TAG else ""
    print(f"[{time.time() - _START:8.1f}s] {tag}{msg}", flush=True)


FEATURES13 = ["rank", "power", "hstr", "cond", "att", "def", "val",
              "form5", "h2h5", "avg_goals", "avg_conceded", "avg_poss", "market"]

FIVE = ["rank", "power", "val", "form5", "market"]
FIVE_IDX = [FEATURES13.index(n) for n in FIVE]

FEATURES = list(FEATURES13)
SEL = None

DRAW_FEATURES = ["poisson_draw", "rank_gap", "total_goals", "market_draw"]

DRAW_ACC_TOL = 0.02


def draw_accept(w, hfa, e, dw, db, Dv, Vn, vy, d):
    new_acc = batch_acc(Vn, vy, w, hfa, d, 1.0, e, dw, db, Dv)
    new_ll = batch_ll(Vn, vy, w, hfa, d, 1.0, 0.0, e, dw, db, Dv)
    old_acc = batch_acc(Vn, vy, w, hfa, d, 1.0, e)
    old_ll = batch_ll(Vn, vy, w, hfa, d, 1.0, 0.0, e)
    keep = bool(new_ll < old_ll - 1e-4 and new_acc >= old_acc - DRAW_ACC_TOL)
    return keep, new_acc, new_ll, old_acc, old_ll


def tune_T_valid(w, hfa, e, dw, db, Dv, Vn, vy, d, T0):
    if dw is None:
        best_T, best_key = T0, None
        t = 0.5
        while t <= 10.001:
            key = (batch_acc(Vn, vy, w, hfa, d, t, e),
                   -batch_ll(Vn, vy, w, hfa, d, t, 0.0, e))
            if best_key is None or key > best_key:
                best_key, best_T = key, t
            t += 0.1
        return best_T
    base_acc = batch_acc(Vn, vy, w, hfa, d, T0, e, dw, db, Dv)
    best_T = T0
    best_ll = batch_ll(Vn, vy, w, hfa, d, T0, 0.0, e, dw, db, Dv)
    t = 0.5
    while t <= 10.001:
        a = batch_acc(Vn, vy, w, hfa, d, t, e, dw, db, Dv)
        ll = batch_ll(Vn, vy, w, hfa, d, t, 0.0, e, dw, db, Dv)
        if a >= base_acc - DRAW_ACC_TOL and ll < best_ll - 1e-6:
            best_ll, best_T = ll, t
        t += 0.1
    return best_T


# ---------------------------------------------------------------- core math
def batch_proba(X, w, hfa: float, d: float, T: float = 1.0, e=None, dw=None, db: float = 0.0,
                Xd=None, cap=None):
    import numpy as np
    Xn = np.asarray(X, dtype=float)
    if e is not None:
        Xn = Xn * np.asarray(e, dtype=float)
    wv = np.asarray(w, dtype=float)
    if cap:
        s = np.clip((cap * np.tanh(Xn * wv / cap)).sum(axis=1) + hfa, -30.0, 30.0)
    else:
        s = np.clip(Xn @ wv + hfa, -30.0, 30.0)
    ph = 1.0 / (1.0 + np.exp(-s))
    if dw is not None:
        Dn = np.asarray(Xd if Xd is not None else X, dtype=float)
        ds = np.clip(Dn @ np.asarray(dw, dtype=float) + db, -30.0, 30.0)
        dd = 1.0 / (1.0 + np.exp(-ds))
    else:
        dd = np.full_like(ph, d)
    P = np.stack([ph * (1 - dd), dd, (1 - ph) * (1 - dd)], axis=1)
    if T != 1.0:
        L = np.log(np.maximum(P, 1e-9)) / T
        L = L - L.max(axis=1, keepdims=True)
        E = np.exp(L)
        P = E / E.sum(axis=1, keepdims=True)
    return P


def batch_ll(X, y, w, hfa: float, d: float, T: float = 1.0, l2: float = 0.0, e=None, dw=None, db=0.0, Xd=None, cap=None):
    import numpy as np
    P = batch_proba(X, w, hfa, d, T, e, dw, db, Xd, cap)
    yy = np.asarray(y, dtype=int)
    ll = float(-np.log(np.maximum(P[np.arange(len(yy)), yy], 1e-12)).mean())
    return ll + l2 * float((np.asarray(w, dtype=float) ** 2).sum())


def batch_acc(X, y, w, hfa: float, d: float, T: float = 1.0, e=None, dw=None, db=0.0, Xd=None, cap=None):
    import numpy as np
    P = batch_proba(X, w, hfa, d, T, e, dw, db, Xd, cap)
    home, dr, away = P[:, 0], P[:, 1], P[:, 2]
    pick = np.where((home >= dr) & (home >= away), 0, np.where(dr >= away, 1, 2))
    yy = np.asarray(y, dtype=int)
    return float((pick == yy).mean())


def parse_score(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def implied_probs(wdl: dict | None):
    if not wdl:
        return None
    try:
        inv = [1.0 / float(wdl[k]) for k in ("home", "draw", "away")]
    except (TypeError, ValueError, ZeroDivisionError, KeyError):
        return None
    s = sum(inv)
    return [v / s for v in inv]


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


def fnum(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def net_wdl(v):
    if not v:
        return 0.0
    return fnum(v.get("win")) - fnum(v.get("loss"))


def row_features(m: dict) -> list:
    st = m.get("strength") or {}
    sh, sa = st.get("home") or {}, st.get("away") or {}
    rf = m.get("recent_form") or {}
    rfh = {f"recent_{i}": (rf.get("home") or {}).get(f"recent_{i}") or {} for i in range(1, 6)}
    rfa = {f"recent_{i}": (rf.get("away") or {}).get(f"recent_{i}") or {} for i in range(1, 6)}

    def side_net(get):
        return sum(net_wdl(get(i)) for i in range(1, 6))

    form_net = side_net(lambda i: rfh[f"recent_{i}"]) - side_net(lambda i: rfa[f"recent_{i}"])
    hh = m.get("head_to_head") or {}
    h2h_net = sum(net_wdl(hh.get(f"recent_{i}")) for i in range(1, 6))
    ti = (m.get("team_info") or {}).get("avg_stats") or {}
    th, ta = ti.get("home") or {}, ti.get("away") or {}
    ph, _, pa = implied_probs(same_odds_wdl(m.get("same_odds"))) or (math.nan, math.nan, math.nan)
    market = (ph - pa) if not (isinstance(ph, float) and math.isnan(ph)) else 0.0
    x = [
        fnum(m.get("home_rank")) - fnum(m.get("away_rank")),
        fnum(sh.get("power")) - fnum(sa.get("power")),
        fnum(sh.get("head_to_head")) - fnum(sa.get("head_to_head")),
        fnum(sh.get("condition")) - fnum(sa.get("condition")),
        fnum(sh.get("attack")) - fnum(sa.get("attack")),
        fnum(sh.get("defense")) - fnum(sa.get("defense")),
        fnum(sh.get("value")) - fnum(sa.get("value")),
        form_net,
        h2h_net,
        fnum(th.get("goals")) - fnum(ta.get("goals")),
        fnum(th.get("conceded")) - fnum(ta.get("conceded")),
        fnum(th.get("possession")) - fnum(ta.get("possession")),
        market,
    ]
    return x if SEL is None else [x[i] for i in SEL]


def apply_emphasis(x, e):
    return [a * b for a, b in zip(x, e)]


def predict_proba(x, w, hfa: float, d: float):
    s = sum(a * b for a, b in zip(x, w)) + hfa
    s = max(-30.0, min(30.0, s))
    ph = 1.0 / (1.0 + math.exp(-s))
    return [ph * (1 - d), d, (1 - ph) * (1 - d)]


def draw_prob(x, dw, db: float):
    s = sum(a * b for a, b in zip(x, dw)) + db
    s = max(-30.0, min(30.0, s))
    return 1.0 / (1.0 + math.exp(-s))


def contrib_score(x, w, hfa: float, cap=None):
    if cap:
        return sum(cap * math.tanh(v * c / cap) for v, c in zip(x, w)) + hfa
    return sum(a * b for a, b in zip(x, w)) + hfa


def full_proba(x, w, hfa: float, d: float, dw=None, db: float = 0.0, xd=None, cap=None):
    s = max(-30.0, min(30.0, contrib_score(x, w, hfa, cap)))
    ph = 1.0 / (1.0 + math.exp(-s))
    dd = draw_prob(xd if xd is not None else x, dw, db) if dw is not None else d
    return [ph * (1 - dd), dd, (1 - ph) * (1 - dd)]


def tune_contrib(Xe_tr, ytr, Xe_va, yva, w, hfa, d, dw=None, db: float = 0.0,
                 Xde_tr=None, Xde_va=None):
    import numpy as _np
    Xtr = _np.asarray(Xe_tr, dtype=float)
    ytr = _np.asarray(ytr, dtype=int)
    has_va = len(Xe_va) > 0 and len(yva) > 0
    Xva = _np.asarray(Xe_va, dtype=float).reshape(-1, len(w)) if has_va else _np.zeros((0, len(w)))
    yva = _np.asarray(yva, dtype=int)

    def key_of(C, T):
        if has_va:
            return (batch_acc(Xva, yva, w, hfa, d, T, None, dw, db, Xde_va, C),
                    -batch_ll(Xva, yva, w, hfa, d, T, 0.0, None, dw, db, Xde_va, C))
        return (-batch_ll(Xtr, ytr, w, hfa, d, T, 0.0, None, dw, db, Xde_tr, C),)

    best_key, best_T, best_C = key_of(None, 1.0), 1.0, None
    for C in (0.5, 1.0, 1.5, 2.0, 3.0, None):
        t = 0.5
        while t <= 10.001:
            key = key_of(C, round(t, 1))
            if key > best_key:
                best_key, best_T, best_C = key, round(t, 1), C
            t += 0.1
    return best_C, best_T


def kmeans(X, k: int, seed: int = 0, iters: int = 60):
    import numpy as _np
    n = len(X)
    rng = _np.random.RandomState(seed)
    C = _np.asarray(X[rng.choice(n, k, replace=False)], dtype=float)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        new = []
        for j in range(k):
            m = X[a == j]
            new.append(m.mean(0) if len(m) else X[rng.choice(n)])
        new = _np.asarray(new, dtype=float)
        if _np.allclose(new, C):
            break
        C = new
    return C


def build_patterns(feats, labels, mu, sd, e, k: int = 5, seed: int = 11):
    import numpy as _np
    Xn = _np.asarray([apply_emphasis(apply_std(x, mu, sd), e) for x in feats], dtype=float)
    y = _np.asarray(labels, dtype=int)
    out: dict = {}
    spread: list = []
    for ci, cname in ((0, "home"), (1, "draw"), (2, "away")):
        Xc = Xn[y == ci]
        if len(Xc) == 0:
            out[cname] = [[0.0] * Xn.shape[1] for _ in range(k)]
            continue
        kk = max(1, min(k, len(Xc)))
        C = kmeans(Xc, kk, seed + ci)
        if kk < k:
            C = _np.vstack([C] + [C[:1]] * (k - kk))
        out[cname] = [list(map(float, row)) for row in C]
        dmin = ((Xc[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(1)
        spread.extend(float(v) for v in _np.sqrt(dmin))
    tau = float(sum(spread) / max(len(spread), 1)) or 1.0
    order = ["home", "draw", "away"]
    all_c = _np.asarray([c for cname in order for c in out[cname]], dtype=float)
    d_all = ((Xn[:, None, :] - all_c[None, :, :]) ** 2).sum(-1).argmin(1)
    stats: dict = {}
    for ci, cname in enumerate(order):
        ranked = []
        for j in range(k):
            members = y[d_all == ci * k + j]
            freq = int(len(members))
            acc = float((members == ci).sum() / freq) if freq else 0.0
            ranked.append((freq, acc, j))
        ranked.sort(key=lambda t: (-t[0], -t[1]))
        out[cname] = [out[cname][j] for _, _, j in ranked]
        stats[cname] = [{"rank": r + 1, "freq": f, "acc": round(a, 4)} for r, (f, a, _) in enumerate(ranked)]
    return out, tau, stats


def pattern_proba(x, patterns, tau: float):
    scores = []
    for cname in ("home", "draw", "away"):
        best = min(sum((a - b) ** 2 for a, b in zip(x, cen)) for cen in patterns[cname])
        scores.append(-math.sqrt(best) / tau)
    mx = max(scores)
    ex = [math.exp(s - mx) for s in scores]
    s = sum(ex)
    return [v / s for v in ex]


def blend_proba(p_lin, x, patterns, tau: float, alpha: float = 0.5):
    if not patterns or not tau:
        return p_lin
    pp = pattern_proba(x, patterns, tau)
    return [alpha * b + (1.0 - alpha) * a for a, b in zip(p_lin, pp)]


def draw_row_features(m: dict) -> list:
    ti = (m.get("team_info") or {}).get("avg_stats") or {}
    th, ta = ti.get("home") or {}, ti.get("away") or {}
    xg_h = max(0.05, (fnum(th.get("goals")) + fnum(ta.get("conceded"))) / 2)
    xg_a = max(0.05, (fnum(ta.get("goals")) + fnum(th.get("conceded"))) / 2)
    import math as _m
    pd = sum(_m.exp(-xg_h) * xg_h ** i / _m.factorial(i) *
             _m.exp(-xg_a) * xg_a ** i / _m.factorial(i) for i in range(11))
    try:
        rg = abs(float(m.get("home_rank")) - float(m.get("away_rank")))
    except (TypeError, ValueError):
        rg = 0.0
    ph, pd_mkt, pa = implied_probs(same_odds_wdl(m.get("same_odds"))) or (math.nan, math.nan, math.nan)
    return [pd, rg, xg_h + xg_a,
            pd_mkt if not (isinstance(pd_mkt, float) and math.isnan(pd_mkt)) else 0.0]


def standardize(Xtr):
    mu = [sum(c) / len(c) for c in zip(*Xtr)]
    sd = []
    for j in range(len(mu)):
        v = sum((r[j] - mu[j]) ** 2 for r in Xtr) / len(Xtr)
        sd.append(math.sqrt(v) if v > 1e-9 else 1.0)
    return mu, sd


def apply_std(x, mu, sd):
    return [(a - b) / s for a, b, s in zip(x, mu, sd)]


def apply_temp(probs, T: float):
    logits = [math.log(max(p, 1e-9)) / T for p in probs]
    mx = max(logits)
    ex = [math.exp(l - mx) for l in logits]
    s = sum(ex)
    return [e / s for e in ex]


def dataset_metrics(rows):
    n = len(rows)
    if not n:
        return 0.0, 0.0
    ll = sum(-math.log(max(r["probs"][r["label"]], 1e-12)) for r in rows) / n
    acc = sum(1 for r in rows if r["probs"].index(max(r["probs"])) == r["label"]) / n
    return acc, ll


def metrics_of(w, hfa, T, d, X, y):
    n = len(X)
    if not n:
        return 0.0, 0.0
    return batch_acc(X, y, w, hfa, d, T), batch_ll(X, y, w, hfa, d, T)


# ---------------------------------------------------------------- io
def model_path(league: str, ver: str) -> str:
    return f"ml/permatch/{league}_{ver}.json"


def cache_path(league: str, seasons: list[str]):
    return os.path.join(os.path.dirname(__file__), ".cache",
                        f"{league}_{'-'.join(sorted(set(seasons)))}.json")


def load_seasons(league: str, seasons: list[str], use_cache: bool = True):
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    cp = cache_path(league, seasons)
    if use_cache and os.path.exists(cp):
        with open(cp) as f:
            cached = json.load(f)
        log(f"cache hit: {cp} ({len(cached)} rows, DB 조회 생략)")
        return sb, cached
    rows = []
    for s in seasons:
        off = 0
        while True:
            res = (sb.table("soccer_matches")
                   .select("id,league_code,season,round,home_team,home_rank,away_team,away_rank,"
                           "home_score,away_score,same_odds,source_match_id,"
                           "team_info,recent_form,head_to_head,strength")
                   .eq("league_code", league).eq("season", s).order("round").order("id")
                   .range(off, off + 999).execute())
            rows += res.data or []
            if not res.data or len(res.data) < 1000:
                break
            off += 1000
    if use_cache:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "w") as f:
            json.dump(rows, f)
    return sb, rows


def tune_weights(league: str, ver: str, tune_s: set, max_sweeps: int = 6,
                 use_cache: bool = True):
    art = json.load(open(model_path(league, ver)))
    w, hfa = list(art["weights"]), art["hfa"]
    d = art["draw_prior"]
    mu, sd = art["mu"], art["sd"]
    e = art.get("emphasis", [1.0] * len(w))
    overlap = set(art.get("train_seasons", [])) & set(tune_s)
    if overlap:
        log(f"경고: 학습시즌 포함됨 {sorted(overlap)} (미학습 권장)")
    _, rows = load_seasons(league, sorted(tune_s), use_cache=use_cache)
    feats, labels, ssn = [], [], []
    for r in rows:
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        if hs is None or aws is None:
            continue
        feats.append(apply_emphasis(apply_std(row_features(r), mu, sd), e))
        labels.append(0 if hs > aws else (1 if hs == aws else 2))
        ssn.append(str(r.get("season")))
    if not feats:
        log("조절용 데이터 없음")
        return False
    import numpy as _np
    Xn = _np.asarray(feats, dtype=float)
    yn = _np.asarray(labels, dtype=int)

    def key_of(wv, hv):
        return (batch_acc(Xn, yn, wv, hv, d), -batch_ll(Xn, yn, wv, hv, d))

    def season_acc(wv, hv):
        out = {}
        for s in sorted(set(ssn)):
            m = _np.asarray([a == s for a in ssn])
            if m.sum() == 0:
                continue
            out[s] = batch_acc(Xn[m], yn[m], wv, hv, d)
        return out

    base = key_of(w, hfa)
    base_se = season_acc(w, hfa)
    log(f"조절 전: acc={base[0]:.3f} ll={-base[1]:.4f} " +
        " ".join(f"{s}={base_se[s]:.3f}" for s in sorted(base_se)))
    best = base
    for sw in range(max_sweeps):
        improved = False
        for j in list(range(len(w))) + ["hfa"]:
            for step in (0.01, 0.05, 0.2):
                for sign in (1.0, -1.0):
                    cw = list(w)
                    ch = hfa + sign * step if j == "hfa" else hfa
                    if j != "hfa":
                        cw[j] += sign * step
                    key = key_of(cw, ch)
                    if key > best:
                        best, w, hfa = key, cw, ch
                        improved = True
        if not improved:
            break
    fin_se = season_acc(w, hfa)
    log(f"조절 후: acc={best[0]:.3f} ll={-best[1]:.4f} " +
        " ".join(f"{s}={fin_se[s]:.3f}" for s in sorted(fin_se)))
    if best > base:
        art["weights"] = w
        art["hfa"] = hfa
        save_artifact(league, ver, art)
        log(f"개선됨 → 저장 (acc {base[0]:.3f} → {best[0]:.3f})")
        return True
    log(f"개선 없음 → 유지 (acc {base[0]:.3f})")
    return False


def all_nontrain_seasons(league: str, train_s: set) -> set:
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    found: set = set()
    off = 0
    while True:
        res = (sb.table("soccer_matches").select("season")
               .eq("league_code", league).range(off, off + 999).execute())
        if not res.data:
            break
        for r in res.data:
            if r.get("season") is not None:
                found.add(str(r["season"]))
        if len(res.data) < 1000:
            break
        off += 1000
    return found - set(train_s)


def auto_tune(league: str, ver: str, tune_s: set, seq_sweeps: int = 1, random_rounds: int = 0,
              max_minutes: float = 0.0, noise: float = 0.1, hfa_noise: float = 0.05,
              wmin: float = -2.0, wmax: float = 2.0, log_every: int = 200,
              seed: int = 7, fresh: bool = False, use_cache: bool = True):
    import numpy as _np
    p = model_path(league, ver)
    if not os.path.exists(p):
        log(f"아티팩트 없음: {p} (먼저 학습 필요)")
        return False
    art = json.load(open(p))
    nw = len(FEATURES)
    d = art["draw_prior"]
    mu, sd = art["mu"], art["sd"]
    e = art.get("emphasis", [1.0] * nw)
    pats = art.get("patterns")
    tau = art.get("pattern_tau", 0.0) or 0.0
    cap = art.get("contrib_cap")
    overlap = set(art.get("train_seasons", [])) & set(tune_s)
    if overlap:
        log(f"경고: 학습시즌 포함됨 {sorted(overlap)} (미학습 권장)")
    if fresh:
        w, hfa = [0.0] * nw, 0.0
        log("--new: 가중치 0부터 시작 (mu/sd/e/패턴은 아티팩트 유지)")
    else:
        w, hfa = list(art["weights"]), art["hfa"]
    dw = art.get("draw_weights")
    db = art.get("draw_bias", 0.0) or 0.0
    mu_d, sd_d = art.get("draw_mu"), art.get("draw_sd")
    use_draw = bool(dw and mu_d and sd_d)
    if not tune_s:
        tune_s = all_nontrain_seasons(league, set(art.get("train_seasons", [])))
        log(f"조절 시즌 자동선택(미학습): {sorted(tune_s)}")
    _, rows = load_seasons(league, sorted(tune_s), use_cache=use_cache)
    feats, labels, ssn, xds = [], [], [], []
    for r in rows:
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        if hs is None or aws is None:
            continue
        feats.append(apply_emphasis(apply_std(row_features(r), mu, sd), e))
        labels.append(0 if hs > aws else (1 if hs == aws else 2))
        ssn.append(str(r.get("season")))
        if use_draw:
            xds.append([(a - b) / s for a, b, s in zip(draw_row_features(r), mu_d, sd_d)])
    if not feats:
        log("조절용 데이터 없음")
        return False
    Xn = _np.asarray(feats, dtype=float)
    yn = _np.asarray(labels, dtype=int)
    Xd = _np.asarray(xds, dtype=float) if use_draw else None
    log("draw 모델 평가 사용" if use_draw else "draw_prior 평가 사용")
    PP = None
    if pats and tau and tau > 0:
        try:
            PP = _np.asarray([pattern_proba(x, pats, tau) for x in feats], dtype=float)
            log(f"패턴 블렌드 평가 사용 (tau={tau:.3f})")
        except (KeyError, TypeError, ValueError):
            PP = None

    def acc_of(wv, hv):
        Plin = batch_proba(Xn, wv, hv, d, 1.0, None, dw if use_draw else None,
                           db if use_draw else 0.0, Xd if use_draw else None, cap)
        P = 0.5 * Plin + 0.5 * PP if PP is not None else Plin
        home, dr, aw = P[:, 0], P[:, 1], P[:, 2]
        pick = _np.where((home >= dr) & (home >= aw), 0, _np.where(dr >= aw, 1, 2))
        return float((pick == yn).mean())

    def season_acc(wv, hv):
        out = {}
        for s in sorted(set(ssn)):
            m = _np.asarray([a == s for a in ssn])
            if m.sum() == 0:
                continue
            Plin = batch_proba(Xn[m], wv, hv, d, 1.0, None, dw if use_draw else None,
                               db if use_draw else 0.0, Xd[m] if use_draw else None, cap)
            P = 0.5 * Plin + 0.5 * PP[m] if PP is not None else Plin
            home, dr, aw = P[:, 0], P[:, 1], P[:, 2]
            pick = _np.where((home >= dr) & (home >= aw), 0, _np.where(dr >= aw, 1, 2))
            out[s] = float((pick == yn[m]).mean())
        return out

    def clamp(v):
        return min(wmax, max(wmin, round(v, 4)))

    def persist():
        try:
            cur = json.load(open(model_path(league, ver)))
            cw, chh = cur.get("weights"), cur.get("hfa")
            if isinstance(cw, list) and len(cw) == nw and isinstance(chh, (int, float)):
                facc = acc_of(cw, chh)
                if facc > best:
                    return ("adopted", list(cw), float(chh), facc)
        except (OSError, ValueError):
            pass
        try:
            cur0 = json.load(open(model_path(league, ver)))
            if cur0.get("weights") == list(bestW) and cur0.get("hfa") == bestHfa:
                return ("same", list(bestW), bestHfa, best)
        except (OSError, ValueError):
            pass
        art["weights"] = list(bestW)
        art["hfa"] = bestHfa
        save_artifact(league, ver, art)
        return ("saved", list(bestW), bestHfa, best)

    def commit(tag):
        nonlocal w, hfa, best, bestW, bestHfa, improved_any
        improved_any = True
        st, bw, bh, ba = persist()
        if st == "adopted":
            best, w, hfa, bestW, bestHfa = ba, bw, bh, list(bw), bh
        ise = season_acc(bestW, bestHfa)
        log(f"{tag} acc={best:.3f} " +
            " ".join(f"{s}={ise[s]:.3f}" for s in sorted(ise)) +
            (" → 외부파일 채택" if st == "adopted" else
             " → 동일, 저장 생략" if st == "same" else " → 저장"))

    order = sorted(range(nw), key=lambda j: -abs(w[j])) + ["hfa"]
    best = acc_of(w, hfa)
    bestW, bestHfa = list(w), hfa
    improved_any = False
    se = season_acc(w, hfa)
    log(f"자동조절 시작: acc={best:.3f} " +
        " ".join(f"{s}={se[s]:.3f}" for s in sorted(se)) +
        f" 순서={[FEATURES[j] if j != 'hfa' else 'hfa' for j in order]}")
    try:
        time_up = False
        for sw in range(max(0, seq_sweeps)):
            for j in order:
                fname = FEATURES[j] if j != "hfa" else "hfa"
                t0 = time.time()
                for sgn in (1.0, -1.0):
                    step = 0
                    while True:
                        step += 1
                        delta = round(step * 0.001, 4)
                        if delta > 4.001:
                            break
                        if max_minutes > 0 and (time.time() - _START) / 60 > max_minutes:
                            time_up = True
                            break
                        cw = list(w)
                        ch = hfa
                        if j == "hfa":
                            ch = clamp(hfa + sgn * delta)
                        else:
                            cw[j] = clamp(cw[j] + sgn * delta)
                        if (ch == hfa) if j == "hfa" else (cw[j] == w[j]):
                            break
                        a = acc_of(cw, ch)
                        if a > best:
                            best, w, hfa = a, cw, ch
                            bestW, bestHfa = list(cw), ch
                            commit(f"★ 순차 {fname} {sgn:+.0f} 값={cw[j] if j != 'hfa' else ch:+.3f}")
                if time_up:
                    break
                log(f"순차 {fname} 완료 ({time.time() - t0:.1f}s) 현재={w[j] if j != 'hfa' else hfa:+.3f} best={best:.3f}")
            if time_up:
                break
            log(f"순차 {sw + 1}바퀴 완료 best={best:.3f}")
        rng = random.Random(seed)
        rnd = 0
        log(f"랜덤워크 시작 (noise={noise} hfa_noise={hfa_noise} rounds=" +
            ("무한" if random_rounds <= 0 else str(random_rounds)) + ", Ctrl+C로 중지)")
        while not time_up:
            rnd += 1
            if random_rounds > 0 and rnd > random_rounds:
                log(f"라운드 {random_rounds} 도달 → 종료")
                break
            if max_minutes > 0 and (time.time() - _START) / 60 > max_minutes:
                log(f"{max_minutes}분 제한 도달 → 종료")
                break
            kw = [clamp(v + rng.uniform(-noise, noise)) for v in w]
            kh = clamp(hfa + rng.uniform(-hfa_noise, hfa_noise))
            a = acc_of(kw, kh)
            w, hfa = kw, kh
            if a > best:
                best, bestW, bestHfa = a, list(kw), kh
                commit(f"★ 랜덤 #{rnd}")
            elif rnd % max(1, log_every) == 0:
                log(f"랜덤 #{rnd} 진행중 best={best:.3f} 현재={a:.3f}")
    except KeyboardInterrupt:
        log("중단됨(Ctrl+C) → 최고점 유지 후 종료")
    if time_up:
        log(f"{max_minutes}분 제한 도달 → 종료")
    if improved_any:
        st, bw, bh, ba = persist()
        if st == "adopted":
            best, w, hfa, bestW, bestHfa = ba, bw, bh, list(bw), bh
            log(f"종료 시점 외부파일 채택 acc={best:.3f}")
    fin_se = season_acc(bestW, bestHfa)
    log(f"자동조절 종료 best={best:.3f} " +
        " ".join(f"{s}={fin_se[s]:.3f}" for s in sorted(fin_se)) +
        (" (개선 저장됨)" if improved_any else " (개선 없음)"))
    return improved_any


def grid_search(league: str, ver: str, tune_s: set, grid_step: float = 0.5,
                max_combos: int = 0, max_minutes: float = 0.0,
                ckpt_every: int = 5000, log_every: int = 20000,
                wmin: float = -2.0, wmax: float = 2.0,
                batch: int = 512, log_secs: float = 10.0,
                split: int = 1, part: int = 0,
                fresh: bool = False, use_cache: bool = True):
    import numpy as _np
    p = model_path(league, ver)
    if not os.path.exists(p):
        log(f"아티팩트 없음: {p} (먼저 학습 필요)")
        return False
    art = json.load(open(p))
    nw = len(FEATURES)
    d = art["draw_prior"]
    mu, sd = art["mu"], art["sd"]
    e = art.get("emphasis", [1.0] * nw)
    pats = art.get("patterns")
    tau = art.get("pattern_tau", 0.0) or 0.0
    cap = art.get("contrib_cap")
    dw = art.get("draw_weights")
    db = art.get("draw_bias", 0.0) or 0.0
    mu_d, sd_d = art.get("draw_mu"), art.get("draw_sd")
    use_draw = bool(dw and mu_d and sd_d)
    ckpt_p = f"ml/permatch/{league}_{ver}.grid.json"
    if split > 1:
        part = max(0, min(part, split - 1))
        ckpt_p = f"ml/permatch/{league}_{ver}.grid.p{part}.json"
    if fresh and os.path.exists(ckpt_p):
        os.remove(ckpt_p)
        log("--new: 체크포인트 삭제, 처음부터 전수탐색")
    if not tune_s:
        tune_s = all_nontrain_seasons(league, set(art.get("train_seasons", [])))
        log(f"조절 시즌 자동선택(미학습): {sorted(tune_s)}")
    _, rows = load_seasons(league, sorted(tune_s), use_cache=use_cache)
    feats, labels, ssn, xds = [], [], [], []
    for r in rows:
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        if hs is None or aws is None:
            continue
        feats.append(apply_emphasis(apply_std(row_features(r), mu, sd), e))
        labels.append(0 if hs > aws else (1 if hs == aws else 2))
        ssn.append(str(r.get("season")))
        if use_draw:
            xds.append([(a - b) / s for a, b, s in zip(draw_row_features(r), mu_d, sd_d)])
    if not feats:
        log("조절용 데이터 없음")
        return False
    Xn = _np.asarray(feats, dtype=float)
    yn = _np.asarray(labels, dtype=int)
    Xd = _np.asarray(xds, dtype=float) if use_draw else None
    PP = None
    if pats and tau and tau > 0:
        try:
            PP = _np.asarray([pattern_proba(x, pats, tau) for x in feats], dtype=float)
        except (KeyError, TypeError, ValueError):
            PP = None

    if use_draw:
        _Dd = _np.asarray(Xd, dtype=float) @ _np.asarray(dw, dtype=float) + db
        ddArr = 1.0 / (1.0 + _np.exp(-_np.clip(_Dd, -30.0, 30.0)))
    else:
        ddArr = _np.full(len(yn), d)

    def _acc_batch(Wmat, hvec):
        B = Wmat.shape[0]
        if cap:
            S = _np.zeros((B, len(yn)))
            Xc = Xn.T
            for j in range(nw):
                S += cap * _np.tanh(Xc[j][None, :] * (Wmat[:, j] / cap)[:, None])
            S += hvec[:, None]
        else:
            S = Wmat @ Xn.T + hvec[:, None]
        Ph = 1.0 / (1.0 + _np.exp(-_np.clip(S, -30.0, 30.0)))
        P0 = Ph * (1.0 - ddArr)
        P1 = _np.broadcast_to(ddArr, Ph.shape)
        P2 = (1.0 - Ph) * (1.0 - ddArr)
        if PP is not None:
            P0 = 0.5 * P0 + 0.5 * PP[:, 0]
            P1 = 0.5 * P1 + 0.5 * PP[:, 1]
            P2 = 0.5 * P2 + 0.5 * PP[:, 2]
        pick = _np.where((P0 >= P1) & (P0 >= P2), 0, _np.where(P1 >= P2, 1, 2))
        return (pick == yn[None, :]).mean(axis=1)

    def acc_of(wv, hv):
        return float(_acc_batch(_np.asarray([wv], dtype=float),
                               _np.asarray([hv], dtype=float))[0])

    def season_acc(wv, hv):
        out = {}
        for s in sorted(set(ssn)):
            m = _np.asarray([a == s for a in ssn])
            if m.sum() == 0:
                continue
            Plin = batch_proba(Xn[m], wv, hv, d, 1.0, None, dw if use_draw else None,
                               db if use_draw else 0.0, Xd[m] if use_draw else None, cap)
            P = 0.5 * Plin + 0.5 * PP[m] if PP is not None else Plin
            home, dr, aw = P[:, 0], P[:, 1], P[:, 2]
            pick = _np.where((home >= dr) & (home >= aw), 0, _np.where(dr >= aw, 1, 2))
            out[s] = float((pick == yn[m]).mean())
        return out

    vals = []
    v = wmin
    while v <= wmax + 1e-9:
        vals.append(round(v, 4))
        v = round(v + grid_step, 4)
    order = sorted(range(nw), key=lambda j: -abs(art["weights"][j])) + ["hfa"]
    nf, nv = len(order), len(vals)
    total = nv ** nf
    artW, artH = list(art["weights"]), art["hfa"]
    best = acc_of(artW, artH)
    bestW, bestHfa = list(artW), artH

    def advance():
        for k in range(nf - 1, -1, -1):
            idx[k] += 1
            if idx[k] < nv:
                return True
            idx[k] = 0
        return False

    def write_ckpt():
        with open(ckpt_p, "w") as f:
            json.dump({"idx": idx, "num": num, "done": ndone, "total": total, "best": best,
                       "bestW": bestW, "bestHfa": bestHfa, "vals": vals,
                       "order": order, "tune": sorted(tune_s),
                       "split": split, "part": part}, f)

    def persist():
        try:
            cur = json.load(open(model_path(league, ver)))
            cw, chh = cur.get("weights"), cur.get("hfa")
            if isinstance(cw, list) and len(cw) == nw and isinstance(chh, (int, float)):
                facc = acc_of(cw, chh)
                if facc > best:
                    return ("adopted", list(cw), float(chh), facc)
        except (OSError, ValueError):
            pass
        try:
            cur0 = json.load(open(model_path(league, ver)))
            if cur0.get("weights") == list(bestW) and cur0.get("hfa") == bestHfa:
                return ("same", list(bestW), bestHfa, best)
        except (OSError, ValueError):
            pass
        art["weights"] = list(bestW)
        art["hfa"] = bestHfa
        save_artifact(league, ver, art)
        return ("saved", list(bestW), bestHfa, best)

    if split < 1:
        split = 1
    start_num = total * part // split
    end_num = total * (part + 1) // split

    def num_to_idx(num):
        dd = [0] * nf
        for k in range(nf):
            dd[k] = (num // nv ** (nf - 1 - k)) % nv
        return dd

    num = start_num
    idx = num_to_idx(num)
    ndone = 0
    if os.path.exists(ckpt_p) and not fresh:
        try:
            c = json.load(open(ckpt_p))
            if (c.get("vals") == vals and c.get("order") == order
                    and set(c.get("tune", [])) == set(tune_s)
                    and c.get("split", 1) == split and c.get("part", 0) == part):
                num = max(start_num, min(int(c.get("num", c.get("done", start_num))), end_num))
                idx = num_to_idx(num)
                ndone = num - start_num
                if isinstance(c.get("bestW"), list) and len(c["bestW"]) == nw:
                    cb = acc_of(c["bestW"], c.get("bestHfa", 0.0))
                    if cb > best:
                        best, bestW, bestHfa = cb, list(c["bestW"]), c.get("bestHfa", 0.0)
                log(f"체크포인트 이어하기: {num:,}/{end_num:,} 완료 best={best:.3f}")
            else:
                log("체크포인트 조건 불일치 → 처음부터")
        except (OSError, ValueError, KeyError, TypeError):
            log("체크포인트 손상 → 처음부터")
    se = season_acc(bestW, bestHfa)
    log(f"전수탐색 시작: dim={nf} 값/축={nv} 전체={total:,}가지 " +
        (f"분할 {part + 1}/{split} [{start_num:,},{end_num:,}) " if split > 1 else "") +
        " ".join(f"{s}={se[s]:.3f}" for s in sorted(se)) + f" base={best:.3f}")
    t0 = time.time()
    last_log = t0
    complete = False
    B = max(1, batch)
    try:
        while num < end_num:
            Wb, hb = [], []
            overflow = False
            for _ in range(B):
                if num >= end_num:
                    break
                row = [0.0] * nw
                hh = 0.0
                for k, feat in enumerate(order):
                    vv = vals[idx[k]]
                    if feat == "hfa":
                        hh = vv
                    else:
                        row[feat] = vv
                Wb.append(row)
                hb.append(hh)
                num += 1
                ndone += 1
                if not advance():
                    overflow = True
                    break
            if not Wb:
                break
            accs = _acc_batch(_np.asarray(Wb, dtype=float), _np.asarray(hb, dtype=float))
            bi = int(_np.argmax(accs))
            ba = float(accs[bi])
            if ba > best:
                best, bestW, bestHfa = ba, list(Wb[bi]), float(hb[bi])
                st, bw, bh, bba = persist()
                if st == "adopted":
                    best, bestW, bestHfa = bba, bw, bh
                ise = season_acc(bestW, bestHfa)
                log(f"★ #{num:,} acc={best:.3f} " +
                    " ".join(f"{s}={ise[s]:.3f}" for s in sorted(ise)) +
                    (" → 외부파일 채택" if st == "adopted" else
                     " → 동일, 저장 생략" if st == "same" else " → 저장"))
                last_log = time.time()
                write_ckpt()
            now = time.time()
            if now - last_log >= log_secs:
                last_log = now
                el = now - t0
                cps = ndone / max(el, 1e-6)
                span = end_num - start_num
                pct = ndone / span * 100 if span else 100.0
                eta_s = (span - ndone) / cps if cps > 0 else float("inf")
                eta_txt = f"{eta_s / 31557600:,.0f}년" if eta_s > 31557600 * 2 else f"{eta_s / 3600:,.1f}h"
                log(f"진행 {ndone:,}/{span:,} ({pct:.4f}%) {cps:,.0f}/s 남은≈{eta_txt} best={best:.3f}")
            if ndone % max(1, ckpt_every) == 0:
                write_ckpt()
            if max_combos > 0 and ndone >= max_combos:
                log(f"최대 {max_combos:,}가지 도달 → 종료")
                break
            if max_minutes > 0 and (time.time() - _START) / 60 > max_minutes:
                log(f"{max_minutes}분 제한 도달 → 종료")
                break
            if overflow or num >= end_num:
                complete = True
                break
    except KeyboardInterrupt:
        log("중단됨(Ctrl+C) → 체크포인트 저장 후 종료")
    write_ckpt()
    if bestW != artW or bestHfa != artH:
        persist()
    span = end_num - start_num
    log(f"전수탐색 종료 {ndone:,}/{span:,} best={best:.3f}" + (" (전체 완료)" if complete else ""))
    return True


def save_artifact(league: str, ver: str, artifact: dict):
    os.makedirs(os.path.dirname(model_path(league, ver)), exist_ok=True)
    with open(model_path(league, ver), "w") as f:
        json.dump(artifact, f)
    log(f"saved {model_path(league, ver)}")


# ---------------------------------------------------------------- single-train (--fast)
def fit_draw(Xm_tr, ty, Xm_va, vy, Xd_tr, Xd_va, w, hfa, d, e, steps=(0.05, 0.2, 0.5)):
    mu_d, sd_d = standardize(Xd_tr if Xd_tr else [[0.0] * len(DRAW_FEATURES)])
    Dn = [[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in Xd_tr]
    Dv = [[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in (Xd_va or [])]
    Xe = [apply_emphasis(x, e) for x in Xm_tr]
    Ve = [apply_emphasis(x, e) for x in Xm_va] if len(Xm_va) else []
    base_tr = []
    for x in Xe:
        s = sum(a * b for a, b in zip(x, w)) + hfa
        ph = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))
        base_tr.append(ph)
    base_va = []
    for x in Ve:
        s = sum(a * b for a, b in zip(x, w)) + hfa
        ph = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))
        base_va.append(ph)
    nw = len(DRAW_FEATURES)
    dw, db = [0.0] * nw, 0.0

    def picks(ph_list, Dn_, dw_, db_):
        out = []
        for ph, x in zip(ph_list, Dn_):
            s = sum(a * b for a, b in zip(x, dw_)) + db_
            dd = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))
            p = [ph * (1 - dd), dd, (1 - ph) * (1 - dd)]
            out.append((p, p.index(max(p))))
        return out

    def key_of(dw_, db_):
        if len(Dv) and len(vy):
            ps = picks(base_va, Dv, dw_, db_)
            hit = sum(1 for (_, pk), y in zip(ps, vy) if pk == y)
            tot = sum(-math.log(max(p[y], 1e-12)) for (p, _), y in zip(ps, vy))
            return (-tot / len(Dv), hit / len(Dv))
        ps = picks(base_tr, Dn, dw_, db_)
        tot = sum(-math.log(max(p[y], 1e-12)) for (p, _), y in zip(ps, ty))
        return (-tot / max(len(Dn), 1),)

    best = key_of(dw, db)
    for _ in range(8):
        improved = False
        for j in list(range(nw)) + ["db"]:
            for step in steps:
                for sign in (1.0, -1.0):
                    if j == "db":
                        cand_w, cand_b = list(dw), db + sign * step
                    else:
                        cand_w = list(dw)
                        cand_w[j] += sign * step
                        cand_b = db
                    key = key_of(cand_w, cand_b)
                    if key > best:
                        dw, db, best = cand_w, cand_b, key
                        improved = True
        if not improved:
            break
    return dw, db, mu_d, sd_d


def fit_linear(train_X, train_y, draw_prior: float, valid_X=None, valid_y=None,
               restarts: int = 40, top_k: int = 5, l2: float = 0.01,
               draw_tr=None, draw_va=None):
    import numpy as np
    mu, sd = standardize(train_X if train_X else [[0.0] * len(FEATURES)])
    Xn = np.asarray([apply_std(x, mu, sd) for x in train_X], dtype=float)
    Vn = np.asarray([apply_std(x, mu, sd) for x in (valid_X or [])], dtype=float)
    train_y = np.asarray(train_y if train_y else [], dtype=int)
    valid_y = np.asarray(valid_y if valid_y else [], dtype=int)
    nw = len(FEATURES)
    import random as _rnd
    scored = []
    for restart in range(restarts):
        _rnd.seed(restart)
        w = [_rnd.gauss(0, 0.3) for _ in range(nw)]
        hfa = _rnd.gauss(0, 0.2)

        def ll_of(wv, hv):
            return batch_ll(Xn, train_y, wv, hv, draw_prior, 1.0, l2)

        best_ll = ll_of(w, hfa)
        for _ in range(12):
            improved = False
            for j in list(range(nw)) + ["hfa"]:
                for step in (0.02, 0.05, 0.15, 0.4):
                    for sign in (1.0, -1.0):
                        if j == "hfa":
                            cand_w, cand_h = list(w), hfa + sign * step
                        else:
                            cand_w = list(w)
                            cand_w[j] += sign * step
                            cand_h = hfa
                        ll = ll_of(cand_w, cand_h)
                        if ll < best_ll - 1e-6:
                            best_ll = ll
                            w, hfa = cand_w, cand_h
                            improved = True
            if not improved:
                break
        if len(Vn) and len(valid_y):
            vacc = batch_acc(Vn, valid_y, w, hfa, draw_prior)
            vll = batch_ll(Vn, valid_y, w, hfa, draw_prior)
            key = (vacc, -vll)
        else:
            key = (-best_ll,)
        scored.append((key, list(w), hfa))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:max(top_k, 1)]
    w = [sum(c[j] for _, c, _ in top) / len(top) for j in range(nw)]
    hfa = sum(c for _, _, c in top) / len(top)
    e = [1.0] * nw

    def _evll(ee):
        return batch_ll(Xn, train_y, w, hfa, draw_prior, 1.0, l2, ee)

    def _evkey(ee):
        if len(Vn) and len(valid_y):
            return (batch_acc(Vn, valid_y, w, hfa, draw_prior, 1.0, ee),
                    -batch_ll(Vn, valid_y, w, hfa, draw_prior, 1.0, 0.0, ee))
        return (-_evll(ee),)
    best_key = _evkey(e)
    for _ in range(8):
        improved = False
        for j in range(nw):
            for step in (0.05, 0.2, 0.5):
                for sign in (1.0, -1.0):
                    cand = list(e)
                    cand[j] = max(0.0, cand[j] + sign * step)
                    key = _evkey(cand)
                    if key > best_key:
                        e = cand
                        best_key = key
                        improved = True
        if not improved:
            break
    log(f"emphasis={json.dumps(dict(zip(FEATURES, [round(v, 2) for v in e])))}")
    if draw_tr:
        dw, db, mu_d, sd_d = fit_draw(Xn, train_y, Vn, valid_y, draw_tr, draw_va,
                                      w, hfa, draw_prior, e)
        Dn = np.asarray([[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in draw_tr],
                        dtype=float)
        Dv = np.asarray([[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in (draw_va or [])],
                        dtype=float).reshape(-1, len(DRAW_FEATURES))
    else:
        dw, db = None, 0.0
        mu_d, sd_d, Dn, Dv = None, None, None, None
    log(f"drawvec={[round(v, 2) for v in (dw or [])]} bias={db:.2f}")
    if dw is not None and len(Vn) and len(valid_y):
        keep, new_acc, new_ll, old_acc, old_ll = draw_accept(
            w, hfa, e, dw, db, Dv, Vn, valid_y, draw_prior)
        if keep:
            log(f"draw 로짓 유지 (ll {old_ll:.4f} → {new_ll:.4f}, acc {old_acc:.3f} → {new_acc:.3f})")
        else:
            log(f"draw 로짓 폐기 (ll {old_ll:.4f} → {new_ll:.4f}, acc {old_acc:.3f} → {new_acc:.3f})")
            dw, db = None, 0.0
            Dn = Dv = None
    T, best_t = 1.0, None
    t = 0.5
    while t <= 10.001:
        ll = batch_ll(Xn, train_y, w, hfa, draw_prior, t, 0.0, e, dw, db, Dn)
        if best_t is None or ll < best_t:
            best_t, T = ll, t
        t += 0.1
    if len(Vn) and len(valid_y):
        T = tune_T_valid(w, hfa, e, dw, db, Dv, Vn, valid_y, draw_prior, T)
    return w, hfa, T, mu, sd, e, dw, db, mu_d, sd_d


def drawfit(league: str, ver: str, extra_s: set, use_cache: bool = True):
    import numpy as _np
    p = model_path(league, ver)
    if not os.path.exists(p):
        log(f"아티팩트 없음: {p}")
        return False
    art = json.load(open(p))
    w, hfa = list(art["weights"]), art["hfa"]
    d = art["draw_prior"]
    mu, sd = art["mu"], art["sd"]
    e = art.get("emphasis", [1.0] * len(w))
    T0 = art.get("T", 1.0)
    train_s = set(art.get("train_seasons", []))
    valid = art.get("valid", "")
    key_s = set(extra_s) | ({valid} if valid else set())
    if not train_s or not key_s:
        log("학습/기준 시즌 정보 없음")
        return False
    _, rows = load_seasons(league, sorted(train_s | key_s), use_cache=use_cache)
    Xm_tr, ty, Xd_tr, Xm_va, vy, Xd_va = [], [], [], [], [], []
    for r in rows:
        s = str(r.get("season"))
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        if hs is None or aws is None:
            continue
        y = 0 if hs > aws else (1 if hs == aws else 2)
        xm = apply_std(row_features(r), mu, sd)
        xd = draw_row_features(r)
        if s in train_s:
            Xm_tr.append(xm)
            ty.append(y)
            Xd_tr.append(xd)
        elif s in key_s:
            Xm_va.append(xm)
            vy.append(y)
            Xd_va.append(xd)
    if not Xm_tr or not Xm_va:
        log("drawfit 데이터 부족")
        return False
    dw, db, mu_d, sd_d = fit_draw(Xm_tr, ty, Xm_va, vy, Xd_tr, Xd_va, w, hfa, d, e)
    Vn = _np.asarray([apply_emphasis(x, e) for x in Xm_va], dtype=float)
    vy_n = _np.asarray(vy, dtype=int)
    Dv = _np.asarray([[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in Xd_va],
                     dtype=float).reshape(-1, len(DRAW_FEATURES))
    keep, new_acc, new_ll, old_acc, old_ll = draw_accept(w, hfa, e, dw, db, Dv, Vn, vy_n, d)
    log(f"drawvec={[round(v, 2) for v in dw]} bias={db:.2f}")
    if not keep:
        log(f"drawfit 개선 없음, 유지 (ll {old_ll:.4f} vs {new_ll:.4f}, acc {old_acc:.3f} vs {new_acc:.3f})")
        return False
    pats, tau = art.get("patterns"), art.get("pattern_tau", 0.0) or 0.0
    cap = art.get("contrib_cap")
    Xe_key = [apply_emphasis(x, e) for x in Xm_va]
    Xd_key = None
    if mu_d and sd_d:
        Xd_key = [[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in Xd_va]

    def served_metrics(Tc):
        n = hit = tot = 0.0
        for i, x in enumerate(Xe_key):
            xd = Xd_key[i] if Xd_key is not None else None
            p = apply_temp(blend_proba(full_proba(x, w, hfa, d, dw, db, xd, cap),
                                       x, pats, tau), Tc)
            y = vy[i]
            n += 1
            hit += (p.index(max(p)) == y)
            tot += -math.log(max(p[y], 1e-12))
        return hit / max(n, 1), tot / max(n, 1)

    base_acc_s, _ = served_metrics(T0)
    T, best_ll_s = T0, served_metrics(T0)[1]
    t = 0.5
    while t <= 10.001:
        a_s, ll_s = served_metrics(round(t, 1))
        if a_s >= base_acc_s - DRAW_ACC_TOL and ll_s < best_ll_s - 1e-6:
            best_ll_s, T = ll_s, round(t, 1)
        t += 0.1
    log(f"drawfit 채택 (ll {old_ll:.4f} → {new_ll:.4f}, acc {old_acc:.3f} → {new_acc:.3f}, T {T0} → {T})")
    with open(p + ".bak", "w") as f:
        json.dump(art, f)
    art["draw_weights"] = dw
    art["draw_bias"] = db
    art["draw_mu"] = mu_d
    art["draw_sd"] = sd_d
    art["draw_features"] = DRAW_FEATURES
    art["T"] = T
    save_artifact(league, ver, art)
    return True


def train_model(matches, train_s: set, valid: str, ver: str, test: str = ""):
    feats, splits, labels, rounds, dfeats = [], [], [], [], []
    for r in matches:
        s = str(r["season"])
        if s not in train_s and s != valid and s != (test or ""):
            continue
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        label = None
        if hs is not None and aws is not None:
            label = 0 if hs > aws else (1 if hs == aws else 2)
        feats.append(row_features(r))
        dfeats.append(draw_row_features(r))
        splits.append("train" if s in train_s else ("valid" if s == valid else "test"))
        labels.append(label)
        try:
            rounds.append(int(r.get("round") or 0))
        except (TypeError, ValueError):
            rounds.append(0)
    if train_s == {valid} or (len(train_s) == 1 and valid in train_s):
        for i, s in enumerate(splits):
            if s in ("train", "valid"):
                splits[i] = "train" if rounds[i] <= 22 else "valid"
        log("single-season round split: train R<=22 / valid R>22")
    n_train = sum(1 for s, y in zip(splits, labels) if s == "train" and y is not None)
    draws = sum(1 for s, y in zip(splits, labels) if s == "train" and y == 1)
    draw_prior = draws / max(n_train, 1)
    log(f"train={n_train} draw_prior={draw_prior:.3f}")
    log(f"features={','.join(FEATURES)}")

    vX = [x for x, s, y in zip(feats, splits, labels) if s == "train" and y is not None]
    vy = [y for s, y in zip(splits, labels) if s == "train" and y is not None]
    vvX = [x for x, s, y in zip(feats, splits, labels) if s == "valid" and y is not None]
    vvy = [y for s, y in zip(splits, labels) if s == "valid" and y is not None]
    d_tr = [x for x, s, y in zip(dfeats, splits, labels) if s == "train" and y is not None]
    d_va = [x for x, s, y in zip(dfeats, splits, labels) if s == "valid" and y is not None]
    w, hfa, T, mu, sd, e, dw, db, mu_d, sd_d = fit_linear(
        vX, vy, draw_prior, vvX, vvy, draw_tr=d_tr, draw_va=d_va)
    log(f"weights={json.dumps(dict(zip(FEATURES, [round(v, 3) for v in w])))} hfa={hfa:.3f} T={T:.1f}")
    Xe_tr = [apply_emphasis(apply_std(x, mu, sd), e) for x in vX]
    Xe_va = [apply_emphasis(apply_std(x, mu, sd), e) for x in vvX]
    if mu_d and sd_d:
        Xde_tr = [[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in d_tr]
        Xde_va = [[(a - b) / s for a, b, s in zip(x, mu_d, sd_d)] for x in d_va]
    else:
        Xde_tr = Xde_va = None
    contrib_cap, T = tune_contrib(Xe_tr, vy, Xe_va, vvy, w, hfa, draw_prior, dw, db, Xde_tr, Xde_va)
    log(f"contrib_cap={contrib_cap} T={T}")

    def _ev(x):
        return apply_emphasis(apply_std(x, mu, sd), e)

    def _evd(xd):
        return [(a - b) / s for a, b, s in zip(xd, mu_d, sd_d)] if mu_d and sd_d else xd
    patterns, pattern_tau, pattern_stats = build_patterns(
        [x for x, s, y in zip(feats, splits, labels) if s == "train" and y is not None],
        [y for s, y in zip(splits, labels) if s == "train" and y is not None],
        mu, sd, e)
    log(f"patterns: home/draw/away x 5, tau={pattern_tau:.3f}")
    for _c in ("home", "draw", "away"):
        _s = pattern_stats[_c][0]
        log(f"  {_c} #1: {_s['freq']}전 acc={_s['acc']:.3f}")
    metrics = {}
    for split in ("train", "valid", "test"):
        rows = [{"probs": apply_temp(blend_proba(full_proba(_ev(x), w, hfa, draw_prior, dw, db, _evd(xd), contrib_cap), _ev(x), patterns, pattern_tau), T), "label": y}
                for x, xd, s, y in zip(feats, dfeats, splits, labels) if s == split and y is not None]
        acc, ll = dataset_metrics(rows)
        metrics[split] = {"n": len(rows), "acc": acc, "ll": ll}
        log(f"[{split}] n={len(rows)} acc={acc:.3f} logloss={ll:.3f}")

    artifact = {"weights": w, "hfa": hfa, "T": T, "draw_prior": draw_prior,
                "mu": mu, "sd": sd, "emphasis": e, "draw_weights": dw, "draw_bias": db,
                "draw_mu": mu_d, "draw_sd": sd_d, "draw_features": DRAW_FEATURES,
                "features": FEATURES, "patterns": patterns, "pattern_tau": pattern_tau,
                "pattern_stats": pattern_stats, "contrib_cap": contrib_cap,
                "train_seasons": sorted(train_s), "valid": valid}
    return artifact, metrics


def eval_artifact(matches, train_s: set, valid: str, artifact: dict):
    w, hfa, T, d = (artifact["weights"], artifact["hfa"], artifact["T"], artifact["draw_prior"])
    mu, sd, e = artifact["mu"], artifact["sd"], artifact.get("emphasis", [1.0] * len(w))
    pats, tau = artifact.get("patterns"), artifact.get("pattern_tau", 0.0) or 0.0
    cap = artifact.get("contrib_cap")
    dw, db = artifact.get("draw_weights"), artifact.get("draw_bias", 0.0)
    mu_d, sd_d = artifact.get("draw_mu"), artifact.get("draw_sd")
    out = {}
    for split, cond in (("train", lambda s: s in train_s), ("valid", lambda s: s == valid)):
        n = hit = tot = 0.0
        for r in matches:
            s = str(r["season"])
            if not cond(s):
                continue
            try:
                hs, aws = float(r.get("home_score")), float(r.get("away_score"))
            except (TypeError, ValueError):
                continue
            y = 0 if hs > aws else (1 if hs == aws else 2)
            x = apply_emphasis(apply_std(row_features(r), mu, sd), e)
            xd = draw_row_features(r)
            if mu_d and sd_d:
                xd = [(a - b) / s_ for a, b, s_ in zip(xd, mu_d, sd_d)]
            p = apply_temp(blend_proba(full_proba(x, w, hfa, d, dw, db, xd, cap), x, pats, tau), T)
            n += 1
            hit += (p.index(max(p)) == y)
            tot += -math.log(max(p[y], 1e-12))
        out[split] = {"n": n, "acc": hit / max(n, 1), "ll": tot / max(n, 1)}
    return out


# ---------------------------------------------------------------- autotune
def sample_steps(rng: random.Random, lo: float = 0.005, hi: float = 0.5, k: int = 4):
    vals = sorted(rng.uniform(math.log(lo), math.log(hi)) for _ in range(k))
    return [round(math.exp(v), 4) for v in vals]


def sample_cfg(rng: random.Random) -> dict:
    return {
        "steps": sample_steps(rng),
        "restarts": rng.choice([8, 15, 25]),
        "sweeps": rng.choice([6, 8, 12]),
        "top_k": rng.choice([1, 3, 5]),
        "l2": rng.choice([0.0, 0.01, 0.05]),
        "emphasis": rng.choice([True, False]),
        "esteps": sample_steps(rng),
    }


def fit_trial(Xn, ty, Vn, vy, d, mu, sd, cfg):
    import numpy as _np
    Xn = _np.asarray(Xn, dtype=float)
    nf = Xn.shape[1]
    Vn = _np.asarray(Vn if Vn else [], dtype=float).reshape(-1, nf) if len(Vn) else _np.zeros((0, nf))
    ty = _np.asarray(ty, dtype=int)
    vy = _np.asarray(vy, dtype=int)
    nw = nf
    rng = random.Random()
    scored = []
    for restart in range(cfg["restarts"]):
        rng.seed(1000 + restart)
        w = [rng.gauss(0, 0.3) for _ in range(nw)]
        hfa = rng.gauss(0, 0.2)

        def ll_of(wv, hv):
            return batch_ll(Xn, ty, wv, hv, d, 1.0, cfg["l2"])

        best_ll = ll_of(w, hfa)
        for _ in range(cfg["sweeps"]):
            improved = False
            for j in list(range(nw)) + ["hfa"]:
                for step in cfg["steps"]:
                    for sign in (1.0, -1.0):
                        cw = list(w)
                        ch = hfa + sign * step if j == "hfa" else hfa
                        if j != "hfa":
                            cw[j] += sign * step
                        ll = ll_of(cw, ch)
                        if ll < best_ll - 1e-6:
                            best_ll = ll
                            w, hfa = cw, ch
                            improved = True
            if not improved:
                break
        if len(Vn) and len(vy):
            key = (batch_acc(Vn, vy, w, hfa, d),
                   -batch_ll(Vn, vy, w, hfa, d))
        else:
            key = (-best_ll,)
        scored.append((key, list(w), hfa))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:max(cfg["top_k"], 1)]
    w = [sum(c[j] for _, c, _ in top) / len(top) for j in range(nw)]
    hfa = sum(c for _, _, c in top) / len(top)

    e = [1.0] * nw
    if cfg["emphasis"]:
        def ekey(ee):
            if len(Vn) and len(vy):
                return (batch_acc(Vn, vy, w, hfa, d, 1.0, ee),
                        -batch_ll(Vn, vy, w, hfa, d, 1.0, 0.0, ee))
            return (-batch_ll(Xn, ty, w, hfa, d, 1.0, cfg["l2"], ee),)
        best_key = ekey(e)
        for _ in range(8):
            improved = False
            for j in range(nw):
                for step in cfg["esteps"]:
                    for sign in (1.0, -1.0):
                        cand = list(e)
                        cand[j] = max(0.0, cand[j] + sign * step)
                        key = ekey(cand)
                        if key > best_key:
                            e, best_key = cand, key
                            improved = True
            if not improved:
                break

    T, best_t = 1.0, None
    t = 0.5
    while t <= 10.001:
        ll = batch_ll(Xn, ty, w, hfa, d, t, 0.0, e)
        if best_t is None or ll < best_t:
            best_t, T = ll, t
        t += 0.1
    if len(Vn) and len(vy):
        best_T, best_key = T, None
        t = 0.5
        while t <= 10.001:
            key = (batch_acc(Vn, vy, w, hfa, d, t, e),
                   -batch_ll(Vn, vy, w, hfa, d, t, 0.0, e))
            if best_key is None or key > best_key:
                best_key, best_T = key, t
            t += 0.1
        T = best_T
    return w, hfa, T, e


def load_existing_best(league: str, ver: str, matches, train_s: set, valid: str):
    """--new가 아닐 때 기존 아티팩트를 최고점으로 복원. 없으면 None."""
    p = model_path(league, ver)
    if not os.path.exists(p):
        return None, None
    try:
        art = json.load(open(p))
    except (OSError, ValueError):
        return None, None
    if not isinstance(art.get("weights"), list) or len(art["weights"]) != len(FEATURES):
        log(f"기존 아티팩트 형식 불일치, 무시: {p}")
        return None, None
    met = eval_artifact(matches, train_s, valid, art)
    vacc = met.get("valid", {}).get("acc", 0.0)
    vll = met.get("valid", {}).get("ll", 0.0)
    log(f"기존 best 로드: {p} valid acc={vacc:.3f} ll={vll:.4f}")
    best = {"acc": vacc, "ll": vll, "cfg": None,
            "weights": art["weights"], "hfa": art["hfa"], "T": art["T"],
            "e": art.get("emphasis", [1.0] * len(FEATURES)),
            "draw_prior": art["draw_prior"], "mu": art["mu"], "sd": art["sd"],
            "draw_weights": art.get("draw_weights"), "draw_bias": art.get("draw_bias", 0.0),
            "draw_mu": art.get("draw_mu"), "draw_sd": art.get("draw_sd"),
            "trial": art.get("best_trial", 0)}
    return best, art


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Prematch 학습 단일 진입점")
    ap.add_argument("--mode", default="train", choices=["train", "predict", "tune", "auto", "grid", "drawfit"])
    ap.add_argument("--tune", default="", help="조절용 시즌, 콤마 구분 (auto에서 비우면 미학습 전체)")
    ap.add_argument("--league", required=True)
    ap.add_argument("--train", default="", help="학습 시즌, 콤마 구분")
    ap.add_argument("--valid", default="", help="검증 시즌 (단일)")
    ap.add_argument("--test", default="", help="테스트 시즌 (참고용)")
    ap.add_argument("--predict", default="", help="--mode predict일 때 대상 시즌")
    ap.add_argument("--ver", required=True)
    ap.add_argument("--fast", action="store_true", help="단발 학습 (autotune 없이 1회 fit)")
    ap.add_argument("--new", action="store_true", help="기존 아티팩트 무시하고 처음부터")
    ap.add_argument("--five", action="store_true", help="5피처(rank,power,val,form5,market) 모드 (ver 뒤에 -f5 자동 추가)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sweeps", type=int, default=6, help="tune 모드용 최대 sweep 수")
    ap.add_argument("--seq-sweeps", type=int, default=1, help="auto 모드 순차 패스 수")
    ap.add_argument("--random-rounds", type=int, default=0, help="auto 모드 랜덤 라운드 수 (0=무한)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="auto 모드 시간 제한(분, 0=무제한)")
    ap.add_argument("--noise", type=float, default=0.1, help="auto 랜덤 가중치 이동폭")
    ap.add_argument("--hfa-noise", type=float, default=0.05, help="auto 랜덤 hfa 이동폭")
    ap.add_argument("--wmin", type=float, default=-2.0)
    ap.add_argument("--wmax", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=200, help="auto 랜덤 진행 로그 간격")
    ap.add_argument("--grid-step", type=float, default=0.5, help="grid 모드 축 간격")
    ap.add_argument("--max-combos", type=int, default=0, help="grid 모드 최대 평가 수 (0=끝까지)")
    ap.add_argument("--ckpt-every", type=int, default=5000, help="grid 모드 체크포인트 간격")
    ap.add_argument("--batch", type=int, default=512, help="grid 모드 묶음 평가 수")
    ap.add_argument("--log-secs", type=float, default=10.0, help="grid 모드 진행 로그 간격(초)")
    ap.add_argument("--split", type=int, default=1, help="grid 모드 분할 수(병렬용)")
    ap.add_argument("--part", type=int, default=0, help="grid 모드 분할 번호(0부터)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    league, ver = args.league, args.ver
    global _LEAGUE_TAG
    _LEAGUE_TAG = league_tag(league)
    if args.five:
        global FEATURES, SEL
        FEATURES = list(FIVE)
        SEL = list(FIVE_IDX)
        if not ver.endswith("-f5"):
            ver = ver + "-f5"
        log(f"5피처 모드: {','.join(FEATURES)} ver={ver}")

    if args.mode == "predict":
        assert args.predict, "--predict 필요"
        with open(model_path(league, ver)) as f:
            art = json.load(f)
        w, hfa, T, d = art["weights"], art["hfa"], art["T"], art["draw_prior"]
        mu, sd = art.get("mu"), art.get("sd")
        e = art.get("emphasis", [1.0] * len(w))
        dw = art.get("draw_weights")
        db = art.get("draw_bias", 0.0)
        mu_d, sd_d = art.get("draw_mu"), art.get("draw_sd")
        pats, tau = art.get("patterns"), art.get("pattern_tau", 0.0) or 0.0
        cap = art.get("contrib_cap")
        _, matches = load_seasons(league, [args.predict], use_cache=not args.no_cache)
        pred_s = set(args.predict.split(","))
        targets = [r for r in matches if str(r["season"]) in pred_s]

        def _px(r):
            x = row_features(r)
            if mu and sd:
                x = apply_std(x, mu, sd)
            xd = draw_row_features(r)
            if mu_d and sd_d:
                xd = [(a - b) / s for a, b, s in zip(xd, mu_d, sd_d)]
            xe = apply_emphasis(x, e)
            return apply_temp(blend_proba(full_proba(xe, w, hfa, d, dw, db, xd, cap), xe, pats, tau), T)
        probas = [_px(r) for r in targets]
        n, hit = len(targets), 0
        for r, p in zip(targets, probas):
            try:
                hs, aws = float(r.get("home_score")), float(r.get("away_score"))
            except (TypeError, ValueError):
                continue
            actual = 0 if hs > aws else (1 if hs == aws else 2)
            if p.index(max(p)) == actual:
                hit += 1
        log(f"predictions recomputed on the fly (no DB write): {n} rows, acc={hit / max(n, 1):.3f}")
        return

    if args.mode == "tune":
        assert args.tune and args.ver, "--tune과 --ver 필요"
        tune_s = {s.strip() for s in args.tune.split(",") if s.strip()}
        tune_weights(league, ver, tune_s, args.sweeps, use_cache=not args.no_cache)
        return

    if args.mode == "drawfit":
        assert args.ver, "--ver 필요"
        extra_s = {s.strip() for s in (args.tune or "").split(",") if s.strip()}
        drawfit(league, ver, extra_s, use_cache=not args.no_cache)
        return

    if args.mode == "auto":
        assert args.ver, "--ver 필요"
        tune_s = {s.strip() for s in (args.tune or "").split(",") if s.strip()}
        auto_tune(league, ver, tune_s, seq_sweeps=args.seq_sweeps,
                  random_rounds=args.random_rounds, max_minutes=args.max_minutes,
                  noise=args.noise, hfa_noise=args.hfa_noise,
                  wmin=args.wmin, wmax=args.wmax, log_every=args.log_every,
                  seed=args.seed, fresh=args.new, use_cache=not args.no_cache)
        return

    if args.mode == "grid":
        assert args.ver, "--ver 필요"
        tune_s = {s.strip() for s in (args.tune or "").split(",") if s.strip()}
        grid_search(league, ver, tune_s, grid_step=args.grid_step,
                    max_combos=args.max_combos, max_minutes=args.max_minutes,
                    ckpt_every=args.ckpt_every, log_every=args.log_every,
                    wmin=args.wmin, wmax=args.wmax,
                    batch=args.batch, log_secs=args.log_secs,
                    split=args.split, part=args.part,
                    fresh=args.new, use_cache=not args.no_cache)
        return

    assert args.train and args.valid, "--train과 --valid 필요"
    train_s = {s.strip() for s in args.train.split(",") if s.strip()}
    valid = args.valid.strip()
    if valid in train_s and not (len(train_s) == 1):
        log("테스트 시즌이 학습시즌과 겹침. 종료")
        sys.exit(2)

    _, rows = load_seasons(league, sorted(train_s | ({valid} if valid else set()) | ({args.test} if args.test else set())),
                           use_cache=not args.no_cache)

    # ---- 단발 모드
    if args.fast:
        artifact, metrics = train_model(rows, train_s, valid, ver, args.test)
        for split in ("train", "valid", "test"):
            m = metrics.get(split, {})
            if m.get("n"):
                log(f"[{split}] n={m['n']} acc={m['acc']:.3f} logloss={m['ll']:.3f}")
        if not args.new:
            try:
                old = json.load(open(model_path(league, ver)))
                old_met = eval_artifact(rows, train_s, valid, old)
                new_acc = metrics.get("valid", {}).get("acc", 0.0)
                old_acc = old_met.get("valid", {}).get("acc", 0.0)
                log(f"기존 {ver}: valid acc={old_acc:.3f} / 신규: valid acc={new_acc:.3f}")
                if old_acc > new_acc:
                    log("기존 유지 (덮어쓰기 안 함). --new로 강제 저장 가능")
                    return
            except FileNotFoundError:
                pass
        save_artifact(league, ver, artifact)
        w = artifact["weights"]
        log(f"FINAL W {{{', '.join(f'{n}:{v:+.3f}' for n, v in zip(FEATURES, w))}}} T={artifact['T']}")
        return

    # ---- autotune 모드
    feats, labels = [], []
    for r in rows:
        if str(r.get("season")) not in train_s:
            continue
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        if hs is None or aws is None:
            continue
        feats.append(row_features(r))
        labels.append(0 if hs > aws else (1 if hs == aws else 2))
    draws = sum(1 for y in labels if y == 1)
    d = draws / max(len(labels), 1)
    mu, sd = standardize(feats)
    Xn = [apply_std(x, mu, sd) for x in feats]
    ty = labels
    vX, vy = [], []
    for r in rows:
        if str(r.get("season")) != valid:
            continue
        hs, aws = parse_score(r.get("home_score")), parse_score(r.get("away_score"))
        if hs is None or aws is None:
            continue
        vX.append(apply_std(row_features(r), mu, sd))
        vy.append(0 if hs > aws else (1 if hs == aws else 2))
    log(f"league={league} train={sorted(train_s)}({len(Xn)}) valid={valid}({len(vX)}) ver={ver}")
    log(f"trials={args.trials} jobs={args.jobs} seed={args.seed} mode={'new' if args.new else 'resume'}")

    best, old_art = (None, None) if args.new else load_existing_best(league, ver, rows, train_s, valid)
    if args.new:
        log("--new: 기존 무시, 처음부터 탐색")
        start_trial = 0
    elif best:
        start_trial = int((old_art or {}).get("trials", 0) or 0)
        log(f"이어하기: best_acc={best['acc']:.3f} trial {start_trial}부터 계속")
    else:
        log("기존 없음: 처음부터 탐색")
        start_trial = 0

    rng = random.Random(args.seed)
    for _ in range(start_trial):
        sample_cfg(rng)
    no_improve = 0

    def ev_metrics(w_, hfa_, T_, e_):
        Xe = [apply_emphasis(x, e_) for x in Xn]
        Ve = [apply_emphasis(x, e_) for x in vX]
        if Ve and vy:
            return metrics_of(w_, hfa_, T_, d, Ve, vy)
        _, ll = metrics_of(w_, hfa_, T_, d, Xe, ty)
        return 0.0, ll

    trial_ids = list(range(start_trial + 1, start_trial + args.trials + 1))
    cfgs = [sample_cfg(rng) for _ in trial_ids]

    from functools import partial
    _run = partial(fit_trial, Xn, ty, vX, vy, d, mu, sd)
    log(f"trials 계산 시작: {len(trial_ids)}건 jobs={args.jobs}")
    results: list = [None] * len(cfgs)
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            fut2i = {ex.submit(_run, c): i for i, c in enumerate(cfgs)}
            done = 0
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                results[i] = fut.result()
                done += 1
                log(f"계산 {done}/{len(cfgs)} 완료 (trial {trial_ids[i]})")
    else:
        for i, c in enumerate(cfgs):
            log(f"trial {trial_ids[i]} 계산 시작 ({i + 1}/{len(cfgs)})")
            results[i] = _run(c)

    for t, cfg, (w, hfa, T, e) in zip(trial_ids, cfgs, results):
        acc, ll = ev_metrics(w, hfa, T, e)
        key = (acc, -ll)
        cur_key = (best["acc"], -best["ll"]) if best else None
        tag = ""
        if cur_key is None or key > cur_key:
            if len(vX) and len(vy):
                import numpy as _np
                _Vn = _np.asarray(vX, dtype=float).reshape(-1, len(FEATURES))
                _vy = _np.asarray(vy, dtype=int)
                _nw = len(FEATURES)
                null_key = (batch_acc(_Vn, _vy, [0.0] * _nw, 0.0, d),
                            -batch_ll(_Vn, _vy, [0.0] * _nw, 0.0, d))
                if null_key > key:
                    log(f"⚠ 경고: null 기준(acc={null_key[0]:.3f} ll={-null_key[1]:.4f})보다 낮음 "
                        f"(acc={acc:.3f} ll={ll:.4f}) — 저장하되 서빙 주의")
            best = {"acc": acc, "ll": ll, "cfg": cfg,
                    "weights": w, "hfa": hfa, "T": T, "e": e,
                    "draw_prior": d, "mu": mu, "sd": sd, "trial": t}
            save_artifact(league, ver, {"weights": w, "hfa": hfa, "T": T, "draw_prior": d,
                                        "mu": mu, "sd": sd, "emphasis": e, "features": FEATURES,
                                        "train_seasons": sorted(train_s),
                                        "valid": valid, "trials": t, "best_trial": t})
            tag = " ★ NEW BEST 저장"
            no_improve = 0
        else:
            no_improve += 1
        done_el = time.time() - _START
        eta = max(0.0, done_el / t * (trial_ids[-1] - t))
        log(f"[trial {t}/{trial_ids[-1]}] valid acc={acc:.3f} ll={ll:.4f} best={best['acc']:.3f} eta={eta / 60:.1f}m cfg={json.dumps(cfg)}{tag}")
        if args.patience > 0 and no_improve >= args.patience:
            log(f"{args.patience}회 연속 미개선 → 조기 종료")
            break
    Xe_best = [apply_emphasis(x, best["e"]) for x in Xn]
    Ve_best = [apply_emphasis(x, best["e"]) for x in vX]
    contrib_cap, best_T = tune_contrib(Xe_best, ty, Ve_best, vy, best["weights"], best["hfa"], d)
    best["T"] = best_T
    log(f"contrib_cap={contrib_cap} T={best_T}")
    pats, tau, pstats = build_patterns(feats, labels, mu, sd, best["e"])
    log(f"patterns: home/draw/away x 5, tau={tau:.3f}")
    try:
        _art = json.load(open(model_path(league, ver)))
        if _art.get("trials", 0) != trial_ids[-1]:
            _art["trials"] = trial_ids[-1]
        _art["T"] = best_T
        _art["contrib_cap"] = contrib_cap
        _art["patterns"] = pats
        _art["pattern_tau"] = tau
        _art["pattern_stats"] = pstats
        save_artifact(league, ver, _art)
    except (OSError, ValueError):
        pass
    log(f"FINAL best acc={best['acc']:.3f} ll={best['ll']:.4f} trial={best['trial']} ver={ver} total={(time.time() - _START) / 60:.1f}m")


if __name__ == "__main__":
    main()
