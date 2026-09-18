"""경우의 수 구간화 공용 모듈. cases.py와 train.py가 함께 사용."""

from __future__ import annotations

import math


def bin_odd_home(p):
    if p != p:
        return "M"
    if p < 0.35:
        return "O1"
    if p < 0.45:
        return "O2"
    if p < 0.55:
        return "O3"
    if p < 0.65:
        return "O4"
    return "O5"


def bin_elo_diff(d):
    if d < -100:
        return "E1"
    if d < -30:
        return "E2"
    if d <= 30:
        return "E3"
    if d <= 100:
        return "E4"
    return "E5"


def bin_form_diff(d):
    if d < -6:
        return "F1"
    if d < -2:
        return "F2"
    if d <= 2:
        return "F3"
    if d <= 6:
        return "F4"
    return "F5"


def bin_same_home(v):
    if v != v:
        return "S0"
    if v < 1.8:
        return "S1"
    if v < 2.2:
        return "S2"
    if v < 2.8:
        return "S3"
    if v < 3.5:
        return "S4"
    return "S5"


def bin_formation(home, away):
    if home != home or away != away:
        return "P0"
    d = home - away
    if d < 0:
        return "홈공"
    if d > 0:
        return "원정공"
    return "비슷"


def raw_home_odd(same_odds_wdl_fn, same) -> float:
    w = same_odds_wdl_fn(same)
    if not w:
        return float("nan")
    try:
        return float(w["home"])
    except (TypeError, ValueError):
        return float("nan")


def combo_key(feat, same_home_raw: float, form_home: float, form_away: float) -> tuple:
    return (
        bin_odd_home(feat["odd_ph"]),
        bin_elo_diff(feat["elo_diff"]),
        bin_form_diff(feat["form_h_pts5"] - feat["form_a_pts5"]),
        bin_same_home(same_home_raw),
        bin_formation(form_home, form_away),
    )


def backoff_keys(key: tuple) -> list:
    o, e, f, s, p = key
    return [key, (o, e, f, s), (o, e, f), (o,)]
