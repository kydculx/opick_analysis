"""5축 경우의 수 전수 분석: 배당확률 x 전력차 x 폼차 x 동일배당 x 포메이션.

사용법:
  SUPABASE_URL=... SUPABASE_SECRET_KEY=... ml/.venv/bin/python ml/cases.py
결과: ml/case_stats.json + 콘솔 요약
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train import build_rows, same_odds_wdl, defense_line  # noqa: E402
from bins import combo_key, raw_home_odd  # noqa: E402


def fetch_all(url: str, key: str):
    from supabase import create_client

    sb = create_client(url, key)
    rows, offset, page = [], 0, 1000
    while True:
        res = (
            sb.table("soccer_matches")
            .select("id,league_code,season,round,home_team,away_team,home_score,away_score,lineup,same_odds")
            .order("id")
            .range(offset, offset + page - 1)
            .execute()
        )
        rows += res.data or []
        print(f"fetched {len(rows)}...", flush=True)
        if not res.data or len(res.data) < page:
            break
        offset += page
    return rows


def main():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SECRET_KEY"]
    matches = fetch_all(url, key)
    print(f"total {len(matches)}")

    by_league: dict[str, list] = {}
    for r in matches:
        by_league.setdefault(r["league_code"], []).append(r)

    combos: Counter = Counter()
    labels: dict[tuple, Counter] = {}
    scored = 0
    for league, rows in by_league.items():
        for b in build_rows(rows):
            if b["label"] is None:
                continue
            scored += 1
            f = b["feat"]
            m = b["match"]
            lin = m.get("lineup") or {}
            key = combo_key(f, raw_home_odd(same_odds_wdl, m.get("same_odds")),
                            defense_line(lin.get("home")), defense_line(lin.get("away")))
            combos[key] += 1
            labels.setdefault(key, Counter())[b["label"]] += 1

    print(f"scored rows: {scored}, observed combos: {len(combos)} / 3750")
    counts = sorted(combos.values())
    import statistics

    print(f"rows/combo: median={statistics.median(counts)}, "
          f"<10행={sum(1 for c in counts if c < 10)} ({sum(1 for c in counts if c < 10)/len(counts)*100:.0f}%), "
          f"<30행={sum(1 for c in counts if c < 30)}")
    print("--- 상위 15 조합 (배당/전력/폼/동일/포메 | n | 홈/무/원정%) ---")
    for key, n in combos.most_common(15):
        c = labels[key]
        t = sum(c.values())
        print(f"{'/'.join(key)} | {n} | {c[0]/t*100:.0f}/{c[1]/t*100:.0f}/{c[2]/t*100:.0f}")

    out = {"/".join(k): {"n": n, "h": labels[k][0], "d": labels[k][1], "a": labels[k][2]}
           for k, n in combos.items()}
    with open("ml/case_stats.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("saved ml/case_stats.json")


if __name__ == "__main__":
    main()
