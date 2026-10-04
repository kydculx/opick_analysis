#!/usr/bin/env python3
"""LSTM 시퀀스 모델 (학습+저장, python-only).

벤치(ml/models_bench.py build_sequences/run_lstm)와 동일한 입력
(팀별 최근 seq경기 [득,실,승점,홈여부], hidden=32, epochs=15)을 사용.
TS 서빙은 미지원 (LSTM 가중치를 TS로 포팅하기 전에는 python 예측만 사용).
아티팩트는 ml/permatch/{league}_{ver}.json + {league}_{ver}.lstm.pt 로 저장.

실행:
    ml/.venv/bin/python ml/lstm_model.py --league premier_league \\
      --train 2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022 \\
      --valid 2024-2025,2025-2026 --ver lstimbase --new
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
    ap = argparse.ArgumentParser(description="LSTM 학습 (python-only)")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--train", default="")
    ap.add_argument("--valid", default="auto")
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--ver", required=True)
    ap.add_argument("--seq", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=32)
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

    print(f"league={args.league} train={train_s} valid={valid_s} seq={args.seq} epochs={args.epochs}")
    rows = B.load_rows(args.league, train_s + valid_s)
    sidx = {s: i for i, s in enumerate(full)}
    scored = sorted((r for r in rows if B.parse_label(r) is not None),
                    key=lambda r: (sidx.get(str(r.get("season")), 0),
                                   int(r.get("round") or 0), r.get("id") or 0))
    Xs, Xa, yy = B.build_sequences(scored, sidx, args.seq)
    cut = sum(1 for r in scored if str(r.get("season")) in set(train_s))
    print(f"seqs train={cut} valid={len(yy) - cut}")
    if cut == 0 or len(yy) - cut == 0:
        print("시퀀스 데이터 부족")
        sys.exit(2)
    try:
        import draw_analysis as _da
        draw_rep = _da.analyze_draws(rows, train_s)
        _da.log_report(draw_rep, print)
    except Exception as e:
        print(f"무분석 스킵: {type(e).__name__}")
        draw_rep = {"seasons": [], "n_train": 0, "draws": 0, "draw_prior": 0.0}

    import numpy as _np
    import torch
    probs = B.run_lstm(Xs[:cut], Xa[:cut], yy[:cut], Xs[cut:], Xa[cut:], yy[cut:],
                       args.seq, args.epochs, args.hidden)
    m = B.metrics_of(probs, yy[cut:])
    print(f"lstm         acc={m['acc']:.3f} ll={m['ll']:.4f} rec={m['draw_rec']:.3f} (n={m['n']})")

    torch.manual_seed(0)
    import torch.nn as nn
    Xtr = _np.asarray(Xs[:cut], dtype=_np.float32)
    mu, sd = Xtr.reshape(-1, 4).mean(0), Xtr.reshape(-1, 4).std(0) + 1e-6

    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(4, args.hidden, batch_first=True)
            self.fc = nn.Linear(args.hidden * 2, 3)

        def forward(self, h, a):
            _, (hn, _) = self.lstm(h)
            _, (an, _) = self.lstm(a)
            return self.fc(torch.cat([hn[-1], an[-1]], dim=1))

    mt = SeqModel()
    opt = torch.optim.Adam(mt.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    Ht = torch.tensor(((Xtr - mu) / sd).tolist())
    At = torch.tensor(((_np.asarray(Xa[:cut], dtype=_np.float32) - mu) / sd).tolist())
    ytr_t = torch.tensor(yy[:cut], dtype=torch.long)
    mt.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = lossf(mt(Ht, At), ytr_t)
        loss.backward()
        opt.step()

    pt_file = f"{args.league}_{args.ver}.lstm.pt"
    pp = os.path.join(ROOT, "ml", "permatch", pt_file)
    os.makedirs(os.path.dirname(pp), exist_ok=True)
    torch.save({"state": mt.state_dict(), "hidden": args.hidden, "seq": args.seq,
                "mu": [float(v) for v in mu], "sd": [float(v) for v in sd]}, pp)
    art = {
        "model_type": "lstm",
        "lstm_file": pt_file,
        "seq_len": args.seq,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "serving": "python-only",
        "train_seasons": sorted(train_s),
        "valid": sorted(valid_s),
        "metrics": {"acc": m["acc"], "ll": m["ll"], "draw_rec": m["draw_rec"]},
        "draw_analysis": draw_rep,
    }
    out_p = os.path.join(ROOT, "ml", "permatch", f"{args.league}_{args.ver}.json")
    json.dump(art, open(out_p, "w", encoding="utf-8"))
    print(f"saved {out_p} + {pt_file}")
    print("TS 서빙 미지원: python 예측만 사용")


if __name__ == "__main__":
    main()
