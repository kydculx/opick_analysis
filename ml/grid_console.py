#!/usr/bin/env python3
"""콘솔 학습 데스크톱 GUI (tkinter + matplotlib).

실행:
    sh ml/train_gui.sh
"""
import json
import os
import hashlib
import queue
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEAGUES = [
    ("premier_league", "프리미어리그"),
    ("k_league_1", "K리그 1"),
    ("k_league_2", "K리그 2"),
    ("j1_league", "J1리그"),
    ("bundesliga", "분데스리가"),
    ("laliga", "라리가"),
    ("ligue_1", "리그1(프랑스)"),
    ("serie_a", "세리에A"),
    ("eredivisie", "에레디비시"),
    ("mls", "MLS"),
    ("a_league", "A리그(호주)"),
]
KO2CODE = {ko: code for code, ko in LEAGUES}
CODE2KO = {code: ko for code, ko in LEAGUES}

EUR_SEASONS = "2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022"
YEAR_SEASONS = "2016,2017,2018,2019,2020,2021"
J1_SEASONS = "2017,2018,2019,2020,2021,2022"
PRESETS = {
    "premier_league": (EUR_SEASONS, "2022-2023"),
    "bundesliga": (EUR_SEASONS, "2022-2023"),
    "eredivisie": (EUR_SEASONS, "2022-2023"),
    "laliga": (EUR_SEASONS, "2022-2023"),
    "ligue_1": (EUR_SEASONS, "2022-2023"),
    "serie_a": (EUR_SEASONS, "2022-2023"),
    "a_league": (EUR_SEASONS, "2022-2023"),
    "k_league_1": (YEAR_SEASONS, "2022"),
    "k_league_2": (YEAR_SEASONS, "2022"),
    "mls": (YEAR_SEASONS, "2022"),
    "j1_league": (J1_SEASONS, "2023"),
}

YEAR_STYLE = {"k_league_1", "k_league_2", "j1_league", "mls"}


def known_seasons(league):
    d = os.path.join(ROOT, "ml", ".cache")
    try:
        files = os.listdir(d)
    except OSError:
        files = []
    prefix = league + "_"
    out = set()
    for f in files:
        if not f.startswith(prefix) or not f.endswith(".json"):
            continue
        toks = [t for t in f[len(prefix):-5].split("-") if t.strip()]
        if league in YEAR_STYLE:
            for t in toks:
                if re.fullmatch(r"\d{4}", t):
                    out.add(t)
        else:
            for i in range(0, len(toks) - 1, 2):
                a, b = toks[i], toks[i + 1]
                if re.fullmatch(r"\d{4}", a) and re.fullmatch(r"\d{4}", b):
                    out.add(f"{a}-{b}")
    if out:
        return sorted(out)
    base, va = PRESETS.get(league, ("", ""))
    fb = [s for s in base.split(",") if s.strip()]
    if va and va.strip():
        fb.append(va.strip())
    return sorted(set(fb))


def split_train_valid(league, n):
    full = known_seasons(league)
    pool = full[:-1] if len(full) > 1 else list(full)
    if not pool:
        return [], []
    n = max(1, min(n, len(pool) - 1)) if len(pool) >= 2 else 1
    tr = pool[:n]
    taken = set(tr)
    return tr, [s for s in pool if s not in taken]


def make_ver(train_csv, feats, model="legacy"):
    feats = list(feats)
    nt = len([s for s in train_csv.split(",") if s.strip()])
    key = f"{train_csv}|{','.join(feats)}"
    h = hashlib.sha1(key.encode()).hexdigest()[:4]
    ver = f"tr{nt}-f{len(feats)}-x{h}"
    return ver + "-s3" if model == "softmax" else ver

MODES = [
    ("train", "새학습/계속학습"),
    ("grid", "전수탐색"),
    ("auto", "랜덤탐색"),
]
MO2CODE = {mo: code for code, mo in MODES}
CODE2MO = {code: mo for code, mo in MODES}

# 웹 DashboardExplorer.tsx FEATURE_KO와 동일 매핑
FEATURE_KO = {
    "rank": "순위차", "power": "전력", "hstr": "H2H강도", "cond": "컨디션",
    "att": "공격", "def": "수비", "val": "가치", "form5": "최근폼",
    "h2h5": "H2H5", "avg_goals": "평균득점", "avg_conceded": "평균실점",
    "avg_poss": "점유율", "market": "배당",
}
FEAT13 = ["rank", "power", "hstr", "cond", "att", "def", "val",
          "form5", "h2h5", "avg_goals", "avg_conceded", "avg_poss", "market"]
FIVESET = {"rank", "power", "val", "form5", "market"}

RE_BEST = re.compile(r"best=([0-9.]+)")
RE_PROG = re.compile(r"진행\s+([\d,]+)/([\d,]+)")
RE_TRIAL = re.compile(r"\[trial\s+(\d+)/(\d+)\]")
RE_CALC = re.compile(r"계산\s+(\d+)/(\d+)\s+완료")
RE_TSTART = re.compile(r"trial\s+(\d+)\s+계산\s+시작\s+\((\d+)/(\d+)\)")


def load_dotenv(path):
    env = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
    except FileNotFoundError:
        pass
    return env


def ckpt_path(league, ver):
    return os.path.join(ROOT, "ml", "permatch", f"{league}_{ver}.grid.json")


def tune_path(league, ver):
    return os.path.join(ROOT, "ml", "permatch", f"{league}_{ver}.json")


def auto_prog_path(league, ver):
    return os.path.join(ROOT, "ml", "permatch", f"{league}_{ver}.auto.json")


def list_versions(league):
    d = os.path.join(ROOT, "ml", "permatch")
    try:
        files = os.listdir(d)
    except OSError:
        return []
    out = []
    for f in files:
        if not f.startswith(league + "_") or not f.endswith(".json"):
            continue
        ver = f[len(league) + 1:-5]
        if ".grid" in ver:
            continue
        out.append(ver)
    return sorted(out)


class Runner:
    def __init__(self, on_line, on_done):
        self.on_line = on_line
        self.on_done = on_done
        self.proc = None
        self.last_lines = []
        self._q = queue.Queue()
        self._stop = threading.Event()

    def start(self, cmd, env):
        self._stop.clear()
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._wait, daemon=True).start()

    def _pump(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._q.put(line)
            self.last_lines.append(line)
            del self.last_lines[:-3]
        self._q.put(None)

    def _wait(self):
        while True:
            try:
                line = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if line is None:
                break
            self.on_line(line)
        rc = self.proc.wait() if self.proc else -1
        self.on_done(rc)

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except ProcessLookupError:
                pass


def build_command(o):
    py = os.path.join(ROOT, "ml", ".venv", "bin", "python")
    if not os.path.exists(py):
        py = sys.executable
    cmd = [py, "ml/permatch_mode.py", "--league", o["league"], "--ver", o["ver"]]
    if o.get("model") == "softmax":
        cmd += ["--model", "softmax"]
    mode = o["mode"]
    if mode == "auto":
        cmd += ["--mode", "auto"]
        if o.get("tune"):
            cmd += ["--tune", o["tune"]]
        if o.get("max_minutes"):
            cmd += ["--max-minutes", o["max_minutes"]]
        cmd += ["--auto-step", o.get("grid_step") or "0"]
        cmd += ["--wmin", o.get("wmin") or "-3.0", "--wmax", o.get("wmax") or "3.0"]
        if o.get("model") == "softmax" and o.get("draw_w"):
            cmd += ["--draw-w", o["draw_w"]]
    elif mode == "grid":
        cmd += ["--mode", "grid"]
        if o.get("tune"):
            cmd += ["--tune", o["tune"]]
        cmd += ["--grid-step", o.get("grid_step") or "0.5"]
        if o.get("max_combos"):
            cmd += ["--max-combos", o["max_combos"]]
        if o.get("max_minutes"):
            cmd += ["--max-minutes", o["max_minutes"]]
        cmd += ["--ckpt-every", o.get("ckpt_every") or "5000"]
        cmd += ["--batch", o.get("batch") or "4096"]
        cmd += ["--log-secs", o.get("log_secs") or "1.0"]
        cmd += ["--wmin", o.get("wmin") or "-3.0", "--wmax", o.get("wmax") or "3.0"]
        cmd += ["--jobs", o.get("jobs") or "1"]
        if o.get("model") == "softmax" and o.get("draw_w"):
            cmd += ["--draw-w", o["draw_w"]]
    else:
        cmd += ["--train", o["train"], "--valid", o["valid"] or "auto",
                "--trials", o.get("trials") or "10000",
                "--jobs", o.get("jobs") or "1"]
        if o.get("model") == "softmax" and o.get("draw_w"):
            cmd += ["--draw-w", o["draw_w"]]
    if o.get("features"):
        cmd += ["--features", o["features"]]
    if o.get("new"):
        cmd.append("--new")
    if o.get("no_cache"):
        cmd.append("--no-cache")
    return cmd


def main():
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import rcParams
    from matplotlib.font_manager import fontManager
    _have = {f.name for f in fontManager.ttflist}
    for _cand in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic"):
        if _cand in _have:
            rcParams["font.family"] = _cand
            break
    rcParams["axes.unicode_minus"] = False
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    app = tk.Tk()
    app.title("학습 콘솔")
    app.geometry("880x820")

    v_league = tk.StringVar(value="premier_league")
    v_league_ko = tk.StringVar(value=CODE2KO["premier_league"])
    v_mode = tk.StringVar(value="grid")
    v_mode_ko = tk.StringVar(value=CODE2MO["grid"])
    v_model = tk.StringVar(value="legacy")
    v_tune = tk.StringVar()
    feat_vars = {n: tk.BooleanVar(value=True) for n in FEAT13}
    v_new = tk.BooleanVar(value=False)
    v_ver = tk.StringVar()
    v_train = tk.StringVar()
    v_valid = tk.StringVar()
    v_traincount = tk.StringVar(value="6")
    v_nocache = tk.BooleanVar(value=False)
    v_trials = tk.StringVar(value="10000")
    v_jobs = tk.StringVar(value="4")
    v_draww = tk.StringVar(value="0")
    v_gridstep = tk.StringVar(value="0.5")
    v_maxcombos = tk.StringVar(value="")
    v_maxmin = tk.StringVar(value="")
    v_ckpt = tk.StringVar(value="5000")
    v_batch = tk.StringVar(value="4096")
    v_logsecs = tk.StringVar(value="1.0")
    v_wmin = tk.StringVar(value="-3.0")
    v_wmax = tk.StringVar(value="3.0")
    v_status = tk.StringVar(value="대기 중")
    v_acc = tk.StringVar(value="−")
    v_train_line = tk.StringVar(value="")
    v_valid_line = tk.StringVar(value="")
    v_detail_open = tk.BooleanVar(value=False)

    def selected_feats():
        return [n for n in FEAT13 if feat_vars[n].get()]

    def apply_count_state():
        if not v_new.get():
            try:
                ent_traincount.state(["disabled"])
            except (tk.TclError, AttributeError, NameError):
                pass

    def sync_from_artifact():
        lg, ver = v_league.get(), v_ver.get().strip()
        if not ver:
            return False
        try:
            d = json.load(open(tune_path(lg, ver), encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(d, dict):
            return False
        tr = [str(s) for s in (d.get("train_seasons") or []) if s]
        if not tr:
            return False
        va = d.get("valid")
        vas = [str(s) for s in (va if isinstance(va, list) else ([va] if va else []))
               if s and s != "auto"]
        v_train.set(",".join(tr))
        v_valid.set(",".join(vas))
        if v_traincount.get().strip() != str(len(tr)):
            v_traincount.set(str(len(tr)))
        return True

    def apply_preset(*_):
        if runner["obj"] is not None:
            return
        lg = v_league.get()
        if not v_new.get():
            cur = v_ver.get().strip()
            if not (cur and os.path.exists(tune_path(lg, cur))):
                want_sm = v_model.get().strip() == "softmax"
                cands = [v for v in list_versions(lg) if (v.endswith("-s3") == want_sm)]
                if cands:
                    v_ver.set(cands[-1])
            sync_from_artifact()
            return
        try:
            n = int(v_traincount.get())
        except (ValueError, AttributeError):
            n = 6
        tr, va = split_train_valid(lg, n)
        v_train.set(",".join(tr))
        v_valid.set(",".join(va))
        sel = selected_feats()
        if v_mode.get() in ("grid", "auto"):
            cur = v_ver.get().strip()
            if not (cur and os.path.exists(tune_path(lg, cur))):
                want_sm = v_model.get().strip() == "softmax"
                cands = [v for v in list_versions(lg) if (v.endswith("-s3") == want_sm)]
                v_ver.set(cands[-1] if cands else "")
        else:
            v_ver.set(make_ver(",".join(tr), sel, v_model.get()))
        try:
            btn_feat.configure(text=f"피처({len(sel)})")
        except (AttributeError, RuntimeError, tk.TclError):
            pass

    def current_options():
        sel = selected_feats()
        return {
            "league": v_league.get(), "mode": v_mode.get(), "ver": v_ver.get().strip(),
            "model": v_model.get().strip() or "legacy",
            "train": v_train.get().strip(), "valid": v_valid.get().strip(),
            "tune": v_tune.get().strip(), "features": ",".join(sel) if len(sel) != len(FEAT13) else "",
            "new": v_new.get(),
            "no_cache": v_nocache.get(), "trials": v_trials.get().strip(),
            "jobs": v_jobs.get().strip(), "draw_w": v_draww.get().strip(), "grid_step": v_gridstep.get().strip(),
            "max_combos": v_maxcombos.get().strip(), "max_minutes": v_maxmin.get().strip(),
            "ckpt_every": v_ckpt.get().strip(), "batch": v_batch.get().strip(),
            "log_secs": v_logsecs.get().strip(), "wmin": v_wmin.get().strip(),
            "wmax": v_wmax.get().strip(),
        }

    opt_widgets = []

    def _reg(w):
        opt_widgets.append(w)
        return w

    def set_opts_enabled(on):
        for w in opt_widgets:
            try:
                if not w.winfo_exists():
                    continue
                w.state(["!disabled"] if on else ["disabled"])
            except tk.TclError:
                pass
        apply_count_state()

    top = ttk.Frame(app, padding=8)
    top.pack(fill="x")
    _reg(ttk.Combobox(top, textvariable=v_league_ko, state="readonly",
                      values=[ko for _, ko in LEAGUES], width=13)).pack(side="left")
    _reg(ttk.Combobox(top, textvariable=v_mode_ko, state="readonly",
                      values=[mo for _, mo in MODES], width=13)).pack(side="left", padx=(6, 0))
    _reg(ttk.Combobox(top, textvariable=v_model, state="readonly",
                      values=["legacy", "softmax"], width=8)).pack(side="left", padx=(6, 0))
    btn_feat = _reg(ttk.Button(top, text="피처(13)", width=8,
                               command=lambda: open_feat_popup()))
    btn_feat.pack(side="left", padx=(6, 0))
    _reg(ttk.Checkbutton(top, text="처음부터", variable=v_new)).pack(side="left", padx=(6, 0))
    btn_stop = ttk.Button(top, text="중지", state="disabled")
    btn_stop.pack(side="right")
    btn_start = ttk.Button(top, text="시작")
    btn_start.pack(side="right", padx=(0, 6))
    ttk.Button(top, text="상세 ▸", width=7,
               command=lambda: toggle_detail()).pack(side="right", padx=(0, 6))

    feat_pop = {"win": None}

    def open_feat_popup():
        try:
            w = feat_pop["win"]
            if w is not None and w.winfo_exists():
                w.deiconify()
                w.lift()
                w.focus_set()
                return
        except tk.TclError:
            pass
        pop = tk.Toplevel(app)
        feat_pop["win"] = pop
        pop.title("피처 선택")
        frm = ttk.Frame(pop, padding=10)
        frm.pack(fill="both", expand=True)
        for n in FEAT13:
            _reg(ttk.Checkbutton(frm, text=f"{FEATURE_KO.get(n, n)}({n})",
                                 variable=feat_vars[n],
                                 command=lambda: apply_preset())).pack(anchor="w")
        brow = ttk.Frame(frm)
        brow.pack(fill="x", pady=(8, 0))
        _reg(ttk.Button(brow, text="전체",
                        command=lambda: ([v.set(True) for v in feat_vars.values()], apply_preset()))).pack(side="left")
        _reg(ttk.Button(brow, text="5피처",
                        command=lambda: ([feat_vars[n].set(n in FIVESET) for n in FEAT13], apply_preset()))).pack(side="left", padx=(6, 0))
        ttk.Button(brow, text="닫기", command=pop.destroy).pack(side="right")

    hint = ttk.Label(app, text="", foreground="gray")
    hint.pack(fill="x", padx=8)
    ttk.Label(app, textvariable=v_status, padding=(8, 2)).pack(fill="x")

    HINTS = {
        "grid": "전수탐색: 미학습 전체 기준 · 축간격/범위로 전 조합 탐색",
        "train": "새학습/계속학습: 학습수만 입력하면 자동배치(학습=오래된순 N개, 검증=나머지·최신 제외)",
        "auto": "랜덤탐색: 미학습 전체 · 축간격 격자 위 랜덤 점프 · 중지로 종료",
    }

    detail = ttk.Frame(app, padding=8)
    f_common = ttk.Frame(detail)
    f_common.pack(fill="x", pady=2)
    fr_common1 = ttk.Frame(f_common)
    fr_common1.pack(fill="x")
    ttk.Label(fr_common1, text="버전", width=6).pack(side="left")
    cb_ver = _reg(ttk.Combobox(fr_common1, textvariable=v_ver, width=44))
    cb_ver.pack(side="left")
    ttk.Label(fr_common1, text="학습수").pack(side="left", padx=(8, 2))
    ent_traincount = ttk.Entry(fr_common1, textvariable=v_traincount, width=4)
    _reg(ent_traincount).pack(side="left")
    ttk.Label(fr_common1, text="적중율").pack(side="left", padx=(8, 2))
    ttk.Label(fr_common1, textvariable=v_acc, width=20).pack(side="left")
    fr_common2 = ttk.Frame(f_common)
    fr_common2.pack(fill="x", pady=(2, 0))
    ttk.Label(fr_common2, textvariable=v_train_line, foreground="gray", wraplength=820).pack(side="top", anchor="w")
    ttk.Label(fr_common2, textvariable=v_valid_line, foreground="gray", wraplength=820).pack(side="top", anchor="w")

    ver_sync = {"on": False}
    train_ver_sync = {"on": False}

    def apply_train_ver(*_):
        if train_ver_sync["on"]:
            return
        if runner["obj"] is not None:
            return
        if not v_new.get():
            return
        tr = v_train.get().strip()
        if not tr:
            return
        ver = make_ver(tr, selected_feats(), v_model.get())
        if ver != v_ver.get().strip():
            train_ver_sync["on"] = True
            try:
                v_ver.set(ver)
            finally:
                train_ver_sync["on"] = False

    def refresh_summary(*_):
        tr = v_train.get().strip() or "-"
        va = v_valid.get().strip()
        va_txt = "auto(미학습 전체)" if (not va or va.lower() == "auto") else va
        v_train_line.set(f"학습: {tr}")
        v_valid_line.set(f"검증: {va_txt}")

    def refresh_ver_list():
        try:
            cb_ver.configure(values=list_versions(v_league.get()))
        except tk.TclError:
            pass

    def apply_ver_features(*_):
        if ver_sync["on"]:
            return
        if runner["obj"] is not None:
            return
        poll_once()
        request_acc()
        if not v_new.get():
            sync_from_artifact()
        ver = v_ver.get().strip()
        if not ver:
            return
        tp = tune_path(v_league.get(), ver)
        if not os.path.exists(tp):
            return
        try:
            names = json.load(open(tp, encoding="utf-8")).get("features")
        except (OSError, ValueError):
            return
        if not isinstance(names, list) or not names:
            return
        known = [n for n in names if n in FEAT13]
        if not known:
            return
        ver_sync["on"] = True
        try:
            for n in FEAT13:
                feat_vars[n].set(n in known)
            btn_feat.configure(text=f"피처({len(known)})")
        except (AttributeError, RuntimeError, tk.TclError):
            pass
        finally:
            ver_sync["on"] = False

    acc_job = {"after_id": None, "token": 0}
    _eval_lock = threading.Lock()

    def request_acc():
        acc_job["token"] += 1
        tok = acc_job["token"]
        if acc_job["after_id"] is not None:
            try:
                app.after_cancel(acc_job["after_id"])
            except tk.TclError:
                pass
        try:
            acc_job["after_id"] = app.after(400, lambda: _acc_spawn(tok))
        except tk.TclError:
            pass

    def _acc_spawn(tok):
        acc_job["after_id"] = None
        threading.Thread(target=lambda: _acc_run(v_league.get(), v_ver.get().strip(), tok),
                         daemon=True).start()

    def _acc_rows(league, seasons):
        base = os.path.join(ROOT, "ml", ".cache")
        try:
            files = os.listdir(base)
        except OSError:
            return []
        cands = [f for f in files if f.startswith(league + "_") and f.endswith(".json")]
        cands = sorted([f for f in cands if any(s in f for s in seasons)],
                       key=lambda f: os.path.getsize(os.path.join(base, f)))
        rows, seen, seen_ids = [], set(), set()
        for f in cands[:12]:
            p = os.path.join(base, f)
            try:
                if os.path.getsize(p) > 50_000_000:
                    continue
                data = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for r in data:
                if str(r.get("season")) in seasons:
                    key = r.get("id", r.get("source_match_id"))
                    if key is not None:
                        if key in seen_ids:
                            continue
                        seen_ids.add(key)
                    seen.add(str(r.get("season")))
                    rows.append(r)
            if len(seen) >= len(seasons):
                break
        return rows

    def _acc_run(league, ver, tok):
        txt = "−"
        try:
            tp = tune_path(league, ver)
            art = json.load(open(tp, encoding="utf-8")) if ver and os.path.exists(tp) else None
        except (OSError, ValueError):
            art = None
        if isinstance(art, dict):
            vs = art.get("valid")
            valid_s = {str(s) for s in (vs if isinstance(vs, list) else ([vs] if vs else [])) if s and s != "auto"}
            feats = art.get("features")
            if valid_s and isinstance(feats, list) and feats:
                vrows = [r for r in _acc_rows(league, valid_s) if str(r.get("season")) in valid_s]
                if vrows:
                    try:
                        if os.path.join(ROOT, "ml") not in sys.path:
                            sys.path.insert(0, os.path.join(ROOT, "ml"))
                        import permatch_mode as pm
                        with _eval_lock:
                            pm.FEATURES = [n for n in feats if n in pm.FEATURES13]
                            pm.SEL = [pm.FEATURES13.index(n) for n in pm.FEATURES]
                            vm = pm.eval_artifact(vrows, set(), valid_s, art).get("valid", {})
                        if vm.get("n"):
                            txt = f"검증 {vm['acc'] * 100:.1f}% (n={int(vm['n'])})"
                    except Exception:
                        pass
            if txt == "−":
                try:
                    cp = ckpt_path(league, ver)
                    if os.path.exists(cp):
                        b = json.load(open(cp, encoding="utf-8")).get("best")
                        if isinstance(b, (int, float)):
                            txt = f"조절 {b * 100:.1f}%"
                except (OSError, ValueError):
                    pass
        def done():
            try:
                if tok == acc_job["token"]:
                    v_acc.set(txt)
            except tk.TclError:
                pass
        try:
            app.after(0, done)
        except tk.TclError:
            pass

    f_train = ttk.Frame(detail)
    fr_train_opt = ttk.Frame(f_train)
    fr_train_opt.pack(fill="x", pady=2)
    ttk.Label(fr_train_opt, text="trials").pack(side="left", padx=(0, 2))
    _reg(ttk.Entry(fr_train_opt, textvariable=v_trials, width=8)).pack(side="left")
    ttk.Label(fr_train_opt, text="jobs").pack(side="left", padx=(8, 2))
    _reg(ttk.Entry(fr_train_opt, textvariable=v_jobs, width=5)).pack(side="left")
    ttk.Label(fr_train_opt, text="무가중").pack(side="left", padx=(8, 2))
    _reg(ttk.Entry(fr_train_opt, textvariable=v_draww, width=4)).pack(side="left")

    f_auto = ttk.Frame(detail)
    fr_auto1 = ttk.Frame(f_auto)
    fr_auto1.pack(fill="x", pady=2)
    for lbl, var, w in [("축간격", v_gridstep, 5), ("wmin", v_wmin, 5), ("wmax", v_wmax, 5)]:
        ttk.Label(fr_auto1, text=lbl).pack(side="left", padx=(0 if lbl == "축간격" else 8, 2))
        _reg(ttk.Entry(fr_auto1, textvariable=var, width=w)).pack(side="left")

    f_grid = ttk.Frame(detail)
    fr_grid2 = ttk.Frame(f_grid)
    fr_grid2.pack(fill="x", pady=2)
    for lbl, var, w in [("jobs", v_jobs, 5), ("축간격", v_gridstep, 5), ("묶음", v_batch, 6),
                        ("wmin", v_wmin, 5), ("wmax", v_wmax, 5)]:
        ttk.Label(fr_grid2, text=lbl).pack(side="left", padx=(0 if lbl == "jobs" else 8, 2))
        _reg(ttk.Entry(fr_grid2, textvariable=var, width=w)).pack(side="left")

    MODE_FRAMES = {"grid": f_grid, "train": f_train, "auto": f_auto}

    def refresh_detail():
        for m, fr in MODE_FRAMES.items():
            fr.pack_forget()
        cur = v_mode.get()
        fr = MODE_FRAMES.get(cur)
        if fr is not None:
            fr.pack(fill="x", pady=2)
        hint.configure(text=HINTS.get(cur, ""))

    def toggle_detail():
        if v_detail_open.get():
            detail.pack_forget()
            v_detail_open.set(False)
        else:
            detail.pack(fill="x", padx=0, after=hint)
            hint.pack_forget()
            hint.pack(fill="x", padx=8, before=detail)
            v_detail_open.set(True)

    mid = ttk.Frame(app, padding=8)
    mid.pack(fill="both", expand=True)
    fig = Figure(figsize=(8.6, 5.4), dpi=100)
    ax_w = fig.add_subplot(2, 1, 1)
    ax_w.set_title("피처 가중치")
    ax_p = fig.add_subplot(2, 1, 2)
    ax_p.set_title("진행")
    ax_p.set_xlabel("진행률(%)")
    ax_p.set_ylabel("정확도")
    ax_p.grid(True, alpha=0.3)
    ax_p.set_autoscalex_on(False)
    ax_p.set_autoscaley_on(False)
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(0, 1)
    line_best, = ax_p.plot([], [], lw=1.4, color="#2563eb", label="best")
    line_cur, = ax_p.plot([], [], lw=1.0, color="#a1a1aa", linestyle="--", label="cur")
    ax_p.legend(fontsize=7, loc="lower right")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.10, hspace=0.50)
    canvas = FigureCanvasTkAgg(fig, master=mid)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    runner = {"obj": None}
    live = {"best": None, "cur_acc": None}
    prog = {"t": 0.0, "ndone": 0, "cps": 0.0}
    hist = {"x": [], "best": [], "cur": []}
    hist_state = {"last_draw": 0.0, "t0": 0.0, "xmode": "%"}
    gfx = {"labels": None, "bars": None, "texts": [],
           "step_lines": [],
           "last_mtime": 0.0, "last_sig": None,
           "last_amtime": 0.0, "last_asig": None}
    chart_state = {"kind": None, "labels": None, "vals": None, "title": None}
    ui_state = {"last_status_t": 0.0, "last_prog_draw": 0.0}

    def _set_status(txt, force=False):
        now = time.monotonic()
        if not force and now - ui_state["last_status_t"] < 0.1:
            return
        ui_state["last_status_t"] = now
        v_status.set(txt)
    feat_cache = {"key": None, "names": None}

    def _draw_step_lines(lo, hi, to_x):
        for ln in gfx["step_lines"]:
            try:
                ln.remove()
            except Exception:
                pass
        gfx["step_lines"] = []
        try:
            step = float(v_gridstep.get())
        except (ValueError, AttributeError):
            return
        if not step > 0 or hi <= lo:
            return
        import math as _m
        k0, k1 = _m.ceil(lo / step), _m.floor(hi / step)
        count = k1 - k0 + 1
        if count <= 0:
            return
        stride = max(1, -(-count // 200))
        for k in range(k0, k1 + 1, stride):
            if k == 0:
                continue
            gfx["step_lines"].append(
                ax_w.axvline(to_x(k * step), color="#e4e4e7", linewidth=0.5, zorder=0.5))

    def refresh_step_lines(*_):
        st = chart_state
        if gfx["bars"] is None or not st["vals"]:
            return
        gfx["labels"] = None
        if st["kind"] == "sm":
            _ensure_bars_sm(st["labels"])
            _update_bars_sm(st["vals"], st["title"])
        else:
            _ensure_bars(st["labels"])
            _update_bars(st["vals"], st["title"])

    def _lim():
        try:
            wmin = float(v_wmin.get())
        except (ValueError, AttributeError):
            wmin = -3.0
        try:
            wmax = float(v_wmax.get())
        except (ValueError, AttributeError):
            wmax = 3.0
        if not (wmin < 0 < wmax):
            wmin, wmax = -3.0, 3.0
        return wmin, wmax, (wmax - wmin) * 0.02

    def _ensure_bars(labels):
        if gfx["labels"] == labels and gfx["bars"] is not None:
            return False
        wmin, wmax, _ = _lim()
        span = wmax - wmin
        ax_w.clear()
        ax_w.set_xlim(0, span)
        n = len(labels)
        gfx["bars"] = ax_w.barh(list(range(n)), [0.0] * n,
                                color=["#2563eb"] * n)
        gfx["texts"] = [
            ax_w.text(0, yy, "", va="center", ha="left",
                      fontsize=7, color="#52525b")
            for yy in range(n)
        ]
        ax_w.set_yticks(list(range(n)))
        ax_w.set_yticklabels(labels, fontsize=8)
        ax_w.invert_yaxis()
        ax_w.axvline(-wmin, color="gray", linewidth=0.8)
        _draw_step_lines(wmin, wmax, lambda w: w - wmin)
        ax_w.set_xticks([span * i / 4 for i in range(5)])
        ax_w.set_xticklabels([f"{'+' if (wmin + span * i / 4) >= 0 else ''}{wmin + span * i / 4:.1f}" for i in range(5)])
        ax_w.set_xlabel("가중치")
        gfx["labels"] = labels
        return True

    def _update_bars(vals, title_acc=None):
        wmin, wmax, pad = _lim()
        span = wmax - wmin
        ax_w.set_xlim(0, span)
        chart_state.update(kind="lin", labels=list(gfx["labels"] or []),
                           vals=list(vals), title=title_acc)
        if title_acc is not None:
            ax_w.set_title(f"피처 가중치 · 탐색 중 {title_acc * 100:.1f}%")
        else:
            ax_w.set_title("피처 가중치")
        for patch, txt, v in zip(gfx["bars"], gfx["texts"], vals):
            d = v - wmin
            patch.set_width(d)
            patch.set_facecolor("#2563eb")
            s = f"{'+' if v >= 0 else ''}{v:.3f}"
            txt.set_text(s)
            if d >= pad * 2:
                txt.set_position((d - pad, txt.get_position()[1]))
                txt.set_ha("right")
                txt.set_color("white")
                txt.set_weight("bold")
            else:
                txt.set_position((d + pad * 0.5, txt.get_position()[1]))
                txt.set_ha("left")
                txt.set_color("#52525b")
                txt.set_weight("normal")
        canvas.draw_idle()

    WIN_PCT = 10.0
    WIN_SEC = 60.0

    def _fit_y(vals):
        nums = [v for v in vals if isinstance(v, (int, float)) and v == v]
        if not nums:
            return
        mn, mx = min(nums), max(nums)
        span = mx - mn
        pad = span * 0.15 if span > 0 else 0.01
        ax_p.set_ylim(max(0.0, mn - pad), min(1.0, mx + pad))

    def _draw_progress():
        xs, yb, yc = hist["x"], hist["best"], hist["cur"]
        line_best.set_data(xs, yb)
        line_cur.set_data(xs, yc)
        if hist_state.get("xmode") == "s":
            win = WIN_SEC
            ax_p.set_xlabel("경과시간(초)")
        else:
            win = WIN_PCT
            ax_p.set_xlabel("진행률(%)")
        if not xs:
            ax_p.set_xlim(0, win)
            ax_p.set_ylim(0, 1)
        else:
            hi = max(win, xs[-1])
            lo = hi - win
            ax_p.set_xlim(lo, hi)
            vis_b = [v for x, v in zip(xs, yb) if x >= lo]
            vis_c = [v for x, v in zip(xs, yc) if x >= lo]
            _fit_y(vis_b + vis_c)
        canvas.draw_idle()
        hist_state["last_draw"] = time.monotonic()

    def _push_hist(x, best, cur=None):
        if not isinstance(x, (int, float)) or not isinstance(best, (int, float)):
            return
        if hist["x"] and x <= hist["x"][-1]:
            if len(hist["x"]) > 1 and x <= hist["x"][-2]:
                return
            hist["x"][-1] = x
            hist["best"][-1] = best
            if cur is None:
                cur = hist["cur"][-1] if hist["cur"] else best
            hist["cur"][-1] = cur
        else:
            hist["x"].append(x)
            hist["best"].append(best)
            hist["cur"].append(cur if isinstance(cur, (int, float)) else best)
            lo = max(WIN_PCT, x) - WIN_PCT
            for _ in range(2):
                if not (len(hist["x"]) > 2 and hist["x"][1] < lo):
                    break
                del hist["x"][0]
                del hist["best"][0]
                del hist["cur"][0]
            if len(hist["x"]) > 5000:
                cut = len(hist["x"]) - 5000
                del hist["x"][:cut]
                del hist["best"][:cut]
                del hist["cur"][:cut]
        if time.monotonic() - hist_state["last_draw"] >= 0.25:
            _draw_progress()

    def _reset_progress():
        hist["x"].clear()
        hist["best"].clear()
        hist["cur"].clear()
        line_best.set_data([], [])
        line_cur.set_data([], [])
        ax_p.set_xlim(0, WIN_PCT)
        ax_p.set_ylim(0, 1)
        hist_state["t0"] = time.monotonic()
        hist_state["xmode"] = "%"

    def _push_time(best, cur=None):
        if not isinstance(best, (int, float)) or best != best:
            return
        if not hist_state.get("t0"):
            hist_state["t0"] = time.monotonic()
        hist_state["xmode"] = "s"
        x = time.monotonic() - hist_state["t0"]
        hist["x"].append(x)
        hist["best"].append(best)
        hist["cur"].append(cur if isinstance(cur, (int, float)) and cur == cur else best)
        cutoff = x - WIN_SEC
        while len(hist["x"]) > 2 and hist["x"][1] < cutoff:
            del hist["x"][0]
            del hist["best"][0]
            del hist["cur"][0]
        if len(hist["x"]) > 20000:
            cut = len(hist["x"]) - 20000
            del hist["x"][:cut]
            del hist["best"][:cut]
            del hist["cur"][:cut]
        _draw_progress()
        hist_state["last_draw"] = 0.0

    SM_CLS = ["홈", "무", "원"]

    def _sm_flat(best_w, feat_names):
        n = len(best_w)
        raw = feat_names or [f"f{i}" for i in range(n)]
        order = sorted(range(n), key=lambda i: -max(abs(v) for v in best_w[i]))
        labels, vals = [], []
        for i in order:
            ko = FEATURE_KO.get(raw[i], raw[i])
            for c in range(3):
                labels.append(f"{ko}/{SM_CLS[c]}")
                vals.append(best_w[i][c])
        return labels, vals

    def _ensure_bars_sm(labels):
        if gfx["labels"] == labels and gfx["bars"] is not None:
            return False
        ax_w.clear()
        n = len(labels)
        gfx["bars"] = ax_w.barh(list(range(n)), [0.0] * n,
                                color=["#2563eb"] * n)
        gfx["texts"] = [
            ax_w.text(0, yy, "", va="center", ha="left",
                      fontsize=7, color="#52525b")
            for yy in range(n)
        ]
        ax_w.set_yticks(list(range(n)))
        ax_w.set_yticklabels(labels, fontsize=8)
        ax_w.invert_yaxis()
        ax_w.axvline(0, color="gray", linewidth=0.8)
        ax_w.set_xlabel("가중치")
        gfx["labels"] = labels
        return True

    def _update_bars_sm(vals, title_acc=None):
        nums = [v for v in vals if isinstance(v, (int, float)) and v == v]
        lo, hi = (min(nums), max(nums)) if nums else (-1.0, 1.0)
        span = hi - lo
        pad = span * 0.15 if span > 0 else 0.5
        ax_w.set_xlim(lo - pad, hi + pad)
        _draw_step_lines(lo - pad, hi + pad, lambda w: w)
        chart_state.update(kind="sm", labels=list(gfx["labels"] or []),
                           vals=list(vals), title=title_acc)
        if title_acc is not None:
            ax_w.set_title(f"피처 가중치(softmax) · 탐색 중 {title_acc * 100:.1f}%")
        else:
            ax_w.set_title("피처 가중치(softmax)")
        for patch, txt, v in zip(gfx["bars"], gfx["texts"], vals):
            patch.set_width(v)
            patch.set_facecolor("#2563eb")
            s = f"{'+' if v >= 0 else ''}{v:.3f}"
            txt.set_text(s)
            frac = abs(v) / span if span > 0 else 0.0
            if frac > 0.12:
                txt.set_position((v - pad * 0.4 if v > 0 else v + pad * 0.4, txt.get_position()[1]))
                txt.set_ha("right" if v > 0 else "left")
                txt.set_color("white")
                txt.set_weight("bold")
            else:
                txt.set_position((v + pad * 0.15 if v >= 0 else v - pad * 0.15, txt.get_position()[1]))
                txt.set_ha("left" if v >= 0 else "right")
                txt.set_color("#52525b")
                txt.set_weight("normal")
        canvas.draw_idle()

    def redraw_graphs(best_w=None, feat_names=None, title_acc=None, order_w=None):
        if not best_w:
            wmin, wmax, _ = _lim()
            ax_w.clear()
            ax_w.set_xlim(0, wmax - wmin)
            ax_w.set_title("피처 가중치")
            gfx["labels"] = None
            gfx["bars"] = None
            gfx["texts"] = []
            canvas.draw_idle()
            return
        if isinstance(best_w[0], list):
            labels, vals = _sm_flat(best_w, feat_names)
            _ensure_bars_sm(labels)
            _update_bars_sm(vals, title_acc)
            return
        raw_names = feat_names or [f"f{i}" for i in range(len(best_w))]
        ref = order_w if isinstance(order_w, list) and len(order_w) == len(best_w) else best_w
        idx = sorted(
            range(len(best_w)),
            key=lambda i: -abs(ref[i]) if isinstance(ref[i], (int, float)) else 0,
        )
        labels = [FEATURE_KO.get(raw_names[i], raw_names[i]) for i in idx]
        vals = [best_w[i] for i in idx]
        _ensure_bars(labels)
        _update_bars(vals, title_acc)

    def _feat_names_for(league, ver):
        key = f"{league}|{ver}"
        if feat_cache["key"] == key and feat_cache["names"] is not None:
            return feat_cache["names"]
        tp = tune_path(league, ver)
        if os.path.exists(tp):
            try:
                names = json.load(open(tp, encoding="utf-8")).get("features")
                feat_cache["key"] = key
                feat_cache["names"] = names
                return names
            except (OSError, ValueError):
                return feat_cache["names"] if feat_cache["key"] == key else None
        return None

    def _fmt_eta(eta_s):
        if eta_s != eta_s or eta_s == float("inf"):
            return "-"
        if eta_s > 31557600 * 2:
            return f"{eta_s / 31557600:,.0f}년"
        return f"{eta_s / 3600:,.1f}h"

    def _refresh_status(ndone, total):
        now = time.monotonic()
        if prog["t"] > 0 and now > prog["t"]:
            inst = (ndone - prog["ndone"]) / max(now - prog["t"], 1e-6)
            if inst >= 0:
                prog["cps"] = prog["cps"] * 0.7 + inst * 0.3 if prog["cps"] else inst
        prog["t"] = now
        prog["ndone"] = ndone
        cps = prog["cps"]
        eta = (total - ndone) / cps if cps > 0 else float("inf")
        pct = ndone / max(total, 1) * 100
        v_status.set(f"진행 {ndone:,}/{total:,} ({pct:.2f}%) {cps:,.0f}/s 남은≈{_fmt_eta(eta)}"
                     + (f" best={live['best']:.3f}" if live["best"] is not None else ""))

    def poll_ckpt():
        o = current_options()
        if o["mode"] != "grid":
            tp = tune_path(o["league"], o["ver"])
            art_mt = 0.0
            if os.path.exists(tp):
                try:
                    try:
                        mt = os.path.getmtime(tp)
                    except OSError:
                        mt = 0.0
                    art_mt = mt
                    if mt != gfx["last_amtime"]:
                        gfx["last_amtime"] = mt
                        d = json.load(open(tp, encoding="utf-8"))
                        bw = d.get("weights")
                        names = d.get("features")
                        sig = (bw, mt) if isinstance(bw, list) else (None, mt)
                        if sig != gfx["last_asig"]:
                            gfx["last_asig"] = sig
                            if isinstance(bw, list) and bw:
                                try:
                                    redraw_graphs(bw, names)
                                except Exception:
                                    pass
                except (OSError, ValueError):
                    pass
            if runner["obj"] is not None:
                try:
                    ap = auto_prog_path(o["league"], o["ver"])
                    amt = os.path.getmtime(ap) if os.path.exists(ap) else 0.0
                except OSError:
                    amt = 0.0
                if amt > art_mt:
                    try:
                        pd = json.load(open(ap, encoding="utf-8"))
                        cw, ca = pd.get("curW"), pd.get("curAcc")
                        psig = (amt, pd.get("ndone"))
                        if isinstance(cw, list) and cw and psig != gfx.get("last_psig"):
                            gfx["last_psig"] = psig
                            now = time.monotonic()
                            if now - ui_state["last_prog_draw"] >= 0.016:
                                ui_state["last_prog_draw"] = now
                                bw2 = pd.get("bestW")
                                try:
                                    redraw_graphs(cw, _feat_names_for(o["league"], o["ver"]),
                                                  title_acc=float(ca) if isinstance(ca, (int, float)) else None,
                                                  order_w=bw2 if isinstance(bw2, list) else None)
                                except Exception:
                                    pass
                                _push_time(pd.get("best"), ca)
                    except (OSError, ValueError, TypeError):
                        pass
                app.after(16, poll_ckpt)
            return
        cp = ckpt_path(o["league"], o["ver"])
        if os.path.exists(cp):
            try:
                try:
                    mt = os.path.getmtime(cp)
                except OSError:
                    mt = 0.0
                if mt != gfx["last_mtime"]:
                    gfx["last_mtime"] = mt
                    d = json.load(open(cp, encoding="utf-8"))
                    if isinstance(d.get("best"), (int, float)):
                        live["best"] = float(d["best"])
                    cur_w = d.get("curW")
                    cur_acc = d.get("curAcc")
                    best_w = d.get("bestW")
                    if isinstance(cur_acc, (int, float)):
                        live["cur_acc"] = float(cur_acc)
                    sig = (d.get("ndone"), d.get("best"), d.get("curAcc"))
                    if isinstance(d.get("ndone"), int) and isinstance(d.get("total"), int):
                        _refresh_status(d["ndone"], d["total"])
                    if isinstance(d.get("ndone"), int) and isinstance(d.get("best"), (int, float)):
                        tot = d.get("total")
                        pct = d["ndone"] / tot * 100 if isinstance(tot, int) and tot > 0 else 0.0
                        _push_hist(pct, float(d["best"]),
                                   float(cur_acc) if isinstance(cur_acc, (int, float)) else None)
                    if sig != gfx["last_sig"]:
                        gfx["last_sig"] = sig
                        running = runner["obj"] is not None
                        try:
                            if running and isinstance(cur_w, list) and cur_w and isinstance(cur_acc, (int, float)):
                                redraw_graphs(cur_w, _feat_names_for(o["league"], o["ver"]), title_acc=float(cur_acc), order_w=best_w)
                            elif isinstance(best_w, list) and best_w:
                                redraw_graphs(best_w, _feat_names_for(o["league"], o["ver"]))
                        except Exception:
                            pass
            except (OSError, ValueError):
                pass
        if runner["obj"] is not None:
            app.after(16, poll_ckpt)

    def append_log(line):
        m = RE_BEST.search(line)
        if m:
            try:
                live["best"] = float(m.group(1))
            except ValueError:
                pass
        m2 = RE_PROG.search(line)
        if m2:
            try:
                nd = int(m2.group(1).replace(",", ""))
                tot = int(m2.group(2).replace(",", ""))
                _refresh_status(nd, tot)
                if live["best"] is not None:
                    _push_hist(nd / max(tot, 1) * 100, float(live["best"]),
                               float(live["cur_acc"]) if isinstance(live.get("cur_acc"), (int, float)) else None)
            except (ValueError, ZeroDivisionError):
                pass
            return
        mt = RE_TRIAL.search(line)
        if mt:
            try:
                nd = int(mt.group(1))
                tot = int(mt.group(2))
                extra = f" best={live['best']:.3f}" if live["best"] is not None else ""
                _set_status(f"trial {nd:,}/{tot:,}{extra}")
                if live["best"] is not None:
                    _push_hist(nd / max(tot, 1) * 100, float(live["best"]))
            except (ValueError, ZeroDivisionError):
                pass
            return
        mc = RE_CALC.search(line)
        if mc:
            try:
                nd = int(mc.group(1))
                tot = int(mc.group(2))
                extra = f" best={live['best']:.3f}" if live["best"] is not None else ""
                _set_status(f"계산 {nd:,}/{tot:,}{extra}")
            except (ValueError, ZeroDivisionError):
                pass
            return
        ms = RE_TSTART.search(line)
        if ms:
            try:
                tid = int(ms.group(1))
                nd = int(ms.group(2))
                tot = int(ms.group(3))
                extra = f" best={live['best']:.3f}" if live["best"] is not None else ""
                _set_status(f"계산 {nd:,}/{tot:,} (trial {tid}){extra}")
            except (ValueError, ZeroDivisionError):
                pass
            return
        txt = line.strip()
        if txt:
            hot = ("★" in txt or "rror" in txt or "Traceback" in txt or "종료" in txt
                   or "중단" in txt or "실패" in txt)
            _set_status(txt[:100], force=hot)

    def on_done(rc):
        def _ui():
            tail = ""
            try:
                rr = runner["obj"]
                if rr is not None:
                    for ln in rr.last_lines:
                        if ln.strip():
                            tail = ln.strip()[:90]
            except (AttributeError, TypeError):
                pass
            runner["obj"] = None
            btn_start.configure(state="normal")
            btn_stop.configure(state="disabled")
            set_opts_enabled(True)
            if tail:
                v_status.set(f"종료 (코드 {rc}) · {tail}")
            else:
                v_status.set(f"종료 (코드 {rc})")
            refresh_ver_list()
            request_acc()
            poll_once()
            _draw_progress()
        app.after(0, _ui)

    def poll_once():
        o = current_options()
        cp = ckpt_path(o["league"], o["ver"])
        tp = tune_path(o["league"], o["ver"])
        bw, names, src = None, None, "아티팩트"
        try:
            try:
                cp_mt = os.path.getmtime(cp)
            except OSError:
                cp_mt = 0.0
            try:
                tp_mt = os.path.getmtime(tp)
            except OSError:
                tp_mt = 0.0
            if os.path.exists(tp):
                try:
                    d = json.load(open(tp, encoding="utf-8"))
                    names = d.get("features")
                    bw = d.get("weights")
                    if bw is None:
                        bw = d.get("W")
                except (OSError, ValueError):
                    pass
            if o["mode"] == "grid" and cp_mt > 0.0 and cp_mt >= tp_mt:
                try:
                    cb = json.load(open(cp, encoding="utf-8")).get("bestW")
                except (OSError, ValueError):
                    cb = None
                if isinstance(cb, list) and cb and (not names or len(cb) == len(names)):
                    bw = cb
                    src = "ckpt"
            if not (isinstance(bw, list) and bw and (not names or len(bw) == len(names))):
                bw = None
                src = "없음"
            print(f"[그래프] {o['league']} {o['ver']} ({o['mode']}) <- {src} "
                  f"bars={len(bw) if isinstance(bw, list) else 0}", flush=True)
            gfx["labels"] = None
            redraw_graphs(bw, names)
        except Exception:
            import traceback as _tb
            _tb.print_exc()

    def on_start():
        v_league.set(KO2CODE.get(v_league_ko.get(), "premier_league"))
        v_mode.set(MO2CODE.get(v_mode_ko.get(), "grid"))
        # 상세 직접 입력을 보존: 프리셋 재적용 금지, 버전 공백시에만 복원
        if not v_ver.get().strip() and v_train.get().strip() and v_mode.get() not in ("grid", "auto"):
            apply_train_ver()
        o = current_options()
        if not selected_feats():
            v_status.set("피처를 1개 이상 선택하세요 (상단 피처 버튼)")
            return
        if o["mode"] == "train" and not o["train"]:
            v_status.set("학습시즌이 필요합니다 (상세에서 확인)")
            return
        if o["mode"] in ("grid", "auto"):
            if not o["ver"]:
                v_status.set("버전을 선택하세요 (상세에서 확인)")
                return
            try:
                art0 = json.load(open(tune_path(o["league"], o["ver"]), encoding="utf-8"))
            except (OSError, ValueError):
                art0 = None
            if not isinstance(art0, dict):
                v_status.set(f"아티팩트 없음: {o['ver']} (버전 확인)")
                return
            if (art0.get("model_type") == "softmax3") != (o["model"] == "softmax"):
                v_status.set("모델과 아티팩트 형식 불일치 (legacy/softmax 확인)")
                return
        set_opts_enabled(False)
        cmd = build_command(o)
        env = dict(os.environ)
        env.update(load_dotenv(os.path.join(ROOT, ".env.local")))
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        if o["mode"] == "grid":
            try:
                j = int(o.get("jobs") or 1)
            except ValueError:
                j = 1
            env["NUMBA_NUM_THREADS"] = str(max(j, 1))
        live["best"] = None
        live["cur_acc"] = None
        prog["t"] = 0.0
        prog["ndone"] = 0
        prog["cps"] = 0.0
        gfx["last_mtime"] = 0.0
        gfx["last_sig"] = None
        gfx["last_amtime"] = 0.0
        gfx["last_asig"] = None
        _reset_progress()
        redraw_graphs()
        v_status.set("실행 중…")
        btn_start.configure(state="disabled")
        btn_stop.configure(state="normal")
        r = Runner(lambda line: app.after(0, append_log, line), on_done)
        runner["obj"] = r
        try:
            r.start(cmd, env)
        except Exception as e:
            runner["obj"] = None
            btn_start.configure(state="normal")
            btn_stop.configure(state="disabled")
            set_opts_enabled(True)
            v_status.set(f"시작 실패: {e}")
            return
        app.after(16, poll_ckpt)

    def on_stop():
        r = runner["obj"]
        if r is not None:
            r.stop()
            v_status.set("중지 요청…")

    def on_league_pick(*_):
        v_league.set(KO2CODE.get(v_league_ko.get(), v_league.get()))
        apply_preset()
        refresh_ver_list()
        request_acc()
        poll_once()

    def on_mode_pick(*_):
        v_mode.set(MO2CODE.get(v_mode_ko.get(), v_mode.get()))
        refresh_detail()
        poll_once()

    def on_fresh_toggle(*_):
        if runner["obj"] is not None:
            return
        if v_new.get():
            try:
                ent_traincount.state(["!disabled"])
            except tk.TclError:
                pass
        else:
            apply_count_state()
        apply_preset()

    btn_start.configure(command=on_start)
    btn_stop.configure(command=on_stop)
    v_league_ko.trace_add("write", on_league_pick)
    v_mode_ko.trace_add("write", on_mode_pick)
    v_ver.trace_add("write", apply_ver_features)
    v_train.trace_add("write", apply_train_ver)
    v_train.trace_add("write", refresh_summary)
    v_valid.trace_add("write", refresh_summary)
    v_traincount.trace_add("write", apply_preset)
    v_model.trace_add("write", apply_preset)
    v_new.trace_add("write", on_fresh_toggle)
    v_gridstep.trace_add("write", refresh_step_lines)
    v_wmin.trace_add("write", refresh_step_lines)
    v_wmax.trace_add("write", refresh_step_lines)
    apply_preset()
    apply_count_state()
    refresh_summary()
    refresh_ver_list()
    refresh_detail()
    toggle_detail()
    request_acc()
    redraw_graphs()
    app.mainloop()


if __name__ == "__main__":
    main()
