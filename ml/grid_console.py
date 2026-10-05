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


def make_ver(train_csv, feats, draw_w="0", ts=""):
    feats = list(feats)
    nt = len([s for s in train_csv.split(",") if s.strip()])
    try:
        dw = float(str(draw_w).strip() or "0")
    except (ValueError, AttributeError):
        dw = 0.0
    if dw == 0.0:
        key = f"{train_csv}|{','.join(feats)}|0"
    else:
        key = f"{train_csv}|{','.join(feats)}|{dw:g}"
    h = hashlib.sha1(key.encode()).hexdigest()[:4]
    tag = ("%g" % dw).replace(".", "_")
    base = f"tr{nt}-f{len(feats)}-d{tag}-x{h}"
    return f"{base}-{ts}" if ts else base


def now_tag():
    return time.strftime("%y-%m-%d_%H-%M", time.localtime())

MODES = [
    ("train", "새학습"),
    ("grid", "전수탐색"),
    ("auto", "랜덤탐색"),
    ("cumu", "누적학습"),
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


def whist_path(league, ver):
    return os.path.join(ROOT, "ml", "permatch", f"{league}_{ver}.whist.json")


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
        if f.endswith((".grid.json", ".live.json", ".auto.json", ".whist.json", ".xgb.json", ".cumu.json", ".wrong.json")):
            continue
        ver = f[len(league) + 1:-5]
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
    mode = o["mode"]
    if mode == "auto":
        cmd += ["--mode", "auto"]
        if o.get("tune"):
            cmd += ["--tune", o["tune"]]
        if o.get("max_minutes"):
            cmd += ["--max-minutes", o["max_minutes"]]
        cmd += ["--auto-step", o.get("grid_step") or "0"]
        cmd += ["--wmin", o.get("wmin") or "-3.0", "--wmax", o.get("wmax") or "3.0"]
        if o.get("draw_w"):
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
        if o.get("draw_w"):
            cmd += ["--draw-w", o["draw_w"]]
    elif mode == "cumu":
        walk = ",".join([s for s in (o.get("train", "") + "," + o.get("valid", "")).split(",") if s.strip()])
        feats = o.get("features", "")
        base = o.get("cumu_base", "")
        if o.get("cumu_ver"):
            outver = o["cumu_ver"]
        elif base:
            outver = base + "-cumu"
        else:
            _ft = f"-f{len(feats.split(','))}" if feats else ""
            outver = f"cumu{_ft}-{now_tag()}"
        cmd = [py, "ml/cumulative.py", "--league", o["league"],
               "--ver", outver, "--lr", o.get("cumu_lr") or "0.05"]
        if base:
            cmd += ["--base", base]
        if o.get("cumu_decay"):
            cmd += ["--lr-decay", o["cumu_decay"]]
        if o.get("cumu_rb"):
            cmd.append("--rollback")
        if feats:
            cmd += ["--features", feats]
        if walk:
            cmd += ["--seasons", walk]
        return cmd
    else:
        cmd += ["--train", o["train"], "--valid", o["valid"] or "auto",
                "--trials", o.get("trials") or "10000",
                "--jobs", o.get("jobs") or "1",
                "--auto-ensemble"]
        if o.get("draw_w"):
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
    from tkinter import messagebox, ttk
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
    app.update_idletasks()
    _sw, _sh, _ww, _wh = app.winfo_screenwidth(), app.winfo_screenheight(), 880, 1000
    app.geometry(f"{_ww}x{_wh}+{max(0, (_sw - _ww) // 2)}+{max(0, (_sh - _wh) // 2)}")

    v_league = tk.StringVar(value="premier_league")
    v_league_ko = tk.StringVar(value=CODE2KO["premier_league"])
    v_mode = tk.StringVar(value="grid")
    v_mode_ko = tk.StringVar(value=CODE2MO["grid"])
    v_tune = tk.StringVar()
    feat_vars = {n: tk.BooleanVar(value=True) for n in FEAT13}
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
    v_cumulr = tk.StringVar(value="0.05")
    v_cumubase = tk.StringVar(value="")
    v_cumudecay = tk.BooleanVar(value=False)
    v_cumurb = tk.BooleanVar(value=False)
    v_status = tk.StringVar(value="대기 중")
    v_acc = tk.StringVar(value="−")
    v_train_line = tk.StringVar(value="")
    v_valid_line = tk.StringVar(value="")
    v_detail_open = tk.BooleanVar(value=False)

    def selected_feats():
        return [n for n in FEAT13 if feat_vars[n].get()]

    _dw_widgets: list = []

    def apply_preset(*_):
        if runner["obj"] is not None:
            return
        if fill_quiet["on"]:
            return
        lg = v_league.get()
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
                cands = list_versions(lg)
                v_ver.set(cands[-1] if cands else "")
        elif v_mode.get() == "cumu":
            pass
        else:
            v_ver.set(make_ver(",".join(tr), sel, v_draww.get(), now_tag()))
        try:
            btn_feat.configure(text=f"피처({len(sel)})")
        except (AttributeError, RuntimeError, tk.TclError):
            pass

    def current_options():
        sel = selected_feats()
        return {
            "league": v_league.get(), "mode": v_mode.get(), "ver": v_ver.get().strip(),
            "train": v_train.get().strip(), "valid": v_valid.get().strip(),
            "tune": v_tune.get().strip(), "features": ",".join(sel) if len(sel) != len(FEAT13) else "",
            "new": v_mode.get() == "train",
            "no_cache": v_nocache.get(), "trials": v_trials.get().strip(),
            "jobs": v_jobs.get().strip(), "draw_w": v_draww.get().strip(), "grid_step": v_gridstep.get().strip(),
            "max_combos": v_maxcombos.get().strip(), "max_minutes": v_maxmin.get().strip(),
            "ckpt_every": v_ckpt.get().strip(), "batch": v_batch.get().strip(),
            "log_secs": v_logsecs.get().strip(), "wmin": v_wmin.get().strip(),
            "wmax": v_wmax.get().strip(), "cumu_lr": v_cumulr.get().strip(),
            "cumu_base": "" if v_cumubase.get().strip() in ("", "(처음부터)") else v_cumubase.get().strip(),
            "cumu_decay": "0.0002" if v_cumudecay.get() else "",
            "cumu_rb": bool(v_cumurb.get()),
        }

    opt_widgets = []

    def _reg(w):
        opt_widgets.append(w)
        return w

    def set_opts_enabled(on):
        st_ttk = ["!disabled"] if on else ["disabled"]
        st_tk = "normal" if on else "disabled"
        for w in opt_widgets:
            try:
                if not w.winfo_exists():
                    continue
                try:
                    w.state(st_ttk)
                except AttributeError:
                    w.configure(state=st_tk)
            except tk.TclError:
                pass

    top = ttk.Frame(app, padding=8)
    top.pack(fill="x")
    _reg(ttk.Combobox(top, textvariable=v_league_ko, state="readonly",
                      values=[ko for _, ko in LEAGUES], width=13)).pack(side="left")
    _reg(ttk.Combobox(top, textvariable=v_mode_ko, state="readonly",
                      values=[mo for _, mo in MODES], width=13)).pack(side="left", padx=(6, 0))
    btn_feat = _reg(ttk.Button(top, text="피처(13)", width=8,
                               command=lambda: open_feat_popup()))
    btn_feat.pack(side="left", padx=(6, 0))
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
        ttk.Button(brow, text="닫기", command=pop.destroy).pack(side="right")

    hint = ttk.Label(app, text="", foreground="gray")
    hint.pack(fill="x", padx=8)
    ttk.Label(app, textvariable=v_status, padding=(8, 2)).pack(fill="x")

    HINTS = {
        "grid": "전수탐색: 미학습 전체 기준 · 축간격/범위로 전 조합 탐색 · 종료 후 아래 슬라이더·진행 클릭으로 시점 이동",
        "train": "새학습: 학습수만 입력하면 자동배치(학습=오래된순 N개, 검증=나머지·최신 제외)",
        "auto": "랜덤탐색: 미학습 전체 · 축간격 격자 위 랜덤 점프 · 중지로 종료",
        "cumu": "누적학습: 버전 선택 없이 콜드스타트 · 오래된 경기부터 1스텝씩 갱신",
    }

    detail = ttk.Frame(app, padding=8)
    f_common = ttk.Frame(detail)
    f_common.pack(fill="x", pady=2)
    fr_common1 = ttk.Frame(f_common)
    fr_common1.pack(fill="x")
    ttk.Label(fr_common1, text="버전", width=6).pack(side="left")
    fr_ver = ttk.Frame(fr_common1)
    fr_ver.pack(side="left")
    cb_ver = _reg(ttk.Combobox(fr_ver, textvariable=v_ver, width=44))
    cb_ver.pack(side="left")
    ver_label = ttk.Label(fr_ver, textvariable=v_ver, foreground="gray")
    btn_del_ver = _reg(ttk.Button(fr_ver, text="삭제", width=5,
                    command=lambda: on_delete_ver()))
    btn_del_ver.pack(side="left", padx=(6, 0))
    ttk.Label(fr_common1, text="학습수").pack(side="left", padx=(8, 2))
    ent_traincount = ttk.Entry(fr_common1, textvariable=v_traincount, width=4)
    _reg(ent_traincount).pack(side="left")
    lbl_acc = ttk.Label(fr_common1, text="적중율")
    lbl_acc.pack(side="left", padx=(8, 2))
    val_acc = ttk.Label(fr_common1, textvariable=v_acc, width=20)
    val_acc.pack(side="left")
    fr_common2 = ttk.Frame(f_common)
    fr_common2.pack(fill="x", pady=(2, 0))
    ttk.Label(fr_common2, textvariable=v_train_line, foreground="gray", wraplength=820).pack(side="top", anchor="w")
    ttk.Label(fr_common2, textvariable=v_valid_line, foreground="gray", wraplength=820).pack(side="top", anchor="w")

    ver_sync = {"on": False}
    train_ver_sync = {"on": False}
    fill_quiet = {"on": False}

    def apply_train_ver(*_):
        if train_ver_sync["on"]:
            return
        if fill_quiet["on"]:
            return
        if runner["obj"] is not None:
            return
        tr = v_train.get().strip()
        if not tr:
            return
        ver = make_ver(tr, selected_feats(), v_draww.get(), now_tag())
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
        try:
            lg = v_league.get()
            legacy = []
            for _v in list_versions(lg):
                if "cumu" not in _v:
                    continue
                try:
                    _a = json.load(open(tune_path(lg, _v), encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(_a, dict) and isinstance(_a.get("weights"), list):
                    legacy.append(_v)
            cb_cumubase.configure(values=["(처음부터)"] + legacy)
            if v_cumubase.get().strip() not in (["(처음부터)"] + legacy):
                v_cumubase.set("(처음부터)")
        except (tk.TclError, NameError):
            pass

    def apply_ver_features(*_):
        if ver_sync["on"]:
            return
        if runner["obj"] is not None:
            return
        poll_once()
        request_acc()
        if v_mode.get() == "train" and runner["obj"] is None:
            return
        _refresh_whist(v_league.get(), v_ver.get().strip())
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
        if v_mode.get() in ("grid", "auto"):
            fill_search_params(v_league.get(), ver)
        refresh_wedit()

    def fill_search_params(league, ver):
        m = re.search(r"-d(\d+(?:_\d+)?)(?=-|$)", ver)
        if m:
            try:
                v_draww.set("%g" % float(m.group(1).replace("_", ".")))
            except (ValueError, AttributeError, tk.TclError):
                pass
        try:
            art = json.load(open(tune_path(league, ver), encoding="utf-8"))
        except (OSError, ValueError):
            art = None
        if isinstance(art, dict):
            trn = art.get("train_seasons")
            va = art.get("valid")
            vas = [str(s) for s in (va if isinstance(va, list) else ([va] if va else []))
                   if s and s != "auto"]
            if isinstance(trn, list) and trn:
                fill_quiet["on"] = True
                try:
                    v_traincount.set(str(len(trn)))
                    v_train.set(",".join(str(s) for s in trn if s))
                    v_valid.set(",".join(vas))
                finally:
                    fill_quiet["on"] = False
        try:
            vals = json.load(open(ckpt_path(league, ver), encoding="utf-8")).get("vals")
        except (OSError, ValueError):
            return
        if not isinstance(vals, list) or len(vals) < 2:
            return
        try:
            lo, hi = min(vals), max(vals)
            step = vals[1] - vals[0]
            if not (step > 0 and hi > lo):
                return
            v_gridstep.set("%g" % step)
            v_wmin.set("%g" % lo)
            v_wmax.set("%g" % hi)
        except (ValueError, TypeError, AttributeError, tk.TclError):
            pass

    acc_job = {"after_id": None, "token": 0}
    _eval_lock = threading.Lock()

    def request_acc():
        if v_mode.get() == "train" and runner["obj"] is None:
            if not ui_state.get("just_done"):
                acc_job["token"] += 1
                v_acc.set("−")
                return
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

    def _acc_text_for(art, league, ver, allow_ckpt=True):
        txt = "−"
        if isinstance(art, dict):
            mt = art.get("model_type") or "permatch"
            if mt in ("poisson", "dixon", "logreg", "xgb", "lstm", "ensemble"):
                mm = art.get("metrics") or {}
                if isinstance(mm.get("acc"), (int, float)):
                    return f"검증 {mm['acc'] * 100:.1f}% (n={int(mm.get('n', 0))})" if mm.get("n") else f"검증 {mm['acc'] * 100:.1f}%"
            cw = (art.get("cumulative") or {}).get("walk") or {}
            if isinstance(cw.get("acc"), (int, float)) and not art.get("valid"):
                return f"누적 {cw['acc'] * 100:.1f}% (n={int(cw.get('n', 0))})"
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
            try:
                train_set = {str(s) for s in (art.get("train_seasons") or [])}
                valid_set = {str(s) for s in (vs if isinstance(vs, list) else ([vs] if vs else [])) if s and s != "auto"}
                cands = [s for s in known_seasons(league) if s not in train_set and s not in valid_set]
                if cands:
                    newest = sorted(cands)[-1]
                    nrows = [r for r in _acc_rows(league, [newest]) if str(r.get("season")) == newest]
                    if nrows:
                        import permatch_mode as _pm2
                        with _eval_lock:
                            _vm = _pm2.eval_artifact(nrows, set(), {newest}, art).get("valid", {})
                        if _vm.get("n"):
                            _nn, _aa = int(_vm["n"]), _vm["acc"]
                            txt += f" · 진행중 {newest} {_aa * 100:.1f}% ({int(round(_aa * _nn))}/{_nn})"
            except Exception:
                pass
            if txt == "−" and allow_ckpt:
                try:
                    cp = ckpt_path(league, ver)
                    if os.path.exists(cp):
                        b = json.load(open(cp, encoding="utf-8")).get("best")
                        if isinstance(b, (int, float)):
                            txt = f"조절 {b * 100:.1f}%"
                except (OSError, ValueError):
                    pass
        return txt

    def _acc_run(league, ver, tok):
        try:
            tp = tune_path(league, ver)
            art = json.load(open(tp, encoding="utf-8")) if ver and os.path.exists(tp) else None
        except (OSError, ValueError):
            art = None
        txt = _acc_text_for(art, league, ver)
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

    def refresh_wedit():
        if runner["obj"] is not None:
            return
        for ch in list(wedit_grid.winfo_children()):
            try:
                ch.destroy()
            except tk.TclError:
                pass
        wedit_vars.clear()
        wedit_names.clear()
        wedit_pending["w"] = None
        try:
            wedit_info.configure(text="")
        except tk.TclError:
            pass
        try:
            old = preview_art.get("line")
            if old is not None:
                old.remove()
            preview_art["line"] = None
        except Exception:
            pass
        if v_mode.get() == "train":
            return
        lg, ver = v_league.get(), v_ver.get().strip()
        if not ver:
            return
        try:
            d = json.load(open(tune_path(lg, ver), encoding="utf-8"))
        except (OSError, ValueError):
            return
        feats = d.get("features")
        w = d.get("weights")
        if not (isinstance(feats, list) and isinstance(w, list) and len(feats) == len(w) and w):
            return
        for idx, (fn, wv) in enumerate(zip(feats, w)):
            var = tk.StringVar(value=f"{wv:.4f}" if isinstance(wv, (int, float)) else "0.0000")
            r, c = divmod(idx, 3)
            base = c * 4
            ttk.Label(wedit_grid, text=FEATURE_KO.get(fn, fn), width=8).grid(
                row=r, column=base, sticky="e", padx=(0, 2), pady=1)
            bdn = _reg(tk.Button(wedit_grid, text="−", width=1, padx=0, pady=0,
                               font=("TkDefaultFont", 8), takefocus=0))
            bdn.grid(row=r, column=base + 1, padx=(0, 1), pady=1)
            bdn.bind("<ButtonPress-1>", lambda e, v=var: wedit_hold_start(v, -1))
            bdn.bind("<ButtonRelease-1>", lambda e: wedit_hold_end())
            bdn.bind("<Leave>", lambda e: wedit_hold_stop())
            ent = _reg(ttk.Entry(wedit_grid, textvariable=var, width=9))
            ent.grid(row=r, column=base + 2, sticky="w", padx=(0, 1), pady=1)
            ent.bind("<Return>", lambda _e: apply_wedit())
            bup = _reg(tk.Button(wedit_grid, text="+", width=1, padx=0, pady=0,
                               font=("TkDefaultFont", 8), takefocus=0))
            bup.grid(row=r, column=base + 3, sticky="w", padx=(0, 8), pady=1)
            bup.bind("<ButtonPress-1>", lambda e, v=var: wedit_hold_start(v, 1))
            bup.bind("<ButtonRelease-1>", lambda e: wedit_hold_end())
            bup.bind("<Leave>", lambda e: wedit_hold_stop())
            wedit_vars.append((fn, var))
            wedit_names.append(fn)

    def apply_wedit():
        if runner["obj"] is not None:
            v_status.set("실행 중에는 편집할 수 없습니다")
            return
        vals = []
        for fn, var in wedit_vars:
            try:
                vals.append(float(var.get().strip()))
            except (ValueError, AttributeError):
                v_status.set(f"숫자 입력 필요: {FEATURE_KO.get(fn, fn)}")
                return
        if not vals:
            v_status.set("편집할 가중치 없음 (버전 확인)")
            return
        lg, ver = v_league.get(), v_ver.get().strip()
        try:
            d = json.load(open(tune_path(lg, ver), encoding="utf-8"))
            feats = d.get("features")
        except (OSError, ValueError):
            feats = None
        if not (isinstance(feats, list) and len(feats) == len(vals)):
            v_status.set("버전 확인 필요")
            return
        wedit_pending["w"] = list(vals)
        wedit_pending["seq"] += 1
        seq = wedit_pending["seq"]
        redraw_graphs(list(vals), list(feats))
        v_status.set("미리보기 적용됨 — 저장하면 버전에 기록")
        try:
            wedit_info.configure(text="편집 미리보기 계산 중…")
        except tk.TclError:
            pass
        threading.Thread(target=lambda: _wedit_preview(lg, ver, list(vals), seq),
                         daemon=True).start()

    def _show_preview_line(acc):
        try:
            old = preview_art.get("line")
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
                preview_art["line"] = None
            if not isinstance(acc, (int, float)):
                canvas.draw_idle()
                return
            lo, hi = ax_p.get_ylim()
            if not (lo <= acc <= hi):
                pad = 0.02
                ax_p.set_ylim(min(lo, acc) - pad, max(hi, acc) + pad)
            preview_art["line"] = ax_p.axhline(acc, color="#16a34a", linestyle="--", linewidth=1.0)
            canvas.draw_idle()
        except Exception:
            pass

    def _wedit_preview(league, ver, weights, seq):
        txt = "편집 미리보기: 검증 시즌 없음"
        acc = None
        try:
            art = json.load(open(tune_path(league, ver), encoding="utf-8"))
            if isinstance(art, dict):
                art = dict(art)
                art["weights"] = list(weights)
                t = _acc_text_for(art, league, ver, allow_ckpt=False)
                if t != "−":
                    txt = f"편집 미리보기 {t}"
                    m = re.search(r"([\d.]+)%", t)
                    if m:
                        acc = float(m.group(1)) / 100.0
        except (OSError, ValueError):
            txt = "편집 미리보기: 계산 실패"
        def done(acc=acc):
            try:
                if seq == wedit_pending.get("seq"):
                    wedit_info.configure(text=txt)
                    _show_preview_line(acc)
            except tk.TclError:
                pass
        try:
            app.after(0, done)
        except tk.TclError:
            pass

    def save_wedit():
        if runner["obj"] is not None:
            v_status.set("실행 중에는 저장할 수 없습니다")
            return
        w = wedit_pending.get("w")
        if not w:
            v_status.set("먼저 적용을 누르세요")
            return
        lg, ver = v_league.get(), v_ver.get().strip()
        if not ver:
            v_status.set("버전 확인 필요")
            return
        try:
            p = tune_path(lg, ver)
            d = json.load(open(p, encoding="utf-8"))
            if not (isinstance(d, dict) and isinstance(d.get("features"), list)
                    and len(d["features"]) == len(w)):
                v_status.set("버전 확인 필요 (피처 수 불일치)")
                return
            d["weights"] = list(w)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f)
        except (OSError, ValueError) as e:
            v_status.set(f"저장 실패: {e}")
            return
        wedit_pending["w"] = None
        try:
            wedit_info.configure(text="")
        except tk.TclError:
            pass
        v_status.set(f"저장됨: {ver}")
        request_acc()
        poll_once()

    f_train = ttk.Frame(detail)
    fr_train_opt = ttk.Frame(f_train)
    fr_train_opt.pack(fill="x", pady=2)
    ttk.Label(fr_train_opt, text="trials").pack(side="left", padx=(0, 2))
    ent_trials = ttk.Entry(fr_train_opt, textvariable=v_trials, width=8)
    _reg(ent_trials).pack(side="left")
    ttk.Label(fr_train_opt, text="jobs").pack(side="left", padx=(8, 2))
    ent_jobs = ttk.Entry(fr_train_opt, textvariable=v_jobs, width=5)
    _reg(ent_jobs).pack(side="left")
    ttk.Label(fr_train_opt, text="무가중").pack(side="left", padx=(8, 2))
    ent_draww = ttk.Entry(fr_train_opt, textvariable=v_draww, width=4)
    _reg(ent_draww).pack(side="left")
    _dw_widgets.append(ent_draww)

    f_auto = ttk.Frame(detail)
    fr_auto1 = ttk.Frame(f_auto)
    fr_auto1.pack(fill="x", pady=2)
    for lbl, var, w in [("축간격", v_gridstep, 5), ("wmin", v_wmin, 5), ("wmax", v_wmax, 5)]:
        ttk.Label(fr_auto1, text=lbl).pack(side="left", padx=(0 if lbl == "축간격" else 8, 2))
        _reg(ttk.Entry(fr_auto1, textvariable=var, width=w)).pack(side="left")
    ttk.Label(fr_auto1, text="무가중").pack(side="left", padx=(8, 2))
    _dw_auto = ttk.Entry(fr_auto1, textvariable=v_draww, width=4)
    _reg(_dw_auto).pack(side="left")
    _dw_widgets.append(_dw_auto)

    f_grid = ttk.Frame(detail)
    fr_grid2 = ttk.Frame(f_grid)
    fr_grid2.pack(fill="x", pady=2)
    for lbl, var, w in [("jobs", v_jobs, 5), ("축간격", v_gridstep, 5), ("묶음", v_batch, 6),
                        ("wmin", v_wmin, 5), ("wmax", v_wmax, 5)]:
        ttk.Label(fr_grid2, text=lbl).pack(side="left", padx=(0 if lbl == "jobs" else 8, 2))
        _reg(ttk.Entry(fr_grid2, textvariable=var, width=w)).pack(side="left")
    ttk.Label(fr_grid2, text="무가중").pack(side="left", padx=(8, 2))
    _dw_grid = ttk.Entry(fr_grid2, textvariable=v_draww, width=4)
    _reg(_dw_grid).pack(side="left")
    _dw_widgets.append(_dw_grid)

    f_cumu = ttk.Frame(detail)
    fr_cumu = ttk.Frame(f_cumu)
    fr_cumu.pack(fill="x", pady=2)
    ttk.Label(fr_cumu, text="베이스").pack(side="left")
    cb_cumubase = _reg(ttk.Combobox(fr_cumu, textvariable=v_cumubase, state="readonly", width=34,
                                    values=["(처음부터)"]))
    cb_cumubase.pack(side="left", padx=(2, 0))
    ttk.Label(fr_cumu, text="lr").pack(side="left", padx=(8, 2))
    _reg(ttk.Entry(fr_cumu, textvariable=v_cumulr, width=6)).pack(side="left")
    _reg(ttk.Checkbutton(fr_cumu, text="lr감쇠", variable=v_cumudecay)).pack(side="left", padx=(8, 0))
    _reg(ttk.Checkbutton(fr_cumu, text="롤백", variable=v_cumurb)).pack(side="left")

    MODE_FRAMES = {"grid": f_grid, "train": f_train, "auto": f_auto, "cumu": f_cumu}

    def refresh_detail():
        for m, fr in MODE_FRAMES.items():
            fr.pack_forget()
        cur = v_mode.get()
        fr = MODE_FRAMES.get(cur)
        if fr is not None:
            fr.pack(fill="x", pady=2)
        if cur == "cumu":
            try:
                wedit_frame.pack_forget()
            except (tk.TclError, AttributeError, NameError):
                pass
        else:
            try:
                if runner["obj"] is None:
                    btn_feat.configure(state="normal")
            except (tk.TclError, AttributeError):
                pass
            try:
                wedit_frame.pack(fill="x", pady=(4, 0))
            except (tk.TclError, AttributeError, NameError):
                pass
        if cur == "train":
            cb_ver.pack_forget()
            btn_del_ver.pack_forget()
            ver_label.pack(side="left")
            lbl_acc.pack_forget()
            val_acc.pack_forget()
            try:
                scrub_scale.state(["disabled"])
            except (tk.TclError, AttributeError):
                pass
        elif cur == "cumu":
            cb_ver.pack_forget()
            btn_del_ver.pack_forget()
            ver_label.pack_forget()
            lbl_acc.pack(side="left", padx=(8, 2))
            val_acc.pack(side="left")
            if v_ver.get().strip():
                ver_sync["on"] = True
                try:
                    v_ver.set("")
                finally:
                    ver_sync["on"] = False
            try:
                scrub_scale.state(["disabled"])
            except (tk.TclError, AttributeError):
                pass
        else:
            ver_label.pack_forget()
            btn_del_ver.pack_forget()
            cb_ver.pack(side="left")
            btn_del_ver.pack(side="left", padx=(6, 0))
            try:
                scrub_scale.state(["!disabled"])
            except (tk.TclError, AttributeError):
                pass
            lbl_acc.pack(side="left", padx=(8, 2))
            val_acc.pack(side="left")
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
    canvas.mpl_connect("button_press_event", lambda event: _on_ax_click(event))
    canvas.mpl_connect("motion_notify_event", lambda event: _on_ax_motion(event))
    canvas.mpl_connect("button_release_event", lambda event: _on_ax_release(event))
    chart_drag = {"on": False}
    scrub_bar = ttk.Frame(mid)
    scrub_bar.pack(fill="x", pady=(4, 0))
    scrub_var = tk.DoubleVar(value=0)
    scrub_scale = ttk.Scale(scrub_bar, from_=0, to=0, orient="horizontal",
                            variable=scrub_var, command=lambda *_: on_scrub())
    scrub_scale.pack(side="left", fill="x", expand=True)
    scrub_scale.state(["disabled"])
    _scrub_drag = {"on": False}

    def _scale_set_from_x(ev):
        try:
            n = len(whist["x"])
            if n == 0:
                return
            w = scrub_scale.winfo_width()
            if w <= 1:
                return
            frac = min(1.0, max(0.0, ev.x / w))
            scrub_var.set(frac * (n - 1))
        except (tk.TclError, AttributeError, ValueError, TypeError):
            pass

    def _on_scale_press(ev):
        if runner["obj"] is not None:
            return
        _scrub_drag["on"] = True
        try:
            scrub_scale.grab_set()
        except tk.TclError:
            pass
        _scale_set_from_x(ev)
        on_scrub()
        return "break"

    def _on_scale_motion(ev):
        if not _scrub_drag["on"]:
            return
        if runner["obj"] is not None:
            return
        _scale_set_from_x(ev)
        on_scrub()
        return "break"

    def _on_scale_release(ev):
        _scrub_drag["on"] = False
        try:
            scrub_scale.grab_release()
        except tk.TclError:
            pass
    scrub_scale.bind("<ButtonPress-1>", _on_scale_press)
    scrub_scale.bind("<B1-Motion>", _on_scale_motion)
    scrub_scale.bind("<ButtonRelease-1>", _on_scale_release)

    def _scrub_keystep(step):
        if runner["obj"] is not None:
            return
        if len(whist["x"]) == 0:
            return
        try:
            cur = int(float(scrub_var.get()))
        except (ValueError, TypeError):
            return
        nxt = max(0, min(len(whist["x"]) - 1, cur + step))
        scrub_var.set(nxt)
        _scrub_to(nxt)

    def _on_scale_key(ev, step):
        _scrub_keystep(step)
        return "break"

    scrub_scale.bind("<Left>", lambda e: _on_scale_key(e, -1))
    scrub_scale.bind("<Right>", lambda e: _on_scale_key(e, 1))

    def _on_key_global(ev):
        try:
            focus = app.focus_get()
            if focus is not None and focus.winfo_class() in ("TEntry", "Entry", "TCombobox", "Text"):
                return
        except tk.TclError:
            pass
        if ev.keysym == "Left":
            _scrub_keystep(-1)
        elif ev.keysym == "Right":
            _scrub_keystep(1)
    app.bind_all("<Left>", _on_key_global)
    app.bind_all("<Right>", _on_key_global)
    scrub_label = ttk.Label(scrub_bar, text="—", width=46)
    scrub_label.pack(side="left", padx=(6, 0))
    scrub = {"active": False, "cursor": None, "order": None,
             "blit_bg": None, "blit_key": None}

    wedit_frame = ttk.LabelFrame(mid, text="가중치 직접편집", padding=6)
    wedit_frame.pack(fill="x", pady=(4, 0))
    wedit_grid = ttk.Frame(wedit_frame)
    wedit_grid.pack(fill="x")
    wedit_vars: list = []
    wedit_names: list = []
    wedit_pending = {"w": None, "seq": 0}
    preview_art = {"line": None}
    wedit_hold = {"after": None, "var": None, "dir": 0, "t0": 0.0}

    def wedit_step():
        try:
            step = float(v_gridstep.get().strip() or "0")
        except (ValueError, AttributeError):
            step = 0.0
        return step if step > 0 else 0.01

    def wedit_nudge(var, direction):
        if runner["obj"] is not None:
            return
        try:
            cur = float(var.get().strip())
        except (ValueError, AttributeError):
            cur = 0.0
        try:
            var.set(f"{cur + direction * wedit_step():.4f}")
        except tk.TclError:
            pass

    def wedit_hold_stop(*_):
        wedit_hold["var"] = None
        try:
            if wedit_hold.get("after") is not None:
                app.after_cancel(wedit_hold["after"])
        except (tk.TclError, ValueError):
            pass
        wedit_hold["after"] = None

    def wedit_hold_start(var, direction):
        if runner["obj"] is not None:
            return
        wedit_hold_stop()
        wedit_nudge(var, direction)
        wedit_hold["var"] = var
        wedit_hold["dir"] = direction
        wedit_hold["t0"] = time.monotonic()
        def _rep():
            wedit_hold["after"] = None
            if wedit_hold.get("var") is not var:
                return
            if runner["obj"] is not None:
                wedit_hold_stop()
                return
            if time.monotonic() - wedit_hold.get("t0", 0.0) > 30.0:
                wedit_hold_stop()
                return
            wedit_nudge(var, direction)
            try:
                wedit_hold["after"] = app.after(90, _rep)
            except tk.TclError:
                pass
        try:
            wedit_hold["after"] = app.after(400, _rep)
        except tk.TclError:
            pass

    def wedit_hold_end(*_):
        wedit_hold_stop()
        apply_wedit()
    wedit_btnrow = ttk.Frame(wedit_frame)
    wedit_btnrow.pack(fill="x", pady=(4, 0))
    _reg(ttk.Button(wedit_btnrow, text="적용", width=8,
                    command=lambda: apply_wedit())).pack(side="left")
    _reg(ttk.Button(wedit_btnrow, text="저장", width=8,
                    command=lambda: save_wedit())).pack(side="left", padx=(6, 0))
    wedit_info = ttk.Label(wedit_btnrow, text="", foreground="gray")
    wedit_info.pack(side="left", padx=(8, 0))

    runner = {"obj": None}
    cumu_out = {"ver": ""}
    live = {"best": None, "cur_acc": None}
    prog = {"t": 0.0, "ndone": 0, "cps": 0.0}
    hist = {"x": [], "best": [], "cur": []}
    hist_state = {"last_draw": 0.0, "t0": 0.0, "xmode": "%"}
    whist = {"x": [], "W": [], "O": [], "a": [], "n": [], "N": [], "t": [],
             "names": None, "ver": None, "t0": 0.0,
             "last_rec": 0.0, "last_best": None}
    gfx = {"labels": None, "bars": None, "texts": [],
           "step_lines": [],
           "last_mtime": 0.0, "last_sig": None,
           "last_amtime": 0.0, "last_asig": None,
           "last_render": 0.0}
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

    def _update_bars(vals, title_acc=None, draw=True):
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
        if draw:
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
        _hide_cursor()
        xs, yb, yc = hist["x"], hist["best"], hist["cur"]
        n = len(xs)
        st = max(1, -(-n // 2000))
        dxs, dyb, dyc = xs[::st], yb[::st], yc[::st]
        line_best.set_data(dxs, dyb)
        line_cur.set_data(dxs, dyc)
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
            vis_b = [vv for xx, vv in zip(dxs, dyb) if xx >= lo]
            vis_c = [vv for xx, vv in zip(dxs, dyc) if xx >= lo]
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
            if len(hist["x"]) > 200000:
                cut = len(hist["x"]) - 200000
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
        try:
            old = preview_art.get("line")
            if old is not None:
                old.remove()
            preview_art["line"] = None
        except Exception:
            pass
        _clear_whist()

    def _clear_whist():
        for k in ("x", "W", "O", "a", "n", "N", "t"):
            whist[k].clear()
        whist["names"] = None
        whist["ver"] = None
        whist["t0"] = 0.0
        whist["last_rec"] = 0.0
        whist["last_best"] = None
        scrub["active"] = False
        scrub["order"] = None
        scrub["blit_bg"] = None
        scrub["blit_key"] = None
        scrub.pop("pending", None)
        _hide_cursor()
        try:
            scrub_scale.configure(to=0)
            scrub_var.set(0)
            scrub_label.configure(text="—")
        except (tk.TclError, AttributeError):
            pass

    def _update_scrub_label(i=None):
        try:
            n = len(whist["x"])
            if n == 0:
                scrub_label.configure(text="—")
                return
            if i is None:
                i = int(float(scrub_var.get()))
            i = max(0, min(n - 1, i))
            a = whist["a"][i]
            atxt = f" · acc {a:.3f}" if isinstance(a, (int, float)) else ""
            try:
                nn, NN = whist["n"][i], whist["N"][i]
                nntxt = f" · {nn:,}/{NN:,}" if isinstance(nn, int) and isinstance(NN, int) and NN > 0 else ""
            except (IndexError, TypeError):
                nntxt = ""
            try:
                tt = whist["t"][i]
                ttxt = f" · {tt:.1f}s" if isinstance(tt, (int, float)) else ""
            except (IndexError, TypeError):
                ttxt = ""
            scrub_label.configure(text=f"{i + 1}/{n} · {whist['x'][i]:.1f}%{nntxt}{ttxt}{atxt}")
        except (tk.TclError, AttributeError, ValueError, TypeError):
            pass

    def _hide_cursor():
        try:
            ln = scrub.get("cursor")
            if ln is not None:
                ln.remove()
        except Exception:
            pass
        scrub["cursor"] = None

    def _enable_scrub():
        try:
            if len(whist["x"]) >= 2:
                scrub_scale.state(["!disabled"])
                scrub_scale.configure(to=len(whist["x"]) - 1)
                scrub_var.set(len(whist["x"]) - 1)
                _update_scrub_label(len(whist["x"]) - 1)
                return True
        except (tk.TclError, AttributeError):
            pass
        return False

    def _refresh_whist(league, ver):
        tag = f"{league}|{ver}" if ver else ""
        if tag and whist.get("ver") == tag and len(whist["x"]) >= 2:
            _enable_scrub()
            return
        _clear_whist()
        if not ver:
            return
        try:
            d = json.load(open(whist_path(league, ver), encoding="utf-8"))
            xs, W = d.get("x"), d.get("W")
            if not (isinstance(xs, list) and isinstance(W, list)) or len(xs) < 2 or len(xs) != len(W):
                return
            O, a = d.get("O"), d.get("a")
            whist["x"] = [float(v) for v in xs]
            whist["W"] = [list(map(float, r)) for r in W]
            whist["O"] = [(list(map(float, r)) if isinstance(r, list) else None) for r in O] if isinstance(O, list) and len(O) == len(xs) else [None] * len(xs)
            whist["a"] = [(float(v) if isinstance(v, (int, float)) else None) for v in a] if isinstance(a, list) and len(a) == len(xs) else [None] * len(xs)
            nn, NN, tt = d.get("n"), d.get("N"), d.get("t")
            whist["n"] = [int(v) for v in nn] if isinstance(nn, list) and len(nn) == len(xs) else [0] * len(xs)
            whist["N"] = [int(v) for v in NN] if isinstance(NN, list) and len(NN) == len(xs) else [0] * len(xs)
            whist["t"] = [(float(v) if isinstance(v, (int, float)) else 0.0) for v in tt] if isinstance(tt, list) and len(tt) == len(xs) else [0.0] * len(xs)
            whist["names"] = list(d.get("names")) if isinstance(d.get("names"), list) else None
            whist["ver"] = tag
            hist["x"].clear()
            hist["best"].clear()
            hist["cur"].clear()
            _b = float("-inf")
            for _i in range(len(whist["x"])):
                _ca = whist["a"][_i]
                _ca = _ca if isinstance(_ca, (int, float)) else 0.0
                _b = max(_b, _ca)
                hist["x"].append(whist["x"][_i])
                hist["best"].append(_b)
                hist["cur"].append(_ca)
            hist_state["xmode"] = "%"
            _draw_progress()
            _enable_scrub()
        except (OSError, ValueError, TypeError):
            _clear_whist()

    def _blit_key(labels):
        try:
            wgt = canvas.get_tk_widget()
            return (tuple(labels),
                    tuple(ax_w.get_xlim()), tuple(ax_w.get_ylim()),
                    tuple(ax_p.get_xlim()), tuple(ax_p.get_ylim()),
                    (wgt.winfo_width(), wgt.winfo_height()))
        except Exception:
            return None

    def _blit_update():
        try:
            bg = scrub.get("blit_bg")
            if bg is None:
                return False
            fig.canvas.restore_region(bg)
            for art in list(gfx["bars"] or []) + list(gfx["texts"] or []):
                ax_w.draw_artist(art)
            ln = scrub.get("cursor")
            if ln is not None and ln.axes is ax_p and ln.get_visible():
                ax_p.draw_artist(ln)
            pv = preview_art.get("line")
            if pv is not None and pv.axes is ax_p and pv.get_visible():
                ax_p.draw_artist(pv)
            fig.canvas.blit(fig.bbox)
            return True
        except Exception:
            scrub["blit_bg"] = None
            return False

    def _move_cursor(x):
        try:
            ln = scrub.get("cursor")
            if ln is None or ln.axes is not ax_p:
                ln = ax_p.axvline(x, color="#f59e0b", linewidth=1.2, zorder=5)
                scrub["cursor"] = ln
                return False
            ln.set_xdata([x, x])
            lo, hi = ax_p.get_xlim()
            if not (lo <= x <= hi):
                span = hi - lo if hi > lo else 10.0
                ax_p.set_xlim(x - span / 2, x + span / 2)
                return False
            return True
        except Exception:
            return False

    def _capture_bg():
        try:
            for art in list(gfx["bars"] or []) + list(gfx["texts"] or []):
                art.set_visible(False)
            ln = scrub.get("cursor")
            if ln is not None:
                try:
                    ln.set_visible(False)
                except Exception:
                    pass
            pv = preview_art.get("line")
            if pv is not None:
                try:
                    pv.set_visible(False)
                except Exception:
                    pass
            fig.canvas.draw()
            scrub["blit_bg"] = fig.canvas.copy_from_bbox(fig.bbox)
            for art in list(gfx["bars"] or []) + list(gfx["texts"] or []):
                art.set_visible(True)
            if ln is not None:
                try:
                    ln.set_visible(True)
                except Exception:
                    pass
            if pv is not None:
                try:
                    pv.set_visible(True)
                except Exception:
                    pass
            return True
        except Exception:
            scrub["blit_bg"] = None
            return False

    def show_w_hist(i):
        n = len(whist["x"])
        if n == 0:
            return
        i = max(0, min(n - 1, int(i)))
        try:
            if scrub.get("order") is None:
                ref = None
                for k in range(n - 1, -1, -1):
                    if isinstance(whist["O"][k], list):
                        ref = whist["O"][k]
                        break
                if ref is None:
                    ref = whist["W"][n - 1]
                scrub["order"] = sorted(range(len(ref)),
                                        key=lambda j: -abs(ref[j]) if isinstance(ref[j], (int, float)) else 0)
            order = scrub["order"]
            raw = whist["names"] or [f"f{j}" for j in range(len(whist["W"][i]))]
            labels = [FEATURE_KO.get(raw[j], raw[j]) if j < len(raw) else f"f{j}" for j in order]
            vals = [whist["W"][i][j] if j < len(whist["W"][i]) else 0.0 for j in order]
            _ensure_bars(labels)
            _update_bars(vals, None, draw=False)
            moved = _move_cursor(whist["x"][i])
            key = _blit_key(labels)
            if (not moved) or scrub.get("blit_bg") is None or key != scrub.get("blit_key"):
                if _capture_bg():
                    scrub["blit_key"] = _blit_key(labels)
            if not _blit_update():
                canvas.draw_idle()
        except Exception:
            pass
        try:
            wrow = whist["W"][i]
            if wedit_vars and len(wrow) == len(wedit_vars):
                for (_, var), v in zip(wedit_vars, wrow):
                    try:
                        var.set(f"{v:.4f}" if isinstance(v, (int, float)) else "0.0000")
                    except tk.TclError:
                        pass
                wedit_pending["w"] = None
                try:
                    wedit_info.configure(text="")
                except tk.TclError:
                    pass
        except (IndexError, TypeError):
            pass
        _update_scrub_label(i)

    def _pump_scrub():
        scrub["scheduled"] = False
        try:
            if runner["obj"] is not None:
                return
            if "pending" not in scrub:
                return
            scrub["last_show"] = time.monotonic()
            show_w_hist(scrub.pop("pending"))
        except (tk.TclError, AttributeError):
            pass

    def _scrub_to(i):
        scrub["active"] = True
        scrub["pending"] = i
        now = time.monotonic()
        if now - scrub.get("last_show", 0.0) < 0.04:
            _update_scrub_label(i)
            if not scrub.get("scheduled"):
                scrub["scheduled"] = True
                try:
                    app.after(40, _pump_scrub)
                except tk.TclError:
                    scrub["scheduled"] = False
            return
        scrub["last_show"] = now
        show_w_hist(scrub.pop("pending"))

    def on_scrub(*_):
        if runner["obj"] is not None:
            return
        if len(whist["x"]) == 0:
            _refresh_whist(v_league.get(), v_ver.get().strip())
            if len(whist["x"]) == 0:
                v_status.set("기록 없음 — 전수탐색 실행 후 생성됨")
                return
        try:
            want = int(float(scrub_var.get()))
        except (ValueError, TypeError):
            return
        _scrub_to(want)

    def _record_w(d, shown, names, league="", ver=""):
        tot, nd = d.get("total"), d.get("ndone")
        if not (isinstance(tot, int) and tot > 0 and isinstance(nd, int)):
            return
        w, o, a = shown
        if len(whist["W"]) > 0 and list(w) == whist["W"][-1]:
            return
        max_c = d.get("max_c", 0)
        eff = max_c if isinstance(max_c, int) and max_c > 0 else tot
        pct = min(100.0, nd / eff * 100)
        now = time.monotonic()
        whist["last_rec"] = now
        whist["last_best"] = d.get("best")
        if league and ver:
            whist["ver"] = f"{league}|{ver}"
        if not whist.get("t0"):
            whist["t0"] = now
        whist["x"].append(pct)
        whist["n"].append(nd)
        whist["N"].append(eff)
        whist["t"].append(now - whist["t0"])
        whist["W"].append(list(w))
        whist["O"].append(list(o) if isinstance(o, list) else None)
        whist["a"].append(a)
        whist["names"] = names
        if len(whist["x"]) > 50000:
            cut = len(whist["x"]) - 50000
            del whist["x"][:cut]
            del whist["W"][:cut]
            del whist["O"][:cut]
            del whist["a"][:cut]
            del whist["n"][:cut]
            del whist["N"][:cut]
            del whist["t"][:cut]
        try:
            n = len(whist["x"])
            scrub_scale.configure(to=max(0, n - 1))
            if not scrub["active"]:
                scrub_var.set(n - 1)
                _update_scrub_label(n - 1)
        except (tk.TclError, AttributeError):
            pass

    def _on_ax_click(event):
        try:
            if event.inaxes is not ax_p:
                return
            if runner["obj"] is not None:
                return
            xs = whist["x"]
            if len(xs) == 0 or event.xdata is None:
                return
            i = min(range(len(xs)), key=lambda k: abs(xs[k] - event.xdata))
            try:
                canvas.get_tk_widget().grab_set()
            except tk.TclError:
                pass
            chart_drag["on"] = True
            scrub_var.set(i)
            _scrub_to(i)
        except Exception:
            pass

    def _on_ax_motion(event):
        try:
            if not chart_drag["on"]:
                return
            if runner["obj"] is not None:
                return
            xs = whist["x"]
            if len(xs) == 0:
                return
            if event.xdata is None:
                try:
                    bbox = ax_p.get_window_extent()
                    i = 0 if event.x <= bbox.x0 else len(xs) - 1
                except Exception:
                    return
            else:
                i = min(range(len(xs)), key=lambda k: abs(xs[k] - event.xdata))
            scrub_var.set(i)
            _scrub_to(i)
        except Exception:
            pass

    def _on_ax_release(event):
        chart_drag["on"] = False
        try:
            canvas.get_tk_widget().grab_release()
        except tk.TclError:
            pass

    def clear_train_view():
        gfx["labels"] = None
        redraw_graphs()
        _reset_progress()
        v_acc.set("−")

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
        if len(hist["x"]) > 200000:
            cut = len(hist["x"]) - 200000
            del hist["x"][:cut]
            del hist["best"][:cut]
            del hist["cur"][:cut]
        _draw_progress()
        hist_state["last_draw"] = 0.0

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
        eta = max(0.0, (total - ndone) / cps) if cps > 0 else float("inf")
        pct = min(100.0, ndone / max(total, 1) * 100)
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
                                if now - gfx.get("last_render", 0.0) >= 0.05:
                                    gfx["last_render"] = now
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
                    tot = d.get("total")
                    max_c = d.get("max_c", 0)
                    eff = max_c if isinstance(max_c, int) and max_c > 0 else tot
                    if isinstance(d.get("ndone"), int) and isinstance(eff, int) and eff > 0:
                        _refresh_status(d["ndone"], eff)
                    if isinstance(d.get("ndone"), int) and isinstance(d.get("best"), (int, float)):
                        pct = min(100.0, d["ndone"] / eff * 100) if isinstance(eff, int) and eff > 0 else 0.0
                        _push_hist(pct, float(d["best"]),
                                   float(cur_acc) if isinstance(cur_acc, (int, float)) else None)
                    if sig != gfx["last_sig"]:
                        gfx["last_sig"] = sig
                        running = runner["obj"] is not None
                        fnames = _feat_names_for(o["league"], o["ver"])
                        shown = None
                        rargs = None
                        if running and isinstance(cur_w, list) and cur_w and isinstance(cur_acc, (int, float)):
                            rargs = (cur_w, fnames, float(cur_acc), best_w)
                            shown = (cur_w, best_w if isinstance(best_w, list) else None, float(cur_acc))
                        elif isinstance(best_w, list) and best_w:
                            rargs = (best_w, fnames, None, None)
                            bb = d.get("best")
                            shown = (best_w, None, float(bb) if isinstance(bb, (int, float)) else None)
                        if shown is not None:
                            try:
                                _record_w(d, shown, fnames, o["league"], o["ver"])
                            except Exception:
                                pass
                            now = time.monotonic()
                            if now - gfx.get("last_render", 0.0) >= 0.05:
                                gfx["last_render"] = now
                                try:
                                    redraw_graphs(rargs[0], rargs[1], title_acc=rargs[2], order_w=rargs[3])
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
            try:
                if v_mode.get() == "cumu":
                    btn_feat.configure(state="disabled")
            except (tk.TclError, AttributeError):
                pass
            try:
                os.remove(live_path(v_league.get(), v_ver.get().strip()))
            except OSError:
                pass
            if tail:
                v_status.set(f"종료 (코드 {rc}) · {tail}")
            else:
                v_status.set(f"종료 (코드 {rc})")
            ui_state["just_done"] = True
            try:
                if v_mode.get() == "train":
                    ev = f"{v_ver.get().strip()}-ens"
                    if ev and os.path.exists(tune_path(v_league.get(), ev)):
                        v_ver.set(ev)
                elif v_mode.get() == "cumu":
                    ev = None
                    try:
                        ev = cumu_out.get("ver")
                    except NameError:
                        ev = None
                    if not ev:
                        ev = f"{v_ver.get().strip()}-cumu"
                    if ev and os.path.exists(tune_path(v_league.get(), ev)):
                        v_ver.set(ev)
            except tk.TclError:
                pass
            refresh_ver_list()
            request_acc()
            poll_once()
            refresh_wedit()
            _draw_progress()
            if len(whist["x"]) >= 2:
                _enable_scrub()
            else:
                _refresh_whist(v_league.get(), v_ver.get().strip())
        app.after(0, _ui)

    def poll_once():
        o = current_options()
        if o["mode"] == "train" and runner["obj"] is None:
            if ui_state.get("just_done"):
                ui_state["just_done"] = False
            else:
                clear_train_view()
                return
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
        if not v_ver.get().strip() and v_train.get().strip() and v_mode.get() not in ("grid", "auto", "cumu"):
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
            if not isinstance(art0.get("weights"), list):
                v_status.set(f"legacy 베이스 필요: {o['ver']} (현재 {art0.get('model_type')})")
                return
        if o["mode"] == "cumu":
            _cb = o.get("cumu_base", "")
            if _cb:
                try:
                    art0 = json.load(open(tune_path(o["league"], _cb), encoding="utf-8"))
                except (OSError, ValueError):
                    art0 = None
                if not isinstance(art0, dict):
                    v_status.set(f"아티팩트 없음: {_cb} (버전 확인)")
                    return
                if not isinstance(art0.get("weights"), list):
                    v_status.set(f"legacy 베이스 필요: {_cb} (현재 {art0.get('model_type')})")
                    return
                o["cumu_ver"] = _cb + "-cumu"
            else:
                _ff = o.get("features", "")
                _ft = f"-f{len(_ff.split(','))}" if _ff else ""
                o["cumu_ver"] = f"cumu{_ft}-{now_tag()}"
            cumu_out["ver"] = o.get("cumu_ver", "") if o["mode"] == "cumu" else ""
        set_opts_enabled(False)
        for _w in list(_dw_widgets):
            try:
                _w.state(["!disabled"])
            except (tk.TclError, AttributeError):
                pass
        cmd = build_command(o)
        write_live_draw_w(force=True)
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
        try:
            scrub_scale.state(["disabled"])
        except (tk.TclError, AttributeError):
            pass
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

    def on_delete_ver():
        if runner["obj"] is not None:
            v_status.set("실행 중에는 삭제할 수 없습니다")
            return
        lg, ver = v_league.get(), v_ver.get().strip()
        if not ver:
            v_status.set("삭제할 버전이 없습니다")
            return
        try:
            ok = messagebox.askyesno("버전 삭제", f"{lg} {ver}\n아티팩트·체크포인트를 삭제할까요?")
        except tk.TclError:
            return
        if not ok:
            return
        base = os.path.join(ROOT, "ml", "permatch", f"{lg}_{ver}")
        gone = []
        for ext in (".json", ".grid.json", ".auto.json", ".live.json", ".whist.json", ".xgb.json", ".cumu.json", ".wrong.json"):
            try:
                os.remove(base + ext)
                gone.append(ext)
            except OSError:
                pass
        v_ver.set("")
        refresh_ver_list()
        request_acc()
        poll_once()
        v_status.set(f"삭제됨: {ver} ({', '.join(gone) if gone else '파일 없음'})")

    def on_league_pick(*_):
        v_league.set(KO2CODE.get(v_league_ko.get(), v_league.get()))
        apply_preset()
        _refresh_whist(v_league.get(), v_ver.get().strip())
        refresh_ver_list()
        request_acc()
        poll_once()

    def on_mode_pick(*_):
        v_mode.set(MO2CODE.get(v_mode_ko.get(), v_mode.get()))
        refresh_detail()
        _refresh_whist(v_league.get(), v_ver.get().strip())
        if v_mode.get() == "train":
            apply_preset()
            ui_state["just_done"] = False
            clear_train_view()
            refresh_wedit()
            request_acc()
        else:
            poll_once()
            request_acc()

    def live_path(league, ver):
        return os.path.join(ROOT, "ml", "permatch", f"{league}_{ver}.live.json")

    def write_live_draw_w(*_, force=False):
        if runner["obj"] is None and not force:
            return
        ver = v_ver.get().strip()
        if not ver:
            return
        try:
            with open(live_path(v_league.get(), ver), "w") as f:
                json.dump({"draw_w": v_draww.get().strip()}, f)
        except (OSError, ValueError):
            pass

    def on_close():
        r = runner.get("obj")
        if r is not None:
            try:
                ok = messagebox.askyesno("종료", "학습이 실행 중입니다. 중지하고 종료할까요?")
            except tk.TclError:
                ok = True
            if not ok:
                return
            try:
                r.stop()
            except Exception:
                pass
            try:
                if r.proc is not None:
                    r.proc.wait(timeout=5)
            except Exception:
                try:
                    if r.proc is not None:
                        r.proc.kill()
                except Exception:
                    pass
            runner["obj"] = None
        try:
            app.destroy()
        except tk.TclError:
            pass

    app.protocol("WM_DELETE_WINDOW", on_close)
    btn_start.configure(command=on_start)
    btn_stop.configure(command=on_stop)
    v_league_ko.trace_add("write", on_league_pick)
    v_mode_ko.trace_add("write", on_mode_pick)
    v_ver.trace_add("write", apply_ver_features)
    v_train.trace_add("write", apply_train_ver)
    v_train.trace_add("write", refresh_summary)
    v_valid.trace_add("write", refresh_summary)
    v_traincount.trace_add("write", apply_preset)
    v_draww.trace_add("write", write_live_draw_w)
    v_gridstep.trace_add("write", refresh_step_lines)
    v_wmin.trace_add("write", refresh_step_lines)
    v_wmax.trace_add("write", refresh_step_lines)
    apply_preset()
    refresh_summary()
    refresh_ver_list()
    refresh_detail()
    toggle_detail()
    request_acc()
    redraw_graphs()
    app.mainloop()


if __name__ == "__main__":
    main()
