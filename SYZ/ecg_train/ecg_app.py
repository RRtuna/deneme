"""ecg_app -- EKG siniflandirma uygulamasi.

    python ecg_app.py

Bir EKG dosyasi ac, 12 derivasyonu gor, modelin ne dedigini gor.
Klasor ac, kayitlar arasinda gez. Hazir oldugunda tum klasoru isleyip
yarisma JSON'unu uret.

EKRAN
-----
  sol   : 12 derivasyon dalga formu
  sag   : modelin 5 sinif olasiligi (cubuk), guven, ana olcumler
  ust   : dosya/klasor acma, kayitlar arasi gezinme
  alt   : klasoru isle -> JSON, dogrula

IKI FARKLI YOL, BILEREK
-----------------------
  Tek kayit  -> `predict.predict_record()` DOGRUDAN cagrilir.
                Etkilesimli, aninda; ayni dogrulanmis fonksiyon.

  Klasor/JSON -> `make_submission.py` ALT SUREC olarak cagrilir.
                Teslim dosyasini ureten yol budur ve arayuz ona hicbir sey
                eklemez. Arayuz cokse bile CLI ayni dosyayi uretir; uretilen
                JSON elle yazilan komutunkiyle BIREBIR aynidir.

Bagimlilik: yalnizca paketin kendisi + tkinter. matplotlib YOK -- dalga
formu dogrudan Canvas'a ciziliyor, boylece teslim paketine yeni bagimlilik
girmiyor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import numpy as np

APP_TITLE = "tkt-26 · EKG Siniflandirma"

DEFAULT_TEAM_NAME = "tkt-26"
DEFAULT_TEAM_ID = "807466"
DEFAULT_APP_ID = "4997133"

CLASSES = ("NORMAL", "AFIB", "AFL", "LBBB", "RBBB")

CLASS_TR = {
    "NORMAL": "Normal ritim",
    "AFIB": "Atriyal fibrilasyon",
    "AFL": "Atriyal flutter",
    "LBBB": "Sol dal blogu",
    "RBBB": "Sag dal blogu",
}

CLASS_COLOR = {
    "NORMAL": "#3aa757",
    "AFIB": "#d9534f",
    "AFL": "#e8a33d",
    "LBBB": "#4a7fd4",
    "RBBB": "#8a63c9",
}

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6")

# Olculen hizlar (750 kayit, gercek paket) -- sure tahmini icin.
MS_PER_RECORD = {"mp11": 497, "mp8": 532, "threads2": 1670, "seri": 7357}
SPEED_FLAGS = {
    "mp11": ["--threads", "1", "--model-parallel", "11"],
    "mp8": ["--threads", "1", "--model-parallel", "8"],
    "threads2": ["--threads", "2"],
    "seri": [],
}
SPEED_LABELS = [
    ("mp11", "MP11 · ~6 dk / 750   (onerilen)"),
    ("mp8", "MP8 · ~7 dk / 750"),
    ("threads2", "threads=2 · ~21 dk / 750"),
    ("seri", "seri · ~92 dk / 750"),
]

# Sag panelde gosterilecek olcumler: (ozellik adi, etiket, birim/bicim)
SHOW_FEATURES = [
    ("hr_mean", "Kalp hizi", "%.0f bpm"),
    ("rr_cv", "RR degiskenligi (CV)", "%.3f"),
    ("rr_rmssd", "RMSSD", "%.3f s"),
    ("rr_irregular_frac", "Duzensiz RR orani", "%.2f"),
    ("rr_shannon", "RR entropi", "%.2f"),
]


# ==========================================================================
# CEKIRDEK -- tkinter'siz test edilebilir
# ==========================================================================

def find_records(root):
    """root altindaki (id, .hea yolu) ciftleri, id'ye gore sirali."""
    out = {}
    if not root or not os.path.isdir(root):
        return []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".hea"):
                out.setdefault(os.path.splitext(fn)[0],
                               os.path.join(dirpath, fn))
    return [(k, out[k]) for k in sorted(out)]


def envelope(y, n_cols):
    """Sinyali n_cols sutuna indir; her sutunda (min, max) tut.

    Duz seyreltme QRS tepelerini kacirir -- 5000 ornegi 900 piksele
    dusururken her 5. ornegi almak, R tepesinin tam ustune denk gelmezse
    tepe kaybolur. Min/max zarfi tepeleri KORUR.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0 or n_cols <= 0:
        return np.zeros(0), np.zeros(0)
    if n <= n_cols:
        return y.copy(), y.copy()
    edges = np.linspace(0, n, n_cols + 1).astype(int)
    lo = np.empty(n_cols)
    hi = np.empty(n_cols)
    for i in range(n_cols):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        seg = y[a:b]
        lo[i] = seg.min()
        hi[i] = seg.max()
    return lo, hi


def trace_scale(sig, n_leads, lane_h, headroom=0.42):
    """Tum derivasyonlar icin TEK olcek (mV -> piksel).

    Ortak olcek klinik olarak dogru: derivasyonlar arasi genlik orani
    korunur. Her derivasyon kendi kendine olceklenirse V1 ile V5 esit
    buyuklukte gorunur ve morfoloji bilgisi kaybolur.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size == 0:
        return 1.0
    amp = np.nanmax(np.abs(sig))
    if not np.isfinite(amp) or amp < 1e-9:
        return 1.0
    return (lane_h * headroom) / amp


def layout_traces(sig, width, height, n_leads=12, pad_left=34, pad_top=6):
    """Her derivasyon icin cizim koordinatlari.

    Dondurur: [(lead_idx, baseline_y, [(x, y_lo, y_hi), ...]), ...]
    """
    sig = np.asarray(sig, dtype=np.float64)
    n_leads = min(n_leads, sig.shape[0]) if sig.ndim == 2 else 0
    if n_leads == 0 or width <= pad_left + 4 or height <= pad_top * 2 + 4:
        return []
    plot_w = width - pad_left - 6
    lane_h = (height - 2 * pad_top) / float(n_leads)
    scale = trace_scale(sig[:n_leads], n_leads, lane_h)
    out = []
    for li in range(n_leads):
        base = pad_top + lane_h * (li + 0.5)
        lo, hi = envelope(sig[li], int(plot_w))
        cols = []
        for i in range(lo.size):
            x = pad_left + i
            cols.append((x, base - hi[i] * scale, base - lo[i] * scale))
        out.append((li, base, cols))
    return out


def sort_probs(prob, classes=CLASSES):
    """[(sinif, olasilik), ...] buyukten kucuge."""
    p = np.asarray(prob, dtype=np.float64).ravel()
    pairs = [(classes[i], float(p[i])) for i in range(min(len(classes), p.size))]
    return sorted(pairs, key=lambda t: -t[1])


def confidence_label(prob):
    """En yuksek olasiliga gore guven etiketi.

    Esikler modelin kendi hata dagilimindan: cift hatalarinin %93.4'u dusuk
    guvenli bolgede. Kullaniciya "model burada kararsiz" demek, sahte bir
    kesinlik sunmaktan iyidir.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    if p.size == 0:
        return "-", "#888"
    top = float(p.max())
    if top >= 0.80:
        return "YUKSEK", "#3aa757"
    if top >= 0.55:
        return "ORTA", "#e8a33d"
    return "DUSUK -- model kararsiz", "#d9534f"


def pick_features(feats, names, wanted=SHOW_FEATURES):
    """Gosterilecek olcumleri (etiket, metin) olarak dondur."""
    out = []
    if feats is None:
        return out
    f = np.asarray(feats, dtype=np.float64).ravel()
    idx = {n: i for i, n in enumerate(names)}
    for key, label, fmt in wanted:
        i = idx.get(key)
        if i is None or i >= f.size or not np.isfinite(f[i]):
            continue
        out.append((label, fmt % f[i]))
    return out


def estimate_seconds(n, speed):
    return n * MS_PER_RECORD.get(speed, MS_PER_RECORD["mp11"]) / 1000.0


def human_time(sec):
    if sec < 90:
        return "%.0f sn" % sec
    if sec < 5400:
        return "%.0f dk" % (sec / 60.0)
    return "%.1f saat" % (sec / 3600.0)


def output_name(team_id):
    tid = (team_id or "").strip()
    stem = tid if tid.upper().startswith("TEAM_") else "TEAM_%s" % tid
    return "%s_FINAL.json" % stem


def build_batch_command(pkg_dir, data_root, team_name, team_id, app_id,
                        speed, ids_file=""):
    cmd = [sys.executable, os.path.join(pkg_dir, "make_submission.py"),
           "--root", data_root, "--team-name", team_name,
           "--team-id", team_id, "--application-id", app_id]
    cmd += SPEED_FLAGS.get(speed, SPEED_FLAGS["mp11"])
    if ids_file:
        cmd += ["--ids", ids_file]
    return cmd


def quote_cmd(cmd):
    return " ".join('"%s"' % p if (" " in p or "\\" in p) else p for p in cmd)


def parse_progress(line, total):
    s = line.strip()
    for tag in ("tahmin ", "on isleme "):
        if s.startswith(tag):
            try:
                done, tot = s[len(tag):].split()[0].split("/")
                return min(1.0, int(done) / float(int(tot) or total))
            except (ValueError, IndexError, ZeroDivisionError):
                return None
    return None


# ==========================================================================
# MODEL -- paketin kendi kodu, sarmalanmadan
# ==========================================================================

class Model:
    """Paketi tembel yukler. Ilk tahmin oturumlari kurar (~4 sn), sonrasi hizli."""

    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.pr = None
        self.ep = None
        self.wl = None
        self.error = None
        self.n_models = 0

    def load(self):
        if self.pr is not None or self.error:
            return self.error is None
        try:
            if self.pkg_dir not in sys.path:
                sys.path.insert(0, self.pkg_dir)
            import predict as pr                  # noqa: PLC0415
            import ecg_preprocess as ep           # noqa: PLC0415
            import wfdb_lite as wl                # noqa: PLC0415
            self.pr, self.ep, self.wl = pr, ep, wl
            man = getattr(pr, "MANIFEST", None) or {}
            for k in ("modeller", "models", "uyeler", "members"):
                v = man.get(k)
                if isinstance(v, (list, tuple)):
                    self.n_models = len(v)
                    break
            return True
        except Exception as exc:                  # noqa: BLE001
            self.error = "%s: %s" % (type(exc).__name__, exc)
            return False

    def read_signal(self, hea_path):
        sig, fs, leads = self.wl.read_record(hea_path)
        return np.asarray(sig, dtype=np.float64), float(fs), leads

    def features(self, sig, fs):
        try:
            return (np.asarray(self.ep.extract_features(sig, fs)),
                    list(self.ep.FEATURE_NAMES))
        except Exception:                         # noqa: BLE001
            return None, []

    def predict(self, hea_path):
        """(sinif, olasilik) -- paketin kendi predict_record'u, degistirilmeden."""
        fn = getattr(self.pr, "predict_record", None)
        if fn is None:
            raise RuntimeError("pakette predict_record yok")
        for cand in (hea_path, hea_path[:-4] if hea_path.lower().endswith(".hea")
                     else hea_path):
            try:
                idx, prob = fn(cand)
                p = np.asarray(prob, dtype=np.float64).ravel()
                s = p.sum()
                if p.size == len(CLASSES) and np.all(np.isfinite(p)) and s > 0:
                    return int(np.argmax(p)), p / s
            except Exception as exc:              # noqa: BLE001
                last = exc
        raise RuntimeError(str(last))


# ==========================================================================
# ARAYUZ
# ==========================================================================

def main():
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        print("tkinter bulunamadi -- arayuz acilamiyor.")
        print("Komut satiri yolu her zaman calisir:")
        print('  python make_submission.py --root <KLASOR> --team-name "%s" '
              "--team-id %s --application-id %s --threads 1 --model-parallel 11"
              % (DEFAULT_TEAM_NAME, DEFAULT_TEAM_ID, DEFAULT_APP_ID))
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    pkg_guess = here
    for cand in ("competition_package", ".", "package"):
        p = os.path.abspath(os.path.join(here, cand))
        if os.path.exists(os.path.join(p, "predict.py")):
            pkg_guess = p
            break

    S = {"records": [], "i": -1, "model": Model(pkg_guess), "sig": None,
         "fs": 500.0, "prob": None, "busy": False, "proc": None}

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1180x760")
    root.minsize(940, 620)

    v_pkg = tk.StringVar(value=pkg_guess)
    v_pos = tk.StringVar(value="kayit yok")
    v_name = tk.StringVar(value="")
    v_status = tk.StringVar(value="Bir EKG dosyasi ya da klasor acin.")
    v_tid = tk.StringVar(value=DEFAULT_TEAM_ID)
    v_aid = tk.StringVar(value=DEFAULT_APP_ID)
    v_tname = tk.StringVar(value=DEFAULT_TEAM_NAME)
    v_speed = tk.StringVar(value="mp11")

    # ---- ust serit -------------------------------------------------------
    top = ttk.Frame(root, padding=(10, 8))
    top.pack(fill="x")
    ttk.Button(top, text="Dosya ac", width=11,
               command=lambda: open_file()).pack(side="left")
    ttk.Button(top, text="Klasor ac", width=11,
               command=lambda: open_dir()).pack(side="left", padx=(6, 14))
    b_prev = ttk.Button(top, text="◀", width=3, command=lambda: step(-1))
    b_prev.pack(side="left")
    ttk.Label(top, textvariable=v_pos, width=14, anchor="center").pack(side="left")
    b_next = ttk.Button(top, text="▶", width=3, command=lambda: step(1))
    b_next.pack(side="left")
    ttk.Label(top, textvariable=v_name, font=("Segoe UI", 10, "bold")).pack(
        side="left", padx=14)

    # ---- govde -----------------------------------------------------------
    body = ttk.Frame(root)
    body.pack(fill="both", expand=True, padx=10)

    left = ttk.LabelFrame(body, text="12 derivasyon", padding=4)
    left.pack(side="left", fill="both", expand=True)
    cv = tk.Canvas(left, background="#0d0f12", highlightthickness=0)
    cv.pack(fill="both", expand=True)

    right = ttk.Frame(body, width=330)
    right.pack(side="right", fill="y", padx=(10, 0))
    right.pack_propagate(False)

    pbox = ttk.LabelFrame(right, text="Model tahmini", padding=8)
    pbox.pack(fill="x")
    lbl_top = tk.Label(pbox, text="—", font=("Segoe UI", 15, "bold"),
                       anchor="w")
    lbl_top.pack(fill="x")
    lbl_conf = tk.Label(pbox, text="", anchor="w")
    lbl_conf.pack(fill="x", pady=(0, 6))
    cv_prob = tk.Canvas(pbox, height=136, highlightthickness=0,
                        background="#f5f5f5")
    cv_prob.pack(fill="x")

    fbox = ttk.LabelFrame(right, text="Olcumler", padding=8)
    fbox.pack(fill="x", pady=(10, 0))
    feat_rows = []
    for _ in range(len(SHOW_FEATURES)):
        r = ttk.Frame(fbox)
        r.pack(fill="x")
        a = ttk.Label(r, text="", width=20, anchor="w")
        a.pack(side="left")
        b = ttk.Label(r, text="", anchor="e", font=("Consolas", 9))
        b.pack(side="right")
        feat_rows.append((a, b))

    ibox = ttk.LabelFrame(right, text="Paket", padding=8)
    ibox.pack(fill="x", pady=(10, 0))
    lbl_pkg = ttk.Label(ibox, text="", wraplength=300, justify="left")
    lbl_pkg.pack(fill="x")
    ttk.Button(ibox, text="Paket klasorunu degistir...",
               command=lambda: change_pkg()).pack(fill="x", pady=(6, 0))

    # ---- alt serit -------------------------------------------------------
    bot = ttk.LabelFrame(root, text="Tum klasoru isle → yarisma JSON'u",
                         padding=8)
    bot.pack(fill="x", padx=10, pady=(8, 10))

    r1 = ttk.Frame(bot)
    r1.pack(fill="x")
    for lab, var, w in (("team_name", v_tname, 12), ("team_id", v_tid, 10),
                        ("application_id", v_aid, 12)):
        ttk.Label(r1, text=lab).pack(side="left")
        ttk.Entry(r1, textvariable=var, width=w).pack(side="left", padx=(4, 12))
    ttk.Label(r1, text="hiz").pack(side="left")
    cmb = ttk.Combobox(r1, values=[l for _k, l in SPEED_LABELS], width=30,
                       state="readonly")
    cmb.current(0)
    cmb.pack(side="left", padx=4)
    cmb.bind("<<ComboboxSelected>>",
             lambda _e: v_speed.set(SPEED_LABELS[cmb.current()][0]))

    r2 = ttk.Frame(bot)
    r2.pack(fill="x", pady=(8, 0))
    b_batch = ttk.Button(r2, text="KLASORU ISLE", width=16,
                         command=lambda: run_batch())
    b_batch.pack(side="left")
    b_stop = ttk.Button(r2, text="Durdur", width=9, state="disabled",
                        command=lambda: stop_batch())
    b_stop.pack(side="left", padx=6)
    bar = ttk.Progressbar(r2, mode="determinate", maximum=100)
    bar.pack(side="left", fill="x", expand=True, padx=8)
    ttk.Label(r2, textvariable=v_status).pack(side="left")

    log = tk.Text(bot, height=7, wrap="none", font=("Consolas", 8),
                  background="#111", foreground="#ddd")
    log.pack(fill="x", pady=(8, 0))
    log.tag_configure("ok", foreground="#5f5")
    log.tag_configure("err", foreground="#f77")
    log.tag_configure("warn", foreground="#fd5")

    def say(t, tag=None):
        log.insert("end", t + "\n", tag)
        log.see("end")

    # ---- cizim -----------------------------------------------------------
    def draw_ecg():
        cv.delete("all")
        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 60 or h < 60:
            return
        if S["sig"] is None:
            cv.create_text(w // 2, h // 2, fill="#556",
                           font=("Segoe UI", 11),
                           text="Dosya ac ya da klasor ac")
            return
        # 1 sn'lik dikey kilavuz
        n = S["sig"].shape[1]
        secs = n / max(S["fs"], 1.0)
        plot_w = w - 40
        for k in range(int(secs) + 1):
            x = 34 + plot_w * (k / max(secs, 1e-9))
            cv.create_line(x, 4, x, h - 4, fill="#1d2530")
        for li, base, cols in layout_traces(S["sig"], w, h):
            cv.create_line(34, base, w - 6, base, fill="#1a2030")
            cv.create_text(16, base, text=LEAD_NAMES[li] if li < 12 else str(li),
                           fill="#7a8798", font=("Segoe UI", 8))
            pts = []
            for x, ylo, yhi in cols:
                pts.extend((x, ylo, x, yhi))
            if len(pts) >= 4:
                cv.create_line(*pts, fill="#4ade80", width=1)

    def draw_probs():
        cv_prob.delete("all")
        w = max(cv_prob.winfo_width(), 240)
        if S["prob"] is None:
            cv_prob.create_text(w // 2, 60, fill="#999",
                                text="tahmin bekleniyor", font=("Segoe UI", 9))
            return
        rows = sort_probs(S["prob"])
        bw = w - 116
        for i, (cls, p) in enumerate(rows):
            y = 12 + i * 26
            cv_prob.create_text(4, y, anchor="w", text=cls,
                                font=("Segoe UI", 9, "bold" if i == 0 else "normal"))
            cv_prob.create_rectangle(58, y - 8, 58 + bw, y + 8,
                                     fill="#e3e3e3", width=0)
            if p > 0.001:
                cv_prob.create_rectangle(58, y - 8, 58 + bw * p, y + 8,
                                         fill=CLASS_COLOR.get(cls, "#666"),
                                         width=0)
            cv_prob.create_text(w - 4, y, anchor="e", text="%5.1f%%" % (100 * p),
                                font=("Consolas", 9))

    def show_prediction(cls, prob, feats, names):
        S["prob"] = prob
        lbl_top.configure(text="%s — %s" % (cls, CLASS_TR.get(cls, "")),
                          fg=CLASS_COLOR.get(cls, "#000"))
        txt, col = confidence_label(prob)
        lbl_conf.configure(text="Guven: %s" % txt, fg=col)
        draw_probs()
        vals = pick_features(feats, names)
        for i, (a, b) in enumerate(feat_rows):
            if i < len(vals):
                a.configure(text=vals[i][0])
                b.configure(text=vals[i][1])
            else:
                a.configure(text="")
                b.configure(text="")

    # ---- kayit yukleme ---------------------------------------------------
    def load_index(i):
        if not S["records"] or not (0 <= i < len(S["records"])):
            return
        S["i"] = i
        rid, path = S["records"][i]
        v_pos.set("%d / %d" % (i + 1, len(S["records"])))
        v_name.set(rid)
        b_prev.configure(state="normal" if i > 0 else "disabled")
        b_next.configure(state="normal" if i < len(S["records"]) - 1 else "disabled")

        m = S["model"]
        if not m.load():
            v_status.set("paket yuklenemedi")
            say("Paket yuklenemedi: %s" % m.error, "err")
            return
        try:
            sig, fs, _leads = m.read_signal(path)
        except Exception as exc:                  # noqa: BLE001
            S["sig"] = None
            draw_ecg()
            say("%s okunamadi: %s" % (rid, exc), "err")
            return
        S["sig"], S["fs"] = sig, fs
        draw_ecg()
        lbl_top.configure(text="hesaplaniyor...", fg="#666")
        lbl_conf.configure(text="")
        S["prob"] = None
        draw_probs()
        root.update_idletasks()

        def work():
            try:
                idx, prob = m.predict(path)
                feats, names = m.features(sig, fs)
                root.after(0, lambda: show_prediction(CLASSES[idx], prob,
                                                      feats, names))
                root.after(0, lambda: v_status.set("hazir"))
            except Exception as exc:              # noqa: BLE001
                root.after(0, lambda e=exc: (
                    lbl_top.configure(text="tahmin basarisiz", fg="#d9534f"),
                    say("%s: %s" % (rid, e), "err")))
        v_status.set("model calisiyor...")
        threading.Thread(target=work, daemon=True).start()

    def step(d):
        load_index(S["i"] + d)

    def open_file():
        p = filedialog.askopenfilename(title="EKG kaydi",
                                       filetypes=[("WFDB header", "*.hea"),
                                                  ("Tum dosyalar", "*.*")])
        if not p:
            return
        S["records"] = [(os.path.splitext(os.path.basename(p))[0], p)]
        load_index(0)

    def open_dir():
        d = filedialog.askdirectory(title="EKG klasoru")
        if not d:
            return
        recs = find_records(d)
        if not recs:
            v_status.set("bu klasorde .hea yok")
            say("Secilen klasorde hic .hea kaydi yok: %s" % d, "warn")
            return
        S["records"] = recs
        S["dir"] = d
        say("%d kayit bulundu · %s" % (len(recs), d))
        say("Tahmini toplu isleme suresi: %s"
            % human_time(estimate_seconds(len(recs), v_speed.get())))
        load_index(0)

    def change_pkg():
        d = filedialog.askdirectory(title="Paket klasoru (predict.py iceren)")
        if not d:
            return
        v_pkg.set(os.path.normpath(d))
        S["model"] = Model(v_pkg.get())
        refresh_pkg()

    def refresh_pkg():
        m = S["model"]
        ok = m.load()
        if ok:
            lbl_pkg.configure(text="%s\n%d ONNX model yuklu"
                              % (os.path.basename(m.pkg_dir) or m.pkg_dir,
                                 m.n_models),
                              foreground="#2a7")
        else:
            lbl_pkg.configure(text="%s\nYUKLENEMEDI: %s"
                              % (m.pkg_dir, m.error), foreground="#c33")

    # ---- toplu isleme ----------------------------------------------------
    def run_batch():
        if S["busy"]:
            return
        d = S.get("dir")
        if not d:
            say("Once bir KLASOR acin (tek dosya yeterli degil).", "warn")
            return
        cmd = build_batch_command(v_pkg.get(), d, v_tname.get(), v_tid.get(),
                                  v_aid.get(), v_speed.get())
        log.delete("1.0", "end")
        say(quote_cmd(cmd))
        say("-" * 70)
        S["busy"] = True
        b_batch.configure(state="disabled")
        b_stop.configure(state="normal")
        bar.configure(value=0)
        v_status.set("isleniyor")
        total = len(S["records"])
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(cmd, cwd=v_pkg.get(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        S["proc"] = proc

        def pump():
            for raw in iter(proc.stdout.readline, ""):
                line = raw.rstrip("\n")
                f = parse_progress(line, total)
                if f is not None:
                    root.after(0, lambda v=f: bar.configure(value=100 * v))
                low = line.lower()
                tag = ("ok" if "all checks passed" in low else
                       "err" if ("hata" in low or "reddedildi" in low) else
                       "warn" if ("uyari" in low or "dikkat" in low) else None)
                root.after(0, lambda l=line, t=tag: say(l, t))
            proc.stdout.close()
            rc = proc.wait()
            root.after(0, lambda: batch_done(rc))
        threading.Thread(target=pump, daemon=True).start()

    def batch_done(rc):
        S["busy"] = False
        b_batch.configure(state="normal")
        b_stop.configure(state="disabled")
        if rc != 0:
            v_status.set("basarisiz")
            say("Kosu basarisiz (kod %d). Yukaridaki hatayi okuyun." % rc, "err")
            return
        bar.configure(value=100)
        v_status.set("bitti")
        out = os.path.join(v_pkg.get(), output_name(v_tid.get()))
        say("")
        say("Dogrulama kosuluyor...")
        try:
            p = subprocess.run(
                [sys.executable, os.path.join(v_pkg.get(), "make_submission.py"),
                 "--validate", out], cwd=v_pkg.get(), capture_output=True,
                text=True, encoding="utf-8", errors="replace")
            lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
            if lines and "all checks passed" in lines[-1]:
                say("DOGRULAMA GECTI · %s" % out, "ok")
            else:
                say("DOGRULAMA GECMEDI -- YUKLEMEYIN:", "err")
                for l in lines[-12:]:
                    say("  " + l, "err")
        except Exception as exc:                  # noqa: BLE001
            say("Dogrulama calistirilamadi: %s" % exc, "err")

    def stop_batch():
        p = S.get("proc")
        if p and p.poll() is None:
            p.terminate()
            v_status.set("durduruldu")
            say("Durduruldu -- JSON yazilmadi.", "warn")

    # ---- baglama ---------------------------------------------------------
    cv.bind("<Configure>", lambda _e: draw_ecg())
    cv_prob.bind("<Configure>", lambda _e: draw_probs())
    root.bind("<Left>", lambda _e: step(-1))
    root.bind("<Right>", lambda _e: step(1))
    root.bind("<Control-o>", lambda _e: open_file())
    root.bind("<Control-d>", lambda _e: open_dir())

    b_prev.configure(state="disabled")
    b_next.configure(state="disabled")
    refresh_pkg()
    root.after(60, draw_ecg)
    say("Kisayol: ← → kayit gez · Ctrl+O dosya · Ctrl+D klasor")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
