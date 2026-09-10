"""ecg_app -- yarisma gunu icin tiklamali arayuz.

    python ecg_app.py

NE YAPAR
--------
Klasor secersin, dugmeye basarsin, JSON cikar. Yarisma gunu uzun bir CMD
komutunu stres altinda yazmak yerine.

NE YAPMAZ -- ve bu en onemli tasarim karari
-------------------------------------------
Bu arayuz cikarim mantigini YENIDEN YAZMAZ. Yaptigi tek sey, dogrulanmis
`make_submission.py`'yi bir ALT SUREC olarak cagirmak ve ciktisini gostermek.

Nedeni: elinizde 750/750 dogrulanmis, bit-birebir kanitlanmis bir yol var.
Onu bir GUI'nin icine kopyalamak yeni hata yuzeyi acar. Boyle:

  * arayuz cokerse CLI ayni dosyayi uretir
  * uretilen JSON, elle yazilan komutunkiyle BIREBIR ayni olur
  * dogrulanmis hatta hicbir sey eklenmez

Arayuz calistirdigi komutu ekranda gosterir; "Komutu kopyala" ile alip
dogrudan CMD'ye yapistirabilirsin. Arayuz bir kolaylik katmanidir, bir
bagimlilik degil.

YARISMA GUNU
------------
  1. Paket klasoru : competition_package
  2. Test verisi   : USB'den acilan klasor
  3. Kimlikler zaten dolu (807466 / 4997133)
  4. Hiz: MP11
  5. CALISTIR -> JSON -> otomatik dogrulama -> yukle

`--ids` VARSAYILAN OLARAK KAPALI. Yarisma verisinde id listesi kullanilmaz;
kutu yalnizca kendi provalarin icin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

APP_TITLE = "tkt-26 · EKG Siniflandirma"

# Kimlikler KYS'den. Yanlis yazilmasi teslimi gecersiz kilar, o yuzden
# onceden dolu geliyor -- ama duzenlenebilir, kilitli degil.
DEFAULT_TEAM_NAME = "tkt-26"
DEFAULT_TEAM_ID = "807466"
DEFAULT_APP_ID = "4997133"

CLASSES = ("NORMAL", "AFIB", "AFL", "LBBB", "RBBB")

# Olculen hiz (750 kayit, gercek pakette). Sure tahmini icin.
MS_PER_RECORD = {
    "mp11": 497,     # --threads 1 --model-parallel 11
    "mp8": 532,      # --threads 1 --model-parallel 8
    "threads2": 1670,  # --threads 2
    "seri": 7357,    # bayraksiz
}

SPEED_LABELS = [
    ("mp11", "MP11  ~6.2 dk / 750 kayit   (ONERILEN)"),
    ("mp8", "MP8   ~6.6 dk / 750 kayit"),
    ("threads2", "threads=2  ~21 dk / 750   (basit yedek)"),
    ("seri", "seri  ~92 dk / 750   (en yavas, en az varsayim)"),
]

SPEED_FLAGS = {
    "mp11": ["--threads", "1", "--model-parallel", "11"],
    "mp8": ["--threads", "1", "--model-parallel", "8"],
    "threads2": ["--threads", "2"],
    "seri": [],
}


# ==========================================================================
# CEKIRDEK -- tkinter'siz test edilebilir
# ==========================================================================

def count_records(root):
    """root altindaki .hea sayisi ve ilk birkac id."""
    n, sample = 0, []
    if not root or not os.path.isdir(root):
        return 0, []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".hea"):
                n += 1
                if len(sample) < 5:
                    sample.append(os.path.splitext(fn)[0])
    return n, sample


def estimate_seconds(n_records, speed):
    return n_records * MS_PER_RECORD.get(speed, MS_PER_RECORD["mp11"]) / 1000.0


def human_time(sec):
    if sec < 90:
        return "%.0f saniye" % sec
    if sec < 5400:
        return "%.0f dakika" % (sec / 60.0)
    return "%.1f saat" % (sec / 3600.0)


def output_name(team_id):
    """Kilavuz onerisi: TEAM_<ID>_FINAL.json. Onek zaten varsa iki kez yazma."""
    tid = (team_id or "").strip()
    stem = tid if tid.upper().startswith("TEAM_") else "TEAM_%s" % tid
    return "%s_FINAL.json" % stem


def build_command(pkg_dir, data_root, team_name, team_id, app_id, speed,
                  ids_file="", out_name=""):
    """make_submission.py komutunu kur. GUI ve CLI ayni seyi calistirir."""
    cmd = [sys.executable, os.path.join(pkg_dir, "make_submission.py"),
           "--root", data_root,
           "--team-name", team_name,
           "--team-id", team_id,
           "--application-id", app_id]
    cmd += SPEED_FLAGS.get(speed, SPEED_FLAGS["mp11"])
    if ids_file:
        cmd += ["--ids", ids_file]
    if out_name:
        cmd += ["--out", out_name]
    return cmd


def build_validate_command(pkg_dir, json_path, ids_file=""):
    cmd = [sys.executable, os.path.join(pkg_dir, "make_submission.py"),
           "--validate", json_path]
    if ids_file:
        cmd += ["--ids", ids_file]
    return cmd


def quote_for_cmd(cmd):
    """Windows CMD'ye yapistirilabilir tek satir."""
    out = []
    for part in cmd:
        out.append('"%s"' % part if (" " in part or "\\" in part) else part)
    return " ".join(out)


def check_ready(pkg_dir, data_root, team_id, app_id):
    """Calistirmadan once engelleyici sorunlar. Bos liste = hazir."""
    problems = []
    if not pkg_dir or not os.path.isdir(pkg_dir):
        problems.append("Paket klasoru secilmedi.")
    else:
        for f in ("make_submission.py", "predict.py", "manifest.json"):
            if not os.path.exists(os.path.join(pkg_dir, f)):
                problems.append("Pakette %s yok -- dogru klasor mu?" % f)
        mdir = os.path.join(pkg_dir, "models")
        if os.path.isdir(mdir):
            n = len([f for f in os.listdir(mdir) if f.endswith(".onnx")])
            if n == 0:
                problems.append("models/ klasorunde .onnx yok.")
        else:
            problems.append("Pakette models/ klasoru yok.")
    if not data_root or not os.path.isdir(data_root):
        problems.append("Test verisi klasoru secilmedi.")
    else:
        n, _ = count_records(data_root)
        if n == 0:
            problems.append("Secilen klasorde hic .hea kaydi yok.")
    if not str(team_id).strip():
        problems.append("team_id bos.")
    if not str(app_id).strip():
        problems.append("application_id bos.")
    return problems


def parse_progress(line, total):
    """'  tahmin 300/750  120 sn' -> 0.40. Bulamazsa None."""
    s = line.strip()
    for tag in ("tahmin ", "on isleme "):
        if s.startswith(tag):
            try:
                frac = s[len(tag):].split()[0]
                done, tot = frac.split("/")
                tot = int(tot) or total
                return min(1.0, int(done) / float(tot))
            except (ValueError, IndexError, ZeroDivisionError):
                return None
    return None


def read_result(json_path):
    """Uretilen JSON'dan ozet: (kayit, sinif dagilimi, hata)."""
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:                     # noqa: BLE001
        return None, None, str(exc)
    preds = doc.get("predictions") or []
    dist = {c: 0 for c in CLASSES}
    for p in preds:
        c = p.get("predicted_class")
        if c in dist:
            dist[c] += 1
    return len(preds), dist, None


# ==========================================================================
# ARAYUZ
# ==========================================================================

def main():
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        print("tkinter bulunamadi. Arayuz acilamiyor.")
        print()
        print("Bu bir engel DEGIL -- komut satiri yolu her zaman calisir:")
        print()
        print("  python make_submission.py --root <TEST_KLASORU> \\")
        print("      --team-name \"%s\" --team-id %s --application-id %s \\"
              % (DEFAULT_TEAM_NAME, DEFAULT_TEAM_ID, DEFAULT_APP_ID))
        print("      --threads 1 --model-parallel 11")
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    guess_pkg = ""
    for cand in ("competition_package", ".", "package"):
        p = os.path.join(here, cand)
        if os.path.exists(os.path.join(p, "make_submission.py")):
            guess_pkg = os.path.abspath(p)
            break

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("880x720")

    state = {"proc": None, "running": False, "out_path": "", "total": 0}

    v_pkg = tk.StringVar(value=guess_pkg)
    v_data = tk.StringVar(value="")
    v_name = tk.StringVar(value=DEFAULT_TEAM_NAME)
    v_tid = tk.StringVar(value=DEFAULT_TEAM_ID)
    v_aid = tk.StringVar(value=DEFAULT_APP_ID)
    v_speed = tk.StringVar(value="mp11")
    v_useids = tk.BooleanVar(value=False)
    v_ids = tk.StringVar(value="")
    v_info = tk.StringVar(value="Test verisi klasoru secin.")
    v_status = tk.StringVar(value="hazir")

    pad = {"padx": 8, "pady": 4}
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    # ---- klasorler -------------------------------------------------------
    box1 = ttk.LabelFrame(frm, text="1 · Klasorler", padding=8)
    box1.pack(fill="x", **pad)

    def pick(var, title):
        d = filedialog.askdirectory(title=title)
        if d:
            var.set(os.path.normpath(d))
            refresh()

    ttk.Label(box1, text="Paket (competition_package)").grid(row=0, column=0, sticky="w")
    ttk.Entry(box1, textvariable=v_pkg, width=72).grid(row=0, column=1, sticky="we", padx=6)
    ttk.Button(box1, text="Sec...",
               command=lambda: pick(v_pkg, "competition_package")).grid(row=0, column=2)

    ttk.Label(box1, text="Test verisi").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(box1, textvariable=v_data, width=72).grid(row=1, column=1, sticky="we",
                                                        padx=6, pady=(6, 0))
    ttk.Button(box1, text="Sec...",
               command=lambda: pick(v_data, "Test verisi klasoru")).grid(row=1, column=2,
                                                                         pady=(6, 0))
    box1.columnconfigure(1, weight=1)

    lbl_info = ttk.Label(box1, textvariable=v_info, foreground="#0a5")
    lbl_info.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

    # ---- kimlik ----------------------------------------------------------
    box2 = ttk.LabelFrame(frm, text="2 · Kimlik  (KYS ile AYNI olmali)", padding=8)
    box2.pack(fill="x", **pad)
    for i, (lab, var) in enumerate((("team_name", v_name),
                                    ("team_id", v_tid),
                                    ("application_id", v_aid))):
        ttk.Label(box2, text=lab).grid(row=0, column=i * 2, sticky="w", padx=(0, 4))
        ttk.Entry(box2, textvariable=var, width=18).grid(row=0, column=i * 2 + 1,
                                                          padx=(0, 16))

    # ---- hiz -------------------------------------------------------------
    box3 = ttk.LabelFrame(frm, text="3 · Hiz", padding=8)
    box3.pack(fill="x", **pad)
    for i, (key, label) in enumerate(SPEED_LABELS):
        ttk.Radiobutton(box3, text=label, value=key, variable=v_speed,
                        command=lambda: refresh()).grid(row=i, column=0, sticky="w")

    idsrow = ttk.Frame(box3)
    idsrow.grid(row=len(SPEED_LABELS), column=0, sticky="w", pady=(8, 0))
    ttk.Checkbutton(idsrow, text="id listesi kullan  (yalniz PROVA icin - "
                                 "yarisma verisinde KULLANMA)",
                    variable=v_useids, command=lambda: refresh()).pack(side="left")
    ttk.Entry(idsrow, textvariable=v_ids, width=30).pack(side="left", padx=6)
    ttk.Button(idsrow, text="...", width=3,
               command=lambda: (v_ids.set(filedialog.askopenfilename(
                   title="id listesi") or v_ids.get()), refresh())).pack(side="left")

    # ---- komut -----------------------------------------------------------
    box4 = ttk.LabelFrame(frm, text="4 · Calistirilacak komut", padding=8)
    box4.pack(fill="x", **pad)
    txt_cmd = tk.Text(box4, height=3, wrap="word", font=("Consolas", 9))
    txt_cmd.pack(fill="x")

    def copy_cmd():
        root.clipboard_clear()
        root.clipboard_append(txt_cmd.get("1.0", "end").strip())
        v_status.set("komut panoya kopyalandi")

    ttk.Button(box4, text="Komutu kopyala", command=copy_cmd).pack(anchor="e", pady=(6, 0))

    # ---- calistir --------------------------------------------------------
    box5 = ttk.Frame(frm)
    box5.pack(fill="x", **pad)
    btn_run = ttk.Button(box5, text="CALISTIR", width=18)
    btn_run.pack(side="left")
    btn_stop = ttk.Button(box5, text="Durdur", width=10, state="disabled")
    btn_stop.pack(side="left", padx=6)
    bar = ttk.Progressbar(box5, mode="determinate", maximum=100)
    bar.pack(side="left", fill="x", expand=True, padx=10)
    ttk.Label(box5, textvariable=v_status).pack(side="left")

    # ---- cikti -----------------------------------------------------------
    box6 = ttk.LabelFrame(frm, text="5 · Cikti", padding=8)
    box6.pack(fill="both", expand=True, **pad)
    txt = tk.Text(box6, wrap="none", font=("Consolas", 9), background="#111",
                  foreground="#ddd", insertbackground="#ddd")
    sb = ttk.Scrollbar(box6, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    txt.pack(fill="both", expand=True)

    def log(line, tag=None):
        txt.insert("end", line if line.endswith("\n") else line + "\n", tag)
        txt.see("end")

    txt.tag_configure("ok", foreground="#5f5")
    txt.tag_configure("err", foreground="#f77")
    txt.tag_configure("warn", foreground="#fd5")

    # ---- mantik ----------------------------------------------------------
    def current_cmd():
        return build_command(
            v_pkg.get(), v_data.get(), v_name.get(), v_tid.get(), v_aid.get(),
            v_speed.get(), v_ids.get() if v_useids.get() else "")

    def refresh():
        n, sample = count_records(v_data.get())
        state["total"] = n
        if n:
            eta = estimate_seconds(n, v_speed.get())
            v_info.set("%d kayit bulundu · tahmini sure %s · cikti: %s"
                       % (n, human_time(eta), output_name(v_tid.get())))
            lbl_info.configure(foreground="#0a5")
        else:
            v_info.set("Test verisi klasoru secin (icinde .hea dosyalari olmali).")
            lbl_info.configure(foreground="#a50")
        txt_cmd.delete("1.0", "end")
        txt_cmd.insert("1.0", quote_for_cmd(current_cmd()))

    for var in (v_tid, v_name, v_aid):
        var.trace_add("write", lambda *_a: refresh())

    def pump(proc):
        """Alt surecin ciktisini satir satir arayuze aktar."""
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            frac = parse_progress(line, state["total"])
            if frac is not None:
                root.after(0, lambda f=frac: bar.configure(value=100 * f))
            tag = None
            low = line.lower()
            if "all checks passed" in low:
                tag = "ok"
            elif "hata" in low or "reddedildi" in low or "error" in low:
                tag = "err"
            elif "uyari" in low or "dikkat" in low or "not:" in low:
                tag = "warn"
            root.after(0, lambda l=line, t=tag: log(l, t))
        proc.stdout.close()
        rc = proc.wait()
        root.after(0, lambda: finish(rc))

    def finish(rc):
        state["running"] = False
        btn_run.configure(state="normal", text="CALISTIR")
        btn_stop.configure(state="disabled")
        bar.configure(value=100 if rc == 0 else 0)
        if rc != 0:
            v_status.set("BASARISIZ (kod %d)" % rc)
            log("", None)
            log("Kosu basarisiz. Yukaridaki hatayi okuyun.", "err")
            log("Komut satirindan da deneyebilirsiniz -- 'Komutu kopyala'.", "warn")
            return
        v_status.set("bitti")
        out = os.path.join(v_pkg.get(), output_name(v_tid.get()))
        state["out_path"] = out
        log("", None)
        n, dist, err = read_result(out)
        if err:
            log("Cikti okunamadi: %s" % err, "err")
            return
        log("=" * 58, "ok")
        log("%s  ·  %d kayit" % (os.path.basename(out), n), "ok")
        log("  " + "  ".join("%s %d" % (c, dist[c]) for c in CLASSES), "ok")
        log("=" * 58, "ok")
        log("Tam yol: %s" % out)
        # Kilavuz md. 3: teslim etmeden once dosyayi denetle.
        log("")
        log("Dogrulama kosuluyor...")
        run_validate(out)

    def run_validate(path):
        cmd = build_validate_command(v_pkg.get(), path,
                                     v_ids.get() if v_useids.get() else "")
        try:
            p = subprocess.run(cmd, cwd=v_pkg.get(), capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
        except Exception as exc:                 # noqa: BLE001
            log("Dogrulama calistirilamadi: %s" % exc, "err")
            return
        last = [l for l in (p.stdout or "").splitlines() if l.strip()]
        if last and "all checks passed" in last[-1]:
            log("DOGRULAMA GECTI -- dosya yuklenmeye hazir.", "ok")
        else:
            log("DOGRULAMA GECMEDI -- YUKLEMEYIN:", "err")
            for l in last[-15:]:
                log("  " + l, "err")

    def start():
        if state["running"]:
            return
        problems = check_ready(v_pkg.get(), v_data.get(), v_tid.get(), v_aid.get())
        if problems:
            txt.delete("1.0", "end")
            log("Baslatilamadi:", "err")
            for pb in problems:
                log("  - " + pb, "err")
            return
        out = os.path.join(v_pkg.get(), output_name(v_tid.get()))
        if os.path.exists(out):
            log("NOT: %s zaten var, uzerine yazilacak." % os.path.basename(out),
                "warn")
        txt.delete("1.0", "end")
        bar.configure(value=0)
        cmd = current_cmd()
        log(quote_for_cmd(cmd))
        log("-" * 58)
        state["running"] = True
        btn_run.configure(state="disabled", text="calisiyor...")
        btn_stop.configure(state="normal")
        v_status.set("calisiyor")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(cmd, cwd=v_pkg.get(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        state["proc"] = proc
        threading.Thread(target=pump, args=(proc,), daemon=True).start()

    def stop():
        p = state.get("proc")
        if p and p.poll() is None:
            p.terminate()
            v_status.set("durduruldu")
            log("Kullanici durdurdu -- JSON YAZILMADI.", "warn")

    btn_run.configure(command=start)
    btn_stop.configure(command=stop)

    refresh()
    log("Yarisma gunu: paket + test klasoru sec, kimlikleri kontrol et,")
    log("MP11 secili birak, CALISTIR'a bas. Bittiginde dogrulama otomatik kosar.")
    log("")
    log("Bu arayuz make_submission.py'yi alt surec olarak cagirir --")
    log("uretilen JSON, elle yazilan komutunkiyle BIREBIR aynidir.")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
