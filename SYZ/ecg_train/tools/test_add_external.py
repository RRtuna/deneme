"""test_add_external -- add_external.py'yi GERCEK yarisma cache semasiyla sina.

    python tools/test_add_external.py
    python tools/test_add_external.py --n 5000      (gercek olcek, yavas)

Gecici bir klasorde su duzeni kurar ve uzerinde add_external.py kosturur:

    <tmp>/SYZ/                          <- YARISMA VERI KOKU
        Normal/NORM_000777/JS36591.hea
        AFIB/AFIB_000001/JS10001.hea
        ...
        ecg_train_v2/                   <- PROJE KOKU
            cache/index.csv             header_path = "data/Normal/NORM_000777/JS36591.hea"
            external_data/ningbo/g26/   <- icinde JS36591.hea AYNISI (SIZINTI)
            external_data/chapman/g1/

Yani index.csv'deki yol ile diskteki gercek yol AYNI DEGIL ve record_id ile
dosya adi da AYNI DEGIL (NORM_000777 vs JS36591) -- gercek kurulumun iki zor
noktasi. Betik bunlarin ikisini de sinar.

Sinanan davranislar
-------------------
  1  header_path cozuluyor, "competition readable: N/N"
  2  JS36591 kopyasi AD cakismasiyla yakalaniyor (stem-only overlap)
  3  yeniden adlandirilmis + olceklenmis kopya KORELASYONLA yakalaniyor
  4  --dry-run hicbir cache dosyasi yazmiyor
  5  yol cozulemezse cache YAZILMIYOR (sert abort)
  6  kismi okuma (N-1/N) durumunda da cache YAZILMIYOR
  7  gercek yazma cokmuyor; cikti index.csv semasi dogru
  8  --single-label coklu etiketli kaydi eliyor
  9  --balance-sources kaynaklari dengeliyor
 10  --per-class kotasi uygulaniyor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ecg_preprocess as ep  # noqa: E402

ADD_EXTERNAL = os.path.join(HERE, "add_external.py")

CODES = {"Normal": "426783006", "AFIB": "164889003", "AFL": "164890007",
         "LBBB": "164909002", "RBBB": "59118001"}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("\n        " + str(detail)[:400]) if detail and not ok else ""))


# --------------------------------------------------------------------------

def write_record(path, sig, fs=500, dx=None):
    """Minimal WFDB kaydi (.hea + .dat)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    dat = base + ".dat"
    with open(path, "w") as fh:
        fh.write("%s %d %d %d\n" % (base, sig.shape[0], fs, sig.shape[1]))
        for lead in range(sig.shape[0]):
            fh.write("%s 16 1000/mV 16 0 0 0 0 lead%d\n" % (dat, lead))
        if dx:
            fh.write("#Dx: %s\n" % dx)
    (sig * 1000).astype("<i2").T.tofile(
        os.path.join(os.path.dirname(path), dat))


def synth(seed, n=2500):
    """Birbirinden GERCEKTEN farkli sentetik EKG: degisken hiz, morfoloji, faz.

    Hepsi ayni sablondan uretilirse korelasyon kapisi her seye takilir ve test
    hicbir sey olcmez -- bu yuzden parametreler genis araliktan cekiliyor.
    """
    r = np.random.RandomState(seed)
    t = np.arange(n) / 500.0
    hr = r.uniform(0.7, 2.4)
    phase = r.uniform(0, 6.28)
    width = int(r.uniform(16, 48))
    amp = r.uniform(1.2, 5.0)
    pol = r.choice([-1, 1])
    sig = np.zeros((12, n))
    for lead in range(12):
        sig[lead] = (r.uniform(0.3, 1.8) * np.sin(2 * np.pi * hr * t
                                                  + phase + lead * 0.37)
                     + 0.3 * r.randn(n))
        step = max(int(500 / hr), width + 5)
        for i in range(int(r.uniform(0, step)), n - width, step):
            sig[lead, i:i + width] += pol * amp * r.uniform(0.6, 1.4) \
                * np.hanning(width)
    return sig


def build(tmp, n_comp, n_ext):
    """Gercek semayla fixture kur. Dondurur: (proje_koku, veri_koku)."""
    data_root = os.path.join(tmp, "SYZ")
    proj_root = os.path.join(data_root, "ecg_train_v2")
    cache = os.path.join(proj_root, "cache")
    os.makedirs(cache, exist_ok=True)

    classes = list(ep.CLASSES)
    rows, sigs = [], {}
    for i in range(n_comp):
        cls = classes[i % len(classes)]
        rid = "%s_%06d" % (cls.upper()[:4], i)
        # record_id ile dosya adi KASTEN farkli -- gercek kurulumdaki gibi
        fname = "JS36591" if i == 0 else "JS%05d" % (10000 + i)
        rel = "data/%s/%s/%s.hea" % (cls, rid, fname)
        real = os.path.join(data_root, cls, rid, fname + ".hea")
        sig = synth(1000 + i * 7)
        sigs[fname] = (sig, cls)
        write_record(real, sig, dx=CODES[cls])
        rows.append({
            "record_id": rid,
            "relative_path": rel[len("data/"):],
            "header_path": rel,
            "signal_path": rel[:-4] + ".dat",
            "label": cls,
            "class_id": classes.index(cls),
            "sampling_rate_hz": 500,
            "lead_count": 12,
            "duration_sec": 5,
            "file_format": "wfdb",
            "split": "test_public" if i % 7 == 0 else "train",
        })

    with open(os.path.join(cache, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(cache, "X.npy"),
            np.zeros((n_comp, 12, ep.TARGET_LEN), dtype=np.float32))
    np.save(os.path.join(cache, "y.npy"),
            np.array([r["class_id"] for r in rows], dtype=np.int64))
    np.save(os.path.join(cache, "F.npy"),
            np.zeros((n_comp, len(ep.FEATURE_NAMES)), dtype=np.float32))
    json.dump({"target_fs": ep.TARGET_FS},
              open(os.path.join(cache, "meta.json"), "w"))

    # ---- dis veri --------------------------------------------------------
    ning = os.path.join(proj_root, "external_data", "ningbo", "g26")
    chap = os.path.join(proj_root, "external_data", "chapman", "g1")

    # (a) SIZINTI: yarisma kaydinin AYNISI, ayni adla
    write_record(os.path.join(ning, "JS36591.hea"), sigs["JS36591"][0],
                 dx=CODES["Normal"])

    # (b) SIZINTI: baska bir yarisma kaydi, FARKLI ad + kazanc/ofset degisik
    src_name = "JS%05d" % 10001
    write_record(os.path.join(ning, "GIZLI_KOPYA.hea"),
                 sigs[src_name][0] * 2.7 + 0.4, dx=CODES[sigs[src_name][1]])

    # (c) gercek yeni kayitlar -- Ningbo cok, Chapman az (dengeleme sinavi)
    for j in range(n_ext):
        write_record(os.path.join(ning, "NEW_%04d.hea" % j), synth(90000 + j * 13),
                     dx=CODES[["AFIB", "AFL", "LBBB", "RBBB"][j % 4]])
    for j in range(max(2, n_ext // 4)):
        write_record(os.path.join(chap, "CH_%04d.hea" % j), synth(40000 + j * 29),
                     dx=CODES[["AFL", "AFIB", "RBBB", "AFL"][j % 4]])

    # (d) COKLU etiketli kayit -- --single-label elemeli
    write_record(os.path.join(chap, "MULTI_0001.hea"), synth(555001),
                 dx="%s,%s" % (CODES["AFIB"], CODES["RBBB"]))

    return proj_root, data_root


def run(proj_root, extra):
    cmd = [sys.executable, ADD_EXTERNAL, "--source", "external_data",
           "--cache", "cache", "--out", "cache_ext"] + extra
    p = subprocess.run(cmd, cwd=proj_root, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def grab(out, label):
    for line in out.splitlines():
        if line.strip().startswith(label):
            return line.split(":", 1)[1].strip()
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=350,
                    help="yarisma kaydi sayisi (gercek olcek icin 5000)")
    ap.add_argument("--ext", type=int, default=60, help="dis kayit sayisi")
    ap.add_argument("--keep", action="store_true", help="gecici klasoru silme")
    args = ap.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix="test_add_ext_")
    try:
        print("fixture kuruluyor: %d yarisma + ~%d dis kayit ..."
              % (args.n, args.ext))
        proj, data = build(tmp, args.n, args.ext)
        print("  proje koku : %s" % proj)
        print("  veri koku  : %s" % data)
        print()

        # ---- 1) dry-run, yollar cozuluyor mu -----------------------------
        print("[1] dry-run + yol cozumu")
        rc, out = run(proj, ["--dry-run"])
        readable = grab(out, "competition readable")
        check("competition readable = %d/%d" % (args.n, args.n),
              readable == "%d/%d" % (args.n, args.n), readable or out[-500:])
        check("cikis kodu 0", rc == 0, "rc=%d" % rc)
        check("external .hea sayisi bildirildi",
              grab(out, "external .hea") is not None)

        # ---- 2) sizinti yakalandi mi -------------------------------------
        print("\n[2] sizinti taramasi")
        stem_ov = grab(out, "stem-only overlaps")
        leaks = grab(out, "exact/signature leaks")
        check("JS36591 ad cakismasiyla yakalandi (stem-only >= 1)",
              stem_ov is not None and int(stem_ov) >= 1, "stem-only=%s" % stem_ov)
        check("yeniden adlandirilmis kopya yakalandi (leaks >= 1)",
              leaks is not None and int(leaks) >= 1, "leaks=%s" % leaks)
        check("GIZLI_KOPYA raporlandi", "GIZLI_KOPYA" in out)

        # ---- 3) coklu etiket elendi --------------------------------------
        print("\n[3] --single-label")
        check("coklu etiketli kayit elendi",
              "birden fazla hedef tani" in out, out[-500:])

        # ---- 4) dry-run hicbir sey yazmadi -------------------------------
        print("\n[4] --dry-run yan etkisiz")
        check("cache_ext olusturulmadi",
              not os.path.exists(os.path.join(proj, "cache_ext")))

        # ---- 5) yol cozulemezse sert abort -------------------------------
        print("\n[5] yanlis --data-root -> sert abort")
        rc2, out2 = run(proj, ["--data-root", os.path.join(tmp, "yok"),
                               "--project-root", os.path.join(tmp, "yok")])
        check("sifir okunabilirlik bildirildi",
              "competition readable: 0/" in out2 or "readable: 0/" in out2,
              out2[-400:])
        check("cache YAZILMADI",
              not os.path.exists(os.path.join(proj, "cache_ext")))

        # ---- 6) kismi okuma -> yine yazma yok ----------------------------
        print("\n[6] kismi okuma (bir .dat silinmis) -> sert abort")
        victim = None
        for dirpath, _d, files in os.walk(data):
            if "ecg_train_v2" in dirpath:
                continue
            for f in files:
                if f.endswith(".dat"):
                    victim = os.path.join(dirpath, f)
                    break
            if victim:
                break
        moved = victim + ".bak"
        os.rename(victim, moved)
        rc3, out3 = run(proj, [])
        check("CACHE YAZILMADI mesaji", "CACHE YAZILMADI" in out3, out3[-400:])
        check("cache_ext olusturulmadi",
              not os.path.exists(os.path.join(proj, "cache_ext")))
        os.rename(moved, victim)

        # ---- 7) gercek yazma ---------------------------------------------
        print("\n[7] gercek yazma + cikti semasi")
        rc4, out4 = run(proj, ["--per-class", "8", "--balance-sources"])
        check("cikis kodu 0", rc4 == 0, out4[-600:])
        idx = os.path.join(proj, "cache_ext", "index.csv")
        check("cache_ext/index.csv olustu", os.path.exists(idx))
        if os.path.exists(idx):
            with open(idx, newline="") as fh:
                rd = csv.DictReader(fh)
                hdr = rd.fieldnames
                orows = list(rd)
            check("basliklar dogru",
                  hdr == ["idx", "record_id", "header_path", "class_id",
                          "label", "split", "ok"], str(hdr))
            check("yarisma record_id korundu",
                  orows[0]["record_id"].startswith(("NORM", "AFIB", "AFL",
                                                    "LBBB", "RBBB")),
                  orows[0])
            check("class_id her satirda dolu",
                  all(r["class_id"] != "" for r in orows))
            n_extra = sum(1 for r in orows if r["split"] == "extra")
            check("extra satirlari eklendi (>0)", n_extra > 0,
                  "extra=%d" % n_extra)
            check("extra sayisi = toplam - yarisma",
                  len(orows) == args.n + n_extra,
                  "%d vs %d+%d" % (len(orows), args.n, n_extra))
            X = np.load(os.path.join(proj, "cache_ext", "X.npy"), mmap_mode="r")
            y = np.load(os.path.join(proj, "cache_ext", "y.npy"))
            check("X/y/index satir sayilari tutarli",
                  X.shape[0] == len(y) == len(orows),
                  "X=%d y=%d idx=%d" % (X.shape[0], len(y), len(orows)))

            # ---- 8/9) per-class + kaynak dengesi -------------------------
            print("\n[8] --per-class ve --balance-sources")
            from collections import Counter
            cnt = Counter(r["label"] for r in orows if r["split"] == "extra")
            check("hicbir sinif kotayi asmadi (<=8)",
                  all(v <= 8 for v in cnt.values()), dict(cnt))
            meta = json.load(open(os.path.join(proj, "cache_ext", "meta.json")))
            per_src = meta.get("external", {}).get("per_source", {})
            check("birden fazla kaynaktan alindi", len(per_src) >= 2,
                  list(per_src))
            check("meta balance_sources=True",
                  meta.get("external", {}).get("balance_sources") is True)

        print()
        n_ok = sum(1 for _n, ok in RESULTS if ok)
        print("=" * 62)
        print("%d/%d gecti" % (n_ok, len(RESULTS)))
        if n_ok != len(RESULTS):
            print("\nBASARISIZ:")
            for name, ok in RESULTS:
                if not ok:
                    print("  - %s" % name)
        return 0 if n_ok == len(RESULTS) else 1
    finally:
        if args.keep:
            print("\ngecici klasor korundu: %s" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
