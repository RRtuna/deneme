"""test_resid -- artik ozelliklerinin SENIN kurulumunda calistigini dogrular.

    python tools/test_resid.py --cache cache

5 saniye surer, hicbir sey yazmaz, hicbir sey egitmez. Uzun bir kosuya
baslamadan once buradaki 12 kontrolun hepsi gecmeli.

Neden gerekli
-------------
`resid_features.py` senin `ecg_preprocess.py`'nin API'sini calisma aninda
kesfediyor: R tepe bulucun `detect_rpeaks` mi `detect_r` mi, tek derivasyon mu
12 derivasyon mu istiyor, derivasyon listesi hangi isimde. Bu kesif yanlis
giderse ozellikler sessizce coplukten ibaret olur ve bunu ancak saatler suren
egitimin sonunda anlarsin. Bu betik onu once yakalar.

Basari durumunda son satir tam olarak: all checks passed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []
N = [0]


def check(name, cond, detail=""):
    N[0] += 1
    ok = bool(cond)
    print("  %-46s %s%s" % (name, "GECTI" if ok else "KALDI",
                            ("   " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--fs", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=12, help="denenecek kayit sayisi")
    args = ap.parse_args(argv)

    print("1) modul yuklenebiliyor mu")
    try:
        import resid_features as rf
    except Exception as exc:                        # noqa: BLE001
        print("  resid_features yuklenemedi: %s" % exc)
        print("  -> resid_features.py ecg_train/ klasorunde mi?")
        return 1
    check("resid_features import", True)
    check("ozellik sayisi 25", rf.N_FEATURES == 25, "%d" % rf.N_FEATURES)
    check("ozellik adi sayisi eslesiyor",
          len(rf.FEATURE_NAMES) == rf.N_FEATURES)

    print()
    print("2) ecg_preprocess API'si taniniyor mu")
    ep = rf.ep
    check("ecg_preprocess yuklendi", ep is not None)
    lead_idx = rf.lead_indices()
    check("4 derivasyon indeksi bulundu", len(lead_idx) == 4,
          "%s -> %s" % (list(rf.RESID_LEADS), lead_idx))
    check("indeksler 0-11 araliginda",
          all(0 <= i < 12 for i in lead_idx))
    check("indeksler benzersiz", len(set(lead_idx)) == 4)

    print()
    print("3) cache okunabiliyor mu")
    x_path = os.path.join(args.cache, "X.npy")
    if not os.path.exists(x_path):
        print("  %s yok -- once 'python prep.py' calistir" % x_path)
        return 1
    X = np.load(x_path, mmap_mode="r")
    check("X.npy yuklendi", X.ndim == 3, "%s" % (X.shape,))
    check("12 derivasyon", X.shape[1] >= 12, "%d" % X.shape[1])

    fs = args.fs
    meta_path = os.path.join(args.cache, "meta.json")
    if not fs and os.path.exists(meta_path):
        fs = float(json.load(open(meta_path)).get("target_fs") or 0.0)
    check("ornekleme hizi bulundu", fs > 0, "%.0f Hz" % fs)
    if not fs:
        return 1

    print()
    print("4) R tepe bulucu")
    sig = np.asarray(X[0], dtype=np.float64)
    pk = rf.rpeaks(sig, fs)
    mode = rf._RPEAK_MODE[0]
    check("R tepeleri bulundu", pk.size >= 3, "%d vurus" % pk.size)
    print("     kullanilan yol: %s" % mode)
    if mode == "fallback":
        print("     NOT: senin modulunun R bulucusu kullanilamadi, betigin")
        print("     kendi basit bulucusu devreye girdi. Calisir ama mevcut 37")
        print("     ozellikle ayni vuruslari kullanmaz. Sorun degil, sadece bil.")
    rate = pk.size / (X.shape[2] / fs) * 60.0
    check("vurus hizi makul (30-250 /dk)", 30 < rate < 250, "%.0f /dk" % rate)

    print()
    print("5) ozellikler uretiliyor mu (%d kayit)" % args.n)
    t0 = time.time()
    n_try = min(args.n, X.shape[0])
    V = np.stack([rf.extract(np.asarray(X[i], dtype=np.float64), fs)
                  for i in range(n_try)])
    dt = (time.time() - t0) / n_try * 1000.0
    check("cikti sekli (%d, 25)" % n_try, V.shape == (n_try, 25), "%s" % (V.shape,))
    check("hepsi sonlu (NaN/inf yok)", np.isfinite(V).all(),
          "%d bozuk" % int((~np.isfinite(V)).sum()))
    check("hepsi ayni degil (sabit sutun yok)",
          int((V.std(axis=0) > 1e-9).sum()) >= 20,
          "%d/25 sutun degisken" % int((V.std(axis=0) > 1e-9).sum()))
    print("     kayit basina maliyet: %.1f ms" % dt)

    print()
    print("6) tekrarlanabilirlik")
    a = rf.extract(np.asarray(X[0], dtype=np.float64), fs)
    b = rf.extract(np.asarray(X[0], dtype=np.float64), fs)
    check("ayni girdi -> ayni cikti", np.array_equal(a, b))

    print()
    print("7) sinif ayirimi var mi (kaba on bakis)")
    idx_path = os.path.join(args.cache, "index.csv")
    if os.path.exists(idx_path):
        rows = list(csv.DictReader(open(idx_path, newline="")))
        y = np.array([int(r["label"]) for r in rows])
        dev = np.array([i for i, r in enumerate(rows)
                        if r["split"] != "test_public"])
        # Her siniftan ayri ayri ornekle. index.csv sinifa gore grupluysa
        # bastan 200 kayit almak tek sinif getirir ve her AUC 0.5 cikar.
        pair = np.concatenate([dev[y[dev] == c][:100] for c in (1, 2)])
        if len(pair) >= 40 and len(np.unique(y[pair])) == 2:
            P = np.stack([rf.extract(np.asarray(X[i], dtype=np.float64), fs)
                          for i in pair])
            L = y[pair]

            def auc(x, l):
                o = np.argsort(x)
                r = np.empty(len(x))
                r[o] = np.arange(len(x)) + 1
                n1, n0 = int((l == 2).sum()), int((l == 1).sum())
                if n1 == 0 or n0 == 0:
                    return 0.5
                return (r[l == 2].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

            a_ = np.array([max(auc(P[:, j], L), 1 - auc(P[:, j], L))
                           for j in range(25)])
            best = int(a_.argmax())
            check("en az bir ozellik AUC > 0.60", a_.max() > 0.60,
                  "%s = %.3f" % (rf.FEATURE_NAMES[best], a_.max()))
            print("     ilk 5:")
            for j in np.argsort(-a_)[:5]:
                print("       %-22s %.3f" % (rf.FEATURE_NAMES[j], a_[j]))
            print("     (bu YALNIZCA %d kayitlik kaba bir bakis; asil karar" % len(pair))
            print("      resid_probe.py'nin fold-disi sayilarindadir)")
        else:
            print("  AFIB/AFL kaydi az, atlandi")
    else:
        print("  index.csv yok, atlandi")

    print()
    print("%d kontrol, %d kaldi" % (N[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  KALDI: %s" % f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
