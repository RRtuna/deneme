"""test_qrst_gain -- QRST ozellikleri GERCEK veride ne katiyor? (ag egitmeden)

    python test_qrst_gain.py                    # cache/ ve F.npy kullanir
    python test_qrst_gain.py --cache cache --limit 400

Neden bu script var
-------------------
QRST iptali fikrini denemek, ozellik yazmak + F.npy'yi yeniden uretmek + tum
aglari yeniden egitmek demek: bir kac gun. Ama fikrin ise yarayip yaramadigini
anlamak icin ag egitmek gerekmez.

Bu script mevcut `F.npy`'deki 37 ozelligi oldugu gibi alir, uzerine 24 QRST
ozelligini hesaplar ve **yalnizca AFIB/AFL ikilisinde** basit bir lojistik
regresyonla ikisini karsilastirir. Ag degil, ozellik uzayi test edilir.

Nasil yorumlanir
----------------
Cikan "USTUNE kattigi" degeri, AFIB/AFL ikili dogruluguna eklenen puandir.

    < 0.01   -> fikri BIRAK. Ozellik uzayinda yoksa agda da olmayacak.
    0.01-0.03 -> sinirda. Tam egitim ~1 gun; kazanc muhtemelen gurultude kalir.
    > 0.03   -> KOS. Projenin tavan taramasina gore ikili dogrulukta +0.07
                genel macro-F1'de ~+0.03, yani olculebilir bir iyilesme demek.

Bu bir on eleme testidir: gecmek tam egitimin ise yarayacagini garanti etmez,
ama kalmak bosuna gun harcamani engeller.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qrst_features as qf

PAIR = ("AFIB", "AFL")


def load_preprocess():
    """ecg_preprocess'i getir ve filtreleme/R-tepe cagrilarini API'ye uyarla."""
    import ecg_preprocess as ep

    if hasattr(ep, "filter_500"):
        filt = ep.filter_500
    elif hasattr(ep, "sosfiltfilt") and hasattr(ep, "butter_highpass_sos"):
        def filt(sig, _ep=ep):
            fs = float(getattr(_ep, "SRC_FS", getattr(_ep, "NATIVE_FS", 500.0)))
            y = _ep.sosfiltfilt(_ep.butter_highpass_sos(
                getattr(_ep, "HP_CUTOFF", 0.5), fs, getattr(_ep, "HP_ORDER", 2)), sig)
            return _ep.sosfiltfilt(_ep.butter_lowpass_sos(
                min(40.0, 0.45 * fs), fs, getattr(_ep, "LP_ORDER", 4)), y)
    else:
        raise SystemExit("ecg_preprocess icinde filter_500 veya butter_* yok")

    detect = getattr(ep, "detect_r", None) or getattr(ep, "detect_rpeaks", None)
    if detect is None:
        raise SystemExit("ecg_preprocess icinde detect_r/detect_rpeaks yok")

    def rpeaks(filtered, fs):
        for arg in (filtered,
                    filtered[1] if filtered.ndim == 2 and filtered.shape[0] > 1
                    else filtered[0]):
            try:
                r = np.asarray(detect(arg), dtype=int).ravel()
                if r.size:
                    return r
            except Exception:                            # noqa: BLE001, S112
                continue
        return np.array([], dtype=int)

    fs = float(getattr(ep, "SRC_FS", getattr(ep, "NATIVE_FS", 500.0)))
    return filt, rpeaks, fs


def cv_scores(X, y, seeds=(0, 1, 2)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    accs = []
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=4000, C=1.0,
                                   random_state=seed).fit(sc.transform(X[tr]), y[tr])
            pred[va] = m.predict(sc.transform(X[va]))
        accs.append(float((pred == y).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--limit", type=int, default=0,
                    help="kac AFIB/AFL kaydi kullanilsin (0 = hepsi)")
    ap.add_argument("--out", default="F_qrst.npy",
                    help="hesaplanan 24 ozelligi buraya yaz (tum cache satirlari)")
    ap.add_argument("--all-rows", action="store_true",
                    help="sadece AFIB/AFL degil, TUM kayitlar icin hesapla ve kaydet")
    args = ap.parse_args(argv)

    index_path = os.path.join(args.cache, "index.csv")
    if not os.path.exists(index_path):
        raise SystemExit("%s yok -- once 'python prep.py' kos" % index_path)
    with open(index_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    f_path = os.path.join(args.cache, "F.npy")
    if not os.path.exists(f_path):
        raise SystemExit("%s yok -- mevcut 37 ozellik olmadan karsilastirilamaz"
                         % f_path)
    F_old = np.load(f_path).astype(np.float64)
    if F_old.shape[0] != len(rows):
        raise SystemExit("F.npy %d satir, index.csv %d satir"
                         % (F_old.shape[0], len(rows)))

    filt, rpeaks, fs = load_preprocess()
    import wfdb_lite as wl

    # test_public HICBIR sekilde kullanilmaz -- bu bir gelistirme testidir.
    dev = [i for i, r in enumerate(rows) if r["split"] != "test_public"]
    targets = dev if args.all_rows else \
        [i for i in dev if rows[i]["label_name"] in PAIR]
    if args.limit:
        rng = np.random.default_rng(0)
        targets = list(np.array(targets)[rng.permutation(len(targets))[:args.limit]])
    if not targets:
        raise SystemExit("AFIB/AFL kaydi bulunamadi")

    print("cache      : %s" % os.path.abspath(args.cache))
    print("gelistirme : %d kayit  (test_public disarida)" % len(dev))
    print("islenecek  : %d kayit" % len(targets))
    print("on isleme  : fs=%g" % fs)
    print()

    F_new = np.zeros((len(rows), qf.N_FEATURES), dtype=np.float32)
    ok = np.zeros(len(rows), dtype=bool)
    t0 = time.time()
    for n, i in enumerate(targets, 1):
        try:
            sig, rec_fs, _ = wl.read_record(rows[i]["path"])
            clean = np.asarray(filt(sig), dtype=np.float64)
            F_new[i] = qf.atrial_features(clean, rpeaks(clean, rec_fs), rec_fs)
            ok[i] = True
        except Exception as exc:                         # noqa: BLE001
            if n < 5:
                print("  HATA %s: %s" % (rows[i]["record"], exc))
        if n % 200 == 0 or n == len(targets):
            rate = n / max(time.time() - t0, 1e-6)
            print("  %d/%d  (%.0f kayit/s)" % (n, len(targets), rate), flush=True)

    if args.out:
        np.save(args.out, F_new)
        print("\nyazildi: %s  (sekil %s)" % (args.out, F_new.shape))

    sel = [i for i in targets if ok[i] and rows[i]["label_name"] in PAIR]
    if len(sel) < 40:
        raise SystemExit("degerlendirme icin yeterli AFIB/AFL kaydi yok (%d)"
                         % len(sel))
    y = np.array([0 if rows[i]["label_name"] == "AFIB" else 1 for i in sel])
    A = F_old[sel]
    B = F_new[sel].astype(np.float64)

    print("\nAFIB vs AFL ikili ayrim  (%d kayit, 5-fold CV x 3 seed)" % len(sel))
    print("%-26s %10s %9s" % ("ozellik kumesi", "dogruluk", "std"))
    a_mean, a_std = cv_scores(A, y)
    print("%-26s %10.4f %9.4f" % ("mevcut %d" % A.shape[1], a_mean, a_std))
    b_mean, b_std = cv_scores(B, y)
    print("%-26s %10.4f %9.4f" % ("yalniz QRST %d" % B.shape[1], b_mean, b_std))
    c_mean, c_std = cv_scores(np.hstack([A, B]), y)
    print("%-26s %10.4f %9.4f" % ("%d + QRST %d" % (A.shape[1], B.shape[1]),
                                  c_mean, c_std))

    gain = c_mean - a_mean
    print()
    print("QRST ozelliklerinin USTUNE kattigi: %+.4f" % gain)
    print()
    if gain < 0.01:
        print("KARAR: BIRAK.")
        print("  Ozellik uzayinda kazanc yok; agi yeniden egitmek de vermeyecek.")
        print("  Bunu DENEY_KAYDI.md'ye yaz -- 'denedik, ise yaramadi' degerli bir")
        print("  sonuctur ve bu testi kosmak sana gunler kazandirdi.")
    elif gain < 0.03:
        print("KARAR: SINIRDA.")
        print("  Tam egitim ~1 gun surer ve kazanc muhtemelen +-0.013'luk test")
        print("  gurultusunun icinde kalir. Baska denenmemis bir fikrin varsa")
        print("  once onu dene.")
    else:
        print("KARAR: KOS.")
        print("  Bu buyuklukteki bir ikili kazanc, genel macro-F1'de olculebilir")
        print("  bir iyilesmeye karsilik gelir. Sonraki adim:")
        print("    1) python test_qrst_gain.py --all-rows   (tum kayitlar icin F_qrst.npy)")
        print("    2) F.npy ile F_qrst.npy'yi birlestir -> 37+24 = 61 ozellik")
        print("    3) model.py'de ozellik dali girisini 61 yap")
        print("    4) train.py --only_fold 0 ile kapi: OOF >= +0.01 mi")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(B)
        m = LogisticRegression(max_iter=4000).fit(sc.transform(B), y)
        order = np.argsort(-np.abs(m.coef_[0]))[:6]
        print("\nen ayirt edici 6 QRST ozelligi:")
        for i in order:
            print("   %-26s %+.3f" % (qf.FEATURE_NAMES[i], m.coef_[0][i]))
    except Exception:                                    # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
