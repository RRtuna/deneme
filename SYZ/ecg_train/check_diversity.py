"""check_diversity -- iki model ailesi GERCEKTEN farkli hatalar mi yapiyor?

    python check_diversity.py runs/cv10 runs/div_inc
    python check_diversity.py runs/cv10 runs/div_inc runs/div_hyb baseline/r18_feat

FAZ 5'in kapisi budur -- tek basina skor DEGIL. Bir model tek basina daha kotu
olabilir ama ensemble'a katki verebilir; onemli olan **farkli** yanilmasidir.

Neden bu olcum kritik
---------------------
Final ensemble'in bes ailesi de ResNet ve 30->20 budama denemesi bunun bedelini
gosterdi: 10 model cikarildi, 750 tahminin hicbiri degismedi. Yeni bir aile
eklemeden once "bu gercekten farkli mi bakiyor" sorusunu ucuza cevaplamak
gerekir.

Basilan sayilar
---------------
  ayni tahmin orani   Klasik cesitlilik olcusu. < 0.85 ise farkli bakiyorlar.
  KURTARILABILIR      Asil sayi. Bir modelin yanildigi ama digerinin dogru
                      bildigi kayit sayisi. Ensemble'in kazanabilecegi ust
                      sinir budur. 0 ise ekleme yapmanin hicbir anlami yok.
  ikisi de yanlis     Hicbir birlestirme kurtaramaz. Sinyal/etiket tavani.

Tum olcumler **gelistirme kumesinin OOF tahminleri** uzerindedir; test_public
hicbir asamada okunmaz.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np


def load_run(path, n_rows):
    """oof_prob.npy'yi yukle ve HANGI SATIRLARI gercekten kapsadigini dondur.

    Kritik: `--only_fold` ile kosulmus bir egitim `oof_prob.npy` yazmaz. Boyle
    bir dosya varsa eski bir kosudan kalmadir ve karsilastirma sessizce yanlis
    olur. Ayrica kismi bir OOF'ta kapsanmayan satirlar tamamen sifirdir; bunlari
    "tahmin" saymak argmax'i sinif 0'a sabitler ve hem dogruluk hem uyum oranini
    anlamsizlastirir. Bu yuzden kapsanan satirlar ayrica dondurulur.
    """
    oof = os.path.join(path, "oof_prob.npy")
    if not os.path.exists(oof):
        raise SystemExit(
            "%s yok.\n"
            "  `--only_fold` ile kosulan egitim oof_prob.npy YAZMAZ.\n"
            "  Cesitlilik kapisi icin ya tam 5-fold kos, ya da iki kosuyu\n"
            "  fold%%d/val_prob.npy uzerinden ayni fold'da karsilastir." % oof)
    p = np.load(oof)
    if p.shape[0] != n_rows:
        raise SystemExit("%s: %d satir, cache %d satir -- ayni cache mi?"
                         % (oof, p.shape[0], n_rows))
    covered = p.sum(axis=1) > 1e-9
    return p, covered


def load_fold(path, fold):
    """Tek fold'un val_prob / val_idx ciftini yukle."""
    d = os.path.join(path, "fold%d" % fold)
    vp, vi = os.path.join(d, "val_prob.npy"), os.path.join(d, "val_idx.npy")
    for f in (vp, vi):
        if not os.path.exists(f):
            raise SystemExit("%s yok -- bu kosu fold %d'i egitmemis" % (f, fold))
    return np.load(vp), np.load(vi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="kosu klasorleri (en az 2)")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--gate", type=float, default=0.85)
    ap.add_argument("--fold", type=int, default=None,
                    help="tek fold karsilastir: her kosunun fold<k>/val_prob.npy "
                         "dosyasi kullanilir. --only_fold ile egitilmis kosular "
                         "icin TEK dogru yol budur (oof_prob.npy yazilmaz).")
    args = ap.parse_args(argv)

    if len(args.runs) < 2:
        raise SystemExit("en az iki kosu ver")

    index_path = os.path.join(args.cache, "index.csv")
    if not os.path.exists(index_path):
        raise SystemExit("%s yok" % index_path)
    with open(index_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] != "test_public"])
    classes = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")

    preds, probs, names = {}, {}, []

    if args.fold is not None:
        idx0 = None
        for run in args.runs:
            vp, vi = load_fold(run, args.fold)
            if idx0 is None:
                idx0 = vi
            elif not np.array_equal(np.sort(idx0), np.sort(vi)):
                raise SystemExit(
                    "fold %d dogrulama satirlari kosular arasinda FARKLI.\n"
                    "  Ayni --seed ve --folds ile egitilmemisler; bu haliyle\n"
                    "  karsilastirma anlamsizdir." % args.fold)
            order = np.argsort(vi)
            name = os.path.basename(run.rstrip("/\\"))
            names.append(name)
            probs[name] = vp[order]
            preds[name] = vp[order].argmax(1)
        dev = np.sort(idx0)
        truth = y[dev]
        print("fold %d dogrulama kumesi: %d kayit  (test_public okunmadi)"
              % (args.fold, len(dev)))
        return report(names, preds, probs, truth, dev, args.gate)

    cover = np.ones(len(dev), dtype=bool)
    for run in args.runs:
        p, covered = load_run(run, len(rows))
        name = os.path.basename(run.rstrip("/\\"))
        names.append(name)
        probs[name] = p[dev]
        preds[name] = p[dev].argmax(1)
        n_missing = int((~covered[dev]).sum())
        if n_missing:
            print("UYARI: %s geliştirme kumesinin %d satirini KAPSAMIYOR"
                  % (name, n_missing))
            print("       (tum-sifir olasilik satiri -- muhtemelen kismi kosu)")
            cover &= covered[dev]

    if not cover.all():
        kept = int(cover.sum())
        if kept < 200:
            raise SystemExit("ortak kapsanan satir cok az (%d) -- "
                             "karsilastirma anlamsiz" % kept)
        print("karsilastirma yalnizca ORTAK kapsanan %d satirda yapiliyor"
              % kept)
        for n in names:
            preds[n] = preds[n][cover]
            probs[n] = probs[n][cover]
        dev = dev[cover]

    truth = y[dev]
    print("gelistirme kumesi: %d kayit  (test_public okunmadi)" % len(dev))
    return report(names, preds, probs, truth, dev, args.gate)


def macro_f1(y, p, k=5):
    f1 = []
    for c in range(k):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1))


def report(names, preds, probs, truth, dev, gate):
    print()
    print("%-16s %8s %8s %10s" % ("kosu", "dogruluk", "hata", "macro-F1"))
    for n in names:
        acc = float((preds[n] == truth).mean())
        print("%-16s %8.4f %8d %10.4f"
              % (n, acc, int((preds[n] != truth).sum()), macro_f1(truth, preds[n])))

    print()
    print("IKILI KARSILASTIRMA")
    verdicts = []
    for a, b in itertools.combinations(names, 2):
        pa, pb = preds[a], preds[b]
        agree = float((pa == pb).mean())
        ea, eb = pa != truth, pb != truth
        both = int((ea & eb).sum())
        only_a = int((ea & ~eb).sum())
        only_b = int((~ea & eb).sum())
        rescuable = only_a + only_b
        union = int((ea | eb).sum())

        # Ic tutarlilik: birlesim, tek tek hata sayilarindan kucuk olamaz.
        # Kucukse iki dizi ayni satirlara denk gelmiyordur.
        if union < max(int(ea.sum()), int(eb.sum())):
            raise SystemExit(
                "IC TUTARSIZLIK: birlesik hata %d, ama %s tek basina %d hata "
                "yapiyor.\n  Iki olasilik dizisi AYNI satirlara denk gelmiyor "
                "-- karsilastirma gecersiz." % (union, a, int(ea.sum())))

        print()
        print("  %s  vs  %s" % (a, b))
        print("    ayni tahmin orani : %.4f   %s"
              % (agree, "dusuk korelasyon (< %.2f)" % gate if agree < gate
                 else "yuksek korelasyon (>= %.2f)" % gate))
        print("    ikisi de yanlis   : %4d   (hicbir ensemble kurtaramaz)" % both)
        print("    KURTARILABILIR    : %4d   (%s: %d, %s: %d)"
              % (rescuable, a, only_b, b, only_a))
        print("    en az biri yanlis : %4d" % union)

        pair = np.isin(truth, [1, 2])
        if pair.any():
            pea, peb = ea & pair, eb & pair
            pr = int((pea & ~peb).sum() + (~pea & peb).sum())
            print("    AFIB/AFL: %s %d hata, %s %d hata, kurtarilabilir %d"
                  % (a, int(pea.sum()), b, int(peb.sum()), pr))

        # ---- ASIL OLCUM: harmanlamak GERCEKTEN kazandiriyor mu --------------
        # Uyum orani bir VEKIL olcudur; karar bu satirda verilir.
        base = max(macro_f1(truth, pa), macro_f1(truth, pb))
        best_w, best_f1 = 0.0, base
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            f1 = macro_f1(truth, ((1 - w) * probs[a] + w * probs[b]).argmax(1))
            if f1 > best_f1:
                best_w, best_f1 = w, f1
        gain = best_f1 - base
        print("    HARMAN: en iyi w=%.1f -> macro-F1 %.4f  (%+.4f, en iyi tekile gore)"
              % (best_w, best_f1, gain))
        verdicts.append((a, b, agree, rescuable, gain))

    print()
    print("KARAR")
    print("  Kapi uyum orani DEGIL, harman kazancidir. Uyum orani yalnizca")
    print("  bir vekil olcudur; iki model %85 ayni karar verip yine de")
    print("  birlestiginde kazandirabilir.")
    print()
    passed = [v for v in verdicts if v[4] > 0.004]
    if not passed:
        best = max((v[4] for v in verdicts), default=0.0)
        print("  Hicbir cift harmanda kazanc vermedi (en iyi %+.4f)." % best)
        print("  Yeni aile ensemble'a katki vermeyecek -- DENEY_KAYDI.md'ye")
        print("  yaz ve BIRAK. Bu savunulabilir bir sonuc: 'mimari cesitliligi")
        print("  denedik, ayni hatalari yapiyorlar'.")
        print()
        print("  UYARI: tek fold uzerinde olculduyse gurultu payi buyuktur")
        print("  (+-0.015). Kazanc 0.004 civarinda kaldiysa ikinci bir fold")
        print("  kos, tek fold'a dayanarak aile atma.")
        return 1
    for a, b, agree, resc, gain in passed:
        print("  %s + %s: harman %+.4f, %d kurtarilabilir -> TAM 5-FOLD KOS"
              % (a, b, gain, resc))
    print()
    print("  Sonraki adim: python ensemble.py  (agirliklari OOF ile yeniden ara)")
    print("  Kapi: ensemble OOF mevcut degerini geciyor mu?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
