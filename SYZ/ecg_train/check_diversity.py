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
    oof = os.path.join(path, "oof_prob.npy")
    if not os.path.exists(oof):
        raise SystemExit("%s yok -- bu kosu tam 5-fold bitmemis olabilir" % oof)
    p = np.load(oof)
    if p.shape[0] != n_rows:
        raise SystemExit("%s: %d satir, cache %d satir -- ayni cache mi?"
                         % (oof, p.shape[0], n_rows))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="kosu klasorleri (en az 2)")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--gate", type=float, default=0.85)
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

    preds, names = {}, []
    for run in args.runs:
        p = load_run(run, len(rows))
        name = os.path.basename(run.rstrip("/\\"))
        names.append(name)
        preds[name] = p[dev].argmax(1)

    truth = y[dev]
    print("gelistirme kumesi: %d kayit  (test_public okunmadi)" % len(dev))
    print()
    print("%-16s %8s %8s" % ("kosu", "dogruluk", "hata"))
    for n in names:
        acc = float((preds[n] == truth).mean())
        print("%-16s %8.4f %8d" % (n, acc, int((preds[n] != truth).sum())))

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

        ok = agree < args.gate
        verdicts.append((a, b, agree, rescuable, ok))
        print()
        print("  %s  vs  %s" % (a, b))
        print("    ayni tahmin orani : %.4f   %s"
              % (agree, "GECTI (< %.2f)" % args.gate if ok
                 else "KALDI (>= %.2f) -- ayni seyi ogreniyorlar" % args.gate))
        print("    ikisi de yanlis   : %4d   (hicbir ensemble kurtaramaz)" % both)
        print("    KURTARILABILIR    : %4d   (%s: %d, %s: %d)"
              % (rescuable, a, only_b, b, only_a))
        if rescuable:
            print("    -> ensemble'in kazanabilecegi ust sinir: %+.4f dogruluk"
                  % (rescuable / len(dev)))

        # AFIB/AFL ikilisinde ozel bakis -- asil darbogaz orada.
        pair = np.isin(truth, [1, 2])
        if pair.any():
            pea, peb = ea & pair, eb & pair
            pr = int((pea & ~peb).sum() + (~pea & peb).sum())
            print("    AFIB/AFL icinde kurtarilabilir: %d / %d hata"
                  % (pr, int((pea | peb).sum())))

    print()
    print("KARAR")
    passed = [v for v in verdicts if v[4] and v[3] > 0]
    if not passed:
        print("  Hicbir cift kapiyi gecmedi. Yeni aile ensemble'a katki")
        print("  vermeyecek -- DENEY_KAYDI.md'ye yaz ve BIRAK.")
        print("  (Bu degerli bir sonuc: 'mimari cesitliligi denedik, ayni")
        print("   hatalari yapiyorlar' cumlesi juride savunulabilir.)")
        return 1
    for a, b, agree, resc, _ in passed:
        print("  %s + %s: korelasyon %.4f, %d kurtarilabilir hata -> TAM 5-FOLD KOS"
              % (a, b, agree, resc))
    print()
    print("  Sonraki adim: python ensemble.py  (agirliklari OOF ile yeniden ara)")
    print("  Kapi: ensemble OOF mevcut degerini geciyor mu?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
