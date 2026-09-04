"""compare_runs -- iki egitim kosusunu ESLESTIRILMIS olarak karsilastir.

    python tools/compare_runs.py runs/hb_base runs/hb_nopair --cache cache_hard

Neden eslestirilmis
-------------------
Iki kosunun OOF skorlarini yan yana koyup "0.7976 vs 0.8033, demek ki iyi"
demek yanlistir: fark, ayni kayitlar uzerinde olcculdugu icin bagimsiz
degildir. Dogru test McNemar'dir -- yalnizca **fikrini degistiren** kayitlara
bakar:

    b01 = taban yanlis, yeni dogru   (duzelen)
    b10 = taban dogru, yeni yanlis   (bozulan)

Bu ikisi arasindaki dengesizlik anlamliysa fark gercektir. Toplam dogruluk
farkina bakmak, ayni bilgiyi cok daha gurultulu bir sekilde okumaktir.

AFIB/AFL ikilisi ayrica raporlanir, cunku projedeki tek darbogaz orasi ve
genel macro-F1 farki bu ikilideki degisimi seyreltir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from math import comb

import numpy as np


def macro_f1(y, p, k=5):
    f1 = []
    for c in range(k):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1)), f1


def sign_test(b01, b10):
    """Iki tarafli isaret testi (McNemar'in tam hali)."""
    n = b01 + b10
    if n == 0:
        return 1.0
    if n > 1000:                       # normal yaklasim
        from math import erfc, sqrt
        z = abs(b01 - b10) / sqrt(n)
        return float(erfc(z / sqrt(2)))
    lo = min(b01, b10)
    tail = sum(comb(n, k) for k in range(lo + 1))
    return float(min(2.0 * tail / (2 ** n), 1.0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="taban kosu klasoru")
    ap.add_argument("others", nargs="+", help="karsilastirilacak kosular")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--bayes", default="", help="meta_hard.json (tavan icin)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.cache, "index.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] != "test_public"])
    yd = y[dev]
    pair = np.isin(yd, [1, 2])

    def load(run):
        p = np.load(os.path.join(run, "oof_prob.npy"))
        if p.shape[0] != len(rows):
            raise SystemExit("%s satir sayisi uyusmuyor" % run)
        return p[dev].argmax(1)

    bayes = None
    if args.bayes and os.path.exists(args.bayes):
        bayes = json.load(open(args.bayes)).get("bayes_pair_accuracy")

    pb = load(args.base)
    base_name = os.path.basename(args.base.rstrip("/\\"))
    m_b, _ = macro_f1(yd, pb)
    acc_b = float((pb[pair] == yd[pair]).mean())

    print("gelistirme: %d kayit  (%d AFIB/AFL)" % (len(dev), int(pair.sum())))
    if bayes:
        print("BAYES cift tavani: %.4f" % bayes)
    print()
    print("%-14s %10s %12s %12s" % ("kosu", "macro-F1", "AFIB/AFL", "tavana"))
    print("%-14s %10.4f %12.4f %12s"
          % (base_name, m_b, acc_b,
             ("%.4f" % (bayes - acc_b)) if bayes else "-"))

    results = []
    for run in args.others:
        pn = load(run)
        name = os.path.basename(run.rstrip("/\\"))
        m_n, _ = macro_f1(yd, pn)
        acc_n = float((pn[pair] == yd[pair]).mean())
        print("%-14s %10.4f %12.4f %12s"
              % (name, m_n, acc_n,
                 ("%.4f" % (bayes - acc_n)) if bayes else "-"))
        results.append((name, pn, m_n, acc_n))

    print()
    print("ESLESTIRILMIS TEST (McNemar / isaret testi)")
    for name, pn, m_n, acc_n in results:
        print()
        print("  %s  ->  %s" % (base_name, name))
        for label, mask in (("tum kayitlar", np.ones_like(pair)),
                            ("AFIB/AFL", pair)):
            cb = pb[mask] == yd[mask]
            cn = pn[mask] == yd[mask]
            b01 = int((~cb & cn).sum())
            b10 = int((cb & ~cn).sum())
            p = sign_test(b01, b10)
            verdict = ("ANLAMLI" if p < 0.05
                       else "sinirda" if p < 0.15 else "gurultu")
            arrow = "+" if b01 > b10 else ("-" if b10 > b01 else "=")
            print("    %-13s duzelen %3d  bozulan %3d  net %s%-3d  p=%.4f  %s"
                  % (label, b01, b10, arrow, abs(b01 - b10), p, verdict))
        print("    macro-F1 %+.4f   AFIB/AFL %+.4f" % (m_n - m_b, acc_n - acc_b))

    print()
    print("Yorum: net fark kucuk ve p yuksekse, iki kosu ayni modeldir --")
    print("skor farki seed gurultusudur. Karar icin p < 0.05 iste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
