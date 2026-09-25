"""v2_karsilastir -- iki (veya daha fazla) kosuyu AYNI fold'lar uzerinde eslestirilmis karsilastir.

    python v2_karsilastir.py --cache cache_extonly_v2 runs\\ext_v2 runs\\ls01_f0
    python v2_karsilastir.py --cache cache_extonly_v2 runs\\ext_v2 runs\\w64 runs\\ep60

tools/compare_runs.py'den farki:
  * tam oof_prob.npy BEKLEMEZ -- fold*/val_idx.npy + val_prob.npy okur. Yani
    `--only_fold 0` ile kosulmus tek-fold tarama deneyi, taban kosunun ayni
    fold'uyla dogrudan karsilastirilir.
  * etiketi y.npy'den okur (gercek cache'te index.csv'nin `label` sutunu sinif
    ADI, sayi degil).
  * ENSEMBLE kazancini da olcer: iki kosunun ayni kayda verdigi OOF
    olasiliklarinin ortalamasi. Iki model de o kaydi egitimde gormedigi icin
    bu, test_public'e bakmadan olculen durust bir ensemble skorudur.

Karar kapisi (v2_analiz.py ile ayni): Delta >= +0.005 VE eslestirilmis
bootstrap P(Delta > 0) >= 0.95.

ONEMLI: v1 train.py'de fold bolunmesi --seed'e bagli (make_folds(..., args.seed)).
Karsilastirilacak kosular AYNI --seed ile egitilmeli; aksi halde fold'lar
farkli olur. Betik bunu denetler ve uyarir.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2_analiz as va  # noqa: E402
from v2_analiz import AFIB, AFL, CLASSES  # noqa: E402


def sign_test_p(b01, b10):
    from math import comb, erfc, sqrt
    n = b01 + b10
    if n == 0:
        return 1.0
    if n > 1000:
        return float(erfc(abs(b01 - b10) / sqrt(n) / sqrt(2)))
    lo = min(b01, b10)
    return float(min(2.0 * sum(comb(n, k) for k in range(lo + 1)) / 2 ** n, 1.0))


def paired(name, y, pa, pb, n_boot, seed):
    fa, ca = va.macro_f1(y, pa)
    fb, cb = va.macro_f1(y, pb)
    d = va.boot_macro_f1(y, pa, pb, n_boot=n_boot, seed=seed)
    p_pos = float(np.mean(d > 0))
    ok_a, ok_b = pa == y, pb == y
    b01, b10 = int(np.sum(~ok_a & ok_b)), int(np.sum(ok_a & ~ok_b))
    pair = np.isin(y, [AFIB, AFL])
    q01 = int(np.sum(~ok_a[pair] & ok_b[pair]))
    q10 = int(np.sum(ok_a[pair] & ~ok_b[pair]))
    v = va.verdict(fb - fa, p_pos)
    print("  %-30s %.4f -> %.4f  Delta %+.4f  %%95 [%+.4f, %+.4f]  P(D>0)=%.3f  -> %s"
          % (name, fa, fb, fb - fa, np.percentile(d, 2.5), np.percentile(d, 97.5), p_pos, v))
    print("  %-30s duzelen %d / bozulan %d (p=%.4f) | AFIB/AFL icinde %d / %d (p=%.4f)"
          % ("", b01, b10, sign_test_p(b01, b10), q01, q10, sign_test_p(q01, q10)))
    print("  %-30s sinif Delta: %s" % ("", "  ".join(
        "%s %+.4f" % (c, b - a) for c, a, b in zip(CLASSES, ca, cb))))
    return {"base": fa, "new": fb, "delta": fb - fa, "p_delta_pos": p_pos,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "fixed": b01, "broken": b10, "verdict": v,
            "class_delta": [float(b - a) for a, b in zip(ca, cb)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="taban kosu klasoru (orn. runs\\ext_v2)")
    ap.add_argument("others", nargs="+", help="karsilastirilacak kosu klasorleri")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    y_all, _Fe, _split, _ok, _src = va.load_cache(args.cache)
    runs = [args.base] + args.others
    loaded = {r: va.load_run(r, len(y_all)) for r in runs}

    # Ortak kayitlar: her kosunun OOF'unda bulunanlar.
    covered = np.ones(len(y_all), dtype=bool)
    for r in runs:
        covered &= loaded[r][1] >= 0
    common = np.flatnonzero(covered)
    if len(common) == 0:
        raise SystemExit("kosularin ortak OOF kaydi yok -- ayni fold'lar bitmis mi?")

    fb = loaded[args.base][1][common]
    for r in args.others:
        fr = loaded[r][1][common]
        if not np.array_equal(fb, fr):
            print("UYARI: %s ile taban arasinda fold atamasi FARKLI (%d kayit). "
                  "Farkli --seed? Eslestirilmis karsilastirma yine gecerli ama "
                  "egitim kumeleri ayni degil." % (r, int(np.sum(fb != fr))))

    y = y_all[common]
    folds = sorted(set(fb.tolist()))
    print("ortak OOF: %d kayit, fold %s" % (len(common), folds))
    probs = {r: loaded[r][0][common] for r in runs}
    preds = {r: probs[r].argmax(1) for r in runs}

    print("\nTEK MODEL (eslestirilmis, taban = %s)" % args.base)
    for i, r in enumerate(args.others):
        paired(os.path.basename(r.rstrip("\\/")), y, preds[args.base], preds[r],
               args.boot, args.seed + i)

    print("\nENSEMBLE (OOF olasilik ortalamasi -- durust, test_public'e bakmaz)")
    for i, r in enumerate(args.others):
        ens = (probs[args.base] + probs[r]) / 2.0
        paired("taban+%s" % os.path.basename(r.rstrip("\\/")), y,
               preds[args.base], ens.argmax(1), args.boot, args.seed + 100 + i)
    if len(args.others) > 1:
        ens = np.mean([probs[r] for r in runs], axis=0)
        paired("hepsi (%d kosu)" % len(runs), y, preds[args.base], ens.argmax(1),
               args.boot, args.seed + 200)

    print("\nKapi: Delta >= +%.3f ve P(Delta>0) >= %.2f." % (va.GAIN_GATE, va.PROB_GATE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
