"""balanced_decode -- sinif dagiliminin bilindigi durumda argmax'tan daha iyi karar.

    python balanced_decode.py --oof ensemble_oof_prob.npy --cache cache
    python balanced_decode.py --oof ensemble_oof_prob.npy --apply ensemble_test_prob.npy

Neden
-----
Veri kumesi HER SINIFTAN TAM ESIT sayida kayit iceriyor: gelistirmede 850,
test_public'te 150. Ama model bunu bilmiyor ve tahmin dagilimi dengesiz cikiyor.
Olculen hali (test_public):

    Normal  150 gercek ->  157 tahmin   (+7)
    AFIB    150 gercek ->  177 tahmin  (+27)
    AFL     150 gercek ->  108 tahmin  (-42)
    LBBB    150 gercek ->  154 tahmin   (+4)
    RBBB    150 gercek ->  154 tahmin   (+4)

Model AFL'yi sistematik olarak az tahmin ediyor. Bagimsiz argmax bu kisiti
kullanamaz; her kaydi tek basina karara baglar.

Yontem
------
Sinif basina bir kaydirma (lambda) araniyor:

    tahmin(i) = argmax_c [ log p(i,c) + lambda_c ]

lambda, tahmin sayilari hedef dagilima esitlenecek sekilde ayarlanir. Bu bir
tasima (transport) probleminin dual cozumudur ve hizlidir.

**Kritik nokta:** lambda yalnizca (a) modelin olasiliklarindan ve (b) bilinen
sinif sayilarindan hesaplanir. **Hicbir etikete bakmaz.** Yani klasik anlamda
"veriye uydurma" degildir; bu yuzden OOF'ta olculen kazanc test'e tasinabilir.

Bu, projede daha once denenen "sinif basina esik/bias kalibrasyonu"ndan
farklidir: o, esikleri OOF macro-F1'ini artiracak sekilde ariyordu (etikete
bakiyordu, 5 serbest parametre, OOF +0.003 / test -0.0015). Buradaki kisit
global ve etiketten bagimsiz.

UYARI -- once bunu dogrula
--------------------------
Bu yontem, degerlendirilecek kumenin dengeli oldugu varsayimina dayanir.
test_public dengeli. Ama yarismanin FINAL/GIZLI kumesi dengeli degilse bu
yontem ZARAR verir (asagidaki tabloda -0.05'e kadar). Kural metnini oku:
final kume dengeli mi? Emin degilsen `--soft` ile kismi duzeltme kullan.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def macro_f1(y_true, y_pred, k=5):
    f1 = []
    for c in range(k):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1)), f1


def balanced_offsets(prob, target_counts, iters=2000, step=0.05, tol=0):
    """Sinif sayilarini hedefe esitleyen lambda kaydirmalarini bul.

    Etikete BAKMAZ. Yalnizca olasiliklar ve hedef sayilar kullanilir.
    """
    logp = np.log(np.clip(prob, 1e-12, 1.0))
    n, k = logp.shape
    target = np.asarray(target_counts, dtype=float)
    lam = np.zeros(k)

    best_lam, best_err = lam.copy(), np.inf
    for it in range(iters):
        pred = (logp + lam).argmax(1)
        counts = np.bincount(pred, minlength=k).astype(float)
        err = np.abs(counts - target).sum()
        if err < best_err:
            best_err, best_lam = err, lam.copy()
        if err <= tol:
            break
        # Az tahmin edilen sinifi yukari, cok tahmin edileni asagi it.
        lam = lam + step * (target - counts) / max(n / k, 1.0)
        step *= 0.999
    return best_lam, best_err


def decode(prob, lam):
    return (np.log(np.clip(prob, 1e-12, 1.0)) + lam).argmax(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default="ensemble_oof_prob.npy",
                    help="(n_cache_rows, 5) OOF olasiliklari")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--apply", default="",
                    help="dogrulandiysa bu test olasilik dosyasina da uygula")
    ap.add_argument("--soft", type=float, default=1.0,
                    help="lambda'nin ne kadari uygulansin (1.0 tam, 0.5 yari)")
    ap.add_argument("--out", default="", help="yeni tahminleri buraya yaz (.npy)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.oof):
        raise SystemExit("%s yok" % args.oof)
    with open(os.path.join(args.cache, "index.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] != "test_public"])

    prob = np.load(args.oof).astype(np.float64)
    if prob.shape[0] != len(rows):
        raise SystemExit("oof %d satir, cache %d satir" % (prob.shape[0], len(rows)))
    P = prob[dev]
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    yd = y[dev]
    k = P.shape[1]

    counts_true = np.bincount(yd, minlength=k)
    counts_pred = np.bincount(P.argmax(1), minlength=k)
    classes = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")[:k]

    print("gelistirme kumesi: %d kayit" % len(dev))
    print()
    print("%-8s %8s %10s %8s" % ("sinif", "gercek", "argmax", "fark"))
    for i, c in enumerate(classes):
        print("%-8s %8d %10d %+8d" % (c, counts_true[i], counts_pred[i],
                                      counts_pred[i] - counts_true[i]))

    base, base_per = macro_f1(yd, P.argmax(1), k)
    print()
    print("argmax macro-F1        : %.6f" % base)

    lam, err = balanced_offsets(P, counts_true)
    lam = lam * args.soft
    pred_bal = decode(P, lam)
    bal, bal_per = macro_f1(yd, pred_bal, k)
    counts_bal = np.bincount(pred_bal, minlength=k)

    print("dengeli cozum macro-F1 : %.6f   (%+.6f)" % (bal, bal - base))
    print("  kalan sayi sapmasi   : %d kayit" % int(err))
    print("  lambda               : %s" % np.round(lam, 4).tolist())
    print("  yeni dagilim         : %s" % counts_bal.tolist())
    print()
    print("%-8s %10s %10s" % ("sinif", "argmax F1", "dengeli F1"))
    for i, c in enumerate(classes):
        print("%-8s %10.4f %10.4f" % (c, base_per[i], bal_per[i]))

    pair = np.isin(yd, [1, 2])
    if pair.any():
        a = float((P.argmax(1)[pair] == yd[pair]).mean())
        b = float((pred_bal[pair] == yd[pair]).mean())
        print()
        print("AFIB/AFL kayitlarinda dogruluk: %.4f -> %.4f (%+.4f)" % (a, b, b - a))

    gain = bal - base
    print()
    if gain > 0.01:
        print("KARAR: UYGULA. OOF'ta %+.4f kazanc var ve lambda etikete bakmadan" % gain)
        print("  hesaplandi, yani bu kazanc test'e tasinmali.")
        print("  ONCE SU SORUYA CEVAP VER: yarismanin final kumesi de dengeli mi?")
        print("  Degilse veya emin degilsen --soft 0.5 ile kismi uygula.")
    elif gain > 0.002:
        print("KARAR: SINIRDA (%+.4f). Kazanc kucuk, riski dusuk." % gain)
        print("  Final kume kesin dengeliyse uygula, degilse birak.")
    else:
        print("KARAR: BIRAK (%+.4f). Model zaten dengeli tahmin ediyor," % gain)
        print("  kisit ek bilgi tasimiyor. DENEY_KAYDI.md'ye yaz.")

    if args.apply and gain > 0.002:
        if not os.path.exists(args.apply):
            raise SystemExit("%s yok" % args.apply)
        T = np.load(args.apply).astype(np.float64)
        T = T / np.clip(T.sum(1, keepdims=True), 1e-12, None)
        n_test = T.shape[0]
        # Test kumesindeki hedef: esit dagilim varsayimi.
        target = np.full(k, n_test / k)
        lam_t, err_t = balanced_offsets(T, target)
        lam_t = lam_t * args.soft
        pred_t = decode(T, lam_t)
        print()
        print("test dosyasina uygulandi: %s (%d kayit)" % (args.apply, n_test))
        print("  argmax dagilimi  : %s" % np.bincount(T.argmax(1), minlength=k).tolist())
        print("  dengeli dagilimi : %s" % np.bincount(pred_t, minlength=k).tolist())
        if args.out:
            np.save(args.out, pred_t)
            print("  yazildi: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
