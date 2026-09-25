"""test_v2_analiz -- v2_analiz.py'yi sahte bir cache + kosu klasoruyle sina.

    python tools/test_v2_analiz.py

Gercek veri gerekmez. Gecici klasorde ext_v2 duzenini taklit eder:
gercek yarisma semasi (record_id + header_path), 6 kaynak, Ningbo'da AFIB yok,
3 bitmis fold + 2 bitmemis fold. Sinanan davranislar:

  1  kaynaklar kayit adindan ve klasor adindan dogru cozuluyor
  2  yalniz biten fold'lar okunuyor, OOF boyu dogru
  3  A'daki macro-F1 train.py'nin macro_f1'iyle birebir ayni
  4  C sondasi: Ningbo "AFL"sinin duzensiz oldugu (ekilen etiket kaymasi) gorunuyor
  5  D2: modele EKILEN sinif yanliligi ic ice CV ile bulunuyor (Delta > 0)
  6  D2: ic ice CV'nin kazanc tahmini, bagimsiz taze veride olculen kazancla
     uyusuyor (secim sizintisi yok); ayni veride sec+olc ise iyimser
  7  JSON yaziliyor, test_public bolumu calisiyor
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import v2_analiz as va  # noqa: E402

K = 5
FI = {n: i for i, n in enumerate(va.FEATURE_NAMES)}

# kaynak -> (kayit adi ureteci, klasor, sinif sayilari Normal/AFIB/AFL/LBBB/RBBB)
SOURCES = {
    "Chapman-Shaoxing": (lambda i: "JS%05d" % (1 + i), "chapman-shaoxing", (500, 300, 80, 10, 100)),
    "Ningbo":           (lambda i: "JS%05d" % (20000 + i), "ningbo", (600, 0, 700, 10, 100)),
    "PTB-XL":           (lambda i: "HR%05d" % (1 + i), "ptb-xl", (700, 250, 10, 40, 100)),
    "Georgia":          (lambda i: "E%05d" % (1 + i), "georgia", (300, 100, 30, 20, 60)),
    "CPSC":             (lambda i: "A%04d" % (1 + i), "cpsc-2018", (200, 200, 0, 40, 150)),
    # adi taninmayan kayit: kaynak klasor adindan cozulmeli
    "CPSC-Extra":       (lambda i: "X%04d" % (1 + i), "cpsc-2018-extra", (100, 30, 10, 10, 30)),
}


def build_fixture(root, biased):
    rng = np.random.default_rng(7)
    rows, ys, feats, srcs = [], [], [], []
    for s, (namef, folder, counts) in SOURCES.items():
        j = 0
        for c, n in enumerate(counts):
            for _ in range(n):
                name = namef(j); j += 1
                rows.append({"record_id": name,
                             "header_path": "D:\\ext\\%s\\g1\\%s.hea" % (folder, name)})
                ys.append(c); srcs.append(s)
                f = rng.normal(0, 1, len(va.FEATURE_NAMES))
                # RR duzensizligi: AFIB duzensiz, gercek AFL duzenli,
                # Ningbo "AFL"si (ekilen konvansiyon kaymasi) AFIB gibi duzensiz.
                irregular = c == 1 or (c == 2 and s == "Ningbo")
                f[FI["rr_cv"]] = abs(rng.normal(0.25 if irregular else 0.03, 0.04))
                feats.append(f)
    y = np.array(ys); F = np.array(feats); src = np.array(srcs)
    n = len(y)
    split = np.where(rng.random(n) < 0.15, "test_public",
                     np.where(rng.random(n) < 0.8, "train", "validation"))

    cache = os.path.join(root, "cache"); os.makedirs(cache)
    with open(os.path.join(cache, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "record_id", "header_path", "class_id", "label", "split", "ok"])
        for i, r in enumerate(rows):
            w.writerow([i, r["record_id"], r["header_path"], y[i],
                        va.CLASSES[y[i]], split[i], 1])
    np.save(os.path.join(cache, "y.npy"), y)
    np.save(os.path.join(cache, "F.npy"), F.astype(np.float32))

    # "model": dogru sinifa kayik, AFIB/AFL arasinda karisik logitler
    logits = rng.normal(0, 1.0, (n, K))
    logits[np.arange(n), y] += 2.2
    pair = np.isin(y, [1, 2])
    logits[pair, 1] += rng.normal(0, 1.2, pair.sum())
    logits[pair, 2] += rng.normal(0, 1.2, pair.sum())
    if biased:
        logits[:, 3] += 1.6        # LBBB'ye ekilmis yanlilik: cok fazla LBBB tahmini
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)

    dev = np.flatnonzero(split != "test_public")
    test = np.flatnonzero(split == "test_public")
    fold_of = np.arange(len(dev)) % 5
    rng.shuffle(fold_of)
    run = os.path.join(root, "runs", "ext_v2"); os.makedirs(run)
    for k in range(5):
        d = os.path.join(run, "fold%d" % k); os.makedirs(d)
        if k >= 3:                  # bitmemis fold: yalniz klasor var
            continue
        idx = dev[fold_of == k]
        np.save(os.path.join(d, "val_idx.npy"), idx)
        np.save(os.path.join(d, "val_prob.npy"), prob[idx])
        np.save(os.path.join(d, "test_prob.npy"), prob[test])
    return cache, run, y, src, prob, dev[fold_of < 3]


def _draw(rng, n):
    pri = np.array([0.49, 0.14, 0.24, 0.023, 0.107])
    y = rng.choice(K, n, p=pri / pri.sum())
    logits = rng.normal(0, 1.0, (n, K))
    logits[np.arange(n), y] += 2.2
    pair = np.isin(y, [1, 2])
    logits[pair, 1] += rng.normal(0, 1.2, pair.sum())
    logits[pair, 2] += rng.normal(0, 1.2, pair.sum())
    return y, logits - np.log(np.exp(logits).sum(1, keepdims=True))


def honesty_check():
    rng = np.random.default_rng(11)
    y, lp = _draw(rng, 6000)
    fold_of = rng.permutation(np.arange(len(y)) % 3)
    base = va.macro_f1(y, lp.argmax(1))[0]
    pred, _ = va.nested(lp, y, fold_of, va.fit_bias, lambda l, b: (l + b).argmax(1))
    est = va.macro_f1(y, pred)[0] - base
    b_all = va.fit_bias(lp, y)
    insample = va.macro_f1(y, (lp + b_all).argmax(1))[0] - base
    yf, lpf = _draw(rng, 60000)
    fresh = va.macro_f1(yf, (lpf + b_all).argmax(1))[0] - va.macro_f1(yf, lpf.argmax(1))[0]
    return est, fresh, insample


def run_analysis(cache, run):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        va.main(["--cache", cache, "--run", run, "--boot", "300"])
    with open(os.path.join(run, "v2_analiz.json")) as fh:
        return json.load(fh), buf.getvalue()


def check(name, cond, detail=""):
    print("%-4s %s %s" % ("OK" if cond else "FAIL", name, detail))
    return bool(cond)


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        cache, run, y, src, prob, used = build_fixture(os.path.join(tmp, "b"), biased=True)
        _y, _F, _split, _ok, got_src = va.load_cache(cache)
        ok &= check("1 kaynak cozumu", np.array_equal(got_src, src),
                    "(%d/%d dogru)" % (np.sum(got_src == src), len(src)))

        rep, text = run_analysis(cache, run)
        ok &= check("2 yalniz biten fold'lar", rep["folds"] == [0, 1, 2]
                    and rep["n_oof"] == len(used), "(n_oof=%d)" % rep["n_oof"])

        tp = y[used]; pp = prob[used].argmax(1)
        ref = []
        for c in range(K):
            t = np.sum((pp == c) & (tp == c)); f = np.sum((pp == c) & (tp != c))
            m = np.sum((pp != c) & (tp == c)); d = 2 * t + f + m
            ref.append(2.0 * t / d if d else 0.0)
        ok &= check("3 macro-F1 train.py ile ayni",
                    abs(rep["A"]["oof_macro_f1"] - float(np.mean(ref))) < 1e-12,
                    "(%.6f)" % rep["A"]["oof_macro_f1"])

        pr = rep["C"]["probe"]
        nb, ch = pr["Ningbo|AFL"]["regular"], pr["Chapman-Shaoxing|AFL"]["regular"]
        ok &= check("4 C sondasi Ningbo AFL'yi duzensiz buluyor", nb < 0.2 and ch > 0.8,
                    "(Ningbo AFL duzenli %.2f, Chapman AFL %.2f)" % (nb, ch))

        d2 = rep["D_E"]["D2 sinif bazli log-bias"]
        ok &= check("5 ekilen yanlilik bulunuyor", d2["delta"] > 0.01
                    and d2["bias_all"][3] < -0.5,
                    "(Delta %+.4f, LBBB bias %+.2f)" % (d2["delta"], d2["bias_all"][3]))
        ok &= check("7 JSON + test_public bolumu", "F" in rep
                    and "ensemble_argmax" in rep["F"])

        # 8: v2_karsilastir -- tek-fold tarama kosusu (yalniz fold0) tabanla
        # eslestiriliyor; daha iyi model Delta > 0, ayni model Delta == 0.
        import v2_karsilastir as vk
        idx = np.load(os.path.join(run, "fold0", "val_idx.npy"))
        rng = np.random.default_rng(3)
        better = np.log(prob[idx] + 1e-9)
        better[np.arange(len(idx)), y[idx]] += rng.normal(1.0, 0.5, len(idx))
        better = np.exp(better - better.max(1, keepdims=True))
        better /= better.sum(1, keepdims=True)
        for tag, p in (("iyi", better), ("ayni", prob[idx])):
            d = os.path.join(tmp, "runs2", tag, "fold0"); os.makedirs(d)
            np.save(os.path.join(d, "val_idx.npy"), idx)
            np.save(os.path.join(d, "val_prob.npy"), p)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            vk.main(["--cache", cache, "--boot", "200", run,
                     os.path.join(tmp, "runs2", "iyi"), os.path.join(tmp, "runs2", "ayni")])
        out = buf.getvalue()
        lines = [l for l in out.splitlines() if "Delta" in l and "->" in l]
        ok &= check("8 v2_karsilastir tek-fold eslestirme",
                    ("ortak OOF: %d kayit, fold [0]" % len(idx)) in out
                    and "Delta +0.0000" in lines[1] and "UYGULA" in lines[0],
                    "(%d ortak kayit)" % len(idx))

        # 6: ic ice CV'nin kazanc tahmini durust mu? Ayni ureticten BAGIMSIZ
        # taze bir kume uret; tum OOF'la secilen bias'in orada getirdigi kazanc,
        # ic ice CV'nin tahminiyle uyusmali. Ic ice olmayan (ayni veride sec +
        # olc) tahmin ise iyimser cikmali -- testin ayirt ediciligi bu.
        est, fresh, insample = honesty_check()
        ok &= check("6 ic ice CV kazanci taze veriyle uyumlu",
                    abs(est - fresh) < 0.015 and insample >= est,
                    "(ic ice %+.4f, taze %+.4f, ayni-veri %+.4f)" % (est, fresh, insample))

    print("\n%s" % ("TUM TESTLER GECTI" if ok else "BASARISIZ TEST VAR"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
