"""v2_analiz -- egitim YAPMADAN, biten fold'larin ciktilarindan macro-F1 kaldiraclarini olcer.

    python v2_analiz.py --cache cache_extonly_v2 --run runs\\ext_v2
    python v2_analiz.py --cache cache_extonly_v2 --run runs\\ext_v2 --boot 2000

Egitim devam ederken de kosulabilir: yalnizca ``fold*/val_idx.npy`` dosyasi
olan fold'lari kullanir. Hicbir dosyaya dokunmaz; tek ciktisi
``<run>/v2_analiz.json``. Birkac dakika surer (en uzun kisim bootstrap ve
yigma). Yalnizca numpy gerekir; scikit-learn varsa yigma (stacking) bolumu da
kosar.

Ne olcer
--------
A  Taban cizgisi: OOF macro-F1, sinif F1, karisiklik matrisi, bootstrap SE.
B  Kaynak kirilimi: AFIB/AFL karismasi hangi hastanede? Yalniz kaynaga bakan
   bir kural AFIB/AFL ciftinde modele ne kadar yaklasiyor?
C  Etiket konvansiyonu sondasi: her (kaynak, sinif) icin RR duzensizligi.
   Gercek AFL cogunlukla sabit iletimle DUZENLI RR verir, AFIB duzensiz.
   Bir kaynagin "AFL"si baska kaynaklarin AFIB'i kadar duzensizse, o kaynakta
   AFL etiketi baska bir seyi (buyuk olasilikla AF) kapsiyor demektir.
D  Karar kurali (egitimsiz): sinif bazli log-bias ve oncul duzeltmesi.
   Egitim --balanced_epoch ile her sinifi esit gordu, degerlendirme verisi
   dengesiz. Parametreler DIGER fold'larin OOF'unda secilir, disarida
   birakilan fold'a uygulanir (ic ice CV) -- yani sayi durust.
E  Yigma: OOF olasiliklari (+ 37 ozellik) uzerine gradyan artirma, yine
   fold'a gore ic ice CV.
F  test_public: D/E'nin tum OOF'la kurulan hali fold-ortalamasi test
   tahminine uygulanir. YALNIZ RAPOR -- hicbir karar buna bakmaz.

Karar kapisi (D ve E icin): Delta >= +0.005 VE eslestirilmis bootstrap
P(Delta > 0) >= 0.95 -> "UYGULA"; aksi halde "gurultu / reddet".
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

CLASSES = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")
K = len(CLASSES)
AFIB, AFL = 1, 2

# ecg_preprocess.FEATURE_NAMES ile ayni sira. Import edilebiliyorsa oradan
# okunur; bu kopya yalnizca betik baska bir klasore tasindiginda devreye girer.
FEATURE_NAMES = (
    "hr_mean", "rr_mean", "rr_std", "rr_cv", "rr_rmssd", "rr_rmssd_norm",
    "rr_pnn50", "rr_pnn20", "rr_median", "rr_iqr", "rr_min", "rr_max",
    "rr_range_norm", "rr_sd1", "rr_sd2", "rr_sd1_sd2", "rr_irregular_frac",
    "rr_shannon", "rr_ac1", "n_beats",
    "p_amp_ii", "p_amp_v1", "p_consistency_ii",
    "flutter_power_ii", "flutter_power_v1", "flutter_power_avf",
    "flutter_peak_freq", "flutter_concentration", "fwave_amp_v1",
    "flutter_autocorr", "atrial_rate_bpm",
    "qrs_duration", "qrs_amp_v1", "qrs_amp_v6", "qrs_v1_v6_ratio",
    "qrs_notch_ratio", "qrs_area_ii",
)

GAIN_GATE = 0.005
PROB_GATE = 0.95


# --------------------------------------------------------------------------
# kaynak tespiti (tools/data_provenance.py ile ayni kural + ek klasor adlari)
# --------------------------------------------------------------------------

_PREFIXES = (("HR", "PTB-XL"), ("JS", "JS"), ("A", "CPSC"), ("Q", "CPSC-Extra"),
             ("I", "INCART"), ("S", "PTB"), ("E", "Georgia"))
_PREFIX_RE = {p: re.compile(r"^%s\d+$" % p, re.I) for p, _ in _PREFIXES}
_NUM = re.compile(r"(\d+)\s*$")
JS_SPLIT = 10646            # JS00001..JS10646 Chapman-Shaoxing, sonrasi Ningbo

# Uzun/ozgul anahtarlar once: "cpsc-2018-extra" "cpsc"den once denenmeli.
_PATH_KEYS = (
    ("cpsc-2018-extra", "CPSC-Extra"), ("cpsc_2018_extra", "CPSC-Extra"),
    ("cpsc2018extra", "CPSC-Extra"), ("mimic", "MIMIC-IV-ECG"),
    ("ptb-xl", "PTB-XL"), ("ptb_xl", "PTB-XL"), ("ptbxl", "PTB-XL"),
    ("ningbo", "Ningbo"), ("chapman", "Chapman-Shaoxing"),
    ("shaoxing", "Chapman-Shaoxing"), ("georgia", "Georgia"),
    ("cpsc", "CPSC"), ("incart", "INCART"), ("st_petersburg", "INCART"),
)


def _by_name(name):
    r = str(name).strip()
    for pre, src in _PREFIXES:
        if _PREFIX_RE[pre].match(r):
            if pre == "JS":
                return ("Chapman-Shaoxing" if int(_NUM.search(r).group(1)) <= JS_SPLIT
                        else "Ningbo")
            return src
    return None


# Iki cache semasi var (bkz. tools/add_external.py): eski/sentetik
# record+path, gercek yarisma record_id+header_path. Yarisma kaydinda
# record_id "NORM_000777" ama dosya "JS36591.hea" -- kaynak dosya adindan cikar.
_ID_COLS = ("record_id", "record", "id")
_PATH_COLS = ("header_path", "path", "relative_path", "hea", "header")


def source_of_row(row):
    if row.get("source"):
        return row["source"]
    paths = [row[k] for k in _PATH_COLS if row.get(k)]
    names = [row[k] for k in _ID_COLS if row.get(k)]
    names += [os.path.splitext(re.split(r"[\\/]+", p)[-1])[0] for p in paths]
    for n in names:
        s = _by_name(n)
        if s:
            return s
    for p in paths:
        parts = [q.lower() for q in re.split(r"[\\/]+", p)]
        for key, src in _PATH_KEYS:
            if any(key in q for q in parts):
                return src
    return "?"


# --------------------------------------------------------------------------
# veri
# --------------------------------------------------------------------------

def load_cache(cache_dir):
    with open(os.path.join(cache_dir, "index.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.load(os.path.join(cache_dir, "y.npy")).astype(np.int64)
    if len(rows) != len(y):
        raise SystemExit("index.csv %d satir, y.npy %d -- cache tutarsiz"
                         % (len(rows), len(y)))
    f_path = os.path.join(cache_dir, "F.npy")
    Fe = np.load(f_path).astype(np.float64) if os.path.exists(f_path) else None
    split = np.array([r.get("split", "") for r in rows])
    ok = np.array([int(r.get("ok") or 1) for r in rows], dtype=bool)
    src = np.array([source_of_row(r) for r in rows])
    return y, Fe, split, ok, src


def load_run(run_dir, n_rows):
    """Biten fold'lari topla. Dondurur: oof (n,K), fold_of (n,) -1=yok, test listesi."""
    oof = np.full((n_rows, K), np.nan)
    fold_of = np.full(n_rows, -1, dtype=np.int64)
    tests, folds = [], []
    for d in sorted(glob.glob(os.path.join(run_dir, "fold*"))):
        m = re.search(r"fold(\d+)$", d)
        vi, vp = os.path.join(d, "val_idx.npy"), os.path.join(d, "val_prob.npy")
        if not m or not (os.path.exists(vi) and os.path.exists(vp)):
            continue
        k = int(m.group(1))
        idx = np.load(vi).astype(np.int64)
        prob = np.load(vp).astype(np.float64)
        if prob.shape != (len(idx), K):
            raise SystemExit("%s: val_prob sekli %s, beklenen (%d, %d)"
                             % (d, prob.shape, len(idx), K))
        if np.any(fold_of[idx] >= 0):
            raise SystemExit("%s: val_idx baska bir fold'la cakisiyor" % d)
        oof[idx] = prob
        fold_of[idx] = k
        tp = os.path.join(d, "test_prob.npy")
        if os.path.exists(tp):
            tests.append(np.load(tp).astype(np.float64))
        folds.append(k)
    if not folds:
        raise SystemExit("%s altinda biten fold yok (fold*/val_idx.npy)" % run_dir)
    return oof, fold_of, tests, folds


# --------------------------------------------------------------------------
# metrikler
# --------------------------------------------------------------------------

def confusion(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def f1_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return f1


def macro_f1(y, pred):
    f1 = f1_from_cm(confusion(y, pred))
    return float(f1.mean()), f1


def pair_acc(y, prob):
    m = np.isin(y, [AFIB, AFL])
    if not m.any():
        return float("nan")
    p = np.where(prob[m, AFIB] >= prob[m, AFL], AFIB, AFL)
    return float(np.mean(p == y[m]))


def boot_macro_f1(y, pred_a, pred_b=None, n_boot=1000, seed=0):
    """Tek sistem icin SE; iki sistem icin eslestirilmis Delta dagilimi."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        fa = f1_from_cm(confusion(y[s], pred_a[s])).mean()
        if pred_b is None:
            out[b] = fa
        else:
            out[b] = f1_from_cm(confusion(y[s], pred_b[s])).mean() - fa
    return out


# --------------------------------------------------------------------------
# D: karar kurallari (ic ice CV)
# --------------------------------------------------------------------------

BIAS_GRID = np.arange(-3.0, 3.0001, 0.05)


def fit_bias(logp, y, grid=BIAS_GRID, passes=3):
    """Macro-F1'i buyuten sinif bazli log-bias (Normal sabit 0), koordinat aramasi."""
    b = np.zeros(K)
    best = macro_f1(y, (logp + b).argmax(1))[0]
    for _ in range(passes):
        improved = False
        for c in range(1, K):
            for v in grid:
                trial = b.copy()
                trial[c] = v
                s = macro_f1(y, (logp + trial).argmax(1))[0]
                if s > best + 1e-12:
                    best, b, improved = s, trial, True
        if not improved:
            break
    return b


def fit_prior_tau(logp, y, log_prior, grid=np.arange(0.0, 1.5001, 0.1)):
    scores = [macro_f1(y, (logp + t * log_prior).argmax(1))[0] for t in grid]
    return float(grid[int(np.argmax(scores))])


def nested(logp, y, fold_of, fit, apply):
    """Her fold icin DIGER fold'larda fit, o fold'a apply. Tahmin dizisi dondurur."""
    pred = np.empty(len(y), dtype=np.int64)
    params = {}
    for k in np.unique(fold_of):
        tr, te = fold_of != k, fold_of == k
        p = fit(logp[tr], y[tr])
        pred[te] = apply(logp[te], p)
        params[int(k)] = p
    return pred, params


# --------------------------------------------------------------------------
# E: yigma
# --------------------------------------------------------------------------

def stack_nested(Xs, y, fold_of, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    prob = np.zeros((len(y), K))
    for k in np.unique(fold_of):
        tr, te = fold_of != k, fold_of == k
        counts = np.bincount(y[tr], minlength=K).astype(np.float64)
        w = (len(y[tr]) / (K * np.maximum(counts, 1)))[y[tr]]
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=seed)
        clf.fit(Xs[tr], y[tr], sample_weight=w)
        prob[te] = clf.predict_proba(Xs[te])
    return prob


def stack_full(Xs, y, Xt, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    counts = np.bincount(y, minlength=K).astype(np.float64)
    w = (len(y) / (K * np.maximum(counts, 1)))[y]
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=seed)
    clf.fit(Xs, y, sample_weight=w)
    return clf.predict_proba(Xt)


# --------------------------------------------------------------------------
# yazdirma yardimcilari
# --------------------------------------------------------------------------

def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def print_cm(cm, title):
    print(title)
    print("  %-14s" % "gercek\\tahmin" + "".join("%8s" % c for c in CLASSES) + "   toplam")
    for i, c in enumerate(CLASSES):
        print("  %-14s" % c + "".join("%8d" % v for v in cm[i]) + "%9d" % cm[i].sum())


def verdict(delta, p_pos):
    if delta >= GAIN_GATE and p_pos >= PROB_GATE:
        return "UYGULA"
    if delta > 0:
        return "sinirda / gurultu"
    return "reddet"


# --------------------------------------------------------------------------
# ana akis
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap tekrari (varsayilan 1000)")
    ap.add_argument("--no-stack", action="store_true",
                    help="E bolumunu (yigma) atla")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0 = time.time()

    feat_names = FEATURE_NAMES
    try:
        sys.path.insert(0, os.getcwd())
        import ecg_preprocess as _ep       # noqa: E402
        feat_names = tuple(_ep.FEATURE_NAMES)
    except Exception:
        pass

    y, Fe, split, ok, src = load_cache(args.cache)
    oof, fold_of, tests, folds = load_run(args.run, len(y))
    dev = np.flatnonzero(fold_of >= 0)
    test_idx = np.flatnonzero((split == "test_public") & ok & (y >= 0))
    yd, Pd, fd = y[dev], oof[dev], fold_of[dev]
    logp = np.log(np.clip(Pd, 1e-7, 1.0))
    report = {"cache": args.cache, "run": args.run, "folds": folds,
              "n_oof": int(len(dev)), "classes": list(CLASSES)}

    # ------------------------------------------------------------------ A
    hdr("A  TABAN CIZGISI  (fold %s, %d OOF kaydi)" % (folds, len(dev)))
    base_pred = Pd.argmax(1)
    base_f1, base_cls = macro_f1(yd, base_pred)
    se = boot_macro_f1(yd, base_pred, n_boot=args.boot, seed=args.seed).std()
    print("OOF macro-F1 : %.4f  (bootstrap SE %.4f)" % (base_f1, se))
    print("sinif F1     : " + "  ".join("%s=%.4f" % (c, v) for c, v in zip(CLASSES, base_cls)))
    print("AFIB/AFL cift: %.4f" % pair_acc(yd, Pd))
    cm = confusion(yd, base_pred)
    print_cm(cm, "\nOOF karisiklik matrisi:")
    off = cm.sum() - np.trace(cm)
    cross = cm[AFIB, AFL] + cm[AFL, AFIB]
    print("\ntoplam hata %d, AFIB<->AFL %d (%%%.1f)" % (off, cross, 100.0 * cross / max(off, 1)))
    oracle = cm.copy()
    oracle[AFIB, AFIB] += oracle[AFIB, AFL]; oracle[AFIB, AFL] = 0
    oracle[AFL, AFL] += oracle[AFL, AFIB]; oracle[AFL, AFIB] = 0
    print("tavan (AFIB<->AFL hatalari sifir olsa): %.4f" % f1_from_cm(oracle).mean())
    pair_m = np.isin(yd, [AFIB, AFL]) & np.isin(base_pred, [AFIB, AFL])
    wrong = pair_m & (base_pred != yd)
    conf = Pd.max(1)
    print("cift hatalarinin guveni: medyan %.3f, %%%.1f'i < 0.6"
          % (np.median(conf[wrong]) if wrong.any() else float("nan"),
             100.0 * np.mean(conf[wrong] < 0.6) if wrong.any() else float("nan")))
    report["A"] = {"oof_macro_f1": base_f1, "se": float(se),
                   "per_class": base_cls.tolist(), "afib_afl": pair_acc(yd, Pd),
                   "cm": cm.tolist(), "afib_afl_share_of_errors": float(cross / max(off, 1)),
                   "oracle_no_pair_errors": float(f1_from_cm(oracle).mean())}

    # ------------------------------------------------------------------ B
    hdr("B  KAYNAK KIRILIMI")
    all_rows = np.flatnonzero(ok & (y >= 0) & np.isin(split, ("train", "validation", "test_public")))
    sources = sorted(set(src[all_rows]), key=lambda s: -np.sum(src[all_rows] == s))
    print("sinif x kaynak (dev+test, tum kayitlar):")
    print("  %-18s" % "kaynak" + "".join("%8s" % c for c in CLASSES) + "   toplam")
    tab = {}
    for s in sources:
        cnt = np.bincount(y[all_rows][src[all_rows] == s], minlength=K)
        tab[s] = cnt.tolist()
        print("  %-18s" % s + "".join("%8d" % v for v in cnt) + "%9d" % cnt.sum())
    if "?" in tab:
        print("  UYARI: kaynagi cozulemeyen kayit var ('?'). index.csv'de 'path' "
              "veya kayit adi beklenen duzende degil.")

    print("\nOOF, kaynak bazinda AFIB/AFL davranisi:")
    print("  %-18s %6s %8s %8s   %6s %8s %8s   %6s" % (
        "kaynak", "nAFIB", "AFIB ok", "AFIB>AFL", "nAFL", "AFL ok", "AFL>AFIB", "acc"))
    by_src = {}
    for s in sources:
        m = src[dev] == s
        if not m.any():
            continue
        ya, pa = yd[m], base_pred[m]
        ra, rl = ya == AFIB, ya == AFL
        row = {
            "n": int(m.sum()), "acc": float(np.mean(ya == pa)),
            "n_afib": int(ra.sum()), "n_afl": int(rl.sum()),
            "afib_recall": float(np.mean(pa[ra] == AFIB)) if ra.any() else None,
            "afib_to_afl": float(np.mean(pa[ra] == AFL)) if ra.any() else None,
            "afl_recall": float(np.mean(pa[rl] == AFL)) if rl.any() else None,
            "afl_to_afib": float(np.mean(pa[rl] == AFIB)) if rl.any() else None,
        }
        by_src[s] = row
        fmt = lambda v: "%8s" % "-" if v is None else "%8.3f" % v   # noqa: E731
        print("  %-18s %6d %s %s   %6d %s %s   %6.3f" % (
            s, row["n_afib"], fmt(row["afib_recall"]), fmt(row["afib_to_afl"]),
            row["n_afl"], fmt(row["afl_recall"]), fmt(row["afl_to_afib"]), row["acc"]))

    # Yalnizca kaynaga bakan kural: her kaynak icin DIGER fold'larda cogunluk
    # (AFIB mi AFL mi) -> o fold'un AFIB/AFL kayitlarina uygula.
    pm = np.isin(yd, [AFIB, AFL])
    rule = np.empty(pm.sum(), dtype=np.int64)
    ys, ss, fs = yd[pm], src[dev][pm], fd[pm]
    for k in np.unique(fs):
        tr, te = fs != k, fs == k
        for s in np.unique(ss[te]):
            mt = tr & (ss == s)
            n_afib, n_afl = np.sum(ys[mt] == AFIB), np.sum(ys[mt] == AFL)
            rule[te & (ss == s)] = AFIB if n_afib >= n_afl else AFL
    src_rule = float(np.mean(rule == ys))
    model_pair = pair_acc(yd, Pd)
    print("\nAFIB/AFL cift dogrulugu: model %.4f | YALNIZ KAYNAK kurali %.4f"
          % (model_pair, src_rule))
    if src_rule >= model_pair - 0.03:
        print("  -> Model ciftte kaynak kuralindan pek iyi degil: AFIB/AFL ayrimi "
              "buyuk olcude HASTANE ayrimi. Bkz. C.")
    report["B"] = {"class_by_source": tab, "oof_by_source": by_src,
                   "pair_model": model_pair, "pair_source_rule": src_rule}

    # ------------------------------------------------------------------ C
    hdr("C  ETIKET KONVANSIYONU SONDASI  (37 ozellikten, egitimsiz)")
    if Fe is None or Fe.shape[1] < len(feat_names):
        print("F.npy yok ya da beklenenden dar -- atlandi.")
    else:
        fi = {n: i for i, n in enumerate(feat_names)}
        cols = [c for c in ("rr_cv", "rr_irregular_frac", "rr_pnn50",
                            "atrial_rate_bpm", "flutter_concentration") if c in fi]
        # "Duzenli RR" esigi veriden: Normal kayitlarin rr_cv 90. yuzdeligi.
        nm = all_rows[y[all_rows] == 0]
        thr = float(np.nanpercentile(Fe[nm, fi["rr_cv"]], 90)) if len(nm) else 0.08
        print("duzenli RR esigi: rr_cv <= %.4f  (Normal kayitlarin %%90'i bunun altinda)" % thr)
        print("\n  %-18s %-6s %6s %9s" % ("kaynak", "sinif", "n", "duzenli%")
              + "".join(" %11s" % c[:11] for c in cols))
        probe = {}
        for cls in (AFIB, AFL, 0):
            for s in sources:
                m = all_rows[(y[all_rows] == cls) & (src[all_rows] == s)]
                if len(m) < 20:
                    continue
                reg = float(np.mean(Fe[m, fi["rr_cv"]] <= thr))
                med = [float(np.nanmedian(Fe[m, fi[c]])) for c in cols]
                probe["%s|%s" % (s, CLASSES[cls])] = {"n": int(len(m)), "regular": reg,
                                                      "median": dict(zip(cols, med))}
                print("  %-18s %-6s %6d %8.1f%%" % (s, CLASSES[cls], len(m), 100 * reg)
                      + "".join(" %11.3f" % v for v in med))
            print()
        print("Okuma: AFL satirlarinda 'duzenli%' yuksek, AFIB satirlarinda dusuk "
              "olmali.\nBir kaynagin AFL'si, diger kaynaklarin AFIB'i kadar "
              "duzensizse o kaynagin AFL\netiketi AF'yi de kapsiyor olabilir -- "
              "bu durumda cift hatalarinin bir kismi\nmodelin degil ETIKETIN "
              "sorunudur ve egitimle kapanmaz.")
        report["C"] = {"regular_threshold_rr_cv": thr, "probe": probe}

    # ------------------------------------------------------------------ D
    hdr("D  KARAR KURALI  (ic ice CV: parametre diger fold'larda secilir)")
    results = {}

    def record(name, pred, extra=None):
        f1, cls = macro_f1(yd, pred)
        d = boot_macro_f1(yd, base_pred, pred, n_boot=args.boot, seed=args.seed + 1)
        delta = f1 - base_f1
        p_pos = float(np.mean(d > 0))
        v = verdict(delta, p_pos)
        results[name] = {"macro_f1": f1, "delta": delta,
                         "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         "p_delta_pos": p_pos, "per_class": cls.tolist(),
                         "afib_afl_f1": [float(cls[AFIB]), float(cls[AFL])],
                         "verdict": v, **(extra or {})}
        print("  %-28s %.4f  Delta %+.4f  %%95 [%+.4f, %+.4f]  P(D>0)=%.3f  -> %s"
              % (name, f1, delta, results[name]["ci95"][0], results[name]["ci95"][1], p_pos, v))
        print("  %-28s AFIB %.4f  AFL %.4f  LBBB %.4f  RBBB %.4f"
              % ("", cls[AFIB], cls[AFL], cls[3], cls[4]))

    print("  %-28s %.4f" % ("argmax (taban)", base_f1))
    if len(np.unique(fd)) < 2:
        print("  (tek fold var -- ic ice CV icin en az 2 fold gerekli, D/E atlandi)")
    else:
        # D1: log-oncul duzeltmesi, tek parametre tau
        def _fit_tau(lp, yy):
            pri = np.bincount(yy, minlength=K) / len(yy)
            return (fit_prior_tau(lp, yy, np.log(np.maximum(pri, 1e-6))),
                    np.log(np.maximum(pri, 1e-6)))
        pred, par = nested(logp, yd, fd, _fit_tau, lambda lp, p: (lp + p[0] * p[1]).argmax(1))
        record("D1 oncul duzeltme (tau)", pred,
               {"tau_by_fold": {k: v[0] for k, v in par.items()}})

        # D2: sinif bazli bias (4 serbest parametre)
        pred, par = nested(logp, yd, fd, fit_bias, lambda lp, b: (lp + b).argmax(1))
        record("D2 sinif bazli log-bias", pred,
               {"bias_by_fold": {k: v.tolist() for k, v in par.items()}})
        bias_all = fit_bias(logp, yd)
        print("  tum OOF'la secilen bias: " +
              "  ".join("%s=%+.2f" % (c, v) for c, v in zip(CLASSES, bias_all)))
        results["D2 sinif bazli log-bias"]["bias_all"] = bias_all.tolist()
        if np.any(np.abs(bias_all) >= BIAS_GRID[-1] - 1e-9):
            print("  UYARI: bir bias izgaranin kenarinda -- o sinifin olasiliklari "
                  "ciddi bicimde kaymis; D2 sayisini temkinli oku.")

        # ------------------------------------------------------------ E
        stack_prob = {}
        if not args.no_stack:
            hdr("E  YIGMA  (HistGradientBoosting, ic ice CV)")
            try:
                import sklearn  # noqa: F401
            except ImportError:
                print("scikit-learn yok -- E atlandi (pip install scikit-learn)")
            else:
                Xp = logp
                stack_prob["E1"] = stack_nested(Xp, yd, fd, args.seed)
                record("E1 yigma: yalniz olasilik", stack_prob["E1"].argmax(1))
                if Fe is not None:
                    Xpf = np.hstack([logp, np.nan_to_num(Fe[dev], nan=0.0)])
                    stack_prob["E2"] = stack_nested(Xpf, yd, fd, args.seed)
                    record("E2 yigma: olasilik+37 ozellik", stack_prob["E2"].argmax(1))

        # ------------------------------------------------------------ F
        hdr("F  test_public  (YALNIZ RAPOR -- secim icin KULLANMA)")
        if tests and len(test_idx) == len(tests[0]):
            Pt = np.mean(tests, axis=0)
            yt = y[test_idx]
            lpt = np.log(np.clip(Pt, 1e-7, 1.0))
            t_base = macro_f1(yt, Pt.argmax(1))[0]
            print("  fold ortalamasi (%d fold) argmax : %.4f" % (len(tests), t_base))
            t_bias = macro_f1(yt, (lpt + bias_all).argmax(1))[0]
            print("  + D2 bias (tum OOF'la)          : %.4f  (%+.4f)" % (t_bias, t_bias - t_base))
            rep_f = {"ensemble_argmax": t_base, "ensemble_bias": t_bias}
            if "E2" in stack_prob and Fe is not None:
                Xt = np.hstack([lpt, np.nan_to_num(Fe[test_idx], nan=0.0)])
                Xs = np.hstack([logp, np.nan_to_num(Fe[dev], nan=0.0)])
                t_st = macro_f1(yt, stack_full(Xs, yd, Xt, args.seed).argmax(1))[0]
                print("  + E2 yigma (tum OOF'la)         : %.4f  (%+.4f)" % (t_st, t_st - t_base))
                rep_f["ensemble_stack_e2"] = t_st
            report["F"] = rep_f
        else:
            print("  test_prob bulunamadi ya da boyu test_public ile uyusmuyor -- atlandi.")

    report["D_E"] = results

    # ------------------------------------------------------------------ ozet
    hdr("OZET")
    print("taban OOF macro-F1 %.4f (SE %.4f), %d fold" % (base_f1, se, len(folds)))
    for name, r in results.items():
        print("  %-32s %+.4f  -> %s" % (name, r["delta"], r["verdict"]))
    print("\nKapi: Delta >= +%.3f ve P(Delta>0) >= %.2f. test_public'e bakilarak "
          "karar VERILMEZ." % (GAIN_GATE, PROB_GATE))

    out = os.path.join(args.run, "v2_analiz.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("\nyazildi: %s  (%.0f sn)" % (out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
