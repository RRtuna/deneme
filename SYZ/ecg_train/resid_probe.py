"""resid_probe -- QRST-iptalli ATRIYAL BANT artigi gercekten yeni bilgi mi?

    python resid_probe.py --cache cache --oof ensemble_oof_prob.npy
    python resid_probe.py --cache cache --oof ensemble_oof_prob.npy --run runs/cv10

EGITIM YAPMAZ. Mevcut cache ve mevcut OOF olasiliklarini kullanir, birkac
dakika surer. Cikardigi tek karar: "AFIB/AFL icin artik kanalina yatirim
yapmaya deger mi?"

`ecg_preprocess.py`'ye DOKUNMAZ, cache'i degistirmez, `test_public`'i okumaz.

Fikir
-----
Mevcut 37 ozellikte flutter olcumleri var (`flutter_power_ii`,
`flutter_peak_freq`, `flutter_concentration`, `flutter_autocorr`,
`fwave_amp_v1`). Hepsi **ham sinyal** uzerinde hesaplaniyor. Sorun: 75 bpm'de
QRS treninin harmonikleri 1.25, 2.5, 3.75, **5.0, 6.25** Hz'e duser -- yani tam
flutter bandinin (4-6 Hz) uzerine. Ham sinyalin o bandindaki enerjiyi olcen bir
sayi, buyuk olcude QRS'i olcer, atriyal aktiviteyi degil.

Bu betik ayni olcumleri once QRS'i cikardiktan sonra yapar:

    1. R tepelerine hizali medyan sablon, vurus basina en kucuk kareler ile
       olceklenip cikarilir  ->  atriyal artik
    2. artik 2.5-12 Hz bandina sinirlanir  ->  gezinme ve yuksek frekans gurultu
       atilir, geriye neredeyse yalnizca atriyal aktivite kalir
    3. 4 derivasyondan (II, III, aVF, V1) 5'er olcum: tepe yogunlugu, spektral
       entropi, baskin frekans, otokorelasyon tepesi, genlik  ->  20 sayi
    4. bu 20 sayiya duz lojistik regresyon (fold-disi) takilir

Sonra tek soru sorulur: bu prob, agin YANILDIGI kayitlarda dogru mu biliyor?

Ne basiliyor
------------
  KURTARILABILIR   Birinin bildigi, digerinin bilmedigi kayit sayisi. Sifirsa
                   prob ayni seyi goruyor demektir, birlestirmenin anlami yok.
  harman kazanci   Agin ikili olasiligini probunkiyle karistirinca ikili
                   dogruluk ve macro-F1 ne oluyor. KARAR bu satirdadir.

Sentetik zor kiyas kumesinde olculen (bkz. tools/make_synth_hard.py):
ag 0.7965 / prob 0.7894 / harman 0.8165, Bayes tavani 0.8408. Yani harman
kalan boslugun %46'sini kapatmisti. Bu betik ayni olcumu GERCEK veride yapar --
sentetikte ise yaramasi gercekte ise yarayacagini GARANTI ETMEZ.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import ecg_preprocess as ep
except Exception:                                   # pragma: no cover
    ep = None

RESID_LEADS = ("II", "III", "aVF", "V1")
BAND = (2.5, 12.0)          # atriyal bant: flutter 4-6 Hz, fib 6-10 Hz
STAT_NAMES = ("conc", "ent", "dom", "ac", "rms")
CLASSES = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")
_EPS = 1e-12


# --------------------------------------------------------------------------
# derivasyon indeksi ve R tepeleri -- iki API yazimini da destekle
# --------------------------------------------------------------------------

def lead_indices():
    default = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
               "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}
    for attr in ("STANDARD_LEADS", "LEADS", "LEAD_NAMES"):
        leads = getattr(ep, attr, None) if ep else None
        if leads and len(leads) == 12:
            up = [str(s).upper() for s in leads]
            try:
                return [up.index(n.upper()) for n in RESID_LEADS]
            except ValueError:
                break
    return [default[n] for n in RESID_LEADS]


def _fallback_rpeaks(x, fs):
    """Basit ama saglam R bulucu -- modulun API'si tutmazsa devreye girer."""
    d = np.diff(x, prepend=x[0])
    e = d * d
    w = max(int(0.12 * fs), 3)
    k = np.ones(w) / w
    s = np.convolve(e, k, mode="same")
    thr = 0.35 * float(np.percentile(s, 99))
    if thr <= _EPS:
        return np.array([], dtype=int)
    above = s > thr
    peaks, i, n = [], 0, s.size
    refractory = int(0.20 * fs)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            peaks.append(i + int(np.argmax(np.abs(x[i:j]))))
            i = j + refractory
        else:
            i += 1
    return np.array(peaks, dtype=int)


_RPEAK_MODE = [None]


def rpeaks(sig, fs):
    if _RPEAK_MODE[0] == "fallback":
        return _fallback_rpeaks(sig[1], fs)
    for fname in ("detect_rpeaks", "detect_r"):
        fn = getattr(ep, fname, None) if ep else None
        if fn is None:
            continue
        for arg in (sig, sig[1]):
            try:
                out = np.asarray(fn(arg, fs), dtype=int).ravel()
            except Exception:
                continue
            if out.size >= 3:
                _RPEAK_MODE[0] = fname
                return out
    _RPEAK_MODE[0] = "fallback"
    return _fallback_rpeaks(sig[1], fs)


# --------------------------------------------------------------------------
# QRST iptali + bant sinirlama
# --------------------------------------------------------------------------

def cancel_qrst(x, peaks, fs, pre=0.25, post=0.45):
    """Medyan sablonu vurus basina olcekleyip cikar; geriye atriyal artik kalir."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    w_pre, w_post = int(round(pre * fs)), int(round(post * fs))
    usable = [int(p) for p in peaks if p - w_pre >= 0 and p + w_post < n]
    if len(usable) < 3:
        return x - x.mean()
    segs = np.stack([x[p - w_pre:p + w_post] for p in usable])
    template = np.median(segs, axis=0)      # medyan: tek bozuk vurus kaydirmasin
    energy = float(template @ template)
    if energy < _EPS:
        return x - x.mean()
    res = x.copy()
    for p in usable:
        a, b = p - w_pre, p + w_post
        seg = x[a:b]
        res[a:b] = seg - (float(seg @ template) / energy) * template
    return res - res.mean()


def bandlimit(x, fs, lo=BAND[0], hi=BAND[1], roll=0.5):
    """FFT ile sifir-fazli bant sinirlama.

    Kendi filtresini kurar: `ecg_preprocess`'in filtre API'sine bagimli degil,
    yani o dosyanin hangi surumu olursa olsun calisir. 10 saniyelik pencerede
    FFT maskesi IIR filtreden hem daha keskin hem de faz kaydirmasiz.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    spec = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    g = np.ones_like(fr)
    g[fr < lo - roll] = 0.0
    g[fr > hi + roll] = 0.0
    lo_edge = (fr >= lo - roll) & (fr < lo)
    hi_edge = (fr > hi) & (fr <= hi + roll)
    g[lo_edge] = 0.5 * (1 - np.cos(np.pi * (fr[lo_edge] - (lo - roll)) / roll))
    g[hi_edge] = 0.5 * (1 + np.cos(np.pi * (fr[hi_edge] - hi) / roll))
    return np.fft.irfft(spec * g, n=n)


def stats(r, fs, lo=BAND[0], hi=BAND[1]):
    """Bir artiktan 5 olcum. Organize flutter -> yuksek conc/ac, dusuk ent."""
    r = np.asarray(r, dtype=np.float64)
    r = r - r.mean()
    if r.size < 32 or not np.any(np.abs(r) > _EPS):
        return [0.0] * 5
    sp = np.abs(np.fft.rfft(r * np.hanning(r.size))) ** 2
    fr = np.fft.rfftfreq(r.size, 1.0 / fs)
    m = (fr > lo) & (fr < hi)
    s, f = sp[m], fr[m]
    total = float(s.sum())
    if total <= _EPS or s.size < 4:
        return [0.0] * 5
    p = s / total
    conc = float(s.max() / total)                       # tepe / toplam
    ent = float(-(p * np.log(p + _EPS)).sum() / np.log(p.size))
    dom = float(f[int(s.argmax())])
    a = np.correlate(r, r, "full")[r.size - 1:]
    a = a / (a[0] + _EPS)
    k0, k1 = max(int(fs / hi), 1), min(int(fs / lo), a.size - 1)
    ac = float(a[k0:k1].max()) if k1 > k0 else 0.0
    return [conc, ent, dom, ac, float(r.std())]


def record_features(sig, fs, lead_idx):
    pk = rpeaks(sig, fs)
    out = []
    for i in lead_idx:
        res = cancel_qrst(sig[i], pk, fs)
        out.extend(stats(bandlimit(res, fs), fs))
    return out


# --------------------------------------------------------------------------
# bagimsiz lojistik regresyon (sklearn surumune bagimli olmamak icin)
# --------------------------------------------------------------------------

def fit_logreg(Xtr, ytr, l2=1.0, iters=800, lr=0.5):
    """L2 cezali ikili lojistik regresyon, tam-toplu gradyan inisi.

    Kendi implementasyonu, cunku sklearn'in surumler arasinda degisen
    imzalari (ornegin `multi_class`) bu betigi kirmamali.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + _EPS
    Z = (Xtr - mu) / sd
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        z = Z @ w + b
        pr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = pr - ytr
        gw = Z.T @ g / n + l2 * w / n
        gb = float(g.mean())
        w -= lr * gw
        b -= lr * gb
    return (w, b, mu, sd)


def predict_logreg(model, Xte):
    w, b, mu, sd = model
    z = ((Xte - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def stratified_folds(y, k, seed=42):
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def macro_f1(y, p, k=5):
    f1 = []
    for c in range(k):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1)), f1


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--oof", default="ensemble_oof_prob.npy",
                    help="(n_cache_rows, 5) mevcut OOF olasiliklari")
    ap.add_argument("--run", default="",
                    help="fold uyeliginin okunacagi kosu klasoru "
                         "(runs/xxx). Verilmezse kendi 5-fold'unu kurar.")
    ap.add_argument("--fs", type=float, default=0.0, help="0 = meta.json'dan")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="resid_features.npy",
                    help="20 ozellik buraya yazilir (tum cache satirlari)")
    args = ap.parse_args(argv)

    x_path = os.path.join(args.cache, "X.npy")
    idx_path = os.path.join(args.cache, "index.csv")
    for p in (x_path, idx_path, args.oof):
        if not os.path.exists(p):
            raise SystemExit("%s yok" % p)

    fs = args.fs
    meta_path = os.path.join(args.cache, "meta.json")
    if not fs and os.path.exists(meta_path):
        fs = float(json.load(open(meta_path)).get("target_fs") or 0.0)
    if not fs:
        raise SystemExit("ornekleme hizi bilinmiyor; --fs ver")

    with open(idx_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] != "test_public"])

    X = np.load(x_path, mmap_mode="r")
    if X.shape[0] != len(rows):
        raise SystemExit("X %d satir, index %d satir" % (X.shape[0], len(rows)))
    prob = np.load(args.oof).astype(np.float64)
    if prob.shape[0] != len(rows):
        raise SystemExit("oof %d satir, cache %d satir -- ayni cache'in OOF'u mu?"
                         % (prob.shape[0], len(rows)))

    lead_idx = lead_indices()
    print("cache %s: %d kayit, %.0f Hz   artik derivasyonlari: %s"
          % (args.cache, len(rows), fs, ", ".join(RESID_LEADS)))
    print("gelistirme %d kayit  (test_public okunmuyor)" % len(dev))
    print("bant %.1f-%.1f Hz" % BAND)
    print()

    t0 = time.time()
    F = np.zeros((len(rows), len(RESID_LEADS) * len(STAT_NAMES)))
    for n, i in enumerate(dev):
        F[i] = record_features(np.asarray(X[i], dtype=np.float64), fs, lead_idx)
        if (n + 1) % 500 == 0 or n + 1 == len(dev):
            print("  ozellik %5d/%d  %.0f sn" % (n + 1, len(dev), time.time() - t0),
                  flush=True)
    print("R tepeleri: %s" % _RPEAK_MODE[0])

    pair = dev[np.isin(y[dev], [1, 2])]
    if len(pair) < 50:
        raise SystemExit("AFIB/AFL kaydi cok az (%d)" % len(pair))
    Fp = F[pair]
    Lp = (y[pair] == 2).astype(float)               # 1 = AFL

    # fold uyeligi: mumkunse agin kendi fold'lari, degilse kendi bolumu
    fold = None
    if args.run:
        fold_full = np.full(len(rows), -1, dtype=int)
        for k in range(20):
            vp = os.path.join(args.run, "fold%d" % k, "val_idx.npy")
            if os.path.exists(vp):
                fold_full[np.load(vp)] = k
        if (fold_full[pair] >= 0).all():
            fold = fold_full[pair]
            print("fold uyeligi %s klasorunden alindi (%d fold)"
                  % (args.run, fold.max() + 1))
    if fold is None:
        fold = stratified_folds(Lp.astype(int), args.folds, args.seed)
        print("fold uyeligi bu betikte kuruldu (%d-fold, seed %d)"
              % (args.folds, args.seed))
        print("  NOT: aginkiyle ayni degil; harman kazanci bir miktar iyimser")
        print("  olabilir. --run runs/<kosu> vererek bunu ortadan kaldir.")

    q = np.zeros(len(pair))                          # fold-disi P(AFL)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        q[te] = predict_logreg(fit_logreg(Fp[tr], Lp[tr]), Fp[te])

    net_pair = prob[pair][:, [1, 2]]
    net_pair = net_pair / np.clip(net_pair.sum(1, keepdims=True), _EPS, None)
    p_net = net_pair[:, 1]                           # agin ikili-ici P(AFL)

    truth = Lp.astype(int)
    ok_net = (p_net > 0.5).astype(int) == truth
    ok_prb = (q > 0.5).astype(int) == truth

    print()
    print("IKILI (AFIB vs AFL), %d kayit" % len(pair))
    print("  ag (mevcut OOF)        : %.4f" % ok_net.mean())
    print("  bant-artik probu       : %.4f" % ok_prb.mean())
    print("  ayni tahmin orani      : %.4f  %s"
          % (float(((p_net > 0.5) == (q > 0.5)).mean()),
             "(< 0.85: farkli bakiyorlar)"))
    print("  ikisi de yanlis        : %4d  (hicbir birlesim kurtaramaz)"
          % int((~ok_net & ~ok_prb).sum()))
    print("  sadece prob dogru      : %4d  <- aga eklenebilecek yeni bilgi"
          % int((~ok_net & ok_prb).sum()))
    print("  sadece ag dogru        : %4d" % int((ok_net & ~ok_prb).sum()))
    print("  KURTARILABILIR         : %4d" % int((ok_net ^ ok_prb).sum()))

    # --- harman: yalnizca ikili KARARI degistir, diger siniflara dokunma ---
    yd, pd_ = y[dev], prob[dev].argmax(1)
    base_f1, _ = macro_f1(yd, pd_)
    pos = {int(g): n for n, g in enumerate(pair)}    # global -> pair sirasi

    print()
    print("%-8s %12s %12s %12s" % ("w(prob)", "ikili dog.", "macro-F1", "fark"))
    best = (0.0, ok_net.mean(), base_f1)
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        mixed = (1 - w) * p_net + w * q
        acc = float(((mixed > 0.5).astype(int) == truth).mean())
        newp = prob[dev].copy()
        for n, g in enumerate(dev):
            j = pos.get(int(g))
            if j is None:
                continue
            s = newp[n, 1] + newp[n, 2]
            newp[n, 1] = s * (1 - mixed[j])
            newp[n, 2] = s * mixed[j]
        f1, _ = macro_f1(yd, newp.argmax(1))
        print("%-8.1f %12.4f %12.4f %+12.4f" % (w, acc, f1, f1 - base_f1))
        if f1 > best[2]:
            best = (w, acc, f1)

    np.save(args.save, F)
    gain = best[2] - base_f1
    print()
    print("en iyi w = %.1f   ikili %.4f   macro-F1 %.4f  (%+.4f)"
          % (best[0], best[1], best[2], gain))
    print("20 ozellik yazildi: %s" % args.save)
    print()
    print("KARAR")
    if gain > 0.02:
        print("  UYGULA. Kazanc buyuk ve kaynagi yeni bilgi (kurtarilabilir > 0).")
        print("  Sonraki adim: bu 20 sayiyi ozellik dalina ekleyip yeniden egit")
        print("  (37 -> 57), ya da w=%.1f harmanini ensemble.py'ye sabitle."
              % best[0])
    elif gain > 0.008:
        print("  SINIRDA (%+.4f). Egitim gerektirmeyen harman ucuz -- OOF'ta" % gain)
        print("  dogrulandiysa uygula, ama tek basina yarismayi cevirmez.")
    else:
        print("  BIRAK (%+.4f). Ham sinyaldeki flutter ozellikleri bu bilgiyi" % gain)
        print("  zaten yakalamis; QRST iptali ek bir sey getirmiyor.")
        print("  DENEY_KAYDI.md'ye yaz -- bu da savunulabilir bir sonuc.")
    print()
    print("UYARI: buradaki w OOF uzerinde secildi. test_public'e yalnizca")
    print("karar verildikten SONRA, tek sefer bakilir (DEGISMEZ KURAL 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
