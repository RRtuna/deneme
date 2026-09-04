"""resid_features -- QRST-iptalli atriyal bant olcumleri + iletim orani.

    import resid_features as rf
    v = rf.extract(sig, fs)          # (25,) -- sig: (12, T) on islenmis sinyal

TEK KAYNAK. Ayni hesap uc yerde kullanilir:
  * `resid_probe.py`            gercek veride egitimsiz dogrulama
  * `tools/make_resid_features.py`  cache'in F.npy'sini genisletir
  * paketin `predict.py`'si     cikarimda ayni sayilari uretir

Egitimde ve cikarimda farkli iki implementasyon olsaydi, aradaki en kucuk fark
sessizce skoru dusururdu -- GOREV.md'nin on isleme icin koydugu "tek kaynak"
kurali burada da gecerli.

`ecg_preprocess.py`'ye DOKUNMAZ: girdisi zaten on islenmis sinyaldir.

Ne olculuyor
------------
1. QRST iptali. R tepelerine hizali medyan sablon, vurus basina en kucuk
   karelerle olceklenip cikarilir. Geriye atriyal artik kalir.
2. Bant sinirlama 2.5-12 Hz. Gezinme ve yuksek frekans gurultu atilir.
   Mevcut 37 ozellikteki flutter olcumleri HAM sinyal uzerinde hesaplaniyor;
   75 bpm'de QRS harmonikleri (5.0 / 6.25 Hz) tam flutter bandina dustugu icin
   o olcumler buyuk olcude QRS'i olcer.
3. Derivasyon basina 5 sayi (II, III, aVF, V1) -> 20.
4. Iletim orani: RR araliklari ortak bir atriyal dongunun tam katlari mi.
   maliyet(T) = ortalama |RR/T - yuvarla(RR/T)|; rastgele RR icin beklenen
   deger her T'de 0.25, tam katlarda 0'a iner. -> 5 sayi.

Toplam 25.
"""

from __future__ import annotations

import numpy as np

try:                                    # paket icinde ecg_preprocess olmayabilir
    import ecg_preprocess as ep
except Exception:                       # pragma: no cover
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


# --------------------------------------------------------------------------
# iletim orani: RR araliklari ortak bir atriyal dongunun tam katlari mi
# --------------------------------------------------------------------------

T_GRID = np.arange(0.150, 0.320, 0.002)     # atriyal dongu: 188-400 vuru/dk
COND_NAMES = ("cond_cost", "cond_T", "cond_int_frac", "cond_ratio_ent",
              "cond_n_ratios")


def conduction_features(peaks, fs):
    """5 olcum. AFL: dusuk maliyet, yuksek tam-kat orani. AFIB: tersi."""
    if peaks.size < 5:
        return [0.25, 0.0, 0.0, 0.0, 0.0]
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / fs
    rr = rr[(rr > 0.20) & (rr < 2.0)]
    if rr.size < 4:
        return [0.25, 0.0, 0.0, 0.0, 0.0]
    k = rr[None, :] / T_GRID[:, None]
    d = np.abs(k - np.round(k))             # olcek-bagimsiz: rastgele -> 0.25
    cost = d.mean(1)
    j = int(cost.argmin())
    T = float(T_GRID[j])
    ratios = np.clip(np.round(rr / T), 1, 8)
    u, c = np.unique(ratios, return_counts=True)
    p = c / c.sum()
    ent = float(-(p * np.log(p + _EPS)).sum() / np.log(max(len(p), 2)))
    return [float(cost[j]), T, float((d[j] < 0.05).mean()), ent, float(len(u))]


def record_features(sig, fs, lead_idx):
    """20 bant-artik olcumu + 5 iletim olcumu = 25 sayi."""
    pk = rpeaks(sig, fs)
    out = []
    for i in lead_idx:
        res = cancel_qrst(sig[i], pk, fs)
        out.extend(stats(bandlimit(res, fs), fs))
    out.extend(conduction_features(pk, fs))
    return out


FEATURE_NAMES = tuple(
    ["resid_%s_%s" % (lead.lower(), stat)
     for lead in RESID_LEADS for stat in STAT_NAMES] + list(COND_NAMES))
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 25, N_FEATURES


def extract(sig, fs):
    """(12, T) on islenmis sinyal -> (25,) float64."""
    return np.asarray(record_features(np.asarray(sig, dtype=np.float64), fs,
                                      lead_indices()), dtype=np.float64)
