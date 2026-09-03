"""qrst_features -- QRST iptali ile atriyal aktiviteyi izole eden ek ozellikler.

    import qrst_features as qf
    extra = qf.atrial_features(filtered_500hz_signal, rpeaks, fs=500.0)   # (24,)

Bu dosya `ecg_preprocess.py`'yi DEGISTIRMEZ. Mevcut 37 ozelligin **yanina**
eklenecek 24 ozellik uretir; boylece cache'in X.npy'si aynen kalir, yalnizca
F.npy genisler.

Neden gerekli
-------------
AFIB ile AFL'yi ayiran sey atriyal dalgadir: AFL'de ~4-6 Hz'lik duzenli bir
testere disi, AFIB'de ayni bantta duzensiz gurultu. Ama bu dalga QRS'ten
10-20 kat kucuktur ve QRS treninin harmonikleri tam o bandin uzerine duser --
75 bpm'de (1.25 Hz) harmonikler 1.25, 2.5, 3.75, **5.0, 6.25** Hz'te. Yani ham
sinyalin 4-6 Hz enerjisini olcen bir ozellik, buyuk olcude QRS'i olcer.

QRST iptali bunu kaynaginda cozer: her vurustan ortalama vurus sablonu
cikarilir, geriye atriyal artik kalir. Bu, klinik sinyal isleme literaturunde
atriyal fibrilasyon analizinin standart on adimidir (average beat
subtraction). Projede denenmemis tek yapisal fikir budur -- diger denenenlerin
hepsi modeli buyutmek/duzenlemekti, bu ise modelin NE GORDUGUNU degistiriyor.

Yontem
------
1. R tepelerine hizali pencereler alinir (-250 ms .. +450 ms).
2. Medyan sablon hesaplanir (ortalama degil: tek bir bozuk vurus sablonu
   kaydirmasin).
3. Her vurusta sablon en kucuk kareler ile olceklenip cikarilir; boylece
   vurustan vurusa genlik degisimi artiga sizmaz.
4. Kalan artik uzerinde spektral ve duzenlilik olculur.

Sablon cikarma islemi QRS bolgesinde artigi neredeyse sifirlar; bu yuzden
olcumler diyastole hapsolmaz, 10 saniyenin tamami kullanilir.
"""

from __future__ import annotations

import numpy as np

# Analiz derivasyonlari: flutter dalgalari en cok burada gorunur.
ANALYSIS_LEADS = {"II": 1, "aVF": 5, "V1": 6}

FEATURE_NAMES = tuple(
    ["qrst_%s_%s" % (lead, stat)
     for lead in ("ii", "avf", "v1")
     for stat in ("flutter_frac", "fib_frac", "dom_freq", "concentration",
                  "spec_entropy", "autocorr", "rms")]
    + ["qrst_dom_freq_spread", "qrst_lead_coherence",
       "qrst_cancel_ratio"]
)
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 24, N_FEATURES

_EPS = 1e-12


def cancel_qrst(lead, peaks, fs, pre=0.25, post=0.45):
    """Bir derivasyondan QRST'yi cikar, atriyal artigi dondur.

    ``lead`` filtrelenmis tek derivasyon (1-B), ``peaks`` R tepe indeksleri.
    Yeterli vurus yoksa sinyal ortalamasi cikarilip aynen dondurulur.
    """
    x = np.asarray(lead, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    n = x.size
    w_pre, w_post = int(round(pre * fs)), int(round(post * fs))

    usable = [p for p in peaks if p - w_pre >= 0 and p + w_post < n]
    if len(usable) < 3:
        return x - x.mean()

    segs = np.stack([x[p - w_pre:p + w_post] for p in usable])
    template = np.median(segs, axis=0)          # medyan: aykiri vurusa dayanikli
    t_energy = float(np.dot(template, template))
    if t_energy < _EPS:
        return x - x.mean()

    res = x.copy()
    for p in usable:
        a, b = p - w_pre, p + w_post
        seg = x[a:b]
        # Sablonu bu vurusun genligine olcekle, sonra cikar.
        k = float(np.dot(seg, template) / t_energy)
        res[a:b] = seg - k * template
    return res - res.mean()


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 32:
        return None, None
    x = x - x.mean()
    if not np.any(np.abs(x) > _EPS):
        return None, None
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return spec, freqs


def _lead_stats(residual, fs):
    """Bir artik sinyalden 7 olcum."""
    out = np.zeros(7, dtype=np.float64)
    spec, freqs = _spectrum(residual, fs)
    if spec is None:
        return out, 0.0

    total = spec[(freqs > 1.0) & (freqs < 20.0)].sum()
    if total <= _EPS:
        return out, 0.0

    flutter = (freqs >= 3.5) & (freqs <= 6.5)      # AFL: ~210-390 /dk
    fib = (freqs > 6.5) & (freqs <= 12.0)          # AFIB: daha yuksek, dagilmis
    out[0] = spec[flutter].sum() / total
    out[1] = spec[fib].sum() / total

    band = (freqs >= 3.0) & (freqs <= 12.0)
    if np.any(band):
        bs, bf = spec[band], freqs[band]
        i = int(np.argmax(bs))
        out[2] = float(bf[i])                       # baskin atriyal frekans
        out[3] = float(bs[i] / (bs.sum() + _EPS))   # dar bantlilik: AFL yuksek
        p = bs / (bs.sum() + _EPS)
        p = p[p > 0]
        # Spektral entropi: AFL tek tepeli (dusuk), AFIB yayvan (yuksek).
        out[4] = float(-np.sum(p * np.log(p)) / np.log(max(p.size, 2)))

    # Otokorelasyon tepesi: atriyal ritim ne kadar duzenli.
    seg = np.asarray(residual, dtype=np.float64)
    seg = seg - seg.mean()
    if seg.size > int(fs / 2) and np.any(np.abs(seg) > _EPS):
        ac = np.correlate(seg, seg, mode="full")[seg.size - 1:]
        if ac[0] > _EPS:
            ac = ac / ac[0]
            lo, hi = int(fs / 12.0), min(int(fs / 3.0), ac.size - 1)
            if hi > lo:
                out[5] = float(np.max(ac[lo:hi]))
    out[6] = float(np.std(residual))                # f/F dalga genligi
    return out, float(out[2])


def atrial_features(sig, peaks, fs=500.0):
    """Filtrelenmis 12 derivasyon + R tepeleri -> 24 atriyal ozellik (float32).

    ``sig`` (12, N) olmali ve **filtrelenmis** 500 Hz sinyal olmalidir --
    yani projedeki `filter_500()` ciktisi.
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peaks = np.asarray(peaks, dtype=int).ravel()

    feats, doms, residuals = [], [], []
    cancel_num, cancel_den = 0.0, 0.0

    for name, idx in ANALYSIS_LEADS.items():
        lead = x[idx] if idx < x.shape[0] else np.zeros(x.shape[-1])
        res = cancel_qrst(lead, peaks, fs)
        stats, dom = _lead_stats(res, fs)
        feats.append(stats)
        doms.append(dom)
        residuals.append(res)
        cancel_num += float(np.var(res))
        cancel_den += float(np.var(lead))

    out = np.concatenate(feats)

    # Baskin frekansin derivasyonlar arasi tutarliligi: organize bir flutter
    # her derivasyonda ayni atriyal hizi gosterir, AFIB gostermez.
    doms = np.asarray([d for d in doms if d > 0], dtype=np.float64)
    spread = float(np.std(doms)) if doms.size >= 2 else 0.0

    # Artiklarin derivasyonlar arasi korelasyonu: organizasyon olcusu.
    coh = 0.0
    if len(residuals) >= 2:
        cs = []
        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                a, b = residuals[i], residuals[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > _EPS and nb > _EPS:
                    cs.append(abs(float(np.dot(a, b) / (na * nb))))
        coh = float(np.mean(cs)) if cs else 0.0

    # Iptal kalitesi: varyansin ne kadari kaldi. Cok yuksekse (QRST
    # cikarilamamissa) diger ozelliklere guven azalir; model bunu ogrenebilsin.
    ratio = float(cancel_num / (cancel_den + _EPS)) if cancel_den > 0 else 1.0

    out = np.concatenate([out, [spread, coh, ratio]])
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert out.size == N_FEATURES, out.size
    return out.astype(np.float32)
