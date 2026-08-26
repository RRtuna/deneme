"""ecg_preprocess -- the one and only preprocessing path.

Training and inference both call into this module, so a change here changes
both at once. That is the point: there is no second filter implementation
anywhere in the project.

Everything is numpy plus the standard library. No scipy, so this file can be
copied verbatim into the delivered ONNX package.

Pipeline for the network input
------------------------------
  raw 12-lead, 500 Hz, 10 s
    -> baseline removal      Butterworth high-pass, 0.5 Hz, order 2
    -> power-line notch      50 Hz and 60 Hz, Q = 30 (optional, on by default)
    -> anti-alias low-pass   Butterworth low-pass, 40 Hz, order 4
    -> resample              band-limited FFT resample to TARGET_FS
    -> normalise             per-record robust scale, see NORM_MODE
    -> fixed length          TARGET_SECONDS, centre-cropped or zero-padded

Pipeline for the 37 hand features
---------------------------------
Features are always computed from the *native* rate signal (500 Hz), never
from the resampled copy, so changing TARGET_FS leaves F.npy untouched and the
resolution experiment stays single-variable.

If you edit this file, regenerate the cache (``python prep.py``) or training
will silently keep using the old arrays.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# configuration -- mirrored into preprocess.json by export.py
# --------------------------------------------------------------------------

PREPROCESS_VERSION = "1.0.0"

NATIVE_FS = 500.0          # rate the raw records are stored at
TARGET_FS = 150.0          # rate the network sees
TARGET_SECONDS = 10.0

HP_CUTOFF = 0.5            # Hz, baseline wander
HP_ORDER = 2
LP_CUTOFF = 40.0           # Hz, anti-alias + muscle noise
LP_ORDER = 4
NOTCH_FREQS = (50.0, 60.0)
NOTCH_Q = 30.0

# "global": every lead divided by one shared robust scale, so the amplitude
#           ratio between V1 and V6 survives -- that ratio is what separates
#           LBBB from RBBB.
# "perlead": each lead scaled on its own. Kept for ablation only.
NORM_MODE = "global"
CLIP_SIGMA = 8.0

CLASSES = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")
N_CLASSES = len(CLASSES)

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6")
N_LEADS = len(STANDARD_LEADS)
_LEAD_IDX = {name: i for i, name in enumerate(STANDARD_LEADS)}

TARGET_LEN = int(round(TARGET_FS * TARGET_SECONDS))

_EPS = 1e-8


def preprocess_config():
    """The exact settings used, for the manifest and for preprocess.json."""
    return {
        "version": PREPROCESS_VERSION,
        "native_fs": NATIVE_FS,
        "target_fs": TARGET_FS,
        "target_seconds": TARGET_SECONDS,
        "target_len": TARGET_LEN,
        "hp_cutoff": HP_CUTOFF, "hp_order": HP_ORDER,
        "lp_cutoff": LP_CUTOFF, "lp_order": LP_ORDER,
        "notch_freqs": list(NOTCH_FREQS), "notch_q": NOTCH_Q,
        "norm_mode": NORM_MODE, "clip_sigma": CLIP_SIGMA,
        "leads": list(STANDARD_LEADS),
        "classes": list(CLASSES),
        "n_features": 37,
        "feature_names": None,      # filled in below, after FEATURE_NAMES exists
    }


# --------------------------------------------------------------------------
# IIR filtering, biquad cascades built by hand
# --------------------------------------------------------------------------
#
# A Butterworth section of even order N is a cascade of N/2 biquads whose only
# difference is the Q factor:  Q_k = 1 / (2 cos((2k+1)pi / 2N)).
# Each biquad uses the standard bilinear-transform (RBJ) coefficients, which
# already fold in frequency prewarping. The cascade therefore reproduces
# scipy.signal.butter exactly, but stays numerically well behaved because no
# high-order polynomial is ever formed -- important at 0.5 Hz / 500 Hz, where a
# single 8th-order transfer function would lose precision near z = 1.

def _butter_qs(order):
    if order % 2 or order < 2:
        raise ValueError("only even Butterworth orders >= 2 are supported, "
                         "got %r" % order)
    return [1.0 / (2.0 * np.cos((2 * k + 1) * np.pi / (2.0 * order)))
            for k in range(order // 2)]


def butter_lowpass_sos(cutoff, fs, order=4):
    """Second-order sections for a Butterworth low-pass."""
    sos = []
    w0 = 2.0 * np.pi * cutoff / fs
    cw, sw = np.cos(w0), np.sin(w0)
    for q in _butter_qs(order):
        alpha = sw / (2.0 * q)
        a0 = 1.0 + alpha
        sos.append([(1.0 - cw) / 2.0 / a0, (1.0 - cw) / a0, (1.0 - cw) / 2.0 / a0,
                    1.0, -2.0 * cw / a0, (1.0 - alpha) / a0])
    return np.asarray(sos, dtype=np.float64)


def butter_highpass_sos(cutoff, fs, order=2):
    """Second-order sections for a Butterworth high-pass."""
    sos = []
    w0 = 2.0 * np.pi * cutoff / fs
    cw, sw = np.cos(w0), np.sin(w0)
    for q in _butter_qs(order):
        alpha = sw / (2.0 * q)
        a0 = 1.0 + alpha
        sos.append([(1.0 + cw) / 2.0 / a0, -(1.0 + cw) / a0, (1.0 + cw) / 2.0 / a0,
                    1.0, -2.0 * cw / a0, (1.0 - alpha) / a0])
    return np.asarray(sos, dtype=np.float64)


def notch_sos(freq, fs, q=30.0):
    """Single biquad band-stop at ``freq``."""
    w0 = 2.0 * np.pi * freq / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    a0 = 1.0 + alpha
    return np.asarray([[1.0 / a0, -2.0 * cw / a0, 1.0 / a0,
                        1.0, -2.0 * cw / a0, (1.0 - alpha) / a0]],
                      dtype=np.float64)


def _sosfilt_reference(sos, x):
    """Forward direct-form-II transposed cascade along the last axis.

    This is the definition of the filter. It is pure numpy and has no
    dependencies, which is what lets the package ship without scipy.
    """
    y = np.asarray(x, dtype=np.float64)
    lead_shape = y.shape[:-1]
    for b0, b1, b2, _, a1, a2 in sos:
        z1 = np.zeros(lead_shape, dtype=np.float64)
        z2 = np.zeros(lead_shape, dtype=np.float64)
        out = np.empty_like(y)
        for n in range(y.shape[-1]):
            xn = y[..., n]
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            out[..., n] = yn
        y = out
    return y


# The recursion above is a Python loop over samples, which costs roughly a
# tenth of a second per record -- fine for inference, painful when building a
# 5000-record cache. When scipy is installed (training machines, never the
# delivered package) its sosfilt evaluates the *same* difference equation with
# the same zero initial conditions, so it is used purely as an accelerator.
# tools/test_preprocess.py asserts the two agree to 1e-9 on real-shaped input;
# set ECG_FORCE_NUMPY_FILTER=1 to pin the reference path.
try:                                            # pragma: no cover
    import os as _os

    if _os.environ.get("ECG_FORCE_NUMPY_FILTER"):
        raise ImportError
    from scipy.signal import sosfilt as _scipy_sosfilt

    FILTER_BACKEND = "scipy"
except Exception:                               # pragma: no cover
    _scipy_sosfilt = None
    FILTER_BACKEND = "numpy"


def _sosfilt(sos, x):
    if _scipy_sosfilt is not None:
        return _scipy_sosfilt(np.asarray(sos, dtype=np.float64),
                              np.asarray(x, dtype=np.float64), axis=-1)
    return _sosfilt_reference(sos, x)


def sosfiltfilt(sos, x, pad=None):
    """Zero-phase filtering: forward, reversed, forward again.

    Odd-reflection padding on both ends keeps the transient out of the signal,
    which matters because a 10 s ECG is short relative to a 0.5 Hz high-pass.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if pad is None:
        pad = min(3 * 6 * len(sos) + 200, max(n - 1, 0))
    if pad <= 0:
        return _sosfilt(sos, _sosfilt(sos, x)[..., ::-1])[..., ::-1]

    left = 2.0 * x[..., :1] - x[..., 1:pad + 1][..., ::-1]
    right = 2.0 * x[..., -1:] - x[..., -pad - 1:-1][..., ::-1]
    ext = np.concatenate([left, x, right], axis=-1)

    ext = _sosfilt(sos, ext)
    ext = _sosfilt(sos, ext[..., ::-1])[..., ::-1]
    return ext[..., pad:pad + n]


def bandpass(x, low, high, fs, order_hp=2, order_lp=4):
    """Zero-phase band-pass, used by the R-peak detector and the features."""
    y = sosfiltfilt(butter_highpass_sos(low, fs, order_hp), x)
    return sosfiltfilt(butter_lowpass_sos(high, fs, order_lp), y)


# --------------------------------------------------------------------------
# resampling and normalisation
# --------------------------------------------------------------------------

def resample_fft(x, n_out):
    """Band-limited resample along the last axis via the real FFT."""
    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[-1]
    if n_in == n_out:
        return x.copy()
    spec = np.fft.rfft(x, axis=-1)
    out_bins = n_out // 2 + 1
    keep = min(spec.shape[-1], out_bins)
    new_spec = np.zeros(x.shape[:-1] + (out_bins,), dtype=complex)
    new_spec[..., :keep] = spec[..., :keep]
    if n_out < n_in and n_out % 2 == 0:
        # Nyquist bin of the shortened spectrum must stay real.
        new_spec[..., -1] = new_spec[..., -1].real
    return np.fft.irfft(new_spec, n=n_out, axis=-1) * (float(n_out) / n_in)


def fix_length(x, n_out):
    """Centre-crop or zero-pad the last axis to exactly ``n_out``."""
    n = x.shape[-1]
    if n == n_out:
        return x
    if n > n_out:
        start = (n - n_out) // 2
        return x[..., start:start + n_out]
    pad = n_out - n
    left = pad // 2
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, pad - left)])


def normalise(x, mode=None, clip=None):
    """Robust per-record scaling. See NORM_MODE for why 'global' is default."""
    mode = NORM_MODE if mode is None else mode
    clip = CLIP_SIGMA if clip is None else clip
    x = np.asarray(x, dtype=np.float64)

    centred = x - np.median(x, axis=-1, keepdims=True)
    if mode == "perlead":
        scale = 1.4826 * np.median(np.abs(centred), axis=-1, keepdims=True)
        flat = np.std(centred, axis=-1, keepdims=True)
    else:
        scale = 1.4826 * np.median(np.abs(centred))
        flat = np.std(centred)
    # MAD collapses on flat-line leads; fall back to std, then to unity.
    scale = np.where(np.asarray(scale) < 1e-6, flat, scale)
    scale = np.where(np.asarray(scale) < 1e-6, 1.0, scale)

    return np.clip(centred / scale, -clip, clip)


# --------------------------------------------------------------------------
# the network input
# --------------------------------------------------------------------------

def preprocess_signal(sig, fs_in=NATIVE_FS, target_fs=None,
                      target_seconds=TARGET_SECONDS, notch=True):
    """Raw 12-lead millivolts -> float32 array shaped (12, target_len).

    ``sig`` is (n_leads, n_samples). ``target_fs`` defaults to TARGET_FS; the
    resolution experiment (FAZ 3) overrides it without touching anything else.
    """
    target_fs = TARGET_FS if target_fs is None else float(target_fs)
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] < 8:
        raise ValueError("signal too short to filter: %d samples" % x.shape[-1])

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = sosfiltfilt(butter_highpass_sos(HP_CUTOFF, fs_in, HP_ORDER), x)
    if notch:
        for f0 in NOTCH_FREQS:
            if f0 < fs_in / 2.0:
                x = sosfiltfilt(notch_sos(f0, fs_in, NOTCH_Q), x)
    x = sosfiltfilt(butter_lowpass_sos(min(LP_CUTOFF, 0.45 * fs_in),
                                       fs_in, LP_ORDER), x)

    n_out = int(round(x.shape[-1] * target_fs / fs_in))
    x = resample_fft(x, max(n_out, 8))
    x = fix_length(x, int(round(target_fs * target_seconds)))
    x = normalise(x)
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------
# R-peak detection (Pan-Tompkins in spirit)
# --------------------------------------------------------------------------

def _moving_average(x, w):
    if w < 2:
        return x.astype(np.float64)
    c = np.cumsum(np.insert(np.asarray(x, dtype=np.float64), 0, 0.0))
    out = (c[w:] - c[:-w]) / float(w)
    pad = len(x) - len(out)
    return np.concatenate([np.full(pad, out[0] if len(out) else 0.0), out])


def detect_rpeaks(sig, fs=NATIVE_FS):
    """Return R-peak sample indices, detected on a lead-II-like channel."""
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]

    # Prefer II; fall back to whichever lead has the strongest QRS band energy.
    if x.shape[0] >= N_LEADS:
        candidates = [_LEAD_IDX["II"], _LEAD_IDX["V1"], _LEAD_IDX["V5"],
                      _LEAD_IDX["I"]]
    else:
        candidates = list(range(x.shape[0]))

    best, best_lead = None, None
    for idx in candidates:
        if idx >= x.shape[0]:
            continue
        band = bandpass(x[idx], 5.0, min(15.0, 0.45 * fs), fs)
        energy = float(np.std(band))
        if best is None or energy > best:
            best, best_lead = energy, band
    if best_lead is None or best < 1e-9:
        return np.array([], dtype=np.int64)

    band = best_lead
    deriv = np.diff(band, prepend=band[0])
    integ = _moving_average(deriv ** 2, max(int(round(0.150 * fs)), 3))
    if not np.any(integ > 0):
        return np.array([], dtype=np.int64)

    # Adaptive threshold: a fraction of a slow running maximum, so the detector
    # survives amplitude drift across the 10 s strip.
    win = max(int(round(2.0 * fs)), 3)
    running = _moving_average(integ, win)
    thresh = np.maximum(0.35 * np.percentile(integ, 98), 1.5 * running)

    above = integ > thresh
    if not np.any(above):
        return np.array([], dtype=np.int64)

    edges = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    refractory = int(round(0.20 * fs))
    search = int(round(0.05 * fs))
    peaks = []
    for s, e in zip(starts, ends):
        if e - s < max(int(round(0.02 * fs)), 2):
            continue
        centre = s + int(np.argmax(integ[s:e]))
        lo = max(centre - search - int(round(0.08 * fs)), 0)
        hi = min(centre + search, len(band))
        if hi <= lo:
            continue
        peak = lo + int(np.argmax(np.abs(band[lo:hi])))
        if peaks and peak - peaks[-1] < refractory:
            if np.abs(band[peak]) > np.abs(band[peaks[-1]]):
                peaks[-1] = peak
            continue
        peaks.append(peak)

    return np.asarray(peaks, dtype=np.int64)


# --------------------------------------------------------------------------
# the 37 hand features
# --------------------------------------------------------------------------

FEATURE_NAMES = (
    # rhythm / heart-rate variability (20)
    "hr_mean", "rr_mean", "rr_std", "rr_cv", "rr_rmssd", "rr_rmssd_norm",
    "rr_pnn50", "rr_pnn20", "rr_median", "rr_iqr", "rr_min", "rr_max",
    "rr_range_norm", "rr_sd1", "rr_sd2", "rr_sd1_sd2", "rr_irregular_frac",
    "rr_shannon", "rr_ac1", "n_beats",
    # atrial activity: P waves vs fibrillatory vs flutter waves (11)
    "p_amp_ii", "p_amp_v1", "p_consistency_ii",
    "flutter_power_ii", "flutter_power_v1", "flutter_power_avf",
    "flutter_peak_freq", "flutter_concentration", "fwave_amp_v1",
    "flutter_autocorr", "atrial_rate_bpm",
    # QRS morphology: bundle branch blocks (6)
    "qrs_duration", "qrs_amp_v1", "qrs_amp_v6", "qrs_v1_v6_ratio",
    "qrs_notch_ratio", "qrs_area_ii",
)
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 37, N_FEATURES


def _safe(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def _band_power(seg, fs, lo, hi):
    """(fraction of power in [lo, hi], peak frequency, peak/band ratio)."""
    seg = np.asarray(seg, dtype=np.float64)
    if seg.size < 8 or np.allclose(seg, seg[0]):
        return 0.0, 0.0, 0.0
    seg = seg - seg.mean()
    window = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * window)) ** 2
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / fs)
    total = spec[1:].sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    band = spec[mask]
    frac = band.sum() / total
    peak_i = int(np.argmax(band))
    peak_f = float(freqs[mask][peak_i])
    conc = float(band[peak_i] / (band.sum() + _EPS))
    return float(frac), peak_f, conc


def _rr_features(rpeaks, fs, n_samples):
    """Twenty rhythm descriptors from the R-peak train."""
    out = np.zeros(20, dtype=np.float64)
    if rpeaks.size < 3:
        # Too few beats to describe a rhythm; leave zeros but record the count.
        out[19] = float(rpeaks.size)
        return out

    rr = np.diff(rpeaks) / float(fs)
    rr = rr[(rr > 0.20) & (rr < 3.0)]
    if rr.size < 2:
        out[19] = float(rpeaks.size)
        return out

    mean = rr.mean()
    median = np.median(rr)
    diffs = np.diff(rr)

    out[0] = 60.0 / mean
    out[1] = mean
    out[2] = rr.std()
    out[3] = rr.std() / (mean + _EPS)
    out[4] = np.sqrt(np.mean(diffs ** 2)) if diffs.size else 0.0
    out[5] = out[4] / (mean + _EPS)
    out[6] = np.mean(np.abs(diffs) > 0.050) if diffs.size else 0.0
    out[7] = np.mean(np.abs(diffs) > 0.020) if diffs.size else 0.0
    out[8] = median
    out[9] = np.percentile(rr, 75) - np.percentile(rr, 25)
    out[10] = rr.min()
    out[11] = rr.max()
    out[12] = (rr.max() - rr.min()) / (median + _EPS)
    # Poincare descriptors: SD1 is beat-to-beat scatter, SD2 long-term scatter.
    out[13] = np.sqrt(0.5) * (diffs.std() if diffs.size else 0.0)
    out[14] = np.sqrt(max(2.0 * rr.var() - 0.5 * (diffs.var() if diffs.size else 0.0), 0.0))
    out[15] = out[13] / (out[14] + _EPS)
    out[16] = np.mean(np.abs(rr - median) > 0.10 * median)

    hist, _ = np.histogram(rr, bins=16, range=(max(rr.min() - 1e-3, 0.0),
                                               rr.max() + 1e-3))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    out[17] = -np.sum(p * np.log(p)) / np.log(16.0) if p.size else 0.0

    if rr.size >= 3:
        centred = rr - mean
        denom = np.sum(centred ** 2)
        out[18] = np.sum(centred[:-1] * centred[1:]) / (denom + _EPS)
    out[19] = float(rpeaks.size)
    return out


def _atrial_features(sig, rpeaks, fs):
    """Eleven descriptors of atrial activity.

    Normal sinus rhythm shows a discrete P wave just before each QRS. AFIB
    replaces it with low-amplitude broadband noise. AFL replaces it with a
    narrow-band sawtooth near 4-6 Hz that is best seen in II, III, aVF and V1 --
    which is exactly why those leads are read here.
    """
    out = np.zeros(11, dtype=np.float64)
    n = sig.shape[-1]
    if rpeaks.size < 2:
        return out

    def lead(name):
        idx = _LEAD_IDX.get(name)
        return sig[idx] if idx is not None and idx < sig.shape[0] else None

    ii, v1, avf = lead("II"), lead("V1"), lead("aVF")
    if ii is None:
        ii = sig[0]

    # ---- P-wave window: 200 ms to 70 ms before each R ----
    p_lo, p_hi = int(round(0.20 * fs)), int(round(0.07 * fs))
    p_segments_ii = []
    p_amp_ii, p_amp_v1, used = 0.0, 0.0, 0
    for r in rpeaks:
        a, b = r - p_lo, r - p_hi
        if a < 0 or b > n or b <= a:
            continue
        seg = ii[a:b]
        p_segments_ii.append(seg - seg.mean())
        p_amp_ii += float(np.mean(np.abs(seg - seg.mean())))
        if v1 is not None:
            p_amp_v1 += float(np.mean(np.abs(v1[a:b] - v1[a:b].mean())))
        used += 1
    if used:
        out[0] = p_amp_ii / used
        out[1] = p_amp_v1 / used

    # Consistency of the P window across beats: high for sinus, low for AFIB.
    if len(p_segments_ii) >= 2:
        length = min(len(s) for s in p_segments_ii)
        stack = np.stack([s[:length] for s in p_segments_ii])
        template = stack.mean(axis=0)
        tnorm = np.linalg.norm(template)
        if tnorm > _EPS:
            corrs = [float(np.dot(s, template) /
                           (np.linalg.norm(s) * tnorm + _EPS)) for s in stack]
            out[2] = float(np.mean(corrs))

    # ---- TQ (diastolic) window: after the T wave, before the next P ----
    tq_ii, tq_v1, tq_avf = [], [], []
    for r0, r1 in zip(rpeaks[:-1], rpeaks[1:]):
        span = r1 - r0
        if span < int(round(0.30 * fs)):
            continue
        a = r0 + int(round(0.36 * span))
        b = r0 + int(round(0.92 * span))
        if b - a < int(round(0.10 * fs)) or b > n:
            continue
        tq_ii.append(ii[a:b])
        if v1 is not None:
            tq_v1.append(v1[a:b])
        if avf is not None:
            tq_avf.append(avf[a:b])

    def concat(chunks):
        return np.concatenate(chunks) if chunks else np.zeros(0)

    cat_ii, cat_v1, cat_avf = concat(tq_ii), concat(tq_v1), concat(tq_avf)

    # Flutter band: 210-390 atrial beats per minute = 3.5-6.5 Hz.
    frac_ii, peak_f, conc = _band_power(cat_ii, fs, 3.5, 6.5)
    out[3] = frac_ii
    out[4] = _band_power(cat_v1, fs, 3.5, 6.5)[0]
    out[5] = _band_power(cat_avf, fs, 3.5, 6.5)[0]

    _, wide_peak, wide_conc = _band_power(cat_ii, fs, 2.0, 8.0)
    out[6] = wide_peak
    out[7] = wide_conc
    out[8] = float(np.std(cat_v1)) if cat_v1.size else 0.0

    # Periodicity of the diastolic segment: flutter is regular, AFIB is not.
    if cat_ii.size >= int(round(0.5 * fs)):
        seg = cat_ii - cat_ii.mean()
        ac = np.correlate(seg, seg, mode="full")[seg.size - 1:]
        if ac[0] > _EPS:
            ac = ac / ac[0]
            lo = int(round(fs / 8.0))
            hi = min(int(round(fs / 2.0)), ac.size - 1)
            if hi > lo:
                out[9] = float(np.max(ac[lo:hi]))
    out[10] = out[6] * 60.0
    return out


def _qrs_features(sig, rpeaks, fs):
    """Six QRS-shape descriptors, aimed at LBBB vs RBBB."""
    out = np.zeros(6, dtype=np.float64)
    n = sig.shape[-1]
    if rpeaks.size < 1:
        return out

    def lead(name):
        idx = _LEAD_IDX.get(name)
        return sig[idx] if idx is not None and idx < sig.shape[0] else None

    ii = lead("II") if lead("II") is not None else sig[0]
    v1, v6 = lead("V1"), lead("V6")

    half = int(round(0.06 * fs))       # +-60 ms covers even a wide QRS
    durations, areas, amps_v1, amps_v6 = [], [], [], []
    hi_energy, lo_energy = 0.0, 0.0

    for r in rpeaks:
        a, b = max(r - half, 0), min(r + half, n)
        if b - a < 8:
            continue
        seg = ii[a:b] - np.median(ii[a:b])

        # Duration: the span where |signal| stays above 20 % of the R peak.
        thresh = 0.20 * np.max(np.abs(seg))
        above = np.flatnonzero(np.abs(seg) > thresh)
        if above.size:
            durations.append((above[-1] - above[0] + 1) / float(fs))
        areas.append(float(np.sum(np.abs(seg))) / fs)

        if v1 is not None:
            amps_v1.append(float(v1[r]) if r < n else 0.0)
        if v6 is not None:
            amps_v6.append(float(v6[r]) if r < n else 0.0)

        # Notching / slurring shows up as extra 15-40 Hz energy inside the QRS.
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
        freqs = np.fft.rfftfreq(seg.size, d=1.0 / fs)
        hi_energy += float(spec[(freqs >= 15.0) & (freqs <= 40.0)].sum())
        lo_energy += float(spec[(freqs > 0.0) & (freqs < 15.0)].sum())

    out[0] = float(np.mean(durations)) if durations else 0.0
    out[1] = float(np.mean(amps_v1)) if amps_v1 else 0.0
    out[2] = float(np.mean(amps_v6)) if amps_v6 else 0.0
    out[3] = out[1] / (abs(out[2]) + _EPS)
    out[4] = hi_energy / (lo_energy + _EPS)
    out[5] = float(np.mean(areas)) if areas else 0.0
    return out


def extract_features(sig, fs=NATIVE_FS):
    """Raw 12-lead millivolts -> the 37 hand features, float32.

    Always called on the native-rate signal. Filtering here is local to the
    feature path and independent of the network input path.
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    clean = sosfiltfilt(butter_highpass_sos(HP_CUTOFF, fs, HP_ORDER), x)
    clean = sosfiltfilt(butter_lowpass_sos(min(45.0, 0.45 * fs), fs, LP_ORDER),
                        clean)

    # Scale-free features: normalise amplitude the same way the network input is
    # normalised, so gain differences between recorders do not leak in.
    scale = 1.4826 * np.median(np.abs(clean - np.median(clean)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(clean)) or 1.0
    clean = clean / scale

    rpeaks = detect_rpeaks(clean, fs)

    feats = np.concatenate([
        _rr_features(rpeaks, fs, x.shape[-1]),
        _atrial_features(clean, rpeaks, fs),
        _qrs_features(clean, rpeaks, fs),
    ])
    feats = np.array([_safe(v) for v in feats], dtype=np.float32)
    assert feats.size == N_FEATURES
    return feats


def preprocess_record(sig, fs_in=NATIVE_FS, target_fs=None):
    """Convenience wrapper: one raw record -> ``(X, F)``."""
    return (preprocess_signal(sig, fs_in, target_fs),
            extract_features(sig, fs_in))


_cfg = preprocess_config()
_cfg["feature_names"] = list(FEATURE_NAMES)


def preprocess_config():  # noqa: F811 -- redefined once FEATURE_NAMES exists
    """The exact settings used, for the manifest and for preprocess.json."""
    return dict(_cfg)
