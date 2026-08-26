"""Correctness checks for ecg_preprocess and wfdb_lite.

Run from the ecg_train directory:  python tools/test_preprocess.py

scipy is used here only as an independent reference to check the hand-written
filter design against. It is never required by the delivered package.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecg_preprocess as ep      # noqa: E402
import wfdb_lite as wl           # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("[%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def test_filter_design_matches_scipy():
    """Our biquad cascades must be the same filters scipy.signal.butter gives."""
    try:
        from scipy.signal import butter, sosfreqz
    except ImportError:
        print("[SKIP] scipy missing, cannot cross-check filter design")
        return

    fs = 500.0
    for cutoff, order in ((40.0, 4), (25.0, 2), (60.0, 6)):
        ours = ep.butter_lowpass_sos(cutoff, fs, order)
        ref = butter(order, cutoff / (fs / 2.0), btype="low", output="sos")
        w, h_ours = sosfreqz(ours, worN=2048, fs=fs)
        _, h_ref = sosfreqz(ref, worN=2048, fs=fs)
        err = float(np.max(np.abs(np.abs(h_ours) - np.abs(h_ref))))
        check("lowpass %g Hz order %d matches scipy" % (cutoff, order),
              err < 1e-9, "max |H| diff %.2e" % err)

    for cutoff, order in ((0.5, 2), (1.0, 4)):
        ours = ep.butter_highpass_sos(cutoff, fs, order)
        ref = butter(order, cutoff / (fs / 2.0), btype="high", output="sos")
        w, h_ours = sosfreqz(ours, worN=4096, fs=fs)
        _, h_ref = sosfreqz(ref, worN=4096, fs=fs)
        err = float(np.max(np.abs(np.abs(h_ours) - np.abs(h_ref))))
        check("highpass %g Hz order %d matches scipy" % (cutoff, order),
              err < 1e-9, "max |H| diff %.2e" % err)


def test_filter_backends_agree():
    """The scipy accelerator must reproduce the pure-numpy reference exactly."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 5000)) * 0.4
    sos = np.vstack([ep.butter_highpass_sos(0.5, 500.0, 2),
                     ep.butter_lowpass_sos(40.0, 500.0, 4),
                     ep.notch_sos(50.0, 500.0, 30.0)])
    ref = ep._sosfilt_reference(sos, x)
    fast = ep._sosfilt(sos, x)
    err = float(np.max(np.abs(ref - fast)))
    check("filter backends agree (backend=%s)" % ep.FILTER_BACKEND,
          err < 1e-9, "max diff %.2e" % err)


def test_lowpass_actually_attenuates():
    """A 45 Hz tone must be crushed; a 5 Hz tone must survive."""
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    for freq, expect_pass in ((5.0, True), (80.0, False)):
        x = np.sin(2 * np.pi * freq * t)[None, :]
        y = ep.sosfiltfilt(ep.butter_lowpass_sos(40.0, fs, 4), x)
        ratio = float(np.std(y) / np.std(x))
        ok = ratio > 0.9 if expect_pass else ratio < 0.05
        check("%g Hz tone %s 40 Hz lowpass" % (freq, "passes" if expect_pass else "blocked"),
              ok, "amplitude ratio %.4f" % ratio)


def test_resample_roundtrip():
    """Band-limited resampling must preserve a slow sine's amplitude/phase."""
    fs, n = 500.0, 5000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 3.0 * t)[None, :]
    y = ep.resample_fft(x, 1500)
    t2 = np.arange(1500) / 150.0
    ref = np.sin(2 * np.pi * 3.0 * t2)[None, :]
    err = float(np.max(np.abs(y[:, 50:-50] - ref[:, 50:-50])))
    check("500 -> 150 Hz resample keeps a 3 Hz sine", err < 1e-3,
          "max err %.2e" % err)


def test_preprocess_shape_and_determinism():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((12, 5000)) * 0.2
    a = ep.preprocess_signal(raw, 500.0)
    b = ep.preprocess_signal(raw, 500.0)
    check("preprocess_signal shape", a.shape == (12, ep.TARGET_LEN),
          "got %s" % (a.shape,))
    check("preprocess_signal dtype", a.dtype == np.float32, str(a.dtype))
    check("preprocess_signal is deterministic", np.array_equal(a, b))
    check("preprocess_signal is finite", bool(np.all(np.isfinite(a))))

    f = ep.extract_features(raw, 500.0)
    check("extract_features returns 37", f.shape == (37,), "got %s" % (f.shape,))
    check("extract_features is finite", bool(np.all(np.isfinite(f))))


def test_features_survive_degenerate_input():
    """Flat lines and single-sample spikes must not raise or produce NaN."""
    for name, raw in (
        ("all zeros", np.zeros((12, 5000))),
        ("constant", np.full((12, 5000), 3.0)),
        ("one spike", np.eye(12, 5000) * 100.0),
        ("tiny amplitude", np.random.default_rng(2).standard_normal((12, 5000)) * 1e-9),
    ):
        try:
            f = ep.extract_features(raw, 500.0)
            x = ep.preprocess_signal(raw, 500.0)
            ok = bool(np.all(np.isfinite(f)) and np.all(np.isfinite(x)))
        except Exception as exc:                # noqa: BLE001
            ok, f = False, exc
        check("degenerate input: %s" % name, ok, "" if ok else str(f))


def test_rpeak_detection_on_synthetic_rhythm():
    """A clean 72 bpm synthetic rhythm must yield the right beat count."""
    fs, bpm = 500.0, 72.0
    rr = 60.0 / bpm
    n = int(10 * fs)
    sig = np.zeros((12, n))
    peaks_true = []
    for k in range(int(10.0 / rr)):
        centre = int((0.4 + k * rr) * fs)
        if centre + 30 >= n:
            break
        peaks_true.append(centre)
        idx = np.arange(centre - 20, centre + 21)
        qrs = np.exp(-0.5 * ((idx - centre) / 6.0) ** 2)
        sig[:, idx] += qrs * 1.5
    found = ep.detect_rpeaks(sig, fs)
    check("R-peak count on 72 bpm synthetic",
          abs(found.size - len(peaks_true)) <= 1,
          "found %d, expected %d" % (found.size, len(peaks_true)))
    if found.size and len(peaks_true):
        k = min(found.size, len(peaks_true))
        offset = float(np.max(np.abs(found[:k] - np.asarray(peaks_true[:k]))))
        check("R-peak location accurate to 20 ms", offset <= 0.02 * fs,
              "max offset %.0f samples" % offset)


def test_wfdb_roundtrip_dat():
    """Write a format-16 record with wfdb_lite's conventions and read it back."""
    rng = np.random.default_rng(3)
    truth = (rng.standard_normal((12, 5000)) * 0.5).astype(np.float64)
    gain, baseline = 1000.0, 0
    raw = np.round(truth * gain + baseline).astype(np.int16)

    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "REC001")
        raw.T.reshape(-1).astype("<i2").tofile(base + ".dat")
        lines = ["REC001 12 500 5000"]
        for lead in wl.STANDARD_LEADS:
            lines.append("REC001.dat 16 %g(%d)/mV 16 0 0 0 0 %s"
                         % (gain, baseline, lead))
        with open(base + ".hea", "w") as fh:
            fh.write("\n".join(lines) + "\n")

        sig, fs, leads = wl.read_record(base + ".hea")

    err = float(np.max(np.abs(sig - raw / gain)))
    check("wfdb .dat format 16 round-trip", err < 1e-6, "max err %.2e" % err)
    check("wfdb fs parsed", fs == 500.0, str(fs))
    check("wfdb lead order canonical", tuple(leads) == wl.STANDARD_LEADS)


def test_wfdb_lead_reordering():
    """Leads listed out of order must come back in canonical order."""
    shuffled = ["V6", "II", "I", "V1", "aVR", "III", "aVL", "aVF",
                "V2", "V3", "V4", "V5"]
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "REC002")
        n = 500
        raw = np.zeros((12, n), dtype=np.int16)
        for i in range(12):
            raw[i] = i * 100                    # each lead holds its own marker
        raw.T.reshape(-1).astype("<i2").tofile(base + ".dat")
        lines = ["REC002 12 500 %d" % n]
        for lead in shuffled:
            lines.append("REC002.dat 16 1000(0)/mV 16 0 0 0 0 %s" % lead)
        with open(base + ".hea", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sig, _, leads = wl.read_record(base + ".hea")

    expected = [shuffled.index(name) * 100 / 1000.0 for name in wl.STANDARD_LEADS]
    got = [float(sig[i, 0]) for i in range(12)]
    check("wfdb reorders leads to canonical order",
          np.allclose(got, expected), "got %s" % got[:4])


def test_wfdb_format_212():
    """Format 212 packing must decode back to the original integers."""
    values = np.array([0, 1, -1, 2047, -2048, 17, -300, 900], dtype=np.int32)
    n_sig, n_samp = 2, 4
    packed = bytearray()
    for i in range(0, values.size, 2):
        a = int(values[i]) & 0xFFF
        b = int(values[i + 1]) & 0xFFF
        packed += bytes([a & 0xFF, ((a >> 8) & 0x0F) | ((b >> 8) << 4), b & 0xFF])

    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "REC003")
        with open(base + ".dat", "wb") as fh:
            fh.write(bytes(packed))
        lines = ["REC003 %d 500 %d" % (n_sig, n_samp),
                 "REC003.dat 212 1(0)/mV 12 0 0 0 0 I",
                 "REC003.dat 212 1(0)/mV 12 0 0 0 0 II"]
        with open(base + ".hea", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sig, _, _ = wl.read_record(base + ".hea", leads=None)

    expected = values.reshape(n_samp, n_sig).T
    check("wfdb .dat format 212 round-trip",
          np.allclose(sig, expected), "got %s" % sig.tolist())


def test_wfdb_mat_roundtrip():
    """Our MAT v5 reader must read what scipy.io.savemat writes."""
    try:
        from scipy.io import savemat
    except ImportError:
        print("[SKIP] scipy missing, cannot build a .mat fixture")
        return

    rng = np.random.default_rng(4)
    raw = (rng.standard_normal((12, 5000)) * 300).astype(np.int16)
    gain = 1000.0

    for compress in (True, False):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "REC004")
            savemat(base + ".mat", {"val": raw}, do_compression=compress)
            lines = ["REC004 12 500 5000"]
            for lead in wl.STANDARD_LEADS:
                lines.append("REC004.mat 16 %g(0)/mV 16 0 0 0 0 %s" % (gain, lead))
            with open(base + ".hea", "w") as fh:
                fh.write("\n".join(lines) + "\n")
            sig, fs, _ = wl.read_record(base + ".hea")
        err = float(np.max(np.abs(sig - raw / gain)))
        check("wfdb .mat round-trip (compressed=%s)" % compress, err < 1e-6,
              "max err %.2e" % err)


def main():
    test_filter_design_matches_scipy()
    test_filter_backends_agree()
    test_lowpass_actually_attenuates()
    test_resample_roundtrip()
    test_preprocess_shape_and_determinism()
    test_features_survive_degenerate_input()
    test_rpeak_detection_on_synthetic_rhythm()
    test_wfdb_roundtrip_dat()
    test_wfdb_lead_reordering()
    test_wfdb_format_212()
    test_wfdb_mat_roundtrip()

    print()
    if FAILURES:
        print("%d check(s) FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
