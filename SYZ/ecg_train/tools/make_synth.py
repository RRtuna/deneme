"""make_synth -- build a synthetic WFDB dataset shaped like the real one.

    python tools/make_synth.py --out /path/to/SYNTH --per-class 200

THIS IS NOT MEDICAL DATA. It exists for one reason: to run prep -> train ->
ensemble -> export -> predict end to end and prove the plumbing works on a
machine that does not have the real records. Every accuracy number produced
from this dataset is meaningless as a clinical claim, and no modelling
decision may be taken from it.

The layout mirrors the description in GOREV.md so the real pipeline needs no
special-casing:

    <out>/Normal/NORM_000001/<id>.hea + .dat      12 leads, 500 Hz, 10 s
    <out>/AFIB/AFIB_000001/...
    <out>/train.csv  validation.csv  test_public.csv     70 / 15 / 15

The five rhythms are drawn with the features that actually distinguish them,
so a model that learns nothing will still score near chance while a working
pipeline will not:

    Normal  regular RR, a clear P before every QRS, narrow QRS
    AFIB    irregular RR, no P, low-amplitude broadband fibrillatory noise
    AFL     regular RR, no P, a 5 Hz sawtooth running through diastole
    LBBB    wide QRS, broad monophasic R in V5/V6, deep S in V1/V2
    RBBB    wide QRS, rSR' in V1/V2, slurred S in I/V6

AFIB and AFL are deliberately drawn with overlapping parameters, because the
whole difficulty of the real task lives in that pair.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

FS = 500.0
SECONDS = 10.0
N = int(FS * SECONDS)
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6")
CLASSES = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")
PREFIX = {"Normal": "NORM", "AFIB": "AFIB", "AFL": "AFLT",
          "LBBB": "LBBB", "RBBB": "RBBB"}
GAIN = 1000.0                                   # ADC units per mV

# Per-lead amplitudes of each wave, roughly the normal 12-lead progression.
A_P = np.array([0.10, 0.15, 0.06, -0.12, 0.04, 0.10,
                0.05, 0.08, 0.06, 0.05, 0.05, 0.04])
A_R = np.array([0.80, 1.20, 0.50, -1.00, 0.30, 0.90,
                0.25, 0.40, 0.90, 1.40, 1.30, 1.00])
A_S = np.array([-0.15, -0.20, -0.10, 0.10, -0.10, -0.15,
                -1.20, -1.40, -0.90, -0.40, -0.20, -0.15])
A_T = np.array([0.20, 0.30, 0.10, -0.25, 0.08, 0.20,
                -0.05, 0.25, 0.35, 0.35, 0.30, 0.22])
IDX = {name: i for i, name in enumerate(LEADS)}


def _gauss(t, centre, width, amp):
    return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)


def _beat(t_rel, cls, rng):
    """One PQRST complex sampled on ``t_rel`` (seconds relative to the R peak).

    Returns a (12, len(t_rel)) array in millivolts.
    """
    out = np.zeros((12, t_rel.size))
    wide = cls in ("LBBB", "RBBB")
    r_width = rng.uniform(0.016, 0.020) * (2.1 if wide else 1.0)

    for lead in range(12):
        wave = np.zeros(t_rel.size)

        # P wave -- absent in both atrial arrhythmias.
        if cls not in ("AFIB", "AFL"):
            wave += _gauss(t_rel, -rng.uniform(0.15, 0.19), 0.022,
                           A_P[lead] * rng.uniform(0.8, 1.2))

        amp_r = A_R[lead] * rng.uniform(0.85, 1.15)
        amp_s = A_S[lead] * rng.uniform(0.85, 1.15)

        if cls == "LBBB":
            # Broad monophasic R laterally, deep wide S over the right precordium.
            if lead in (IDX["V5"], IDX["V6"], IDX["I"], IDX["aVL"]):
                wave += _gauss(t_rel, 0.0, r_width * 1.5, amp_r * 1.15)
                wave += _gauss(t_rel, 0.030, r_width * 1.3, amp_r * 0.45)
            elif lead in (IDX["V1"], IDX["V2"], IDX["V3"]):
                wave += _gauss(t_rel, 0.0, r_width * 0.7, abs(amp_r) * 0.15)
                wave += _gauss(t_rel, 0.030, r_width * 1.8, amp_s * 1.35)
            else:
                wave += _gauss(t_rel, 0.0, r_width * 1.4, amp_r)
                wave += _gauss(t_rel, 0.032, r_width * 1.2, amp_s)
        elif cls == "RBBB":
            # rSR' in V1/V2: a second, later R prime is the whole signature.
            if lead in (IDX["V1"], IDX["V2"]):
                wave += _gauss(t_rel, -0.012, r_width * 0.6, abs(amp_r) * 0.55)
                wave += _gauss(t_rel, 0.008, r_width * 0.8, amp_s * 0.5)
                wave += _gauss(t_rel, 0.045, r_width * 1.2,
                               abs(amp_r) * rng.uniform(0.9, 1.4))
            elif lead in (IDX["I"], IDX["V5"], IDX["V6"]):
                wave += _gauss(t_rel, 0.0, r_width * 0.9, amp_r)
                wave += _gauss(t_rel, 0.048, r_width * 1.9, -abs(amp_r) * 0.35)
            else:
                wave += _gauss(t_rel, 0.0, r_width, amp_r)
                wave += _gauss(t_rel, 0.030, r_width * 1.3, amp_s)
        else:
            wave += _gauss(t_rel, -0.028, r_width * 0.7, -abs(amp_r) * 0.08)
            wave += _gauss(t_rel, 0.0, r_width, amp_r)
            wave += _gauss(t_rel, 0.026, r_width * 1.1, amp_s)

        t_off = 0.30 if not wide else 0.36
        t_amp = A_T[lead] * rng.uniform(0.8, 1.2)
        if wide:
            t_amp *= -0.8                       # discordant T, as in real BBB
        wave += _gauss(t_rel, t_off, 0.055, t_amp)

        out[lead] = wave
    return out


def _rr_series(cls, rng, hard=False):
    """Beat times over the 10 s strip, with class-appropriate regularity."""
    if cls == "AFIB":
        base = rng.uniform(0.55, 0.95)
        # In hard mode a slice of AFIB records is drawn nearly regular, which
        # is what organised AFIB looks like and why it gets called flutter.
        spread = 0.55 if not hard else rng.choice([0.55, 0.80, 0.92],
                                                  p=[0.55, 0.25, 0.20])
        times, t = [], rng.uniform(0.2, 0.6)
        while t < SECONDS - 0.35:
            times.append(t)
            # Irregularly irregular: the defining feature of AFIB.
            t += float(np.clip(base * rng.uniform(spread, 2.0 - spread),
                               0.28, 1.6))
        return np.asarray(times)

    if cls == "AFL":
        # Flutter conducts in fixed ratios, so ventricular rhythm stays regular.
        atrial = rng.uniform(4.6, 5.6)          # Hz, ~280-335 per minute
        ratio = rng.choice([2, 3, 4], p=[0.5, 0.2, 0.3])
        rr = ratio / atrial
        jitter = 0.012 if rng.random() < 0.35 else 0.002
        if hard and rng.random() < 0.30:
            # Variable block: flutter with an irregular ventricular response,
            # the single most common reason AFL is mistaken for AFIB.
            times, t = [], rng.uniform(0.2, 0.5)
            while t < SECONDS - 0.35:
                times.append(t)
                t += float(np.clip(rng.choice([2, 3, 4]) / atrial, 0.28, 1.6))
            return np.asarray(times)
        times, t = [], rng.uniform(0.2, 0.5)
        while t < SECONDS - 0.35:
            times.append(t)
            t += rr + rng.normal(0.0, jitter)
        return np.asarray(times)

    base = rng.uniform(0.62, 1.00)
    times, t = [], rng.uniform(0.2, 0.6)
    while t < SECONDS - 0.35:
        times.append(t)
        t += base + rng.normal(0.0, 0.022)      # ordinary sinus variability
    return np.asarray(times)


def synth_record(cls, rng, hard=False):
    """Generate one 12-lead record, returned as float millivolts (12, N)."""
    t = np.arange(N) / FS
    sig = np.zeros((12, N))

    beats = _rr_series(cls, rng, hard=hard)
    half = int(0.45 * FS)
    for r_time in beats:
        centre = int(round(r_time * FS))
        lo, hi = max(centre - half, 0), min(centre + half, N)
        if hi - lo < 10:
            continue
        t_rel = (np.arange(lo, hi) - centre) / FS
        sig[:, lo:hi] += _beat(t_rel, cls, rng)

    if cls == "AFL":
        # Sawtooth flutter waves, largest in II / III / aVF and visible in V1.
        f = rng.uniform(4.6, 5.6)
        phase = rng.uniform(0, 2 * np.pi)
        saw = 2.0 * ((f * t + phase / (2 * np.pi)) % 1.0) - 1.0
        saw = saw - saw.mean()
        amp = rng.uniform(0.06, 0.16)
        if hard:
            # Fine flutter waves and a wandering rate blur the sawtooth toward
            # the fibrillatory end of the spectrum.
            amp *= rng.uniform(0.35, 1.0)
            drift = np.cumsum(rng.standard_normal(N)) / FS
            drift *= rng.uniform(0.0, 0.35) / (np.std(drift) + 1e-9)
            saw = 2.0 * ((f * t + drift + phase / (2 * np.pi)) % 1.0) - 1.0
            saw -= saw.mean()
        for lead, weight in ((IDX["II"], 1.0), (IDX["III"], 0.9),
                             (IDX["aVF"], 0.95), (IDX["V1"], 0.55),
                             (IDX["I"], 0.2), (IDX["aVR"], -0.6)):
            sig[lead] += amp * weight * saw
    elif cls == "AFIB":
        # Fibrillatory waves: same band, but noise-like instead of periodic.
        amp = rng.uniform(0.02, 0.09)
        noise = rng.standard_normal(N)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(N, 1.0 / FS)
        # Coarse AFIB concentrates its f-waves into a narrower band, which is
        # precisely where it starts to look like flutter.
        lo, hi = (4.0, 9.0) if not hard else (
            (4.5, 6.5) if rng.random() < 0.35 else (4.0, 9.0))
        spec[(freqs < lo) | (freqs > hi)] = 0
        fwave = np.fft.irfft(spec, n=N)
        fwave /= (np.std(fwave) + 1e-9)
        for lead, weight in ((IDX["V1"], 1.0), (IDX["II"], 0.6),
                             (IDX["III"], 0.5), (IDX["aVF"], 0.55)):
            sig[lead] += amp * weight * fwave

    # Recording artefacts every real strip carries.
    noise_gain = 1.0 if not hard else rng.uniform(1.0, 3.5)
    for lead in range(12):
        sig[lead] += 0.05 * rng.uniform(0.4, 1.6) * noise_gain * np.sin(
            2 * np.pi * rng.uniform(0.15, 0.4) * t + rng.uniform(0, 6.28))
        sig[lead] += rng.uniform(0.004, 0.020) * noise_gain * rng.standard_normal(N)
        sig[lead] += rng.uniform(0.002, 0.012) * np.sin(
            2 * np.pi * 50.0 * t + rng.uniform(0, 6.28))
        if hard and rng.random() < 0.10:
            # Lead dropout / saturation, both common in real 12-lead exports.
            if rng.random() < 0.5:
                sig[lead] *= rng.uniform(0.0, 0.15)
            else:
                sig[lead] = np.clip(sig[lead], -0.5, 0.5)
    return sig


def write_record(directory, record_id, sig):
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, record_id)
    raw = np.clip(np.round(sig * GAIN), -32768, 32767).astype("<i2")
    raw.T.reshape(-1).tofile(base + ".dat")

    lines = ["%s 12 %g %d" % (record_id, FS, N)]
    for lead in LEADS:
        lines.append("%s.dat 16 %g(0)/mV 16 0 0 0 0 %s" % (record_id, GAIN, lead))
    with open(base + ".hea", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return base + ".hea"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset root to create")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--difficulty", choices=("easy", "hard"), default="hard",
                    help="'easy' is trivially separable and only proves the "
                         "code runs; 'hard' overlaps AFIB/AFL and adds "
                         "artefacts so scores land in a realistic band")
    args = ap.parse_args(argv)

    hard = args.difficulty == "hard"
    rng = np.random.default_rng(args.seed)
    root = os.path.abspath(args.out)
    os.makedirs(root, exist_ok=True)

    rows = []
    for cls in CLASSES:
        for k in range(args.per_class):
            rec_id = "%s_%06d" % (PREFIX[cls], k + 1)
            folder = os.path.join(root, cls, rec_id)
            sig = synth_record(cls, rng, hard=hard)
            hea = write_record(folder, rec_id, sig)
            rows.append({"id": os.path.relpath(hea, root).replace("\\", "/"),
                         "diagnosis": cls, "cls": cls})
        print("  %-7s %d kayit" % (cls, args.per_class), flush=True)

    # Stratified 70 / 15 / 15, matching the 3500 / 750 / 750 proportions.
    splits = {"train": [], "validation": [], "test_public": []}
    for cls in CLASSES:
        subset = [r for r in rows if r["cls"] == cls]
        order = rng.permutation(len(subset))
        n_tr = int(round(0.70 * len(subset)))
        n_va = int(round(0.15 * len(subset)))
        for rank, i in enumerate(order):
            split = ("train" if rank < n_tr
                     else "validation" if rank < n_tr + n_va else "test_public")
            splits[split].append(subset[i])

    for split, items in splits.items():
        path = os.path.join(root, "%s.csv" % split)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "diagnosis"])
            for r in items:
                w.writerow([r["id"], r["diagnosis"]])
        print("%-14s %4d kayit -> %s" % (split, len(items), path))

    print("\nSENTETIK VERI -- gercek EKG degil, sadece boru hattini dogrulamak icin.")
    print("root: %s" % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
