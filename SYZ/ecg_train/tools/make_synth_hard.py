"""make_synth_hard -- AFIB/AFL zorlugunu YENIDEN URETEN kiyas kumesi.

    python tools/make_synth_hard.py --out /path/HARD --per-class 500 --theta-sep 0.30

Amac: gercek projedeki ~0.75'lik AFIB/AFL ikili dogrulugunu taklit eden, ama
**tavani analitik olarak bilinen** bir veri kumesi uretmek. Boylece bir fikir
denendiginde "iyilesti mi" degil, "tavana ne kadar yaklasti" sorusu
cevaplanabilir.

Neden gerekli
-------------
Kolay sentetik veride (macro-F1 0.99) hicbir fikir ayirt edilemez: her sey
calisir gorunur. Gercek veride ise her deneme saatler suruyor. Arada bir
tezgah lazim.

Nasil zorlastiriliyor
---------------------
AFIB ve AFL'yi ayri parametre kumeleriyle uretmek yerine, **tek bir gizli
degisken** kullaniliyor:

    theta in [0,1]   atriyal aktivitenin "organize olma" derecesi
      theta = 1  ->  saf testere disi (klasik AFL)
      theta = 0  ->  saf bant gurultusu (klasik AFIB)

    AFL  : theta ~ Beta(a, b)      ortalamasi yuksek
    AFIB : theta ~ Beta(b, a)      ortalamasi dusuk

Iki dagilim ORTUSUR. Ortusme miktari --theta-sep ile ayarlanir. Ventrikuler
duzenlilik de theta'ya bagli (organize flutter daha sabit blok yapar), yani
RR ipucu ile atriyal ipucu ayni gizli degiskenden gelir -- gercekte oldugu gibi.

Bu kurulumda **Bayes tavani hesaplanabilir**: theta'yi mukemmel olcen bir
gozlemcinin ulasabilecegi en yuksek ikili dogruluk. Uretici bunu ekrana yazar
ve meta.json'a kaydeder.

Onemli sinir
------------
Bu hala sentetik veridir. Burada ise yarayan bir fikir gercek veride ise
yaramayabilir. Ama burada ise YARAMAYAN bir fikir, gercek veride de buyuk
olasilikla yaramaz -- ve elemek, bulmaktan ucuzdur. Tezgahin isi budur.
"""

from __future__ import annotations

import argparse
import csv
import json
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
GAIN = 1000.0
IDX = {n: i for i, n in enumerate(LEADS)}

A_P = np.array([0.10, 0.15, 0.06, -0.12, 0.04, 0.10,
                0.05, 0.08, 0.06, 0.05, 0.05, 0.04])
A_R = np.array([0.80, 1.20, 0.50, -1.00, 0.30, 0.90,
                0.25, 0.40, 0.90, 1.40, 1.30, 1.00])
A_S = np.array([-0.15, -0.20, -0.10, 0.10, -0.10, -0.15,
                -1.20, -1.40, -0.90, -0.40, -0.20, -0.15])
A_T = np.array([0.20, 0.30, 0.10, -0.25, 0.08, 0.20,
                -0.05, 0.25, 0.35, 0.35, 0.30, 0.22])


def _g(t, c, w, a):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def bayes_pair_accuracy(a, b, n=400000, seed=0):
    """theta mukemmel olculse ikili dogruluk en fazla ne olurdu."""
    rng = np.random.default_rng(seed)
    afl = rng.beta(a, b, n)          # AFL: organize
    afib = rng.beta(b, a, n)         # AFIB: dagini
    # Esit onsel, optimal kural theta > 0.5
    return float(((afl > 0.5).mean() + (afib <= 0.5).mean()) / 2.0)


def _beat(t_rel, cls, rng, wide):
    out = np.zeros((12, t_rel.size))
    r_width = rng.uniform(0.016, 0.020) * (2.1 if wide else 1.0)
    for lead in range(12):
        w = np.zeros(t_rel.size)
        if cls not in ("AFIB", "AFL"):
            w += _g(t_rel, -rng.uniform(0.15, 0.19), 0.022,
                    A_P[lead] * rng.uniform(0.8, 1.2))
        ar = A_R[lead] * rng.uniform(0.85, 1.15)
        as_ = A_S[lead] * rng.uniform(0.85, 1.15)
        if cls == "LBBB":
            if lead in (IDX["V5"], IDX["V6"], IDX["I"], IDX["aVL"]):
                w += _g(t_rel, 0.0, r_width * 1.5, ar * 1.15)
                w += _g(t_rel, 0.030, r_width * 1.3, ar * 0.45)
            elif lead in (IDX["V1"], IDX["V2"], IDX["V3"]):
                w += _g(t_rel, 0.0, r_width * 0.7, abs(ar) * 0.15)
                w += _g(t_rel, 0.030, r_width * 1.8, as_ * 1.35)
            else:
                w += _g(t_rel, 0.0, r_width * 1.4, ar)
                w += _g(t_rel, 0.032, r_width * 1.2, as_)
        elif cls == "RBBB":
            if lead in (IDX["V1"], IDX["V2"]):
                w += _g(t_rel, -0.012, r_width * 0.6, abs(ar) * 0.55)
                w += _g(t_rel, 0.008, r_width * 0.8, as_ * 0.5)
                w += _g(t_rel, 0.045, r_width * 1.2, abs(ar) * rng.uniform(0.9, 1.4))
            elif lead in (IDX["I"], IDX["V5"], IDX["V6"]):
                w += _g(t_rel, 0.0, r_width * 0.9, ar)
                w += _g(t_rel, 0.048, r_width * 1.9, -abs(ar) * 0.35)
            else:
                w += _g(t_rel, 0.0, r_width, ar)
                w += _g(t_rel, 0.030, r_width * 1.3, as_)
        else:
            w += _g(t_rel, -0.028, r_width * 0.7, -abs(ar) * 0.08)
            w += _g(t_rel, 0.0, r_width, ar)
            w += _g(t_rel, 0.026, r_width * 1.1, as_)
        t_amp = A_T[lead] * rng.uniform(0.8, 1.2) * (-0.8 if wide else 1.0)
        w += _g(t_rel, 0.36 if wide else 0.30, 0.055, t_amp)
        out[lead] = w
    return out


def synth(cls, rng, a, b):
    """Bir kayit uret. AFIB/AFL icin gizli theta kullanilir."""
    t = np.arange(N) / FS
    sig = np.zeros((12, N))
    theta = None

    # --- ventrikuler ritim ---
    if cls in ("AFIB", "AFL"):
        theta = rng.beta(a, b) if cls == "AFL" else rng.beta(b, a)
        atrial = rng.uniform(4.4, 5.8)                 # Hz
        # Organize (theta yuksek) -> sabit blok, duzenli RR.
        # Dagini (theta dusuk)   -> degisken blok, duzensiz RR.
        ratio = rng.choice([2, 3, 4], p=[0.5, 0.2, 0.3])
        times, tt = [], rng.uniform(0.2, 0.6)
        while tt < SECONDS - 0.35:
            times.append(tt)
            if rng.random() < theta:
                step = ratio / atrial                  # sabit blok
            else:
                step = rng.choice([2, 3, 4]) / atrial  # degisken blok
            jitter = (1.0 - theta) * 0.10
            tt += float(np.clip(step * (1 + rng.normal(0, jitter)), 0.28, 1.6))
        beats = np.asarray(times)
    else:
        base = rng.uniform(0.62, 1.00)
        times, tt = [], rng.uniform(0.2, 0.6)
        while tt < SECONDS - 0.35:
            times.append(tt)
            tt += base + rng.normal(0.0, 0.022)
        beats = np.asarray(times)

    wide = cls in ("LBBB", "RBBB")
    half = int(0.45 * FS)
    for r in beats:
        c = int(round(r * FS))
        lo, hi = max(c - half, 0), min(c + half, N)
        if hi - lo < 10:
            continue
        sig[:, lo:hi] += _beat((np.arange(lo, hi) - c) / FS, cls, rng, wide)

    # --- atriyal dalga: theta ile testere disi <-> gurultu arasinda gecis ---
    if theta is not None:
        phase = rng.uniform(0, 2 * np.pi)
        saw = 2.0 * ((atrial * t + phase / (2 * np.pi)) % 1.0) - 1.0
        saw -= saw.mean()
        saw /= (np.std(saw) + 1e-9)

        noise = rng.standard_normal(N)
        spec = np.fft.rfft(noise)
        fr = np.fft.rfftfreq(N, 1.0 / FS)
        spec[(fr < atrial - 1.5) | (fr > atrial + 3.0)] = 0
        fib = np.fft.irfft(spec, n=N)
        fib /= (np.std(fib) + 1e-9)

        atrial_wave = theta * saw + (1.0 - theta) * fib
        amp = rng.uniform(0.05, 0.14)
        for lead, wgt in ((IDX["II"], 1.0), (IDX["III"], 0.9), (IDX["aVF"], 0.95),
                          (IDX["V1"], 0.6), (IDX["I"], 0.2), (IDX["aVR"], -0.6)):
            sig[lead] += amp * wgt * atrial_wave

    # --- artefaktlar ---
    ng = rng.uniform(1.0, 3.0)
    for lead in range(12):
        sig[lead] += 0.05 * rng.uniform(0.4, 1.6) * ng * np.sin(
            2 * np.pi * rng.uniform(0.15, 0.4) * t + rng.uniform(0, 6.28))
        sig[lead] += rng.uniform(0.006, 0.022) * ng * rng.standard_normal(N)
        sig[lead] += rng.uniform(0.002, 0.012) * np.sin(
            2 * np.pi * 50.0 * t + rng.uniform(0, 6.28))
        if rng.random() < 0.08:
            sig[lead] *= rng.uniform(0.0, 0.2)
    return sig, theta


def write_record(directory, rec_id, sig):
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, rec_id)
    np.clip(np.round(sig * GAIN), -32768, 32767).astype("<i2").T.reshape(-1).tofile(base + ".dat")
    lines = ["%s 12 %g %d" % (rec_id, FS, N)]
    for lead in LEADS:
        lines.append("%s.dat 16 %g(0)/mV 16 0 0 0 0 %s" % (rec_id, GAIN, lead))
    with open(base + ".hea", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return base + ".hea"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--other-per-class", type=int, default=0,
                    help="Normal/LBBB/RBBB icin (0 = per-class'in yarisi)")
    ap.add_argument("--theta-sep", type=float, default=0.30,
                    help="dusuk = daha cok ortusme = daha zor. Beta(a,b) icin "
                         "a=1+4*sep, b=1+4*(1-sep)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    a = 1.0 + 4.0 * args.theta_sep
    b = 1.0 + 4.0 * (1.0 - args.theta_sep)
    bayes = bayes_pair_accuracy(a, b)
    print("theta dagilimlari: AFL ~ Beta(%.2f, %.2f)   AFIB ~ Beta(%.2f, %.2f)"
          % (a, b, b, a))
    print("BAYES TAVANI (ikili dogruluk, theta mukemmel olculurse): %.4f" % bayes)
    print("  gercek projedeki deger 0.75 -- tavan buna yakinsa tezgah temsili")
    print()

    rng = np.random.default_rng(args.seed)
    root = os.path.abspath(args.out)
    os.makedirs(root, exist_ok=True)
    other = args.other_per_class or max(args.per_class // 2, 1)

    rows, thetas = [], {}
    for cls in CLASSES:
        n = args.per_class if cls in ("AFIB", "AFL") else other
        for k in range(n):
            rec = "%s_%06d" % (PREFIX[cls], k + 1)
            sig, th = synth(cls, rng, a, b)
            hea = write_record(os.path.join(root, cls, rec), rec, sig)
            rows.append({"id": os.path.relpath(hea, root).replace("\\", "/"),
                         "diagnosis": cls, "cls": cls})
            if th is not None:
                thetas[rec] = float(th)
        print("  %-7s %d kayit" % (cls, n), flush=True)

    splits = {"train": [], "validation": [], "test_public": []}
    for cls in CLASSES:
        sub = [r for r in rows if r["cls"] == cls]
        order = rng.permutation(len(sub))
        n_tr = int(round(0.70 * len(sub)))
        n_va = int(round(0.15 * len(sub)))
        for rank, i in enumerate(order):
            s = ("train" if rank < n_tr
                 else "validation" if rank < n_tr + n_va else "test_public")
            splits[s].append(sub[i])
    for s, items in splits.items():
        with open(os.path.join(root, "%s.csv" % s), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "diagnosis"])
            for r in items:
                w.writerow([r["id"], r["diagnosis"]])
        print("%-13s %4d kayit" % (s, len(items)))

    with open(os.path.join(root, "meta_hard.json"), "w") as fh:
        json.dump({"beta_a": a, "beta_b": b, "bayes_pair_accuracy": bayes,
                   "theta_sep": args.theta_sep, "theta": thetas}, fh)
    print("\nSENTETIK KIYAS KUMESI -- fikir elemek icin, skor iddiasi icin degil.")
    print("Bayes tavani: %.4f  (meta_hard.json)" % bayes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
