"""afib_afl_diag -- FAZ 2.5: is the AFIB/AFL wall a model problem or a label problem?

    python afib_afl_diag.py --oof ensemble_oof_prob.npy
    python afib_afl_diag.py --oof baseline/r18_feat/oof_prob.npy --plot 30
    python afib_afl_diag.py --oof runs/main_v2/oof_prob.npy --write-exclude suspects.npy

Trains nothing. Reads an existing out-of-fold probability matrix and answers
three questions in a few seconds:

  1. How much macro-F1 is actually sitting in the AFIB/AFL pair? (a ceiling
     sweep, so "fix the pair" turns into a number instead of a feeling)
  2. Which records does the model call wrong *with high confidence*? Those are
     the label-error candidates -- a model is rarely confidently wrong about a
     record whose label is right.
  3. What do those records look like? Leads II, III, aVF and V1 are written out
     as SVG, because that is where flutter waves show up.

Leak safety: every prediction used here is out-of-fold, so the fold that judged
a record never trained on it. Records from test_public are excluded outright --
they may not inform any decision, including which records to drop.

The exclusion list is written as cache row indices for ``train.py --exclude``,
which removes them from TRAINING ONLY; validation and test folds keep them, so
the OOF comparison before and after stays honest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ecg_preprocess as ep
import wfdb_lite as wl
from train import DEV_SPLITS, TEST_SPLIT, load_cache, macro_f1

AFIB, AFL = 1, 2
PLOT_LEADS = ("II", "III", "aVF", "V1")


# --------------------------------------------------------------------------
# how much is the pair worth?
# --------------------------------------------------------------------------

def ceiling_sweep(y_dev, prob, rng_seed=0):
    """macro-F1 if the AFIB/AFL decision were right a given fraction of the time.

    Everything outside the pair is left exactly as the model predicted; only
    the within-pair choice is replaced. That isolates the pair's contribution
    instead of assuming a uniformly better model.
    """
    rng = np.random.default_rng(rng_seed)
    pred = prob.argmax(1)
    pair_rows = np.flatnonzero(np.isin(y_dev, [AFIB, AFL]))

    # Rows the model already routes into the pair correctly; only these can be
    # rescued by a better within-pair decision.
    routed = pair_rows[np.isin(pred[pair_rows], [AFIB, AFL])]

    out = []
    for target in (0.70, 0.76, 0.80, 0.85, 0.90, 0.95, 1.00):
        trials = []
        for _ in range(20):
            adjusted = pred.copy()
            correct = rng.random(len(routed)) < target
            adjusted[routed] = np.where(correct, y_dev[routed],
                                        np.where(y_dev[routed] == AFIB, AFL, AFIB))
            trials.append(macro_f1(y_dev, adjusted)[0])
        out.append((target, float(np.mean(trials))))
    return out, len(pair_rows), len(routed)


# --------------------------------------------------------------------------
# suspects
# --------------------------------------------------------------------------

def find_suspects(y, prob, dev_idx, threshold=0.80):
    """Development rows whose true label is one of the pair but the OOF model
    confidently says the other."""
    rows = []
    for i in dev_idx:
        truth = int(y[i])
        if truth not in (AFIB, AFL):
            continue
        other = AFL if truth == AFIB else AFIB
        p = prob[i]
        if p.sum() <= 1e-9:
            continue
        p = p / p.sum()
        if p[other] >= threshold and p[other] > p[truth]:
            rows.append({"idx": int(i), "true": ep.CLASSES[truth],
                         "predicted": ep.CLASSES[other],
                         "confidence": float(p[other]),
                         "p_true": float(p[truth])})
    rows.sort(key=lambda r: -r["confidence"])
    return rows


# --------------------------------------------------------------------------
# SVG strips -- no plotting library needed
# --------------------------------------------------------------------------

def ecg_svg(sig, fs, leads, title, width=1100, lead_height=130):
    """Render selected leads as a standalone SVG string."""
    idx = {name: i for i, name in enumerate(ep.STANDARD_LEADS)}
    rows = [(name, sig[idx[name]]) for name in leads if name in idx]
    if not rows:
        return ""

    n = len(rows[0][1])
    height = lead_height * len(rows) + 40
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d" font-family="monospace">'
             % (width, height, width, height),
             '<rect width="100%%" height="100%%" fill="#ffffff"/>',
             '<text x="8" y="20" font-size="14" fill="#111">%s</text>'
             % _escape(title)]

    # 200 ms grid, the standard ECG paper spacing.
    step = width * (0.2 * fs) / n
    x = 0.0
    while x < width:
        parts.append('<line x1="%.1f" y1="30" x2="%.1f" y2="%d" stroke="#f0d0d0" '
                     'stroke-width="0.5"/>' % (x, x, height))
        x += step

    for row, (name, wave) in enumerate(rows):
        top = 40 + row * lead_height
        mid = top + lead_height / 2.0
        scale = np.percentile(np.abs(wave - np.median(wave)), 99) or 1.0
        ys = mid - (wave - np.median(wave)) / (scale * 3.0) * (lead_height / 2.0)
        ys = np.clip(ys, top + 4, top + lead_height - 4)
        xs = np.linspace(0, width, n)

        step_pts = max(n // width and 1, 1)
        pts = " ".join("%.1f,%.1f" % (xs[i], ys[i]) for i in range(0, n, step_pts))
        parts.append('<line x1="0" y1="%.1f" x2="%d" y2="%.1f" stroke="#e8e8e8" '
                     'stroke-width="0.5"/>' % (mid, width, mid))
        parts.append('<polyline points="%s" fill="none" stroke="#111" '
                     'stroke-width="0.9"/>' % pts)
        parts.append('<text x="6" y="%.1f" font-size="12" fill="#c00">%s</text>'
                     % (top + 14, name))

    parts.append('</svg>')
    return "".join(parts)


def _escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def plot_suspects(suspects, paths, out_dir, limit):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for row in suspects[:limit]:
        path = paths[row["idx"]]
        try:
            sig, fs, _ = wl.read_record(path)
            clean = ep.sosfiltfilt(
                ep.butter_highpass_sos(ep.HP_CUTOFF, fs, ep.HP_ORDER), sig)
            clean = ep.sosfiltfilt(
                ep.butter_lowpass_sos(min(40.0, 0.45 * fs), fs, ep.LP_ORDER), clean)
        except Exception as exc:                # noqa: BLE001
            print("  cizilemedi %s: %s" % (path, exc))
            continue

        record = os.path.splitext(os.path.basename(path))[0]
        title = ("%s | etiket=%s | model=%s (%.0f%%) | testere disi F dalgasi "
                 "var mi?" % (record, row["true"], row["predicted"],
                              100 * row["confidence"]))
        svg = ecg_svg(clean, fs, PLOT_LEADS, title)
        name = "%s_%s_as_%s.svg" % (record, row["true"], row["predicted"])
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(svg)
        written.append(name)
    return written


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default="ensemble_oof_prob.npy",
                    help="(n_cache_rows, 5) out-of-fold probabilities")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--plot", type=int, default=24,
                    help="how many suspect records to render as SVG")
    ap.add_argument("--plot-dir", default="diag/afib_afl")
    ap.add_argument("--write-exclude", default="",
                    help="write suspect row indices here for train.py --exclude")
    ap.add_argument("--report", default="diag/afib_afl_report.json")
    args = ap.parse_args(argv)

    if not os.path.exists(args.oof):
        raise SystemExit("%s yok. Once tam 5-fold kos veya --oof ile baska bir "
                         "matris ver." % args.oof)

    X, Fe, y, split, records, ok = load_cache(args.cache, mmap=True)
    del X, Fe
    prob = np.load(args.oof).astype(np.float64)
    if prob.shape[0] != len(y):
        raise SystemExit("oof_prob satir sayisi %d, cache %d -- ayni cache mi?"
                         % (prob.shape[0], len(y)))

    usable = ok & (y >= 0)
    dev_idx = np.flatnonzero(np.isin(split, DEV_SPLITS) & usable)
    n_test_excluded = int(np.sum(split == TEST_SPLIT))

    dev_prob = prob[dev_idx]
    dev_prob = dev_prob / np.where(dev_prob.sum(1, keepdims=True) < 1e-9, 1.0,
                                   dev_prob.sum(1, keepdims=True))
    y_dev = y[dev_idx]
    pred = dev_prob.argmax(1)

    base_f1 = macro_f1(y_dev, pred)[0]
    pair_mask = np.isin(y_dev, [AFIB, AFL])
    routed_mask = pair_mask & np.isin(pred, [AFIB, AFL])
    pair_pred = np.where(dev_prob[pair_mask][:, AFIB] >= dev_prob[pair_mask][:, AFL],
                         AFIB, AFL)
    pair_acc = float(np.mean(pair_pred == y_dev[pair_mask]))
    routing_acc = float(np.mean(np.isin(pred[pair_mask], [AFIB, AFL])))

    print("=== AFIB/AFL TESHISI (OOF, %d gelistirme kaydi) ===" % len(dev_idx))
    print("  test_public disarida birakildi: %d kayit" % n_test_excluded)
    print("  OOF macro-F1                   : %.4f" % base_f1)
    print("  'bu ikisinden biri' dogrulugu  : %.4f" % routing_acc)
    print("  ikilinin ICINDE dogruluk       : %.4f" % pair_acc)
    print("  ikili kayit sayisi             : %d (%d tanesi dogru yonlendirildi)"
          % (int(pair_mask.sum()), int(routed_mask.sum())))

    print("\n  tavan taramasi -- ikili dogruluk su olsaydi macro-F1 ne olurdu:")
    sweep, n_pair, n_routed = ceiling_sweep(y_dev, dev_prob)
    for target, score in sweep:
        marker = "  <- su an" if abs(target - round(pair_acc, 2)) < 0.011 else ""
        print("    %.2f -> %.4f  (%+.4f)%s" % (target, score, score - base_f1,
                                               marker))

    suspects = find_suspects(y, prob, dev_idx, args.threshold)
    print("\n  yuksek guvenli ters tahmin (p >= %.2f): %d kayit"
          % (args.threshold, len(suspects)))
    by_kind = {}
    for row in suspects:
        by_kind.setdefault((row["true"], row["predicted"]), []).append(row)
    for (truth, predicted), rows in sorted(by_kind.items()):
        print("    gercek %-5s -> model %-5s : %3d kayit  (ort. guven %.2f)"
              % (truth, predicted, len(rows),
                 float(np.mean([r["confidence"] for r in rows]))))

    if suspects:
        print("\n  en guvenli 10 supheli:")
        for row in suspects[:10]:
            print("    %-22s %-5s -> %-5s  %.3f"
                  % (records[row["idx"]], row["true"], row["predicted"],
                     row["confidence"]))

    written = []
    if args.plot and suspects:
        import csv as _csv

        with open(os.path.join(args.cache, "index.csv"), newline="") as fh:
            paths = [r["path"] for r in _csv.DictReader(fh)]
        written = plot_suspects(suspects, paths, args.plot_dir, args.plot)
        print("\n  %d SVG yazildi: %s/" % (len(written), args.plot_dir))
        print("  Bunlari ac ve BAK: II/III/aVF'de duzenli testere disi F dalgasi")
        print("  goruyorsan etiket AFL olmali. Duzensiz ince dalgalanma varsa")
        print("  AFIB. Ayirt edemiyorsan etiket zaten tartismali demektir.")

    if args.write_exclude:
        idx = np.array([r["idx"] for r in suspects], dtype=np.int64)
        np.save(args.write_exclude, idx)
        print("\n  %d indeks yazildi: %s" % (len(idx), args.write_exclude))
        print("  Sonraki adim (SADECE egitimden cikarir, val/test dokunulmaz):")
        print("    python train.py --preset <p> --tag clean_v1 --exclude %s"
              % args.write_exclude)
        print("  Karar: OOF belirgin artiyorsa etiket sorunu dogrulanmis olur.")
        print("  Artmiyorsa sinyal tabanli tavana carpilmistir -- FAZ 2/3/5'e don.")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump({
            "oof_source": os.path.abspath(args.oof),
            "n_dev": int(len(dev_idx)),
            "n_test_excluded": n_test_excluded,
            "oof_macro_f1": base_f1,
            "routing_accuracy": routing_acc,
            "within_pair_accuracy": pair_acc,
            "n_pair_records": int(pair_mask.sum()),
            "ceiling_sweep": [{"pair_accuracy": t, "macro_f1": s} for t, s in sweep],
            "threshold": args.threshold,
            "suspects": [dict(r, record=records[r["idx"]]) for r in suspects],
            "svg_written": written,
        }, fh, indent=2)
    print("\nyazildi: %s" % args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
