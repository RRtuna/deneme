"""predict -- ECG classification with onnxruntime. No PyTorch required.

    python predict.py /path/Normal/NORM_000508/48090046.hea
    python predict.py --batch %ECG_ROOT%\\test_public.csv --root %ECG_ROOT%
    python predict.py --batch list.csv --root DATA --out predictions.csv

Requires only ``numpy`` and ``onnxruntime``. Preprocessing comes from the
``ecg_preprocess.py`` sitting next to this file, which is a byte-identical
copy of the module used during training -- that is the whole reason the
package reproduces the training score.

When the batch CSV carries labels, macro-F1, per-class F1, the confusion
matrix and the AFIB/AFL binary accuracy are printed, and the score is checked
against the one recorded in manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ecg_preprocess as ep       # noqa: E402
import wfdb_lite as wl            # noqa: E402

try:                              # QRST-artik olcumleri (yalnizca genisletilmis
    import resid_features as rf   # modellerde bulunur; yoksa paket 37 ozelliklidir)
except Exception:                 # noqa: BLE001
    rf = None

CLASSES = list(ep.CLASSES)

_LABEL_ALIASES = {
    "normal": "Normal", "norm": "Normal", "sr": "Normal", "nsr": "Normal",
    "sinus": "Normal", "sinusrhythm": "Normal", "0": "Normal",
    "afib": "AFIB", "af": "AFIB", "atrialfibrillation": "AFIB", "1": "AFIB",
    "afl": "AFL", "aff": "AFL", "atrialflutter": "AFL", "flutter": "AFL",
    "2": "AFL",
    "lbbb": "LBBB", "clbbb": "LBBB", "3": "LBBB",
    "rbbb": "RBBB", "crbbb": "RBBB", "4": "RBBB",
}


def normalise_label(value):
    if value is None:
        return None
    key = str(value).strip().replace(" ", "").replace("_", "").replace("-", "").lower()
    if not key or key in ("nan", "none"):
        return None
    return _LABEL_ALIASES.get(key)


# --------------------------------------------------------------------------
# model bundle
# --------------------------------------------------------------------------

class Bundle:
    """The exported ensemble, loaded from manifest.json."""

    def __init__(self, root=HERE, threads=0):
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise SystemExit("manifest.json bulunamadi: %s" % manifest_path)
        with open(manifest_path) as fh:
            self.manifest = json.load(fh)

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads

        self.sessions = []
        for entry in self.manifest["models"]:
            path = os.path.join(root, entry["file"])
            if not os.path.exists(path):
                raise SystemExit("model dosyasi yok: %s" % path)
            self.sessions.append({
                "member": entry["member"],
                "session": ort.InferenceSession(
                    path, opts, providers=["CPUExecutionProvider"]),
            })

        self.member_weights = self.manifest.get("member_weights") or {}
        self.input_len = int(self.manifest.get("input_len", ep.TARGET_LEN))
        self.n_features = int(self.manifest.get("n_features", ep.N_FEATURES))
        # 37'den fazlaysa modeller QRST-artik olcumlerini de bekliyor demektir.
        self.extra_features = self.n_features > len(ep.FEATURE_NAMES)
        self.classes = self.manifest.get("classes", CLASSES)
        self.stacker = self.manifest.get("stacker")
        self.stacker_order = self.manifest.get("stacker_member_order")

    def __len__(self):
        return len(self.sessions)

    def predict_proba(self, signals, features, batch=32):
        """(n, 12, T) + (n, 37) -> (n, 5) ensemble probabilities."""
        signals = np.ascontiguousarray(signals, dtype=np.float32)
        features = np.ascontiguousarray(features, dtype=np.float32)
        n = signals.shape[0]

        per_member = {}
        for item in self.sessions:
            probs = np.zeros((n, len(self.classes)), dtype=np.float64)
            for start in range(0, n, batch):
                sl = slice(start, min(start + batch, n))
                logits = item["session"].run(
                    ["logits"], {"signal": signals[sl], "features": features[sl]})[0]
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs[sl] = e / e.sum(axis=1, keepdims=True)
            per_member.setdefault(item["member"], []).append(probs)

        names = list(per_member)
        mats = [_normalise(np.mean(per_member[name], axis=0)) for name in names]

        if self.stacker and self.stacker_order:
            order = [names.index(m) for m in self.stacker_order if m in names]
            if len(order) == len(names):
                feats = np.concatenate(
                    [np.log(np.clip(mats[i], 1e-9, 1.0)) for i in order], axis=1)
                coef = np.asarray(self.stacker["coef"], dtype=np.float64)
                intercept = np.asarray(self.stacker["intercept"], dtype=np.float64)
                logits = feats @ coef.T + intercept
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)

        weights = np.array([self.member_weights.get(name, 1.0) for name in names],
                           dtype=np.float64)
        weights = weights / (weights.sum() or 1.0)
        out = np.zeros_like(mats[0])
        for w, m in zip(weights, mats):
            out += w * m
        return _normalise(out)


def _normalise(prob):
    s = prob.sum(axis=1, keepdims=True)
    return prob / np.where(s < 1e-12, 1.0, s)


# --------------------------------------------------------------------------
# record loading
# --------------------------------------------------------------------------

def load_one(path, target_fs=None, extra=False):
    """Header path -> (signal_for_network, features).

    ``extra`` acikken 25 QRST-artik olcumu de eklenir. Bunlar **on islenmis**
    sinyalden hesaplanir (egitimde cache'in X.npy'sinden hesaplandigi gibi);
    ham sinyalden hesaplamak farkli sayilar uretir ve skoru sessizce dusurur.
    """
    sig, fs, _leads = wl.read_record(path)
    x = ep.preprocess_signal(sig, fs, target_fs=target_fs)
    f = ep.extract_features(sig, fs)
    if extra:
        if rf is None:
            raise SystemExit("manifest 37'den fazla ozellik istiyor ama "
                             "resid_features.py pakette yok")
        f = np.concatenate([np.asarray(f, dtype=np.float32),
                            rf.extract(x, target_fs or ep.TARGET_FS
                                       ).astype(np.float32)])
    return x, f


def scan_records(root):
    """Index every .hea under ``root`` by the spellings a CSV might use."""
    lookup = {}

    def add(key, path):
        if key:
            key = str(key).strip().strip("/\\").lower()
            lookup.setdefault(key, path)
            lookup.setdefault(key.replace("\\", "/"), path)

    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".hea"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            stem = os.path.splitext(fn)[0]
            parent = os.path.basename(dirpath)
            add(rel, path)
            add(os.path.splitext(rel)[0], path)
            add(stem, path)
            add(parent, path)
            add(os.path.join(parent, stem), path)
    return lookup


def read_batch_csv(csv_path, lookup):
    """Resolve a split CSV into (header_path, true_label_or_None) pairs."""
    with open(csv_path, newline="", errors="replace") as fh:
        rows = [r for r in csv.reader(fh) if any(str(c).strip() for c in r)]
    if not rows:
        return []

    header = [str(c).strip() for c in rows[0]]
    has_header = any(c and not c.replace(".", "").isdigit()
                     and normalise_label(c) is None for c in header)
    body = rows[1:] if has_header else rows
    n_col = max(len(r) for r in rows)

    def key_of(value):
        v = str(value).strip().strip("/\\").lower().replace("\\", "/")
        if v.endswith((".hea", ".dat", ".mat")):
            v = os.path.splitext(v)[0]
        return v

    best_col, best_hits = 0, -1
    for col in range(n_col):
        hits = sum(1 for r in body if col < len(r) and key_of(r[col]) in lookup)
        if hits > best_hits:
            best_col, best_hits = col, hits
    if best_hits <= 0:
        raise SystemExit("%s icindeki hicbir sutun %s altindaki kayitlarla "
                         "eslesmedi" % (csv_path, "kok"))

    label_col, label_hits = None, 0
    for col in range(n_col):
        if col == best_col:
            continue
        hits = sum(1 for r in body if col < len(r) and normalise_label(r[col]))
        if hits > label_hits:
            label_col, label_hits = col, hits
    if label_hits < 0.9 * len(body):
        label_col = None

    out, missing = [], 0
    for r in body:
        path = lookup.get(key_of(r[best_col])) if best_col < len(r) else None
        if path is None:
            missing += 1
            continue
        label = normalise_label(r[label_col]) if label_col is not None \
            and label_col < len(r) else None
        out.append((path, label))
    if missing:
        print("UYARI: %d satir diskte bulunamadi" % missing)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)), f1s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", nargs="?", help="a single .hea file")
    ap.add_argument("--batch", help="CSV listing records to score")
    ap.add_argument("--root", default=".", help="data root for --batch")
    ap.add_argument("--out", default="", help="write per-record predictions here")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--models", default=HERE, help="package directory")
    args = ap.parse_args(argv)

    if not args.record and not args.batch:
        ap.error("bir kayit yolu ver veya --batch kullan")

    bundle = Bundle(args.models, args.threads)
    target_fs = bundle.manifest.get("target_fs")
    print("model: %d ONNX grafigi, birlestirme=%s"
          % (len(bundle), bundle.manifest.get("combination", "flat")))

    # ---- single record ----
    if args.record:
        x, f = load_one(args.record, target_fs, bundle.extra_features)
        prob = bundle.predict_proba(x[None, ...], f[None, ...])[0]
        order = np.argsort(-prob)
        print("\n%s" % os.path.basename(args.record))
        print("  tahmin: %s  (%.1f%%)" % (bundle.classes[order[0]],
                                          100 * prob[order[0]]))
        for i in order:
            print("    %-8s %6.2f%%" % (bundle.classes[i], 100 * prob[i]))
        if not args.batch:
            return 0

    # ---- batch ----
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("kok dizin yok: %s" % root)

    print("\n%s taraniyor..." % root)
    lookup = scan_records(root)
    pairs = read_batch_csv(args.batch, lookup)
    if not pairs:
        raise SystemExit("%s icinden hicbir kayit cozulemedi" % args.batch)
    print("%d kayit islenecek" % len(pairs))

    signals = np.zeros((len(pairs), ep.N_LEADS, bundle.input_len), dtype=np.float32)
    features = np.zeros((len(pairs), bundle.n_features), dtype=np.float32)
    failed = []
    t0 = time.time()

    for i, (path, _label) in enumerate(pairs):
        try:
            x, f = load_one(path, target_fs, bundle.extra_features)
            signals[i], features[i] = x, f
        except Exception as exc:                # noqa: BLE001
            failed.append((path, str(exc)))
        if (i + 1) % 100 == 0:
            print("  on isleme %d/%d" % (i + 1, len(pairs)), flush=True)

    prep_sec = time.time() - t0
    t1 = time.time()
    prob = bundle.predict_proba(signals, features)
    infer_sec = time.time() - t1
    pred = prob.argmax(1)

    print("\non isleme %.1fs  cikarim %.1fs  toplam %.1fs  (%.1f ms/kayit)"
          % (prep_sec, infer_sec, prep_sec + infer_sec,
             1000 * (prep_sec + infer_sec) / len(pairs)))
    if failed:
        print("HATA: %d kayit okunamadi" % len(failed))
        for path, err in failed[:5]:
            print("  %s -> %s" % (os.path.basename(path), err))

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["record", "prediction"] + bundle.classes)
            for (path, _l), p, pr in zip(pairs, pred, prob):
                w.writerow([os.path.splitext(os.path.basename(path))[0],
                            bundle.classes[p]] + ["%.6f" % v for v in pr])
        print("yazildi: %s" % args.out)

    labels = [lbl for _p, lbl in pairs]
    if all(lbl is not None for lbl in labels):
        y_true = np.array([bundle.classes.index(lbl) for lbl in labels])
        f1, per_class = macro_f1(y_true, pred, len(bundle.classes))

        print("\n=== SKOR (%d kayit) ===" % len(pairs))
        print("  macro-F1: %.4f" % f1)
        for name, value in zip(bundle.classes, per_class):
            print("    %-8s %.4f" % (name, value))

        mask = np.isin(y_true, [1, 2])
        if np.any(mask):
            pair_prob = prob[mask][:, [1, 2]]
            pair_pred = np.where(pair_prob[:, 0] >= pair_prob[:, 1], 1, 2)
            print("  AFIB/AFL ikili ic dogruluk: %.4f"
                  % float(np.mean(pair_pred == y_true[mask])))

        print("\n  karisiklik matrisi (satir=gercek, sutun=tahmin)")
        print("           " + "".join("%8s" % c for c in bundle.classes))
        for i, name in enumerate(bundle.classes):
            row = [int(np.sum((y_true == i) & (pred == j)))
                   for j in range(len(bundle.classes))]
            print("  %-8s " % name + "".join("%8d" % v for v in row))

        recorded = (bundle.manifest.get("validation") or {}).get(
            "test_public_macro_f1_onnx")
        if recorded is not None:
            gap = abs(f1 - recorded)
            print("\n  manifest.json kayitli skor: %.4f  (fark %.4f)"
                  % (recorded, gap))
            if gap > 0.001:
                print("  UYARI: fark 0.001'i asiyor -- ayni kumeyi mi "
                      "skorluyorsun? Ayni ise on isleme uyusmuyor demektir.")
    else:
        print("\n(CSV'de etiket yok, skor hesaplanmadi)")
        counts = {c: int(np.sum(pred == i)) for i, c in enumerate(bundle.classes)}
        print("  tahmin dagilimi: %s"
              % "  ".join("%s=%d" % (k, v) for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
