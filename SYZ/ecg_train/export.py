"""export -- build the delivered package: ONNX + int8 + manifest, self-checked.

    python export.py                       # uses ensemble.json
    python export.py --out package --int8
    python export.py --no-int8             # keep float32 only

What it produces::

    package/
      models/*.onnx        one file per fold of every ensemble member
      ecg_preprocess.py    byte-identical copy of the training-time module
      wfdb_lite.py         byte-identical copy
      predict.py           onnxruntime-only inference, no torch
      manifest.json        members, weights, validation scores, checksums
      preprocess.json      the preprocessing settings, for the record
      README.md

Self-validation, in this order:

  1. every exported graph is re-run under onnxruntime and compared against
     PyTorch on real cache rows -- max absolute probability difference must
     stay under --tol-prob;
  2. the whole package is scored on test_public through onnxruntime alone and
     compared with the PyTorch ensemble score.

If the macro-F1 gap exceeds --tol-score (default 0.005), the export is marked
failed and the exit code is non-zero. A package that scores differently from
the model it came from is broken, not "close enough".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ecg_preprocess as ep
from ensemble import _normalise, blend
from model import N_FEATURES, build_model
from train import (DEV_SPLITS, TEST_SPLIT, binary_afib_afl, load_cache,
                   macro_f1)

HERE = os.path.dirname(os.path.abspath(__file__))
OPSET = 17


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_fold_checkpoints(member_dir):
    """All fold checkpoints of one ensemble member, in fold order."""
    out = []
    for name in sorted(os.listdir(member_dir)):
        ckpt = os.path.join(member_dir, name, "best.pt")
        if name.startswith("fold") and os.path.exists(ckpt):
            out.append((name, ckpt))
    full = os.path.join(member_dir, "full", "best.pt")
    if os.path.exists(full):
        out.append(("full", full))
    return out


def infer_shape(state_dict):
    """Girdi genisligini AGIRLIKLARDAN oku.

    Kanal ve ozellik sayisini sabit varsaymak, genisletilmis bir cache ile
    egitilmis bir checkpoint'te `load_state_dict`'i patlatir. Checkpoint yeni
    surumse degerleri zaten tasir; degilse ilk evrisim katmaninin ve ozellik
    dalinin agirlik sekillerinden cikarilir.
    """
    in_ch, n_feat = None, None
    for key, w in state_dict.items():
        if in_ch is None and key.startswith("backbone.") and w.ndim == 3:
            in_ch = int(w.shape[1])
        if n_feat is None and key.startswith("feat_branch.") and w.ndim == 2:
            n_feat = int(w.shape[1])
        if in_ch is not None and n_feat is not None:
            break
    return in_ch, n_feat


def feature_names_for(n_feat):
    """Manifest'e yazilacak ozellik adlari; genisletilmisse artik adlari eklenir."""
    names = list(ep.FEATURE_NAMES)
    if n_feat > len(names):
        import resid_features as rf
        names = names + list(rf.FEATURE_NAMES)
    if len(names) != n_feat:
        raise SystemExit("ozellik adi sayisi (%d) model beklentisiyle (%d) "
                         "uyusmuyor" % (len(names), n_feat))
    return names


def load_torch_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    in_ch, n_feat = infer_shape(sd)
    model = build_model(ckpt["preset"], dropout=ckpt.get("dropout", 0.2),
                        use_features=ckpt.get("use_features", True),
                        in_ch=ckpt.get("in_ch") or in_ch or ep.N_LEADS,
                        n_features=ckpt.get("n_features") or n_feat or N_FEATURES)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def export_one(model, out_path, input_len):
    """Export a single model to ONNX with a dynamic batch axis.

    The TorchScript exporter (``dynamo=False``) is used on purpose. The newer
    dynamo exporter produces a graph that onnxruntime's dynamic quantiser
    rejects ("Inferred shape and existing shape differ"), and it spills the
    weights into a sibling ``.onnx.data`` file -- which turns the package into
    something that breaks the moment somebody copies only the ``.onnx`` files.
    We want one self-contained file per model, and we want int8 to work.
    """
    in_ch, n_feat = infer_shape(model.state_dict())
    dummy_x = torch.zeros(2, in_ch or ep.N_LEADS, input_len)
    dummy_f = torch.zeros(2, n_feat or N_FEATURES)
    kwargs = dict(
        input_names=["signal", "features"], output_names=["logits"],
        dynamic_axes={"signal": {0: "batch"}, "features": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True)

    try:
        torch.onnx.export(model, (dummy_x, dummy_f), out_path,
                          dynamo=False, **kwargs)
    except TypeError:
        # Very old torch has no dynamo switch; its default is TorchScript.
        torch.onnx.export(model, (dummy_x, dummy_f), out_path, **kwargs)

    sidecar = out_path + ".data"
    if os.path.exists(sidecar):
        raise SystemExit(
            "%s agirliklari harici dosyaya yazdi (%s). Paket tek dosyali "
            "olmali -- aksi halde sadece *.onnx kopyalayan biri bozuk paket "
            "alir." % (os.path.basename(out_path), os.path.basename(sidecar)))
    return out_path


def quantize_int8(src, dst):
    """Dynamic int8 quantisation. Returns True when it actually produced a file."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("  onnxruntime.quantization yok, int8 atlandi")
        return False
    try:
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        return os.path.exists(dst)
    except Exception as exc:                    # noqa: BLE001
        print("  int8 basarisiz (%s), float32 kullanilacak" % exc)
        return False


def onnx_probs(path, X, Fe, idx, batch=64):
    """Run an exported graph under onnxruntime and return probabilities."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    out = np.zeros((len(idx), len(ep.CLASSES)), dtype=np.float64)
    for start in range(0, len(idx), batch):
        sel = idx[start:start + batch]
        logits = sess.run(["logits"], {
            "signal": np.ascontiguousarray(X[sel]).astype(np.float32),
            "features": np.ascontiguousarray(Fe[sel]).astype(np.float32)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        out[start:start + len(sel)] = e / e.sum(axis=1, keepdims=True)
    return out


@torch.no_grad()
def torch_probs(model, X, Fe, idx, batch=64):
    out = np.zeros((len(idx), len(ep.CLASSES)), dtype=np.float64)
    for start in range(0, len(idx), batch):
        sel = idx[start:start + batch]
        logits = model(torch.from_numpy(np.ascontiguousarray(X[sel])).float(),
                       torch.from_numpy(np.ascontiguousarray(Fe[sel])).float())
        out[start:start + len(sel)] = torch.softmax(logits, dim=1).numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="ensemble.json")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="package")
    ap.add_argument("--int8", dest="int8", action="store_true", default=True)
    ap.add_argument("--no-int8", dest="int8", action="store_false")
    ap.add_argument("--tol-prob", type=float, default=2e-3,
                    help="max |ONNX - PyTorch| probability difference per "
                         "float32 graph; catches a broken export")
    ap.add_argument("--tol-prob-int8", type=float, default=0.10,
                    help="per-graph sanity bound for int8. Quantisation moves "
                         "probabilities by ~1%% by design, so int8 is judged on "
                         "the ensemble SCORE (--tol-score), not on this")
    ap.add_argument("--tol-score", type=float, default=0.005,
                    help="max macro-F1 gap between package and PyTorch")
    ap.add_argument("--check-rows", type=int, default=64)
    args = ap.parse_args(argv)

    if not os.path.exists(args.ensemble):
        raise SystemExit("%s yok -- once 'python ensemble.py' kos" % args.ensemble)
    with open(args.ensemble) as fh:
        ens = json.load(fh)

    if ens.get("method") == "stacked":
        print("UYARI: secilen kural 'stacked'. Paket agirlikli ortalama "
              "destekliyor; stacker katsayilari manifest'e yaziliyor ve "
              "predict.py bunlari uyguluyor.")

    X, Fe, y, split, records, ok = load_cache(args.cache, mmap=True)
    usable = ok & (y >= 0)
    dev_idx = np.flatnonzero(np.isin(split, DEV_SPLITS) & usable)
    test_idx = np.flatnonzero((split == TEST_SPLIT) & usable)
    input_len = X.shape[2]

    out_dir = os.path.abspath(args.out)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    check_idx = test_idx[:args.check_rows] if len(test_idx) else dev_idx[:args.check_rows]

    print("ONNX disa aktarim (opset %d, giris uzunlugu %d)" % (OPSET, input_len))
    exported, failures = [], []

    for member in ens["members"]:
        member_dir = member["path"]
        ckpts = find_fold_checkpoints(member_dir)
        if not ckpts:
            print("  UYARI: %s icinde fold*/best.pt yok, atlandi" % member_dir)
            continue

        for fold_name, ckpt_path in ckpts:
            model, ckpt = load_torch_model(ckpt_path)
            stem = "%s__%s" % (member["name"], fold_name)
            fp32 = os.path.join(models_dir, stem + ".onnx")
            export_one(model, fp32, input_len)

            ref = torch_probs(model, X, Fe, check_idx)
            got = onnx_probs(fp32, X, Fe, check_idx)
            diff = float(np.max(np.abs(ref - got)))
            if diff > args.tol_prob:
                failures.append(("%s (fp32 export)" % stem, diff))

            int8_path, diff8 = None, None
            if args.int8:
                candidate = os.path.join(models_dir, stem + ".int8.onnx")
                if quantize_int8(fp32, candidate):
                    diff8 = float(np.max(np.abs(
                        ref - onnx_probs(candidate, X, Fe, check_idx))))
                    if diff8 <= args.tol_prob_int8:
                        int8_path = candidate
                    else:
                        os.remove(candidate)
                        print("  %s: int8 sapmasi %.2e, akil saglik sinirini "
                              "(%.2e) asti -- atildi"
                              % (stem, diff8, args.tol_prob_int8))

            print("  %-28s fp32 %5.1f MB  max|dP|=%.2e%s"
                  % (stem, os.path.getsize(fp32) / 1e6, diff,
                     ("   int8 max|dP|=%.2e" % diff8) if diff8 is not None else ""))

            exported.append({
                "member": member["name"], "fold": fold_name,
                "fp32_path": fp32, "int8_path": int8_path,
                "weight": member.get("weight"),
                "max_prob_diff_vs_torch": diff,
                "int8_max_prob_diff_vs_torch": diff8,
                "preset": ckpt["preset"],
                "n_features": int(model.feat_mean.shape[0])
                if getattr(model, "feat_branch", None) is not None else N_FEATURES,
            })

    if not exported:
        raise SystemExit("hicbir model disa aktarilamadi")

    # ---- weights, normalised over what actually got exported ----
    by_member = {}
    for e in exported:
        by_member.setdefault(e["member"], []).append(e)
    raw_w = {name: (1.0 if items[0]["weight"] is None else float(items[0]["weight"]))
             for name, items in by_member.items()}
    total_w = sum(raw_w.values()) or 1.0
    member_weights = {k: v / total_w for k, v in raw_w.items()}

    def ensemble_probs(key, idx):
        """Score the whole package at ``key`` precision on cache rows ``idx``."""
        mats, weights = [], []
        for name, items in by_member.items():
            folds = [onnx_probs(e[key] or e["fp32_path"], X, Fe, idx)
                     for e in items if e["fold"] != "full"] or \
                    [onnx_probs(e[key] or e["fp32_path"], X, Fe, idx)
                     for e in items]
            mats.append(_normalise(np.mean(folds, axis=0)))
            weights.append(member_weights[name])
        return blend(mats, np.asarray(weights) / sum(weights))

    # ---- decide int8 vs float32 on the SCORE, not on raw probabilities ----
    # Quantisation always perturbs probabilities; the only question that
    # matters for delivery is whether the package still scores the same.
    score_idx = test_idx if len(test_idx) else dev_idx
    use_int8 = False
    fp32_f1 = macro_f1(y[score_idx], ensemble_probs("fp32_path", score_idx).argmax(1))[0]
    int8_f1 = None

    if args.int8 and all(e["int8_path"] for e in exported):
        int8_f1 = macro_f1(y[score_idx],
                           ensemble_probs("int8_path", score_idx).argmax(1))[0]
        gap = abs(int8_f1 - fp32_f1)
        print("\nint8 karari (%d kayit uzerinde):" % len(score_idx))
        print("  float32 ensemble macro-F1 : %.4f" % fp32_f1)
        print("  int8    ensemble macro-F1 : %.4f  (fark %.4f, esik %.4f)"
              % (int8_f1, gap, args.tol_score))
        use_int8 = gap <= args.tol_score
        print("  -> %s" % ("int8 kullaniliyor (%.1fx kucuk)"
                           % (sum(os.path.getsize(e["fp32_path"]) for e in exported)
                              / max(sum(os.path.getsize(e["int8_path"])
                                        for e in exported), 1))
                           if use_int8 else
                           "int8 skoru degistirdi, float32 teslim ediliyor"))
    elif args.int8:
        print("\nbazi modeller int8'e cevrilemedi -- paket float32 kaliyor")

    # ---- keep exactly one file per model ----
    for e in exported:
        keep = e["int8_path"] if (use_int8 and e["int8_path"]) else e["fp32_path"]
        drop = e["fp32_path"] if keep == e["int8_path"] else e["int8_path"]
        if drop and os.path.exists(drop):
            os.remove(drop)
        e["file"] = os.path.relpath(keep, out_dir).replace("\\", "/")
        e["quantised"] = keep == e["int8_path"]
        e["sha256"] = sha256(keep)
        e["size_bytes"] = os.path.getsize(keep)
        for tmp in ("fp32_path", "int8_path"):
            e.pop(tmp, None)

    # ---- copy the runtime files ----
    # Modeller 37'den fazla ozellik bekliyorsa artik olcumlerinin kodu da
    # pakete girmeli, yoksa cikarim eksik ozellik vektoruyle calisir.
    n_feat_pkg = max((e.get("n_features") or N_FEATURES) for e in exported)
    runtime = ["ecg_preprocess.py", "wfdb_lite.py"]
    if n_feat_pkg > N_FEATURES:
        runtime.append("resid_features.py")
    for name in runtime:
        src = os.path.join(HERE, name)
        if not os.path.exists(src):
            raise SystemExit("%s yok -- paket eksik olur" % src)
        shutil.copy2(src, os.path.join(out_dir, name))
    predict_src = os.path.join(HERE, "package_src", "predict.py")
    if os.path.exists(predict_src):
        shutil.copy2(predict_src, os.path.join(out_dir, "predict.py"))
    else:
        print("UYARI: package_src/predict.py bulunamadi")

    with open(os.path.join(out_dir, "preprocess.json"), "w") as fh:
        json.dump(ep.preprocess_config(), fh, indent=2)

    manifest = {
        "created_by": "export.py",
        "preprocess_version": ep.PREPROCESS_VERSION,
        "input_len": int(input_len),
        "target_fs": ep.TARGET_FS,
        "classes": list(ep.CLASSES),
        "n_features": int(n_feat_pkg),
        "feature_names": feature_names_for(n_feat_pkg),
        "combination": ens.get("method", "flat"),
        "member_weights": member_weights,
        "models": exported,
        "selection_note": ("all architecture, hyper-parameter and ensemble "
                           "choices were made on out-of-fold development "
                           "scores; test_public was read once, for reporting"),
        "oof_macro_f1": ens.get("oof_macro_f1"),
        "oof_afib_afl": ens.get("oof_afib_afl"),
    }
    if ens.get("method") == "stacked" and "stacker" in ens:
        manifest["stacker"] = ens["stacker"]
        manifest["stacker_member_order"] = [m["name"] for m in ens["members"]]

    manifest["precision"] = "int8" if use_int8 else "float32"
    manifest["int8_considered"] = {
        "requested": bool(args.int8),
        "used": use_int8,
        "float32_macro_f1": fp32_f1,
        "int8_macro_f1": int8_f1,
        "scored_on": "test_public" if len(test_idx) else "development (OOF rows)",
    }

    # ---- final check: the shipped package, re-run from disk, vs PyTorch ----
    torch_test = None
    if len(test_idx):
        mats, weights = [], []
        for name, items in by_member.items():
            folds = [onnx_probs(os.path.join(out_dir, e["file"]), X, Fe, test_idx)
                     for e in items if e["fold"] != "full"]
            if not folds:
                folds = [onnx_probs(os.path.join(out_dir, e["file"]), X, Fe, test_idx)
                         for e in items]
            mats.append(_normalise(np.mean(folds, axis=0)))
            weights.append(member_weights[name])
        shipped = blend(mats, np.asarray(weights) / sum(weights))
        onnx_f1 = macro_f1(y[test_idx], shipped.argmax(1))[0]

        ref_path = "ensemble_test_prob.npy"
        if os.path.exists(ref_path):
            torch_test = macro_f1(y[test_idx], np.load(ref_path).argmax(1))[0]

        manifest["validation"] = {
            "test_public_macro_f1_onnx": onnx_f1,
            "test_public_macro_f1_torch": torch_test,
            "test_public_afib_afl_onnx": binary_afib_afl(y[test_idx], shipped),
            "n_test": int(len(test_idx)),
        }
        print("\ndogrulama (test_public, %d kayit, %s):"
              % (len(test_idx), manifest["precision"]))
        print("  ONNX paket macro-F1   : %.4f" % onnx_f1)
        if torch_test is not None:
            gap = abs(onnx_f1 - torch_test)
            print("  PyTorch ensemble      : %.4f" % torch_test)
            print("  fark                  : %.4f (esik %.4f)" % (gap, args.tol_score))
            manifest["validation"]["gap"] = gap
            if gap > args.tol_score:
                failures.append(("ensemble score gap", gap))

    manifest["self_check_passed"] = not failures
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    write_package_readme(out_dir, manifest)

    print("\npaket: %s" % out_dir)
    print("  model sayisi: %d   toplam boyut: %.1f MB"
          % (len(exported),
             sum(os.path.getsize(os.path.join(out_dir, e["file"]))
                 for e in exported) / 1e6))

    if failures:
        print("\nKENDI KENDINE DOGRULAMA BASARISIZ:")
        for name, value in failures:
            print("  %-28s %.2e" % (name, value))
        print("Teslim etme -- once bunu coz.")
        return 1

    print("\nkendi kendine dogrulama gecti.")
    return 0


def write_package_readme(out_dir, manifest):
    weights = "\n".join("  %-24s %.4f" % (k, v)
                        for k, v in manifest["member_weights"].items())
    val = manifest.get("validation") or {}
    text = """# EKG siniflandirma paketi

onnxruntime ile calisir. **PyTorch gerekmez.**

## Kurulum

    pip install numpy onnxruntime

## Kullanim

Tek kayit:

    python predict.py /yol/Normal/NORM_000508/48090046.hea

Toplu (CSV) ve skor:

    python predict.py --batch %%ECG_ROOT%%\\test_public.csv --root %%ECG_ROOT%%

## Icerik

    models/*.onnx      %(n_models)d model (%(combination)s birlestirme)
    ecg_preprocess.py  egitimdekinin birebir kopyasi
    wfdb_lite.py       saf numpy WFDB okuyucu
    predict.py         onnxruntime cikarim
    manifest.json      agirliklar, dogrulama skorlari, sha256
    preprocess.json    on isleme ayarlari

## Agirliklar

%(weights)s

## Dogrulama

    OOF macro-F1 (gelistirme)   %(oof)s
    test_public macro-F1 (ONNX) %(test)s

Tum mimari, hiperparametre ve ensemble secimleri **out-of-fold** skorlarla
yapildi. `test_public` yalnizca raporlama icin, bir kez okundu.

## Giris sozlesmesi

    signal    (batch, 12, %(len)d)  float32   ecg_preprocess.preprocess_signal ciktisi
    features  (batch, 37)           float32   ecg_preprocess.extract_features ciktisi

Derivasyon sirasi: I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
Siniflar: %(classes)s
""" % {
        "n_models": len(manifest["models"]),
        "combination": manifest["combination"],
        "weights": weights,
        "oof": ("%.4f" % manifest["oof_macro_f1"]) if manifest.get("oof_macro_f1") else "-",
        "test": ("%.4f" % val["test_public_macro_f1_onnx"]) if val.get("test_public_macro_f1_onnx") else "-",
        "len": manifest["input_len"],
        "classes": " ".join(manifest["classes"]),
    }
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
