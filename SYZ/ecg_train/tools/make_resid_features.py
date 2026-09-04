"""make_resid_features -- mevcut cache'in F.npy'sine 20 bant-artik olcumu ekler.

    python tools/make_resid_features.py --in cache --out cache_f57

X.npy'ye **dokunmaz** (sembolik baglanti ya da kopya), yalnizca F.npy 37'den
57'ye genisler. `ecg_preprocess.py` degismez, on isleme davranisi ayni kalir.

Neden bu yol, artik KANALI eklemekten daha ucuz
-----------------------------------------------
Artik kanali eklemek girdiyi 12'den 16'ya cikarir; ag yeniden egitilmek
zorundadir ve ONNX girdi sekli degisir. Ozellik yolu ise yalnizca ozellik
dalinin ilk katmanini genisletir: girdi sekli ayni kalir, cikarimda ek maliyet
kayit basina ~4 ms'dir (mevcut 660 ms on islemenin yaninda gorunmez).

Olcumler `resid_probe.py` ile bire bir aynidir; orada OOF uzerinde
dogrulandiktan sonra buraya tasinir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resid_probe import (RESID_LEADS, STAT_NAMES, lead_indices,  # noqa: E402
                         record_features, _RPEAK_MODE)

NEW_NAMES = tuple("resid_%s_%s" % (lead.lower(), stat)
                  for lead in RESID_LEADS for stat in STAT_NAMES)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="cache")
    ap.add_argument("--out", dest="dst", default="cache_f57")
    ap.add_argument("--fs", type=float, default=0.0)
    ap.add_argument("--link", action="store_true",
                    help="X.npy'yi kopyalamak yerine sembolik bagla (yer kazanir)")
    args = ap.parse_args(argv)

    x_path = os.path.join(args.src, "X.npy")
    f_path = os.path.join(args.src, "F.npy")
    idx_path = os.path.join(args.src, "index.csv")
    for p in (x_path, f_path, idx_path):
        if not os.path.exists(p):
            raise SystemExit("%s yok" % p)

    fs = args.fs
    meta_path = os.path.join(args.src, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    fs = fs or float(meta.get("target_fs") or 0.0)
    if not fs:
        raise SystemExit("ornekleme hizi bilinmiyor; --fs ver")

    with open(idx_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.load(x_path, mmap_mode="r")
    F = np.load(f_path)
    if X.shape[0] != len(rows) or F.shape[0] != len(rows):
        raise SystemExit("satir sayilari uyusmuyor")

    lead_idx = lead_indices()
    print("kaynak: %s  (%d kayit, %.0f Hz, %d ozellik)"
          % (args.src, len(rows), fs, F.shape[1]))
    print("eklenen: %d olcum -> %d" % (len(NEW_NAMES), F.shape[1] + len(NEW_NAMES)))

    t0 = time.time()
    extra = np.zeros((len(rows), len(NEW_NAMES)), dtype=np.float32)
    for i in range(len(rows)):
        extra[i] = record_features(np.asarray(X[i], dtype=np.float64), fs, lead_idx)
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            print("  %5d/%d  %.0f sn" % (i + 1, len(rows), time.time() - t0),
                  flush=True)
    print("R tepeleri: %s" % _RPEAK_MODE[0])

    os.makedirs(args.dst, exist_ok=True)
    np.save(os.path.join(args.dst, "F.npy"),
            np.concatenate([F.astype(np.float32), extra], axis=1))
    for name in ("y.npy", "index.csv"):
        shutil.copy2(os.path.join(args.src, name), os.path.join(args.dst, name))

    dst_x = os.path.join(args.dst, "X.npy")
    if os.path.exists(dst_x) or os.path.islink(dst_x):
        os.remove(dst_x)
    if args.link:
        os.symlink(os.path.abspath(x_path), dst_x)
    else:
        shutil.copy2(x_path, dst_x)

    meta = dict(meta)
    meta["resid_features"] = list(NEW_NAMES)
    meta["resid_from"] = os.path.abspath(args.src)
    meta["n_features_total"] = int(F.shape[1] + len(NEW_NAMES))
    with open(os.path.join(args.dst, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print()
    print("yazildi: %s  (%.0f sn)" % (args.dst, time.time() - t0))
    print("sonraki: python train.py --cache %s --tag f57 ..." % args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
