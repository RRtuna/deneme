"""prep_fs -- rebuild X.npy at a different sample rate (FAZ 3).

    python prep_fs.py 250          # writes ./cache_250/X.npy
    python prep_fs.py 500 --out cache_500

Only X.npy is recomputed. y.npy, F.npy and index.csv are copied straight from
the 150 Hz cache, because the 37 features are always derived from the native
500 Hz signal and do not depend on the network's input rate. That keeps the
resolution comparison single-variable: the only thing that changes between
``cache`` and ``cache_250`` is the time resolution the network sees.

Memory note: at 500 Hz, X.npy is about 600 MB for 5000 records. It is written
through a memmap, so building it never holds more than one record in RAM; only
training needs the headroom.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fs", type=float, help="target sample rate in Hz, e.g. 250")
    ap.add_argument("--src", default="cache",
                    help="existing 150 Hz cache to take the record list from")
    ap.add_argument("--out", default=None,
                    help="output cache directory (default: cache_<fs>)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--allow-errors", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out or ("cache_%d" % int(round(args.fs)))
    src = args.src

    index_path = os.path.join(src, "index.csv")
    if not os.path.exists(index_path):
        raise SystemExit("%s not found -- run 'python prep.py' first so the "
                         "record list and F.npy exist" % index_path)

    rows = prep.load_index(src)
    entries = [{"path": r["path"], "label_name": r["label_name"],
                "split": r["split"]} for r in rows]
    missing = [e["path"] for e in entries if not os.path.exists(e["path"])]
    if missing:
        raise SystemExit("%d record(s) listed in %s no longer exist, e.g. %s"
                         % (len(missing), index_path, missing[0]))

    print("kaynak: %s (%d kayit)  hedef: %s @ %g Hz"
          % (src, len(entries), out_dir, args.fs))

    meta = prep.build_cache(root=".", out_dir=out_dir, target_fs=args.fs,
                            workers=args.workers or None, want_features=False,
                            reuse_index=entries)

    # F.npy is rate-independent by construction; reuse it rather than recompute.
    for name in ("F.npy",):
        src_file = os.path.join(src, name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(out_dir, name))
            print("kopyalandi: %s -> %s" % (src_file, out_dir))
        else:
            print("UYARI: %s yok, ozellik dali bu cache ile calismaz" % src_file)

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["derived_from"] = os.path.abspath(src)
    meta["has_features"] = os.path.exists(os.path.join(out_dir, "F.npy"))
    meta["features_source"] = "copied from %s (computed at native 500 Hz)" % src
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nhatali=%d" % meta["hatali"])
    print("kullanim: python train.py --preset <p> --tag res_%d --cache %s"
          % (int(round(args.fs)), out_dir))

    if meta["hatali"] and not args.allow_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
