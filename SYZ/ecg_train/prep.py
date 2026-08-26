"""prep -- build the training cache: raw records -> X.npy, F.npy, y.npy, index.csv.

    python prep.py                     # uses %ECG_ROOT%, writes ./cache
    python prep.py --root D:\...\SYZ --out cache --workers 8
    python prep.py --limit 50          # quick smoke test

A clean run prints ``hatali=0`` and the per-split record counts. Anything else
means records failed to load, and training on a broken cache wastes hours, so
the exit code is non-zero when failures occur unless --allow-errors is given.

Re-run this whenever ecg_preprocess.py changes. Training reads the cache, not
the raw records, and will otherwise keep using stale arrays without complaint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ecg_preprocess as ep
import wfdb_lite as wl

SPLIT_FILES = OrderedDict([
    ("train", "train.csv"),
    ("validation", "validation.csv"),
    ("test_public", "test_public.csv"),
])

# Folder / label spellings mapped onto the five canonical classes.
_LABEL_ALIASES = {
    "normal": "Normal", "norm": "Normal", "sr": "Normal", "sinus": "Normal",
    "sinusrhythm": "Normal", "nsr": "Normal", "0": "Normal",
    "afib": "AFIB", "af": "AFIB", "atrialfibrillation": "AFIB",
    "fibrillation": "AFIB", "1": "AFIB",
    "afl": "AFL", "aff": "AFL", "atrialflutter": "AFL", "flutter": "AFL",
    "2": "AFL",
    "lbbb": "LBBB", "clbbb": "LBBB", "leftbundlebranchblock": "LBBB",
    "3": "LBBB",
    "rbbb": "RBBB", "crbbb": "RBBB", "rightbundlebranchblock": "RBBB",
    "4": "RBBB",
}

CLASS_TO_INDEX = {name: i for i, name in enumerate(ep.CLASSES)}


def normalise_label(value):
    """Map a raw label or folder name onto a canonical class name, or None."""
    if value is None:
        return None
    key = str(value).strip()
    if not key or key.lower() in ("nan", "none"):
        return None
    key = key.replace(" ", "").replace("_", "").replace("-", "").lower()
    return _LABEL_ALIASES.get(key)


# --------------------------------------------------------------------------
# locating records on disk
# --------------------------------------------------------------------------

def scan_records(root):
    """Index every ``.hea`` under ``root`` by the keys a CSV might use.

    Returns ``(lookup, records)`` where ``lookup`` maps several spellings of a
    record id onto its header path, and ``records`` is the sorted list of paths.
    """
    root = os.path.abspath(root)
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".hea"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()

    lookup = {}

    def add(key, path):
        if not key:
            return
        key = str(key).strip().strip("/\\").lower()
        # First writer wins, so an exact relative path is never shadowed by a
        # bare basename collision discovered later.
        lookup.setdefault(key, path)
        lookup.setdefault(key.replace("\\", "/"), path)

    for path in paths:
        rel = os.path.relpath(path, root)
        stem = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(path))
        rel_noext = os.path.splitext(rel)[0]

        add(rel, path)
        add(rel_noext, path)
        add(stem, path)
        add(parent, path)
        add(os.path.join(parent, stem), path)
        add(rel_noext.split(os.sep, 1)[-1] if os.sep in rel_noext else rel_noext,
            path)
    return lookup, paths


def label_from_path(path, root):
    """Class implied by the first folder under the data root."""
    rel = os.path.relpath(path, os.path.abspath(root))
    parts = rel.replace("\\", "/").split("/")
    for part in parts[:-1]:
        name = normalise_label(part)
        if name:
            return name
    # Some layouts encode the class in the record id itself (NORM_000508).
    stem = os.path.splitext(parts[-1])[0]
    for token in stem.replace("-", "_").split("_"):
        name = normalise_label(token)
        if name:
            return name
    return None


def _read_csv_rows(path):
    """Read a CSV as a list of dicts, tolerating a missing header row."""
    import csv

    with open(path, "r", newline="", errors="replace") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(fh, dialect))

    rows = [r for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        return [], []

    header = [str(c).strip() for c in rows[0]]
    # A header whose cells look like data (no letters-only field, or the row
    # count matches a known total) means the file has no header line.
    looks_like_header = any(
        c and not c.replace(".", "").replace("-", "").isdigit() and
        normalise_label(c) is None for c in header)
    if looks_like_header:
        body = rows[1:]
        names = [c if c else "col%d" % i for i, c in enumerate(header)]
    else:
        body = rows
        names = ["col%d" % i for i in range(len(header))]

    out = []
    for r in body:
        r = list(r) + [""] * (len(names) - len(r))
        out.append({names[i]: str(r[i]).strip() for i in range(len(names))})
    return out, names


def resolve_split(csv_path, lookup, root):
    """Map one split CSV onto ``(header_path, class_name)`` pairs.

    The record and label columns are found by scoring every column against the
    on-disk index rather than by assuming column names, because the CSV layout
    is not documented anywhere in the task.
    """
    rows, names = _read_csv_rows(csv_path)
    if not rows:
        return [], {}

    def key_of(value):
        v = str(value).strip().strip("/\\").lower().replace("\\", "/")
        if v.endswith(".hea") or v.endswith(".dat") or v.endswith(".mat"):
            v = os.path.splitext(v)[0]
        return v

    # --- record column: the one whose values most often hit the index ---
    best_col, best_hits = None, 0
    for name in names:
        hits = sum(1 for r in rows if key_of(r.get(name, "")) in lookup)
        if hits > best_hits:
            best_col, best_hits = name, hits
    if best_col is None or best_hits == 0:
        raise SystemExit(
            "%s: no column matches any record on disk under %s.\n"
            "  columns seen: %s\n  first row: %s"
            % (csv_path, root, names, rows[0]))

    # --- label column: the one whose values map onto class names ---
    label_col, label_hits = None, 0
    for name in names:
        if name == best_col:
            continue
        hits = sum(1 for r in rows if normalise_label(r.get(name, "")))
        if hits > label_hits:
            label_col, label_hits = name, hits
    if label_hits < 0.9 * len(rows):
        label_col = None                        # fall back to the folder name

    pairs, unresolved = [], 0
    for r in rows:
        key = key_of(r.get(best_col, ""))
        path = lookup.get(key)
        if path is None:
            unresolved += 1
            continue
        name = normalise_label(r.get(label_col, "")) if label_col else None
        if name is None:
            name = label_from_path(path, root)
        pairs.append((path, name))

    info = {
        "csv": os.path.basename(csv_path),
        "rows": len(rows),
        "record_column": best_col,
        "record_column_hits": best_hits,
        "label_column": label_col,
        "unresolved_rows": unresolved,
    }
    return pairs, info


# --------------------------------------------------------------------------
# per-record work
# --------------------------------------------------------------------------

_WORKER = {}


def _init_worker(target_fs, want_features):
    _WORKER["target_fs"] = target_fs
    _WORKER["features"] = want_features


def _process(job):
    """Load one record and return ``(idx, X, F, error)``."""
    idx, path = job
    try:
        sig, fs, _leads = wl.read_record(path)
        if sig.shape[-1] < 8:
            raise wl.WFDBError("record has %d samples" % sig.shape[-1])
        x = ep.preprocess_signal(sig, fs, target_fs=_WORKER["target_fs"])
        f = ep.extract_features(sig, fs) if _WORKER["features"] else None
        return idx, x, f, None
    except Exception as exc:                    # noqa: BLE001
        return idx, None, None, "%s: %s" % (type(exc).__name__, exc)


def build_cache(root, out_dir, target_fs=None, workers=None, limit=0,
                want_features=True, reuse_index=None, quiet=False):
    """Build the cache and return the meta dict that was written."""
    root = os.path.abspath(root)
    target_fs = ep.TARGET_FS if target_fs is None else float(target_fs)
    os.makedirs(out_dir, exist_ok=True)

    if reuse_index is not None:
        entries = reuse_index
        split_info = [{"csv": "(reused index.csv)", "rows": len(entries)}]
    else:
        if not os.path.isdir(root):
            raise SystemExit("data root not found: %s\n"
                             "Set ECG_ROOT or pass --root." % root)
        lookup, all_paths = scan_records(root)
        if not all_paths:
            raise SystemExit("no .hea files found under %s" % root)
        if not quiet:
            print("taranan kayit (.hea): %d" % len(all_paths))

        entries, split_info, seen = [], [], set()
        for split, fname in SPLIT_FILES.items():
            csv_path = os.path.join(root, fname)
            if not os.path.exists(csv_path):
                print("UYARI: %s bulunamadi, bu split atlandi" % csv_path)
                continue
            pairs, info = resolve_split(csv_path, lookup, root)
            info["split"] = split
            split_info.append(info)
            for path, name in pairs:
                if path in seen:
                    continue                    # a record belongs to one split
                seen.add(path)
                entries.append({"path": path, "label_name": name,
                                "split": split})

        if not entries:
            raise SystemExit("no records resolved from the split CSVs")

    if limit:
        entries = entries[:limit]

    n = len(entries)
    target_len = int(round(target_fs * ep.TARGET_SECONDS))
    x_path = os.path.join(out_dir, "X.npy")
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                  shape=(n, ep.N_LEADS, target_len))
    F = np.zeros((n, ep.N_FEATURES), dtype=np.float32) if want_features else None

    jobs = [(i, e["path"]) for i, e in enumerate(entries)]
    workers = workers or min(os.cpu_count() or 1, 8)
    errors = []
    t0 = time.time()

    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(target_fs, want_features)) as pool:
            for done, (idx, x, f, err) in enumerate(
                    pool.imap_unordered(_process, jobs, chunksize=8), 1):
                if err is None:
                    X[idx] = x
                    if want_features:
                        F[idx] = f
                else:
                    errors.append((entries[idx]["path"], err))
                if not quiet and (done % 250 == 0 or done == n):
                    rate = done / max(time.time() - t0, 1e-6)
                    print("  %d/%d  (%.1f kayit/s)" % (done, n, rate), flush=True)
    else:
        _init_worker(target_fs, want_features)
        for done, job in enumerate(jobs, 1):
            idx, x, f, err = _process(job)
            if err is None:
                X[idx] = x
                if want_features:
                    F[idx] = f
            else:
                errors.append((entries[idx]["path"], err))
            if not quiet and (done % 250 == 0 or done == n):
                print("  %d/%d" % (done, n), flush=True)

    X.flush()
    del X

    bad = {p for p, _ in errors}
    y = np.array([CLASS_TO_INDEX.get(e["label_name"], -1) for e in entries],
                 dtype=np.int64)

    np.save(os.path.join(out_dir, "y.npy"), y)
    if want_features:
        np.save(os.path.join(out_dir, "F.npy"), F)

    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as fh:
        import csv

        w = csv.writer(fh)
        w.writerow(["idx", "record", "path", "label", "label_name", "split", "ok"])
        for i, e in enumerate(entries):
            w.writerow([i, os.path.splitext(os.path.basename(e["path"]))[0],
                        e["path"], int(y[i]), e["label_name"] or "",
                        e["split"], int(e["path"] not in bad)])

    split_counts = Counter(e["split"] for e in entries)
    class_counts = Counter(e["label_name"] for e in entries)
    per_split_class = {}
    for e in entries:
        per_split_class.setdefault(e["split"], Counter())[e["label_name"]] += 1

    meta = {
        "root": root,
        "out_dir": os.path.abspath(out_dir),
        "n_records": n,
        "hatali": len(errors),
        "target_fs": target_fs,
        "target_len": target_len,
        "has_features": bool(want_features),
        "split_counts": dict(split_counts),
        "class_counts": {k or "?": v for k, v in class_counts.items()},
        "per_split_class": {s: {k or "?": v for k, v in c.items()}
                            for s, c in per_split_class.items()},
        "unlabelled": int((y < 0).sum()),
        "split_info": split_info,
        "preprocess": ep.preprocess_config(),
        "elapsed_sec": round(time.time() - t0, 1),
        "errors": [{"path": p, "error": e} for p, e in errors[:50]],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if not quiet:
        print()
        print("hatali=%d" % len(errors))
        print("kayit=%d  sure=%.1fs  cache=%s" % (n, meta["elapsed_sec"], out_dir))
        for split in SPLIT_FILES:
            if split in per_split_class:
                counts = per_split_class[split]
                detail = "  ".join("%s=%d" % (c, counts.get(c, 0))
                                   for c in ep.CLASSES)
                print("  %-12s %4d   %s" % (split, split_counts[split], detail))
        if meta["unlabelled"]:
            print("UYARI: %d kaydin etiketi cozulemedi (label=-1)"
                  % meta["unlabelled"])
        for path, err in errors[:10]:
            print("  HATA %s -> %s" % (os.path.basename(path), err))
        if len(errors) > 10:
            print("  ... %d hata daha (meta.json'da)" % (len(errors) - 10))
    return meta


def load_index(cache_dir):
    """Read back index.csv as a list of dicts."""
    import csv

    path = os.path.join(cache_dir, "index.csv")
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ECG_ROOT", "."),
                    help="data root holding the class folders and split CSVs")
    ap.add_argument("--out", default="cache", help="cache directory to write")
    ap.add_argument("--fs", type=float, default=None,
                    help="target sample rate (default: ecg_preprocess.TARGET_FS)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-features", action="store_true",
                    help="skip the 37 features (used by prep_fs.py)")
    ap.add_argument("--allow-errors", action="store_true",
                    help="exit 0 even when records fail to load")
    args = ap.parse_args(argv)

    meta = build_cache(args.root, args.out, target_fs=args.fs,
                       workers=args.workers or None, limit=args.limit,
                       want_features=not args.no_features)

    if meta["hatali"] and not args.allow_errors:
        print("\nhatali!=0 -- bozuk veriyle egitim bosa gider, once nedenini bul.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
