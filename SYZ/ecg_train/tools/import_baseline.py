"""import_baseline -- mevcut modellerinin OOF/test matrislerini bu cache'e tasi.

    python tools/import_baseline.py ^
        --name r18_feat ^
        --old-index ..\eski_ecg_train\cache\index.csv ^
        --oof       ..\eski_ecg_train\runs\r18_feat\oof_prob.npy ^
        --test      ..\eski_ecg_train\runs\r18_feat\test_prob.npy

Neden gerekli
-------------
`oof_prob.npy` bir olasilik matrisidir ve satirlarinin **hangi kayda ait
oldugunu kendisi bilmez** -- bu bilgi, o matrisi ureten cache'in index.csv
sirasindadir. Bu depodaki `prep.py` kayitlari kendi sirasiyla diziyor ve bu
sira senin eski cache'ininkiyle neredeyse kesinlikle ayni degil.

Satirlari kaymis iki matrisi `ensemble.py`'ye verirsen hata almazsin: makul
gorunen ama tamamen anlamsiz bir OOF skoru alirsin. Bu script kaymayi kayit
adindan eslestirerek onler ve eslesmeyen her satiri raporlar.

Ne yapar
--------
1. Eski index'ten "hangi satir hangi kayit" listesini cikarir (sutun adini
   varsaymaz, yeni index'teki kayit adlariyla en cok eslesen sutunu secer).
2. Yeni `cache/index.csv` icin bir permutasyon kurar.
3. `oof_prob.npy`'yi tam cache uzunlugunda yeniden dizer, gelistirme disi
   satirlari sifir birakir.
4. `test_prob.npy`'yi yeni test_public sirasina gore yeniden dizer.
5. `baseline/<name>/` altina yazar; `ensemble.py` oradan kendiliginde bulur.

Ne yapmaz
---------
On isleme farkini duzeltmez. Eski matrislerin baska bir `ecg_preprocess.py`
ile uretildiyse, iceri aktarilan olasiliklar o eski on islemeye aittir. Bu
genelde sorun degildir -- ensemble uyeleri zaten farkli modellerdir -- ama
`SONUC.md`'ye yazilmasi gereken bir gercektir.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecg_preprocess as ep

N_CLASSES = len(ep.CLASSES)


def _norm_key(value):
    """Kayit adini karsilastirilabilir hale getir."""
    v = str(value).strip().strip("/\\").replace("\\", "/").lower()
    v = v.rsplit("/", 1)[-1]
    for ext in (".hea", ".dat", ".mat"):
        if v.endswith(ext):
            v = v[:-len(ext)]
    return v


def read_old_index(path, expected_keys):
    """Eski index dosyasindan satir sirasina gore kayit adlarini cikar.

    Sutun adi varsayilmaz: yeni cache'teki kayit adlariyla en cok eslesen
    sutun secilir. Duz metin dosyasi (satir basina bir ad) da kabul edilir.
    """
    if not os.path.exists(path):
        raise SystemExit("eski index bulunamadi: %s" % path)

    if path.lower().endswith((".txt", ".lst")):
        with open(path, errors="replace") as fh:
            return [_norm_key(line) for line in fh if line.strip()]

    with open(path, newline="", errors="replace") as fh:
        rows = [r for r in csv.reader(fh) if any(str(c).strip() for c in r)]
    if not rows:
        raise SystemExit("%s bos" % path)

    n_col = max(len(r) for r in rows)
    header_is_data = any(_norm_key(c) in expected_keys for c in rows[0])
    body = rows if header_is_data else rows[1:]

    best_col, best_hits = None, 0
    for col in range(n_col):
        hits = sum(1 for r in body
                   if col < len(r) and _norm_key(r[col]) in expected_keys)
        if hits > best_hits:
            best_col, best_hits = col, hits

    if best_col is None or best_hits == 0:
        header = rows[0] if not header_is_data else ["(basliksiz)"]
        raise SystemExit(
            "%s icindeki hicbir sutun yeni cache'teki kayit adlariyla "
            "eslesmedi.\n  sutunlar: %s\n  ilk satir: %s\n"
            "Ayni veri kumesinin index'i mi verdin?" % (path, header, body[0]))

    name = (rows[0][best_col] if not header_is_data else "sutun %d" % best_col)
    print("  eski index: %d satir, kayit sutunu = %r (%d/%d eslesti)"
          % (len(body), name, best_hits, len(body)))
    return [_norm_key(r[best_col]) if best_col < len(r) else "" for r in body]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="baseline/<name>/ altina yazilir, orn. r18_feat")
    ap.add_argument("--old-index", required=True,
                    help="eski cache'in index.csv'si (veya kayit adi listesi)")
    ap.add_argument("--oof", required=True, help="eski oof_prob.npy")
    ap.add_argument("--test", default="", help="eski test_prob.npy (istege bagli)")
    ap.add_argument("--cache", default="cache", help="yeni cache dizini")
    ap.add_argument("--out", default="baseline")
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="gelistirme satirlarinin en az bu kadari dolmali")
    args = ap.parse_args(argv)

    new_index_path = os.path.join(args.cache, "index.csv")
    if not os.path.exists(new_index_path):
        raise SystemExit("%s yok -- once 'python prep.py' kos" % new_index_path)

    with open(new_index_path, newline="") as fh:
        new_rows = list(csv.DictReader(fh))
    new_keys = [_norm_key(r["record"]) for r in new_rows]
    key_to_new = {}
    for i, k in enumerate(new_keys):
        key_to_new.setdefault(k, i)
    expected = set(new_keys)

    dev_rows = [i for i, r in enumerate(new_rows)
                if r["split"] in ("train", "validation")]
    test_rows = [i for i, r in enumerate(new_rows) if r["split"] == "test_public"]
    print("yeni cache: %d kayit (%d gelistirme, %d test_public)"
          % (len(new_rows), len(dev_rows), len(test_rows)))

    old_keys = read_old_index(args.old_index, expected)

    old_oof = np.load(args.oof).astype(np.float64)
    if old_oof.ndim != 2 or old_oof.shape[1] != N_CLASSES:
        raise SystemExit("oof_prob sekli %s, (N, %d) bekleniyordu"
                         % (old_oof.shape, N_CLASSES))
    if old_oof.shape[0] != len(old_keys):
        raise SystemExit(
            "oof_prob %d satir, eski index %d satir -- ayni kosunun ciktilari "
            "mi? (Eski matris sadece gelistirme satirlarini iceriyorsa, eski "
            "index'ten de sadece o satirlari veren bir dosya hazirla.)"
            % (old_oof.shape[0], len(old_keys)))

    # --- OOF'u yeniden diz ---
    new_oof = np.zeros((len(new_rows), N_CLASSES), dtype=np.float64)
    moved, unmatched = 0, []
    for old_row, key in enumerate(old_keys):
        target = key_to_new.get(key)
        if target is None:
            unmatched.append(key)
            continue
        new_oof[target] = old_oof[old_row]
        moved += 1

    filled = new_oof[dev_rows].sum(axis=1) > 1e-9
    coverage = float(filled.mean()) if len(dev_rows) else 0.0
    print("  tasinan satir: %d/%d   eslesmeyen: %d"
          % (moved, len(old_keys), len(unmatched)))
    print("  gelistirme kapsamasi: %.1f%%" % (100 * coverage))

    if unmatched:
        print("  eslesmeyen ornekler: %s" % ", ".join(unmatched[:5]))
    if coverage < args.min_coverage:
        raise SystemExit(
            "kapsama %.1f%% < %.1f%% -- boyle bir matris ensemble'a "
            "girerse agirlik aramasi bos satirlar uzerinde calisir ve OOF "
            "skoru yaniltici cikar. Eski index'in dogru dosya oldugundan emin ol."
            % (100 * coverage, 100 * args.min_coverage))

    # --- test_prob'u yeniden diz ---
    new_test = np.zeros((len(test_rows), N_CLASSES), dtype=np.float64)
    if args.test:
        old_test = np.load(args.test).astype(np.float64)
        if old_test.shape[1] != N_CLASSES:
            raise SystemExit("test_prob sekli %s" % (old_test.shape,))

        # Eski test matrisi ya tam cache uzunlugunda ya da sadece test satirlari.
        if old_test.shape[0] == len(old_keys):
            old_test_keys = old_keys
        else:
            old_test_split = [k for k, r in zip(old_keys, range(len(old_keys)))]
            old_test_keys = None
            if old_test.shape[0] == len(test_rows):
                print("  test_prob sadece test satirlarini iceriyor; eski "
                      "index'teki test_public sirasi varsayiliyor")
                old_test_keys = [k for k in old_keys if key_to_new.get(k) in
                                 set(test_rows)]
                if len(old_test_keys) != old_test.shape[0]:
                    raise SystemExit(
                        "test_prob %d satir ama eski index'te %d test kaydi "
                        "bulundu -- eslestirme guvenli degil"
                        % (old_test.shape[0], len(old_test_keys)))
            else:
                raise SystemExit(
                    "test_prob %d satir; ne eski index (%d) ne de yeni "
                    "test_public (%d) ile uyusuyor"
                    % (old_test.shape[0], len(old_keys), len(test_rows)))

        pos_in_new_test = {row: i for i, row in enumerate(test_rows)}
        moved_test = 0
        for old_row, key in enumerate(old_test_keys):
            target = key_to_new.get(key)
            if target is None or target not in pos_in_new_test:
                continue
            new_test[pos_in_new_test[target]] = old_test[old_row]
            moved_test += 1
        print("  test_prob tasinan satir: %d/%d" % (moved_test, len(test_rows)))
        if moved_test < len(test_rows):
            print("  UYARI: %d test satiri bos kaldi -- ensemble.py bu uyeyi "
                  "reddedebilir" % (len(test_rows) - moved_test))

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "oof_prob.npy"), new_oof)
    np.save(os.path.join(out_dir, "test_prob.npy"), new_test)

    import json

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({
            "tag": args.name,
            "imported_from": {"index": os.path.abspath(args.old_index),
                              "oof": os.path.abspath(args.oof),
                              "test": os.path.abspath(args.test) or None},
            "rows_moved": moved,
            "unmatched": len(unmatched),
            "dev_coverage": coverage,
            "note": ("Bu matrisler baska bir kosudan iceri aktarildi. On "
                     "isleme bu depodakinden farkli olabilir; SONUC.md'ye yaz."),
        }, fh, indent=2)

    print("\nyazildi: %s/" % out_dir)
    print("Sonraki adim: python ensemble.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
