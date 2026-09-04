"""make_resid_cache -- mevcut cache'e QRST-iptalli ARTIK KANALLARI ekler.

    python tools/make_resid_cache.py --in cache --out cache_resid

Girdi cache'in X.npy'si (n, 12, T); cikti (n, 16, T). y.npy, F.npy, index.csv
oldugu gibi kopyalanir. **`ecg_preprocess.py`'ye dokunmaz**: bu adim zaten
on islenmis sinyalin uzerinde calisir, on isleme davranisi degismez.

Neden bu, 24 skaler ozellikten farkli
-------------------------------------
`qrst_features.py` ayni iptali yapip 24 SAYI cikariyor ve ozellik dalina
veriyor. Ama agin gucu sayilarda degil, DALGA SEKLI tanimada. AFL'nin testere
disi F dalgasi ile AFIB'in duzensiz f dalgasi arasindaki fark tam olarak bir
sekil farkidir; onu 7 spektral skalere sikistirmak bilginin cogunu atar.

Asil sorun ise dinamik aralik. Kayit basina normalizasyon QRS genligine gore
yapilir (R ~ 1.0). Atriyal dalga ondan 10-20 kat kucuktur, yani agin gordugu
sayi araliginda ~0.05-0.1'lik bir dalgalanmadir ve uzerine QRS treninin
harmonikleri biner. Ag once QRS'i modellemek zorunda kalir; atriyal dalga
artik bir seydir.

Artik kanali bunu tersine cevirir: QRS sablonu cikarilir, geriye kalan sinyal
**kendi olceginde** aga verilir. Ayni bilgi, 10-20 kat daha buyuk genlikte.

Olcekleme -- neden kayit basina DEGIL
-------------------------------------
Her kaydin artigini kendi std'sine bolmek cazip ama zararli: Normal/LBBB/RBBB
kayitlarinda gercek atriyal aktivite yok, geriye yalnizca gurultu kalir; kayit
basina normalizasyon o gurultuyu birim varyansa sisirir ve ag 4 kanal saf
gurultu gorur. Bunun yerine TEK bir kume-genelinde sabit kullanilir; boylece
"artik ne kadar buyuk" bilgisi (AFIB/AFL'de buyuk, digerlerinde kucuk)
korunur, yalnizca olcek buyutulur.

Sabit **yalnizca gelistirme kayitlarindan** hesaplanir; `test_public` hicbir
asamada okunmaz (DEGISMEZ KURAL 4).
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

import ecg_preprocess as ep  # noqa: E402

# Flutter dalgalari en cok bu derivasyonlarda gorunur (alt duvar + V1).
RESID_LEADS = ("II", "III", "aVF", "V1")

_EPS = 1e-12


def _lead_index(name):
    """Derivasyon indeksini modulun kendi listesinden al (sabit yazma)."""
    for attr in ("STANDARD_LEADS", "LEADS", "LEAD_NAMES"):
        leads = getattr(ep, attr, None)
        if leads:
            up = [str(s).upper() for s in leads]
            if name.upper() in up:
                return up.index(name.upper())
    raise SystemExit("ecg_preprocess icinde derivasyon listesi bulunamadi")


def _detect(sig, fs):
    """R tepeleri -- iki API yazimini da destekle."""
    for fname in ("detect_rpeaks", "detect_r"):
        fn = getattr(ep, fname, None)
        if fn is None:
            continue
        try:
            return np.asarray(fn(sig, fs), dtype=int)
        except TypeError:
            pass
        try:                                   # bazi surumler tek derivasyon ister
            return np.asarray(fn(sig[1], fs), dtype=int)
        except Exception:
            continue
    raise SystemExit("R tepe bulucu yok (detect_rpeaks / detect_r)")


def cancel_qrst(x, peaks, fs, pre=0.25, post=0.45):
    """Medyan sablonu vurus basina en kucuk kareler ile olcekleyip cikar."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    w_pre, w_post = int(round(pre * fs)), int(round(post * fs))
    usable = [int(p) for p in peaks if p - w_pre >= 0 and p + w_post < n]
    if len(usable) < 3:
        return x - x.mean()

    segs = np.stack([x[p - w_pre:p + w_post] for p in usable])
    template = np.median(segs, axis=0)      # medyan: tek bozuk vurus kaydirmasin
    t_energy = float(template @ template)
    if t_energy < _EPS:
        return x - x.mean()

    res = x.copy()
    for p in usable:
        a, b = p - w_pre, p + w_post
        seg = x[a:b]
        res[a:b] = seg - (float(seg @ template) / t_energy) * template
    return res - res.mean()


def residual_block(sig, fs, lead_idx):
    """(k, T) artik kanallari -- olceklenmemis, ham genlikte."""
    peaks = _detect(sig, fs)
    return np.stack([cancel_qrst(sig[i], peaks, fs) for i in lead_idx])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="cache")
    ap.add_argument("--out", dest="dst", default="cache_resid")
    ap.add_argument("--fs", type=float, default=0.0,
                    help="cache'in ornekleme hizi (0 = meta.json'dan oku)")
    ap.add_argument("--target-rms", type=float, default=0.35,
                    help="gelistirme kumesindeki MEDYAN artik std'sinin "
                         "esleneceği deger")
    ap.add_argument("--clip", type=float, default=8.0)
    args = ap.parse_args(argv)

    x_path = os.path.join(args.src, "X.npy")
    idx_path = os.path.join(args.src, "index.csv")
    for p in (x_path, idx_path):
        if not os.path.exists(p):
            raise SystemExit("%s yok" % p)

    fs = args.fs
    meta_path = os.path.join(args.src, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        fs = fs or float(meta.get("target_fs") or 0.0)
    if not fs:
        raise SystemExit("ornekleme hizi bilinmiyor; --fs ver")

    with open(idx_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    dev = np.array([i for i, r in enumerate(rows) if r["split"] != "test_public"])

    X = np.load(x_path, mmap_mode="r")
    n, n_lead, T = X.shape
    if n != len(rows):
        raise SystemExit("X %d satir, index %d satir" % (n, len(rows)))

    lead_idx = [_lead_index(name) for name in RESID_LEADS]
    k = len(lead_idx)
    print("kaynak : %s  (%d kayit, %d derivasyon, %d ornek, %.0f Hz)"
          % (args.src, n, n_lead, T, fs))
    print("artik  : %s  ->  %d ek kanal (toplam %d)"
          % (", ".join(RESID_LEADS), k, n_lead + k))
    print("test_public satirlari olcek hesabina GIRMEZ (%d gelistirme kaydi)"
          % len(dev))

    os.makedirs(args.dst, exist_ok=True)
    out_path = os.path.join(args.dst, "X.npy")
    Y = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                  shape=(n, n_lead + k, T))

    t0 = time.time()
    stds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sig = np.asarray(X[i], dtype=np.float64)
        res = residual_block(sig, fs, lead_idx)
        Y[i, :n_lead] = X[i]
        Y[i, n_lead:] = res.astype(np.float32)      # olcek sonra uygulanir
        stds[i] = float(np.median(res.std(axis=1)))
        if (i + 1) % 250 == 0 or i + 1 == n:
            print("  %5d/%d  %.1f sn" % (i + 1, n, time.time() - t0), flush=True)

    med = float(np.median(stds[dev]))
    if med < _EPS:
        raise SystemExit("artik std sifir -- R tepeleri bulunamamis olabilir")
    scale = args.target_rms / med
    print()
    print("gelistirme medyan artik std : %.5f" % med)
    print("olcek carpani               : %.2f x" % scale)

    for i in range(n):                              # olcek + kirp
        block = Y[i, n_lead:] * scale
        Y[i, n_lead:] = np.clip(block, -args.clip, args.clip)
    Y.flush()

    for name in ("y.npy", "F.npy", "index.csv"):
        src = os.path.join(args.src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dst, name))

    meta = dict(meta)
    meta["resid_from"] = os.path.abspath(args.src)
    meta["resid_leads"] = list(RESID_LEADS)
    meta["resid_scale"] = scale
    meta["resid_target_rms"] = args.target_rms
    meta["n_channels"] = n_lead + k
    with open(os.path.join(args.dst, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Kanallarin gercekten sinif ayirici olup olmadigini ucuza goster.
    y = np.array([int(r["label"]) for r in rows])
    print()
    print("%-8s %12s %12s" % ("sinif", "artik std", "n"))
    for c, name in enumerate(("Normal", "AFIB", "AFL", "LBBB", "RBBB")):
        m = (y == c) & np.isin(np.arange(n), dev)
        if m.any():
            print("%-8s %12.4f %12d" % (name, float(np.mean(stds[m]) * scale),
                                        int(m.sum())))
    print()
    print("yazildi: %s  (%.1f sn)" % (args.dst, time.time() - t0))
    print("sonraki: python train.py --cache %s --run runs/resid ..." % args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
