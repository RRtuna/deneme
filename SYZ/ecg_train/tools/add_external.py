"""add_external -- acik kaynak EKG kayitlarini cache'e EK EGITIM VERISI olarak ekle.

    python tools/add_external.py --source D:\\ECG_ARRHYTHMIA --cache cache --out cache_ext --dry-run
    python tools/add_external.py --source D:\\ECG_ARRHYTHMIA --cache cache --out cache_ext

Sartname madde 3.1.1 dis veri kullanimina acikca izin veriyor ve ornek veri
olarak PhysioNet **ECG Arrhythmia Dataset 1.0.0**'i gosteriyor (~45.000 kayit,
12 derivasyon, 500 Hz, SNOMED-CT etiketleri). Yarismanin kendi kumesi bu
kaynakla ayni etiketleme duzenini kullandigi icin, klasik "farkli veri setinden
gelen etiket konvansiyonu uyusmuyor" riski burada yok.

Eklenen kayitlar `split="extra"` alir. `train.py` bunlari **her fold'un egitim
kismina** koyar, **hicbir fold'un dogrulamasina** koymaz. Boylece OOF skoru
yalnizca yarismanin kendi verisinde olculmeye devam eder ve onceki tum
deneylerle karsilastirilabilir kalir. `test_public`'e zaten hic dokunulmaz.

ASIL TEHLIKE: SIZINTI
---------------------
Senin `test_public` kayitlarin da bu acik veri setinin icinde. Disaridan
eklenen bir kayit onlardan biriyle ayni ise, test uzerinde egitmis olursun ve
bunu OOF'ta GOREMEZSIN -- skor yukselir, gercek basari duser. Bu yuzden betik
uc kademeli cakisma taramasi yapar:

  1. kayit adi        (uzanti ve buyuk/kucuk harf yok sayilir)
  2. sekil imzasi     her derivasyon z-skorlanip yuvarlanir ve ozetlenir;
                      kazanc/ofset degisimine duyarsizdir
  3. korelasyon       II derivasyonu 128 noktaya indirilip z-skorlanir; senin
                      TUM kayitlarinla capraz korelasyon alinir, esigi asan
                      aday kopya sayilir (yeniden orneklenmis/kirpilmis
                      kopyalari da yakalar)

Bir kayit bu uc testten HERHANGI birine takilirsa eklenmez.

ETIKET HARITASI -- tahmin edilmez, SENIN VERINDEN cikarilir
-----------------------------------------------------------
SNOMED kodlarini elle yazmak risklidir (ornegin RBBB icin 59118001 mi,
713427006 mi?). Bunun yerine betik once senin cache'indeki kayitlarin
kaynaktaki karsiliklarini bulur ve "senin hangi etiketin, kaynakta hangi kodla
gorunuyor" tablosunu cikarir. Harita boylece kendi verinden dogar. Ortusme
bulunamazsa yerlesik tabloya duser ve bunu **yuksek sesle** soyler.

Once daima --dry-run ile calistir: hicbir sey yazmaz, ne ekleyecegini gosterir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecg_preprocess as ep  # noqa: E402
import wfdb_lite as wl       # noqa: E402

# Yalnizca ortusme bulunamazsa kullanilir. PhysioNet/CinC 2020-2021
# esleniklerini icerir; dogrulanmis degildir, bu yuzden uyari basilir.
FALLBACK_SNOMED = {
    "Normal": ("426783006",),
    "AFIB":   ("164889003",),
    "AFL":    ("164890007",),
    "LBBB":   ("164909002", "733534002"),
    "RBBB":   ("59118001", "713427006"),
}

FP_LEN = 128          # imza uzunlugu (II derivasyonu, yeniden orneklenmis)
CORR_GATE = 0.995     # bunun ustu kopya sayilir


# --------------------------------------------------------------------------
# kaynak tarama
# --------------------------------------------------------------------------

def scan_source(root):
    """Kaynak agacindaki tum .hea dosyalari."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".hea"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


_DX = re.compile(r"#\s*Dx\s*:\s*(.+)", re.I)


def dx_codes(hea_path):
    """.hea yorum satirlarindan SNOMED kodlari. Yoksa bos kume."""
    try:
        hdr = wl.read_header(hea_path)
    except Exception:                            # noqa: BLE001
        return None
    for line in hdr.comments:
        m = _DX.match(line.strip())
        if m:
            return {c.strip() for c in m.group(1).split(",") if c.strip()}
    return set()


def stem(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


# --------------------------------------------------------------------------
# imzalar
# --------------------------------------------------------------------------

def fingerprint(sig):
    """(sekil imzasi, korelasyon vektoru).

    Ikisi de derivasyon kazancina ve ofsetine duyarsizdir: sinyal once
    z-skorlanir. Kaynak ile yarisma kopyasi farkli olcekte kaydedilmis olsa
    bile ayni imzayi verir.
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    z = (x - x.mean(axis=1, keepdims=True))
    z = z / (z.std(axis=1, keepdims=True) + 1e-12)

    shape = tuple(np.round(z[:, ::max(z.shape[1] // 16, 1)][:, :16], 2).ravel())

    lead = z[1] if z.shape[0] > 1 else z[0]
    idx = np.linspace(0, lead.size - 1, FP_LEN)
    v = np.interp(idx, np.arange(lead.size), lead)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return shape, (v / n if n > 1e-12 else v)


def load_signal(path):
    sig, fs, _leads = wl.read_record(path)
    return np.asarray(sig, dtype=np.float64), float(fs)


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="indirilmis acik veri setinin kok klasoru")
    ap.add_argument("--cache", default="cache", help="mevcut yarisma cache'i")
    ap.add_argument("--out", default="cache_ext")
    ap.add_argument("--dry-run", action="store_true",
                    help="hicbir sey yazma, ne eklenecegini goster")
    ap.add_argument("--per-class", type=int, default=0,
                    help="sinif basina en fazla kac kayit eklensin (0 = sinirsiz)")
    ap.add_argument("--only", default="",
                    help="yalnizca bu siniflar, virgulle (or. AFIB,AFL)")
    ap.add_argument("--corr-gate", type=float, default=CORR_GATE)
    ap.add_argument("--single-label", action="store_true", default=True,
                    help="yalnizca TEK hedef tani tasiyan kayitlari al (varsayilan)")
    ap.add_argument("--allow-multi", dest="single_label", action="store_false",
                    help="birden fazla hedef tani tasiyan kayitlari da al")
    args = ap.parse_args(argv)

    classes = list(ep.CLASSES)
    want = set(c.strip() for c in args.only.split(",") if c.strip()) or set(classes)
    for c in want:
        if c not in classes:
            raise SystemExit("bilinmeyen sinif %r; mevcut: %s"
                             % (c, ", ".join(classes)))

    idx_path = os.path.join(args.cache, "index.csv")
    if not os.path.exists(idx_path):
        raise SystemExit("%s yok" % idx_path)
    with open(idx_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    meta_path = os.path.join(args.cache, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    target_fs = float(meta.get("target_fs") or ep.TARGET_FS)

    print("yarisma cache'i : %s  (%d kayit)" % (args.cache, len(rows)))
    print("kaynak          : %s" % args.source)
    hea = scan_source(args.source)
    print("kaynakta bulunan: %d .hea dosyasi" % len(hea))
    if not hea:
        raise SystemExit("kaynakta .hea yok -- yol dogru mu?")

    # ---- 1) senin kayitlarinin imzalari -----------------------------------
    print()
    print("1/4  yarisma kayitlarinin imzalari cikariliyor")
    t0 = time.time()
    own_stem, own_shape, own_vecs, own_label, own_split = {}, {}, [], [], []
    for i, r in enumerate(rows):
        p = r.get("path") or ""
        if not p or not os.path.exists(p):
            continue
        try:
            sig, _fs = load_signal(p)
        except Exception:                        # noqa: BLE001
            continue
        sh, v = fingerprint(sig)
        own_stem[stem(p)] = i
        own_shape[sh] = i
        own_vecs.append(v)
        own_label.append(r.get("label_name") or "")
        own_split.append(r.get("split") or "")
        if (i + 1) % 1000 == 0:
            print("     %5d/%d  %.0f sn" % (i + 1, len(rows), time.time() - t0),
                  flush=True)
    if not own_vecs:
        raise SystemExit(
            "Hicbir yarisma kaydi okunamadi.\n"
            "  index.csv'deki 'path' sutunu bu makinede gecerli mi?\n"
            "  Cakisma taramasi yapilamadan dis veri EKLENEMEZ -- sizinti riski.")
    OWN = np.stack(own_vecs)
    own_label = np.array(own_label)
    own_split = np.array(own_split)
    print("     %d kayit imzalandi  (%.0f sn)" % (len(OWN), time.time() - t0))

    # ---- 2) etiket haritasini SENIN verinden cikar -------------------------
    print()
    print("2/4  etiket haritasi cikariliyor (kaynaktaki karsiliklarindan)")
    src_by_stem = {stem(h): h for h in hea}
    votes = defaultdict(Counter)
    n_matched = 0
    for st, i in own_stem.items():
        h = src_by_stem.get(st)
        if h is None:
            continue
        codes = dx_codes(h)
        if not codes:
            continue
        n_matched += 1
        lab = rows[i].get("label_name") or ""
        for c in codes:
            votes[lab][c] += 1

    label_map, derived = {}, False
    if n_matched >= 50:
        derived = True
        # Bir kodu, yalnizca TEK bir etikette baskinsa o etikete bagla.
        totals = Counter()
        for lab, cnt in votes.items():
            for c, n in cnt.items():
                totals[c] += n
        for lab, cnt in votes.items():
            n_lab = sum(1 for st, i in own_stem.items()
                        if (rows[i].get("label_name") or "") == lab
                        and stem(rows[i].get("path", "")) in src_by_stem)
            keep = [c for c, n in cnt.items()
                    if n >= 0.5 * max(n_lab, 1) and n >= 0.9 * totals[c]]
            if keep:
                label_map[lab] = tuple(sorted(keep))
        print("     %d kayit kaynakta bulundu, harita SENIN verinden turedi:"
              % n_matched)
    else:
        label_map = dict(FALLBACK_SNOMED)
        print("     UYARI: yalnizca %d ortak kayit bulundu (>=50 gerekli)."
              % n_matched)
        print("     Harita SENIN verinden turetilemedi, yerlesik tablo")
        print("     kullaniliyor. Asagidaki kodlari kendi veri setinin")
        print("     dokumantasyonuyla KARSILASTIR:")

    for lab in classes:
        print("       %-8s <- %s" % (lab, ", ".join(label_map.get(lab, ())) or "(yok)"))
    missing = [c for c in want if not label_map.get(c)]
    if missing:
        raise SystemExit("su siniflar icin SNOMED kodu belirlenemedi: %s\n"
                         "  --only ile bunlari disarida birak ya da kodlari"
                         " elle dogrula." % ", ".join(missing))

    code_to_label = {}
    for lab, codes in label_map.items():
        for c in codes:
            code_to_label[c] = lab

    # ---- 3) adaylari sec + cakisma taramasi -------------------------------
    print()
    print("3/4  aday secimi ve cakisma taramasi")
    cand, skipped = [], Counter()
    for h in hea:
        codes = dx_codes(h)
        if codes is None:
            skipped["basliк okunamadi"] += 1
            continue
        hits = {code_to_label[c] for c in codes if c in code_to_label}
        if not hits:
            skipped["hedef tani yok"] += 1
            continue
        if args.single_label and len(hits) > 1:
            skipped["birden fazla hedef tani"] += 1
            continue
        lab = sorted(hits)[0]
        if lab not in want:
            skipped["istenmeyen sinif"] += 1
            continue
        if stem(h) in own_stem:
            skipped["ad cakismasi (senin kaydin)"] += 1
            continue
        cand.append((h, lab))

    print("     aday: %d kayit" % len(cand))
    for k, v in skipped.most_common():
        print("     elendi: %-32s %6d" % (k, v))
    if not cand:
        raise SystemExit("eklenecek aday yok")

    print()
    print("     sekil ve korelasyon taramasi (%d aday)" % len(cand))
    accepted, dup_shape, dup_corr, unreadable = [], 0, 0, 0
    t0 = time.time()
    BATCH, buf, bufmeta = 512, [], []

    def flush():
        nonlocal dup_corr
        if not buf:
            return
        M = np.stack(buf)                       # (b, FP_LEN), birim norm
        C = M @ OWN.T                           # (b, n_own) korelasyon
        best = C.max(axis=1)
        arg = C.argmax(axis=1)
        for k, (h, lab, sig, fs) in enumerate(bufmeta):
            if best[k] >= args.corr_gate:
                dup_corr += 1
                if dup_corr <= 5:
                    print("       KOPYA %s ~ %s (%s, r=%.4f)"
                          % (os.path.basename(h), own_split[arg[k]],
                             own_label[arg[k]], best[k]))
                continue
            accepted.append((h, lab, sig, fs))
        buf.clear()
        bufmeta.clear()

    for n, (h, lab) in enumerate(cand):
        try:
            sig, fs = load_signal(h)
        except Exception:                        # noqa: BLE001
            unreadable += 1
            continue
        sh, v = fingerprint(sig)
        if sh in own_shape:
            dup_shape += 1
            continue
        buf.append(v)
        bufmeta.append((h, lab, sig, fs))
        if len(buf) >= BATCH:
            flush()
        if (n + 1) % 2000 == 0:
            print("       %6d/%d  %.0f sn" % (n + 1, len(cand), time.time() - t0),
                  flush=True)
    flush()

    print()
    print("     okunamadi           : %d" % unreadable)
    print("     sekil imzasi ayni   : %d  <- SIZINTI olurdu" % dup_shape)
    print("     korelasyon >= %.3f  : %d  <- SIZINTI olurdu"
          % (args.corr_gate, dup_corr))
    print("     KABUL EDILEN        : %d" % len(accepted))

    if args.per_class:
        by = defaultdict(list)
        for item in accepted:
            by[item[1]].append(item)
        accepted = [it for lab in by for it in by[lab][:args.per_class]]
        print("     --per-class %d sonrasi: %d" % (args.per_class, len(accepted)))

    print()
    print("%-8s %10s %10s %10s" % ("sinif", "senin", "eklenen", "toplam"))
    own_cnt = Counter(own_label)
    add_cnt = Counter(lab for _h, lab, _s, _f in accepted)
    for c in classes:
        print("%-8s %10d %10d %10d" % (c, own_cnt[c], add_cnt[c],
                                       own_cnt[c] + add_cnt[c]))

    if args.dry_run:
        print()
        print("--dry-run: hicbir sey yazilmadi.")
        print("Sayilar makul gorunuyorsa --dry-run'i kaldirip tekrar calistir.")
        return 0
    if not accepted:
        raise SystemExit("kabul edilen kayit yok, cache yazilmadi")

    # ---- 4) yeni cache'i yaz ---------------------------------------------
    print()
    print("4/4  genisletilmis cache yaziliyor")
    X_old = np.load(os.path.join(args.cache, "X.npy"), mmap_mode="r")
    y_old = np.load(os.path.join(args.cache, "y.npy"))
    f_path = os.path.join(args.cache, "F.npy")
    F_old = np.load(f_path) if os.path.exists(f_path) else None

    n_old, n_lead, T = X_old.shape
    n_new = n_old + len(accepted)
    os.makedirs(args.out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(args.out, "X.npy"), mode="w+",
                                  dtype=np.float32, shape=(n_new, n_lead, T))
    X[:n_old] = X_old
    y = np.concatenate([y_old,
                        np.array([classes.index(l) for _h, l, _s, _f in accepted],
                                 dtype=y_old.dtype)])
    F = (np.zeros((n_new, F_old.shape[1]), dtype=np.float32)
         if F_old is not None else None)
    if F is not None:
        F[:n_old] = F_old

    t0 = time.time()
    bad = []
    for k, (h, lab, sig, fs) in enumerate(accepted):
        i = n_old + k
        try:
            X[i] = ep.preprocess_signal(sig, fs, target_fs=target_fs)
            if F is not None:
                F[i] = ep.extract_features(sig, fs)
        except Exception as exc:                 # noqa: BLE001
            bad.append((h, "%s: %s" % (type(exc).__name__, exc)))
        if (k + 1) % 500 == 0 or k + 1 == len(accepted):
            print("     %5d/%d  %.0f sn" % (k + 1, len(accepted),
                                            time.time() - t0), flush=True)
    X.flush()
    np.save(os.path.join(args.out, "y.npy"), y)
    if F is not None:
        np.save(os.path.join(args.out, "F.npy"), F)

    bad_paths = {h for h, _e in bad}
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "record", "path", "label", "label_name", "split", "ok"])
        for i, r in enumerate(rows):
            w.writerow([i, r["record"], r.get("path", ""), r["label"],
                        r.get("label_name", ""), r["split"], r.get("ok", 1)])
        for k, (h, lab, _s, _f) in enumerate(accepted):
            w.writerow([n_old + k, stem(h), h, classes.index(lab), lab,
                        "extra", int(h not in bad_paths)])

    meta = dict(meta)
    meta["external"] = {
        "source": os.path.abspath(args.source),
        "added": len(accepted),
        "failed": len(bad),
        "per_class": dict(add_cnt),
        "label_map": {k: list(v) for k, v in label_map.items()},
        "label_map_derived_from_own_data": derived,
        "dup_by_name": skipped.get("ad cakismasi (senin kaydin)", 0),
        "dup_by_shape": dup_shape,
        "dup_by_corr": dup_corr,
        "corr_gate": args.corr_gate,
        "single_label_only": args.single_label,
    }
    meta["n_records"] = n_new
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if bad:
        print()
        print("     on isleme hatasi: %d kayit ok=0 isaretlendi" % len(bad))
        for h, e in bad[:5]:
            print("       %s  %s" % (os.path.basename(h), e))

    print()
    print("yazildi: %s  (%d kayit = %d yarisma + %d dis)"
          % (args.out, n_new, n_old, len(accepted)))
    print()
    print("sonraki:")
    print("  python train.py --cache %s --tag ext --preset <mevcut> "
          "--folds 5 --epochs 40 --patience 99" % args.out)
    print()
    print("Kapi: OOF hala YALNIZCA yarisma verisinde olculuyor, yani bu sayi")
    print("mevcut kosunla dogrudan karsilastirilabilir. Artmiyorsa dis veri")
    print("yardim etmiyor demektir -- DENEY_KAYDI.md'ye yaz ve geri don.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
