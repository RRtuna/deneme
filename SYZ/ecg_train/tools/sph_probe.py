"""sph_probe -- SPH veri setinde SENIN 5 sinifindan kac kayit var?

    python tools/sph_probe.py --dir D:\\SPH_meta

Yalnizca IKI KUCUK CSV okur: `attributes.csv` ve `code.csv`. 25.770 adet .h5
dosyasini indirmene GEREK YOK -- once bu iki dosyayi indir, betigi calistir,
sayilari gor, sonra indirmeye deger mi karar ver.

    https://data.mendeley.com/datasets/dvb5mnhfc4/1

Neden ayri bir betik
--------------------
SPH etiketleri SNOMED degil, **AHA/ACC/HRS ifadeleri** (44 ana ifade + 15
niteleyici). Yani `add_external.py`'nin SNOMED yolu burada calismaz; once
"hangi AHA kodu senin hangi sinifina denk geliyor" sorusu cevaplanmali.

Betik kolon adlarini ve ayraci **calisma aninda kesfeder** -- dokumantasyondan
tahmin etmez. Bulduklarini basar, boylece yanlis eslesme sessizce gecmez.

Karar
-----
AFL sayisi belirleyici. Senin darbogazin orasi ve acik veri setlerinde en kit
sinif o. Birkac yuzun altindaysa donusturucu yazmaya degmez; Challenge 2021
havuzunda zaten ~7.800 eklenebilir AFL var.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter

# Senin siniflarina karsilik gelebilecek AHA ifadeleri -- ANAHTAR KELIMEYLE
# aranir, kod numarasi tahmin edilmez.
KEYWORDS = {
    "AFL":    ("atrial flutter",),
    "AFIB":   ("atrial fibrillation",),
    "LBBB":   ("left bundle branch block",),
    "RBBB":   ("right bundle branch block",),
    "Normal": ("normal ecg", "normal electrocardiogram", "sinus rhythm"),
}
ORDER = ("Normal", "AFIB", "AFL", "LBBB", "RBBB")


def sniff(path):
    """Ayraci ve satirlari bul."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        head = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:                            # noqa: BLE001
            sep = ","
        fh.seek(0)
        rows = list(csv.DictReader(fh, delimiter=sep))
    return rows, sep


def find_file(directory, *names):
    for n in names:
        p = os.path.join(directory, n)
        if os.path.exists(p):
            return p
    # esnek arama
    for fn in os.listdir(directory):
        low = fn.lower()
        if any(n.split(".")[0] in low for n in names) and low.endswith(".csv"):
            return os.path.join(directory, fn)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="attributes.csv ve code.csv'nin bulundugu klasor")
    ap.add_argument("--attributes", default="")
    ap.add_argument("--codes", default="")
    args = ap.parse_args(argv)

    attr_p = args.attributes or find_file(args.dir, "attributes.csv", "metadata.csv")
    code_p = args.codes or find_file(args.dir, "code.csv", "codes.csv")
    if not attr_p or not code_p:
        raise SystemExit("attributes.csv ve/veya code.csv bulunamadi (%s)" % args.dir)

    codes, sep_c = sniff(code_p)
    attrs, sep_a = sniff(attr_p)
    print("code.csv      : %s  (%d satir, ayrac %r)" % (code_p, len(codes), sep_c))
    print("attributes.csv: %s  (%d satir, ayrac %r)" % (attr_p, len(attrs), sep_a))
    if not codes or not attrs:
        raise SystemExit("dosyalardan biri bos")
    print("code.csv kolonlari      : %s" % ", ".join(codes[0].keys()))
    print("attributes.csv kolonlari: %s" % ", ".join(attrs[0].keys()))

    # ---- code.csv: hangi kolon kod, hangisi aciklama ----------------------
    ckeys = list(codes[0].keys())
    code_col = next((k for k in ckeys if re.search(r"code|id|no", k, re.I)), ckeys[0])
    desc_col = next((k for k in ckeys
                     if re.search(r"desc|statement|name|diagn|text", k, re.I)),
                    ckeys[-1])
    print()
    print("kod kolonu: %r   aciklama kolonu: %r" % (code_col, desc_col))

    # ---- anahtar kelimeyle sinif -> kod eslemesi --------------------------
    mapping = {}
    print()
    print("AHA IFADESI -> SENIN SINIFIN")
    for cls in ORDER:
        hits = []
        for r in codes:
            d = (r.get(desc_col) or "").strip().lower()
            if any(k in d for k in KEYWORDS[cls]):
                hits.append((str(r.get(code_col)).strip(), r.get(desc_col).strip()))
        mapping[cls] = [h[0] for h in hits]
        if hits:
            for code, desc in hits:
                print("  %-7s <- %-6s %s" % (cls, code, desc))
        else:
            print("  %-7s <- (eslesen ifade YOK)" % cls)

    unmatched = [cls for cls in ORDER if not mapping[cls]]
    if unmatched:
        print()
        print("UYARI: su siniflar icin AHA ifadesi bulunamadi: %s"
              % ", ".join(unmatched))
        print("code.csv'deki tum ifadeler asagida -- eslemeyi elle kontrol et:")
        for r in codes[:80]:
            print("   %-6s %s" % (str(r.get(code_col)).strip(),
                                  (r.get(desc_col) or "").strip()))

    # ---- attributes.csv: hangi kolon kodlari tasiyor ----------------------
    akeys = list(attrs[0].keys())
    all_codes = {c for v in mapping.values() for c in v}
    best, best_hits = None, -1
    for k in akeys:
        hits = sum(1 for r in attrs
                   if any(c in re.split(r"[;,\s]+", str(r.get(k) or ""))
                          for c in all_codes))
        if hits > best_hits:
            best, best_hits = k, hits
    print()
    print("kod tasiyan kolon: %r  (%d kayitta hedef kod bulundu)" % (best, best_hits))
    if best_hits == 0:
        raise SystemExit("Hicbir kolonda hedef kod bulunamadi -- esleme basarisiz.\n"
                         "  Yukaridaki kolon adlarini ve code.csv ciktisini bana gonder.")

    # ---- sayim -----------------------------------------------------------
    code_to_cls = {}
    for cls, cl in mapping.items():
        for c in cl:
            code_to_cls[c] = cls

    single, multi, none_ = Counter(), 0, 0
    for r in attrs:
        toks = set(re.split(r"[;,\s]+", str(r.get(best) or "")))
        hit = {code_to_cls[c] for c in toks if c in code_to_cls}
        if not hit:
            none_ += 1
        elif len(hit) > 1:
            multi += 1
        else:
            single[hit.pop()] += 1

    print()
    print("SENIN 5 SINIFINDA KAC KAYIT (tek etiketli)")
    print("%-8s %10s" % ("sinif", "kayit"))
    for cls in ORDER:
        print("%-8s %10d" % (cls, single[cls]))
    print("%-8s %10d" % ("TOPLAM", sum(single.values())))
    print()
    print("birden fazla hedef tani  : %d  (tek etiketli kuralda elenir)" % multi)
    print("hedef tani tasimayan     : %d" % none_)

    print()
    print("KARAR")
    afl = single["AFL"]
    print("  Belirleyici sayi AFL: %d" % afl)
    if afl >= 500:
        print("  DEGER. Challenge 2021 disindan, TAMAMEN FARKLI bir hastaneden")
        print("  %d AFL demek -- kaynak cesitliligi icin bu cok degerli," % afl)
        print("  cunku final degerlendirmesi (md. 7.2) yeni bir veri setinde.")
        print("  Donusturucuyu yazmaya deger; bana bu ciktiyi gonder.")
        return 0
    if afl >= 150:
        print("  SINIRDA. %d AFL az ama farkli bir kaynaktan geldigi icin" % afl)
        print("  cesitlilik degeri var. Challenge 2021 islerin bittikten")
        print("  SONRA, vakit kalirsa yap.")
        return 0
    print("  BIRAK. %d AFL, donusturucu + etiket eslemesi isine degmez." % afl)
    print("  Challenge 2021 havuzunda ~7.800 eklenebilir AFL zaten var.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
