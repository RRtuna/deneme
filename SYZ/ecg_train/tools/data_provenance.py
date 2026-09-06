"""data_provenance -- yarisma verin hangi hastanelerden geliyor, ve bu bir tuzak mi?

    python tools/data_provenance.py --cache cache

Egitim yapmaz, dosya yazmaz, birkac saniye surer.

Neden bu soru kritik
--------------------
PhysioNet/CinC Challenge 2021 egitim kumesi sekiz ayri veri tabanindan olusur
ve siniflar bu kaynaklara COK dengesiz dagilmistir. Resmi sayimlar
(physionetchallenges/evaluation-2021, dx_mapping_scored.csv):

    sinif    CPSC  CPSCx  PTB-XL  Georgia  Chapman  Ningbo
    AFIB     1221    153    1514      570     1780       0
    AFL         0     54      73      186      445    7615

Yani **Ningbo'da hic AFIB yok, ama AFL'nin %91'i orada.** Eger senin 1000 AFL
kaydinin cogu Ningbo'dan, 1000 AFIB kaydin baska hastanelerden geliyorsa, iki
sinif buyuk olcude **kaynak hastaneye gore ayrisiyor** demektir.

Bu bir tuzaktir: ag, "AFIB mi AFL mi" yerine "hangi hastane" sorusunu
ogrenebilir. Kendi test_public'inde bu ise yarar (ayni dagilim), ama sartname
md. 7.2'ye gore final degerlendirmesi **TEKNOFEST'in ozgun ve yeni bir EKG veri
setinde** yapilacak -- orada hastane imzasi diye bir sey yok ve kisayol coker.

Bu betik o riski olcer. Bulursa ne yapilacagini da soyler.

Kayit adi -> kaynak esleme
--------------------------
Challenge 2021 kayit adlari kaynagi tasir:

    A#####   CPSC (China Physiological Signal Challenge 2018)
    Q#####   CPSC-Extra
    I#####   St Petersburg INCART
    S#####   PTB
    HR#####  PTB-XL
    E#####   Georgia
    JS#####  Chapman-Shaoxing (dusuk numaralar) / Ningbo (yuksek numaralar)

JS ayrimindaki esik yaklasiktir; betik bunu ayrica isaretler. Adlar bu duzene
uymuyorsa (TEKNOFEST yeniden adlandirmis olabilir) betik klasor yoluna duser ve
eslesmeyenleri listeler -- o listeyi bana gonder.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

# (onek, kaynak). Sirali: uzun onekler once denenir.
PREFIXES = (
    ("HR", "PTB-XL"),
    ("JS", "Chapman/Ningbo"),
    ("A", "CPSC"),
    ("Q", "CPSC-Extra"),
    ("I", "INCART"),
    ("S", "PTB"),
    ("E", "Georgia"),
)

# Ad onegi olmayan kaynaklar yol/klasor adindan cozulur.
PATH_KEYS = (
    ("mimic", "MIMIC-IV-ECG"), ("ptb-xl", "PTB-XL"), ("ptb_xl", "PTB-XL"),
    ("ningbo", "Ningbo"), ("chapman", "Chapman-Shaoxing"), ("shaoxing", "Chapman-Shaoxing"),
    ("georgia", "Georgia"), ("cpsc", "CPSC"), ("incart", "INCART"),
    ("st_petersburg", "INCART"), ("ptb", "PTB"),
)

# JS kayitlarinda Chapman-Shaoxing 10.646 kayittir; sonrasi Ningbo.
JS_SPLIT = 10646

_NUM = re.compile(r"(\d+)\s*$")


_PREFIX_RE = {pre: re.compile(r"^%s\d+$" % pre, re.I) for pre, _n in PREFIXES}


def source_of(record, path=""):
    """Kayit adindan kaynak veri tabani. Bilinmiyorsa klasore duser.

    Onek TAM eslesmeli: "E" ardindan yalnizca rakam gelmeli. Aksi halde
    "EXT_AFIB_001" gibi bir ad "E" onekine takilip Georgia sanilir -- kaynak
    dagilimi tablosu sessizce yanlis cikar.
    """
    r = str(record).strip()
    for pre, name in PREFIXES:
        if _PREFIX_RE[pre].match(r):
            if pre == "JS":
                m = _NUM.search(r)
                if m:
                    n = int(m.group(1))
                    return ("Chapman-Shaoxing" if n <= JS_SPLIT else "Ningbo")
                return "Chapman/Ningbo(?)"
            return name
    # Ad taninmadi: klasor adindan tahmin et.
    if path:
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
        for key, name in PATH_KEYS:
            if any(key in p for p in parts):
                return name
    return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--show-unknown", type=int, default=12,
                    help="taninmayan kayit adlarindan kac tanesi basilsin")
    args = ap.parse_args(argv)

    idx = os.path.join(args.cache, "index.csv")
    if not os.path.exists(idx):
        raise SystemExit("%s yok" % idx)
    with open(idx, newline="") as fh:
        rows = list(csv.DictReader(fh))

    per = defaultdict(Counter)          # sinif -> kaynak -> sayi
    per_split = defaultdict(Counter)    # split -> kaynak -> sayi
    unknown = []
    for r in rows:
        src = source_of(r.get("record", ""), r.get("path", ""))
        lab = r.get("label_name") or r.get("label") or "?"
        per[lab][src] += 1
        per_split[r.get("split", "?")][src] += 1
        if src in ("?", "Chapman/Ningbo(?)"):
            unknown.append(r.get("record", ""))

    sources = sorted({s for c in per.values() for s in c},
                     key=lambda s: -sum(c[s] for c in per.values()))
    labels = [l for l in ("Normal", "AFIB", "AFL", "LBBB", "RBBB") if l in per]
    labels += [l for l in sorted(per) if l not in labels]

    print("cache: %s  (%d kayit)" % (args.cache, len(rows)))
    print()
    print("SINIF x KAYNAK")
    w = max(14, max((len(s) for s in sources), default=8) + 1)
    print("%-8s" % "sinif" + "".join(("%%%ds" % w) % s for s in sources) + "%9s" % "toplam")
    for lab in labels:
        tot = sum(per[lab].values())
        print("%-8s" % lab
              + "".join(("%%%dd" % w) % per[lab][s] for s in sources)
              + "%9d" % tot)

    if unknown:
        print()
        print("kaynagi COZULEMEYEN kayit: %d" % len(unknown))
        print("  ornek: %s" % ", ".join(unknown[:args.show_unknown]))
        print("  Bu listeyi bana gonder -- ad duzenini eslerim.")

    # ---- karisiklik olcusu: sinif ile kaynak ne kadar ic ice ---------------
    print()
    print("KARISIKLIK (confounding) OLCUSU")
    print("Her sinif icin: baskin kaynagin payi. %100'e yaklasan bir deger,")
    print("o sinifin tek bir hastaneden geldigini soyler.")
    print()
    print("%-8s %-20s %10s" % ("sinif", "baskin kaynak", "pay"))
    dom = {}
    for lab in labels:
        tot = sum(per[lab].values())
        if not tot:
            continue
        s, n = per[lab].most_common(1)[0]
        dom[lab] = (s, n / tot)
        print("%-8s %-20s %9.1f%%" % (lab, s, 100.0 * n / tot))

    risk = []
    if "AFIB" in dom and "AFL" in dom:
        a_src, a_p = dom["AFIB"]
        f_src, f_p = dom["AFL"]
        # Iki sinifin kaynak dagilimlari ortusuyor mu (Bhattacharyya benzeri).
        srcs = set(per["AFIB"]) | set(per["AFL"])
        ta, tf = sum(per["AFIB"].values()), sum(per["AFL"].values())
        overlap = sum(min(per["AFIB"][s] / max(ta, 1),
                          per["AFL"][s] / max(tf, 1)) for s in srcs)
        print()
        print("AFIB ile AFL'nin kaynak dagilimlari ortusmesi: %.1f%%"
              % (100.0 * overlap))
        print("  (%100 = ikisi ayni hastanelerden esit gelir, kisayol yok)")
        print("  (  %0 = tamamen ayri hastaneler, kisayol acik)")
        if overlap < 0.30:
            risk.append(
                "AFIB ve AFL buyuk olcude FARKLI hastanelerden geliyor "
                "(ortusme %.1f%%)." % (100.0 * overlap))

    print()
    print("KARAR")
    if risk:
        for r in risk:
            print("  RISK: %s" % r)
        print()
        print("  Ag 'AFIB mi AFL mi' yerine 'hangi hastane' sorusunu ogreniyor")
        print("  olabilir. Kendi test_public'inde bu ise yarar; sartname md.")
        print("  7.2'deki YENI veri setinde coker.")
        print()
        print("  YAPILACAK: dis veri eklerken sayiyi degil, KAYNAK CESITLILIGINI")
        print("  hedefle. Az bulunan sinifi cok bulundugu hastaneden daha da")
        print("  cogaltmak karisikligi ARTIRIR. Bunun yerine:")
        print("    - AFIB'i Ningbo disindaki kaynaklardan ekle")
        print("    - AFL'yi Ningbo DISINDAKI kaynaklardan ekle (PTB-XL 73,")
        print("      Georgia 186, CPSC-Extra 54, Chapman 445)")
        print("  Amac: her sinif birden fazla hastaneden gelsin.")
        return 1
    print("  Belirgin bir kaynak-sinif karisikligi gorulmedi. Dis veri")
    print("  eklerken yine de sinif dengesini koru.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
