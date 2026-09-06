"""source_breakdown -- model sinyale mi bakiyor, kaynak hastaneye mi?

    python tools/source_breakdown.py --cache cache --oof ensemble_oof_prob.npy

Egitim yapmaz, dosya yazmaz, saniyeler surer.

Neden
-----
`data_provenance.py` ciktisi su tabloyu verdi:

    kaynak          AFIB   AFL      n   sadece-kaynak kurali
    MIMIC-IV-ECG     475   471    946   0.502   <- DENGELI, kisayol YOK
    PTB-XL            48    56    104   0.538   <- dengeli
    Chapman          477    22    499   0.956   <- neredeyse hep AFIB
    Ningbo             0   451    451   1.000   <- hep AFL

Sinyale hic bakmayan, yalnizca "bu kayit hangi hastaneden" diye soran bir kural
ikilide **0.7295** tutturuyor. Modelin gercek degeri **0.7376**. Fark 0.008.

Bu tek basina bir kanit degil -- tesadüf olabilir. Ama ayirt edici bir test var:

**MIMIC-IV-ECG kayitlarinda modelin dogrulugu.**

MIMIC dengeli (475 AFIB / 471 AFL), yani orada kaynak bilgisi HICBIR sey
soylemiyor. Model gercekten sinyalden ogrendiyse orada da ~0.74 tutturur.
Yalnizca kaynak imzasini ogrendiyse orada ~0.50'ye duser.

Ne basiliyor
------------
Her kaynak icin: modelin ikili dogrulugu, o kaynakta sadece-kaynak kuralinin
dogrulugu, ve ikisinin farki. Karar, DENGELI kaynaklardaki (MIMIC, PTB-XL)
sayidan okunur -- kisayolun ise yaramadigi tek yer orasi.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_provenance import source_of  # noqa: E402

PAIR = (1, 2)              # AFIB, AFL
BALANCED_GATE = 0.65       # baskin sinif payi bunun altindaysa kaynak "dengeli"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--oof", default="ensemble_oof_prob.npy")
    ap.add_argument("--classes", default="Normal,AFIB,AFL,LBBB,RBBB")
    args = ap.parse_args(argv)

    classes = args.classes.split(",")
    idx_path = os.path.join(args.cache, "index.csv")
    for p in (idx_path, args.oof):
        if not os.path.exists(p):
            raise SystemExit("%s yok" % p)
    with open(idx_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    prob = np.load(args.oof)
    if prob.shape[0] != len(rows):
        raise SystemExit("oof %d satir, cache %d satir -- ayni cache'in OOF'u mu?"
                         % (prob.shape[0], len(rows)))

    y = np.array([int(r["label"]) for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] != "test_public"])
    src = np.array([source_of(r.get("record", ""), r.get("path", "")) for r in rows])
    pred = prob.argmax(1)

    pair = dev[np.isin(y[dev], PAIR)]
    if len(pair) < 50:
        raise SystemExit("AFIB/AFL kaydi cok az (%d)" % len(pair))

    print("gelistirme kumesi: %d kayit, bunlarin %d tanesi AFIB/AFL"
          % (len(dev), len(pair)))
    print("(test_public okunmadi)")
    print()

    # ---- kaynak basina ----------------------------------------------------
    by = defaultdict(list)
    for i in pair:
        by[src[i]].append(i)

    print("AFIB/AFL IKILISI, KAYNAK BASINA")
    print("%-16s %6s %6s %7s %11s %11s %9s"
          % ("kaynak", "AFIB", "AFL", "n", "model", "kaynak-kur.", "fark"))

    rows_out, bal_hit, bal_n = [], 0, 0
    tot_rule = 0
    for s in sorted(by, key=lambda s: -len(by[s])):
        ii = np.array(by[s])
        n_a = int((y[ii] == 1).sum())
        n_f = int((y[ii] == 2).sum())
        n = len(ii)
        acc = float((pred[ii] == y[ii]).mean())
        rule = max(n_a, n_f) / n                 # o kaynakta cogunluk kurali
        tot_rule += max(n_a, n_f)
        balanced = rule < BALANCED_GATE
        rows_out.append((s, n, acc, rule, balanced))
        if balanced:
            bal_hit += int((pred[ii] == y[ii]).sum())
            bal_n += n
        print("%-16s %6d %6d %7d %11.4f %11.4f %+9.4f%s"
              % (s, n_a, n_f, n, acc, rule, acc - rule,
                 "   <- DENGELI" if balanced else ""))

    overall = float((pred[pair] == y[pair]).mean())
    rule_overall = tot_rule / len(pair)
    print()
    print("%-16s %13s %7d %11.4f %11.4f %+9.4f"
          % ("TOPLAM", "", len(pair), overall, rule_overall, overall - rule_overall))

    # ---- karar ------------------------------------------------------------
    print()
    print("KARAR")
    if bal_n < 100:
        print("  Dengeli kaynak yok ya da cok kucuk (%d kayit) -- bu test" % bal_n)
        print("  sonuc veremiyor. Kaynak dagilimini elle incele.")
        return 2

    bal_acc = bal_hit / bal_n
    print("  Dengeli kaynaklar (kisayolun ISE YARAMADIGI tek yer):")
    print("    %d kayit, model dogrulugu %.4f" % (bal_n, bal_acc))
    print()
    print("  Dogru kiyas SANS SEVIYESIDIR (0.50), genel dogruluk degil --")
    print("  genel deger zaten kolay kaynaklarla sismis durumda.")
    print()
    print("  ** Yeni bir veri setinde (sartname md. 7.2) kaynak ipucu")
    print("     olmayacagi icin BEKLENEN ikili dogruluk ~ %.4f'tir," % bal_acc)
    print("     gelistirmede gordugun %.4f degil. **" % overall)
    print()
    if bal_acc < 0.58:
        print("  CIDDI: dengeli kaynaklarda model neredeyse sansa dusuyor.")
        print("  Genel skorun buyuk olcude 'hangi hastane' kisayolundan")
        print("  geliyor. Final degerlendirmede bu kisayol YOK.")
        print()
        print("  YAPILACAK: kaynak-sinif korelasyonunu kir -- Ningbo'dan AFIB,")
        print("  Chapman'dan AFL ekle; her sinif birden fazla kaynaktan gelsin.")
        return 1
    if bal_acc < 0.65:
        print("  SINIRDA: sanstan belirgin yukarida ama zayif. Gercek sinyal")
        print("  ogrenmesi var, kisayolun da payi var.")
        print("  Dis veri eklerken kaynak cesitliligini oncelikle hedefle.")
        return 1
    print("  TEMIZ: model kaynak ipucunun HIC olmadigi kayitlarda %.4f" % bal_acc)
    print("  tutturuyor. Ogrendigi sey sinyal, hastane imzasi degil.")
    print("  Sadece-kaynak kuralinin %.4f'e ulasmasi bir TESADUF --" % rule_overall)
    print("  kaynak dagilimindan dogan bir artefakt, modelin davranisi degil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
