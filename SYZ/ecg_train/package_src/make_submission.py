"""make_submission -- final JSON sonuc dosyasini uretir ve DOGRULAR.

    python make_submission.py --root <test verisi klasoru> \
        --team-name "tkt-26" --team-id TEAM_001 --application-id BASVURU_001

    python make_submission.py --validate TEAM_001_FINAL.json --root <test klasoru>

16 Eylul finalinde teslim edilecek dosya budur. Kilavuzun 3. bolumu cok kati:
tek bir hatali alan dosyayi gecersiz kilar ve sure durmaz.

Kilavuzdan cikarilan ZORUNLU kurallar (madde 3 ve 3.1)
------------------------------------------------------
  * UTF-8, `.json` uzantisi, onerilen ad: TEAM_<TAKIM_ID>_FINAL.json
  * ust alanlar: team_name, team_id, application_id, competition_level
  * competition_level lise icin tam olarak "LISE"
  * sinif adlari YALNIZCA ve BUYUK HARFLE: NORMAL, AFIB, AFL, LBBB, RBBB
  * her test ornegi `predictions` icinde TAM OLARAK BIR KEZ
  * hicbir id eksik olmayacak, fazladan id eklenmeyecek, id degistirilmeyecek
  * probabilities ZORUNLU ve TUM siniflar icin deger icerecek
  * olasiliklar 0-1 arasinda, her kayit icin toplami 1 (+-0.001 tolerans)
  * ondalik ayirici NOKTA
  * NaN, Infinity, bos olasilik, yorum satiri, gecersiz sinif adi, tekrar eden
    id veya bozuk JSON KABUL EDILMEZ

Neden olasiliklari ezme
-----------------------
Kilavuz madde 4: "Takimlar arasi esitlik olmasi durumunda esitligin bozulmasi
icin takimlar tarafindan saglanan sinif olasiliklarindan hesaplanan **PR-AUC**
metrigi dikkate alinacaktir." Yani olasiliklar yalnizca bicimsel bir zorunluluk
degil, esitlik bozucu. Sert 0/1 yazma -- modelin gercek olasiliklarini ver.

Internet YOK
------------
Kilavuz: "Yarisma Test asamasinda internet erisimi yasaktir." Bu betik yalnizca
paketin icindeki dosyalari ve standart kutuphaneyi + numpy + onnxruntime
kullanir. Hicbir sey indirmez.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CLASS_ORDER = ("NORMAL", "AFIB", "AFL", "LBBB", "RBBB")
LEVEL = "LISE"
TOL = 0.001


# --------------------------------------------------------------------------
# dogrulama -- teslimden once HER SEY buradan gecmeli
# --------------------------------------------------------------------------

def validate(doc, expected_ids=None):
    """Kilavuz madde 3'e gore denetle. Hata listesi dondurur (bos = temiz)."""
    err = []

    if not isinstance(doc, dict):
        return ["kok nesne bir JSON object degil"]

    for field in ("team_name", "team_id", "application_id", "competition_level"):
        v = doc.get(field)
        if not isinstance(v, str) or not v.strip():
            err.append("ust alan eksik veya bos: %s" % field)
    if doc.get("competition_level") != LEVEL:
        err.append("competition_level %r olmali, %r bulundu"
                   % (LEVEL, doc.get("competition_level")))

    preds = doc.get("predictions")
    if not isinstance(preds, list) or not preds:
        return err + ["predictions bir liste degil ya da bos"]

    seen = {}
    for n, p in enumerate(preds):
        where = "predictions[%d]" % n
        if not isinstance(p, dict):
            err.append("%s bir object degil" % where)
            continue

        rid = p.get("id")
        if not isinstance(rid, str) or not rid.strip():
            err.append("%s: id eksik veya metin degil" % where)
        elif rid in seen:
            err.append("%s: TEKRAR EDEN id %r (ilk gorulme %d)"
                       % (where, rid, seen[rid]))
        else:
            seen[rid] = n

        cls = p.get("predicted_class")
        if cls not in CLASS_ORDER:
            err.append("%s: gecersiz predicted_class %r (izinli: %s)"
                       % (where, cls, ", ".join(CLASS_ORDER)))

        pr = p.get("probabilities")
        if not isinstance(pr, dict):
            err.append("%s: probabilities eksik" % where)
            continue
        missing = [c for c in CLASS_ORDER if c not in pr]
        extra = [c for c in pr if c not in CLASS_ORDER]
        if missing:
            err.append("%s: eksik sinif(lar) %s" % (where, ", ".join(missing)))
        if extra:
            err.append("%s: fazladan anahtar %s" % (where, ", ".join(map(str, extra))))
        total = 0.0
        for c in CLASS_ORDER:
            v = pr.get(c)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                err.append("%s: %s sayi degil (%r)" % (where, c, v))
                continue
            if math.isnan(v) or math.isinf(v):
                err.append("%s: %s NaN/Infinity" % (where, c))
                continue
            if v < 0.0 or v > 1.0:
                err.append("%s: %s 0-1 disinda (%r)" % (where, c, v))
            total += float(v)
        if abs(total - 1.0) > TOL:
            err.append("%s: olasilik toplami %.6f (1.0 +-%.3f olmali)"
                       % (where, total, TOL))

        # predicted_class en yuksek olasilikla tutarli mi (uyari niteliginde)
        if cls in CLASS_ORDER and isinstance(pr, dict):
            try:
                top = max(CLASS_ORDER, key=lambda c: float(pr.get(c, -1)))
                if top != cls:
                    err.append("%s: predicted_class %r ama en yuksek olasilik %r"
                               % (where, cls, top))
            except Exception:                    # noqa: BLE001
                pass

    if expected_ids is not None:
        exp = set(expected_ids)
        got = set(seen)
        for rid in sorted(exp - got):
            err.append("EKSIK id: %s" % rid)
        for rid in sorted(got - exp):
            err.append("FAZLADAN id: %s" % rid)

    return err


def load_json_strict(path):
    """NaN/Infinity iceren dosyayi sessizce kabul etme -- kilavuz yasakliyor."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    def _no_const(x):
        raise ValueError("JSON icinde %s bulundu -- kilavuz yasakliyor" % x)

    return json.loads(text, parse_constant=_no_const)


# --------------------------------------------------------------------------

def find_records(root):
    """Test klasorundeki kayitlari bul: (id, header_yolu)."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".hea"):
                rid = os.path.splitext(fn)[0]
                out.setdefault(rid, os.path.join(dirpath, fn))
    return out


def read_id_list(path):
    """Kuyruk dosyasindan id listesi (csv ya da duz metin, ilk kolon)."""
    ids = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        import csv as _csv
        sniff = fh.read(2048)
        fh.seek(0)
        sep = ";" if sniff.count(";") > sniff.count(",") else ","
        rdr = _csv.reader(fh, delimiter=sep)
        rows = list(rdr)
    if not rows:
        return ids
    start = 0
    head = [c.strip().lower() for c in rows[0]]
    col = 0
    for cand in ("id", "record", "kayit", "filename", "file"):
        if cand in head:
            col = head.index(cand)
            start = 1
            break
    else:
        if any(h in ("predicted_class", "label", "diagnosis") for h in head):
            start = 1
    for r in rows[start:]:
        if r and str(r[col]).strip():
            ids.append(os.path.splitext(str(r[col]).strip())[0])
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="", help="test verisi kok klasoru")
    ap.add_argument("--ids", default="",
                    help="id listesi (csv/txt). Verilmezse --root taranir.")
    ap.add_argument("--models", default=HERE, help="paket klasoru")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--team-name", default="")
    ap.add_argument("--team-id", default="")
    ap.add_argument("--application-id", default="")
    ap.add_argument("--out", default="", help="cikti json (bos = otomatik ad)")
    ap.add_argument("--max-fail", type=int, default=-1,
                    help="on islemede kac kayit duserse teslim REDDEDILSIN "
                         "(-1 = otomatik: kayitlarin %%1'i, en az 5)")
    ap.add_argument("--validate", default="",
                    help="uretme, VAR OLAN bir json'u denetle")
    args = ap.parse_args(argv)

    # ---- yalnizca dogrulama modu ------------------------------------------
    if args.validate:
        try:
            doc = load_json_strict(args.validate)
        except Exception as exc:                 # noqa: BLE001
            print("JSON okunamadi: %s" % exc)
            return 1
        exp = None
        if args.ids:
            exp = read_id_list(args.ids)
        elif args.root:
            exp = sorted(find_records(args.root))
        err = validate(doc, exp)
        n = len(doc.get("predictions") or [])
        print("dosya : %s" % args.validate)
        print("kayit : %d" % n)
        if exp is not None:
            print("beklenen id sayisi: %d" % len(exp))
        if err:
            print()
            print("%d HATA:" % len(err))
            for e in err[:60]:
                print("  - %s" % e)
            if len(err) > 60:
                print("  ... ve %d hata daha" % (len(err) - 60))
            return 1
        print()
        print("all checks passed")
        return 0

    # ---- uretme modu ------------------------------------------------------
    for req in ("team_name", "team_id", "application_id"):
        if not getattr(args, req.replace("-", "_")):
            raise SystemExit("--%s zorunlu" % req.replace("_", "-"))
    if not args.root:
        raise SystemExit("--root zorunlu")

    import inspect                                # noqa: PLC0415
    import predict as pr                          # paketin kendi cikarim kodu
    import ecg_preprocess as ep                   # noqa: PLC0415

    def bundle_attr(b, name, manifest_keys, fallback):
        """Bundle'in ic alanlarina BAGIMLI OLMA.

        Paketin `predict.py`'si farkli bir surumden olabilir (ornegin
        `n_features` / `extra_features` alanlari yok, manifest anahtarlari
        Turkce). Once nesneden, sonra manifestten, en son varsayilandan al.
        """
        v = getattr(b, name, None)
        if v is not None:
            return v
        man = getattr(b, "manifest", {}) or {}
        for k in manifest_keys:
            if man.get(k) not in (None, ""):
                return man[k]
        return fallback

    def call_load_one(path, target_fs, extra):
        """load_one imzasi surumden surume degisti -- kac argument aliyorsa o."""
        try:
            n = len(inspect.signature(pr.load_one).parameters)
        except (TypeError, ValueError):
            n = 2
        return pr.load_one(path, target_fs) if n < 3 \
            else pr.load_one(path, target_fs, extra)

    ids_paths = find_records(args.root)
    if args.ids:
        want = read_id_list(args.ids)
        missing = [i for i in want if i not in ids_paths]
        if missing:
            raise SystemExit("id listesindeki %d kayit --root altinda yok, "
                             "ornek: %s" % (len(missing), ", ".join(missing[:5])))
        order = want
    else:
        order = sorted(ids_paths)
    if not order:
        raise SystemExit("%s altinda .hea kaydi bulunamadi" % args.root)

    print("test kaydi     : %d" % len(order))
    bundle = pr.Bundle(args.models, args.threads or None)
    man = getattr(bundle, "manifest", {}) or {}
    target_fs = man.get("target_fs") or getattr(ep, "TARGET_FS", None)
    n_feat = int(bundle_attr(bundle, "n_features", ("n_features", "ozellik_sayisi"),
                             len(getattr(ep, "FEATURE_NAMES", range(37)))))
    input_len = int(bundle_attr(bundle, "input_len", ("input_len", "giris_uzunlugu"),
                                getattr(ep, "TARGET_LEN", 1500)))
    n_lead = int(getattr(ep, "N_LEADS", 12))
    extra = bool(getattr(bundle, "extra_features", False))

    print("model          : %d ONNX grafigi" % len(bundle))
    print("ozellik sayisi : %d%s" % (n_feat, "  (+artik olcumleri)" if extra else ""))

    # ---- ON KONTROL: tek kayit isle, gercek sekilleri OGREN ----------------
    # Manifest ile paketin `predict.py`'si farkli surumlerden olabilir. Ozellik
    # uzunlugunu manifestten VARSAYMAK, her kaydin sessizce dusmesine ve
    # yapisal olarak gecerli ama ICI COP bir teslim dosyasina yol acar.
    try:
        x0, f0 = call_load_one(ids_paths[order[0]], target_fs, extra)
    except Exception as exc:                     # noqa: BLE001
        raise SystemExit("ILK KAYIT ISLENEMEDI (%s): %s\n"
                         "  Paket ile test verisi uyusmuyor olabilir."
                         % (order[0], exc))
    x0 = np.asarray(x0)
    f0 = np.asarray(f0).ravel()
    if x0.shape[0] != n_lead or x0.shape[1] != input_len:
        print("  NOT: sinyal sekli %s, manifest (%d, %d) diyordu -- sinyale uyuldu"
              % (x0.shape, n_lead, input_len))
        n_lead, input_len = int(x0.shape[0]), int(x0.shape[1])
    if f0.size != n_feat:
        print("  NOT: ozellik uzunlugu %d, manifest %d diyordu -- sinyale uyuldu"
              % (f0.size, n_feat))
        n_feat = int(f0.size)

    signals = np.zeros((len(order), n_lead, input_len), dtype=np.float32)
    feats = np.zeros((len(order), n_feat), dtype=np.float32)
    failed = []
    t0 = time.time()
    for i, rid in enumerate(order):
        try:
            x, f = call_load_one(ids_paths[rid], target_fs, extra)
            signals[i], feats[i] = x, f
        except Exception as exc:                 # noqa: BLE001
            failed.append((rid, "%s: %s" % (type(exc).__name__, exc)))
        if (i + 1) % 100 == 0 or i + 1 == len(order):
            print("  on isleme %d/%d  %.0f sn" % (i + 1, len(order), time.time() - t0),
                  flush=True)

    prob = bundle.predict_proba(signals, feats)
    elapsed = time.time() - t0
    print("toplam %.1f sn  (%.0f ms/kayit)" % (elapsed, 1000 * elapsed / len(order)))

    limit = args.max_fail if args.max_fail >= 0 else max(5, len(order) // 100)
    if failed:
        print()
        print("UYARI: %d kayit on islenemedi (esik %d)" % (len(failed), limit))
        for rid, e in failed[:5]:
            print("   %s  %s" % (rid, e))
        if len(failed) > limit:
            raise SystemExit(
                "\nTESLIM REDDEDILDI: %d/%d kayit on islenemedi.\n"
                "  Bu kayitlara esit olasilik yazmak, yapisal olarak gecerli ama\n"
                "  ICI COP bir dosya uretir. Once sebebi bul.\n"
                "  Bilerek devam etmek icin: --max-fail %d"
                % (len(failed), len(order), len(failed)))
        print("  esigin altinda -- bu kayitlara esit olasilik yaziliyor")
        # Bu kayitlar icin duzgun bir sey yapmak sart: id EKSIK BIRAKILAMAZ.
        fail_idx = {order.index(r) for r, _e in failed}
        for i in fail_idx:
            prob[i] = 1.0 / len(CLASS_ORDER)

    # ---- JSON kur ---------------------------------------------------------
    pkg_classes = [str(c).upper() for c in
                   bundle_attr(bundle, "classes", ("classes", "siniflar"),
                               list(getattr(ep, "CLASSES", CLASS_ORDER)))]
    if sorted(pkg_classes) != sorted(CLASS_ORDER):
        raise SystemExit("paket siniflari %s, kilavuz %s bekliyor"
                         % (pkg_classes, list(CLASS_ORDER)))

    predictions = []
    for i, rid in enumerate(order):
        p = np.asarray(prob[i], dtype=np.float64)
        p = np.clip(p, 0.0, 1.0)
        s = p.sum()
        p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        # 6 haneye yuvarla, sonra toplam farkini en buyuk sinifa yedir --
        # yuvarlama yuzunden toplam 1'den kaymasin.
        r = {pkg_classes[k]: round(float(p[k]), 6) for k in range(len(p))}
        drift = round(1.0 - sum(r.values()), 6)
        top = max(r, key=r.get)
        r[top] = round(r[top] + drift, 6)
        predictions.append({
            "id": rid,
            "predicted_class": top,
            "probabilities": {c: r[c] for c in CLASS_ORDER},
        })

    doc = {
        "team_name": args.team_name,
        "team_id": args.team_id,
        "application_id": args.application_id,
        "competition_level": LEVEL,
        "predictions": predictions,
    }

    err = validate(doc, order)
    if err:
        print()
        print("DOGRULAMA BASARISIZ -- dosya YAZILMADI (%d hata):" % len(err))
        for e in err[:40]:
            print("  - %s" % e)
        return 1

    # Kilavuz onerisi: TEAM_<TAKIM_ID>_FINAL.json. team_id zaten "TEAM_"
    # ile basliyorsa oneki iki kez yazma -> TEAM_TEAM_001_FINAL.json olmasin.
    tid = args.team_id
    stem = tid if tid.upper().startswith("TEAM_") else "TEAM_%s" % tid
    out = args.out or ("%s_FINAL.json" % stem)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, allow_nan=False,
                  separators=(",", ":"))

    # Yazdiktan sonra DISKTEN geri okuyup tekrar dogrula.
    doc2 = load_json_strict(out)
    err2 = validate(doc2, order)
    if err2:
        print("YAZILDIKTAN SONRA dogrulama basarisiz:")
        for e in err2[:20]:
            print("  - %s" % e)
        return 1

    size = os.path.getsize(out)
    print()
    print("yazildi: %s  (%d kayit, %.1f KB)" % (out, len(predictions), size / 1024))
    print("diskten geri okundu ve tekrar dogrulandi.")
    print()
    print("sinif dagilimi:")
    from collections import Counter
    cnt = Counter(p["predicted_class"] for p in predictions)
    for c in CLASS_ORDER:
        print("  %-7s %6d" % (c, cnt[c]))
    top_share = max(cnt.values()) / len(predictions)
    if top_share > 0.90:
        print()
        print("  !! DIKKAT: tahminlerin %%%.0f'i TEK sinifta." % (100 * top_share))
        print("  !! Bu genellikle modelin degil, BORU HATTININ bozuk oldugunu")
        print("  !! gosterir (ozellik uzunlugu, on isleme, yanlis paket).")
        print("  !! Teslim etmeden once sebebini bul.")
    print()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
