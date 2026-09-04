"""add_inception -- model.py'ye Inception1D/Hybrid presetlerini EKLER.

    python tools/add_inception.py            # uygular ve dogrular
    python tools/add_inception.py --dry-run  # ne yapacagini goster, dokunma
    python tools/add_inception.py --undo     # yedekten geri al

Neden elle yapistirmak yerine betik
-----------------------------------
Eklenecek blok `model.py`'nin **en sonuna** gitmeli: `PRESETS` ve `build_model`
tanimlandiktan sonra. Yanlis yere yapistirilirsa ya NameError verir ya da
sessizce eski `build_model`'i sarmalamaz -- ikincisi kotudur, cunku
`--preset inception` bilinmeyen preset hatasi yerine yanlis mimariyi kurmaya
calisir. Betik yeri kendisi bulur, tekrar calistirilirsa iki kez eklemez ve
sonunda dogrulama yapar.

Ne DEGISMEZ
-----------
Mevcut `PRESETS` girdileri, `ECGNet` ve blok siniflari **hic ellenmez**. Blok
yalnizca dosyanin sonuna eklenir ve `build_model`'i sarmalar: preset
`PRESETS_DIVERSE` icinde degilse cagri aynen eski yola gider. Bu yuzden eski
checkpoint'ler (r18 / wide / w64 / w80 ...) `strict=True` ile yuklenmeye devam
eder -- betik bunu kendisi test eder.

`ecg_preprocess.py`, cache ve on isleme davranisi degismez.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "# FAZ 5 -- mimari cesitliligi (model_diverse.py)"

BLOCK = '''

# --------------------------------------------------------------------------
''' + MARKER + '''
#
# Bu blok EKLEME yapar, hicbir seyi degistirmez: yukaridaki PRESETS girdileri,
# ECGNet ve blok siniflari aynen durur. Bu yuzden eski checkpoint'ler
# (r18 / wide / w64 / w80 ...) bozulmadan yuklenmeye devam eder -- build_model
# bilinmeyen bir preset gormedikce eski yola gider.
#
# model_diverse.py yoksa dosya sessizce eski haliyle calisir.
# --------------------------------------------------------------------------

try:
    from model_diverse import PRESETS_DIVERSE, build_diverse
except ImportError:                                  # eklenti kurulu degil
    PRESETS_DIVERSE = {}
else:
    PRESETS.update(PRESETS_DIVERSE)
    _build_model_resnet = build_model

    def build_model(preset="r18", dropout=0.2, use_features=True, **kwargs):
        if preset in PRESETS_DIVERSE:
            return build_diverse(preset, dropout=dropout,
                                 use_features=use_features, **kwargs)
        return _build_model_resnet(preset, dropout=dropout,
                                   use_features=use_features, **kwargs)
'''


def verify(model_path):
    """Yamadan sonra: yeni presetler kuruluyor mu, eskiler bozuldu mu."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(model_path)))
    for mod in ("model", "model_diverse"):
        sys.modules.pop(mod, None)
    import torch                                     # noqa: PLC0415
    from model import PRESETS, build_model, count_parameters  # noqa: PLC0415

    print()
    print("DOGRULAMA")
    ok = True

    for name in ("inception", "hybrid"):
        if name not in PRESETS:
            print("  KALDI: %s preset'i eklenmemis" % name)
            ok = False
    print("  PRESETS: %s" % ", ".join(PRESETS))

    for name in ("r18", "w64", "inception", "hybrid"):
        if name not in PRESETS:
            continue
        try:
            m = build_model(name).eval()
            with torch.no_grad():
                out = m(torch.zeros(2, 12, 1500), torch.zeros(2, 37))
            print("  %-11s %-11s %11s parametre  cikti %s"
                  % (name, type(m).__name__,
                     "{:,}".format(count_parameters(m)), tuple(out.shape)))
            if tuple(out.shape) != (2, 5):
                print("     KALDI: cikti sekli (2, 5) olmali")
                ok = False
        except Exception as exc:                     # noqa: BLE001
            print("  KALDI: %s kurulamadi -- %s" % (name, exc))
            ok = False

    # Eski checkpoint hala yukleniyor mu -- asil risk bu.
    found = None
    for root, _dirs, files in os.walk(os.path.join(os.path.dirname(
            os.path.abspath(model_path)), "runs")):
        if "best.pt" in files:
            found = os.path.join(root, "best.pt")
            break
    if found:
        try:
            ck = torch.load(found, map_location="cpu", weights_only=False)
            m = build_model(ck["preset"], dropout=ck.get("dropout", 0.2),
                            use_features=ck.get("use_features", True))
            m.load_state_dict(ck["state_dict"], strict=True)
            print("  eski checkpoint (%s, preset=%s) strict=True yuklendi"
                  % (os.path.relpath(found), ck["preset"]))
        except Exception as exc:                     # noqa: BLE001
            print("  KALDI: eski checkpoint yuklenemedi -- %s" % exc)
            ok = False
    else:
        print("  (runs/ altinda checkpoint bulunamadi, uyumluluk testi atlandi)")

    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "model.py"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--undo", action="store_true")
    args = ap.parse_args(argv)

    path = args.model
    backup = path + ".yedek"

    if args.undo:
        if not os.path.exists(backup):
            raise SystemExit("%s yok -- geri alinacak bir sey bulunamadi" % backup)
        shutil.copy2(backup, path)
        print("geri alindi: %s <- %s" % (path, backup))
        return 0

    if not os.path.exists(path):
        raise SystemExit("%s yok" % path)
    src = open(path, encoding="utf-8").read()

    for need in ("PRESETS", "def build_model"):
        if need not in src:
            raise SystemExit("%s icinde %r bulunamadi -- dosya beklenen "
                             "model.py degil" % (path, need))

    if not os.path.exists(os.path.join(HERE, "model_diverse.py")):
        raise SystemExit("model_diverse.py yok -- once onu ecg_train/ "
                         "klasorune koy")

    if MARKER in src:
        print("blok zaten ekli (%s) -- dosyaya dokunulmadi" % path)
        return 0 if verify(path) else 1

    print("hedef  : %s  (%d satir)" % (path, src.count("\n") + 1))
    print("islem  : dosyanin SONUNA %d satirlik blok eklenecek"
          % BLOCK.strip().count("\n"))
    print("yedek  : %s" % backup)
    if args.dry_run:
        print()
        print("--- eklenecek blok ---")
        print(BLOCK)
        print("(--dry-run: hicbir sey yazilmadi)")
        return 0

    shutil.copy2(path, backup)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(BLOCK)
    print("eklendi.")

    if not verify(path):
        print()
        print("DOGRULAMA KALDI -- geri almak icin:")
        print("  python tools/add_inception.py --undo")
        return 1

    print()
    print("all checks passed")
    print()
    print("sonraki:")
    print("  python train.py --preset inception --tag div_inc --only_fold 0 "
          "--epochs 30 --lr 0.002 --no_bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
