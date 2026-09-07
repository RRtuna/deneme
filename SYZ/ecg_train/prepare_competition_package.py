"""prepare_competition_package -- yarisma teslim klasorunu guvenle hazirla.

    python prepare_competition_package.py
    python prepare_competition_package.py --force
    python prepare_competition_package.py --check

Ne yapar
--------
1. Dogrulanmis final release paketini `competition_package` olarak KOPYALAR.
2. `make_submission.py`'yi `predict.py` ve `manifest.json` ile ayni seviyeye koyar.
3. Kritik dosya eksikse HATA verir ve hicbir sey birakmaz.
4. Sonunda klasor yapisini, model sayisini ve SHA-256 saglamalarini dogrular.
5. Tekrar calistirilirsa guvenlidir: var olan klasoru silmez, eksikleri tamamlar
   ve dogrular. Sifirdan kurmak icin `--force`.

Degismezler
-----------
* Kaynak paket **asla** degistirilmez. Bu betik kaynaga tek bir bayt yazmaz.
* Yarisma test verisi kopyalanmaz. Ham kayit uzantilari (.mat/.dat/.hea) ve
  split csv'leri kopyalama disi tutulur; boyle bir dosya kaynakta bulunursa
  atlanir ve raporlanir. (Kilavuz md. 6: "Yarisma test verisi model paketinin
  parcasi olarak kopyalanmamali.")
* Mevcut ONNX dosyalarina dokunulmaz; yalnizca kopyalanir ve sagalamalari
  manifest ile karsilastirilir.

Windows CMD icin yazildi. Sembolik baglanti kullanmaz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

DEFAULT_SOURCE = os.path.join("release", "verified_20model_final_2026-09-03",
                              "package")
DEFAULT_DEST = "competition_package"

# Pakette bulunmasi ZORUNLU dosyalar. Biri eksikse kurulum durur.
REQUIRED = ("manifest.json", "predict.py", "ecg_preprocess.py", "wfdb_lite.py")

# Yarisma verisi ya da turevi olabilecek her sey -- kopyalanmaz.
BLOCKED_EXT = (".mat", ".dat", ".hea", ".npy", ".npz", ".pt", ".pth")
BLOCKED_NAMES = ("train.csv", "validation.csv", "test_public.csv", "index.csv")
SKIP_DIRS = ("__pycache__", ".git", ".ipynb_checkpoints")

SUBMISSION = "make_submission.py"


def die(msg):
    print()
    print("HATA: %s" % msg)
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_submission(explicit, script_dir):
    """make_submission.py'yi bul: once --submission, sonra olasi yerler."""
    if explicit:
        if not os.path.exists(explicit):
            die("--submission ile verilen dosya yok: %s" % explicit)
        return explicit
    candidates = [
        os.path.join(script_dir, SUBMISSION),
        os.path.join(script_dir, "package_src", SUBMISSION),
        os.path.join(os.getcwd(), SUBMISSION),
        os.path.join(os.getcwd(), "package_src", SUBMISSION),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    die("%s bulunamadi. Su yerlerden birine koy ya da --submission ile yolunu "
        "ver:\n  %s" % (SUBMISSION, "\n  ".join(candidates)))


def blocked(name):
    low = name.lower()
    return low.endswith(BLOCKED_EXT) or low in BLOCKED_NAMES


def copy_tree(src, dst):
    """Kaynagi hedefe kopyala. Engelli dosyalari atla, atlananlari dondur."""
    skipped, copied = [], 0
    for dirpath, dirnames, files in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, src)
        out_dir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out_dir, exist_ok=True)
        for fn in files:
            s = os.path.join(dirpath, fn)
            if blocked(fn) and not fn.lower().endswith(".onnx"):
                skipped.append(os.path.relpath(s, src))
                continue
            d = os.path.join(out_dir, fn)
            if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
                continue                        # zaten var, ayni boyutta
            shutil.copy2(s, d)
            copied += 1
    return copied, skipped


def write_docs(dest, manifest):
    """Kilavuz md. 6'nin istedigi requirements.txt ve README -- yoksa uret."""
    made = []
    req = os.path.join(dest, "requirements.txt")
    if not os.path.exists(req):
        with open(req, "w", encoding="utf-8") as fh:
            fh.write("# TEKNOFEST 2026 -- EKG siniflandirma paketi\n")
            fh.write("# PyTorch GEREKMEZ. Yalnizca bu ikisi:\n")
            fh.write("numpy>=1.21\n")
            fh.write("onnxruntime>=1.14\n")
        made.append("requirements.txt")

    readme = os.path.join(dest, "README_TESLIM.md")
    if not os.path.exists(readme):
        n = len(manifest.get("models") or [])
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write("""# Yarisma teslim paketi

12 derivasyonlu EKG kayitlarini 5 sinifa ayiran ONNX modeli.
**PyTorch gerekmez, internet gerekmez.**

## Kurulum

    pip install -r requirements.txt

Yalnizca `numpy` ve `onnxruntime` gerekir.

## Sonuc JSON'u uretme

    python make_submission.py --root <TEST_VERISI_KLASORU> ^
        --team-name "<TAKIM ADI>" --team-id <TEAM_ID> --application-id <BASVURU_ID>

Cikti: `TEAM_<TEAM_ID>_FINAL.json` (bu klasorde).

`--root` altindaki tum `.hea` kayitlari taranir. Belirli bir id listesi
kullanmak icin `--ids <liste.csv>` ekle.

## Uretilen dosyayi denetleme

    python make_submission.py --validate TEAM_<TEAM_ID>_FINAL.json --root <TEST_VERISI_KLASORU>

Basari durumunda son satir: `all checks passed`

## Tek kayit / toplu skor

    python predict.py <kayit.hea>
    python predict.py --batch <liste.csv> --root <VERI_KOKU>

## Paket icerigi

    models/           %d ONNX grafigi (int8)
    manifest.json     uyeler, agirliklar, SHA-256 saglamalar
    predict.py        onnxruntime ile cikarim
    make_submission.py  final JSON ureteci ve dogrulayicisi
    ecg_preprocess.py Egitimdekiyle BIREBIR ayni on isleme
    wfdb_lite.py      saf numpy WFDB okuyucu
    requirements.txt  bagimliliklar

## Not

Yarisma test verisi bu pakete dahil DEGILDIR ve edilmemelidir.
""" % n)
        made.append("README_TESLIM.md")
    return made


def verify(dest, source, strict_sha=True):
    """Yapiyi, model sayisini ve saglamalari denetle. Hata listesi dondurur."""
    errs, warns = [], []

    for f in REQUIRED + (SUBMISSION,):
        if not os.path.exists(os.path.join(dest, f)):
            errs.append("eksik dosya: %s" % f)
    if errs:
        return errs, warns

    models_dir = os.path.join(dest, "models")
    if not os.path.isdir(models_dir):
        return ["models/ klasoru yok"], warns

    on_disk = sorted(f for f in os.listdir(models_dir)
                     if f.lower().endswith(".onnx"))
    try:
        manifest = json.load(open(os.path.join(dest, "manifest.json"),
                                  encoding="utf-8"))
    except Exception as exc:                     # noqa: BLE001
        return ["manifest.json okunamadi: %s" % exc], warns

    entries = manifest.get("models") or []
    if not entries:
        errs.append("manifest.json icinde model listesi yok")

    listed = []
    for e in entries:
        rel = (e.get("file") or "").replace("\\", "/")
        if not rel:
            errs.append("manifest girdisinde 'file' yok: %s" % e.get("member"))
            continue
        listed.append(os.path.basename(rel))
        p = os.path.join(dest, *rel.split("/"))
        if not os.path.exists(p):
            errs.append("manifest'te var ama diskte yok: %s" % rel)
            continue
        want = e.get("sha256")
        if want:
            got = sha256(p)
            if got != want:
                (errs if strict_sha else warns).append(
                    "SAGLAMA TUTMUYOR: %s\n      manifest %s\n      dosya   %s"
                    % (rel, want, got))

    extra = sorted(set(on_disk) - set(listed))
    for x in extra:
        warns.append("models/ icinde manifest'te olmayan dosya: %s" % x)

    # Kaynakla hedef ayni sayida ONNX tasiyor mu
    src_models = os.path.join(source, "models")
    if os.path.isdir(src_models):
        n_src = len([f for f in os.listdir(src_models)
                     if f.lower().endswith(".onnx")])
        if n_src != len(on_disk):
            errs.append("kaynakta %d ONNX var, kopyada %d" % (n_src, len(on_disk)))

    return errs, warns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--submission", default="",
                    help="make_submission.py yolu (bos = otomatik ara)")
    ap.add_argument("--force", action="store_true",
                    help="hedef klasoru SIL ve sifirdan kur")
    ap.add_argument("--check", action="store_true",
                    help="hicbir sey kopyalama, yalnizca var olani dogrula")
    ap.add_argument("--no-docs", action="store_true",
                    help="requirements.txt / README uretme")
    args = ap.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.abspath(args.source)
    dest = os.path.abspath(args.dest)

    print("kaynak : %s" % source)
    print("hedef  : %s" % dest)
    print()

    if os.path.normcase(source) == os.path.normcase(dest):
        die("kaynak ve hedef ayni klasor olamaz")
    if os.path.normcase(dest).startswith(os.path.normcase(source) + os.sep):
        die("hedef kaynagin ICINDE olamaz")

    # ---- salt dogrulama ---------------------------------------------------
    if args.check:
        if not os.path.isdir(dest):
            die("%s yok -- once --check olmadan calistir" % dest)
        errs, warns = verify(dest, source)
        for w in warns:
            print("  UYARI: %s" % w)
        if errs:
            for e in errs:
                print("  HATA : %s" % e)
            return 1
        print("all checks passed")
        return 0

    # ---- kaynak denetimi --------------------------------------------------
    if not os.path.isdir(source):
        die("kaynak paket bulunamadi: %s\n"
            "  --source ile dogru yolu ver." % source)
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(source, f))]
    if missing:
        die("kaynak pakette su zorunlu dosyalar YOK: %s\n"
            "  Yanlis klasoru mu gosteriyorsun?" % ", ".join(missing))
    src_models = os.path.join(source, "models")
    if not os.path.isdir(src_models):
        die("kaynakta models/ klasoru yok")
    n_src = len([f for f in os.listdir(src_models) if f.lower().endswith(".onnx")])
    if n_src == 0:
        die("kaynak models/ icinde hic .onnx yok")
    print("kaynak dogrulandi: %d ONNX grafigi, zorunlu dosyalarin hepsi var" % n_src)

    sub_src = find_submission(args.submission, script_dir)
    print("make_submission.py: %s" % sub_src)

    # ---- kopyalama --------------------------------------------------------
    if args.force and os.path.isdir(dest):
        print()
        print("--force: mevcut %s siliniyor" % dest)
        shutil.rmtree(dest)

    fresh = not os.path.isdir(dest)
    print()
    print("%s..." % ("kopyalaniyor" if fresh else "eksikler tamamlaniyor"))
    copied, skipped = copy_tree(source, dest)
    print("  %d dosya kopyalandi" % copied)
    if skipped:
        print("  %d dosya ATLANDI (yarisma verisi olabilir):" % len(skipped))
        for s in skipped[:10]:
            print("     %s" % s)
        if len(skipped) > 10:
            print("     ... ve %d dosya daha" % (len(skipped) - 10))

    # make_submission.py -- predict.py ile ayni seviyeye
    sub_dst = os.path.join(dest, SUBMISSION)
    if (not os.path.exists(sub_dst)
            or sha256(sub_src) != sha256(sub_dst)):
        shutil.copy2(sub_src, sub_dst)
        print("  %s yerlestirildi" % SUBMISSION)
    else:
        print("  %s zaten guncel" % SUBMISSION)

    manifest = json.load(open(os.path.join(dest, "manifest.json"),
                              encoding="utf-8"))
    if not args.no_docs:
        made = write_docs(dest, manifest)
        for m in made:
            print("  %s uretildi" % m)

    # ---- dogrulama --------------------------------------------------------
    print()
    print("DOGRULAMA")
    errs, warns = verify(dest, source)
    for w in warns:
        print("  UYARI: %s" % w)
    if errs:
        for e in errs:
            print("  HATA : %s" % e)
        print()
        print("Paket EKSIK ya da BOZUK. --force ile sifirdan kurmayi dene.")
        return 1

    models_dir = os.path.join(dest, "models")
    n_onnx = len([f for f in os.listdir(models_dir) if f.lower().endswith(".onnx")])
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _d, fs in os.walk(dest) for f in fs)

    print("  zorunlu dosyalarin hepsi yerinde")
    print("  models/ : %d ONNX, manifest ile ayni" % n_onnx)
    print("  SHA-256 : %d model dogrulandi" % len(manifest.get("models") or []))
    print("  toplam boyut: %.1f MB" % (total / 1e6))

    # ---- klasor yapisi ----------------------------------------------------
    print()
    print("KLASOR YAPISI")
    for name in sorted(os.listdir(dest)):
        p = os.path.join(dest, name)
        if os.path.isdir(p):
            k = len(os.listdir(p))
            print("  %-24s <klasor, %d dosya>" % (name + os.sep, k))
        else:
            print("  %-24s %8.1f KB" % (name, os.path.getsize(p) / 1024))

    # ---- son kontrol: istenen komut gercekten calisiyor mu ----------------
    print()
    print("SON KONTROL: python make_submission.py --help")
    try:
        r = subprocess.run([sys.executable, SUBMISSION, "--help"],
                           cwd=dest, capture_output=True, text=True, timeout=120)
    except Exception as exc:                     # noqa: BLE001
        print("  calistirilamadi: %s" % exc)
        return 1
    if r.returncode != 0:
        print("  BASARISIZ (cikis kodu %d)" % r.returncode)
        print((r.stderr or r.stdout or "")[-1500:])
        return 1
    print("  calisiyor.")

    print()
    print("all checks passed")
    print()
    print("Simdi su komutlar calisir:")
    print("  cd %s" % args.dest)
    print("  python make_submission.py --help")
    print()
    print("Gercek teslim komutu (team_id ve application_id'yi KYS'den al):")
    print("  python make_submission.py --root <TEST_KLASORU> ^")
    print("      --team-name \"tkt-26\" --team-id TEAM_XXX --application-id BASVURU_XXX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
