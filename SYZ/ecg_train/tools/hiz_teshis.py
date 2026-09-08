"""hiz_teshis -- 7357 ms/kayit nereye gidiyor?

    cd competition_package
    python hiz_teshis.py --root "D:\\TUNA_ISPIR\\Documents\\Claude\\Projects\\SYZ"

Bu betik HICBIR SEYI DEGISTIRMEZ. Yalnizca olcer ve teshis koyar.
predict.py, manifest.json ve ONNX modellere dokunmaz.

Olculen dort sey
----------------
  1. sessions()      ilk ve ikinci cagri  -> oturumlar onbelleklenmis mi?
  2. predict_record  ayni kayitta iki kez -> kayit basina sabit maliyet var mi?
  3. predict_arrays  hazir dizide         -> saf cikarim ne kadar?
  4. WFDB okuma      (varsa)              -> disk + ayristirma ne kadar?

Cikan tablo, hangi hizlandirmanin ise yarayacagini soyler.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())


def find_one_record(root):
    """root altinda ilk .hea kaydini bul."""
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith(".hea"):
                return os.path.join(dirpath, fn)
    return None


def timeit(fn, *a, **k):
    t = time.perf_counter()
    try:
        out = fn(*a, **k)
        return time.perf_counter() - t, out, None
    except Exception as exc:                     # noqa: BLE001
        return time.perf_counter() - t, None, "%s: %s" % (type(exc).__name__, exc)


def sn(x):
    return "%8.3f sn" % x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="", help="test verisi kok klasoru")
    ap.add_argument("--record", default="", help="tek bir .hea yolu (root yerine)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="predict_record kac kez tekrarlansin (varsayilan 3)")
    args = ap.parse_args(argv)

    if args.record:
        rec = args.record
    elif args.root:
        rec = find_one_record(args.root)
        if not rec:
            raise SystemExit("%s altinda .hea bulunamadi" % args.root)
    else:
        raise SystemExit("--root veya --record ver")
    if not os.path.exists(rec):
        raise SystemExit("kayit yok: %s" % rec)

    print("kayit  : %s" % rec)
    print("python : %s" % sys.version.split()[0])

    t, _, err = timeit(__import__, "predict")
    if err:
        raise SystemExit("predict import edilemedi: %s" % err)
    import predict as pr                          # noqa: PLC0415
    print("import predict : %s" % sn(t))

    try:
        import onnxruntime as ort                 # noqa: PLC0415
        print("onnxruntime    : %s  %s"
              % (ort.__version__, ort.get_available_providers()))
    except Exception:                             # noqa: BLE001
        pass
    print()

    rows = []

    # ---- 1. sessions() onbellekli mi --------------------------------------
    t_sess1 = t_sess2 = None
    if hasattr(pr, "sessions"):
        t_sess1, _, e1 = timeit(pr.sessions)
        t_sess2, _, e2 = timeit(pr.sessions)
        rows.append(("sessions() 1. cagri", t_sess1, e1))
        rows.append(("sessions() 2. cagri", t_sess2, e2))
    else:
        print("NOT: pr.sessions yok, atlandi")

    # ---- 2. predict_record tekrarli ---------------------------------------
    t_rec = []
    if hasattr(pr, "predict_record"):
        cand = [rec, rec[:-4] if rec.lower().endswith(".hea") else rec]
        form = None
        for c in cand:
            _t, _o, e = timeit(pr.predict_record, c)
            if e is None:
                form = c
                break
        if form is None:
            print("predict_record iki yol bicimiyle de calismadi:")
            print("  %s" % e)
            return 1
        for i in range(max(1, args.repeat)):
            ti, _, ei = timeit(pr.predict_record, form)
            t_rec.append(ti)
            rows.append(("predict_record %d." % (i + 1), ti, ei))
    else:
        print("NOT: pr.predict_record yok, atlandi")

    # ---- 3. predict_arrays (saf cikarim) ----------------------------------
    t_arr = None
    if hasattr(pr, "predict_arrays"):
        sig = (np.random.RandomState(0).randn(12, 5000) * 0.3).astype(np.float32)
        timeit(pr.predict_arrays, sig, 500)       # isinma
        t_arr, _, ea = timeit(pr.predict_arrays, sig, 500)
        rows.append(("predict_arrays (hazir dizi)", t_arr, ea))

    # ---- 4. ham WFDB okuma ------------------------------------------------
    t_read = None
    try:
        import wfdb_lite as W                     # noqa: PLC0415
        fn = getattr(W, "rdrecord", None) or getattr(W, "read_record", None)
        if fn:
            timeit(fn, rec)
            t_read, _, er = timeit(fn, rec)
            rows.append(("WFDB okuma", t_read, er))
    except Exception:                             # noqa: BLE001
        pass

    # ---- tablo ------------------------------------------------------------
    print("=" * 58)
    for name, t, err in rows:
        print("  %-30s %s%s" % (name, sn(t), "   HATA: " + err if err else ""))
    print("=" * 58)
    print()

    # ---- teshis -----------------------------------------------------------
    print("TESHIS")
    print("-" * 58)
    verdict = []

    # sessions() onbellekli mi? Ikinci cagri hala pahaliysa degildir.
    sess_cached = None
    if t_sess1 is not None and t_sess2 is not None:
        sess_cached = not (t_sess2 > 0.25 * t_sess1 and t_sess1 > 0.5)
        verdict.append(
            "sessions() %s  (1. %.3f sn, 2. %.3f sn)"
            % ("ONBELLEKLI" if sess_cached else "ONBELLEKSIZ", t_sess1, t_sess2))

    if not t_rec:
        for v in verdict:
            print("  " + v.replace("\n", "\n  "))
        return 0

    med = sorted(t_rec)[len(t_rec) // 2]
    verdict.append("predict_record ortanca: %.3f sn/kayit" % med)

    # Kayit basina maliyeti UC parcaya ayir. sessions() onbelleksizse
    # yeniden yukleme maliyeti her kayitta odenir -- bunu on islemenin
    # hanesine yazmak yanlis teshise goturur.
    pay_sess = (t_sess2 or 0.0) if sess_cached is False else 0.0
    pay_infer = t_arr or 0.0
    pay_rest = max(0.0, med - pay_sess - pay_infer)
    parts = [
        ("model yeniden yukleme (sessions)", pay_sess,
         "sessions()'i memoize et -- predict.py'ye DOKUNMADAN, calisma\n"
         "aninda sarmalayarak. Tek satirlik degisiklik, kayip tamamen kalkar.\n"
         "Beklenen: %.3f -> ~0 sn/kayit" % pay_sess),
        ("saf cikarim (20 ONNX)", pay_infer,
         "onnxruntime thread ayari + surec paralelligi. int8 modeller\n"
         "VNNI'siz CPU'da yavas olabilir. Beklenen: cekirdek sayisi kadar."),
        ("WFDB okuma + on isleme", pay_rest,
         "surec paralelligi (multiprocessing) -- her kayit bagimsiz,\n"
         "tahminler birebir ayni kalir. Beklenen: 4-8x."),
    ]
    verdict.append("kayit basina dagilim:")
    for name, val, _fix in parts:
        if val > 0:
            verdict.append("    %-34s %6.3f sn  (%%%.0f)"
                           % (name, val, 100.0 * val / med if med else 0))

    parts.sort(key=lambda p: -p[1])
    top_name, top_val, top_fix = parts[0]
    verdict.append("")
    verdict.append("EN BUYUK KALEM: %s (%%%.0f)"
                   % (top_name, 100.0 * top_val / med if med else 0))
    verdict.append("COZUM: " + top_fix)

    if sess_cached is False:
        verdict.append("")
        verdict.append(
            "NOT: sessions() onbelleksiz oldugu icin bu kalem HER kayitta\n"
            "odeniyor. Once onu duzelt, sonra kalanı olcup paralellige karar ver.")

    if len(t_rec) >= 2:
        ilk = t_rec[0]
        kalan = sorted(t_rec[1:])
        son = kalan[len(kalan) // 2]
        if son > 0 and ilk > 2 * son:
            verdict.append("")
            verdict.append(
                "Ilk cagri sonrakilerin %.1f kati -- isinma maliyeti,\n"
                "750 kayitta amorti oluyor, sorun degil." % (ilk / son))

    for v in verdict:
        print("  " + v.replace("\n", "\n  "))
    print()
    print("Bu sayilari oldugu gibi paylas -- hangi hizlandirmanin")
    print("yapilacagini bunlar belirleyecek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
