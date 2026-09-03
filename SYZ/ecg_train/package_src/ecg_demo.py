"""ecg_demo -- EKG siniflandirma paketinin sunum arayuzu.

    python ecg_demo.py                      # paketi bulundugu klasorde arar
    python ecg_demo.py --root D:\...\SYZ    # kayitlarin bulundugu veri koku
    python ecg_demo.py --package package_pruned --root D:\...\SYZ
    python ecg_demo.py --no-browser --port 8080

Tarayicida acilan yerel bir arayuz. **Ek bagimlilik yoktur**: yalnizca Python
standart kutuphanesi + paketin zaten ihtiyac duydugu numpy ve onnxruntime.
Sunucu 127.0.0.1'e baglanir, disari hicbir istek atmaz, internet gerektirmez.
Jurinin onunde Wi-Fi'yi kapatip calistirabilirsin.

Neden tkinter degil: tkinter bazi Python kurulumlarinda gelmez ve cizim
yetenegi sinirlidir. Tarayici her makinede vardir, EKG'yi gercek kagit
gorunumunde cizebiliriz ve tek satir kod kurulumu gerekmez.

Bu dosya paketin icine, `predict.py`'nin yanina konur; ondan farkli olarak
kayit gezinme, dalga formu gorsellestirme ve canli toplu skorlama sunar.
Ayni `manifest.json` ve ayni `ecg_preprocess.py` kullanilir, yani ekranda
gordugun sonuc `predict.py`'nin verdigi sonucun aynisidir.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# on isleme adaptoru
# --------------------------------------------------------------------------

class Preprocess:
    """`ecg_preprocess` modulunu iki farkli API sekliyle de kullanabilir.

    Projenin gecmisinde modulun iki bicimi dolasima girdi:

      A)  prepare(sig, fs) -> (X, F)      filter_500 / to_target / features_for
      B)  preprocess_signal(sig, fs) + extract_features(sig, fs)

    Hangisi varsa o kullanilir. Boylece bu arayuz, paketin icindeki
    `ecg_preprocess.py` hangi surumse ona uyar -- ve *kesinlikle* onu cagirir,
    kendi filtresini yazmaz. Ekrandaki sonucun `predict.py` ile ayni olmasinin
    sebebi budur.
    """

    def __init__(self, module):
        self.ep = module
        self.mode = None
        if hasattr(module, "prepare"):
            self.mode = "prepare"
        elif hasattr(module, "preprocess_signal") and hasattr(module, "extract_features"):
            self.mode = "split"
        else:
            raise SystemExit(
                "ecg_preprocess.py icinde ne prepare() ne de "
                "preprocess_signal()+extract_features() bulundu -- bu paketle "
                "calisamam.")

        self.classes = list(getattr(module, "CLASSES",
                                    ("Normal", "AFIB", "AFL", "LBBB", "RBBB")))
        self.src_fs = float(getattr(module, "SRC_FS",
                                    getattr(module, "NATIVE_FS", 500.0)))

    def run(self, sig, fs):
        """Ham mV -> (X, F). F yoksa None doner."""
        if self.mode == "prepare":
            out = self.ep.prepare(sig, fs)
            if isinstance(out, (tuple, list)):
                x = np.asarray(out[0], dtype=np.float32)
                f = np.asarray(out[1], dtype=np.float32) if len(out) > 1 else None
            else:
                x, f = np.asarray(out, dtype=np.float32), None
            return x, f
        x = np.asarray(self.ep.preprocess_signal(sig, fs), dtype=np.float32)
        f = np.asarray(self.ep.extract_features(sig, fs), dtype=np.float32)
        return x, f

    def filtered(self, sig, fs):
        """Ekranda gostermek icin filtrelenmis 500 Hz sinyal.

        Ham sinyali cizmek taban kaymasi yuzunden okunaksiz olur; modelin
        gordugu temizlenmis hali daha durustur ve daha iyi gorunur.
        """
        ep = self.ep
        try:
            if hasattr(ep, "filter_500"):
                return np.asarray(ep.filter_500(sig), dtype=np.float64)
            if hasattr(ep, "sosfiltfilt") and hasattr(ep, "butter_highpass_sos"):
                y = ep.sosfiltfilt(ep.butter_highpass_sos(
                    getattr(ep, "HP_CUTOFF", 0.5), fs,
                    getattr(ep, "HP_ORDER", 2)), sig)
                return np.asarray(ep.sosfiltfilt(ep.butter_lowpass_sos(
                    min(40.0, 0.45 * fs), fs, getattr(ep, "LP_ORDER", 4)), y),
                    dtype=np.float64)
        except Exception:                                # noqa: BLE001
            pass
        return np.asarray(sig, dtype=np.float64)

    def rpeaks(self, filtered, fs):
        for name in ("detect_r", "detect_rpeaks"):
            fn = getattr(self.ep, name, None)
            if fn is None:
                continue
            for arg in (filtered, filtered[1] if filtered.ndim == 2 and
                        filtered.shape[0] > 1 else filtered[0]):
                try:
                    r = np.asarray(fn(arg), dtype=int).ravel()
                    if r.size:
                        return r
                except Exception:                        # noqa: BLE001
                    continue
        return np.array([], dtype=int)


# --------------------------------------------------------------------------
# model paketi
# --------------------------------------------------------------------------

class Bundle:
    def __init__(self, package_dir):
        self.dir = os.path.abspath(package_dir)
        manifest_path = os.path.join(self.dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise SystemExit("manifest.json bulunamadi: %s\n"
                             "--package ile paket klasorunu goster." % manifest_path)
        with open(manifest_path, encoding="utf-8") as fh:
            self.manifest = json.load(fh)

        sys.path.insert(0, self.dir)
        import ecg_preprocess                            # noqa: PLC0415
        import wfdb_lite                                 # noqa: PLC0415
        self.pre = Preprocess(ecg_preprocess)
        self.wl = wfdb_lite

        import onnxruntime as ort                        # noqa: PLC0415
        self.ort_version = ort.__version__
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sessions = []
        for entry in self.manifest["models"]:
            path = os.path.join(self.dir, entry["file"])
            if not os.path.exists(path):
                raise SystemExit("model dosyasi yok: %s" % path)
            # Saglayici ACIKCA sabitlenir. Aksi halde onnxruntime kurulumda
            # bulunan baska bir saglayiciyi (orn. Azure) secebilir; yerel ve
            # cevrimdisi kaldigimizdan emin olmak icin buna izin vermiyoruz.
            self.sessions.append((entry["member"], ort.InferenceSession(
                path, opts, providers=["CPUExecutionProvider"])))

        self.classes = self.manifest.get("classes", self.pre.classes)
        self.weights = self.manifest.get("member_weights") or {}
        self.input_len = int(self.manifest.get("input_len", 1500))
        self.target_fs = float(self.manifest.get("target_fs", 150.0))
        self.stacker = self.manifest.get("stacker")
        self.stacker_order = self.manifest.get("stacker_member_order")

    # -- tahmin ------------------------------------------------------------

    def predict(self, X, F):
        per_member = {}
        for name, sess in self.sessions:
            feed = {"signal": X.astype(np.float32)}
            names = {i.name for i in sess.get_inputs()}
            if "features" in names:
                feed["features"] = (F if F is not None
                                    else np.zeros((X.shape[0], 37),
                                                  dtype=np.float32)).astype(np.float32)
            logits = sess.run(["logits"], feed)[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            per_member.setdefault(name, []).append(e / e.sum(axis=1, keepdims=True))

        names = list(per_member)
        mats = [_norm(np.mean(per_member[n], axis=0)) for n in names]

        if self.stacker and self.stacker_order and \
                all(n in names for n in self.stacker_order):
            order = [names.index(n) for n in self.stacker_order]
            feats = np.concatenate([np.log(np.clip(mats[i], 1e-9, 1.0))
                                    for i in order], axis=1)
            logits = feats @ np.asarray(self.stacker["coef"]).T \
                + np.asarray(self.stacker["intercept"])
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        w = np.array([float(self.weights.get(n, 1.0)) for n in names])
        w = w / (w.sum() or 1.0)
        out = np.zeros_like(mats[0])
        for wi, m in zip(w, mats):
            out += wi * m
        return _norm(out)

    def analyse(self, path, waveform_points=1400):
        """Tek kaydi bastan sona isle ve arayuzun ihtiyaci olan her seyi dondur."""
        t0 = time.perf_counter()
        sig, fs, leads = self.wl.read_record(path)
        t_read = time.perf_counter() - t0

        t0 = time.perf_counter()
        X, F = self.pre.run(sig, fs)
        t_prep = time.perf_counter() - t0

        t0 = time.perf_counter()
        prob = self.predict(X[None, ...], None if F is None else F[None, ...])[0]
        t_infer = time.perf_counter() - t0

        shown = self.pre.filtered(sig, fs)
        if shown.ndim == 1:
            shown = shown[None, :]
        peaks = self.pre.rpeaks(shown, fs)

        # Cizim icin seyrelt: ekranda 1400 noktadan fazlasi zaten gorunmez.
        n = shown.shape[-1]
        step = max(n // waveform_points, 1)
        thin = shown[:, ::step]
        scale = float(np.percentile(np.abs(thin - np.median(thin)), 99)) or 1.0

        order = np.argsort(-prob)
        return {
            "record": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "fs": fs,
            "seconds": round(n / fs, 2),
            "leads": list(leads),
            "waveform": [[round(float(v), 4) for v in row] for row in thin / scale],
            "rpeaks": [int(p // step) for p in peaks if p < n],
            "n_beats": int(peaks.size),
            "heart_rate": (round(float(60.0 * fs / np.mean(np.diff(peaks))), 1)
                           if peaks.size > 2 else None),
            "prediction": self.classes[int(order[0])],
            "confidence": round(float(prob[order[0]]), 6),
            "probabilities": [{"label": self.classes[i],
                               "value": round(float(prob[i]), 6)}
                              for i in order],
            "timings": {"read_ms": round(1000 * t_read, 1),
                        "preprocess_ms": round(1000 * t_prep, 1),
                        "inference_ms": round(1000 * t_infer, 1),
                        "total_ms": round(1000 * (t_read + t_prep + t_infer), 1)},
        }


def _norm(p):
    s = p.sum(axis=1, keepdims=True)
    return p / np.where(s < 1e-12, 1.0, s)


# --------------------------------------------------------------------------
# kayit tarama
# --------------------------------------------------------------------------

_LABEL_ALIASES = {
    "normal": "Normal", "norm": "Normal", "sr": "Normal", "nsr": "Normal",
    "sinus": "Normal", "0": "Normal",
    "afib": "AFIB", "af": "AFIB", "1": "AFIB",
    "afl": "AFL", "aflt": "AFL", "flutter": "AFL", "2": "AFL",
    "lbbb": "LBBB", "clbbb": "LBBB", "3": "LBBB",
    "rbbb": "RBBB", "crbbb": "RBBB", "4": "RBBB",
}


def norm_label(v):
    if v is None:
        return None
    k = str(v).strip().replace(" ", "").replace("_", "").replace("-", "").lower()
    return _LABEL_ALIASES.get(k) if k and k not in ("nan", "none") else None


def scan_records(root, limit=6000):
    """Veri kokundeki tum .hea kayitlarini, klasorden turetilen etiketle listele."""
    out, lookup = [], {}
    root = os.path.abspath(root)
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".hea"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            stem = os.path.splitext(fn)[0]
            label = None
            for part in rel.replace("\\", "/").split("/")[:-1]:
                label = label or norm_label(part)
            if label is None:
                for tok in stem.replace("-", "_").split("_"):
                    label = label or norm_label(tok)
            out.append({"record": stem, "path": path,
                        "rel": rel.replace("\\", "/"), "label": label})
            for key in (stem, rel, os.path.splitext(rel)[0],
                        os.path.basename(dirpath)):
                lookup.setdefault(str(key).strip().lower().replace("\\", "/"), path)
            if len(out) >= limit:
                out.sort(key=lambda r: r["rel"])
                return out, lookup
    out.sort(key=lambda r: r["rel"])
    return out, lookup


# --------------------------------------------------------------------------
# toplu skorlama isi
# --------------------------------------------------------------------------

class BatchJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.running = False
        self.done = 0
        self.total = 0
        self.failed = 0
        self.phase = ""
        self.result = None
        self.error = None
        self.started = 0.0

    def snapshot(self):
        with self.lock:
            return {"running": self.running, "done": self.done,
                    "total": self.total, "failed": self.failed,
                    "phase": self.phase,
                    "result": self.result, "error": self.error,
                    "elapsed": round(time.time() - self.started, 1)
                    if self.started else 0.0}


def stratified_sample(items, limit, seed=0):
    """Siniflardan dengeli ornek al.

    Kayitlari yola gore siralayip ilk N'i almak, klasor duzeni yuzunden tek
    sinif getirir; o kume uzerinde macro-F1 anlamsiz cikar (5 siniftan biri
    varsa tavan 0.2'dir) ve sunumda sistem bozukmus gibi gorunur. Bu yuzden
    hizli kosu her siniftan esit sayida ornek alir.
    """
    if not limit or limit >= len(items):
        return items
    rng = np.random.default_rng(seed)
    by_label = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)

    labels = sorted(by_label, key=lambda k: (k is None, k))
    per = max(limit // max(len(labels), 1), 1)
    out = []
    for lab in labels:
        pool = by_label[lab]
        take = min(per, len(pool))
        idx = rng.permutation(len(pool))[:take]
        out.extend(pool[i] for i in idx)

    # Bolunmeden artan kontenjani, havuzu bitmemis siniflara dagit.
    leftover = [it for lab in labels for it in by_label[lab] if it not in out]
    if len(out) < limit and leftover:
        idx = rng.permutation(len(leftover))[:limit - len(out)]
        out.extend(leftover[i] for i in idx)
    return out


def macro_f1(y_true, y_pred, k):
    f1 = []
    for c in range(k):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1)), f1


def run_batch(app, items, job):
    """Kayit listesini sirayla skorla; ilerlemeyi job uzerinden bildir."""
    try:
        # Ilerleme iki asamayi da kapsar: on isleme + cikarim. Yalnizca on
        # islemeyi saymak, cubugun %100'de donup beklemesine yol aciyordu.
        with job.lock:
            job.running, job.total, job.done = True, 2 * len(items), 0
            job.failed, job.result, job.error = 0, None, None
            job.phase = "on isleme"
            job.started = time.time()

        classes = app.bundle.classes
        X = np.zeros((len(items), 12, app.bundle.input_len), dtype=np.float32)
        F = np.zeros((len(items), app.bundle.manifest.get("n_features", 37)),
                     dtype=np.float32)
        ok = np.zeros(len(items), dtype=bool)
        t_prep = 0.0

        for i, it in enumerate(items):
            try:
                t = time.perf_counter()
                sig, fs, _ = app.bundle.wl.read_record(it["path"])
                x, f = app.bundle.pre.run(sig, fs)
                t_prep += time.perf_counter() - t
                X[i] = x
                if f is not None:
                    F[i] = f
                ok[i] = True
            except Exception:                            # noqa: BLE001
                with job.lock:
                    job.failed += 1
            with job.lock:
                job.done = i + 1

        idx = np.flatnonzero(ok)
        with job.lock:
            job.phase = "cikarim"
        t = time.perf_counter()
        prob = np.zeros((len(items), len(classes)))
        step = 32
        for s in range(0, len(idx), step):
            sel = idx[s:s + step]
            prob[sel] = app.bundle.predict(X[sel], F[sel])
            with job.lock:
                job.done = len(items) + min(s + step, len(idx))
        t_infer = time.perf_counter() - t

        pred = prob.argmax(1)
        res = {
            "n": int(len(items)), "scored": int(ok.sum()), "failed": int((~ok).sum()),
            "classes": classes,
            "prep_ms": round(1000 * t_prep / max(int(ok.sum()), 1), 1),
            "infer_ms": round(1000 * t_infer / max(len(idx), 1), 1),
            "counts": {c: int(np.sum(pred[idx] == i)) for i, c in enumerate(classes)},
        }

        labelled = [i for i in idx if items[i]["label"] in classes]
        if len(labelled) == len(idx) and len(idx):
            y = np.array([classes.index(items[i]["label"]) for i in labelled])
            p = pred[labelled]
            m, per = macro_f1(y, p, len(classes))
            cm = [[int(np.sum((y == a) & (p == b))) for b in range(len(classes))]
                  for a in range(len(classes))]
            res.update({
                "macro_f1": round(m, 6),
                "accuracy": round(float((y == p).mean()), 6),
                "per_class": [round(v, 4) for v in per],
                "confusion": cm,
            })
            pair = np.isin(y, [1, 2]) & np.isin(p, [1, 2])
            if pair.any():
                res["afib_afl"] = round(float((y[pair] == p[pair]).mean()), 4)

        with job.lock:
            job.result = res
            job.running = False
    except Exception as exc:                             # noqa: BLE001
        with job.lock:
            job.error = "%s: %s" % (type(exc).__name__, exc)
            job.running = False
            traceback.print_exc()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class App:
    def __init__(self, bundle, root):
        self.bundle = bundle
        self.root = os.path.abspath(root) if root else None
        self.records, self.lookup = ([], {})
        if self.root and os.path.isdir(self.root):
            self.records, self.lookup = scan_records(self.root)
        self.job = BatchJob()

    def info(self):
        try:
            import torch                                 # noqa: F401, PLC0415
            torch_state = "KURULU"
        except ImportError:
            torch_state = "yok"
        m = self.bundle.manifest
        val = m.get("validation") or {}
        size = 0
        for e in m.get("models", []):
            p = os.path.join(self.bundle.dir, e["file"])
            if os.path.exists(p):
                size += os.path.getsize(p)
        return {
            "package": self.bundle.dir,
            "n_models": len(m.get("models", [])),
            "families": self.bundle.weights,
            "combination": m.get("combination", "flat"),
            "precision": m.get("precision", "-"),
            "size_mb": round(size / 1e6, 1),
            "classes": self.bundle.classes,
            "input": "12 x %d @ %g Hz" % (self.bundle.input_len, self.bundle.target_fs),
            "oof_macro_f1": m.get("oof_macro_f1"),
            "test_macro_f1": val.get("test_public_macro_f1_onnx"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": self.bundle.ort_version,
            "torch": torch_state,
            "providers": ["CPUExecutionProvider"],
            "preprocess_api": self.bundle.pre.mode,
            "root": self.root,
            "n_records": len(self.records),
            "offline": True,
        }


class Handler(BaseHTTPRequestHandler):
    app: App = None
    server_version = "ecg-demo"

    def log_message(self, *a):                            # sunum sirasinda sessiz
        pass

    # -- yardimcilar -------------------------------------------------------

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError:
            return {}

    def _resolve(self, given):
        """Istemciden gelen yolu, taranmis kayitlarla dogrula.

        Rastgele bir dosya yolunun sunucuya okutulmasini engellemek icin
        yalnizca veri kokunun altinda taranmis kayitlar kabul edilir.
        """
        if not given:
            return None
        key = str(given).strip().lower().replace("\\", "/")
        if key.endswith(".hea"):
            key = key[:-4]
        hit = self.lookup_get(key)
        if hit:
            return hit
        for rec in self.app.records:
            if rec["path"] == given or rec["record"].lower() == key:
                return rec["path"]
        return None

    def lookup_get(self, key):
        return self.app.lookup.get(key)

    # -- yonlendirme -------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/info":
                return self._json(self.app.info())
            if path == "/api/records":
                return self._json({"records": [
                    {k: r[k] for k in ("record", "rel", "label")}
                    for r in self.app.records]})
            if path == "/api/batch":
                return self._json(self.app.job.snapshot())
            return self._json({"error": "bulunamadi"}, 404)
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": "%s: %s" % (type(exc).__name__, exc)}, 500)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
            if path == "/api/predict":
                target = self._resolve(body.get("path") or body.get("record"))
                if not target:
                    return self._json({"error": "kayit bulunamadi"}, 404)
                return self._json(self.app.analyse_cached(target))
            if path == "/api/batch/start":
                if self.app.job.snapshot()["running"]:
                    return self._json({"error": "zaten calisiyor"}, 409)
                items = self.app.records
                only = body.get("labelled_only")
                if only:
                    items = [r for r in items if r["label"]]
                items = stratified_sample(items, int(body.get("limit") or 0))
                if not items:
                    return self._json({"error": "skorlanacak kayit yok"}, 400)
                threading.Thread(target=run_batch,
                                 args=(self.app, items, self.app.job),
                                 daemon=True).start()
                return self._json({"started": True, "total": len(items)})
            return self._json({"error": "bulunamadi"}, 404)
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": "%s: %s" % (type(exc).__name__, exc)}, 500)


def _analyse_cached(self, path):
    key = os.path.abspath(path)
    with self._cache_lock:
        if key in self._cache:
            return self._cache[key]
    data = self.bundle.analyse(path)
    with self._cache_lock:
        self._cache[key] = data
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
    return data


App.analyse_cached = _analyse_cached
App._cache = {}
App._cache_lock = threading.Lock()


# --------------------------------------------------------------------------
# arayuz -- tek parca, disaridan hicbir kaynak yuklemez
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="tr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EKG Siniflandirma - Canli Demo</title>
<!-- Bos veri-URI: tarayicinin /favicon.ico istegini ve konsoldaki 404'u onler.
     Sunum sirasinda gelistirici konsolu acilirsa temiz gorunsun. -->
<link rel="icon" href="data:,">
<style>
:root{
  --bg:#0d1518; --panel:#132025; --panel2:#18292f; --line:#22383e;
  --ink:#e8f2f1; --ink2:#a9c2c2; --muted:#7793959;
  --accent:#2fd0ae; --accent2:#0f3a35;
  --grid:#3a1f22; --grid2:#5a2b30; --trace:#e8f2f1;
  --normal:#2fd0ae; --afib:#d9a441; --afl:#e0637f; --lbbb:#7aa7e8; --rbbb:#b58ae0;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Consolas,"Liberation Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
button{font-family:inherit;font-size:13px;cursor:pointer;border-radius:4px;
  border:1px solid var(--line);background:var(--panel2);color:var(--ink);
  padding:7px 13px;transition:border-color .12s,background .12s}
button:hover{border-color:var(--accent);background:#1d3238}
button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--accent);color:#06231e;border-color:var(--accent);
  font-weight:650}
button.primary:hover{background:#43ddbc}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

header{display:flex;align-items:center;gap:16px;padding:11px 18px;
  border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:650;letter-spacing:-.01em}
header h1 span{color:var(--accent)}
.badge{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
  padding:3px 8px;border-radius:3px;background:var(--accent2);color:var(--accent);
  border:1px solid #1d5a50;text-transform:uppercase}
.badge.grey{background:#1b2b30;color:var(--ink2);border-color:var(--line)}
.spacer{flex:1}
.stat{font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.stat b{color:var(--ink);font-weight:600}

.tabs{display:flex;gap:2px;padding:0 18px;background:var(--panel);
  border-bottom:1px solid var(--line)}
.tab{padding:9px 16px;font-size:13px;color:var(--ink2);cursor:pointer;
  border-bottom:2px solid transparent;background:none;border-radius:0;border-top:0;
  border-left:0;border-right:0}
.tab:hover{color:var(--ink);background:none;border-color:transparent}
.tab.on{color:var(--accent);border-bottom-color:var(--accent)}
.view{display:none}.view.on{display:block}

.layout{display:grid;grid-template-columns:250px 1fr 290px;gap:0;
  height:calc(100vh - 92px);min-height:520px}
.col{overflow-y:auto;padding:14px}
.col.mid{background:#0a1114}
.col.left,.col.right{background:var(--panel);border-right:1px solid var(--line)}
.col.right{border-right:0;border-left:1px solid var(--line)}

.lbl{font-family:var(--mono);font-size:10px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.lbl:not(:first-child){margin-top:20px}

#search{width:100%;padding:7px 10px;border-radius:4px;border:1px solid var(--line);
  background:var(--panel2);color:var(--ink);font-family:var(--mono);font-size:12px}
#search:focus{outline:none;border-color:var(--accent)}
.reclist{margin-top:9px;display:flex;flex-direction:column;gap:2px}
.rec{display:flex;align-items:center;gap:7px;padding:6px 8px;border-radius:3px;
  cursor:pointer;font-family:var(--mono);font-size:11.5px;color:var(--ink2);
  border:1px solid transparent}
.rec:hover{background:var(--panel2);color:var(--ink)}
.rec.on{background:var(--accent2);color:var(--ink);border-color:#1d5a50}
.dot{width:7px;height:7px;border-radius:50%;flex:none}
.rec span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

#paper{width:100%;height:auto;display:block;border-radius:4px;background:#150c0d}
.wavehead{display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap}
.wavehead h2{margin:0;font-size:16px;font-weight:650;font-family:var(--mono)}
.empty{display:flex;align-items:center;justify-content:center;height:100%;
  color:var(--muted);font-size:14px;text-align:center;padding:40px}

.result{border-radius:5px;padding:14px;background:var(--panel2);
  border:1px solid var(--line);border-left:3px solid var(--accent)}
.result .cls{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:2px 0 1px}
.result .conf{font-family:var(--mono);font-size:12px;color:var(--ink2)}
.bars{margin-top:6px;display:flex;flex-direction:column;gap:7px}
.bar{display:grid;grid-template-columns:52px 1fr 50px;gap:9px;align-items:center;
  font-family:var(--mono);font-size:11.5px}
.bar .track{height:9px;background:#0d1b1f;border-radius:5px;overflow:hidden}
.bar .fill{height:100%;border-radius:5px;transition:width .35s ease}
.bar .pct{text-align:right;color:var(--ink2);font-variant-numeric:tabular-nums}
.kv{display:flex;justify-content:space-between;gap:10px;padding:5px 0;
  border-bottom:1px solid var(--line);font-size:12.5px}
.kv:last-child{border-bottom:0}
.kv span{color:var(--ink2)}
.kv b{font-family:var(--mono);font-weight:600;font-variant-numeric:tabular-nums}

.pad{padding:22px 26px;max-width:1000px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));
  gap:11px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  padding:13px 15px}
.card dt{font-family:var(--mono);font-size:10px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.card dd{margin:0;font-size:20px;font-weight:650;font-family:var(--mono);
  font-variant-numeric:tabular-nums}
.card dd.sm{font-size:14px;font-weight:500}
.card dd.ok{color:var(--accent)}

table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);font-weight:500;padding:0 12px 8px 0;
  border-bottom:1px solid var(--line)}
td{padding:8px 12px 8px 0;border-bottom:1px solid var(--line);color:var(--ink2)}
td:first-child{color:var(--ink)}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;
  padding-right:16px}
tr:last-child td{border-bottom:0}
.cm td.d{color:var(--accent);font-weight:650}
.cm td.err{color:var(--afl)}

.prog{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;
  margin:12px 0 6px}
.prog i{display:block;height:100%;background:var(--accent);width:0;
  transition:width .25s}
.note{color:var(--muted);font-size:12.5px;margin-top:8px}
.err{color:var(--afl);font-family:var(--mono);font-size:12px;margin-top:10px}
.spin{display:inline-block;width:11px;height:11px;border:2px solid var(--line);
  border-top-color:var(--accent);border-radius:50%;animation:sp .7s linear infinite;
  vertical-align:-1px}
@keyframes sp{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spin{animation:none}
  .bar .fill,.prog i{transition:none}}
@media (max-width:1100px){.layout{grid-template-columns:210px 1fr}
  .col.right{grid-column:1/-1;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>

<header>
  <h1>EKG Siniflandirma <span>&#9679;</span> Canli Demo</h1>
  <span class="badge" id="b-offline">Cevrimdisi</span>
  <span class="badge grey" id="b-torch">PyTorch: -</span>
  <span class="badge grey" id="b-models">- model</span>
  <span class="spacer"></span>
  <span class="stat" id="hdr-stat"></span>
</header>

<div class="tabs">
  <button class="tab on" data-v="single">Tek Kayit</button>
  <button class="tab" data-v="batch">Toplu Skor</button>
  <button class="tab" data-v="sys">Sistem</button>
</div>

<div class="view on" id="v-single">
  <div class="layout">
    <div class="col left">
      <p class="lbl">Kayitlar</p>
      <input id="search" placeholder="ara..." autocomplete="off">
      <div style="display:flex;gap:6px;margin-top:9px">
        <button id="rnd" style="flex:1">Rastgele</button>
      </div>
      <div class="reclist" id="reclist"></div>
    </div>

    <div class="col mid">
      <div id="wavewrap"><div class="empty" id="wave-empty">
        Soldan bir kayit sec &mdash; ya da <b style="color:var(--accent)">Rastgele</b>.
      </div></div>
    </div>

    <div class="col right">
      <p class="lbl">Sonuc</p>
      <div id="res"><div class="note">Henuz kayit secilmedi.</div></div>
    </div>
  </div>
</div>

<div class="view" id="v-batch">
  <div class="pad">
    <p class="lbl">Toplu skorlama</p>
    <p class="note" style="margin:0 0 14px">Veri kokundeki etiketli tum kayitlari
    bastan sona isler ve macro-F1, karisiklik matrisi ile hiz olcumlerini
    hesaplar. Tum islem bu bilgisayarda yapilir.</p>
    <button class="primary" id="run">Tumunu Skorla</button>
    <button id="run100">Ilk 100</button>
    <div class="prog" id="prog" hidden><i id="progbar"></i></div>
    <div id="progtxt" class="note"></div>
    <div id="batchres" style="margin-top:20px"></div>
  </div>
</div>

<div class="view" id="v-sys">
  <div class="pad">
    <p class="lbl">Sistem ve paket</p>
    <div id="sys"></div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const CLS_COLOR={Normal:'var(--normal)',AFIB:'var(--afib)',AFL:'var(--afl)',
                 LBBB:'var(--lbbb)',RBBB:'var(--rbbb)'};
let INFO=null, RECORDS=[], CUR=null;

const esc=s=>String(s).replace(/[&<>"']/g,c=>(
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

$$('.tab').forEach(t=>t.onclick=()=>{
  $$('.tab').forEach(x=>x.classList.toggle('on',x===t));
  $$('.view').forEach(v=>v.classList.toggle('on',v.id==='v-'+t.dataset.v));
});

async function api(url,opt){
  const r=await fetch(url,opt);
  const j=await r.json().catch(()=>({error:'yanit okunamadi'}));
  if(!r.ok) throw new Error(j.error||('HTTP '+r.status));
  return j;
}

// ---------- baslangic ----------
(async()=>{
  INFO=await api('/api/info');
  $('#b-torch').textContent='PyTorch: '+INFO.torch;
  $('#b-models').textContent=INFO.n_models+' model';
  $('#hdr-stat').innerHTML='<b>'+esc(INFO.combination)+'</b> ensemble &middot; '+
    esc(INFO.input)+' &middot; <b>'+INFO.size_mb+'</b> MB';
  renderSys();
  const rr=await api('/api/records');
  RECORDS=rr.records;
  renderList();
  if(!RECORDS.length){
    $('#wave-empty').innerHTML='Veri kokunde kayit bulunamadi.<br>'+
      '<code style="color:var(--ink2)">--root</code> ile klasoru goster.';
  }
})().catch(e=>{document.body.insertAdjacentHTML('afterbegin',
  '<div class="err" style="padding:14px">Baslatma hatasi: '+esc(e.message)+'</div>')});

// ---------- kayit listesi ----------
function renderList(){
  const q=$('#search').value.trim().toLowerCase();
  const items=RECORDS.filter(r=>!q||r.record.toLowerCase().includes(q)||
    (r.label||'').toLowerCase().includes(q)).slice(0,400);
  $('#reclist').innerHTML=items.map(r=>
    '<div class="rec'+(CUR&&CUR.record===r.record?' on':'')+'" data-r="'+esc(r.record)+'">'+
    '<i class="dot" style="background:'+(CLS_COLOR[r.label]||'var(--line)')+'"></i>'+
    '<span>'+esc(r.record)+'</span></div>').join('')||
    '<div class="note">eslesme yok</div>';
  $$('.rec').forEach(el=>el.onclick=()=>load(el.dataset.r));
}
$('#search').oninput=renderList;
$('#rnd').onclick=()=>{ if(RECORDS.length)
  load(RECORDS[Math.floor(Math.random()*RECORDS.length)].record); };

// ---------- tek kayit ----------
async function load(rec){
  $('#res').innerHTML='<div class="note"><span class="spin"></span> isleniyor...</div>';
  try{
    const d=await api('/api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({record:rec})});
    CUR=d; renderList(); drawWave(d); renderResult(d);
  }catch(e){ $('#res').innerHTML='<div class="err">'+esc(e.message)+'</div>'; }
}

function drawWave(d){
  const W=1000, GUT=46, LH=62, n=d.waveform[0].length, rows=d.waveform.length;
  const H=rows*LH+16, PW=W-GUT;
  const X=k=>GUT+k*PW/(n-1);

  // Tum derivasyonlar TEK olcekle cizilir; boylece V1/V6 genlik orani -- yani
  // dal blogunu ayirt eden sey -- ekranda korunur. Bedeli, dusuk genlikli bir
  // derivasyonun duz gorunmesidir; onu bozuk sanmamak icin asagida isaretliyoruz.
  const pk=d.waveform.map(w=>Math.max(...w)-Math.min(...w));
  const pkMax=Math.max(...pk);
  const weak=[];

  let g='';
  for(let x=GUT;x<=W;x+=PW/50) g+='<line x1="'+x.toFixed(1)+'" y1="0" x2="'+
    x.toFixed(1)+'" y2="'+H+'" stroke="var(--grid)" stroke-width="0.6"/>';
  for(let y=0;y<H;y+=LH/4) g+='<line x1="'+GUT+'" y1="'+y.toFixed(1)+'" x2="'+W+
    '" y2="'+y.toFixed(1)+'" stroke="var(--grid)" stroke-width="0.6"/>';
  for(let x=GUT;x<=W;x+=PW/10) g+='<line x1="'+x.toFixed(1)+'" y1="0" x2="'+
    x.toFixed(1)+'" y2="'+H+'" stroke="var(--grid2)" stroke-width="0.9"/>';

  let tr='';
  d.waveform.forEach((w,i)=>{
    const top=8+i*LH, mid=top+LH/2;
    const name=d.leads[i]||('L'+i);
    const low=pkMax>0 && pk[i]<0.05*pkMax;
    if(low) weak.push(name);
    let pts='';
    for(let k=0;k<n;k++){
      const y=Math.max(top+3,Math.min(top+LH-3,mid-w[k]*(LH*0.40)));
      pts+=(k?' ':'')+X(k).toFixed(1)+','+y.toFixed(1);
    }
    tr+='<polyline points="'+pts+'" fill="none" stroke="var(--trace)" '+
        'stroke-width="1.15" stroke-linejoin="round"/>'+
        // Etiket olukta durur, dalganin uzerine binmez.
        '<text x="'+(GUT-8)+'" y="'+(mid+4)+'" font-size="12" text-anchor="end" '+
        'font-family="var(--mono)" fill="'+(low?'var(--afib)':'var(--accent)')+
        '">'+esc(name)+'</text>';
  });

  let rp='';
  (d.rpeaks||[]).forEach(p=>{ const x=X(p);
    if(x>=GUT&&x<=W) rp+='<line x1="'+x.toFixed(1)+'" y1="3" x2="'+x.toFixed(1)+
      '" y2="11" stroke="var(--accent)" stroke-width="1.5"/>'; });

  $('#wavewrap').innerHTML=
    '<div class="wavehead"><h2>'+esc(d.record)+'</h2>'+
    '<span class="badge grey">'+d.seconds+' sn &middot; '+d.fs+' Hz</span>'+
    (d.heart_rate?'<span class="badge grey">'+d.heart_rate+' bpm</span>':'')+
    (d.n_beats?'<span class="badge grey">'+d.n_beats+' vurus</span>':'')+
    (weak.length?'<span class="badge" style="background:#2c2412;color:var(--afib);'+
      'border-color:#4a3a18">dusuk sinyal: '+esc(weak.join(' '))+'</span>':'')+
    '</div><svg id="paper" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+
    esc(d.record)+' 12 derivasyon EKG, tahmin '+esc(d.prediction)+'">'+
    g+rp+tr+'</svg>'+
    '<p class="note">Filtrelenmis 500 Hz sinyal; yesil cizgiler saptanan R '+
    'tepeleri, izgara ~200 ms. Tum derivasyonlar ayni olcekte cizilir, bu '+
    'yuzden aralarindaki genlik orani korunur.'+
    (weak.length?' <b style="color:var(--afib)">'+esc(weak.join(', '))+'</b> '+
      'cok dusuk genlikli &mdash; kayitta kopuk/zayif elektrot olabilir; model '+
      'yine de 12 derivasyonun tamamini girdi olarak alir.':'')+'</p>';
}

function renderResult(d){
  const c=CLS_COLOR[d.prediction]||'var(--accent)';
  let bars=d.probabilities.map(p=>
    '<div class="bar"><span>'+esc(p.label)+'</span><div class="track">'+
    '<i class="fill" style="width:'+(p.value*100).toFixed(1)+'%;background:'+
    (CLS_COLOR[p.label]||'var(--accent)')+'"></i></div>'+
    '<span class="pct">'+(p.value*100).toFixed(1)+'%</span></div>').join('');
  const t=d.timings;
  $('#res').innerHTML=
    '<div class="result" style="border-left-color:'+c+'">'+
      '<div class="conf">TAHMIN</div>'+
      '<div class="cls" style="color:'+c+'">'+esc(d.prediction)+'</div>'+
      '<div class="conf">guven %'+(d.confidence*100).toFixed(1)+'</div></div>'+
    '<p class="lbl">Sinif olasiliklari</p><div class="bars">'+bars+'</div>'+
    '<p class="lbl">Sure</p>'+
    '<div class="kv"><span>Okuma</span><b>'+t.read_ms+' ms</b></div>'+
    '<div class="kv"><span>On isleme</span><b>'+t.preprocess_ms+' ms</b></div>'+
    '<div class="kv"><span>Cikarim ('+INFO.n_models+' model)</span><b>'+
      t.inference_ms+' ms</b></div>'+
    '<div class="kv"><span>Toplam</span><b>'+t.total_ms+' ms</b></div>'+
    '<p class="lbl">Kayit</p>'+
    '<div class="kv"><span>Kalp hizi</span><b>'+(d.heart_rate||'-')+' bpm</b></div>'+
    '<div class="kv"><span>Vurus</span><b>'+d.n_beats+'</b></div>'+
    '<div class="kv"><span>Derivasyon</span><b>'+d.leads.length+'</b></div>';
}

// ---------- toplu ----------
let poll=null;
async function startBatch(limit){
  $('#batchres').innerHTML=''; $('#prog').hidden=false;
  $('#run').disabled=$('#run100').disabled=true;
  try{
    await api('/api/batch/start',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({labelled_only:true,limit:limit||0})});
    poll=setInterval(tick,400); tick();
  }catch(e){
    $('#progtxt').innerHTML='<span class="err">'+esc(e.message)+'</span>';
    $('#run').disabled=$('#run100').disabled=false;
  }
}
$('#run').onclick=()=>startBatch(0);
$('#run100').onclick=()=>startBatch(100);

async function tick(){
  const s=await api('/api/batch');
  const pct=s.total?100*s.done/s.total:0;
  $('#progbar').style.width=pct.toFixed(1)+'%';
  // Ilerleme iki asamali: done 0..N on isleme, N..2N cikarim. Hangi asamada
  // oldugumuzu sayidan cikarmak yerine phase alanindan okuyoruz; aksi halde
  // gecis aninda (done tam N iken) sayac 100'den 0'a atliyor.
  const half=Math.max(Math.round(s.total/2),1);
  const inInfer=(s.phase||'').indexOf('cikarim')===0;
  const cur=Math.min(inInfer?s.done-half:s.done,half);
  $('#progtxt').innerHTML=s.running
    ? '<span class="spin"></span> '+esc(s.phase||'')+' &middot; '+cur+' / '+half+
      ' kayit &middot; '+s.elapsed+' sn'
    : (s.error?'<span class="err">'+esc(s.error)+'</span>'
              :'Bitti &middot; '+half+' kayit &middot; '+s.elapsed+' sn');
  if(!s.running){
    clearInterval(poll); poll=null;
    $('#run').disabled=$('#run100').disabled=false;
    if(s.result) renderBatch(s.result);
  }
}

function renderBatch(r){
  let h='';
  // macro-F1, kumede bulunmayan sinifi 0 sayar; tek sinifli bir orneklemde
  // tavan 1/5'tir. Bunu sessizce gostermek sistemi bozukmus gibi gosterir.
  const present=r.confusion
    ? r.confusion.filter(row=>row.reduce((a,b)=>a+b,0)>0).length : null;
  if(present!==null && present<r.classes.length){
    h+='<div class="result" style="border-left-color:var(--afib);margin-bottom:16px">'+
       '<b>Dikkat:</b> bu orneklemde '+r.classes.length+' siniftan yalnizca '+
       present+' tanesi var. macro-F1 eksik siniflari 0 sayar, bu yuzden '+
       'asagidaki deger karsilastirilabilir degildir. Tam degerlendirme icin '+
       '<b>Tumunu Skorla</b>.</div>';
  }
  h+='<div class="cards">';
  if(r.macro_f1!==undefined){
    h+='<div class="card"><dt>macro-F1</dt><dd class="ok">'+
       r.macro_f1.toFixed(4)+'</dd></div>'+
       '<div class="card"><dt>Dogruluk</dt><dd>'+r.accuracy.toFixed(4)+'</dd></div>';
    if(r.afib_afl!==undefined)
      h+='<div class="card"><dt>AFIB/AFL ikili</dt><dd>'+
         r.afib_afl.toFixed(4)+'</dd></div>';
  }
  h+='<div class="card"><dt>Kayit</dt><dd>'+r.scored+'</dd></div>'+
     '<div class="card"><dt>Hata</dt><dd'+(r.failed?'':' class="ok"')+'>'+
       r.failed+'</dd></div>'+
     '<div class="card"><dt>Kayit basina</dt><dd class="sm">'+
       (r.prep_ms+r.infer_ms).toFixed(0)+' ms</dd></div></div>';

  if(r.confusion){
    h+='<p class="lbl">Karisiklik matrisi <span style="text-transform:none;'+
       'letter-spacing:0">(satir = gercek, sutun = tahmin)</span></p>'+
       '<table class="cm"><tr><th></th>'+
       r.classes.map(c=>'<th style="text-align:right">'+esc(c)+'</th>').join('')+
       '<th style="text-align:right">F1</th></tr>';
    r.confusion.forEach((row,i)=>{
      h+='<tr><td>'+esc(r.classes[i])+'</td>'+row.map((v,j)=>
        '<td class="n '+(i===j?'d':(v?'err':''))+'">'+v+'</td>').join('')+
        '<td class="n">'+r.per_class[i].toFixed(3)+'</td></tr>';
    });
    h+='</table>';
  }
  h+='<p class="lbl">Hiz</p><div style="max-width:420px">'+
     '<div class="kv"><span>On isleme / kayit</span><b>'+r.prep_ms+' ms</b></div>'+
     '<div class="kv"><span>Cikarim / kayit</span><b>'+r.infer_ms+' ms</b></div></div>';
  $('#batchres').innerHTML=h;
}

// ---------- sistem ----------
function renderSys(){
  const i=INFO;
  // manifest'teki skorlar tam float hassasiyetinde tutulur; ekranda dort
  // ondalik yeter (0.9976504683537752 degil 0.9977).
  const val=v=>(v===null||v===undefined)?'-':
    (typeof v==='number'?v.toFixed(4):v);
  const fam=Object.entries(i.families||{}).map(([k,v])=>
    '<div class="kv"><span>'+esc(k)+'</span><b>'+(+v).toFixed(4)+'</b></div>').join('');
  $('#sys').innerHTML=
   '<div class="cards">'+
    '<div class="card"><dt>ONNX model</dt><dd>'+i.n_models+'</dd></div>'+
    '<div class="card"><dt>Paket boyutu</dt><dd>'+i.size_mb+' MB</dd></div>'+
    '<div class="card"><dt>Hassasiyet</dt><dd class="sm">'+esc(i.precision)+'</dd></div>'+
    '<div class="card"><dt>PyTorch</dt><dd class="sm'+
      (i.torch==='yok'?' ok':'')+'">'+esc(i.torch)+'</dd></div></div>'+
   '<p class="lbl">Dogrulanmis skorlar</p><div style="max-width:460px">'+
    '<div class="kv"><span>OOF macro-F1 (gelistirme)</span><b>'+
      val(i.oof_macro_f1)+'</b></div>'+
    '<div class="kv"><span>test_public macro-F1 (ONNX)</span><b>'+
      val(i.test_macro_f1)+'</b></div></div>'+
   (fam?'<p class="lbl">Ensemble agirliklari</p><div style="max-width:460px">'+
      fam+'</div>':'')+
   '<p class="lbl">Calisma ortami</p><div style="max-width:460px">'+
    '<div class="kv"><span>Python</span><b>'+esc(i.python)+'</b></div>'+
    '<div class="kv"><span>numpy</span><b>'+esc(i.numpy)+'</b></div>'+
    '<div class="kv"><span>onnxruntime</span><b>'+esc(i.onnxruntime)+'</b></div>'+
    '<div class="kv"><span>Saglayici</span><b>'+esc(i.providers.join(', '))+'</b></div>'+
    '<div class="kv"><span>On isleme API</span><b>'+esc(i.preprocess_api)+'</b></div>'+
    '<div class="kv"><span>Ag baglantisi</span><b style="color:var(--accent)">'+
      'yok (127.0.0.1)</b></div></div>'+
   '<p class="lbl">Yollar</p><div style="max-width:700px">'+
    '<div class="kv"><span>Paket</span><b style="font-size:11px">'+
      esc(i.package)+'</b></div>'+
    '<div class="kv"><span>Veri koku</span><b style="font-size:11px">'+
      esc(val(i.root))+'</b></div>'+
    '<div class="kv"><span>Bulunan kayit</span><b>'+i.n_records+'</b></div></div>'+
   '<p class="note" style="margin-top:18px">Bu arayuz paketin icindeki ayni '+
    '<code>ecg_preprocess.py</code> ve ayni ONNX modellerini kullanir; '+
    'ekrandaki sonuc <code>predict.py</code> ciktisiyla ayni olmalidir.</p>';
}
</script></body></html>
"""


# --------------------------------------------------------------------------

def find_package(start):
    for cand in (start, os.path.join(start, "package"),
                 os.path.join(start, "package_pruned"),
                 os.path.join(start, "..", "package_pruned"),
                 os.path.join(start, "..", "package")):
        if os.path.exists(os.path.join(cand, "manifest.json")):
            return os.path.abspath(cand)
    return None


def free_port(preferred):
    for port in [preferred] + list(range(preferred + 1, preferred + 25)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("bos port bulunamadi")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default=None, help="manifest.json'un bulundugu klasor")
    ap.add_argument("--root", default=os.environ.get("ECG_ROOT", ""),
                    help="EKG kayitlarinin bulundugu veri koku")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    package = args.package or find_package(HERE)
    if not package:
        raise SystemExit(
            "Paket bulunamadi. Bu dosyayi manifest.json ile ayni klasore koy "
            "veya --package ile yolunu ver.")

    print("paket   : %s" % package)
    bundle = Bundle(package)
    print("model   : %d (%s, %s)" % (len(bundle.sessions),
                                     bundle.manifest.get("combination", "flat"),
                                     bundle.manifest.get("precision", "-")))
    print("on isl. : ecg_preprocess.%s()" % bundle.pre.mode)

    root = args.root or os.path.dirname(package)
    app = App(bundle, root)
    print("veri    : %s  (%d kayit)" % (app.root, len(app.records)))
    if not app.records:
        print("UYARI: kayit bulunamadi -- --root ile veri kokunu goster")

    port = free_port(args.port)
    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d/" % port

    print()
    print("  %s" % url)
    print("  (yerel adres, internet gerekmez -- Wi-Fi kapaliyken de calisir)")
    print("  durdurmak icin Ctrl+C")
    print()

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatiliyor")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
