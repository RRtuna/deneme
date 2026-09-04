"""train -- 5-fold cross-validated training on the development set.

    python train.py --preset w64 --tag cap_b64 --only_fold 0 --epochs 40 --patience 99
    python train.py --preset w64 --tag main_v2 --epochs 40 --patience 99
    python train.py --preset w64 --tag main_v2 --cache cache_250

The development set is train.csv + validation.csv. test_public is loaded only
so that each fold can emit a prediction for the final report; it never enters a
training batch, never fits the feature scaler, and never selects a checkpoint.

Resuming: every fold writes ``runs/<tag>/fold<k>/done.json`` when it finishes.
Re-running the same command skips completed folds, so a machine that goes to
sleep mid-run costs you one fold, not the whole job.

``summary.json`` is written only when every requested fold is present. If it is
missing, the run did not finish and its numbers are not results.

A note on the FAZ 2 gate: this script prints both the fold's validation
(out-of-fold) macro-F1 and its test_public macro-F1. Use the validation number
to decide anything. GOREV.md rule 3 forbids selecting on test_public, and the
FAZ 2 gate as written ("fold 0 test > 0.855") contradicts it -- the OOF number
is the one that is safe to act on.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ecg_preprocess as ep
from model import N_CLASSES, N_FEATURES, PRESETS, build_model, count_parameters

# Girdi genisligi cache'ten okunur: kanal sayisi (12 derivasyon + varsa
# QRST-iptalli artik kanallari) ve ozellik sayisi (37 + varsa bant-artik
# olcumleri). Sabit yazmak, genisletilmis bir cache'te sessizce yanlis model
# kurar -- bu yuzden tek kaynak X.npy ile F.npy'nin kendisidir.
IN_CH = 12
N_FEAT = N_FEATURES


def net(args):
    return build_model(args.preset, dropout=args.dropout,
                       use_features=not args.no_features,
                       in_ch=IN_CH, n_features=N_FEAT)


DEV_SPLITS = ("train", "validation")
TEST_SPLIT = "test_public"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_cache(cache_dir, mmap=False):
    """Load a cache directory into ``(X, F, y, split, records)``."""
    for name in ("X.npy", "y.npy", "index.csv"):
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            raise SystemExit("%s not found -- run 'python prep.py' first" % path)

    X = np.load(os.path.join(cache_dir, "X.npy"),
                mmap_mode="r" if mmap else None)
    if not mmap:
        X = np.ascontiguousarray(X)

    y = np.load(os.path.join(cache_dir, "y.npy"))

    f_path = os.path.join(cache_dir, "F.npy")
    if os.path.exists(f_path):
        Fe = np.ascontiguousarray(np.load(f_path)).astype(np.float32)
    else:
        print("UYARI: %s yok -- ozellik dali sifirlarla calisacak" % f_path)
        Fe = np.zeros((len(y), N_FEATURES), dtype=np.float32)

    with open(os.path.join(cache_dir, "index.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    split = np.array([r["split"] for r in rows])
    records = np.array([r["record"] for r in rows])
    ok = np.array([int(r.get("ok", 1)) for r in rows], dtype=bool)

    if len(rows) != len(y):
        raise SystemExit("index.csv has %d rows but y.npy has %d"
                         % (len(rows), len(y)))
    return X, Fe, y, split, records, ok


def make_folds(y_dev, n_folds, seed):
    """Stratified fold assignment; deterministic for a given (n_folds, seed)."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y_dev), dtype=np.int64)
    for cls in np.unique(y_dev):
        idx = np.flatnonzero(y_dev == cls)
        idx = idx[rng.permutation(len(idx))]
        fold_of[idx] = np.arange(len(idx)) % n_folds
    return fold_of


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

class Aug:
    """Batch-level augmentation, applied on the torch tensor.

    Every transform has its own flag so FAZ 6.5 can ablate them one at a time:
    ``--aug_off noise,freq_mask``. The defaults are the aggressive set the task
    describes; if a model overfits, reach for dropout and weight decay first --
    turning augmentation *up* is the wrong lever here.
    """

    ALL = ("shift", "scale", "lead_scale", "noise", "baseline",
           "lead_drop", "freq_mask")

    def __init__(self, disabled=(), max_shift=0.10, scale_range=0.20,
                 lead_scale=0.15, noise=0.05, baseline=0.10,
                 lead_drop_p=0.20, freq_mask_p=0.25):
        self.disabled = set(disabled)
        self.max_shift = max_shift
        self.scale_range = scale_range
        self.lead_scale = lead_scale
        self.noise = noise
        self.baseline = baseline
        self.lead_drop_p = lead_drop_p
        self.freq_mask_p = freq_mask_p

    def on(self, name):
        return name not in self.disabled

    def __call__(self, x):
        b, c, t = x.shape

        if self.on("shift"):
            max_shift = int(self.max_shift * t)
            if max_shift > 0:
                shifts = torch.randint(-max_shift, max_shift + 1, (b,))
                idx = (torch.arange(t).unsqueeze(0) - shifts.unsqueeze(1)) % t
                x = torch.gather(x, 2, idx.unsqueeze(1).expand(b, c, t))

        if self.on("scale"):
            g = 1.0 + (torch.rand(b, 1, 1) * 2 - 1) * self.scale_range
            x = x * g

        if self.on("lead_scale"):
            g = 1.0 + (torch.rand(b, c, 1) * 2 - 1) * self.lead_scale
            x = x * g

        if self.on("baseline"):
            # Slow sinusoidal wander, the artefact the high-pass is fighting.
            freq = torch.rand(b, 1, 1) * 0.5 + 0.05
            phase = torch.rand(b, 1, 1) * 6.2832
            grid = torch.arange(t, dtype=x.dtype).view(1, 1, t) / t
            x = x + self.baseline * torch.sin(6.2832 * freq * grid * 10 + phase)

        if self.on("noise"):
            x = x + torch.randn_like(x) * self.noise

        if self.on("lead_drop"):
            keep = (torch.rand(b, c, 1) > self.lead_drop_p).to(x.dtype)
            # Never drop every lead at once.
            all_gone = keep.sum(dim=1, keepdim=True) == 0
            keep = torch.where(all_gone, torch.ones_like(keep), keep)
            x = x * keep

        if self.on("freq_mask"):
            mask = torch.rand(b) < self.freq_mask_p
            if bool(mask.any()):
                sel = x[mask]
                spec = torch.fft.rfft(sel.float(), dim=2)
                n_bins = spec.shape[2]
                width = max(int(0.08 * n_bins), 1)
                starts = torch.randint(0, max(n_bins - width, 1),
                                       (sel.shape[0],))
                bins = torch.arange(n_bins).view(1, 1, n_bins)
                lo = starts.view(-1, 1, 1)
                band = (bins >= lo) & (bins < lo + width)
                spec = spec * (~band).to(spec.real.dtype)
                x = x.clone()
                x[mask] = torch.fft.irfft(spec, n=x.shape[2], dim=2).to(x.dtype)

        return x


def mixup(x, f, target, alpha):
    """Convex-combine a batch with a shuffled copy of itself."""
    if alpha <= 0:
        return x, f, target
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)               # keep the dominant sample dominant
    perm = torch.randperm(x.shape[0])
    return (lam * x + (1 - lam) * x[perm],
            lam * f + (1 - lam) * f[perm],
            lam * target + (1 - lam) * target[perm])


class EMA:
    """Exponential moving average of the weights; evaluated instead of raw.

    The decay is warmed up as ``(1 + n) / (10 + n)`` capped at ``decay``. Without
    that ramp a short run is dominated by the random initialisation -- at decay
    0.999 after 54 steps the average is still 95 % init, which shows up as a
    model that confidently predicts one class.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.steps = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self, model):
        out = {}
        for k, v in model.state_dict().items():
            out[k] = self.shadow[k].to(v.dtype) if k in self.shadow \
                else v.detach().clone()
        return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def macro_f1(y_true, y_pred, n_classes=N_CLASSES):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)), [float(v) for v in f1s]


def binary_afib_afl(y_true, prob):
    """Accuracy inside the AFIB/AFL pair -- the metric that actually moves."""
    mask = np.isin(y_true, [1, 2])
    if not np.any(mask):
        return float("nan")
    pair = prob[mask][:, [1, 2]]
    pred = np.where(pair[:, 0] >= pair[:, 1], 1, 2)
    return float(np.mean(pred == y_true[mask]))


# --------------------------------------------------------------------------
# evaluation with test-time augmentation
# --------------------------------------------------------------------------

def _shift(x, frac):
    if frac == 0:
        return x
    n = int(round(frac * x.shape[2]))
    return torch.roll(x, shifts=n, dims=2)


LIMB_LEADS = [0, 1, 2, 3, 4, 5]
PRECORDIAL_LEADS = [6, 7, 8, 9, 10, 11]


@torch.no_grad()
def evaluate(model, X, Fe, idx, batch=64, tta="shift3", autocast=False):
    """Return class probabilities for ``idx``, averaged over TTA views."""
    model.eval()
    views = [("shift", 0.0)]
    parts = set(tta.split(",")) if tta else set()
    if "shift3" in parts or "shift" in parts:
        views = [("shift", -0.05), ("shift", 0.0), ("shift", 0.05)]
    if "shift5" in parts:
        views = [("shift", f) for f in (-0.08, -0.04, 0.0, 0.04, 0.08)]
    if "scale" in parts:
        views += [("scale", 0.9), ("scale", 1.1)]
    if "leads" in parts:
        views += [("leads", "limb"), ("leads", "precordial")]

    out = np.zeros((len(idx), N_CLASSES), dtype=np.float64)
    for start in range(0, len(idx), batch):
        sel = idx[start:start + batch]
        xb = torch.from_numpy(np.ascontiguousarray(X[sel])).float()
        fb = torch.from_numpy(np.ascontiguousarray(Fe[sel])).float()

        acc = torch.zeros(len(sel), N_CLASSES, dtype=torch.float64)
        for kind, arg in views:
            xv = xb
            if kind == "shift":
                xv = _shift(xb, arg)
            elif kind == "scale":
                xv = xb * arg
            elif kind == "leads":
                xv = xb.clone()
                keep = LIMB_LEADS if arg == "limb" else PRECORDIAL_LEADS
                mask = torch.zeros(xb.shape[1], 1)
                mask[keep] = 1.0
                xv = xv * mask

            if autocast:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = model(xv, fb)
            else:
                logits = model(xv, fb)
            acc += torch.softmax(logits.float(), dim=1).double()

        out[start:start + len(sel)] = (acc / len(views)).numpy()
    return out


# --------------------------------------------------------------------------
# one fold
# --------------------------------------------------------------------------

def train_one_fold(args, fold, X, Fe, y, dev_idx, fold_of, test_idx, run_dir):
    fold_dir = os.path.join(run_dir, "fold%d" % fold)
    os.makedirs(fold_dir, exist_ok=True)
    done_path = os.path.join(fold_dir, "done.json")

    if os.path.exists(done_path) and not args.force:
        with open(done_path) as fh:
            done = json.load(fh)
        print("fold %d zaten bitmis, atlaniyor (val macro-F1 %.4f)"
              % (fold, done.get("val_f1", float("nan"))))
        done["val_prob"] = np.load(os.path.join(fold_dir, "val_prob.npy"))
        done["test_prob"] = np.load(os.path.join(fold_dir, "test_prob.npy"))
        done["val_idx"] = np.load(os.path.join(fold_dir, "val_idx.npy"))
        return done

    tr_idx = dev_idx[fold_of != fold]
    va_idx = dev_idx[fold_of == fold]

    if args.exclude:
        excluded = set(np.load(args.exclude).tolist())
        before = len(tr_idx)
        tr_idx = np.array([i for i in tr_idx if int(i) not in excluded])
        print("  --exclude: egitimden %d kayit cikarildi (val/test dokunulmadi)"
              % (before - len(tr_idx)))

    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)

    model = net(args)

    # Feature scaler from the training fold only. Fitting it on the whole dev
    # set would leak the fold's own validation rows into its normalisation.
    mean = Fe[tr_idx].mean(axis=0)
    std = Fe[tr_idx].std(axis=0)
    model.set_feature_scaler(mean, std)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    ema = EMA(model, decay=args.ema)
    aug = Aug(disabled=args.aug_off.split(",") if args.aug_off else ())

    steps_per_epoch = max(int(np.ceil(len(tr_idx) / args.batch)), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup = steps_per_epoch * args.warmup_epochs

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    y_tr = torch.from_numpy(y[tr_idx]).long()
    onehot = torch.zeros(len(tr_idx), N_CLASSES)
    onehot[torch.arange(len(tr_idx)), y_tr] = 1.0
    if args.label_smoothing > 0:
        onehot = onehot * (1 - args.label_smoothing) \
            + args.label_smoothing / N_CLASSES

    best = {"val_f1": -1.0, "epoch": -1}
    history, step, stale = [], 0, 0
    t_start = time.time()

    # One reusable module for evaluating the EMA weights, rather than rebuilding
    # the whole network every epoch.
    eval_model = net(args)

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        epoch_loss, t_epoch = 0.0, time.time()

        for bstart in range(0, len(perm), args.batch):
            sel = perm[bstart:bstart + args.batch]
            if len(sel) < 2:
                continue                       # BatchNorm needs at least two
            rows = tr_idx[sel]

            xb = torch.from_numpy(np.ascontiguousarray(X[rows])).float()
            fb = torch.from_numpy(np.ascontiguousarray(Fe[rows])).float()
            tb = onehot[sel]

            xb = aug(xb)
            xb, fb, tb = mixup(xb, fb, tb, args.mixup)

            for group in opt.param_groups:
                group["lr"] = lr_at(step)

            if args.bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = model(xb, fb)
                    loss = -(tb * torch.log_softmax(logits.float(), dim=1)).sum(1).mean()
            else:
                logits = model(xb, fb)
                loss = -(tb * torch.log_softmax(logits, dim=1)).sum(1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            ema.update(model)

            epoch_loss += float(loss.detach()) * len(sel)
            step += 1

        # Evaluate the EMA weights; they are what gets saved and exported.
        eval_model.load_state_dict(ema.state_dict(model))
        val_prob = evaluate(eval_model, X, Fe, va_idx, args.eval_batch,
                            args.tta, args.bf16)
        f1, per_class = macro_f1(y[va_idx], val_prob.argmax(1))
        pair_acc = binary_afib_afl(y[va_idx], val_prob)

        history.append({"epoch": epoch, "loss": epoch_loss / max(len(perm), 1),
                        "val_f1": f1, "afib_afl": pair_acc,
                        "lr": lr_at(step), "sec": round(time.time() - t_epoch, 1)})
        print("  ep %2d/%d  loss %.4f  val macro-F1 %.4f  AFIB/AFL %.4f  %.0fs"
              % (epoch + 1, args.epochs, history[-1]["loss"], f1, pair_acc,
                 history[-1]["sec"]), flush=True)

        if f1 > best["val_f1"]:
            best = {"val_f1": f1, "epoch": epoch, "per_class": per_class,
                    "afib_afl": pair_acc}
            torch.save({"state_dict": eval_model.state_dict(),
                        "preset": args.preset, "dropout": args.dropout,
                        "use_features": not args.no_features,
                        "in_ch": IN_CH, "n_features": N_FEAT,
                        "feat_mean": mean.tolist(), "feat_std": std.tolist(),
                        "epoch": epoch, "val_f1": f1},
                       os.path.join(fold_dir, "best.pt"))
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("  erken durdurma (patience %d)" % args.patience)
                break

    # Reload the best checkpoint for the fold's final predictions.
    ckpt = torch.load(os.path.join(fold_dir, "best.pt"), weights_only=False)
    final = net(args)
    final.load_state_dict(ckpt["state_dict"])

    val_prob = evaluate(final, X, Fe, va_idx, args.eval_batch, args.tta, args.bf16)
    test_prob = evaluate(final, X, Fe, test_idx, args.eval_batch, args.tta,
                         args.bf16) if len(test_idx) else np.zeros((0, N_CLASSES))

    val_f1, per_class = macro_f1(y[va_idx], val_prob.argmax(1))
    test_f1 = macro_f1(y[test_idx], test_prob.argmax(1))[0] if len(test_idx) else float("nan")

    np.save(os.path.join(fold_dir, "val_prob.npy"), val_prob)
    np.save(os.path.join(fold_dir, "test_prob.npy"), test_prob)
    np.save(os.path.join(fold_dir, "val_idx.npy"), va_idx)

    done = {
        "fold": fold, "val_f1": val_f1, "test_f1": test_f1,
        "val_per_class": per_class,
        "val_afib_afl": binary_afib_afl(y[va_idx], val_prob),
        "test_afib_afl": binary_afib_afl(y[test_idx], test_prob) if len(test_idx) else float("nan"),
        "best_epoch": int(ckpt["epoch"]), "epochs_run": len(history),
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "minutes": round((time.time() - t_start) / 60.0, 2),
        "history": history,
    }
    with open(done_path, "w") as fh:
        json.dump(done, fh, indent=2)

    print("  fold %d bitti: val %.4f | test %.4f | en iyi epoch %d | %.1f dk"
          % (fold, val_f1, test_f1, done["best_epoch"], done["minutes"]))

    done["val_prob"] = val_prob
    done["test_prob"] = test_prob
    done["val_idx"] = va_idx
    return done


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="r18", choices=sorted(PRESETS))
    ap.add_argument("--tag", default=None, help="run name under runs/")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=99,
                    help="99 lets cosine fully anneal, which beat early stopping")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mixup", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", default="shift3",
                    help="comma list: shift3, shift5, scale, leads, or none")
    ap.add_argument("--aug_off", default="",
                    help="comma list of augmentations to disable (FAZ 6.5)")
    ap.add_argument("--no_features", action="store_true",
                    help="ablate the 37-feature branch")
    ap.add_argument("--exclude", default="",
                    help=".npy of cache row indices to drop from TRAINING only")
    ap.add_argument("--full", action="store_true",
                    help="after CV, also train one model on all dev data")
    ap.add_argument("--full_epochs", type=int, default=0,
                    help="epochs for --full (default: median best epoch + 1)")
    ap.add_argument("--bf16", dest="bf16", action="store_true", default=None)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--mmap", action="store_true",
                    help="keep X.npy on disk; use when RAM is under 8 GB")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="retrain folds that already have done.json")
    return ap


def cpu_supports_bf16():
    """AVX512-BF16 or AMX. Without one of them bf16 autocast only adds overhead."""
    try:
        with open("/proc/cpuinfo") as fh:
            flags = fh.read()
        return ("avx512_bf16" in flags) or ("amx_bf16" in flags)
    except OSError:
        return False


def main(argv=None):
    args = build_argparser().parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.bf16 is None:
        args.bf16 = cpu_supports_bf16()
        if not args.bf16:
            print("bf16 kapali: CPU'da AVX512-BF16/AMX yok, bf16 hizlandirmaz")
    elif args.bf16 and not cpu_supports_bf16():
        print("UYARI: --bf16 istendi ama CPU destegi yok; muhtemelen yavaslar")

    args.tag = args.tag or ("%s_f%d" % (args.preset, args.folds))
    run_dir = os.path.join("runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)

    X, Fe, y, split, records, ok = load_cache(args.cache, mmap=args.mmap)

    global IN_CH, N_FEAT
    IN_CH = int(X.shape[1])
    N_FEAT = int(Fe.shape[1])
    if IN_CH != 12:
        print("cache %d kanalli (12 derivasyon + %d ek kanal)"
              % (IN_CH, IN_CH - 12))
    if N_FEAT != N_FEATURES:
        print("cache %d ozellikli (%d + %d ek olcum)"
              % (N_FEAT, N_FEATURES, N_FEAT - N_FEATURES))

    usable = ok & (y >= 0)
    dev_idx = np.flatnonzero(np.isin(split, DEV_SPLITS) & usable)
    test_idx = np.flatnonzero((split == TEST_SPLIT) & usable)
    if len(dev_idx) == 0:
        raise SystemExit("development set is empty")

    fold_of = make_folds(y[dev_idx], args.folds, args.seed)

    print("preset=%s params=%s cache=%s" % (
        args.preset, "{:,}".format(count_parameters(net(args))),
        args.cache))
    print("gelistirme=%d  test_public=%d  fold=%d  epoch=%d  bf16=%s"
          % (len(dev_idx), len(test_idx), args.folds, args.epochs, args.bf16))
    print("giris=%s  ozellik=%s" % ((X.shape[1], X.shape[2]),
                                    "kapali" if args.no_features else "37"))

    folds = [args.only_fold] if args.only_fold is not None else list(range(args.folds))
    results, t0 = [], time.time()

    for fold in folds:
        print("\n--- fold %d/%d ---" % (fold, args.folds))
        results.append(train_one_fold(args, fold, X, Fe, y, dev_idx, fold_of,
                                      test_idx, run_dir))

    with open(os.path.join(run_dir, "args.json"), "w") as fh:
        json.dump({k: v for k, v in vars(args).items()}, fh, indent=2)

    partial = args.only_fold is not None or len(folds) < args.folds
    if partial:
        r = results[0]
        print("\n=== TEK FOLD (%s, fold %d) ===" % (args.tag, r["fold"]))
        print("  val (OOF) macro-F1  : %.4f   <- kararlari BUNUNLA ver" % r["val_f1"])
        print("  val AFIB/AFL        : %.4f" % r["val_afib_afl"])
        print("  test_public macro-F1: %.4f   (sadece rapor icin)" % r["test_f1"])
        print("\nsummary.json YAZILMADI -- tek fold bir sonuc degil.")
        with open(os.path.join(run_dir, "summary_fold%d.json" % r["fold"]),
                  "w") as fh:
            json.dump({k: v for k, v in r.items()
                       if k not in ("val_prob", "test_prob", "val_idx")},
                      fh, indent=2)
        return 0

    # --- assemble the out-of-fold matrix over the whole development set ---
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    covered = np.zeros(len(y), dtype=bool)
    for r in results:
        oof[r["val_idx"]] = r["val_prob"]
        covered[r["val_idx"]] = True

    test_prob = np.mean([r["test_prob"] for r in results], axis=0) \
        if len(test_idx) else np.zeros((0, N_CLASSES))

    oof_f1, oof_per_class = macro_f1(y[dev_idx], oof[dev_idx].argmax(1))
    oof_pair = binary_afib_afl(y[dev_idx], oof[dev_idx])
    test_f1, test_per_class = (macro_f1(y[test_idx], test_prob.argmax(1))
                               if len(test_idx) else (float("nan"), []))
    test_pair = binary_afib_afl(y[test_idx], test_prob) if len(test_idx) else float("nan")

    np.save(os.path.join(run_dir, "oof_prob.npy"), oof)
    np.save(os.path.join(run_dir, "test_prob.npy"), test_prob)
    np.save(os.path.join(run_dir, "dev_idx.npy"), dev_idx)
    np.save(os.path.join(run_dir, "test_idx.npy"), test_idx)

    if args.full:
        train_full(args, X, Fe, y, dev_idx, results, run_dir)

    summary = {
        "tag": args.tag, "preset": args.preset, "cache": args.cache,
        "folds": args.folds, "epochs": args.epochs, "seed": args.seed,
        "params": count_parameters(net(args)),
        "oof_macro_f1": oof_f1, "oof_per_class": oof_per_class,
        "oof_afib_afl": oof_pair,
        "test_macro_f1": test_f1, "test_per_class": test_per_class,
        "test_afib_afl": test_pair,
        "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
        "coverage": float(covered[dev_idx].mean()),
        "fold_val_f1": [r["val_f1"] for r in results],
        "fold_test_f1": [r["test_f1"] for r in results],
        "median_best_epoch": int(np.median([r["best_epoch"] for r in results])),
        "total_minutes": round((time.time() - t0) / 60.0, 2),
        "args": {k: v for k, v in vars(args).items()},
        "classes": list(ep.CLASSES),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== %s ===" % args.tag)
    print("  OOF macro-F1     : %.4f   <- tum kararlar bununla" % oof_f1)
    print("  OOF AFIB/AFL     : %.4f" % oof_pair)
    print("  OOF sinif F1     : %s"
          % "  ".join("%s=%.3f" % (c, v) for c, v in zip(ep.CLASSES, oof_per_class)))
    print("  fold val F1      : %s"
          % "  ".join("%.4f" % v for v in summary["fold_val_f1"]))
    print("  test_public F1   : %.4f   (rapor icin, secim icin degil)" % test_f1)
    print("  sure             : %.1f dk" % summary["total_minutes"])
    print("\nyazildi: %s/summary.json" % run_dir)
    return 0


def train_full(args, X, Fe, y, dev_idx, results, run_dir):
    """FAZ 6.1: retrain on the whole development set for the median epoch count."""
    epochs = args.full_epochs or int(np.median([r["best_epoch"] for r in results])) + 1
    print("\n--- tum veriyle yeniden egitim (%d epoch, %d kayit) ---"
          % (epochs, len(dev_idx)))

    full_dir = os.path.join(run_dir, "full")
    os.makedirs(full_dir, exist_ok=True)

    torch.manual_seed(args.seed + 999)
    np.random.seed(args.seed + 999)

    model = net(args)
    model.set_feature_scaler(Fe[dev_idx].mean(axis=0), Fe[dev_idx].std(axis=0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = EMA(model, decay=args.ema)
    aug = Aug(disabled=args.aug_off.split(",") if args.aug_off else ())

    steps_per_epoch = max(int(np.ceil(len(dev_idx) / args.batch)), 1)
    total = steps_per_epoch * epochs
    warmup = steps_per_epoch * args.warmup_epochs

    onehot = torch.zeros(len(dev_idx), N_CLASSES)
    onehot[torch.arange(len(dev_idx)), torch.from_numpy(y[dev_idx]).long()] = 1.0

    step = 0
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(dev_idx))
        t_epoch = time.time()
        for bstart in range(0, len(perm), args.batch):
            sel = perm[bstart:bstart + args.batch]
            if len(sel) < 2:
                continue
            rows = dev_idx[sel]
            xb = aug(torch.from_numpy(np.ascontiguousarray(X[rows])).float())
            fb = torch.from_numpy(np.ascontiguousarray(Fe[rows])).float()
            xb, fb, tb = mixup(xb, fb, onehot[sel], args.mixup)

            lr = (args.lr * (step + 1) / max(warmup, 1) if step < warmup
                  else args.lr * 0.5 * (1 + np.cos(np.pi * min(
                      (step - warmup) / max(total - warmup, 1), 1.0))))
            for g in opt.param_groups:
                g["lr"] = lr

            logits = model(xb, fb)
            loss = -(tb * torch.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            ema.update(model)
            step += 1
        print("  full ep %2d/%d  %.0fs" % (epoch + 1, epochs, time.time() - t_epoch),
              flush=True)

    final = net(args)
    final.load_state_dict(ema.state_dict(model))
    torch.save({"state_dict": final.state_dict(), "preset": args.preset,
                "dropout": args.dropout, "use_features": not args.no_features,
                "in_ch": IN_CH, "n_features": N_FEAT,
                "epochs": epochs, "trained_on": "all dev"},
               os.path.join(full_dir, "best.pt"))
    print("  yazildi: %s/best.pt  (OOF'u YOK -- ensemble'a ek uye olarak kullan)"
          % full_dir)


if __name__ == "__main__":
    raise SystemExit(main())
