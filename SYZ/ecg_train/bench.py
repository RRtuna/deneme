"""bench -- how long will training take on THIS machine? Needs no data.

    python bench.py
    python bench.py --presets r18 wide w64 --dev 4250 --epochs 40

Times a real forward+backward step on random tensors of the exact training
shape, then extrapolates to seconds per epoch and hours for a full 5-fold run.
Use it before committing to a plan, so the phase budget comes from measurement
rather than hope.

Watch the bf16 column. bf16 autocast only pays off with AVX512-BF16 or AMX; on
a CPU without either it adds conversion overhead and runs *slower*. When the
report says bf16 is not supported, train with ``--no_bf16``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from model import N_FEATURES, PRESETS, build_model

N_LEADS = 12


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.lower().startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def describe_machine():
    flags = cpu_flags()
    name = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": name,
        "cores_logical": torch.get_num_threads(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "amx_bf16": "amx_bf16" in flags or "amx_tile" in flags,
    }


def time_steps(model, batch, length, bf16, n_steps, warmup=2):
    """Median seconds per forward+backward step."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, N_LEADS, length)
    f = torch.randn(batch, N_FEATURES)
    target = torch.randint(0, 5, (batch,))

    times = []
    for i in range(n_steps + warmup):
        t0 = time.perf_counter()
        if bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(x, f)
                loss = torch.nn.functional.cross_entropy(logits.float(), target)
        else:
            logits = model(x, f)
            loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--length", type=int, default=1500,
                    help="input samples: 1500 at 150 Hz, 2500 at 250 Hz")
    ap.add_argument("--dev", type=int, default=4250,
                    help="development-set size (train + validation)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", default="", help="also write the table here")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    machine = describe_machine()
    print("makine")
    print("  CPU          : %s" % machine["cpu"])
    print("  torch is parcacigi: %d" % machine["cores_logical"])
    print("  AVX2 %s | AVX512F %s | AVX512-BF16 %s | AMX %s"
          % (machine["avx2"], machine["avx512f"],
             machine["avx512_bf16"], machine["amx_bf16"]))

    bf16_useful = machine["avx512_bf16"] or machine["amx_bf16"]
    if not bf16_useful:
        print("\n  bf16 DESTEKLENMIYOR -- egitimi --no_bf16 ile kos.")
        print("  (bf16 autocast bu CPU'da hizlandirmaz, yavaslatir)")

    presets = args.presets or list(PRESETS)
    steps_per_epoch = int(np.ceil(args.dev * (args.folds - 1) / args.folds
                                  / args.batch))

    print("\nolcum: batch=%d  giris=%d ornek  fold basina %d adim/epoch"
          % (args.batch, args.length, steps_per_epoch))
    print("\n%-10s %10s %10s %10s %10s %10s"
          % ("preset", "params", "s/adim", "s/epoch", "5fx%de saat" % args.epochs,
             "bf16 s/adim"))
    print("-" * 66)

    rows = []
    for name in presets:
        if name not in PRESETS:
            print("%-10s bilinmeyen preset, atlandi" % name)
            continue
        try:
            model = build_model(name)
            params = sum(p.numel() for p in model.parameters())

            sec = time_steps(model, args.batch, args.length, False, args.steps)
            per_epoch = sec * steps_per_epoch
            hours = per_epoch * args.epochs * args.folds / 3600.0

            bf16_sec = float("nan")
            if bf16_useful or args.presets:
                try:
                    bf16_sec = time_steps(build_model(name), args.batch,
                                          args.length, True, max(args.steps // 2, 3))
                except Exception:               # noqa: BLE001
                    bf16_sec = float("nan")

            print("%-10s %10s %10.3f %10.1f %10.2f %10s"
                  % (name, "{:,}".format(params), sec, per_epoch, hours,
                     ("%.3f" % bf16_sec) if bf16_sec == bf16_sec else "-"))
            rows.append({"preset": name, "params": params, "sec_per_step": sec,
                         "sec_per_epoch": per_epoch, "hours_5fold": hours,
                         "bf16_sec_per_step": (bf16_sec if bf16_sec == bf16_sec
                                               else None)})
        except Exception as exc:                # noqa: BLE001
            print("%-10s HATA: %s" % (name, exc))

    if rows:
        print("\nbutce rehberi (bu makinenin olcumlerine gore)")
        for row in rows:
            single = row["hours_5fold"] / args.folds
            print("  %-10s tek fold %.2f saat  |  tam 5-fold %.2f saat"
                  % (row["preset"], single, row["hours_5fold"]))

        print("\nnot: FAZ 2 kapasite taramasi tek fold kosar, yani 5-fold "
              "maliyetinin 1/5'i.")
        print("     250 Hz'e gecince giris 1500 -> 2500 olur; sureyi yeniden")
        print("     olcmek icin: python bench.py --length 2500")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"machine": machine, "settings": vars(args), "rows": rows},
                      fh, indent=2)
        print("\nyazildi: %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
