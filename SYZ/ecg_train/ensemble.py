"""ensemble -- pick the combination rule, using out-of-fold scores only.

    python ensemble.py
    python ensemble.py --members runs/main_v2 baseline/r18_feat --out ensemble.json

Every candidate rule is scored on the development set's out-of-fold matrix.
test_public is loaded only at the very end, printed once for the report, and
takes no part in choosing the rule, the weights, or the members. That ordering
is enforced in code: ``choose_rule`` is never handed the test arrays.

Three rules are compared:

  flat      plain average of member probabilities
  weighted  per-member weights from a Dirichlet random search plus coordinate
            refinement, maximising OOF macro-F1
  stacked   multinomial logistic regression over the concatenated member
            probabilities

The stacked score is measured with an inner cross-validation over the
development rows. Fitting a stacker on the OOF matrix and scoring it on that
same matrix is optimistic by roughly the amount stacking appears to win, which
is how "stacking beats weighting" turns into a result that does not reproduce.

Member contract
---------------
A member directory holds:
  oof_prob.npy   (n_cache_rows, 5)  out-of-fold probabilities, zeros off-dev
  test_prob.npy  (n_test, 5)        fold-averaged test_public probabilities
  summary.json   optional, for the printed table
Row order matches the cache's index.csv. train.py writes exactly this layout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ecg_preprocess as ep
from train import DEV_SPLITS, TEST_SPLIT, binary_afib_afl, load_cache, macro_f1

N_CLASSES = len(ep.CLASSES)


# --------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------

def discover_members(explicit=None, run_glob="runs/*", baseline_glob="baseline/*"):
    """Find directories that carry a usable OOF matrix."""
    if explicit:
        candidates = list(explicit)
    else:
        candidates = sorted(glob.glob(run_glob)) + sorted(glob.glob(baseline_glob))

    members = []
    for path in candidates:
        oof = os.path.join(path, "oof_prob.npy")
        test = os.path.join(path, "test_prob.npy")
        if not (os.path.exists(oof) and os.path.exists(test)):
            continue
        summary = {}
        spath = os.path.join(path, "summary.json")
        if os.path.exists(spath):
            with open(spath) as fh:
                summary = json.load(fh)
        members.append({"name": os.path.basename(path.rstrip("/\\")),
                        "path": path,
                        "oof": np.load(oof).astype(np.float64),
                        "test": np.load(test).astype(np.float64),
                        "summary": summary})
    return members


def validate_members(members, n_rows, n_test, dev_idx):
    """Drop members whose arrays do not line up with the cache."""
    keep = []
    for m in members:
        if m["oof"].shape != (n_rows, N_CLASSES):
            print("  ATLANDI %-22s oof_prob %s, beklenen (%d, %d)"
                  % (m["name"], m["oof"].shape, n_rows, N_CLASSES))
            continue
        if n_test and m["test"].shape != (n_test, N_CLASSES):
            print("  ATLANDI %-22s test_prob %s, beklenen (%d, %d)"
                  % (m["name"], m["test"].shape, n_test, N_CLASSES))
            continue
        covered = m["oof"][dev_idx].sum(axis=1)
        if np.mean(covered > 1e-6) < 0.99:
            print("  ATLANDI %-22s OOF kapsamasi %.1f%% (tam 5-fold degil?)"
                  % (m["name"], 100 * np.mean(covered > 1e-6)))
            continue
        keep.append(m)
    return keep


def _normalise(prob):
    s = prob.sum(axis=1, keepdims=True)
    return prob / np.where(s < 1e-12, 1.0, s)


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

def blend(mats, weights):
    out = np.zeros_like(mats[0])
    for w, m in zip(weights, mats):
        out += w * m
    return _normalise(out)


def search_weights(oof_mats, y, n_random=4000, refine_rounds=3, seed=0):
    """Dirichlet random search, then coordinate refinement. OOF only."""
    rng = np.random.default_rng(seed)
    k = len(oof_mats)
    best_w = np.ones(k) / k
    best = macro_f1(y, blend(oof_mats, best_w).argmax(1))[0]

    for _ in range(n_random):
        w = rng.dirichlet(np.ones(k) * 0.7)
        score = macro_f1(y, blend(oof_mats, w).argmax(1))[0]
        if score > best:
            best, best_w = score, w

    for step in (0.20, 0.10, 0.05):
        for _ in range(refine_rounds):
            improved = False
            for i in range(k):
                for delta in (step, -step):
                    w = best_w.copy()
                    w[i] = max(w[i] + delta, 0.0)
                    total = w.sum()
                    if total <= 0:
                        continue
                    w /= total
                    score = macro_f1(y, blend(oof_mats, w).argmax(1))[0]
                    if score > best + 1e-9:
                        best, best_w, improved = score, w, True
            if not improved:
                break
    return best_w, best


def _fit_stacker(features, y, seed=0):
    from sklearn.linear_model import LogisticRegression

    # multi_class= was removed in scikit-learn 1.9; multinomial is the default.
    return LogisticRegression(max_iter=2000, C=1.0,
                              random_state=seed).fit(features, y)


def stacked_score(oof_mats, y, n_inner=5, seed=0):
    """Honest stacking estimate via inner CV over the development rows."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("  (sklearn yok, stacking atlandi)")
        return None, float("-inf")

    features = np.concatenate([np.log(np.clip(m, 1e-9, 1.0)) for m in oof_mats],
                              axis=1)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.int64)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        fold_of[idx] = np.arange(len(idx)) % n_inner

    pred = np.zeros(len(y), dtype=np.int64)
    for f in range(n_inner):
        tr, va = fold_of != f, fold_of == f
        clf = _fit_stacker(features[tr], y[tr], seed)
        pred[va] = clf.predict(features[va])

    score = macro_f1(y, pred)[0]
    final = _fit_stacker(features, y, seed)      # refit on everything for use
    return final, score


def choose_rule(oof_mats, y_dev, seed=0, weight_iters=4000):
    """Compare the three rules on OOF alone. Never sees test_public."""
    rules = {}

    flat_w = np.ones(len(oof_mats)) / len(oof_mats)
    rules["flat"] = {"weights": flat_w,
                     "oof": macro_f1(y_dev, blend(oof_mats, flat_w).argmax(1))[0]}

    w, score = search_weights(oof_mats, y_dev, n_random=weight_iters, seed=seed)
    rules["weighted"] = {"weights": w, "oof": score}

    stacker, sscore = stacked_score(oof_mats, y_dev, seed=seed)
    if stacker is not None:
        rules["stacked"] = {"model": stacker, "oof": sscore}

    best_name = max(rules, key=lambda k: rules[k]["oof"])

    # Prefer the simpler rule when the difference is inside fold noise. A 0.001
    # OOF edge for stacking is not a reason to ship a second model.
    margin = 0.005
    if best_name == "stacked" and rules["stacked"]["oof"] - max(
            rules["flat"]["oof"], rules["weighted"]["oof"]) < margin:
        best_name = "weighted" if rules["weighted"]["oof"] > rules["flat"]["oof"] \
            else "flat"
    if best_name == "weighted" and rules["weighted"]["oof"] - rules["flat"]["oof"] < 0.002:
        best_name = "flat"

    return best_name, rules


def apply_rule(name, rules, mats):
    if name == "stacked":
        features = np.concatenate([np.log(np.clip(m, 1e-9, 1.0)) for m in mats],
                                  axis=1)
        return rules["stacked"]["model"].predict_proba(features)
    return blend(mats, rules[name]["weights"])


# --------------------------------------------------------------------------
# optional AFIB/AFL expert
# --------------------------------------------------------------------------

def integrate_expert(prob_dev, prob_test, expert_dir, y_dev, dev_idx, test_idx):
    """Blend a two-class AFIB/AFL specialist in, with alpha chosen on OOF.

    Kept because GOREV.md asks for it, and because the honest answer is worth
    recording: on the existing package this bought +0.001, i.e. nothing. The
    search below will report the same shape of result rather than hiding it.
    """
    oof_path = os.path.join(expert_dir, "oof_prob.npy")
    test_path = os.path.join(expert_dir, "test_prob.npy")
    if not os.path.exists(oof_path):
        print("uzman bulunamadi: %s" % oof_path)
        return prob_dev, prob_test, None

    e_oof = np.load(oof_path).astype(np.float64)
    e_test = np.load(test_path).astype(np.float64)
    if e_oof.shape[1] != 2:
        print("uzman 2 sinifli olmali, %d bulundu -- atlandi" % e_oof.shape[1])
        return prob_dev, prob_test, None

    e_dev = _normalise(e_oof[dev_idx])
    base = macro_f1(y_dev, prob_dev.argmax(1))[0]

    best_alpha, best_score = 0.0, base
    for alpha in np.arange(0.0, 1.01, 0.05):
        mixed = prob_dev.copy()
        pair = mixed[:, [1, 2]]
        total = pair.sum(axis=1, keepdims=True)
        blended = (1 - alpha) * _normalise(pair) + alpha * e_dev
        mixed[:, [1, 2]] = blended * total
        score = macro_f1(y_dev, _normalise(mixed).argmax(1))[0]
        if score > best_score + 1e-9:
            best_alpha, best_score = float(alpha), score

    print("uzman entegrasyonu: alpha=%.2f  OOF %.4f -> %.4f  (%+.4f)"
          % (best_alpha, base, best_score, best_score - base))
    if best_alpha == 0.0:
        print("  alpha=0 secildi -- uzman katki vermiyor, disarida birakildi")
        return prob_dev, prob_test, {"alpha": 0.0, "oof_delta": 0.0}

    def mix(prob, expert):
        out = prob.copy()
        pair = out[:, [1, 2]]
        total = pair.sum(axis=1, keepdims=True)
        out[:, [1, 2]] = ((1 - best_alpha) * _normalise(pair)
                          + best_alpha * _normalise(expert)) * total
        return _normalise(out)

    return (mix(prob_dev, e_dev), mix(prob_test, e_test[:len(test_idx)]),
            {"alpha": best_alpha, "oof_delta": best_score - base})


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--members", nargs="*", default=None,
                    help="member directories (default: runs/* and baseline/*)")
    ap.add_argument("--out", default="ensemble.json")
    ap.add_argument("--expert", default="", help="2-class AFIB/AFL run directory")
    ap.add_argument("--weight_iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_members", type=int, default=1)
    args = ap.parse_args(argv)

    X, Fe, y, split, records, ok = load_cache(args.cache, mmap=True)
    del X, Fe
    usable = ok & (y >= 0)
    dev_idx = np.flatnonzero(np.isin(split, DEV_SPLITS) & usable)
    test_idx = np.flatnonzero((split == TEST_SPLIT) & usable)
    y_dev, y_test = y[dev_idx], y[test_idx]

    print("uye araniyor...")
    members = discover_members(args.members)
    members = validate_members(members, len(y), len(test_idx), dev_idx)
    if len(members) < args.min_members:
        raise SystemExit("kullanilabilir uye yok -- once 'python train.py' ile "
                         "tam 5-fold kos (summary.json ve oof_prob.npy gerekli)")

    print("\n%-24s %10s %10s  %s" % ("uye", "OOF F1", "AFIB/AFL", "kaynak"))
    oof_mats, test_mats = [], []
    for m in members:
        dev_prob = _normalise(m["oof"][dev_idx])
        oof_mats.append(dev_prob)
        test_mats.append(_normalise(m["test"]) if len(test_idx) else m["test"])
        m["oof_f1"] = macro_f1(y_dev, dev_prob.argmax(1))[0]
        m["oof_pair"] = binary_afib_afl(y_dev, dev_prob)
        print("%-24s %10.4f %10.4f  %s"
              % (m["name"], m["oof_f1"], m["oof_pair"], m["path"]))

    # --- error correlation: the FAZ 5 diversity gate ---
    if len(members) > 1:
        print("\nayni tahmini verme orani (dusuk = gercek cesitlilik):")
        preds = [m.argmax(1) for m in oof_mats]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                agree = float((preds[i] == preds[j]).mean())
                flag = "  <- 0.85 alti, cesitlilik var" if agree < 0.85 else ""
                print("  %-18s %-18s %.4f%s"
                      % (members[i]["name"], members[j]["name"], agree, flag))

    print("\nkural secimi (SADECE OOF)...")
    best_name, rules = choose_rule(oof_mats, y_dev, args.seed, args.weight_iters)
    for name in ("flat", "weighted", "stacked"):
        if name in rules:
            mark = " <- secildi" if name == best_name else ""
            print("  %-9s OOF %.4f%s" % (name, rules[name]["oof"], mark))

    dev_prob = apply_rule(best_name, rules, oof_mats)
    test_prob = apply_rule(best_name, rules, test_mats) if len(test_idx) \
        else np.zeros((0, N_CLASSES))

    expert_info = None
    if args.expert:
        dev_prob, test_prob, expert_info = integrate_expert(
            dev_prob, test_prob, args.expert, y_dev, dev_idx, test_idx)

    oof_f1, oof_per_class = macro_f1(y_dev, dev_prob.argmax(1))
    oof_pair = binary_afib_afl(y_dev, dev_prob)

    weights = rules.get(best_name, {}).get("weights")
    result = {
        "method": best_name,
        "members": [{"name": m["name"], "path": m["path"],
                     "oof_macro_f1": m["oof_f1"], "oof_afib_afl": m["oof_pair"],
                     "weight": (float(weights[i]) if weights is not None else None)}
                    for i, m in enumerate(members)],
        "oof_macro_f1": oof_f1,
        "oof_per_class": dict(zip(ep.CLASSES, oof_per_class)),
        "oof_afib_afl": oof_pair,
        "rule_scores": {k: v["oof"] for k, v in rules.items()},
        "expert": expert_info,
        "cache": os.path.abspath(args.cache),
        "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
        "selected_on": "out-of-fold development set only",
    }

    if best_name == "stacked":
        clf = rules["stacked"]["model"]
        result["stacker"] = {"coef": clf.coef_.tolist(),
                             "intercept": clf.intercept_.tolist(),
                             "input": "log of concatenated member probabilities"}

    print("\n=== SECILEN KURAL: %s ===" % best_name)
    print("  ensemble OOF macro-F1 : %.4f" % oof_f1)
    print("  ensemble OOF AFIB/AFL : %.4f" % oof_pair)
    print("  sinif F1: %s" % "  ".join("%s=%.3f" % (c, v)
                                       for c, v in zip(ep.CLASSES, oof_per_class)))
    best_single = max(m["oof_f1"] for m in members)
    print("  en iyi tek uye OOF    : %.4f  (ensemble farki %+.4f)"
          % (best_single, oof_f1 - best_single))

    # test_public is touched only here, after every choice is locked in.
    if len(test_idx):
        test_f1, test_per_class = macro_f1(y_test, test_prob.argmax(1))
        result["test_macro_f1"] = test_f1
        result["test_per_class"] = dict(zip(ep.CLASSES, test_per_class))
        result["test_afib_afl"] = binary_afib_afl(y_test, test_prob)
        np.save("ensemble_test_prob.npy", test_prob)
        print("\n  test_public macro-F1  : %.4f   (rapor icin, secim icin degil)"
              % test_f1)
        print("  test_public AFIB/AFL  : %.4f" % result["test_afib_afl"])

    # Written at full cache length with zeros off-development, matching the
    # member contract in baseline/README.md, so this file can be fed straight
    # back into afib_afl_diag.py or reused as an ensemble member.
    full_oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    full_oof[dev_idx] = dev_prob
    np.save("ensemble_oof_prob.npy", full_oof)

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nyazildi: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
