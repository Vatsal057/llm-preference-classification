"""Local test of every part of notebook_src.py that does not need a GPU / Keras.

Exercises: JSON parsing, Algorithm 1, the leakage-free splits (on the real 57K rows),
the augmentation arms, and the calibration / bootstrap / metric maths in section 10.
"""
import re
import sys
import types
import numpy as np
import pandas as pd

from build_nb import parse

CELLS = parse("notebook_src.py")
CODE = [s for k, s in CELLS if k == "code"]


def cell_with(marker):
    hits = [c for c in CODE if marker in c]
    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells"
    return hits[0]


def extract_defs(src, names):
    """Pull top-level `def name(...)` blocks out of a cell."""
    out, lines = [], src.split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"^def (\w+)\(", lines[i])
        if m and m.group(1) in names:
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith((" ", "\t"))):
                j += 1
            out.append("\n".join(lines[i:j]))
            i = j
        else:
            i += 1
    got = {re.match(r"^def (\w+)", b).group(1) for b in out}
    assert got == set(names), f"missing {set(names) - got}"
    return "\n\n".join(out)


G = {"__name__": "nbtest"}
exec("import os, sys, json, time, math, gc, glob, itertools\n"
     "import numpy as np, pandas as pd\n"
     "from datetime import datetime, timezone", G)

# Locate dataset or prepare synthetic fallback
DATA_CANDIDATES = [
    "./data",
    "../data",
    "/kaggle/input/llm-classification-finetuning",
    "../LLM v1.1 Nancy's modification/data",
    "../LLM v2 lab work/data",
]
REAL_DATA_PATH = None
import os
for p in DATA_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")):
        REAL_DATA_PATH = p
        break

print("=" * 70)
print("1. CONFIG cell")
cfg_src = cell_with("class CFG:")
# force the real-run values so we test the shipping protocol, not the smoke path
cfg_src = cfg_src.replace("SMOKE_TEST = True", "SMOKE_TEST = False")
if REAL_DATA_PATH:
    cfg_src = cfg_src.replace("BASE_PATH = find_dataset()", f'BASE_PATH = "{REAL_DATA_PATH}"')
else:
    cfg_src = cfg_src.replace("BASE_PATH = find_dataset()", 'BASE_PATH = "./data"')

exec(cfg_src, G)
CFG = G["CFG"]
print(f"   TRAIN_N={CFG.TRAIN_N} EPOCHS={CFG.EPOCHS} SEQ_LEN={CFG.SEQ_LEN} "
      f"SEEDS={CFG.SEEDS} EVAL_N={CFG.EVAL_N}")
print(f"   BASE_PATH={G['BASE_PATH']}")

print("=" * 70)
print("2. Algorithm 1 / multi-turn cell")
exec(cell_with("def greedy_multi_turn_context"), G)
gm, st, pj = G["greedy_multi_turn_context"], G["single_turn_context"], G["parse_json_list"]

assert pj(None) == [""]
assert pj('["a", "b"]') == ["a", "b"]
assert pj('["a", null]') == ["a", ""]
assert pj("not json at all") == ["not json at all"]
assert pj("[]") == [""]
p3 = ["q1", "q2", "q3"]
r3 = ["a1", "a2", "a3"]
full = gm(p3, r3, 400)
assert full.count("[TURN]") == 3 and full.index("q1") < full.index("q3"), "chronological order broken"
tiny = gm(p3, r3, 30)
assert "q3" in tiny and "q1" not in tiny, "greedy must keep the most recent turn"
assert len(gm(p3, r3, 12)) <= 12, "budget violated"
assert gm(p3, r3, 0) == "", "zero budget must not crash"
assert gm([], [], 100) == "" and gm(["x"], [], 100) == "", "empty conversation must not crash"
assert gm(["a", "b"], ["x"], 400).count("[TURN]") == 1, "must zip to the shorter side"
assert "q1" in st(p3, r3, 400) and "q2" not in st(p3, r3, 400), "single-turn control leaked later turns"
print("   Algorithm 1 OK (order, recency, budget, ragged/empty inputs)")

print("=" * 70)
print("3. Splits and leakage check")
if REAL_DATA_PATH:
    print(f"   Loading real dataset from: {REAL_DATA_PATH}")
    exec(cell_with("raw = pd.read_csv"), G)
else:
    print("   [NOTE] train.csv not found locally -- synthesizing 57,477-row LMSYS benchmark...")
    _orig_read_csv = G["pd"].read_csv
    def _mock_read_csv(filepath, **kwargs):
        rng = np.random.default_rng(42)
        n = 57477
        labels = rng.choice([0, 1, 2], size=n, p=[0.347, 0.340, 0.313])
        models = [f"model_{i}" for i in range(25)]
        return pd.DataFrame({
            "id": np.arange(n),
            "prompt": ['["What is quantum computing?", "Explain superposition."]'] * n,
            "response_a": ['["Quantum computing uses qubits...", "Superposition allows..."]'] * n,
            "response_b": ['["It uses quantum mechanics...", "Superposition means states..."]'] * n,
            "winner_model_a": (labels == 0).astype(int),
            "winner_model_b": (labels == 1).astype(int),
            "winner_tie": (labels == 2).astype(int),
            "model_a": rng.choice(models, n),
            "model_b": rng.choice(models, n),
        })
    G["pd"].read_csv = _mock_read_csv
    exec(cell_with("raw = pd.read_csv"), G)
    G["pd"].read_csv = _orig_read_csv

raw, tp = G["raw"], G["train_pool"]
assert raw.id.is_unique, "ids are not unique -- the partition assertion would be meaningless"
print(f"   rows={len(raw):,}  pool={len(tp):,}  MAX_TRAIN_N={G['MAX_TRAIN_N']:,}")
print(f"   unseen-pair eval rows={G['UNSEEN_PAIR_MASK'].sum():,}")
for name in ("eval_df", "calib_df", "dev_df"):
    d = G[name]
    bal = d.label.value_counts(normalize=True).sort_index().values
    assert abs(bal - raw.label.value_counts(normalize=True).sort_index().values).max() < 0.01, \
        f"{name} is not stratified"
print("   stratification holds in every split; id-disjointness asserted inside the cell")

# determinism: rebuilding must reproduce the identical evaluation split
from sklearn.model_selection import train_test_split
_r2, e2 = train_test_split(raw, test_size=CFG.EVAL_N, stratify=raw.label,
                           random_state=CFG.SPLIT_SEED)
assert set(e2.id) == set(G["eval_df"].id), "split is not reproducible from SPLIT_SEED"
print("   split is deterministic across processes")

print("=" * 70)
print("4. Augmentation arms")
exec(cell_with("def build_train_arrays"), G)
bta = G["build_train_arrays"]
tok = np.random.randint(1, 100, size=(50, 2, 8))
msk = np.ones_like(tok)
y = np.random.randint(0, 3, 50)
for arm, mult in (("none", 1), ("dup", 2), ("swap", 2)):
    a, b, c = bta(tok, msk, y, arm, None)
    assert len(a) == 50 * mult == len(c), arm
n_a, n_b, n_t = np.bincount(y, minlength=3)
_, _, ys = bta(tok, msk, y, "swap", None)
sa, sb, stt = np.bincount(ys, minlength=3)
assert (sa, sb, stt) == (n_a + n_b, n_a + n_b, 2 * n_t), "swap must balance the A/B classes"
_, _, yd = bta(tok, msk, y, "dup", None)
assert np.bincount(yd, minlength=3).tolist() == [2 * n_a, 2 * n_b, 2 * n_t], \
    "dup must preserve the class prior"
ts, ms, _ = bta(tok, msk, y, "swap", None)
assert (ts[50:, 0] == tok[:, 1]).all() and (ts[50:, 1] == tok[:, 0]).all()
print(f"   none/dup/swap OK. swap balances A vs B ({n_a},{n_b}) -> ({sa},{sb}); "
      f"dup preserves the prior")

print("=" * 70)
print("5. Calibration / metrics / bootstrap maths")
sec = cell_with("def load_all_runs") + "\n" + cell_with("def apply_T") + "\n" + \
      cell_with("def paired_bootstrap")
G2 = dict(G)
exec("from sklearn.metrics import log_loss as sk_log_loss, f1_score\n"
     "from scipy import stats\nfrom scipy.optimize import minimize\nEPS = 1e-7", G2)
exec(extract_defs(sec, ["norm", "metrics", "apply_T", "fit_temperature",
                        "expected_calibration_error", "paired_bootstrap"]), G2)

rng = np.random.default_rng(0)
n = 4000
y_true = rng.integers(0, 3, n)
logits = rng.normal(0, 1, (n, 3))
logits[np.arange(n), y_true] += 1.1
over = np.exp(logits * 2.2)
p_over = over / over.sum(1, keepdims=True)          # deliberately over-confident

T = G2["fit_temperature"](p_over, y_true, per_class=False)
Tc = G2["fit_temperature"](p_over, y_true, per_class=True)
ll_raw = G2["metrics"](p_over, y_true)["log_loss"]
ll_cal = G2["metrics"](G2["apply_T"](p_over, T), y_true)["log_loss"]
ll_pc = G2["metrics"](G2["apply_T"](p_over, Tc), y_true)["log_loss"]
assert T[0] > 1.3, f"should detect over-confidence, got T={T}"
assert ll_cal < ll_raw and ll_pc <= ll_cal + 1e-4
ece_raw = G2["expected_calibration_error"](p_over, y_true)
ece_cal = G2["expected_calibration_error"](G2["apply_T"](p_over, T), y_true)
assert ece_cal < ece_raw
print(f"   temperature fit: T={T[0]:.3f}  per-class={np.round(Tc,3).tolist()}")
print(f"   log loss {ll_raw:.4f} -> {ll_cal:.4f} (global) / {ll_pc:.4f} (per-class)")
print(f"   ECE      {ece_raw:.4f} -> {ece_cal:.4f}")
assert abs(G2["apply_T"](p_over, [1.0]) - G2["norm"](p_over)).max() < 1e-9, "T=1 must be identity"

# bootstrap: a real difference is detected, an identical pair is not.
# `p_worse` mixes in each row's neighbour's prediction, which destroys signal rather
# than merely softening confidence.
p_norm = G2["norm"](p_over)
p_worse = G2["norm"](0.5 * p_norm + 0.5 * np.roll(p_norm, 1, axis=0))
bs = G2["paired_bootstrap"](p_norm, p_worse, y_true, n_boot=2000)
assert bs["log_loss"]["significant"] and bs["log_loss"]["delta"] > 0, bs["log_loss"]
assert bs["accuracy"]["delta"] < 0, "the degraded model must also lose accuracy"
same = G2["paired_bootstrap"](G2["norm"](p_over), G2["norm"](p_over), y_true, n_boot=2000)
assert not same["log_loss"]["significant"] and abs(same["log_loss"]["delta"]) < 1e-12
assert same["mcnemar"]["p"] == 1.0 and same["mcnemar"]["b"] == same["mcnemar"]["c"] == 0
print(f"   bootstrap detects a real gap (d={bs['log_loss']['delta']:+.4f}, "
      f"CI [{bs['log_loss']['ci_low']:+.4f},{bs['log_loss']['ci_high']:+.4f}], "
      f"McNemar p={bs['mcnemar']['p']:.2e}) and rejects a null gap")

print("=" * 70)
print("6. Mirroring / swap symmetry conventions")
exec(extract_defs(cell_with("def mirror_labels"), ["mirror_labels"]), G2)
assert G2["mirror_labels"](np.array([0, 1, 2])).tolist() == [1, 0, 2]
p = G2["norm"](rng.random((5, 3)))
assert np.allclose(p[:, [1, 0, 2]][:, [1, 0, 2]], p), "mirroring must be an involution"
print("   label mirroring and probability mirroring are consistent involutions")

print("=" * 70)
print("7. Warmup-cosine schedule shape (numpy re-implementation of the Keras class)")
lr_max, total = CFG.LR_MAX, 1000
w = max(1, int(CFG.WARMUP_FRAC * total))
d = max(1, total - w)


def lr_at(s):
    if s < w:
        return lr_max * (s + 1) / w
    prog = min(max((s - w) / d, 0.0), 1.0)
    return 1e-7 + 0.5 * (lr_max - 1e-7) * (1 + np.cos(np.pi * prog))


vals = [lr_at(s) for s in range(total)]
assert vals[0] < vals[w - 1] and abs(vals[w - 1] - lr_max) < 1e-12, "warmup must reach lr_max"
assert vals[-1] < lr_max * 0.01, "cosine must decay to ~0"
assert all(vals[i] >= vals[i + 1] - 1e-15 for i in range(w, total - 1)), "decay must be monotone"
print(f"   peak {max(vals):.2e} at step {int(np.argmax(vals))} (warmup={w}), "
      f"final {vals[-1]:.2e}")

print("=" * 70)
print("8. Manifest / cost model")
G3 = dict(G)
G3["find_existing"] = lambda rid: None
# PRESET_COST/PRESET_FAMILY/ARCH_PRESETS live in the tokenization cell, which needs Keras.
# Take just the module-level constants.
_tokcell = cell_with("PRESET_FAMILY = {")
exec(_tokcell[:_tokcell.index("def resolve_classes")], G3)
exec(re.search(r"^ARCH_PRESETS = \[.*?\]$", _tokcell, re.M | re.S).group(0), G3)
exec(cell_with("def run_id_of"), G3)
exec(cell_with("BLOCKS = {"), G3)
runs = G3["RUNS"]
ids = [G3["run_id_of"](c) for c in runs]
assert len(ids) == len(set(ids)), "duplicate run ids -- runs would overwrite each other"
assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+", i) for i in ids), "run id is not filename-safe"
import collections
print(f"   {len(runs)} runs, unique ids OK")
print("   per block:", dict(collections.Counter(c["block"] for c in runs)))
est = G3["estimate_hours"](runs)
for k, v in est.items():
    print(f"     {k}: {v:.2f} h")
print(f"   TOTAL estimate: {sum(est.values()):.2f} h "
      f"(at the default {G3['THROUGHPUT_GUESS']:.0f} samples/s)")

print("\n" + "=" * 70)
print("ALL LOCAL TESTS PASSED")
print("=" * 70)
