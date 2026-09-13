# %% [markdown]
# # LMSYS Preference Classification — Revision Experiments (v2)
#
# **Paper:** *Efficient LLM Preference Classification Through Position Bias Mitigation and Architectural Symmetry* (APIN-D-26-05359)
#
# This notebook replaces the original single-run starter notebook. It is an **experiment runner**:
# it executes a manifest of training configurations under *matched conditions*, over multiple seeds,
# stores every run's predictions, and then recomputes all paper tables from those stored predictions.
#
# ### What this fixes
#
# | # | Reviewer comment | Fix in this notebook |
# |---|---|---|
# | — | *(not raised by reviewers, but fatal)* the original notebook applied swap augmentation **before** the train/val split, so swapped twins of training rows leaked into validation | §3 splits **first** on original rows, augments **train only**, and asserts zero `id` overlap between splits |
# | R1-2, R2-1 | baseline trained 1 epoch / 57K vs. proposed 5 epochs / 115K | §8 Block A: full 2×2 factorial (augment × Siamese) **at identical epochs**, plus a duplicate-augmentation arm that matches data volume without swapping |
# | R2-7 | ablation confounds data volume with position-bias mitigation | the `dup` arm doubles data with *no* positional swap — the swap-vs-dup contrast isolates bias mitigation |
# | R1-5, R2-3 | no statistical analysis | every config runs 3 seeds; §10 reports mean ± std, paired bootstrap 95% CIs, McNemar tests |
# | R1 (synergy Q) | combined effect exceeds sum of parts, unexplained | the 2×2 factorial yields an explicit **interaction term** with a CI |
# | R2-6 | Table 4 training conditions unspecified | Block C trains every architecture with *identical* data, epochs, LR, seeds |
# | R2-8 | position-bias analysis only reports prediction rates | §10 reports accuracy **conditioned on the true winner's position**, plus flip-consistency under input swap |
# | R2-9 | temperature fitted and evaluated on the same set | a disjoint `calib` split fits the temperature; the `eval` split is never used for fitting. Global vs. per-class scaling compared (answers R1-Q4) |
# | R1 (multi-turn) | greedy context strategy too vague | §2 implements it as `greedy_multi_turn_context()` and prints reproducible pseudocode + the exact separator token IDs (answers R1-Q2) |
# | R1-4, R2-2 | single dataset | §9 evaluates zero-shot transfer to **MT-Bench Human Judgments**, plus an unseen-model-pair split of LMSYS |
# | R2-4 | software stack unspecified | §1 records every version, CUDA/driver, and device into the results file |
#
# ### How to run it
#
# 1. Settings → **Accelerator: GPU**, **Internet: On**. Add the competition dataset
#    (*+ Add Input → Competitions → LLM Classification Finetuning*; join the competition first).
# 2. **Always launch with *Save Version → Save & Run All (Commit)*, not the interactive editor.**
#    An interactive session can be reset and will take `/kaggle/working` with it, destroying every
#    completed run. A commit run writes a permanent output, which is what makes resuming possible.
# 3. First commit: leave `CFG.SMOKE_TEST = True`. Tiny models on a few hundred rows, ~30-45 min.
#    The results are meaningless by design — you are only checking that nothing crashes.
# 4. Then set `CFG.SMOKE_TEST = False`, choose blocks in §8, and commit again.
# 5. If a session hits the 12-hour limit: add the **previous version's output** as an input dataset
#    and commit again. Finished runs are detected and skipped; nothing is recomputed.
#
# **Cells must run in order.** The analysis in §10 reads the predictions written by §9.1, so it
# cannot run before training has produced them — it will stop at `assert ALL` if you try.
#
# ### Scaling up later
#
# Only `CFG.TRAIN_N`, `CFG.EPOCHS`, `CFG.SEQ_LEN` and `CFG.SEEDS` control cost. Raising `TRAIN_N` to
# `None` uses the full training pool. Everything else — splits, eval set, analysis, tables — is unchanged,
# so results from a small run and a large run are directly comparable and can share a results directory.

# %% [markdown]
# ---
# # 1 | Environment, versions and reproducibility record
#
# Addresses **R2-4**. Everything printed here belongs in the paper's reproducibility paragraph.

# %%
import os
import sys
import json
import time
import math
import gc
import glob
import shutil
import platform
import subprocess
import warnings
import itertools
from datetime import datetime, timezone

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf

# tf.data is used only as an input pipeline. Keeping TensorFlow off the GPU stops it from
# competing with JAX for device memory -- the single most common OOM cause in this setup.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception as _e:
    print("Could not hide GPU from TensorFlow:", _e)

import keras

try:
    import keras_hub as kh

    KH_NAME, KH_VERSION = "keras_hub", kh.__version__
except ImportError:
    import keras_nlp as kh

    KH_NAME, KH_VERSION = "keras_nlp", kh.__version__


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def _pkg(name):
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return "n/a"


# "Batch" for Save & Run All commits, "Interactive" in the editor. The two have different
# permissions on Kaggle -- notably, only interactive sessions may download Kaggle Models.
KAGGLE_RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Unknown")

ENV_INFO = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "kaggle_run_type": KAGGLE_RUN_TYPE,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "keras": keras.__version__,
    "keras_backend": keras.backend.backend(),
    KH_NAME: KH_VERSION,
    "tensorflow": tf.__version__,
    "jax": _pkg("jax"),
    "jaxlib": _pkg("jaxlib"),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": _pkg("scikit-learn"),
    "scipy": _pkg("scipy"),
    "datasets": _pkg("datasets"),
    "nvidia_smi": _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "cuda_toolkit": _sh("nvcc --version | tail -n 1"),
}

try:
    import jax

    ENV_INFO["jax_devices"] = [str(d) for d in jax.devices()]
except Exception:
    ENV_INFO["jax_devices"] = []

print(json.dumps(ENV_INFO, indent=2))

keras.mixed_precision.set_global_policy("mixed_float16")
print("\nglobal dtype policy:", keras.mixed_precision.global_policy().name)

# %% [markdown]
# ---
# # 2 | Configuration
#
# `SMOKE_TEST = True` runs the entire pipeline on a few hundred rows in ~10 minutes. **Always do this first**
# on a fresh Kaggle image: it catches library/preset issues before you spend GPU quota.
#
# The **scale knobs** are the only settings that affect cost. The evaluation, calibration and dev splits are
# deliberately *fixed and large* — they cost inference time only, and a large eval set is what makes the
# statistical claims (R1-5, R2-3) tight.

# %%
class CFG:
    # ------------------------------------------------------------------ scale knobs
    # The smoke path has been validated end to end (all four backbones, both tokenizer variants,
    # every block boundary), so this now defaults to the real run. Set it back to True only if you
    # change the pipeline and want to re-verify cheaply.
    SMOKE_TEST = False
    TRAIN_N = 12000            # training rows drawn from the pool (None = use the whole pool)
    ARCH_TRAIN_N = 5000        # Block C only -- see the note in section 8
    EPOCHS = 3
    SEQ_LEN = 256
    BATCH_SIZE = 16
    LR_MAX = 1.5e-5            # peak LR of the warmup+cosine schedule (identical for every architecture)
    WARMUP_FRAC = 0.10
    SEEDS = [42, 1337, 2024]   # >= 3 independent runs per config (R1-5, R2-3)

    # ------------------------------------------------------------------ fixed protocol (do not scale)
    EVAL_N = 8000              # held-out evaluation split; all reported metrics come from here
    CALIB_N = 3000             # held-out calibration split; temperature is fitted ONLY here (R2-9)
    DEV_N = 2000               # per-epoch monitoring only; never used for reporting
    SPLIT_SEED = 20260101      # fixed forever, so every run shares identical splits

    # ------------------------------------------------------------------ model / task
    MAIN_PRESET = "deberta_v3_extra_small_en"
    LABEL_SMOOTHING = 0.02
    DROPOUT = 0.1
    HIDDEN = 512
    N_CLASSES = 3
    LABEL2NAME = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}

    # ------------------------------------------------------------------ multi-turn context
    TURN_SEP = " [TURN] "      # literal separator between conversation turns
    RESP_SEP = " [RESPONSE] "  # literal separator between a turn's prompt and that turn's response
    CHARS_PER_TOKEN = 3.8      # character-budget proxy used by the greedy context selector

    # ------------------------------------------------------------------ generalization (R1-4, R2-2)
    USE_MTBENCH = True         # requires Internet: On
    MTBENCH_MAX = 3500

    # ------------------------------------------------------------------ session budget
    # Kaggle kills a commit at 12 h (exit 137). A killed notebook may lose its output, which
    # would throw away the whole session's training. Stop launching new runs early enough that
    # the notebook finishes normally and Kaggle saves /kaggle/working.
    SESSION_BUDGET_H = 10.0

    # ------------------------------------------------------------------ hardware
    # Kaggle's "GPU T4 x2" gives two devices, but Keras uses only the first unless told otherwise.
    # Data-parallel training shards each batch across both, roughly halving wall-clock time.
    # Measure it with the throughput probe before trusting it, and keep BATCH_SIZE divisible by
    # the device count.
    USE_MULTI_GPU = False

    # ------------------------------------------------------------------ bookkeeping
    BOOTSTRAP_N = 10000


if CFG.SMOKE_TEST:
    CFG.TRAIN_N = 240
    CFG.ARCH_TRAIN_N = 240
    CFG.EPOCHS = 1
    # sequence length is deliberately NOT reduced: the smoke run's measured throughput is
    # what calibrates the cost estimate in section 8, so it must be realistic.
    CFG.EVAL_N = 400
    CFG.CALIB_N = 300
    CFG.DEV_N = 200
    CFG.SEEDS = [42]
    CFG.MTBENCH_MAX = 200
    print(">>> SMOKE TEST MODE -- tiny data, results are NOT for the paper <<<")
else:
    print(f">>> REAL RUN -- {CFG.TRAIN_N:,} training rows, {CFG.EPOCHS} epochs, "
          f"seeds {CFG.SEEDS} <<<")

# ---------------------------------------------------------------------- multi-GPU
# Must be set before any model is built, so it lives here rather than next to build_model().
N_DEVICES = 1
if CFG.USE_MULTI_GPU:
    try:
        import jax

        _devs = jax.devices()
        if len(_devs) > 1 and CFG.BATCH_SIZE % len(_devs) == 0:
            keras.distribution.set_distribution(keras.distribution.DataParallel(devices=_devs))
            N_DEVICES = len(_devs)
            print(f"data-parallel training across {N_DEVICES} devices: {_devs}")
            print(f"  global batch {CFG.BATCH_SIZE} -> {CFG.BATCH_SIZE // N_DEVICES} per device")
        else:
            print(f"multi-GPU not applied: {len(_devs)} device(s), "
                  f"batch {CFG.BATCH_SIZE} must be divisible by the device count")
    except Exception as e:
        print("multi-GPU setup failed, continuing on a single device:", repr(e)[:200])

# ---------------------------------------------------------------------- paths
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS_DIR = os.path.join(WORK, "results")
PREDS_DIR = os.path.join(WORK, "predictions")
CACHE_DIR = os.path.join(WORK, "tokcache")
FAILED_DIR = os.path.join(WORK, "failed")
for _d in (RESULTS_DIR, PREDS_DIR, CACHE_DIR, FAILED_DIR):
    os.makedirs(_d, exist_ok=True)

# Results carried over from an earlier Kaggle session (add that version's output as an input).
def find_input_dirs(name, root="/kaggle/input", max_depth=8):
    """Locate directories called `name` anywhere under /kaggle/input.

    Kaggle mounts inputs under category folders whose depth varies by type
    (/kaggle/input/competitions/..., /kaggle/input/models/<owner>/<model>/...), so a
    one-level glob silently misses attached notebook output -- which would silently
    retrain everything instead of resuming.
    """
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= max_depth:
            dirnames[:] = []
        if os.path.basename(dirpath) == name:
            out.append(dirpath)
    return sorted(out)


PRIOR_RESULT_DIRS = find_input_dirs("results")
PRIOR_PRED_DIRS = find_input_dirs("predictions")
PRIOR_TOKCACHE_DIRS = find_input_dirs("tokcache")

# Make this session's output CUMULATIVE by copying prior runs into /kaggle/working.
# A Kaggle version's output contains only what that session wrote; attached inputs are not
# carried through. Without this, resuming needs every previous version attached at once, and
# forgetting one silently retrains those runs.
_copied = 0
for _srcs, _dst in ((PRIOR_RESULT_DIRS, RESULTS_DIR), (PRIOR_PRED_DIRS, PREDS_DIR)):
    for _d in _srcs:
        if os.path.abspath(_d) == os.path.abspath(_dst):
            continue
        for _f in glob.glob(os.path.join(_d, "*")):
            _t = os.path.join(_dst, os.path.basename(_f))
            if os.path.isfile(_f) and not os.path.exists(_t):
                shutil.copy2(_f, _t)
                _copied += 1
if _copied:
    print(f"carried {_copied} files forward from earlier sessions into this session's output")

_prior_runs = len(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
if _prior_runs:
    print(f"Found {_prior_runs} completed runs (these will be REUSED, not retrained):")
    for d in PRIOR_RESULT_DIRS:
        print("  ", d, "->", len(glob.glob(os.path.join(d, "*.json"))), "runs")
else:
    print("No prior results found in /kaggle/input -- every run will be trained from scratch.")
    if os.path.isdir("/kaggle/input"):
        print("  Top-level inputs:", sorted(os.listdir("/kaggle/input")))
        print("  If you attached an earlier version's output expecting to resume, stop and")
        print("  check it here before spending GPU time.")

# ---------------------------------------------------------------------- competition data
def find_dataset():
    """Locate train.csv, and fail with an actionable message rather than a None further downstream."""
    candidates = ["/kaggle/input/llm-classification-finetuning", "./data", "../data"]
    candidates += [os.path.dirname(p) for p in glob.glob("/kaggle/input/*/train.csv")]
    candidates += [os.path.dirname(p) for p in glob.glob("/kaggle/input/*/*/train.csv")]
    # depth-independent fallback, for whatever layout Kaggle uses next
    if os.path.isdir("/kaggle/input"):
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if dirpath[len("/kaggle/input"):].count(os.sep) >= 6:
                dirnames[:] = []
            if "train.csv" in filenames:
                candidates.append(dirpath)
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")):
            return p

    attached = sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else []
    raise FileNotFoundError(
        "\n\n"
        "  train.csv was not found.\n\n"
        f"  Currently attached inputs: {attached if attached else 'NONE'}\n"
        f"  Looked in: {candidates}\n\n"
        "  Fix: in the right-hand sidebar click '+ Add Input' -> Competitions tab ->\n"
        "  search 'LLM Classification Finetuning' -> click '+'.\n"
        "  You must have joined the competition first (kaggle.com/competitions/\n"
        "  llm-classification-finetuning -> Join Competition -> accept the rules).\n\n"
        "  Then RE-RUN THIS CELL before continuing -- BASE_PATH is defined here.\n")


BASE_PATH = find_dataset()
print("\nDataset path:", BASE_PATH)

# %% [markdown]
# ---
# # 3 | Multi-turn context construction
#
# Addresses **R1's minor comment on §3.3** and **R1-Q2** ("what special token is used as a turn separator?").
#
# The original code joined the prompt turns but kept only `response[0]`, so the multi-turn ablation was
# measuring an inconsistent input. Here each conversation is rebuilt as an interleaved turn sequence and
# truncated by an explicit greedy rule that always preserves the most recent turn.
#
# ```text
# Algorithm 1  GreedyMultiTurnContext
# ------------------------------------------------------------------
# Input : prompts P = [p_1..p_T], responses R = [r_1..r_T],
#         character budget B  ( B = SEQ_LEN * CHARS_PER_TOKEN )
# Output: context string C
#
#  1  blocks <- []                       # kept turns, most recent first
#  2  budget <- B
#  3  for t = T down to 1:               # iterate newest -> oldest
#  4      b <- TURN_SEP + p_t + RESP_SEP + r_t
#  5      if |b| <= budget:
#  6          blocks.append(b);  budget <- budget - |b|
#  7      else if t = T:                 # final turn is never dropped
#  8          blocks.append(tail(b, budget));  budget <- 0;  break
#  9      else:
# 10          break                      # stop at the first turn that does not fit
# 11  C <- concat(reverse(blocks))       # restore chronological order
# 12  return C
# ------------------------------------------------------------------
# ```
#
# Truncation happens oldest-first, which is the opposite of the tokenizer's default right-truncation. The
# separators are **literal text markers, not vocabulary tokens** — the cell below prints the exact sub-word
# IDs they map to so the paper can state them precisely.

# %%
def clean_text(s):
    """Force a string to valid UTF-8.

    `json.loads` happily produces lone surrogates from escapes like \\ud83d, and the LMSYS dump
    contains them. TensorFlow encodes those to invalid UTF-8 byte sequences, and SentencePiece
    then aborts the whole batch with a RuntimeError. Dropping the offending code points costs
    nothing -- they carry no text -- and makes tokenization total.
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")


def parse_json_list(x):
    """Parse the competition's string-encoded list columns. Never raises."""
    if pd.isna(x):
        return [""]
    s = str(x)
    try:
        v = json.loads(s.replace("null", '""'))
    except Exception:
        try:
            v = json.loads(s.replace("null", '""').replace("\\/", "/"))
        except Exception:
            return [clean_text(s)]
    if not isinstance(v, list):
        return [clean_text(v)]
    return [clean_text(e) for e in v] or [""]


def greedy_multi_turn_context(prompts, responses, budget_chars,
                              turn_sep=CFG.TURN_SEP, resp_sep=CFG.RESP_SEP):
    """Algorithm 1. Returns the newest turns that fit in `budget_chars`, chronologically ordered."""
    T = min(len(prompts), len(responses))
    if T == 0:
        return ""
    blocks, budget = [], budget_chars
    for t in range(T - 1, -1, -1):
        b = f"{turn_sep}{prompts[t]}{resp_sep}{responses[t]}"
        if len(b) <= budget:
            blocks.append(b)
            budget -= len(b)
        elif t == T - 1:
            blocks.append(b[-budget:] if budget > 0 else b[:0])
            budget = 0
            break
        else:
            break
    return "".join(reversed(blocks))


def single_turn_context(prompts, responses, budget_chars,
                        turn_sep=CFG.TURN_SEP, resp_sep=CFG.RESP_SEP):
    """Ablation control: first turn only (what the original notebook effectively used)."""
    if not prompts or not responses:
        return ""
    b = f"{turn_sep}{prompts[0]}{resp_sep}{responses[0]}"
    return b[:budget_chars]


def build_side_texts(df, multiturn, seq_len):
    """Build the two model inputs (side A, side B) for every row."""
    budget = int(seq_len * CFG.CHARS_PER_TOKEN)
    fn = greedy_multi_turn_context if multiturn else single_turn_context
    a = [clean_text(fn(p, r, budget)) for p, r in zip(df["prompt_turns"], df["resp_a_turns"])]
    b = [clean_text(fn(p, r, budget)) for p, r in zip(df["prompt_turns"], df["resp_b_turns"])]
    return a, b


print("Algorithm 1 self-check")
_demo_p = ["turn one question", "turn two question", "turn three question"]
_demo_r = ["answer one", "answer two", "answer three"]
print("  full   :", greedy_multi_turn_context(_demo_p, _demo_r, 400))
print("  budget50:", greedy_multi_turn_context(_demo_p, _demo_r, 50))
print("  single :", single_turn_context(_demo_p, _demo_r, 400))

# %% [markdown]
# ---
# # 4 | Data loading and **leakage-free** splits
#
# This is the most important correction in the notebook.
#
# **Original ordering:** `augment(df)` → `train_test_split(df)`. A swapped copy of a training row could land
# in validation, so validation contained near-duplicates of training data. The original run's
# `metrics.json` shows 91,963 train / 22,991 val — an 80/20 split of the **114,954 augmented** rows,
# confirming the leak.
#
# **Corrected ordering:** split the *original* rows into disjoint `eval` / `calib` / `dev` / `train_pool`
# sets by `id`, then augment `train_pool` only. Assertions below prove the splits are disjoint.

# %%
t0 = time.time()
raw = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
print(f"loaded {len(raw):,} rows in {time.time()-t0:.1f}s")

raw["prompt_turns"] = raw["prompt"].map(parse_json_list)
raw["resp_a_turns"] = raw["response_a"].map(parse_json_list)
raw["resp_b_turns"] = raw["response_b"].map(parse_json_list)
raw["n_turns"] = [min(len(p), len(a), len(b))
                  for p, a, b in zip(raw.prompt_turns, raw.resp_a_turns, raw.resp_b_turns)]
raw = raw[raw.n_turns > 0].reset_index(drop=True)

raw["label"] = raw[["winner_model_a", "winner_model_b", "winner_tie"]].values.argmax(axis=1)
raw["pair_key"] = [
    "|".join(sorted([str(a), str(b)])) for a, b in zip(raw.model_a, raw.model_b)
]

print("class balance:", raw.label.value_counts(normalize=True).sort_index().round(4).to_dict())
print("multi-turn share:", float((raw.n_turns > 1).mean()).__round__(4))
print("distinct model pairings:", raw.pair_key.nunique())

# ------------------------------------------------------------------ stratified, id-disjoint splits
from sklearn.model_selection import train_test_split

_rest, eval_df = train_test_split(
    raw, test_size=CFG.EVAL_N, stratify=raw.label, random_state=CFG.SPLIT_SEED
)
_rest, calib_df = train_test_split(
    _rest, test_size=CFG.CALIB_N, stratify=_rest.label, random_state=CFG.SPLIT_SEED
)
train_pool, dev_df = train_test_split(
    _rest, test_size=CFG.DEV_N, stratify=_rest.label, random_state=CFG.SPLIT_SEED
)

train_pool = train_pool.sample(frac=1.0, random_state=CFG.SPLIT_SEED).reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
calib_df = calib_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)

# ------------------------------------------------------------------ leakage assertions
_sets = {"train_pool": set(train_pool.id), "dev": set(dev_df.id),
         "calib": set(calib_df.id), "eval": set(eval_df.id)}
for x, y in itertools.combinations(_sets, 2):
    overlap = _sets[x] & _sets[y]
    assert not overlap, f"LEAK: {x} and {y} share {len(overlap)} ids"
assert sum(len(v) for v in _sets.values()) == len(raw), "splits do not partition the data"
print("\nleakage check passed -- all four splits are id-disjoint and partition the dataset")

SPLIT_SIZES = {k: len(v) for k, v in _sets.items()}
print("split sizes:", SPLIT_SIZES)

# ------------------------------------------------------------------ how much of the pool do we train on
MAX_TRAIN_N = len(train_pool) if CFG.TRAIN_N is None else min(CFG.TRAIN_N, len(train_pool))
ARCH_TRAIN_N = min(CFG.ARCH_TRAIN_N, MAX_TRAIN_N)
print(f"training rows in use: {MAX_TRAIN_N:,} of {len(train_pool):,} available in the pool")
print(f"backbone comparison (Block C) uses the first {ARCH_TRAIN_N:,} of those rows")

# ------------------------------------------------------------------ unseen-model-pair generalization split
_train_pairs = set(train_pool.pair_key.iloc[:MAX_TRAIN_N])
UNSEEN_PAIR_MASK = ~eval_df.pair_key.isin(_train_pairs).values
print(f"eval rows whose model pairing never appears in training: "
      f"{UNSEEN_PAIR_MASK.sum():,} / {len(eval_df):,}")

Y_EVAL = eval_df.label.values.astype("int32")
Y_CALIB = calib_df.label.values.astype("int32")
Y_DEV = dev_df.label.values.astype("int32")

# %% [markdown]
# ---
# # 5 | Augmentation arms
#
# Three mutually exclusive training-set constructions, all applied **to the training split only**:
#
# | arm | rows | what it controls for |
# |---|---|---|
# | `none` | N | the honest baseline |
# | `dup` | 2N | **data volume matched to `swap`, with no positional change** — this is the control R2-7 asked for |
# | `swap` | 2N | swap augmentation: A/B exchanged and the label mirrored |
#
# Because `dup` and `swap` see exactly the same number of gradient updates on exactly the same underlying
# examples, any difference between them is attributable to position-bias mitigation alone, not to data volume.

# %%
def mirror_labels(y):
    """0 (A wins) <-> 1 (B wins); 2 (tie) is invariant."""
    m = y.copy()
    m[y == 0] = 1
    m[y == 1] = 0
    return m


def build_train_arrays(tok, mask, y, arm, rng):
    """tok/mask: (N,2,L). Returns augmented (M,2,L) arrays plus labels."""
    if arm == "none":
        return tok, mask, y
    if arm == "dup":
        return (np.concatenate([tok, tok]),
                np.concatenate([mask, mask]),
                np.concatenate([y, y]))
    if arm == "swap":
        tok_s = tok[:, ::-1, :]
        mask_s = mask[:, ::-1, :]
        return (np.concatenate([tok, tok_s]),
                np.concatenate([mask, mask_s]),
                np.concatenate([y, mirror_labels(y)]))
    raise ValueError(f"unknown augmentation arm: {arm}")


# self-check on toy data
_t = np.arange(2 * 2 * 3).reshape(2, 2, 3)
_m = np.ones_like(_t)
_y = np.array([0, 2])
_at, _am, _ay = build_train_arrays(_t, _m, _y, "swap", None)
assert (_at[2, 0] == _t[0, 1]).all() and (_at[2, 1] == _t[0, 0]).all(), "swap did not exchange sides"
assert _ay.tolist() == [0, 2, 1, 2], "swap did not mirror labels correctly"
print("swap augmentation self-check passed:", _ay.tolist())

# %% [markdown]
# ---
# # 6 | Tokenization (cached once per architecture)
#
# The original pipeline re-tokenized on every epoch of every run. Here each split is tokenized once per
# `(preset, sequence length, multi-turn flag)` and cached to disk as `int32` arrays, so 30+ training runs
# share the work. Swapped inputs need no re-tokenization at all — they are `arr[:, ::-1, :]`.

# %%
PRESET_FAMILY = {
    "deberta_v3_extra_small_en": "DebertaV3",
    "deberta_v3_small_en": "DebertaV3",
    "deberta_v3_base_en": "DebertaV3",
    "bert_base_en_uncased": "Bert",
    "roberta_base_en": "Roberta",
    "distil_bert_base_en_uncased": "DistilBert",
}

# rough relative training cost per sample vs. deberta_v3_extra_small_en (used only for time estimates)
PRESET_COST = {
    "deberta_v3_extra_small_en": 1.0,
    "deberta_v3_small_en": 2.2,
    "deberta_v3_base_en": 3.8,
    "bert_base_en_uncased": 3.6,
    "roberta_base_en": 3.6,
    "distil_bert_base_en_uncased": 1.9,
}


def resolve_classes(preset):
    fam = PRESET_FAMILY[preset]
    backbone_cls = getattr(kh.models, f"{fam}Backbone", None)
    assert backbone_cls is not None, f"{KH_NAME}.models has no {fam}Backbone"
    prep_cls = None
    for nm in (f"{fam}TextClassifierPreprocessor", f"{fam}Preprocessor"):
        prep_cls = getattr(kh.models, nm, None)
        if prep_cls is not None:
            break
    assert prep_cls is not None, f"{KH_NAME}.models has no preprocessor for {fam}"
    return backbone_cls, prep_cls


# Kaggle model slug that carries each preset, for the "+ Add Input -> Models" step
PRESET_KAGGLE_MODEL = {
    "deberta_v3_extra_small_en": "keras/deberta-v3",
    "deberta_v3_small_en": "keras/deberta-v3",
    "deberta_v3_base_en": "keras/deberta-v3",
    "bert_base_en_uncased": "keras/bert",
    "roberta_base_en": "keras/roberta",
    "distil_bert_base_en_uncased": "keras/distil-bert",
}

def scan_attached_presets(root="/kaggle/input", max_depth=8):
    """Index every attached Keras preset by walking /kaggle/input.

    Kaggle nests inputs under category folders whose exact layout varies
    (`/kaggle/input/competitions/...`, `/kaggle/input/models/<owner>/<model>/<fw>/<var>/<ver>/`,
    and older flat forms), so hard-coded glob patterns are unreliable. A preset directory is
    identified by containing `config.json`; it is indexed under every component of its path,
    which means it is found under its variation name wherever Kaggle chose to mount it.
    """
    index = {}
    if not os.path.isdir(root):
        return index
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= max_depth:
            dirnames[:] = []
        if "config.json" in filenames:
            for part in dirpath[len(root):].split(os.sep):
                if part:
                    index.setdefault(part, set()).add(dirpath)
    return {k: sorted(v) for k, v in index.items()}


ATTACHED_PRESETS = scan_attached_presets()
_PATH_CACHE = {}


def preset_location(preset):
    """Return a local directory for `preset` if one is attached, else the bare preset name.

    A committed (non-interactive) Kaggle run is forbidden from attaching new Kaggle Models, so
    `from_preset("deberta_v3_extra_small_en")` raises BackendError there even though it works
    fine in the interactive editor. `from_preset` also accepts a directory, so an attached copy
    is always preferred.
    """
    if preset in _PATH_CACHE:
        return _PATH_CACHE[preset]
    hits = ATTACHED_PRESETS.get(preset, [])
    if hits:
        # deepest path wins: .../<variation>/<version>/ rather than .../<variation>/
        found = sorted(hits, key=lambda p: (p.count(os.sep), p))[-1]
    else:
        found = preset
    _PATH_CACHE[preset] = found
    return found


def preset_report(presets):
    """Print where each preset will load from, and fail loudly if a committed run would break."""
    print(f"{'preset':<32}{'source':<12}location")
    missing = []
    for p in sorted(set(presets)):
        loc = preset_location(p)
        local = loc != p
        if not local:
            missing.append(p)
        print(f"  {p:<30}{'attached' if local else 'DOWNLOAD':<12}{loc}")
    if missing:
        # self-diagnosing: show what actually is mounted, so a wrong guess is visible in the log
        print("\n--- what is actually under /kaggle/input ---")
        if os.path.isdir("/kaggle/input"):
            shown = 0
            for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
                d = dirpath[len("/kaggle/input"):].count(os.sep)
                if d >= 5:
                    dirnames[:] = []
                if d <= 4 and shown < 60:
                    marker = "  <-- preset dir" if "config.json" in filenames else ""
                    print(f"  {dirpath}{marker}")
                    shown += 1
        else:
            print("  /kaggle/input does not exist")
        print("--- preset directories found:",
              sorted(k for k, v in ATTACHED_PRESETS.items()
                     if any(k in p.split(os.sep) for p in v))[:40])
    if missing:
        print("\n" + "!" * 78)
        print("These presets are NOT attached and would be downloaded at run time:")
        for p in missing:
            print(f"  {p:<30} -> + Add Input -> Models -> {PRESET_KAGGLE_MODEL.get(p, '?')}")
        print("\nThat works in the interactive editor but FAILS in a committed run with:")
        print('  BackendError: New Models cannot be attached in non-interactive sessions')
        print("Attach them before using Save & Run All (Commit).")
        print("!" * 78)
    return missing


_PREP_CACHE = {}


def get_preprocessor(preset, seq_len):
    key = (preset, seq_len)
    if key not in _PREP_CACHE:
        _, prep_cls = resolve_classes(preset)
        _PREP_CACHE[key] = prep_cls.from_preset(preset_location(preset), sequence_length=seq_len)
    return _PREP_CACHE[key]


TOKENIZE_FAILURES = []


def _run_preprocessor(prep, batch):
    out = prep(tf.constant(batch))
    if isinstance(out, tuple):
        out = out[0]
    return (np.asarray(out["token_ids"], dtype="int32"),
            np.asarray(out["padding_mask"]).astype("int32"))


def tokenize_sides(texts_a, texts_b, preset, seq_len, chunk=1024):
    """-> token_ids (N,2,L) int32, padding_mask (N,2,L) int32

    A single malformed string makes SentencePiece abort the whole batch. Rather than lose the
    run, a failing chunk is retried one item at a time and the offending strings are replaced
    with empty text (and recorded in TOKENIZE_FAILURES so the count can be reported).
    """
    prep = get_preprocessor(preset, seq_len)
    flat = [t for pair in zip(texts_a, texts_b) for t in pair]
    ids, masks = [], []
    for i in range(0, len(flat), chunk):
        batch = flat[i:i + chunk]
        try:
            bi, bm = _run_preprocessor(prep, batch)
        except Exception as e:
            # A stale tokenizer resource (e.g. after clear_session) fails on everything, so
            # rebuild it once and retry the whole chunk before blaming the data.
            print(f"  tokenizer failed on chunk at offset {i} ({type(e).__name__}); "
                  f"rebuilding the preprocessor and retrying")
            _PREP_CACHE.pop((preset, seq_len), None)
            prep = get_preprocessor(preset, seq_len)
            try:
                bi, bm = _run_preprocessor(prep, batch)
            except Exception:
                print("  still failing -- isolating the offending items")
                rows_i, rows_m, n_bad = [], [], 0
                for j, t in enumerate(batch):
                    try:
                        ti, tm = _run_preprocessor(prep, [t])
                    except Exception:
                        TOKENIZE_FAILURES.append((preset, i + j, t[:120]))
                        n_bad += 1
                        ti, tm = _run_preprocessor(prep, [""])
                    rows_i.append(ti)
                    rows_m.append(tm)
                # a systemic failure must not be silently converted into empty training text
                if n_bad > 0.25 * len(batch):
                    raise RuntimeError(
                        f"tokenizer rejected {n_bad}/{len(batch)} items -- this is an environment "
                        f"problem, not bad data. Refusing to continue with blanked-out text.")
                bi, bm = np.concatenate(rows_i), np.concatenate(rows_m)
        ids.append(bi)
        masks.append(bm)
    n = len(texts_a)
    return (np.concatenate(ids).reshape(n, 2, seq_len),
            np.concatenate(masks).reshape(n, 2, seq_len))


def cached_tokens(name, df, preset, seq_len, multiturn):
    """Tokenize a split, caching to /kaggle/working/tokcache."""
    # v2 = text is UTF-8 sanitised; bump this whenever the text pipeline changes so that
    # caches written by an older version are never silently reused
    tag = f"{name}__{preset}__L{seq_len}__{'mt' if multiturn else 'st'}__n{len(df)}__v2"
    path = os.path.join(CACHE_DIR, tag + ".npz")
    for prior in PRIOR_TOKCACHE_DIRS:
        p = os.path.join(prior, tag + ".npz")
        if os.path.exists(p) and not os.path.exists(path):
            path = p
            break
    if os.path.exists(path):
        z = np.load(path)
        return z["ids"], z["mask"]
    ta, tb = build_side_texts(df, multiturn, seq_len)
    ids, mask = tokenize_sides(ta, tb, preset, seq_len)
    np.savez_compressed(os.path.join(CACHE_DIR, tag + ".npz"), ids=ids, mask=mask)
    return ids, mask


# Every backbone this notebook can train. Defined here (not in section 8) so the preset
# availability check below can run before anything expensive happens.
ARCH_PRESETS = [CFG.MAIN_PRESET, "bert_base_en_uncased", "roberta_base_en", "deberta_v3_small_en"]

# ------------------------------------------------------------------ preflight: are presets available?
_missing_presets = preset_report(ARCH_PRESETS)
if _missing_presets and KAGGLE_RUN_TYPE == "Batch":
    raise RuntimeError(
        "\n\n  This is a committed (Save & Run All) session, which is not allowed to download\n"
        "  Kaggle Models. The presets listed above must be attached as inputs first:\n\n"
        + "".join(f"    + Add Input -> Models -> search '{PRESET_KAGGLE_MODEL.get(p, p)}'\n"
                 f"      -> framework Keras -> variation '{p}' -> Add\n"
                 for p in _missing_presets)
        + "\n  Failing now rather than hours into training.\n")

# ------------------------------------------------------------------ report the separator token IDs (R1-Q2)
_tk = get_preprocessor(CFG.MAIN_PRESET, CFG.SEQ_LEN)
_probe = _tk(tf.constant([f"x{CFG.TURN_SEP}y{CFG.RESP_SEP}z"]))
if isinstance(_probe, tuple):
    _probe = _probe[0]
_ids = np.asarray(_probe["token_ids"])[0]
_ids = _ids[_ids != 0][:24]
try:
    _tok_strings = [_tk.tokenizer.id_to_token(int(i)) for i in _ids]
except Exception:
    _tok_strings = ["<id_to_token unavailable>"]
SEPARATOR_TOKENS = {
    "turn_sep_literal": CFG.TURN_SEP,
    "resp_sep_literal": CFG.RESP_SEP,
    "probe_string": f"x{CFG.TURN_SEP}y{CFG.RESP_SEP}z",
    "probe_token_ids": [int(i) for i in _ids],
    "probe_token_strings": _tok_strings,
    "note": "separators are literal text markers segmented by the SentencePiece vocabulary, "
            "not reserved special tokens",
}
print(json.dumps(SEPARATOR_TOKENS, indent=2)[:1200])

# %% [markdown]
# ---
# # 7 | Model, schedule, and the single training function
#
# One `build_model` serves every architecture and every ablation. Two changes from the original beyond the
# `use_siamese` flag, both of which are corrections rather than new contributions:
#
# * **Masked** mean pooling. The original averaged over padding positions, which at 256 tokens dilutes short
#   responses toward zero. Pooling now divides by the true token count.
# * The classifier head outputs **float32** under the mixed-precision policy, so log-loss is not computed
#   from float16 probabilities.

# %%
class MaskedMeanPool(keras.layers.Layer):
    """Mean over non-padding positions, computed in float32."""

    def call(self, inputs):
        seq, mask = inputs
        seq = keras.ops.cast(seq, "float32")
        m = keras.ops.expand_dims(keras.ops.cast(mask, "float32"), -1)
        total = keras.ops.sum(seq * m, axis=1)
        count = keras.ops.maximum(keras.ops.sum(m, axis=1), 1.0)
        return total / count

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to lr_max, then cosine decay to lr_min. Defined in steps, not epochs."""

    def __init__(self, lr_max, total_steps, warmup_frac=0.1, lr_min=1e-7):
        self.lr_max = float(lr_max)
        self.total_steps = int(max(1, total_steps))
        self.warmup_frac = float(warmup_frac)
        self.warmup_steps = int(max(1, warmup_frac * self.total_steps))
        self.decay_steps = int(max(1, self.total_steps - self.warmup_steps))
        self.lr_min = float(lr_min)

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        w = float(self.warmup_steps)
        warm = self.lr_max * (step + 1.0) / w
        prog = keras.ops.clip((step - w) / float(self.decay_steps), 0.0, 1.0)
        cos = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + keras.ops.cos(math.pi * prog))
        return keras.ops.where(step < w, warm, cos)

    def get_config(self):
        return {"lr_max": self.lr_max, "total_steps": self.total_steps,
                "warmup_frac": self.warmup_frac, "lr_min": self.lr_min}


def build_model(preset, use_siamese, seq_len, lr_max, total_steps):
    backbone_cls, _ = resolve_classes(preset)
    backbone = backbone_cls.from_preset(preset_location(preset))

    tok = keras.Input(shape=(2, seq_len), dtype="int32", name="token_ids")
    pad = keras.Input(shape=(2, seq_len), dtype="int32", name="padding_mask")

    try:
        bb_keys = set(backbone.input.keys())
    except AttributeError:
        bb_keys = {"token_ids", "padding_mask"}

    pooled = []
    for i in range(2):
        t_i = keras.layers.Lambda(lambda z, i=i: z[:, i, :], name=f"slice_tok_{i}")(tok)
        p_i = keras.layers.Lambda(lambda z, i=i: z[:, i, :], name=f"slice_pad_{i}")(pad)
        args = {"token_ids": t_i, "padding_mask": p_i}
        if "segment_ids" in bb_keys:
            args["segment_ids"] = keras.layers.Lambda(
                lambda z: keras.ops.zeros_like(z), name=f"seg_{i}")(t_i)
        out = backbone(args)
        if isinstance(out, dict):
            out = out.get("sequence_output", list(out.values())[0])
        pooled.append(MaskedMeanPool(name=f"pool_{i}")([out, p_i]))

    e_a, e_b = pooled
    if use_siamese:
        # V = [e_A ; e_B ; |e_A - e_B|]
        absdiff = keras.layers.Lambda(
            lambda z: keras.ops.abs(z[0] - z[1]), name="abs_diff")([e_a, e_b])
        feats = keras.layers.Concatenate(name="fusion")([e_a, e_b, absdiff])
    else:
        # V = [e_A ; e_B]
        feats = keras.layers.Concatenate(name="fusion")([e_a, e_b])

    x = keras.layers.Dense(CFG.HIDDEN, activation="relu", name="fc")(feats)
    x = keras.layers.Dropout(CFG.DROPOUT, name="drop")(x)
    out = keras.layers.Dense(CFG.N_CLASSES, activation="softmax",
                             dtype="float32", name="classifier")(x)

    model = keras.Model({"token_ids": tok, "padding_mask": pad}, out)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=WarmupCosine(lr_max, total_steps, CFG.WARMUP_FRAC)),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=CFG.LABEL_SMOOTHING),
        metrics=[keras.metrics.CategoricalCrossentropy(name="log_loss"),
                 keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model, backbone


def make_ds(ids, mask, y=None, batch_size=None, shuffle=False, seed=0, drop_remainder=False):
    bs = batch_size or CFG.BATCH_SIZE
    x = {"token_ids": ids, "padding_mask": mask}
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        yc = keras.utils.to_categorical(y, num_classes=CFG.N_CLASSES).astype("float32")
        ds = tf.data.Dataset.from_tensor_slices((x, yc))
    if shuffle:
        ds = ds.shuffle(min(len(ids), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(bs, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)


def free_memory():
    """Release device memory between runs. 30+ model builds in one session will OOM without this.

    Clearing the Keras session also invalidates the TensorFlow resource handle held inside a
    SentencePiece tokenizer, so any cached preprocessor must be dropped at the same time --
    otherwise the next tokenize call fails with an opaque RuntimeError.
    """
    _PREP_CACHE.clear()
    for fn in (getattr(keras.backend, "clear_session", None),
               getattr(keras.utils, "clear_session", None)):
        if fn is not None:
            try:
                fn()
                break
            except Exception:
                pass
    gc.collect()
    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass

# %% [markdown]
# ## 7.1 | `run_experiment` — trains one configuration and stores its predictions
#
# Every run writes two files and nothing else:
#
# * `results/<run_id>.json` — config, timings, parameter counts, per-epoch history, throughput
# * `predictions/<run_id>.npz` — probabilities on `eval`, `calib` and MT-Bench, in **both input orders**
#
# All reported metrics are recomputed from the `.npz` files in §10. That means tables, calibration,
# significance tests and position-bias analysis can be revised **without retraining anything**.

# %%
def run_id_of(cfg):
    # smoke runs are prefixed so they can never be mistaken for, or mixed into, real results
    prefix = "SMOKE__" if CFG.SMOKE_TEST else ""
    return (f"{prefix}{cfg['block']}__{cfg['name']}__{cfg['preset']}"
            f"__n{cfg['train_n']}__e{cfg['epochs']}__s{cfg['seed']}")


def find_existing(run_id):
    for d in [RESULTS_DIR] + PRIOR_RESULT_DIRS:
        p = os.path.join(d, run_id + ".json")
        if os.path.exists(p):
            return p
    return None


def find_preds(run_id):
    for d in [PREDS_DIR] + PRIOR_PRED_DIRS:
        p = os.path.join(d, run_id + ".npz")
        if os.path.exists(p):
            return p
    return None


def predict_both_orders(model, ids, mask, batch_size=None):
    """Returns (p_orig, p_swap_mirrored). p_swap_mirrored is the swapped-input prediction
    mapped back into the original label space, so it is directly comparable to p_orig."""
    bs = batch_size or (CFG.BATCH_SIZE * 2)
    p_orig = model.predict(make_ds(ids, mask, batch_size=bs), verbose=0)
    p_swap = model.predict(make_ds(ids[:, ::-1, :], mask[:, ::-1, :], batch_size=bs), verbose=0)
    p_swap_mirrored = p_swap[:, [1, 0, 2]]
    return np.asarray(p_orig, dtype="float32"), np.asarray(p_swap_mirrored, dtype="float32")


def run_experiment(cfg, tokens):
    """cfg keys: block, name, preset, arm, siamese, multiturn, train_n, epochs, seed"""
    rid = run_id_of(cfg)
    t_start = time.time()
    keras.utils.set_random_seed(cfg["seed"])

    n = cfg["train_n"]
    tr_ids, tr_mask = tokens["train"][0][:n], tokens["train"][1][:n]
    tr_y = tokens["train_y"][:n]
    a_ids, a_mask, a_y = build_train_arrays(tr_ids, tr_mask, tr_y, cfg["arm"], None)

    steps_per_epoch = max(1, len(a_ids) // CFG.BATCH_SIZE)
    total_steps = steps_per_epoch * cfg["epochs"]

    train_ds = make_ds(a_ids, a_mask, a_y, shuffle=True, seed=cfg["seed"], drop_remainder=True)
    dev_ds = make_ds(tokens["dev"][0], tokens["dev"][1], Y_DEV, batch_size=CFG.BATCH_SIZE * 2)

    model, backbone = build_model(cfg["preset"], cfg["siamese"], CFG.SEQ_LEN, CFG.LR_MAX, total_steps)

    t_fit = time.time()
    hist = model.fit(train_ds, epochs=cfg["epochs"], validation_data=dev_ds, verbose=2)
    fit_seconds = time.time() - t_fit

    # ---------------- predictions (both input orders) on every evaluation set
    payload = {}
    p, ps = predict_both_orders(model, tokens["eval"][0], tokens["eval"][1])
    payload["eval_orig"], payload["eval_swap"] = p, ps
    # calibration needs the natural order only -- fitting a temperature on swapped inputs would
    # calibrate a distribution we never report from
    payload["calib_orig"] = np.asarray(
        model.predict(make_ds(tokens["calib"][0], tokens["calib"][1],
                              batch_size=CFG.BATCH_SIZE * 2), verbose=0), dtype="float32")
    if "mtbench" in tokens:
        p, ps = predict_both_orders(model, tokens["mtbench"][0], tokens["mtbench"][1])
        payload["mtbench_orig"], payload["mtbench_swap"] = p, ps

    # ---------------- honest inference-throughput measurement (warmup excluded, median of repeats)
    probe = make_ds(tokens["eval"][0][:CFG.BATCH_SIZE * 8],
                    tokens["eval"][1][:CFG.BATCH_SIZE * 8], batch_size=CFG.BATCH_SIZE)
    model.predict(probe, verbose=0)  # warmup / compile
    laps = []
    for _ in range(5):
        t = time.time()
        model.predict(probe, verbose=0)
        laps.append((time.time() - t) / (CFG.BATCH_SIZE * 8))
    ms_per_sample = float(np.median(laps) * 1000)

    result = {
        "run_id": rid,
        "config": dict(cfg),
        "protocol": {
            "seq_len": CFG.SEQ_LEN, "batch_size": CFG.BATCH_SIZE, "lr_max": CFG.LR_MAX,
            "warmup_frac": CFG.WARMUP_FRAC, "label_smoothing": CFG.LABEL_SMOOTHING,
            "hidden": CFG.HIDDEN, "dropout": CFG.DROPOUT, "split_seed": CFG.SPLIT_SEED,
            "eval_n": len(Y_EVAL), "calib_n": len(Y_CALIB), "dev_n": len(Y_DEV),
        },
        "data": {
            "train_rows_original": int(n),
            "train_rows_after_augmentation": int(len(a_ids)),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
        },
        "params": {
            "total": int(model.count_params()),
            "backbone": int(backbone.count_params()),
            "head": int(model.count_params() - backbone.count_params()),
        },
        "timing": {
            "fit_seconds": fit_seconds,
            "total_seconds": time.time() - t_start,
            "inference_ms_per_sample": ms_per_sample,
            "train_samples_per_second": float(len(a_ids) * cfg["epochs"] / max(fit_seconds, 1e-6)),
        },
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        "env": ENV_INFO,
    }

    np.savez_compressed(os.path.join(PREDS_DIR, rid + ".npz"), **payload)
    with open(os.path.join(RESULTS_DIR, rid + ".json"), "w") as f:
        json.dump(result, f, indent=2)

    # drop this frame's references before clearing the session, or the device arrays survive it
    del model, backbone, train_ds, dev_ds, probe, a_ids, a_mask, a_y, payload
    free_memory()
    return result

# %% [markdown]
# ---
# # 8 | Experiment manifest
#
# Blocks are ordered by how directly they answer the review. **Leave A, B and D on.** Turn C and E on only
# if the GPU quota allows — the estimator below prints hours before anything runs.
#
# **Block A — Table 3 replacement.** A full 2×2 factorial of {no augmentation, swap augmentation} ×
# {no Siamese fusion, Siamese fusion}, plus the `dup` data-volume control. Every arm gets identical epochs,
# LR, schedule and seeds. This is what makes the improvement attributable (R1-2, R2-1) and it yields the
# **interaction term** that explains the synergy R1 asked about.
#
# **Block B — Table 7 ablation.** Adds the multi-turn arm and an all-off reference. Temperature calibration
# is post-hoc, so its contribution is measured in §10 without extra training.
#
# **Block C — Table 4 replacement.** Alternative architectures under conditions identical to Block A (R2-6).
#
# **Block D — data efficiency.** Nested training subsets, so the curve is monotone in data by construction.
#
# **Block E — scale-up.** One full-pool run of the final configuration, for a headline number closer to the
# original paper's scale. Expensive; enable last.

# %%
BLOCKS = {
    # A and B are COMPLETE (24 runs, results already collected). They are switched off so this
    # session trains only what is left, regardless of which earlier output happens to be attached.
    # Set both back to True if you ever need to reproduce them from scratch.
    "A_main": False,    # DONE -- 2x2 factorial + data-volume control  (R1-2, R2-1, R2-7)
    "B_abl": False,     # DONE -- multi-turn + all-off reference       (R1 ablation, R2-7)
    "C_arch": True,     # alternative architectures               (R2-6)
    "D_eff": True,      # data-efficiency curve
    "E_scale": False,   # full-pool run of the final config       (enable only with spare quota)
}

# Block C is by far the costliest (BERT/RoBERTa cost ~3.6x DeBERTa-v3-xsmall per sample).
# Two seeds keeps the whole manifest inside one weekly GPU quota; raise to CFG.SEEDS when you
# have the budget, and re-run -- the two completed seeds will be reused, not recomputed.
ARCH_SEEDS = CFG.SEEDS   # all three seeds: Reviewer 2 asked for a minimum of three runs
# ARCH_PRESETS is defined in section 6 so the preset availability check can run early.
EFF_FRACTIONS = [0.1, 0.3, 0.6]


def eff_n(fr):
    """Training rows for a data-efficiency point. Nested subsets: eff_10 is a prefix of eff_30."""
    return max(CFG.BATCH_SIZE * 4, int(MAX_TRAIN_N * fr))


def manifest():
    runs = []

    def add(block, name, seed, preset=CFG.MAIN_PRESET, arm="swap", siamese=True,
            multiturn=True, train_n=None, epochs=None):
        runs.append({
            "block": block, "name": name, "preset": preset, "arm": arm,
            "siamese": bool(siamese), "multiturn": bool(multiturn),
            "train_n": int(train_n or MAX_TRAIN_N), "epochs": int(epochs or CFG.EPOCHS),
            "seed": int(seed),
        })

    if BLOCKS["A_main"]:
        for s in CFG.SEEDS:
            add("A", "noaug_nosiam", s, arm="none", siamese=False)   # baseline
            add("A", "noaug_siam", s, arm="none", siamese=True)      # Siamese only
            add("A", "dupaug_nosiam", s, arm="dup", siamese=False)   # data volume only
            add("A", "swapaug_nosiam", s, arm="swap", siamese=False)  # swap only
            add("A", "swapaug_siam", s, arm="swap", siamese=True)    # full model

    if BLOCKS["B_abl"]:
        for s in CFG.SEEDS:
            add("B", "full_nomultiturn", s, arm="swap", siamese=True, multiturn=False)
            add("B", "all_off", s, arm="none", siamese=False, multiturn=False)

    if BLOCKS["C_arch"]:
        # Block C runs at ARCH_TRAIN_N, not MAX_TRAIN_N. Table 4 compares *backbones*, so it only has
        # to be internally matched -- and BERT/RoBERTa cost ~3.6x per sample, so running them at the
        # main scale would consume the entire quota. Our own backbone is included here at the same
        # reduced scale, which is what makes the table self-contained and fair.
        for s in ARCH_SEEDS:
            for preset in ARCH_PRESETS:
                add("C", f"arch_{preset}", s, preset=preset, arm="swap", siamese=True,
                    train_n=ARCH_TRAIN_N)

    if BLOCKS["D_eff"]:
        for s in CFG.SEEDS:
            for fr in EFF_FRACTIONS:
                add("D", f"eff_{int(fr*100)}pct", s, arm="swap", siamese=True, train_n=eff_n(fr))

    if BLOCKS["E_scale"]:
        for s in CFG.SEEDS[:1]:
            add("E", "fullpool", s, arm="swap", siamese=True, train_n=len(train_pool))

    return runs


RUNS = manifest()

# ---------------------------------------------------------------- cost estimate
THROUGHPUT_GUESS = 23.0  # samples/s, deberta_v3_extra_small_en, seq 256, batch 16, T4 (measured)


def fit_cost_model():
    """Separate the one-off XLA compile cost from the marginal per-sample cost.

    A run's wall time is `overhead + samples / throughput`. Naively dividing samples by fit time
    conflates the two and, on short runs, understates throughput by 5x -- which would make the
    budget estimate wildly pessimistic. Fitting a line across runs of differing size recovers
    both terms honestly. Short smoke runs are useful here precisely because they pin the intercept.
    """
    by_preset = {}
    for d in [RESULTS_DIR] + PRIOR_RESULT_DIRS:
        for p in glob.glob(os.path.join(d, "*.json")):
            try:
                r = json.load(open(p))
                if not isinstance(r, dict) or "data" not in r or "config" not in r or "timing" not in r:
                    continue
            except Exception:
                continue
            n = r["data"]["train_rows_after_augmentation"] * r["config"]["epochs"]
            by_preset.setdefault(r["config"]["preset"], []).append((n, r["timing"]["fit_seconds"]))

    model = {}
    for preset, pts in by_preset.items():
        xs = np.array([a for a, _ in pts], dtype="float64")
        ys = np.array([b for _, b in pts], dtype="float64")
        if len(np.unique(xs)) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            if slope > 1e-9:
                model[preset] = {"overhead_s": float(max(intercept, 0.0)),
                                 "samples_per_s": float(1.0 / slope), "runs": len(pts)}

    # Presets seen at only one training size (every Block C run uses the same size) cannot be
    # regressed. Borrow the compile overhead measured on the main preset and back out the
    # marginal rate from the remainder -- far better than guessing a FLOP-ratio multiplier.
    main = model.get(CFG.MAIN_PRESET)
    if main:
        for preset, pts in by_preset.items():
            if preset in model:
                continue
            n = float(np.mean([a for a, _ in pts]))
            marginal = float(np.mean([b for _, b in pts])) - main["overhead_s"]
            if marginal > 2.0 and n > 0:
                rate = n / marginal
                if 1.0 < rate < 500.0:
                    model[preset] = {"overhead_s": main["overhead_s"], "samples_per_s": rate,
                                     "runs": len(pts), "approximate": True}
    return model


COST_MODEL = fit_cost_model()
if COST_MODEL:
    print("cost model fitted from completed runs (compile overhead separated from throughput):")
    for k, v in sorted(COST_MODEL.items()):
        print(f"  {k:<30} {v['samples_per_s']:6.1f} samples/s   "
              f"compile {v['overhead_s']:5.0f} s   ({v['runs']} runs)")
else:
    print(f"no completed runs yet -- using defaults ({THROUGHPUT_GUESS:.0f} samples/s "
          f"for {CFG.MAIN_PRESET})")


INFER_SAMPLES = (2 * CFG.EVAL_N + CFG.CALIB_N
                 + (2 * CFG.MTBENCH_MAX if CFG.USE_MTBENCH else 0))
INFER_SPEEDUP = 2.5   # forward-only passes vs. a training step


def estimate_hours(runs):
    """Training + evaluation + per-run compile overhead, using measured costs where available."""
    per_block = {}
    for c in runs:
        mult = 2 if c["arm"] in ("dup", "swap") else 1
        train_samples = c["train_n"] * mult * c["epochs"]
        equiv = train_samples + INFER_SAMPLES / INFER_SPEEDUP
        m = COST_MODEL.get(c["preset"])
        if m:
            sps, overhead = m["samples_per_s"], m["overhead_s"]
        else:
            sps = THROUGHPUT_GUESS / PRESET_COST[c["preset"]]
            overhead = 90 * PRESET_COST[c["preset"]]
        per_block[c["block"]] = per_block.get(c["block"], 0.0) + equiv / sps + overhead
    return {k: v / 3600 for k, v in sorted(per_block.items())}


todo = [c for c in RUNS if find_existing(run_id_of(c)) is None]
est = estimate_hours(todo)
print(f"\nmanifest: {len(RUNS)} runs total, {len(todo)} not yet completed")
print(f"{'block':<10}{'hours (est.)':>14}")
for k, v in est.items():
    print(f"{k:<10}{v:>14.2f}")
print(f"{'TOTAL':<10}{sum(est.values()):>14.2f}")
print("\nKaggle allows ~9-12 h per session and ~30 h of GPU per week.")
print("If the total exceeds one session, run it repeatedly: completed runs are skipped.")
pd.DataFrame(todo).head(40)

# %% [markdown]
# ## 8.1 | Tokenize every split the manifest needs

# %%
def tokens_for(preset, multiturn, need_train_n):
    tk = {}
    tk["train"] = cached_tokens("train", train_pool.iloc[:need_train_n], preset, CFG.SEQ_LEN, multiturn)
    tk["train_y"] = train_pool.label.values[:need_train_n].astype("int32")
    tk["dev"] = cached_tokens("dev", dev_df, preset, CFG.SEQ_LEN, multiturn)
    tk["calib"] = cached_tokens("calib", calib_df, preset, CFG.SEQ_LEN, multiturn)
    tk["eval"] = cached_tokens("eval", eval_df, preset, CFG.SEQ_LEN, multiturn)
    return tk


NEEDED = sorted({(c["preset"], c["multiturn"]) for c in RUNS})
MAX_N_FOR = {}
for c in RUNS:
    k = (c["preset"], c["multiturn"])
    MAX_N_FOR[k] = max(MAX_N_FOR.get(k, 0), c["train_n"])
print("tokenization variants required:", NEEDED)

# %% [markdown]
# ---
# # 9 | Second dataset — MT-Bench Human Judgments (R1-4, R2-2)
#
# `lmsys/mt_bench_human_judgments` is an independent human-preference benchmark with the **same three-class
# label space** (`model_a` / `model_b` / `tie`) and native multi-turn conversations. Models trained on LMSYS
# are evaluated on it **zero-shot** — no fine-tuning, no label leakage — which tests cross-dataset
# generalization rather than in-distribution fit.
#
# Note in the paper that MT-Bench's class prior differs from LMSYS (far fewer ties), so **accuracy and
# macro-F1 are the comparable metrics**; absolute log loss across the two datasets is not.
#
# If the download fails, the notebook continues and the unseen-model-pair split (§4) carries the
# generalization claim alone.

# %%
mtbench_df = None
if CFG.USE_MTBENCH:
    def _load_mtbench():
        try:
            from datasets import load_dataset

            d = load_dataset("lmsys/mt_bench_human_judgments", split="human")
            return d.to_pandas()
        except Exception as e1:
            print("  load_dataset failed:", repr(e1)[:200])
        try:
            from huggingface_hub import list_repo_files, hf_hub_download

            files = [f for f in list_repo_files("lmsys/mt_bench_human_judgments", repo_type="dataset")
                     if f.endswith(".parquet") and "gpt4" not in f.lower()]
            parts = [pd.read_parquet(hf_hub_download("lmsys/mt_bench_human_judgments", f,
                                                     repo_type="dataset"))
                     for f in files]
            return pd.concat(parts, ignore_index=True)
        except Exception as e2:
            print("  hub download failed:", repr(e2)[:200])
        return None

    print("fetching MT-Bench human judgments ...")
    m = _load_mtbench()
    if m is not None and len(m):
        def _turns(conv, role):
            """conversation_* arrives as a list/ndarray of {'role','content'} records."""
            if conv is None or (hasattr(conv, "__len__") and len(conv) == 0):
                return []
            out = []
            for t in conv:
                if not hasattr(t, "get"):
                    continue
                if str(t.get("role", "")) == role:
                    out.append(str(t.get("content", "")))
            return out

        rows = []
        for _, r in m.iterrows():
            ca, cb = r.get("conversation_a"), r.get("conversation_b")
            p = _turns(ca, "user")
            ra = _turns(ca, "assistant")
            rb = _turns(cb, "assistant")
            w = str(r.get("winner", "")).lower()
            lab = 0 if w == "model_a" else 1 if w == "model_b" else 2 if "tie" in w else -1
            if lab < 0 or not p or not ra or not rb:
                continue
            rows.append({"prompt_turns": p, "resp_a_turns": ra, "resp_b_turns": rb, "label": lab})
        mtbench_df = pd.DataFrame(rows)
        if len(mtbench_df) > CFG.MTBENCH_MAX:
            mtbench_df = mtbench_df.sample(CFG.MTBENCH_MAX, random_state=CFG.SPLIT_SEED)
        mtbench_df = mtbench_df.reset_index(drop=True)
        print(f"  MT-Bench rows usable: {len(mtbench_df):,}")
        print("  class balance:",
              mtbench_df.label.value_counts(normalize=True).sort_index().round(4).to_dict())
    else:
        print("  MT-Bench unavailable -- continuing with the unseen-model-pair split only.")

Y_MTBENCH = mtbench_df.label.values.astype("int32") if mtbench_df is not None else None

# %% [markdown]
# ---
# # 9.1 | The runner
#
# Resumable and fault-isolated: a run that raises is logged to `failed/` and the loop continues, so one bad
# architecture cannot cost you a whole session.

# %%
def execute(runs):
    todo = [c for c in runs if find_existing(run_id_of(c)) is None]
    print(f"{len(runs) - len(todo)} already done, {len(todo)} to run\n")
    t_session = time.time()
    current_key, tokens = None, None

    ordered = sorted(todo, key=lambda c: (c["block"], c["preset"], c["name"], c["seed"]))
    for i, cfg in enumerate(ordered):
        elapsed_h = (time.time() - t_session) / 3600
        next_h = sum(estimate_hours([cfg]).values())
        if elapsed_h + next_h > CFG.SESSION_BUDGET_H:
            print("")
            print("=" * 78)
            print("STOPPING EARLY -- session budget reached")
            print(f"  {elapsed_h:.2f} h elapsed, next run needs ~{next_h:.2f} h, "
                  f"budget {CFG.SESSION_BUDGET_H} h")
            print(f"  {len(ordered) - i} runs left; they will be picked up next session.")
            print("  Exiting cleanly so Kaggle saves the output instead of killing the notebook.")
            print("=" * 78)
            break
        rid = run_id_of(cfg)
        key = (cfg["preset"], cfg["multiturn"])
        if key != current_key:
            free_memory()
            print(f"[tokenize] {key} up to {MAX_N_FOR[key]:,} training rows")
            tokens = tokens_for(key[0], key[1], MAX_N_FOR[key])
            if mtbench_df is not None:
                tokens["mtbench"] = cached_tokens("mtbench", mtbench_df, key[0], CFG.SEQ_LEN, key[1])
            current_key = key

        print(f"\n{'='*78}\n[{i+1}/{len(todo)}] {rid}\n"
              f"  elapsed this session: {(time.time()-t_session)/3600:.2f} h\n{'='*78}")
        try:
            r = run_experiment(cfg, tokens)
            print(f"  done in {r['timing']['total_seconds']/60:.1f} min "
                  f"| {r['timing']['train_samples_per_second']:.1f} samples/s "
                  f"| dev log_loss {r['history'].get('val_log_loss', [float('nan')])[-1]:.4f}")
            free_memory()
        except Exception as e:
            import traceback

            msg = traceback.format_exc()
            print("  FAILED:\n", msg[-2000:])
            with open(os.path.join(FAILED_DIR, rid + ".json"), "w") as f:
                json.dump({"run_id": rid, "config": cfg, "error": msg}, f, indent=2)
            free_memory()

    print(f"\nsession finished in {(time.time()-t_session)/3600:.2f} h")


execute(RUNS)

# %% [markdown]
# ---
# # 10 | Analysis
#
# Everything below reads only the stored `.npz` predictions. It runs in seconds on CPU and can be re-run
# after any session to regenerate every table.

# %%
from sklearn.metrics import log_loss as sk_log_loss, f1_score, confusion_matrix, classification_report
from scipy import stats
from scipy.optimize import minimize

EPS = 1e-7


def load_all_runs():
    seen, out = set(), []
    for d in [RESULTS_DIR] + PRIOR_RESULT_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(p))
                if not isinstance(r, dict) or "run_id" not in r:
                    continue
            except Exception:
                continue
            if r["run_id"] in seen:
                continue
            npz = find_preds(r["run_id"])
            if npz is None:
                continue
            r["_npz"] = npz
            seen.add(r["run_id"])
            out.append(r)
    return out


ALL = load_all_runs()
print(f"loaded {len(ALL)} completed runs with predictions")
assert ALL, "no completed runs found -- run section 9.1 first"


def probs(run, key):
    with np.load(run["_npz"]) as z:
        return z[key].astype("float64") if key in z else None


def norm(p):
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def metrics(p, y, mask=None):
    p = norm(p)
    if mask is not None:
        p, y = p[mask], y[mask]
    pred = p.argmax(1)
    return {
        "log_loss": float(sk_log_loss(y, p, labels=[0, 1, 2])),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "n": int(len(y)),
    }

# %% [markdown]
# ## 10.1 | Temperature calibration on a **held-out** split (R2-9, R1-Q4)
#
# The original notebook fitted the temperature on the validation set and then reported calibrated metrics on
# that same set. Here the temperature is fitted on `calib` and applied, unchanged, to `eval`. Two families
# are compared, which answers R1's question about why a single global temperature was used:
#
# * **global**: one scalar `T` for all classes
# * **per-class (vector scaling)**: one `T_k` per class, i.e. three parameters

# %%
def apply_T(p, T):
    z = np.log(np.clip(p, EPS, 1.0))
    z = z / np.asarray(T, dtype="float64")
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p_cal, y_cal, per_class=False):
    def nll(theta):
        T = np.exp(theta)  # keeps T > 0
        return sk_log_loss(y_cal, np.clip(apply_T(p_cal, T), EPS, 1), labels=[0, 1, 2])

    x0 = np.zeros(3 if per_class else 1)
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return np.exp(res.x)


def expected_calibration_error(p, y, bins=15):
    p = norm(p)
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype("float64")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

# %% [markdown]
# ## 10.2 | Aggregation across seeds and paired bootstrap significance testing (R1-5, R2-3)
#
# Two complementary statistics are produced for every comparison:
#
# * **across-seed** mean ± sample std, and a paired *t*-test over seeds — this captures run-to-run variance
#   from initialization and shuffling, which is what the reviewers explicitly asked for;
# * **paired bootstrap over evaluation instances** (10,000 resamples of the 8,000-row eval set, using
#   seed-averaged probabilities) — this captures sampling variance of the benchmark itself and gives a
#   confidence interval on the *difference*, which is far tighter than a 3-point *t*-test.
#
# A difference is reported as significant only when the bootstrap CI excludes zero.

# %%
def group_runs(runs):
    g = {}
    for r in runs:
        c = r["config"]
        k = (c["block"], c["name"], c["preset"], c["train_n"], c["epochs"])
        g.setdefault(k, []).append(r)
    return g


GROUPS = group_runs(ALL)

CALIB_FIT = {}


def eval_probs_for(run, calibrated=True, tta=False, dataset="eval"):
    """Returns the probability matrix a paper table should quote."""
    p = probs(run, f"{dataset}_orig")
    if p is None:
        return None
    if tta:
        ps = probs(run, f"{dataset}_swap")
        if ps is not None:
            p = 0.5 * (norm(p) + norm(ps))
    if calibrated:
        key = (run["run_id"], tta)
        if key not in CALIB_FIT:
            pc = probs(run, "calib_orig")
            if tta:
                pcs = probs(run, "calib_swap")
                if pcs is not None:
                    pc = 0.5 * (norm(pc) + norm(pcs))
            CALIB_FIT[key] = fit_temperature(norm(pc), Y_CALIB, per_class=False)
        p = apply_T(norm(p), CALIB_FIT[key])
    return norm(p)


def group_metrics(runs, y, calibrated=True, tta=False, dataset="eval", mask=None):
    per_seed = []
    for r in runs:
        p = eval_probs_for(r, calibrated=calibrated, tta=tta, dataset=dataset)
        if p is None:
            continue
        per_seed.append(metrics(p, y, mask))
    if not per_seed:
        return None
    out = {"n_seeds": len(per_seed)}
    for k in ("log_loss", "accuracy", "macro_f1"):
        v = np.array([m[k] for m in per_seed])
        out[k + "_mean"] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[k + "_runs"] = v.tolist()
    out["n"] = per_seed[0]["n"]
    return out


def seed_avg_probs(runs, calibrated=True, tta=False, dataset="eval"):
    ps = [eval_probs_for(r, calibrated, tta, dataset) for r in runs]
    ps = [p for p in ps if p is not None]
    return norm(np.mean(ps, axis=0)) if ps else None


def paired_bootstrap(p_a, p_b, y, n_boot=None, seed=7):
    """Bootstrap the paired difference (b - a) in log loss and accuracy over eval instances."""
    n_boot = n_boot or CFG.BOOTSTRAP_N
    rng = np.random.default_rng(seed)
    n = len(y)
    ll_a = -np.log(np.clip(p_a[np.arange(n), y], EPS, 1))
    ll_b = -np.log(np.clip(p_b[np.arange(n), y], EPS, 1))
    ok_a = (p_a.argmax(1) == y).astype("float64")
    ok_b = (p_b.argmax(1) == y).astype("float64")
    d_ll, d_acc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_ll.append(ll_b[idx].mean() - ll_a[idx].mean())
        d_acc.append(ok_b[idx].mean() - ok_a[idx].mean())
    d_ll, d_acc = np.array(d_ll), np.array(d_acc)

    def summarize(d, point):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_two_sided = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi),
                "p_bootstrap": float(min(1.0, p_two_sided)),
                "significant": bool(lo > 0 or hi < 0)}

    # McNemar (exact binomial) on the accuracy disagreements
    b = int(((ok_a == 1) & (ok_b == 0)).sum())
    c = int(((ok_a == 0) & (ok_b == 1)).sum())
    mcnemar_p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0
    return {
        "log_loss": summarize(d_ll, ll_b.mean() - ll_a.mean()),
        "accuracy": summarize(d_acc, ok_b.mean() - ok_a.mean()),
        "mcnemar": {"b": b, "c": c, "p": mcnemar_p},
    }


def seed_ttest(runs_a, runs_b, y, metric="log_loss", calibrated=True):
    """Paired t-test across the shared seeds of two configurations."""
    da = {r["config"]["seed"]: metrics(eval_probs_for(r, calibrated), y)[metric] for r in runs_a}
    db = {r["config"]["seed"]: metrics(eval_probs_for(r, calibrated), y)[metric] for r in runs_b}
    shared = sorted(set(da) & set(db))
    if len(shared) < 2:
        return {"n_pairs": len(shared), "p": None, "note": "need >= 2 shared seeds"}
    a = np.array([da[s] for s in shared])
    b = np.array([db[s] for s in shared])
    t, p = stats.ttest_rel(a, b)
    return {"n_pairs": len(shared), "mean_delta": float((b - a).mean()),
            "t": float(t), "p": float(p)}

# %% [markdown]
# ## 10.3 | Table 3 replacement — matched-condition comparison and the interaction term
#
# Every row trains for the same number of epochs with the same schedule and seeds. `dup` doubles the data
# without any positional change, so `swap − dup` is the position-bias effect with data volume held constant
# (**R2-7**).
#
# The 2×2 factorial also gives the **interaction** the reviewer asked about:
#
# `interaction = (swap+siam − swap-only) − (siam-only − baseline)`
#
# A negative interaction (in log loss) means the two components are *super-additive*: swap augmentation makes
# the `|e_A − e_B|` fusion more useful than it is on its own. That is a testable statement with a CI,
# instead of the paper's unexplained arithmetic.

# %%
def g(name, block="A", preset=CFG.MAIN_PRESET, train_n=None, epochs=None):
    train_n = train_n or MAX_TRAIN_N
    epochs = epochs or CFG.EPOCHS
    return GROUPS.get((block, name, preset, train_n, epochs), [])


TABLE3_ROWS = [
    ("Baseline (no aug, no Siamese)", "noaug_nosiam"),
    ("+ Siamese fusion only", "noaug_siam"),
    ("+ Duplicate aug (data volume control)", "dupaug_nosiam"),
    ("+ Swap aug only", "swapaug_nosiam"),
    ("+ Swap aug + Siamese (full model)", "swapaug_siam"),
]

table3 = []
for label, name in TABLE3_ROWS:
    runs = g(name)
    if not runs:
        continue
    m = group_metrics(runs, Y_EVAL, calibrated=True)
    m_unc = group_metrics(runs, Y_EVAL, calibrated=False)
    table3.append({
        "row": label, "name": name, "seeds": m["n_seeds"],
        "train_rows": runs[0]["config"]["train_n"],
        "aug_rows": runs[0]["data"]["train_rows_after_augmentation"],
        "epochs": runs[0]["config"]["epochs"],
        "log_loss": m["log_loss_mean"], "log_loss_std": m["log_loss_std"],
        "accuracy": m["accuracy_mean"], "accuracy_std": m["accuracy_std"],
        "macro_f1": m["macro_f1_mean"], "macro_f1_std": m["macro_f1_std"],
        "log_loss_uncal": m_unc["log_loss_mean"],
    })
table3_df = pd.DataFrame(table3)
print("TABLE 3 (replacement) -- all rows trained under identical conditions\n")
if len(table3_df):
    print(table3_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------------ significance vs. baseline
sig3 = []
base_runs = g("noaug_nosiam")
if base_runs:
    p_base = seed_avg_probs(base_runs)
    for label, name in TABLE3_ROWS[1:]:
        runs = g(name)
        if not runs:
            continue
        p_x = seed_avg_probs(runs)
        bs = paired_bootstrap(p_base, p_x, Y_EVAL)
        tt = seed_ttest(base_runs, runs, Y_EVAL)
        sig3.append({
            "comparison": f"{name} vs baseline",
            "d_log_loss": bs["log_loss"]["delta"],
            "ll_ci": f"[{bs['log_loss']['ci_low']:+.4f}, {bs['log_loss']['ci_high']:+.4f}]",
            "ll_sig": bs["log_loss"]["significant"],
            "d_acc_pp": 100 * bs["accuracy"]["delta"],
            "acc_ci_pp": f"[{100*bs['accuracy']['ci_low']:+.2f}, {100*bs['accuracy']['ci_high']:+.2f}]",
            "mcnemar_p": bs["mcnemar"]["p"],
            "seed_ttest_p": tt.get("p"),
        })
sig3_df = pd.DataFrame(sig3)
print("\n\nPaired bootstrap vs. the matched baseline (10,000 resamples of the eval set)\n")
if len(sig3_df):
    print(sig3_df.to_string(index=False))

# ------------------------------------------------------------------ isolated effects and interaction
effects = {}
if g("dupaug_nosiam") and g("swapaug_nosiam"):
    effects["position_bias_effect (swap - dup, data volume held constant)"] = paired_bootstrap(
        seed_avg_probs(g("dupaug_nosiam")), seed_avg_probs(g("swapaug_nosiam")), Y_EVAL)
if g("noaug_nosiam") and g("dupaug_nosiam"):
    effects["data_volume_effect (dup - baseline)"] = paired_bootstrap(
        seed_avg_probs(g("noaug_nosiam")), seed_avg_probs(g("dupaug_nosiam")), Y_EVAL)

INTERACTION = None
if all(g(k) for k in ["noaug_nosiam", "noaug_siam", "swapaug_nosiam", "swapaug_siam"]):
    def _ll(runs):
        p = seed_avg_probs(runs)
        return -np.log(np.clip(p[np.arange(len(Y_EVAL)), Y_EVAL], EPS, 1))

    ll00, ll01 = _ll(g("noaug_nosiam")), _ll(g("noaug_siam"))
    ll10, ll11 = _ll(g("swapaug_nosiam")), _ll(g("swapaug_siam"))
    inter_vec = (ll11 - ll10) - (ll01 - ll00)
    rng = np.random.default_rng(11)
    boots = [inter_vec[rng.integers(0, len(inter_vec), len(inter_vec))].mean()
             for _ in range(CFG.BOOTSTRAP_N)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    INTERACTION = {
        "interaction_log_loss": float(inter_vec.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "simple_effect_swap_alone": float(ll10.mean() - ll00.mean()),
        "simple_effect_siamese_alone": float(ll01.mean() - ll00.mean()),
        "combined_effect": float(ll11.mean() - ll00.mean()),
        "sum_of_simple_effects": float((ll10.mean() - ll00.mean()) + (ll01.mean() - ll00.mean())),
    }
    print("\n\nFactorial decomposition (log loss; negative = improvement)")
    print(json.dumps(INTERACTION, indent=2))
    verdict = ("super-additive (synergy confirmed)" if INTERACTION["significant"]
               and INTERACTION["interaction_log_loss"] < 0
               else "sub-additive" if INTERACTION["significant"]
               else "NOT distinguishable from additive at 95% confidence")
    print(f"\n  => components are {verdict}")
    print("  This is the honest answer to R1-Q3: the paper's -0.0488 vs -0.0338 gap was a single-run "
          "artefact until now.")

# %% [markdown]
# ## 10.3b | Symmetrised inference and seed ensembling — free improvements
#
# Both of these use predictions that have **already been computed**; neither costs extra training.
#
# **Symmetrised inference.** The model is run on `(A,B)` and on `(B,A)`, and the two predictions are
# averaged after mirroring. For a symmetric-by-design architecture this is the natural inference rule, and
# it makes the classifier *exactly* position-invariant rather than approximately so — flip consistency
# becomes 1.000 by construction. Report the doubled inference cost alongside it.
#
# **Seed ensembling.** Averaging the predictions of the runs you already trained. Standard practice, and
# honest as long as the table says how many members the ensemble has.
#
# These belong in the paper as a separate "inference-time" table, not folded silently into the main
# comparison — the main comparison must stay single-model so it isolates the architecture.

# %%
full = g("swapaug_siam")   # the full model's runs; also reused by 10.7 and 10.10

sym_rows = []
if full:
    for label, tta, ensemble in [
        ("Single input order, single model", False, False),
        ("Symmetrised inference (both orders)", True, False),
        (f"Seed ensemble ({len(full)} models)", False, True),
        ("Seed ensemble + symmetrised", True, True),
    ]:
        if ensemble:
            p = seed_avg_probs(full, calibrated=True, tta=tta)
            m = metrics(p, Y_EVAL)
            row = {"inference rule": label, "log_loss": m["log_loss"], "log_loss_std": 0.0,
                   "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        else:
            gm = group_metrics(full, Y_EVAL, calibrated=True, tta=tta)
            row = {"inference rule": label, "log_loss": gm["log_loss_mean"],
                   "log_loss_std": gm["log_loss_std"], "accuracy": gm["accuracy_mean"],
                   "macro_f1": gm["macro_f1_mean"]}
        row["relative_inference_cost"] = (2 if tta else 1) * (len(full) if ensemble else 1)
        sym_rows.append(row)

    # is symmetrised inference a real gain, or noise?
    p_single = seed_avg_probs(full, calibrated=True, tta=False)
    p_tta = seed_avg_probs(full, calibrated=True, tta=True)
    bs_tta = paired_bootstrap(p_tta, p_single, Y_EVAL)

    # flip consistency is 1.0 by construction under symmetrised inference
    sym_df = pd.DataFrame(sym_rows)
    print("INFERENCE-TIME IMPROVEMENTS (no extra training)\n")
    print(sym_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsymmetrised vs single order: delta log loss "
          f"{-bs_tta['log_loss']['delta']:+.4f} "
          f"[{-bs_tta['log_loss']['ci_high']:+.4f}, {-bs_tta['log_loss']['ci_low']:+.4f}], "
          f"significant={bs_tta['log_loss']['significant']}")
    print("Symmetrised inference makes the classifier exactly position-invariant "
          "(flip consistency = 1.000 by construction) at 2x inference cost.")

# %% [markdown]
# ## 10.4 | Position-bias analysis (R2-8)
#
# The reviewer objected that Figure 9(b) showed only *prediction rates*, which can look balanced while the
# model is still positionally biased. Reported here instead:
#
# * **accuracy conditioned on the true winner's position** — `acc | winner = A` vs `acc | winner = B`;
#   a genuinely position-invariant model has no gap;
# * **flip consistency** — how often the prediction on `(A,B)` is the exact mirror of the prediction on
#   `(B,A)`. This is the strongest direct measure of positional invariance and requires no labels;
# * **positional preference index** — mean `P(A wins) − P(B wins)` averaged over both input orders. It is
#   exactly zero for a perfectly symmetric model.

# %%
def calibrated_swap_probs(run):
    """Swapped-order predictions, mirrored back and calibrated with this run's temperature."""
    key = (run["run_id"], False)
    if key not in CALIB_FIT:
        CALIB_FIT[key] = fit_temperature(norm(probs(run, "calib_orig")), Y_CALIB, per_class=False)
    ps = probs(run, "eval_swap")
    return apply_T(norm(ps), CALIB_FIT[key]) if ps is not None else None


def position_report(runs):
    # p_o   : prediction on (A, B) as presented
    # p_s   : prediction on (B, A), already mirrored back into the original label space
    p_o = seed_avg_probs(runs, calibrated=True, tta=False)
    # The swapped-order predictions must get the SAME temperature as the original-order ones,
    # or the two log-loss columns below are not comparable. (Temperature scaling cannot change
    # an argmax, so flip consistency is unaffected either way.)
    ps = [calibrated_swap_probs(r) for r in runs]
    ps = [x for x in ps if x is not None]
    p_s_mirror = norm(np.mean(ps, axis=0)) if ps else None
    if p_o is None or p_s_mirror is None:
        return None
    pred_o, pred_s = p_o.argmax(1), p_s_mirror.argmax(1)
    a_mask, b_mask, t_mask = (Y_EVAL == 0), (Y_EVAL == 1), (Y_EVAL == 2)
    acc_a = float((pred_o[a_mask] == 0).mean())
    acc_b = float((pred_o[b_mask] == 1).mean())
    rates = np.bincount(pred_o, minlength=3) / len(pred_o)
    ppi = float(np.mean(0.5 * ((p_o[:, 0] - p_o[:, 1]) + (p_s_mirror[:, 0] - p_s_mirror[:, 1]))))
    return {
        "acc_given_winner_A": acc_a,
        "acc_given_winner_B": acc_b,
        "position_accuracy_gap_pp": 100 * (acc_a - acc_b),
        "acc_given_tie": float((pred_o[t_mask] == 2).mean()),
        "pred_rate_A": float(rates[0]), "pred_rate_B": float(rates[1]),
        "pred_rate_tie": float(rates[2]),
        "flip_consistency": float((pred_o == pred_s).mean()),
        "positional_preference_index": ppi,
        "logloss_original_order": float(sk_log_loss(Y_EVAL, p_o, labels=[0, 1, 2])),
        "logloss_swapped_order": float(sk_log_loss(Y_EVAL, p_s_mirror, labels=[0, 1, 2])),
    }


pos_rows = []
for label, name in [("Baseline (no swap aug)", "noaug_nosiam"),
                    ("Duplicate aug (no swap)", "dupaug_nosiam"),
                    ("Swap aug only", "swapaug_nosiam"),
                    ("Full model (swap + Siamese)", "swapaug_siam")]:
    runs = g(name)
    if not runs:
        continue
    rep = position_report(runs)
    if rep:
        rep = {"model": label, **rep}
        pos_rows.append(rep)
position_df = pd.DataFrame(pos_rows)
print("POSITION-BIAS TABLE (replaces Figure 9b)\n")
if len(position_df):
    cols = ["model", "acc_given_winner_A", "acc_given_winner_B", "position_accuracy_gap_pp",
            "flip_consistency", "positional_preference_index", "logloss_original_order",
            "logloss_swapped_order"]
    print(position_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRead this as: a model with no position bias has a gap near 0 pp, flip consistency near 1.00,")
    print("a preference index near 0, and equal log loss in both input orders.")

# %% [markdown]
# ## 10.5 | Calibration table — global vs per-class temperature (R2-9, R1-Q4)

# %%
calib_rows = []
_full = g("swapaug_siam") or g("noaug_nosiam")
for r in _full:
    p_cal, p_ev = norm(probs(r, "calib_orig")), norm(probs(r, "eval_orig"))
    T_g = fit_temperature(p_cal, Y_CALIB, per_class=False)
    T_c = fit_temperature(p_cal, Y_CALIB, per_class=True)
    for tag, p in [("uncalibrated", p_ev),
                   ("global T (held-out fit)", apply_T(p_ev, T_g)),
                   ("per-class T (held-out fit)", apply_T(p_ev, T_c))]:
        m = metrics(p, Y_EVAL)
        calib_rows.append({
            "seed": r["config"]["seed"], "method": tag,
            "T": np.round(T_g, 4).tolist() if tag.startswith("global")
                 else (np.round(T_c, 4).tolist() if tag.startswith("per-class") else None),
            "eval_log_loss": m["log_loss"], "eval_accuracy": m["accuracy"],
            "eval_ece": expected_calibration_error(p, Y_EVAL),
        })
calib_df_out = pd.DataFrame(calib_rows)
if len(calib_df_out):
    print("CALIBRATION (temperature fitted on the disjoint calib split, evaluated on eval)\n")
    print(calib_df_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summ = calib_df_out.groupby("method")[["eval_log_loss", "eval_accuracy", "eval_ece"]].agg(
        ["mean", "std"])
    print("\nacross seeds:\n", summ.round(4).to_string())
    print("\nPer-class scaling adds two parameters. Quote it in the paper only if it beats global "
          "scaling by more than the seed-to-seed std -- otherwise state that the extra parameters "
          "did not help, which is a complete answer to R1-Q4.")

# %% [markdown]
# ## 10.6 | Table 4 replacement — architectures under identical conditions (R2-6)

# %%
arch_rows = []
for preset in ARCH_PRESETS:
    label = preset.replace("_en_uncased", "").replace("_en", "").replace("_", "-")
    if preset == CFG.MAIN_PRESET:
        label += " (ours)"
    runs = g(f"arch_{preset}", block="C", preset=preset, train_n=ARCH_TRAIN_N)
    if not runs:
        continue
    m = group_metrics(runs, Y_EVAL, calibrated=True)
    arch_rows.append({
        "model": label,
        "params_M": round(runs[0]["params"]["total"] / 1e6, 1),
        "backbone_M": round(runs[0]["params"]["backbone"] / 1e6, 1),
        "seeds": m["n_seeds"],
        "log_loss": m["log_loss_mean"], "log_loss_std": m["log_loss_std"],
        "accuracy": m["accuracy_mean"], "accuracy_std": m["accuracy_std"],
        "macro_f1": m["macro_f1_mean"],
        "train_min": round(np.mean([r["timing"]["fit_seconds"] for r in runs]) / 60, 1),
        "infer_ms": round(float(np.median([r["timing"]["inference_ms_per_sample"] for r in runs])), 2),
    })
arch_df = pd.DataFrame(arch_rows)
print(f"TABLE 4 (replacement) -- every backbone trained on the same {ARCH_TRAIN_N:,} rows, "
      f"same augmentation,\nsame {CFG.EPOCHS} epochs, same LR schedule, same {len(ARCH_SEEDS)} seeds\n")
if len(arch_df):
    print(arch_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nNOTE FOR THE PAPER: this table uses {ARCH_TRAIN_N:,} training rows, whereas Table 3 uses "
          f"{MAX_TRAIN_N:,}.\nSay so in the caption. The backbone comparison only needs to be matched "
          "*within itself*, and the\nlarger models cost ~3.6x per sample; our own backbone is included "
          "here at the reduced scale so the\ntable is self-contained. Its numbers will differ from "
          "Table 3 for that reason -- that is expected,\nnot an inconsistency, but it must be stated.")
    print("\nNOTE FOR THE PAPER: the Gemma-2-9b-it row in the submitted Table 4 was not produced by this")
    print("pipeline. Either drop it or label it explicitly as a figure quoted from the competition")
    print("leaderboard under a different protocol -- it cannot be presented as a controlled comparison.")

# %% [markdown]
# ## 10.7 | Table 7 replacement — ablation with the data-volume confound removed (R2-7)
#
# Each row removes exactly one component from the full model, holding everything else fixed. The swap row
# is measured against `dup` rather than against `none`, so it reports position-bias mitigation with data
# volume already controlled — this is the correction R2-7 required.

# %%
abl_rows = []
full = g("swapaug_siam")
if full:
    p_full = seed_avg_probs(full)
    m_full = group_metrics(full, Y_EVAL)
    abl_defs = [
        ("Full model", None, None),
        ("- Swap augmentation (vs. volume-matched dup)", "dupaug_nosiam", "A"),
        ("- Siamese fusion |e_A - e_B|", "swapaug_nosiam", "A"),
        ("- Multi-turn context", "full_nomultiturn", "B"),
    ]
    for label, name, block in abl_defs:
        if name is None:
            abl_rows.append({"component_removed": label,
                             "log_loss": m_full["log_loss_mean"],
                             "log_loss_std": m_full["log_loss_std"],
                             "accuracy": m_full["accuracy_mean"],
                             "delta_log_loss": 0.0, "ci": "-", "significant": "-"})
            continue
        runs = g(name, block=block)
        if not runs:
            continue
        m = group_metrics(runs, Y_EVAL)
        bs = paired_bootstrap(p_full, seed_avg_probs(runs), Y_EVAL)
        abl_rows.append({
            "component_removed": label,
            "log_loss": m["log_loss_mean"], "log_loss_std": m["log_loss_std"],
            "accuracy": m["accuracy_mean"],
            "delta_log_loss": bs["log_loss"]["delta"],
            "ci": f"[{bs['log_loss']['ci_low']:+.4f}, {bs['log_loss']['ci_high']:+.4f}]",
            "significant": bs["log_loss"]["significant"],
        })

# temperature calibration contributes post hoc -- measure it without retraining
if full:
    m_unc = group_metrics(full, Y_EVAL, calibrated=False)
    m_cal = group_metrics(full, Y_EVAL, calibrated=True)
    abl_rows.append({
        "component_removed": "- Temperature calibration (post hoc)",
        "log_loss": m_unc["log_loss_mean"], "log_loss_std": m_unc["log_loss_std"],
        "accuracy": m_unc["accuracy_mean"],
        "delta_log_loss": m_unc["log_loss_mean"] - m_cal["log_loss_mean"],
        "ci": "see 10.5", "significant": "see 10.5",
    })

ablation_df = pd.DataFrame(abl_rows)
print("TABLE 7 (replacement) -- leave-one-out ablation, positive delta = the component helps\n")
if len(ablation_df):
    print(ablation_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDo NOT convert these into '% of total contribution' as the submitted Table 7 did: the")
    print("components are not additive, and the percentages implied a decomposition that does not exist.")

# %% [markdown]
# ## 10.8 | Data efficiency

# %%
eff_specs = [(f"eff_{int(fr*100)}pct", "D", eff_n(fr)) for fr in EFF_FRACTIONS]
eff_specs.append(("swapaug_siam", "A", MAX_TRAIN_N))
if BLOCKS["E_scale"]:
    eff_specs.append(("fullpool", "E", len(train_pool)))

eff_rows = []
for name, block, n in eff_specs:
    runs = g(name, block=block, train_n=n)
    if not runs:
        continue
    m = group_metrics(runs, Y_EVAL)
    eff_rows.append({"train_rows": n, "augmented_rows": 2 * n, "seeds": m["n_seeds"],
                     "log_loss": m["log_loss_mean"], "log_loss_std": m["log_loss_std"],
                     "accuracy": m["accuracy_mean"], "accuracy_std": m["accuracy_std"]})
eff_df = pd.DataFrame(eff_rows).sort_values("train_rows").reset_index(drop=True)
if len(eff_df):
    best = eff_df.log_loss.min()
    eff_df["pct_of_best"] = (best / eff_df.log_loss * 100).round(2)
    print("DATA EFFICIENCY\n")
    print(eff_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n'% of best' is best_log_loss / this_log_loss. State the definition in the paper -- the")
    print("submitted 95.1% figure never defined it.")

# %% [markdown]
# ## 10.9 | Generalization (R1-4, R2-2)
#
# Two independent tests:
#
# 1. **MT-Bench Human Judgments, zero-shot** — a different dataset, different annotators, different model
#    population, no fine-tuning.
# 2. **Unseen model pairings within LMSYS** — eval rows whose `(model_a, model_b)` pairing never occurs in
#    training. This isolates whether the model learned transferable quality judgements or memorized
#    per-model idiosyncrasies.

# %%
gen_rows = []
for label, name, block, preset in [("Full model", "swapaug_siam", "A", CFG.MAIN_PRESET),
                                   ("Baseline", "noaug_nosiam", "A", CFG.MAIN_PRESET)]:
    runs = g(name, block=block, preset=preset)
    if not runs:
        continue
    m_all = group_metrics(runs, Y_EVAL)
    row = {"model": label, "eval_log_loss": m_all["log_loss_mean"],
           "eval_accuracy": m_all["accuracy_mean"]}
    m_uns = group_metrics(runs, Y_EVAL, mask=UNSEEN_PAIR_MASK)
    if m_uns:
        row["unseen_pairs_n"] = m_uns["n"]
        row["unseen_pairs_log_loss"] = m_uns["log_loss_mean"]
        row["unseen_pairs_accuracy"] = m_uns["accuracy_mean"]
    if Y_MTBENCH is not None:
        m_mt = group_metrics(runs, Y_MTBENCH, dataset="mtbench", calibrated=False)
        if m_mt:
            row["mtbench_n"] = m_mt["n"]
            row["mtbench_accuracy"] = m_mt["accuracy_mean"]
            row["mtbench_accuracy_std"] = m_mt["accuracy_std"]
            row["mtbench_macro_f1"] = m_mt["macro_f1_mean"]
            row["mtbench_log_loss"] = m_mt["log_loss_mean"]
    gen_rows.append(row)

gen_df = pd.DataFrame(gen_rows)
print("GENERALIZATION\n")
if len(gen_df):
    print(gen_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
if Y_MTBENCH is not None:
    maj = float(np.bincount(Y_MTBENCH, minlength=3).max() / len(Y_MTBENCH))
    print(f"\nMT-Bench majority-class accuracy (the floor any claim must beat): {maj:.4f}")
    print("MT-Bench class prior differs from LMSYS, so compare accuracy and macro-F1 across datasets,")
    print("not absolute log loss.")

# %% [markdown]
# ## 10.10 | Per-class report, confusion matrix and training curves for the full model

# %%
if full:
    p = seed_avg_probs(full)
    pred = p.argmax(1)
    print(classification_report(Y_EVAL, pred, target_names=list(CFG.LABEL2NAME.values()), digits=4))
    cm = confusion_matrix(Y_EVAL, pred)
    print("confusion matrix (rows = true, cols = predicted):\n",
          pd.DataFrame(cm, index=list(CFG.LABEL2NAME.values()),
                       columns=list(CFG.LABEL2NAME.values())).to_string())

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for r in full:
        h = r["history"]
        ep = range(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], alpha=.7)
        axes[0].plot(ep, h.get("val_loss", []), "--", alpha=.7)
        axes[1].plot(ep, h.get("val_log_loss", []), alpha=.8,
                     label=f"seed {r['config']['seed']}")
        axes[2].plot(ep, h.get("val_accuracy", []), alpha=.8,
                     label=f"seed {r['config']['seed']}")
    axes[0].set_title("loss (solid=train, dashed=dev)")
    axes[1].set_title("dev log loss")
    axes[2].set_title("dev accuracy")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK, "training_curves_v2.png"), dpi=160)
    plt.show()

    fig, ax = plt.subplots(figsize=(4.6, 4))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=cmn.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > cmn.max() * .6 else "black")
    ax.set_xticks(range(3), ["A", "B", "tie"])
    ax.set_yticks(range(3), ["A", "B", "tie"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("row-normalized confusion matrix")
    plt.colorbar(im, fraction=.046)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK, "confusion_matrix_v2.png"), dpi=160)
    plt.show()

# %% [markdown]
# ---
# # 11 | Export — LaTeX tables, a machine-readable dump, and reviewer-response notes
#
# Download these three files from the notebook output:
#
# * `paper_tables.tex` — table bodies to paste into the manuscript
# * `paper_results.json` — every number, for checking the text against the tables
# * `reviewer_response_notes.md` — what each table now supports, and what claims must be softened

# %%
def fmt(m, s, d=4):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}" if s else f"{m:.{d}f}"


tex = []
tex.append("% ==== Table 3 replacement: matched-condition comparison ====")
tex.append("\\begin{table}[!htbp]\\centering")
tex.append("\\caption{Matched-condition comparison. All rows use identical training data, epochs, "
           "learning-rate schedule and random seeds; only the marked component differs. "
           "Values are mean $\\pm$ standard deviation over %d seeds on a held-out set of %d "
           "examples.}" % (len(CFG.SEEDS), len(Y_EVAL)))
tex.append("\\label{tab:main_results}")
tex.append("\\begin{tabular}{@{}lccccc@{}}\\toprule")
tex.append("\\textbf{Configuration} & \\textbf{Train rows} & \\textbf{Epochs} & "
           "\\textbf{Log Loss} & \\textbf{Accuracy} & \\textbf{Macro F1} \\\\ \\midrule")
for r in table3:
    tex.append(f"{r['row']} & {r['aug_rows']:,} & {r['epochs']} & "
               f"{fmt(r['log_loss'], r['log_loss_std'])} & "
               f"{fmt(100*r['accuracy'], 100*r['accuracy_std'], 2)}\\% & "
               f"{fmt(100*r['macro_f1'], 100*r['macro_f1_std'], 2)}\\% \\\\")
tex.append("\\bottomrule\\end{tabular}\\end{table}\n")

if len(arch_df):
    tex.append("% ==== Table 4 replacement: architectures under identical conditions ====")
    tex.append("\\begin{table}[!htbp]\\centering")
    tex.append("\\caption{Alternative backbones. Every model uses the same %d-row swap-augmented "
               "training set, the same %d epochs, the same learning-rate schedule and the same %d "
               "random seeds; only the backbone differs. This table uses a smaller training set than "
               "Table~\\ref{tab:main_results} because the larger backbones cost roughly %.1f times more "
               "per sample; the comparison is matched within the table.}"
               % (ARCH_TRAIN_N, CFG.EPOCHS, len(ARCH_SEEDS), PRESET_COST["bert_base_en_uncased"]))
    tex.append("\\label{tab:comparison}")
    tex.append("\\begin{tabular}{@{}lcccc@{}}\\toprule")
    tex.append("\\textbf{Backbone} & \\textbf{Params} & \\textbf{Log Loss} & "
               "\\textbf{Accuracy} & \\textbf{Inference (ms)} \\\\ \\midrule")
    for _, r in arch_df.iterrows():
        tex.append(f"{r['model']} & {r['params_M']}M & "
                   f"{fmt(r['log_loss'], r['log_loss_std'])} & "
                   f"{fmt(100*r['accuracy'], 100*r['accuracy_std'], 2)}\\% & {r['infer_ms']} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}\n")

if len(position_df):
    tex.append("% ==== Position-bias table (replaces Figure 9b) ====")
    tex.append("\\begin{table}[!htbp]\\centering")
    tex.append("\\caption{Position-bias analysis. \\emph{Acc$\\mid$A} and \\emph{Acc$\\mid$B} are accuracy "
               "restricted to examples whose true winner is Model A and Model B respectively; a "
               "position-invariant classifier has a gap of zero. \\emph{Flip consistency} is the fraction "
               "of examples whose prediction is the exact mirror image when the two responses are "
               "exchanged at the input.}")
    tex.append("\\label{tab:position}")
    tex.append("\\begin{tabular}{@{}lcccc@{}}\\toprule")
    tex.append("\\textbf{Model} & \\textbf{Acc$\\mid$A} & \\textbf{Acc$\\mid$B} & "
               "\\textbf{Gap (pp)} & \\textbf{Flip cons.} \\\\ \\midrule")
    for _, r in position_df.iterrows():
        tex.append(f"{r['model']} & {100*r['acc_given_winner_A']:.2f}\\% & "
                   f"{100*r['acc_given_winner_B']:.2f}\\% & "
                   f"{r['position_accuracy_gap_pp']:+.2f} & {r['flip_consistency']:.4f} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}\n")

if len(ablation_df):
    tex.append("% ==== Table 7 replacement: leave-one-out ablation ====")
    tex.append("\\begin{table}[!htbp]\\centering")
    tex.append("\\caption{Leave-one-out ablation with bootstrap confidence intervals. The swap-augmentation "
               "row is measured against a duplicate-augmentation control with identical data volume, so it "
               "reports position-bias mitigation alone.}")
    tex.append("\\label{tab:ablation}")
    tex.append("\\begin{tabular}{@{}lccc@{}}\\toprule")
    tex.append("\\textbf{Component removed} & \\textbf{Log Loss} & \\textbf{$\\Delta$} & "
               "\\textbf{95\\% CI} \\\\ \\midrule")
    for _, r in ablation_df.iterrows():
        tex.append(f"{r['component_removed']} & {fmt(r['log_loss'], r['log_loss_std'])} & "
                   f"{r['delta_log_loss']:+.4f} & {r['ci']} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}\n")

tex.append("% ==== Reproducibility (R2-4) ====")
tex.append("% " + json.dumps(ENV_INFO))

with open(os.path.join(WORK, "paper_tables.tex"), "w") as f:
    f.write("\n".join(tex))
print("wrote paper_tables.tex")

PAPER = {
    "protocol": {k: v for k, v in vars(CFG).items() if k.isupper()},
    "environment": ENV_INFO,
    "separator_tokens": SEPARATOR_TOKENS,
    "split_sizes": SPLIT_SIZES,
    "train_rows_used": int(MAX_TRAIN_N),
    "tokenizer_failures": len(TOKENIZE_FAILURES),
    "inference_time_variants": sym_rows,
    "table3_matched_comparison": table3,
    "table3_significance": sig3,
    "factorial_interaction": INTERACTION,
    "isolated_effects": effects,
    "position_bias": pos_rows,
    "calibration": calib_rows,
    "table4_architectures": arch_df.to_dict("records") if len(arch_df) else [],
    "table7_ablation": ablation_df.to_dict("records") if len(ablation_df) else [],
    "data_efficiency": eff_df.to_dict("records") if len(eff_df) else [],
    "generalization": gen_rows,
    "n_runs_completed": len(ALL),
}
with open(os.path.join(WORK, "paper_results.json"), "w") as f:
    json.dump(PAPER, f, indent=2, default=str)
print("wrote paper_results.json")

# %%
notes = []
notes.append("# Notes for the revision and the response letter\n")
notes.append(f"Generated {datetime.now(timezone.utc).isoformat()} from {len(ALL)} completed runs.\n")

notes.append("## 1. A correction you must disclose\n")
notes.append(
    "The submitted results were computed on a validation set that contained swapped duplicates of "
    "training rows: the original notebook applied swap augmentation before `train_test_split`. "
    "`metrics.json` from that run shows 91,963 train / 22,991 validation rows, an 80/20 split of the "
    "114,954 **augmented** rows. Every number in the submitted Tables 3-7 and Figures 8-9 is affected.\n\n"
    "All results in this notebook come from a corrected protocol: the original rows are partitioned "
    f"into disjoint train / dev / calibration / evaluation splits ({SPLIT_SIZES}) and augmentation is "
    "applied to the training split only. The notebook asserts that the splits share no `id`.\n\n"
    "Disclose this in the response letter. It strengthens the paper: it shows the pipeline was audited, "
    "and the corrected comparisons are the ones the reviewers asked for.\n")

notes.append("## 2. Scale of the revised experiments\n")
notes.append(
    f"Trained on {MAX_TRAIN_N:,} of {len(train_pool):,} available rows, {CFG.EPOCHS} epochs, "
    f"sequence length {CFG.SEQ_LEN}, {len(CFG.SEEDS)} seeds per configuration, evaluated on a fixed "
    f"held-out set of {len(Y_EVAL):,} examples.\n\n"
    "State this plainly in the experimental setup. The reduced scale is a deliberate trade: a matched, "
    "replicated, leakage-free comparison at moderate scale answers the reviewers' questions, whereas a "
    "single unmatched run at full scale does not. Absolute log loss is therefore higher than the 0.9871 "
    "reported in the submission; the *comparisons* are now valid, which is what was being challenged.\n")

notes.append("## 3. Claims that must change\n")
notes.append(
    "- **Abstract (R2-10).** Remove 'as competitive as a DeBERTa-v3-extra-small model'. The model *is* "
    "one. Say it is compared against an identically sized, identically trained baseline without the "
    "symmetric architecture and swap augmentation.\n"
    "- **Table 4 Gemma-2-9b-it row.** Not produced by this pipeline and not reproducible here. Drop it, "
    "or move it to related work labelled as a leaderboard figure obtained under a different protocol.\n"
    "- **Table 7 percentages.** Do not report '% of total contribution'. The components are not additive; "
    "report leave-one-out deltas with confidence intervals instead.\n"
    "- **R1-3, absolute performance.** Report the majority-class and random baselines alongside the "
    "headline accuracy, and describe the result as a modest but statistically reliable improvement over a "
    "matched baseline rather than as competitive with large models.\n"
    "- **Synergy claim (R1-Q3).** Replaced by the factorial interaction term in section 10.3, which "
    "carries a confidence interval. If that interval includes zero, state that the components are "
    "additive within measurement error and delete the synergy discussion.\n")

notes.append("## 4. Where each reviewer point is now answered\n")
notes.append(
    "| Reviewer point | Evidence |\n|---|---|\n"
    "| R1-2 / R2-1 unfair baseline | Table 3 replacement: 2x2 factorial at matched epochs and matched "
    "data volume |\n"
    "| R2-7 data vs. architecture confound | `dup` arm; the swap-minus-dup contrast in section 10.3 |\n"
    "| R1-5 / R2-3 no statistics | mean +- std over seeds, paired bootstrap CIs, McNemar, paired t-test |\n"
    "| R2-6 Table 4 conditions | Block C trains every backbone identically; conditions printed in the "
    "table caption |\n"
    "| R2-8 position bias | accuracy conditioned on the winner's position, flip consistency, preference "
    "index |\n"
    "| R2-9 calibration leakage | temperature fitted on a disjoint calibration split; global vs per-class "
    "compared |\n"
    "| R1-Q2 turn separator | exact literal separators and their sub-word IDs are recorded in "
    "`paper_results.json` |\n"
    "| R1 section 3.3 vagueness | Algorithm 1 pseudocode, implemented as `greedy_multi_turn_context` |\n"
    "| R1-4 / R2-2 single dataset | MT-Bench zero-shot transfer plus the unseen-model-pair split |\n"
    "| R2-4 reproducibility | full version record in `paper_results.json`; publish this notebook and the "
    "results directory on GitHub or Zenodo and cite the DOI |\n")

notes.append("## 5. Still to do outside this notebook\n")
notes.append(
    "- **R2-5 / R1 writing quality.** A professional language edit is required. The specific sentences the "
    "reviewers quoted must all be rewritten.\n"
    "- **R1-1 novelty.** Add a paragraph stating precisely what differs from prior uses of swap "
    "augmentation in pairwise ranking, and cite those works specifically rather than gesturing at [9].\n"
    "- **R1 minor, Figure 2.** Redraw the pipeline diagram at larger font size.\n"
    "- **R1 minor, references.** Remove references [17]-[32] that are never cited in the text.\n"
    "- **R2-4 code release.** Make the Kaggle notebook public and mirror it to GitHub or Zenodo.\n")

with open(os.path.join(WORK, "reviewer_response_notes.md"), "w") as f:
    f.write("\n".join(notes))
print("wrote reviewer_response_notes.md")

print("\n" + "=" * 78)
print("OUTPUT FILES")
for fn in ["paper_tables.tex", "paper_results.json", "reviewer_response_notes.md",
           "training_curves_v2.png", "confusion_matrix_v2.png"]:
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        print(f"  {fn:<32}{os.path.getsize(p)/1024:8.1f} KB")
print(f"  results/          {len(glob.glob(os.path.join(RESULTS_DIR,'*.json'))):>4} run records")
print(f"  predictions/      {len(glob.glob(os.path.join(PREDS_DIR,'*.npz'))):>4} prediction files")
_failed = glob.glob(os.path.join(FAILED_DIR, "*.json"))
if _failed:
    print(f"  failed/           {len(_failed):>4} FAILED RUNS -- inspect these")
print("=" * 78)
