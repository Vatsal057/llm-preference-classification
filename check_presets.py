#!/usr/bin/env python3
"""check_presets.py - Pre-flight Preset Check & Model Cache Downloader

Run this script on the HPC login node (or any machine with internet access)
BEFORE submitting GPU jobs to confirm that all 4 backbone architectures can
be downloaded, cached, and loaded without error.

Backbones checked:
  1. deberta_v3_extra_small_en (Ours)
  2. bert_base_en_uncased
  3. roberta_base_en
  4. deberta_v3_small_en

Usage:
    python3 check_presets.py
"""

import os
import sys

# Set backend before importing keras
os.environ.setdefault("KERAS_BACKEND", "jax")

PRESETS = [
    ("deberta_v3_extra_small_en", "DebertaV3"),
    ("bert_base_en_uncased", "Bert"),
    ("roberta_base_en", "Roberta"),
    ("deberta_v3_small_en", "DebertaV3"),
]


def check_all():
    print("=" * 80)
    print("  PRE-FLIGHT BACKBONE & PRESET CACHE VERIFICATION")
    print("=" * 80)

    try:
        import keras
        import keras_hub as kh
        print(f"Keras version:     {keras.__version__} (backend: {keras.backend.backend()})")
        print(f"KerasHub version:  {kh.__version__}")
    except ImportError as e:
        print(f"Error importing Keras/KerasHub: {e}", file=sys.stderr)
        print("Please install requirements: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    all_ok = True
    results = []

    test_text = "Prompt [TURN] Response A [RESPONSE] Response B"

    for preset, fam in PRESETS:
        print(f"\n--> Checking preset: {preset} ({fam})")
        prep_cls = getattr(kh.models, f"{fam}TextClassifierPreprocessor", None)
        bb_cls = getattr(kh.models, f"{fam}Backbone", None)

        if prep_cls is None or bb_cls is None:
            print(f"  [FAIL] Missing classes for family {fam} in keras_hub.models")
            all_ok = False
            results.append((preset, "FAILED: Class missing", 0))
            continue

        # 1. Preprocessor
        try:
            print(f"  Downloading/loading preprocessor for {preset}...")
            prep = prep_cls.from_preset(preset, sequence_length=128)
            tokens = prep(test_text)
            print(f"  [OK] Preprocessor loaded and tokenized test input")
        except Exception as e:
            print(f"  [FAIL] Preprocessor load failed: {e}")
            all_ok = False
            results.append((preset, f"FAILED preprocessor: {str(e)[:50]}", 0))
            continue

        # 2. Backbone
        try:
            print(f"  Downloading/loading backbone weights for {preset}...")
            bb = bb_cls.from_preset(preset)
            params = bb.count_params()
            print(f"  [OK] Backbone loaded: {params / 1e6:.1f}M parameters")
            results.append((preset, "OK (Cached)", round(params / 1e6, 1)))
        except Exception as e:
            print(f"  [FAIL] Backbone load failed: {e}")
            all_ok = False
            results.append((preset, f"FAILED backbone: {str(e)[:50]}", 0))

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"{'Preset':<32} {'Status':<25} {'Params (M)'}")
    print("-" * 68)
    for p, s, pm in results:
        print(f"{p:<32} {s:<25} {pm}")

    if all_ok:
        print("\n>>> SUCCESS: All 4 backbones verified and cached. Ready for HPC training! <<<")
        sys.exit(0)
    else:
        print("\n>>> ERROR: One or more backbones failed. Check internet access / Kaggle credentials. <<<", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    check_all()
