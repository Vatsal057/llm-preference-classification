#!/bin/bash
#SBATCH --job-name=llm_pref_all
#SBATCH --output=slurm_all_%j.out
#SBATCH --error=slurm_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # 2 GPUs. For a single-GPU node use gpu:1 and USE_MULTI_GPU=0.
#SBATCH --time=40:00:00
#SBATCH --mem=32G

# ==============================================================================
# Single-allocation alternative to job1_ad.slurm + job2_cb.slurm.
# Blocks A, B, C, D = 42 runs. Estimated ~37.5 h single-GPU, ~19 h dual-GPU.
#
# Check your partition's wall-clock cap before submitting: 40 h with 2 GPUs is
# rejected outright on many partitions and can queue for a long time. If the cap
# is lower, use the two-job route instead -- it is resumable and produces the
# same result.
#
# SESSION_BUDGET_H must stay BELOW the --time wall limit so the analysis in
# section 10 runs and the tables are written. If SLURM kills the job first, the
# completed runs survive on disk but no tables are produced; resubmitting picks
# up where it stopped.
# ==============================================================================

set -u

echo "=== Full suite (Blocks A, B, C, D) starting at $(date) on $(hostname) ==="
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

export KERAS_BACKEND="jax"
export SESSION_BUDGET_H=37.0       # < 40 h wall limit, leaves room for section 10
export USE_MULTI_GPU=1             # 0 for single-GPU; keep BATCH_SIZE divisible by device count

export RUN_BLOCK_A=1
export RUN_BLOCK_B=1
export RUN_BLOCK_C=1
export RUN_BLOCK_D=1
export RUN_BLOCK_E=0

mkdir -p results predictions tokcache failed

python3 check_presets.py || { echo "Pre-flight check FAILED -- fix presets/credentials first."; exit 1; }

echo "Running all 42 runs ..."
python3 notebook_src.py
train_status=$?

echo "=== Training stage done (exit ${train_status}). Enforcing strict verification ==="
python3 reproduce_tables.py --strict
strict_status=$?

if [ ${strict_status} -ne 0 ]; then
    echo "############################################################"
    echo "STRICT VERIFICATION FAILED (exit ${strict_status})."
    echo "Runs are missing or under-seeded -- most likely the session budget"
    echo "or the wall limit was reached. Inspect failed/ and resubmit;"
    echo "completed runs are reused, not recomputed."
    echo "Do NOT copy numbers into the manuscript until this passes."
    echo "############################################################"
    exit ${strict_status}
fi

echo "=== Finished at $(date). All 42 runs present, strict verification PASSED ==="
exit 0
