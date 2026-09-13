#!/bin/bash
#SBATCH --job-name=llm_pref_job1_ad
#SBATCH --output=slurm_job1_ad_%j.out
#SBATCH --error=slurm_job1_ad_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # 2 GPUs. For a single-GPU node use gpu:1 and USE_MULTI_GPU=0.
#SBATCH --time=20:00:00
#SBATCH --mem=32G

# ==============================================================================
# Job 1 of 2: Block A (main factorial, 15 runs) + Block D (data efficiency, 9 runs)
#
# Estimated: ~16.7 h single-GPU, ~8.5 h dual-GPU.
# Run this FIRST. Every table except the backbone comparison depends on Block A,
# and Block D's 12,000-row anchor row is Block A's swapaug_siam run.
#
# SESSION_BUDGET_H must stay BELOW the --time wall limit. The runner stops
# launching new runs once the budget is reached and then proceeds to the analysis
# in section 10, which is what writes the tables. If SLURM kills the job first,
# the completed runs survive on disk but no tables are produced.
# ==============================================================================

set -u

echo "=== Job 1 (Blocks A & D) starting at $(date) on $(hostname) ==="
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

export KERAS_BACKEND="jax"
export SESSION_BUDGET_H=18.0       # < 20 h wall limit, leaves room for section 10
export USE_MULTI_GPU=1             # 0 for single-GPU; keep BATCH_SIZE divisible by device count

export RUN_BLOCK_A=1
export RUN_BLOCK_B=0
export RUN_BLOCK_C=0
export RUN_BLOCK_D=1
export RUN_BLOCK_E=0

# tokcache is the directory name the code actually uses (CFG/CACHE_DIR), not "cache".
mkdir -p results predictions tokcache failed

# Pre-flight: confirm all backbones resolve before spending GPU time.
python3 check_presets.py || { echo "Pre-flight check FAILED -- fix presets/credentials first."; exit 1; }

echo "Running Blocks A & D ..."
python3 notebook_src.py
train_status=$?

echo "=== Job 1 finished at $(date), notebook exit status ${train_status} ==="
echo "Run job2_cb.slurm next. Completed runs here are detected and skipped."
echo "Do NOT run 'reproduce_tables.py --strict' yet: Blocks B and C are still missing,"
echo "so the strict gate is expected to fail until Job 2 completes."
exit ${train_status}
