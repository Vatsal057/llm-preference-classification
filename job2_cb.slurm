#!/bin/bash
#SBATCH --job-name=llm_pref_job2_cb
#SBATCH --output=slurm_job2_cb_%j.out
#SBATCH --error=slurm_job2_cb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # 2 GPUs. For a single-GPU node use gpu:1 and USE_MULTI_GPU=0.
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# ==============================================================================
# Job 2 of 2: Block C (4 backbones x 3 seeds, 12 runs) + Block B (ablation, 6 runs)
#
# Estimated: ~20.8 h single-GPU, ~10.5 h dual-GPU.
# All blocks are enabled: the runner skips Job 1's completed runs and trains only
# what is missing. Section 10 then recompiles paper_results.json from ALL runs
# (A, B, C, D) together, which is why the strict gate can be enforced here.
#
# SESSION_BUDGET_H must stay BELOW the --time wall limit so the analysis in
# section 10 actually runs. See the note in job1_ad.slurm.
# ==============================================================================

set -u

echo "=== Job 2 (Blocks C & B) starting at $(date) on $(hostname) ==="
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

export KERAS_BACKEND="jax"
export SESSION_BUDGET_H=22.0       # < 24 h wall limit, leaves room for section 10
export USE_MULTI_GPU=1             # 0 for single-GPU; keep BATCH_SIZE divisible by device count

# Enable everything; already-completed runs from Job 1 are detected and reused.
export RUN_BLOCK_A=1
export RUN_BLOCK_B=1
export RUN_BLOCK_C=1
export RUN_BLOCK_D=1
export RUN_BLOCK_E=0

mkdir -p results predictions tokcache failed

python3 check_presets.py || { echo "Pre-flight check FAILED -- fix presets/credentials first."; exit 1; }

echo "Running Blocks C & B (plus any Block A/D runs still missing) ..."
python3 notebook_src.py
train_status=$?

echo "=== Training stage done (exit ${train_status}). Enforcing strict verification ==="
python3 reproduce_tables.py --strict
strict_status=$?

if [ ${strict_status} -ne 0 ]; then
    echo "############################################################"
    echo "STRICT VERIFICATION FAILED (exit ${strict_status})."
    echo "Some runs are missing or under-seeded. Inspect failed/ and the"
    echo "run counts printed above, then resubmit -- completed runs are reused."
    echo "Do NOT copy numbers into the manuscript until this passes."
    echo "############################################################"
    exit ${strict_status}
fi

echo "=== Job 2 finished at $(date). All 42 runs present, strict verification PASSED ==="
echo "Copy the regenerated tables from results/paper_results.json into the manuscript."
exit 0
