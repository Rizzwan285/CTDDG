#!/bin/bash
# ============================================================================
# CTDDG — Headless Full Pipeline via Slurm (no Jupyter UI needed)
# ============================================================================
# Runs all pipeline notebooks sequentially on a GPU node via nbconvert.
# Use this when you don't need an interactive Jupyter session.
#
# Usage:
#   sbatch cluster/run_pipeline_batch.sh
#   sbatch -p gpu04 --gres=gpu:2 cluster/run_pipeline_batch.sh  # L40 GPUs
# ============================================================================

#SBATCH --job-name=ctddg_pipeline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=cluster/pipeline_%j.log
#SBATCH --error=cluster/pipeline_%j.err

set -euo pipefail

echo "============================================"
echo "  CTDDG Full Pipeline — Batch Execution"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "============================================"

# ─── Environment ─────────────────────────────────────────────────────
module load cuda/11.2
module load anaconda3/2022.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MXNET_CUDNN_LIB_CHECKING=0
export CTDDG_ROOT="$HOME/repo/CTDDG"

cd "$CTDDG_ROOT"

echo "Python: $(which python) — $(python --version 2>&1)"
echo "MXNet:  $(python -c 'import mxnet; print(mxnet.__version__)' 2>/dev/null)"
echo "GPUs:   $(python -c 'import mxnet as mx; print(mx.context.num_gpus())' 2>/dev/null)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ─── Setup ───────────────────────────────────────────────────────────
echo ""
echo "▶ Step 0: Setup & Path Patching"
python scripts/patch_paths.py "$CTDDG_ROOT"

if [ -f data/chembl/chembl.txt ] && [ ! -f data/chembl/chembl_final.txt ]; then
    python scripts/convert_chembl_format.py data/chembl/chembl.txt data/chembl/chembl_final.txt
fi

mkdir -p outputs/pretrain/logs outputs/docking

run_notebook() {
    local NB_PATH="$1"
    local LABEL="$2"
    local TIMEOUT="${3:-172800}"
    
    echo ""
    echo "============================================"
    echo "  ▶ $LABEL"
    echo "  Started: $(date)"
    echo "============================================"
    
    local NB_NAME
    NB_NAME=$(basename "$NB_PATH" .ipynb)
    
    python -m jupyter nbconvert \
        --to notebook --execute \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        --ExecutePreprocessor.kernel_name=ctddg_env \
        --output="${NB_NAME}_executed.ipynb" \
        "$NB_PATH"
    
    echo "  ✅ $LABEL completed at $(date)"
}

# ─── Pipeline ────────────────────────────────────────────────────────
run_notebook "code/pretraining.ipynb"         "Stage 1: Pretraining"    172800
# Note: Fine-tuning requires pretraining classes. Run 02_finetuning.ipynb interactively.
run_notebook "code/Generating_samples.ipynb"  "Stage 3: Generation"     86400
run_notebook "code/Evaluation_metrics.ipynb"  "Stage 4: Evaluation"     86400

echo ""
echo "============================================"
echo "  🎉 Pipeline Complete!"
echo "  Finished: $(date)"
echo "============================================"
