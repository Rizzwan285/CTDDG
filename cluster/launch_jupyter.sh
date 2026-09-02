#!/bin/bash
# ============================================================================
# CTDDG — Slurm Jupyter Notebook Launcher for Bhavani GPU Cluster
# ============================================================================
# Submits a Jupyter notebook server as a Slurm job on a GPU node.
# After submission, check the output file for the SSH tunnel command.
#
# Usage:
#   sbatch cluster/launch_jupyter.sh                    # Default: 1 GPU, normal partition
#   sbatch --gres=gpu:2 cluster/launch_jupyter.sh       # Use both GPUs on the node
#   sbatch -p gpu04 cluster/launch_jupyter.sh           # Specifically target node004 (L40 GPUs)
#   sbatch -p gpu04 --gres=gpu:2 cluster/launch_jupyter.sh  # Both L40 GPUs
# ============================================================================

#SBATCH --job-name=ctddg_jupyter
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/jupyter_%j.log
#SBATCH --error=cluster/jupyter_%j.err

echo "============================================"
echo "  CTDDG Jupyter Server — Slurm Job"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "============================================"

# ─── Environment Setup ──────────────────────────────────────────────
module load cuda/11.2
module load anaconda3/2022.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

# CRITICAL: Force Conda paths on compute nodes
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# MXNet settings
export MXNET_CUDNN_LIB_CHECKING=0
export MXNET_ENFORCE_DETERMINISM=0

# Set project root
export CTDDG_ROOT="$HOME/repo/CTDDG"
cd "$CTDDG_ROOT"

# ─── Verify Environment ─────────────────────────────────────────────
echo ""
echo "Environment verification:"
echo "  Python: $(which python) — $(python --version 2>&1)"
echo "  MXNet:  $(python -c 'import mxnet; print(mxnet.__version__)' 2>/dev/null || echo 'IMPORT FAILED')"
echo "  GPUs:   $(python -c 'import mxnet as mx; print(mx.context.num_gpus())' 2>/dev/null || echo 'CHECK FAILED')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
echo ""

# ─── Pick a free port ────────────────────────────────────────────────
# Use a port in the 8800-8999 range based on job ID to avoid collisions
JUPYTER_PORT=$(( 8800 + (SLURM_JOB_ID % 200) ))

# ─── Print connection instructions ──────────────────────────────────
NODE_HOSTNAME=$(hostname)
echo "============================================"
echo "  📋 CONNECTION INSTRUCTIONS"
echo "============================================"
echo ""
echo "  Run this on your LOCAL machine to create an SSH tunnel:"
echo ""
echo "    ssh -N -L ${JUPYTER_PORT}:${NODE_HOSTNAME}:${JUPYTER_PORT} $(whoami)@bhavani.iitpkd.ac.in"
echo ""
echo "  Then open in your browser:"
echo ""
echo "    http://localhost:${JUPYTER_PORT}"
echo ""
echo "  (Check below for the token URL)"
echo "============================================"
echo ""

# ─── Launch Jupyter ──────────────────────────────────────────────────
jupyter notebook \
    --no-browser \
    --port="${JUPYTER_PORT}" \
    --ip="0.0.0.0" \
    --NotebookApp.token="" \
    --NotebookApp.password="" \
    --notebook-dir="$CTDDG_ROOT"
