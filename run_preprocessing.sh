#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=CTDDG-preprocess
#SBATCH --output=preprocessing_%j.out
#SBATCH --error=preprocessing_%j.err

module load cuda/11.2
module load anaconda3/2022.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MXNET_CUDNN_LIB_CHECKING=0

cd ~/repo/CTDDG

echo "========================================"
echo "CTDDG preprocessing started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

python -c "import mxnet as mx; print('MXNet:', mx.__version__); print('GPUs:', mx.context.num_gpus())"

jupyter nbconvert --to notebook --execute \
    code/data_preprocessing_1.ipynb \
    --output data_preprocessing_1_executed.ipynb \
    2>&1 | grep --line-buffered -v "Needleman-Wunsch"

NBEXIT=${PIPESTATUS[0]}

if [ $NBEXIT -eq 0 ]; then
    echo "========================================"
    echo "CTDDG preprocessing finished successfully"
    echo "Date: $(date)"
    echo "========================================"
else
    echo "========================================"
    echo "CTDDG preprocessing FAILED"
    echo "nbconvert exit code: $NBEXIT"
    echo "Date: $(date)"
    echo "========================================"
    exit $NBEXIT
fi