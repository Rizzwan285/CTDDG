#!/bin/bash
# ============================================================================
# CTDDG — Register Jupyter Kernel for ctddg_env on Bhavani Cluster
# ============================================================================
# Run this ONCE on the login/master node to register the ctddg_env conda
# environment as a Jupyter kernel. After this, Jupyter notebooks will show
# "CTDDG (Python 3.9 / MXNet GPU)" as a kernel option.
#
# Usage:
#   bash cluster/setup_jupyter_kernel.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo "  CTDDG Jupyter Kernel Registration"
echo "============================================"

# 1. Load cluster modules
module load cuda/11.2
module load anaconda3/2022.10

# 2. Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

# 3. Force conda paths (critical for Bhavani cluster)
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 4. Install ipykernel if not present
pip install ipykernel --quiet 2>/dev/null || true

# 5. Register the kernel
python -m ipykernel install --user \
    --name ctddg_env \
    --display-name "CTDDG (Python 3.9 / MXNet GPU)"

echo ""
echo "✅ Kernel registered successfully!"
echo "   Name: ctddg_env"
echo "   Display: CTDDG (Python 3.9 / MXNet GPU)"
echo ""

# 6. Patch the kernel.json to include environment setup
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/ctddg_env"

if [ -d "$KERNEL_DIR" ]; then
    # Create a wrapper script that sets up the full environment
    cat > "$KERNEL_DIR/kernel_launcher.sh" << 'LAUNCHER_EOF'
#!/bin/bash
# CTDDG Kernel Launcher — ensures correct env on GPU compute nodes
module load cuda/11.2 2>/dev/null || true
module load anaconda3/2022.10 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ctddg_env 2>/dev/null || true

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MXNET_CUDNN_LIB_CHECKING=0
export MXNET_ENFORCE_DETERMINISM=0

# Execute the actual Python kernel
exec python -m ipykernel_launcher "$@"
LAUNCHER_EOF
    chmod +x "$KERNEL_DIR/kernel_launcher.sh"

    # Update kernel.json to use the launcher
    python3 -c "
import json, os
kernel_json_path = os.path.join('$KERNEL_DIR', 'kernel.json')
with open(kernel_json_path) as f:
    kj = json.load(f)
kj['argv'] = ['$KERNEL_DIR/kernel_launcher.sh', '-f', '{connection_file}']
kj['env'] = {
    'MXNET_CUDNN_LIB_CHECKING': '0',
    'MXNET_ENFORCE_DETERMINISM': '0'
}
with open(kernel_json_path, 'w') as f:
    json.dump(kj, f, indent=2)
print('✅ kernel.json patched with environment launcher')
"
fi

echo ""
echo "============================================"
echo "  Setup Complete! Next steps:"
echo "  1. Submit a Jupyter job:  sbatch cluster/launch_jupyter.sh"
echo "  2. Check the output file for the SSH tunnel command"
echo "  3. Open Jupyter in your browser and select the CTDDG kernel"
echo "============================================"
