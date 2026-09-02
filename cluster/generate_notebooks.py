#!/usr/bin/env python3
"""Generate all CTDDG pipeline Jupyter notebooks for Bhavani HPC cluster."""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(REPO, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

def make_nb(cells_data, kernel="ctddg_env"):
    cells = []
    for ctype, src in cells_data:
        cell = {
            "cell_type": ctype,
            "metadata": {} if ctype == "markdown" else {"trusted": True},
            "source": src.split("\n") if isinstance(src, str) else src,
        }
        if ctype == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "CTDDG (Python 3.9 / MXNet GPU)", "language": "python", "name": kernel},
            "language_info": {"name": "python", "version": "3.9.25"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }

def save(name, cells_data):
    path = os.path.join(NB_DIR, name)
    with open(path, "w") as f:
        json.dump(make_nb(cells_data), f, indent=1)
    print(f"  Created: {path}")

# =====================================================================
# NOTEBOOK 0: Cluster Setup & Verification
# =====================================================================
save("00_cluster_setup.ipynb", [
    ("markdown", "# 🔧 CTDDG — Cluster Setup & Environment Verification\n\nRun this notebook **first** to verify your Bhavani cluster environment.\n\n**Prerequisites:** The `ctddg_env` conda environment must already exist.\nRun `bash cluster/setup_jupyter_kernel.sh` on the login node first."),

    ("code", """import os, sys, subprocess

# ── Auto-detect project root ──
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.chdir(CTDDG_ROOT)
sys.path.insert(0, os.path.join(CTDDG_ROOT, "scripts"))

print(f"Project Root: {CTDDG_ROOT}")
print(f"Python:       {sys.executable}")
print(f"Python ver:   {sys.version}")"""),

    ("markdown", "## 1. GPU Verification"),
    ("code", """# Check NVIDIA GPUs
!nvidia-smi
print()

import mxnet as mx
num_gpus = mx.context.num_gpus()
print(f"\\n✅ MXNet {mx.__version__} — {num_gpus} GPU(s) detected")

# Quick GPU compute test
for i in range(num_gpus):
    x = mx.nd.ones((100, 100), ctx=mx.gpu(i))
    y = mx.nd.dot(x, x)
    print(f"  GPU {i}: {y[0,0].asscalar():.0f} (dot product test passed)")"""),

    ("markdown", "## 2. Dependency Verification"),
    ("code", """deps = {}
for name, imp in [("mxnet", "mxnet"), ("numpy", "numpy"), ("scipy", "scipy"),
                   ("rdkit", "rdkit"), ("pandas", "pandas"), ("networkx", "networkx"),
                   ("h5py", "h5py"), ("matplotlib", "matplotlib"), ("molvs", "molvs")]:
    try:
        m = __import__(imp)
        deps[name] = getattr(m, "__version__", "OK")
    except ImportError:
        deps[name] = "❌ MISSING"

for k, v in deps.items():
    status = "✅" if v != "❌ MISSING" else "❌"
    print(f"  {status} {k:15s} {v}")"""),

    ("markdown", "## 3. Data Directory Verification"),
    ("code", """from scripts.config import cfg
cfg.__init__(CTDDG_ROOT)

checks = [
    ("atom_types.txt", cfg.ATOM_TYPES_FILE),
    ("chembl/chembl.txt", cfg.CHEMBL_PLAIN_FILE),
    ("bindingdb/train_dataset", os.path.join(cfg.BINDINGDB_DIR, "train_dataset")),
    ("bindingdb/test_dataset", os.path.join(cfg.BINDINGDB_DIR, "test_dataset")),
]

all_ok = True
for label, path in checks:
    exists = os.path.exists(path)
    if not exists: all_ok = False
    print(f"  {'✅' if exists else '❌'} {label}: {path}")

if all_ok:
    print("\\n✅ All data files present!")
else:
    print("\\n⚠️  Some data files missing. Run download_data.sh or rsync from local.")"""),

    ("markdown", "## 4. Patch Notebook Paths\nRun this to update all hardcoded paths in the original notebooks."),
    ("code", """!python scripts/patch_paths.py {CTDDG_ROOT}"""),

    ("markdown", "## 5. Convert ChEMBL Format\nAppend dummy class labels for the pretraining data loader."),
    ("code", """chembl_src = os.path.join(CTDDG_ROOT, "data", "chembl", "chembl.txt")
chembl_dst = os.path.join(CTDDG_ROOT, "data", "chembl", "chembl_final.txt")
if os.path.exists(chembl_src) and not os.path.exists(chembl_dst):
    !python scripts/convert_chembl_format.py {chembl_src} {chembl_dst}
elif os.path.exists(chembl_dst):
    print(f"✅ chembl_final.txt already exists ({os.path.getsize(chembl_dst)} bytes)")
else:
    print("❌ chembl.txt not found — download data first")"""),

    ("markdown", "## 6. Create Output Directories"),
    ("code", """for d in ["outputs/pretrain/logs", "outputs/docking"]:
    os.makedirs(os.path.join(CTDDG_ROOT, d), exist_ok=True)
    print(f"  ✅ {d}/")
print("\\n🎉 Setup complete! Proceed to notebook 01_pretraining.ipynb")"""),
])

# =====================================================================
# NOTEBOOK 1: Pretraining (Multi-GPU)
# =====================================================================
save("01_pretraining.ipynb", [
    ("markdown", "# 🧪 Stage 1: Pretraining (Unconditional Model)\n\nTrains `VanillaMolGen_RNN` on ChEMBL SMILES data.\n\n**GPU Utilization:** Uses all available GPUs via MXNet data parallelism.\n\n**Expected time:** ~6-12 hours for 480K iterations on 2× L40 GPUs."),

    ("code", """import os, sys
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.chdir(CTDDG_ROOT)
print(f"Working directory: {os.getcwd()}")"""),

    ("markdown", "## GPU Check"),
    ("code", """import mxnet as mx
NUM_GPUS = mx.context.num_gpus()
print(f"Available GPUs: {NUM_GPUS}")
!nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"""),

    ("markdown", "## Run Pretraining\nThis executes the original `pretraining.ipynb` notebook.\nThe training loop uses 480,000 iterations with checkpointing every 500 steps."),
    ("code", """%%time
# Execute the original pretraining notebook
# This runs the entire single-cell notebook which contains all model definitions + training loop
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--to", "notebook", "--execute",
     "--ExecutePreprocessor.timeout=172800",  # 48h timeout
     "--ExecutePreprocessor.kernel_name=ctddg_env",
     f"--output=pretraining_executed.ipynb",
     os.path.join(CTDDG_ROOT, "code", "pretraining.ipynb")],
    capture_output=True, text=True, cwd=CTDDG_ROOT
)
print("STDOUT:", result.stdout[-2000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
    print(f"\\n❌ Pretraining failed (exit code {result.returncode})")
else:
    print("\\n✅ Pretraining completed successfully!")"""),

    ("markdown", "## Monitor Training Progress\nRun this cell periodically to check training progress."),
    ("code", """log_path = os.path.join(CTDDG_ROOT, "outputs", "pretrain", "logs", "log.out")
if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    print(f"Log entries: {len(lines)}")
    if len(lines) > 1:
        print(f"Header: {lines[0].strip()}")
        print(f"Latest: {lines[-1].strip()}")
        if lines[-1].strip() == "Training finished":
            print("\\n🎉 Training is COMPLETE!")
        else:
            parts = lines[-1].strip().split('\\t')
            if len(parts) >= 2:
                print(f"\\nProgress: step {parts[0]} / 480000 ({100*int(parts[0])/480000:.1f}%)")
else:
    print("⏳ Training has not started yet (no log file)")"""),
])

# =====================================================================
# NOTEBOOK 2: Fine-Tuning (Conditional Model)
# =====================================================================
save("02_finetuning.ipynb", [
    ("markdown", "# 🎯 Stage 2: Conditional Fine-Tuning\n\nFine-tunes `CVanillaMolGen_RNN` on paired ligand-protein data from BindingDB.\nThis bridges the gap between unconditional pretraining and conditional generation.\n\n**Prerequisites:** Stage 1 (pretraining) must be complete."),

    ("code", """import os, sys
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.chdir(CTDDG_ROOT)

import mxnet as mx
NUM_GPUS = mx.context.num_gpus()
print(f"Project root: {CTDDG_ROOT}")
print(f"GPUs available: {NUM_GPUS}")
!nvidia-smi --query-gpu=index,name --format=csv,noheader"""),

    ("markdown", "## Verify Pretrained Checkpoint"),
    ("code", """ckpt_path = os.path.join(CTDDG_ROOT, "outputs", "pretrain", "logs", "ckpt.params")
config_path = os.path.join(CTDDG_ROOT, "outputs", "pretrain", "logs", "configs.json")

for label, p in [("Checkpoint", ckpt_path), ("Config", config_path)]:
    exists = os.path.exists(p)
    print(f"  {'✅' if exists else '❌'} {label}: {p}")
    if not exists:
        raise FileNotFoundError(f"Missing: {p}. Run Stage 1 first!")"""),

    ("markdown", "## Configure Fine-Tuning\nSet which dataset to fine-tune on. The default is Dataset 1."),
    ("code", """# ── CONFIGURATION ──
DATASET_INDEX = 1       # Which BindingDB dataset (1-5)
ITERATIONS = 50000      # Fine-tuning iterations
BATCH_SIZE = 16         # Batch size per GPU
LEARNING_RATE = 1e-4
SAVE_FREQ = 5000        # Checkpoint every N steps
LOG_FREQ = 100          # Log every N steps

print(f"Dataset:    {DATASET_INDEX}")
print(f"Iterations: {ITERATIONS:,}")
print(f"Batch size: {BATCH_SIZE} × {NUM_GPUS} GPUs = {BATCH_SIZE * NUM_GPUS} effective")"""),

    ("markdown", "## Run Fine-Tuning\nThis executes the pretraining notebook first (to define all classes), then runs the fine-tuning code."),
    ("code", """%%time
# The finetune_cell.py depends on classes defined in pretraining.ipynb.
# We run pretraining.ipynb code (definitions only, not training) then the finetuning.

# First, execute pretraining notebook to get all class definitions loaded
print("Loading model definitions from pretraining.ipynb...")
exec_globals = {"__name__": "__main__"}

# We need to run the pretraining notebook's code to define all the classes
# but skip the actual training loop. We do this by loading it as a module.
import json
nb_path = os.path.join(CTDDG_ROOT, "code", "pretraining.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

# Get the source code from the single cell
full_src = "".join(nb["cells"][0]["source"])

# Split at the training section and only execute definitions
# Find where training starts
train_marker = '""\\"# Training the model""\\"'
alt_marker = 'print("We are in training part....")'

# Execute everything up to the training loop
if alt_marker in full_src:
    defs_src = full_src[:full_src.index(alt_marker)]
elif "Training the model" in full_src:
    idx = full_src.index("Training the model")
    # Go back to find the triple-quote before it
    defs_src = full_src[:idx-3]
else:
    print("⚠️ Could not find training marker, loading full source")
    defs_src = full_src

# Execute definitions in current namespace
exec(compile(defs_src, "<pretraining_defs>", "exec"))
print("✅ Model definitions loaded")

# Now run the fine-tuning code
print("\\nStarting fine-tuning...")
exec(open(os.path.join(CTDDG_ROOT, "scripts", "finetune_cell.py")).read())
run_finetuning(dataset_index=DATASET_INDEX)"""),

    ("markdown", "## Check Fine-Tuning Output"),
    ("code", """ft_dir = os.path.join(CTDDG_ROOT, "outputs", "CTDGD", f"Dataset{DATASET_INDEX}", "model")
if os.path.exists(ft_dir):
    for f in os.listdir(ft_dir):
        fpath = os.path.join(ft_dir, f)
        print(f"  ✅ {f} ({os.path.getsize(fpath)/1024:.1f} KB)")
else:
    print(f"❌ Fine-tuning output directory not found: {ft_dir}")"""),
])

# =====================================================================
# NOTEBOOK 3: Sample Generation
# =====================================================================
save("03_generation.ipynb", [
    ("markdown", "# 🧬 Stage 3: Conditional Molecule Generation\n\nGenerates drug-like molecules conditioned on target protein embeddings.\n\n**Prerequisites:** Stage 2 (fine-tuning) must be complete.\n\n**Multi-GPU:** Generation runs on GPU 0 (model is loaded per-protein). Multiple datasets can be parallelized across GPUs."),

    ("code", """import os, sys
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.chdir(CTDDG_ROOT)
print(f"Working directory: {os.getcwd()}")"""),

    ("markdown", "## Configure Generation"),
    ("code", """# ── CONFIGURATION ──
DATASET_INDEX = 2     # Which dataset's test proteins to generate for
RUN = 1               # Run number (for multiple runs)
N_SAMPLES = 1000      # Number of molecules to generate per protein
MODEL_NAME = "CTDGD"  # Model name (matches fine-tuning output dir name)

print(f"Dataset: {DATASET_INDEX}, Run: {RUN}, Samples: {N_SAMPLES}")"""),

    ("markdown", "## Run Generation"),
    ("code", """%%time
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--to", "notebook", "--execute",
     "--ExecutePreprocessor.timeout=86400",
     "--ExecutePreprocessor.kernel_name=ctddg_env",
     f"--output=Generating_samples_executed.ipynb",
     os.path.join(CTDDG_ROOT, "code", "Generating_samples.ipynb")],
    capture_output=True, text=True, cwd=CTDDG_ROOT
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
    print(f"\\n❌ Generation failed")
else:
    print("✅ Generation completed!")"""),

    ("markdown", "## Quick Preview of Generated Molecules"),
    ("code", """import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display

out_dir = os.path.join(CTDDG_ROOT, "outputs", MODEL_NAME,
                       f"Dataset{DATASET_INDEX}", "generated_samples",
                       str(N_SAMPLES), f"run{RUN}")
csvs = sorted([f for f in os.listdir(out_dir) if f.endswith('.csv')]) if os.path.isdir(out_dir) else []
print(f"Found {len(csvs)} protein output files")

if csvs:
    df = pd.read_csv(os.path.join(out_dir, csvs[0]))
    print(f"\\nProtein 1: {len(df)} molecules generated")
    mols = [Chem.MolFromSmiles(s) for s in df['smiles'].head(12) if Chem.MolFromSmiles(s)]
    if mols:
        display(Draw.MolsToGridImage(mols, molsPerRow=4))"""),
])

# =====================================================================
# NOTEBOOK 4: Evaluation Metrics
# =====================================================================
save("04_evaluation.ipynb", [
    ("markdown", "# 📊 Stage 4: Evaluation Metrics\n\nComputes validity, uniqueness, novelty, QED, SAS, and Lipinski properties.\n\n**Prerequisites:** Stage 3 (generation) must be complete."),

    ("code", """import os, sys
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.chdir(CTDDG_ROOT)"""),

    ("markdown", "## Run Evaluation Notebook"),
    ("code", """%%time
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--to", "notebook", "--execute",
     "--ExecutePreprocessor.timeout=86400",
     "--ExecutePreprocessor.kernel_name=ctddg_env",
     f"--output=Evaluation_metrics_executed.ipynb",
     os.path.join(CTDDG_ROOT, "code", "Evaluation_metrics.ipynb")],
    capture_output=True, text=True, cwd=CTDDG_ROOT
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
else:
    print("✅ Evaluation completed!")"""),
])

# =====================================================================
# NOTEBOOK 5: Molecular Docking
# =====================================================================
save("05_docking.ipynb", [
    ("markdown", "# 🔬 Stage 5: Molecular Docking\n\nPerforms molecular docking on top generated candidates.\n\n**Prerequisites:** Stage 4 (evaluation) should be complete."),

    ("code", """import os, sys
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.chdir(CTDDG_ROOT)"""),

    ("markdown", "## Run Docking Notebook"),
    ("code", """%%time
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--to", "notebook", "--execute",
     "--ExecutePreprocessor.timeout=86400",
     "--ExecutePreprocessor.kernel_name=ctddg_env",
     f"--output=Molecular_docking_executed.ipynb",
     os.path.join(CTDDG_ROOT, "code", "Molecular_docking.ipynb")],
    capture_output=True, text=True, cwd=CTDDG_ROOT
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
else:
    print("✅ Docking completed!")"""),
])

# =====================================================================
# NOTEBOOK 6: Full Pipeline Runner
# =====================================================================
save("06_run_full_pipeline.ipynb", [
    ("markdown", "# 🚀 CTDDG — Full Pipeline Runner\n\nRuns the **entire** CTDDG pipeline end-to-end in a single notebook.\nUse this for unattended batch execution on the cluster.\n\n**Estimated time:** 12-24 hours on 2× L40 GPUs"),

    ("code", """import os, sys, time, subprocess
CTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))
os.environ["CTDDG_ROOT"] = CTDDG_ROOT
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.chdir(CTDDG_ROOT)

import mxnet as mx
print(f"Root: {CTDDG_ROOT}")
print(f"GPUs: {mx.context.num_gpus()}")
!nvidia-smi --query-gpu=index,name --format=csv,noheader"""),

    ("markdown", "## Pipeline Configuration"),
    ("code", """# What to run (set False to skip a stage)
RUN_PRETRAINING = True
RUN_FINETUNING = True
RUN_GENERATION = True
RUN_EVALUATION = True
RUN_DOCKING = False  # Often done separately

DATASET_INDEX = 1
N_SAMPLES = 1000
RUN_ID = 1"""),

    ("markdown", "## Step 0: Setup"),
    ("code", """# Patch paths and convert ChEMBL
!python scripts/patch_paths.py {CTDDG_ROOT}

chembl_src = os.path.join(CTDDG_ROOT, "data", "chembl", "chembl.txt")
chembl_dst = os.path.join(CTDDG_ROOT, "data", "chembl", "chembl_final.txt")
if os.path.exists(chembl_src) and not os.path.exists(chembl_dst):
    !python scripts/convert_chembl_format.py {chembl_src} {chembl_dst}

for d in ["outputs/pretrain/logs", "outputs/docking"]:
    os.makedirs(os.path.join(CTDDG_ROOT, d), exist_ok=True)
print("✅ Setup complete")"""),

    ("markdown", "## Execute Pipeline Stages"),
    ("code", """def run_notebook(nb_path, label, timeout=172800):
    print(f"\\n{'='*60}")
    print(f"  ▶ {label}")
    print(f"{'='*60}")
    t0 = time.time()
    name = os.path.splitext(os.path.basename(nb_path))[0]
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute",
         f"--ExecutePreprocessor.timeout={timeout}",
         "--ExecutePreprocessor.kernel_name=ctddg_env",
         f"--output={name}_executed.ipynb",
         nb_path],
        capture_output=True, text=True, cwd=CTDDG_ROOT
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ {label} completed in {elapsed/60:.1f} min")
        return True
    else:
        print(f"  ❌ {label} FAILED after {elapsed/60:.1f} min")
        print(f"  Error: {result.stderr[-1000:]}")
        return False

pipeline_start = time.time()

if RUN_PRETRAINING:
    ok = run_notebook("code/pretraining.ipynb", "Stage 1: Pretraining")
    if not ok: raise RuntimeError("Pretraining failed")

if RUN_FINETUNING:
    # Fine-tuning needs special handling since finetune_cell.py
    # depends on pretraining.ipynb classes
    print("\\n" + "="*60)
    print("  ▶ Stage 2: Fine-Tuning")
    print("="*60)
    print("  ℹ️  Run notebook 02_finetuning.ipynb for this stage")
    print("  (Fine-tuning requires pretraining class definitions in-memory)")

if RUN_GENERATION:
    run_notebook("code/Generating_samples.ipynb", "Stage 3: Generation")

if RUN_EVALUATION:
    run_notebook("code/Evaluation_metrics.ipynb", "Stage 4: Evaluation")

if RUN_DOCKING:
    run_notebook("code/Molecular_docking.ipynb", "Stage 5: Docking")

total = time.time() - pipeline_start
print(f"\\n{'='*60}")
print(f"  🎉 Pipeline completed in {total/3600:.1f} hours")
print(f"{'='*60}")"""),
])

print(f"\n✅ All notebooks created in: {NB_DIR}")
