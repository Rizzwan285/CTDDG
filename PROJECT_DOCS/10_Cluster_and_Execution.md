# 10 — Cluster and Execution (Bhavani)

Everything here is drawn from `cluster/*.sh`, the Slurm job logs in `cluster/`, the verified
`ctddg_env` installation, and `WEEK 1 PROGRESS - TASK 2.md`. Commands marked ✅ **verified**
have demonstrably run on this cluster.

---

## 1. The cluster

| Item | Value | Source |
|---|---|---|
| Login node | `bhavani.iitpkd.ac.in` | `launch_jupyter.sh`, Week-1 notes |
| GPU nodes seen | `node001`, `node002`, `node003`, `node004` | job logs 11943/11968/11969/11970/11973, 11849 |
| Partitions | `normal` (default), `gpu01`–`gpu04` | `launch_jupyter_preferred.sh` |
| GPUs per node | **2 × NVIDIA A30, 24576 MiB** | every `cluster/jupyter_*.log` |
| Driver | 530.30.02 (driver CUDA 12.1) | Week-1 notes §3 |
| Scheduler | Slurm | job scripts |
| Modules used | `cuda/11.2`, `anaconda3/2022.10` | all job scripts |
| Conda | 23.1.0 | Week-1 notes §3 |

⚠ **No L40 GPUs exist in any observed log**, despite `notebooks/01_pretraining.ipynb`,
`notebooks/06_run_full_pipeline.ipynb` and comments in `launch_jupyter.sh` /
`run_pipeline_batch.sh` referring to "L40 GPUs" on `gpu04`/`node004`. Job 11849 ran on node004
and reported 2 GPUs; its log does not print the model name. **Whether node004 has L40s is NOT
DETERMINED FROM AVAILABLE SOURCES.** Every node whose GPU model *is* logged shows A30.

## 2. The environment

| Item | Value |
|---|---|
| Name | `ctddg_env` |
| Prefix | `/home/muhamed/.conda/envs/ctddg_env` |
| Python | **3.9.25** |
| MXNet | **1.9.1** (`mxnet-cu112`, built against CUDA 11.2) |

Verified installed (queried directly):

| Package | Version |
|---|---|
| mxnet | 1.9.1 |
| rdkit | 2022.09.5 |
| numpy | 1.23.5 |
| scipy | 1.10.1 |
| pandas | 1.5.3 |
| networkx | 3.2.1 |
| h5py | 3.14.0 |
| molvs | 0.1.1 |
| scikit-learn | 0.24.2 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| bio_embeddings | installed |
| torch | 1.9.1+cu102 |
| EMBOSS `needle` | `$CONDA_PREFIX/bin/needle` |

**Absent** (all docking-related): `vina`, `meeko`, `prolif`, `MDAnalysis`, `pymol`,
`openbabel`, `py3Dmol`.

### Why CUDA 11.2 and not 12.1
The driver's "CUDA 12.1" is the *maximum* it supports. `mxnet-cu112==1.9.1` is compiled
against CUDA 11.2 and MXNet 1.9.1 has no newer build, so `module load cuda/11.2` is mandatory
(Week-1 notes §3).

### ⚠ The `PATH` trap — the single most important operational fact

`conda activate` alone is **not sufficient** on Bhavani's compute nodes: the cluster's system
Anaconda shadows the environment, and Python silently resolves to the wrong interpreter. The
symptoms are misleading — `No module named 'mxnet'` when MXNet is installed perfectly.

**Every job script must contain this block** ✅ verified:

```bash
module load cuda/11.2
module load anaconda3/2022.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

# CRITICAL: force Conda's Python and libraries ahead of the system Anaconda
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export MXNET_CUDNN_LIB_CHECKING=0     # silences a harmless cuDNN version warning
```

Then always verify before trusting anything:
```bash
which python      # expect /home/muhamed/.conda/envs/ctddg_env/bin/python
python --version  # expect Python 3.9.25
```

`source .../conda.sh` is used rather than `conda init` so the block is self-contained and safe
to repeat in every job script.

### Rebuilding the environment
```bash
module load anaconda3/2022.10
conda create -n ctddg_env python=3.9 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env
conda install -c bioconda emboss -y
pip install mxnet-cu112==1.9.1
pip install -r requirements.txt
pip install bio-embeddings[all]
```
Snapshots of the known-good state were taken (Week-1 notes §9): `~/ctddg_env.yml`,
`~/ctddg_pip_freeze.txt`.

---

## 3. The six things people confuse

| Concept | What it is | Dies when… |
|---|---|---|
| **SSH connection** | Your terminal on the login node | you close the terminal / laptop sleeps |
| **SSH tunnel** | `ssh -N -L` forwarding a local port to a remote port. Carries no computation | the same |
| **Jupyter server** | A process on the **GPU node** serving notebooks over HTTP | its Slurm job ends |
| **Jupyter kernel** | The Python process the server spawns; holds your variables | the server dies, or you restart the kernel |
| **Slurm job** | The scheduler's reservation of node + GPU + time | the time limit expires, or `scancel` |
| **Training process** | The Python computation inside a kernel or a batch script | its parent process dies |

**The critical implication:** closing your laptop kills the SSH tunnel — but the tunnel is
just a pipe. The Jupyter server, the kernel and the training all keep running on the GPU node
because they belong to the **Slurm job**, not to your session. Reconnect the tunnel later and
everything is still there.

**What actually kills long jobs:** the Slurm `--time` limit. Job 11973 was terminated by
exactly this after 5 days (`slurmstepd: error: *** JOB 11973 ON node003 CANCELLED AT
2026-09-07T22:50:00 DUE TO TIME LIMIT ***`) — harmlessly, because the 46-hour training had
finished 3 days earlier.

---

## 4. Path from laptop to GPU

```
Laptop (VS Code Remote-SSH, or a plain terminal)
   │
   │  ssh muhamed@bhavani.iitpkd.ac.in
   ▼
Login / master node        ← NO GPU HERE. mx.context.num_gpus() returns 0.
   │                          Only edit files, submit jobs, run light scripts.
   │  sbatch cluster/launch_jupyter_preferred.sh
   ▼
Slurm scheduler
   │  allocates node + 2 GPUs + walltime
   ▼
GPU node (node001…node004, 2 × A30)
   │  module load · conda activate · export PATH/LD_LIBRARY_PATH
   ▼
Jupyter server on port 8800 + (JOBID % 200)
   │       ▲
   │       └── ssh -N -L <port>:<node>:<port> muhamed@bhavani.iitpkd.ac.in   (from laptop)
   ▼
kernel "ctddg_env"
   │
   ▼
training process  →  outputs/pretrain/logs/
```

---

## 5. Verified working commands

### 5.1 One-time: register the Jupyter kernel (login node) ✅
```bash
bash cluster/setup_jupyter_kernel.sh
```
Registers kernel `ctddg_env` ("CTDDG (Python 3.9 / MXNet GPU)"), then writes
`~/.local/share/jupyter/kernels/ctddg_env/kernel_launcher.sh` and rewrites `kernel.json` to
launch through it — so the kernel re-applies the module loads and `PATH`/`LD_LIBRARY_PATH`
exports on whatever compute node it lands on. Without this the kernel inherits the system
Anaconda (see §2).

### 5.2 Start a Jupyter session on a GPU node ✅
```bash
sbatch cluster/launch_jupyter_preferred.sh
```
Walks `node004 → node003 → node001 → node002`, takes the first whose `sinfo -h -n <node> -o "%T"`
reports `idle`, maps it to its partition, and submits `launch_jupyter.sh` with `--nodelist`.
Walltime: `4-23:59:59` for node003, `24:00:00` otherwise.

Direct alternatives:
```bash
sbatch cluster/launch_jupyter.sh                        # defaults: partition normal, 2 GPUs, 23:59:59
sbatch -p gpu03 --gres=gpu:2 cluster/launch_jupyter.sh
sbatch -p gpu03 --time=4-23:59:59 cluster/launch_jupyter.sh
```

`launch_jupyter.sh` requests: 1 node, 1 task, 8 CPUs, `--gres=gpu:2`, 60 G RAM, 23:59:59.

### 5.3 Get the connection details ✅
```bash
squeue -u $USER
cat cluster/jupyter_<JOBID>.log
```
The log prints, e.g.:
```
    ssh -N -L 8973:node003:8973 muhamed@bhavani.iitpkd.ac.in
    http://localhost:8973
```
Port formula: `8800 + (SLURM_JOB_ID % 200)`.

### 5.4 Open the tunnel (from your laptop) ✅
```bash
ssh -N -L 8973:node003:8973 muhamed@bhavani.iitpkd.ac.in
```
`-N` = no remote command, just forward. Then browse to `http://localhost:8973`.

⚠ **The server runs with `--NotebookApp.token="" --NotebookApp.password=""`** — no
authentication. Jupyter itself warns: "All authentication is disabled. Anyone who can connect
to this server will be able to run code." It also binds `--ip=0.0.0.0`, i.e. every interface
on the compute node, not just localhost. Anyone with access to the cluster's internal network
can reach it. Consider setting a token.

### 5.5 Headless batch execution ✅ (pattern verified; this exact script not yet run)
```bash
sbatch cluster/run_pipeline_batch.sh
```
72-hour walltime; runs pretraining → generation → evaluation via `jupyter nbconvert`.
⚠ **Skips fine-tuning**, and ⚠ **would re-run `code/pretraining.ipynb` on the finished
checkpoint** — see §8.

`run_preprocessing.sh` is the verified pattern for a single notebook:
```bash
sbatch run_preprocessing.sh
```
```bash
jupyter nbconvert --to notebook --execute \
    code/data_preprocessing_1.ipynb \
    --output data_preprocessing_1_executed.ipynb
NBEXIT=${PIPESTATUS[0]}
```
✅ This has run twice (jobs 11771, 11849).

### 5.6 Interactive GPU shell
```bash
salloc -p gpu03 --gres=gpu:1 --time=02:00:00
srun --pty bash
```
(Week-1 notes §2. Useful for quick tests; **not** for long training — it dies with your SSH
session.)

### 5.7 Monitoring ✅
```bash
python check_pretraining.py            # progress / "Training is COMPLETE!"
tail -f outputs/pretrain/logs/log.out
tail -f cluster/jupyter_<JOBID>.err
squeue -u $USER
sinfo -h -o "%P %n %T %G"
scancel <JOBID>
nvidia-smi                             # only meaningful ON the GPU node
```

`check_pretraining.py` is standalone (no MXNet import), so it runs fine on the login node.

---

## 6. Running long jobs without keeping the laptop open

**The key fact: you already can.** The Slurm job owns the processes; your laptop owns only the
tunnel. Close the laptop, reconnect the tunnel tomorrow, the work continues.

That is exactly what happened with pretraining: job 11973 started 2026-09-02 22:49, training
ran through to 2026-09-04 21:40 — nearly two days — inside a Jupyter session.

### Two options, ranked

**Option A — plain `sbatch`, no Jupyter (recommended for long runs).**
Fewest moving parts; no browser, no tunnel, no websocket to drop. Model on `run_preprocessing.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ctddg-stage
#SBATCH --partition=gpu03
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                 # the code uses one GPU; don't reserve two
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=4-23:59:59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load cuda/11.2
module load anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MXNET_CUDNN_LIB_CHECKING=0
cd ~/repo/CTDDG

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ctddg_env \
    --output <name>_executed.ipynb \
    code/<notebook>.ipynb
```

⚠ Set the walltime generously — pretraining alone needed ~47 h, so 24 h is not enough.
node003/gpu03 allows `4-23:59:59`.

**Option B — Jupyter in a Slurm job (what was used).**
Convenient for exploration; the browser tab is disposable. If the websocket drops, the kernel
keeps running and reconnecting reattaches to it. Its real risk is that *you* must remember the
job's walltime.

### Practical rules
1. **Walltime > expected runtime, generously.** This is the only thing that actually kills a run.
2. **Checkpoint frequently.** Pretraining saves every 500 steps (~3 min) and resumes
   automatically ([06](06_Pretraining_and_Checkpoints.md) §9).
3. **Back up checkpoints before re-running anything.**
   ```bash
   cp -r outputs/pretrain/logs outputs/pretrain/logs.bak.$(date +%F)
   ```
4. **Log to a file, not just stdout.** Notebook stdout only reaches you via the browser.
5. **Never train on the login node** — it has no GPU, and `mx.gpu()` would fail immediately.

---

## 7. Checkpoint recovery

The pretraining loop resumes automatically when all three of `log.out`, `ckpt.params` and
`trainer.status` exist in `outputs/pretrain/logs/`: it reads the last step and elapsed time
from `log.out`, loads the weights and the Adam state, and appends to the log. Full mechanics
in [06](06_Pretraining_and_Checkpoints.md) §9.

To restart genuinely from scratch you must **move those three files aside** — there is no flag.

---

## 8. ⚠ Cluster-specific hazards

| Hazard | Detail |
|---|---|
| **Re-running pretraining mutates a finished 47-hour artefact** | `run_pipeline_batch.sh` and `notebooks/06` both invoke `code/pretraining.ipynb` unconditionally. It will resume at 480,000, take one extra step, and overwrite `ckpt.params` / `trainer.status`. **Back up first.** |
| **`notebooks/` cannot be executed** | Every multi-line cell is malformed ([02](02_Repository_and_Codebase.md) §5.2). Run `code/*.ipynb` directly |
| **Jupyter has no authentication and binds 0.0.0.0** | Anyone on the cluster network can execute code as you |
| **2 GPUs requested, 1 used** | `mx.gpu()` is device 0 only. Requesting `--gres=gpu:2` wastes an allocation and lengthens queue time |
| **`nbconvert` timeouts** | `run_pipeline_batch.sh` uses 172800 s (48 h) — **shorter than pretraining's 47 h + margin**. Prefer `--ExecutePreprocessor.timeout=-1` |
| **`--output` is relative to `cwd`** | With `cwd=$CTDDG_ROOT`, `--output=pretraining_executed.ipynb` and an input under `code/` writes to `code/pretraining_executed.ipynb`. That is where it actually landed |
| **`set -euo pipefail` in `run_pipeline_batch.sh`** | Any stage failing aborts the rest — desirable, but means a missing `fpscores.pkl.gz` kills the whole job |
| **`data/` has its own `.git`** | Remote `git://git-lfs.github.com/mshik/DGGNP.git`. Don't `git add data/` |
| **Internet access from compute nodes** | Needed by `cmd.fetch` (docking) and by `bio_embeddings` on first use. **NOT DETERMINED FROM AVAILABLE SOURCES**; ProtBert weights are already cached at `~/.cache/bio_embeddings/prottrans_bert_bfd/`, so embeddings are safe. Pre-download PDB files from the login node if `fetch` fails |
| **`~/datasets/` vs `~/repo/CTDDG/`** | Deliberately separate so multi-GB archives never enter git (Week-1 notes §16) |

---

## 9. Job history

| Job | Node | Date (2026) | Purpose | Outcome |
|---|---|---|---|---|
| 11771 | node003 | 08-17 → 08-18 | preprocessing | ❌ killed by TIME LIMIT during `needle` |
| 11849 | node004 | 08-24 → 08-25 | preprocessing | ❌ `TypeError` at the similarity-histogram cell |
| 11943 | node001 | 09-02 12:14 | Jupyter | cancelled 22:40 |
| 11968 | node002 | 09-02 22:39 | Jupyter | cancelled 22:40 |
| 11969 | node001 | 09-02 22:40 | Jupyter | cancelled 22:43 |
| 11970 | node003 | 09-02 22:43 | Jupyter | cancelled 22:44 |
| **11973** | **node003** | **09-02 22:49 → 09-07 22:50** | **Jupyter — pretraining ran inside** | **✅ 480,000 iterations complete;** job later hit TIME LIMIT |

The four rapid cancellations on 09-02 are the node-selection loop in
`launch_jupyter_preferred.sh` being retried until node003 was free.

---

## 10. VS Code Remote-SSH

This session is running through VS Code Remote-SSH
(`~/.vscode-server/cli/servers/Stable-.../` is on `PATH`), connected to the **login node**.

- ✅ Good for editing, `git`, submitting jobs, reading logs.
- ❌ There is **no GPU** on the login node — `mx.context.num_gpus()` returns 0. Do not attach
  a Python kernel here and expect MXNet GPU code to run.
- To use a GPU from VS Code you would either connect a second Remote-SSH session to the
  allocated compute node, or use the Jupyter server + tunnel route (§5.4).

## 11. Cluster status summary

| Component | Status |
|---|---|
| Login access | ✅ working |
| `ctddg_env` | ✅ built and verified |
| MXNet GPU | ✅ verified (2 GPUs detected, matmul test passed) |
| Slurm submission | ✅ verified |
| Jupyter kernel registration | ✅ verified |
| Jupyter + SSH tunnel | ✅ verified |
| Headless `nbconvert` in a Slurm job | ✅ verified (`run_preprocessing.sh`) |
| Long-run survivability | ✅ demonstrated (46.7 h) |
| `bio_embeddings` weights cached | ✅ `~/.cache/bio_embeddings/prottrans_bert_bfd/` |
| Docking toolchain | ❌ not installed |
| `notebooks/` wrappers | ❌ structurally broken |
