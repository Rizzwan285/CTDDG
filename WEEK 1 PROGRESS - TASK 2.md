# WEEK 1 PROGRESS — TASK 2

**Task:** Codebase Audit and Environment Setup for CTDDG

**Objective:** Set up a working Python/CUDA environment and fix the CTDDG code repository (`sahelybhadra/CTDDG`) so that the baseline model runs without errors.

**Cluster:** Bhavani, IIT Palakkad (`bhavani.iitpkd.ac.in`) — GPU work done on `node002` (NVIDIA A30)

**Working repository:** `https://github.com/Rizzwan285/CTDDG.git`

---

## 1. What this task actually involved

Before any model can be trained, two things have to be true: the machine must be able to import and run the code, and the code must actually be complete. Week 1 was spent establishing both, and neither was straightforward.

The environment half turned into a version-pinning exercise. CTDDG is built on **Apache MXNet 1.9.1**, a framework that is now retired, so it drags along a chain of older libraries (NumPy, SciPy, Gensim, Transformers) that break against anything modern. On top of that, the cluster's own Anaconda installation silently shadows the Conda environment on compute nodes, which produced errors that looked like missing packages but were not.

The audit half turned out to be the more serious finding. The repository, as published, **cannot be run end to end** — an entire training stage is missing from it, and several notebooks crash before they finish.

Both halves are now resolved to the point where the pipeline is logically continuous and the GPU stack is verified working.

---

## 2. Background and terminology

* **Apache MXNet:** The deep-learning framework the CTDDG model is written in (`mxnet-cu112==1.9.1`, the CUDA 11.2 build). It is no longer actively maintained, which is the root cause of most dependency conflicts in this project.

* **Conda environment:** An isolated Python installation with its own interpreter and libraries. Here the environment is `ctddg_env` (Python 3.9), created so that the old dependency stack does not conflict with the cluster's system Python.

* **Environment Modules (`module load`):** The cluster's mechanism for exposing software such as CUDA and Anaconda. Nothing is on the `PATH` by default — CUDA and Anaconda have to be explicitly loaded in every session and every job script.

* **Slurm:** The cluster's job scheduler. GPUs are not visible from the master node; a job must be requested (`salloc`) and entered (`srun --pty bash`) before any GPU exists for the process.

* **cuDNN and NCCL:** NVIDIA libraries that MXNet links against at import time — cuDNN for the deep-learning primitives, NCCL for multi-GPU communication. MXNet refuses to import if either shared object is missing, **even when only one GPU is used.**

* **bio-embeddings / ProtTrans BERT-BFD:** The protein language model used by the preprocessing notebook. It converts a protein amino-acid sequence into a numeric vector — one 1024-dimensional vector per residue. This vector is what "conditions" the molecule generator on a specific protein target.

* **SMILES:** A plain-text string encoding of a molecule's structure (e.g. `CC(=O)Oc1ccccc1C(=O)O` for aspirin). The pretraining stage of CTDDG learns from millions of these.

* **Unconditional vs. conditional generation:** An *unconditional* model learns the general grammar of valid drug-like molecules. A *conditional* model additionally takes the protein embedding as an input, so it generates molecules aimed at a **specific** target pocket. The distinction between these two matters a great deal here — it is exactly where the repository is broken (see Section 5.1).

---

# PART A — ENVIRONMENT SETUP

## 3. Cluster and target configuration

| Item | Value |
| ----- | ----- |
| **Master node** | `bhavani.iitpkd.ac.in` |
| **GPU node used** | `node002` |
| **GPU** | NVIDIA A30, 24 GB, compute capability `sm_80` |
| **NVIDIA driver** | 530.30.02 (driver-level CUDA 12.1) |
| **Modules loaded** | `anaconda3/2022.10`, `cuda/11.2` |
| **Conda version** | 23.1.0 |
| **Environment name** | `ctddg_env` |
| **Python** | 3.9.25 |

### Why CUDA 11.2 and not the driver's CUDA 12.1

The driver reports CUDA 12.1, which is only the *maximum* version it supports. The repository pins `mxnet-cu112==1.9.1`, which is compiled against **CUDA 11.2**, so the `cuda/11.2` module is the one that must be loaded. Loading a newer CUDA module would not help — MXNet 1.9.1 has no build for it. The cluster fortunately provides modules from CUDA 10.1 through 11.3, so 11.2 was available.

## 4. Environment build

```bash
module load anaconda3/2022.10

conda create -n ctddg_env python=3.9 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

conda install -c bioconda emboss -y          # -> emboss 6.5.7
pip install mxnet-cu112==1.9.1               # ~500 MB download
pip install -r requirements.txt
```

`conda activate` failed on the first attempt with `CommandNotFoundError: Your shell has not been properly configured`. Rather than permanently modifying the cluster's shell configuration, the `source .../conda.sh` line above was used — it is self-contained and safe to repeat in job scripts.

## 5. Problems encountered and how each was resolved

This is the substance of the environment work. Every one of these was a hard blocker at the time it appeared.

| # | Problem and symptom | Root cause | Fix applied |
| ----- | ----- | ----- | ----- |
| **1** | **Conda unavailable** — `bash: conda: command not found` | Anaconda is provided as a module, not as a default install | `module load anaconda3/2022.10` |
| **2** | **Gensim would not build** — `AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__'` | `gensim==3.8.3` uses a legacy `setup.py` that breaks under pip's build isolation | Installed through Conda instead: `conda install -c conda-forge gensim=3.8.3` |
| **3** | **bio-embeddings install failed** — dependency resolution collapsed on the Gensim build | Same as above, cascading | `pip install "bio-embeddings[all]" --no-deps`, then supplied dependencies manually → `bio-embeddings 0.2.2` |
| **4** | **No GPU on the master node** — `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | The master node has no GPU assigned; this is expected behaviour | Request a GPU job with `salloc`, then enter it with `srun --pty bash` (full command below) |
| **5** | **Conda PATH overridden on the GPU node** — `ModuleNotFoundError: No module named 'mxnet'` / `'torch'`, *even with the environment activated* | `$CONDA_PREFIX` pointed at `ctddg_env`, but `which python` returned `/home/apps/anaconda/anaconda3/bin/python` — the cluster's system Anaconda was winning the `PATH` | `export PATH="$CONDA_PREFIX/bin:$PATH"` |
| **6** | **Wrong system libraries loaded** — SciPy import failure | The Anaconda module puts `/home/apps/anaconda/anaconda3/lib` at the front of `LD_LIBRARY_PATH`, so an older `libstdc++.so.6` was loaded instead of the environment's | `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` |
| **7** | **MXNet import crash** — `OSError: libcudnn.so.8: cannot open shared object file` | The cluster's `cuda/11.2` module ships CUDA but **not** cuDNN | `conda install -c conda-forge cudnn=8.9.7` |
| **8** | **MXNet import crash (again)** — `OSError: libnccl.so.2: cannot open shared object file` | No NCCL module exists on the cluster (searching `module avail` for `nccl` returns nothing), and the `nvidia` channel did not carry the required version | Installed `nccl 2.11.4.1` from conda-forge → `$CONDA_PREFIX/lib/libnccl.so.2` |
| **9** | **NumPy incompatibility** — `AttributeError: module 'numpy' has no attribute 'bool'` | MXNet 1.9.1 still uses the `np.bool` alias, which was removed in NumPy 1.24 | Pinned `numpy=1.23.5` (satisfies the repo's own `numpy>=1.21,<1.24`) |
| **10** | **SciPy / Gensim incompatibility** — `ImportError: cannot import name 'triu' from 'scipy.linalg.special_matrices'` | Gensim 3.8.3 calls a SciPy function removed in newer releases | Pinned `scipy 1.10.1` |
| **11** | **Transformers incompatibility** — `AttributeError: module transformers has no attribute modeling_utils` | bio-embeddings 0.2.2 expects an older Transformers internal API | Pinned `transformers 4.21.2` |
| **12** | **ProtTrans appeared to hang** — no output for 20+ minutes on first init | Not a hang. It was silently downloading the **1.6 GB** model into `~/.cache/bio_embeddings/prottrans_bert_bfd/`; `ps -u $USER -f` confirmed there was no stuck process | Waited for the cache to populate. Subsequent initialisations are fast |

The GPU job referred to in #4 is requested as:

```bash
salloc --partition=normal -N 1 --gres=gpu:1 --mem=30000 --ntasks-per-node=1
srun --pty bash          # enter the allocated node
hostname; nvidia-smi     # confirm you are on a GPU node
```

### Diagnosing #8 (missing NCCL)

The specific missing symbol was confirmed by inspecting MXNet's shared object directly rather than guessing:

```bash
ldd ~/.conda/envs/ctddg_env/lib/python3.9/site-packages/mxnet/libmxnet.so \
    | grep -E "not found|nccl|cudnn|cuda"
```

which returned:

```text
libcudnn.so.8   => /home/muhamed/.conda/envs/ctddg_env/lib/libcudnn.so.8
libcuda.so.1    => /lib64/libcuda.so.1
libnccl.so.2    => not found
libcudart.so.11.0 => /opt/ohpc/pub/apps/cuda/11.2/lib64/libcudart.so.11.0
```

That single line — `libnccl.so.2 => not found` — turned a vague import error into a one-command fix.

## 6. The standard environment block

This is the most reusable output of the environment work. **Every future Slurm job script must begin with this**, because `conda activate` alone is demonstrably not sufficient on the compute nodes:

```bash
module load cuda/11.2
module load anaconda3/2022.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env

# CRITICAL: force the node to use Conda's Python and libraries,
# not the cluster's system Anaconda
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Suppress the harmless MXNet cuDNN version-mismatch warning
export MXNET_CUDNN_LIB_CHECKING=0
```

Then verify before running anything real:

```bash
which python     # expect ~/.conda/envs/ctddg_env/bin/python
python --version # expect Python 3.9.25
```

## 7. Verification tests

The environment was not declared "done" on the basis of a successful `pip install`. Each component was tested for the behaviour actually needed downstream.

| # | What is being tested | Command (abbreviated) | Result | Status |
| ----- | ----- | ----- | ----- | ----- |
| **1** | Interpreter identity | `which python` | `~/.conda/envs/ctddg_env/bin/python` | PASS |
| **2** | MXNet import + GPU count | `mxnet.context.num_gpus()` | `MXNet: 1.9.1`, `GPUs: 1` | PASS |
| **3** | **Real GPU computation** | `mx.nd.dot()` on a 100×100 matrix on `mx.gpu()` | `(100, 100)`, `gpu(0)`, `100.0` | PASS |
| **4** | Gensim | `gensim.__version__` | `3.8.3` | PASS |
| **5** | SciPy | `scipy.__version__` | `1.10.1` | PASS |
| **6** | bio-embeddings import | `import bio_embeddings` | OK | PASS |
| **7** | Embedder initialisation | `ProtTransBertBFDEmbedder()` | `Embedder init: OK` | PASS |
| **8** | **Real protein embedding** | `e.embed('MKTAYIAKQRQISFVKSHFSRQ')` | `numpy.ndarray`, shape **`(22, 1024)`** | PASS |

Tests 3 and 8 are the meaningful ones. Test 3 proves MXNet is not merely importable but is genuinely executing arithmetic on the A30. Test 8 proves the protein embedding pipeline produces the exact tensor shape the model expects — 22 residues in the test sequence, 1024 features per residue, which matches the `N_C = 1024` conditioning dimension used later in the model.

## 8. Warnings that appear but are not blockers

Several warnings show up during normal operation and were investigated rather than ignored:

* **`cuDNN lib mismatch: linked-against version 8907 != compiled-against version 8101`** — MXNet was compiled against cuDNN 8.1 but is running against 8.9.7. The GPU computation test still returned the correct result, so this is cosmetic. Suppressed with `MXNET_CUDNN_LIB_CHECKING=0`.

* **`NVIDIA A30 with CUDA capability sm_80 is not compatible with the current PyTorch installation`** — the installed PyTorch is `1.9.1+cu102`, which predates the Ampere architecture. This matters less than it appears: PyTorch is only a *dependency of bio-embeddings*, not the framework CTDDG trains in. This was confirmed by searching the repository:

  ```bash
  grep -RniE "import torch|from torch|torch\." code scripts
  ```

  No CTDDG model code uses PyTorch. All model code (`pretraining.ipynb`, `Generating_samples.ipynb`, `finetune_cell.py`) uses MXNet. If protein embedding later proves too slow on CPU, PyTorch can be upgraded independently without touching the MXNet stack.

* **`Some weights of the model checkpoint were not used when initializing BertModel`** — expected, because the ProtTrans checkpoint was trained with a different task head. Embeddings still generate correctly.

* **`RequestsDependencyWarning`** and **`pkg_resources is deprecated`** — artefacts of the older package stack, non-fatal.

## 9. Environment snapshots

After the stack stabilised, it was captured so it can be rebuilt without repeating Section 5:

```bash
conda env export --no-builds > ~/ctddg_env.yml   # 7.4 K
pip freeze > ~/ctddg_pip_freeze.txt              # 5.2 K
```

Both snapshots were taken **after** the cuDNN/NCCL/NumPy/SciPy fixes and after the ProtTrans embedding test passed, so they represent a known-good state.

---

# PART B — CODEBASE AUDIT

## 10. What the repository contains

```text
CTDDG/
├── code/
│   ├── data_preprocessing_1.ipynb   # SMILES cleanup + ProtTrans protein embeddings
│   ├── pretraining.ipynb            # Unconditional molecule generator (ChEMBL)
│   ├── Generating_samples.ipynb     # Conditional generation for a target
│   ├── Evaluation_metrics.ipynb     # Validity, novelty, SAS, etc.
│   └── Molecular_docking.ipynb      # Docking of generated molecules
├── scripts/                         # (added this week)
├── data/                            # not tracked in git
└── requirements.txt
```

The intended pipeline is: preprocess data → pretrain a general molecule generator → adapt it to a specific protein target → generate candidate molecules → evaluate them → dock them.

## 11. Audit findings

All five notebooks were read cell by cell against that intended flow. Four distinct problems were found.

### 11.1 The missing fine-tuning stage — **CRITICAL**

This is the most important finding of the week.

* `pretraining.ipynb` trains a class called **`VanillaMolGen_RNN`**. This is an **unconditional** model — it learns to produce valid drug-like molecules in general, with no knowledge of any protein.

* `Generating_samples.ipynb` loads a class called **`CVanillaMolGen_RNN`** — a **conditional** model, which takes a 1024-dimensional protein embedding as an extra input and produces molecules aimed at that specific target.

* These are **not the same model**, and **the script that trains the second one does not exist anywhere in the repository.**

In other words the repository contains step 1 and step 3 of a three-step process, and step 2 is simply absent. Running the notebooks in the documented order would fail at generation time, because there is no checkpoint of the conditional architecture for it to load — and no code capable of producing one. This is a genuine gap in the published work, not a configuration mistake on our side.

### 11.2 Data format mismatch, and dead toxicity code — **HIGH**

* `data_preprocessing_1.ipynb` writes the ChEMBL dataset as **plain SMILES**, one molecule per line.
* `pretraining.ipynb` reads it expecting **`<SMILES> <class_label>`** and does `smiles, smiles_class = smiles.split(" ")`.

On plain SMILES this raises a `ValueError` and the data loader crashes immediately — so the pipeline cannot proceed past this point as shipped.

Tracing the label further through the code produced an unexpected result: **the parsed class label is never actually used.** The loader parses it, then hardcodes `tox_class = [1]*k`, and the loss function uses the hardcoded value. So the "detoxification" conditioning that the project name implies is, in this baseline, **dead code** — it is parsed, discarded, and replaced with a constant.

This is worth flagging clearly because it changes what the baseline actually is. The published baseline is a *target-conditional* generator; the *toxicity-conditional* part is not functional in this code. That is useful to know now rather than after training runs, and it defines a concrete extension point for the rest of the project.

### 11.3 Fragmented hardcoded paths — **HIGH**

The notebooks contain **12 distinct hardcoded absolute paths**, left over from at least three different machines the original authors used:

```text
/workspace/mtp_data/bindingdb
/workspace/mtp_data/data/atom_types.txt
/workspace/binding_data/atom_types.txt
/workspace/data/atom_types.txt
/home/iit/CDGCN/data/chembl
/workspace/Toxicity_experiment/chembl_final.txt
/workspace/Toxicity_experiment/March_experiment/just_test
/workspace/fpscores.pkl.gz
/workspace/CTDGD/outputs
/workspace/CTDGD/data
/workspace/finetune
/workspace/Jupyter_Dock
```

Note the inconsistency even within a single project — `atom_types.txt` is read from three different locations, and the output directory is spelled `CTDGD` rather than `CTDDG`. None of these paths exist on Bhavani, so the notebooks cannot run out of the box on any machine other than the authors'.

### 11.4 Outright bugs — **MEDIUM**

* `Molecular_docking.ipynb` contains an **empty `all_smiles_df_beams.append()`** call immediately after a valid one. `list.append()` requires exactly one argument, so this raises `TypeError` and kills the docking loop.

* `Generating_samples.ipynb` creates its output directories with six consecutive `os.system(f'mkdir ...')` shell calls. These fail silently if a parent directory is absent or a directory already exists, and the notebook then proceeds to write into a path that does not exist.

---

# PART C — FIXES IMPLEMENTED

## 12. Scripts written

Each fix is deliberately **non-invasive** — the original notebook logic and training behaviour are preserved. Nothing was "improved" beyond what was needed to make the pipeline run, so that the reproduction remains a faithful reproduction.

| Script | Fixes finding | What it does |
| ----- | ----- | ----- |
| **`scripts/finetune_cell.py`** | 11.1 | Supplies the missing stage: the `CMoleculeGenerator_RNN` / `CVanillaMolGen_RNN` conditional architecture, a conditional data loader, and the MXNet training loop |
| **`scripts/convert_chembl_format.py`** | 11.2 | Appends a dummy `" 1"` label to each plain SMILES line, satisfying the loader's `split(" ")` without altering training behaviour (since the label is discarded anyway) |
| **`scripts/config.py`** | 11.3 | Centralises every path in one place, derived from a single `PROJECT_ROOT` (overridable via the `CTDDG_ROOT` environment variable) |
| **`scripts/patch_paths.py`** | 11.3 | Rewrites all 12 hardcoded paths inside the notebook JSON to the current machine's absolute repository path. Run as `python scripts/patch_paths.py $(pwd)` |
| **`scripts/patch_notebooks.py`** | 11.4 | Removes the empty `.append()` call and replaces the six `os.system('mkdir')` calls with `os.makedirs(..., exist_ok=True)` |
| **`scripts/download_data.sh`** | data setup | Automates fetching the Google Drive datasets via `gdown`, downloading `fpscores.pkl.gz` for the SAS metric, and cloning `Jupyter_Dock` for the docking notebook |
| **`.gitignore` / `README.md`** | housekeeping | Keeps multi-GB datasets out of git; documents the full cluster build and execution order |

## 13. How the missing fine-tuning stage was reconstructed

Since finding 11.1 is the critical one, the approach taken is worth stating explicitly. The conditional model was not invented freely — it was **derived from the two endpoints that do exist in the repository**, so that it is compatible with both:

1. **Architecture.** `CVanillaMolGen_RNN` subclasses the existing `VanillaMolGen_RNN` and adds exactly what the conditional path needs: a `_policy_0` layer that produces the first-atom distribution from the protein vector `c`, and a per-layer `linear_c` projection that injects `c` into each graph-convolution layer. The constructor signature was matched to what `Generating_samples.ipynb` already expects, so the downstream notebook needs no changes.

2. **Weight transfer.** The pretrained unconditional weights are loaded with `allow_missing=True, ignore_extra=True`. This carries over everything the two models share and leaves only the genuinely new conditioning layers to be initialised fresh (Xavier) — which is the whole point of fine-tuning rather than training from scratch.

3. **Data loading.** `CMolRNNLoader` extends the existing loader to yield `(SMILES, protein_embedding)` pairs, reusing the parent's collate logic by appending the same dummy `" 1"` label described in 11.2. The condition tensor `c` is appended to the batch.

4. **Training loop.** Adam at `1e-4` with gradient clipping at 10.0, batch size 16, periodic checkpointing to `outputs/<model>/Dataset<i>/model/`. `N_C` is read from the data (1024, matching the ProtTrans embedding width verified in test 8) and written into `configs.json` so the generation notebook can reconstruct the model.

The script is written to be pasted as a final cell in `pretraining.ipynb`, where the parent classes (`MoleculeGenerator_RNN`, `MolRNNLoader`, `GraphConv`, `BalancedSampler`) are already in scope.

## 14. Corrected execution order

The documented order in the original repository skips the missing stage. The corrected order is:

1. Environment setup (Section 6) and path patching — `python scripts/patch_paths.py $(pwd)`
2. Format conversion — `python scripts/convert_chembl_format.py data/chembl/chembl.txt data/chembl/chembl_final.txt`
3. `code/data_preprocessing_1.ipynb` — SMILES cleanup + ProtTrans embeddings
4. `code/pretraining.ipynb` — **unconditional** training
5. **`scripts/finetune_cell.py`** — **conditional fine-tuning** ← *the stage that was missing*
6. `code/Generating_samples.ipynb` — conditional generation
7. `code/Evaluation_metrics.ipynb`
8. `code/Molecular_docking.ipynb`

## 15. Work committed

| Commit | Contents |
| ----- | ----- |
| `adb684b` | Safe fixes, `requirements.txt`, and the fine-tuning cell — 5 new scripts, 558 insertions |
| `67fd94a` | `scripts/download_data.sh` |
| `28c9b75` | `.gitignore` and expanded `README.md` with dataset and cluster instructions |

---

# PART D — DATASET AND REPOSITORY HOUSEKEEPING

## 16. Repository cleanup

Two structural problems were found and resolved on the cluster copy:

* **A nested duplicate repository.** `find . -maxdepth 2 -type d` revealed `./.git`, `./CTDDG/.git` *and* `./data/.git` — a complete second copy of CTDDG sitting inside the first. `diff -qr` across `code/`, `scripts/`, `README.md` and `requirements.txt` produced no output, and both repositories sat on the same commit (`28c9b75`), confirming it was an accidental clone-inside-a-clone rather than a divergent version. The outer repository was kept as the working copy.

* **Stale git metadata inside `data/`.** The `data/` directory carried its own `.git` pointing at an unrelated remote (`git://git-lfs.github.com/mshik/DGGNP.git`). Treated as leftover metadata from the original data distribution, not project history.

Repository and datasets are deliberately kept separate (`~/repo/CTDDG` and `~/datasets/`) so that multi-GB archives never enter git history.

## 17. Dataset status

| Dataset | Purpose | Status |
| ----- | ----- | ----- |
| **ChEMBL** | Unconditional pretraining (SMILES) | Available; format conversion script ready |
| **BindingDB** | Conditional fine-tuning (ligand–protein pairs) | Available |
| **`fpscores.pkl.gz`** | SAS score in evaluation | Download automated |
| **CrossDocked2020 — types archive** (21 MB) | Metadata index | Extracted successfully |
| **CrossDocked2020 — main archive** (~4.0 GB on cluster) | 3D structures | **Extraction failed — under investigation** |

### The CrossDocked extraction problem

Extraction of the main archive failed with:

```text
gzip: stdin: decompression OK, trailing garbage ignored
tar: Child returned status 2
tar: Error is not recoverable, exiting now
```

The likely cause is an incomplete or corrupted transfer. The original plan was to `rsync` the archive to the cluster, but the cluster responded `bash: rsync: command not found` and the transfer aborted with protocol error code 12, so the copy currently on the cluster cannot be assumed intact.

Rather than repeatedly retrying extraction on a possibly damaged file, the next step is an integrity check first:

```bash
cd ~/datasets
gzip -t downsampled_CrossDocked2020_v1.3.tgz
echo "Exit code: $?"
```

Exit code 0 means the gzip layer is sound and extraction can proceed; anything else means the archive must be re-transferred. **The original archive will not be deleted until extraction is confirmed.**

This does not block current progress: the immediate goal is the **single-target, single-property baseline**, which runs on ChEMBL and BindingDB. CrossDocked2020 (Task 1's dataset) is needed for the later multi-target extension.

---

# PART E — STATUS AND NEXT STEPS

## 18. Current status

| Component | Status | Evidence |
| ----- | ----- | ----- |
| **Conda environment** | **READY** | Python 3.9.25, all imports clean |
| **MXNet on GPU** | **READY** | Real matrix multiplication executed on the A30 |
| **Protein embeddings** | **READY** | `(22, 1024)` ProtTrans embedding generated |
| **Repository** | **READY** | Synchronised with `origin/main`, duplicates resolved |
| **Codebase audit** | **COMPLETE** | 4 findings documented, all addressed |
| **Fixes** | **COMMITTED** | 6 scripts across 3 commits |
| **ChEMBL / BindingDB** | **AVAILABLE** | — |
| **CrossDocked2020 (main)** | **NOT READY** | Archive integrity check pending |
| **Training runs** | **NOT STARTED** | Next phase |

## 19. Pending work — Week 2

1. **Validate and extract the CrossDocked2020 archive** (`gzip -t` first; re-transfer if it fails).
2. **Confirm the notebooks' real data-path expectations** by reading the loader cells directly, rather than assuming the layout in the README matches the code.
3. **Run `data_preprocessing_1.ipynb`** on the cluster — the first genuinely compute-heavy stage, since it invokes ProtTrans over every protein sequence.
4. **Run `pretraining.ipynb`** to produce the unconditional checkpoint.
5. **Execute the reconstructed fine-tuning stage** and verify that the checkpoint it produces actually loads in `Generating_samples.ipynb` — this is the real test of whether finding 11.1 has been correctly resolved.
6. **Write a reusable Slurm batch script** wrapping the environment block from Section 6, so the notebooks can be run headless via `jupyter nbconvert` instead of interactively.

## 20. The main lesson from this week

The single biggest time sink was not a hard technical problem — it was a misleading one. For a long stretch, the Conda environment was correctly created, correctly activated, and reporting the right `$CONDA_PREFIX`, while the compute node was quietly executing the **cluster's system Anaconda Python** the whole time. Every error it produced (`No module named 'mxnet'`, `No module named 'torch'`) pointed at a broken installation, when the installation was fine and only the `PATH` was wrong.

The practical takeaway, now baked into the standard environment block and the README: on this cluster, `conda activate` is never sufficient on its own. Always verify with

```bash
which python
which pip
```

before trusting anything else, and always export `PATH` and `LD_LIBRARY_PATH` explicitly in Slurm jobs.

The parallel lesson from the audit half is that a published repository running "as documented" cannot be assumed. Reading all five notebooks end to end before attempting to run them is what surfaced the missing fine-tuning stage — which would otherwise have appeared much later, as a confusing checkpoint-loading error after hours of training.
