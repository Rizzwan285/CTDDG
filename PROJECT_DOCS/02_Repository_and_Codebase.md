# 02 — Repository and Codebase Map

## 1. Top-level layout

```
/home/muhamed/repo/CTDDG/
├── papers/                       # The two research papers (authoritative for methodology)
│   ├── CTDDG.pdf
│   └── CDGCN.pdf
├── code/                         # ORIGINAL implementation (authors'). 6 notebooks.
├── notebooks/                    # GENERATED wrappers (00–06). See §5 — currently broken.
├── scripts/                      # Utility + reconstruction scripts added during Week 1.
├── cluster/                      # Slurm job scripts, Jupyter launchers, job logs.
├── data/                         # Datasets (git-ignored). Also a stale nested .git.
├── outputs/                      # Checkpoints and results (git-ignored).
├── Molecular_dynamics_files/     # Authors' published MD structures (CDK2 case study)
├── Moleculsr_docking_top10GeneratedMol/  # Authors' published top-10 docked poses [sic]
├── supplimentaryAPIN.pdf         # CTDDG supplementary (Tables S1, S2) [sic]
├── check_pretraining.py          # Standalone progress checker
├── run_preprocessing.sh          # Slurm job for the preprocessing notebook
├── requirements.txt
├── README.md
├── WEEK 1 PROGRESS - TASK 2.{md,docx}   # Project history / decisions
└── preprocessing_118*.{out,err}  # Slurm output from preprocessing attempts
```

## 2. `code/` — the original implementation

This is the source of truth for what the model does. All six files are Jupyter notebooks;
there are **no `.py` modules**. Model definitions are duplicated verbatim across notebooks
rather than imported.

### 2.1 `code/pretraining.ipynb` — **the core file**

**Structure:** a *single* code cell, 51,461 characters. Everything — imports,
hyperparameters, data loading, model classes, training loop — is in that one cell.

**Purpose:** train the unconditional generator `VanillaMolGen_RNN` on ChEMBL.

**Inputs**
| Path | Purpose |
|---|---|
| `data/chembl/chembl_final.txt` | `<SMILES> <class>` per line, 1,090,529 lines |
| `data/atom_types.txt` | 65 atom-type triples, read by `MoleculeSpec.__init__` |
| `outputs/pretrain/logs/{configs.json, ckpt.params, trainer.status, log.out}` | read on resume |

**Outputs:** the same four files in `outputs/pretrain/logs/`.

**Classes and functions, in file order**

| Name | Role |
|---|---|
| `MoleculeSpec` | Atom/bond type registry. Reads `atom_types.txt`; `bond_orders = [AROMATIC, SINGLE, DOUBLE, TRIPLE]`; `max_iter = 120` |
| `get_mol_spec()` | Lazy global singleton `_mol_spec` |
| `BalancedSampler(Sampler)` | Buckets molecules by SMILES length, then draws one per bucket → batches of similar-size molecules (memory efficiency) |
| `get_graph_from_smiles(smiles)` | RDKit → (nx.Graph, atom_types, atom_ranks, bonds, bond_types). `atom_ranks` from `Chem.CanonicalRankAtoms` |
| `traverse_graph(...)` | Implements `Q_α`: recursive DFS, canonical-rank neighbour w.p. `p`, else uniform random. Returns `step_ids` and `log_p` |
| `single_reorder`, `single_expand` | Convert one traversal into the per-step action tensors (`action_type`, `atom_type`, `bond_type`, `append_pos`, `connect_pos`) |
| `get_d(A, X)` | Sparse computation of distance-2 and distance-3 neighbour sets (`D_2`, `D_3`) |
| `merge_single_0`, `merge_single` | Batch offsetting; splits adjacency into `num_bond_types` slices + `D_2` + `D_3` → the 6-element `A_list` |
| `process_single(smiles, k, p)` | Per-molecule pipeline; samples `k` traversals. ⚠ **`tox_class = [1] * k` hardcoded here** |
| `get_mol_from_graph(_list)` | Graph → RDKit Mol, with optional `SanitizeMol` |
| `MolLoader(DataLoader)` | Collation + `from_numpy_to_tensor` (hardcodes `mx.gpu()`) |
| `MolRNNLoader(MolLoader)` | Adds `graph_to_rnn`, `rnn_to_graph`, `NX_cum` for the GRU |
| `GraphConvFn`, `EfficientGraphConvFn` | Custom autograd `Function`s; the "efficient" one recomputes in backward to save memory |
| `SegmentSumFn(GraphConvFn)` | Segment-sum via a sparse CSR matrix — used for graph pooling |
| `logsumexp`, `squeeze`, `unsqueeze`, `get_activation` | Utilities |
| `Linear_BN`, `BatchNorm`, `GraphConv`, `Policy` | Layers. `BatchNorm` is a **custom** `nn.Block`, not `gluon.nn.BatchNorm` |
| `MoleculeGenerator(nn.Block)` | Abstract base: embeddings, dense stack, `policy_0`, `_likelihood`, `forward` |
| `MoleculeGenerator_RNN` | Adds `gluon.rnn.GRU` and `_rnn_train` / `_rnn_test` |
| `VanillaMolGen_RNN` | Concrete unconditional model — **what was trained** |

Then: dataset load → sampler/loader → `is_continuous` detection → model construction →
Adam trainer → training loop.

### 2.2 `code/pretraining_executed.ipynb` — the completed run

Source is **byte-identical** to `pretraining.ipynb` (verified by diff of extracted sources).
Difference: `execution_count = 1` and **965 stored outputs**. Metadata records Python
3.9.25 (vs 3.8.18 in the un-executed file). This is the artefact of the finished
480,000-iteration run. Git-ignored via `*_executed.ipynb`.

### 2.3 `code/Generating_samples.ipynb` — **the conditional model lives here**

25 cells. Re-declares most of `pretraining.ipynb` (`MoleculeSpec` … `MoleculeGenerator_RNN`),
then adds the conditional layer.

| Cell | Contents |
|---|---|
| 6–14 | Duplicated from `pretraining.ipynb`, then **cell 14** additionally defines `_TwoLayerDense`, `CMoleculeGenerator_RNN`, `CVanillaMolGen_RNN` |
| 16 | `_decode_step(...)` — the sequential decoder; `Builder` (abstract); `CVanilla_RNN_Builder` with `.sample()` and `.to_nd()` |
| 18 | `generate_samples(protein_data_path, dataset_index, n_samples, run, model_name='CTDGD')` |
| 19 | Driver — **hardcodes** `dataset_index=2, run=1, n_samples=1000, model_name='CTDGD'` |
| 21–23 | Visualisation with `Draw.MolsToGridImage` |

**Inputs:** `data/d{i}_te_embeddings_run{r}.txt` (tab-separated floats, one protein per line);
`outputs/CTDGD/Dataset{i}/model/{configs.json, ckpt.params}`.
**Outputs:** `outputs/CTDGD/Dataset{i}/generated_samples/{n}/run{r}/Protein{p}_generated_samples.csv`
plus `samples_evaluation.txt` (total wall time).

### 2.4 `code/data_preprocessing_1.ipynb`

108 cells. Order of operations:

1. **cells 3–25** — load the five BindingDB folds; discover fold 1 has 1514 train / 99 test
   proteins instead of 1042 / 104; repair by sampling with `random.seed(17)`.
2. **cells 26–37** — EMBOSS `needle` pairwise alignments (train↔test, within-train,
   within-test) and similarity histograms. ⚠ **cell 37 contains the bug that killed the
   last preprocessing run** (see §7).
3. **cells 44–54** — ligand filtering: `filter_by_atom_count` drops molecules with >100 atoms.
4. **cells 55–57** — collect atom types from BindingDB ligands.
5. **cells 58–70** — ChEMBL: read `chembl.csv`, drop >100-atom molecules, keep the longest
   salt fragment, subtract BindingDB ligands, write `chembl.txt`; merge atom types → write
   `data/atom_types.txt` (65 entries).
6. **cells 71–79** — protein embeddings via `ProtTransBertBFDEmbedder`.
7. **cells 80–89** — split the test proteins in half into validation / test.
8. **cells 90–107** — assemble per-model dataset files; the **CDGCN** section (cells 99–107)
   writes `d1_tr_cdgcn.txt` as `<ligand>\t<embedding floats>`.

### 2.5 `code/Evaluation_metrics.ipynb`

20 cells. Property functions (`lipsinki_properties` [sic], `molecular_properties`), a vendored
copy of RDKit's SA_Score (`readFragmentScores`, `calculateScore`, `processMols`) reading
`data/fpscores.pkl.gz`, then aggregation. Also plots the fine-tuning loss from
`outputs/finetune/log.out`.

### 2.6 `code/Molecular_docking.ipynb`

41 cells, largely adapted from the Jupyter Dock tutorial. PyMOL fetch → LePro/`fix_protein`
→ `prepare_receptor` → obabel/pybel ligand prep → `prepare_ligand` → smina.

## 3. `scripts/`

| File | What it is | Status |
|---|---|---|
| `config.py` | Centralised path config, `cfg` singleton, `PROJECT_ROOT` from `$CTDDG_ROOT` (default `/workspace`) | Works, but **nothing in `code/` imports it** — the notebooks still use absolute paths |
| `convert_chembl_format.py` | Appends `" 1"` to every SMILES line | **Has been run** — produced `chembl_final.txt` |
| `patch_paths.py` | Rewrites 12 hardcoded absolute paths inside `code/*.ipynb` | **Has been run** — visible in the uncommitted diff |
| `patch_notebooks.py` | Removes an empty `.append()` in the docking notebook; replaces six `os.system('mkdir')` with `os.makedirs` | **Has been run** (both fixes present in current files) |
| `download_data.sh` | gdown the authors' Drive folder, fetch `fpscores.pkl.gz`, clone Jupyter_Dock | **Not fully run** — neither artefact exists |
| `finetune_cell.py` | **Reconstruction** of the missing conditional fine-tuning stage | **Non-functional** — see [07](07_Finetuning_and_Conditioning.md) §5 |

⚠ **Ordering constraint:** `patch_notebooks.py` matches against `/workspace/...` strings, so
it must run **before** `patch_paths.py`. That is the order that was actually used.

## 4. `cluster/`

| File | Role |
|---|---|
| `setup_jupyter_kernel.sh` | Registers `ctddg_env` as Jupyter kernel `ctddg_env`; writes a `kernel_launcher.sh` wrapper that re-loads modules and forces `PATH`/`LD_LIBRARY_PATH` |
| `launch_jupyter.sh` | Slurm batch script: 1 node, 8 CPUs, `--gres=gpu:2`, 60 G, 23:59:59; port `8800 + (JOBID % 200)`; prints the SSH tunnel command; starts Jupyter with **no token and no password** |
| `launch_jupyter_preferred.sh` | Picks the first idle node from `node004 node003 node001 node002`, maps it to partition `gpu04/gpu03/gpu01/gpu02`, resubmits `launch_jupyter.sh` |
| `run_pipeline_batch.sh` | Headless nbconvert chain: pretraining → generation → evaluation. **Skips fine-tuning** |
| `generate_notebooks.py` | **The script that generated `notebooks/00–06`** — see §5 |
| `jupyter_119*.{log,err}` | Job logs from five Jupyter sessions (11943, 11968, 11969, 11970, 11973) |

## 5. `notebooks/` — the generated wrappers

### 5.1 Provenance — definitively established

`cluster/generate_notebooks.py` emits all seven notebooks. It builds each cell as:

```python
"source": src.split("\n") if isinstance(src, str) else src
```

### 5.2 ⚠ They are structurally broken

`nbformat` requires each element of the `source` list to **retain its trailing newline**,
because Jupyter reconstructs the cell as `"".join(source)`. `src.split("\n")` strips them.
Result: every multi-line cell collapses onto one line.

Concretely, `notebooks/01_pretraining.ipynb` cell 1 becomes:

```
import os, sysCTDDG_ROOT = os.environ.get("CTDDG_ROOT", os.path.dirname(os.getcwd()))os.environ[...
```

→ `SyntaxError: invalid syntax`.

Measured across all seven (parsed with `ast`, treating `!`/`%` lines as no-ops):

| Notebook | code cells | multi-line | raise SyntaxError |
|---|---|---|---|
| `00_cluster_setup.ipynb` | 7 | 6 | 4 |
| `01_pretraining.ipynb` | 4 | 4 | 3 |
| `02_finetuning.ipynb` | 5 | 5 | 3 |
| `03_generation.ipynb` | 4 | 4 | 2 |
| `04_evaluation.ipynb` | 2 | 2 | 1 |
| `05_docking.ipynb` | 2 | 2 | 1 |
| `06_run_full_pipeline.ipynb` | 4 | 4 | 2 |

The cells that *don't* raise only escape because the whole cell reduced to a single
shell-magic line. **Every multi-line cell is wrong.** This is strong evidence — alongside the
absence of any output from them — that **`notebooks/` has never been successfully executed**.

### 5.3 What each wrapper intends to do

| Notebook | Intent | Relationship to `code/` |
|---|---|---|
| `00_cluster_setup.ipynb` | GPU check, dependency check, data check; run `patch_paths.py`; run `convert_chembl_format.py`; `mkdir` outputs | **New behaviour** — no counterpart in `code/` |
| `01_pretraining.ipynb` | `subprocess.run(jupyter nbconvert --execute code/pretraining.ipynb)` → `pretraining_executed.ipynb`; then tail `log.out` | **Pure wrapper**, zero duplication |
| `02_finetuning.ipynb` | `exec()` the *prefix* of `pretraining.ipynb` up to `print("We are in training part....")` to get class definitions, then `exec()` `scripts/finetune_cell.py`, then `run_finetuning(DATASET_INDEX)` | **Wraps a reconstruction.** The only notebook that introduces genuinely new model code |
| `03_generation.ipynb` | nbconvert `code/Generating_samples.ipynb` | **Pure wrapper.** ⚠ Its `DATASET_INDEX / RUN / N_SAMPLES / MODEL_NAME` config cell is **decorative** — the subprocess re-reads the hardcoded values in `Generating_samples.ipynb` cell 19 |
| `04_evaluation.ipynb` | nbconvert `code/Evaluation_metrics.ipynb` | Pure wrapper |
| `05_docking.ipynb` | nbconvert `code/Molecular_docking.ipynb` | Pure wrapper |
| `06_run_full_pipeline.ipynb` | Chain of the above | ⚠ **Skips fine-tuning entirely** — prints "Run notebook 02_finetuning.ipynb for this stage" and continues. "Full pipeline" is a misnomer |

### 5.4 Inaccurate claims inside the generated notebooks

| Claim | Reality |
|---|---|
| `01`: "Uses all available GPUs via MXNet data parallelism" | `code/pretraining.ipynb` hardcodes `mx.gpu()` = device 0 only. No parallelism exists |
| `01`: "~6-12 hours for 480K iterations on 2× L40 GPUs" | Actual: **~46.7 h** on **2× NVIDIA A30** (only one used). No L40 appears in any job log |
| `02`: "Batch size 16 × 2 GPUs = 32 effective" | Single-GPU; the trainer never sees two devices |
| `06`: "Estimated time: 12-24 hours" | Pretraining alone took 46.7 h |
| `04`: "Computes validity, uniqueness, **novelty**, QED, SAS…" | Novelty is **not implemented** in `Evaluation_metrics.ipynb` |

### 5.5 What was inferred when the notebooks were generated

| Item | Inferred? | Evidence |
|---|---|---|
| Paths (`outputs/pretrain/logs/...`) | ✅ Correct | Match `pretraining.ipynb` after patching |
| Checkpoint location for fine-tuning | ✅ Correct | `outputs/pretrain/logs/ckpt.params` is right |
| Fine-tuning output location `outputs/CTDGD/Dataset{i}/model` | ✅ Correct | Matches what `generate_samples()` loads |
| Fine-tuning hyperparameters (50,000 iters, batch 16, lr 1e-4, clip 10.0, k=10, p=0.9) | ⚠ **Invented** | No source in repo or papers. CDGCN §3.3 says fine-tuning ran ~10 epochs with early stopping, batch size 8, α=0.8, K=5 |
| Execution order | ✅ Correct | Matches CDGCN §3.3 and the papers' two-phase description |
| Conditional architecture | ❌ **Reinvented incorrectly** | A correct version already existed in `Generating_samples.ipynb` |

## 6. `data/` and `outputs/`

Both git-ignored. Contents and statistics: [03](03_Data_and_Preprocessing.md);
checkpoint semantics: [06](06_Pretraining_and_Checkpoints.md).

⚠ `data/` contains its own `.git` pointing at `git://git-lfs.github.com/mshik/DGGNP.git` —
the CDGCN author's data repository. `data/data_preprocessing.ipynb` is the **upstream
CDGCN/DGGNP preprocessing notebook with its original executed outputs intact**, and is the
best available source of real dataset statistics.

## 7. Known execution flow and where it broke

```
Laptop / VS Code Remote SSH
        │
        ▼
bhavani.iitpkd.ac.in  (login node — no GPU)
        │  sbatch cluster/launch_jupyter_preferred.sh
        ▼
Slurm → gpu03 / node003  (2× A30 24 GB)
        │  module load cuda/11.2 anaconda3/2022.10
        │  conda activate ctddg_env ; export PATH/LD_LIBRARY_PATH
        ▼
Jupyter Server  (port 8800 + JOBID%200, no token)
        │  ← SSH tunnel from laptop
        ▼
kernel "ctddg_env"  →  jupyter nbconvert --execute code/pretraining.ipynb
        ▼
   480,000 iterations, ~46.7 h  →  outputs/pretrain/logs/
```

**Verified job:** 11973, node003, started 2026-09-02 22:49:55 IST, cancelled
2026-09-07 22:50:00 on the 5-day time limit. Training itself ran 2026-09-02 23:17 →
2026-09-04 21:40 (`outputs/pretrain/training_monitor.log`), finishing well inside the job.

**Exactly how nbconvert was invoked is NOT DETERMINED FROM AVAILABLE SOURCES.** It was not
`cluster/run_pipeline_batch.sh` (no `cluster/pipeline_*.log` exists) and it cannot have been
`notebooks/01_pretraining.ipynb` (broken). INFERRED — NOT EXPLICITLY CONFIRMED: run manually
from a terminal inside the Jupyter session.

**Preprocessing broke** at `data_preprocessing_1.ipynb` cell 37:

```python
for p1 in range(1, len(ds_len[i][0])+1):     # ds_len[i][0] is an int
```
→ `TypeError: object of type 'int' has no len()` (`preprocessing_11849.out`, job 11849,
node004, 2026-08-25 08:49). The `needle` computation *before* it completed successfully.

## 8. Local modifications to `code/` (uncommitted)

`git diff HEAD -- code/` shows changes to all five committed notebooks. Every change is one
of two kinds:

1. **Path rewrites** from `scripts/patch_paths.py` — `/workspace/...` → `/home/muhamed/repo/CTDDG/...`.
2. **A resume guard in `data_preprocessing_1.ipynb`** — the three `needle` loops gained
   `if not os.path.exists(output_file):` and switched `>>` to `>`.

**No model logic, hyperparameter, dataset or training behaviour has been modified locally.**
Verified by inspecting the full diff of `code/pretraining.ipynb`, which contains exactly
three changed lines, all paths.
