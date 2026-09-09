# PROJECT_CONTEXT

## Purpose
CTDDG = **Conditional Target-based Detoxed Drug Generation**. A graph-based deep generative
model that takes a **protein target** (amino-acid sequence) and generates novel drug-like
**molecular graphs** intended to bind it while being less toxic (Ames mutagenicity).

This is a **BTP / research reproduction** project at IIT Palakkad, run on the Bhavani cluster.
Immediate goal: reproduce the **single-target, single-property baseline**. Stated future
direction: **multi-target extension using CrossDocked2020**.

## Papers (papers/)
| File | Role |
|---|---|
| `CDGCN.pdf` | Mallick & Bhadra. **The architecture** — graph generator, protein conditioning, sequential decoding, loss. Code: github.com/mshik/CDGCN |
| `CTDDG.pdf` | Singh, Bhadra & Bhadra. **The extension** — reuses CDGCN's architecture verbatim, adds a toxicity-weighted loss (Eq. 8). Code: github.com/sahelybhadra/CTDDG |
| `../supplimentaryAPIN.pdf` | CTDDG supplementary — Table S1 (top-10 CDK2 SMILES), Table S2 (MD interactions) |

To understand *how the model works* → CDGCN. To understand *what CTDDG adds* → CTDDG §2.3–2.4.

## ⚠ The single most important fact
**The code implements CDGCN's loss, not CTDDG's.** Verified three ways:
1. `data/chembl/chembl_final.txt` — all 1,090,529 molecules labelled class `1` (dummy label
   from `scripts/convert_chembl_format.py`).
2. `process_single` parses `smiles_class` then hardcodes `tox_class = [1] * k`.
3. `MoleculeGenerator._likelihood` takes `tox_class_batch` and never uses it; `forward`
   returns `-l.mean()`.

The completed pretrained checkpoint is a **CDGCN unconditional generator**. Do not call it a
CTDDG model. Implementing Eq. 8 is the highest-value BTP opportunity.

## Repository structure
```
code/         ORIGINAL implementation (authors'). 6 notebooks, no .py modules.
              Model definitions duplicated across notebooks, not imported.
notebooks/    GENERATED wrappers 00–06 from cluster/generate_notebooks.py. ⚠ BROKEN — see below.
scripts/      Utility + reconstruction scripts (Week 1).
cluster/      Slurm scripts, Jupyter launchers, job logs.
data/         Datasets (git-ignored). Has a stale nested .git → mshik/DGGNP.
outputs/      Checkpoints & results (git-ignored).
papers/       The two PDFs.
PROJECT_DOCS/ Full human documentation (13 files).
```

## Original code vs generated notebooks
- `notebooks/*.ipynb` were **machine-generated** by `cluster/generate_notebooks.py`.
- ⚠ **They are structurally broken.** The generator does `src.split("\n")` without re-adding
  newlines, so Jupyter's `"".join(source)` collapses every multi-line cell into one line →
  `SyntaxError`. **Every multi-line cell in all 7 notebooks is affected. They have never run.**
- 01, 03, 04, 05 are thin `nbconvert` wrappers around `code/*.ipynb`.
- 03's config cell is decorative — the subprocess re-reads hardcoded values in
  `Generating_samples.ipynb` cell 19.
- 06 "full pipeline" **skips fine-tuning entirely**.
- 02 is the only one with new model code, and it wraps `scripts/finetune_cell.py`, which is
  non-functional.
- **Work directly with `code/*.ipynb`.**

## Model
- `VanillaMolGen_RNN` — **unconditional**, in `code/pretraining.ipynb`. **This is what was trained.**
- `CVanillaMolGen_RNN` — **conditional**, in `code/Generating_samples.ipynb` cell 14.
  Architecture exists; **no training loop for it exists anywhere.**
- 6,986,506 parameters (verified against the run's own printout). GRU is 79% of them.
- MXNet 1.9.1 Gluon, single GPU (`mx.gpu()` hardcoded).

## Datasets
| Dataset | Role | Numbers (verified) |
|---|---|---|
| ChEMBL | pretraining | `chembl.txt` **1,090,529** SMILES; `chembl.csv` 1,207,360 rows |
| BindingDB (Grechishnikova, via mshik/DGGNP) | fine-tuning | 5 folds; fold 1 = **1042 train / 104 test** proteins, **227,093 / 11,049** pairs |
| atom types | model spec | **65** (= 49 BindingDB ∪ 45 ChEMBL) — matches paper |
| CrossDocked2020 | future multi-target | `~/datasets/downsampled_CrossDocked2020_v1.3/`, 2,801 dirs, extracted |

## Training stages
1. **Pretraining** — ChEMBL, unconditional, NLL with importance sampling over K=5 paths. **✅ DONE: 480,000 iters, ~46.7 h.**
2. **Fine-tuning** — BindingDB pairs, conditional. **❌ NOT STARTED, BLOCKED.**
3. **Generation** → 4. **Evaluation** → 5. **Docking**. All ❌ NOT STARTED.

## Conditioning — what the condition actually is
**A single fixed-length real vector `c` derived from the protein's amino-acid sequence.**
Nothing else. NOT properties, NOT toxicity, NOT multi-target, NOT a 3D pocket.

Enters at two points in `CVanillaMolGen_RNN`:
- `dense_policy_0 = _TwoLayerDense(N_C, N_A*3, N_A)` → distribution over the **first atom**.
- `linear_c[i](c)[ids, :]` added to the output of **every** GCN layer (6 of them).
  `ids` maps each atom to its molecule — essential for variable-size molecules.

Encoder is **frozen and offline** — embeddings are computed in preprocessing and read back as
text. Never part of the MXNet graph.

## Terminology (preserve it)
`known protein` (has known drugs, in train) · `novel protein` (no known drug, in test) ·
`lead` · `generation path T` · `Q_α` (proposal over paths) · `α`/`p` = 0.8 (randomness) ·
`K`/`k` = 5 (paths per molecule) · `λ` = 1 (toxicity trade-off) · `S_N` (set of N molecules) ·
`target awareness` (% docking < −7.0 kcal/mol) · **Ames mutagenicity** (the toxicity endpoint) ·
actions: `init` / `append` / `connect` / `end`.

⚠ `beam_size` in the evaluation and docking notebooks means **number of stochastic samples**.
**There is no beam search in this codebase.**
⚠ `CTDGD` (in output paths) is a typo for CTDDG, but it is load-bearing across three notebooks.

## Constraints
- MXNet 1.9.1 is retired → the whole dependency stack is pinned and fragile. Do not upgrade
  NumPy (1.23.5; `np.bool` is used in `_decode_step` and was removed in NumPy ≥1.24).
- `conda activate` is **not sufficient** on Bhavani compute nodes — must also
  `export PATH="$CONDA_PREFIX/bin:$PATH"` and `LD_LIBRARY_PATH`.
- Single GPU only.
- `data/atom_types.txt` is effectively immutable (indices are line numbers; checkpoint depends on them).

## Known facts vs assumptions
**Verified:** pretraining complete (480,000, "Training finished", 46.65 h, 6,986,506 params) ·
toxicity loss not implemented · all ChEMBL labels are 1 · `notebooks/` broken ·
`finetune_cell.py` non-functional · needle alignments complete · 65 atom types · env working.

**NOT DETERMINED:** which `bio_embeddings` model/dimension the CTDDG authors truly used · how
`d{i}_te_embeddings_run{r}.txt` was produced · whether results use 104 or 52 test proteins ·
whether the paper means 166,400 or 66,400 pairs · exactly how pretraining was launched · what
wrote `training_monitor.log` · whether node004 has L40 GPUs · the ChEMBL export filters ·
compute-node internet access.
