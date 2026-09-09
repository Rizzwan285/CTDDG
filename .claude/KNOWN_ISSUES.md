# KNOWN_ISSUES

Severity: 🔴 blocker · 🟠 high · 🟡 medium · ⚪ low/cosmetic
All verified during the 2026-09-08 audit unless marked otherwise.

---

## 🔴 Blockers

### 1. Protein embeddings are per-residue, not per-protein
`code/data_preprocessing_1.ipynb` cell 75 calls `embedder.embed(protein)` → shape `(L, 1024)`.
The fixed-size vector needs `reduce_per_protein` (a mean over axis 0), which is never called.
Cell 78 then writes `str(e)` where `e` is a 1024-vector → `'[0.1 0.2 ... 0.9]'`, bracketed and
NumPy-truncated → unparseable by the consumer's `float(e_i)`.
**Evidence:** installed `bio_embeddings/embed/prottrans_base_embedder.py`; upstream
`data/data_preprocessing.ipynb` cell 79 output `6165` (ProSE, avg-pooled).
**Fix:** one line — `embedder.reduce_per_protein(embedder.embed(protein))`, plus a proper float writer.
**Blocks:** everything downstream of pretraining.

### 2. No conditional fine-tuning loop exists
The architecture (`CVanillaMolGen_RNN`) exists in `code/Generating_samples.ipynb` cell 14, but
there is no conditional data loader and no training loop anywhere in `code/`.

### 3. `scripts/finetune_cell.py` is non-functional
Five independent runtime blockers:
1. `_TwoLayerDense` undefined (it lives only in `Generating_samples.ipynb`) → `NameError`
2. `_build_graph_conv` never creates `bn_skip` / `linear_skip`, but `_graph_conv_forward` uses them → `AttributeError`
3. `if len(parts) == 2` — a real line has `1 + N_C` tab-separated fields → dataset always empty
4. `batch_size_sampler=` is not a valid kwarg (`batch_sampler=`) → `TypeError`
5. `PROJECT_ROOT = "/workspace"` does not exist on Bhavani → `FileNotFoundError`

Plus five architectural divergences that would make any resulting checkpoint incompatible with
`Generating_samples.ipynb`: 5 vs **6** `linear_c` layers · `gluon.nn.BatchNorm` vs the custom
`BatchNorm` (different parameter names) · BN placement shifted by one layer ·
`nd.repeat(..., X.shape[0] // c.shape[0])` instead of `linear_c(c)[ids, :]` (**wrong for
variable-size molecules**) · double softmax in `_policy_0`, dropping the `+ 0.0*policy_0` term.
**Recommendation: do not repair — rewrite reusing the existing architecture.**

### 4. `data/d{i}_te_embeddings_run{r}.txt` is produced by nothing
`Generating_samples.ipynb` cell 19 reads it; no code in the repository writes it.
The preprocessing notebook writes `d{i}_te_unique_bioembeddings.txt` inside the fold
directories instead. **NOT DETERMINED FROM AVAILABLE SOURCES** how the authors produced it.

---

## 🟠 High

### 5. The toxicity loss (CTDDG's whole contribution) is not implemented
- `data/chembl/chembl_final.txt`: **all 1,090,529 labels are `1`** (dummy, from `convert_chembl_format.py`)
- `process_single`: parses `smiles_class`, then hardcodes `tox_class = [1] * k`
- `MoleculeGenerator._likelihood`: takes `tox_class_batch`, **never references it**
- `forward`: returns `-l.mean()` = CDGCN Eq. 6, not CTDDG Eq. 8
- The paper's Random Forest Ames classifier does not exist — only a bare `# Load ML classifier` comment

**The trained checkpoint is a CDGCN model.** This is also the best BTP opportunity.

### 6. `notebooks/*.ipynb` are structurally broken
`cluster/generate_notebooks.py` does `src.split("\n")` without re-adding newlines. Jupyter
joins `source` with `""`, so every multi-line cell collapses to one line → `SyntaxError`.
Affects **all 7 notebooks**; none has ever been executed.
**Fix:** in the generator, `[l + "\n" for l in src.split("\n")]` (last line without), then regenerate.

### 7. Preprocessing crashes at the similarity histogram
`code/data_preprocessing_1.ipynb` cell 37: `range(1, len(ds_len[i][0])+1)` where `ds_len[i][0]`
is an `int` → `TypeError: object of type 'int' has no len()`.
Observed in `preprocessing_11849.out` (job 11849). **Original-code bug**, not a porting artefact.
Only affects plotting — the alignments themselves are complete.

### 8. The CDGCN dataset section re-inserts SMILES tokenisation spaces
Ligands are loaded correctly (cell 40 strips spaces; cell 42 salt-strips), but
`code/data_preprocessing_1.ipynb` cell 100 puts them back:
`x[0] + x[1:-1].replace('', ' ') + x[-1]` turns `CCO` into `C C O`. The line is copied from the
Grechishnikova section (cell 93), where character tokenisation is correct. The graph model
needs plain SMILES, and `process_single`'s `smiles.split(" ")` would shred it.
Upstream `data/data_preprocessing.ipynb` cell 102 round-trips through `.space_sep_seq` and
strips on read.

### 9. ✅ FIXED 2026-09-08 — re-running pretraining used to overwrite the finished checkpoint
**Was:** `is_continuous` became True, `global_counter = 480000`, one extra step ran, then
`ckpt.params` / `trainer.status` were rewritten. `notebooks/06_run_full_pipeline.ipynb` and
`cluster/run_pipeline_batch.sh` triggered this unconditionally.

**Now:** `code/pretraining.ipynb` has a checkpoint-protection block (config section) that
(1) gives each experiment its own directory via `CTDDG_RUN_NAME`, (2) **raises `RuntimeError`**
if the target directory holds a run whose `log.out` ends with `Training finished`, and
(3) snapshots any existing checkpoint into `outputs/pretrain/archive/` before resuming it.
Because every caller executes this notebook, all of them are protected.
⚠ Side effect: `run_pipeline_batch.sh` (which has `set -euo pipefail`) will now **abort** on
the pretraining stage rather than silently clobbering — this is intended.
⚠ Each launch writes one ~81 MB snapshot; prune `outputs/pretrain/archive/` occasionally.
A write-protected copy of the 480k run is at
`outputs/pretrain/archive/base_cdgcn_step480000_FINISHED_20260908-235622/` (md5-verified).

### 10. `data/atom_types.txt` is order-dependent and not reproducible
`MoleculeSpec.get_atom_type` returns `self.atom_types.index(...)` — the integer is the **line
number**. The checkpoint's `embedding_atom` (65×16) and `policy_0` (65) are tied to it. The
generator builds the file from a Python `set()`, so regeneration is not order-stable.
**Treat as immutable.**

---

## 🟡 Medium

### 11. Evaluation loop bounds are inconsistent
`Evaluation_metrics.ipynb` cells 11/12 iterate `range(1, 11)` (10 proteins); cell 13 iterates
`range(nprot)` with `nprot = 104` → `IndexError`. **NOT DETERMINED** which the reported results used.

### 12. Property percentages are compounded with uniqueness
`100 × (unique/N) × (satisfying/N)` — **not** the fraction of molecules satisfying the property.
Applies to logP, MW, HD, HA, RB, TPSA, SAS (not QED). Not comparable to differently-defined
literature numbers.

### 13. Novelty and `R_c` are not implemented
Both are reported in the papers (CDGCN Table 1: 99.7% novel). ⚠ `notebooks/04_evaluation.ipynb`
falsely claims novelty is computed.

### 14. No score parsing in the docking notebook
`Molecular_docking.ipynb` writes `*_smina_out.pdbqt` and stops. The papers' target-awareness
metric (% < −7.0 kcal/mol) and top-10 ranking cannot be reproduced without new code.
The notebook is also interactive-only: hardcodes `beam=0, run=3, prot_id=6, pdb_id='1cqp'`,
and `os.chdir` + relative paths.

### 15. `log.out` time column is mislabelled
Header says `time(h)`; the code writes `(time.time() - t0) / 60` = **minutes**. Confirmed by
the resume logic (`t0 = time.time() - t_final*60`) and by `training_monitor.log` wall-clock.
Read 2799.06 as ≈46.65 hours.

### 16. Paper/code discrepancies
| # | Paper | Code / data |
|---|---|---|
| a | Toxicity-weighted loss Eq. 8 | not implemented |
| b | RF Ames classifier for ChEMBL | absent |
| c | pkCSM labels for BindingDB | absent |
| d | "166400 pairs" (§2.1) vs "66400 pairs" (§2.4) | fold 1 has 227,093 + 11,049 = 238,142; no fold matches either |
| e | `Embed_p` = "Dallago et al." (a toolkit, unnamed model, no dimension) | CDGCN used **ProSE/Bepler 6165-d**; CTDDG uses **ProtBert-BFD 1024-d** |
| f | "default docking parameters" | `--exhaustiveness 16` (smina's default is 8) |
| g | Results on 104 test proteins | preprocessing halves 104 → 52 val + 52 test |
| h | Novelty, `R_c` reported | not implemented |

### 17. Missing external assets
`data/fpscores.pkl.gz` (SAS) · `Jupyter_Dock/` · `pymol`, `openbabel`, `vina`, `meeko`,
`prolif`, `MDAnalysis`, `py3Dmol`.

### 18. Output path conventions conflict
`generate_samples` → `outputs/CTDGD/Dataset{i}/model/`;
`Evaluation_metrics.ipynb` cell 16 reads `outputs/finetune/log.out`;
`scripts/config.py` proposes `outputs/{model_name}/Dataset{i}/model`. Pick one.

### 19. Dependency fragility
MXNet 1.9.1 is retired. `np.bool` in `_decode_step` was removed in NumPy ≥1.24 (env has 1.23.5).
`sklearn` 0.24.2 is very old. Do not upgrade casually.

### 20. `notebooks/` contain false performance claims
"all available GPUs via MXNet data parallelism" (single GPU only) · "6–12 hours on 2× L40"
(actual: 46.7 h on 1× A30) · "batch 16 × 2 GPUs = 32 effective" · "12–24 hours" for the full
pipeline. No L40 appears in any job log.

---

## ⚪ Low / cosmetic

### 21. `samples_evaluation.txt` timing is nonsense
`startTime` in ms, `endTime` in s → the reported difference is meaningless.

### 22. `molvs.standardize_smiles` imported but never called
`Generating_samples.ipynb` cell 2. Dead import.

### 23. `getbox()` result computed then discarded
`Molecular_docking.ipynb` cell 40 — `--autobox_ligand` makes smina compute the box itself.

### 24. The second atom is always greedy
`_decode_step`'s `A.shape[0] == 0` branch uses `argmax` even when `random=True`. Molecules
sharing a first atom also share their second. **INFERRED — NOT EXPLICITLY CONFIRMED** whether intentional.

### 25. Jupyter runs with no authentication
`launch_jupyter.sh` uses `--NotebookApp.token="" --NotebookApp.password="" --ip=0.0.0.0`.
Anyone on the cluster network can execute code as you.

### 26. 2 GPUs requested, 1 used
`--gres=gpu:2` throughout, but `mx.gpu()` is device 0 only. Wastes allocation and queue time.

### 27. `scripts/config.py` is unused
Nothing imports it; the notebooks still carry absolute paths patched by `patch_paths.py`.

### 28. Patch-script ordering is implicit
`patch_notebooks.py` matches `/workspace/…` strings, so it must run **before** `patch_paths.py`.
Nothing enforces this.

### 29. `data/` contains a stale nested `.git`
Remote `git://git-lfs.github.com/mshik/DGGNP.git`. Do not `git add data/`.

### 30. `CTDGD` typo in output paths
Load-bearing across `Generating_samples.ipynb`, `Evaluation_metrics.ipynb`,
`Molecular_docking.ipynb`. Changing it means changing all three.

### 31. `nbconvert` timeout too short in `run_pipeline_batch.sh`
172,800 s (48 h) against a 47-hour pretraining run leaves almost no margin. Prefer `-1`.

---

## Unresolved questions
1. Which `bio_embeddings` model/dimension did the CTDDG authors actually use?
2. How was `d{i}_te_embeddings_run{r}.txt` produced?
3. Do the papers' results use 104 test proteins or the 52-protein half?
4. 166,400 or 66,400 BindingDB pairs? Neither matches any fold.
5. Was the toxicity loss ever implemented in the authors' private code?
6. Exactly how was pretraining launched? (not `run_pipeline_batch.sh`, not `notebooks/01`)
7. What wrote `outputs/pretrain/training_monitor.log`?
8. Does node004 have L40 GPUs? (every logged model is A30)
9. What physicochemical filters produced `chembl.csv`?
10. Do compute nodes have outbound internet (needed for `cmd.fetch`)?
