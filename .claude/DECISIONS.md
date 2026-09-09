# DECISIONS

Only decisions with evidence in the repository (commits, job logs, `WEEK 1 PROGRESS - TASK 2.md`,
or files on disk). **Nothing here is inferred without being labelled.**

---

**DATE:** 2026-08 (Week 1)
**DECISION:** Target the **single-target, single-property baseline** first; defer the
multi-target extension.
**REASON:** The baseline runs on ChEMBL + BindingDB, which were already available; CrossDocked2020
(needed for multi-target) had failed to extract at the time.
**ALTERNATIVES:** Start directly on multi-target.
**IMPACT:** Defines the current scope. CrossDocked2020 has since extracted successfully
(`~/datasets/downsampled_CrossDocked2020_v1.3/`, 2,801 dirs), so the blocker is gone.
**SOURCE:** `WEEK 1 PROGRESS - TASK 2.md` §17

---

**DATE:** 2026-08 (Week 1)
**DECISION:** Build a dedicated conda env `ctddg_env` on Python 3.9 with `mxnet-cu112==1.9.1`
and `module load cuda/11.2`.
**REASON:** MXNet 1.9.1 is compiled against CUDA 11.2 and has no newer build. The driver's
"CUDA 12.1" is only the maximum it supports.
**ALTERNATIVES:** A newer CUDA module (no MXNet build exists); porting to PyTorch (would
invalidate the whole codebase).
**IMPACT:** Pins the entire dependency stack (NumPy 1.23.5, sklearn 0.24.2, torch 1.9.1+cu102).
Do not upgrade NumPy — `np.bool` is used in `_decode_step` and was removed in ≥1.24.
**SOURCE:** `WEEK 1 PROGRESS - TASK 2.md` §3–§4; verified in the installed env

---

**DATE:** 2026-08 (Week 1)
**DECISION:** Export `PATH` and `LD_LIBRARY_PATH` explicitly in every job script, rather than
relying on `conda activate`.
**REASON:** The cluster's system Anaconda silently shadows the environment on compute nodes,
producing misleading `No module named 'mxnet'` errors while the install was fine.
**ALTERNATIVES:** `conda init` (modifies shell config permanently, less portable).
**IMPACT:** Every job script in `cluster/` carries this block. Omitting it is the single most
likely cause of a confusing failure.
**SOURCE:** `WEEK 1 PROGRESS - TASK 2.md` §5, §6, §20

---

**DATE:** 2026-08-16
**DECISION:** Keep the repository (`~/repo/CTDDG`) and the datasets (`~/datasets/`) separate;
`.gitignore` `data/`, `outputs/`, `Jupyter_Dock/`.
**REASON:** Multi-GB archives must never enter git history.
**ALTERNATIVES:** git-lfs (as the upstream `mshik/DGGNP` repo does).
**IMPACT:** Data must be transferred manually. ⚠ `.gitignore` also excludes `*.json` and
`*.params`, so `configs.json` and checkpoints are unprotected by git.
**SOURCE:** commit `28c9b75`; `WEEK 1 PROGRESS - TASK 2.md` §16

---

**DATE:** 2026-08-16 (commit `adb684b`)
**DECISION:** Non-invasive fixes only — patch paths and add missing scaffolding, but do not
alter model logic, hyperparameters or training behaviour.
**REASON:** "so that the reproduction remains a faithful reproduction".
**ALTERNATIVES:** Refactor the notebooks into modules.
**IMPACT:** ✅ **Held.** `git diff HEAD -- code/pretraining.ipynb` contains exactly three changed
lines, all paths. The research implementation is intact.
**SOURCE:** `WEEK 1 PROGRESS - TASK 2.md` §12

---

**DATE:** 2026-08-16 (commit `adb684b`)
**DECISION:** Satisfy the pretraining loader's `smiles.split(" ")` by appending a **dummy** `" 1"`
label to every ChEMBL SMILES.
**REASON:** The label is discarded downstream (`tox_class = [1]*k`), so a dummy changes nothing
about training behaviour while unblocking the loader.
**ALTERNATIVES:** Train the paper's Random Forest Ames classifier and write real labels.
**IMPACT:** ⚠ **Consequential.** The completed 480,000-iteration run therefore has **no toxicity
signal**. The checkpoint implements CDGCN's objective, not CTDDG's. This is now the clearest
BTP opportunity (`PROJECT_DOCS/12` §1).
**SOURCE:** `scripts/convert_chembl_format.py` docstring; `WEEK 1 PROGRESS - TASK 2.md` §11.2;
verified — all 1,090,529 labels are `1`

---

**DATE:** 2026-08-16 (commit `adb684b`)
**DECISION:** Reconstruct the missing conditional fine-tuning stage in `scripts/finetune_cell.py`,
deriving the architecture from the two existing endpoints.
**REASON:** The repository has pretraining and generation but no code that trains the conditional
model.
**ALTERNATIVES:** Reuse `CVanillaMolGen_RNN`, which **already exists** in
`code/Generating_samples.ipynb` cell 14.
**IMPACT:** ⚠ **The reconstruction is non-functional and architecturally divergent** — 5 runtime
blockers plus 5 semantic differences (5 vs 6 `linear_c` layers, `gluon.nn.BatchNorm` vs the
custom `BatchNorm`, shifted BN placement, `nd.repeat` instead of the `ids` gather, double
softmax). A checkpoint from it would not load in `Generating_samples.ipynb`.
**RECOMMENDATION:** Do not repair it; write a fresh loop reusing the existing architecture.
**SOURCE:** `WEEK 1 PROGRESS - TASK 2.md` §11.1, §13; audit 2026-09-08 (`PROJECT_DOCS/07` §5)

---

**DATE:** 2026-08-17 → 2026-08-25
**DECISION:** Add a resume guard (`if not os.path.exists(output_file)`, and `>` instead of `>>`)
to the three `needle` loops.
**REASON:** Job 11771 was killed by the Slurm time limit mid-alignment; restarting from scratch
would have wasted days. `>>` also risked appending to partial files.
**ALTERNATIVES:** A longer walltime; parallelising `needle`.
**IMPACT:** ✅ Worked — all 5,809,835 alignments are now complete. This is an **uncommitted**
local modification to `code/data_preprocessing_1.ipynb`.
**SOURCE:** `git diff HEAD -- code/data_preprocessing_1.ipynb`; `preprocessing_11771.err`

---

**DATE:** 2026-09-02
**DECISION:** Run pretraining inside a long-lived Jupyter Slurm job on node003 (partition
gpu03, walltime `4-23:59:59`) rather than as a plain batch job.
**REASON:** node003/gpu03 is the only partition in `launch_jupyter_preferred.sh` allowing
almost 5 days — enough headroom for a ~47-hour run.
**ALTERNATIVES:** `sbatch cluster/run_pipeline_batch.sh` (72 h, but nbconvert timeout 48 h).
**IMPACT:** ✅ Succeeded. Training ran 2026-09-02 23:17 → 09-04 21:40. The job was later killed
by the time limit, long after training had saved.
**SOURCE:** `cluster/jupyter_11973.{log,err}`, `outputs/pretrain/training_monitor.log`

---

**DATE:** 2026-09-02
**DECISION:** Run pretraining for the hardcoded **480,000** iterations rather than the computed
681,580 (= 5 epochs).
**REASON:** 480,000 is the value hardcoded in the authors' own notebook, overriding the
computed expression.
**ALTERNATIVES:** 681,580 (≈5 epochs).
**IMPACT:** ≈3.52 epochs over ChEMBL, ~46.7 h. Neither paper states a pretraining iteration
count, so this cannot be checked against the literature.
**SOURCE:** `code/pretraining.ipynb`; `outputs/pretrain/logs/log.out`

---

**DATE:** 2026-09-02
**DECISION:** Generate the `notebooks/00–06` wrapper series via `cluster/generate_notebooks.py`.
**REASON:** Give the pipeline a documented, ordered, cluster-aware entry point.
**ALTERNATIVES:** Shell scripts only.
**IMPACT:** ⚠ **The generator emits structurally invalid notebooks** (`src.split("\n")` drops the
newlines that `nbformat` requires), so every multi-line cell collapses to one line and raises
`SyntaxError`. None of them has ever run. Work directly with `code/*.ipynb`, or fix the
generator and regenerate.
**SOURCE:** `cluster/generate_notebooks.py`; verified by `ast.parse` across all 7 notebooks

---

**DATE:** 2026-09-08
**DECISION:** Produce `PROJECT_DOCS/` (human documentation) and `.claude/` (machine context)
from a full audit, and change **nothing** in the research implementation.
**REASON:** The project was runnable but not understood; several defects needed recording
before any modification could be reasoned about.
**ALTERNATIVES:** Fix the defects immediately.
**IMPACT:** All findings are documented; `code/`, `notebooks/`, `scripts/` and `data/` are
untouched. The identified fixes are scoped in `PROJECT_DOCS/12_BTP_Modification_Guide.md` but
deliberately not applied.
**SOURCE:** this audit

---

**DATE:** 2026-09-08
**DECISION:** Add checkpoint protection to `code/pretraining.ipynb` (`CTDDG_RUN_NAME`
directories + a refuse-to-overwrite guard + an automatic pre-resume snapshot).
**REASON:** The notebook auto-resumes and re-saves, so re-running it on the completed 480,000
-iteration run would have destroyed ~47 GPU-hours of work. `notebooks/06` and
`cluster/run_pipeline_batch.sh` both invoked it unconditionally.
**ALTERNATIVES:** Rely on manual `cp -r` backups (error-prone); make the notebook read-only
(would block legitimate resumes).
**IMPACT:** The only change to the research implementation so far beyond path patching.
Training behaviour, hyperparameters, model and loss are untouched — the block runs before the
dataset is even read and either raises or falls through. All four paths tested
(finished/refuse, force/snapshot, new-run/fresh, unfinished/resume+snapshot).
⚠ `run_pipeline_batch.sh` (`set -euo pipefail`) will now abort at pretraining instead of
clobbering — intended. ⚠ ~81 MB snapshot per launch.
**SOURCE:** `code/pretraining.ipynb` config block; archive at
`outputs/pretrain/archive/base_cdgcn_step480000_FINISHED_20260908-235622/`

