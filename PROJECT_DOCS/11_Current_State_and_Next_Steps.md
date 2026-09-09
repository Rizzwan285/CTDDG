# 11 — Current State and Next Steps

**Snapshot date:** 2026-09-08
**Repository:** `/home/muhamed/repo/CTDDG` (branch `main`, HEAD `422c8fc`)

---

## 1. Status at a glance

| Component | Status | Evidence |
|---|---|---|
| Cluster access (Bhavani) | **DONE** | 7 Slurm jobs in `cluster/` |
| `ctddg_env` conda environment | **DONE** | Python 3.9.25, MXNet 1.9.1, verified by direct query |
| MXNet GPU | **DONE** | 2 × A30 detected; matmul test passed |
| Repository / path patching | **DONE** | `git diff HEAD -- code/` shows patched paths |
| ChEMBL data | **DONE** | `chembl.csv` 295 MB, `chembl.txt` 1,090,529 lines |
| ChEMBL label conversion | **DONE (dummy labels)** | `chembl_final.txt` — all class `1` |
| BindingDB data (5 folds) | **DONE** | 1042/104 etc. FASTA files present |
| `atom_types.txt` | **DONE** | 65 lines |
| Needleman-Wunsch similarity | **DONE** | 5,809,835 alignment files; counts match exactly |
| Similarity histograms | **NOT STARTED** (blocked by a bug) | `preprocessing_11849.out` — `TypeError` |
| **Protein embeddings** | **NOT STARTED** | no `*bioembeddings*` file exists |
| **Paired fine-tuning file** `d1_tr_cdgcn.txt` | **NOT STARTED** | absent |
| **Pretraining** | **✅ DONE** | **480,000 / 480,000 · `Training finished`** |
| Pretrained checkpoint | **DONE** | `ckpt.params` 27.9 MB, `trainer.status` 55.9 MB |
| Checkpoint validation (sampling test) | **NOT STARTED** | never sampled from |
| **Fine-tuning** | **NOT STARTED — BLOCKED** | no loader, no loop, no data; reconstruction non-functional |
| Generation | **NOT STARTED** | needs a fine-tuned checkpoint |
| Evaluation | **NOT STARTED** | needs molecules + `fpscores.pkl.gz` (absent) |
| Docking | **NOT STARTED** | needs molecules + `Jupyter_Dock/` + 6 packages (all absent) |
| `notebooks/` wrappers | **BROKEN** | every multi-line cell is malformed |
| CrossDocked2020 (future work) | **DONE (extracted)** | `~/datasets/downsampled_CrossDocked2020_v1.3/` — 2,801 dirs |

---

## 2. The completed pretraining run

```
480000	2799.0602951129276	15.987691879272476	8.210006192943532e-06
Training finished
```

| Property | Value |
|---|---|
| Iterations | **480,000 / 480,000** |
| Wall time | **2799.06 minutes ≈ 46.65 hours ≈ 1.94 days** |
| Log records | 960 (one per 500 steps) + header + terminator = 962 lines |
| Final logged loss | 15.99 (single-batch, noisy — see [06](06_Pretraining_and_Checkpoints.md) §8) |
| Final learning rate | 8.210006192943532e-06 |
| Parameters | **6,986,506** |
| Node / GPU | node003 · 1 × NVIDIA A30 (2 allocated, 1 used) |
| Slurm job | 11973, partition gpu03 |
| Dates | 2026-09-02 23:17 → 2026-09-04 21:40 IST |
| Model class | `VanillaMolGen_RNN` — **unconditional** |
| Objective | plain NLL (CDGCN Eq. 6) — **not** CTDDG's toxicity-weighted Eq. 8 |
| Artefact | `code/pretraining_executed.ipynb` (965 outputs) |

⚠ **The time column in `log.out` is labelled `time(h)` but contains MINUTES.** Confirmed from
the source (`float(time.time() - t0)/60`), from the resume logic, and against
`training_monitor.log` wall-clock. See [06](06_Pretraining_and_Checkpoints.md) §8.

⚠ **This is a CDGCN model, not a CTDDG model** — the toxicity mechanism is inert
([06](06_Pretraining_and_Checkpoints.md) §7). The checkpoint is valid and useful; it simply is
not what the paper's title describes.

---

## 3. Key paths

```
Repository        /home/muhamed/repo/CTDDG
Conda env         /home/muhamed/.conda/envs/ctddg_env       (Python 3.9.25, MXNet 1.9.1)
Pretrain output   outputs/pretrain/logs/{ckpt.params, trainer.status, configs.json, log.out}
ChEMBL            data/chembl/{chembl.csv, chembl.txt, chembl_final.txt}
BindingDB         data/bindingdb/{train,test}_dataset/*_4_org_*/
Atom types        data/atom_types.txt                        (65 lines — DO NOT REGENERATE)
Papers            papers/{CTDDG.pdf, CDGCN.pdf}
Supplementary     supplimentaryAPIN.pdf
Upstream prep nb  data/data_preprocessing.ipynb              (has original executed outputs)
CrossDocked       ~/datasets/downsampled_CrossDocked2020_v1.3/
ProtBert cache    ~/.cache/bio_embeddings/prottrans_bert_bfd/
```

---

## 4. What is blocking progress

A single dependency chain. Item 1 blocks everything.

```
1. Protein embeddings are not produced, and the producer is defective
   (give_bioembeddings omits reduce_per_protein → (L,1024) not (1024,);
    the writer str()s a numpy array)                          [04 §4]
        │
        ▼
2. d1_tr_cdgcn.txt cannot be assembled
   (also: the CDGCN section writes space-separated SMILES)    [03 §6.1]
        │
        ▼
3. Fine-tuning cannot run
   (no loader, no loop; scripts/finetune_cell.py has 5
    independent blockers and diverges from the real
    architecture in Generating_samples.ipynb)                 [07 §5]
        │
        ▼
4. No conditional checkpoint → no generation                  [08 §9]
        │
        ▼
5. No molecules → no evaluation, no docking                   [09]
```

Independent side-blockers (not on the critical path):
- `data/fpscores.pkl.gz` absent → SAS metric fails.
- `Jupyter_Dock/` and six packages absent → docking impossible.
- `data/d{i}_te_embeddings_run{r}.txt` produced by no code → generation input undefined.

---

## 5. What should I do next?

Ordered, with prerequisites verified. Steps 0–2 are cheap and de-risk everything after.

### Step 0 — ✅ DONE (2026-09-08): the checkpoint is protected
A write-protected, md5-verified copy now lives at
`outputs/pretrain/archive/base_cdgcn_step480000_FINISHED_20260908-235622/`, and
`code/pretraining.ipynb` now **refuses** to overwrite a finished run
([06](06_Pretraining_and_Checkpoints.md) §9).

To start a new pretraining experiment, give it a name:
```bash
export CTDDG_RUN_NAME=exp_toxloss      # -> outputs/pretrain/exp_toxloss/logs/
```

### Step 1 — Validate the pretrained checkpoint (half a day)
Write a small unconditional sampler (`Vanilla_RNN_Builder`, ~30 lines, mirroring
`CVanilla_RNN_Builder` but calling `_policy_0(ctx)` instead of `_policy_0(c)`), sample ~500
molecules, and measure RDKit validity.

**Why before anything else:** 46 hours were spent on this checkpoint and nothing has ever been
sampled from it. If validity is near CDGCN's reported ~95%, everything downstream is worth
building. If it is near zero, you have found a much bigger problem cheaply.
Details: [06](06_Pretraining_and_Checkpoints.md) §11, [12](12_BTP_Modification_Guide.md) §8.

### Step 2 — Fix and run the protein embeddings (1 day + compute)
Two edits in `code/data_preprocessing_1.ipynb`:
- cell 75: `embedding_list.append(embedder.reduce_per_protein(embedder.embed(protein)))`
- cell 78 (and the test-set equivalent): write floats, e.g.
  `f.write('\t'.join(str(float(x)) for x in embed) + '\n')`

Then run **only** the embedding cells for fold 1 (1042 train + 104 test proteins).

**Why:** this is the one-line root cause of the whole blockage
([04](04_Protein_Embeddings.md) §4). ProtBert weights are already cached, so no download is
needed. ⚠ Do **not** re-run the whole notebook — it would recompute 5.8 M needle alignments
and re-derive `atom_types.txt` (which must not change,
[03](03_Data_and_Preprocessing.md) §4).

⚠ Decide and record: **ProtBert-BFD (1024-d)** as the CTDDG code does, or **ProSE/Bepler
(6165-d)** as upstream CDGCN did. They are not interchangeable — `N_C` propagates into
`configs.json` and every conditioning layer. ProtBert is the pragmatic choice (installed,
cached, matches the CTDDG authors' committed code).

### Step 3 — Build the paired fine-tuning file (half a day)
Run the CDGCN section (cells 99–107) for fold 1, with the space-separation fixed — mirror
upstream `data/data_preprocessing.ipynb` cell 102 (`line.strip().replace(' ', '')`).

**Verify before proceeding:**
```bash
head -1 data/bindingdb/train_dataset/train_4_org_1042_104/d1_tr_cdgcn.txt | cut -c1-120
awk -F'\t' '{print NF}' data/bindingdb/train_dataset/train_4_org_1042_104/d1_tr_cdgcn.txt \
  | sort -u                      # expect a single value: 1 + N_C
wc -l data/bindingdb/train_dataset/train_4_org_1042_104/d1_tr_cdgcn.txt   # expect ~227,093
```
Field 1 must be a **space-free** SMILES; fields 2…N_C+1 must all parse as floats.

### Step 4 — Write a correct conditional fine-tuning stage (2–4 days)
Scoped in [07](07_Finetuning_and_Conditioning.md) §7. Essentials:
- **Reuse** `CVanillaMolGen_RNN` from `code/Generating_samples.ipynb`. Do **not** repair
  `scripts/finetune_cell.py` — it reimplements the architecture incorrectly ([07](07_Finetuning_and_Conditioning.md) §5.6).
- Write `CMolRNNLoader` that yields `(smiles, embedding)` and appends **both** `c` and the
  per-atom `ids` tensor (`NX_rep`).
- Load with `allow_missing=True, ignore_extra=True`.
- Follow the papers where they are explicit: `k=5`, `p=0.8`, `batch_size=8`.
- **Write `N_C` into `configs.json`** — generation fails without it
  ([08](08_Generation.md) §2).
- Mirror pretraining's `log.out` / `ckpt.params` / `trainer.status` conventions so resume and
  the evaluation plot both work.
- Start with a **short smoke run** (e.g. 200 iterations) and confirm the loss is finite and
  decreasing before committing a multi-day job.

### Step 5 — Generate, then evaluate
Generation ([08](08_Generation.md)); then `wget` `fpscores.pkl.gz` and reconcile the
`range(1,11)` vs `nprot=104` loop bounds ([09](09_Evaluation_and_Docking.md) §A.2).

### Step 6 — Docking (optional, only for the five PDB targets)
Install the toolchain and clone Jupyter_Dock. Note the docking notebook is an interactive
worksheet for one protein, not a batch stage ([09](09_Evaluation_and_Docking.md) §B.8).

---

## 6. Things NOT to do

| Don't | Why |
|---|---|
| **Re-run `code/pretraining.ipynb`** without backing up first | Resumes at 480,000 and overwrites the checkpoint |
| **Run `notebooks/06_run_full_pipeline.ipynb` or `cluster/run_pipeline_batch.sh`** | Both re-invoke pretraining unconditionally; the notebooks are broken anyway |
| **Regenerate `data/atom_types.txt`** | Built from a Python `set()`, so line order is not reproducible; the checkpoint's atom embeddings are tied to it |
| **Re-run the whole preprocessing notebook** | Would redo 5.8 M needle alignments and touch `atom_types.txt` |
| **Try to fix `notebooks/*.ipynb` by hand** | Regenerate from `cluster/generate_notebooks.py` after fixing its `src.split("\n")`, or just run `code/*.ipynb` directly |
| **Repair `scripts/finetune_cell.py`** | The architecture it defines is not checkpoint-compatible with `Generating_samples.ipynb` |
| **Report docking scores as binding affinities** | [09](09_Evaluation_and_Docking.md) §B.1 |
| **Describe the current checkpoint as "CTDDG"** | It implements the CDGCN objective |

---

## 7. Open questions

| # | Question | Status |
|---|---|---|
| 1 | Which `bio_embeddings` model and dimension did the CTDDG authors actually use? | **NOT DETERMINED** — papers say only "Dallago et al."; code says ProtBert (unpooled); upstream used ProSE 6165-d |
| 2 | How was `data/d{i}_te_embeddings_run{r}.txt` produced? | **NOT DETERMINED** — no code produces it |
| 3 | Do the papers' results use 104 test proteins or the 52-protein half-split? | **NOT DETERMINED** — `Evaluation_metrics.ipynb` sets `nprot=104` |
| 4 | Does the paper mean 166,400 or 66,400 BindingDB pairs? Neither matches any fold | **NOT DETERMINED** — internal contradiction in `papers/CTDDG.pdf` |
| 5 | Was CTDDG's toxicity loss ever implemented in the authors' private code? | **NOT DETERMINED** — the published repository has only the plumbing |
| 6 | How exactly was pretraining launched? | **NOT DETERMINED** — not via `run_pipeline_batch.sh` (no log) nor `notebooks/01` (broken). INFERRED: manual `nbconvert` |
| 7 | What wrote `outputs/pretrain/training_monitor.log`? | **NOT DETERMINED** — no script in the repo emits it |
| 8 | Does node004 have L40 GPUs? | **NOT DETERMINED** — every logged GPU model is A30 |
| 9 | What physicochemical filters produced `chembl.csv`? | **NOT DETERMINED** — applied at ChEMBL export; thresholds unrecorded |
| 10 | Do compute nodes have outbound internet (needed for `cmd.fetch`)? | **NOT DETERMINED** |

---

## 8. Realistic timeline to a first end-to-end result

| Step | Effort |
|---|---|
| 0. Back up checkpoint | 5 min |
| 1. Validate pretrained checkpoint | 0.5 day |
| 2. Fix + run protein embeddings | 1 day + ~1–3 h GPU |
| 3. Build `d1_tr_cdgcn.txt` | 0.5 day |
| 4. Write + smoke-test fine-tuning | 2–4 days |
| 5. Fine-tuning run | 1–3 days GPU |
| 6. Generation (104 proteins × 100 samples) | ~1 h GPU |
| 7. Evaluation | 0.5 day |
| **Total to first evaluated numbers** | **≈ 1.5–2.5 weeks** |

Docking adds several days, mostly toolchain installation.

Only after that is it worth starting BTP modifications
([12_BTP_Modification_Guide.md](12_BTP_Modification_Guide.md)) — with one exception: the
toxicity-loss work (§1 there) can be prototyped against pretraining alone, independently of
this whole chain.
