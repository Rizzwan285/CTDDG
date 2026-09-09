# CTDDG — Project Documentation (Master Index)

**Repository:** `/home/muhamed/repo/CTDDG`
**Audited:** 2026-09-08
**Audit method:** every notebook read cell-by-cell; both papers read from `papers/`; logs, checkpoints, data files and the conda environment inspected directly.

> **How to read this documentation.** Every non-obvious claim below is tagged with where it
> came from. Four tags recur throughout all documents:
>
> | Tag | Meaning |
> |---|---|
> | **PAPER** | Stated in `papers/CTDDG.pdf` or `papers/CDGCN.pdf` |
> | **CODE** | Verified by reading the original implementation in `code/` |
> | **NEW NOTEBOOKS** | Behaviour of the generated wrappers in `notebooks/` |
> | **EXECUTED** | Confirmed by an artefact on disk (log, checkpoint, notebook output) |
>
> Where something could not be established, it says
> **NOT DETERMINED FROM AVAILABLE SOURCES** or **INFERRED — NOT EXPLICITLY CONFIRMED**.

---

## 1. What this project is

CTDDG stands for **Conditional Target-based Detoxed Drug Generation**. It is a deep
generative model that takes a **protein target** as input and produces **novel small
drug-like molecules**, built atom-by-atom as a **molecular graph**, that are intended to
(a) bind the target and (b) be less likely to be toxic.

The problem it addresses: finding a new drug for a disease-causing protein normally means
searching a chemical space estimated at ~10^60 molecules, over 10–15 years and >$1.8B
(**PAPER**, `papers/CTDDG.pdf` §1 Introduction). A generative model that proposes a small
set of plausible candidates for a *previously untargeted* protein shrinks that search.

The distinguishing claim of CTDDG over its predecessor is toxicity: it is described as
"the first method designed to generate less toxic de novo drugs for novel protein targets"
(**PAPER**, `papers/CTDDG.pdf` Abstract).

## 2. The two papers and how they relate

| Paper | File | Role |
|---|---|---|
| **CDGCN** — *Conditional de novo Drug generative model using Graph Convolution Networks*, Mallick & Bhadra | `papers/CDGCN.pdf` | **The architecture.** Defines the graph generator, the protein conditioning mechanism, the sequential decoding scheme and the training loss. |
| **CTDDG** — *Graph-Based Deep Generative Model for Low-Toxic Drug Design…*, Singh, Bhadra & Bhadra | `papers/CTDDG.pdf` | **The extension.** Reuses CDGCN's architecture verbatim and adds a toxicity-weighted loss plus toxicity labelling of both datasets. |

> "Architecture for CTDDG is based on that of CDGCN (Mallick and Bhadra, 2023)."
> — **PAPER**, `papers/CTDDG.pdf` §2.2

**Practical consequence:** to understand *how the model works*, read CDGCN. To understand
*what CTDDG adds*, read only §2.3 and §2.4 of CTDDG. A third file,
`supplimentaryAPIN.pdf` (repo root), is the CTDDG supplementary — Table S1 lists the SMILES
of the ten reported CDK2 molecules, Table S2 their MD interactions.

## 3. The single most important finding

**The completed pretraining run implements the CDGCN loss, not the CTDDG loss.**

The toxicity mechanism that gives CTDDG its name is present in the code as plumbing but is
inert at three independent points:

1. `data/chembl/chembl_final.txt` labels **all 1,090,529 molecules as class `1`** — verified
   by counting; it is byte-identical to `chembl.txt` with `" 1"` appended (**EXECUTED**).
2. `process_single()` parses that label into `smiles_class` and then **discards it**,
   hardcoding `tox_class = [1] * k` (**CODE**, `code/pretraining.ipynb`).
3. `MoleculeGenerator._likelihood()` accepts `tox_class_batch` as an argument and **never
   references it**; the returned loss is the plain negative log-likelihood (**CODE**).

So the model on disk is a faithful **CDGCN** unconditional pretrained generator. This is not
a defect to be fixed silently — it is the honest starting point, and it is arguably the
clearest opening for BTP work. See [12_BTP_Modification_Guide.md](12_BTP_Modification_Guide.md).

## 4. Verified end-to-end pipeline

Only the stages below are drawn from code that actually exists. Status reflects the
repository as of this audit.

```
   data/chembl/chembl.csv                data/bindingdb/{train,test}_dataset/
   (1,207,360 ChEMBL rows)               (5 folds, FASTA + SMILES pairs)
            │                                          │
            ▼                                          ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ PREPROCESSING       code/data_preprocessing_1.ipynb           │  ◑ PARTIAL
   │  • fix fold-1 protein counts (1514→1042, 99→104)              │
   │  • Needleman-Wunsch similarity (EMBOSS needle)   ✅ complete   │
   │  • filter ligands >100 atoms, strip salts                     │
   │  • build atom_types.txt (65 types)               ✅ on disk    │
   │  • ChEMBL minus BindingDB ligands                ✅ on disk    │
   │  • ProtTrans protein embeddings                  ❌ not run    │
   │  • assemble d{i}_tr_cdgcn.txt                    ❌ not run    │
   └───────────────────────────────────────────────────────────────┘
            │                                          │
   chembl_final.txt                          d{i}_tr_cdgcn.txt
   (1,090,529 SMILES + dummy label)          (SMILES \t protein embedding)
            │                                          │
            ▼                                          │
   ┌─────────────────────────────────┐                 │
   │ PRETRAINING  code/pretraining   │  ✅ DONE         │
   │ VanillaMolGen_RNN (no protein)  │  480000/480000   │
   │ 6,986,506 params · ~46.7 h      │  "Training       │
   │ → outputs/pretrain/logs/        │   finished"      │
   └─────────────────────────────────┘                 │
            │ ckpt.params (27.9 MB)                     │
            ▼                                          ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ FINE-TUNING (conditional)                        ❌ BLOCKED    │
   │ CVanillaMolGen_RNN — architecture EXISTS in                   │
   │ code/Generating_samples.ipynb, but NO training loop           │
   │ for it exists anywhere in the original repository.            │
   │ scripts/finetune_cell.py is a reconstruction and is           │
   │ non-functional (see 07_Finetuning_and_Conditioning.md).       │
   └───────────────────────────────────────────────────────────────┘
            │ outputs/CTDGD/Dataset{i}/model/ckpt.params
            ▼
   ┌─────────────────────────────────┐
   │ GENERATION  Generating_samples  │  ❌ NOT STARTED (needs above)
   │ stochastic sequential decoding  │
   │ → Protein{i}_generated_samples  │
   │   .csv                          │
   └─────────────────────────────────┘
            ▼
   ┌─────────────────────────────────┐
   │ EVALUATION  Evaluation_metrics  │  ❌ NOT STARTED
   │ validity, uniqueness, logP, MW, │  (also needs data/fpscores.pkl.gz)
   │ HBD, HBA, RB, TPSA, QED, SAS    │
   └─────────────────────────────────┘
            ▼
   ┌─────────────────────────────────┐
   │ DOCKING  Molecular_docking      │  ❌ NOT STARTED
   │ smina, autobox from co-crystal  │  (needs Jupyter_Dock + 6 packages)
   │ ligand + 5 Å                    │
   └─────────────────────────────────┘
```

Legend: ✅ done · ◑ partial · ❌ not started or blocked

**The pipeline is currently cut in half at fine-tuning.** Everything upstream of it is done
or nearly done; nothing downstream of it can run until it is solved.

## 5. Where the important code lives

| You want to understand… | Read this file |
|---|---|
| The model, the graph convolution, the training loop | `code/pretraining.ipynb` — one 51 KB cell, contains everything |
| The **conditional** model (protein-aware) | `code/Generating_samples.ipynb`, cell 14 — `_TwoLayerDense`, `CMoleculeGenerator_RNN`, `CVanillaMolGen_RNN` |
| Sampling / decoding | `code/Generating_samples.ipynb`, cell 16 — `_decode_step`, `CVanilla_RNN_Builder.sample` |
| Datasets and how they were built | `code/data_preprocessing_1.ipynb` |
| What the *upstream* authors actually ran | `data/data_preprocessing.ipynb` — **retains its original executed outputs**, the best source of real dataset statistics |
| Metrics | `code/Evaluation_metrics.ipynb` |
| Docking | `code/Molecular_docking.ipynb` |

## 6. Document index

| Document | Covers |
|---|---|
| [01_Project_and_Research_Context.md](01_Project_and_Research_Context.md) | Beginner primer, then the full scientific methodology; PAPER vs CODE vs NOTEBOOKS vs EXECUTED for every major component |
| [02_Repository_and_Codebase.md](02_Repository_and_Codebase.md) | File-by-file map, execution flow, how `notebooks/` relates to `code/` |
| [03_Data_and_Preprocessing.md](03_Data_and_Preprocessing.md) | ChEMBL, BindingDB/Grechishnikova, splits, filters, real statistics |
| [04_Protein_Embeddings.md](04_Protein_Embeddings.md) | How protein information enters the model — and the confirmed gap here |
| [05_Model_Architecture.md](05_Model_Architecture.md) | Full reverse-engineered architecture, tensor shapes, forward pass |
| [06_Pretraining_and_Checkpoints.md](06_Pretraining_and_Checkpoints.md) | Objective, hyperparameters, checkpoint file semantics, the completed run |
| [07_Finetuning_and_Conditioning.md](07_Finetuning_and_Conditioning.md) | The missing stage, what the condition actually is, why the reconstruction fails |
| [08_Generation.md](08_Generation.md) | Sequential decoding step-by-step, sampling, outputs |
| [09_Evaluation_and_Docking.md](09_Evaluation_and_Docking.md) | Every implemented metric; docking protocol and what it does/doesn't prove |
| [10_Cluster_and_Execution.md](10_Cluster_and_Execution.md) | Bhavani setup, Slurm, SSH tunnels, running long jobs safely |
| [11_Current_State_and_Next_Steps.md](11_Current_State_and_Next_Steps.md) | Status snapshot and the exact next actions |
| [12_BTP_Modification_Guide.md](12_BTP_Modification_Guide.md) | Concrete, code-grounded modification points for research work |

A companion set of compact machine-oriented context files lives in `.claude/` for future
Claude Code sessions. `PROJECT_DOCS/` is for humans; `.claude/` is the short version.

## 7. What has already been completed

- **Environment** — `ctddg_env` conda env built and verified on the Bhavani cluster.
- **Data acquisition** — ChEMBL and BindingDB present; CrossDocked2020 extracted to `~/datasets/`.
- **Needleman-Wunsch similarity** — all 5,809,835 pairwise alignments computed.
- **Pretraining** — 480,000 / 480,000 iterations, "Training finished", checkpoint on disk.

## 8. What remains

1. Produce fixed-size per-protein embeddings (the blocking gap — see [04](04_Protein_Embeddings.md)).
2. Assemble the paired fine-tuning file `d{i}_tr_cdgcn.txt`.
3. Write a *correct* conditional fine-tuning loop (see [07](07_Finetuning_and_Conditioning.md)).
4. Generate, evaluate, dock.
5. Then begin BTP modifications (see [12](12_BTP_Modification_Guide.md)).

## 9. Where future BTP work should happen

The Week-1 progress notes state the current goal is the **single-target, single-property
baseline**, with a **multi-target extension using CrossDocked2020** as the eventual
direction (`WEEK 1 PROGRESS - TASK 2.md` §17, §19). Given the audit, the three highest-value
and best-grounded modification points are:

1. **Activate the toxicity loss** — the paper's Eq. 8 is fully specified and the code has all
   the plumbing already threaded through. This turns the repo from CDGCN into actual CTDDG.
2. **Replace or fix the protein conditioning path** — currently the weakest-verified link.
3. **Write the conditional fine-tuning stage properly**, reusing the architecture that
   already exists in `Generating_samples.ipynb` rather than reimplementing it.

Each is broken down with file, function, difficulty and risk in
[12_BTP_Modification_Guide.md](12_BTP_Modification_Guide.md).
