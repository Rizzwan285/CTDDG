# WORKFLOW

Verified operational pipeline. **Run `code/*.ipynb` directly — `notebooks/*.ipynb` are broken.**

```
preprocessing → pretraining → fine-tuning → generation → evaluation → docking
     ◑ partial      ✅ DONE      ❌ blocked     ❌            ❌           ❌
```

## Environment block (required in EVERY job script)
```bash
module load cuda/11.2
module load anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctddg_env
export PATH="$CONDA_PREFIX/bin:$PATH"                       # ← without this the system
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" #   Anaconda silently shadows it
export MXNET_CUDNN_LIB_CHECKING=0
cd ~/repo/CTDDG
which python    # MUST print ~/.conda/envs/ctddg_env/bin/python
```

---

## Stage 0 — Cluster session
| | |
|---|---|
| Once | `bash cluster/setup_jupyter_kernel.sh` (login node) |
| Start | `sbatch cluster/launch_jupyter_preferred.sh` |
| Connect | `cat cluster/jupyter_<JOBID>.log` → copy the `ssh -N -L …` line → run on laptop → `http://localhost:<port>` |
| Port | `8800 + (SLURM_JOB_ID % 200)` |
| GPU | 2 × A30 allocated, **1 used** |
| ⚠ | Jupyter runs with **no token/password** and binds `0.0.0.0` |

Long runs survive laptop disconnection — the tunnel is only a pipe; the Slurm job owns the
processes. What kills runs is the `--time` limit.

---

## Stage 1 — Preprocessing ◑ PARTIAL
| | |
|---|---|
| Notebook | `code/data_preprocessing_1.ipynb` (108 cells) |
| Batch | `sbatch run_preprocessing.sh` ✅ verified pattern |
| GPU | 1 (only for the embedding step) |
| Done | fold-1 repair · **all 5,809,835 needle alignments** · `atom_types.txt` (65) · `chembl.txt` |
| ❌ Fails | cell 37 — `TypeError: object of type 'int' has no len()` (histogram plotting only) |
| ❌ Not run | protein embeddings (cells 74–79) · `d1_tr_cdgcn.txt` (cells 99–107) |
| Outputs | `data/atom_types.txt`, `data/chembl/chembl.txt`, `data/bindingdb/needle_outputs/` |

⚠ **Never re-run the whole notebook** — it would recompute 5.8 M alignments and rewrite
`atom_types.txt` (whose line order the trained checkpoint depends on). Run only the cells you need.

### ChEMBL label conversion ✅ done
```bash
python scripts/convert_chembl_format.py data/chembl/chembl.txt data/chembl/chembl_final.txt
```
Appends `" 1"` — a **dummy** label. Output: 1,090,529 lines, all class 1.

### Path patching ✅ already applied
```bash
python scripts/patch_notebooks.py        # MUST run BEFORE patch_paths (matches /workspace/…)
python scripts/patch_paths.py $(pwd)
```

---

## Stage 2 — Pretraining ✅ DONE
| | |
|---|---|
| Notebook | `code/pretraining.ipynb` (single 51 KB cell) |
| Input | `data/chembl/chembl_final.txt` (1,090,529), `data/atom_types.txt` |
| Output | `outputs/pretrain/logs/{ckpt.params, trainer.status, configs.json, log.out}` |
| Model | `VanillaMolGen_RNN`, 6,986,506 params |
| Result | **480,000/480,000 · "Training finished" · 2799.06 min ≈ 46.65 h · 1 × A30** |
| Monitor | `python check_pretraining.py` or `tail -f outputs/pretrain/logs/log.out` |
| Resume | automatic when `log.out` + `ckpt.params` + `trainer.status` all exist |

```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ctddg_env \
    --output pretraining_executed.ipynb \
    code/pretraining.ipynb
```
⚠ **Re-running overwrites the finished checkpoint.** Back up first:
```bash
cp -r outputs/pretrain/logs outputs/pretrain/logs.bak.$(date +%F)
```
⚠ Budget > 47 h of walltime. `run_pipeline_batch.sh`'s 48 h nbconvert timeout is too tight.

---

## Stage 3 — Fine-tuning ❌ BLOCKED
| | |
|---|---|
| Architecture | `CVanillaMolGen_RNN` — **exists** in `code/Generating_samples.ipynb` cell 14 |
| Loader + loop | **do not exist** |
| `scripts/finetune_cell.py` | **non-functional** — 5 runtime blockers + 5 architectural divergences. **Do not repair it** |
| Needs | `d{i}_tr_cdgcn.txt` = `<SMILES>\t<N_C floats>` per pair |
| Would output | `outputs/CTDGD/Dataset{i}/model/{ckpt.params, configs.json incl. N_C}` |
| Paper params | `papers/CDGCN.pdf` §3.3 — batch 8, α=0.8, K=5, ~10 epochs with early stopping |
| Transfer | `load_parameters(..., allow_missing=True, ignore_extra=True)` → 86% of params carry over; nothing frozen |

Blocked on protein embeddings. Scope: `PROJECT_DOCS/07_Finetuning_and_Conditioning.md` §7.

---

## Stage 4 — Generation ❌
| | |
|---|---|
| Notebook | `code/Generating_samples.ipynb` |
| Driver | cell 19 — hardcodes `dataset_index=2, run=1, n_samples=1000, model_name='CTDGD'` |
| Input | `data/d{i}_te_embeddings_run{r}.txt` ⚠ **no code produces this file** |
| Input | `outputs/CTDGD/Dataset{i}/model/{configs.json (needs N_C), ckpt.params}` |
| Output | `outputs/CTDGD/Dataset{i}/generated_samples/{n}/run{r}/Protein{p}_generated_samples.csv` |
| Sampling | stochastic joint categorical, `random=True`, **max 100 steps**, no beam search |
| Runtime | CDGCN Table 1: ≈142 s per protein at S_1000 → ≈4 h for 104 proteins per run |

---

## Stage 5 — Evaluation ❌
| | |
|---|---|
| Notebook | `code/Evaluation_metrics.ipynb` |
| Needs | ⚠ `data/fpscores.pkl.gz` (**absent** — `wget` from rdkit Contrib/SA_Score) |
| Metrics | validity, uniqueness, logP, MW, HBD, HBA, RB, TPSA, QED, SAS |
| Missing | **novelty** and `R_c` are NOT implemented (despite `notebooks/04` claiming novelty) |
| ⚠ Bug | cells 11/12 loop `range(1,11)` proteins; cell 13 uses `nprot=104` → `IndexError` |
| ⚠ Formula | property % = `100 × (unique/N) × (satisfying/N)` — **compounded with uniqueness**, not a plain rate |
| Side effect | overwrites the generated CSVs in place, adding property columns |

---

## Stage 6 — Docking ❌
| | |
|---|---|
| Notebook | `code/Molecular_docking.ipynb` — an **interactive worksheet**, not a batch stage |
| Needs | `Jupyter_Dock/` (absent) + pymol, openbabel, vina, meeko, prolif, MDAnalysis (all absent) |
| Protein | PyMOL fetch → LePro / `fix_protein` → `prepare_receptor` → `.pdbqt` |
| Ligand | `obabel --gen3D` → pybel `addh` → `prepare_ligand` → `.pdbqt` |
| Engine | `smina --autobox_ligand <pdb>_lig.mol2 --autobox_add 5 --exhaustiveness 16 --num_modes 5` |
| Targets | 4nos, 2nru, 1lqf, 1k2r, 1cqp (the five with known binding sites) |
| ⚠ | **No score parsing / ranking code exists.** Hardcodes `beam=0, run=3, prot_id=6, pdb_id='1cqp'` |
| ⚠ | A docking score is **not** a binding affinity and **not** evidence of binding |

---

## Scripts reference
| Script | Purpose | Run? |
|---|---|---|
| `scripts/convert_chembl_format.py` | append dummy `" 1"` label | ✅ |
| `scripts/patch_paths.py` | rewrite 12 hardcoded `/workspace/…` paths | ✅ |
| `scripts/patch_notebooks.py` | remove empty `.append()`, `os.system(mkdir)` → `os.makedirs` | ✅ |
| `scripts/download_data.sh` | gdown datasets, `fpscores.pkl.gz`, clone Jupyter_Dock | ❌ partial |
| `scripts/config.py` | path config singleton | ⚠ nothing imports it |
| `scripts/finetune_cell.py` | reconstructed fine-tuning | ❌ non-functional |
| `check_pretraining.py` | progress checker (no MXNet) | ✅ |
| `cluster/generate_notebooks.py` | generates `notebooks/00–06` | ⚠ produces broken notebooks |
| `cluster/run_pipeline_batch.sh` | headless chain | ⚠ skips fine-tuning; re-runs pretraining |
