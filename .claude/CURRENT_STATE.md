# CURRENT_STATE

**Last verified: 2026-09-08** · repo `/home/muhamed/repo/CTDDG`, branch `main`, HEAD `422c8fc`
*(Re-verify against the repository before relying on this.)*

## Status

| Stage | Status |
|---|---|
| Cluster + `ctddg_env` | ✅ DONE |
| ChEMBL data + label conversion | ✅ DONE (labels are dummies — all `1`) |
| BindingDB data (5 folds) | ✅ DONE |
| `atom_types.txt` (65) | ✅ DONE |
| Needle similarity (5,809,835 alignments) | ✅ DONE |
| Similarity histograms | ❌ blocked by a bug in `data_preprocessing_1.ipynb` cell 37 |
| **Pretraining** | ✅ **DONE — 480,000/480,000, "Training finished"** |
| Checkpoint validation (ever sampled from?) | ❌ NOT STARTED |
| Protein embeddings | ❌ NOT STARTED — producer defective |
| `d1_tr_cdgcn.txt` (paired data) | ❌ NOT STARTED |
| Fine-tuning | ❌ NOT STARTED — **BLOCKED** |
| Generation / Evaluation / Docking | ❌ NOT STARTED |
| `notebooks/` wrappers | ❌ BROKEN (never executed) |
| CrossDocked2020 | ✅ extracted at `~/datasets/` |

## Current checkpoint — `outputs/pretrain/logs/`

| | |
|---|---|
| Model | `VanillaMolGen_RNN` — **unconditional**, CDGCN objective |
| Iterations | 480,000 / 480,000 · `Training finished` |
| Wall time | 2799.06 **minutes** ≈ 46.65 h (⚠ `log.out` header says `time(h)` but writes minutes) |
| Final loss / lr | 15.99 (single-batch, noisy) / 8.21e-06 |
| Parameters | 6,986,506 |
| Files | `ckpt.params` 27.9 MB · `trainer.status` 55.9 MB · `configs.json` 129 B · `log.out` 962 lines |
| Run | Slurm job 11973, node003, 1 × A30 (2 allocated), 2026-09-02 23:17 → 09-04 21:40 |
| Artefact | `code/pretraining_executed.ipynb` (965 outputs) |

✅ **PROTECTED (2026-09-08).** `code/pretraining.ipynb` now refuses to overwrite a finished
run. Re-running it with the default run name raises `RuntimeError` and prints instructions.
- New experiment:  `export CTDDG_RUN_NAME=my_experiment` → `outputs/pretrain/my_experiment/logs/`
- Deliberate overwrite: `export CTDDG_FORCE_RETRAIN=1` (still snapshots first)
- Resuming an *unfinished* run snapshots it to `outputs/pretrain/archive/` first.
Write-protected copy of the 480k run:
`outputs/pretrain/archive/base_cdgcn_step480000_FINISHED_20260908-235622/` (md5-verified).

## Blocking chain
```
protein embeddings not produced (producer omits reduce_per_protein)   ← ROOT CAUSE
  → d1_tr_cdgcn.txt cannot be built
    → fine-tuning cannot run (no loader, no loop; finetune_cell.py broken)
      → no conditional checkpoint → no generation → no evaluation → no docking
```
Side-blockers: `data/fpscores.pkl.gz` absent (SAS) · `Jupyter_Dock/` + 6 packages absent
(docking) · `data/d{i}_te_embeddings_run{r}.txt` produced by no code.

## Next actions (see PROJECT_DOCS/11 for detail)
0. ✅ done — checkpoint archived and the notebook now guards itself.
1. Write a `Vanilla_RNN_Builder` and validate the pretrained checkpoint by sampling (~0.5 day, zero risk).
2. Fix `give_bioembeddings` (add `reduce_per_protein`) + the float writer; run embedding cells for fold 1.
3. Build `d1_tr_cdgcn.txt` (strip the injected SMILES spaces, mirroring upstream cell 102).
4. Write a correct conditional fine-tuning stage reusing `CVanillaMolGen_RNN` from `Generating_samples.ipynb`.
5. Generate → evaluate.

## Environment
```
Env      /home/muhamed/.conda/envs/ctddg_env      Python 3.9.25
MXNet    1.9.1 (mxnet-cu112)      rdkit 2022.09.5      numpy 1.23.5 (do NOT upgrade)
scipy 1.10.1 · pandas 1.5.3 · networkx 3.2.1 · h5py 3.14.0 · molvs 0.1.1
sklearn 0.24.2 · matplotlib 3.9.4 · seaborn 0.13.2 · bio_embeddings + torch 1.9.1+cu102
EMBOSS needle at $CONDA_PREFIX/bin/needle
ABSENT: vina, meeko, prolif, MDAnalysis, pymol, openbabel, py3Dmol
Cluster  bhavani.iitpkd.ac.in · node001-004 · 2 × NVIDIA A30 24 GB · partitions normal, gpu01-04
```

## Working commands (verified)
```bash
# environment block — REQUIRED in every job script
module load cuda/11.2 && module load anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ctddg_env
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MXNET_CUDNN_LIB_CHECKING=0
which python    # MUST be ~/.conda/envs/ctddg_env/bin/python

bash cluster/setup_jupyter_kernel.sh        # once, login node
sbatch cluster/launch_jupyter_preferred.sh  # start Jupyter on an idle GPU node
cat cluster/jupyter_<JOBID>.log             # prints the ssh -N -L tunnel command
sbatch run_preprocessing.sh                 # headless nbconvert pattern
python check_pretraining.py                 # progress checker (no MXNet needed)
squeue -u $USER ; scancel <JOBID>
```

## Key paths
```
outputs/pretrain/logs/{ckpt.params,trainer.status,configs.json,log.out}
data/chembl/{chembl.csv,chembl.txt,chembl_final.txt}
data/bindingdb/{train,test}_dataset/*_4_org_*/
data/atom_types.txt                      # 65 lines — DO NOT REGENERATE
data/data_preprocessing.ipynb            # upstream nb WITH original outputs = real statistics
papers/{CTDDG.pdf,CDGCN.pdf} · supplimentaryAPIN.pdf
~/datasets/downsampled_CrossDocked2020_v1.3/
~/.cache/bio_embeddings/prottrans_bert_bfd/
```

## Uncommitted changes
`git diff HEAD -- code/` touches all 5 committed notebooks. **All changes are path rewrites
(`scripts/patch_paths.py`) plus a needle-resume guard in `data_preprocessing_1.ipynb`.**
No model logic, hyperparameter or dataset behaviour has been modified locally.
Untracked: `PROJECT_DOCS/`, `.claude/`, `cluster/launch_jupyter_preferred.sh`,
`check_pretraining.py`, `run_preprocessing.sh`, `preprocessing_*.{out,err}`,
`Molecular_dynamics_files/`, `Moleculsr_docking_top10GeneratedMol/`.
