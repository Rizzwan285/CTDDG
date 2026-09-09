# 06 — Pretraining and Checkpoints

**This stage is complete.** 480,000 / 480,000 iterations, `Training finished` written to the
log, checkpoint on disk. This document records exactly what was trained and what the
artefacts mean.

---

## 1. Purpose

Learn `P(molecule)` — general drug-like chemistry — from ~1.09 M ChEMBL molecules, with **no
protein involved**. The papers justify the two-phase split by data scarcity:

> "BindingDB contains … pairs of proteins and ligands … This data is not enough to train the
> large architecture of CTDDG. Hence CTDDG get trained in two phases following Grechishnikova
> (2021); first phase is pre-training…" — **PAPER**, `papers/CTDDG.pdf` §2.4

CDGCN says the same, and adds that during pretraining the protein path is replaced:

> "CDGCN is first pre-trained to generate chemically valid molecules without protein
> constraint. In this case, a learnable weight vector followed by Softmax activation is used
> instead of `Embed_p` and `FNN_0`." — **PAPER**, `papers/CDGCN.pdf` §2.2

**CODE matches this exactly.** `MoleculeGenerator._policy_0(ctx)` returns
`exp(policy_0) / exp(policy_0).sum()` over the learnable `[65]` parameter `policy_0` — a
learnable weight vector followed by softmax.

---

## 2. Dataset

| Item | Value |
|---|---|
| File | `data/chembl/chembl_final.txt` |
| Lines | **1,090,529** (verified by `wc -l`, and by the run's own stdout) |
| Format | `<SMILES> <class>` |
| Class distribution | **1,090,529 × class `1`** — no class-0 molecules exist |
| Loader | `read_data()` → list of raw lines |

Run's stdout (`code/pretraining_executed.ipynb`) confirms:
```
GPU is available. Number of GPUs: 2
1090529
<class 'list'>
['c1ccc(-n2cc(CNC3CCc4ncnn4C3)cn2)cc1 1', 'Cc1cccc(CCN2CCC[C@@H]2C)c1 1', …]
681580
```

`681580` is the *computed* `iterations = (len(dataset)//batch_size)*5 = (1090529//8)*5`.
It is then **overwritten** by the hardcoded `iterations = 480000` just above the training
loop. 480,000 iterations × 8 = 3.84 M molecule-draws ≈ **3.52 epochs**.

---

## 3. Model

`VanillaMolGen_RNN(N_A=65, N_B=4, D=2, **configs)` — see
[05_Model_Architecture.md](05_Model_Architecture.md). 6,986,506 parameters.

Initialisation: `mx.init.Xavier()` on a fresh run; `model.load_parameters(ckpt.params)` on
resume.

---

## 4. Objective

Negative log-likelihood with importance sampling over `K = 5` generation paths — CTDDG Eq. 4
/ CDGCN Eq. 6. Implementation in `MoleculeGenerator._likelihood`, detailed in
[05](05_Model_Architecture.md) §B.5.

```python
with autograd.record():
    loss = [(model(*inputs)).as_in_context(mx.gpu())]
    loss = sum(loss)
    loss.backward()
```

`model.mode == 'loss'` → `forward` returns `-l.mean()`, a scalar averaged over the 8
molecules in the batch.

**No toxicity term.** See §7.

---

## 5. Hyperparameters

Verbatim from `code/pretraining.ipynb`:

```python
batch_size      = 8       # training batch size
batch_size_test = 8
k               = 5       # number of generation paths (K)
p               = 0.8     # randomness parameter alpha
F_e             = 16
F_h             = [32,64,128,128,256,256]
F_skip          = 256
F_c             = [512, ]
Fh_policy       = 128
activation      = 'relu'
lr              = 1e-3    # initial learning rate
decay           = 1e-3    # decay factor
decay_step      = 100     # decay every 100 steps
clip_grad       = 3.0
summary_step    = 500     # checkpoint + log every 500 steps
N_rnn           = 3
is_continuous   = False   # (auto-detected later)
model_name      = 'base_cdgcn'
iterations      = 480000  # hardcoded, overrides the computed value
```

Note `model_name = 'base_cdgcn'` — the authors' own naming acknowledges this is the CDGCN
baseline.

### Optimiser
```python
opt     = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
trainer = gluon.Trainer(model.collect_params(), opt)
...
trainer.step(batch_size=1)
```

⚠ `trainer.step(batch_size=1)` — the gradient is **not** divided by 8. This is correct here
because `forward` already returns `l.mean()`; dividing again would shrink the effective
learning rate 8×. Worth knowing before changing the batch size.

### Learning-rate schedule
```python
if global_counter % decay_step == 0:
    trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))
```
Multiplicative: `lr(t) = 1e-3 × 0.999^(t/100)`. Not an MXNet `LRScheduler` — it is applied by
hand inside the loop.

**Arithmetic check against the actual log:**
| Step | Predicted `1e-3 × 0.999^(step/100)` | Logged |
|---|---|---|
| 500 | 9.950100e-04 | `0.000995009990004999` ✅ |
| 480,000 | 8.2050e-06 | `8.210006192943532e-06` ✅ |

Confirms the schedule ran as written for the full duration.

### CDGCN paper's stated hyperparameters
`papers/CDGCN.pdf` §3.3: MXNet 1.7.0, single RTX 2080 Ti (8 GB), batch size 8 chosen by
5-fold CV ("Batch sizes higher than 8 could not be tested due to memory constraint"),
α = 0.8, K = 5, "All other hyperparameters and optimizer for CDGCN follow Li et al.".

**Agreement:** batch size 8 ✅, α = 0.8 ✅, K = 5 ✅. Framework differs (1.9.1 here vs 1.7.0);
neither paper states the pretraining iteration count, so **480,000 is NOT DETERMINED FROM
AVAILABLE SOURCES as a paper-derived value** — it is the number hardcoded in the authors'
notebook.

---

## 6. GPU usage

```python
ctx = mx.gpu()                      # device 0
X = nd.array(X, ctx=mx.gpu(), ...)  # hardcoded throughout MolLoader.from_numpy_to_tensor
loss = [(model(*inputs)).as_in_context(mx.gpu())]
```

**Single GPU only.** `mx.gpu()` defaults to device 0. There is no `gluon.utils.split_and_load`,
no device list, no `kvstore`. The `loss = [ ... ]; loss = sum(loss)` pattern looks like a
vestige of a multi-GPU template but wraps exactly one element.

⚠ `notebooks/01_pretraining.ipynb` claims "Uses all available GPUs via MXNet data
parallelism". **That is false.** The job requested 2 GPUs; one sat idle.

---

## 7. ⚠ The toxicity objective is not implemented

Three independent verifications:

**(a) The data has no labels.**
```
$ awk '{print $NF}' data/chembl/chembl_final.txt | sort | uniq -c
1090529 1
```
Every molecule is class 1. `scripts/convert_chembl_format.py` states it appends "a dummy class
label". The paper's Random Forest Ames-mutagenicity classifier does not exist in this
repository — `code/pretraining.ipynb` contains the bare comment `# Load ML classifier` with
nothing beneath it.

**(b) The label is parsed then discarded.**
```python
def process_single(smiles, k, p):
    smiles, smiles_class = smiles.split(" ")
    smiles_class = int(smiles_class)          # ← parsed
    ...
    tox_class = [1] * k                       # ← hardcoded; smiles_class never used again
```

**(c) The loss ignores it.**
`_likelihood(..., batch_size, iw_size, tox_class_batch)` never references `tox_class_batch`;
`forward` returns `-l.mean()`.

### What this means

| | |
|---|---|
| **PAPER** | Eq. 8: weight each molecule's log-likelihood by `(1 − (1+λ)·ToxicClass)`, λ=1 |
| **CODE** | Eq. 4: unweighted mean negative log-likelihood |
| **EXECUTED** | Eq. 4 |

**The checkpoint in `outputs/pretrain/logs/ckpt.params` is a CDGCN pretrained model.** It is a
perfectly valid unconditional molecular graph generator; it simply carries none of CTDDG's
toxicity behaviour. Do not describe it as a CTDDG model. Making it one is the most
paper-grounded BTP opportunity available — [12](12_BTP_Modification_Guide.md) §1.

---

## 8. Checkpoint files — what each one is

All in `outputs/pretrain/logs/` (`ckpt_dir = f'{model_dir}/logs/'`,
`model_dir = /home/muhamed/repo/CTDDG/outputs/pretrain`).

### `configs.json` — 129 bytes
```json
{"F_e": 16, "F_h": [32, 64, 128, 128, 256, 256], "F_skip": 256,
 "F_c": [512], "Fh_policy": 128, "activation": "relu", "N_rnn": 3}
```
The **architecture spec**, written once at the start of a fresh run and re-read on resume:
```python
model = VanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)
```
Note what is **not** in it: `N_A`, `N_B` (derived from `atom_types.txt` at runtime) and `D`
(passed literally). ⚠ Consequence: this file only reconstructs the model correctly if
`atom_types.txt` is unchanged and `D` is still 2.

⚠ `.gitignore` contains `*.json`, so `configs.json` is not tracked. Losing it means losing the
architecture spec for the checkpoint.

### `ckpt.params` — 27,950,253 bytes
Model weights, written by `model.save_parameters()`. 6,986,506 × 4 bytes = 27,946,024, plus
~4 KB of MXNet header — consistent.

### `trainer.status` — 55,859,209 bytes
Adam **optimiser state**, written by `trainer.save_states()`: the first and second moment
estimates (two per parameter) plus the current learning rate. ≈2× the weights file, as
expected. Required for a *seamless* resume; without it, resuming restarts Adam's moments from
zero.

### `log.out` — 63,602 bytes, 962 lines
```
step	time(h)	loss	lr
500	2.951296663284302	35.5838737487793	0.000995009990004999
1000	5.8902428110440574	37.04569625854492	0.000990044880209748
…
480000	2799.0602951129276	15.987691879272476	8.210006192943532e-06
Training finished
```
1 header + 960 records (480,000 / 500) + the terminator. Written by:
```python
f.write('{}\t{}\t{}\t{}\n'.format(global_counter,
                                  float(time.time() - t0)/60,
                                  loss,
                                  trainer.learning_rate))
```

### ⚠ The time column is MINUTES, not hours

The header says `time(h)`. The expression is `(time.time() - t0) / 60` — seconds divided by
60, i.e. **minutes**. The header is simply wrong.

Corroboration, three ways:
1. The resume logic reads it back as minutes: `t0 = time.time() - t_final * 60`. Self-consistent.
2. Final value 2799.06 → 46.65 h. `outputs/pretrain/training_monitor.log` shows the run
   spanning 2026-09-02 23:17 → 2026-09-04 21:40 ≈ **46.4 h** for steps 2,000→479,500. ✅
   If the column were hours, the run would have taken 116 days.
3. Rate check: 500 steps in the first 2.95 units → 2.82 steps/s if minutes. 480,000 / 2.82 ≈
   47 h. ✅

**Read the time column as minutes.**

### `loss` column caveat
The logged loss is the **single mini-batch loss at that exact step**, not a running average.
With batch size 8 it is very noisy — the last four records are 15.43, 13.94, 21.65, 15.99.
Do not read a single value as "final loss"; look at the trend over the 960 records.

---

## 9. Resume mechanism

Detection:
```python
if all([os.path.isfile(os.path.join(ckpt_dir, _n))
        for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
    is_continuous = True
else:
    is_continuous = False
```

On resume:
```python
with open(os.path.join(ckpt_dir, 'log.out')) as f:
    records = f.readlines()
    if records[-1] != 'Training finished\n':
        final_record = records[-1]
    else:
        final_record = records[-2]                 # skip the terminator
count, t_final = int(final_record.split('\t')[0]), float(final_record.split('\t')[1])
t0 = time.time() - t_final * 60                    # restore the elapsed-time origin
global_counter = count
```
Then `configs.json` is read (not rewritten), `ckpt.params` loaded, `trainer.load_states()`
called, and `log.out` opened in append mode.

### ✅ Checkpoint protection (added 2026-09-08)

`code/pretraining.ipynb`'s configuration block now contains a protection block:

```python
RUN_NAME = os.environ.get('CTDDG_RUN_NAME', model_name)      # default 'base_cdgcn'
# legacy outputs/pretrain/logs/ is kept for the default name (the finished 480k run);
# any other RUN_NAME -> outputs/pretrain/<RUN_NAME>/logs/
...
if _finished and os.environ.get('CTDDG_FORCE_RETRAIN', '0') != '1':
    raise RuntimeError(...)                                   # refuses to clobber
...
shutil.copytree(ckpt_dir, _archive)                           # snapshot before resuming
```

| Situation | Behaviour |
|---|---|
| Default name, `log.out` ends `Training finished` | **`RuntimeError`** with instructions — nothing written |
| `CTDDG_RUN_NAME=exp1` | fresh `outputs/pretrain/exp1/logs/`, trains from scratch, original untouched |
| Unfinished run | snapshot to `outputs/pretrain/archive/<run>_step<N>_<stamp>/`, then resume normally |
| `CTDDG_FORCE_RETRAIN=1` | snapshot, then proceed (deliberate override) |

All four paths were tested. Because every caller (`notebooks/06`,
`cluster/run_pipeline_batch.sh`, manual `nbconvert`) executes this notebook, all are covered.
⚠ `run_pipeline_batch.sh` uses `set -euo pipefail`, so it will now **abort** at pretraining
instead of clobbering — intended.
⚠ One ~81 MB snapshot per launch; prune `outputs/pretrain/archive/` occasionally.

### ⚠ Resume risks — historical (now mitigated)

`outputs/pretrain/logs/` currently contains all three files and `log.out` ends with
`Training finished`. So **any re-execution of `code/pretraining.ipynb` right now will**:

1. set `is_continuous = True`;
2. read `records[-2]` → `global_counter = 480000`;
3. enter the loop, increment to 480,001, perform **one more training step**, then break at
   `if global_counter >= iterations`;
4. **overwrite `ckpt.params` and `trainer.status`**;
5. append a second `Training finished` to `log.out`.

Not catastrophic, but it does mutate a completed 46-hour artefact. **Back up
`outputs/pretrain/logs/` before running anything that touches this notebook** — including
`notebooks/06_run_full_pipeline.ipynb` and `cluster/run_pipeline_batch.sh`, both of which
re-invoke `code/pretraining.ipynb` unconditionally.

Also note: a fresh run would need `is_continuous == False`, which means **deleting** the three
files. The notebook has no `--from-scratch` flag.

---

## 10. The completed run — verified record

| Property | Value | Source |
|---|---|---|
| Iterations | **480,000 / 480,000** | `log.out` final record |
| Terminator | **`Training finished`** | `log.out` line 962 |
| Log records | 960 (every 500 steps) | `wc -l` |
| Wall time | **2799.06 minutes ≈ 46.65 hours ≈ 1.94 days** | `log.out` time column (minutes, §8) |
| Independent time check | 2026-09-02 23:17 → 2026-09-04 21:40 ≈ 46.4 h | `outputs/pretrain/training_monitor.log` |
| Final logged loss | 15.987691879272476 (single batch; noisy) | `log.out` |
| Final learning rate | 8.210006192943532e-06 | `log.out` |
| Parameters | **6,986,506** | notebook stdout |
| Node | **node003** | `cluster/jupyter_11973.log` |
| GPUs on node | 2 × NVIDIA A30, 24576 MiB | same |
| GPUs used | **1** (`mx.gpu()` = device 0) | code |
| Slurm job | **11973**, partition gpu03 | `cluster/jupyter_11973.log` |
| Job window | 2026-09-02 22:49:55 → 2026-09-07 22:50:00 (TIME LIMIT) | `cluster/jupyter_11973.err` |
| MXNet | 1.9.1 | job log |
| Python | 3.9.25 (`ctddg_env`) | notebook metadata |
| Started from | scratch (`Training is starting from scratch....`) | notebook stdout |
| Throughput | ≈ 2.86 steps/s ≈ 22.9 molecules/s | derived |
| Executed notebook | `code/pretraining_executed.ipynb`, 965 outputs | file |

The Slurm job outlived the training by ~3 days and was then killed by the 5-day limit; the
training had already finished and saved.

⚠ **How nbconvert was invoked is NOT DETERMINED FROM AVAILABLE SOURCES.** It was not
`cluster/run_pipeline_batch.sh` (that writes `cluster/pipeline_%j.log`, which does not exist)
and it cannot have been `notebooks/01_pretraining.ipynb` (structurally broken, see
[02](02_Repository_and_Codebase.md) §5.2). INFERRED — NOT EXPLICITLY CONFIRMED: run by hand
from a terminal inside the Jupyter session on node003.

`outputs/pretrain/training_monitor.log` is a **separate** watcher (`=== CTDDG Training Monitor
Started: Wed Sep 2 23:17:27 IST 2026 ===`, then timestamped `Step N` lines). It is not written
by any script in the repository — **NOT DETERMINED FROM AVAILABLE SOURCES**; presumably an
ad-hoc shell loop tailing `log.out`.

---

## 11. Was the checkpoint validated?

**No.** No validation loss, no held-out evaluation, no sample-quality check was run against
this checkpoint. The only evidence it trained sensibly is the loss trend in `log.out`.

A cheap, high-value sanity check before investing in fine-tuning: load `ckpt.params` into
`VanillaMolGen_RNN` and sample a few hundred molecules **unconditionally**, then measure
validity with RDKit. CDGCN reports ~95% validity after pretraining + fine-tuning
(`papers/CDGCN.pdf` Table 1); an unconditional pretrained model in the same ballpark is
strong evidence the 46 hours were well spent.

⚠ Doing so needs a small unconditional sampler. `CVanilla_RNN_Builder` in
`Generating_samples.ipynb` is hardcoded for the **conditional** model, and `_decode_step`
calls `get_init()` / `get_action()` closures that pass `c`. A `Vanilla_RNN_Builder`
equivalent does not exist in this repository and would have to be written — roughly 30 lines,
reusing `_decode_step` with `mode='decode_0'` / `'decode_step'` on `VanillaMolGen_RNN`.
See [12](12_BTP_Modification_Guide.md) §8.
