# 07 — Fine-tuning and Conditioning

This is the broken link in the pipeline. Everything upstream is done; nothing downstream can
run until this is solved. Read this before attempting any fine-tuning.

---

## 1. What fine-tuning is supposed to do

```
outputs/pretrain/logs/ckpt.params        (VanillaMolGen_RNN — unconditional, 6,986,506 params)
        │
        ▼  load with allow_missing=True, ignore_extra=True
CVanillaMolGen_RNN                       (adds dense_policy_0 + 6 × linear_c, ≈+1.1 M params)
        │
        ▼  train on paired (ligand SMILES, protein embedding) data
outputs/CTDGD/Dataset{i}/model/{ckpt.params, configs.json}
        │
        ▼
Generating_samples.ipynb  →  CVanilla_RNN_Builder(model_loc)
```

**PAPER** (`papers/CTDDG.pdf` §2.4): "During second phase of training CTDDG is trained to
generate ligand specific to a target protein. For fine-tuning, BindingDB ligands were
labelled using pkCSM to identify ligand with Ames mutagenicity."

**PAPER** (`papers/CDGCN.pdf` §3.3): "Finetuning of CDGCN was done on the BindingDB datasets
by passing pairs of proteins and their known ligands to the model. … Finetuning lasted on
average for 10 epochs with early stopping for the five different folds."

---

## 2. ⚠ The core problem

**The original repository contains the conditional *architecture* but no code that trains it.**

| Component | Where it lives | Present? |
|---|---|---|
| `_TwoLayerDense` | `code/Generating_samples.ipynb` cell 14 | ✅ |
| `CMoleculeGenerator_RNN` | `code/Generating_samples.ipynb` cell 14 | ✅ |
| `CVanillaMolGen_RNN` | `code/Generating_samples.ipynb` cell 14 | ✅ |
| `CVanilla_RNN_Builder` (loads & samples) | `code/Generating_samples.ipynb` cell 16 | ✅ |
| **Conditional data loader** (`CMolRNNLoader`) | — | ❌ **absent** |
| **Conditional training loop** | — | ❌ **absent** |
| **`d{i}_tr_cdgcn.txt` paired data file** | — | ❌ **absent** |

Note the precise framing. `WEEK 1 PROGRESS - TASK 2.md` §11.1 describes this as "the script
that trains the second one does not exist anywhere in the repository" — correct — but goes on
to reconstruct the *architecture* as well. That was unnecessary and, as §5 shows, harmful:
a correct architecture was already sitting in `Generating_samples.ipynb`.

`CMoleculeGenerator_RNN.forward` in the original takes `..., c, ids` and calls
`self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size,
iw_size)` — i.e. it is already wired for a `mode='loss'` training path. Only the loader and
the loop are missing.

---

## 3. What exactly is the condition?

**A single fixed-length real-valued vector `c ∈ ℝ^{N_C}` derived from the target protein's
amino-acid sequence, computed offline by a frozen protein language model.**

Established by reading the consumer code:

```python
# Generating_samples.ipynb cell 18 — generate_samples()
with open(protein_data_path) as f:
    for line in f:
        embeds.append(line.strip().split('\t'))
for e in embeds:
    c.append([float(e_i) for e_i in e])
c = np.array(c)
...
samples = model.sample(n_samples, c=c[i], output_type='graph')
```

One line of tab-separated floats per protein → one vector → one `sample()` call.

### It is NOT:
| Candidate | Verdict | Evidence |
|---|---|---|
| Molecular properties (logP, QED, MW, SAS) | ❌ | No property tensor enters any model. These are computed only in `Evaluation_metrics.ipynb`, after generation |
| A toxicity flag | ❌ | `tox_class` never enters the network. In the paper it weights the **loss**, not the input |
| Multiple targets | ❌ | `c[i]` is one row; `sample()` takes one vector |
| A 3D binding pocket | ❌ | Both papers position this as an advantage over Pocket2Mol — sequence only |
| Protein identity (one-hot) | ❌ for CTDDG | That is the **Li et al.** baseline; its dataset (`d{i}_tr_li.txt`, `np.identity(1000)`) is built separately in `data/data_preprocessing.ipynb` cell 115 |

**So: single-target, single-condition, sequence-derived.** The project's own framing —
"single-target, single-property baseline" (`WEEK 1 PROGRESS - TASK 2.md` §17) — is accurate,
except that the "single property" (toxicity) is not actually active either.

### Where it enters
Two places, both in `CVanillaMolGen_RNN` — see [04](04_Protein_Embeddings.md) §6:
1. `dense_policy_0(c)` → distribution over the first atom type.
2. `linear_c[i](c)[ids, :]` added to the output of **every** graph-convolution layer.

---

## 4. Expected data format

From `code/data_preprocessing_1.ipynb` cells 100–106:

```
<ligand SMILES>\t<e_1>\t<e_2>\t…\t<e_NC>\n
```

written to `data/bindingdb/train_dataset/train_4_org_1042_104/d1_tr_cdgcn.txt`, one line per
protein–ligand pair (227,093 lines for fold 1). The upstream equivalent
(`data/data_preprocessing.ipynb` cell 110) shows a real line:

```
CC[C@@H](C)[C@@H]1NC(=O)…N2C1=O	0.074268565	0.047261816	0.038259566	…
```

Two blockers on producing this file — both documented in
[03](03_Data_and_Preprocessing.md) §6.1 and [04](04_Protein_Embeddings.md) §4:

1. The CDGCN assembly cell (100) **re-inserts** character-tokenisation spaces (`C C O`) into
   ligands that were already loaded correctly as plain SMILES — a line copied from the
   Grechishnikova section.
2. The embedding is **per-residue `(L, 1024)`**, not a pooled `(1024,)` vector, and is written
   with `str(numpy_array)` — producing unparseable bracketed text.

---

## 5. ⚠ `scripts/finetune_cell.py` is non-functional

The reconstruction has **five independent blockers**, any one of which stops it. Listing them
all so nobody spends a day rediscovering them one at a time.

### 5.1 `_TwoLayerDense` is undefined at runtime
`CMoleculeGenerator_RNN.__init__` calls `_TwoLayerDense(self.N_C, self.N_A*3, self.N_A)`, but
that class is defined **only** in `code/Generating_samples.ipynb`, and
`notebooks/02_finetuning.ipynb` execs only the *prefix* of `code/pretraining.ipynb`, which
does not contain it. Verified: `grep -c "_TwoLayerDense" pretraining.ipynb` → **0**.
→ `NameError` on model construction.

### 5.2 `bn_skip` / `linear_skip` are never created
```python
def _build_graph_conv(self, F_h):
    self.conv, self.bn, self.linear_c = [], [], []
    for i, (f_in, f_out) in enumerate(...):
        conv = GraphConv(...); ...
        if i != len(self.F_h) - 1:
            bn = nn.BatchNorm(); ...
            linear_c = nn.Dense(...); ...
    # ← no self.bn_skip, no self.linear_skip
```
but `_graph_conv_forward` ends with
`self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))`.
Because this method **overrides** `VanillaMolGen_RNN._build_graph_conv` (which does create
them), they never exist. → `AttributeError`.

### 5.3 The data parser cannot read the data
```python
parts = line.strip().split('\t')
if len(parts) == 2:
    smi = parts[0]
    embed = ast.literal_eval(parts[1])
```
A real line has `1 + N_C` tab-separated fields, so `len(parts)` is 1025 (or 6166), never 2.
Every line is skipped → `dataset == []` → `IndexError` on `dataset[0][1]`.
`ast.literal_eval` on a single float string would also not yield a list.

### 5.4 Wrong keyword argument
```python
loader_train = CMolRNNLoader(dataset, batch_size_sampler=sampler_train, num_workers=0, k=10, p=0.9)
```
`MolLoader.__init__` accepts `batch_sampler`, not `batch_size_sampler`. → `TypeError`.

### 5.5 Hardcoded non-existent root
```python
PROJECT_ROOT = "/workspace"
data_path = f"{PROJECT_ROOT}/data/bindingdb/train_dataset/train_4_org_1042_104/d{dataset_index}_tr_cdgcn.txt"
```
`/workspace` does not exist on Bhavani. → `FileNotFoundError`. (Note also that the path pins
the fold-1 *directory* while varying the file's `d{i}` prefix — inconsistent for `i ≠ 1`.)

### 5.6 Semantic divergences from the original architecture

Even if the crashes were fixed, the resulting model would **not be checkpoint-compatible**
with `Generating_samples.ipynb`:

| Aspect | Original (`Generating_samples.ipynb`) | Reconstruction (`finetune_cell.py`) | Consequence |
|---|---|---|---|
| `linear_c` count | **6** (one per `F_h` entry) | **5** (skips the last layer) | Different parameter set; last GCN layer receives no protein signal |
| BatchNorm class | custom `BatchNorm(in_channels=f_in)` | `gluon.nn.BatchNorm()` | **Different parameter names** → checkpoint won't load |
| BatchNorm placement | `None` for `i==0`, BN for `i=1..5` | BN for `i=0..4`, none for `i=5` | Shifted by one layer |
| Condition gather | `linear_c(c)[ids, :]` using per-atom molecule index | `nd.repeat(linear_c(c), repeats=X.shape[0] // c.shape[0], axis=0)` | **Wrong for variable-size molecules** — integer division assumes every molecule has the same atom count |
| `_policy_0` | `dense_policy_0(c) + 0.0*policy_0.data(...)` | `nd.softmax(self.dense_policy_0(c), axis=-1)` | **Double softmax** (`_TwoLayerDense.forward` already softmaxes) and drops the term that keeps `policy_0` registered |
| `init` tiling | `nd.tile(unsqueeze(_policy_0(c), 1), [1, iw_size, 1])` | `nd.repeat(c, repeats=iw_size, axis=0)` then `_policy_0` | Different ordering semantics vs `iw_ids` |
| `_likelihood` args | no `tox_class` | passes `tox_class_batch` | Harmless (ignored) but signals confusion |

Item 4 is the conceptually serious one: `linear_c(c)[ids, :]` is precisely the mechanism that
makes conditioning work across molecules of different sizes, and the reconstruction replaced
it with an assumption that is false for essentially every batch.

**Recommendation: do not repair `finetune_cell.py`.** Write a fresh loop that *imports*
`CVanillaMolGen_RNN` from `Generating_samples.ipynb`. See §7.

---

## 6. Hyperparameters — invented vs. paper

`finetune_cell.py` / `notebooks/02_finetuning.ipynb` propose:

| Parameter | Reconstruction | Paper | Verdict |
|---|---|---|---|
| Iterations | 50,000 | "~10 epochs with early stopping" (`CDGCN` §3.3) | ⚠ **invented** |
| Batch size | 16 | 8 (`CDGCN` §3.3, chosen by 5-fold CV; larger untested due to memory) | ⚠ **diverges** |
| Learning rate | 1e-4 | not stated; "follow Li et al." | ⚠ invented (plausible: 10× below pretraining's 1e-3) |
| Gradient clip | 10.0 | not stated; pretraining uses 3.0 | ⚠ invented |
| `k` (paths, K) | 10 | **5** | ⚠ **diverges from the paper** |
| `p` (randomness, α) | 0.9 | **0.8** | ⚠ **diverges from the paper** |
| Early stopping | none | present in paper | ⚠ missing |
| Checkpoint every | 5,000 steps | — | reasonable |
| Trainer step | `trainer.step(batch_size=batch_size)` | pretraining uses `batch_size=1` | ⚠ **inconsistent** — the loss is already meaned, so this divides the gradient by 16 |

10 epochs × 227,093 pairs ÷ batch 8 ≈ **283,866 iterations** for fold 1 — over 5× the proposed
50,000. If you want to follow the paper, use `k=5`, `p=0.8`, `batch_size=8`, and either early
stopping on a validation split or ~280 k iterations.

---

## 7. What a correct fine-tuning stage needs

Not implemented here — this pass is documentation only. Listed so the work is scoped.

1. **Fix the embedding producer** — add `reduce_per_protein` (one line,
   [04](04_Protein_Embeddings.md) §4) and write real floats.
2. **Fix the SMILES writer** — strip the injected spaces in the CDGCN section
   ([03](03_Data_and_Preprocessing.md) §6.1), matching upstream cell 102.
3. **Produce `d1_tr_cdgcn.txt`** for fold 1.
4. **Write `CMolRNNLoader`** — subclass `MolRNNLoader`, yield `(smiles, embedding)`, append
   the dummy `" 1"` label so `process_single` still parses, and append `c` **and** the
   per-atom `ids` tensor (`NX_rep`) to the batch. The `ids` tensor is the part the existing
   reconstruction omits entirely.
5. **Write the loop** — construct `CVanillaMolGen_RNN` with `N_C` read from the data;
   `model.collect_params().initialize(Xavier(), ctx=mx.gpu())`;
   `model.load_parameters(pretrain_ckpt, ctx, allow_missing=True, ignore_extra=True)`;
   Adam; mirror pretraining's logging/checkpoint/resume conventions.
6. **Write `configs.json` including `N_C`** — `CVanilla_RNN_Builder._get_model` does
   `CVanillaMolGen_RNN(N_A, N_B, D=2, **configs)`, so `N_C` **must** be a key in the
   fine-tuned model's `configs.json` or generation fails.
7. **Decide the output path.** `generate_samples()` loads
   `outputs/{model_name}/Dataset{i}/model/` with `model_name='CTDGD'` (note the typo — CTD**G**D).
   ⚠ `Evaluation_metrics.ipynb` cell 16 instead reads `outputs/finetune/log.out`. These are
   different directories; pick one convention and make both agree.

### Weight transfer — what actually carries over

`allow_missing=True, ignore_extra=True` is correct. Loading `VanillaMolGen_RNN` weights into
`CVanillaMolGen_RNN`:

| Carried over from pretraining | Freshly initialised |
|---|---|
| `embedding_atom`, `embedding_mask` | `dense_policy_0.*` (≈213 k params) |
| all 6 `GraphConv.W` | `linear_c[0..5]` (≈885 k params) |
| all `BatchNorm` layers, `bn_skip`, `linear_skip` | |
| `dense` | |
| `policy_0` (kept alive by the `0.0 *` term) | |
| `Policy.*` | |
| `rnn` (GRU, 5.51 M params — 79% of the model) | |

≈6.99 M of ≈8.08 M parameters (86%) transfer. That is exactly the point of the two-phase
design: the expensive GRU and graph encoder are already trained; fine-tuning mostly teaches
the two new projections how to use `c`, while everything adapts.

### Nothing is frozen

Neither paper mentions freezing, and no `grad_req='null'` appears in any conditional class.
`gluon.Trainer(model.collect_params(), opt)` optimises **every** parameter. So fine-tuning
updates the whole network, not just the conditioning layers. (`BatchNorm.running_mean` /
`running_var` carry `grad_req='null'` — but those are buffers, not trainable weights.)

---

## 8. Fine-tuning status

| Item | Status |
|---|---|
| Conditional architecture | ✅ exists (`Generating_samples.ipynb` cell 14) |
| Conditional data loader | ❌ absent |
| Conditional training loop | ❌ absent (reconstruction non-functional) |
| Paired data file | ❌ absent |
| Protein embeddings | ❌ absent, and producer is defective |
| Fine-tuned checkpoint | ❌ absent |
| `outputs/CTDGD/` | ❌ does not exist |
| `outputs/finetune/` | ❌ does not exist |

**NOT STARTED — and blocked on the protein-embedding fix first.**
