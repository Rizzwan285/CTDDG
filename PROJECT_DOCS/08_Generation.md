# 08 — Generation

Source: `code/Generating_samples.ipynb`, cells 14 (model), 16 (decoder + builder),
18 (driver), 19 (parameters).

**Status: NOT STARTED.** Generation requires a fine-tuned conditional checkpoint, which does
not exist ([07](07_Finetuning_and_Conditioning.md)). This document describes what the code
would do.

---

## 1. Overview

```
d{i}_te_embeddings_run{r}.txt        outputs/CTDGD/Dataset{i}/model/
  (one protein per line,              ├── configs.json   (architecture + N_C)
   N_C tab-separated floats)          └── ckpt.params    (fine-tuned weights)
        │                                       │
        └──────────────┬────────────────────────┘
                       ▼
              CVanilla_RNN_Builder(model_loc, gpu_id=0)
                       │
          for each protein i:  model.sample(n_samples, c=c[i], output_type='graph')
                       │
                       ▼
   ┌──────────────────────────────────────────────────────┐
   │ empty graph                                          │
   │   → decode_0  : sample first atom from _policy_0(c)   │
   │   → decode_step (repeat ≤ 99 times):                  │
   │        embed → GCN(+c) → dense → GRU → Policy         │
   │        → sample from {append, connect, end}           │
   │   → stop when all molecules emit `end` or hit 100     │
   └──────────────────────────────────────────────────────┘
                       │
                       ▼
   graph list → get_mol_from_graph_list(sanitize=True) → Chem.MolToSmiles
                       │
                       ▼
   Protein{i+1}_generated_samples.csv    (column: smiles)
```

---

## 2. Inputs

### Model checkpoint
```python
class Builder:
    def __init__(self, model_loc, gpu_id=0):
        with open(os.path.join(model_loc, 'configs.json')) as f:
            configs = json.load(f)
        self.mdl = self.__class__._get_model(configs)
        self.ctx = mx.gpu(gpu_id) if gpu_id is not None else mx.cpu()
        self.mdl.load_parameters(os.path.join(model_loc, 'ckpt.params'), ctx=self.ctx)

class CVanilla_RNN_Builder(Builder):
    @staticmethod
    def _get_model(configs):
        return CVanillaMolGen_RNN(get_mol_spec().num_atom_types,
                                  get_mol_spec().num_bond_types, D=2, **configs)
```

⚠ Note: **no** `allow_missing` / `ignore_extra` here — the checkpoint must match the
architecture exactly. ⚠ `configs.json` **must contain `N_C`**, since it is splatted straight
into the constructor. The pretraining `configs.json` does not have it.

`model_loc` = `outputs/{model_name}/Dataset{dataset_index}/model/` with
`model_name = 'CTDGD'` (a typo for CTDDG that is now load-bearing — it appears in
`Generating_samples.ipynb`, `Evaluation_metrics.ipynb` and `Molecular_docking.ipynb`).

### Protein input
```python
with open(protein_data_path) as f:
    for line in f:
        embeds.append(line.strip().split('\t'))
for e in embeds:
    c.append([float(e_i) for e_i in e])
c = np.array(c)          # shape [n_proteins, N_C]
```
Path: `data/d{dataset_index}_te_embeddings_run{run}.txt`. ⚠ **No code in this repository
produces this file** — see [03](03_Data_and_Preprocessing.md) §8.

The protein's identity in the output is purely positional: row `i` of the file becomes
`Protein{i+1}_generated_samples.csv`. Keep the row order stable and record it.

### Parameters (hardcoded, `Generating_samples.ipynb` cell 19)
```python
dataset_index = 2
run           = 1
n_samples     = 1000
model_name    = 'CTDGD'
```
⚠ `notebooks/03_generation.ipynb` presents its own `DATASET_INDEX / RUN / N_SAMPLES /
MODEL_NAME` cell, but then launches `Generating_samples.ipynb` in a **subprocess**, which
re-reads these hardcoded values. The wrapper's configuration has no effect.

---

## 3. Sampling — step by step

### 3.1 Initialisation (`_decode_step` with `X=None`)
```python
init = get_init()                                    # [num_samples, 65] probabilities
if random:
    X = [np.random.choice(np.arange(init.shape[1]), 1, p=init[i, :])[0]
         for i in range(init.shape[0])]
else:
    X = np.argmax(init, axis=1)
A  = np.zeros((0, 3)); NX = last_action = np.ones([n]); NA = np.zeros([n])
finished = np.array([False] * n)
```
`get_init()` sets `mdl.mode = 'decode_0'` and calls `mdl(c)` → `_policy_0(c)`, the
`_TwoLayerDense` softmax over the 65 atom types. Every molecule in the batch starts from the
**same** protein vector but samples its own first atom.

### 3.2 The decoding loop
```python
count = 1
h = np.zeros([self.mdl.N_rnn, num_samples, self.mdl.F_c[-1]], dtype=np.float32)   # [3, n, 512]
while not np.all(finished) and count < 100:
    def get_action(inputs):
        self.mdl.mode = 'decode_step'
        _h = nd.array(h[:, ~finished, :], ...)
        _c = nd.array(c[~finished, :], ...)
        _X, _A_sparse, _NX, _NX_rep, _mask, _NX_cum = self.to_nd(inputs)
        _append, _connect, _end, _h = self.mdl(_X, _A_sparse, _NX, _NX_rep, _mask,
                                               _NX_cum, _h, _c, _NX_rep)
        h[:, ~finished, :] = _h[0].asnumpy()
        return _append.asnumpy(), _connect.asnumpy(), _end.asnumpy()
    outputs = _decode_step(X, A, NX, NA, last_action, finished,
                           get_init=None, get_action=get_action, random=random)
    X, A, NX, NA, last_action, finished = outputs
    count += 1
```

Notes:
- **Hard cap of 100 steps.** A molecule that never emits `end` is truncated at 100 actions and
  still returned. `MoleculeSpec.max_iter = 120` is a *different* cap, used only for the
  training-time RNN tensor width.
- Only unfinished molecules are fed forward (`~finished` masking), so the batch shrinks.
- The GRU hidden state `h` lives in NumPy between steps and is spliced back per-molecule.
- The last positional argument is `_NX_rep` passed as `ids` — the per-atom → per-molecule map
  that makes `linear_c(c)[ids, :]` correct for variable-size molecules
  ([04](04_Protein_Embeddings.md) §6.2).

### 3.3 Choosing an action

**Stochastic (`random=True`, the default used by `generate_samples`):**
```python
def _rand_id(*_x):
    _x_reshaped = [np.reshape(_xi, [-1]) for _xi in _x]
    _p = np.concatenate(_x_reshaped)
    _p = _p / np.sum(_p)
    _rand_index = np.random.choice(np.arange(_p.shape[0]), 1, p=_p)[0]
    ...
action_type, action_index, p_step = _rand_id(append_i, connect_i, np.array([end_i]))
```
All action probabilities for a molecule — `append` `[NX, 65, 4]`, `connect` `[NX, 4]`,
`end` scalar — are flattened into one vector, renormalised, and sampled once. So the choice of
*action type*, *which atom*, *which new atom type* and *which bond type* is a **single joint
categorical draw**.

**Greedy (`random=False`):**
```python
_argmax = lambda _x: np.unravel_index(np.argmax(_x), _x.shape)
if end_val >= append_val and end_val >= connect_val:   action_type = 2
elif append_val >= connect_val and append_val >= end_val: action_type = 0
else: action_type = 1
```

⚠ **There is no beam search and no temperature parameter.** The only knobs are `random`
(sample vs argmax) and `num_samples`. See §7.

### 3.4 Applying the action
| Action | State update |
|---|---|
| `end` (type 2) | `finished[unfinished_id] = True` |
| `append` (type 0) | insert `atom_type` into `X` at `cumsum[id]`; `NX += 1`; add bond `[append_pos, NX[id], bond_type]`; `last_action = 1` |
| `connect` (type 1) | add bond `[NX[id]-1, connect_ps, bond_type]`; `NA += 1`; `last_action = 0` |

`last_action` feeds `last_append_mask` on the next step, so the model knows whether the last
operation added an atom (mask 1) or a bond (mask 2) — see [05](05_Model_Architecture.md) §B.4.

A special case handles the second atom: when `A.shape[0] == 0`, only `append` is legal, so the
code takes `argmax` over `append` reshaped to `[-1, 65*4]` **for every molecule at once** —
⚠ note this branch is **always deterministic**, even when `random=True`. So the second atom is
greedy while every subsequent step is sampled. Consequence: molecules sharing a first atom
also share their second. INFERRED — NOT EXPLICITLY CONFIRMED that this is intentional; it is
not mentioned in either paper.

---

## 4. Graph → molecule → SMILES

```python
cumsum_X_ = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
cumsum_A_ = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()
for ...:
    graph_list.append([X[cumsum_X_pre:cumsum_X_post], A[cumsum_A_pre:cumsum_A_post, :]])
```
Then:
```python
def get_mol_from_graph(X, A, sanitize=True):
    try:
        mol = Chem.RWMol(Chem.Mol())
        for i, atom_type in enumerate(X.tolist()):
            mol.AddAtom(get_mol_spec().index_to_atom(atom_type))
        for atom_id1, atom_id2, bond_type in A.tolist():
            get_mol_spec().index_to_bond(mol, atom_id1, atom_id2, bond_type)
    except:
        return None
    if sanitize:
        try:
            mol = mol.GetMol(); Chem.SanitizeMol(mol); return mol
        except:
            return None
    return mol
```

`index_to_atom` restores the `(symbol, formal_charge, num_explicit_Hs)` triple.
`Chem.SanitizeMol` performs valence checking, aromaticity perception and kekulisation —
**this is where chemical validity is enforced**, after the fact. Failures return `None`.

---

## 5. Validity, duplicates, sanitization

### Validity
Nothing during decoding prevents an impossible molecule (e.g. pentavalent carbon). Validity
is purely `SanitizeMol` succeeding. `generate_samples` filters twice:

```python
samples = [m for m in model.sample(n_samples, c=c[i], output_type='graph') if m is not None]
smiles_list = [Chem.MolToSmiles(m) for m in get_mol_from_graph_list(samples, sanitize=True) if m is not None]
```

⚠ The first filter is a no-op: with `output_type='graph'`, `sample()` returns raw
`[X, A]` pairs and never `None`. The real filter is the second.

So `len(smiles_list) ≤ n_samples`, and the shortfall **is** the invalidity rate. That is
exactly how `Evaluation_metrics.ipynb` computes validity:
`valid % = 100 * len(df) / beam_size`.

### Duplicates
**Not removed at generation time.** The CSV can contain repeated SMILES.
`Evaluation_metrics.ipynb` measures uniqueness via `df['smiles'].nunique()` rather than
deduplicating.

⚠ This differs from CDGCN §3.5: "Duplicate molecules were discarded from the sets of generated
molecules." The published metric definitions and this code therefore treat duplicates
differently — see [09](09_Evaluation_and_Docking.md) §2.

### Canonicalisation
`Chem.MolToSmiles(m)` emits RDKit canonical SMILES, so duplicate detection by string equality
is sound.

⚠ `molvs.standardize_smiles` is **imported** at the top of `Generating_samples.ipynb`
(cell 2) but **never called**. Dead import.

---

## 6. Outputs

```python
output_base = f'/home/muhamed/repo/CTDDG/outputs/{model_name}/Dataset{dataset_index}'
os.makedirs(f'{output_base}/model', exist_ok=True)
os.makedirs(f'{output_base}/generated_samples/{n_samples}/run{run}', exist_ok=True)
...
smiles_df.to_csv(f'.../generated_samples/{n_samples}/run{run}/Protein{i+1}_generated_samples.csv', index=False)
...
with open(f'.../run{run}/samples_evaluation.txt', 'a+') as eval_f:
    eval_f.write(f'Total time: {endTime - startTime} seconds')
```

```
outputs/CTDGD/Dataset{i}/generated_samples/{n_samples}/run{r}/
├── Protein1_generated_samples.csv        # single column: smiles
├── Protein2_generated_samples.csv
├── …
└── samples_evaluation.txt
```

`Evaluation_metrics.ipynb` later **overwrites each CSV in place**, adding MW, logP, QED, HD,
HA, RB, TPSA, SAS columns.

⚠ **Timing bug in `samples_evaluation.txt`:**
```python
startTime = int(round(time.time() * 1000))     # milliseconds
endTime   = int(round(time.time()))            # seconds
eval_f.write(f'Total time: {endTime - startTime} seconds')
```
Mismatched units — the reported number is meaningless (a large negative value). Cosmetic only.

⚠ The `os.makedirs(..., exist_ok=True)` calls are a **local fix** applied by
`scripts/patch_notebooks.py`, replacing six fragile `os.system('mkdir ...')` calls.

---

## 7. How many samples, and what "beam size" means

The papers report `S_10`, `S_100`, `S_1000` — sets of 10, 100 and 1000 generated molecules
(`papers/CDGCN.pdf` §3.5). The CTDDG tables use 100 samples per protein across 104 test
proteins × 10 runs (`papers/CTDDG.pdf` Tables 1–3).

`Evaluation_metrics.ipynb` and `Molecular_docking.ipynb` loop over
`for beam_size in [10, 100, 1000]` and read
`generated_samples/{beam_size}/run{run}/...`.

⚠ **"beam_size" is a misnomer here.** It is `n_samples` — the number of *stochastic samples*.
The name is inherited from the Grechishnikova transformer baseline, where beam size genuinely
controlled generation: "The generation of `S_N` by Grechishnikova was done by setting the beam
size hyperparameter equal to `N`" (`papers/CDGCN.pdf` §3.5). **No beam search exists in this
codebase.**

To produce the 10 runs the papers average over, call `generate_samples(..., run=r, ...)` for
`r = 1..10` — each run re-samples stochastically and writes to its own `run{r}/` directory.

---

## 8. Runtime expectations

`papers/CDGCN.pdf` Table 1 (RTX 2080 Ti, averaged over proteins):

| Set | CDGCN, novel proteins |
|---|---|
| `S_10` | 20.7 ± 0.91 s |
| `S_100` | 35.1 ± 1.47 s |
| `S_1000` | 141.7 ± 5.72 s |

For 104 test proteins at `S_1000`: ≈ 104 × 142 s ≈ **4.1 hours per run** on that hardware.
Ten runs ≈ 41 hours. Plan accordingly on Bhavani ([10](10_Cluster_and_Execution.md)).

⚠ Sub-linear scaling with `N` is expected — molecules are decoded in parallel within one
`sample()` call, so larger `N` amortises the per-step overhead.

---

## 9. Blockers before generation can run

| # | Blocker | See |
|---|---|---|
| 1 | No fine-tuned conditional checkpoint | [07](07_Finetuning_and_Conditioning.md) |
| 2 | No `outputs/CTDGD/Dataset{i}/model/configs.json` containing `N_C` | §2 |
| 3 | No `data/d{i}_te_embeddings_run{r}.txt`, and no code produces it | [03](03_Data_and_Preprocessing.md) §8 |
| 4 | Protein embedding producer defective | [04](04_Protein_Embeddings.md) §4 |

---

## 10. Additional issues to be aware of

| Issue | Location | Impact |
|---|---|---|
| `np.bool` used | `_decode_step`, `finished = np.array(..., dtype=np.bool)` | **Removed in NumPy ≥1.24.** Env has 1.23.5, so it works — but pinning matters. Would raise `AttributeError` on a newer NumPy |
| Second atom always greedy | `_decode_step`, `A.shape[0] == 0` branch | Reduces diversity; molecules sharing atom 1 share atom 2 |
| 100-step cap silently truncates | `while ... count < 100` | Large molecules emerge incomplete but are still returned and counted |
| `model` reloaded once, reused for all proteins | `generate_samples` | Fine — `c` is the only per-protein input |
| Single GPU | `gpu_id=0` | Multiple datasets/runs could be parallelised across GPUs by launching separate processes |
| `samples_evaluation.txt` opened `'a+'` | | Re-running a run appends rather than overwrites |
