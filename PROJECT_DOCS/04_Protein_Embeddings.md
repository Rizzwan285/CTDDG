# 04 — Protein Embeddings

This is the **weakest-verified link in the whole pipeline** and the immediate blocker on
fine-tuning. Read it carefully before doing any work downstream of pretraining.

---

## 1. Intended flow

```
protein amino-acid sequence  (e.g. "MEPGSDDFLPPPECPVFEPSW…")
        │
        ▼
  pretrained protein language model            ← frozen; never trained here
        │
        ▼
  fixed-length vector  c ∈ ℝ^{N_C}             ← one vector per protein
        │
        ├──► _TwoLayerDense → softmax → P(first atom type)          [init action]
        │
        └──► linear_c_i(c)  added into every graph-convolution layer [all other actions]
        │
        ▼
  molecule generator (CVanillaMolGen_RNN)
```

The key requirement, imposed by the consumer code, is **one fixed-length vector per
protein** — not per residue.

---

## 2. What the papers say

**PAPER**, `papers/CDGCN.pdf` §2.2:

> "`Embed_p` is based on a pre-trained network from Dallago et al.[5] which was trained on
> protein sequences for the task of predicting protein function from its sequence."

Reference [5] is **Dallago, Schütze, Heinzinger, Olenyi, Littmann, Lu, Yang, Min, Yoon,
Morton, Rost — "Learned embeddings from deep learning to visualize and predict protein sets",
Curr Protoc 1(5):e113 (2021)**. That paper is the **`bio_embeddings` toolkit** — a *collection*
of protein language models, not a single model.

⚠ **Neither paper names which embedder inside `bio_embeddings` was used, nor states the
embedding dimension `N_C`.** This is a genuine under-specification in the published work.

`papers/CTDDG.pdf` inherits the architecture wholesale and adds nothing on this point beyond
"CTDDG takes input a protein sequence transferred into embedding that contains useful
information about its function" (§2.2).

---

## 3. What the code does — and the two versions disagree

### 3.1 Upstream CDGCN / DGGNP: **ProSE (Bepler et al.)**, pooled

`data/data_preprocessing.ipynb` (the upstream notebook, with outputs intact), markdown cell
76: *"Obtaining protein embeddings using pretrained ProSE model from Bepler et al."*

```python
# cell 77
def give_embeddings_bepler(filename):
    os.system(f'python prose/embed_sequences.py --pool avg -o {filename}_embeddings_bepler.h5 {filename}_proteins_bepler.fa')
    embeddings = []
    with h5py.File(filename+'_embeddings_bepler.h5', "r") as f:
        for key in f.keys():
            embeddings.append(list(f[key]))
    return embeddings
```

Note `--pool avg`: ProSE averages over residues and emits **one fixed-length vector per
protein**. Stored output of cell 79 (`len(d1_tr_embeds[100])`) is **6165** — the ProSE
multi-task representation width.

So for the upstream model: **N_C = 6165**, and the file written is genuinely
`<float>\t<float>\t…`. Confirmed by cell 108's stored output:
```
'0.049526587\t0.059723236\t0.06336489\t0.045884926\t…'
```
and cell 107 showing exactly 122 lines for fold 3's 122 test proteins. This is precisely the
format `generate_samples()` parses.

### 3.2 CTDDG `code/`: **ProtTrans BERT-BFD**, *not* pooled

`code/data_preprocessing_1.ipynb` markdown cell 74 keeps the title *"Obtaining protein
embeddings using pretrained model from Dallago et al."* but the code changed:

```python
# cell 75
from bio_embeddings.embed import ProtTransBertBFDEmbedder

def give_bioembeddings(protein_list):
    embedder = ProtTransBertBFDEmbedder()
    embedding_list = []
    for protein in protein_list:
        embedding_list.append(embedder.embed(protein))
    return embedding_list
```

**Provenance check:** `git show HEAD:code/data_preprocessing_1.ipynb | grep -c
ProtTransBertBFDEmbedder` → 2, and `grep -c 'bepler\|prose'` → 0. The substitution is in the
CTDDG authors' committed upload (commit `47d37e3`, 2024-08-09). It is **their change**, not a
local porting artefact.

---

## 4. ⚠ The confirmed defect

`ProtTransBertBFDEmbedder.embed()` returns a **per-residue** array of shape `(L, 1024)`,
where `L` is the sequence length. The fixed-size vector requires a separate call.

Verified directly in the installed package
(`~/.conda/envs/ctddg_env/lib/python3.9/site-packages/bio_embeddings/`):

```python
# embed/prottrans_base_embedder.py
@staticmethod
def reduce_per_protein(embedding):
    return embedding.mean(axis=0)

def embed(self, sequence: str) -> ndarray:
    [embedding] = self.embed_batch([sequence])
    return embedding
```

and class metadata:
```
ProtTransBertBFDEmbedder.name = 'prottrans_bert_bfd'
ProtTransBertBFDEmbedder.embedding_dimension = 1024
```

**`give_bioembeddings` never calls `reduce_per_protein`.** So it returns `(L, 1024)` arrays.
The writer then does:

```python
# cell 78
for embed in d1_tr_embeds:
    for e in embed:            # e is a 1024-vector, NOT a float
        f.write(str(e)+'\t')
    f.write('\n')
```

`str(numpy_array_of_1024_floats)` produces `'[0.123 0.456 ... 0.789]'` — bracketed and, for
1024 elements, **NumPy-truncated with an ellipsis**. The resulting file cannot be parsed by
the consumer:

```python
# Generating_samples.ipynb cell 18
embeds.append(line.strip().split('\t'))
c.append([float(e_i) for e_i in e])        # ValueError on '[0.123'
```

**Verdict: as written, `code/data_preprocessing_1.ipynb` cannot produce a usable protein
embedding file.** The missing operation is a single call:

```python
embedding_list.append(embedder.reduce_per_protein(embedder.embed(protein)))   # → (1024,)
```

Corroborating evidence that the pooling is what was intended:
- The upstream ProSE version used `--pool avg`.
- `scripts/finetune_cell.py` comments `# Number of conditional variables (e.g. 1024 for protein embedding)`.
- `WEEK 1 PROGRESS - TASK 2.md` §18 records a verification test producing a `(22, 1024)`
  embedding — i.e. a 22-residue peptide, per-residue — showing the shape was observed but its
  implication not followed through.

⚠ **The embedding files are not in this repository**, so what the CTDDG authors *actually*
produced is **NOT DETERMINED FROM AVAILABLE SOURCES**. INFERRED — NOT EXPLICITLY CONFIRMED:
they applied a mean-pool somewhere not captured in the committed notebook.

---

## 5. Dimensions summary

| Source | Model | Per-protein dimension `N_C` | Status |
|---|---|---|---|
| `papers/CDGCN.pdf` §2.2 | "Dallago et al." (unnamed) | **not stated** | ⚠ under-specified |
| `papers/CTDDG.pdf` §2.2 | same, inherited | **not stated** | ⚠ under-specified |
| `data/data_preprocessing.ipynb` (upstream) | ProSE / Bepler, `--pool avg` | **6165** | ✅ confirmed by stored output |
| `code/data_preprocessing_1.ipynb` (CTDDG) | ProtTrans BERT-BFD | **1024** *if pooled* | ⚠ not pooled as written |
| `scripts/finetune_cell.py` | reads `len(dataset[0][1])` at runtime | comment says 1024 | reconstruction |
| `outputs/pretrain/logs/configs.json` | — | **absent** — the pretrained model is unconditional and has no `N_C` | ✅ |

`N_C` is not a free choice at generation time: `CVanilla_RNN_Builder._get_model` passes
`**configs` straight into `CVanillaMolGen_RNN`, so `N_C` must be present in the *fine-tuned*
model's `configs.json` and must match the width of every row in the embedding file.

---

## 6. Where the embedding enters the model

Two entry points, both in `code/Generating_samples.ipynb` cell 14.

### 6.1 The first atom — `_policy_0`

```python
class CMoleculeGenerator_RNN(MoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, *a, **kw):
        self.N_C = N_C
        super().__init__(...)
        with self.name_scope():
            self.dense_policy_0 = _TwoLayerDense(self.N_C, self.N_A * 3, self.N_A)

    def _policy_0(self, c):
        return self.dense_policy_0(c) + 0.0 * self.policy_0.data(c.context)
```

`_TwoLayerDense(input_size=N_C, hidden_size=3·N_A, output_size=N_A)`:
```
c  →  Dense(3·N_A, no bias)  →  BatchNorm  →  ReLU  →  Dense(N_A, bias)  →  softmax
```
Output: a probability distribution over the 65 atom types — *which atom to start the molecule
with, given this protein*. This corresponds exactly to `FNN_0` in the papers.

The `+ 0.0 * self.policy_0.data(...)` term is a deliberate trick: `policy_0` is the
*unconditional* learnable start-atom vector inherited from `MoleculeGenerator`. Multiplying by
zero keeps the parameter registered (so a pretrained checkpoint containing it still loads
cleanly) while contributing nothing to the forward pass.

### 6.2 Every graph-convolution layer — `linear_c`

```python
class CVanillaMolGen_RNN(CMoleculeGenerator_RNN):
    def _build_graph_conv(self, F_h):
        ...
        self.linear_c = []
        for i, f_out in enumerate(self.F_h):                     # 6 layers
            linear_c = nn.Dense(f_out, use_bias=False, in_units=self.N_C)
            self.register_child(linear_c)
            self.linear_c.append(linear_c)

    def _graph_conv_forward(self, X, A, c, ids):
        X_out = [X]
        for conv, bn, linear_c in zip(self.conv, self.bn, self.linear_c):
            X = X_out[-1]
            if bn is not None:
                X_out.append(conv(self.activation(bn(X)), A) + linear_c(c)[ids, :])
            else:
                X_out.append(conv(X, A) + linear_c(c)[ids, :])
        X_out = nd.concat(*X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))
```

One `nn.Dense(f_out, in_units=N_C)` per GCN layer, so **6 projections** for
`F_h = [32, 64, 128, 128, 256, 256]`. These are `FNN_1 … FNN_l` in the papers, and the
addition is the papers' "the task-specific protein embedding is summed with the output node
embedding for each FNN and GCN layer".

**The `ids` index is essential.** `linear_c(c)` has shape `[n_molecules, f_out]`; `ids` is a
per-atom vector mapping each atom row to its molecule, so `linear_c(c)[ids, :]` has shape
`[n_atoms, f_out]` and broadcasts correctly across molecules of *different sizes*. At
generation time `CVanilla_RNN_Builder.sample` passes `_NX_rep` as `ids`:

```python
_append, _connect, _end, _h = self.mdl(_X, _A_sparse, _NX, _NX_rep, _mask, _NX_cum, _h, _c, _NX_rep)
#                                                     ↑ NX_rep                            ↑ ids
```

---

## 7. Parameter count added by conditioning

For `N_C = 1024`, `N_A = 65`, `F_h = [32, 64, 128, 128, 256, 256]`:

| Component | Shape | Parameters |
|---|---|---|
| `dense_policy_0.input` | 1024 × 195 | 199,680 |
| `dense_policy_0.bn_input` | 195 × 4 | 780 |
| `dense_policy_0.output` | 195 × 65 + 65 | 12,740 |
| `linear_c[0..5]` | 1024 × (32+64+128+128+256+256) | 884,736 |
| **Total new** | | **≈ 1,097,936** |

Against 6,986,506 pretrained parameters, conditioning adds ~15.7%. All of it is randomly
initialised at the start of fine-tuning — the pretrained checkpoint contains none of it.

*(These are arithmetic from the layer definitions, not measured — no conditional model has
been instantiated in this repository.)*

---

## 8. Is the protein encoder frozen?

**Yes — it is not part of the network at all.**

Both papers describe `Embed_p` as "a pre-trained network", and in the implementation it never
appears inside any MXNet block. Embeddings are computed **offline** in the preprocessing
notebook (PyTorch / `bio_embeddings`), written to a text file, and read back as plain floats.
The MXNet model only ever sees the numbers.

Consequences worth internalising:
- No gradient can flow into the protein encoder. Only `dense_policy_0` and `linear_c[i]` learn
  how to *use* `c`.
- Changing the protein embedding model means regenerating the data files and retraining the
  conditional layers — but **not** the pretrained unconditional weights.
- Generation for a brand-new target requires running the embedder on that FASTA first. The
  CDK2 case study did exactly this (**PAPER**, `papers/CTDDG.pdf` §4: "The FASTA sequence of
  human CDK2 (Uniprot ID: P24941) was utilized to generate inhibitors").

---

## 9. Environment status

Verified in `ctddg_env`:

| Item | Status |
|---|---|
| `bio_embeddings` | ✅ installed |
| `torch` | ✅ 1.9.1+cu102 |
| `ProtTransBertBFDEmbedder` import | ✅ works |
| `embedding_dimension` | ✅ 1024 |
| Model weights cached | ✅ `~/.cache/bio_embeddings/prottrans_bert_bfd/` |
| ProSE / Bepler (`prose/embed_sequences.py`) | ❌ not present |

So the **ProtTrans path is immediately runnable** (weights already downloaded); the upstream
**ProSE path is not** without installing ProSE separately.

---

## 10. Summary of discrepancies

| # | Discrepancy | Severity |
|---|---|---|
| 1 | Papers cite "Dallago et al." (a toolkit) without naming the embedder or dimension | Medium — blocks exact reproduction |
| 2 | Upstream CDGCN used **ProSE/Bepler, 6165-d, avg-pooled**; CTDDG `code/` uses **ProtBert-BFD, 1024-d** | High — CTDDG results are not directly comparable to CDGCN's on this axis |
| 3 | `give_bioembeddings` omits `reduce_per_protein` → per-residue `(L,1024)` instead of `(1024,)` | **Critical** — blocks fine-tuning |
| 4 | Writer `str(e)` on an array produces unparseable bracketed/truncated text | **Critical** — same root cause |
| 5 | `data/d{i}_te_embeddings_run{r}.txt` (generation input) is produced by no code in the repo | High — blocks generation |
| 6 | No embedding file survives in the repo, so what the authors ran cannot be confirmed | Medium |

Fixing #3 is a one-line change and is the **single highest-leverage action** available right
now. It is deliberately **not** applied — this pass is documentation only. See
[12_BTP_Modification_Guide.md](12_BTP_Modification_Guide.md) §2 for the proposed change and
[11_Current_State_and_Next_Steps.md](11_Current_State_and_Next_Steps.md) for sequencing.
