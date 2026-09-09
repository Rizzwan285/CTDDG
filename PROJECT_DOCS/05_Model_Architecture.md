# 05 — Model Architecture

Everything below is reverse-engineered from `code/pretraining.ipynb` (unconditional) and
`code/Generating_samples.ipynb` cell 14 (conditional). Only components that actually exist
are documented.

**Validation of this reading:** summing the parameters of every layer described here gives
**6,986,506**, which matches exactly the value printed by the completed run
(`code/pretraining_executed.ipynb`, final output: `Number of parameters in the model:
6986506`). The breakdown is in §9.

---
---

# PART A — INTUITIVE

## A.1 The model draws a molecule like a person sketching

It starts with a blank page and repeatedly asks *"what next?"*, choosing one of four moves:

| Move | Meaning |
|---|---|
| `init` | place the very first atom |
| `append` | draw a **new atom** and connect it to an existing atom |
| `connect` | draw a **new bond** between the atom I just placed and some earlier atom (closes a ring) |
| `end` | stop; the molecule is finished |

Because it grows the molecule one move at a time, it can produce molecules of any size —
unlike models that emit a fixed-size adjacency matrix in one shot. Both papers call this out
as a strength (`papers/CDGCN.pdf` §1).

## A.2 How it decides

Before every move the model re-reads the partially drawn molecule:

1. **Look up each atom.** Each atom's type becomes a 16-dimensional vector.
2. **Let atoms talk to their neighbours.** Six rounds of graph convolution. Each round mixes
   an atom's vector with those of its neighbours — *separately per bond type*, and separately
   for atoms 2 and 3 bonds away. After six rounds every atom "knows" a good deal about its
   surroundings.
3. **Summarise the whole molecule.** Average all atom vectors → one vector for the molecule.
4. **Remember the story so far.** A GRU consumes that summary at every step, so the model
   remembers what it has been building rather than judging only the current snapshot.
5. **Score every possible move.** For each atom, score every (new atom type, bond type)
   pairing and every (bond type) ring closure; plus one score for "stop".
6. **Sample a move** from those scores and apply it.

## A.3 Where the protein comes in

In the **conditional** model, the protein's vector `c` is injected twice: it decides the
probability distribution over the *first* atom, and it is added into *every* graph
convolution layer, so it colours the representation of every atom at every step.

In the model that was **actually trained** (unconditional), there is no `c`: the first-atom
distribution is a single learnable 65-vector, and nothing is injected into the convolutions.

---
---

# PART B — TECHNICAL

## B.1 Configuration

From `code/pretraining.ipynb` (hyperparameter block) and `outputs/pretrain/logs/configs.json`:

```json
{"F_e": 16, "F_h": [32, 64, 128, 128, 256, 256], "F_skip": 256,
 "F_c": [512], "Fh_policy": 128, "activation": "relu", "N_rnn": 3}
```

| Symbol | Value | Meaning |
|---|---|---|
| `N_A` | **65** | atom types (`len(atom_types.txt)`) |
| `N_B` | **4** | bond types — AROMATIC, SINGLE, DOUBLE, TRIPLE |
| `D` | **2** | receptive-field size; adds distance-2 and distance-3 neighbourhoods |
| `F_e` | 16 | atom embedding width |
| `F_h` | [32,64,128,128,256,256] | GCN layer widths (6 layers) |
| `F_skip` | 256 | skip-connection width |
| `F_c` | [512] | dense stack after graph conv |
| `Fh_policy` | 128 | hidden width inside the policy head |
| `N_rnn` | 3 | GRU layers |
| `k` / `K` | 5 | generation paths sampled per molecule |
| `p` / `α` | 0.8 | coefficient of randomness in `Q_α` |
| batch size | 8 | molecules per step |

`D=2` is passed positionally at construction:
`VanillaMolGen_RNN(num_atom_types, num_bond_types, D=2, **configs)`.

## B.2 Molecular representation

### Atoms
`MoleculeSpec` reads `data/atom_types.txt` into a list of `(symbol, formal_charge,
num_explicit_Hs)` triples. `get_atom_type(atom)` returns `self.atom_types.index(triple)`.
**The integer identity of an atom type is its line number in that file** — see
[03](03_Data_and_Preprocessing.md) §4 for why this makes the file effectively immutable.

### Bonds
```python
self.bond_orders = [Chem.BondType.AROMATIC, Chem.BondType.SINGLE,
                    Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
```
Hardcoded, order-significant, index 0–3.

### Graph → tensors
`get_graph_from_smiles` produces `X_0` (int32 atom-type array) and
`A_0` (`[n_bonds, 3]` = `[begin_idx, end_idx, bond_type]`), plus `atom_ranks` from
`Chem.CanonicalRankAtoms`.

### The 6-slice adjacency `A_list`
`merge_single_0` splits the batch adjacency into six sparse matrices:
```python
A_split = [A_0[A_0[:,2] == i, :2] for i in range(num_bond_types)]   # 4 bond-type slices
A_split.extend([D_0_2, D_0_3])                                       # distance-2, distance-3
```
`get_d(A, X)` computes `D_2` and `D_3` by sparse matrix powers (`A²`, `A³`), removing
direct-bond pairs from `D_3`. So `len(A_list) == N_B + D == 6`, exactly the second term of
the CDGCN graph-convolution equation.

## B.3 Layer inventory

| Class | Role |
|---|---|
| `GraphConvFn(Function)` | Sparse `A @ X` with a matching backward (assumes `A` symmetric) |
| `EfficientGraphConvFn(Function)` | Concatenates `[X, A₁X, …, A₆X]` then one dense multiply; recomputes in backward to save memory |
| `SegmentSumFn(GraphConvFn)` | Segment sum via a CSR matrix — used for pooling atoms → molecules |
| `GraphConv(nn.Block)` | Holds `W` of shape `(F_in·(D+1), F_out)` where `D = N_B + D = 6`, i.e. `(F_in·7, F_out)` |
| `Linear_BN(nn.Sequential)` | `Dense(no bias)` → custom `BatchNorm` |
| `BatchNorm(nn.Block)` | Custom; params `bn_weight`, `bn_bias`, `running_mean`, `running_var`; switches on `autograd.is_training()` |
| `Policy(nn.Block)` | The action head |
| `MoleculeGenerator` → `MoleculeGenerator_RNN` → `VanillaMolGen_RNN` | Unconditional model |
| `_TwoLayerDense`, `CMoleculeGenerator_RNN`, `CVanillaMolGen_RNN` | Conditional model (in `Generating_samples.ipynb` only) |

## B.4 Forward pass, step by step

Let `n` = total atoms in the flattened batch, `m` = number of intermediate graphs
(molecule-states) in the batch.

### Step 1 — atom embedding
```python
X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
```
| Tensor | Shape | Meaning |
|---|---|---|
| `X` (in) | `[n]` int32 | atom-type index per atom |
| `last_append_mask` | `[n]` int32 ∈ {0,1,2} | 0 = ordinary, 1 = last atom added by `append`, 2 = last touched by `connect` |
| `X` (out) | `[n, 16]` | atom embedding + positional-role embedding |

`embedding_mask` is `nn.Embedding(3, F_e)` — this is how the network knows *where it just
was*, which is essential because `connect` is defined relative to the last appended node.

### Step 2 — graph convolution (`VanillaMolGen_RNN._graph_conv_forward`)
```python
X_out = [X]
for conv, bn in zip(self.conv, self.bn):
    X = X_out[-1]
    if bn is not None:
        X_out.append(conv(self.activation(bn(X)), A))
    else:
        X_out.append(conv(X, A))
X_out = nd.concat(*X_out[1:], dim=1)
return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))
```

Per layer `i`, `EfficientGraphConvFn` computes
`concat([X, A₁X, A₂X, A₃X, A₄X, D₂X, D₃X]) @ W_i` — the `[n, F_in·7] × [F_in·7, F_out]`
product that realises CDGCN Eq. 1 with `W_i`, the four `θ_b` and the two `θ_d` folded into a
single weight matrix.

⚠ Note `bn is None` for `i == 0`: **no batch norm before the first convolution**.

| Tensor | Shape |
|---|---|
| after layer 0…5 | `[n,32] [n,64] [n,128] [n,128] [n,256] [n,256]` |
| `X_out` concat | `[n, 864]` (= Σ F_h) |
| after `linear_skip` | `[n, 256]` |

### Step 3 — dense stack
```python
X = self.dense(X)          # Linear_BN(256 → 512)
```
→ `[n, 512]`. This is `FNN_{l+1}` in the papers.

### Step 4 — pooling and GRU (`_rnn_train`)
```python
X_avg  = SegmentSumFn(NX_rep, NX.shape[0])(X) / nd.cast(unsqueeze(NX, 1), 'float32')
X_curr = nd.take(X, indices=NX_cum - 1)
X = nd.concat(X_avg, X_curr, dim=1)              # [m, 1024]
X = nd.take(X, indices=graph_to_rnn)             # [batch, iw, length, 1024]
X = X.reshape([batch*iw, length, 1024])
X = self.rnn(X)                                  # GRU → [batch*iw, length, 512]
X = X.reshape([batch, iw, length, -1])
X = nd.gather_nd(X, indices=rnn_to_graph)        # [m, 512]
```

| Tensor | Shape | Meaning |
|---|---|---|
| `NX` | `[m]` | atom count of each intermediate graph |
| `NX_rep` | `[n]` | molecule index of each atom |
| `NX_cum` | `[m]` | cumulative atom counts; `NX_cum-1` indexes each graph's **last** atom |
| `X_avg` | `[m, 512]` | mean-pooled graph embedding (`h_G`) |
| `X_curr` | `[m, 512]` | embedding of the last appended atom (`h_{v*}`) |
| GRU input | `[·, ·, 1024]` | `h_{v*} ‖ h_G` — matches CDGCN Eq. 2 exactly |
| `X_mol` | `[m, 512]` | per-step recurrent state |

`graph_to_rnn` / `rnn_to_graph` are index maps built in `MolRNNLoader._collate_fn` that
scatter the flat list of intermediate graphs into `[batch, iw, time]` for the GRU and gather
the result back. `MoleculeSpec.max_iter = 120` caps the time axis.

At generation time `_rnn_test` does the same for a single step and threads the hidden state
`h` (shape `[N_rnn, num_samples, 512]`) through the loop manually.

### Step 5 — the policy head (`Policy.forward`)
```python
X_end = X_mol                                       # [m, 512]
X = nd.concat(X, X_end[NX_rep, :], dim=1)           # [n, 1024]
X_h     = nd.relu(self.linear_h(X)).reshape([-1, 128])       # [n, 128]
X_h_end = nd.relu(self.linear_h_t(X_end)).reshape([-1, 128]) # [m, 128]
X_x     = nd.exp(self.linear_x(X_h)).reshape([-1, 1, 264])   # [n, 1, 264]
X_x_end = nd.exp(self.linear_x_t(X_h_end)).reshape([-1,1,1]) # [m, 1, 1]
X_sum   = nd.sum(SegmentSumFn(NX_rep, NX.shape[0])(X_x), -1, keepdims=True) + X_x_end
X_softmax     = X_x / X_sum[NX_rep, :, :]
X_softmax_end = X_x_end / X_sum
connect, append = X_softmax[:, :4], X_softmax[:, 4:]
append = append.reshape([-1, 65, 4])
end = squeeze(X_softmax_end, -1)
```

Key point: `264 = N_B + N_B·N_A = 4 + 260`. The **exponential activation plus explicit
division by a per-molecule sum** implements a softmax **jointly over every possible action of
every atom, plus `end`** — not a per-atom softmax. This matches the papers: `FNN_{l+2}` and
`FNN_{l+3}` use exponential activation and share one softmax.

| Output | Shape | Meaning |
|---|---|---|
| `append` | `[n, 65, 4]` | P(attach new atom of type a with bond b at this atom) |
| `connect` | `[n, 4]` | P(bond of type b from last appended atom to this atom) |
| `end` | `[m]` | P(stop) |

Per molecule the three sum to 1.

### Step 6 — first-atom distribution
- **Unconditional:** `_policy_0(ctx)` = `softmax(exp(policy_0))` over a learnable `[65]`
  parameter, tiled to `[batch·iw, 65]`.
- **Conditional:** `_policy_0(c)` = `_TwoLayerDense(N_C, 195, 65)(c)`, then
  `nd.tile(unsqueeze(·, 1), [1, iw, 1])`.

## B.5 Loss computation (`MoleculeGenerator._likelihood`)

```python
action_type, node_type, edge_type, append_pos, connect_pos = actions[:,0..4]
_log_mask = lambda x, mask: mask*nd.log(x + 1e-10) + (1-mask)*nd.zeros_like(x)

loss_init    = nd.log(nd.gather_nd(init, stack(arange, action_0)) + 1e-10)
loss_end     = _log_mask(end,     cast(action_type == 2, 'float32'))
loss_append  = _log_mask(gather_nd(append,  stack(append_pos, node_type, edge_type)),
                         cast(action_type == 0, 'float32'))
loss_connect = _log_mask(gather_nd(connect, stack(connect_pos, edge_type)),
                         cast(action_type == 1, 'float32'))

log_p_x = loss_end + loss_append + loss_connect
log_p_x = squeeze(SegmentSumFn(iw_ids, batch_size*iw_size)(unsqueeze(log_p_x, -1)), -1)
log_p_x = log_p_x + loss_init
log_p_x     = log_p_x.reshape([batch_size, iw_size])
log_p_sigma = log_p_sigma.reshape([batch_size, iw_size])
l = log_p_x - log_p_sigma
l = logsumexp(l, axis=1) - math.log(float(iw_size))
return l
```

Mapping to the papers:
- `log_p_x` = `log P_θ(G, T | P)` (CTDDG Eq. 1) — masked so each step contributes only its
  actual action type, summed over the path by `SegmentSumFn(iw_ids, …)`.
- `log_p_sigma` = `log Q_α(T | G, P)`, accumulated during traversal in `traverse_graph`.
- `logsumexp(log_p_x − log_p_sigma, axis=1) − log(K)` = the importance-sampling bound,
  CTDDG Eq. 3 / CDGCN Eq. 5.
- `forward` returns `-l.mean()` = CTDDG Eq. 4 / CDGCN Eq. 6.

### ⚠ `tox_class_batch` is accepted and ignored

`_likelihood`'s signature ends `…, batch_size, iw_size, tox_class_batch)`. The identifier
appears **nowhere else in the method body**. `forward()` returns `-l.mean()` unmodified.

CTDDG Eq. 8 would require multiplying the per-molecule `l` by
`(1 − (1+λ)·ToxicClass_i)` before the mean. That multiplication does not exist. **The
implemented objective is CDGCN's, not CTDDG's.** See
[06](06_Pretraining_and_Checkpoints.md) §7 and
[12](12_BTP_Modification_Guide.md) §1.

## B.6 `Q_α` — the proposal distribution over generation paths

```python
def traverse_graph(graph, atom_ranks, current_node=None, step_ids=None, p=0.9, log_p=0.0):
    ...
    next_nodes = [n for n, r in sorted(zip(next_nodes, next_node_ranks), key=lambda _x:_x[1])]
    while len(next_nodes) > 0:
        if len(next_nodes) == 1:
            next_node = next_nodes[0]                                     # forced, no log-prob
        elif random.random() >= (1 - p):
            next_node = next_nodes[0];  log_p += np.log(p)                # canonical rank
        else:
            next_node = next_nodes[randint(1, len(next_nodes)-1)]
            log_p += np.log((1.0 - p) / (len(next_nodes) - 1))            # uniform among rest
        step_ids[next_node] = max(step_ids) + 1
        _, log_p = traverse_graph(graph, atom_ranks, next_node, step_ids, p, log_p)
        next_nodes = [n for n in next_nodes if step_ids[n] < 0]
```

A recursive DFS. `p` (the papers' `α`) is the probability of following canonical atom rank;
otherwise uniform over the remaining unvisited neighbours. `log_p` accumulates
`log Q_α(T | G)`. Called with `p = 0.8` and `k = 5` paths per molecule.

⚠ The function's own default is `p=0.9`, and `MolLoader.__init__`'s default is also `p=0.9`;
the module-level `p = 0.8` is what is actually passed at construction. `finetune_cell.py` uses
the 0.9 default — a silent divergence.

## B.7 Conditional model — the diff

`code/Generating_samples.ipynb` cell 14. Relative to `VanillaMolGen_RNN`:

| Added | Purpose |
|---|---|
| `self.N_C` | condition width |
| `dense_policy_0 = _TwoLayerDense(N_C, N_A*3, N_A)` | protein → first-atom distribution |
| `linear_c[i] = nn.Dense(F_h[i], use_bias=False, in_units=N_C)`, one per GCN layer (**6**) | protein → per-layer additive bias |
| `_graph_conv_forward(X, A, c, ids)` gains `+ linear_c(c)[ids, :]` | injection point |
| `_policy_0(c) = dense_policy_0(c) + 0.0 * policy_0.data(...)` | keeps the unconditional parameter registered so old checkpoints load |
| `forward` signature gains `c, ids` | |
| `init = nd.tile(unsqueeze(self._policy_0(c), axis=1), [1, iw_size, 1])` | per-molecule init distribution |

Everything else — `Embed_V`, GCN stack, dense stack, GRU, `Policy`, `_likelihood` — is
inherited unchanged. ⚠ Note `CMoleculeGenerator_RNN.forward` calls
`self._likelihood(..., batch_size, iw_size)` **without** `tox_class`, so the conditional model
as written in `Generating_samples.ipynb` is CDGCN-shaped too.

## B.8 Decoding (generation)

`_decode_step` in `code/Generating_samples.ipynb` cell 16 operates on **NumPy**, not MXNet,
and maintains batched state across `num_samples` molecules simultaneously:

| Array | Shape | Meaning |
|---|---|---|
| `X` | `[Σ NX]` | concatenated atom types |
| `A` | `[Σ NA, 3]` | concatenated bonds |
| `NX`, `NA` | `[num_samples]` | per-molecule atom / bond counts |
| `last_action` | `[num_samples]` | 1 if last op was `append`, else 0 |
| `finished` | `[num_samples]` bool | termination flags |

Only unfinished molecules are passed to the network each step (`X_u = X[np.repeat(~finished,
NX)]`). Details in [08](08_Generation.md).

## B.9 Parameter budget — verified

| Component | Arithmetic | Parameters |
|---|---|---|
| `embedding_atom` | 65 × 16 | 1,040 |
| `embedding_mask` | 3 × 16 | 48 |
| `GraphConv` W ×6 | 16·7·32 + 32·7·64 + 64·7·128 + 128·7·128 + 128·7·256 + 256·7·256 | 878,080 |
| `BatchNorm` layers 1–5 | 4 × (32+64+128+128+256) | 2,432 |
| `bn_skip` | 4 × 864 | 3,456 |
| `linear_skip` | 864·256 + 4·256 | 222,208 |
| `dense` | 256·512 + 4·512 | 133,120 |
| `policy_0` | 65 | 65 |
| `Policy.linear_h` | 1024·128 + 512 | 131,584 |
| `Policy.linear_h_t` | 512·128 + 512 | 66,048 |
| `Policy.linear_x` | 128·264 + 264 | 34,056 |
| `Policy.linear_x_t` | 128 + 1 | 129 |
| `GRU` (3 layers, hidden 512, input 1024) | 2,362,368 + 1,575,936 + 1,575,936 | 5,514,240 |
| **Total** | | **6,986,506** |

Matches the logged value exactly — a strong end-to-end check on this reconstruction. Note the
GRU is **79%** of the model.

Adding conditioning at `N_C = 1024` would add ≈1,097,936 more (see [04](04_Protein_Embeddings.md) §7).

## B.10 What does *not* exist in this architecture

To prevent over-claiming:

- ❌ No attention mechanism of any kind (no transformer, no graph attention).
- ❌ No property conditioning (logP, QED, MW, SAS are computed only at evaluation time).
- ❌ No toxicity input to the network — `tox_class` was designed to weight the loss.
- ❌ No 3D coordinates, no pocket geometry, no docking-in-the-loop.
- ❌ No beam search — decoding is stochastic sampling or greedy argmax.
- ❌ No multi-target conditioning — one protein vector per molecule.
- ❌ No variational latent, no adversarial component, no reinforcement learning.
- ❌ No explicit valence checking during decoding; validity is enforced *post hoc* by
  `Chem.SanitizeMol`, and invalid graphs are dropped.
