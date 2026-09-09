# 12 — BTP Modification Guide

Every modification below is grounded in something concrete: a paper equation that is
unimplemented, a defect verified in the code, or a stated future direction. **No speculative
"you could try a transformer" suggestions.**

Difficulty is calibrated for one student with cluster access. Risk means *risk to existing
working artefacts*, principally the 47-hour pretrained checkpoint.

**Read [11_Current_State_and_Next_Steps.md](11_Current_State_and_Next_Steps.md) first.** Most
of these need a working end-to-end pipeline. The exception is §1, which is the most
scientifically interesting and can be prototyped immediately.

---

## Priority ordering

| # | Modification | Category | Difficulty | Risk | Scientific value |
|---|---|---|---|---|---|
| 1 | **Implement the toxicity-weighted loss (paper Eq. 8)** | training objective | Medium | Low | ★★★★★ |
| 2 | Fix the protein-embedding pipeline | preprocessing | **Low** | Low | ★★★★★ (unblocks everything) |
| 3 | Write a correct conditional fine-tuning stage | fine-tuning | Medium-High | Low | ★★★★★ (unblocks everything) |
| 4 | Swap the protein encoder | conditioning | Medium | Medium | ★★★★ |
| 5 | Add property conditioning | conditioning | Medium-High | Medium | ★★★★ |
| 6 | Multi-target conditioning | conditioning | High | High | ★★★★★ (stated BTP direction) |
| 7 | CrossDocked2020 dataset | data | High | Medium | ★★★★ (stated BTP direction) |
| 8 | Unconditional sampler for checkpoint validation | tooling | **Low** | **None** | ★★★ (enabling) |
| 9 | Implement the novelty metric | evaluation | **Low** | **None** | ★★★ |
| 10 | Docking score extraction + batch loop | docking | Medium | **None** | ★★★ |
| 11 | λ sweep for the toxicity trade-off | experiment | Low (after §1) | Low | ★★★★ |

---

## 1. Implement the toxicity-weighted loss — CTDDG's actual contribution

**File:** `code/pretraining.ipynb` (single cell)
**Class/function:** `MoleculeGenerator._likelihood`, and `MoleculeGenerator.forward` /
`MoleculeGenerator_RNN.forward`
**Also:** `process_single`, and the ChEMBL labelling

### Current behaviour
```python
def _likelihood(self, init, append, connect, end, action_0, actions, iw_ids,
                log_p_sigma, batch_size, iw_size, tox_class_batch):
    ...
    l = logsumexp(log_p_x - log_p_sigma, axis=1) - math.log(float(iw_size))
    return l          # tox_class_batch NEVER used

def forward(self, *input):
    ...
    return -l.mean()  # = CTDDG Eq. 4 / CDGCN Eq. 6
```
and upstream, `process_single` parses `smiles_class` then discards it:
```python
smiles, smiles_class = smiles.split(" ")
smiles_class = int(smiles_class)
...
tox_class = [1] * k         # hardcoded
```
and `data/chembl/chembl_final.txt` is **100% class 1** (verified).

### Possible modification
Three coordinated changes:

**(a) Make labels real.** Train the paper's Random Forest on Ames-mutagenicity data
(`papers/CTDDG.pdf` §2.4 cites Wu et al. 2021 — the *MoleculeNet / TDC Ames* dataset) with
ECFP-style fingerprints, apply it to `chembl.txt`, and write genuine 0/1 labels. `sklearn`
0.24.2 and RDKit are already installed.

**(b) Propagate the label.** In `process_single`:
```python
tox_class = [smiles_class] * k       # instead of [1] * k
```

**(c) Apply Eq. 8.** In `forward`, before the mean:
```python
# l : [batch_size]      tox : [batch_size]  (one label per molecule)
weight = 1.0 - (1.0 + lam) * tox          # lam = 1.0 per the paper
return -(l * weight).mean()
```
Per `papers/CTDDG.pdf` §2.3.3 this yields weight `+1` for non-toxic (maximise likelihood) and
`−λ` for toxic (minimise likelihood).

⚠ `tox_class` currently arrives with **`k` entries per molecule** (`process_single` returns
`np.array([...]*k)`), so it has shape `[batch_size * k]`, not `[batch_size]`. `l` has shape
`[batch_size]` after `logsumexp(..., axis=1)`. You must reduce `tox_class` to one value per
molecule — e.g. `tox.reshape([batch_size, iw_size])[:, 0]` — since all `k` paths of a molecule
share its label. **Verify this alignment on a tiny batch before any long run.**

### Why this location
It is the **entire scientific delta between CDGCN and CTDDG**. The paper specifies the
equation exactly; the code already threads `tox_class` from `process_single` → `MolLoader` →
`from_numpy_to_tensor` → `forward` → `_likelihood` and simply never uses it. The plumbing is
done; the last multiplication is missing.

### Downstream effects
- Pretraining must be re-run from scratch (~47 h). **Back up the existing checkpoint first.**
- Fine-tuning would want the same treatment, with BindingDB ligands labelled by pkCSM
  (`papers/CTDDG.pdf` §2.4).
- Evaluation gains a real toxicity axis, comparable to the paper's Table 3.
- The `end`/`append`/`connect` head, decoding and generation are all unaffected.

⚠ **A minimised likelihood is unbounded below.** Pushing `P(toxic graph)` toward 0 makes
`log P → −∞`, so the weighted loss can diverge. The paper does not discuss this. Watch for
loss divergence; a clipped or hinge form of the toxic term is a defensible, reportable
contribution if plain Eq. 8 is unstable.

**Difficulty:** Medium (the RF classifier is the bulk of the work).
**Risk:** Low, if the existing checkpoint is backed up.

---

## 2. Fix the protein-embedding pipeline

**File:** `code/data_preprocessing_1.ipynb`
**Cells:** 75 (`give_bioembeddings`), 78 and its test-set counterpart

### Current behaviour
```python
def give_bioembeddings(protein_list):
    embedder = ProtTransBertBFDEmbedder()
    embedding_list = []
    for protein in protein_list:
        embedding_list.append(embedder.embed(protein))     # → (L, 1024), per-residue
    return embedding_list
...
for embed in d1_tr_embeds:
    for e in embed:                 # e is a 1024-vector
        f.write(str(e)+'\t')        # writes '[0.1 0.2 ... 0.9]'
```
Confirmed: `embed()` returns per-residue; `reduce_per_protein` (a mean over axis 0) is the
pooling step and is never called. The upstream CDGCN notebook used ProSE with `--pool avg`,
and its stored output shows a single 6165-d vector per protein.

### Possible modification
```python
embedding_list.append(embedder.reduce_per_protein(embedder.embed(protein)))   # → (1024,)
...
f.write('\t'.join(str(float(x)) for x in embed) + '\n')
```

### Why this location
It is the sole root cause of the fine-tuning blockage, and it is a one-line fix.
[04_Protein_Embeddings.md](04_Protein_Embeddings.md) §4 has the full evidence.

### Downstream effects
Sets `N_C = 1024`, which then propagates into `configs.json`, `dense_policy_0` and all six
`linear_c` layers. **Everything downstream depends on this.**

**Difficulty:** Low. **Risk:** Low — but ⚠ run *only* the embedding cells; re-running the whole
notebook would recompute 5.8 M needle alignments and rewrite `atom_types.txt` (see §"Do not
change" below).

---

## 3. Write a correct conditional fine-tuning stage

**Files:** new script; reuse classes from `code/Generating_samples.ipynb` cell 14
**Do NOT extend:** `scripts/finetune_cell.py`

### Current behaviour
No conditional loader or training loop exists. The reconstruction has five independent
runtime blockers and diverges architecturally in five more ways
([07](07_Finetuning_and_Conditioning.md) §5).

### Possible modification
Full scope in [07](07_Finetuning_and_Conditioning.md) §7. The two things most likely to be got
wrong:
- The loader must emit the per-atom `ids` tensor (`NX_rep`) alongside `c`, because
  `linear_c(c)[ids, :]` is what makes conditioning correct for variable-size molecules.
- `configs.json` **must** contain `N_C`, since `CVanilla_RNN_Builder._get_model` splats it
  into the constructor.

### Why this location
`CVanillaMolGen_RNN` in `Generating_samples.ipynb` is the authors' own architecture and is
already compatible with the generation code. Reusing it guarantees checkpoint compatibility.

**Difficulty:** Medium-High. **Risk:** Low (writes to a new directory).

---

## 4. Swap the protein encoder

**Files:** `code/data_preprocessing_1.ipynb` cell 75; `N_C` in the fine-tuned `configs.json`

### Current behaviour
`ProtTransBertBFDEmbedder`, 1024-d. Upstream CDGCN used ProSE/Bepler, 6165-d. Frozen; computed
offline; never part of the MXNet graph.

### Possible modification
`bio_embeddings` offers alternatives already importable in `ctddg_env`, e.g.
`ProtTransT5XLU50Embedder`, `ESMEmbedder`, `ProtTransXLNetUniRef100Embedder`. All expose
`reduce_per_protein`. Change one class name and regenerate the embedding files.

A defensible experiment: hold everything else fixed and compare target awareness across two or
three encoders — an ablation neither paper performs.

### Why this location
The papers under-specify this (they cite only "Dallago et al."), and CDGCN and CTDDG demonstrably
used **different** encoders. That makes it a genuine open question, not a manufactured one.

### Downstream effects
`N_C` changes → all conditioning layers resize → fine-tuning must be redone.
**Pretraining is unaffected** (it has no protein path), so the 47-hour checkpoint is safe.

⚠ Larger encoders (T5-XL) need substantial GPU memory and re-download weights; only ProtBert
is cached locally.

**Difficulty:** Medium. **Risk:** Medium (invalidates fine-tuned checkpoints, not the pretrained one).

---

## 5. Add property conditioning

**File:** `code/Generating_samples.ipynb` cell 14 — `CVanillaMolGen_RNN.__init__`, `_policy_0`,
`_graph_conv_forward`

### Current behaviour
`c` is the protein embedding alone. No property vector enters the network anywhere.

### Possible modification
Concatenate desired properties onto `c`:
```python
c_full = np.concatenate([protein_embedding, [logP_target, qed_target, sas_target]])
# N_C becomes 1024 + n_properties
```
No architectural change is needed — `dense_policy_0` and `linear_c[i]` already take `in_units=N_C`.
The work is in preprocessing (compute per-ligand properties with RDKit during dataset assembly)
and in choosing target values at generation time.

### Why this location
`papers/CTDDG.pdf` §1 explicitly names this gap when discussing Kang & Cho's SSVAE, which
"generates molecules with desired properties by conditioning the distribution on these
properties" but "does not explicitly incorporate target protein conditions". Combining both is
the natural synthesis, and the paper's own conclusion points there: "the graph encoding scheme
of CTDDG will be improved upon by exploring more efficient methods … such as maximizing binding
affinity, optimizing drug like properties etc." (§5).

### Downstream effects
Fine-tuning data format changes; `N_C` changes; generation needs a property-vector choice per
sample. Evaluation becomes much more informative (you can test whether requesting QED 0.8
actually raises QED).

**Difficulty:** Medium-High. **Risk:** Medium.

---

## 6. Multi-target conditioning — the stated BTP direction

**Files:** `CVanillaMolGen_RNN` (conditioning path); preprocessing (dataset construction)

### Current behaviour
Strictly one protein → one vector → one molecule. `generate_samples` loops over proteins
independently; nothing in the model represents more than one target.

### Possible modification
Options, roughly in increasing ambition:

- **(a) Pooled multi-target.** `c = mean(c_1, …, c_n)` or a learned attention pool over
  several target embeddings. Minimal architectural change (`N_C` unchanged), and directly
  testable: does a molecule conditioned on {CDK2, CDK4} dock well against both?
- **(b) Concatenated targets.** `c = [c_primary ‖ c_antitarget]` with `N_C = 2 × 1024`, so the
  model can learn *selectivity* — bind A, avoid B.
- **(c) Signed multi-target loss.** Reuse the Eq. 8 weighting idea: maximise likelihood for
  on-target ligands, minimise it for known anti-target ligands. This composes neatly with §1
  and is arguably the most novel.

### Why this location
`WEEK 1 PROGRESS - TASK 2.md` §17 records that CrossDocked2020 was acquired specifically for
"the later multi-target extension", and §5 of `papers/CTDDG.pdf` leaves the conditioning scheme
open as future work. Option (c) also generalises the paper's own loss construction rather than
bolting on an unrelated mechanism.

### Downstream effects
Needs a multi-target paired dataset (CrossDocked2020, §7). Docking evaluation must run against
multiple receptors. `N_C` and the fine-tuning data format both change.

**Difficulty:** High. **Risk:** High — this is genuine research, not reproduction. Only start
once the single-target baseline reproduces.

---

## 7. Move to CrossDocked2020

**Files:** preprocessing (new); `~/datasets/downsampled_CrossDocked2020_v1.3/` (2,801 pocket dirs, extracted)

### Current behaviour
BindingDB (Grechishnikova), sequence-only, no 3D structures.

### Possible modification
CrossDocked2020 provides many ligands per pocket with 3D poses, which enables multi-target
grouping (§6) and much broader docking evaluation. Extracting `(pocket → [ligand SMILES])`
pairs and running the protein sequences through the embedder reuses the whole existing pipeline.

### Why this location
Already downloaded and extracted; explicitly earmarked for the multi-target extension.

### Downstream effects
New preprocessing; `atom_types.txt` may need extending if CrossDocked introduces atom types
outside the 65 — ⚠ **which would invalidate the pretrained checkpoint's atom embeddings**.
Check coverage *before* committing:
```python
# pseudocode: for each CrossDocked ligand, assert every (symbol, charge, nHs) is in atom_types.txt
```
If new types appear, the safest route is to **drop** those ligands rather than extend the file.

**Difficulty:** High. **Risk:** Medium.

---

## 8. Unconditional sampler (enabling tool)

**File:** new, ~30 lines, modelled on `CVanilla_RNN_Builder` in `Generating_samples.ipynb` cell 16

### Current behaviour
The only sampler is conditional. There is no way to draw molecules from the pretrained
checkpoint, so it has never been validated.

### Possible modification
```python
class Vanilla_RNN_Builder(Builder):
    @staticmethod
    def _get_model(configs):
        return VanillaMolGen_RNN(get_mol_spec().num_atom_types,
                                 get_mol_spec().num_bond_types, D=2, **configs)
    # sample(): get_init() → mdl.mode='decode_0'; mdl(self.ctx)  (no c)
    #           get_action() → mdl.mode='decode_step'; mdl(_X,_A,_NX,_NX_rep,_mask,_NX_cum,_h)
```
Reuse `_decode_step` unchanged. Note `VanillaMolGen_RNN.forward`'s `decode_0` branch takes
`input[0]` as a **context**, and its `decode_step` branch takes 7 args (no `c`, no `ids`).

### Why this location
Highest value per line of code in the whole repository. 46 hours of GPU time is currently
unverified.

**Difficulty:** Low. **Risk:** None — read-only with respect to the checkpoint.

---

## 9. Implement the novelty metric

**File:** `code/Evaluation_metrics.ipynb`, alongside the aggregation in cell 13

### Current behaviour
Not implemented, despite both papers reporting it (CDGCN Table 1: 99.7%) and
`notebooks/04_evaluation.ipynb` claiming it is computed.

### Possible modification
```python
training_smiles = set()   # canonical SMILES from chembl.txt + all BindingDB ligand files
novel_pct = 100 * sum(s not in training_smiles for s in df['smiles'].unique()) / len(df['smiles'].unique())
```
CDGCN §3.4 specifies "exact string matching algorithm on the molecules in the dataset", so
canonicalise both sides with `Chem.MolToSmiles` first.

### Why this location
It is a headline metric in both papers and a one-function addition. Without it, results here
cannot be compared with either paper's Table 1.

**Difficulty:** Low. **Risk:** None.

---

## 10. Docking score extraction and batch loop

**File:** `code/Molecular_docking.ipynb`

### Current behaviour
Writes `*_smina_out.pdbqt` files and stops. **No score parsing, no ranking, no threshold
statistic.** Also hardcodes `beam=0, run=3, prot_id=6, pdb_id='1cqp'`, and `os.chdir` +
relative paths make it interactive-only.

### Possible modification
Parse the score from each output (smina writes `REMARK minimizedAffinity` per MODEL), take the
best pose per ligand, then compute the paper's target-awareness metric (% below −7.0 kcal/mol,
`papers/CTDDG.pdf` Table 2) and the top-10 ranking used in the case study (Table 4). Wrap the
whole thing in a loop over the five evaluation targets **4nos, 2nru, 1lqf, 1k2r, 1cqp**.

### Why this location
Table 2 is the paper's principal target-awareness result and currently cannot be reproduced at
all.

**Difficulty:** Medium (mostly toolchain installation).
**Risk:** None to existing artefacts.

---

## 11. λ sweep for the toxicity trade-off

**File:** wherever §1 lands
**Depends on:** §1

### Current behaviour
λ is fixed to 1 with a stated but untested rationale:

> "Higher lambda indicates excessive suppression of toxic molecules which might limit the
> model's exploration of certain important molecular scaffolds. Hence for our experiment, we
> have fixed λ to 1." — `papers/CTDDG.pdf` §2.3.3

### Possible modification
Train with λ ∈ {0, 0.5, 1, 2, 5} and plot toxicity % against validity, uniqueness, QED and
target awareness. λ = 0 recovers CDGCN exactly, giving a clean controlled comparison on
*identical* data and code — something neither paper has.

### Why this location
The paper asserts a trade-off and then does not measure it. A Pareto curve is a small,
well-defined, genuinely novel contribution.

⚠ Cost: one pretraining run (~47 h) per λ. Consider a reduced iteration budget (e.g. 100 k)
held constant across λ, and say so when reporting.

**Difficulty:** Low once §1 works. **Risk:** Low (compute only).

---

## What should probably NOT be changed

| Component | Why not |
|---|---|
| **`data/atom_types.txt`** | Atom-type integers are line numbers; the checkpoint's `embedding_atom` (65×16) and `policy_0` (65) are tied to this ordering. The generator builds it from a `set()`, so regeneration is not order-reproducible. **Treat as immutable.** |
| **`MoleculeSpec.bond_orders`** | Hardcoded `[AROMATIC, SINGLE, DOUBLE, TRIPLE]`; indices are baked into `A_list` construction, `Policy`'s `N_B + N_B*N_A` output width, and the checkpoint |
| **`GraphConv` / `EfficientGraphConvFn`** | Custom autograd with hand-written backward and recomputation. Correct and memory-tuned. Changing it risks silently wrong gradients |
| **`traverse_graph` / `single_reorder` / `single_expand`** | Dense index arithmetic implementing `Q_α` and the action encoding. Subtle, and a bug here corrupts training without any error |
| **`_likelihood`'s masking logic** | The `_log_mask` + `SegmentSumFn` construction implements the importance-sampling bound. §1's change should be applied in `forward`, *outside* this method |
| **`configs.json` schema** | Consumed by `**configs` splat in three places. Add keys (`N_C`) rather than renaming |
| **The pretrained checkpoint** | 47 GPU-hours. Back up before anything touches `outputs/pretrain/` |
| **`trainer.step(batch_size=1)`** | Deliberate — `forward` already means the loss. "Fixing" it to `batch_size=8` silently divides the learning rate by 8 |
| **`k=5`, `p=0.8`** | Selected by 5-fold CV in `papers/CDGCN.pdf` §3.3; a sensitivity study exists in its supplementary. Change only as a deliberate experiment |

---

## Suggested BTP arc

**Phase 1 — Reproduce (weeks 1–3).** §2 → §8 → §3 → generation → §9. Goal: a table of
validity/uniqueness/QED/SAS comparable to `papers/CTDDG.pdf` Table 1.

**Phase 2 — Complete the paper (weeks 4–7).** §1 + §11. Goal: implement Eq. 8, produce the
λ trade-off curve, and reproduce Table 3's toxicity comparison. This alone is a defensible BTP:
it turns a published-but-unimplemented method into a working, measured one.

**Phase 3 — Extend (weeks 8+).** Pick **one** of §4 (encoder ablation), §5 (property
conditioning) or §6 (multi-target). §6 is the stated direction and has the CrossDocked data
ready, but only start it once Phase 1 reproduces — otherwise you cannot tell a research
failure from a plumbing failure.
