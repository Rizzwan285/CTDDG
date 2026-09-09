# 01 — Project and Research Context

---
---

# LEVEL 1 — BEGINNER / INTUITIVE

Assumes basic ML knowledge, no computational chemistry background.

## 1.1 What is drug molecule generation?

A **drug** is usually a small molecule that sticks to a **protein** in the body and changes
what that protein does. Most diseases have a protein behind them; a drug that jams that
protein can treat the disease.

Historically, finding such a molecule meant physically screening libraries of chemicals.
**De novo drug design** instead *designs* a molecule from scratch. **Drug molecule
generation** is the machine-learning version: train a model on millions of known molecules
until it has learnt what "a chemically valid, drug-like molecule" looks like, then ask it to
produce new ones.

The space is absurdly large — the CDGCN paper puts drug-like chemical space at ~10^60
molecules, of which perhaps 10^8 are actually synthesizable (**PAPER**, `papers/CDGCN.pdf` §1).
The point of a generative model is not to explore that space but to *skip* it: propose a few
hundred plausible candidates instead of screening billions.

## 1.2 What is *conditional* drug molecule generation?

An **unconditional** generator produces "some valid drug-like molecule" — it has no idea what
disease you care about. Useful for learning chemistry, useless on its own.

A **conditional** generator takes an extra input — here, a description of your target protein
— and produces molecules *aimed at that protein*. Formally you move from learning `P(molecule)`
to learning `P(molecule | protein)`.

This distinction is the single most important thing to hold on to in this repository,
because the two models are separate classes with separate checkpoints, and only the
unconditional one has actually been trained. See [06](06_Pretraining_and_Checkpoints.md)
and [07](07_Finetuning_and_Conditioning.md).

## 1.3 Why do proteins matter?

The protein *is* the target. A molecule that is beautifully drug-like but doesn't fit your
protein's binding pocket is worthless.

Both papers distinguish two cases, and the terminology is used throughout:

- **known protein** — a protein that already has known drugs against it.
- **novel protein** — a protein with no known drug.

> "we mention proteins known to interact with some known drugs as known proteins, and
> proteins not known to interact with any known drug as novel proteins"
> — **PAPER**, `papers/CDGCN.pdf` §1

The headline claim of CDGCN (and inherited by CTDDG) is that it works for **novel** proteins.
That is only possible because the protein is fed in as a *learned representation of its amino
acid sequence*, not as an identity label. A model conditioned on a one-hot "protein #37"
label can never generalise to protein #1043; a model conditioned on a sequence embedding can.
This is exactly the criticism both papers level at Li et al. (**PAPER**, `papers/CDGCN.pdf` §1).

## 1.4 What are SMILES?

**SMILES** (Simplified Molecular-Input Line-Entry System) writes a molecule as a text string.
Aspirin is `CC(=O)Oc1ccccc1C(=O)O`. Letters are atoms, digits open and close rings,
parentheses are branches, lower case means aromatic.

SMILES are how molecules are *stored* in this project — `data/chembl/chembl.txt` is a million
lines of them. But they are **not** how the model works internally (see next).

## 1.5 What molecular representation does this model use?

A **molecular graph**: atoms are nodes, bonds are edges.

- Each node carries an **atom type**, drawn from a fixed set of 65 (`data/atom_types.txt`).
  An "atom type" here is the triple *(element symbol, formal charge, number of explicit
  hydrogens)* — so `N,0,0` and `N,1,0` are different types.
- Each edge carries a **bond type**, from a fixed set of 4: aromatic, single, double, triple.

Why graphs rather than SMILES strings? Because a SMILES-based model must first learn SMILES
*grammar* before it can learn chemistry, and it frequently emits strings that don't parse.
Graph models constrain the output to be a graph by construction and get much higher validity
— roughly 94–95% vs 64–89% in the CDGCN comparison (**PAPER**, `papers/CDGCN.pdf` Table 1).

The graph is built **one action at a time** — see [05](05_Model_Architecture.md).

## 1.6 What is pretraining?

Training on a very large, cheap, *unlabelled-for-your-task* dataset to learn general
structure. Here: ~1.09 million ChEMBL molecules, with no protein involved at all, teaching
the model "what molecules look like". This is the stage that **has been completed**.

## 1.7 What is fine-tuning?

Taking those pretrained weights and continuing training on a smaller, more specific dataset —
here, BindingDB protein–ligand *pairs* — so the model learns "what molecules look like *for
this protein*". Both papers are explicit that this two-phase split exists because the paired
data is too small to train the architecture from scratch:

> "This data is not enough to train the large architecture of CTDDG. Hence CTDDG get trained
> in two phases" — **PAPER**, `papers/CTDDG.pdf` §2.4

## 1.8 What is conditioning?

The mechanical act of injecting the extra information into the network. In CDGCN/CTDDG the
protein is turned into a fixed-length vector `c`, and that vector is added into the model at
two places: it produces the probability distribution over the *first* atom, and it is
projected and added into *every* graph convolution layer. See [04](04_Protein_Embeddings.md).

## 1.9 What is generation?

Running the trained model forwards: start from an empty graph, repeatedly sample one of four
actions (add first atom / attach a new atom / add a bond / stop), and stop when the model
says stop. Convert the finished graph to a molecule with RDKit, then to a SMILES string.

## 1.10 What is evaluation?

Automated checks on the generated set. Is it a chemically valid molecule at all? Is it
different from the others? Does it satisfy drug-likeness rules of thumb (Lipinski's rule of
five, TPSA, QED)? Does it look synthesizable (SAS)? See [09](09_Evaluation_and_Docking.md).

## 1.11 What is docking?

**Molecular docking** is a physics-based simulation that takes a 3D protein structure and a
3D molecule and searches for the best way to fit the molecule into a pocket, returning a
**docking score** in kcal/mol (more negative = predicted tighter fit).

Docking is a *computational prediction*, not a measurement. It is a cheap filter, not
evidence of binding. This distinction is laboured deliberately in
[09](09_Evaluation_and_Docking.md) because it is the most commonly overstated result in
this field.

---
---

# LEVEL 2 — TECHNICAL

## 2.1 Research motivation

**PAPER** (`papers/CTDDG.pdf` §1): existing generative models fall into groups, each with a
gap the paper positions against —

- **VAE-based** (CharVAE, JT-VAE, HierVAE) — learn a distribution over drug-like molecules,
  no target awareness.
- **GAN-based** (ORGAN, MolGAN, Mol-CycleGAN) — generate molecules similar to the training
  set with optimised properties, no target awareness.
- **Flow-based** (GraphNVP, MolFlow, MolGrow) — invertible mappings, no target awareness.
- **3D structure-based** (Pocket2Mol, Luo et al.) — target-aware, but *require the binding
  pocket's 3D structure*, so restricted to known proteins.
- **Property-conditional** (Kang & Cho's SSVAE) — conditions on properties including
  toxicity, but has no protein conditioning at all.
- **DrugEx** — RNN + RL with multi-objective optimisation including toxicity, but built
  specifically for adenosine receptors.

The stated gap CTDDG fills: a model that is target-aware **from sequence alone** (so it works
for novel proteins), **and** optimises toxicity, **and** preserves drug-likeness.

## 2.2 Problem formulation

A molecule is a graph `G = (V, E)`; each node has an atom type from fixed set `A`, each edge
a bond type from fixed set `B`. Given a protein `P` (as an amino-acid sequence), generate `G`.

Generation is a **sequence of actions** `T = (t_1 … t_m)` applied to an intermediate graph:

| Action | Effect |
|---|---|
| `init` | add the first node to the empty graph |
| `append` | attach a **new** node with a **new** edge to an existing node |
| `connect` | add a **new edge** between an existing node and the last appended node |
| `end` | terminate generation |

**PAPER**, `papers/CTDDG.pdf` §2.2 / `papers/CDGCN.pdf` §2.1.

The joint log-likelihood of a graph *and* a specific generation path, given the protein:

```
log P_θ(G, T | P) = Σ_{j=1..m} log P_θ(t_{j+1} | G_j, …, G_1, P)          (CTDDG Eq. 1)
```

The marginal `log P_θ(G | P) = log Σ_{T ∈ S(G)} P_θ(G, T | P)` is intractable because a
molecule has combinatorially many valid construction orders. The papers resolve this with
**importance sampling** over a predefined proposal distribution `Q_α(T | G, P)`:

```
log P_θ(G|P) ≥ log (1/K) Σ_{k=1..K} P_θ(G, T_k | P) / Q_α(T_k | G, P)     (CTDDG Eq. 3)
```

with the mini-batch training loss

```
L̂(θ) = −(1/N) Σ_{i=1..N} log (1/K) Σ_{k=1..K} P_θ(G_i, T_i_k | P_i) / Q_α(T_i_k | G_i, P_i)   (CTDDG Eq. 4)
```

`Q_α` is a Bernoulli-controlled traversal: at each step, with probability `α` take the
neighbour with the lowest **canonical atom rank**, otherwise take a uniformly random unvisited
neighbour. `α` is the "coefficient of randomness" (**PAPER**, `papers/CTDDG.pdf` §2.3.2).

## 2.3 Architecture

Identical in both papers. The clearest statement is `papers/CDGCN.pdf` §2.1–§2.2:

```
protein sequence ──► Embed_p ──► protein embedding c
                                      │
      ┌───────────────────────────────┼──────────────────────────┐
      ▼                               ▼                          ▼
   FNN_0                        FNN_1 … FNN_l              (per-layer projections)
      │                               │
      ▼                               ▼
 init-atom distribution      summed into GCN_1 … GCN_l outputs
                                      │
 intermediate graph ─► Embed_V ─► GCN_1 … GCN_l ─► concat ─► FNN_{l+1} ─► node embeddings
                                                                 │
                                                          avg pool ─► graph embedding
                                                                 │
                                                                RNN (GRU)
                                                                 │
                        ┌────────────────────────────────────────┴──────────┐
                        ▼                                                   ▼
     concat(RNN out, node emb) ─► FNN_{l+2}                            FNN_{l+3}
        |V| × (|A||B| + |B|)  [append & connect]                      scalar [end]
                        └──────────────► softmax over all ◄────────────┘
```

Component detail (**PAPER**, `papers/CDGCN.pdf` §2.2):

- `Embed_p` — a **pretrained** protein network from Dallago et al. (the `bio_embeddings`
  toolkit), trained to predict protein function from sequence.
- `FNN_0` — dense layer + softmax → probability per atom type for `init`.
- `Embed_V` — learnable `|A| × d_0` atom-type embedding matrix.
- `GCN_1 … GCN_l` — graph convolution + batch norm + ReLU (last layer: GC only).
- `RNN` — GRU, carrying state across generation steps:
  `h^GRU_m = GRU(h^GRU_{m−1}, h_{v*} ‖ h_G)` where `v*` is the last appended node.
- `FNN_{l+2}` — dense → BN → ReLU → dense → **exponential** activation.
- `FNN_{l+3}` — dense → **exponential** activation.

The graph convolution is the molecule-adapted form (CDGCN Eq. 1):

```
h_i(v) = W_i h_{i−1}(v)  +  Σ_{b∈B} θ^i_b Σ_{u∈N_b(v)} h_{i−1}(u)
                         +  Σ_{1<d≤D} θ^i_d Σ_{u∈N_d(v)} h_{i−1}(u)
```

`N_b(v)` = neighbours joined by a bond of type `b`; `N_d(v)` = nodes at *path length* `d`.
`D` is the receptive field size. So each layer aggregates **per-bond-type** neighbourhoods
*and* **distance-2/3** neighbourhoods separately — this is what makes it a molecule-specific
GCN rather than a generic one.

## 2.4 Training objective — the one thing CTDDG changes

**PAPER** — CDGCN minimises Eq. 4 above (plain NLL). CTDDG replaces it with a
toxicity-weighted variant (`papers/CTDDG.pdf` §2.3.3):

```
L̂(θ) = (1/N) Σ_i [ log (1/K) Σ_k P_θ(G_i,T_ik|P_i) / Q_α(T_ik|G_i,P_i) ] · (1 − (1+λ)·ToxicClass_i)   (Eq. 8)
```

where `ToxicClass_i ∈ {0, 1}` is 1 for a toxic ligand. Read the weight term:

| ToxicClass | weight `(1 − (1+λ)·c)` | with λ=1 | effect |
|---|---|---|---|
| 0 (non-toxic) | 1 | +1 | **maximise** likelihood |
| 1 (toxic) | −λ | −1 | **minimise** likelihood |

> "for an individual data point … only one aspect of the loss function is activated at a
> time, either the toxic or non-toxic component."
> — **PAPER**, `papers/CTDDG.pdf` §2.3.3

λ is fixed to 1 in the paper's experiments; higher λ is said to risk "excessive suppression
of toxic molecules which might limit the model's exploration of certain important molecular
scaffolds."

**CODE** — **not implemented.** `code/pretraining.ipynb`, method
`MoleculeGenerator._likelihood(self, init, append, connect, end, action_0, actions, iw_ids,
log_p_sigma, batch_size, iw_size, tox_class_batch)` receives `tox_class_batch` and never uses
it. `forward()` returns `-l.mean()`, i.e. Eq. 4 exactly. Verified by reading the whole method.

**NEW NOTEBOOKS** — do not change this; `notebooks/01_pretraining.ipynb` merely shells out to
`code/pretraining.ipynb`.

**CURRENT EXECUTED STATE** — the 480,000-iteration run used Eq. 4. The checkpoint at
`outputs/pretrain/logs/ckpt.params` is a CDGCN-style unconditional model.

## 2.5 Conditioning

**PAPER** — the condition is the protein, encoded by `Embed_p` into a task-agnostic embedding,
then routed two ways: through `FNN_0` to seed the first atom, and through `FNN_1…FNN_l` to be
summed into each GCN layer's node embeddings (**PAPER**, `papers/CDGCN.pdf` §2.1).

**CODE** — matches, and is more specific. In `code/Generating_samples.ipynb`, class
`CVanillaMolGen_RNN`:

```python
def _graph_conv_forward(self, X, A, c, ids):
    X_out = [X]
    for conv, bn, linear_c in zip(self.conv, self.bn, self.linear_c):
        X = X_out[-1]
        if bn is not None:
            X_out.append(conv(self.activation(bn(X)), A) + linear_c(c)[ids, :])
        else:
            X_out.append(conv(X, A) + linear_c(c)[ids, :])
    ...
```

and `_policy_0(self, c) → self.dense_policy_0(c) + 0.0 * self.policy_0.data(c.context)`,
where `dense_policy_0` is a `_TwoLayerDense(N_C, N_A*3, N_A)` ending in softmax.

Two details worth noting:
- `linear_c(c)[ids, :]` — `ids` maps every *atom* to its molecule, so each atom receives its
  own molecule's protein vector. Variable-size molecules are handled correctly.
- The `+ 0.0 * self.policy_0` term is deliberate: it keeps the unconditional `policy_0`
  parameter registered in the conditional model so that the pretrained checkpoint still
  loads, while contributing nothing to the output.

**NEW NOTEBOOKS** — `scripts/finetune_cell.py` reimplements this differently and incorrectly;
see [07](07_Finetuning_and_Conditioning.md) §5.

**CURRENT EXECUTED STATE** — no conditional model has ever been instantiated or trained here.

### So what *is* the condition, precisely?

**A single fixed-length real-valued vector `c` derived from the target protein's amino-acid
sequence.** Nothing else.

- It is **not** molecular properties (logP, QED, MW). No property conditioning exists.
- It is **not** a toxicity flag. `tox_class` is threaded through the loaders but never enters
  the network — it was designed to weight the *loss*, not to be an input.
- It is **not** multi-target. One protein, one vector, one molecule at a time.
- It is **not** a 3D pocket. The papers stress this as an advantage over Pocket2Mol.

## 2.6 Datasets

**PAPER** (`papers/CTDDG.pdf` §2.1):

| Dataset | Role | Paper's numbers |
|---|---|---|
| **ChEMBL** | pretraining | ~2.3M curated; **1,093,589** extracted after physicochemical filtering |
| **BindingDB** (Grechishnikova's version) | fine-tuning | **166,400** protein–ligand pairs; **1042** train proteins, **104** test proteins |

Also stated: 65 unique atom types, 4 bond types; train-set proteins mostly <40% pairwise
similarity; train↔test similarity kept <20% "to make test set diverse enough".

⚠ **Internal inconsistency in the paper.** §2.1 says "166400 pairs"; §2.4 says "BindingDB
contains 66400 pairs of proteins and ligands paris for 1146 unique protein sequences".
1042 + 104 = 1146, so the protein count is consistent, but the pair count is not.

**CODE / EXECUTED** — real numbers, read from the preserved outputs of the upstream
`data/data_preprocessing.ipynb` and from files on disk:

| Quantity | Value | Source |
|---|---|---|
| ChEMBL rows in `chembl.csv` | 1,207,360 | file, `wc -l` |
| ChEMBL after filtering & BindingDB removal (`chembl.txt`) | **1,090,529** | file, `wc -l` |
| Fold-1 train pairs | **227,093** | `data/data_preprocessing.ipynb` cell 40 output |
| Fold-1 test pairs | **11,049** | same, cell 41 output |
| Unique BindingDB ligands (all folds) | **154,899** | same, cell 64 output |
| Atom types (BindingDB) / (ChEMBL) / union | 49 / 45 / **65** | same, cells 57 & 69 |

The union of 65 matches the paper exactly; `data/atom_types.txt` has 65 lines. The pair
counts do not match either of the paper's figures — see [03](03_Data_and_Preprocessing.md).

### Toxicity labelling

**PAPER** (`papers/CTDDG.pdf` §2.4):
- **Pretraining** — a **Random Forest classifier** splits ChEMBL into toxic / non-toxic by
  **Ames mutagenicity**; the RF's own training data comes from Wu et al. (2021).
- **Fine-tuning** — BindingDB ligands labelled using **pkCSM** for Ames mutagenicity.

**CODE** — no Random Forest exists anywhere in the repository. `code/pretraining.ipynb`
contains only the comment `# Load ML classifier` with nothing beneath it.

**EXECUTED** — `data/chembl/chembl_final.txt` carries label `1` on all 1,090,529 lines,
produced by `scripts/convert_chembl_format.py`, whose docstring states plainly that it
appends "a dummy label". No toxicity classification has ever been run in this repository.

## 2.7 Generation

**PAPER** — an action is *sampled* from the softmax at each step; the process is explicitly
probabilistic, and CDGCN §3.5 notes "Different sets of molecules were generated … by repeated
sampling as the generative processes here are probabilistic", with duplicates discarded.

**CODE** — `CVanilla_RNN_Builder.sample(num_samples, c, output_type='mol', sanitize=True,
random=True)` in `code/Generating_samples.ipynb`. `random=True` samples from the joint
normalised distribution over all (append, connect, end) options; `random=False` is greedy
argmax. Hard cap of 100 decoding steps. Full walkthrough in [08](08_Generation.md).

⚠ **There is no beam search in this codebase.** The variable named `beam_size` in
`Evaluation_metrics.ipynb` and `Molecular_docking.ipynb` is the *number of stochastic
samples*. The term is inherited from the Grechishnikova transformer baseline, where beam size
genuinely was the sampling control (**PAPER**, `papers/CDGCN.pdf` §3.5: "The generation of S_N
by Grechishnikova was done by setting the beam size hyperparameter equal to N").

## 2.8 Evaluation

**PAPER** — validity, uniqueness, novelty, Lipinski's rule of five (logP<5, MW<500, HBD<5,
HBA<10), rotatable bonds<10, TPSA<140, QED, SAS<6; plus `R_c` (a random-forest
"binding/active" classifier score) in CDGCN only; plus docking-derived target awareness.

**CODE** — `code/Evaluation_metrics.ipynb` implements validity, uniqueness, logP, MW, HBD,
HBA, RB, TPSA, QED, SAS. **Novelty and `R_c` are not implemented** in this repository.

**Reported results** (**PAPER**, `papers/CTDDG.pdf` Tables 1–3, 100 samples × 104 test
proteins × 10 runs):

| Metric | CDGCN | SSVAE | CTDDG |
|---|---|---|---|
| Validity (%) ↑ | 91.7 ± 0.22 | 88.4 ± 3.29 | **94.4 ± 0.32** |
| Uniqueness (%) ↑ | 91.7 ± 0.22 | 58.1 ± 4.13 | **94.4 ± 0.32** |
| logP < 5 (%) ↑ | 68.6 | 51.4 | **74.8** |
| MW < 500 (%) ↑ | **73.5** | 51.4 | 71.1 |
| QED (absolute) ↑ | **0.55** | 0.44 | 0.51 |
| SAS < 6 (%) ↑ | 84.7 | 51.4 | **88.6** |
| Binding affinity < −7 kcal/mol, 5 proteins, avg (%) ↑ | **95.0** | 8.04 | 93.8 |
| Toxic samples, 5 proteins, avg (%) ↓ | 23.30 | **11.7** | 19.32 |

Read honestly, the paper's own numbers say: CTDDG reduces toxicity from 23.3% to 19.3% versus
CDGCN while *slightly losing* target awareness (95.0 → 93.8) and gaining validity/uniqueness.
SSVAE has the lowest toxicity but essentially no target awareness.

**CURRENT EXECUTED STATE** — no evaluation has been run in this repository.

## 2.9 Docking and the CDK2 case study

**PAPER** (`papers/CTDDG.pdf` §2.5, §4) — Smina; box set 5 Å larger than the reference ligand;
CDK2 pocket from PDB **1AQ1**; comparison inhibitors Staurosporine, Dinaciclib, Roscovitine.
100 molecules generated from the human CDK2 FASTA (UniProt **P24941**); top 10 by binding
affinity taken forward; the 5 with zero Lipinski violations analysed with PLIP; mol3 and mol7
(both forming an H-bond with LEU83, as all three known inhibitors do) carried into 50 ns
GROMACS MD; MM-PBSA binding energies −12.05 (mol3), −6.28 (mol7), −13.42 (Roscovitine)
kcal/mol.

These artefacts are **in this repository**: `Moleculsr_docking_top10GeneratedMol/mol1–10.pdb`
and `Molecular_dynamics_files/{mol3,mol7,roso}_{initial,final}.pdb`. The SMILES are in
`supplimentaryAPIN.pdf` Table S1. They are the authors' published results — **not** anything
reproduced here.

**CODE** — `code/Molecular_docking.ipynb` uses `--autobox_ligand <pdb>_lig.mol2
--autobox_add 5`, i.e. the box is derived from the co-crystallised ligand + 5 Å. This matches
the paper's description. It uses `--exhaustiveness 16 --num_modes 5`; the paper says "default
docking parameters", and smina's default exhaustiveness is 8 — a minor discrepancy.

**CURRENT EXECUTED STATE** — no docking has been run here; the required tools
(`Jupyter_Dock/`, pymol, vina, meeko, prolif, MDAnalysis, openbabel) are all absent.

## 2.10 Terminology to preserve

| Term | Meaning in these papers |
|---|---|
| **known protein** | has known drugs; appears in training data |
| **novel protein** | no known drug; appears in test data |
| **lead** | a molecule with drug-like properties worth optimising |
| **generation path `T`** | the ordered action sequence that builds graph `G` |
| **`Q_α`** | predefined proposal distribution over generation paths |
| **`α`** (paper) / `p` (code) | coefficient of randomness; canonical-rank vs random neighbour |
| **`K`** (paper) / `k` (code) | number of sampled generation paths per molecule |
| **`λ`** | toxicity trade-off parameter in CTDDG Eq. 8; fixed to 1 |
| **`R_c`** | % "binding/active" predictions from RF classifiers (CDGCN only, not implemented) |
| **`S_N`** | a set of N generated molecules |
| **Ames mutagenicity** | the specific toxicity endpoint CTDDG targets |
| **target awareness** | % of generated molecules docking below −7.0 kcal/mol |
