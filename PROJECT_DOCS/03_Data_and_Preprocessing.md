# 03 — Data and Preprocessing

All statistics in this document are either read from files on disk or taken from the
**preserved execution outputs** of `data/data_preprocessing.ipynb` (the upstream
CDGCN/DGGNP notebook, which still carries its original outputs). Nothing is invented; where a
number is unavailable it says so.

---

## 1. Dataset inventory

| Dataset | Role | Location | On disk? |
|---|---|---|---|
| **ChEMBL** (raw export) | source for pretraining | `data/chembl/chembl.csv` | ✅ 295 MB |
| **ChEMBL** (filtered SMILES) | pretraining | `data/chembl/chembl.txt` | ✅ 48 MB |
| **ChEMBL** (labelled) | pretraining input | `data/chembl/chembl_final.txt` | ✅ 51 MB |
| **BindingDB** (Grechishnikova version) | fine-tuning | `data/bindingdb/{train,test}_dataset/` | ✅ |
| **Atom-type vocabulary** | model input spec | `data/atom_types.txt` | ✅ 65 lines |
| **SMILES token vocab** (Grechishnikova baseline) | not used by CTDDG | `data/bindingdb/vocab.protein_specific_ligand_generation.tokens` | ✅ 62 tokens |
| **Needleman-Wunsch alignments** | similarity QC | `data/bindingdb/needle_outputs/` | ✅ 5,809,835 files |
| **Protein embeddings** | conditioning | *(would be `d{i}_{tr,te}_unique_bioembeddings.txt`)* | ❌ **absent** |
| **Paired fine-tuning file** | fine-tuning | *(would be `d{i}_tr_cdgcn.txt`)* | ❌ **absent** |
| **`fpscores.pkl.gz`** | SAS metric | `data/fpscores.pkl.gz` | ❌ **absent** |
| **CrossDocked2020** | future multi-target work | `~/datasets/downsampled_CrossDocked2020_v1.3/` | ✅ 2,801 dirs, extracted |

---

## 2. ChEMBL

**Source.** `papers/CTDDG.pdf` §2.1: "ChEMBL is a manually curated public database of
approximately 2.3 million molecules. 1093589 molecules satisfying the desired physicochemical
properties (Section 3.4) were extracted and used for the pre-training". Citation: Mendez et
al., 2018.

**Purpose.** Unconditional pretraining — teach the generator general drug-like chemistry.
No protein is involved.

**File format.**

`chembl.csv` — semicolon-delimited export with 31 columns, 1,207,360 data rows:
```
"ChEMBL ID";"Name";…;"QED Weighted";…;"Heavy Atoms";…;"Smiles"
"CHEMBL1902090";"";…;"0.88";…;"25";…;"Cc1ccc(CCN2CC(C(=O)NCc3cccnc3)CC2=O)cc1"
```
The physicochemical filtering the paper mentions (following Grechishnikova) was applied
**when this CSV was exported from the ChEMBL web interface**, not in code. `code/data_preprocessing_1.ipynb`
cell 62 says as much: *"We have the ChEMBL dataset based on the physiochemical filters as
suggested by Grechishnikova."* **The exact filter thresholds are NOT DETERMINED FROM AVAILABLE
SOURCES** — they are not recorded anywhere in this repository.

`chembl.txt` — one plain SMILES per line, **1,090,529 lines**.

`chembl_final.txt` — `<SMILES> <class>`, **1,090,529 lines**:
```
c1ccc(-n2cc(CNC3CCc4ncnn4C3)cn2)cc1 1
Cc1cccc(CCN2CCC[C@@H]2C)c1 1
```

**In-code processing** (`code/data_preprocessing_1.ipynb` cells 61–65):

```python
for s in chembl_df['Smiles']:
    mol = Chem.MolFromSmiles(s)
    if len(mol.GetAtoms()) <= 100:
        chembl_mols.append([b for b in s.split('.') if len(b) == max(...)][0])   # keep longest salt fragment
...
f.write(s+'\n') for s in set(chembl_mols) - set(bindingdb_ligs)
```

Three operations: (1) drop molecules with >100 atoms; (2) **salt stripping** — split on `.`
and keep the longest component; (3) **subtract all BindingDB ligands** to prevent leakage into
the fine-tuning/test data. (Salt stripping is applied to the BindingDB ligands too, in cell 42.) Step (3) is required by both papers
(`papers/CDGCN.pdf` §3.1: "hence they were removed from the modified ChEMBL dataset to avoid
data leakage during evaluation").

### ⚠ The toxicity label is a placeholder

`chembl_final.txt` was produced by `scripts/convert_chembl_format.py`, not by the
preprocessing notebook. Verified facts:

- `diff <(sed 's/ 1$//' chembl_final.txt) chembl.txt` → **identical**.
- `awk '{print $NF}' chembl_final.txt | sort | uniq -c` → **`1090529 1`**. Every molecule is
  class 1; there are zero class-0 molecules.
- The script's own docstring: *"So we just append a dummy label ' 1' to each line."*

**PAPER** requires a Random Forest classifier trained on Wu et al. (2021) data to split
ChEMBL by Ames mutagenicity. **No such classifier exists in this repository** —
`code/pretraining.ipynb` has a bare comment `# Load ML classifier` and nothing under it.

Consequence: the pretraining dataset carries **no toxicity signal**. See
[06](06_Pretraining_and_Checkpoints.md) §7.

---

## 3. BindingDB — the Grechishnikova version

**Source.** Both papers use the BindingDB preprocessing of **Grechishnikova (2021)**,
*"Transformer neural network for protein-specific de novo drug generation as a machine
translation problem"*, Scientific Reports 11:321. In it, each protein is a FASTA sequence and
each ligand a SMILES string (**PAPER**, `papers/CTDDG.pdf` §2.1).

Physically, the copy here came from the CDGCN author's data repository — `data/.git`'s remote
is `git://git-lfs.github.com/mshik/DGGNP.git`.

**Purpose.** Conditional fine-tuning: teach the pretrained generator to produce ligands
*for a given protein*.

### 3.1 The five cross-validation folds

CDGCN §3.1: "The BindingDB dataset is divided into five cross-validation folds where the
validation folds are further split in half to obtain validation data … and separate test
data". Directory naming is `{train,test}_4_org_<Ntrain>_<Ntest>`:

| Fold | Directory suffix | Train proteins | Test proteins | Train pairs | Test pairs |
|---|---|---|---|---|---|
| 1 | `1042_104` | 1042 | 104 | **227,093** | **11,049** |
| 2 | `1000_112` | 1000 | 112 | **144,000** | **16,200** |
| 3 | `1004_122` | 1004 | 122 | **146,060** | **26,640** |
| 4 | `1002_103` | 1002 | 103 | **144,885** | **16,015** |
| 5 | `1036_124` | 1036 | 124 | **142,687** | **25,875** |

Protein counts verified by counting `protein*.fasta` files in each directory. Pair counts
from `data/data_preprocessing.ipynb` cells 40/41 stored outputs.

**Fold 1 is the one both papers report on** (1042 / 104).

### 3.2 The fold-1 repair

`code/data_preprocessing_1.ipynb` markdown cells 6 and 8 record the problem:

> "Dataset 1 actually has 1514 proteins in train dataset, whereas in Grechishnikova paper it
> is given as 1042. All other datasets are correct."
> "Dataset 1 actually has 99 proteins in test dataset, whereas … 104."

The repair (cells 11–18) is deterministic under `random.seed(17)`:
- Train: keep the 456 proteins unique to fold 1, top up from the other folds' train proteins
  to reach 1042.
- Test: keep the 22 proteins unique to fold 1's test set, top up from the other folds' test
  proteins to reach 104.

Confirmed by the stored outputs (cells 13 → `1042`, cell 17 → `104`) and by the current
directories containing exactly 1042 and 104 FASTA files. **This repair has already been
applied to the data on disk.**

### 3.3 File formats inside a fold directory

| File | Format |
|---|---|
| `protein{1..N}.fasta` | Single line, bare amino-acid sequence, **no `>` header** |
| `proteins_train_4_org_corrected` | One **space-separated** amino-acid sequence per pair (`M A F M K K Y…`), 227,093 lines for fold 1 |
| `ligands_train_4_org_corrected` | One **space-separated** SMILES per pair (`O = C ( N 1 C C 2 …`), same line count, row-aligned with the proteins file |
| `d{i}_{tr,te}_unique_proteins.txt` | One space-separated sequence per **unique** protein |

The space separation is character-level tokenisation for Grechishnikova's transformer. The
graph model does **not** want it — see §6.

### 3.4 Similarity control

**PAPER** (`papers/CTDDG.pdf` §2.1): majority of training sequences share <40% pairwise
similarity; train↔test similarity kept within 20%. Measured with Needleman-Wunsch global
alignment from the **EMBOSS** package.

**CODE** (`code/data_preprocessing_1.ipynb` cells 30–35):
```
needle <train>.fasta <test>.fasta stdout -gapopen 10.0 -gapextend 0.5 > <out>.txt
```
run for all three combinations. Cell 37 then parses `line[24]` of each output and plots a
histogram.

**EXECUTED — complete.** File counts match the expected totals exactly:

| Comparison | Files present | Expected (Σ over folds) |
|---|---|---|
| train↔test | 574,526 | 1042·104 + 1000·112 + 1004·122 + 1002·103 + 1036·124 = **574,526** ✅ |
| within-train | 5,171,080 | 1042² + 1000² + 1004² + 1002² + 1036² = **5,171,080** ✅ |
| within-test | 64,229 | 104² + 112² + 122² + 103² + 124² = **64,229** ✅ |

The similarity *histograms* were never produced — cell 37 crashes (§7). The alignments
themselves are all on disk and need not be recomputed.

### 3.5 Ligand filtering

`filter_by_atom_count` (cell 49) drops molecules with >100 atoms. Stored output from
`data/data_preprocessing.ipynb` cell 50 shows the counts **unchanged** for all ten
train/test sets — i.e. **the filter removed nothing**; every BindingDB ligand already had
≤100 atoms.

### 3.6 ⚠ Pair counts do not match the paper

| Source | Value |
|---|---|
| `papers/CTDDG.pdf` §2.1 | "166400 pairs of proteins and ligands" |
| `papers/CTDDG.pdf` §2.4 | "BindingDB contains 66400 pairs … for 1146 unique protein sequences" |
| Fold 1 on disk | 227,093 + 11,049 = **238,142** |
| Fold 2 | 144,000 + 16,200 = 160,200 |
| Fold 3 | 146,060 + 26,640 = 172,700 |
| Fold 4 | 144,885 + 16,015 = 160,900 |
| Fold 5 | 142,687 + 25,875 = 168,562 |

The paper contradicts itself (166,400 vs 66,400) and neither figure matches any fold. The
protein count *is* consistent: 1042 + 104 = 1146. **Which pair count the paper intends is NOT
DETERMINED FROM AVAILABLE SOURCES.** INFERRED — NOT EXPLICITLY CONFIRMED: 166,400 may refer to
a de-duplicated count, or to a fold other than 1.

---

## 4. Atom types

`data/atom_types.txt`, 65 lines, `symbol,formal_charge,num_explicit_Hs`:
```
P,-1,0
Br,0,0
Sb,0,0
Cl,1,0
...
```

Built in `code/data_preprocessing_1.ipynb` cells 56–70 as the **union** of the atom types in
BindingDB ligands and in ChEMBL molecules.

**EXECUTED** (`data/data_preprocessing.ipynb` cells 57, 69):
- BindingDB ligands → **49** distinct types
- ChEMBL → **45** distinct types
- Union → **65**

This matches `papers/CTDDG.pdf` §2.1 ("65 unique atom types and four unique bond types")
exactly. Bond types come from `MoleculeSpec.bond_orders`, hardcoded as
`[AROMATIC, SINGLE, DOUBLE, TRIPLE]`.

⚠ **`data/atom_types.txt` is line-order dependent.** `MoleculeSpec.get_atom_type` returns
`self.atom_types.index(...)`, so the integer index of every atom type is determined by the
file's line order. The trained checkpoint's `embedding_atom` matrix and `policy_0` vector are
tied to this exact ordering. **Regenerating this file would silently invalidate the trained
checkpoint** — the preprocessing notebook builds it from a Python `set()`, so the order is
not reproducible. Treat the current file as immutable.

---

## 5. Train / validation / test splits

**PAPER** (`papers/CDGCN.pdf` §3.1): validation folds are split in half → validation for
hyperparameter tuning, test for evaluation. Training proteins = "known"; validation/test
proteins = "novel".

**CODE** (`code/data_preprocessing_1.ipynb` cells 84–87), `random.seed(17)`:
```python
d1_va_uprots = random.sample(d1_te_uprots, len(d1_te_uprots)//2)
d1_te_uprots = list(set(d1_te_uprots) - set(d1_va_uprots))
```

**EXECUTED** (upstream notebook cells 86–87):

| Fold | Validation proteins | Test proteins |
|---|---|---|
| 1 | 52 | 52 |
| 2 | 56 | 56 |
| 3 | 61 | 61 |
| 4 | 51 | 52 |
| 5 | 62 | 62 |

⚠ Note the tension: the **papers report on 104 test proteins**, but this split halves the 104
into 52 + 52. **Whether the reported results use all 104 or only the 52-protein test half is
NOT DETERMINED FROM AVAILABLE SOURCES.** `code/Evaluation_metrics.ipynb` sets `nprot = 104`,
which suggests all 104 were used for the reported tables.

---

## 6. Building the model-ready fine-tuning file

**Target format** (`code/data_preprocessing_1.ipynb` cells 100–106, the section titled
`### CDGCN`):

```
<ligand SMILES>\t<e_1>\t<e_2>\t…\t<e_NC>\n
```
written to `d1_tr_cdgcn.txt`, one line per protein–ligand pair.

The pairing loop:
```python
lines_d1.append(ligands_d1[i] + '\t' + embeds_d1[j])   # cell 105
```

### 6.1 ⚠ Two problems in this section

**(a) The ligand gets re-spaced.** The ligands are loaded *correctly*: cell 40 reads
`ligands_train_4_org_corrected` and strips the tokenisation spaces
(`line.strip().replace(' ', '')`), and cell 42 additionally salt-strips (longest `.`-fragment)
before writing `d1_tr_ligands.txt`. So `d1_tr_ligs` holds plain SMILES.

But the CDGCN assembly cell then puts the spaces **back**:
```python
# cell 100
ligands_d1.append(d1_tr_ligs[i][0] + d1_tr_ligs[i][1:-1].replace('', ' ') + d1_tr_ligs[i][-1])
```
`.replace('', ' ')` inserts a space between every character, so `CCO` → `C C O`. That line is
copied verbatim from the Grechishnikova section (cell 93), where character tokenisation is
exactly what the transformer wants — but the graph model does not want it, and
`process_single`'s `smiles.split(" ")` would shred it.

The upstream notebook avoids this by round-tripping through the `.space_sep_seq` file and
stripping on read: `data/data_preprocessing.ipynb` cell 102 does
`ligands.append(line.strip().replace(' ', ''))`, and its cell 103 output confirms plain SMILES
(`'CC(C)[C@@]1(NC(=O)…'`). The CTDDG version uses the in-memory spaced string directly.

**(b) The embeddings are the wrong shape.** See [04](04_Protein_Embeddings.md) — this is the
larger of the two problems.

### 6.2 The other two model formats

The same notebook builds datasets for the two baselines, useful context for reading the
directory:

| Suffix | Model | Conditioning |
|---|---|---|
| `.space_sep_seq` | Grechishnikova transformer | character-tokenised sequence ↔ SMILES |
| `d{i}_tr_li.txt` | Li et al. | **one-hot** protein identity, `np.identity(1000)` — works only for known proteins |
| `d{i}_tr_cdgcn.txt` / `d{i}_*_dggnp.txt` | CDGCN / CTDDG | **protein embedding vector** |

Upstream cell 122's output shows the Li et al. file: `SMILES\t1\t0\t0\t0…`.

---

## 7. Preprocessing execution status

| Stage | Status | Evidence |
|---|---|---|
| Fold-1 protein repair | ✅ DONE | 1042 / 104 FASTA files present |
| Needleman-Wunsch alignments | ✅ DONE | 5,809,835 output files, counts match exactly |
| Similarity histograms | ❌ FAILED | `preprocessing_11849.out` — `TypeError` at cell 37 |
| Ligand filtering (>100 atoms) | ✅ DONE upstream | filter is a no-op; ligand files present |
| `atom_types.txt` | ✅ ON DISK | 65 lines |
| `chembl.txt` | ✅ ON DISK | 1,090,529 lines |
| `chembl_final.txt` | ✅ ON DISK | via `convert_chembl_format.py` (dummy labels) |
| Protein embeddings | ❌ NOT RUN | no `*bioembeddings*` file anywhere |
| `d{i}_tr_cdgcn.txt` | ❌ NOT RUN | absent |
| `d{i}_te_embeddings_run{r}.txt` | ❌ NOT RUN | absent — **and not produced by any code here** (§8) |

### The blocking bug

`code/data_preprocessing_1.ipynb` cell 37:
```python
for p1 in range(1, len(ds_len[i][0])+1):      # ds_len[i][0] == 1042, an int
    for p2 in range(1, len(ds_len[i][1])+1):
```
`ds_len` is `[(1042, 104), (1000, 112), …]`, so `ds_len[i][0]` is an `int`.
→ `TypeError: object of type 'int' has no len()`

Recorded in `preprocessing_11849.out` (job 11849, node004, 2026-08-25 08:49:18).
This is a **bug in the original notebook**, not a porting artefact. It only affects
histogram plotting; per §3.4 the alignments themselves are complete.

An earlier attempt, job 11771 (2026-08-17, node003), was killed by the Slurm time limit while
still running `needle`. The `if not os.path.exists(output_file)` resume guard now in the
notebook is a **local uncommitted modification** added in response.

---

## 8. Files the code expects but nothing produces

| Expected by | Path | Produced by |
|---|---|---|
| `Generating_samples.ipynb` cell 19 | `data/d{i}_te_embeddings_run{r}.txt` | ❌ **nothing in this repository** |
| `finetune_cell.py` | `data/bindingdb/train_dataset/train_4_org_1042_104/d{i}_tr_cdgcn.txt` | `data_preprocessing_1.ipynb` cell 106 (for `i=1` only) |
| `Evaluation_metrics.ipynb` | `data/fpscores.pkl.gz` | `scripts/download_data.sh` (not run) |
| `Evaluation_metrics.ipynb` cell 16 | `outputs/finetune/log.out` | ❌ nothing — note this is a *different* directory from where `finetune_cell.py` writes |

The `run{r}` file naming implies a per-run subsample of the test-protein embeddings.
**How it was constructed is NOT DETERMINED FROM AVAILABLE SOURCES.** INFERRED — NOT EXPLICITLY
CONFIRMED: a subset of the rows of `d{i}_te_unique_bioembeddings.txt`, one line per protein,
in the same order that `generate_samples()` then labels `Protein1…ProteinN`.

---

## 9. Data flow summary

```
ChEMBL web export (filters applied at export — thresholds unrecorded)
   │  data/chembl/chembl.csv                        1,207,360 rows
   ▼  drop >100 atoms · keep longest salt fragment · subtract BindingDB ligands
   │  data/chembl/chembl.txt                        1,090,529 SMILES
   ▼  scripts/convert_chembl_format.py  (append " 1" — DUMMY)
   │  data/chembl/chembl_final.txt                  1,090,529 lines
   ▼
   └──────────────► PRETRAINING (VanillaMolGen_RNN)  ✅ done


BindingDB / Grechishnikova (5 folds)
   │  proteins_*_corrected + ligands_*_corrected   (space-separated, row-aligned)
   ▼  fold-1 repair (seed 17) → 1042 train / 104 test proteins
   ▼  needle similarity QC  ✅ 5,809,835 alignments
   ▼  drop >100-atom ligands (no-op)
   ├──► atom types (49) ─┐
   │                     ├─► union 65 ─► data/atom_types.txt  ✅
   │   ChEMBL types (45)─┘
   ▼  ProtTransBertBFDEmbedder(protein)             ❌ NOT RUN — and shape is wrong (see 04)
   ▼  pair: <ligand SMILES>\t<embedding>            ❌ NOT RUN
   │  d1_tr_cdgcn.txt
   ▼
   └──────────────► FINE-TUNING                     ❌ BLOCKED
```
