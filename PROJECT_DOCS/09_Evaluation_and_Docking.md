# 09 — Evaluation and Docking

Sources: `code/Evaluation_metrics.ipynb`, `code/Molecular_docking.ipynb`, and both papers.

**Status: NOT STARTED.** Both stages depend on generated molecules, which depend on
fine-tuning. Evaluation additionally needs `data/fpscores.pkl.gz`; docking needs
`Jupyter_Dock/` and six absent Python packages.

---
---

# PART A — EVALUATION

## A.1 Metrics actually implemented

`code/Evaluation_metrics.ipynb` computes ten quantities. For each: what it is, how it is
calculated *here*, how to read it, and where it misleads.

### A.1.1 Validity

**Definition.** Fraction of requested molecules that RDKit can parse and sanitize into a real
molecule.

**Calculation** (cell 13): `valid % = 100 * len(df) / beam_size`, averaged over proteins, then
over 10 runs, with `statistics.pstdev` for the spread.

Because `generate_samples` writes only successfully sanitized SMILES, `len(df)` *is* the valid
count and the shortfall from `beam_size` is the failure rate.

**Interpretation.** Whether the model has learned chemical grammar — valences, ring closures,
aromaticity. Paper: CTDDG 94.4 ± 0.32%, CDGCN 91.7 ± 0.22%, SSVAE 88.4 ± 3.29%
(`papers/CTDDG.pdf` Table 1).

**Limitations.** A weak bar. Methane is valid. Validity says nothing about drug-likeness,
novelty or binding. Graph models are structurally advantaged here versus SMILES models
(`papers/CDGCN.pdf` §1).

### A.1.2 Uniqueness

**Definition.** Fraction of the requested molecules that are distinct canonical SMILES.

**Calculation:** `100 * df['smiles'].nunique() / beam_size`.

⚠ Note the denominator is `beam_size`, not `len(df)`. So uniqueness is **bounded above by
validity** and the two are numerically identical whenever every valid molecule is distinct —
which is exactly what the paper's Table 1 shows (CTDDG 94.4/94.4, CDGCN 91.7/91.7). That is a
property of the metric definition, not a coincidence.

**Limitations.** Measures within-set diversity only. A model emitting 100 distinct but
near-identical scaffolds scores 100%.

### A.1.3 Novelty — ⚠ NOT IMPLEMENTED

Both papers report it (`papers/CDGCN.pdf` Table 1: "Novel (%)", 99.7% for CDGCN, computed by
"exact string matching algorithm on the molecules in the dataset"). **No novelty computation
exists in `Evaluation_metrics.ipynb`.**

⚠ `notebooks/04_evaluation.ipynb` nevertheless advertises "Computes validity, uniqueness,
**novelty**, QED, SAS, and Lipinski properties." That claim is false.

Implementing it is straightforward: load the BindingDB + ChEMBL SMILES into a `set()` and
test membership. See [12](12_BTP_Modification_Guide.md) §9.

### A.1.4 Lipinski's Rule of Five and bioavailability filters

```python
def lipsinki_properties(samples_df):        # [sic]
    hd_s   = Chem.Lipinski.NumHDonors(m)
    ha_s   = Chem.Lipinski.NumHAcceptors(m)
    rb_s   = Chem.Lipinski.NumRotatableBonds(m)
    tpsa_s = Chem.Descriptors.TPSA(m)

def molecular_properties(samples_df):
    mw_s   = Descriptors.ExactMolWt(m)
    logp_s = Crippen.MolLogP(m, includeHs=True)
    qed_s  = QED.qed(m)
```

| Property | Threshold | Rationale (paper) |
|---|---|---|
| logP | < 5 | octanol–water partition coefficient; lipophilicity |
| MW | < 500 Da | molecular weight |
| H-bond donors | < 5 | Lipinski |
| H-bond acceptors | < 10 | Lipinski |
| Rotatable bonds | < 10 | oral bioavailability (Veber et al.) |
| TPSA | < 140 Å² | oral bioavailability (Veber et al.) |

**PAPER**, `papers/CTDDG.pdf` §3.2.1, citing Ghose et al. (1999) and Veber et al. (2002).

**Limitations.** Rules of thumb for *orally active* drugs, with well-known exceptions
(antibiotics, peptides, kinase inhibitors). Satisfying them is neither necessary nor
sufficient for a real drug.

### A.1.5 ⚠ How the property percentages are actually computed

This is the subtlest part of the evaluation and is easy to misread:

```python
logPi_.append(100 * (df['smiles'].nunique() / ns) * len(df.loc[df['logP'] < 5]) / ns)
#              ↑ uniqueness fraction ↑            ↑ count satisfying the property ↑
# where ns = beam_size
```

The reported "logP < 5 (%)" is **not** the fraction of generated molecules satisfying
logP < 5. It is

```
100 × (unique / N) × (satisfying / N)
```

— the property-satisfaction rate **multiplied by** the uniqueness rate. So a model with 90%
uniqueness and 90% logP-compliance reports **81%**, not 90%.

This compounding applies to logP, MW, HD, HA, RB, TPSA and SAS. It does **not** apply to QED,
which is reported as a plain mean.

Consequence: these numbers are **jointly** measuring diversity and property compliance and
cannot be compared against a differently-defined literature number. When reporting your own
results, state the formula explicitly. It does explain why the paper's Table 1 numbers look
lower than one might expect from RDKit property distributions.

### A.1.6 QED — Quantitative Estimate of Drug-likeness

`QED.qed(m)` (Bickerton et al. 2012). Range 0–1, higher = more drug-like; a desirability-weighted
combination of MW, logP, HBA, HBD, PSA, rotatable bonds, aromatic rings and structural alerts.

**Calculation:** `qedi_.append(df['QED'].sum() / ns)` — the mean over `ns = beam_size`
(⚠ not over valid molecules, so invalid ones drag the mean down as zeros).

Paper: CTDDG 0.51, CDGCN 0.55, SSVAE 0.44 (`papers/CTDDG.pdf` Table 1). For reference, the
three known CDK2 inhibitors score 0.40–0.58 (Table 4) — so ~0.5 is a realistic target, not a
weak one.

### A.1.7 SAS — Synthetic Accessibility Score

Ertl & Schuffenhauer (2009). Range 1–10, **lower = easier to synthesise**; threshold used
here is < 6.

The notebook **vendors RDKit's `SA_Score` implementation inline** (`readFragmentScores`,
`numBridgeheadsAndSpiro`, `calculateScore`, `processMols`), including the Novartis copyright
header. It reads a fragment-score table:

```python
data = pickle.load(gzip.open('/home/muhamed/repo/CTDDG/data/fpscores.pkl.gz'))
```

Score = fragment-contribution term (Morgan fingerprint radius 2, scored against the table)
− size/stereo/spiro/bridgehead/macrocycle penalties + a symmetry correction, rescaled to 1–10.

⚠ **`data/fpscores.pkl.gz` is absent.** `scripts/download_data.sh` fetches it from
`https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz`.
Without it, `readFragmentScores` raises and all SAS values fail.

**Limitations.** A heuristic trained on PubChem fragment frequencies. It rewards *common*
substructures, so it under-rates novel-but-tractable chemistry and over-rates familiar-but-hard
chemistry. It is not a retrosynthesis engine.

### A.1.8 `R_c` — ⚠ NOT IMPLEMENTED

CDGCN §3.4 reports `R_c`: the percentage of "binding/active" predictions from per-protein
binary random-forest classifiers built on ECFP6 fingerprints (100 known ligands positive, 100
non-ligands negative, classifiers retained only at ≥95% accuracy).

Not implemented anywhere here, and CTDDG does not report it either — it uses docking-based
target awareness instead (§B.6).

## A.2 Aggregation procedure

```python
nprot = 104
for beam_size in [10, 100, 1000]:
    for run in range(10):
        for i in range(nprot):
            ...
        metric_.append(sum(metric_i) / nprot)          # average over proteins
    print(round(sum(metric_)/10, 1), '+-', round(statistics.pstdev(metric_), 2))
```

Mean over 104 proteins → then mean ± **population** standard deviation over 10 runs. The `±`
in the papers' tables is therefore **run-to-run variability**, not per-protein spread.

### ⚠ Loop-bound inconsistency
Cells 11 and 12 iterate `for prot_id in range(1, 11)` — **10 proteins**. Cell 13 iterates
`for i in range(nprot)` with `nprot = 104`. As shipped, cell 13 raises `IndexError` on the
11th protein. Either the ranges were edited between runs, or the cells were run against
different data. **NOT DETERMINED FROM AVAILABLE SOURCES.** Fix both to the same value before
running.

## A.3 Fine-tuning loss plot

```python
with open('/home/muhamed/repo/CTDDG/outputs/finetune/log.out') as f:
    for i, line in enumerate(f):
        if i > 1 and len(line.split('\t')) == 4 and line.split('\t')[0] != 'step':
            ctdgd_losses.append(float(line.split('\t')[2]))
```

Confirms the fine-tuning stage was expected to write a `log.out` in the **same four-column
`step\ttime\tloss\tlr` format** as pretraining.

⚠ Path conflict: this reads `outputs/finetune/log.out`, but `generate_samples` loads the model
from `outputs/CTDGD/Dataset{i}/model/` and `scripts/finetune_cell.py` writes there. Pick one
convention ([07](07_Finetuning_and_Conditioning.md) §7 item 7).

## A.4 Evaluation prerequisites

| Requirement | Status |
|---|---|
| Generated CSVs under `outputs/CTDGD/Dataset1/generated_samples/{10,100,1000}/run{1..10}/` | ❌ |
| `data/fpscores.pkl.gz` | ❌ absent |
| RDKit, pandas, numpy | ✅ in `ctddg_env` |
| `molvs` | ✅ (imported; `standardize_smiles` unused) |
| Consistent protein loop bounds | ⚠ needs a one-line edit |

---
---

# PART B — DOCKING

## B.1 What docking is, and is not

**Molecular docking** searches for the best geometric and energetic fit of a small molecule
into a protein's binding site, and scores it with an empirical function calibrated against
known complexes. Output: a pose plus a score in kcal/mol, more negative = better predicted fit.

### What a docking score **can** support
- A **relative ranking** of many ligands against one target under one protocol.
- A **coarse filter**: −11 kcal/mol is more promising than −4.
- Structural hypotheses: which residues a ligand might contact.
- A sanity check that a generator is producing pocket-compatible chemistry.

### What a docking score **cannot** establish
- ❌ **It is not a binding affinity.** Correlation between docking scores and measured Kd/Ki
  is famously weak. A score is not a ΔG measurement.
- ❌ **It is not evidence of biological binding.** No solvent, no full protein flexibility,
  no entropy, no kinetics, no cell.
- ❌ **It is not efficacy.** Binding ≠ inhibition ≠ therapeutic effect.
- ❌ **It is not comparable across targets or protocols.** Scores depend on the box,
  exhaustiveness, protonation, and the receptor structure used.
- ❌ **It says nothing about selectivity, ADMET, or safety.**

The CTDDG paper is appropriately careful. Its docking-derived metric is called
**"target awareness"** — "This metric serves to quantify the model's awareness of the target
to some degree" (`papers/CTDDG.pdf` §3.2.2) — not "binding affinity", and the conclusion
states plainly: "experimental validation through wet-lab studies is a next crucial step to
demonstrate the real-life benefits of the model. At present, this is beyond the scope of the
current manuscript" (§5).

**Use the same register in your own write-ups.** Correct: "93.8% of generated molecules
achieved a Smina docking score below −7.0 kcal/mol against the target structure." Incorrect:
"93.8% of generated molecules bind the target."

## B.2 Protein preparation

```python
# cell 10 — fetch and split with PyMOL
cmd.fetch(code=pdb_id, type='pdb')
cmd.select(name='Prot', selection='polymer.protein')
cmd.select(name='Lig',  selection='organic')
cmd.save(filename=pdb_id+'_clean.pdb', format='pdb',  selection='Prot')
cmd.save(filename=pdb_id+'_lig.mol2',  format='mol2', selection='Lig')
```

The co-crystallised ligand is saved separately — it becomes the docking-box reference (§B.5).

Two alternative protonation routes are provided:
```python
# cell 16 — LePro (preferred per the notebook's markdown)
os.system('.../Jupyter_Dock/bin/lepro_linux_x86 {}_clean.pdb'.format(pdb_id))
os.rename('pro.pdb', '{}_clean_H.pdb'.format(pdb_id))

# cell 17 — Jupyter Dock's fix_protein
fix_protein(filename=f'{pdb_id}_clean.pdb', addHs_pH=7.4,
            try_renumberResidues=True, output=f'{pdb_id}_clean_H.pdb')
```
Then to PDBQT:
```python
os.system('.../bin/prepare_receptor -v -r {}_clean_H.pdb -o {}_clean_H.pdbqt'.format(pdb_id, pdb_id))
```

## B.3 Ligand preparation

```python
os.system(f'obabel -:"{smiles}" --gen3D -O {pdb_id}_{prot_id+1}_gen_mol{i+1}_{modelname}.mol2')
mol = [m for m in pybel.readfile(filename=..., format='mol2')][0]
mol.addh()
out = pybel.Outputfile(filename=f'..._H.mol2', format='mol2', overwrite=True)
out.write(mol); out.close()
...
os.system(f'.../bin/prepare_ligand -v -l ..._H.mol2 -o ..._H.pdbqt')
```

SMILES → 3D conformer (`obabel --gen3D`) → add hydrogens (pybel) → PDBQT.

⚠ `--gen3D` produces **one** conformer with default settings. CDGCN §3.4 used OpenBabel the
same way. Single-conformer docking is a known accuracy limitation, mitigated somewhat by
Vina/Smina's own torsional search.

## B.4 File formats

| Format | Role |
|---|---|
| `.pdb` | protein structure from RCSB |
| `.mol2` | ligand with explicit bond orders; also the box reference |
| `.pdbqt` | AutoDock format — coordinates + partial charges + AutoDock atom types |
| `_clean.pdb` / `_clean_H.pdb` | protein only / protonated |
| `_smina_out.pdbqt` | docked poses, score in the header of each MODEL |

## B.5 Docking box

```python
cmd.load(filename='{}_clean_H.pdb'.format(pdb_id), format='pdb', object='prot')
cmd.load(filename=f'..._H.mol2', format='mol2', object='lig')
center, size = getbox(selection='lig', extending=5.0, software='vina')
cmd.delete('all')
```
and the actual invocation:
```python
os.system(f'.../bin/smina -r {pdb_id}_clean_H.pdbqt -l ..._H.pdbqt '
          f'--autobox_ligand {pdb_id}_lig.mol2 --autobox_add 5 '
          f'--exhaustiveness 16 --num_modes 5 -o ..._smina_out.pdbqt')
```

The box is derived from the **co-crystallised ligand** (`{pdb_id}_lig.mol2`) extended by
**5 Å** — i.e. this is **site-directed** docking into a known pocket, not blind docking.

**Matches the paper:** "the docking box was set to 5 larger than the ligand, using the CDK2
binding pocket (PDB ID: 1AQ1) as the docking site" (`papers/CTDDG.pdf` §2.5).

⚠ Note the `getbox(...)` result is computed and then **discarded** — `--autobox_ligand` makes
smina compute the box itself. The `getbox` call is vestigial (it would be needed for the
commented-out explicit `--center_x/--size_x` variant).

⚠ **Requires a known binding site.** Both papers restrict docking evaluation to five test
proteins with available binding sites: **4nos, 2nru, 1lqf, 1k2r, 1cqp** (`papers/CTDDG.pdf`
Table 2). You cannot dock against a protein whose pocket you do not know.

## B.6 Parameters, scoring and ranking

| Parameter | Value | Note |
|---|---|---|
| Engine | **smina** (AutoDock Vina fork) | binary from `Jupyter_Dock/bin/` |
| Scoring function | Vina default | not overridden |
| `--exhaustiveness` | **16** | ⚠ smina's default is 8; the paper says "default docking parameters" |
| `--num_modes` | 5 | 5 poses retained per ligand |
| `--autobox_add` | 5 (Å) | matches paper |

**Ranking.** The paper's two docking-derived analyses:
1. **Target awareness** — % of generated molecules scoring **< −7.0 kcal/mol**
   (`papers/CTDDG.pdf` Table 2). Threshold rationale from CDGCN §3.5: "Binding energy of −6.0
   kcal/mol is considered as minimum threshold for any molecule to be used for drug
   development."
2. **Case study** — the **top 10 lowest-scoring** of 100 CDK2-generated molecules
   (`papers/CTDDG.pdf` Table 4), then filtered by zero Lipinski violations, then by PLIP
   interaction fingerprint, then MD.

⚠ **No score-parsing code exists in `Molecular_docking.ipynb`.** It writes `_smina_out.pdbqt`
files and stops. Extracting scores, ranking, and the −7.0 threshold statistic all have to be
written. See [12](12_BTP_Modification_Guide.md) §10.

## B.7 The published CDK2 case study (authors' results, not reproduced here)

**PAPER**, `papers/CTDDG.pdf` §4:
- Target: human CDK2, UniProt **P24941**; docking site from PDB **1AQ1**; MD reference pose
  from PDB **2A4L**.
- 100 molecules generated → top 10 by score (Table 4, −13.09 to −11.26 kcal/mol).
- mol3, mol4, mol6, mol8, mol10 have zero Lipinski violations → PLIP analysis (Table 5).
- All three known inhibitors H-bond with **LEU83**; mol3 and mol7 also do → selected for MD.
- 50 ns GROMACS 2023.1, CHARMM36m, TIP3P, 303.15 K, 150 mM NaCl, CHARMM-GUI setup.
- MM-PBSA over the last 5 ns (50 frames): **mol3 −12.05**, **mol7 −6.28**,
  **Roscovitine −13.42** kcal/mol.
- mol3 vs Roscovitine Tanimoto: **0.46** (MACCS), **0.21** (2D pharmacophore) — structurally
  distinct despite similar interactions.

**Artefacts present in this repository:**
- `Moleculsr_docking_top10GeneratedMol/mol{1..10}.pdb` — the docked poses
- `Molecular_dynamics_files/{mol3,mol7,roso}_{initial,final}.pdb` — MD start/end frames
- `supplimentaryAPIN.pdf` Table S1 — the SMILES; Table S2 — final-frame interactions

These are the **authors' published outputs**. They were not produced here and are not
reproducible without the missing pipeline stages. Treat them as reference data.

⚠ Scientific reading of the case study: MM-PBSA is a more rigorous estimate than a docking
score, but still a computational one. The paper's own claim is measured — mol3 "may be a
potential inhibitor for CDK2" — and its conclusion calls for wet-lab validation. Even the
strongest result here is a **hypothesis**, not a demonstrated inhibitor.

## B.8 Docking prerequisites — none currently met

| Requirement | Status |
|---|---|
| `Jupyter_Dock/` (binaries `smina`, `prepare_receptor`, `prepare_ligand`, `lepro_linux_x86`; `utilities/utils.py`) | ❌ absent |
| `pymol` | ❌ not installed |
| `openbabel` / `pybel` | ❌ not installed |
| `vina` (Python bindings) | ❌ not installed |
| `meeko` | ❌ not installed |
| `prolif` | ❌ not installed |
| `MDAnalysis` | ❌ not installed |
| `py3Dmol` | ❌ not installed |
| Generated molecules | ❌ absent |
| Internet access from a compute node (for `cmd.fetch`) | ⚠ **NOT DETERMINED FROM AVAILABLE SOURCES** — pre-download PDBs from the login node if it is blocked |

`scripts/download_data.sh` clones Jupyter_Dock from
`https://github.com/osvaldo2927/Jupyter_Dock.git`.

⚠ The docking notebook is also **not a batch script**: cells 19, 31 and 40 hardcode
`beam = 0; run = 3; prot_id = 6; pdb_id = '1cqp'`, and cell 8 does `os.chdir(data_path)` with
subsequent cells relying on relative paths. It is an interactive worksheet for one
protein/run, not a pipeline stage. Wrapping it in a loop over the five PDB targets is part of
the work.

⚠ Docking is CPU-bound and embarrassingly parallel. CDGCN §3.4 used **Jupyter Dock** for
exactly this reason: "for parallelizing the heavy computation of the molecular docking for
multiple molecules". At `exhaustiveness 16`, budget on the order of tens of seconds per
ligand per core.
