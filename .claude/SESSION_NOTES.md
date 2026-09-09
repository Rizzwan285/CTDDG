# SESSION_NOTES

Append newest at the top. Keep entries short.

---

## 2026-09-08 (later) — Checkpoint protection added

**Question answered:** pretraining used **CDGCN-style** data/objective, not toxicity-labelled.
Re-verified: `chembl_final.txt` is 1,090,529 × label `1`; `tox_class = [1]*k` hardcoded;
`tox_class_batch` occurs at exactly 3 lines (1150 signature, 1206 unpack, 1211 call) and never
in the body of `_likelihood`; both `forward()` methods return `-l.mean()`.
**Conclusion: the 480k checkpoint does NOT need re-running for the baseline** — it is a valid
CDGCN pretrained generator. It only needs re-running if/when the toxicity loss is implemented,
and that should go to a NEW run name.

**Changed (implementation — first time this session):**
- `code/pretraining.ipynb`: added `import shutil` and a checkpoint-protection block in the
  config section (`CTDDG_RUN_NAME`, refuse-to-overwrite guard, pre-resume snapshot).
  Diff vs original: 2 lines removed, 84 added. AST-parses. No training logic touched.
- Archived the finished run (write-protected, md5-verified) to
  `outputs/pretrain/archive/base_cdgcn_step480000_FINISHED_20260908-235622/`.
- Updated `.claude/{KNOWN_ISSUES,CURRENT_STATE,DECISIONS}.md` and `PROJECT_DOCS/{06,11}`.

**Tested:** all four guard paths in a sandbox (finished→refuse, FORCE→snapshot+proceed,
new RUN_NAME→fresh dir, unfinished→snapshot+resume). Confirmed `patch_paths.py` remains a
no-op on the edited notebook. No GPU code was run.

---

## 2026-09-08 — Full audit + documentation pass

**Investigated**
- Both papers, read end-to-end. No PDF tooling exists on this cluster (no poppler, no
  pypdf/fitz/pdfminer), so a pure-Python extractor was written in the scratchpad
  (scan `stream…endstream` → `zlib.decompress` → parse `Tj`/`TJ` with `ToUnicode` CMaps).
  ⚠ Ligatures (fi/fl/ffi) are dropped — "specic" = "specific".
- All 6 notebooks in `code/`, all 7 in `notebooks/`, all of `scripts/` and `cluster/`.
- `data/data_preprocessing.ipynb` — the **upstream** CDGCN/DGGNP notebook, which still has its
  original executed outputs. Best source of real dataset statistics.
- Logs, checkpoints, `ctddg_env` package versions, `bio_embeddings` internals.

**Key findings**
1. 🔴 **The toxicity loss is not implemented.** All 1,090,529 ChEMBL labels are `1`;
   `process_single` hardcodes `tox_class=[1]*k`; `_likelihood` ignores `tox_class_batch`.
   **The 480,000-iteration checkpoint is a CDGCN model, not CTDDG.**
2. 🔴 **Protein embeddings are the blocker.** `give_bioembeddings` omits `reduce_per_protein`
   → `(L,1024)` per-residue instead of `(1024,)`; the writer `str()`s a numpy array.
   Upstream CDGCN used **ProSE/Bepler avg-pooled, 6165-d** (confirmed by cell 79 output);
   CTDDG switched to **ProtBert-BFD 1024-d** in the authors' own commit `47d37e3`.
3. 🔴 **`notebooks/*.ipynb` are structurally broken** — `generate_notebooks.py` does
   `src.split("\n")` without re-adding newlines → every multi-line cell collapses to one line.
   Verified with `ast.parse` across all 7. **They have never run.**
4. 🔴 **`scripts/finetune_cell.py` is non-functional** — 5 runtime blockers + 5 architectural
   divergences from the real `CVanillaMolGen_RNN`. Notably it replaces the `ids`-based gather
   with `nd.repeat`, which is wrong for variable-size molecules.
   The correct architecture **already exists** in `Generating_samples.ipynb` cell 14.
5. ✅ **Pretraining verified complete** — 480,000/480,000, "Training finished", 2799.06
   **minutes** (⚠ header says `time(h)`) ≈ 46.65 h, 6,986,506 params, node003, 1 × A30.
6. ✅ **Architecture reading validated** — hand-summing every layer's parameters gives exactly
   **6,986,506**, matching the run's own printout. The GRU is 79% of the model.
7. ✅ **Needle alignments complete** — 5,809,835 files; counts match the expected per-fold
   products exactly. Only the histogram cell fails (`TypeError`, an original-code bug).
8. Real dataset statistics recovered from the upstream notebook's stored outputs: fold 1 =
   227,093 train / 11,049 test pairs; 154,899 unique BindingDB ligands; atom types 49 ∪ 45 = **65**
   (matches the paper).
9. Paper internal contradiction: 166,400 vs 66,400 BindingDB pairs; neither matches any fold.
10. `~/datasets/downsampled_CrossDocked2020_v1.3/` is now **extracted** (2,801 dirs) — the
    Week-1 blocker is resolved.

**Changed**
- Created `PROJECT_DOCS/` (README + 12 numbered documents).
- Created `.claude/` (README, PROJECT_CONTEXT, CURRENT_STATE, ARCHITECTURE, WORKFLOW,
  DECISIONS, KNOWN_ISSUES, SESSION_NOTES).
- **Nothing in `code/`, `notebooks/`, `scripts/` or `data/` was modified.**

**Tested**
- Verified `ctddg_env` package versions by direct import.
- Verified `ProtTransBertBFDEmbedder.embedding_dimension == 1024` and that
  `reduce_per_protein` is `embedding.mean(axis=0)`.
- Verified label distribution in `chembl_final.txt` (`awk`/`uniq -c`).
- Verified notebook cell integrity with `ast.parse`.
- Verified the LR schedule arithmetic against `log.out` at steps 500 and 480,000.

**Did not test**
- Any GPU execution. Nothing was run on a compute node this session.
- Sampling from the pretrained checkpoint (**still never validated** — highest-value next step).

**Remaining**
0. Back up `outputs/pretrain/logs/` before anything touches it.
1. Write a `Vanilla_RNN_Builder` and validate the checkpoint by sampling (~0.5 day, zero risk).
2. Fix `give_bioembeddings` + the float writer; run embedding cells for fold 1 only.
3. Build `d1_tr_cdgcn.txt` (strip the injected SMILES spaces).
4. Write a correct conditional fine-tuning stage reusing `CVanillaMolGen_RNN`.
5. Generate → evaluate → (optionally) dock.
Then BTP work — see `PROJECT_DOCS/12_BTP_Modification_Guide.md`; §1 (toxicity loss) is the
highest-value and can be prototyped independently of the blocking chain.
