# .claude/ — Persistent Project Context

These files exist so a new Claude Code session in this repository does not have to be
re-briefed. They are **compact context**, not documentation. Full human documentation lives in
`PROJECT_DOCS/`.

## Before doing significant work, read in this order

1. `PROJECT_CONTEXT.md` — what the project is, model, datasets, terminology, constraints
2. `CURRENT_STATE.md` — what is done, what is blocked, working commands, key paths
3. `ARCHITECTURE.md` — classes, tensor flow, checkpoint flow
4. `WORKFLOW.md` — the verified stage-by-stage pipeline
5. `KNOWN_ISSUES.md` — bugs, hardcoded paths, paper/code discrepancies **← check before "fixing" anything**
6. The relevant `PROJECT_DOCS/NN_*.md` for the area you are touching
7. `papers/CTDDG.pdf` / `papers/CDGCN.pdf` — **only** when scientific context is genuinely needed

## Source-of-truth priority

```
1. Actual current code                    ← highest
2. Actual experiment outputs / logs
3. Research papers (papers/*.pdf)          ← for intended methodology
4. PROJECT_DOCS/
5. .claude/                                ← lowest; a cache, not truth
```

**If `.claude/` contradicts the current code, trust the code and update `.claude/`.**

## Rules

- **Scientific methodology question** → consult the actual paper in `papers/`.
- **Implementation question** → read the actual code in `code/`.
- **Paper and code disagree** → do **not** silently pick one. Report the discrepancy.
  Several are already catalogued in `KNOWN_ISSUES.md`.
- **Project-state question** → check `CURRENT_STATE.md` **and** verify against the repository.
  Never answer from memory alone.
- **Do not modify the research implementation** (`code/`, hyperparameters, datasets) unless the
  user explicitly asks. Document concerns in `KNOWN_ISSUES.md` instead.
- **Never re-run `code/pretraining.ipynb`** without an explicit backup — it auto-resumes and
  overwrites a completed 47-hour checkpoint.
- **Do not regenerate `data/atom_types.txt`** — atom-type indices are line numbers and the
  trained checkpoint depends on the current ordering.

## Reading the PDFs

No `pdftotext`, `poppler`, or Python PDF library is installed on this cluster. A working
pure-Python extractor was written during the 2026-09-08 audit; if you need paper text again,
either re-create one (scan `stream…endstream`, `zlib.decompress`, parse `Tj`/`TJ` with
`ToUnicode` CMaps) or use the `Read` tool if PDF rendering has since become available.
⚠ Ligature characters (fi, fl, ffi) are dropped by naive extraction — "specic" means "specific".

## Maintenance

- Update `CURRENT_STATE.md` whenever a stage completes, a checkpoint changes, or a blocker clears.
- Append to `SESSION_NOTES.md` at the end of a meaningful session.
- Add to `DECISIONS.md` only for decisions actually made — never invent them.
- Add to `KNOWN_ISSUES.md` whenever a new defect or discrepancy is found.
