#!/usr/bin/env python3
"""
CTDDG Safe Bug Fixes — Patches notebook JSON to fix known bugs.

Fixes applied:
1. Molecular_docking.ipynb: Remove `all_smiles_df_beams.append()` (no-arg crash)
2. Generating_samples.ipynb: Replace `os.system('mkdir ...')` with `os.makedirs(..., exist_ok=True)`

Usage:
    python patch_notebooks.py
"""

import json
import os
import re
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")


def patch_notebook(path, patches):
    """Apply text-level patches to notebook cell sources."""
    with open(path, "r") as f:
        nb = json.load(f)

    patched_count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        new_source = source
        for desc, old, new in patches:
            if old in new_source:
                new_source = new_source.replace(old, new)
                print(f"  ✓ {desc}")
                patched_count += 1
        if new_source != source:
            # Rebuild source as list of lines (notebook JSON format)
            lines = new_source.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]]
            if lines[-1]:  # last line without trailing newline
                cell["source"].append(lines[-1])

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

    return patched_count


def main():
    # ── Fix 1: Molecular_docking.ipynb ───────────────────────────────
    docking_path = os.path.join(CODE_DIR, "Molecular_docking.ipynb")
    if os.path.isfile(docking_path):
        print(f"\nPatching {docking_path}:")
        count = patch_notebook(docking_path, [
            (
                "Remove empty .append() crash bug",
                "    all_smiles_df_beams.append(smiles_df_beams)\n    all_smiles_df_beams.append()\n",
                "    all_smiles_df_beams.append(smiles_df_beams)\n",
            ),
        ])
        print(f"  Applied {count} patch(es)")
    else:
        print(f"SKIP: {docking_path} not found")

    # ── Fix 2: Generating_samples.ipynb ──────────────────────────────
    gen_path = os.path.join(CODE_DIR, "Generating_samples.ipynb")
    if os.path.isfile(gen_path):
        print(f"\nPatching {gen_path}:")
        # Replace fragile os.system('mkdir ...') with os.makedirs
        # The original code has 6 consecutive mkdir lines
        old_mkdir = (
            "    os.system(f'mkdir /workspace/CTDGD/outputs/{model_name}')\n"
            "    os.system(f'mkdir /workspace/CTDGD/outputs/{model_name}/Dataset{dataset_index}')\n"
            "    os.system(f'mkdir /workspace/CTDGD/outputs/{model_name}/Dataset{dataset_index}/model')\n"
            "    os.system(f'mkdir /workspace/CTDGD/outputs/{model_name}/Dataset{dataset_index}/generated_samples')\n"
            "    os.system(f'mkdir /workspace/CTDGD/outputs/{model_name}/Dataset{dataset_index}/generated_samples/{n_samples}')\n"
            "    os.system(f'mkdir /workspace/CTDGD/outputs/{model_name}/Dataset{dataset_index}/generated_samples/{n_samples}/run{run}')\n"
        )
        new_mkdir = (
            "    output_base = f'/workspace/CTDGD/outputs/{model_name}/Dataset{dataset_index}'\n"
            "    os.makedirs(f'{output_base}/model', exist_ok=True)\n"
            "    os.makedirs(f'{output_base}/generated_samples/{n_samples}/run{run}', exist_ok=True)\n"
        )
        count = patch_notebook(gen_path, [
            ("Replace os.system('mkdir') with os.makedirs(exist_ok=True)", old_mkdir, new_mkdir),
        ])
        print(f"  Applied {count} patch(es)")
    else:
        print(f"SKIP: {gen_path} not found")

    print("\nDone. Review the patched notebooks before committing.")


if __name__ == "__main__":
    main()
