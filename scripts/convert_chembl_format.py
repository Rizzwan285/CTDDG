#!/usr/bin/env python3
"""
Convert plain-SMILES ChEMBL file to the <SMILES> <class_label> format
expected by pretraining.ipynb.

The original pretraining code does:
    smiles, smiles_class = smiles.split(" ")
    smiles_class = int(smiles_class)
But then NEVER uses smiles_class (tox_class is hardcoded to [1]*k).
So we just append a dummy label " 1" to each line.

Usage:
    python convert_chembl_format.py <input_plain_smiles> <output_labeled_smiles>

Example:
    python convert_chembl_format.py ../data/chembl/chembl.txt ../data/chembl/chembl_final.txt
"""

import sys
import os


def convert(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            smiles = line.strip()
            if smiles:
                fout.write(f"{smiles} 1\n")
                count += 1

    print(f"Converted {count} SMILES from '{input_path}' -> '{output_path}'")
    print(f"Format: <SMILES> 1  (dummy class label appended)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
