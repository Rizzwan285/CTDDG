#!/usr/bin/env python3
"""
CTDDG Path Patcher
Replaces all hardcoded absolute paths in the notebooks with the 
correct paths for the current environment.

Usage:
    python scripts/patch_paths.py <PATH_TO_CTDDG_ROOT>

Example on cluster:
    python scripts/patch_paths.py /home/username/CTDDG
"""

import os
import glob
import sys
import json

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
        
    project_root = os.path.abspath(sys.argv[1])
    code_dir = os.path.join(project_root, "code")
    
    if not os.path.isdir(code_dir):
        print(f"Error: Could not find 'code' directory at {code_dir}")
        sys.exit(1)

    replacements = {
        '/workspace/mtp_data/bindingdb': f'{project_root}/data/bindingdb',
        '/workspace/mtp_data/data/atom_types.txt': f'{project_root}/data/atom_types.txt',
        '/workspace/binding_data/atom_types.txt': f'{project_root}/data/atom_types.txt',
        '/workspace/data/atom_types.txt': f'{project_root}/data/atom_types.txt',
        '/home/iit/CDGCN/data/chembl': f'{project_root}/data/chembl',
        '/workspace/Toxicity_experiment/chembl_final.txt': f'{project_root}/data/chembl/chembl_final.txt',
        '/workspace/Toxicity_experiment/March_experiment/just_test': f'{project_root}/outputs/pretrain',
        '/workspace/fpscores.pkl.gz': f'{project_root}/data/fpscores.pkl.gz',
        '/workspace/CTDGD/outputs': f'{project_root}/outputs',
        '/workspace/CTDGD/data': f'{project_root}/data',
        '/workspace/finetune': f'{project_root}/outputs/finetune',
        '/workspace/Jupyter_Dock': f'{project_root}/Jupyter_Dock'
    }

    notebooks = glob.glob(os.path.join(code_dir, '*.ipynb'))
    
    patched_count = 0
    for nb_file in notebooks:
        with open(nb_file, 'r') as f:
            nb_data = f.read()
        
        original_data = nb_data
        for old, new in replacements.items():
            nb_data = nb_data.replace(old, new)
            
        if nb_data != original_data:
            with open(nb_file, 'w') as f:
                f.write(nb_data)
            print(f"Patched paths in: {os.path.basename(nb_file)}")
            patched_count += 1
            
    print(f"\nSuccessfully patched {patched_count} notebooks to use root: {project_root}")

if __name__ == "__main__":
    main()
