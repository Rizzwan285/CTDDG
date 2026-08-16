# CTDDG: Conditional Target-based Detoxed Drug Generation

This repository contains the code for reproducing the single-target, single-property baseline of the CTDDG pipeline using Apache MXNet.

## Project Structure

```
CTDDG/
├── code/                   # Jupyter Notebooks for pipeline execution
├── scripts/                # Utility scripts (path patching, downloading, fine-tuning)
├── data/                   # MUST BE CREATED: Contains the datasets (ignored by git)
├── outputs/                # MUST BE CREATED: Model checkpoints & results (ignored by git)
├── requirements.txt        # Conda/pip dependencies for MXNet GPU
├── README.md               # This file
└── .gitignore              
```

## Dataset Setup (`data/` folder)

**Note:** Due to their massive size, the datasets are **not included** in this repository. 
You must acquire them and place them in the `data/` directory.

### Method 1: Using your existing local `data/` folder (Recommended for Clusters)
If you already have the datasets downloaded on your local machine in the `data/` folder (e.g., `/home/username/Documents/BTP/CTDDG/data`), you can transfer this folder directly to your cluster workspace using `rsync` or `scp`.

**Command to transfer from local to the Bhavani cluster:**
```bash
rsync -avzP /home/username/Documents/BTP/CTDDG/data/ your_username@bhavani.iitpkd.ac.in:/path/to/your/cluster/CTDDG/
```

### Method 2: Automated Download Script
If you are starting from a fresh environment, run the provided download script to fetch the Google Drive datasets, `fpscores.pkl.gz`, and clone the `Jupyter_Dock` repository automatically:
```bash
conda activate ctddg_env
./scripts/download_data.sh
```

**The final structure of your `data/` folder must look exactly like this:**
```
CTDDG/data/
├── chembl/
│   ├── chembl.txt          # Plain SMILES from preprocessing
│   └── chembl_final.txt    # Labeled SMILES from `convert_chembl_format.py`
├── bindingdb/
│   ├── train_dataset/      # Paired ligand and protein embeddings for training
│   └── test_dataset/       # Unseen targets for generation
└── fpscores.pkl.gz         # Used for SAS calculation in Evaluation Metrics
```

## Execution Instructions (Bhavani Cluster)

1. **Pull the Codebase:**
   Clone or pull this repository to your cluster node.
   
2. **Build the Environment (Master Node):**
   ```bash
   conda create -n ctddg_env python=3.9 -y
   conda activate ctddg_env
   conda install -c bioconda emboss -y
   pip install mxnet-cu112==1.9.1
   pip install -r requirements.txt
   pip install bio-embeddings[all]
   ```

3. **Get the Data:**
   Use one of the two Dataset Setup methods above to populate your `data/` folder. Ensure the folders exist and are populated.

4. **Convert ChEMBL Format:**
   ```bash
   python scripts/convert_chembl_format.py data/chembl/chembl.txt data/chembl/chembl_final.txt
   ```

5. **Patch the Notebook Paths (CRITICAL):**
   Run the following script to overwrite the hardcoded paths in all Jupyter notebooks to point to your cluster's exact directory:
   ```bash
   python scripts/patch_paths.py $(pwd)
   ```

6. **Submit Slurm Job (A30 Node):**
   You can now run the notebooks headless via `jupyter nbconvert`. Ensure you run them in order: Preprocessing -> Pretraining -> Fine-Tuning (using `scripts/finetune_cell.py`) -> Sample Generation -> Evaluation -> Docking.
