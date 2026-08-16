#!/bin/bash
# CTDDG Dataset & Dependencies Downloader
# Run this script on the Bhavani cluster to fetch all datasets and external tools.

echo "1. Installing 'gdown' to download from Google Drive..."
pip install gdown

echo "2. Setting up data directories..."
mkdir -p data/bindingdb data/chembl outputs/pretrain/logs outputs/docking

echo "3. Downloading Datasets from Google Drive (This may take a while)..."
# Download the main datasets folder provided by the authors
gdown --folder https://drive.google.com/drive/folders/1KniQd6NFCiNNE6X1PvCrpy4McDicBXp -O data/raw_gdrive

echo "4. Moving files to correct directories..."
# Note: Adjust these move commands depending on the exact folder structure inside the Google Drive zip
mv data/raw_gdrive/ChEMBL/* data/chembl/ 2>/dev/null || echo "Please move ChEMBL files manually from data/raw_gdrive to data/chembl/"
mv data/raw_gdrive/BindingDB/* data/bindingdb/ 2>/dev/null || echo "Please move BindingDB files manually from data/raw_gdrive to data/bindingdb/"

echo "5. Downloading SAS Scoring file (fpscores.pkl.gz) from RDKit..."
wget -O data/fpscores.pkl.gz https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz

echo "6. Downloading Jupyter Dock for Molecular Docking..."
git clone https://github.com/osvaldo2927/Jupyter_Dock.git

echo "Download complete! Please verify the files in the 'data/' directory."
