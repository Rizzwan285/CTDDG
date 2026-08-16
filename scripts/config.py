"""
CTDDG Pipeline — Centralized Path Configuration

All hardcoded paths from the 5 original notebooks are unified here.
Set PROJECT_ROOT to your local working directory, and all other paths
will resolve automatically.

Usage:
    from config import cfg
    chembl_path = cfg.CHEMBL_FILE
"""

import os

# ─── SET THIS TO YOUR LOCAL WORKING DIRECTORY ───────────────────────
# This replaces /workspace/mtp_data/, /workspace/CTDGD/, /home/iit/CDGCN/, etc.
PROJECT_ROOT = os.environ.get(
    "CTDDG_ROOT",
    "/workspace"  # Default; override via env var or edit this line
)


class _Config:
    """Singleton config object that lazily builds all paths from PROJECT_ROOT."""

    def __init__(self, root):
        self.ROOT = root

        # ── Raw / preprocessed data ──────────────────────────────────
        self.DATA_DIR = os.path.join(root, "data")
        self.BINDINGDB_DIR = os.path.join(self.DATA_DIR, "bindingdb")
        self.CHEMBL_DIR = os.path.join(self.DATA_DIR, "chembl")

        # Key data files
        self.ATOM_TYPES_FILE = os.path.join(self.DATA_DIR, "atom_types.txt")
        self.CHEMBL_PLAIN_FILE = os.path.join(self.CHEMBL_DIR, "chembl.txt")
        self.CHEMBL_LABELED_FILE = os.path.join(self.CHEMBL_DIR, "chembl_final.txt")
        self.FPSCORES_FILE = os.path.join(self.DATA_DIR, "fpscores.pkl.gz")

        # ── Training outputs ─────────────────────────────────────────
        self.OUTPUTS_DIR = os.path.join(root, "outputs")

        # Pretraining
        self.PRETRAIN_DIR = os.path.join(self.OUTPUTS_DIR, "pretrain")
        self.PRETRAIN_CKPT_DIR = os.path.join(self.PRETRAIN_DIR, "logs")

        # Fine-tuning (per-dataset)
        # Use: cfg.finetune_model_dir(model_name, dataset_index)
        # Use: cfg.finetune_samples_dir(model_name, dataset_index, n_samples, run)

        # ── Docking ──────────────────────────────────────────────────
        self.DOCKING_DIR = os.path.join(self.OUTPUTS_DIR, "docking")

    # ── Helper methods for per-dataset paths ─────────────────────────

    def train_dataset_dir(self, dataset_id):
        """e.g. data/bindingdb/train_dataset/train_4_org_1042_104/"""
        mapping = {1: "train_4_org_1042_104", 2: "train_4_org_1000_112",
                   3: "train_4_org_1004_122", 4: "train_4_org_1002_103",
                   5: "train_4_org_1036_124"}
        return os.path.join(self.BINDINGDB_DIR, "train_dataset", mapping[dataset_id])

    def test_dataset_dir(self, dataset_id):
        mapping = {1: "test_4_org_1042_104", 2: "test_4_org_1000_112",
                   3: "test_4_org_1004_122", 4: "test_4_org_1002_103",
                   5: "test_4_org_1036_124"}
        return os.path.join(self.BINDINGDB_DIR, "test_dataset", mapping[dataset_id])

    def finetune_model_dir(self, model_name, dataset_index):
        return os.path.join(self.OUTPUTS_DIR, model_name,
                            f"Dataset{dataset_index}", "model")

    def finetune_samples_dir(self, model_name, dataset_index, n_samples, run):
        return os.path.join(self.OUTPUTS_DIR, model_name,
                            f"Dataset{dataset_index}", "generated_samples",
                            str(n_samples), f"run{run}")

    def protein_embedding_file(self, dataset_index, run):
        return os.path.join(self.DATA_DIR,
                            f"d{dataset_index}_te_embeddings_run{run}.txt")

    def ensure_dirs(self):
        """Create all base directories if they don't exist."""
        for d in [self.DATA_DIR, self.BINDINGDB_DIR, self.CHEMBL_DIR,
                  self.OUTPUTS_DIR, self.PRETRAIN_DIR, self.PRETRAIN_CKPT_DIR,
                  self.DOCKING_DIR]:
            os.makedirs(d, exist_ok=True)


cfg = _Config(PROJECT_ROOT)
