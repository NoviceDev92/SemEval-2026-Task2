"""Patch notebooks/final_semeval_task2.ipynb for local/GitHub reproduction."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "final_semeval_task2.ipynb"

CELL0_REPRO = """# Reproducibility — run this cell first (before any training or inference).
from __future__ import annotations
from pathlib import Path
import os
import random

import numpy as np
import torch


def _repo_root() -> Path:
    p = Path.cwd().resolve()
    if (p / "configs" / "default.yaml").is_file():
        return p
    if p.name == "notebooks" and (p.parent / "configs" / "default.yaml").is_file():
        return p.parent
    return p


REPO_ROOT = Path(os.environ.get("SEMEVAL_REPO_ROOT", _repo_root()))
DATA_DIR = Path(os.environ.get("SEMEVAL_DATA_DIR", REPO_ROOT / "data"))
TRAIN_CSV = str(Path(os.environ.get("SEMEVAL_TRAIN_CSV", DATA_DIR / "train_subtask1.csv")))
TEST_CSV = str(Path(os.environ.get("SEMEVAL_TEST_CSV", DATA_DIR / "test_subtask1.csv")))

WORKDIR = Path(os.environ.get("SEMEVAL_WORKDIR", REPO_ROOT / "results" / "notebook_workspace"))
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED = int(os.environ.get("SEMEVAL_SEED", "42"))


def set_seed(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RNG_SEED)

MODEL_DIR = Path(os.environ.get("SEMEVAL_MODEL_DIR", REPO_ROOT / "models"))


def artifact_path(name: str) -> str:
    w = WORKDIR / name
    if w.is_file():
        return str(w)
    m = MODEL_DIR / name
    if m.is_file():
        return str(m)
    return str(w)


PATH_PRED_CSV = str(REPO_ROOT / "results" / "pred_subtask1.csv")
PATH_PRED_ZIP = str(REPO_ROOT / "results" / "submission_notebook.zip")

Path(REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)

print("REPO_ROOT", REPO_ROOT)
print("DATA_DIR", DATA_DIR)
print("WORKDIR (cwd)", WORKDIR)
print("TRAIN_CSV", TRAIN_CSV)
print("TEST_CSV", TEST_CSV)
print("TORCH_DEVICE", TORCH_DEVICE)
print("RNG_SEED", RNG_SEED)
"""

CELL_LOAD_DATA = """import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================
# Uses TRAIN_CSV / TEST_CSV from the reproducibility cell.

train_df = pd.read_csv(TRAIN_CSV)
print(f"Loaded Train: {len(train_df)} rows")

test_df = pd.read_csv(TEST_CSV)
print(f"Loaded Test:  {len(test_df)} rows")

# ==========================================
# 2. CALCULATE OVERLAP (SEEN vs UNSEEN)
# ==========================================
train_users = set(train_df['user_id'].unique())
test_users = set(test_df['user_id'].unique())

seen_users = test_users.intersection(train_users)
unseen_users = test_users - train_users

test_df['user_type'] = test_df['user_id'].apply(lambda u: 'Seen' if u in seen_users else 'Unseen')
text_counts = test_df['user_type'].value_counts()

# ==========================================
# 3. PRINT REPORT
# ==========================================
print("\\n" + "="*50)
print("FINAL SPLIT STATISTICS (Generalization Gap)")
print("="*50)

n_total_users = len(test_users)
n_seen_users = len(seen_users)
n_unseen_users = len(unseen_users)

print(f"Total Users in Test: {n_total_users}")
print(f"  Seen Users:    {n_seen_users} ({n_seen_users/n_total_users*100:.1f}%)")
print(f"  Unseen Users:  {n_unseen_users} ({n_unseen_users/n_total_users*100:.1f}%)")

n_total_texts = len(test_df)
n_seen_texts = text_counts.get('Seen', 0)
n_unseen_texts = text_counts.get('Unseen', 0)

print(f"\\nTotal Texts in Test: {n_total_texts}")
print(f"  Seen Texts:    {n_seen_texts} ({n_seen_texts/n_total_texts*100:.1f}%)")
print(f"  Unseen Texts:  {n_unseen_texts} ({n_unseen_texts/n_total_texts*100:.1f}%)")
print("="*50)
"""


def cell_source(code: str) -> list[str]:
    code = code.rstrip() + "\n"
    return [code]


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb.get("metadata", {}).pop("kaggle", None)

    cells = nb["cells"]
    first_src = "".join(cells[0].get("source", []))
    already = "SEMEVAL_REPO_ROOT" in first_src and "artifact_path" in first_src
    if not already:
        cells.insert(
            0,
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": cell_source(CELL0_REPRO),
            },
        )

    cells[0]["source"] = cell_source(CELL0_REPRO)
    cells[1]["source"] = cell_source(CELL_LOAD_DATA)

    def src(i: int) -> str:
        return "".join(cells[i].get("source", []))

    # [0]=repro [1]=load [2]=sanitation [3]=stats [4]=gpu [5]=arousal [6]=valence [7]=tsne [8]=submit [9]=forensic [10]=ensemble

    # sanitation
    s = src(2)
    s = s.replace(
        'TRAIN_PATH = "/kaggle/input/samuval/TRAIN_RELEASE_3SEP2025/train_subtask1.csv"',
        "TRAIN_PATH = TRAIN_CSV  # reproducibility cell",
    )
    cells[2]["source"] = cell_source(s)

    # EDA stats
    s = src(3)
    s = s.replace(
        'train = pd.read_csv("/kaggle/input/samuval/TRAIN_RELEASE_3SEP2025/train_subtask1.csv")',
        "train = pd.read_csv(TRAIN_CSV)",
    )
    s = s.replace(
        'test = pd.read_csv("/kaggle/input/samevaltest/test_subtask1.csv") # Or whatever your test file is',
        "test = pd.read_csv(TEST_CSV)",
    )
    cells[3]["source"] = cell_source(s)

    # valence ValConfig
    s = src(6)
    s = s.replace('DEVICE = "cuda"', "DEVICE = TORCH_DEVICE")
    cells[6]["source"] = cell_source(s)

    # submission
    s = src(8)
    s = s.replace(
        "# --- CONFIG ---\n"
        'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
        "BATCH_SIZE = 32\n"
        'TEST_PATH = "/kaggle/input/samevaltest/test_subtask1.csv"\n'
        'OUTPUT_CSV = "pred_subtask1.csv"  # <--- REQUIRED FILENAME\n'
        'OUTPUT_ZIP = "submission.zip"\n',
        "# --- CONFIG ---\n"
        "DEVICE = TORCH_DEVICE\n"
        "BATCH_SIZE = 32\n"
        "TEST_PATH = TEST_CSV\n"
        "OUTPUT_CSV = PATH_PRED_CSV\n"
        "OUTPUT_ZIP = PATH_PRED_ZIP\n",
    )
    if "from pathlib import Path" not in s:
        s = s.replace("import zipfile\n", "import zipfile\nfrom pathlib import Path\n")
    s = s.replace(
        'state = torch.load("arousal_base_unified.pth", map_location=DEVICE)',
        "state = torch.load(artifact_path('arousal_base_unified.pth'), map_location=DEVICE)",
    )
    s = s.replace(
        'model_val.load_state_dict(torch.load("valence_base_final.pth", map_location=DEVICE))',
        "model_val.load_state_dict(torch.load(artifact_path('valence_base_final.pth'), map_location=DEVICE))",
    )
    s = s.replace('with open("valence_iso.pkl", "rb") as f:', "with open(artifact_path('valence_iso.pkl'), 'rb') as f:")
    s = s.replace(
        "with zipfile.ZipFile(OUTPUT_ZIP, 'w') as z:\n    z.write(OUTPUT_CSV)\n",
        "with zipfile.ZipFile(OUTPUT_ZIP, 'w') as z:\n    z.write(OUTPUT_CSV, arcname=Path(OUTPUT_CSV).name)\n",
    )
    cells[8]["source"] = cell_source(s)

    # forensic
    s = src(9)
    s = s.replace(
        '    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
        "    DEVICE = TORCH_DEVICE\n",
    )
    cells[9]["source"] = cell_source(s)

    # ensemble
    s = src(10)
    s = s.replace('DEVICE = "cuda"', "DEVICE = TORCH_DEVICE")
    cells[10]["source"] = cell_source(s)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
