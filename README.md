# SemEval 2026 Task 2 - Multimodal Sentiment Analysis

Multimodal sentiment analysis for SemEval 2026 Task 2, combining textual and visual information to predict arousal and valence sentiment dimensions.

## 📁 Project Structure

```
├── paper/                          # LaTeX paper source and published versions
│   ├── main.tex                   # Main paper source
│   ├── references.bib             # Bibliography
│   ├── acl.sty, acl_natbib.bst    # ACL formatting templates
│   ├── figures/                   # All figures and diagrams
│   └── published_papers/          # Published PDF versions
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── clean-semeval-final.ipynb  # Final cleaned dataset notebook
│   ├── molecular_eda.ipynb        # EDA for molecular data
│   └── final_semeval_task2.ipynb  # Main task pipeline
├── src/                           # Python source code
│   ├── molecular_mcc_pipeline.py  # MCC pipeline implementation
│   └── requirements-molecular.txt # Python dependencies
├── data/                          # Dataset releases
│   ├── TRAIN_RELEASE_3SEP2025.zip
│   ├── TEST_RELEASE_5JAN2026.zip
│   └── TEST_LABELS_RELEASE_23FEB2026.zip
├── results/                       # Predictions and scoring results
│   ├── pred_subtask1.csv          # Predictions
│   ├── prediction_result.zip
│   └── scoring_result.zip
├── models/                        # Pretrained model weights (not in repo)
└── archive/                       # Old submissions and archives
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r src/requirements-molecular.txt
```

### 2. Download Model Weights

Model weights are NOT included in the repository due to size constraints. Download them manually:

- `arousal_base_unified.pth` (702 MB)
- `valence_base_final.pth` (701 MB)

Place them in the `models/` directory before running inference.

### 3. Run Notebooks

```bash
jupyter notebook notebooks/
```

## 📝 Paper

The final camera-ready paper is located in `paper/published_papers/SemEval_2026_Task2_Paper.pdf`

## 📊 Data

- **Training**: `data/TRAIN_RELEASE_3SEP2025.zip`
- **Test (unlabeled)**: `data/TEST_RELEASE_5JAN2026.zip`
- **Test (labeled)**: `data/TEST_LABELS_RELEASE_23FEB2026.zip`

## 🔧 Key Files

| File | Purpose |
|------|---------|
| `src/molecular_mcc_pipeline.py` | MCC pipeline for multimodal sentiment prediction |
| `notebooks/final_semeval_task2.ipynb` | Complete task workflow and analysis |
| `paper/main.tex` | LaTeX source for the paper |

## 📦 Model Information

Models are large (1.4 GB total) and stored separately. See `models/README.md` for download instructions.

## 📄 License

This project is part of SemEval 2026 Task 2. Please refer to the original task guidelines for licensing information.

## 👤 Author

GitHub User (user@example.com)

---

For more information, see the individual README files in each subdirectory.
