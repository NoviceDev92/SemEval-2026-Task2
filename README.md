# 🧠📊 LexMachina at SemEval-2026 Task 2: Longitudinal Affect Assessment

**Composite Correlation: r=0.645 (Valence) | r=0.434 (Arousal)** | **Jadavpur University** | **SemEval 2026 Workshop**

Predicting variation in emotional valence and arousal over time from ecological essays using a bifurcated optimization strategy combining Domain-Adversarial Neural Networks (DANN) and Isotonic Calibration.

---

## 🎯 Problem Statement

Longitudinal affect assessment poses significant challenges in NLP:

1. **User Generalization Gap**: Models trained on seen users often fail dramatically on unseen users due to overfitting to individual user-specific affective patterns
2. **Conservative Bias**: Standard MSE-optimized regression heads produce predictions clustered toward the mean, failing to capture extreme emotional states
3. **Domain Shift**: Transitioning from training to test distributions requires robust feature representations

Our system addresses these challenges through a dual-stream architecture leveraging recent advances in domain adaptation and calibration theory.

---

## 🏗️ System Architecture

Our solution employs a bifurcated optimization strategy:

<div align="center">
<img src="paper/figures/architecture_diagram.png" alt="System Architecture" width="800"/>
</div>

### Arousal Stream: Domain-Adversarial Neural Network (DANN)

To solve the **User Generalization Gap**, we implemented a DANN with a Gradient Reversal Layer (GRL) that forces the model to learn user-invariant affective representations:

$$\mathcal{L}_{total} = \mathcal{L}_{regression} - \lambda \cdot \mathcal{L}_{adversarial}$$

The GRL dynamically schedules adversarial weight, significantly boosting zero-shot performance on unseen users.

### Valence Stream: Isotonic Calibration

To counteract **regression to the mean**, we applied post-hoc isotonic calibration to stretch predictions back to the extreme boundaries of the $[-2,2]$ space:

$$f_{calibrated}(x) = \text{IsotonicRegression}(f_{raw}(x), y_{val})$$

### Data Sanitation: Out-of-Fold Protocol

An automated 3-fold cross-validation pipeline purges approximately 9% of training data containing severe label noise before final model training.

---

## 📊 Official Results

Evaluated using the official SemEval composite correlation metric ($r_c$), which aggregates inter-user traits ($r_b$) and temporal fluctuations ($r_w$):

| System | Valence $r$ | Valence $r_b$ | Valence $r_w$ | Arousal $r$ | Arousal $r_b$ | Arousal $r_w$ |
|--------|-------------|---------------|---------------|------------|---------------|---------------|
| **Overall System** | **0.645** | **0.712** | **0.567** | **0.434** | **0.461** | **0.406** |
| Seen Users | 0.636 | 0.718 | 0.537 | 0.343 | 0.323 | 0.363 |
| Unseen Users | 0.669 | 0.734 | 0.593 | 0.574 | 0.681 | 0.443 |
| Words Only | 0.655 | 0.730 | 0.563 | 0.572 | 0.631 | 0.507 |
| Essay Only | 0.627 | 0.665 | 0.586 | 0.307 | 0.315 | 0.298 |

---

## 🔬 Visualizing the Framework

### 1. DANN Latent Space Visualization

By disabling user identity markers during training, our Gradient Reversal Layer successfully forced the network to learn a user-agnostic affective space. The highly overlapping clusters below prove the effectiveness of adversarial disentanglement:

<div align="center">
<img src="paper/figures/tsne_plot.png" alt="t-SNE Visualization of Arousal Latent Space" width="700"/>
</div>

**Key Finding**: Top 5 users show significant overlap in latent space, indicating successful user-invariant representation learning.

### 2. Isotonic Calibration Error Analysis

While isotonic regression successfully cured conservative bias in the valence stream, our error analysis revealed a "staircase effect" (quantization artifacts) as the continuous variable was mapped to the validation set's step-function distribution:

<div align="center">
<img src="paper/figures/valence_forensic_analysis_highres.png" alt="Valence Forensic Analysis" width="700"/>
</div>

**Insight**: The quantization artifacts are mitigated through ensemble averaging across multiple seeds, explaining the 0.645 correlation score.

### 3. Additional Diagnostic Visualizations

<div align="center">
<img src="paper/figures/Figure1_Forensic.png" alt="Forensic Analysis" width="600"/>
</div>

---

## 📁 Project Structure

```
SemEval-2026-Task2/
├── README.md                              # This file
├── .gitignore                             # Git configuration
├── LICENSE                                # MIT License
│
├── paper/                                 # Academic publication
│   ├── main.tex                          # LaTeX source
│   ├── references.bib                    # Bibliography
│   ├── acl.sty                           # ACL template styles
│   ├── acl_natbib.bst                    # Bibliography style
│   ├── figures/                          # Visualizations (14 images)
│   │   ├── architecture_diagram.png      # System architecture
│   │   ├── tsne_plot.png                 # DANN latent space
│   │   ├── valence_forensic_analysis_highres.png
│   │   ├── Figure1_Forensic.png
│   │   └── [10 additional analysis figures]
│   └── published_papers/                 # Final versions
│       ├── SemEval_2026_Task2_Paper.pdf
│       ├── SemEval_2026_Task2_finalpaper.pdf
│       └── SemEval_2026_Task2_CameraReady.pdf
│
├── notebooks/                            # Jupyter notebooks
│   ├── final_semeval_task2.ipynb         # 🎯 Main pipeline (START HERE)
│   ├── clean-semeval-final.ipynb         # Data cleaning & EDA
│   └── clean-semeval.ipynb               # Initial exploration
│
├── src/                                  # Python source code
│   ├── molecular_mcc_pipeline.py         # Core MCC implementation
│   └── requirements-molecular.txt        # Dependencies
│
├── data/                                 # Official SemEval datasets
│   ├── TRAIN_RELEASE_3SEP2025.zip       # Training data
│   ├── TEST_RELEASE_5JAN2026.zip        # Test data (unlabeled)
│   ├── TEST_LABELS_RELEASE_23FEB2026.zip # Test labels
│   └── README.md                         # Dataset documentation
│
├── results/                              # Predictions & metrics
│   ├── pred_subtask1.csv                # Model predictions
│   ├── prediction_result.zip
│   ├── scoring_result.zip
│   └── README.md                         # Results documentation
│
├── models/                               # Pretrained weights (not tracked)
│   ├── .gitkeep
│   ├── arousal_base_unified.pth         # 702 MB (ignored)
│   ├── valence_base_final.pth           # 701 MB (ignored)
│   └── README.md                         # Download instructions
│
└── archive/                              # Legacy submissions
    ├── old_submissions/
    │   ├── submission.zip
    │   └── submission_v3_fixed.zip
    ├── SemEval2026_Task2_CameraReady_LaTeX_Source.zip
    └── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with ≥12GB VRAM (T4 or better)
- **CUDA**: 12.x
- **PyTorch**: Compatible with your CUDA version

### Installation

```bash
# Clone repository
git clone https://github.com/NoviceDev92/SemEval-2026-Task2.git
cd SemEval-2026-Task2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm
# Or use: pip install -r src/requirements-molecular.txt
```

### Data Setup

1. **Get Official Datasets**
   - Register at the [SemEval 2026 Task 2 official website](https://semeval.github.io/SemEval2026/)
   - Download the three dataset ZIPs
   - Extract to `data/` directory

2. **Download Model Weights**
   - `arousal_base_unified.pth` (702 MB)
   - `valence_base_final.pth` (701 MB)
   - Place in `models/` directory

### Running the Pipeline

```bash
# Option 1: Run complete notebook (Recommended)
jupyter notebook notebooks/final_semeval_task2.ipynb

# Option 2: Run from command line
python src/molecular_mcc_pipeline.py --config configs/default.yaml

# Option 3: Exploratory analysis
jupyter notebook notebooks/clean-semeval-final.ipynb
```

The notebook will automatically:
- Load and preprocess data
- Train models across 3 random seeds
- Generate ensemble predictions
- Produce analytical visualizations

---

## 📚 Citation

If you use our code, the Out-of-Fold sanitation protocol, or build upon our Instance-Weighted Domain Adaptation theories, please cite our paper:

```bibtex
@inproceedings{ganguli2026lexmachina,
   title={LexMachina at SemEval-2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays},
   author={Ganguli, Somdev and Dutta, Vibhan and Datta, Romit and Barman, Amit and Naskar, Sudip Kr},
   booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
   pages={XX--XX},
   year={2026},
   organization={Association for Computational Linguistics}
}
```

---

## 📖 References

Key papers and resources cited in this work:

- Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation". ICML.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks". ICML.
- SemEval-2026 Task 2: Longitudinal Affect Assessment. https://semeval.github.io/SemEval2026/

---

## ⚙️ Technical Details

### Key Implementation Details

| Component | Details |
|-----------|---------|
| **Base Model** | DeBERTa-v3-base |
| **Optimization** | AdamW with warmup |
| **Training Strategy** | 3-seed ensemble with cross-validation |
| **Data Sanitation** | Out-of-Fold (OOF) protocol removing ~9% noisy samples |
| **Hardware** | NVIDIA Tesla T4 or equivalent |
| **Training Time** | ~4-6 hours per seed |

### Model Architecture Details

**Arousal Stream**:
- DeBERTa embeddings → DANN head with GRL
- Gradient reversal coefficient: dynamically scheduled
- User adversarial loss: Binary cross-entropy

**Valence Stream**:
- DeBERTa embeddings → Regression head → Isotonic calibration
- Calibration curve learned on validation fold
- Final predictions: Ensemble average across 3 seeds

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 👥 Authors

**Team LexMachina** - Jadavpur University & University of Glasgow
- Somdev Ganguli
- Vibhan Dutta
- Romit Datta
- Amit Barman
- Sudip Kr Naskar

---

## 🤝 Contributing

This is an academic research project. For bug reports or feature requests, please open an issue on GitHub.

---

## ❓ FAQ

**Q: Why are model weights not in the repo?**
A: Model weights exceed GitHub's large file limit (1.4 GB total). See `models/README.md` for download instructions.

**Q: Can I reproduce the exact results?**
A: Yes! Set random seeds to the values in the paper and follow the Quick Start guide. Some variance is expected due to hardware differences.

**Q: What GPU do I need?**
A: Minimum 12GB VRAM. The code works on T4 (16GB), but faster GPUs will reduce training time.

**Q: How do I use this for my own data?**
A: Adapt the preprocessing in `notebooks/clean-semeval-final.ipynb` to match your data format, then use the training pipeline in `notebooks/final_semeval_task2.ipynb`.

---

**Last Updated**: May 12, 2026 | **Status**: Published at SemEval 2026

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
