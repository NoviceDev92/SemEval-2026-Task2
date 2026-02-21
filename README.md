# LexMachina at SemEval-2026 Task 2 🧠📝
**Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)

This repository contains the official code and models for the **LexMachina** system, submitted to **SemEval-2026 Task 2: Longitudinal Affect Assessment (Subtask 1)** by the team from Jadavpur University. 

Our system achieved competitive composite correlation scores of **$r = 0.645$ for Valence** and **$r = 0.434$ for Arousal** by utilizing a bifurcated optimization strategy to handle domain shift and conservative prediction biases.

---

## 🏗️ System Architecture

Our dual-stream architecture leverages **DeBERTa-v3-base** as the core linguistic feature extractor, followed by two distinct pipelines tailored to the unique challenges of Valence and Arousal prediction:

1. **Arousal Stream (Domain-Adversarial Neural Network - DANN):**
   To solve the "User Generalization Gap" (where models overfit to seen users and fail on strangers), we implemented a DANN. A Gradient Reversal Layer (GRL) forces the model to learn *user-invariant* affective representations, significantly boosting zero-shot performance on unseen users.
2. **Valence Stream (Isotonic Calibration):**
   To counteract the "regression to the mean" phenomenon inherent in standard MSE-optimized regression heads, we applied Post-Hoc Isotonic Calibration to stretch predictions back to the extreme boundaries of the $[-2, 2]$ space.
3. **Out-of-Fold (OOF) Data Sanitation:**
   An automated 3-fold cross-validation pipeline purges approximately 9% of the training data containing severe label noise/semantic inversions prior to final model training.

*(Note: Upload your `architecture_diagram.png` to the repository to display it here).*
---

## 📊 Official Results

Our system was evaluated using the official SemEval composite correlation metric ($r_c$), which aggregates inter-user traits ($r_b$) and temporal fluctuations ($r_w$).

| Evaluation Slice | Valence ($r_c$) | Valence ($r_b$) | Valence ($r_w$) | Arousal ($r_c$) | Arousal ($r_b$) | Arousal ($r_w$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall System** | **0.645** | **0.712** | **0.567** | **0.434** | **0.461** | **0.406** |
| Seen Users | 0.636 | 0.718 | 0.537 | 0.343 | 0.323 | 0.363 |
| Unseen Users | 0.669 | 0.734 | 0.593 | 0.574 | 0.681 | 0.443 |
| Words Only | 0.655 | 0.730 | 0.563 | 0.572 | 0.631 | 0.507 |
| Essay Only | 0.627 | 0.665 | 0.586 | 0.307 | 0.315 | 0.298 |

---

## 📂 Repository Structure

* `clean-semeval-final.ipynb`: The primary Jupyter Notebook containing the entire pipeline end-to-end. This includes:
  * OOF Data Sanitation Protocol.
  * Model class definitions (DANN and Vanilla regressors).
  * Training loops with dynamic GRL scheduling.
  * 3-Seed Ensemble generation.
  * t-SNE latent space extraction and plotting.
* `architecture_diagram.png`: Visual representation of our dual-stream system.
* `tsne_plot.png`: t-SNE projection proving the DANN successfully stripped user identity markers.
* `valence_forensic_analysis.png`: Error analysis illustrating Isotonic Calibration quantization artifacts.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.10+ installed. The required libraries can be installed via:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm
