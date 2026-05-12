# Model Weights

Due to Git repository size constraints, pretrained model weights are **not stored in this repository**.

## 📥 Download Instructions

### Required Models

1. **arousal_base_unified.pth** (702 MB)
   - Pretrained model for arousal sentiment dimension
   - Used for: Arousal prediction in final_semeval_task2.ipynb

2. **valence_base_final.pth** (701 MB)
   - Pretrained model for valence sentiment dimension
   - Used for: Valence prediction in final_semeval_task2.ipynb

3. **valence_iso.pkl** (small)
   - Isotonic calibration fitted during training; required for CLI inference and the submission notebook cell
   - Place in this directory next to the `.pth` files (or update `configs/default.yaml`)

### Setup

1. Download both `.pth` files and `valence_iso.pkl` from your release artifact or reproduce via the training notebook
2. Place them in this directory (`models/`)
3. Point `configs/default.yaml` and notebook paths at this folder if you use a different layout

### Usage in Code

```python
# Checkpoints are state_dicts — load into the model classes used in
# notebooks/final_semeval_task2.ipynb or src/molecular_mcc_pipeline.py
import torch
from pathlib import Path

state = torch.load(Path("models/arousal_base_unified.pth"), map_location="cpu", weights_only=True)
# model.load_state_dict({k: v for k, v in state.items() if "adv" not in k}, strict=False)
```

## ✅ Tracking Models

Models are ignored from Git tracking via `.gitignore`:
```
*.pth
*.pt
*.bin
```

This prevents accidentally committing large binary files.

## 💾 Total Size

- arousal_base_unified.pth: 702.99 MB
- valence_base_final.pth: 701.35 MB
- **Total: ~1.4 GB**

---

**Do not commit `.pth` files to the repository!**
