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

### Setup

1. Download both `.pth` files from your model repository/source
2. Place them in this directory (`models/`)
3. Update paths in notebooks if necessary

### Usage in Code

```python
# Example loading
import torch
arousal_model = torch.load('models/arousal_base_unified.pth')
valence_model = torch.load('models/valence_base_final.pth')
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
