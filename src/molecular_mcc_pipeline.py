"""
Command-line inference for SemEval-2026 Task 2 (Subtask 1).

Trains in `notebooks/final_semeval_task2.ipynb`; this module reproduces the
submission path (DeBERTa arousal + valence + isotonic calibration).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _torch_load_state(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ArousalDANNInference(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.adv = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, ids, mask):
        emb = self.base(ids, mask).last_hidden_state[:, 0, :]
        return self.head(emb)


class ValenceModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(768, 1)

    def forward(self, ids, mask):
        emb = self.base(ids, mask).last_hidden_state[:, 0, :]
        return self.head(emb)


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {"ids": enc["input_ids"][0], "mask": enc["attention_mask"][0]}


def cmd_predict(config_path: Path) -> int:
    root = repo_root()
    cfg = load_yaml_config(config_path)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    clip_cfg = cfg.get("clip", {})

    test_csv = resolve_path(root, paths["test_csv"])
    out_csv = resolve_path(root, paths["output_csv"])
    out_zip = resolve_path(root, paths.get("submission_zip", "results/submission.zip"))
    aro_path = resolve_path(root, paths["arousal_weights"])
    val_path = resolve_path(root, paths["valence_weights"])
    iso_path = resolve_path(root, paths["valence_iso"])

    for p, label in [
        (test_csv, "test_csv"),
        (aro_path, "arousal_weights"),
        (val_path, "valence_weights"),
        (iso_path, "valence_iso"),
    ]:
        if not p.is_file():
            print(f"Missing {label}: {p}", file=sys.stderr)
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_cfg["name"]
    batch_size = int(model_cfg.get("batch_size", 32))
    max_length = int(model_cfg.get("max_length", 128))

    test_df = pd.read_csv(test_csv)

    required = {"user_id", "text_id", "text"}
    missing = required - set(test_df.columns)
    if missing:
        print(f"test CSV missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    def sanitize_text(t):
        return str(t).replace("\n", " ").strip()

    texts = test_df["text"].astype(str).map(sanitize_text).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    loader = DataLoader(
        TestDataset(texts, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )

    aro_preds = np.zeros(len(test_df))
    val_preds = np.zeros(len(test_df))

    print("Loading arousal checkpoint…", flush=True)
    model_aro = ArousalDANNInference(model_name).to(device)
    state = _torch_load_state(aro_path, map_location=device)
    state = {k: v for k, v in state.items() if "adv" not in k}
    model_aro.load_state_dict(state, strict=False)
    model_aro.eval()
    buf = []
    with torch.no_grad():
        for b in tqdm(loader, desc="Arousal"):
            ids = b["ids"].to(device)
            mask = b["mask"].to(device)
            buf.extend(model_aro(ids, mask).flatten().cpu().numpy())
    aro_preds = np.array(buf)

    print("Loading valence checkpoint + isotonic…", flush=True)
    model_val = ValenceModel(model_name).to(device)
    model_val.load_state_dict(_torch_load_state(val_path, map_location=device))
    model_val.eval()
    with open(iso_path, "rb") as f:
        iso_reg = pickle.load(f)

    raw_val = []
    with torch.no_grad():
        for b in tqdm(loader, desc="Valence"):
            ids = b["ids"].to(device)
            mask = b["mask"].to(device)
            raw_val.extend(model_val(ids, mask).squeeze().cpu().numpy())
    val_preds = iso_reg.transform(np.array(raw_val))

    v_lo = float(clip_cfg.get("valence_min", -2))
    v_hi = float(clip_cfg.get("valence_max", 2))
    a_lo = float(clip_cfg.get("arousal_min", 0))
    a_hi = float(clip_cfg.get("arousal_max", 2))

    sub = pd.DataFrame(
        {
            "user_id": test_df["user_id"],
            "text_id": test_df["text_id"],
            "pred_valence": np.clip(val_preds, v_lo, v_hi),
            "pred_arousal": np.clip(aro_preds, a_lo, a_hi),
        }
    )

    if sub.isnull().values.any():
        sub["pred_valence"] = sub["pred_valence"].fillna(0.0)
        sub["pred_arousal"] = sub["pred_arousal"].fillna(1.0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} shape={sub.shape}", flush=True)

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, arcname=out_csv.name)
    print(f"Wrote {out_zip}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SemEval-2026 Task 2 inference pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pred = sub.add_parser("predict", help="Run test inference and write pred_subtask1.csv + zip")
    p_pred.add_argument(
        "--config",
        "-c",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
        help="Path to YAML config",
    )

    args = parser.parse_args()
    if args.command == "predict":
        return cmd_predict(args.config.resolve())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
