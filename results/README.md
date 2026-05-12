# Results

## Submission format (Subtask 1)

Official uploads expect a CSV with columns:

`user_id`, `text_id`, `pred_valence`, `pred_arousal`

Valence is typically clipped to `[-2, 2]` and arousal to `[0, 2]` to match the task scale.

The CLI writes this shape to `pred_subtask1.csv` (see `configs/default.yaml`). Older exports that omit `user_id` may not match the current evaluator; regenerate with `python src/molecular_mcc_pipeline.py predict` when weights are available.
