from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from vcoach_ai.food101_labels import FOOD101_LABELS


def evaluate_model(model, dataset, top_k: int) -> tuple[dict, pd.DataFrame]:
    y_true = []
    y_score = []

    for images, labels in dataset:
        y_true.append(labels.numpy())
        y_score.append(model.predict(images, verbose=0))

    y_true_array = np.concatenate(y_true)
    y_score_array = np.concatenate(y_score)
    y_pred_array = np.argmax(y_score_array, axis=1)

    top1_accuracy = float(np.mean(y_pred_array == y_true_array))
    topk_indices = np.argsort(y_score_array, axis=1)[:, -top_k:]
    topk_accuracy = float(np.mean([truth in topk for truth, topk in zip(y_true_array, topk_indices)]))

    metrics = {
        "top1_accuracy": top1_accuracy,
        f"top{top_k}_accuracy": topk_accuracy,
        "sample_count": int(len(y_true_array)),
    }

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=list(range(len(FOOD101_LABELS))),
        target_names=FOOD101_LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})
    return metrics, per_class


def save_metrics(metrics: dict, per_class: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    per_class.to_csv(output_path / "per_class_metrics.csv", index=False)

