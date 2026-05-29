from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from vcoach_ai.config import ensure_dir, load_config
from vcoach_ai.datasets import load_food101_datasets
from vcoach_ai.food101_labels import FOOD101_LABELS
from vcoach_ai.metrics import evaluate_model, save_metrics
from vcoach_ai.model import build_food101_model, compile_model, unfreeze_backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/food101.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tf.keras.utils.set_random_seed(config["seed"])

    run_name = config["run_name"]
    experiment_dir = ensure_dir(Path(config["outputs"]["experiment_dir"]) / run_name)

    train_ds, val_ds = load_food101_datasets(config)

    model = build_food101_model(
        image_size=config["data"]["image_size"],
        num_classes=len(FOOD101_LABELS),
        dropout_rate=config["model"]["dropout_rate"],
        imagenet_weights=config["model"]["imagenet_weights"],
    )

    top_k = config["model"]["top_k"]
    compile_model(
        model,
        learning_rate=config["training"]["frozen_learning_rate"],
        top_k=top_k,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(experiment_dir / "best.keras"),
            monitor=f"val_top{top_k}_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(experiment_dir / "history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor=f"val_top{top_k}_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    frozen_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["frozen_epochs"],
        callbacks=callbacks,
    )

    unfreeze_backbone(
        model,
        fine_tune_at_layer=config["training"]["fine_tune_at_layer"],
    )
    compile_model(
        model,
        learning_rate=config["training"]["fine_tune_learning_rate"],
        top_k=top_k,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(frozen_history.history["loss"]),
        epochs=config["training"]["frozen_epochs"] + config["training"]["fine_tune_epochs"],
        callbacks=callbacks,
    )

    final_model_path = experiment_dir / "final.keras"
    model.save(final_model_path)

    metrics, per_class_metrics = evaluate_model(model, val_ds, top_k=top_k)
    save_metrics(metrics, per_class_metrics, experiment_dir)

    write_model_card(
        experiment_dir=experiment_dir,
        run_name=run_name,
        config=config,
        metrics=metrics,
    )


def write_model_card(experiment_dir: Path, run_name: str, config: dict, metrics: dict) -> None:
    metric_lines = "\n".join(f"- {key}: {value}" for key, value in metrics.items())
    content = f"""# Model Card: {run_name}

## Task

Food-101 single-label food classification.

## Model

- Backbone: {config["model"]["backbone"]}
- Input size: {config["data"]["image_size"]}
- Classes: 101
- Output: softmax probabilities

## Dataset

- Source: {config["data"]["source"]}
- TFDS name: {config["data"]["tfds_name"]}

## Metrics

{metric_lines}

## App Integration

Export `best.keras` or `final.keras` to TFLite using:

```powershell
python -m vcoach_ai.export_tflite --model experiments/{run_name}/best.keras --output models/food101_classifier.tflite
```
"""
    (experiment_dir / "model_card.md").write_text(content, encoding="utf-8")
    (experiment_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

