from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from vcoach_ai.config import ensure_dir
from vcoach_ai.food101_labels import FOOD101_LABELS, display_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="models/food101_classifier.tflite")
    parser.add_argument("--float16", action="store_true")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.float16:
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_bytes(tflite_model)

    labels_path = output_path.with_name("food101_labels.json")
    labels_path.write_text(
        json.dumps(
            {
                "labels": FOOD101_LABELS,
                "display_labels": [display_name(label) for label in FOOD101_LABELS],
                "output": "[1, 101] softmax probabilities in labels order",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = output_path.with_name("export_report.json")
    report_path.write_text(
        json.dumps(
            {
                "source_model": args.model,
                "output_model": str(output_path),
                "labels_file": str(labels_path),
                "float16": args.float16,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

