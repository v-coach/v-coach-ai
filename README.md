# v-coach-ai

Food image classification project for V-Coach.

This project trains an AI model that looks at a food photo and predicts **which
food it is** among 101 Food-101 categories. This is different from the earlier
restricted-ingredient detector idea. This project is for **food name
classification**.

## Goal

Input:

```text
food photo
```

Output:

```text
top food class, top-k food classes, confidence scores
```

Example:

```json
{
  "top1": "pizza",
  "confidence": 0.91,
  "top5": ["pizza", "lasagna", "cheesecake", "hamburger", "garlic_bread"]
}
```

## Folder Structure

```text
v-coach-ai/
  configs/                  # Training settings
  data/                     # Dataset notes and optional local Food-101 data
  docs/                     # Model choice and app integration notes
  experiments/              # Training run outputs
  models/                   # Final app-ready TFLite models
  reports/                  # Accuracy reports
  src/vcoach_ai/            # Training, evaluation, export, prediction code
```

## Recommended Dataset

Use Food-101: 101 classes, 101,000 images.

This project supports:

1. `tfds`: download Food-101 through `tensorflow_datasets`
2. `directory`: train from local folders

## Setup

```powershell
cd C:\Users\chall\OneDrive\Desktop\v-coach-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Train

```powershell
python -m vcoach_ai.train --config configs/food101.yaml
```

Outputs are saved under:

```text
experiments/food101_mobilenetv3/
```

## Export TFLite

```powershell
python -m vcoach_ai.export_tflite --model experiments/food101_mobilenetv3/best.keras --output models/food101_classifier.tflite
```

The app needs:

- `models/food101_classifier.tflite`
- `models/food101_labels.json`
- input size and preprocessing notes from `models/export_report.json`

## Predict One Image

```powershell
python -m vcoach_ai.predict --model experiments/food101_mobilenetv3/best.keras --image path\to\food.jpg
```

