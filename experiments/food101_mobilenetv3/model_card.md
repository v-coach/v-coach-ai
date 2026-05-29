# Model Card: food101_mobilenetv3

## Task

Food-101 single-label food classification.

## Model

- Backbone: MobileNetV3Small
- Input size: 224
- Classes: 101
- Output: softmax probabilities

## Dataset

- Source: food101_archive
- TFDS name: food101

## Metrics

- top1_accuracy: 0.5150891089108911
- top5_accuracy: 0.8057029702970298
- sample_count: 25250

## App Integration

Export `best.keras` or `final.keras` to TFLite using:

```powershell
python -m vcoach_ai.export_tflite --model experiments/food101_mobilenetv3/best.keras --output models/food101_classifier.tflite
```
