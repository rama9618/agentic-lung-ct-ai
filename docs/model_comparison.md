# Model Comparison

This project compares three CNN architectures for lung CT image classification.

## Summary Table

| Model | Purpose | Strengths | Weaknesses | Test Accuracy | Weighted F1 |
|---|---|---|---|---:|---:|
| ResNet50 | Strong baseline classifier | Stable, widely used, Grad-CAM friendly | Larger and less efficient than newer compact models | 91.43% | 91.43% |
| DenseNet121 | Medical-imaging candidate with feature reuse | Dense feature propagation, parameter efficient, good for subtle patterns | Can use more memory; checkpoint not trained yet in this repo snapshot | Placeholder | Placeholder |
| EfficientNet-B0 | Lightweight efficient classifier | Good speed/accuracy tradeoff, deployment friendly | May require tuning; checkpoint not trained yet in this repo snapshot | Placeholder | Placeholder |

## How Comparison Works

Run:

```bash
python src/compare_models.py
```

The script evaluates:

```text
models/best_resnet50.pth
models/best_densenet121.pth
models/best_efficientnet_b0.pth
```

It writes:

```text
reports/model_comparison.csv
reports/model_comparison.txt
reports/best_model.txt
```

The best model is selected using weighted F1-score.

## Metric Notes

- **Accuracy**: Overall fraction of correct predictions.
- **Precision**: How often predicted positives are correct.
- **Recall**: How often true class examples are found.
- **Weighted F1**: F1-score weighted by class support; useful when class counts differ.

## Practical Interpretation

ResNet50 is currently the evaluated baseline in this repository snapshot. DenseNet121 and EfficientNet-B0 are supported by the model factory and training script, but their checkpoints must be trained before final comparison metrics are available.

## Safety Note

Higher test metrics do not imply clinical validation. All model outputs remain research-only and require qualified medical review.
