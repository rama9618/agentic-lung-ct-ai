# Usage Guide

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If your environment uses `python3`:

```bash
python3 -m pip install -r requirements.txt
```

## Run App

Start the Streamlit dashboard:

```bash
streamlit run src/app.py
```

Then upload a CT image through the UI.

The app displays:

- Prediction
- Confidence
- Grad-CAM heatmap
- Heatmap overlay
- Pseudo-3D visualizations
- AI clinical support note
- AI Medical Assistant chatbot

## Train Model

Choose a model in `src/train.py`:

```python
MODEL_NAME = "resnet50"
```

Supported values:

```text
resnet50
densenet121
efficientnet_b0
```

Run:

```bash
python src/train.py
```

Best checkpoint output:

```text
models/best_<model_name>.pth
```

## Evaluate Model

Run:

```bash
python src/evaluate.py
```

Output:

```text
reports/test_evaluation.txt
```

## Compare Models

Run:

```bash
python src/compare_models.py
```

Outputs:

```text
reports/model_comparison.csv
reports/model_comparison.txt
reports/best_model.txt
```

The app automatically loads the model listed in:

```text
reports/best_model.txt
```

## Generate Grad-CAM Images

Run:

```bash
python src/gradcam.py
```

Outputs:

```text
reports/gradcam/
```
