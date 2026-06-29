# DriveSeg

> Autonomous driving scene segmentation quality assessment with hazard prediction.

Trains four classifiers on synthetic segmentation metrics (mean IoU, Dice coefficient, boundary F1, per-class IoU, frame blur) to predict hazardous driving scenes. Dashboard provides segmentation quality analysis, BEV perception theory, and Kalman filter tracking visualisation.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.716 |
| Gini | 0.431 |
| KS Statistic | 0.337 |
| F1 Score | 0.364 |
| Accuracy | 0.651 |

5-fold CV AUC: 0.710 ± 0.012. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Frame records overview, hazard distribution, per-pixel segmentation feature descriptions |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Segmentation Quality** | mIoU and Dice distributions, per-class IoU analysis, boundary F1 metrics |
| **Safety** | Scene risk scoring, Kalman filter tracking theory, blur analysis, BEV projection visualisation |

## Repo Structure

```
DriveSeg/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic autonomous driving dataset: per-pixel segmentation metrics (mean IoU, Dice coefficient, boundary F1, per-class IoU for vehicle/pedestrian/road), frame blur, edge intensity, texture variance, depth variance.

## License

MIT
