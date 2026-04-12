# F1 Predictive Analysis Project

Data science project applying machine learning to Formula One race strategy using the [FastF1](https://docs.fastf1.dev/) API.

**Authors:** Cooper Kerr, Isaac Middlemas — University of Utah

---

## Research Questions

| # | Question | Model | Status |
|---|---|---|---|
| 1 | At what lap can a driver's final finishing position be predicted? | Random Forest Classifier | ✅ |
| 2 | When will a driver's pit window open? | Gradient Boosted Machine (GBM) | ✅ |
| 3 | Will pitting now (undercut) beat the car directly ahead? | Logistic Regression + XGBoost | ✅ |

---

## Repository Structure

```
F1-Predictive-Analysis-Project/
├── notebooks/
│   ├── 01_milestone_report.ipynb       # Full milestone report (all three questions)
│   ├── 02_pit_window_forecasting.ipynb # Pit window forecasting pipeline
│   └── 03_undercut_prediction.ipynb    # Undercut/overcut strategy prediction
│
├── scripts/
│   ├── build_undercut_dataset.py       # Builds f1_undercut_dataset.csv + trains model
│   └── test_undercut_model.py          # Interactive model test harness (4 modes)
│
├── data/
│   ├── f1_pit_window_labels.csv        # 47K lap-level observations (pit window model)
│   └── f1_undercut_dataset.csv         # 966 labeled undercut attempts (2022-2024)
│
├── models/
│   ├── f1_pit_window_model.pkl         # Trained pit window GBM
│   ├── f1_pit_window_model_tuned.pkl   # Hyperparameter-tuned pit window GBM
│   └── f1_undercut_model.pkl           # Trained undercut XGBoost classifier
│
├── outputs/
│   └── figures/                        # All generated plots
│
└── f1-cache/                           # FastF1 local cache (gitignored for 2022-2024)
```

---

## Undercut Model Results (2024 holdout)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 77.1% | 0.825 |
| XGBoost (primary) | 73.8% | 0.806 |

**Key predictors (SHAP):** gap to car ahead, pace delta, tire age advantage, degradation rate, closing rate.

---

## Usage

**Test the undercut model interactively:**

```bash
# Browse all 2024 holdout predictions sorted by model confidence
python scripts/test_undercut_model.py inspect

# Examine false positives and false negatives
python scripts/test_undercut_model.py errors

# Filter by driver
python scripts/test_undercut_model.py driver VER

# Predict success probability for a custom scenario
python scripts/test_undercut_model.py scenario
```

**Rebuild the dataset and retrain from scratch:**

```bash
python scripts/build_undercut_dataset.py
```

**Requirements:** Python 3.12, FastF1 3.8.x, pandas 2.2.x, scikit-learn, xgboost, shap
