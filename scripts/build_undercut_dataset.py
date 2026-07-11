"""
Standalone runner for the F1 undercut prediction pipeline.
Mirrors the notebook logic exactly — run this to generate results
without needing a Jupyter kernel.
"""

import sys
import fastf1
import pandas as pd
from pathlib import Path
import warnings
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline

import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_runner import build_schedule_dataset
from strategy_attempts import build_undercut_records
from strategy_features import prepare_strategy_frame

warnings.filterwarnings('ignore')

# ── Project root paths (run this script from the project root) ───────────────
ROOT         = Path(__file__).parent.parent
CACHE_DIR    = ROOT / 'f1-cache'
DATA_DIR     = ROOT / 'data'
MODEL_DIR    = ROOT / 'models'
FIGURES_DIR  = ROOT / 'outputs' / 'figures'

fastf1.Cache.enable_cache(str(CACHE_DIR))

RANDOM_STATE    = 42
TRAIN_YEARS     = [2022, 2023]
TEST_YEARS      = [2024]
ALL_YEARS       = TRAIN_YEARS + TEST_YEARS
DATASET_PATH    = DATA_DIR  / 'f1_undercut_dataset.csv'
MODEL_PATH      = MODEL_DIR / 'f1_undercut_model.pkl'

plt.style.use('seaborn-v0_8-darkgrid')


def build_full_dataset(years):
    return build_schedule_dataset(
        years,
        build_undercut_records,
        'undercut attempts',
        schedule_loader=fastf1.get_event_schedule,
        session_loader=fastf1.get_session,
        session_load_kwargs={
            'laps': True,
            'telemetry': False,
            'weather': False,
            'messages': False,
        },
        header_style='box',
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    rebuild = '--rebuild' in sys.argv

    # 1. Dataset
    if Path(DATASET_PATH).exists() and not rebuild:
        print('Loading cached dataset...  (pass --rebuild to regenerate from FastF1)')
        df = pd.read_csv(DATASET_PATH)
    else:
        if rebuild:
            print('--rebuild: ignoring cached dataset, pulling from FastF1...')
        else:
            print('Building dataset — this will take several minutes...')
        df = build_full_dataset(ALL_YEARS)
        df.to_csv(DATASET_PATH, index=False)
        print(f'\nSaved: {DATASET_PATH}')

    print(f'\nDataset shape: {df.shape}')
    print(f'Years covered: {sorted(df["year"].unique())}')
    vc = df['undercut_success'].value_counts()
    print(f'Label distribution: Success {vc.get(1,0):,} ({vc.get(1,0)/len(df):.1%})  '
          f'| Failure {vc.get(0,0):,} ({vc.get(0,0)/len(df):.1%})')

    # 2. Feature engineering
    df_model, FEATURES = prepare_strategy_frame(df, 'undercut')

    print(f'\nModelling dataset: {len(df_model):,} samples, {len(FEATURES)} features')
    print(f'Class balance: {df_model["undercut_success"].mean():.1%} successful undercuts')

    # 3. EDA — gap vs success
    df_model['gap_bin'] = pd.cut(df_model['gap_ahead'], bins=[0, 2, 4, 6, 8, 12, 20, 30])
    gap_success = df_model.groupby('gap_bin', observed=True)['undercut_success'].agg(['mean', 'count'])
    gap_success.columns = ['success_rate', 'n_attempts']
    print('\nGap vs undercut success rate:')
    print(gap_success.to_string())

    df_model['age_adv_bin'] = pd.cut(df_model['tire_age_advantage'], bins=[-30, -10, -5, 0, 5, 10, 20, 40])
    age_success = df_model.groupby('age_adv_bin', observed=True)['undercut_success'].agg(['mean', 'count'])
    age_success.columns = ['success_rate', 'n_attempts']
    print('\nTire age advantage vs undercut success rate:')
    print(age_success.to_string())

    # 4. Train / test split
    train_mask = df_model['year'].isin(TRAIN_YEARS)
    test_mask  = df_model['year'].isin(TEST_YEARS)

    X_train = df_model.loc[train_mask, FEATURES].values.astype(float)
    y_train = df_model.loc[train_mask, 'undercut_success'].values.astype(int)
    X_test  = df_model.loc[test_mask,  FEATURES].values.astype(float)
    y_test  = df_model.loc[test_mask,  'undercut_success'].values.astype(int)

    print(f'\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}')

    # 5. Logistic Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_test)
    lr_probs = lr_pipe.predict_proba(X_test)[:, 1]

    print('\n=== Logistic Regression (Baseline) ===')
    print(f'Accuracy: {accuracy_score(y_test, lr_preds):.3f}  |  ROC-AUC: {roc_auc_score(y_test, lr_probs):.3f}')
    print(classification_report(y_test, lr_preds, target_names=['No Gain', 'Position Gain']))

    coef_df = pd.DataFrame({
        'feature': FEATURES,
        'coefficient': lr_pipe.named_steps['clf'].coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print('Top 10 LR coefficients:')
    print(coef_df.head(10).to_string(index=False))

    # 6. XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, min_child_weight=5,
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    print('\n=== XGBoost (Primary) ===')
    print(f'Accuracy: {accuracy_score(y_test, xgb_preds):.3f}  |  ROC-AUC: {roc_auc_score(y_test, xgb_probs):.3f}')
    print(classification_report(y_test, xgb_preds, target_names=['No Gain', 'Position Gain']))

    joblib.dump({'model': xgb_model, 'features': FEATURES}, MODEL_PATH)
    print(f'Model saved: {MODEL_PATH}')

    # 7. Summary table
    summary = pd.DataFrame([
        {
            'Model': 'Logistic Regression',
            'Accuracy':  round(accuracy_score(y_test, lr_preds),  3),
            'ROC-AUC':   round(roc_auc_score(y_test, lr_probs),   3),
            'Precision': round(precision_score(y_test, lr_preds), 3),
            'Recall':    round(recall_score(y_test, lr_preds),    3),
            'F1':        round(f1_score(y_test, lr_preds),        3),
        },
        {
            'Model': 'XGBoost',
            'Accuracy':  round(accuracy_score(y_test, xgb_preds),  3),
            'ROC-AUC':   round(roc_auc_score(y_test, xgb_probs),   3),
            'Precision': round(precision_score(y_test, xgb_preds), 3),
            'Recall':    round(recall_score(y_test, xgb_preds),    3),
            'F1':        round(f1_score(y_test, xgb_preds),        3),
        },
    ])
    print('\n=== Final Comparison (2024 holdout) ===')
    print(summary.to_string(index=False))

    # 8. Plots
    print('\nGenerating plots...')

    # EDA distributions
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    eda_feats = ['gap_ahead', 'tire_age', 'car_ahead_tire_age', 'tire_age_advantage',
                 'pace_delta', 'deg_delta', 'closing_rate', 'race_progress']
    colours = {1: '#2ecc71', 0: '#e74c3c'}
    for ax, feat in zip(axes, eda_feats):
        for label, name in {1: 'Success', 0: 'Failure'}.items():
            subset = df_model[df_model['undercut_success'] == label][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=colours[label], label=name, density=True)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle('Feature Distributions by Undercut Outcome', fontsize=13, fontweight='bold')
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'undercut_eda.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrices + ROC
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, lr_preds,
        display_labels=['No Gain', 'Position Gain'], ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Logistic Regression\nConfusion Matrix')
    ConfusionMatrixDisplay.from_predictions(y_test, xgb_preds,
        display_labels=['No Gain', 'Position Gain'], ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title('XGBoost\nConfusion Matrix')
    RocCurveDisplay.from_predictions(y_test, lr_probs,  ax=axes[2], name='Logistic Regression')
    RocCurveDisplay.from_predictions(y_test, xgb_probs, ax=axes[2], name='XGBoost')
    axes[2].plot([0, 1], [0, 1], 'k--')
    axes[2].set_title('ROC Curves — 2024 Holdout')
    plt.suptitle('Undercut Success Prediction — Model Evaluation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'undercut_model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # SHAP
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                      show=False, max_display=12, plot_type='bar')
    axes[0].set_title('Mean |SHAP| — Feature Importance')
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                      show=False, max_display=12)
    axes[1].set_title('SHAP Beeswarm — Direction & Magnitude')
    plt.suptitle('XGBoost SHAP Analysis — Drivers of Undercut Success',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'undercut_shap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Gap success rate bar chart
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(range(len(gap_success)), gap_success['success_rate'],
            color='steelblue', alpha=0.8)
    ax2.plot(range(len(gap_success)), gap_success['n_attempts'], 'o--', color='tomato')
    ax1.set_xticks(range(len(gap_success)))
    ax1.set_xticklabels([str(b) for b in gap_success.index], rotation=30, fontsize=8)
    ax1.set_xlabel('Gap to Car Ahead (s)')
    ax1.set_ylabel('Undercut Success Rate', color='steelblue')
    ax2.set_ylabel('Number of Attempts', color='tomato')
    ax1.set_title('Undercut Success Rate by Gap to Car Ahead', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'undercut_gap_vs_success.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Tire age advantage chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colours_bar = ['#c0392b' if r < 0.5 else '#27ae60' for r in age_success['success_rate']]
    ax.bar(range(len(age_success)), age_success['success_rate'],
           color=colours_bar, alpha=0.85)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(age_success)))
    ax.set_xticklabels([str(b) for b in age_success.index], rotation=30, fontsize=8)
    ax.set_xlabel('Tire Age Advantage (car_ahead_age − driver_age, laps)')
    ax.set_ylabel('Undercut Success Rate')
    ax.set_title('Undercut Success Rate by Relative Tire Age Advantage',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'undercut_age_vs_success.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Plots saved to {FIGURES_DIR}/')
    print('\nDone.')
