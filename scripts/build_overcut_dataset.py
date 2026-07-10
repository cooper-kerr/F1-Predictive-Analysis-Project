"""
F1 Overcut Strategy Prediction Pipeline
========================================
Overcut: the car directly AHEAD pits first. The stay-out driver decides to
remain on track, hoping to come out ahead after both cars have stopped.

Label: stay-out driver is ahead of the formerly-ahead car, resolved
STABILIZATION laps after the stay-out driver's own pit stop.

Decision point: the lap on which the car ahead pits (the moment the stay-out
driver chooses not to pit). Features are extracted from the lap before that
(dec_lap = pit_lap - 1) to avoid contamination from the in-lap itself.

Train: 2022–2023  |  Test: 2024  (temporal holdout, consistent with undercut)
"""

import sys
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline

import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from f1_strategy_common import (
    build_gap_timeseries,
    compute_pit_loss,
    get_deg_delta,
    get_pace,
    has_sc_between,
)

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent.parent
CACHE_DIR   = ROOT / 'f1-cache'
DATA_DIR    = ROOT / 'data'
MODEL_DIR   = ROOT / 'models'
FIGURES_DIR = ROOT / 'outputs' / 'figures'

fastf1.Cache.enable_cache(str(CACHE_DIR))

RANDOM_STATE    = 42
STABILIZATION   = 4
GAP_WINDOW      = 3
MAX_GAP         = 30.0
MIN_HIST_LAPS   = 3
TRAIN_YEARS     = [2022, 2023]
TEST_YEARS      = [2024]
ALL_YEARS       = TRAIN_YEARS + TEST_YEARS
DATASET_PATH    = DATA_DIR  / 'f1_overcut_dataset.csv'
MODEL_PATH      = MODEL_DIR / 'f1_overcut_model.pkl'

plt.style.use('seaborn-v0_8-darkgrid')


# ── Overcut label construction ───────────────────────────────────────────────

def build_overcut_records(session, year, circuit_name):
    """
    For each pit stop in the session, treat the pitting car as the one
    that WAS ahead. Find the driver directly behind who stays out.
    Label = 1 if the stay-out driver ends up ahead after both stop.

    Decision point: the lap before the car ahead pits (dec_lap = pit_lap - 1).
    Evaluation: STABILIZATION laps after the stay-out driver pits.
    """
    laps = session.laps.copy()
    laps = laps[laps['LapTime'].notna()].copy()
    total_laps = int(laps['LapNumber'].max())

    gaps_df  = build_gap_timeseries(session)
    pit_loss = compute_pit_loss(session)

    gaps_idx = gaps_df.set_index(['Driver', 'LapNumber'])
    laps_idx = laps.set_index(['Driver', 'LapNumber'])

    records  = []

    # Iterate over every pit stop — treat each pitter as the "car ahead that pitted"
    pit_rows = laps[laps['PitInTime'].notna()][['Driver', 'LapNumber']].copy()

    for _, pit_row in pit_rows.iterrows():
        pitting_car = pit_row['Driver']      # car ahead that pits
        pit_lap     = int(pit_row['LapNumber'])
        dec_lap     = pit_lap - 1            # decision point for stay-out driver

        if dec_lap < MIN_HIST_LAPS:
            continue

        # ── Who is directly behind the pitting car? ───────────────────────
        try:
            gap_row = gaps_idx.loc[(pitting_car, dec_lap)]
            if isinstance(gap_row, pd.DataFrame):
                gap_row = gap_row.iloc[0]
        except KeyError:
            continue

        stay_out = gap_row['car_behind_driver']
        if stay_out is None or pd.isna(stay_out):
            continue   # pitting car is last — no one behind

        # Gap from stay-out driver's perspective = gap_behind of pitting car
        gap_behind_pitter = gap_row['gap_behind']
        if pd.isna(gap_behind_pitter) or gap_behind_pitter > MAX_GAP or gap_behind_pitter <= 0:
            continue

        # ── Confirm stay-out driver pits AFTER the pitting car ───────────
        stay_out_pits = laps[
            (laps['Driver'] == stay_out) &
            (laps['PitInTime'].notna()) &
            (laps['LapNumber'] > pit_lap)
        ]['LapNumber']

        if stay_out_pits.empty:
            continue   # stay-out driver never pits after — last stint, skip

        stay_out_pit_lap  = int(stay_out_pits.min())
        stay_out_laps_num = stay_out_pit_lap - pit_lap   # how long they stayed out
        eval_lap          = min(stay_out_pit_lap + STABILIZATION, total_laps)

        if stay_out_laps_num < 1:
            continue   # stayed out for 0 laps — not a real overcut

        # ── SC/VSC contamination ──────────────────────────────────────────
        if has_sc_between(laps, stay_out, pit_lap, eval_lap):
            continue

        # ── Outcome: relative order at eval_lap ───────────────────────────
        eval_laps_session = laps[
            laps['LapNumber'] == eval_lap
        ][['Driver', 'LapStartTime']].dropna().sort_values('LapStartTime').reset_index(drop=True)

        if eval_laps_session.empty:
            continue

        order = {row['Driver']: idx for idx, row in eval_laps_session.iterrows()}

        if stay_out not in order or pitting_car not in order:
            continue   # one retired

        # Overcut success = stay-out driver is now AHEAD of the pitting car
        overcut_success = 1 if order[stay_out] < order[pitting_car] else 0

        # ── Feature extraction at dec_lap ─────────────────────────────────
        stay_out_laps_df = laps[laps['Driver'] == stay_out].sort_values('LapNumber')
        pitting_laps_df  = laps[laps['Driver'] == pitting_car].sort_values('LapNumber')

        own_pace     = get_pace(stay_out_laps_df, dec_lap + 1)
        threat_pace  = get_pace(pitting_laps_df,  dec_lap + 1)
        deg_delta    = get_deg_delta(stay_out_laps_df, dec_lap + 1)
        ca_deg_delta = get_deg_delta(pitting_laps_df,  dec_lap + 1)

        try:
            so_dec  = laps_idx.loc[(stay_out, dec_lap)]
            if isinstance(so_dec, pd.DataFrame): so_dec = so_dec.iloc[0]
            pit_dec = laps_idx.loc[(pitting_car, dec_lap)]
            if isinstance(pit_dec, pd.DataFrame): pit_dec = pit_dec.iloc[0]
        except KeyError:
            continue

        tire_age         = so_dec.get('TyreLife',  np.nan)
        compound         = so_dec.get('Compound',  'UNKNOWN')
        ca_tire_age      = pit_dec.get('TyreLife', np.nan)   # pitting car's tire age

        # Tire age delta from stay-out driver's view:
        # positive = stay-out has MORE laps on tires (disadvantage vs fresh rubber)
        tire_age_delta = (float(tire_age) - float(ca_tire_age)) \
            if not pd.isna(tire_age) and not pd.isna(ca_tire_age) else np.nan

        # Closing rate on dec_lap window
        recent_gaps = gaps_df[
            (gaps_df['Driver'] == pitting_car) &   # gap_behind of pitting car = gap_ahead of stay-out
            (gaps_df['LapNumber'] >= dec_lap - GAP_WINDOW) &
            (gaps_df['LapNumber'] <= dec_lap)
        ].sort_values('LapNumber').dropna(subset=['gap_behind'])

        if len(recent_gaps) >= 2:
            closing_rate = float(np.polyfit(
                recent_gaps['LapNumber'].values,
                recent_gaps['gap_behind'].values, 1
            )[0])
        else:
            closing_rate = np.nan

        pace_delta        = (own_pace - threat_pace) \
            if not pd.isna(own_pace) and not pd.isna(threat_pace) else np.nan
        pit_loss_fraction = pit_loss / own_pace \
            if not pd.isna(own_pace) and own_pace > 0 else np.nan

        records.append({
            # Metadata
            'year':            year,
            'circuit':         circuit_name,
            'stay_out_driver': stay_out,
            'pitting_car':     pitting_car,
            'pit_lap':         pit_lap,            # when car-ahead pitted
            'stay_out_laps':   stay_out_laps_num,  # how many extra laps stayed out
            # Features
            'gap_ahead':          gap_behind_pitter,  # stay-out driver's gap to (now-pitted) car
            'tire_age':           float(tire_age)   if not pd.isna(tire_age)    else np.nan,
            'ca_tire_age':        float(ca_tire_age) if not pd.isna(ca_tire_age) else np.nan,
            'tire_age_delta':     tire_age_delta,     # positive = stay-out more worn
            'compound':           str(compound),
            'own_pace':           own_pace,
            'threat_pace':        threat_pace,
            'pace_delta':         pace_delta,
            'deg_delta':          deg_delta,
            'ca_deg_delta':       ca_deg_delta,
            'closing_rate':       closing_rate,
            'pit_loss':           pit_loss,
            'pit_loss_fraction':  pit_loss_fraction,
            'race_progress':      dec_lap / total_laps if total_laps > 0 else np.nan,
            # Label
            'overcut_success':    overcut_success,
        })

    return records


# ── Dataset construction loop ────────────────────────────────────────────────

def build_full_dataset(years):
    all_records = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        gp_names = schedule['EventName'].tolist()
        print(f'\n── {year}: {len(gp_names)} races ──')
        for gp in gp_names:
            try:
                session = fastf1.get_session(year, gp, 'R')
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                if session.laps is None or len(session.laps) == 0:
                    print(f'  {gp:40s}  SKIP: no lap data')
                    continue
                records = build_overcut_records(session, year, gp)
                all_records.extend(records)
                print(f'  {gp:40s}  {len(records):3d} overcut attempts')
            except Exception as e:
                print(f'  {gp:40s}  FAILED: {str(e)[:80]}')
    return pd.DataFrame(all_records)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    rebuild = '--rebuild' in sys.argv

    # 1. Dataset
    if DATASET_PATH.exists() and not rebuild:
        print('Loading cached dataset...  (pass --rebuild to regenerate from FastF1)')
        df = pd.read_csv(DATASET_PATH)
    else:
        if rebuild:
            print('--rebuild: ignoring cached dataset, pulling from FastF1...')
        else:
            print('Building overcut dataset — this will take several minutes...')
        df = build_full_dataset(ALL_YEARS)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f'\nSaved: {DATASET_PATH}')

    print(f'\nDataset shape: {df.shape}')
    print(f'Years covered: {sorted(df["year"].unique())}')
    vc = df['overcut_success'].value_counts()
    print(f'Label distribution: Success {vc.get(1,0):,} ({vc.get(1,0)/len(df):.1%})'
          f'  | Failure {vc.get(0,0):,} ({vc.get(0,0)/len(df):.1%})')

    # 2. Feature engineering
    compound_dummies = pd.get_dummies(df['compound'], prefix='compound')
    df_feat = pd.concat([df, compound_dummies], axis=1)

    BASE_FEATURES = [
        'gap_ahead', 'tire_age', 'ca_tire_age', 'tire_age_delta',
        'own_pace', 'threat_pace', 'pace_delta',
        'deg_delta', 'ca_deg_delta',
        'closing_rate', 'pit_loss', 'pit_loss_fraction', 'race_progress',
    ]
    FEATURES = BASE_FEATURES + [c for c in compound_dummies.columns]

    REQUIRED = ['gap_ahead', 'tire_age', 'ca_tire_age', 'own_pace', 'threat_pace']
    df_model = df_feat.dropna(subset=REQUIRED + ['overcut_success']).copy()
    for col in FEATURES:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
        else:
            df_model[col] = 0

    print(f'\nModelling dataset: {len(df_model):,} samples, {len(FEATURES)} features')
    print(f'Class balance: {df_model["overcut_success"].mean():.1%} successful overcuts')

    # 3. EDA — gap vs success rate
    df_model['gap_bin'] = pd.cut(df_model['gap_ahead'], bins=[0, 2, 4, 6, 8, 12, 20, 30])
    gap_success = df_model.groupby('gap_bin', observed=True)['overcut_success'].agg(['mean', 'count'])
    gap_success.columns = ['success_rate', 'n_attempts']
    print('\nGap vs overcut success rate:')
    print(gap_success.to_string())

    df_model['age_delta_bin'] = pd.cut(df_model['tire_age_delta'], bins=[-20, -10, -5, 0, 5, 10, 20, 40])
    age_success = df_model.groupby('age_delta_bin', observed=True)['overcut_success'].agg(['mean', 'count'])
    age_success.columns = ['success_rate', 'n_attempts']
    print('\nTire age delta (stay-out worn-ness) vs success rate:')
    print(age_success.to_string())

    stay_success = df_model.groupby('stay_out_laps')['overcut_success'].agg(['mean', 'count'])
    stay_success.columns = ['success_rate', 'n_attempts']
    print('\nStay-out laps vs success rate (top 15):')
    print(stay_success.sort_values('n_attempts', ascending=False).head(15).to_string())

    # 4. Train / test split
    train_mask = df_model['year'].isin(TRAIN_YEARS)
    test_mask  = df_model['year'].isin(TEST_YEARS)

    X_train = df_model.loc[train_mask, FEATURES].values.astype(float)
    y_train = df_model.loc[train_mask, 'overcut_success'].values.astype(int)
    X_test  = df_model.loc[test_mask,  FEATURES].values.astype(float)
    y_test  = df_model.loc[test_mask,  'overcut_success'].values.astype(int)

    print(f'\nTrain (2022–2023): {len(X_train):,}  |  Test (2024): {len(X_test):,}')

    # 5. Logistic Regression baseline
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': xgb_model, 'features': FEATURES}, MODEL_PATH)
    print(f'Model saved: {MODEL_PATH}')

    # 7. Summary
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # EDA distributions
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    eda_feats = ['gap_ahead', 'tire_age', 'ca_tire_age', 'tire_age_delta',
                 'pace_delta', 'deg_delta', 'closing_rate', 'race_progress']
    colours = {1: '#2ecc71', 0: '#e74c3c'}
    for ax, feat in zip(axes, eda_feats):
        for label, name in {1: 'Success', 0: 'Failure'}.items():
            subset = df_model[df_model['overcut_success'] == label][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=colours[label], label=name, density=True)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle('Feature Distributions by Overcut Outcome', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overcut_eda.png', dpi=150, bbox_inches='tight')
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
    plt.suptitle('Overcut Success Prediction — Model Evaluation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overcut_model_evaluation.png', dpi=150, bbox_inches='tight')
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
    plt.suptitle('XGBoost SHAP Analysis — Drivers of Overcut Success',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overcut_shap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Stay-out laps vs success rate (bar chart, top 12 by count)
    top_so = stay_success[stay_success['n_attempts'] >= 5].sort_index()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(range(len(top_so)), top_so['success_rate'], color='steelblue', alpha=0.8)
    ax2.plot(range(len(top_so)), top_so['n_attempts'], 'o--', color='tomato')
    ax1.set_xticks(range(len(top_so)))
    ax1.set_xticklabels([str(int(x)) for x in top_so.index], fontsize=9)
    ax1.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Laps Stayed Out After Car-Ahead Pitted')
    ax1.set_ylabel('Overcut Success Rate', color='steelblue')
    ax2.set_ylabel('Number of Attempts', color='tomato')
    ax1.set_title('Overcut Success Rate by Number of Extra Laps Stayed Out',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overcut_stay_out_laps.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Gap vs success
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(range(len(gap_success)), gap_success['success_rate'], color='steelblue', alpha=0.8)
    ax2.plot(range(len(gap_success)), gap_success['n_attempts'], 'o--', color='tomato')
    ax1.set_xticks(range(len(gap_success)))
    ax1.set_xticklabels([str(b) for b in gap_success.index], rotation=30, fontsize=8)
    ax1.set_xlabel('Gap to Car Ahead (s) at Decision Point')
    ax1.set_ylabel('Overcut Success Rate', color='steelblue')
    ax2.set_ylabel('Number of Attempts', color='tomato')
    ax1.set_title('Overcut Success Rate by Gap to (Pitting) Car Ahead',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overcut_gap_vs_success.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Plots saved to {FIGURES_DIR}/')
    print('\nDone.')
