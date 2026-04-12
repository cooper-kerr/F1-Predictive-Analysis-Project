"""
Standalone runner for the F1 undercut prediction pipeline.
Mirrors the notebook logic exactly — run this to generate results
without needing a Jupyter kernel.
"""

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
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline

import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
fastf1.Cache.enable_cache('./f1-cache')

RANDOM_STATE    = 42
FUEL_BURN_RATE  = 1.8
FUEL_LAP_EFFECT = 0.035
PACE_WINDOW     = 4
DEG_WINDOW      = 5
STABILIZATION   = 4
GAP_WINDOW      = 3
MAX_GAP         = 30.0
MIN_HIST_LAPS   = 3
TRAIN_YEARS     = [2022, 2023]
TEST_YEARS      = [2024]
ALL_YEARS       = TRAIN_YEARS + TEST_YEARS
DATASET_PATH    = './f1_undercut_dataset.csv'
MODEL_PATH      = './f1_undercut_model.pkl'

plt.style.use('seaborn-v0_8-darkgrid')


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_gap_timeseries(session):
    laps = session.laps.copy()
    laps = laps[laps['IsAccurate'] == True].copy()
    laps = laps.dropna(subset=['LapStartTime'])
    laps = laps.sort_values(['LapNumber', 'LapStartTime']).reset_index(drop=True)
    grp  = laps.groupby('LapNumber', sort=False)
    laps['ahead_LapStartTime']  = grp['LapStartTime'].shift(1)
    laps['car_ahead_driver']    = grp['Driver'].shift(1)
    laps['behind_LapStartTime'] = grp['LapStartTime'].shift(-1)
    laps['gap_ahead']  = (laps['LapStartTime'] - laps['ahead_LapStartTime']).dt.total_seconds()
    laps['gap_behind'] = (laps['behind_LapStartTime'] - laps['LapStartTime']).dt.total_seconds()
    return laps[['Driver', 'LapNumber', 'gap_ahead', 'gap_behind', 'car_ahead_driver']].copy()


def compute_pit_loss(session):
    laps     = session.laps.copy()
    in_laps  = laps[laps['PitInTime'].notna()][['Driver', 'LapNumber', 'PitInTime']].copy()
    out_laps = laps[laps['PitOutTime'].notna()][['Driver', 'LapNumber', 'PitOutTime']].copy()
    in_laps['OutLapNumber'] = in_laps['LapNumber'] + 1
    out_laps = out_laps.rename(columns={'LapNumber': 'OutLapNumber'})
    merged   = in_laps.merge(out_laps, on=['Driver', 'OutLapNumber'], how='inner')
    if merged.empty:
        return 22.0
    return float((merged['PitOutTime'] - merged['PitInTime']).dt.total_seconds().median())


def get_pace(driver_laps, up_to_lap, window=PACE_WINDOW):
    recent = driver_laps[
        (driver_laps['LapNumber'] < up_to_lap) &
        (driver_laps['LapNumber'] >= up_to_lap - window) &
        (driver_laps['IsAccurate'] == True) &
        (driver_laps['TrackStatus'].astype(str) == '1')
    ].copy()
    if recent.empty:
        return np.nan
    lt = recent['LapTime'].dt.total_seconds()
    corrected = lt - (recent['LapNumber'] - 1) * FUEL_BURN_RATE * FUEL_LAP_EFFECT
    return float(corrected.median())


def get_deg_delta(driver_laps, up_to_lap, window=DEG_WINDOW):
    recent = driver_laps[
        (driver_laps['LapNumber'] < up_to_lap) &
        (driver_laps['LapNumber'] >= up_to_lap - window) &
        (driver_laps['IsAccurate'] == True) &
        (driver_laps['TrackStatus'].astype(str) == '1') &
        (driver_laps['TyreLife'].notna())
    ].copy()
    if len(recent) < 3:
        return np.nan
    lt = recent['LapTime'].dt.total_seconds()
    corrected = lt - (recent['LapNumber'] - 1) * FUEL_BURN_RATE * FUEL_LAP_EFFECT
    x = recent['TyreLife'].values.astype(float)
    y = corrected.values
    if np.ptp(x) < 0.5:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def has_sc_between(all_laps, driver, lap_start, lap_end):
    window = all_laps[
        (all_laps['Driver'] == driver) &
        (all_laps['LapNumber'] >= lap_start) &
        (all_laps['LapNumber'] <= lap_end)
    ]['TrackStatus'].dropna().astype(str)
    return any(any(c in s for c in ['4', '5', '6']) for s in window)


# ── Label construction ───────────────────────────────────────────────────────

def build_undercut_records(session, year, circuit_name):
    laps = session.laps.copy()
    laps = laps[laps['LapTime'].notna()].copy()
    total_laps = int(laps['LapNumber'].max())

    gaps_df  = build_gap_timeseries(session)
    pit_loss = compute_pit_loss(session)

    gaps_idx = gaps_df.set_index(['Driver', 'LapNumber'])
    laps_idx = laps.set_index(['Driver', 'LapNumber'])

    records  = []
    pit_rows = laps[laps['PitInTime'].notna()][['Driver', 'LapNumber']].copy()

    for _, pit_row in pit_rows.iterrows():
        driver  = pit_row['Driver']
        pit_lap = int(pit_row['LapNumber'])
        dec_lap = pit_lap - 1

        if dec_lap < MIN_HIST_LAPS:
            continue

        try:
            gap_row = gaps_idx.loc[(driver, dec_lap)]
            if isinstance(gap_row, pd.DataFrame):
                gap_row = gap_row.iloc[0]
        except KeyError:
            continue

        car_ahead = gap_row['car_ahead_driver']
        gap_ahead = gap_row['gap_ahead']

        if car_ahead is None or pd.isna(car_ahead) or pd.isna(gap_ahead):
            continue
        if gap_ahead > MAX_GAP or gap_ahead <= 0:
            continue

        car_ahead_pits = laps[
            (laps['Driver'] == car_ahead) &
            (laps['PitInTime'].notna()) &
            (laps['LapNumber'] > pit_lap)
        ]['LapNumber']

        if car_ahead_pits.empty:
            continue

        ca_pit_lap = int(car_ahead_pits.min())
        eval_lap   = min(ca_pit_lap + STABILIZATION, total_laps)

        if has_sc_between(laps, driver, pit_lap, eval_lap):
            continue

        eval_laps_session = laps[
            laps['LapNumber'] == eval_lap
        ][['Driver', 'LapStartTime']].dropna().sort_values('LapStartTime').reset_index(drop=True)

        if eval_laps_session.empty:
            continue

        order = {row['Driver']: idx for idx, row in eval_laps_session.iterrows()}

        if driver not in order or car_ahead not in order:
            continue

        undercut_success = 1 if order[driver] < order[car_ahead] else 0

        driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        ca_laps     = laps[laps['Driver'] == car_ahead].sort_values('LapNumber')

        own_pace     = get_pace(driver_laps, dec_lap + 1)
        threat_pace  = get_pace(ca_laps,     dec_lap + 1)
        deg_delta    = get_deg_delta(driver_laps, dec_lap + 1)
        ca_deg_delta = get_deg_delta(ca_laps,     dec_lap + 1)

        try:
            dec_row = laps_idx.loc[(driver, dec_lap)]
            if isinstance(dec_row, pd.DataFrame):
                dec_row = dec_row.iloc[0]
            ca_dec_row = laps_idx.loc[(car_ahead, dec_lap)]
            if isinstance(ca_dec_row, pd.DataFrame):
                ca_dec_row = ca_dec_row.iloc[0]
        except KeyError:
            continue

        tire_age    = dec_row.get('TyreLife',  np.nan)
        compound    = dec_row.get('Compound',  'UNKNOWN')
        ca_tire_age = ca_dec_row.get('TyreLife', np.nan)

        recent_gaps = gaps_df[
            (gaps_df['Driver'] == driver) &
            (gaps_df['LapNumber'] >= dec_lap - GAP_WINDOW) &
            (gaps_df['LapNumber'] <= dec_lap)
        ].sort_values('LapNumber').dropna(subset=['gap_ahead'])

        closing_rate = float(np.polyfit(
            recent_gaps['LapNumber'].values,
            recent_gaps['gap_ahead'].values, 1
        )[0]) if len(recent_gaps) >= 2 else np.nan

        pace_delta         = (own_pace - threat_pace) if not pd.isna(own_pace) and not pd.isna(threat_pace) else np.nan
        tire_age_advantage = (float(ca_tire_age) - float(tire_age)) if not pd.isna(ca_tire_age) and not pd.isna(tire_age) else np.nan
        pit_loss_fraction  = pit_loss / own_pace if not pd.isna(own_pace) and own_pace > 0 else np.nan

        records.append({
            'year':               year,
            'circuit':            circuit_name,
            'driver':             driver,
            'car_ahead':          car_ahead,
            'pit_lap':            pit_lap,
            'gap_ahead':          gap_ahead,
            'tire_age':           float(tire_age) if not pd.isna(tire_age) else np.nan,
            'car_ahead_tire_age': float(ca_tire_age) if not pd.isna(ca_tire_age) else np.nan,
            'tire_age_advantage': tire_age_advantage,
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
            'undercut_success':   undercut_success,
        })

    return records


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
                # Guard: session.laps must be non-empty for this session to be useful
                if session.laps is None or len(session.laps) == 0:
                    print(f'  {gp:40s}  SKIP: no lap data')
                    continue
                records = build_undercut_records(session, year, gp)
                all_records.extend(records)
                print(f'  {gp:40s}  {len(records):3d} undercut attempts')
            except Exception as e:
                short = str(e)[:80]
                print(f'  {gp:40s}  FAILED: {short}')
    return pd.DataFrame(all_records)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Dataset
    if Path(DATASET_PATH).exists():
        print('Loading cached dataset...')
        df = pd.read_csv(DATASET_PATH)
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
    compound_dummies = pd.get_dummies(df['compound'], prefix='compound')
    df_feat = pd.concat([df, compound_dummies], axis=1)

    BASE_FEATURES = [
        'gap_ahead', 'tire_age', 'car_ahead_tire_age', 'tire_age_advantage',
        'own_pace', 'threat_pace', 'pace_delta', 'deg_delta', 'ca_deg_delta',
        'closing_rate', 'pit_loss', 'pit_loss_fraction', 'race_progress',
    ]
    FEATURES = BASE_FEATURES + [c for c in compound_dummies.columns]

    REQUIRED = ['gap_ahead', 'tire_age', 'car_ahead_tire_age', 'own_pace', 'threat_pace']
    df_model = df_feat.dropna(subset=REQUIRED + ['undercut_success']).copy()
    for col in FEATURES:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
        else:
            df_model[col] = 0

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
    plt.savefig('undercut_eda.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('undercut_model_evaluation.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('undercut_shap.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('undercut_gap_vs_success.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('undercut_age_vs_success.png', dpi=150, bbox_inches='tight')
    plt.close()

    print('Plots saved:')
    print('  undercut_eda.png')
    print('  undercut_model_evaluation.png')
    print('  undercut_shap.png')
    print('  undercut_gap_vs_success.png')
    print('  undercut_age_vs_success.png')
    print('\nDone.')
