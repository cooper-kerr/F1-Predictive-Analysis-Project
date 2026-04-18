"""
F1 Race Position Prediction Pipeline
=====================================
Research question: At what lap can a driver's final finishing position be
predicted with meaningful accuracy?

Target: 4-bucket final-position class — Podium (P1–P3), Top Points (P4–P6),
Low Points (P7–P10), Out of Points (P11+).

Features at lap N: current position, tire compound (one-hot), tire age,
stint number, lap number, gap to leader, rolling pace trend.

Train: 2022–2023  |  Test: 2024  (temporal holdout — consistent with the
undercut and overcut pipelines).

Outputs:
  data/f1_position_dataset.csv            Feature matrix at every checkpoint
  models/f1_position_model.pkl            Dict: {checkpoint_lap: RF classifier}
  outputs/figures/position_curve.png      Predictability vs lap
  outputs/figures/position_confusion.png  Confusion matrices at key laps
  outputs/figures/position_importance.png Feature importance at lap 30
"""

import sys
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT         = Path(__file__).parent.parent
CACHE_DIR    = ROOT / 'f1-cache'
DATA_DIR     = ROOT / 'data'
MODEL_DIR    = ROOT / 'models'
FIGURES_DIR  = ROOT / 'outputs' / 'figures'

fastf1.Cache.enable_cache(str(CACHE_DIR))

RANDOM_STATE     = 42
LAP_CHECKPOINTS  = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
TRAIN_YEARS      = [2022, 2023]
TEST_YEARS       = [2024]
ALL_YEARS        = TRAIN_YEARS + TEST_YEARS
DATASET_PATH     = DATA_DIR  / 'f1_position_dataset.csv'
MODEL_PATH       = MODEL_DIR / 'f1_position_model.pkl'

plt.style.use('seaborn-v0_8-darkgrid')


# ── Feature extraction ──────────────────────────────────────────────────────

def position_bucket(pos: float) -> str:
    if pd.isna(pos):
        return np.nan
    pos = int(pos)
    if pos <= 3:   return 'Podium'
    if pos <= 6:   return 'Top Points'
    if pos <= 10:  return 'Low Points'
    return 'Out of Points'


def extract_race_snapshot(session, year: int, circuit: str) -> pd.DataFrame:
    """
    At every checkpoint lap, build one row per driver with their live state
    and their final finishing position from the session results.
    """
    laps = session.laps.copy()
    if laps is None or len(laps) == 0:
        return pd.DataFrame()

    results = session.results
    if results is None or len(results) == 0:
        return pd.DataFrame()

    final_pos = dict(zip(results['Abbreviation'], results['Position']))

    total_laps = int(laps['LapNumber'].max())
    rows = []

    for checkpoint in LAP_CHECKPOINTS:
        if checkpoint > total_laps:
            continue

        cp_laps = laps[laps['LapNumber'] == checkpoint].copy()
        if cp_laps.empty:
            continue

        # Leader's LapStartTime at this lap defines gap_to_leader
        cp_laps_sorted = cp_laps.dropna(subset=['LapStartTime']).sort_values('LapStartTime')
        if cp_laps_sorted.empty:
            continue
        leader_start = cp_laps_sorted['LapStartTime'].iloc[0]

        for _, lap_row in cp_laps.iterrows():
            driver = lap_row['Driver']
            if driver not in final_pos or pd.isna(final_pos[driver]):
                continue

            # Rolling pace trend: mean lap time over last 5 green laps
            recent = laps[
                (laps['Driver'] == driver) &
                (laps['LapNumber'] < checkpoint) &
                (laps['LapNumber'] >= checkpoint - 5) &
                (laps['IsAccurate'] == True) &
                (laps['TrackStatus'].astype(str) == '1')
            ]
            if len(recent) >= 2:
                pace_mean = float(recent['LapTime'].dt.total_seconds().mean())
                pace_trend = float(np.polyfit(
                    recent['LapNumber'].values.astype(float),
                    recent['LapTime'].dt.total_seconds().values, 1
                )[0]) if len(recent) >= 3 else 0.0
            else:
                pace_mean = np.nan
                pace_trend = np.nan

            start_time = lap_row.get('LapStartTime')
            if pd.isna(start_time):
                gap_to_leader = np.nan
            else:
                gap_to_leader = float((start_time - leader_start).total_seconds())

            rows.append({
                'year':            year,
                'circuit':         circuit,
                'driver':          driver,
                'checkpoint_lap':  checkpoint,
                'race_progress':   checkpoint / total_laps,
                'current_position': lap_row.get('Position'),
                'tire_compound':   str(lap_row.get('Compound', 'UNKNOWN')),
                'tire_age':        lap_row.get('TyreLife'),
                'stint_number':    lap_row.get('Stint'),
                'gap_to_leader':   gap_to_leader,
                'pace_mean':       pace_mean,
                'pace_trend':      pace_trend,
                'final_position':  float(final_pos[driver]),
                'final_bucket':    position_bucket(final_pos[driver]),
            })

    return pd.DataFrame(rows)


def build_full_dataset(years):
    all_rows = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        gp_names = schedule['EventName'].tolist()
        print(f'\n── {year}: {len(gp_names)} races ──')
        for gp in gp_names:
            try:
                session = fastf1.get_session(year, gp, 'R')
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                df = extract_race_snapshot(session, year, gp)
                all_rows.append(df)
                print(f'  {gp:40s}  {len(df):4d} rows')
            except Exception as e:
                print(f'  {gp:40s}  FAILED: {str(e)[:80]}')
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ── Modelling helpers ───────────────────────────────────────────────────────

BASE_FEATURES = [
    'current_position', 'tire_age', 'stint_number',
    'gap_to_leader', 'pace_mean', 'pace_trend', 'race_progress',
]


def prep_features(df: pd.DataFrame):
    compound_dummies = pd.get_dummies(df['tire_compound'], prefix='compound')
    feat = pd.concat([df, compound_dummies], axis=1)
    feature_cols = BASE_FEATURES + list(compound_dummies.columns)
    for col in feature_cols:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors='coerce')
            feat[col] = feat[col].fillna(feat[col].median())
        else:
            feat[col] = 0
    return feat, feature_cols


def train_per_checkpoint(df: pd.DataFrame):
    df = df.dropna(subset=['final_bucket']).copy()
    feat, feature_cols = prep_features(df)

    results = []
    models  = {}

    for cp in LAP_CHECKPOINTS:
        cp_df = feat[feat['checkpoint_lap'] == cp]
        if cp_df.empty:
            continue

        train = cp_df[cp_df['year'].isin(TRAIN_YEARS)]
        test  = cp_df[cp_df['year'].isin(TEST_YEARS)]

        if len(train) < 50 or len(test) < 20:
            print(f'  lap {cp:3d}: too few rows (train={len(train)}, test={len(test)})')
            continue

        X_train = train[feature_cols].values.astype(float)
        y_train = train['final_bucket'].values
        X_test  = test[feature_cols].values.astype(float)
        y_test  = test['final_bucket'].values

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_leaf=3, random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        results.append({
            'checkpoint_lap': cp,
            'n_train':        len(train),
            'n_test':         len(test),
            'accuracy':       acc,
            'f1_weighted':    f1,
        })
        models[cp] = {'model': rf, 'features': feature_cols, 'y_test': y_test, 'preds': preds}
        print(f'  lap {cp:3d}: train={len(train):5d}  test={len(test):5d}  acc={acc:.3f}  f1={f1:.3f}')

    return pd.DataFrame(results), models


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rebuild = '--rebuild' in sys.argv

    if DATASET_PATH.exists() and not rebuild:
        print('Loading cached position dataset... (pass --rebuild to regenerate)')
        df = pd.read_csv(DATASET_PATH)
    else:
        if rebuild:
            print('--rebuild: pulling from FastF1...')
        else:
            print('Building position dataset — first run will take several minutes...')
        df = build_full_dataset(ALL_YEARS)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f'\nSaved: {DATASET_PATH}')

    print(f'\nDataset shape: {df.shape}')
    print(f'Years covered: {sorted(df["year"].unique())}')
    print(f'Checkpoints: {sorted(df["checkpoint_lap"].unique())}')
    print('\nBucket distribution:')
    print(df['final_bucket'].value_counts(dropna=False))

    # Train one RF per checkpoint
    print('\n── Training Random Forest per checkpoint (2022–23 → 2024) ──')
    summary, models = train_per_checkpoint(df)

    print('\n=== Predictability curve ===')
    print(summary.to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'checkpoints': models, 'features': BASE_FEATURES}, MODEL_PATH)
    print(f'\nModels saved: {MODEL_PATH}')

    # ── Plots ──────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Predictability curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary['checkpoint_lap'], summary['accuracy'], 'o-',
            color='steelblue', linewidth=2, markersize=8, label='Accuracy')
    ax.plot(summary['checkpoint_lap'], summary['f1_weighted'], 's--',
            color='tomato', linewidth=1.5, markersize=6, label='Weighted F1')
    ax.axhline(0.25, color='grey', linestyle=':', label='Random baseline (4 classes)')
    ax.set_xlabel('Lap Checkpoint')
    ax.set_ylabel('Score')
    ax.set_title('Race Position Predictability — Trained on 2022–23, Tested on 2024',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'position_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrices at laps 10, 30, 50
    key_laps = [cp for cp in [10, 30, 50] if cp in models]
    if key_laps:
        fig, axes = plt.subplots(1, len(key_laps), figsize=(6 * len(key_laps), 5))
        if len(key_laps) == 1:
            axes = [axes]
        for ax, cp in zip(axes, key_laps):
            m = models[cp]
            ConfusionMatrixDisplay.from_predictions(
                m['y_test'], m['preds'],
                labels=['Podium', 'Top Points', 'Low Points', 'Out of Points'],
                ax=ax, colorbar=False, cmap='Blues',
                xticks_rotation=30,
            )
            acc = summary.loc[summary['checkpoint_lap'] == cp, 'accuracy'].iloc[0]
            ax.set_title(f'Lap {cp}  (acc={acc:.2f})')
        fig.suptitle('Final-Position Predictions on 2024 Holdout',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'position_confusion.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Feature importance at lap 30
    if 30 in models:
        m = models[30]
        importances = pd.Series(m['model'].feature_importances_,
                                index=m['features']).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importances)), importances.values, color='steelblue')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances.index)
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title('Random Forest Feature Importance at Lap 30',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'position_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Save predictability curve as CSV for notebook loading
    summary.to_csv(DATA_DIR / 'f1_position_curve.csv', index=False)
    print(f'Curve saved: {DATA_DIR / "f1_position_curve.csv"}')
    print(f'Figures saved to {FIGURES_DIR}/')
    print('\nDone.')
