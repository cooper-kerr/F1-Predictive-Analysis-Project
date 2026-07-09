"""
F1 Race Position Predictability Comparison
==========================================
Script-backed version of notebook 05's leakage-aware position model.

Research question: using only race-state features available at a lap
checkpoint, how predictable is a driver's final finishing-position bucket?

Approach:
  - Extract one row per driver at selected lap checkpoints
  - Use FIA classified final position as the target
  - Exclude current Position from the headline model to avoid leakage
  - Evaluate with leave-one-race-out GroupKFold by Grand Prix
  - Save the comparison curve, fold accuracies, and confusion matrices
"""

import warnings
from pathlib import Path

import fastf1
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent.parent
CACHE_DIR   = ROOT / 'f1-cache'
DATA_DIR    = ROOT / 'data'
MODEL_DIR   = ROOT / 'models'
FIGURES_DIR = ROOT / 'outputs' / 'figures'

fastf1.Cache.enable_cache(str(CACHE_DIR))

RACES = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
    'Miami', 'Emilia Romagna', 'Monaco', 'Canada', 'Spain',
    'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
    'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico',
    'Sao Paulo', 'Las Vegas', 'Qatar', 'Abu Dhabi',
]
YEARS = [2024, 2025]
LAP_CHECKPOINTS = [5, 10, 20, 30, 40, 50]
POSITION_BINS = [0, 3, 6, 10, 99]
POSITION_LABELS = [
    'Podium (P1-P3)',
    'Top Points (P4-P6)',
    'Low Points (P7-P10)',
    'Out of Points (P11+)',
]
BASE_NUMERIC_FEATURES = [
    'TyreLife',
    'Stint',
    'PitStopsSoFar',
    'GapToLeader',
    'PaceDelta3',
]
RANDOM_STATE = 42

CURVE_PATH  = DATA_DIR / 'f1_position_comparison_curve.csv'
MODEL_PATH  = MODEL_DIR / 'f1_position_comparison_analysis.pkl'
FIGURE_PATH = FIGURES_DIR / 'position_comparison_curve.png'

plt.style.use('seaborn-v0_8-darkgrid')


def classified_final_positions(session) -> pd.Series:
    """Return DriverNumber (str) -> FIA classified finishing position."""
    results = session.results.copy()
    results.index = results.index.astype(str)
    return results['Position']


def extract_race_features(
    year: int,
    grand_prix: str,
    lap_checkpoints=LAP_CHECKPOINTS,
) -> pd.DataFrame | None:
    try:
        session = fastf1.get_session(year, grand_prix, 'R')
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as exc:
        print(f'FAILED: {year} {grand_prix}: {str(exc)[:80]}')
        return None

    laps = session.laps
    if laps.empty:
        return None

    final_pos = classified_final_positions(session)

    laps = laps.sort_values(['DriverNumber', 'LapNumber']).copy()
    laps['DriverNumber'] = laps['DriverNumber'].astype(str)
    laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
    laps['TimeSec'] = laps['Time'].dt.total_seconds()
    laps['IsPitLap'] = laps['PitInTime'].notna() | laps['PitOutTime'].notna()
    laps['PitInFlag'] = laps['PitInTime'].notna().astype(int)
    laps['PitStopsSoFar'] = laps.groupby('DriverNumber')['PitInFlag'].cumsum()

    rows = []
    for lap_n in lap_checkpoints:
        at_lap = laps[laps['LapNumber'] == lap_n].copy()
        if at_lap.empty:
            continue

        leader_time = at_lap['TimeSec'].min(skipna=True)
        at_lap['GapToLeader'] = at_lap['TimeSec'] - leader_time

        window = laps[
            (laps['LapNumber'] <= lap_n) &
            (laps['LapNumber'] > lap_n - 3) &
            (~laps['IsPitLap'])
        ]
        if not window.empty:
            field_median = window['LapTimeSec'].median()
            driver_pace = (
                window.groupby('DriverNumber')['LapTimeSec']
                .median()
                .rename('DriverPace')
            )
            at_lap = at_lap.merge(driver_pace, on='DriverNumber', how='left')
            at_lap['PaceDelta3'] = at_lap['DriverPace'] - field_median
        else:
            at_lap['PaceDelta3'] = np.nan

        at_lap['Year'] = year
        at_lap['GrandPrix'] = grand_prix
        at_lap['LapCheckpoint'] = lap_n
        at_lap['FinalPosition'] = at_lap['DriverNumber'].map(final_pos)

        cols = [
            'Year', 'GrandPrix', 'LapCheckpoint', 'DriverNumber',
            'Position', 'Compound', 'TyreLife', 'Stint',
            'PitStopsSoFar', 'GapToLeader', 'PaceDelta3', 'FinalPosition',
        ]
        rows.append(at_lap[cols])

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def assign_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=POSITION_BINS, labels=POSITION_LABELS, right=True)


def majority_class_baseline(df: pd.DataFrame) -> float:
    buckets = assign_bucket(df['FinalPosition']).dropna()
    if buckets.empty:
        return float('nan')
    return float(buckets.value_counts(normalize=True).max())


def evaluate_checkpoint(df_year: pd.DataFrame, lap_n: int) -> dict:
    df = df_year.copy()
    df['PosBucket'] = assign_bucket(df['FinalPosition'])
    df = df.dropna(subset=['PosBucket'])

    sub = df[df['LapCheckpoint'] == lap_n].copy()
    if sub.empty:
        return empty_result(lap_n)

    sub = pd.get_dummies(sub, columns=['Compound'], dummy_na=False)
    compound_cols = [c for c in sub.columns if c.startswith('Compound_')]
    feature_cols = BASE_NUMERIC_FEATURES + compound_cols
    sub = sub.dropna(subset=feature_cols + ['PosBucket'])

    if sub['GrandPrix'].nunique() < 2 or len(sub) < 10:
        return empty_result(lap_n)

    X = sub[feature_cols].astype(float)
    y = sub['PosBucket'].astype(str)
    groups = sub['GrandPrix']

    gkf = GroupKFold(n_splits=groups.nunique())
    fold_accs: dict[str, float] = {}
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        if pd.Series(y.iloc[train_idx]).nunique() < 2:
            continue

        held_race = groups.iloc[test_idx].iloc[0]
        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = rf.predict(X.iloc[test_idx])

        fold_accs[held_race] = float(accuracy_score(y.iloc[test_idx], preds))
        all_y_true.extend(y.iloc[test_idx].tolist())
        all_y_pred.extend(preds.tolist())

    if not fold_accs:
        return empty_result(lap_n)

    arr = np.array(list(fold_accs.values()), dtype=float)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=POSITION_LABELS)
    return {
        'checkpoint_lap': lap_n,
        'accuracy': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'se': float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        'n_folds': int(len(arr)),
        'n_rows': int(len(sub)),
        'feature_cols': feature_cols,
        'fold_accs': fold_accs,
        'confusion_matrix': cm,
        'labels': POSITION_LABELS,
    }


def empty_result(lap_n: int) -> dict:
    return {
        'checkpoint_lap': lap_n,
        'accuracy': np.nan,
        'std': np.nan,
        'se': np.nan,
        'n_folds': 0,
        'n_rows': 0,
        'feature_cols': [],
        'fold_accs': {},
        'confusion_matrix': np.zeros((len(POSITION_LABELS), len(POSITION_LABELS)), dtype=int),
        'labels': POSITION_LABELS,
    }


def build_feature_dataset(year: int) -> pd.DataFrame:
    frames = []
    for race in RACES:
        df = extract_race_features(year, race, LAP_CHECKPOINTS)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_curve() -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    details = {}
    baselines = {}

    for year in YEARS:
        print(f'\n--- Loading {year} data ---')
        df_year = build_feature_dataset(year)
        if df_year.empty:
            baselines[year] = np.nan
            continue

        baselines[year] = majority_class_baseline(df_year)
        details[year] = {}
        print(f'{year}: {df_year["GrandPrix"].nunique()} races, {len(df_year):,} rows')

        for lap in LAP_CHECKPOINTS:
            result = evaluate_checkpoint(df_year, lap)
            details[year][lap] = result
            rows.append({
                'year': year,
                'checkpoint_lap': lap,
                'accuracy': result['accuracy'],
                'std': result['std'],
                'se': result['se'],
                'n_folds': result['n_folds'],
                'n_rows': result['n_rows'],
            })
            print(
                f'year={year} lap={lap:2d} rows={result["n_rows"]:4d} '
                f'folds={result["n_folds"]:2d} acc={result["accuracy"]:.3f}'
            )

    return pd.DataFrame(rows), details, baselines


def plot_curve(curve: pd.DataFrame, baselines: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for year, color, marker in [(2024, 'steelblue', 'o'), (2025, 'darkorange', 's')]:
        sub = curve[curve['year'] == year].dropna(subset=['accuracy']).sort_values('checkpoint_lap')
        if sub.empty:
            continue
        ax.plot(sub['checkpoint_lap'], sub['accuracy'],
                marker=marker, linewidth=2, color=color, label=f'{year} accuracy')
        ax.fill_between(sub['checkpoint_lap'], sub['accuracy'] - sub['se'],
                        sub['accuracy'] + sub['se'], color=color, alpha=0.18)
        if year in baselines and not np.isnan(baselines[year]):
            ax.axhline(baselines[year], color=color, linestyle='--', alpha=0.5,
                       label=f'{year} majority baseline ({baselines[year]:.0%})')

    ax.set_xlabel('Lap checkpoint')
    ax.set_ylabel('Mean accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Race Position Predictability: Leave-One-Race-Out CV',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    curve, details, baselines = build_curve()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    curve.to_csv(CURVE_PATH, index=False)
    plot_curve(curve, baselines, FIGURE_PATH)
    joblib.dump({
        'curve': curve,
        'details': details,
        'baselines': baselines,
        'races': RACES,
        'years': YEARS,
        'lap_checkpoints': LAP_CHECKPOINTS,
        'feature_base': BASE_NUMERIC_FEATURES,
        'position_included': False,
        'evaluation': 'leave-one-race-out GroupKFold by GrandPrix',
        'figure_path': str(FIGURE_PATH),
    }, MODEL_PATH)

    print(f'\nSaved: {CURVE_PATH}')
    print(f'Saved: {MODEL_PATH}')
    print(f'Saved: {FIGURE_PATH}')
