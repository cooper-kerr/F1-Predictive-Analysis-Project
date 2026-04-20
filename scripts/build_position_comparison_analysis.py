"""
F1 Race Position Predictability Comparison
==========================================
Script-backed version of the core idea from notebook 05.

Research question: using a simple race-state snapshot, how does finishing-
bucket predictability evolve through the race, and how does that compare
between 2024 and 2025 on a fixed race subset?

Approach:
  - Extract a driver's state at selected lap checkpoints
  - Use only current position, tyre life, and tyre compound
  - One-hot encode compound
  - Train / evaluate via row-level train_test_split for each season + checkpoint
  - Save a comparison curve and plot
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent.parent
CACHE_DIR   = ROOT / 'f1-cache'
DATA_DIR    = ROOT / 'data'
MODEL_DIR   = ROOT / 'models'
FIGURES_DIR = ROOT / 'outputs' / 'figures'

fastf1.Cache.enable_cache(str(CACHE_DIR))

RACES = ['Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'Miami']
YEARS = [2024, 2025]
LAP_CHECKPOINTS = [5, 10, 20, 30, 40, 50]
RANDOM_STATE = 42

CURVE_PATH  = DATA_DIR / 'f1_position_comparison_curve.csv'
MODEL_PATH  = MODEL_DIR / 'f1_position_comparison_analysis.pkl'
FIGURE_PATH = FIGURES_DIR / 'position_comparison_curve.png'

plt.style.use('seaborn-v0_8-darkgrid')


def extract_features_at_lap(year: int, grand_prix: str, target_lap: int) -> pd.DataFrame | None:
    try:
        session = fastf1.get_session(year, grand_prix, 'R')
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as exc:
        print(f'FAILED: {year} {grand_prix} lap {target_lap}: {str(exc)[:80]}')
        return None

    laps = session.laps
    if laps.empty:
        return None

    current_state = laps[laps['LapNumber'] == target_lap].copy()
    if current_state.empty:
        return None

    final_positions = []
    for driver in current_state['DriverNumber']:
        driver_laps = laps[laps['DriverNumber'] == driver]
        final_positions.append(driver_laps.iloc[-1]['Position'] if not driver_laps.empty else np.nan)

    current_state['FinalPosition'] = final_positions
    current_state['Year'] = year
    current_state['GrandPrix'] = grand_prix

    cols = ['Year', 'GrandPrix', 'DriverNumber', 'Position', 'Compound', 'TyreLife', 'FinalPosition']
    return current_state[cols].dropna()


def bucket_position(pos: float) -> str:
    bins = [0, 3, 6, 10, 25]
    labels = ['Podium', 'Top Points', 'Low Points', 'Out of Points']
    return pd.cut(pd.Series([pos]), bins=bins, labels=labels).iloc[0]


def train_and_eval(df_features: pd.DataFrame) -> float:
    df_features = df_features.copy()
    df_features['PosBucket'] = df_features['FinalPosition'].apply(bucket_position)
    df_encoded = pd.get_dummies(df_features, columns=['Compound'], drop_first=True)
    feature_cols = ['Position', 'TyreLife'] + [c for c in df_encoded.columns if c.startswith('Compound_')]

    X = df_encoded[feature_cols]
    y = df_encoded['PosBucket']
    if len(X) < 10:
        return np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return float(accuracy_score(y_test, preds))


def build_curve() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for lap in LAP_CHECKPOINTS:
            race_frames = []
            for race in RACES:
                df = extract_features_at_lap(year, race, lap)
                if df is not None and not df.empty:
                    race_frames.append(df)

            if not race_frames:
                rows.append({'year': year, 'checkpoint_lap': lap, 'accuracy': np.nan})
                continue

            df_year = pd.concat(race_frames, ignore_index=True)
            acc = train_and_eval(df_year)
            rows.append({
                'year': year,
                'checkpoint_lap': lap,
                'accuracy': acc,
                'n_rows': int(len(df_year)),
            })
            print(f'year={year} lap={lap:2d} rows={len(df_year):4d} acc={acc:.3f}')
    return pd.DataFrame(rows)


def plot_curve(curve: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for year, color, marker in [(2024, 'steelblue', 'o'), (2025, 'darkorange', 's')]:
        sub = curve[curve['year'] == year].dropna(subset=['accuracy']).sort_values('checkpoint_lap')
        if sub.empty:
            continue
        ax.plot(sub['checkpoint_lap'], sub['accuracy'],
                marker=marker, linewidth=2, color=color, label=f'{year} accuracy')

        ax.axhline(0.25, color='grey', linestyle=':', label='Random baseline (4 classes)')
    ax.set_xlabel('Lap checkpoint')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Race Position Predictability: 2024 vs 2025',
                 fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    curve = build_curve()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    curve.to_csv(CURVE_PATH, index=False)
    plot_curve(curve, FIGURE_PATH)
    joblib.dump({
        'curve': curve,
        'races': RACES,
        'years': YEARS,
        'lap_checkpoints': LAP_CHECKPOINTS,
        'figure_path': str(FIGURE_PATH),
    }, MODEL_PATH)

    print(f'\nSaved: {CURVE_PATH}')
    print(f'Saved: {MODEL_PATH}')
    print(f'Saved: {FIGURE_PATH}')
