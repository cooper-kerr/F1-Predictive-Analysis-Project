"""
F1 Tire Degradation Pipeline
============================
Research question: For a given tire compound at a given circuit, how does
lap pace decay as a function of tire age?

Approach: fit per-(compound, circuit) quadratic curves
  delta_pace(age) = a * age + b * age**2
on 2022-2023 green-lap data (fuel-corrected, baseline-subtracted), evaluate
on 2024 holdout.

Outputs:
  data/f1_degradation_dataset.csv       Filtered, fuel-corrected lap dataset
  models/f1_degradation_curves.pkl      joblib dict of fitted quadratics
  outputs/figures/degradation_curves.png  Three-panel curve overlay
"""

import sys
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib

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

FUEL_BURN_RATE  = 1.8
FUEL_LAP_EFFECT = 0.035
BASELINE_WINDOW = 3
MIN_LAPS_PER_CELL = 30
TRAIN_YEARS = [2022, 2023]
TEST_YEARS  = [2024]
ALL_YEARS   = TRAIN_YEARS + TEST_YEARS
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

DATASET_PATH = DATA_DIR  / 'f1_degradation_dataset.csv'
MODEL_PATH   = MODEL_DIR / 'f1_degradation_curves.pkl'

plt.style.use('seaborn-v0_8-darkgrid')


def is_green(track_status: str) -> bool:
    """Return True iff the lap has no SC/VSC flag."""
    if pd.isna(track_status):
        return False
    s = str(track_status)
    return not any(c in s for c in ['4', '5', '6'])


def extract_degradation_laps(session, year: int, circuit: str) -> pd.DataFrame:
    """
    Return one row per valid green lap with:
      year, circuit, driver, stint, compound, tire_age,
      lap_time_seconds, fuel_corrected_pace, delta_pace
    """
    laps = session.laps.copy()
    if laps is None or len(laps) == 0:
        return pd.DataFrame()

    # Drop laps without a lap time
    laps = laps[laps['LapTime'].notna()].copy()

    # Green flag only
    laps = laps[laps['TrackStatus'].apply(is_green)].copy()

    # Exclude pit-in, pit-out laps
    laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()].copy()

    # Need these columns populated
    laps = laps.dropna(subset=['Driver', 'Stint', 'Compound',
                               'TyreLife', 'LapNumber']).copy()

    laps['lap_time_seconds'] = laps['LapTime'].dt.total_seconds()
    laps = laps[(laps['lap_time_seconds'] >= 60) &
                (laps['lap_time_seconds'] <= 200)].copy()

    # Fuel correction
    laps['fuel_corrected_pace'] = (
        laps['lap_time_seconds']
        - (laps['LapNumber'] - 1) * FUEL_BURN_RATE * FUEL_LAP_EFFECT
    )

    # Per (driver, stint): baseline = median of first BASELINE_WINDOW green laps
    laps = laps.sort_values(['Driver', 'Stint', 'LapNumber']).copy()
    laps['_stint_rank'] = laps.groupby(['Driver', 'Stint']).cumcount() + 1
    baseline = (
        laps[laps['_stint_rank'] <= BASELINE_WINDOW]
        .groupby(['Driver', 'Stint'])['fuel_corrected_pace']
        .median()
        .rename('baseline_pace')
    )
    laps = laps.join(baseline, on=['Driver', 'Stint'])

    # Drop stints with no baseline (insufficient opening laps)
    laps = laps.dropna(subset=['baseline_pace']).copy()

    # Drop the baseline laps themselves so they don't bias the fit
    laps = laps[laps['_stint_rank'] > BASELINE_WINDOW].copy()

    laps['delta_pace'] = laps['fuel_corrected_pace'] - laps['baseline_pace']

    out = laps[['Driver', 'Stint', 'Compound', 'TyreLife', 'LapNumber',
                'lap_time_seconds', 'fuel_corrected_pace',
                'baseline_pace', 'delta_pace']].copy()
    out['year']    = year
    out['circuit'] = circuit
    out = out.rename(columns={
        'Driver':   'driver',
        'Stint':    'stint',
        'Compound': 'compound',
        'TyreLife': 'tire_age',
        'LapNumber': 'lap_number',
    })
    return out


def build_full_dataset(years):
    rows = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        gp_names = schedule['EventName'].tolist()
        print(f'\n-- {year}: {len(gp_names)} races --')
        for gp in gp_names:
            try:
                session = fastf1.get_session(year, gp, 'R')
                session.load(laps=True, telemetry=False,
                             weather=False, messages=False)
                df = extract_degradation_laps(session, year, gp)
                if df.empty:
                    print(f'  {gp:40s}  SKIP: no valid laps')
                    continue
                rows.append(df)
                print(f'  {gp:40s}  {len(df):4d} laps')
            except Exception as e:
                print(f'  {gp:40s}  FAILED: {str(e)[:80]}')
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, robust to zero-variance targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-9:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def fit_quadratic(df_group: pd.DataFrame) -> dict | None:
    """Fit delta_pace = a*age + b*age^2; return None if insufficient data."""
    x = df_group['tire_age'].to_numpy(dtype=float)
    y = df_group['delta_pace'].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < MIN_LAPS_PER_CELL:
        return None
    if np.ptp(x) < 1.0:
        return None
    # polyfit returns [c2, c1, c0]; we want [c1, c2] for a*age + b*age^2 (no intercept)
    # Fit with intercept free, then drop it to match the design spec shape.
    coefs = np.polyfit(x, y, 2)  # [b, a, c]
    b, a, _c = float(coefs[0]), float(coefs[1]), float(coefs[2])
    y_pred = a * x + b * x * x
    r2 = r2_score_manual(y, y_pred)
    return {'coef': [a, b], 'n_laps': int(len(x)), 'r2_train': r2}


def fit_all_curves(df_train: pd.DataFrame) -> dict:
    """
    Returns:
      {
        ('SOFT',   'Bahrain Grand Prix'): {coef:[a,b], n_laps, r2_train},
        ...
        ('SOFT',   '__GLOBAL__'):        {coef:[a,b], n_laps, r2_train},
        ...
      }
    The __GLOBAL__ entry per compound is the compound-wide fallback for
    holdout circuits not seen in training (or skipped for sample size).
    """
    curves = {}
    df_train = df_train[df_train['compound'].isin(DRY_COMPOUNDS)].copy()

    # Per (compound, circuit)
    for (compound, circuit), grp in df_train.groupby(['compound', 'circuit']):
        fit = fit_quadratic(grp)
        if fit is None:
            continue
        curves[(compound, circuit)] = fit

    # Per compound fallback (pool across all circuits)
    for compound, grp in df_train.groupby('compound'):
        fit = fit_quadratic(grp)
        if fit is None:
            continue
        curves[(compound, '__GLOBAL__')] = fit

    return curves


def lookup_curve(curves: dict, compound: str, circuit: str) -> dict | None:
    if (compound, circuit) in curves:
        return curves[(compound, circuit)]
    if (compound, '__GLOBAL__') in curves:
        return curves[(compound, '__GLOBAL__')]
    return None


def predict_delta_pace(curve: dict, tire_age: float) -> float:
    a, b = curve['coef']
    return a * tire_age + b * tire_age * tire_age


if __name__ == '__main__':
    rebuild = '--rebuild' in sys.argv

    if DATASET_PATH.exists() and not rebuild:
        print('Loading cached dataset...  (pass --rebuild to regenerate)')
        df = pd.read_csv(DATASET_PATH)
    else:
        if rebuild:
            print('--rebuild: ignoring cache, pulling from FastF1...')
        else:
            print('Building dataset -- this will take several minutes...')
        df = build_full_dataset(ALL_YEARS)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f'\nSaved: {DATASET_PATH}')

    print(f'\nDataset shape: {df.shape}')
    print(f'Years covered: {sorted(df["year"].unique())}')
    print(f'Compound distribution:')
    print(df['compound'].value_counts().to_string())

    # Fit curves on training years only
    df_train = df[df['year'].isin(TRAIN_YEARS)].copy()
    print(f'\nTraining observations: {len(df_train):,}')

    curves = fit_all_curves(df_train)
    print(f'Fitted {len(curves)} (compound, circuit) curves '
          f'(includes compound-global fallbacks).')

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'curves': curves,
                 'train_years': TRAIN_YEARS,
                 'test_years':  TEST_YEARS}, MODEL_PATH)
    print(f'Saved: {MODEL_PATH}')

    # Print a compact training R2 summary
    rows = []
    for (compound, circuit), fit in curves.items():
        rows.append({'compound': compound, 'circuit': circuit,
                     'n_laps': fit['n_laps'],
                     'r2_train': round(fit['r2_train'], 3)})
    summary = pd.DataFrame(rows).sort_values(['compound', 'circuit'])
    print('\nTraining fit summary (top 15 rows):')
    print(summary.head(15).to_string(index=False))
