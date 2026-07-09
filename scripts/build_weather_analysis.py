"""
F1 Weather / Temperature Analysis
=================================
Script-backed version of notebook 06.

Research question: how do weather and temperature variables relate to race
lap time after controlling for compound, tire life, race progression, race,
and year?

Approach:
  - Load the notebook 06 race/year scope: six races across 2021-2025
  - Merge lap-level timing with session weather observations by timestamp
  - Remove pit-in/pit-out laps and undefined compounds
  - Fit the notebook 06 interaction OLS formula with HC2 robust standard errors
  - Save a compact coefficient table, artifact dict, and a scatter plot

Outputs:
  data/f1_weather_dataset.csv
  data/f1_weather_coefficients.csv
  models/f1_weather_analysis.pkl
  outputs/figures/weather_laptime_scatter.png
"""

import sys
import warnings
from pathlib import Path

import fastf1
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent.parent
CACHE_DIR   = ROOT / 'f1-cache'
DATA_DIR    = ROOT / 'data'
MODEL_DIR   = ROOT / 'models'
FIGURES_DIR = ROOT / 'outputs' / 'figures'

fastf1.Cache.enable_cache(str(CACHE_DIR))

RACES = ['United States', 'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China']
YEARS = [2021, 2022, 2023, 2024, 2025]

DATASET_PATH = DATA_DIR / 'f1_weather_dataset.csv'
COEF_PATH    = DATA_DIR / 'f1_weather_coefficients.csv'
MODEL_PATH   = MODEL_DIR / 'f1_weather_analysis.pkl'
FIGURE_PATH  = FIGURES_DIR / 'weather_laptime_scatter.png'

plt.style.use('seaborn-v0_8-darkgrid')


def build_full_dataset() -> pd.DataFrame:
    frames = []
    for race in RACES:
        for year in YEARS:
            try:
                session = fastf1.get_session(year, race, 'R')
                session.load(telemetry=False, weather=True, messages=False)

                laps = session.laps.copy()
                weather = session.weather_data.copy()
                if laps.empty or weather.empty:
                    print(f'{year} {race:20s}  SKIP: missing laps/weather')
                    continue

                laps = laps[laps['LapTime'].notna()].copy()
                if laps.empty:
                    print(f'{year} {race:20s}  SKIP: no timed laps')
                    continue

                laps['Race'] = race
                laps['Year'] = year
                laps = laps.sort_values('Time')
                weather = weather.sort_values('Time')

                merged = pd.merge_asof(laps, weather, on='Time', direction='backward')
                merged['LapTime_Seconds'] = merged['LapTime'].dt.total_seconds()
                merged = merged.dropna(subset=[
                    'LapTime_Seconds', 'TrackTemp', 'WindSpeed', 'Rainfall',
                    'TyreLife', 'LapNumber', 'Compound'
                ]).copy()
                merged = merged[
                    merged['PitInTime'].isna() &
                    merged['PitOutTime'].isna() &
                    (merged['Compound'] != 'None')
                ].copy()

                frames.append(merged)
                print(f'{year} {race:20s}  {len(merged):4d} rows')
            except Exception as exc:
                print(f'{year} {race:20s}  FAILED: {str(exc)[:80]}')

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fit_weather_model(df: pd.DataFrame):
    model = smf.ols(
        formula='''LapTime_Seconds ~
                    + C(Compound)
                    + TrackTemp: C(Compound) - 1
                    + TyreLife
                    + WindSpeed
                    + Rainfall
                    + LapNumber
                    + C(Year)
                    + C(Race)''',
        data=df,
    ).fit(cov_type='HC2')
    return model


def coefficient_table(model) -> pd.DataFrame:
    tbl = pd.DataFrame({
        'term': model.params.index,
        'coef': model.params.values,
        'std_err': model.bse.values,
        'p_value': model.pvalues.values,
    })
    tbl['coef'] = tbl['coef'].round(4)
    tbl['std_err'] = tbl['std_err'].round(4)
    tbl['p_value'] = tbl['p_value'].round(4)
    return tbl


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for race, group in df[df['Year'] == 2024].groupby('Race'):
        ax.scatter(group['TrackTemp'], group['LapTime_Seconds'],
                   label=race, alpha=0.45, s=14)

    ax.set_title('2024 Canonical Weather Sample by Track Temperature and Lap Time',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Track Temperature')
    ax.set_ylabel('Lap Time (seconds)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    rebuild = '--rebuild' in sys.argv

    if DATASET_PATH.exists() and not rebuild:
        print('Loading cached weather dataset... (pass --rebuild to regenerate)')
        df = pd.read_csv(DATASET_PATH)
    else:
        print('Building weather dataset from FastF1...')
        df = build_full_dataset()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f'Saved: {DATASET_PATH}')

    if df.empty:
        raise RuntimeError('Weather dataset is empty')

    model = fit_weather_model(df)
    coef_df = coefficient_table(model)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    coef_df.to_csv(COEF_PATH, index=False)
    plot_scatter(df, FIGURE_PATH)

    joblib.dump({
        'model': model,
        'races': RACES,
        'years': YEARS,
        'n_rows': int(len(df)),
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        'coefficients': coef_df,
        'figure_path': str(FIGURE_PATH),
    }, MODEL_PATH)

    print(f'\nRows: {len(df):,}')
    print(f'R-squared: {model.rsquared:.3f}')
    print(f'Saved: {COEF_PATH}')
    print(f'Saved: {MODEL_PATH}')
    print(f'Saved: {FIGURE_PATH}')
