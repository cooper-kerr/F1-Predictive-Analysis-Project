"""
F1 Tire Degradation Pipeline
============================
Research question: how does tire pace evolve across a stint, and what does
that tell us about compound-specific peak-grip windows and pit timing?

Approach: per-lap green-flag lap dataset with fuel correction and
peak-pace baseline (= median of the stint's fastest 3 fuel-corrected laps).
`delta_pace` is lost time versus that peak. Three summary analyses:

  (1) Pace profile across a stint — per-compound mean delta_pace by tire-age
      bucket, 2022-2023 training profile compared to 2024 holdout. Exposes
      the U-shape: warm-up slow, peak in the middle, degradation at the end.

  (2) Peak-pace age by compound — the tire age at which each stint's
      fastest lap occurs. Strategic input: when is this tire at its best?

  (3) Stint-length vs degradation magnitude — for each stint, the difference
      between worst-3-laps and best-3-laps fuel-corrected pace; correlated
      with stint length to size the pit-timing tradeoff.

Outputs:
  data/f1_degradation_dataset.csv       Per-lap dataset (used by §5.1 notebook)
  models/f1_degradation_analysis.pkl    Summary dict of all three analyses
  outputs/figures/degradation_pace_profile.png
  outputs/figures/peak_age_by_compound.png
  outputs/figures/stint_length_vs_degradation.png
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

from dataset_runner import build_schedule_dataset

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
TRAIN_YEARS = [2022, 2023]
TEST_YEARS  = [2024]
ALL_YEARS   = TRAIN_YEARS + TEST_YEARS
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

DATASET_PATH = DATA_DIR  / 'f1_degradation_dataset.csv'
MODEL_PATH   = MODEL_DIR / 'f1_degradation_analysis.pkl'

AGE_BINS   = [0, 5, 10, 15, 20, 25, 30, 60]
AGE_LABELS = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
MIN_STINT_LAPS   = 8    # for stint summaries (peak age, degradation)
MIN_CELL_STINTS  = 5    # for per-(compound, circuit) stint aggregates

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

    # Per (driver, stint): baseline = median of the fastest BASELINE_WINDOW
    # fuel-corrected laps of the stint (peak-pace anchor, avoids tire warm-up
    # contamination that biases a first-N-laps baseline). Requires the stint
    # to have at least BASELINE_WINDOW laps.
    laps = laps.sort_values(['Driver', 'Stint', 'LapNumber']).copy()
    stint_sizes = laps.groupby(['Driver', 'Stint']).size()
    valid_stints = stint_sizes[stint_sizes >= BASELINE_WINDOW].index
    laps = laps.set_index(['Driver', 'Stint']).loc[valid_stints].reset_index()

    baseline = (
        laps.groupby(['Driver', 'Stint'])['fuel_corrected_pace']
        .apply(lambda s: s.nsmallest(BASELINE_WINDOW).median())
        .rename('baseline_pace')
    )
    laps = laps.join(baseline, on=['Driver', 'Stint'])
    laps = laps.dropna(subset=['baseline_pace']).copy()

    # delta_pace = lost time vs peak (>= 0 for non-peak laps, ~0 at peak).
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
    return build_schedule_dataset(
        years,
        extract_degradation_laps,
        'laps',
        schedule_loader=fastf1.get_event_schedule,
        session_loader=fastf1.get_session,
        session_load_kwargs={
            'laps': True,
            'telemetry': False,
            'weather': False,
            'messages': False,
        },
        skip_empty_extractions=True,
        empty_extraction_message='no valid laps',
    )


def compute_stint_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-lap dataset to one row per (year, circuit, driver, stint).

    Columns returned:
      year, circuit, driver, stint, compound, stint_length,
      peak_age          -- tire age of the fastest fuel-corrected lap
      best_pace, worst_pace
      degradation_s     -- median(worst 3 fuel-corrected laps)
                           − median(best 3 fuel-corrected laps)

    Only stints with >= MIN_STINT_LAPS green laps are included, so that both
    peak_age and degradation_s are stable.
    """
    df = df[df['compound'].isin(DRY_COMPOUNDS)].copy()
    sizes = df.groupby(['year', 'circuit', 'driver', 'stint']).size()
    keep = sizes[sizes >= MIN_STINT_LAPS].index
    df = df.set_index(['year', 'circuit', 'driver', 'stint']).loc[keep].reset_index()

    rows = []
    for (year, circuit, driver, stint), grp in df.groupby(
            ['year', 'circuit', 'driver', 'stint']):
        fcp = grp['fuel_corrected_pace'].to_numpy()
        ages = grp['tire_age'].to_numpy()
        best_idx = int(np.argmin(fcp))
        best_3  = float(np.median(np.sort(fcp)[:3]))
        worst_3 = float(np.median(np.sort(fcp)[-3:]))
        rows.append({
            'year':         int(year),
            'circuit':      circuit,
            'driver':       driver,
            'stint':        int(stint),
            'compound':     grp['compound'].iloc[0],
            'stint_length': int(len(grp)),
            'peak_age':     int(ages[best_idx]),
            'best_pace':    best_3,
            'worst_pace':   worst_3,
            'degradation_s': worst_3 - best_3,
        })
    return pd.DataFrame(rows)


def compute_pace_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket laps by (compound, era, tire-age bin) and return the mean
    delta_pace in each cell. `era` ∈ {'2022-2023', '2024'}.
    """
    df = df[df['compound'].isin(DRY_COMPOUNDS)].copy()
    df['era'] = np.where(df['year'].isin(TRAIN_YEARS), '2022-2023', '2024')
    df['age_bin'] = pd.cut(df['tire_age'], bins=AGE_BINS, labels=AGE_LABELS,
                           right=False, include_lowest=True)
    grouped = (df.groupby(['compound', 'era', 'age_bin'], observed=True)
                 ['delta_pace']
                 .agg(['mean', 'median', 'count'])
                 .reset_index()
                 .rename(columns={'mean': 'mean_delta_s',
                                  'median': 'median_delta_s',
                                  'count': 'n_laps'}))
    return grouped


def compute_peak_age_stats(stint_summaries: pd.DataFrame) -> pd.DataFrame:
    """Per-compound central tendency of peak_age, split by era."""
    s = stint_summaries.copy()
    s['era'] = np.where(s['year'].isin(TRAIN_YEARS), '2022-2023', '2024')
    rows = []
    for (compound, era), grp in s.groupby(['compound', 'era']):
        rows.append({
            'compound':   compound,
            'era':        era,
            'n_stints':   int(len(grp)),
            'median_peak_age': float(grp['peak_age'].median()),
            'mean_peak_age':   round(float(grp['peak_age'].mean()), 2),
            'std_peak_age':    round(float(grp['peak_age'].std(ddof=1)), 2),
        })
    return pd.DataFrame(rows)


def compute_stint_length_correlation(stint_summaries: pd.DataFrame) -> pd.DataFrame:
    """Per-compound Pearson correlation between stint_length and degradation_s."""
    rows = []
    for compound, grp in stint_summaries.groupby('compound'):
        if len(grp) < MIN_CELL_STINTS:
            continue
        x = grp['stint_length'].to_numpy(dtype=float)
        y = grp['degradation_s'].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < MIN_CELL_STINTS or np.std(x[mask]) == 0:
            continue
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        # Linear fit for plotting / reporting
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        rows.append({
            'compound':        compound,
            'n_stints':        int(mask.sum()),
            'pearson_r':       round(r, 3),
            'slope_s_per_lap': round(float(slope), 3),
            'intercept_s':     round(float(intercept), 3),
        })
    return pd.DataFrame(rows)


COMPOUND_COLOR = {'SOFT': '#e74c3c', 'MEDIUM': '#f1c40f', 'HARD': '#2c3e50'}


def plot_pace_profile(profile: pd.DataFrame, out_path: Path) -> None:
    """U-shape: mean delta_pace per age-bucket, per compound, per era."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, era in zip(axes, ['2022-2023', '2024']):
        era_df = profile[profile['era'] == era]
        for compound in DRY_COMPOUNDS:
            sub = (era_df[era_df['compound'] == compound]
                   .sort_values('age_bin'))
            if sub.empty:
                continue
            x = np.arange(len(sub))
            ax.plot(x, sub['mean_delta_s'].to_numpy(),
                    marker='o', linewidth=2,
                    color=COMPOUND_COLOR[compound],
                    label=compound)
            ax.set_xticks(x)
            ax.set_xticklabels(sub['age_bin'].astype(str), rotation=0)
        ax.set_title(f'{era} green laps')
        ax.set_xlabel('Tire age bucket (laps)')
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.legend(loc='upper left', fontsize=9, title='Compound')
    axes[0].set_ylabel('Mean delta vs peak pace (s)')
    fig.suptitle('Tire pace profile across a stint '
                 '— U-shape: warm-up → peak → degradation',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_peak_age_distribution(stint_summaries: pd.DataFrame,
                               out_path: Path) -> None:
    """Per-compound box plot of peak_age (stint-level)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [stint_summaries.loc[stint_summaries['compound'] == c, 'peak_age']
                            .to_numpy()
            for c in DRY_COMPOUNDS]
    bp = ax.boxplot(data, labels=DRY_COMPOUNDS, patch_artist=True,
                    showmeans=True, widths=0.55)
    for patch, compound in zip(bp['boxes'], DRY_COMPOUNDS):
        patch.set_facecolor(COMPOUND_COLOR[compound])
        patch.set_alpha(0.75)
    medians = [float(np.median(d)) for d in data]
    for i, m in enumerate(medians, start=1):
        ax.annotate(f'median {m:.0f}', xy=(i, m),
                    xytext=(i + 0.18, m),
                    fontsize=9, va='center', color='black')
    ax.set_ylabel('Tire age of fastest lap (laps)')
    ax.set_xlabel('Compound')
    ax.set_title('Peak-pace tire age by compound (per stint, all years)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_stint_length_vs_degradation(stint_summaries: pd.DataFrame,
                                     corr_table: pd.DataFrame,
                                     out_path: Path) -> None:
    """Three-panel scatter with per-compound regression line."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    corr_lookup = corr_table.set_index('compound').to_dict('index')
    for ax, compound in zip(axes, DRY_COMPOUNDS):
        sub = stint_summaries[stint_summaries['compound'] == compound]
        ax.scatter(sub['stint_length'], sub['degradation_s'],
                   s=14, alpha=0.35,
                   color=COMPOUND_COLOR[compound],
                   edgecolor='none')
        if compound in corr_lookup:
            info = corr_lookup[compound]
            xgrid = np.linspace(sub['stint_length'].min(),
                                sub['stint_length'].max(), 50)
            ygrid = info['intercept_s'] + info['slope_s_per_lap'] * xgrid
            ax.plot(xgrid, ygrid, color='black', linewidth=1.8,
                    linestyle='--',
                    label=f"r={info['pearson_r']:+.2f}, "
                          f"slope={info['slope_s_per_lap']:+.2f}s/lap")
            ax.legend(loc='upper left', fontsize=9)
        ax.set_title(compound)
        ax.set_xlabel('Stint length (laps)')
        if compound == 'SOFT':
            ax.set_ylabel('Degradation (worst-3 − best-3, s)')
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
    fig.suptitle('Stint length vs. end-of-stint degradation magnitude',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


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

    # (1) Stint-level summaries
    stint_summaries = compute_stint_summaries(df)
    print(f'\nStint summaries: {len(stint_summaries):,} stints '
          f'(>= {MIN_STINT_LAPS} green laps each)')
    print(stint_summaries.groupby('compound')['stint_length']
          .agg(['count', 'mean', 'median', 'max'])
          .round(2).to_string())

    # (2) Pace profile
    profile = compute_pace_profile(df)

    # (3) Peak age + stint length correlation
    peak_stats = compute_peak_age_stats(stint_summaries)
    print('\n=== Peak-pace tire age (by compound, era) ===')
    print(peak_stats.to_string(index=False))

    corr_table = compute_stint_length_correlation(stint_summaries)
    print('\n=== Stint length vs degradation (Pearson r, per compound) ===')
    print(corr_table.to_string(index=False))

    # Save artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'stint_summaries':       stint_summaries,
        'pace_profile':          profile,
        'peak_age_stats':        peak_stats,
        'stint_length_corr':     corr_table,
        'age_bins':              AGE_BINS,
        'age_labels':            AGE_LABELS,
        'train_years':           TRAIN_YEARS,
        'test_years':            TEST_YEARS,
    }, MODEL_PATH)
    print(f'\nSaved: {MODEL_PATH}')

    # Figures
    plot_pace_profile(profile,
                      FIGURES_DIR / 'degradation_pace_profile.png')
    plot_peak_age_distribution(stint_summaries,
                               FIGURES_DIR / 'peak_age_by_compound.png')
    plot_stint_length_vs_degradation(stint_summaries, corr_table,
                                     FIGURES_DIR / 'stint_length_vs_degradation.png')
    print(f'\nFigures saved to {FIGURES_DIR}/')
    print('  - degradation_pace_profile.png')
    print('  - peak_age_by_compound.png')
    print('  - stint_length_vs_degradation.png')
    print('\nDone.')
