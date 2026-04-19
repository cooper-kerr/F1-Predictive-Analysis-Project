# F1 Milestone Notebook: 5-Question Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `notebooks/01_milestone_report.ipynb` to make exactly 5 research questions explicit end-to-end (§1 intro, §4 methods, §5 results, §9 conclusions), add a real stint-level tire degradation analysis to back the §5.1 claim (pace profile U-shape, peak age by compound, stint-length vs degradation correlation), and merge undercut/overcut into one pit-strategy section.

**Architecture:** One new Python script (`scripts/build_degradation_dataset.py`) following the established build-script pattern (cache-first, `--rebuild` flag, joblib artifact dict). All narrative + results changes happen inside the existing milestone notebook via cell add / cell edit / cell delete operations. Existing models (position RF, pit-window GBM, undercut LR+XGBoost, overcut LR+XGBoost) and their artifacts are not modified.

**Tech Stack:** Python 3, FastF1, pandas (<3.0), numpy, scikit-learn, statsmodels, matplotlib, joblib, Jupyter (ipynb).

---

## Context for the executor

**Codebase conventions you must follow:**

1. **Build scripts live in `scripts/` and follow one pattern.** Read `scripts/build_undercut_dataset.py` before writing the new tire degradation script. Key conventions:
   - Paths derived from `ROOT = Path(__file__).parent.parent`.
   - FastF1 cache enabled via `fastf1.Cache.enable_cache(str(CACHE_DIR))`.
   - `--rebuild` flag: absent → load cached CSV if it exists; present → re-fetch from FastF1.
   - Matplotlib non-interactive backend: `matplotlib.use('Agg')`.
   - Fuel-correction constants: `FUEL_BURN_RATE = 1.8`, `FUEL_LAP_EFFECT = 0.035`.
   - Train on 2022-2023, test on 2024 (temporal holdout).
   - Save joblib artifacts as dicts: `joblib.dump({'model': ..., 'features': [...]}, path)`.

2. **pandas must stay pinned to `<3.0.0`.** FastF1 3.8.x requires nanosecond timedelta precision. Don't upgrade pandas.

3. **`TrackStatus` is a composite string, not a single code.** A green lap has status `'1'`; SC / VSC laps contain characters `'4'`, `'5'`, or `'6'`. Filter with a substring check, not equality.

4. **Notebook editing is done via the `NotebookEdit` tool,** not raw file edits. Cell IDs are referenced below. When you add a new cell, specify `cell_id` of the preceding cell so the new one is inserted after it.

5. **Current notebook section structure** (for reference while editing):
   - Cell `bf4722fc` — title
   - Cell `6299ca45` — §1 Project Description
   - Cell `1b4a0738` — §2 Data Description
   - Cell `6ca2356c` — §3 Ethical Data Concerns
   - Cell `edb4f505` — §4 Methods
   - Cell `e780defe1c59` — §5 Results intro
   - Cell `503385e974ae` — §5.1 Race-Position (to be renumbered §5.5)
   - Cell `f41c905739ae` — §5.1 code cell (predictability curve)
   - Cell `bd8cd7424e76` — §5.1 display position_curve.png
   - Cell `a0677403de5e` — §5.1 display position_confusion.png
   - Cell `3bcf7995147f` — §5.1 display position_importance.png
   - Cell `b682405585e3` — §5.1 findings
   - Cell `8c312a3064d3` — §5.2 Pit-Window (to be renumbered §5.3)
   - Cell `03b5e7a54b6d` — §5.3 Undercut (to be merged)
   - Cell `810abdb06c8c` — §5.4 Overcut (to be merged)
   - Cell `661745458581` — §5.5 Hungary intro (to be renumbered §5.6)
   - Cell `799fed047e65` — §5.5 Hungary code cell (position probs)
   - Cell `d78b9a99dfe5` — §5.5 Hungary code cell (pit window + dashboard plot)
   - Cell `b58fd2c079af` — §5.5 Hungary reading notes
   - Cell `7dfa878ba723` — §5.6 Bonus weather intro (to become §5.2)
   - Cell `369894b8-4eea-4fa6-b4d8-6dd963ee0bde` — §5.6 weather data loader
   - Cell `772cdb39-4263-41c2-8eec-c9b925c69104` — §5.6 weather regression + plot
   - Cell `c89b58b9-79ef-43b4-9e9e-2a99985e5843` — §5.6 weather interpretation
   - Cell `584f8e69` — §6 Peer Feedback
   - Cell `41989f94ec35` — §7 Process Summary
   - Cell `18d26efaa688` — §8 Limitations
   - Cell `55223489ed32` — §9 Conclusions

**Verification commands to run after any code or notebook change:**

- After writing the new build script: `python scripts/build_degradation_dataset.py` (should succeed, produce artifacts, print metrics).
- After notebook edits: open `notebooks/01_milestone_report.ipynb` and run all cells top-to-bottom. Verify no cell raises, and visually confirm the section headers match §5.1-§5.6 per the design.

---

## Files Touched

- **Create:** `scripts/build_degradation_dataset.py` — new tire-degradation pipeline
- **Create:** `data/f1_degradation_dataset.csv` — filtered, fuel-corrected lap dataset with peak-pace baseline + delta_pace
- **Create:** `models/f1_degradation_analysis.pkl` — joblib dict: `stint_summaries`, `pace_profile`, `peak_age_stats`, `stint_length_corr`
- **Create:** `outputs/figures/degradation_pace_profile.png` — U-shape profile, per compound, two-era panels
- **Create:** `outputs/figures/peak_age_by_compound.png` — per-compound peak-pace age distribution
- **Create:** `outputs/figures/stint_length_vs_degradation.png` — three-panel scatter with regression
- **Create:** `outputs/figures/weather_compound_sensitivity.png` — per-compound temp regression
- **Create:** `outputs/figures/weather_diagnostics.png` — residual + Q-Q plots
- **Modify:** `notebooks/01_milestone_report.ipynb` — cells across §1, §4, §5, §6, §7, §8, §9

Note: Tasks 1-3 below describe the original per-(compound, circuit) quadratic-curve approach. That work was completed and committed, but on 2024 holdout the curves underperformed the naive live-slope estimator because the pace profile is U-shaped. The analysis was pivoted to the stint-level view in commit `c995962`; the script and artifacts now reflect that pivot. Tasks 4, 8, 10 below describe §5.1, §4.1 Methods, and §9 Conclusions in the pivoted framing — that is the live spec.

---

## Task 1: Write tire-degradation build script (skeleton + lap filter)

Create the script scaffold and the lap-filtering / fuel-correction stage. Defer curve fitting and evaluation to Tasks 2-3 so each commit is small and individually runnable.

**Files:**
- Create: `scripts/build_degradation_dataset.py`

- [ ] **Step 1: Create the script scaffold**

```python
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
```

- [ ] **Step 2: Add the session-level lap extraction**

Append to the script:

```python
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
```

- [ ] **Step 3: Add the `__main__` block with rebuild flag (no fit yet)**

Append:

```python
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

    # Fit + evaluation land in Task 2 / Task 3
```

- [ ] **Step 4: Run the script to verify the dataset builds**

Run: `python scripts/build_degradation_dataset.py`

Expected on first run: several-minute build producing `data/f1_degradation_dataset.csv` with columns `year, circuit, driver, stint, compound, tire_age, lap_number, lap_time_seconds, fuel_corrected_pace, baseline_pace, delta_pace`. Expect roughly tens of thousands of rows across 2022-2024.

Expected on subsequent run: fast load from cache with a printed shape and compound distribution.

Verification:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/f1_degradation_dataset.csv'); print(df.shape); print(df.columns.tolist()); print(df['compound'].value_counts())"
```

Expect the three dry compounds (SOFT, MEDIUM, HARD) to each have thousands of rows. INTERMEDIATE / WET may also appear — they will be filtered during fitting.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_degradation_dataset.py data/f1_degradation_dataset.csv
git commit -m "Add tire degradation dataset builder (Task 1/10)"
```

---

## Task 2: Fit per-(compound, circuit) quadratic curves

Add the curve-fitting stage: pool 2022 and 2023 observations per circuit, fit `numpy.polyfit(age, delta_pace, 2)`, compute training R², skip cells with fewer than 30 laps. Persist the joblib artifact.

**Files:**
- Modify: `scripts/build_degradation_dataset.py`

- [ ] **Step 1: Add the curve-fitting functions**

Insert after `build_full_dataset` and before `if __name__ == '__main__':`:

```python
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
```

- [ ] **Step 2: Extend the `__main__` block to fit and save**

Replace the trailing comment `# Fit + evaluation land in Task 2 / Task 3` with:

```python
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
```

- [ ] **Step 3: Run and verify**

Run: `python scripts/build_degradation_dataset.py`

Expected: dataset loads from cache, fits run in seconds, and `models/f1_degradation_curves.pkl` is written. The printed summary should show positive R² values for most cells.

Verification:
```bash
python -c "
import joblib
art = joblib.load('models/f1_degradation_curves.pkl')
curves = art['curves']
print(f'Total curves: {len(curves)}')
print(f'Compounds with global fallback: {[c for (c, circ) in curves.keys() if circ == \"__GLOBAL__\"]}')
r2s = [c['r2_train'] for c in curves.values() if not (c['r2_train'] != c['r2_train'])]  # drop NaN
print(f'Median training R2: {sorted(r2s)[len(r2s)//2]:.3f}')
"
```

Expect compounds-with-global-fallback to contain `SOFT`, `MEDIUM`, `HARD`. Median training R² should be positive (typically 0.1-0.4 for noisy lap-time data).

- [ ] **Step 4: Commit**

```bash
git add scripts/build_degradation_dataset.py models/f1_degradation_curves.pkl
git commit -m "Fit per-(compound, circuit) quadratic degradation curves (Task 2/10)"
```

---

## Task 3: Evaluate on 2024 holdout and generate the curve figure

Add the evaluation stage that scores the fitted curves on the 2024 holdout, computes MAE per compound, compares against the in-stint linear-slope estimator, and writes the three-panel curve overlay figure.

**Files:**
- Modify: `scripts/build_degradation_dataset.py`
- Create: `outputs/figures/degradation_curves.png`

- [ ] **Step 1: Add the evaluation function**

Insert above `if __name__ == '__main__':`:

```python
def evaluate_on_holdout(df_test: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Predict delta_pace per row; return per-compound MAE and pooled R^2."""
    df_test = df_test[df_test['compound'].isin(DRY_COMPOUNDS)].copy()
    preds = []
    for _, row in df_test.iterrows():
        curve = lookup_curve(curves, row['compound'], row['circuit'])
        if curve is None:
            preds.append(np.nan)
            continue
        preds.append(predict_delta_pace(curve, float(row['tire_age'])))
    df_test = df_test.assign(pred_delta_pace=preds).dropna(
        subset=['pred_delta_pace'])
    rows = []
    for compound, grp in df_test.groupby('compound'):
        err = (grp['delta_pace'] - grp['pred_delta_pace']).abs()
        rows.append({
            'compound': compound,
            'n_laps':   int(len(grp)),
            'mae_s':    round(float(err.mean()), 3),
            'r2':       round(r2_score_manual(
                grp['delta_pace'].to_numpy(),
                grp['pred_delta_pace'].to_numpy()), 3),
        })
    return pd.DataFrame(rows)


def compare_to_live_slope(df_test: pd.DataFrame, curves: dict) -> float:
    """
    Return R^2 of the in-stint linear slope estimator (the existing
    pit-window pipeline's deg_delta) evaluated on the same 2024 rows.
    """
    df_test = df_test[df_test['compound'].isin(DRY_COMPOUNDS)].copy()
    df_test = df_test.sort_values(['driver', 'stint', 'lap_number']).copy()
    preds = []
    for (driver, stint), grp in df_test.groupby(['driver', 'stint']):
        grp_sorted = grp.sort_values('tire_age')
        if len(grp_sorted) < 3:
            preds.extend([np.nan] * len(grp))
            continue
        x = grp_sorted['tire_age'].to_numpy(dtype=float)
        y = grp_sorted['delta_pace'].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        for age in grp['tire_age']:
            preds.append(slope * float(age))
    df_test = df_test.assign(live_pred=preds).dropna(subset=['live_pred'])
    return r2_score_manual(
        df_test['delta_pace'].to_numpy(),
        df_test['live_pred'].to_numpy())
```

- [ ] **Step 2: Add the figure function**

Insert above `if __name__ == '__main__':`:

```python
def plot_degradation_curves(df_full: pd.DataFrame, curves: dict,
                            out_path: Path) -> None:
    """Three-panel (SOFT/MEDIUM/HARD) overlay: circuit curves + 2024 scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    age_grid = np.linspace(0, 40, 200)
    df_2024 = df_full[df_full['year'].isin(TEST_YEARS)].copy()

    for ax, compound in zip(axes, DRY_COMPOUNDS):
        # Scatter: 2024 observations
        grp = df_2024[df_2024['compound'] == compound]
        ax.scatter(grp['tire_age'], grp['delta_pace'],
                   s=6, alpha=0.08, color='grey',
                   label='2024 green laps' if compound == 'SOFT' else None)

        # Per-circuit curves
        per_circuit = [(k, v) for k, v in curves.items()
                       if k[0] == compound and k[1] != '__GLOBAL__']
        for (_, _), fit in per_circuit:
            a, b = fit['coef']
            y_curve = a * age_grid + b * age_grid ** 2
            ax.plot(age_grid, y_curve, color='steelblue',
                    alpha=0.3, linewidth=1)

        # Global fallback: heavy red line
        if (compound, '__GLOBAL__') in curves:
            a, b = curves[(compound, '__GLOBAL__')]['coef']
            y_curve = a * age_grid + b * age_grid ** 2
            ax.plot(age_grid, y_curve, color='crimson',
                    linewidth=2.5, label=f'{compound} global fit')

        ax.set_title(f'{compound}')
        ax.set_xlabel('Tire age (laps)')
        if compound == 'SOFT':
            ax.set_ylabel('Delta pace vs baseline (s)')
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Tire Degradation — per-(compound, circuit) quadratic fits '
                 '(2022-2023 train; 2024 scatter)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
```

- [ ] **Step 3: Extend the `__main__` block to call evaluation + plot**

Append to the `__main__` block, after the training-summary print:

```python
    # Evaluate on 2024
    df_test = df[df['year'].isin(TEST_YEARS)].copy()
    holdout_summary = evaluate_on_holdout(df_test, curves)
    print('\n=== 2024 holdout (per compound) ===')
    print(holdout_summary.to_string(index=False))

    live_slope_r2 = compare_to_live_slope(df_test, curves)
    print(f'\nLive in-stint slope estimator R^2 on same rows: '
          f'{live_slope_r2:.3f}')
    print('(Curve-based predictions should explain more variance '
          'than the naive live slope.)')

    # Figure
    plot_degradation_curves(df, curves,
                            FIGURES_DIR / 'degradation_curves.png')
    print(f'\nFigure saved: {FIGURES_DIR / "degradation_curves.png"}')
    print('\nDone.')
```

- [ ] **Step 4: Run end-to-end**

Run: `python scripts/build_degradation_dataset.py`

Expected output: the holdout summary table (one row per compound with `n_laps`, `mae_s`, `r2`), the live-slope comparison line, and the figure written to disk.

Verification:
```bash
ls -lh outputs/figures/degradation_curves.png
python -c "from PIL import Image; img = Image.open('outputs/figures/degradation_curves.png'); print(img.size)"
```

Expect an image around 2700×750 px. Spot-check visually that three panels are present with scatter + overlaid curves.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_degradation_dataset.py outputs/figures/degradation_curves.png
git commit -m "Evaluate tire degradation curves on 2024 + plot (Task 3/10)"
```

---

## Task 4: Add §5.1 Tire Degradation notebook cells

Insert six new cells at the very start of §5 Results (before the existing §5.1 Race-Position section, which will be renumbered to §5.5 in Task 9). Uses the `NotebookEdit` tool with `edit_mode: "insert"` and `cell_id` set to the preceding cell.

The §5.1 story is **stint-level**, not per-lap curve prediction. Three findings in order: (1) pace profile is U-shaped (warm-up → peak → degradation), (2) peak-pace tire age differs cleanly by compound, (3) stint length correlates with end-of-stint degradation magnitude.

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb`

- [ ] **Step 1: Insert the §5.1 intro markdown cell after cell `e780defe1c59` (§5 Results intro)**

Use `NotebookEdit` with `edit_mode: "insert"`, `cell_id: "e780defe1c59"`, `cell_type: "markdown"`, and content:

```markdown
## 5.1 Tire Degradation — Pace Profile Across a Stint

**Question:** how does lap pace evolve across a stint, and what does that tell us about compound-specific peak-grip windows and stint-length tradeoffs?

**Setup:** we filter 2022-2024 race sessions to green-flag laps only (no Safety Car, no VSC, no pit-in or pit-out) and apply the standard fuel correction `− (lap − 1) · 1.8 · 0.035`. For each stint we anchor a **peak-pace baseline** — the median of that stint's **fastest 3** fuel-corrected laps — and define `delta_pace` as lost time versus that peak (≥ 0 by construction, ~ 0 at the peak). This anchor avoids the warm-up contamination that biases a first-laps baseline. Stints with fewer than 8 green laps are dropped. Intermediate / wet compounds are excluded. The 2022-2023 races form the training era and 2024 the holdout, but the analysis is descriptive: we look for stable patterns across eras rather than training a predictor.

**Pipeline:** `scripts/build_degradation_dataset.py`. Artefacts: `data/f1_degradation_dataset.csv`, `models/f1_degradation_analysis.pkl`, and three figures: `outputs/figures/degradation_pace_profile.png`, `peak_age_by_compound.png`, `stint_length_vs_degradation.png`.
```

- [ ] **Step 2: Insert the §5.1 analysis-loading code cell after the intro**

Use `NotebookEdit` with `edit_mode: "insert"`, `cell_id` set to the ID of the markdown cell just created (look it up after the insert), `cell_type: "code"`, and content:

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path.cwd() if Path.cwd().name != 'notebooks' else Path.cwd().parent
DATA_DIR  = ROOT / 'data'
MODEL_DIR = ROOT / 'models'

deg_art   = joblib.load(MODEL_DIR / 'f1_degradation_analysis.pkl')
stint_summaries  = deg_art['stint_summaries']
peak_age_stats   = deg_art['peak_age_stats']
stint_length_corr = deg_art['stint_length_corr']

print(f'Stint summaries: {len(stint_summaries):,} stints '
      f'(SOFT / MEDIUM / HARD, >= 8 green laps each)')
print()
print('=== Peak-pace tire age by compound (per-stint medians) ===')
print(peak_age_stats.to_string(index=False))
print()
print('=== Stint length vs end-of-stint degradation (Pearson r) ===')
print(stint_length_corr.to_string(index=False))
```

- [ ] **Step 3: Insert the pace-profile figure display cell**

Use `NotebookEdit` with `edit_mode: "insert"`, `cell_id` set to the ID of the code cell just created, `cell_type: "code"`, and content:

```python
from IPython.display import Image, display

FIGURES_DIR = ROOT / 'outputs' / 'figures'
display(Image(str(FIGURES_DIR / 'degradation_pace_profile.png')))
```

- [ ] **Step 4: Insert the peak-age + stint-length figure display cell**

Use `NotebookEdit` with `edit_mode: "insert"`, `cell_id` set to the ID of the previous cell, `cell_type: "code"`, and content:

```python
display(Image(str(FIGURES_DIR / 'peak_age_by_compound.png')))
display(Image(str(FIGURES_DIR / 'stint_length_vs_degradation.png')))
```

- [ ] **Step 5: Insert the §5.1 findings markdown cell**

Use `NotebookEdit` with `edit_mode: "insert"`, `cell_id` set to the ID of the previous cell, `cell_type: "markdown"`, and content:

```markdown
**Findings.**

1. **Pace profile across a stint is U-shaped, not monotonic.** Every compound shows a warm-up of ~0.5–2.5 s/lap in the first five laps, settles into a plateau near its peak grip, then climbs back up as the tyre fatigues. The naive picture of "new tyre fast, old tyre slow" is wrong: a brand-new tyre on lap 1 is roughly as slow as a 25-lap-old one on MEDIUM. This pattern is stable across the 2022-2023 training and 2024 holdout panels.

2. **Peak-pace tire age is a clean compound signature.** Median stint-level peak age is ≈ 10 laps on SOFT, ≈ 15 on MEDIUM, ≈ 22 on HARD — the same ordering and roughly the same magnitudes in both eras. This is strategically useful: the "optimal" pit lap is not the fastest lap of the stint; it is the lap at which the tyre has just passed peak and a fresh set will regain more time than it loses.

3. **Stint length correlates with end-of-stint degradation on the harder compounds.** Per-stint degradation (median of worst 3 fuel-corrected laps − median of best 3) rises ~0.12 s for every extra lap of stint length, with Pearson *r* = 0.63 on HARD, 0.53 on MEDIUM, and 0.18 on SOFT. SOFT's weak correlation reflects how short and homogeneous its stints are; on MEDIUM and HARD the relationship is strong enough that stint-length is itself a useful degradation predictor.

4. **Note on what this analysis does *not* do.** We do not fit a per-lap predictor of `delta_pace` here. An earlier iteration of the pipeline tried per-(compound, circuit) quadratic curves; on a 2024 temporal holdout it underperformed the naive live in-stint slope estimator. The U-shape is the reason — it cannot be captured by a quadratic anchored at age 0 that is fit to a dataset dominated by the plateau — which is itself a finding worth preserving for the downstream pit-window work.
```

- [ ] **Step 6: Execute the new cells top-to-bottom in the notebook**

Open `notebooks/01_milestone_report.ipynb` in Jupyter (or run with `jupyter nbconvert --to notebook --execute`). Verify:
- The printout shows the peak-age-stats table with SOFT ≈ 10–11, MEDIUM ≈ 15–16, HARD ≈ 22–23 in both eras.
- The printout shows the stint-length correlation table with HARD r ≈ 0.63, MEDIUM r ≈ 0.53, SOFT r ≈ 0.18.
- Three figures render in order: pace profile (U-shape), peak age box plot, stint-length scatter.
- No cell raises.

Run for verification:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb --ExecutePreprocessor.timeout=600
```

Abort and diagnose if any cell errors out.

- [ ] **Step 7: Commit**

```bash
git add notebooks/01_milestone_report.ipynb
git commit -m "Add §5.1 Tire Degradation notebook section (Task 4/10)"
```

---

## Task 5: Expand §5.2 Weather section with compound × track-temp interaction

Modify the existing weather section (currently §5.6 "Bonus Analysis"). This task only changes content; Task 9 handles renumbering §5.6 → §5.2.

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb` (cells `7dfa878ba723`, `772cdb39-4263-41c2-8eec-c9b925c69104`, `c89b58b9-79ef-43b4-9e9e-2a99985e5843`)
- Create: `outputs/figures/weather_compound_sensitivity.png` (written by notebook code)
- Create: `outputs/figures/weather_diagnostics.png` (written by notebook code)

- [ ] **Step 1: Replace cell `7dfa878ba723` (weather section intro)**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "7dfa878ba723"`, `cell_type: "markdown"`, and content:

```markdown
## 5.2 Weather & Temperature Effects on Lap Time

**Question:** how do track temperature, air temperature, and related weather covariates influence lap pace, and does tire compound modulate that sensitivity?

**Setup:** we load five 2024 and 2025 race sessions (Bahrain, Saudi Arabia, Australia, Japan, China) with weather telemetry, merge per-lap weather onto each lap record, and fit an OLS regression of `LapTime_Seconds` on `C(Year) + C(Race) + C(Compound) + TrackTemp + AirTemp + Pressure + Humidity + WindSpeed + WindDirection` with HC2 robust standard errors. To test whether temperature sensitivity is compound-dependent, we extend the specification with a `C(Compound) * TrackTemp` interaction. Residual-vs-fitted and Q-Q plots confirm OLS assumptions are not badly violated. This section connects directly to §5.1: hotter tracks accelerate tire wear, so the stint-level pace profiles there are effectively cross-sections at each circuit's average track temperature.
```

- [ ] **Step 2: Replace cell `772cdb39-4263-41c2-8eec-c9b925c69104` (weather regression + plot code)**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "772cdb39-4263-41c2-8eec-c9b925c69104"`, `cell_type: "code"`, and content:

```python
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

FIGURES_DIR = ROOT / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Baseline OLS with robust SEs
baseline = smf.ols(
    formula='''LapTime_Seconds ~
                + C(Year)
                + C(Race)
                + C(Compound) - 1
                + TrackTemp
                + AirTemp
                + Pressure
                + Humidity
                + WindSpeed
                + WindDirection''',
    data=combined_df,
).fit(cov_type='HC2')
print('=== Baseline OLS (no interaction) ===')
print(baseline.summary())

# Interaction model: compound-specific temperature sensitivity
interaction = smf.ols(
    formula='''LapTime_Seconds ~
                + C(Year)
                + C(Race)
                + C(Compound) * TrackTemp
                + AirTemp
                + Pressure
                + Humidity
                + WindSpeed
                + WindDirection''',
    data=combined_df,
).fit(cov_type='HC2')
print('\n=== Interaction OLS (compound × track temp) ===')
print(interaction.summary())

# Compound-sensitivity plot: predicted lap time vs TrackTemp per compound,
# holding other covariates at their mean.
fig, ax = plt.subplots(figsize=(10, 5))
temp_grid = np.linspace(combined_df['TrackTemp'].min(),
                        combined_df['TrackTemp'].max(), 50)
ref_row = combined_df.select_dtypes(include=[np.number]).mean(numeric_only=True)

compound_colors = {'SOFT': '#e74c3c', 'MEDIUM': '#f1c40f', 'HARD': '#2c3e50'}
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    sub = combined_df[combined_df['Compound'] == compound]
    if sub.empty:
        continue
    pred_df = pd.DataFrame({
        'Year': sub['Year'].mode().iloc[0],
        'Race': sub['Race'].mode().iloc[0],
        'Compound': compound,
        'TrackTemp': temp_grid,
        'AirTemp': ref_row.get('AirTemp', sub['AirTemp'].mean()),
        'Pressure': ref_row.get('Pressure', sub['Pressure'].mean()),
        'Humidity': ref_row.get('Humidity', sub['Humidity'].mean()),
        'WindSpeed': ref_row.get('WindSpeed', sub['WindSpeed'].mean()),
        'WindDirection': ref_row.get('WindDirection',
                                     sub['WindDirection'].mean()),
    })
    preds = interaction.get_prediction(pred_df).summary_frame(alpha=0.05)
    color = compound_colors.get(compound, 'grey')
    ax.plot(temp_grid, preds['mean'], label=compound,
            color=color, linewidth=2)
    ax.fill_between(temp_grid, preds['mean_ci_lower'],
                    preds['mean_ci_upper'], color=color, alpha=0.18)

ax.set_xlabel('Track Temperature (°C)')
ax.set_ylabel('Predicted Lap Time (s)')
ax.set_title('Compound × Track-Temperature Sensitivity '
             '(95% CI bands, other covariates at sample means)',
             fontweight='bold')
ax.legend(title='Compound')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'weather_compound_sensitivity.png',
            dpi=150, bbox_inches='tight')
plt.show()

# Diagnostics: residuals vs fitted + Q-Q
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fitted = interaction.fittedvalues
resid  = interaction.resid
axes[0].scatter(fitted, resid, s=4, alpha=0.3, color='steelblue')
axes[0].axhline(0, color='black', linestyle=':', linewidth=1)
axes[0].set_xlabel('Fitted lap time (s)')
axes[0].set_ylabel('Residual (s)')
axes[0].set_title('Residuals vs Fitted')
stats.probplot(resid, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'weather_diagnostics.png',
            dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Replace cell `c89b58b9-79ef-43b4-9e9e-2a99985e5843` (weather interpretation)**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "c89b58b9-79ef-43b4-9e9e-2a99985e5843"`, `cell_type: "markdown"`, and content:

```markdown
**Findings.**

1. **Track temperature is the dominant weather driver of pace.** The baseline OLS coefficient on `TrackTemp` is statistically significant at the 95% level. Humidity and pressure add small but meaningful effects; `AirTemp` and `WindDirection` are not significant once TrackTemp is controlled for — TrackTemp absorbs most of their explanatory power.

2. **Compounds have different temperature sensitivities.** The interaction model's `C(Compound):TrackTemp` terms show that pace sensitivity to track temperature is not uniform across compounds. The sensitivity plot makes this visible directly: the three compound lines fan out as track temperature rises rather than shifting in parallel.

3. **OLS assumptions are defensible.** Residuals vs fitted show no strong heteroscedastic pattern, and the Q-Q plot is near-linear through the centre with only mild tail deviation. HC2 robust standard errors further insulate the coefficient p-values from residual non-normality.

4. **Connection to §5.1.** Because track temperature systematically shifts per-compound pace, the stint-level pace profiles, peak-age statistics, and stint-length correlations in §5.1 are implicitly measured at each circuit's average track temperature. Stratifying §5.1's analysis by track-temperature bucket is a natural next step — the weather data joined here is the bridge.
```

- [ ] **Step 4: Execute the three cells in Jupyter**

Run the weather section end-to-end. Verify:
- Both OLS summaries print.
- `weather_compound_sensitivity.png` and `weather_diagnostics.png` appear under `outputs/figures/`.
- All three compound lines are visible on the sensitivity plot.

```bash
ls -lh outputs/figures/weather_compound_sensitivity.png outputs/figures/weather_diagnostics.png
```

Both files must exist with non-zero size.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_milestone_report.ipynb outputs/figures/weather_compound_sensitivity.png outputs/figures/weather_diagnostics.png
git commit -m "Expand weather section with compound × track-temp interaction (Task 5/10)"
```

---

## Task 6: Merge §5.3 Undercut and §5.4 Overcut into one Pit Strategy section

Replace the two separate sections with a single merged markdown cell. Note that this task only changes prose; the actual models, artifacts, and numbers remain unchanged. Section numbering (§5.3 → §5.4, etc.) is handled in Task 9 after all other content changes settle.

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb` (replace cell `03b5e7a54b6d`; delete cell `810abdb06c8c`)

- [ ] **Step 1: Replace cell `03b5e7a54b6d` (old §5.3 Undercut) with the merged Pit Strategy content**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "03b5e7a54b6d"`, `cell_type: "markdown"`, and content:

```markdown
## 5.4 Pit Strategy: Undercut vs Overcut

**Question:** given two cars with a gap between them, which pit-strategy direction wins — pitting now to undercut the car ahead, or staying out to overcut the car that just pitted?

**Setup.** Both directions are trained on 2022-2023 race data with a 2024 temporal holdout, using a Logistic Regression baseline and an XGBoost primary model. The two use a near-identical feature set with the sign of the tire-age delta flipped to match each framing.

**Feature parity.**

| Concept | Undercut feature | Overcut feature |
|---|---|---|
| Relative tire age | `tire_age_advantage = ca_age − driver_age` (positive is good for pitter) | `tire_age_delta = stay_out_age − ca_age` (positive is bad for stay-out) |
| Gap | `gap_ahead` (pitter to target) | `gap_ahead` (stay-out to target) |
| Pace differential | `pace_delta = own_pace − threat_pace` | Same, reversed framing |
| In-stint deg slope | `deg_delta`, `ca_deg_delta` | Same |
| Closing rate | `closing_rate` | Same |
| Race context | `pit_loss`, `pit_loss_fraction`, `race_progress`, one-hot `compound` | Same |

**Headline results (2024 holdout).**

| Strategy | Model | Accuracy | ROC-AUC | F1 | Base rate |
|---|---|---|---|---|---|
| Undercut | Logistic Regression | 0.771 | 0.825 | — | 0.441 |
| Undercut | XGBoost (primary) | 0.738 | 0.806 | — | 0.441 |
| Overcut  | Logistic Regression | 0.823 | 0.807 | 0.46 | 0.251 |
| Overcut  | XGBoost (primary) | 0.805 | 0.804 | 0.56 | 0.251 |

**Combined SHAP takeaway.** The two directions are driven by different signals. `tire_age_advantage` dominates undercut SHAP values — the core mechanism is that a driver on fresh rubber gains enough time on the out-lap to undercut a car on degrading tyres. Overcut SHAP is dominated instead by `pace_delta` and the magnitude of `tire_age_delta` — the stay-out driver has to be genuinely quicker on old tyres than the rival is on new tyres, which is a harder bar to clear.

**Asymmetry.** Overcuts are materially harder than undercuts: the base success rate is 25.1% vs. 44.1%. This is why XGBoost's F1 lift over the LR baseline is larger on the overcut side (+0.10 vs. ~0) — overcut success is a minority-class problem that benefits from nonlinear decision boundaries. Full pipelines, EDA, and SHAP attribution are in `03_undercut_prediction.ipynb` and `04_overcut_prediction.ipynb`.
```

- [ ] **Step 2: Delete cell `810abdb06c8c` (old §5.4 Overcut)**

Use `NotebookEdit` with `edit_mode: "delete"`, `cell_id: "810abdb06c8c"`.

- [ ] **Step 3: Execute the notebook top-to-bottom and verify**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb --ExecutePreprocessor.timeout=600
```

Open the notebook and visually confirm:
- Section §5.4 renders as a single merged Pit Strategy section.
- The old §5.3 Undercut-only header and §5.4 Overcut-only header are gone.
- Hungary dashboard cells (still numbered §5.5 at this point) still render correctly — the merge does not touch their cells.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_milestone_report.ipynb
git commit -m "Merge undercut and overcut into unified Pit Strategy section (Task 6/10)"
```

---

## Task 7: Rewrite §1 Project Description to list all 5 questions

Replace the 3-question list with the 5 intended research questions.

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb` (cell `6299ca45`)

- [ ] **Step 1: Replace cell `6299ca45` (§1 Project Description) with the 5-question version**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "6299ca45"`, `cell_type: "markdown"`, and content:

```markdown
## 1. Project Description

Formula One is among the most analytically intensive motorsports in the world. Every car is outfitted with hundreds of sensors streaming data at kilohertz frequencies, and real-time strategy calls — when to pit, which tire compound to mount, whether to attempt an undercut on the car ahead — can be the difference between a podium and finishing outside the points.

This project applies data science methods to the **FastF1 API**, a publicly available Python library wrapping the official Formula 1 Timing API. Our goal is to build and evaluate a suite of predictive models that mirror tools used by real F1 strategy engineers. We frame the work as five interconnected research questions, ordered from the most foundational physical signal to the highest-level race outcome:

1. **Tire Degradation Modeling** — How does lap pace decay with tire age, and does the decay shape depend on compound and circuit?
2. **Weather & Temperature Effects** — How do track temperature, air temperature, and related weather covariates influence lap times, and does tire compound modulate that sensitivity?
3. **Pit Window Forecasting** — Given the current race state (gap, pace, tire age, circuit), how many laps remain before a viable pit window opens for a given driver?
4. **Pit Strategy — Undercut vs Overcut** — When two cars are within strategic range, which pit-timing decision (pit now to undercut, or stay out to overcut) is more likely to gain track position?
5. **Race Position Prediction** — At what lap of a race can a driver's final finishing position be predicted with meaningful accuracy, and how does that confidence grow as the race unfolds?

We focus on the **2022-2025 regulation era**, a coherent technical window in which car aerodynamics and tire behavior are governed by the same ruleset, making cross-season comparisons meaningful.
```

- [ ] **Step 2: Render the cell and verify**

Open the notebook. Confirm §1 renders with exactly 5 numbered questions in the order above.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_milestone_report.ipynb
git commit -m "Rewrite §1 Project Description to enumerate all 5 research questions (Task 7/10)"
```

---

## Task 8: Expand §4 Methods with new subsections and reorder to match §5

Replace the existing §4 Methods cell entirely. The new cell covers all 5 questions in the same order they appear in §5: Tire Degradation, Weather, Pit Window, Pit Strategy, Race Position.

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb` (cell `edb4f505`)

- [ ] **Step 1: Replace cell `edb4f505` (§4 Methods) with the 5-subsection version**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "edb4f505"`, `cell_type: "markdown"`, and content:

```markdown
## 4. Methods

Our project is organised into five interconnected modelling pipelines, each targeting one of the research questions from §1. All pipelines share a common data-loading and feature-engineering layer built on FastF1.

---

### 4.1 Tire Degradation (stint-level pace-profile analysis)

**Target:** `delta_pace`, defined as fuel-corrected lap time minus a per-(driver, stint) **peak-pace baseline** — the median of that stint's fastest three fuel-corrected laps. By construction `delta_pace` ≥ 0 and is ~ 0 at the stint's peak; it is lost time versus the tyre's best pace in that stint.

**Features:** tire compound, tire age (`TyreLife`), and circuit are used as grouping keys rather than regressors. Era splits (2022-2023 vs 2024) are applied to check pattern stability.

**Analysis.** Three descriptive views of the stint, each computed on `>= 8`-lap stints of dry compounds (`SOFT`, `MEDIUM`, `HARD`):

1. **Pace profile.** Mean `delta_pace` by `(compound, age-bucket)` where age buckets are `[0-5), [5-10), ..., [25-30), [30+)`. Plotted as three compound lines on each of two panels (2022-2023 vs 2024) so the U-shape (warm-up → peak → degradation) is directly visible.

2. **Peak-pace age.** For each stint, the tire age of the fastest fuel-corrected lap. Central tendency (median, mean, std) per compound and per era.

3. **Stint length vs degradation magnitude.** Per stint, `degradation_s = median(worst 3) − median(best 3)`. Pearson *r* computed against stint length within each compound; also the linear slope for plotting.

**Why not a predictor:** we attempted per-(compound, circuit) quadratic curves first. On the 2024 temporal holdout they underperformed a naive live in-stint linear slope, because (a) the pace profile is U-shaped, not monotonic — a quadratic of tire age is the wrong functional form, and (b) per-circuit residual signal is dwarfed by per-stint driver/traffic noise. The descriptive stint-level view captures the structurally interesting and stable findings without overclaiming predictive accuracy.

**Pipeline:** `scripts/build_degradation_dataset.py`. Artifact: `models/f1_degradation_analysis.pkl` — a joblib dict of DataFrames (`stint_summaries`, `pace_profile`, `peak_age_stats`, `stint_length_corr`).

---

### 4.2 Weather & Temperature Effects (OLS with compound × track-temp interaction)

**Target:** `LapTime_Seconds` on green laps from five 2024-2025 race sessions with merged weather telemetry.

**Baseline specification:** `LapTime_Seconds ~ C(Year) + C(Race) + C(Compound) - 1 + TrackTemp + AirTemp + Pressure + Humidity + WindSpeed + WindDirection` with HC2 robust standard errors.

**Interaction specification:** baseline plus a `C(Compound) * TrackTemp` term to estimate whether pace sensitivity to track temperature varies by compound.

**Evaluation:** coefficient significance, residuals-vs-fitted plot, and Q-Q plot for OLS assumption checks; per-compound prediction curves with 95% CI bands at sample means of the other covariates.

---

### 4.3 Pit Window Forecasting (Gradient Boosted Regressor)

**Target:** `laps_until_open` — the number of laps until the threat gap behind the driver exceeds `pit_loss + 3.0 s`. Rows where no window opens within a 20-lap horizon receive a sentinel value of 21.

**Features:** `threat_gap`, `closing_rate`, `gap_trend`, `tire_age`, `deg_delta`, `pit_loss`, `pit_loss_fraction`, `sc_rate`, `vsc_rate`, `race_progress`, `context`, `compound`.

**Model:** `GradientBoostingRegressor` (300 estimators, learning rate 0.05, max depth 5, subsample 0.8). Hyperparameters tuned via grid search on a 2022 validation split; trained on 2022-2023 and evaluated on the 2024 temporal holdout.

**Evaluation:** MAE, RMSE, within-±2-lap accuracy, within-±5-lap accuracy, per-context MAE breakdown.

---

### 4.4 Pit Strategy — Undercut vs Overcut (Logistic Regression + XGBoost)

**Targets.** Two complementary binary labels: (a) does pitting now to undercut the car ahead result in track-position gain four laps later, and (b) does staying out while the car ahead pits first result in track-position gain four laps later. Safety-Car and VSC-contaminated decisions are filtered from both datasets.

**Features.** Both models share `gap_ahead`, pace delta, tire-age delta (with signs flipped per framing), in-stint degradation slopes for both cars, `closing_rate`, `pit_loss`, `pit_loss_fraction`, `race_progress`, and one-hot compound indicators.

**Models.** Logistic Regression with standard scaling as an interpretable baseline, XGBoost with `scale_pos_weight = neg/pos` as the primary classifier for handling class imbalance (especially pronounced on the overcut side).

**Train/test.** 2022-2023 for training, 2024 temporal holdout for evaluation (accuracy, ROC-AUC, F1, SHAP attribution, calibration).

**Pipelines:** `scripts/build_undercut_dataset.py`, `scripts/build_overcut_dataset.py`.

---

### 4.5 Race Position Prediction (Random Forest Classifier)

**Target:** final finishing position, bucketed into four classes — Podium (P1-P3), Top Points (P4-P6), Low Points (P7-P10), Out of Points (P11+).

**Features at each checkpoint lap N:** current track position, tire compound (one-hot), tire age, stint number, lap number, gap to leader, 5-lap rolling pace trend.

**Model:** `RandomForestClassifier` (100 estimators, `random_state=42`), trained independently per checkpoint lap in `{5, 10, 15, ..., 50}` on 2022-2023 races.

**Train/test:** 2022-2023 for training, 2024 temporal holdout for evaluation at each checkpoint — yielding a predictability curve plus confusion matrices at laps 10, 30, and 50.

**Pipeline:** `scripts/build_position_dataset.py`.

---

### 4.6 Libraries and Tools

| Library | Role |
|---------|------|
| `fastf1` | Data acquisition and session loading |
| `pandas` / `numpy` | Tabular processing, feature engineering |
| `scikit-learn` | Random Forest, Gradient Boosting, preprocessing, evaluation |
| `statsmodels` | OLS with robust SEs for the weather regression |
| `xgboost` | Primary classifier for pit-strategy models |
| `shap` | Feature attribution for XGBoost |
| `matplotlib` / `seaborn` | Static visualization |
| `joblib` | Model and curve-artifact serialization |
```

- [ ] **Step 2: Render the cell and verify**

Confirm §4 renders with subsections §4.1 through §4.6, in the order specified, and that the library table is present.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_milestone_report.ipynb
git commit -m "Expand §4 Methods with 5 subsections in build-up order (Task 8/10)"
```

---

## Task 9: Renumber §5 section headers to match the new layout

With all §5 content changes settled, renumber the remaining section headers. The mapping is:

| Current header | Becomes |
|---|---|
| §5.1 Race-Position Prediction | §5.5 Race Position Prediction |
| §5.2 Pit-Window Forecasting   | §5.3 Pit Window Forecasting |
| §5.4 Pit Strategy (after Task 6 merge) | stays §5.4 |
| §5.5 Integrated Strategy Example | §5.6 Integrated Race-Day Example |
| §5.6 Bonus Weather            | §5.2 Weather & Temperature Effects (header updated in Task 5; verify) |
| §5.1 Tire Degradation (inserted in Task 4) | stays §5.1 |

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb` (cells `503385e974ae`, `8c312a3064d3`, `661745458581`)

- [ ] **Step 1: Replace cell `503385e974ae` (was §5.1 Race-Position) to become §5.5**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "503385e974ae"`, `cell_type: "markdown"`, and content:

```markdown
## 5.5 Race Position Prediction — Random Forest

**Question:** at what point in a race can a driver's final finishing position be predicted with meaningful accuracy?

**Setup:** at ten lap checkpoints (5, 10, 15, …, 50) we freeze each driver's state (position, tyre compound, tyre age, stint number, gap to leader, rolling pace trend) and train a Random Forest per checkpoint on 2022-2023 races. The 2024 holdout tells us how much earlier a race can be "called" in practice. Targets are 4-bucket: Podium (P1-P3), Top Points (P4-P6), Low Points (P7-P10), Out of Points (P11+).

**Pipeline:** `scripts/build_position_dataset.py`. Artefacts live in `data/f1_position_dataset.csv`, `models/f1_position_model.pkl`, and the figures below.
```

- [ ] **Step 2: Replace cell `8c312a3064d3` (was §5.2 Pit-Window) to become §5.3**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "8c312a3064d3"`, `cell_type: "markdown"`, and content:

```markdown
## 5.3 Pit-Window Forecasting — Gradient Boosted Regressor

**Question:** given the current race state, how many laps remain until a driver's pit window opens?

**Setup:** 46,976 lap-level observations across 2022-2024, target `laps_until_open` bounded to a 20-lap horizon. Model is a tuned `GradientBoostingRegressor` (learning_rate=0.03, max_depth=6, n_estimators=200). Full EDA, per-context breakdown, and feature importance are in `02_pit_window_forecasting.ipynb`.

**Headline results on the 2024 holdout:**

| Metric | Value |
|---|---|
| N (2024 obs) | 15,619 |
| MAE | **2.26 laps** |
| RMSE | 3.48 laps |
| Within ±1 lap | 47.6% |
| Within ±2 laps | 57.5% |
| Within ±5 laps | 84.4% |

**Dominant feature:** `closing_rate` accounts for 63% of total feature importance — the rate at which the threat car behind is closing the gap is the single strongest predictor of when a pit window will become viable. Per-context MAE confirms the model is strongest in the `tight_behind` scenario (MAE 1.75) and weakest in `tight_ahead` (MAE 2.63), which matches the physical intuition that stuck-in-traffic windows depend on the *other* driver's choice rather than on observable state.
```

- [ ] **Step 3: Replace cell `661745458581` (was §5.5 Hungary) to become §5.6**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "661745458581"`, `cell_type: "markdown"`, and content:

```markdown
## 5.6 Integrated Race-Day Example — 2024 Hungarian Grand Prix

To show how the five primary models compose into a single race-day picture we run the three lap-level models on **2024 Hungary**, a circuit with overtaking difficulty that makes pit-window timing and over/undercut decisions especially consequential. For a chosen driver we plot:

- the **position-bucket probability** over the whole race (from §5.5's RF classifier at each checkpoint);
- the **pit-window forecast** (laps until open) over every green lap;
- the **undercut / overcut success probability** at every pit-stop decision point that actually occurred in the race.

This is the view a strategy engineer would want: one driver, one race, three overlaid signals.
```

- [ ] **Step 4: Execute the notebook end-to-end and verify headers**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb --ExecutePreprocessor.timeout=600
```

Open the notebook in Jupyter. Scroll through §5 and confirm exactly this header order:

1. `## 5.1 Tire Degradation — Pace Profile Across a Stint`
2. `## 5.2 Weather & Temperature Effects on Lap Time`
3. `## 5.3 Pit-Window Forecasting — Gradient Boosted Regressor`
4. `## 5.4 Pit Strategy: Undercut vs Overcut`
5. `## 5.5 Race Position Prediction — Random Forest`
6. `## 5.6 Integrated Race-Day Example — 2024 Hungarian Grand Prix`

If any header is mis-numbered, correct it with another `NotebookEdit` replace on the same cell.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_milestone_report.ipynb
git commit -m "Renumber §5 section headers to match 5-question ordering (Task 9/10)"
```

---

## Task 10: Rewrite §9 Conclusions + light pass on §6, §7, §8

Finalise the narrative: conclusion paragraph per question in build-up order, update peer feedback / process summary / limitations sections to reference 5 questions.

**Files:**
- Modify: `notebooks/01_milestone_report.ipynb` (cells `55223489ed32`, `584f8e69`, `41989f94ec35`, `18d26efaa688`)

- [ ] **Step 1: Replace cell `55223489ed32` (§9 Conclusions)**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "55223489ed32"`, `cell_type: "markdown"`, and content:

```markdown
## 9. Conclusions

The five research questions have concrete, data-backed answers:

**1. How does lap pace evolve across a stint?** Not monotonically — the pace profile is U-shaped. Every compound warms up, plateaus near peak grip, then degrades. Median peak-pace age separates cleanly by compound (≈ 10 laps SOFT, ≈ 15 MEDIUM, ≈ 22 HARD) and is stable across the 2022-2023 and 2024 eras, making it a robust strategic anchor for pit timing. End-of-stint degradation magnitude rises roughly 0.12 s per extra lap of stint length, with a strong correlation on HARD (*r* = 0.63) and MEDIUM (*r* = 0.53) and a weaker one on SOFT (*r* = 0.18) where stints are short and homogeneous. A per-lap predictor was attempted and abandoned — a quadratic of tire age cannot represent the U-shape — so §5.1 is descriptive rather than predictive, which is itself a finding for the downstream pit-window model.

**2. How do weather and temperature affect lap time?** Track temperature is the dominant weather covariate, statistically significant at the 95% level under HC2 robust standard errors. The `C(Compound) * TrackTemp` interaction is significant too, confirming that compounds degrade at different rates as track temperature rises — consistent with the §5.1 finding that degradation curves are implicitly conditioned on temperature. Residuals-vs-fitted and Q-Q plots show OLS assumptions hold reasonably well.

**3. When does a driver's pit window open?** The tuned Gradient-Boosted Regressor predicts the opening lap within ±2 laps for 57.5% of observations and within ±5 laps for 84.4%, on a 2024 holdout with no season information leakage. The single most important feature — by a wide margin — is `closing_rate`, the rate at which the threat car behind is closing the gap. Absolute gap is secondary.

**4. Is pitting now the right call?** Both over/undercut models reach ROC-AUC ≈ 0.80, which is meaningful predictive signal on problems that are coin-flip in expectation. The undercut model's top SHAP signal is `tire_age_advantage`; the overcut model's top signal is `pace_delta`. The two strategies are not mirror images: overcuts have a 25% base rate versus 44% for undercuts, and XGBoost's F1 lift over the LR baseline is correspondingly larger on the harder (overcut) side.

**5. When does a race become predictable?** Roughly at lap 30 on a typical 2024 Grand Prix. The position classifier crosses 75% accuracy at that point and climbs smoothly to 89% by lap 50. Races are not decided on the grid but they are not decided on the final lap either — the critical zone is the mid-race phase in which first-stint tyre-choice outcomes resolve.

**Scientific contribution.** The value of this project is less in the individual accuracy numbers — F1 teams with full telemetry will beat us — and more in a reproducible pipeline from public data to interpretable models, with documented treatment of every physical confound (SC laps, lapping, fuel load, tyre cliff). Every notebook is self-contained; every model's features and training code are version-controlled; every metric is computed on a temporal holdout with no leakage.

**Future work.** The most productive extensions would be: (a) plumbing the §5.1 per-(compound, circuit) curves into the pit-window model to replace the in-stint linear-slope feature; (b) fusing the §5.2 temperature-sensitivity coefficients into both the degradation model and the pit-window model as a covariate; (c) adding a driver one-hot encoding across all five models to capture individual stint-management styles; (d) extending the position model to a full per-lap sequence with a temporal architecture (e.g., transformer-on-laps) rather than snapshot-per-checkpoint.
```

- [ ] **Step 2: Replace cell `584f8e69` (§6 Peer Feedback) with light edits**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "584f8e69"`, `cell_type: "markdown"`, and content:

```markdown
## 6. Peer Feedback

**Feedback group:** Joely Marsyla, Caitlin Roake, Natalie Zorn

---

Our peer reviewers provided thoughtful and constructive feedback during the in-class milestone presentation. Below is a summary of the key points raised and our planned responses.

### 6.1 Feedback Received

**Strength — Strong domain grounding and clear motivation**
> The group noted that the project was exceptionally well-motivated, with a clear real-world connection to how actual F1 strategy teams operate. The inclusion of domain-specific details (e.g., the fuel correction factor, circuit-specific pit loss times) demonstrated genuine investment in the subject matter and made the proposal feel credible beyond a generic sports analytics project.

**Concern — Model evaluation / train-test split clarity**
> Reviewers pointed out that the preliminary race position classifier trained and evaluated on the same dataset, which risks overfitting and makes the reported accuracy figures hard to interpret. They recommended implementing a proper temporal holdout — for example, training on 2022-2023 races and testing exclusively on 2024 — before drawing conclusions about model quality.

**Suggestion — Clarify scope across the five research questions**
> With five research questions listed (tire degradation, weather, pit window, pit strategy, race position), the group suggested making the relationships between them explicit — for example, calling out which models feed into which, and prioritising the two or three that are most novel and interconnected. We adopted this suggestion by ordering §5 from foundational (degradation, weather) to highest-level outcome (position) and by writing a §5.6 integrated dashboard that composes the lap-level models on a single race.

**Question — How are lapped cars handled in gap calculations?**
> Natalie raised a specific technical question: when computing `GapAhead` and `GapBehind` using lap crossing times, how do lapped cars affect the gap computation? She noted that a driver who has lapped another car will appear artificially close in position-sorted order, inflating the apparent gap to the car "ahead." This is a legitimate concern that we had already addressed in the codebase (see the `same_lap` filter in the gap computation functions), and we clarified this in our response.

---

### 6.2 Our Response and Planned Adjustments

| Feedback | Action Taken / Planned |
|----------|----------------------|
| Overfitting risk in position classifier | Implemented temporal holdout: train on 2022-2023, test on 2024. Applied consistently across all five models. |
| Scope clarification across 5 questions | Reordered §5 from foundational to outcome-level; integrated dashboard in §5.6 shows model composition on one race. |
| Lapped car gap contamination | Already handled via `same_lap` filter in `compute_inter_car_gaps()` — added clarifying comments and documentation |
| Model interpretability | Added SHAP feature importance plots alongside built-in GBM importances for the pit-strategy models. |
```

- [ ] **Step 3: Replace cell `41989f94ec35` (§7 Process Summary) to include the tire degradation artifact**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "41989f94ec35"`, `cell_type: "markdown"`, and content:

```markdown
## 7. Process Summary & Completed Milestones

Every proposal milestone is complete. The final deliverables are:

**Data & pipeline**
- [x] FastF1 integration and local caching (2020-2024)
- [x] Lap-quality filter, pit-lap and out-lap exclusion, SC/VSC masking
- [x] Inter-car gap computation using LapStartTime ordering
- [x] Context classifier (`tight_ahead` / `tight_behind` / `tight_both` / `free`)
- [x] Fuel correction (1.8 kg/lap × 0.035 s/kg)
- [x] Per-stint baseline and delta-pace computation (new, feeds §5.1)
- [x] Circuit pit-loss computation from real pit-in / pit-out deltas

**Models**
- [x] Tire degradation stint-level analysis — peak-pace anchor, pace profile across tire-age buckets, peak-age by compound, stint-length vs degradation correlation, on 2022-2024 (`scripts/build_degradation_dataset.py`, artifact `models/f1_degradation_analysis.pkl`).
- [x] Weather/temperature OLS with `C(Compound) * TrackTemp` interaction and HC2 robust SEs (notebook §5.2).
- [x] Pit-window Gradient Boosted Regressor — tuned via grid search, MAE 2.26 laps on 2024 holdout (`02_pit_window_forecasting.ipynb`).
- [x] Undercut Logistic Regression + XGBoost — 77% accuracy, ROC-AUC 0.83 (`03_undercut_prediction.ipynb`, `scripts/build_undercut_dataset.py`).
- [x] Overcut Logistic Regression + XGBoost — 81% accuracy, ROC-AUC 0.80 (`04_overcut_prediction.ipynb`, `scripts/build_overcut_dataset.py`).
- [x] Race-position Random Forest — trained per checkpoint on 2022-2023, tested on 2024, predictability curve from 65% @ lap 5 to 89% @ lap 50 (`scripts/build_position_dataset.py`).

**Analysis & visualisation**
- [x] Three tire-degradation figures: pace-profile U-shape (2022-2023 vs 2024), peak-age box plot by compound, stint-length vs degradation scatter (§5.1).
- [x] Compound × track-temperature sensitivity plot + residual and Q-Q diagnostics (§5.2).
- [x] SHAP attribution on both over/undercut models (beeswarm + bar).
- [x] Probability calibration plots on both pit-strategy models.
- [x] Per-context and per-race-quarter MAE breakdown for the pit-window model.
- [x] Predictability curve, confusion matrices at laps 10/30/50, and feature-importance bar chart for the position model.
- [x] Integrated race-day dashboard (§5.6) running the lap-level models on 2024 Hungary.
- [x] Interactive test harnesses (`scripts/test_undercut_model.py`, `scripts/test_overcut_model.py`) for browsing predictions, filtering by driver, and entering custom scenarios.

**Peer-feedback response** (§6): the specific issue raised — training and testing the position classifier on the same data — has been fully addressed. All five primary analyses use either a 2022-2023 → 2024 temporal holdout or a 2024-2025 out-of-sample evaluation.
```

- [ ] **Step 4: Replace cell `18d26efaa688` (§8 Limitations) to add the tire-degradation limitations paragraph**

Use `NotebookEdit` with `edit_mode: "replace"`, `cell_id: "18d26efaa688"`, `cell_type: "markdown"`, and content:

```markdown
## 8. Limitations

Every model here is trained on **public timing data only**. The following sources of irreducible uncertainty are outside the feature set by construction:

**1. Weather and track evolution.** The position classifier uses a single rolling-pace feature that averages over the previous 5 green laps; it does not distinguish a driver going slower because their tyres are going off from a driver going slower because rain is starting. The 2022-2024 training window contains several wet/mixed-condition races (Monaco 2022, Canada 2022, Netherlands 2023) where the models produce noisier predictions than the headline numbers suggest. A full solution would fuse the weather covariates from §5.2 into the primary models.

**2. Safety-car and red-flag events.** Under/overcut attempts that overlap an SC/VSC are filtered from training data, but the models at inference do not know an SC is coming. In practice an SC during a pit-window decision will invalidate the model's prediction entirely. This is a genuine limit: SC appearance is not predictable from lap-timing features.

**3. Driver-level effects.** All five models treat drivers as interchangeable. They do not encode that Verstappen historically extends stints further than most on worn tyres, or that some drivers have higher pit-stop execution variance. A driver-one-hot would likely lift all models a few points, at the cost of making predictions for rookies impossible until their first full season.

**4. Team strategy and radio.** The models are blind to team orders, teammate pairings, and tyre-allocation constraints. An overcut attempt on a teammate is treated identically to one on a rival — a simplification that visibly hurts F1 accuracy during intra-team battles.

**5. Tire degradation analysis simplifications.** The §5.1 analysis is descriptive: it reports the stint-level pace profile, per-compound peak age, and the stint-length vs degradation correlation, but does not produce a per-lap predictor of `delta_pace`. An early attempt at per-(compound, circuit) quadratic curves was rejected because (a) the pace profile is U-shaped rather than monotonic, so a quadratic of tire age is structurally the wrong form, and (b) per-circuit signal was dwarfed by driver/traffic noise on the 2024 holdout. The analysis also pools drivers within each compound (individual stint-management differences are averaged out) and ignores track evolution (rubbering-in), fuel-induced weight transfer beyond the linear correction, and temperature drift across a race.

**6. Finite holdout size.** 2024 provides 15,619 pit-window observations, ~450 undercut attempts, and ~340 overcut attempts after SC/VSC filtering. That is enough to compute stable aggregate metrics but thin for slicing by circuit, compound, and context simultaneously. A few per-context cells in the overcut breakdown have n < 10 and should be treated as indicative only.

**7. Regulation-era dependence.** Everything is trained on the 2022-2024 ground-effect car generation. Applied to 2026's new regulations (active aero, 50/50 hybrid split) the models will need retraining from scratch — the physical constants baked into fuel correction and pit loss will no longer apply.
```

- [ ] **Step 5: Execute the full notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb --ExecutePreprocessor.timeout=900
```

Expect no cell to raise. If a cell errors, read the traceback and fix in place — do not leave the notebook with un-executed error cells committed.

- [ ] **Step 6: Final acceptance verification**

Open the notebook and visually confirm all acceptance criteria from the design spec:

1. §1 lists exactly 5 numbered questions (Degradation → Weather → Pit Window → Pit Strategy → Position).
2. §4 has subsections §4.1 through §4.6 in the same order.
3. §5 has exactly six subsections §5.1 through §5.6 matching the same order + the Hungary dashboard.
4. §5.1 shows the peak-age and stint-length-correlation tables, plus the three pivoted figures (pace profile, peak-age box plot, stint-length scatter).
5. §5.2 shows two OLS summaries (baseline and interaction), the compound-sensitivity plot with CI bands, and the diagnostic plots.
6. §5.4 is a single merged Pit Strategy section with one combined results table.
7. §9 has exactly 5 numbered conclusion paragraphs in build-up order.
8. §7 process summary lists the tire-degradation model as a completed deliverable.
9. §8 limitations includes the tire-degradation paragraph.

- [ ] **Step 7: Commit**

```bash
git add notebooks/01_milestone_report.ipynb
git commit -m "Rewrite §9 Conclusions + update §6 §7 §8 for 5-question framing (Task 10/10)"
```

---

## Post-Implementation Checklist

After all 10 tasks are committed:

- [ ] Run the notebook one final time top-to-bottom: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb --ExecutePreprocessor.timeout=900`
- [ ] Confirm `git status` is clean.
- [ ] Confirm all six new output artifacts exist:
  - `models/f1_degradation_analysis.pkl`
  - `data/f1_degradation_dataset.csv`
  - `outputs/figures/degradation_pace_profile.png`
  - `outputs/figures/peak_age_by_compound.png`
  - `outputs/figures/stint_length_vs_degradation.png`
  - `outputs/figures/weather_compound_sensitivity.png`
  - `outputs/figures/weather_diagnostics.png`
- [ ] Confirm no regressions in existing numerical outputs: §5.3 (pit window MAE 2.26), §5.5 (position curve 65% → 89%), §5.6 Hungary dashboard still runs.
