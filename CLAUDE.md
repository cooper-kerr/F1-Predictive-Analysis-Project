# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 race strategy prediction project (Cooper Kerr & Isaac Middlemas, University of Utah). Uses the FastF1 API to answer five research questions, ordered from foundational physical signal to race-level outcome:
1. How does lap pace evolve across a stint? (stint-level descriptive analysis — intentionally not a predictor)
2. How do weather/temperature affect lap time? (OLS with `C(Compound) * TrackTemp` interaction, HC2 robust SEs)
3. When does a pit window open? (tuned Gradient Boosted Regressor)
4. Will pitting now (undercut) or staying out (overcut) gain track position? (XGBoost + LR, two models with feature signs flipped per framing)
5. At what lap can final finishing position be predicted? (Random Forest per checkpoint lap)

The integrated deliverable is `notebooks/01_milestone_report.ipynb`, which covers all five questions in §5.1–§5.6 and a §5.6 Hungary dashboard composing the lap-level models on one race.

## Commands

**Rebuild datasets and retrain models from scratch:**
```bash
python scripts/build_degradation_dataset.py --rebuild   # stint-level analysis → models/f1_degradation_analysis.pkl + 3 figures
python scripts/build_undercut_dataset.py --rebuild      # → models/f1_undercut_model.pkl
python scripts/build_overcut_dataset.py --rebuild       # → models/f1_overcut_model.pkl
python scripts/build_position_dataset.py --rebuild      # → models/f1_position_model.pkl
```
Pit-window training lives in `notebooks/02_pit_window_forecasting.ipynb` (no standalone build script).

Without `--rebuild`, the scripts skip the FastF1 pull and load the existing CSV in `data/`. This is intentional for fast re-training after feature changes, but the raw data is not refreshed. Each script cache-checks first — if its CSV already exists it skips the FastF1 data pull and jumps straight to training/analysis.

**Re-execute the integrated milestone notebook end-to-end:**
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb --ExecutePreprocessor.timeout=900
```
This is the canonical verification — if it runs top-to-bottom without errors, the five-question pipeline is healthy.

**Test models interactively:**
```bash
python scripts/test_undercut_model.py inspect        # all 2024 holdout predictions
python scripts/test_undercut_model.py errors         # false positives / false negatives
python scripts/test_undercut_model.py driver VER     # filter by driver code
python scripts/test_undercut_model.py scenario       # interactive custom scenario

python scripts/test_overcut_model.py inspect
python scripts/test_overcut_model.py driver SAI
python scripts/test_overcut_model.py scenario
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```
All version constraints are enforced there. See the pandas constraint note below.

## Critical Dependency Constraints

**pandas must stay at `<3.0.0` (use `pandas==2.2.3`).** FastF1 3.8.x caches timedeltas with nanosecond precision (`dtype('<m8[ns]')`); pandas 3.x uses microsecond precision (`dtype('<m8[us]')`), causing `MergeError: incompatible merge keys` inside FastF1's `_load_laps_data`. This is not a project bug — it is a FastF1 upstream limitation.

**statsmodels must be ≥ 0.14.6.** The §5.2 weather OLS uses `statsmodels.formula.api`. On anaconda envs with scipy ≥ 1.16, statsmodels ≤ 0.14.5 fails to import because scipy removed the private `_lazywhere` symbol it relied on. `requirements.txt` does not currently pin statsmodels — if a fresh env hits an `ImportError` from statsmodels, `pip install --upgrade 'statsmodels>=0.14.6'` resolves it.

## Architecture

### Data pipeline (build scripts)

Two pipeline shapes live side-by-side:

- **Predictor pipelines** (`build_undercut_dataset.py`, `build_overcut_dataset.py`, `build_position_dataset.py`): FastF1 load → feature engineering → train/test split (2022–2023 vs 2024 temporal holdout) → model fit → joblib artefact `{'model': ..., 'features': [...]}`.
- **Descriptive pipeline** (`build_degradation_dataset.py`): FastF1 load → per-stint summaries → DataFrame aggregates (`pace_profile`, `peak_age_stats`, `stint_length_corr`) → PNG figures. No train/test split and no model object — §5.1 is descriptive by design. An early attempt at per-(compound, circuit) quadratic curves was rejected because the pace profile is U-shaped, not monotonic, and per-circuit signal is dwarfed by driver/traffic noise.

The predictor pipelines all follow the same structure:

1. **Gap timeseries** — `build_gap_timeseries(session)`: sorts laps by `(LapNumber, LapStartTime)`. Because `LapStartTime` is the absolute time a lap *began*, an earlier value within the same lap number means that driver is further ahead on track. `shift(1)` within each lap group gives the car directly ahead; `shift(-1)` gives the car directly behind.

2. **Label construction** — `build_undercut_records` / `build_overcut_records`: iterates pit stops, finds the relevant pairing, resolves outcome at `eval_lap = ca_pit_lap + STABILIZATION (4 laps)`, and filters events where a Safety Car / VSC appeared between the two pit stops (TrackStatus chars `'4'`, `'5'`, `'6'`). Position proxy is LapStartTime ordering at `eval_lap`, not the `Position` column (which can be unreliable).

3. **Feature engineering**: fuel-corrected pace (`lap_time − (lap_number − 1) × 1.8 × 0.035`), degradation slope via `np.polyfit` over the last 5 green laps, closing rate via `np.polyfit` over the last 3 gap observations.

4. **Models**: `StandardScaler + LogisticRegression` pipeline as interpretable baseline; `XGBClassifier` with `scale_pos_weight = neg/pos` for class imbalance as primary. Both trained on 2022–2023, evaluated on 2024 (temporal holdout). Saved with `joblib.dump({'model': ..., 'features': [...]}, path)`.

### Undercut vs overcut framing

| | Undercut | Overcut |
|---|---|---|
| Who pits first? | `driver` (the pitting car) | `pitting_car` (the car that was ahead) |
| Who stays out? | `car_ahead` stays out longer | `stay_out_driver` stays out |
| Key feature | `tire_age_advantage = ca_tire_age − driver_tire_age` | `tire_age_delta = stay_out_tire_age − ca_tire_age` |
| Success | `driver` is ahead at `eval_lap` | `stay_out_driver` is ahead at `eval_lap` |

### Saved artefact format

Predictor `.pkl` files (`f1_undercut_model.pkl`, `f1_overcut_model.pkl`, `f1_position_model.pkl`, `f1_pit_window_model_tuned.pkl`) are `joblib` dicts of the form `{'model': <estimator>, 'features': [list of feature names]}`. The test harnesses load this and reconstruct the feature vector in the same order — feature list is the source of truth, not any hardcoded ordering.

`f1_degradation_analysis.pkl` is the exception: it is a joblib dict of DataFrames — `stint_summaries`, `pace_profile`, `peak_age_stats`, `stint_length_corr` — plus metadata keys `age_bins`, `age_labels`, `train_years`, `test_years`. There is no model object and no feature list. §5.1 of the milestone notebook loads these DataFrames directly.

### Notebooks vs scripts

The notebooks in `notebooks/` are narrative mirrors of the build scripts. `01_milestone_report.ipynb` is the integrated 5-question report — the one to re-execute when verifying an end-to-end change. `02_pit_window_forecasting.ipynb`, `03_undercut_prediction.ipynb`, and `04_overcut_prediction.ipynb` hold the full EDA, feature-importance, SHAP, and calibration work for their individual models; the milestone notebook points to them for detail rather than reproducing it.

Path resolution in notebooks uses `os.path.abspath('')` to detect whether running from `notebooks/` or from the project root, then walks up accordingly. The scripts use `Path(__file__).parent.parent` (always correct regardless of working directory).

### FastF1 cache

`f1-cache/` is gitignored for years 2022–2024. The `f1-cache/2021/` directory is NOT gitignored (only specific year subdirs are listed in `.gitignore`). FastF1's HTTP cache (`fastf1_http_cache.sqlite`) is also gitignored. On a fresh clone, the first `build_*.py` run will be slow (fetching all sessions from the API); subsequent runs are fast (served from cache).
