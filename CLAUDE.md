# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 race strategy prediction project (Cooper Kerr & Isaac Middlemas, University of Utah). Uses the FastF1 API to build ML classifiers answering three questions:
1. At what lap can a driver's final finishing position be predicted? (Random Forest)
2. When will a driver's pit window open? (GBM)
3. Will pitting now (undercut) or staying out (overcut) beat the car directly ahead? (XGBoost + LR)

## Commands

**Rebuild datasets and retrain models from scratch:**
```bash
python scripts/build_undercut_dataset.py --rebuild
python scripts/build_overcut_dataset.py --rebuild
```
Without `--rebuild`, the scripts skip the FastF1 pull and load the existing CSV in `data/`. This is intentional for fast re-training after feature changes, but the raw data is not refreshed.
Both scripts cache-check first — if the CSV already exists they skip the FastF1 data pull and jump straight to training.

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

## Critical Dependency Constraint

**pandas must stay at `<3.0.0` (use `pandas==2.2.3`).** FastF1 3.8.x caches timedeltas with nanosecond precision (`dtype('<m8[ns]')`); pandas 3.x uses microsecond precision (`dtype('<m8[us]')`), causing `MergeError: incompatible merge keys` inside FastF1's `_load_laps_data`. This is not a project bug — it is a FastF1 upstream limitation.

## Architecture

### Data pipeline (build scripts)

Both `build_undercut_dataset.py` and `build_overcut_dataset.py` follow the same structure:

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

All `.pkl` files are `joblib` dicts: `{'model': XGBClassifier, 'features': [list of feature names]}`. The test harnesses load this and reconstruct the feature vector in the same order — feature list is the source of truth, not any hardcoded ordering.

### Notebooks vs scripts

The notebooks in `notebooks/` are narrative mirrors of the build scripts. Path resolution in notebooks uses `os.path.abspath('')` to detect whether running from `notebooks/` or from the project root, then walks up accordingly. The scripts use `Path(__file__).parent.parent` (always correct regardless of working directory).

### FastF1 cache

`f1-cache/` is gitignored for years 2022–2024. The `f1-cache/2021/` directory is NOT gitignored (only specific year subdirs are listed in `.gitignore`). FastF1's HTTP cache (`fastf1_http_cache.sqlite`) is also gitignored. On a fresh clone, the first `build_*.py` run will be slow (fetching all sessions from the API); subsequent runs are fast (served from cache).
