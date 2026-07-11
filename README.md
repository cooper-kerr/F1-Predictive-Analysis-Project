# F1 Predictive Analysis Project

Machine learning and statistical analysis of Formula One race strategy using the [FastF1](https://docs.fastf1.dev/) API.

**Maintained by Cooper Kerr.** Minh Le contributed probability modeling work incorporated into this project as part of a University of Utah class project.  
**Institution:** University of Utah

## Recruiter Quick Read

This project turns public Formula One timing data into an explainable race-strategy demo. The Streamlit app lets a reviewer pick a 2024 race, driver, and lap, then see:

- when the model thinks the driver's pit window is opening
- whether an undercut or overcut looks more favorable against the direct rival ahead
- how tire degradation and weather context help explain the strategy recommendation

The strongest engineering signal is the full pipeline around the demo: FastF1 data collection, feature engineering, leakage-aware modeling, saved model artifacts, reproducible build scripts, notebook-backed analysis, and a deployed-demo-friendly Streamlit interface that reads bundled static lap data instead of depending on live API calls.

## Dashboard Value Proposition

The dashboard is designed as a recruiter-facing slice of the larger analysis, not as a replacement for a real F1 strategy tool. It connects three model outputs into one race-state view:

1. **Pit-window forecast:** estimates how many laps remain before a stop becomes strategically viable.
2. **Undercut / overcut probabilities:** compares two common rival-focused strategy options using trained classifiers.
3. **Context panels:** shows degradation and weather analysis so the prediction is easier to interpret.

For a hiring manager, the app demonstrates practical machine-learning product work: translating a noisy sports dataset into features, training and evaluating models, packaging artifacts, and presenting model output in a way a non-specialist can inspect quickly.

## Run The App

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal. The app uses `data/f1_2024_static_laps.csv.gz` at runtime, so the demo can load without making live FastF1 requests.

If the static runtime lap file needs to be regenerated from a populated local FastF1 cache:

```bash
python scripts/build_static_laps_cache.py
```

## Contributions

Cooper Kerr built and maintains the repository structure, data engineering workflow, reproducible build scripts, notebook orchestration, pit-window forecasting pipeline, undercut and overcut strategy pipelines, tire-degradation analysis, and leakage fixes. Minh Le contributed probability modeling work used in the race-position predictability section.

## Technical Highlights

- **Data pipeline:** uses FastF1 timing data, cached local artifacts, and section-specific datasets under `data/`.
- **Model packaging:** stores trained models and analysis summaries under `models/` with `joblib`, allowing the report and app to load the same artifacts.
- **Evaluation discipline:** uses holdout seasons or leave-one-race-out validation where appropriate, and explicitly documents leakage fixes in the race-position analysis.
- **Demo architecture:** keeps Streamlit runtime lightweight by loading prebuilt model/data artifacts instead of rebuilding datasets on page load.
- **Explainability:** includes feature-importance, SHAP, calibration, and context plots for strategy sections where they are useful.

## Model And Evaluation Highlights

| Area | Current Result |
|---|---|
| Weather / temperature | Compound-specific OLS interaction model with R² = 0.380 on 27,998 filtered laps |
| Pit-window forecasting | 2024 holdout MAE 2.21 laps, RMSE 3.41 laps, 58.5% within ±2 laps, 85.2% within ±5 laps |
| Undercut model | Strategy classification with ROC-AUC around 0.80; Logistic Regression reaches 0.771 accuracy / 0.825 ROC-AUC, while XGBoost favors stronger success-case recall |
| Overcut model | Strategy classification with roughly 0.81 accuracy and 0.80 ROC-AUC in the report summary |
| Race predictability | Leave-one-race-out comparison with corrected lap-50 accuracy of 66.5% for 2024 and 63.3% for 2025 |

## Scope

This branch is organized around the final report notebook:

- canonical deliverable: [notebooks/01_milestone_report.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/01_milestone_report.ipynb)
- report philosophy: keep the final report notebook light
- heavy data pulls, feature engineering, and figure generation should live in `scripts/` or in the companion analysis notebooks that already own a section

The repo answers five research questions plus one integrated race-day example:

1. Tire degradation across a stint
2. Weather and temperature effects on lap time
3. Pit-window forecasting
4. Pit strategy: undercut vs overcut
5. Race position predictability
6. Integrated Hungary race-day strategy example

## Final Report Map

This is the important ownership model for the current branch.

| Report Section | Topic | Canonical Source | What The Report Loads |
|---|---|---|---|
| `5.1` | Tire degradation | `scripts/build_degradation_dataset.py` | `models/f1_degradation_analysis.pkl` and three saved figures |
| `5.2` | Weather / temperature | `scripts/build_weather_analysis.py` | `data/f1_weather_dataset.csv`, `data/f1_weather_coefficients.csv`, `models/f1_weather_analysis.pkl`, `weather_laptime_scatter.png` |
| `5.3` | Pit-window forecasting | `notebooks/02_pit_window_forecasting.ipynb` | `data/f1_pit_window_labels.csv`, `models/f1_pit_window_model_tuned.pkl` |
| `5.4` | Undercut / overcut | `scripts/build_undercut_dataset.py`, `scripts/build_overcut_dataset.py`, plus notebooks `03` and `04` for analysis | `data/f1_undercut_dataset.csv`, `data/f1_overcut_dataset.csv`, `models/f1_undercut_model.pkl`, `models/f1_overcut_model.pkl` |
| `5.5` | Race predictability | `scripts/build_position_comparison_analysis.py` | `data/f1_position_comparison_curve.csv`, `models/f1_position_comparison_analysis.pkl`, `position_comparison_curve.png` |
| `5.6` | Integrated dashboard | inline final report logic using `5.3` and `5.4` artifacts | pit-window, undercut, overcut artifacts and `integrated_race_dashboard.png` |

## Research Questions

| # | Question | Current Branch Implementation |
|---|---|---|
| 1 | How does lap pace evolve across a stint, and what does that imply for pit timing? | Descriptive degradation analysis with stint-level summaries |
| 2 | How do weather and temperature variables relate to lap time? | Lightweight merged lap-weather OLS based on notebook `06` |
| 3 | When will a driver's pit window open? | Gradient-boosted regressor on lap-level race state |
| 4 | When is an undercut or overcut likely to succeed? | Binary classification pipelines for both decisions |
| 5 | How predictable is final finishing position as a race unfolds? | Lightweight checkpoint comparison based on notebook `05` |

## Repository Structure

```text
F1-Predictive-Analysis-Project/
├── notebooks/
│   ├── 01_milestone_report.ipynb
│   ├── 02_pit_window_forecasting.ipynb
│   ├── 03_undercut_prediction.ipynb
│   ├── 04_overcut_prediction.ipynb
│   ├── 05_race_position_predictability_modeling.ipynb
│   └── 06_temperature_effect_modeling.ipynb
├── scripts/
│   ├── build_degradation_dataset.py
│   ├── build_weather_analysis.py
│   ├── build_position_comparison_analysis.py
│   ├── f1_strategy_common.py
│   ├── build_undercut_dataset.py
│   ├── build_overcut_dataset.py
│   ├── test_undercut_model.py
│   └── test_overcut_model.py
├── data/
│   ├── f1_degradation_dataset.csv
│   ├── f1_weather_dataset.csv
│   ├── f1_weather_coefficients.csv
│   ├── f1_position_comparison_curve.csv
│   ├── f1_pit_window_labels.csv
│   ├── f1_2024_static_laps.csv.gz
│   ├── f1_undercut_dataset.csv
│   └── f1_overcut_dataset.csv
├── models/
│   ├── f1_degradation_analysis.pkl
│   ├── f1_weather_analysis.pkl
│   ├── f1_position_comparison_analysis.pkl
│   ├── f1_pit_window_model_tuned.pkl
│   ├── f1_undercut_model.pkl
│   └── f1_overcut_model.pkl
├── outputs/
│   └── figures/
├── requirements.txt
└── f1-cache/
```

## Notebook Roles

### [01_milestone_report.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/01_milestone_report.ipynb)

The final deliverable. It should remain mostly report-facing:

- load saved artifacts
- print compact summaries
- display saved figures
- avoid becoming the main training notebook

### [02_pit_window_forecasting.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/02_pit_window_forecasting.ipynb)

Owns the pit-window pipeline. This is the one section that is still notebook-driven rather than script-driven on this branch.

### [03_undercut_prediction.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/03_undercut_prediction.ipynb)

Analysis notebook for the undercut dataset and classifier. The build script owns artifact generation; the notebook owns most of the evaluation narrative and visuals.

### [04_overcut_prediction.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/04_overcut_prediction.ipynb)

Analysis notebook for the overcut dataset and classifier. Same role as notebook `03`.

### [05_race_position_predictability_modeling.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/05_race_position_predictability_modeling.ipynb)

Companion notebook for the lighter race-predictability framing now used in report `5.5`. It is useful context, but the report-facing artifact contract is owned by `scripts/build_position_comparison_analysis.py`.

### [06_temperature_effect_modeling.ipynb](/Users/cooperkerr/F1-Predictive-Analysis-Project/notebooks/06_temperature_effect_modeling.ipynb)

Source notebook for the canonical weather interaction model used in report `5.2`; the script now mirrors this notebook's race/year scope, filters, and formula for artifact generation.

## Script Roles

### [build_degradation_dataset.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/build_degradation_dataset.py)

Builds the degradation analysis artifacts consumed by report `5.1`:

- stint summaries
- pace-profile table
- peak-age statistics
- stint-length vs degradation correlation
- three report figures

The degradation finding is compound-dependent: pace typically improves after a brief warm-up and then flattens, with HARD and MEDIUM showing little to no late-stint median uptick and SOFT showing only a possible late uptick in sparsely supported tail bins.

### [build_weather_analysis.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/build_weather_analysis.py)

Builds the canonical weather section used in report `5.2`:

- six-race, 2021-2025 notebook `06` sample
- lap / weather merge by timestamp
- pit-lap and undefined-compound filtering
- compound-specific track-temperature interaction OLS with HC2 robust standard errors
- compact coefficient table
- single scatter plot used by the report
- current fit: R² = 0.380 on 27,998 filtered laps

### [build_position_comparison_analysis.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/build_position_comparison_analysis.py)

Builds the simplified race-position comparison used in report `5.5`:

- seasons compared: `2024` vs `2025`
- race subset: full shared notebook `05` race list
- checkpoints: `5, 10, 20, 30, 40, 50`
- features: tyre life, stint, pit stops so far, gap to leader, 3-lap pace delta, and tyre compound
- evaluation: leave-one-race-out `GroupKFold` by Grand Prix
- leakage control: current `Position` is excluded from the headline model
- current corrected lap-50 accuracy: `66.5%` for 2024 and `63.3%` for 2025

### [build_undercut_dataset.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/build_undercut_dataset.py)

Builds the undercut dataset and trains the undercut model artifact used in report `5.4` and `5.6`.

### [build_overcut_dataset.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/build_overcut_dataset.py)

Builds the overcut dataset and trains the overcut model artifact used in report `5.4` and `5.6`.

### [f1_strategy_common.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/f1_strategy_common.py)

Shared helper functions for strategy dataset generation, including gap time series construction, pit-loss calculation, short-window pace estimates, degradation deltas, and Safety Car / VSC interval checks.

### [test_undercut_model.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/test_undercut_model.py) and [test_overcut_model.py](/Users/cooperkerr/F1-Predictive-Analysis-Project/scripts/test_overcut_model.py)

Interactive inspection harnesses for the trained strategy models. These are developer tools, not report dependencies.

## Important Artifacts

### Degradation

- `data/f1_degradation_dataset.csv`
- `models/f1_degradation_analysis.pkl`
- `outputs/figures/degradation_pace_profile.png`
- `outputs/figures/peak_age_by_compound.png`
- `outputs/figures/stint_length_vs_degradation.png`

### Weather

- `data/f1_weather_dataset.csv`
- `data/f1_weather_coefficients.csv`
- `models/f1_weather_analysis.pkl`
- `outputs/figures/weather_laptime_scatter.png`

### Pit Window

- `data/f1_pit_window_labels.csv`
- `models/f1_pit_window_model_tuned.pkl`
- supporting figures under `outputs/figures/pit_window_*`
- Current 2024 holdout: MAE 2.21 laps, RMSE 3.41 laps, 58.5% within ±2 laps, 85.2% within ±5 laps.

### Undercut

- `data/f1_undercut_dataset.csv`
- `models/f1_undercut_model.pkl`
- supporting figures under `outputs/figures/undercut_*`

### Overcut

- `data/f1_overcut_dataset.csv`
- `models/f1_overcut_model.pkl`
- supporting figures under `outputs/figures/overcut_*`

### Position Comparison

- `data/f1_position_comparison_curve.csv`
- `models/f1_position_comparison_analysis.pkl`
- `outputs/figures/position_comparison_curve.png`

### Integrated Dashboard

- `outputs/figures/integrated_race_dashboard.png`

This figure is generated from the final report notebook using the pit-window, undercut, and overcut artifacts.

## Environment

- Python `3.12` expected
- dependencies listed in [requirements.txt](/Users/cooperkerr/F1-Predictive-Analysis-Project/requirements.txt)
- FastF1 local cache directory: `f1-cache/`
- Streamlit app runtime lap data: `data/f1_2024_static_laps.csv.gz`

The Streamlit app uses the bundled 2024 lap data file instead of making live FastF1 calls at page load. This keeps deployed demos independent of F1 live-timing endpoint availability and hosting-provider IP blocks. Regenerate it from a populated local FastF1 cache with:

```bash
python scripts/build_static_laps_cache.py
```

Basic setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild Commands

### Degradation

```bash
python scripts/build_degradation_dataset.py
```

### Weather

```bash
python scripts/build_weather_analysis.py
```

### Position comparison used in report `5.5`

```bash
python scripts/build_position_comparison_analysis.py
```

### Undercut and overcut

```bash
python scripts/build_undercut_dataset.py
python scripts/build_overcut_dataset.py
```

### Pit-window section

The pit-window workflow is still notebook-driven:

```bash
jupyter notebook notebooks/02_pit_window_forecasting.ipynb
```

### Final report sanity check

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_milestone_report.ipynb
```

## Recommended Execution Order

If you need to rebuild the branch from artifacts:

1. Run `build_degradation_dataset.py`
2. Run `build_weather_analysis.py`
3. Ensure `02_pit_window_forecasting.ipynb` has already produced `f1_pit_window_labels.csv` and `f1_pit_window_model_tuned.pkl`
4. Run `build_undercut_dataset.py`
5. Run `build_overcut_dataset.py`
6. Run `build_position_comparison_analysis.py`
7. Execute `01_milestone_report.ipynb`

## Notes And Constraints

- This branch intentionally uses the lighter `05` and `06` ideas for report sections `5.5` and `5.2`.
- The older full position-model pipeline has been removed from this branch.
- The integrated dashboard in `5.6` no longer uses an older position-bucket model; it now uses only the pit-window, undercut, and overcut pipelines.
- `05` and `06` are still present as companion notebooks, but they are not the source of truth for saved report artifacts.
- `CLAUDE.md` may contain local workflow notes that are not part of the final deliverable.

## Project Limitations

- The models use public timing data only; they do not include team radio, tire inventory constraints, pit-crew execution details, or proprietary simulator estimates.
- Safety Car, VSC, red-flag, and weather-shift effects can still create race outcomes that are hard to predict from pre-event lap-state features.
- Driver and team effects are simplified, so the strategy models should be treated as decision-support signals rather than tactical guarantees.
- The dashboard is a demo over saved 2024 lap data; it is intentionally optimized for reliability and reviewability rather than live race operation.

## Remaining Technical Debt

- The pit-window pipeline is still notebook-driven while the other report-facing sections are mostly script-backed.
- Companion notebooks `05` and `06` still duplicate some executable logic from their corresponding scripts.

## Suggested Next Steps

- Move the pit-window training workflow from notebook-only execution into a script-backed artifact builder.
- Add a compact dashboard demo note or hosted screenshot for reviewers who do not want to run the app locally.
- Add driver/team features and richer traffic context to the strategy models.
- Add a small smoke test that verifies all required Streamlit artifacts exist before deployment.
