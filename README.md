# F1 Predictive Analysis Project

Data science project applying machine learning and statistical analysis to Formula One race strategy using the [FastF1](https://docs.fastf1.dev/) API.

**Authors:** Cooper Kerr, Isaac Middlemas, Minh Le — University of Utah

## Research Questions

| # | Question | Primary Implementation | Status |
|---|---|---|---|
| 1 | How does lap pace evolve across a stint, and what does that imply for pit timing? | `scripts/build_degradation_dataset.py` | ✅ |
| 2 | How do weather and temperature variables relate to lap time? | `scripts/build_weather_analysis.py` | ✅ |
| 3 | When will a driver's pit window open? | `notebooks/02_pit_window_forecasting.ipynb` | ✅ |
| 4 | When is an undercut or overcut likely to succeed? | `scripts/build_undercut_dataset.py`, `scripts/build_overcut_dataset.py` | ✅ |
| 5 | How predictable is final finishing position as a race unfolds? | `scripts/build_position_dataset.py`, `scripts/build_position_comparison_analysis.py` | ✅ |

## Canonical Report Workflow

The final report notebook is [01_milestone_report.ipynb](notebooks/01_milestone_report.ipynb). It is intended to stay lightweight:

- heavy data pulls and figure generation should happen in `scripts/`
- the report notebook should mainly load saved datasets, models, and figures
- notebooks `05` and `06` are companion notebooks that document the lighter framing now used in report sections `5.5` and `5.2`

Current report-facing script ownership:

- Tire degradation: [build_degradation_dataset.py](scripts/build_degradation_dataset.py)
- Weather / temperature: [build_weather_analysis.py](scripts/build_weather_analysis.py)
- Position comparison used in report `5.5`: [build_position_comparison_analysis.py](scripts/build_position_comparison_analysis.py)
- Full position model artifacts still used elsewhere in the project: [build_position_dataset.py](scripts/build_position_dataset.py)

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
│
├── scripts/
│   ├── build_degradation_dataset.py
│   ├── build_weather_analysis.py
│   ├── build_position_dataset.py
│   ├── build_position_comparison_analysis.py
│   ├── build_undercut_dataset.py
│   ├── build_overcut_dataset.py
│   ├── test_undercut_model.py
│   └── test_overcut_model.py
│
├── data/
│   ├── f1_degradation_dataset.csv
│   ├── f1_weather_dataset.csv
│   ├── f1_weather_coefficients.csv
│   ├── f1_position_dataset.csv
│   ├── f1_position_curve.csv
│   ├── f1_position_comparison_curve.csv
│   ├── f1_pit_window_labels.csv
│   ├── f1_undercut_dataset.csv
│   └── f1_overcut_dataset.csv
│
├── models/
│   ├── f1_degradation_analysis.pkl
│   ├── f1_weather_analysis.pkl
│   ├── f1_position_model.pkl
│   ├── f1_position_comparison_analysis.pkl
│   ├── f1_pit_window_model.pkl
│   ├── f1_pit_window_model_tuned.pkl
│   ├── f1_undercut_model.pkl
│   └── f1_overcut_model.pkl
│
├── outputs/
│   └── figures/
│
├── docs/
├── requirements.txt
└── f1-cache/
```

## Key Artifacts

### Weather / Temperature

- Dataset: `data/f1_weather_dataset.csv`
- Coefficient table: `data/f1_weather_coefficients.csv`
- Artifact: `models/f1_weather_analysis.pkl`
- Figure: `outputs/figures/weather_laptime_scatter.png`

### Position Prediction

There are now two position-related outputs:

- Full model pipeline:
  - `data/f1_position_dataset.csv`
  - `data/f1_position_curve.csv`
  - `models/f1_position_model.pkl`
  - `outputs/figures/position_curve.png`
- Simplified 2024 vs 2025 comparison used in report `5.5`:
  - `data/f1_position_comparison_curve.csv`
  - `models/f1_position_comparison_analysis.pkl`
  - `outputs/figures/position_comparison_curve.png`

## Usage

### Rebuild degradation artifacts

```bash
python scripts/build_degradation_dataset.py
```

### Rebuild weather artifacts

```bash
python scripts/build_weather_analysis.py
```

### Rebuild full position-model artifacts

```bash
python scripts/build_position_dataset.py
```

### Rebuild the simplified report-facing position comparison

```bash
python scripts/build_position_comparison_analysis.py
```

### Rebuild strategy datasets and models

```bash
python scripts/build_undercut_dataset.py
python scripts/build_overcut_dataset.py
```

### Inspect trained strategy models interactively

```bash
python scripts/test_undercut_model.py inspect
python scripts/test_undercut_model.py driver VER
python scripts/test_overcut_model.py inspect
python scripts/test_overcut_model.py driver SAI
```

## Notes

- Python 3.12 is expected.
- FastF1 caching is enabled in the build scripts.
- The pit-window pipeline currently remains notebook-driven rather than script-driven.
- The worktree may contain additional generated artifacts not listed above; the table reflects the intended core pipeline outputs.
