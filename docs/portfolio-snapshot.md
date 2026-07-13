# Portfolio Snapshot: F1 Predictive Analysis Project

Snapshot date: 2026-07-12 MST

Portfolio repo inspected: `/Users/cooperkerr/cooper-kerr.github.io`

Portfolio files changed: none

## Project Identity

- Project name: F1 Predictive Analysis Project
- Local path: `/Users/cooperkerr/F1-Predictive-Analysis-Project`
- Public repo: `https://github.com/cooper-kerr/F1-Predictive-Analysis-Project`
- GitHub API check: repo is public, default branch is `Main`, no homepage URL is configured, pushed at `2026-07-12T17:07:15Z`
- Local branch: `Main`
- Dirty state at snapshot time: one user-owned uncommitted deletion, `D docs/research/model_actionability_and_uncertainty_review.md`
- Primary stack: Python, FastF1, pandas, scikit-learn, XGBoost, statsmodels, SHAP, Plotly, Streamlit, joblib, SQLite for live snapshot storage
- Product purpose: an explainable Formula One race-strategy intelligence demo using public timing data, saved model artifacts, static replay data, and early live-race ingestion support.

## Recruiter-Visible Surfaces

- Public GitHub repo: `https://github.com/cooper-kerr/F1-Predictive-Analysis-Project`
- Public README: checked in at `README.md`; it includes a recruiter quick read, model metrics, app run command, contribution note, technical highlights, repository map, limitations, and next steps.
- Live app URL: not evidenced. GitHub repo `homepage` is `null`, and the existing portfolio entry says no public demo URL is listed.
- Portfolio entry: existing project match in `/Users/cooperkerr/cooper-kerr.github.io/data/projects.json` by title, slug `f1-predictive-analysis`, and repo URL.
- Screenshots/assets: `outputs/figures/integrated_race_dashboard.png` plus saved model/evaluation figures under `outputs/figures/`. Portfolio currently has no image paths for this project.

## Evidence Ledger

### Repo And Git

- `git remote -v`: origin is `https://github.com/cooper-kerr/F1-Predictive-Analysis-Project`
- `git branch --show-current`: `Main`
- `git status --short`: `D docs/research/model_actionability_and_uncertainty_review.md`
- `git log --oneline --decorate -n 30`: latest commits include `06c71ed update`, `d9d8957 live analysis`, `d8e227b code refactor`, `b933faf design update`, `3c1b6eb dashboard update`, `246bd20 front end`, `d5f6129 fix: pin numpy/sklearn/shap/xgboost/pandas to exact training-environment versions to fix sklearn _loss unpickling error on Streamlit Cloud`, and `abe41d1 Refresh milestone report with all corrected pipelines`.

### Recent Material Changes

- `06c71ed update`: changed `app.py`, `scripts/live/predict.py`, `tests/test_app_smoke.py`, and added `docs/research/model_actionability_and_uncertainty_review.md`; 335 insertions and 19 deletions.
- `d9d8957 live analysis`: added live race-state architecture: `scripts/live/*`, `scripts/race_state.py`, OpenF1 provider normalization, SQLite store, quality gates, live prediction adapter, and tests; 1,671 insertions across 19 files.
- `d8e227b code refactor`: introduced shared strategy modules and tests: `scripts/race_strategy.py`, `scripts/strategy_attempts.py`, `scripts/strategy_features.py`, `scripts/dataset_runner.py`, and corresponding tests; 6,282 insertions and 772 deletions across 83 files.
- `b933faf design update`: substantially restyled `app.py`; 303 insertions and 75 deletions.
- `3c1b6eb dashboard update`: expanded `app.py`, added README content, and added `tests/test_app_smoke.py`; 790 insertions and 130 deletions.
- `246bd20 front end`: added Streamlit/Plotly-facing frontend changes; `app.py` and `requirements.txt`.
- `d5f6129`: pinned model-runtime dependency versions in `requirements.txt` for Streamlit Cloud unpickling compatibility.
- `abe41d1`: refreshed `notebooks/01_milestone_report.ipynb` and saved report figures, including `outputs/figures/integrated_race_dashboard.png`.

### Runtime And App Evidence

- `README.md` describes the app as a recruiter-facing race-strategy demo that lets reviewers choose a 2024 race, driver, and lap, then inspect pit-window timing, undercut/overcut pressure, tyre degradation, and weather context.
- `app.py` loads saved artifacts for pit window, undercut, overcut, degradation, weather, and position comparison from `models/`.
- `app.py` uses `data/f1_2024_static_laps.csv.gz` through `ReplayRaceStateAdapter`, so replay mode does not need live FastF1 calls at page load.
- `app.py` exposes Data Mode options `Replay` and `Live`; live mode reads cached snapshots from `data/live_race.sqlite` or `LIVE_DB_PATH`, and falls back to replay when no active snapshot is available.
- `scripts/race_state.py` defines a provider-neutral `RaceStateSnapshot` plus `ReplayRaceStateAdapter` for bundled 2024 static laps.
- `scripts/live/providers/openf1.py` maps OpenF1-shaped endpoint payloads into `RaceStateSnapshot`.
- `scripts/live/store.py` persists normalized live snapshots to SQLite tables for sessions, laps, positions, stints, pits, weather, track status, and metadata.
- `scripts/live/quality.py` suppresses predictions for stale feeds, missing driver/rival/gap/pace fields, wet mode, Safety Car/VSC, and other quality flags.
- `scripts/live/predict.py` converts pit-window, undercut, overcut, and race-state context into calls such as `Attack with undercut`, `Extend for overcut`, `Hold position`, `Marginal call`, or `Suppressed`.

### Data And Model Evidence

CSV row counts from checked-in `data/` files:

- `data/f1_pit_window_labels.csv`: 46,976 rows, 27 columns.
- `data/f1_undercut_dataset.csv`: 966 rows, 20 columns.
- `data/f1_overcut_dataset.csv`: 932 rows, 21 columns.
- `data/f1_degradation_dataset.csv`: 64,955 rows, 11 columns.
- `data/f1_weather_dataset.csv`: 27,998 rows, 41 columns.
- `data/f1_weather_coefficients.csv`: 23 rows, 4 columns.
- `data/f1_position_comparison_curve.csv`: 12 rows, 7 columns.
- `data/f1_2024_static_laps.csv.gz`: 25,574 rows, 11 columns.

README-reported metrics:

- Weather / temperature: compound-specific OLS interaction model with `R² = 0.380` on 27,998 filtered laps.
- Pit-window forecasting: 2024 holdout MAE `2.21` laps, RMSE `3.41` laps, `58.5%` within +/-2 laps, `85.2%` within +/-5 laps.
- Pit-window full-horizon audit in `app.py`: MAE `2.82` laps, RMSE `4.54` laps, `54.9%` within +/-2 laps, `79.9%` within +/-5 laps.
- Undercut model: ROC-AUC around `0.80`; Logistic Regression reaches `0.771` accuracy and `0.825` ROC-AUC, while XGBoost favors stronger success-case recall.
- Overcut model: roughly `0.81` accuracy and `0.80` ROC-AUC in the report summary.
- Race predictability: corrected lap-50 accuracy of `66.5%` for 2024 and `63.3%` for 2025, evaluated with leave-one-race-out comparison.

### Tests And Verification Evidence

- `tests/test_app_smoke.py` checks that app artifacts load with expected schemas, bundled pit-window prediction works, static laps support strategy context, action recommendations are actionable, and position comparison artifacts exist.
- `tests/test_live_openf1.py` checks OpenF1 payload normalization into provider-neutral live snapshots.
- `tests/test_live_quality.py` checks suppression for stale feed, Safety Car/VSC, and wet mode.
- `tests/test_live_store.py` covers SQLite snapshot storage.
- `tests/test_race_state.py` checks replay adapter behavior and session compatibility.
- Verification attempted during this snapshot: `python3 -m pytest -q` failed because the current local Python does not have `pytest` installed. Earlier CSV inspection using only the standard library succeeded. Direct `pandas`/`joblib` artifact loading was not available in the current local Python because those packages were not installed.

## Current-State Summary

This project is a public, active Formula One strategy analytics repo with both research/report depth and a demo-oriented Streamlit app. The current product story is strongest when framed as "decision support from public timing data" rather than live race automation.

The project now demonstrates:

- Public timing-data ingestion and feature engineering through FastF1-backed datasets.
- Multiple strategy analytics pipelines: tire degradation, weather effects, pit-window forecasting, undercut/overcut classification, and race-position predictability.
- Model packaging with saved `joblib` artifacts that the app and report both consume.
- A recruiter-facing Streamlit dashboard that uses bundled 2024 lap data for reliable replay inspection.
- A provider-neutral race-state boundary and early OpenF1 live-ingestion layer, including SQLite persistence and quality gates.
- Testing around app smoke behavior, strategy features, live provider normalization, live-store persistence, and quality-gate suppression.

Important wording boundary: the app is deployed-demo-friendly and Streamlit Cloud compatibility was addressed, but no public hosted app URL is evidenced. Portfolio wording should not say "shipped live demo" unless a URL is added and verified.

## Portfolio Match And Staleness

Matched existing portfolio entry:

- Target file: `/Users/cooperkerr/cooper-kerr.github.io/data/projects.json`
- Existing title: `F1 Predictive Analysis Project`
- Existing slug: `f1-predictive-analysis`
- Existing repo URL: `https://github.com/cooper-kerr/F1-Predictive-Analysis-Project`

Current stale or incomplete portfolio facts:

- Replace: the current description says "The Streamlit demo is currently local-only; no public demo URL is listed yet." This remains true about no public URL, but it undersells the current bundled replay app, static runtime data, and live-ingestion work.
- Add/replace: the current tech stack only says `Python`, `FastF1`, `Streamlit`, `machine learning`. It should add recruiter-readable terms for `scikit-learn`, `XGBoost`, `statsmodels`, `Plotly`, and `SQLite` if space allows.
- Add: `image_paths` should include an F1 dashboard screenshot copied into the portfolio repo, likely from `outputs/figures/integrated_race_dashboard.png` or a fresh app screenshot.
- Verify: whether a public Streamlit URL exists outside this repo. None is evidenced in GitHub metadata, README, or portfolio files.
- Add/replace: career context should mention the live/replay architecture only as "built early live-ingestion support" or "added provider-neutral live race-state architecture", not as a fully operational live race product.

## Portfolio-Ready Project Entry Fields

Suggested replacement description for `data/projects.json`:

```json
[
  "Built an explainable Formula One strategy analytics demo on public FastF1 timing data, combining tire degradation, weather-effects analysis, pit-window forecasting, undercut/overcut strategy classification, and race-position predictability.",
  "Packaged the project like a reviewable ML product: saved model artifacts, reproducible build scripts, notebook-backed analysis, static 2024 replay data, and a Streamlit dashboard that lets reviewers inspect race, driver, lap, model output, and supporting evidence without live API calls.",
  "Extended the dashboard with provider-neutral race-state architecture, OpenF1 snapshot normalization, SQLite-backed live snapshot storage, quality gates for stale/wet/Safety Car states, and tests around app smoke behavior, live ingestion, race-state adapters, and strategy features."
]
```

Suggested `tech_stack`:

```json
[
  "Python",
  "FastF1",
  "Streamlit",
  "scikit-learn",
  "XGBoost",
  "statsmodels",
  "Plotly",
  "SQLite"
]
```

Suggested `relevance_tags`:

```json
[
  "data-science",
  "machine-learning",
  "sports-analytics",
  "python",
  "dashboard"
]
```

Suggested links:

```json
[
  {
    "label": "GitHub Repo",
    "url": "https://github.com/cooper-kerr/F1-Predictive-Analysis-Project"
  }
]
```

Add a live app link only after a hosted URL is verified.

Suggested image fields after copying or generating the asset in the portfolio repo:

```json
"image_paths": [
  "images/F1_Strategy_Dashboard.png"
],
"image_captions": [
  "Streamlit race-strategy dashboard showing pit-window timing, undercut/overcut probabilities, and supporting race-state evidence."
]
```

## Suggested `career_context.md` Bullets

Use these under the F1 project section after updating `data/projects.json`:

- Built an explainable Formula One strategy analytics demo on public FastF1 timing data, combining tire degradation, weather-effects analysis, pit-window forecasting, undercut/overcut strategy classification, and race-position predictability.
- Packaged saved model artifacts and static 2024 replay lap data into a Streamlit dashboard so reviewers can inspect race, driver, lap, model output, and supporting evidence without depending on live API calls.
- Added early live-race infrastructure with provider-neutral race snapshots, OpenF1 payload normalization, SQLite-backed snapshot storage, and quality gates for stale feeds, wet conditions, Safety Car/VSC states, and missing direct-rival features.
- Current evidenced scale: 46,976 pit-window labels, 966 undercut attempts, 932 overcut attempts, 64,955 degradation lap rows, 27,998 weather lap rows, and 25,574 bundled 2024 replay lap rows.

## Suggested Skill Additions

Target file: `/Users/cooperkerr/cooper-kerr.github.io/data/skills.json`

Suggested add/replace actions:

- Add to `ml_data_libraries`: `XGBoost`, `statsmodels`, `SHAP`, `Plotly`
- Add to `tools`: `Streamlit`, `GitHub CLI`
- Optional add to `programming_languages` or a future project-specific stack only: keep `FastF1` in the project tech stack rather than global skills unless the portfolio supports domain libraries.

## Screenshot Or Asset Recommendations

- Best immediate portfolio asset: copy `outputs/figures/integrated_race_dashboard.png` into the portfolio repo as `images/F1_Strategy_Dashboard.png`, then reference it in `data/projects.json`.
- Better asset if time allows: run `streamlit run app.py` in a dependency-ready environment and capture the first screen of the interactive dashboard with race, driver, lap, pit-window, undercut, overcut, and direct-rival metrics visible.
- Do not use notebook-only figures as the first image if the goal is recruiter product signal; the dashboard screenshot better communicates deployable interface work.

## Risks, Unknowns, And Claims To Verify

- Public hosted app URL: unknown and not evidenced. Do not add a live demo link until verified.
- Test status: local `python3 -m pytest -q` could not run because `pytest` is missing from the current Python environment.
- Pickle artifact internals: not loaded during this snapshot because local Python lacks `joblib` and `pandas`; model metric claims are based on README/app constants, CSV row counts, tests, and checked-in file presence.
- Dirty worktree: `docs/research/model_actionability_and_uncertainty_review.md` is deleted in the working tree. This should be committed, restored, or intentionally left out before portfolio updates cite it.
- Live mode: code supports cached OpenF1 snapshots and quality-gated predictions, but no evidence shows a public live service running continuously.
- Authorship: README states Cooper maintains the project and Minh Le contributed probability modeling work incorporated from a University of Utah class project. Keep attribution careful in recruiter wording.

## Recommended Portfolio Changes

- `/Users/cooperkerr/cooper-kerr.github.io/data/projects.json`: replace the F1 project `description` with the three suggested bullets above.
- `/Users/cooperkerr/cooper-kerr.github.io/data/projects.json`: replace the F1 project `tech_stack` with the suggested expanded stack.
- `/Users/cooperkerr/cooper-kerr.github.io/data/projects.json`: add `dashboard` to F1 `relevance_tags`.
- `/Users/cooperkerr/cooper-kerr.github.io/data/projects.json`: add `image_paths` and `image_captions` after an image is copied into the portfolio repo.
- `/Users/cooperkerr/cooper-kerr.github.io/career_context.md`: regenerate or manually mirror the revised F1 project bullets and stack after JSON edits.
- `/Users/cooperkerr/cooper-kerr.github.io/data/skills.json`: add `XGBoost`, `statsmodels`, `SHAP`, `Plotly`, and `Streamlit` if the global skill model should reflect this project.
- Verify action: check whether a public Streamlit or hosted demo URL exists. If yes, add a `Live Demo` link; if no, keep only the GitHub repo link.
