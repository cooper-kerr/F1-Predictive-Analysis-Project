# F1 Milestone Notebook: 5-Question Restructure — Design

**Date:** 2026-04-17
**Scope:** `notebooks/01_milestone_report.ipynb` and one new script
**Authors:** Cooper Kerr, Isaac Middlemas

## 1. Problem

The milestone notebook claims to answer a set of research questions, but the framing is internally inconsistent:

- **§1 Project Description** lists 3 questions (position, tire degradation, pit window).
- **§4 Methods** covers only 2 questions (position + pit window).
- **§5 Results** delivers 5 distinct analyses (position, pit window, undercut, overcut, weather).
- **§9 Conclusions** wraps up only 3 questions.

The intended framing is 5 research questions, each with its own section:

1. Pit window forecasting
2. Race position prediction
3. Undercut + overcut (combined as one pit-strategy question)
4. Weather/temperature effects on lap times
5. Tire degradation model

The tire degradation model is the biggest gap: §2.5 and §7 claim "per-compound, per-circuit quadratic degradation fits," but the codebase only has a linear in-stint slope estimator (`get_deg_delta` in `build_undercut_dataset.py` and `build_overcut_dataset.py`) used as an input feature. No per-compound, per-circuit curves are fitted or saved anywhere.

## 2. Goals

- Restructure `notebooks/01_milestone_report.ipynb` so that exactly 5 research questions are explicit, each with its own section under §5.
- Build a real tire degradation model so §5.1 matches the notebook's existing prose claims.
- Rewrite §1, §4, and §9 so the narrative (intro → methods → results → conclusions) is internally consistent and lists the same 5 questions in the same order throughout.
- Promote the weather/temperature analysis from "Bonus Analysis" to a first-class research question, with moderate analytical expansion.
- Merge the undercut and overcut sections into a single pit-strategy section that frames them as complementary decisions.
- Keep the Hungarian GP integrated dashboard as a showcase section after the 5 research questions.

## 3. Non-Goals

- **No changes to existing models.** The position RF, pit-window GBM, and undercut/overcut LR+XGBoost models stay as they are — artifacts, features, and metrics untouched.
- **No changes to companion notebooks** (`02_pit_window_forecasting.ipynb`, `03_undercut_prediction.ipynb`, `04_overcut_prediction.ipynb`).
- **Tire degradation curves are NOT plumbed into other models.** The existing pit-window pipeline continues to use the in-stint linear-slope `deg_delta` feature. Swapping in the new curves is a future improvement, not part of this restructure.
- **No new FastF1 pulls.** Everything runs from the existing local cache.
- **No changes to `data/`, `models/`, or `outputs/figures/`** beyond the new tire-degradation artifacts listed in §5.

## 4. Restructured §5 Layout

Final Results ordering, build-up from foundational to highest-level:

```
5. Results
   5.1 Tire Degradation Model                      (NEW — biggest new work)
   5.2 Weather & Temperature Effects on Lap Time   (EXPANDED from former §5.6 "Bonus")
   5.3 Pit Window Forecasting                      (existing, minor edits)
   5.4 Pit Strategy: Undercut vs Overcut           (MERGED from former §5.3 + §5.4)
   5.5 Race Position Prediction                    (existing, minor edits)
   5.6 Integrated Race-Day Example — 2024 Hungary  (existing §5.5, renumbered)
```

Rationale: foundational mechanical models first (degradation curves, weather regression), then the model that consumes them as a concept (pit window), then the decision layer (pit strategy), then the top-level outcome (race position). The Hungary dashboard stays at the end as a composition demo.

## 5. §5.1 — Tire Degradation Model (new work)

### Research question

*"For a given tire compound at a given circuit, how does lap pace decay as a function of tire age?"*

### Pipeline — `scripts/build_degradation_dataset.py` (new)

1. **Data loading.** Seasons 2022-2023 for training, 2024 for holdout. Uses the existing FastF1 cache. A `--rebuild` flag matches the other build scripts: absent → load cached CSV; present → re-fetch from FastF1.
2. **Lap filtering.** Exclude any lap whose `TrackStatus` string contains `'4'`, `'5'`, or `'6'` (Safety Car / VSC). Exclude pit-in laps (`PitInTime` not null), pit-out laps (`PitOutTime` not null), and the first lap of each stint (out-lap). Drop laps outside [60 s, 200 s]. Same filter convention as the existing `build_undercut_dataset.py` / `build_overcut_dataset.py`.
3. **Fuel correction.** Apply `LapTimeSeconds − (LapNumber − 1) × 1.8 × 0.035` to strip out weight-related pace gains.
4. **Baseline.** Per (driver, stint): baseline pace = median of the first 3 green laps of the stint. The regression target is `Δpace = fuel_corrected_pace − baseline`.
5. **Per-(compound, circuit) quadratic fit.** Circuits are identified by `EventName` string; 2022 and 2023 observations at the same circuit are pooled into one fit. For each `(compound, circuit)` pair in `{SOFT, MEDIUM, HARD} × (distinct 2022-2023 circuits)`, fit `Δpace(age) = a·age + b·age²` using `numpy.polyfit(age, Δpace, 2)` where `age = TyreLife`. Intermediate/wet compounds excluded — insufficient dry-race data for a stable fit.
6. **Artifacts.**
   - `data/f1_degradation_dataset.csv` — the filtered, fuel-corrected lap dataset used for fitting.
   - `models/f1_degradation_curves.pkl` — `joblib` dict: `{(compound, circuit): {'coef': [a, b], 'n_laps': int, 'r2_train': float}}`. Follows the existing "joblib dict with features list" artifact convention.
7. **Minimum-sample guard.** Skip any `(compound, circuit)` cell with fewer than 30 training laps; log these as skipped. On the 2024 holdout, any skipped cell — and any 2024 circuit not seen in 2022-2023 — defaults to the compound-global curve (pooled across all circuits for that compound).

### Evaluation (2024 holdout)

| Metric | Meaning |
|---|---|
| **R² per (compound, circuit)** | Variance explained by tire age alone |
| **MAE per compound** (pooled across circuits) | Mean absolute pace-loss prediction error, s/lap |
| **Curve figure** | Three-panel matplotlib figure (SOFT / MEDIUM / HARD), each overlaying per-circuit quadratic curves, 2024 green-lap scatter underneath |
| **Live-slope comparison** | Table showing variance explained by the fitted curves vs. the in-stint linear-slope `deg_delta` estimator currently used in the pit-window pipeline |

### Notebook section layout

- Intro paragraph with the research question.
- Code cell loading the fitted curves + running 2024 evaluation.
- Metrics table (R² summary + MAE per compound).
- Curve-overlay figure (saved to `outputs/figures/degradation_curves.png`).
- 3-4 bullet findings (steepest-degrading compound, circuit variance, etc.).

## 6. §5.2 — Weather & Temperature Effects (expanded)

### Research question

*"How do track temperature, air temperature, and weather conditions affect lap pace, and does tire compound modulate that sensitivity?"*

### Changes from current §5.6

- Remove the "Bonus Analysis" header and the disclaimer sentence that says it does not feed into the primary questions.
- Keep the existing OLS regression (lap time on TrackTemp, AirTemp, Compound, etc. with HC2 robust SEs) and the existing scatter plot.
- **Add** `C(Compound) * TrackTemp` interaction terms to the OLS formula to estimate per-compound temperature sensitivity.
- **Add** a compound-sensitivity plot: three regression lines (SOFT / MEDIUM / HARD) of fuel-corrected pace vs. track temp on shared axes, with 95% confidence bands.
- **Add** residual diagnostics: residuals-vs-fitted plot and Q-Q plot to confirm OLS assumptions are not badly violated.
- **Add** a closing paragraph tying the temperature sensitivity back to §5.1: hotter tracks accelerate degradation, so the quadratic curves from §5.1 are effectively a cross-section at average temperature.

## 7. §5.4 — Pit Strategy: Undercut vs Overcut (merged)

### Research question

*"Given two cars with a gap between them, which pit-strategy direction wins — pitting now to undercut the car ahead, or staying out to overcut the car that just pitted?"*

### Section structure

- **Shared setup paragraph** — defines the two sides, frames them as complementary decisions, notes both models train on 2022-2023 with 2024 holdout and use LR baseline + XGBoost primary.
- **Feature parity table** — single table showing both models share most features (gap, pace delta, tire-age delta, degradation slope), with the sign-flip explained (undercut uses `tire_age_advantage = ca_age − driver_age`; overcut uses `tire_age_delta = stay_out_age − ca_age`).
- **Side-by-side results table** — 4 rows (Undercut LR / Undercut XGBoost / Overcut LR / Overcut XGBoost), columns for accuracy / ROC-AUC / F1 / base-rate.
- **Combined SHAP takeaway** — one paragraph naming the top drivers for each direction: undercut wins on `tire_age_advantage`, overcut wins on `pace_delta` + large `tire_age_delta`.
- **Asymmetry finding** — one paragraph on why the two aren't mirror images: undercut 44% base-rate success vs. overcut 25%; overcuts are materially harder, which is why the XGBoost F1 lift over LR is larger on the overcut side.

Word count roughly halves vs. current §5.3 + §5.4 combined by removing redundant framing, but all existing numbers are preserved.

## 8. §1, §4, §9 Narrative Updates

### §1 Project Description — rewrite the numbered list

Replace the current 3-question list with all 5, each with a one-sentence motivation:

1. **Tire Degradation Modeling** — How does lap pace decay with tire age, and does the decay shape depend on compound and circuit?
2. **Weather & Temperature Effects** — How do track temperature, air temperature, and related weather covariates influence lap times, and does tire compound modulate that sensitivity?
3. **Pit Window Forecasting** — Given the current race state (gap, pace, tire age, circuit), how many laps remain before a viable pit window opens?
4. **Pit Strategy — Undercut vs Overcut** — When two cars are within strategic range, which pit-timing decision (pit now to undercut, or stay out to overcut) is more likely to gain track position?
5. **Race Position Prediction** — At what lap of a race can a driver's final finishing position be predicted with meaningful accuracy, and how does that confidence grow as the race unfolds?

### §4 Methods — three new subsections, reorder to match §5

Final §4 ordering after restructure:
- **§4.1 Tire Degradation** (new, ~150 words) — target, features (compound, circuit, tire age, fuel-corrected pace), model (per-group quadratic via `polyfit`), train/test split, evaluation.
- **§4.2 Weather Effects** (new, ~100 words) — OLS with `C(Compound) * TrackTemp` interaction, HC2 robust SEs, diagnostics plan.
- **§4.3 Pit Window Forecasting** (existing — renumbered from §4.2).
- **§4.4 Pit Strategy — Undercut vs Overcut** (new merged section, ~200 words) — unified framing, feature parity with sign conventions, LR baseline + XGBoost primary, 2022-2023 → 2024 holdout.
- **§4.5 Race Position Prediction** (existing — renumbered from §4.1).

### §9 Conclusions — one paragraph per question, build-up order

1. **Tire degradation.** Headline R² / MAE per compound. Which compound has the steepest curve. How much circuit variance remains after grouping.
2. **Weather.** Lead coefficient for track temp on pace, compound interaction sign and magnitude. Confirmation that OLS assumptions hold.
3. **Pit window.** Existing language kept — MAE 2.26 laps on 2024 holdout, `closing_rate` dominates feature importance.
4. **Pit strategy.** Existing undercut + overcut conclusions combined, names the asymmetry (overcuts harder) and the different dominant signals per direction.
5. **Race position.** Existing language kept — 65% @ lap 5, 89% @ lap 50, mid-race resolution finding.

Closing paragraph (scientific contribution + future work) stays largely as-is; future-work list updated to reference all 5 questions.

### §6, §7, §8 — light pass

- **§6 Peer feedback.** Update references to "three questions" to match the 5-question framing. The peer-raised overfitting concern (position classifier) is already addressed — keep that text.
- **§7 Process summary.** Update the models checklist to include the new tire-degradation model script and artifact.
- **§8 Limitations.** Add a short paragraph about tire-degradation curve limitations: curves assume in-stint degradation is the dominant signal, ignore driver style and track evolution, pool across drivers so individual stint management differences are averaged out.

## 9. Execution Order

1. Write `scripts/build_degradation_dataset.py` — new pipeline, fits curves on 2022-2023, saves `models/f1_degradation_curves.pkl` and `data/f1_degradation_dataset.csv`. Includes `--rebuild` flag.
2. Generate §5.1 figures: three-panel compound curves, saved to `outputs/figures/degradation_curves.png`.
3. Add §5.1 notebook cells: intro, evaluation code, metrics table, figure display, findings bullets.
4. Expand §5.2 weather: add compound × track-temp interaction, compound-sensitivity plot, residual diagnostics; remove "Bonus" framing.
5. Merge §5.3 + §5.4 into new §5.4 Pit Strategy; renumber.
6. Renumber §5.5 Hungary dashboard → §5.6.
7. Rewrite §1 Project Description — 5-question list with motivation.
8. Expand §4 Methods — add §4.1 Degradation, §4.2 Weather, §4.4 Pit Strategy (merged); renumber existing subsections.
9. Rewrite §9 Conclusions — one paragraph per question, build-up order.
10. Light pass on §6, §7, §8 — reference 5 questions; add tire-degradation limitations paragraph to §8.

## 10. Acceptance Criteria

After the restructure, running `notebooks/01_milestone_report.ipynb` top-to-bottom produces:

- **§5.1** with a filled R²/MAE metrics table, per-compound curves figure, findings text referencing real numbers from the 2024 holdout.
- **§5.2** with interaction coefficients in the OLS output, compound-sensitivity regression plot, and Q-Q + residual diagnostic plots.
- **§5.4** as a single cohesive section with both undercut and overcut models compared side-by-side in one results table.
- **§1, §4, §9** all mention exactly 5 questions, in the same order (Degradation → Weather → Pit Window → Pit Strategy → Position), with matching section numbers.
- **No regressions** in existing §5.3 (pit window), §5.5 (position), §5.6 (Hungary dashboard) numerical outputs.

## 11. Dependencies

Hard dependency: step 1 (build script) must complete and write artifacts before step 3 (§5.1 notebook cells) can be authored or executed. Steps 4-10 are independent of the tire-degradation build and can proceed in parallel if useful.

No other external dependencies — FastF1 cache is already populated locally, all Python dependencies are in `requirements.txt`.
