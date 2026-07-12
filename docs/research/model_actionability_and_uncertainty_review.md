# Model Actionability and Uncertainty Review

Research date: 2026-07-12

## Goal

Evaluate whether the app's actionable insights are logically sound, whether the models explain what is going on clearly enough, whether the data is sufficient, and how the current 2026 F1 calendar shape affects replay, model, app, and live features.

## Bottom Line

The app is directionally sound as a recruiter-facing race-strategy intelligence demo. It has a coherent decision layer: pit-window urgency, undercut/overcut comparison against a direct rival, degradation context, weather evidence, position predictability, and live quality gates. The current logic is strongest when framed as "decision support from public timing data" and weakest when the UI wording implies calibrated race-engineering confidence.

The main issue is not that the app is making nonsense calls. The issue is that several outputs are point estimates or heuristic confidence labels without enough local support, calibrated uncertainty, or missingness context. The app should downgrade the language from "high confidence" to "strong model separation" unless it also displays calibration/support evidence.

## Current Evidence Base

| Area | Current support | Assessment |
|---|---:|---|
| Pit window | 46,976 rows from 2022-2024; 2024 replay data at runtime | Useful, but current headline metric is optimistic because unresolved horizon cases are excluded from evaluation. |
| Undercut | 966 attempts from 2022-2024; 278 successes | Good for directional rival-specific modeling; thin for precise probability claims by circuit/compound/gap bin. |
| Overcut | 932 attempts from 2022-2024; 234 successes | Same as undercut; useful signal, not enough for race-team-grade calibration. |
| Degradation | 64,955 lap rows and 2,868 stint summaries | Strong descriptive support; needs interval/support visibility when used in live action explanations. |
| Weather | 27,998 filtered laps from six races, 2021-2025 | Useful explanatory regression; too narrow for broad live weather adjustment. |
| Position predictability | 2024 and 2025 comparison, leave-one-race-out GroupKFold | Statistically the most honest section because it shows fold SE, baselines, rows, and leakage control. |

Dataset counts above come from the checked-in CSVs and saved artifacts under `data/` and `models/`.

## Logical Soundness of Actionable Insights

The action layer in `scripts/live/predict.py` is well-shaped: it turns model outputs into calls like `Attack with undercut`, `Extend for overcut`, `Hold position`, `Marginal call`, or `Suppressed`. It compares undercut and overcut probabilities against historical base rates, measures the probability edge, and gives reasons using pit-window urgency, gap, recent pace, and degradation advantage.

The quality gate in `scripts/live/quality.py` is a strong design choice. It blocks or flags stale live feeds, Safety Car/VSC, wet mode, missing rival/gap/pace/tyre state, low clean-lap support, estimated pit loss, and unsupported live providers.

The weak point is the word `confidence`. In the current implementation, confidence is based on the absolute undercut-vs-overcut probability edge, not a confidence interval, local sample support, calibration error, or model posterior uncertainty. That makes `High` a model-separation label, not a statistical confidence label.

Recommendation: rename or qualify it as `Signal strength`, or compute a support-aware confidence using probability edge, calibration-bin error, gap-bin count, quality flags, and whether critical features were median-filled.

## Pit-Window Caveat

The pit-window notebook reports a 2024 holdout MAE of 2.21 laps, RMSE of 3.41 laps, 58.5% within +/-2 laps, and 85.2% within +/-5 laps. That is useful, but it is evaluated only on resolved rows where `laps_until_open` exists.

A pit-window audit found that using a sentinel of 21 laps for rows where the window does not open within the 20-lap horizon changes the 2024 holdout to:

- MAE: 2.82 laps
- RMSE: 4.54 laps
- within +/-2 laps: 54.9%
- within +/-5 laps: 79.9%

That does not invalidate the model. It means the UI should say "typical error is about 2-3 laps, and longer/no-window cases are harder" rather than presenting a single pit-window number as equally reliable everywhere.

There are also leakage risks to audit before treating the pit-window model as live-ready: future-looking "pits soon" features, full-race Safety Car/VSC rates, full-race median gap/pit-loss summaries, and end-of-lap state semantics.

## Explanation Quality

The app explains decisions better than most score-only demos. It shows the direct rival, gap, pit-window estimate, undercut/overcut probabilities, recent pace advantage, degradation advantage, feature vectors, historical gap-bin success rates, and evidence tabs.

The explanations still need three upgrades:

1. Show the distinction between model reasons and race facts. Example: "gap is 3.2s" is an observed fact; "undercut likely succeeds" is a model inference.
2. Show support next to each inference. Example: `gap band 2-4s, n=48 historical undercut attempts`.
3. Surface uncertainty where the decision depends on it. Example: pit-window estimate should show typical error, and undercut/overcut probabilities should show calibration/support labels.

## Data Sufficiency and Model Expansion

The data is enough for the current portfolio/demo scope. It is not enough for high-trust live decision automation.

Priority model expansion:

1. Rebuild pit-window evaluation with unresolved horizon rows included and remove future-derived features from live-facing feature sets.
2. Persist calibration metrics for undercut and overcut: Brier score, calibration slope/intercept, ECE, and calibration-bin counts.
3. Add local support features to action calls: nearest gap-bin count, compound count, circuit count, and whether the selected context is outside training support.
4. Expand live features around OpenF1 availability: real intervals, pits, stints, race control, weather, and provider timestamp age.
5. Split weather into explanatory historical context versus live weather-adjusted prediction. The current weather model should stay explanatory until live weather coverage and residual diagnostics are stronger.
6. Add missingness reports by model: rows dropped for required fields, median-filled fields, wet/Safety Car exclusions, and low-support degradation bins.

## 2026 Season Breaks and App Impact

The official F1 calendar page for 2026 lists the current date context clearly: as of 2026-07-12, Great Britain on 2026-07-03 to 2026-07-05 is complete, Belgium is next on 2026-07-17 to 2026-07-19, Hungary follows on 2026-07-24 to 2026-07-26, and the Netherlands resumes on 2026-08-21 to 2026-08-23. The current calendar page lists 22 rounds from Australia to Abu Dhabi.

Operational impact:

- The app's replay mode is still 2024-only, so 2026 calendar changes do not affect static replay correctness unless the UI claims to represent the current season.
- Live mode should not hard-code race order or assume 2024 circuits. It should treat schedule/session metadata as provider data.
- The post-Hungary break from 2026-07-26 to 2026-08-21 is useful for ingestion and model work: add 2026 race fixtures progressively, validate live OpenF1 snapshots, and refresh race/session dropdowns.
- New or renamed calendar entries such as Barcelona-Catalunya and Spain/Madrid should be tested against `circuit` naming assumptions before they are used in pit-loss or circuit-median features.
- Calendar gaps matter for freshness UX: during off weeks, the app should say "no active session" rather than showing stale live calls.

## Visualization and Uncertainty Recommendations

Use uncertainty-aware visuals where the decision depends on estimation:

- Pit window: show point estimate plus typical error band and context-specific MAE. Avoid a naked "opens in 3.1 laps" metric.
- Undercut/overcut: show probability, lift over base rate, calibration/support label, and gap-bin count. Avoid treating probability difference alone as confidence.
- Degradation: keep the current support table, but add per-bin interval bands or at least a visible low-support state in the action board.
- Weather: show coefficients with standard errors/p-values as explanatory evidence; do not imply live causal adjustment unless current weather is in the live feature vector.
- Position: keep the fold SE and baseline line. It is the clearest statistically honest view in the app.

## Implementation-Safe Next Steps

1. Change `Confidence` to `Signal strength` in the Action Board, or compute a support-aware confidence score.
2. Add pit-window error language to the Pit Window tab: resolved-row MAE plus full-horizon MAE after sentinel handling.
3. Promote quality flags from caption text into visible badges.
4. Add support labels to undercut/overcut probabilities using the existing gap-bin `count` in `load_strategy_action_benchmarks()`.
5. Persist calibration metrics in `models/f1_undercut_model.pkl` and `models/f1_overcut_model.pkl`.
6. Add a schedule/session availability state for live mode so off-week periods do not look like broken or stale predictions.

## Sources

- Repo source: `app.py`, `scripts/live/predict.py`, `scripts/live/quality.py`, `scripts/strategy_attempts.py`, `scripts/strategy_features.py`, `scripts/build_undercut_dataset.py`, `scripts/build_overcut_dataset.py`, `scripts/build_weather_analysis.py`, `scripts/build_position_comparison_analysis.py`, and checked-in model/data artifacts.
- Prior repo research: `docs/research/actionable_f1_strategy_modeling.md` and `docs/research/live_architecture_implementation.md`.
- Official F1 2026 calendar: https://www.formula1.com/en/racing/2026
- OpenF1 API capability and access notes: https://openf1.org/
