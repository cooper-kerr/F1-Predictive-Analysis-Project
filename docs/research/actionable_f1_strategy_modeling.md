# Actionable F1 Strategy Modeling Notes

Research date: 2026-07-11

## Goal

Improve the recruiter-facing Formula 1 strategy dashboard by making each prediction read as an actionable race-state judgment, not only a model score. The dashboard should explain what decision is being considered, what public data supports it, and where the assumption stops being defensible.

## Source Base

- [FastF1 core docs](https://docs.fastf1.dev/core.html): `Session` exposes laps, weather, telemetry, track status, race control messages, and result data after `Session.load`; `Laps` includes lap time, stint, pit-in/out timestamps, compound, tyre life, fresh-tyre flag, track status, end-of-lap position, deletion flags, FastF1-generated rows, and `IsAccurate`.
- [FastF1 weather API docs](https://docs.fastf1.dev/api.html#fastf1.api.weather_data): weather samples include air temperature, humidity, pressure, rainfall, track temperature, wind direction, and wind speed, updated once per minute.
- [FastF1 lap weather join docs](https://docs.fastf1.dev/core.html#fastf1.core.Laps.get_weather_data): `Laps.get_weather_data()` returns one weather point per lap, using the first value within the lap or the last known value before lap end; use raw `Session.weather_data` when finer control is needed.
- [FastF1 Ergast interface docs](https://docs.fastf1.dev/ergast.html): Ergast-like data can be returned as raw JSON-like data or flattened pandas frames, with pagination and a maximum request limit of 1000 results.
- [Jolpica Ergast-compatible API root](https://api.jolpi.ca/ergast/): Ergast-like endpoints expose seasons, races, results, pit stops, laps, standings, and status resources.
- [Jolpica pit stop endpoint example](https://api.jolpi.ca/ergast/f1/2026/1/pitstops/): pit stop records include driver id, lap, stop number, clock time, and duration.
- [Jolpica lap endpoint example](https://api.jolpi.ca/ergast/f1/2026/1/laps/): lap records include lap number and per-driver timing plus position.
- [Jolpica result endpoint example](https://api.jolpi.ca/ergast/f1/2026/results/): race result records include final position, grid, completed laps, finishing status, result time, and fastest-lap details.
- [FIA F1 regulations archive](https://www.fia.com/regulation/category/110): FIA publishes Formula 1 sporting/technical regulations and archives the 2024 sporting regulations used by this project season.
- [Pirelli F1 tyre guide](https://www.pirelli.com/tyres/en-ww/motorsport/car/formula-1): current F1 slick compounds are ordered from hardest to softest, with harder compounds supporting longer stints at lower peak performance and softer compounds favoring grip and warm-up on low-severity circuits; intermediates and wets are separate wet-condition tyres.
- [Todd et al., 2025](https://arxiv.org/abs/2501.04067): race strategy decisions are framed around when to stop and which compound to use; their Mercedes-AMG PETRONAS-linked study models tyre energy/degradation from telemetry and uses explainability methods.
- [Thomas et al., 2025](https://arxiv.org/abs/2501.04068): race strategy can be modeled as selecting tyre compounds and pit timing during a race; the paper emphasizes fast inference, generalization, feature importance, surrogate trees, and counterfactuals for trust.

## Defensible Modeling Assumptions

### Pit Window

Pit-window recommendations are defensible when framed as "is a stop becoming attractive soon?" rather than "this is the optimal lap." Public FastF1 data directly supports the key state features: current lap, scheduled race length, stint, tyre life, compound, pit-in/out markers, lap time, position, track status, and weather context.

Recommended features:

- `laps_remaining`, `stint_lap`, `TyreLife`, `Compound`, `FreshTyre`, `Stint`, `current_position`, and `pit_stops_so_far`.
- Rolling clean-air pace deltas over the previous 3-5 accurate, non-box laps.
- Gap to nearest rival ahead/behind where available from lap positions and timing-derived gaps.
- Pit-loss estimate by race: median in-lap plus out-lap penalty or observed pit-stop duration/gap loss from historical stops in that event.
- Rule and feasibility context: compounds already used, wet/dry state, safety car/VSC status, and whether a stop would rejoin into traffic.

Dashboard action:

- Show a "pit window pressure" band: `Too early`, `Opening`, `Prime window`, `Late risk`.
- Pair the predicted laps-to-window with the top three reasons: tyre age/degradation trend, rival gap/pit-loss feasibility, and weather/neutralization context.
- Label any output as "decision support from public timing data"; do not imply access to fuel load, tyre set history beyond FastF1 `FreshTyre`/`TyreLife`, team simulation, or live tyre carcass data.

### Undercut and Overcut

Undercut/overcut should be modeled as rival-specific attempts, matching the repo language in `CONTEXT.md`. This is defensible because public data can identify the direct rival, relative position, stop lap, stint age, and post-stop position change. It should not be presented as a universal passing probability.

Recommended undercut features:

- Rival ahead, gap to rival, attacker's tyre age/compound, rival's tyre age/compound, attacker's recent pace delta, and expected pit-loss.
- Out-lap vulnerability: attacker's historical out-lap pace on the target compound, excluding inaccurate laps and safety car/VSC periods.
- Traffic risk: expected rejoin position and number of cars within pit-loss range.
- Track status and race-control filters, because FastF1 marks timing issues around safety car/VSC laps and offers `pick_track_status()` and `IsAccurate` for filtering.

Recommended overcut features:

- Remaining tyre performance of the car staying out: recent pace stability, stint age, compound, and degradation slope.
- Rival's cold-tyre/out-lap risk: rival's out-lap pace on new compound and traffic after stopping.
- Track evolution/weather signal: improving lap-time baseline, drying track, falling fuel load proxy by lap number, and rainfall/track temperature changes.

Dashboard action:

- Present "Undercut case" and "Overcut case" as side-by-side cards for the selected rival: probability, required gap swing, observed gap, and one-line reason.
- Add a "fragility" label when success depends on traffic, safety car/VSC, wet tyres, or sparse historical examples.
- Keep the target label as "gained position after both cars completed relevant stops" rather than "overtook on track."

### Degradation

Public lap data supports an approximate degradation model, not a full tyre-energy model. FastF1 provides compound, tyre life, stint, lap timing, track status, weather, and telemetry channels, while the Mercedes-linked tyre-energy paper shows that optimal stop decisions are tightly coupled to tyre degradation/energy but uses private team telemetry not available in this repo.

Recommended features:

- Stint-relative lap time adjusted by race/lap baseline, compound, tyre life, fresh tyre flag, and track status.
- Rolling pace slope over clean laps only; exclude pit in/out laps, deleted laps, FastF1-generated rows, inaccurate rows, and safety car/VSC windows.
- Compound interaction terms: `Compound * TyreLife`, `Compound * TrackTemp`, and optionally `Compound * stint_phase`.
- Driver/team normalization to avoid treating car pace as tyre wear.

Dashboard action:

- Show "degradation signal" as a trend with confidence/coverage, not a single absolute tyre-health number.
- Separate "pace improving from fuel burn/track evolution" from "tyre degradation worsening" by comparing the selected driver with field or teammate lap baselines.
- Flag sparse tail stints where long tyre-life examples are rare.

### Weather and Track Evolution

FastF1 weather data is usable for lap-level context but is minute-sampled, so sub-lap weather changes should not drive precise claims. Track evolution is not directly observed as a single FastF1 channel; it should be inferred from field-normalized lap-time movement after filtering pit laps, inaccurate laps, safety car/VSC periods, and rainfall transitions.

Recommended features:

- Weather: air temperature, track temperature, humidity, pressure, rainfall, wind direction, and wind speed.
- Weather deltas: rolling changes in track temperature, rainfall transitions, and wind changes rather than only absolute values.
- Track evolution proxy: median clean-lap pace by lap, within race and compound, compared to each driver's expected pace.
- Wet/dry mode flag: treat slick, intermediate, and wet tyre phases as separate regimes.

Dashboard action:

- Add a weather/track-evolution strip: `Track temp`, `Rainfall`, `Wind`, `Field pace trend`.
- If rainfall or wet compounds appear, switch strategy language from dry-compound pit window to "condition-change window."
- Use raw `Session.weather_data` for event-level trend plots; use `Laps.get_weather_data()` for per-lap model rows.

### Position Predictability

Final position predictability is defensible as a checkpoint model if it is explicitly leakage-controlled. FastF1 lap rows include end-of-lap `Position`; Ergast-like results provide final classification, grid, laps completed, and status. Current position is informative but can dominate the task, so the dashboard should distinguish "race-state predictability with current position" from a harder "pace/context-only" model.

Recommended features:

- Checkpoint lap, grid position, current position if using a stateful model, pit stops so far, tyre age, compound, gap to leader/rival, recent pace delta, and status/retirement handling.
- Grouped validation by race or season to avoid learning event-specific leakage.
- Separate classes for DNF/retired or filter to classified finishers, then disclose the choice.

Dashboard action:

- Show position predictability as "confidence in holding/improving/losing position range" rather than exact finishing place.
- Include a baseline: current position at checkpoint, grid-only, or previous-lap position. Recruiters can see whether the model beats a simple heuristic.
- Add a leakage note when `Position` is included; make the pace/context-only score available as the cleaner modeling signal.

## Recruiter-Facing Recommendations

1. Convert model outputs into a compact strategy call: "Box soon", "Cover rival", "Extend", or "Hold".
2. For every call, show the decisive race-state facts: gap to rival, pit-loss estimate, tyre age/compound, recent pace delta, and weather/track-status flag.
3. Add data-quality badges: clean laps available, safety car/VSC nearby, wet/dry regime, sparse sample, and FastF1 accuracy filter.
4. Put the rival at the center of undercut/overcut views. A recruiter should immediately see "against whom?" and "what gap swing is required?"
5. Use explainability that matches the project level: permutation/SHAP for trained models, plus deterministic reason strings for domain features.
6. Preserve the current public-data honesty. State that FastF1/Ergast-like data supports timing, lap, stint, weather, pit stop, and result analysis, but not private tyre energy, exact fuel mass, team tyre allocations beyond exposed fields, or proprietary simulations.

## Implementation-Safe Next Steps

- Add a dashboard "strategy call" layer that consumes existing pit-window, undercut, overcut, degradation, weather, and position artifacts without retraining first.
- Add a shared feature glossary in the UI so feature names map to race concepts: pit loss, clean pace, rejoin traffic, degradation slope, track evolution.
- Add validation cards beside each model card: primary metric, baseline metric, holdout/grouping method, and sample size.
- Prefer incremental model improvements after the UI makes assumptions visible: first improve rival/gap and pit-loss features, then weather/track-evolution features, then position-predictability leakage variants.
