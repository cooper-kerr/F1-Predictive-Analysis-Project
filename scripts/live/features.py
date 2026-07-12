from __future__ import annotations

import pandas as pd

from race_strategy import build_strategy_context
from race_state import RaceStateSnapshot


def latest_lap_for_driver(snapshot: RaceStateSnapshot, driver: str) -> int | None:
    laps = snapshot.laps
    if laps.empty:
        return None
    rows = laps[laps["Driver"] == driver]
    if rows.empty:
        return None
    return int(rows["LapNumber"].max())


def strategy_context_from_snapshot(
    snapshot: RaceStateSnapshot,
    driver: str,
    lap: int | None = None,
):
    """Build the model context using the same code path for replay and live data."""

    selected_lap = lap if lap is not None else latest_lap_for_driver(snapshot, driver)
    if selected_lap is None:
        return None, "No lap state exists for the selected driver."
    return build_strategy_context(snapshot.to_session(), driver, int(selected_lap))


def strategy_feature_rows(context: dict | None) -> pd.DataFrame:
    if context is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "driver": context["driver"],
                "rival": context["rival"],
                "lap": context["lap"],
                "compound": context["compound"],
                "gap_ahead": context["gap_ahead"],
                "tire_age": context["tire_age"],
                "rival_tire_age": context["rival_tire_age"],
                "own_pace": context["own_pace"],
                "threat_pace": context["threat_pace"],
                "pace_delta": context["pace_delta"],
                "deg_delta": context["deg_delta"],
                "ca_deg_delta": context["ca_deg_delta"],
                "pit_loss": context["pit_loss"],
                "race_progress": context["race_progress"],
            }
        ]
    )

