from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

SAFETY_CAR_STATUS_CODES = {"4", "5", "6", "7"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def snapshot_age_seconds(snapshot, now: datetime | None = None) -> float | None:
    if not snapshot.timestamp_utc:
        return None
    now = now or datetime.now(timezone.utc)
    try:
        timestamp = datetime.fromisoformat(str(snapshot.timestamp_utc).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0.0, (now - timestamp).total_seconds())


def latest_track_status(snapshot) -> str | None:
    if not snapshot.laps.empty and "TrackStatus" in snapshot.laps:
        values = snapshot.laps["TrackStatus"].dropna().astype(str)
        if not values.empty:
            return values.iloc[-1]
    if not snapshot.track_status.empty:
        text = " ".join(
            snapshot.track_status.astype(str).fillna("").to_numpy().ravel().tolist()
        ).upper()
        if "SAFETY CAR" in text:
            return "4"
        if "VSC" in text or "VIRTUAL" in text:
            return "6"
    return None


def quality_flags_for_prediction(
    snapshot,
    context: dict | None,
    context_reason: str | None = None,
    max_staleness_seconds: float = 30.0,
    min_clean_laps: int = 3,
) -> list[str]:
    flags = list(snapshot.quality_flags or ())

    age = snapshot_age_seconds(snapshot)
    if snapshot.mode == "live" and age is None:
        flags.append("missing_provider_timestamp")
    if snapshot.mode == "live" and age is not None and age > max_staleness_seconds:
        flags.append("stale_feed")

    status = latest_track_status(snapshot)
    if status and any(code in status for code in SAFETY_CAR_STATUS_CODES):
        flags.append("safety_car_or_vsc")

    if context is None:
        flags.append("missing_driver_state")
        if context_reason:
            flags.append("missing_rival")
        return list(dict.fromkeys(flags))

    for key in ["rival", "gap_ahead", "own_pace", "threat_pace", "tire_age", "rival_tire_age"]:
        if key not in context or pd.isna(context[key]):
            flags.append(f"missing_{key}")

    if str(context.get("compound", "")).upper() in WET_COMPOUNDS:
        flags.append("wet_mode")

    lap = int(context["lap"])
    clean_laps = snapshot.laps[
        (snapshot.laps["Driver"] == context["driver"])
        & (snapshot.laps["LapNumber"] <= lap)
        & (snapshot.laps.get("IsAccurate", True) == True)
        & (snapshot.laps.get("TrackStatus", "1").astype(str) == "1")
    ]
    if len(clean_laps) < min_clean_laps:
        flags.append("low_clean_lap_support")

    if pd.isna(context.get("pit_loss")) or float(context.get("pit_loss", 0)) == 22.0:
        flags.append("estimated_pit_loss")

    if snapshot.mode == "live" and snapshot.provider != "openf1":
        flags.append("unsupported_provider_fields")

    return list(dict.fromkeys(flags))


def suppress_prediction(flags: list[str]) -> bool:
    blocking = {
        "stale_feed",
        "missing_driver_state",
        "missing_rival",
        "missing_gap_ahead",
        "missing_own_pace",
        "missing_threat_pace",
        "wet_mode",
        "safety_car_or_vsc",
    }
    return any(flag in blocking for flag in flags)

