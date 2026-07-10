"""
Shared helpers for undercut and overcut strategy pipelines.

Only generic timing/lap utilities live here. Strategy-specific event
construction and success labels stay in the individual pipeline scripts.
"""

import numpy as np

FUEL_BURN_RATE = 1.8
FUEL_LAP_EFFECT = 0.035
PACE_WINDOW = 4
DEG_WINDOW = 5


def build_gap_timeseries(session):
    """
    Return same-lap gaps and adjacent drivers using LapStartTime ordering.

    Undercut uses car_ahead_driver. Overcut also needs car_behind_driver, so
    this shared version returns both.
    """
    laps = session.laps.copy()
    laps = laps[laps['IsAccurate'] == True].copy()
    laps = laps.dropna(subset=['LapStartTime'])
    laps = laps.sort_values(['LapNumber', 'LapStartTime']).reset_index(drop=True)
    grp = laps.groupby('LapNumber', sort=False)

    laps['ahead_LapStartTime'] = grp['LapStartTime'].shift(1)
    laps['car_ahead_driver'] = grp['Driver'].shift(1)
    laps['behind_LapStartTime'] = grp['LapStartTime'].shift(-1)
    laps['car_behind_driver'] = grp['Driver'].shift(-1)

    laps['gap_ahead'] = (
        laps['LapStartTime'] - laps['ahead_LapStartTime']
    ).dt.total_seconds()
    laps['gap_behind'] = (
        laps['behind_LapStartTime'] - laps['LapStartTime']
    ).dt.total_seconds()

    return laps[[
        'Driver',
        'LapNumber',
        'gap_ahead',
        'gap_behind',
        'car_ahead_driver',
        'car_behind_driver',
    ]].copy()


def compute_pit_loss(session):
    laps = session.laps.copy()
    in_laps = laps[
        laps['PitInTime'].notna()
    ][['Driver', 'LapNumber', 'PitInTime']].copy()
    out_laps = laps[
        laps['PitOutTime'].notna()
    ][['Driver', 'LapNumber', 'PitOutTime']].copy()
    in_laps['OutLapNumber'] = in_laps['LapNumber'] + 1
    out_laps = out_laps.rename(columns={'LapNumber': 'OutLapNumber'})
    merged = in_laps.merge(out_laps, on=['Driver', 'OutLapNumber'], how='inner')
    if merged.empty:
        return 22.0
    return float((merged['PitOutTime'] - merged['PitInTime']).dt.total_seconds().median())


def get_pace(driver_laps, up_to_lap, window=PACE_WINDOW):
    recent = driver_laps[
        (driver_laps['LapNumber'] < up_to_lap) &
        (driver_laps['LapNumber'] >= up_to_lap - window) &
        (driver_laps['IsAccurate'] == True) &
        (driver_laps['TrackStatus'].astype(str) == '1')
    ].copy()
    if recent.empty:
        return np.nan
    lt = recent['LapTime'].dt.total_seconds()
    corrected = lt - (recent['LapNumber'] - 1) * FUEL_BURN_RATE * FUEL_LAP_EFFECT
    return float(corrected.median())


def get_deg_delta(driver_laps, up_to_lap, window=DEG_WINDOW):
    recent = driver_laps[
        (driver_laps['LapNumber'] < up_to_lap) &
        (driver_laps['LapNumber'] >= up_to_lap - window) &
        (driver_laps['IsAccurate'] == True) &
        (driver_laps['TrackStatus'].astype(str) == '1') &
        (driver_laps['TyreLife'].notna())
    ].copy()
    if len(recent) < 3:
        return np.nan
    lt = recent['LapTime'].dt.total_seconds()
    corrected = lt - (recent['LapNumber'] - 1) * FUEL_BURN_RATE * FUEL_LAP_EFFECT
    x = recent['TyreLife'].values.astype(float)
    y = corrected.values
    if np.ptp(x) < 0.5:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def has_sc_between(all_laps, driver, lap_start, lap_end):
    window = all_laps[
        (all_laps['Driver'] == driver) &
        (all_laps['LapNumber'] >= lap_start) &
        (all_laps['LapNumber'] <= lap_end)
    ]['TrackStatus'].dropna().astype(str)
    return any(any(c in s for c in ['4', '5', '6']) for s in window)
