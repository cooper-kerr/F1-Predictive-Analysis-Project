from pathlib import Path

import fastf1
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "f1-cache"
OUTPUT_PATH = DATA_DIR / "f1_2024_static_laps.csv.gz"
YEAR = 2024
SESSION_TYPE = "R"
LAP_COLUMNS = [
    "Driver",
    "LapNumber",
    "LapTime",
    "LapStartTime",
    "IsAccurate",
    "TrackStatus",
    "TyreLife",
    "Compound",
    "PitInTime",
    "PitOutTime",
]


def main():
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    pit_labels = pd.read_csv(DATA_DIR / "f1_pit_window_labels.csv")
    races = sorted(pit_labels[pit_labels["year"] == YEAR]["circuit"].dropna().unique())
    frames = []

    for race in races:
        session = fastf1.get_session(YEAR, race, SESSION_TYPE)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()
        frame = laps[LAP_COLUMNS].copy()
        frame.insert(0, "Race", race)
        frames.append(frame)
        print(f"{race}: {frame.shape[0]} laps")

    out = pd.concat(frames, ignore_index=True)
    for col in ["LapTime", "LapStartTime", "PitInTime", "PitOutTime"]:
        out[col] = out[col].dt.total_seconds()

    out.to_csv(OUTPUT_PATH, index=False, compression="gzip")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with shape {out.shape}")


if __name__ == "__main__":
    main()
