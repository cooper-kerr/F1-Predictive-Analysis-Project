"""
Schedule-driven FastF1 dataset runner.

Build scripts provide the domain extractor. This module owns schedule lookup,
session loading, empty-session handling, failure reporting, and concatenation.
"""

import pandas as pd


def build_schedule_dataset(
    years,
    extractor,
    item_label,
    *,
    schedule_loader,
    session_loader,
    session_load_kwargs,
    skip_empty_extractions=False,
    empty_extraction_message="no rows",
    header_style="dash",
):
    chunks = []
    for year in years:
        schedule = schedule_loader(year, include_testing=False)
        gp_names = schedule["EventName"].tolist()
        if header_style == "box":
            print(f"\n── {year}: {len(gp_names)} races ──")
        else:
            print(f"\n-- {year}: {len(gp_names)} races --")

        for gp in gp_names:
            try:
                session = session_loader(year, gp, "R")
                session.load(**session_load_kwargs)
                if session.laps is None or len(session.laps) == 0:
                    print(f"  {gp:40s}  SKIP: no lap data")
                    continue

                extracted = extractor(session, year, gp)
                if isinstance(extracted, pd.DataFrame):
                    count = len(extracted)
                    is_empty = extracted.empty
                else:
                    count = len(extracted)
                    is_empty = count == 0

                if skip_empty_extractions and is_empty:
                    print(f"  {gp:40s}  SKIP: {empty_extraction_message}")
                    continue

                chunks.append(extracted)
                print(f"  {gp:40s}  {count:3d} {item_label}")
            except Exception as exc:
                print(f"  {gp:40s}  FAILED: {str(exc)[:80]}")

    if not chunks:
        return pd.DataFrame()
    if all(isinstance(chunk, pd.DataFrame) for chunk in chunks):
        return pd.concat(chunks, ignore_index=True)

    records = []
    for chunk in chunks:
        records.extend(chunk)
    return pd.DataFrame(records)


def build_race_year_dataset(
    races,
    years,
    extractor,
    item_label,
    *,
    session_loader,
    session_load_kwargs,
    skip_empty_extractions=True,
    empty_extraction_message="no rows",
):
    frames = []
    for race in races:
        for year in years:
            try:
                session = session_loader(year, race, "R")
                session.load(**session_load_kwargs)
                extracted = extractor(session, year, race)
                if skip_empty_extractions and extracted.empty:
                    print(f"{year} {race:20s}  SKIP: {empty_extraction_message}")
                    continue
                frames.append(extracted)
                print(f"{year} {race:20s}  {len(extracted):4d} {item_label}")
            except Exception as exc:
                print(f"{year} {race:20s}  FAILED: {str(exc)[:80]}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
