from __future__ import annotations

import argparse
import os
import time

from live.providers.openf1 import OpenF1Client
from live.store import LiveRaceStore


def ingest_once(provider, store: LiveRaceStore, session: dict, as_of_lap: int | None = None):
    snapshot = provider.load_snapshot(session, as_of_lap=as_of_lap)
    store.write_snapshot(snapshot)
    return snapshot


def ingest_loop(
    provider,
    store: LiveRaceStore,
    session: dict,
    poll_seconds: float,
    as_of_lap: int | None = None,
):
    while True:
        ingest_once(provider, store, session, as_of_lap=as_of_lap)
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Poll OpenF1 and persist normalized live snapshots.")
    parser.add_argument("--db", default=os.environ.get("LIVE_DB_PATH", "data/live_race.sqlite"))
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--race-name", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("LIVE_POLL_SECONDS", "5")))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    provider = OpenF1Client(api_key=os.environ.get("F1_PROVIDER_API_KEY"))
    store = LiveRaceStore(args.db)
    session = {
        "session_key": args.session_key,
        "race_name": args.race_name,
        "year": args.year,
        "session_status": "live",
    }
    if args.once:
        ingest_once(provider, store, session)
    else:
        ingest_loop(provider, store, session, args.poll_seconds)


if __name__ == "__main__":
    main()
