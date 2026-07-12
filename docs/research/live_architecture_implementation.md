# Live Architecture Implementation

Research date: 2026-07-12

## Provider Capability Matrix

| Source | Useful fields | Live fit | Caveats |
|---|---|---:|---|
| OpenF1 | Laps, positions, intervals, pit stops, weather, race control, drivers, sessions | Primary MVP provider | OpenF1 documents historical access without authentication on the free tier, with JSON/CSV responses and rate limits of 3 requests/second and 30 requests/minute. Live access is sponsor-tier, with higher limits and REST/MQTT/WebSocket support. |
| FastF1 live timing recorder | Raw live timing capture for post-session replay | Archive/replay only | FastF1 says the live timing client saves live data but cannot process it in real time. It also has runtime and recording-continuity constraints. |
| FastF1 low-level API | Timing, car data, position, session status, race control, lap count, driver info, weather | Historical/replay support | FastF1 marks low-level API functions as private/future-unstable, so they should not be the primary live contract. |
| Streamlit cache/database reads | Cached model artifacts, cached DB snapshots | UI integration | Streamlit recommends `st.cache_data` for serializable data/API/DB query results with TTL, and `st.cache_resource` for global resources such as ML models and DB connections. |
| SQLite WAL | Local normalized live store | MVP storage | SQLite WAL improves local read/write concurrency because readers and writers can proceed concurrently, while applications should still tolerate occasional busy states. |

## Recommendation

Use a replay-first architecture with one live ingest worker:

1. `RaceStateSnapshot` remains the only app-facing race-state shape.
2. Replay mode uses `ReplayRaceStateAdapter` and bundled `data/f1_2024_static_laps.csv.gz`.
3. Live mode reads the latest normalized snapshot from SQLite.
4. One ingest worker polls OpenF1, maps payloads into canonical tables, and writes SQLite snapshots.
5. Streamlit never polls OpenF1 directly; it reads cached SQLite snapshots with a short TTL.
6. Strategy Signals consume the same `RaceStateSnapshot` contract for replay and live-shaped data.
7. Quality Gates suppress actionable calls when the feed is stale, Safety Car/VSC is active, wet mode is detected, or direct-rival/model inputs are missing.

## MVP Storage

The implemented SQLite store creates these tables:

- `sessions`
- `drivers`
- `laps`
- `positions_intervals`
- `stints`
- `pits`
- `weather`
- `track_status`
- `snapshot_metadata`

This is intentionally local-first. A hosted production version can swap the persistence layer later while preserving the `RaceStateSnapshot` boundary.

## Rate-Limit Notes

OpenF1 free historical access is enough for fixture-based development and replay experiments. Live mode may require sponsor access. The local client centralizes request pacing with a minimum interval between requests, but production polling should still group endpoints by cadence:

- Positions/intervals/race control/pits: 2-5 seconds.
- Weather: 30-60 seconds.
- Session metadata/drivers: before session start and then infrequently.

## Sources

- OpenF1 documents live timing/session capabilities, JSON/CSV support, free historical access, and sponsor live access: https://openf1.org/
- FastF1 live timing docs state that the recorder saves live data but cannot process it in real time: https://docs.fastf1.dev/livetiming.html
- FastF1 API docs mark low-level API functions as private/future-unstable: https://docs.fastf1.dev/api.html
- Streamlit caching docs distinguish cached data, cached resources, TTL, API calls, DB queries, and ML model caching: https://docs.streamlit.io/develop/concepts/architecture/caching
- SQLite WAL docs describe improved read/write concurrency and `SQLITE_BUSY` caveats: https://www.sqlite.org/wal.html
