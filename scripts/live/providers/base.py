from abc import ABC, abstractmethod

from race_state import RaceStateSnapshot


class LiveRaceStateProvider(ABC):
    """Minimal provider contract for ingest workers and live snapshots."""

    provider_name: str

    @abstractmethod
    def available_sessions(self) -> list[dict]:
        """Return provider session descriptors."""

    @abstractmethod
    def load_snapshot(self, session, as_of_lap: int | None = None) -> RaceStateSnapshot:
        """Return a normalized snapshot for a session."""

