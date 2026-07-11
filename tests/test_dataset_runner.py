import pandas as pd

from dataset_runner import build_race_year_dataset, build_schedule_dataset


class FakeSession:
    def __init__(self, laps):
        self.laps = laps
        self.loaded_with = None

    def load(self, **kwargs):
        self.loaded_with = kwargs


def _schedule_loader(year, include_testing=False):
    assert include_testing is False
    return pd.DataFrame({"EventName": ["Race A", "Race B"]})


def test_build_schedule_dataset_collects_record_lists():
    sessions = {
        (2024, "Race A", "R"): FakeSession(pd.DataFrame({"lap": [1]})),
        (2024, "Race B", "R"): FakeSession(pd.DataFrame({"lap": [1, 2]})),
    }

    def session_loader(year, gp, kind):
        return sessions[(year, gp, kind)]

    def extractor(session, year, gp):
        return [{"year": year, "race": gp, "n_laps": len(session.laps)}]

    df = build_schedule_dataset(
        [2024],
        extractor,
        "records",
        schedule_loader=_schedule_loader,
        session_loader=session_loader,
        session_load_kwargs={"laps": True},
    )

    assert df.to_dict("records") == [
        {"year": 2024, "race": "Race A", "n_laps": 1},
        {"year": 2024, "race": "Race B", "n_laps": 2},
    ]
    assert sessions[(2024, "Race A", "R")].loaded_with == {"laps": True}


def test_build_schedule_dataset_concatenates_dataframes_and_skips_empty():
    sessions = {
        (2024, "Race A", "R"): FakeSession(pd.DataFrame({"lap": [1]})),
        (2024, "Race B", "R"): FakeSession(pd.DataFrame({"lap": [1]})),
    }

    def session_loader(year, gp, kind):
        return sessions[(year, gp, kind)]

    def extractor(session, year, gp):
        if gp == "Race B":
            return pd.DataFrame()
        return pd.DataFrame([{"year": year, "race": gp}])

    df = build_schedule_dataset(
        [2024],
        extractor,
        "rows",
        schedule_loader=_schedule_loader,
        session_loader=session_loader,
        session_load_kwargs={},
        skip_empty_extractions=True,
        empty_extraction_message="no rows",
    )

    assert df.to_dict("records") == [{"year": 2024, "race": "Race A"}]


def test_build_schedule_dataset_skips_empty_laps_and_failed_sessions():
    sessions = {
        (2024, "Race A", "R"): FakeSession(pd.DataFrame()),
    }

    def session_loader(year, gp, kind):
        if gp == "Race B":
            raise RuntimeError("boom")
        return sessions[(year, gp, kind)]

    def extractor(session, year, gp):
        raise AssertionError("empty and failed sessions should not extract")

    df = build_schedule_dataset(
        [2024],
        extractor,
        "records",
        schedule_loader=_schedule_loader,
        session_loader=session_loader,
        session_load_kwargs={},
    )

    assert df.empty


def test_build_race_year_dataset_collects_fixed_grid_dataframes():
    sessions = {
        (2024, "Race A", "R"): FakeSession(pd.DataFrame({"lap": [1]})),
        (2025, "Race A", "R"): FakeSession(pd.DataFrame({"lap": [1, 2]})),
    }

    def session_loader(year, race, kind):
        return sessions[(year, race, kind)]

    def extractor(session, year, race):
        return pd.DataFrame([{"year": year, "race": race, "n_laps": len(session.laps)}])

    df = build_race_year_dataset(
        ["Race A"],
        [2024, 2025],
        extractor,
        "rows",
        session_loader=session_loader,
        session_load_kwargs={"weather": True},
    )

    assert df.to_dict("records") == [
        {"year": 2024, "race": "Race A", "n_laps": 1},
        {"year": 2025, "race": "Race A", "n_laps": 2},
    ]
    assert sessions[(2024, "Race A", "R")].loaded_with == {"weather": True}
