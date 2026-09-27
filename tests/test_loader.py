"""Tests für src.loader.DataLoader.

Jeder Test baut sich seinen eigenen Loader – keine geteilte Fixture, damit
die Tests unabhängig voneinander laufen und einzeln aussagekräftig sind.
"""

import pandas as pd

from src.loader import DataLoader


def test_stations_shape():
    loader = DataLoader()
    df = loader.load_stations()
    assert df.shape == (168, 5)
    assert df["station_id"].isna().sum() == 0


def test_flows_timestamp():
    df = DataLoader().load_flows()
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.index.isna().sum() == 0


def test_flows_col_count():
    df = DataLoader().load_flows()
    assert df.shape[1] == 168


def test_flows_stadtmitte_mapping():
    loader = DataLoader()
    loader.load_flows()
    assert len(loader.col_to_station_id) == 168
    # Beide Stadtmitte-Spalten muessen auf verschiedene station_id zeigen.
    assert len(set(loader.col_to_station_id.values())) == 168


def test_closures_duration():
    df = DataLoader().load_closures()
    assert df["duration"].apply(lambda x: isinstance(x, pd.Timedelta)).all()


def test_closures_end_time():
    df = DataLoader().load_closures()
    assert "end_time" in df.columns
    assert ((df["end_time"] - df["when"]) == df["duration"]).all()


def test_weather_join():
    loader = DataLoader()
    flows = loader.load_flows()
    weather = loader.load_weather()
    assert set(flows.index) == set(weather.index)


def test_energy_shape():
    df = DataLoader().load_energy()
    # Mindestens die 104 Trainingstage; mit Testdaten (22.09.–30.09.) mehr.
    assert df.shape[0] >= 104
    assert df.shape[1] == 8  # 8 Linien-Spalten (U1-U9, kein U4)
    for line in ["U1", "U2", "U3", "U5", "U6", "U7", "U8", "U9"]:
        assert line in df.columns
