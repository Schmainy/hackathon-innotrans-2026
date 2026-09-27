"""Tests für Wochentags-/Wochenauswertungen und die Energie-Zuordnung.

Laufen echt gegen die Rohdaten – Regressionen im Datenpfad fallen so auf.
"""

from src.legacy_agent import TrainAgent
from src.tools.energy_tool import get_energy
from src.tools.temporal_tool import (
    get_busiest_station_by_weekday,
    get_station_weekday_pattern,
    get_weekly_summary,
)


def test_monday_morning_peak():
    result = get_busiest_station_by_weekday(weekday=0, hour_start=7, hour_end=9)
    top = result["top_5_stations"]
    assert result["time_window"] == "07:00–09:00"
    assert len(top) == 5
    assert top[0]["station_name"]
    assert top[0]["mean_window_total"] >= top[-1]["mean_window_total"]


def test_weekly_summary():
    result = get_weekly_summary()
    assert 5 <= result["days_with_data"] <= 7
    assert all(day["total_pax"] > 0 for day in result["daily_totals"].values())
    assert result["total_weekly_pax"] == sum(
        day["total_pax"] for day in result["daily_totals"].values()
    )


def test_station_weekday_pattern():
    result = get_station_weekday_pattern("Alexanderplatz")
    assert result["weekday_mean"] > result["weekend_mean"]


def test_last_week_routes_to_weekly_summary():
    plan = TrainAgent(llm=lambda prompt: "ok").route(
        "How many passengers passed through the network last week?"
    )
    names = [call["name"] for call in plan["calls"]]
    assert names == ["get_weekly_summary"]


def test_energy_reports_allocation_sensitivity():
    result = get_energy()
    assert set(result["allocation_sensitivity"]) == {
        "equal_split", "first_line", "full_count",
    }
    assert result["worst_efficiency_line"] == result["allocation_sensitivity"][
        "equal_split"]["worst"]
