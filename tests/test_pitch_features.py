"""Tests für Zeitkontext, Stationsnormalisierung, Routing und Gesamtnetz-Fallback."""

import pytest

from src.legacy_agent import TrainAgent
from src.tools._common import normalize_station
from src.tools.network_tool import TRANSIT_DIR, find_alternative_routes, find_transit_route
from src.tools.temporal_tool import get_peak_hours, get_temporal_context, is_service_running


def _calls(question: str) -> list[str]:
    return [c["name"] for c in TrainAgent(llm=lambda p: "ok").route(question)["calls"]]


@pytest.mark.parametrize("ts, slot, rush, weekend", [
    ("2026-09-21 08:15", "morning_peak", True, False),   # Montag
    ("2026-09-21 17:30", "evening_peak", True, False),
    ("2026-09-21 12:00", "off_peak", False, False),
    ("2026-09-26 08:15", "off_peak", False, True),       # Samstag: keine Stoßzeit
    ("2026-09-22 02:30", "night", False, False),
])
def test_temporal_context(ts, slot, rush, weekend):
    ctx = get_temporal_context(ts)
    assert (ctx["time_slot"], ctx["ist_rushhour"], ctx["ist_wochenende"]) == (slot, rush, weekend)


def test_service_pause_boundaries():
    assert is_service_running("2026-09-22 00:59")
    assert not is_service_running("2026-09-22 01:00")
    assert not is_service_running("2026-09-22 04:44")
    assert is_service_running("2026-09-22 04:45")
    assert "error" in get_temporal_context("kein datum")


def test_peak_hours_definition():
    peaks = get_peak_hours()
    assert peaks["morning_peak"] == {"start": "07:00", "end": "09:00"}
    assert peaks["evening_peak"] == {"start": "16:00", "end": "19:00"}


def test_bahnhof_is_stopword():
    assert normalize_station("Bahnhof Zoologischer Garten") == normalize_station(
        "S+U Zoologischer Garten Bhf (Berlin)")


@pytest.mark.parametrize("question, hours", [
    ("between 8 and 10", (8, 10)),
    ("from 8 to 10", (8, 10)),
    ("from 5pm to 7pm", (17, 19)),
])
def test_extract_hours(question, hours):
    assert TrainAgent._extract_hours(question) == hours


def test_count_questions_route_correctly():
    assert _calls("How many passengers at Alexanderplatz?") == ["get_station_flow"]
    assert _calls("What is the total ridership?") == ["get_network_flow_summary"]
    assert "get_stops_for_line" in _calls("How many stations does U7 have?")
    assert "get_peak_hours" in _calls("What are the peak hours for U Stadtmitte?")
    assert "get_weekday_profile" in _calls(
        "How many passengers used U Alexanderplatz on Monday morning?")


needs_graph = pytest.mark.skipif(not (TRANSIT_DIR / "adjacency.json").exists(),
                                 reason="Gesamtnetz-Graph nicht gebaut")


@needs_graph
def test_transit_route_uses_rail():
    result = find_transit_route("Hertzallee", "U Rudow")
    assert "ubahn" in result["modes_used"]
    assert result["total_minutes"] < 60


@needs_graph
def test_suspension_gets_multimodal_alternative():
    result = find_alternative_routes("Pankow", "Alexanderplatz", avoid_direct=True, line="U2")
    assert "error" not in result
    legs = result["transit_alternative"]["routes"][0]["legs"]
    assert all(leg["line"] != "U2" for leg in legs)


@needs_graph
def test_station_outside_ubahn_dataset():
    result = find_alternative_routes("Ostkreuz", "Alexanderplatz")
    assert "fallback_reason" in result and result["routes"]


# --------------------------------------------------------------------- #
# Robustheit: Tools liefern Fehler-dicts statt Exceptions
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("hours", [(10, 8), (-1, 5), (0, 30), (8, 8), (None, None), ("x", 5)])
def test_invalid_hour_window_is_rejected(hours):
    from src.tools.flow_tool import get_network_flow_summary, get_station_flow
    from src.tools.weather_tool import get_weather
    for result in (get_station_flow("Alexanderplatz", "2026-07-21", *hours),
                   get_network_flow_summary("2026-07-21", *hours),
                   get_weather("2026-07-21", *hours)):
        assert result["error"] == "Invalid hour window"


@pytest.mark.parametrize("date", ["", None, "kaputt", "2026-02-31"])
def test_invalid_date_returns_error(date):
    from src.tools.flow_tool import get_network_flow_summary
    assert get_network_flow_summary(date)["error"] == "Invalid date"


def test_fallback_answer_shows_multimodal_alternative():
    def llm_down(prompt):
        raise RuntimeError("endpoint down")
    result = TrainAgent(llm=llm_down).answer(
        "U2 suspended between Pankow and Alexanderplatz – what are the alternatives?")
    if (TRANSIT_DIR / "adjacency.json").exists():
        assert "Multimodal alternative" in result["answer"]
    assert "transit_alternative" not in result["answer"]


# --------------------------------------------------------------------- #
# Antwortqualität (Befunde aus dem LLM-End-to-End-Test)
# --------------------------------------------------------------------- #

def test_weekday_profile_window_total():
    from src.tools.temporal_tool import get_weekday_profile
    result = get_weekday_profile("Alexanderplatz", "Monday", 6, 10)
    window = result["window_total"]
    assert window["hours"] == "06:00–10:00"
    assert 0 < window["mean_total"] < result["mean_daily_total"]


def test_peak_hours_are_backed_by_measurement():
    measured = get_peak_hours()["measured"]
    top_hours = {h["hour"] for h in measured["busiest_hours_mon_fri"][:4]}
    assert top_hours <= {"07:00", "08:00", "16:00", "17:00", "18:00"}


def test_rail_replacement_buses_are_not_mixed_into_bus_lines():
    from src.tools.gtfs_tool import get_lines_for_station
    result = get_lines_for_station("Alexanderplatz")
    all_lines = [line for lines in result["lines_by_type"].values() for line in lines]
    assert len(all_lines) == len(set(all_lines)) == result["total_lines"]


def test_closure_listing_keeps_high_confidence():
    plan = TrainAgent(llm=lambda p: "ok").route("Are there any closures affecting U6?")
    assert plan["assumptions"] == []
