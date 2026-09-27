"""Tests für die Analysen hinter den Trainingsfragen 1, 2, 7, 8, 9 und Bonus 2."""

import pytest

from src.legacy_agent import TrainAgent
from src.tools.dependency_tool import analyze_disruption_routing, find_station_dependencies
from src.tools.flow_tool import get_station_flow
from src.tools.network_tool import TRANSIT_DIR, find_diverse_transit_routes


@pytest.fixture(scope="module")
def agent():
    return TrainAgent(llm=lambda prompt: "ok")


def test_station_flow_baseline_shows_concert_peak():
    result = get_station_flow("Warschauer Str.", "2026-06-23", 16, 24, compare_baseline=True)
    assert result["baseline"]["days"] > 5
    assert result["baseline"]["peak_ratio"] > 1.5  # Konzertabend deutlich über Normal


def test_concert_question_covers_neighbouring_stations(agent):
    q = ("There's a Guns N' Roses concert on June 23rd at the Uber Arena. What will the "
         "passenger flow look like at the neighboring stations?")
    flows = [e for e in agent.gather(agent.route(q))["results"] if e["name"] == "get_station_flow"]
    assert len(flows) >= 2
    assert all("baseline" in e["result"] for e in flows)


def test_weather_week_question_uses_flows_at_weather_peak(agent):
    q = ("Give me an example of a passenger flow peak caused by bad weather in the week of "
         "July 20-26 and provide the time and station where the peak took place.")
    plan = agent.route(q)
    assert plan["weather_followup"]
    results = agent.gather(plan)["results"]
    peak = next(e["result"]["peak"]["timestamp"] for e in results
                if e["name"] == "find_weather_anomalies")
    flow_dates = {e["result"]["date"] for e in results if e["name"] == "get_network_flow_summary"}
    assert flow_dates == {peak[:10]}  # nicht der Wochenbeginn


def test_dependencies_remove_daily_rhythm():
    result = find_station_dependencies()
    assert result["median_correlation_raw_flows"] > 0.2  # gemeinsamer Tagesgang
    assert abs(result["median_correlation_after_daily_pattern"]) < 0.1
    assert all(p["graph_hops"] >= 3 for p in result["pairs"])


def test_disruption_routing_summary():
    summary = analyze_disruption_routing()["summary"]
    assert summary["segment_closures_analysed"] > 0
    assert "of" in summary["closures_where_nearby_stations_rose_more_than_distant"]


@pytest.mark.skipif(not (TRANSIT_DIR / "adjacency.json").exists(), reason="Gesamtnetz fehlt")
def test_diverse_routes_use_different_corridors():
    routes = find_diverse_transit_routes("S Messe Nord/ICC", "S+U Alexanderplatz")["routes"]
    assert len(routes) >= 2
    vias = [set(v for leg in r["legs"] for v in leg["via"]) for r in routes]
    main_first = max(routes[0]["legs"], key=lambda leg: leg["minutes"])
    assert not set(main_first["via"]) & vias[1]  # Korridor der schnellsten Route gemieden


@pytest.mark.parametrize("question, tool", [
    ("Are there stations whose passenger demand appears strongly dependent on another "
     "station despite no direct connection between them?", "find_station_dependencies"),
    ("During disruptions, which alternative routes do passengers actually prefer compared "
     "to the theoretically shortest routes?", "analyze_disruption_routing"),
    ("During InnoTrans we expect a major surge in passenger flow around Messe Berlin towards "
     "the city center. Can you suggest an unconventional alternative route not based on the "
     "shortest path?", "find_diverse_transit_routes"),
])
def test_analysis_questions_route(agent, question, tool):
    assert tool in {c["name"] for c in agent.route(question)["calls"]}
