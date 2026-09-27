"""Routing-Matrix: 30 typische Operator-Fragen -> erwartete Tools (ohne LLM).

Pro Frage: Tools, die aufgerufen werden MÜSSEN, und Tools, die NICHT
aufgerufen werden dürfen (typische Fehlrouten wie der Tagesreport eines
unterstellten Einzeltags).
"""

import pytest

from src.legacy_agent import TrainAgent

CASES = [
    ("How many passengers used U Alexanderplatz on Monday morning?", {"get_weekday_profile"}, set()),
    ("What is the busiest station on Friday evening?", {"get_busiest_station_by_weekday"}, set()),
    ("How many total passengers were recorded last week?", {"get_weekly_summary"}, set()),
    ("Show me the flow at U Stadtmitte between 8 and 10am", {"get_station_flow"}, set()),
    ("Which stations have more than 2000 passengers per hour?",
     {"find_stations_above_threshold"}, {"get_network_flow_summary", "network_info"}),
    ("How many people use the U-Bahn on weekends?", {"get_weekday_profile"}, {"get_network_flow_summary"}),
    ("How many stations does U7 have?", {"get_stops_for_line", "network_info"}, set()),
    ("How many stations does U6 have?", {"get_stops_for_line"}, set()),
    ("What lines serve Alexanderplatz?", {"get_lines_for_station"}, set()),
    ("Find an alternative route from U Pankow to U Tempelhof if U2 is closed",
     {"find_alternative_routes"}, set()),
    ("What is the fastest route from Ostkreuz to Hertzallee?", {"find_transit_route"}, set()),
    ("Which are the most critical stations in the network?", {"get_critical_stations"}, set()),
    ("What stations are on U1?", {"get_stops_for_line"}, set()),
    ("How many lines does Berlin have?", {"network_info", "get_all_ubahn_lines"}, set()),
    ("Is the U-Bahn running at 3am?", {"get_temporal_context"}, set()),
    ("Is the U-Bahn running at 2:30am?", {"get_temporal_context"}, set()),
    ("What are the peak hours?", {"get_peak_hours"}, set()),
    ("Is Monday morning busier than Friday evening at U Zoologischer Garten?",
     {"get_weekday_profile"}, set()),
    ("What time slot is 17:45?", {"get_temporal_context"}, set()),
    ("When does service stop at night?", {"get_peak_hours"}, set()),
    ("Are there any current closures?", {"get_closures"}, set()),
    ("What is the impact of the U2 closure?", {"get_closures"}, set()),
    ("Which closures affect Alexanderplatz?", {"get_closures"}, set()),
    ("What is the energy consumption of U6?", {"get_energy"}, set()),
    ("Which line uses the most energy?", {"get_energy"}, set()),
    ("How does rain affect passenger numbers?", {"get_rain_impact"},
     {"get_weather", "get_network_flow_summary"}),
    ("Are there any events this week?", {"get_events"}, {"get_weekly_summary"}),
    ("What happened during the InnoTrans event?", {"get_events"}, set()),
    ("What is the passenger flow at U Zoologischer Garten on Monday during peak hours?",
     {"get_weekday_profile", "get_peak_profile"}, set()),
    ("What are the most crowded stations and which lines serve them?",
     {"get_busiest_station_by_weekday"}, {"get_events"}),
]


@pytest.fixture(scope="module")
def agent():
    return TrainAgent(llm=lambda prompt: "ok")


@pytest.mark.parametrize("question, required, forbidden", CASES, ids=[c[0][:45] for c in CASES])
def test_routing(agent, question, required, forbidden):
    plan = agent.route(question)
    names = {c["name"] for c in plan["calls"]}
    assert not plan.get("out_of_scope"), "Frage landet ohne Tools beim LLM"
    assert required <= names, f"fehlend: {required - names}"
    assert not (forbidden & names), f"unerwünscht: {forbidden & names}"


def test_whole_line_closure_is_passed_to_routing(agent):
    plan = agent.route("Find an alternative route from U Pankow to U Tempelhof if U2 is closed")
    call = next(c for c in plan["calls"] if c["name"] == "find_alternative_routes")
    assert call["kwargs"]["closed_line"] == "U2"


def test_current_closures_use_last_data_day(agent):
    plan = agent.route("Are there any current closures?")
    call = next(c for c in plan["calls"] if c["name"] == "get_closures")
    assert call["kwargs"]["date_str"] == "today"


def test_crowded_is_not_an_event(agent):
    assert "event" not in agent.route("What are the most crowded stations?")["matched"]
    assert "event" in agent.route("How big was the crowd at the concert?")["matched"]


def test_line_list_without_other_keywords(agent):
    names = {c["name"] for c in agent.route("U7 stations please")["calls"]}
    assert "get_stops_for_line" in names
