"""Routing-Tests für src.legacy_agent.TrainAgent.

Das LLM wird injiziert – kein echter HTTP-Call, keine Credentials nötig.
Getestet wird ausschliesslich der ROUTE/GATHER-Teil: welche Tools bei welcher
Frageformulierung greifen. Die Tools selbst laufen dabei echt gegen die
Rohdaten, damit die Tests auch Regressionen im Datenpfad fangen.
"""

import pytest

from src.legacy_agent import TrainAgent


class FakeLLM:
    """Ersetzt den Azure-Endpoint. Zählt Aufrufe, damit Tests belegen können,
    dass Out-of-Scope-Fragen gar nicht erst beim LLM landen."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "Test answer based on provided data."


@pytest.fixture
def agent() -> TrainAgent:
    return TrainAgent(llm=FakeLLM())


# --------------------------------------------------------------------- #
# Routing je Themenbereich
# --------------------------------------------------------------------- #

def test_route_closure_keyword(agent):
    result = agent.answer("Was ist mit der U6 Sperrung am 13. Juli?")
    assert "get_closures" in result["tools_used"]


def test_route_event_keyword(agent):
    result = agent.answer("Guns N Roses Konzert am 23. Juni")
    assert "get_events" in result["tools_used"]


def test_route_weather_keyword(agent):
    result = agent.answer("Wie war das Wetter in der Woche 20.-26. Juli?")
    assert ("get_weather" in result["tools_used"]
            or "find_weather_anomalies" in result["tools_used"])


def test_route_energy_keyword(agent):
    result = agent.answer("Welche Linie hat den schlechtesten Energieverbrauch?")
    assert "get_energy" in result["tools_used"]


def test_route_critical_stations(agent):
    result = agent.answer("Welche Stationen sind am kritischsten für das Netz?")
    assert "get_critical_stations" in result["tools_used"]


def test_route_peak_profile(agent):
    result = agent.answer("Wann ist der Pendler-Peak an Rudow?")
    assert "get_peak_profile" in result["tools_used"]
    # Ohne Datum darf KEIN Einzeltag unterstellt werden (Fix C).
    assert "get_station_flow" not in result["tools_used"]


def test_route_compare_to_network(agent):
    result = agent.answer("Liegt der Peak an Rudow über dem Netzwerkdurchschnitt?")
    assert "compare_station_to_network" in result["tools_used"]


def test_route_alternative_route(agent):
    result = agent.answer(
        "Gibt es eine Alternativroute von Hermannplatz nach Alexanderplatz?"
    )
    assert "find_alternative_routes" in result["tools_used"]


# --------------------------------------------------------------------- #
# Ablehnung und Unsicherheit
# --------------------------------------------------------------------- #

def test_out_of_scope_poem():
    llm = FakeLLM()
    result = TrainAgent(llm=llm).answer("Write me a poem about trains.")
    assert result["confidence"] == "none"
    assert result["tools_used"] == []
    # Out-of-Scope darf den Endpoint gar nicht erst belasten.
    assert llm.calls == []


def test_out_of_scope_weather_general(agent):
    # "today" liegt ausserhalb des Datensatzes und ist kein parsebares Datum.
    result = agent.answer("What is the weather like in Berlin today?")
    assert result["confidence"] in ["none", "low"]


# --------------------------------------------------------------------- #
# Datumsextraktion
# --------------------------------------------------------------------- #

def test_date_extraction_german(agent):
    result = agent.answer("Was passierte am 23. Juni 2026 an der Uber Arena?")
    assert "get_events" in result["tools_used"]


def test_date_extraction_iso(agent):
    result = agent.answer("Flow at Alexanderplatz on 2026-07-21")
    assert "get_station_flow" in result["tools_used"]


# --------------------------------------------------------------------- #
# LLM-Zeitlimit und Gesamtnetz (GTFS)
# --------------------------------------------------------------------- #

def test_llm_timeout_returns_data_answer(monkeypatch):
    import time

    import src.legacy_agent as agent_module

    monkeypatch.setattr(agent_module, "LLM_TIMEOUT_SECONDS", 0.1)
    slow = lambda prompt: (time.sleep(1), "too late")[1]  # noqa: E731
    result = TrainAgent(llm=slow).answer(
        "Which stations are critical articulation points?"
    )
    assert result["llm_timeout"] is True
    assert "timed out" in result["note"]
    assert "Alexanderplatz" in result["answer"]
    assert result["confidence"] in ("medium", "low")


def test_route_gtfs_lines_for_station(agent):
    result = agent.answer("Which lines serve Alexanderplatz?")
    assert "get_lines_for_station" in result["tools_used"]


def test_last_as_verb_keeps_closure_history():
    # "how long will it last" darf das Datum nicht auf den letzten Datentag
    # setzen; mehrere passende Sperrungen duerfen den Plan nicht zerlegen.
    agent = TrainAgent(llm=lambda prompt: "ok")
    plan = agent.route("There is a closure on U3. How long will it last?")
    assert "recent" not in plan["matched"]
    assert isinstance(plan["dates"], dict)
    result = agent.answer("There is a closure on U3. How long will it last?")
    assert result["error"] is None
