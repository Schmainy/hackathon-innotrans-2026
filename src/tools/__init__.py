"""Agent-Tools – ein Modul je Datensatz bzw. Fähigkeit.

Jede Tool-Funktion ist eine pure Funktion ohne globalen State, lädt ihre Daten
über src.loader.DataLoader und gibt ein strukturiertes dict zurück. Fehler
kommen als {"error": ...} zurück statt als Exception – einzige Ausnahme ist
call_llm(), das einen LLMError wirft, damit der Agent einen Endpoint-Ausfall
explizit behandeln muss.

Die Dicts sind so geschnitten, dass sie direkt als MCP-Tool-Responses
registriert werden können (Phase 3).
"""

from src.tools.closure_tool import get_closure_impact, get_closures
from src.tools.dependency_tool import analyze_disruption_routing, find_station_dependencies
from src.tools.energy_tool import get_energy
from src.tools.event_tool import get_events
from src.tools.gtfs_tool import (
    find_route_between_stations,
    find_stations_in_text,
    get_all_ubahn_lines,
    get_lines_for_station,
    get_stops_for_line,
)
from src.tools.flow_tool import (
    detect_anomalies,
    get_network_flow_summary,
    get_station_flow,
    get_station_peak_15min,
    get_station_peak_vs_baseline,
)
from src.tools.llm_client import LLMConfigError, LLMError, call_llm
from src.tools.network_tool import (
    find_alternative_routes,
    find_diverse_transit_routes,
    find_transit_route,
    get_critical_stations,
)
from src.tools.peak_tool import compare_station_to_network, get_peak_profile
from src.tools.temporal_tool import (
    get_busiest_station_by_weekday,
    get_peak_hours,
    get_station_weekday_pattern,
    find_stations_above_threshold,
    get_temporal_context,
    get_weekday_profile,
    get_weekly_summary,
    is_service_running,
)
from src.tools.weather_tool import find_weather_anomalies, get_rain_impact, get_weather

__all__ = [
    # flow_tool
    "get_station_flow",
    "get_station_peak_15min",
    "get_station_peak_vs_baseline",
    "get_network_flow_summary",
    "detect_anomalies",
    # closure_tool
    "get_closures",
    "get_closure_impact",
    # event_tool
    "get_events",
    # weather_tool
    "get_weather",
    "find_weather_anomalies",
    "get_rain_impact",
    # energy_tool
    "get_energy",
    # network_tool
    "get_critical_stations",
    "find_alternative_routes",
    "find_transit_route",
    "find_diverse_transit_routes",
    # dependency_tool
    "find_station_dependencies",
    "analyze_disruption_routing",
    # peak_tool
    "get_peak_profile",
    "compare_station_to_network",
    # temporal_tool
    "get_weekday_profile",
    "get_busiest_station_by_weekday",
    "get_station_weekday_pattern",
    "get_weekly_summary",
    "get_temporal_context",
    "find_stations_above_threshold",
    "get_peak_hours",
    "is_service_running",
    # gtfs_tool
    "get_lines_for_station",
    "get_stops_for_line",
    "get_all_ubahn_lines",
    "find_route_between_stations",
    "find_stations_in_text",
    # llm_client
    "call_llm",
    "LLMError",
    "LLMConfigError",
]
