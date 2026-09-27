"""Tool 9 – Gesamtnetz aus dem VBB-Fahrplan (GTFS): U-Bahn, S-Bahn, Tram, Bus.

get_lines_for_station       welche Linien halten an einer Station
get_stops_for_line          Haltefolge einer Linie (U2, S41, M10, 100 ...)
get_all_ubahn_lines         alle U-Bahn-Linien laut Fahrplan (inkl. U4)
find_route_between_stations Linien, die beide Stationen direkt verbinden
find_stations_in_text       GTFS-Stationsnamen in einer Frage (auch S-Bahn-only)

Liest nur die vorverarbeiteten Tabellen aus data/derived/ (wenige MB, siehe
scripts/build_gtfs_layer.py). Die rohen 639 MB GTFS werden zur Laufzeit nie
angefasst. Statischer Fahrplan, keine Echtzeitdaten. Das ist Fahrplan-WISSEN
aus einer Datei, kein Teil des Hackathon-Datensatzes – deshalb trägt jedes
Ergebnis eine source-Angabe.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from src.loader import REPO_ROOT
from src.tools._common import normalize_station

DERIVED_DIR = REPO_ROOT / "data" / "derived"

NETWORK_MODES = {
    "ubahn": ["U-Bahn"],
    "sbahn": ["S-Bahn"],
    "tram": ["Tram"],
    "bus": ["Bus"],
    "regional": ["Regional"],
    "all": ["U-Bahn", "S-Bahn", "Tram", "Bus", "Regional", "Ferry"],
}

# Bei gleichnamigen Linien (RE2 als Zug und als Ersatzbus) gewinnt die Schiene.
MODE_PRIORITY = {"U-Bahn": 0, "S-Bahn": 1, "Regional": 2, "Tram": 3, "Bus": 4, "Ferry": 5}

BERLIN_PREFIX = "de:11000:"
MAX_MATCHED_NAMES = 8


class GTFSNotBuiltError(FileNotFoundError):
    """Die abgeleiteten GTFS-Tabellen fehlen – Build-Skript ausführen."""


@lru_cache(maxsize=1)
def _tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lädt die abgeleiteten Tabellen einmal pro Prozess (zusammen ~3.5 MB)."""
    route_stops_path = DERIVED_DIR / "gtfs_route_stops.csv"
    stop_routes_path = DERIVED_DIR / "gtfs_stop_routes.csv"
    if not route_stops_path.is_file() or not stop_routes_path.is_file():
        raise GTFSNotBuiltError(
            f"GTFS-Tabellen fehlen unter {DERIVED_DIR} – "
            "python scripts/build_gtfs_layer.py ausführen."
        )
    route_stops = pd.read_csv(route_stops_path, dtype=str, encoding="utf-8")
    route_stops["stop_sequence"] = route_stops["stop_sequence"].astype(int)
    stop_routes = pd.read_csv(stop_routes_path, dtype=str, encoding="utf-8")
    stop_routes["norm"] = stop_routes["stop_name"].map(_station_norm)
    meta_path = DERIVED_DIR / "gtfs_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return route_stops, stop_routes, meta


def _station_norm(name: str) -> str:
    """Vergleichsschlüssel wie im U-Bahn-Datensatz, ohne "[Tram]"-Zusätze.

    Ein führendes "Berlin" fällt weg: "S+U Berlin Hauptbahnhof" soll unter
    "Hauptbahnhof" gefunden werden.
    """
    key = normalize_station(re.sub(r"\[.*?\]", "", str(name)))
    if key.startswith("berlin") and len(key) > len("berlin") + 3:
        key = key[len("berlin"):]
    return key


def _source(meta: dict) -> str:
    return (
        f"{meta.get('source', 'VBB GTFS static timetable')}, valid "
        f"{meta.get('valid_from', '?')}–{meta.get('valid_to', '?')}. Static "
        "timetable, not part of the hackathon dataset, no real-time data."
    )


def _line_sort_key(line: str) -> tuple:
    match = re.match(r"([A-Za-z]*)(\d*)(.*)", str(line))
    prefix, number, rest = match.groups() if match else (str(line), "", "")
    return (prefix, int(number) if number else -1, rest)


def _match_station(stop_routes: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """Alle Halte-Zeilen zu einem Stationsnamen.

    Bevorzugt Namen, die mit dem Suchbegriff BEGINNEN ("alexanderplatz" trifft
    "S+U Alexanderplatz Bhf/Memhardstr.", die Bushaltestelle am selben Platz),
    sonst Teilstring-Treffer. Berliner Halte haben Vorrang vor gleichnamigen
    in Brandenburg.
    """
    key = _station_norm(station_name)  # "Bhf"/"Bahnhof" entfernt normalize_station
    if not key:
        return stop_routes.iloc[0:0]
    hits = stop_routes[stop_routes["norm"].str.startswith(key)]
    if hits.empty:
        hits = stop_routes[stop_routes["norm"].str.contains(key, regex=False)]
    berlin = hits[hits["station_key"].str.startswith(BERLIN_PREFIX)]
    return berlin if not berlin.empty else hits


def _modes_for(network: str) -> list[str]:
    return NETWORK_MODES.get(str(network).lower(), NETWORK_MODES["all"])


def get_lines_for_station(station_name: str, network: str = "all") -> dict:
    """Alle Linien (U-Bahn, S-Bahn, Tram, Bus, Regional), die an einer Station halten.

    network: "ubahn" | "sbahn" | "tram" | "bus" | "regional" | "all"
    """
    try:
        _, stop_routes, meta = _tables()
    except GTFSNotBuiltError as exc:
        return {"error": "GTFS not built", "detail": str(exc)}

    hits = _match_station(stop_routes, station_name)
    if hits.empty:
        return {"error": "Station not found", "input": station_name,
                "detail": "Station not found in VBB GTFS."}

    hits = hits[hits["mode"].isin(_modes_for(network))]
    # Busse unter Bahn-Liniennummern ("S7", "M1", "RE1" als Bus) sind
    # Schienenersatzverkehr. Unter "Bus" gemischt erschienen S7/M1 doppelt.
    rail_lines = _rail_line_names()
    is_sev = (hits["mode"] == "Bus") & hits["line"].isin(rail_lines)
    lines_by_type = {
        mode: sorted(group["line"].dropna().unique().tolist(), key=_line_sort_key)
        for mode, group in sorted(hits[~is_sev].groupby("mode"),
                                  key=lambda kv: MODE_PRIORITY.get(kv[0], 9))
    }
    replacement = sorted(hits.loc[is_sev, "line"].dropna().unique().tolist(), key=_line_sort_key)
    names = hits["stop_name"].dropna().unique().tolist()
    result = {
        "station": station_name,
        "network_filter": network,
        "matched_stop_names": names[:MAX_MATCHED_NAMES],
        "matched_stop_count": len(names),
        "lines_by_type": lines_by_type,
        "total_lines": sum(len(v) for v in lines_by_type.values()),
        "source": _source(meta),
        "note": (
            "All lines are VBB timetable data [VBB timetable], not general "
            "knowledge. Lines stopping at any stop whose name matches (incl. "
            "bus/tram stops at the same square). Includes variants and night lines."
        ),
    }
    if replacement:
        result["rail_replacement_buses"] = replacement
        result["rail_replacement_note"] = (
            "Buses running under a rail/tram line number (Schienenersatzverkehr, "
            "SEV) in the timetable period – not counted in total_lines."
        )
    return result


@lru_cache(maxsize=1)
def _rail_line_names() -> frozenset[str]:
    """Liniennamen, die im Fahrplan als Schiene/Tram/Fähre verkehren."""
    route_stops, _, _ = _tables()
    return frozenset(route_stops.loc[route_stops["mode"] != "Bus", "line"].dropna().unique())


def get_stops_for_line(line_name: str) -> dict:
    """Haltefolge einer Linie in Fahrtrichtung, z. B. "U2", "S41", "M10", "100".

    Referenz ist die längste Fahrt der Linie (Richtung 0 bevorzugt) – Zweige
    und Kurzläufer-Varianten sind darin nicht vollständig abgebildet.
    """
    try:
        route_stops, _, meta = _tables()
    except GTFSNotBuiltError as exc:
        return {"error": "GTFS not built", "detail": str(exc)}

    want = str(line_name).strip().upper()
    rows = route_stops[route_stops["line"].str.upper() == want]
    if rows.empty:
        return {"error": "Line not found", "input": line_name,
                "detail": f"Line '{line_name}' not found in VBB GTFS (Berlin routes)."}

    mode = min(rows["mode"].unique(), key=lambda m: MODE_PRIORITY.get(m, 9))
    rows = rows[rows["mode"] == mode]
    sizes = rows.groupby(["trip_id", "direction_id"]).size().reset_index(name="n")
    sizes = sizes.sort_values(["n", "direction_id"], ascending=[False, True])
    trip_id = sizes.iloc[0]["trip_id"]
    trip = rows[rows["trip_id"] == trip_id].sort_values("stop_sequence")

    names = [re.sub(r"\s*\(Berlin\)", "", str(n)) for n in trip["stop_name"]]
    times = pd.to_timedelta(trip["departure_time"], errors="coerce").dropna()
    minutes = int((times.iloc[-1] - times.iloc[0]).total_seconds() // 60) if len(times) > 1 else None

    return {
        "line": want,
        "type": mode,
        "stop_count": len(names),
        "stops": names,
        "end_to_end_minutes": minutes,
        "route_variants": int(rows["route_id"].nunique()),
        "source": _source(meta),
        "note": (
            f"Representative trip ({len(names)} stops, longest trip of the "
            "line). Branches and short-turn variants may differ."
        ),
    }


def get_all_ubahn_lines() -> dict:
    """Alle U-Bahn-Linien laut Fahrplan mit Haltezahl der Referenzfahrt."""
    try:
        route_stops, _, meta = _tables()
    except GTFSNotBuiltError as exc:
        return {"error": "GTFS not built", "detail": str(exc)}

    ubahn = route_stops[route_stops["mode"] == "U-Bahn"]
    counts = ubahn.groupby(["line", "trip_id"]).size().groupby("line").max()
    lines = sorted(counts.index.tolist(), key=_line_sort_key)
    return {
        "ubahn_lines": lines,
        "count": len(lines),
        "stops_per_line": {line: int(counts[line]) for line in lines},
        "source": _source(meta),
        "note": (
            "All U-Bahn lines in the VBB timetable. U4 runs in reality but is "
            "NOT part of the hackathon passenger-flow dataset."
        ),
    }


def find_route_between_stations(
    from_station: str, to_station: str, prefer_network: str = "all"
) -> dict:
    """Linien, die beide Stationen ohne Umstieg verbinden.

    Nur Direktverbindungen – Wege mit Umstieg plant find_alternative_routes
    auf dem U-Bahn-Graphen.
    """
    from_data = get_lines_for_station(from_station, network=prefer_network)
    if "error" in from_data:
        return from_data
    to_data = get_lines_for_station(to_station, network=prefer_network)
    if "error" in to_data:
        return to_data

    def flatten(data: dict) -> set[str]:
        return {line for lines in data["lines_by_type"].values() for line in lines}

    from_lines, to_lines = flatten(from_data), flatten(to_data)
    common = sorted(from_lines & to_lines, key=_line_sort_key)
    return {
        "from": from_station,
        "to": to_station,
        "network_filter": prefer_network,
        "direct_lines": common,
        "has_direct_connection": bool(common),
        "from_lines": sorted(from_lines, key=_line_sort_key),
        "to_lines": sorted(to_lines, key=_line_sort_key),
        "source": from_data["source"],
        "note": (
            "Direct lines only (no transfers). Transfer routes on the U-Bahn "
            "graph: find_alternative_routes."
        ),
    }


@lru_cache(maxsize=1)
def _station_aliases() -> list[tuple[str, str]]:
    """(normalisierter Alias, Anzeigename) für Berliner Halte, längster zuerst.

    Alias ist der ganze Name und der Teil vor "/" ("S Messe Nord/ICC" ->
    "messenord"), damit Operatoren-Kurzformen treffen.
    """
    _, stop_routes, _ = _tables()
    berlin = stop_routes[stop_routes["station_key"].str.startswith(BERLIN_PREFIX)]
    aliases: dict[str, str] = {}
    for name in berlin["stop_name"].dropna().unique():
        # Anzeigename = der jeweils gematchte Teil: "S Südkreuz Bhf/Ostseite"
        # soll über "Südkreuz" als "S Südkreuz Bhf" zurückkommen, nicht als
        # die Bushaltestelle an der Ostseite.
        for part in {name, name.split("/")[0]}:
            display = re.sub(r"\s*\(Berlin\)|\s*\[.*?\]", "", part).strip()
            key = _station_norm(part)
            if len(key) >= 6 and (key not in aliases or len(display) < len(aliases[key])):
                aliases[key] = display
    return sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True)


def find_stations_in_text(text: str, limit: int = 2) -> list[str]:
    """GTFS-Stationsnamen in einer Frage, in Nennungsreihenfolge.

    Rückfallebene, wenn der U-Bahn-Datensatz die Station nicht kennt
    (Ostkreuz, Messe Nord/ICC, Südkreuz – S-Bahn-only).
    """
    try:
        aliases = _station_aliases()
    except GTFSNotBuiltError:
        return []
    norm_text = _station_norm(text)
    found: list[tuple[int, str]] = []
    consumed = ""
    for key, display in aliases:
        pos = norm_text.find(key)
        if pos < 0 or key in consumed:
            continue
        found.append((pos, display))
        consumed += key + "|"
        if len(found) >= limit * 3:
            break
    found.sort()
    names: list[str] = []
    for _, display in found:
        if display not in names:
            names.append(display)
    return names[:limit]
