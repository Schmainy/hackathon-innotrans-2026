"""Tool 2 – Störungen und deren Flow-Impact.

get_closures        gefilterte Sperrungsliste mit geparster Struktur
get_closure_impact  Vergleich Flow während Sperrung vs. Wochentags-Baseline
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from src.loader import DataLoader
from src.tools._common import (
    TS_OUT,
    as_float,
    as_int,
    extract_cause,
    extract_lines,
    extract_stations,
    flow_column_for,
    full_day_index,
    normalize_station,
    parse_date,
    resolve_station,
)


def _enrich(row, reference_day: pd.Timestamp | None) -> dict:
    """Wandelt eine closures-Zeile in das strukturierte Ausgabeformat.

    is_active: die Sperrung überlappt den Referenztag (angefragter Tag bzw.
    letzter Datentag). Eine Sperrung bis exakt 00:00 zählt nicht mehr für
    den Folgetag.
    """
    active = None
    if reference_day is not None:
        day_start = reference_day.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        active = bool(row.when < day_end and row.end_time > day_start)
    return {
        "when": row.when.strftime(TS_OUT),
        "end_time": row.end_time.strftime(TS_OUT),
        "duration_minutes": as_int(row.duration.total_seconds() / 60),
        "description": row.description,
        "affected_stations": extract_stations(row.description),
        "affected_lines": extract_lines(row.description),
        "cause": extract_cause(row.description),
        "closure_type": (
            "line_segment" if "suspended" in row.description.lower() else "station"
        ),
        "is_active": active,
    }


# Stichwoerter fuer "jetzt". Echtzeitdaten gibt es nicht – sie werden auf den
# letzten vollstaendigen Datentag abgebildet und so ausgewiesen.
TODAY_WORDS = {"today", "now", "current", "currently", "heute", "aktuell", "jetzt"}


def _last_data_day(loader: DataLoader) -> pd.Timestamp:
    """Letzter vollständiger Betriebstag im Flow-Datensatz."""
    return full_day_index(loader.load_flows()).max().normalize()


def _summary(rows: list[dict]) -> dict:
    """Zählungen je Linie, Ursache und Typ – kompakt genug für den LLM-Kontext."""
    by_line: dict[str, int] = {}
    by_cause: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for r in rows:
        for ln in r["affected_lines"] or ["(station only)"]:
            by_line[ln] = by_line.get(ln, 0) + 1
        by_cause[r["cause"]] = by_cause.get(r["cause"], 0) + 1
        by_type[r["closure_type"]] = by_type.get(r["closure_type"], 0) + 1
    return {
        "by_line": dict(sorted(by_line.items(), key=lambda kv: -kv[1])),
        "by_cause": dict(sorted(by_cause.items(), key=lambda kv: -kv[1])),
        "by_type": by_type,
        "total_hours": as_float(sum(r["duration_minutes"] for r in rows) / 60, 1),
    }


def get_closures(
    date_str: str | None = None,
    station_name: str | None = None,
    line: str | None = None,
) -> dict:
    """Sperrungen, gefiltert nach Datum, Station und/oder Linie (AND-verknüpft).

    date_str:
        None            gesamte Historie (für "Wie lange dauerte ...?")
        "YYYY-MM-DD"    nur am Tag aktive Sperrungen
        "today"/"now"   letzter vollständiger Datentag, als solcher ausgewiesen

    Die Stationsfilterung ist schreibweisen-tolerant: "Kaiserin-Augusta-Strasse"
    findet den Datensatz-Eintrag "Kaiserin-Augusta-Str.".
    """
    loader = DataLoader()
    try:
        closures = loader.load_closures()
        last_day = _last_data_day(loader)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    filters = {"date": date_str, "station": station_name, "line": line}
    note = None
    day = None
    as_of_last_day = False

    if date_str is not None:
        if str(date_str).strip().lower() in TODAY_WORDS:
            day = last_day
            as_of_last_day = True
            note = (
                f"Showing closures as of last data day ({day:%Y-%m-%d}). "
                "Real-time data not available."
            )
        else:
            day = parse_date(date_str)
            if day is None:
                return {"error": "Invalid date", "input": date_str}

    # Ohne Datum bezieht sich is_active auf den letzten Datentag.
    reference = day if day is not None else last_day
    rows = [_enrich(r, reference) for r in closures.itertuples()]
    if day is not None:
        rows = [r for r in rows if r["is_active"]]

    if line is not None:
        want = str(line).strip().upper()
        rows = [r for r in rows if want in r["affected_lines"]]

    if station_name is not None:
        key = normalize_station(station_name)
        rows = [
            r for r in rows
            if any(normalize_station(s) == key for s in r["affected_stations"])
        ]

    result = {
        "filters": filters,
        "count": len(rows),
        "reference_day": reference.strftime("%Y-%m-%d"),
        "as_of_last_data_day": as_of_last_day,
        # Zusammenfassung vor der Liste: bei knappem Kontextbudget ueberlebt
        # sie die Kuerzung, die lange Liste nicht.
        "summary": _summary(rows),
        "closures": rows,
        "data_limitation": (
            "Ursachen stammen aus der englischen Freitextbeschreibung "
            "(\"due to ...\") und werden auf deutsche Kategorien gemappt. "
            "affected_stations sind die in der Beschreibung genannten "
            "Endpunkte, nicht der gesamte Streckenabschnitt. Keine "
            "Echtzeitdaten – der Datensatz endet am "
            f"{last_day:%Y-%m-%d}."
        ),
    }
    if note:
        result["note"] = note
    if day is not None and not rows:
        result["active_closures"] = []
        result["message"] = f"No active closures on {day:%Y-%m-%d}."
    return result


def get_closure_impact(closure_description: str) -> dict:
    """Flow-Impact einer konkreten Sperrung gegen die Wochentags-Baseline.

    Baseline je Station: Mittelwert derselben Tageszeit-Slots an allen anderen
    Tagen mit demselben Wochentag (unvollständige Randtage ausgeschlossen).
    """
    loader = DataLoader()
    try:
        closures = loader.load_closures()
        stations = loader.load_stations()
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    needle = str(closure_description).strip().lower()
    hits = [r for r in closures.itertuples() if needle in r.description.lower()]
    if not hits:
        hits = [r for r in closures.itertuples() if r.description.lower() in needle]
    if not hits:
        return {"error": "Closure not found", "input": closure_description}

    row = hits[0]
    named = extract_stations(row.description)
    is_segment = "suspended" in row.description.lower()

    graph = loader.load_graph()

    # Direkt betroffene station_ids bestimmen.
    affected_ids: list[str] = []
    for name in named:
        matches = resolve_station(stations, name)
        affected_ids.extend(m["station_id"] for m in matches)

    # Bei Streckensperrungen die Zwischenstationen entlang des Abschnitts ergaenzen.
    if is_segment and len(named) == 2:
        ends = [resolve_station(stations, n) for n in named]
        if ends[0] and ends[1]:
            try:
                path = nx.shortest_path(
                    graph, ends[0][0]["station_id"], ends[1][0]["station_id"]
                )
                affected_ids = list(dict.fromkeys(affected_ids + path))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    affected_ids = list(dict.fromkeys(affected_ids))
    if not affected_ids:
        return {"error": "No stations resolved", "closure": row.description}

    neighbor_ids = sorted(
        {n for sid in affected_ids for n in graph.neighbors(sid)} - set(affected_ids)
    )

    full_idx = set(full_day_index(flows))
    mask_window = (flows.index >= row.when) & (flows.index < row.end_time)
    during = flows[mask_window]
    if during.empty:
        return {"error": "No flow data in closure window", "closure": row.description}

    times = during.index.time
    baseline_mask = (
        (flows.index.dayofweek == row.when.dayofweek)
        & pd.Index(flows.index.time).isin(set(times))
        & ~mask_window
        & flows.index.isin(full_idx)
    )
    baseline = flows[baseline_mask]

    id_to_name = dict(zip(stations["station_id"], stations["station_name"]))

    def _block(ids: list[str], pct_key: str) -> list[dict]:
        out = []
        for sid in ids:
            col = flow_column_for(loader.col_to_station_id, sid)
            if col is None:
                continue
            obs = float(during[col].sum())
            base_slots = float(baseline[col].mean()) * len(during) if len(baseline) else 0.0
            pct = ((obs - base_slots) / base_slots * 100.0) if base_slots > 0 else None
            out.append(
                {
                    "station": id_to_name.get(sid, sid),
                    "station_id": sid,
                    "flow_during": as_int(obs),
                    "baseline": as_float(base_slots),
                    pct_key: as_float(pct) if pct is not None else None,
                }
            )
        out.sort(key=lambda d: d["baseline"], reverse=True)
        return out

    during_block = _block(affected_ids, "drop_pct")
    for entry in during_block:
        # drop_pct positiv = Rueckgang gegenueber Baseline.
        if entry["drop_pct"] is not None:
            entry["drop_pct"] = as_float(-entry["drop_pct"])

    return {
        "closure": row.description,
        "when": row.when.strftime(TS_OUT),
        "end_time": row.end_time.strftime(TS_OUT),
        "cause": extract_cause(row.description),
        "affected_lines": extract_lines(row.description),
        "affected_stations": [id_to_name.get(s, s) for s in affected_ids],
        "during_closure": during_block,
        "neighboring_stations": _block(neighbor_ids, "change_pct"),
        "baseline_slots_used": int(len(baseline)),
        "data_limitation": (
            "Keine OD-Matrix verfügbar – Umleitungsströme lassen sich nicht "
            "direkt zuordnen. Veränderungen an Nachbarstationen sind "
            "Korrelation, nicht nachgewiesene Umleitung. Baseline ist der "
            "Mittelwert derselben Slots an allen übrigen Tagen desselben "
            "Wochentags."
        ),
    }
