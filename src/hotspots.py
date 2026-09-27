"""Hotspots für die Seitenleiste der Operator-Oberfläche.

Jeder Hotspot ist ein Ort mit bekannt hoher Belastung plus eine fertige
Frage (auto_query), die per Klick an den Agenten geht. Abgeleitet wird
ausschliesslich aus den vorhandenen Datensätzen – nichts wird erfunden:

    konzert      Music-Venues mit den meisten Events (Eventdatei), je mit
                 dem Event der höchsten (skalierten) Attendance
    messe        Messe-/Konferenz-Events (InnoTrans auf dem Messegelände)
    transit_hub  die am stärksten frequentierten Stationen (Flow-Daten)
                 und die kritischsten Netzknoten

Kategorien wie sport, demo oder politik kennt das Frontend zwar, die
Eventdaten enthalten dafür aber keine Einträge – sie bleiben deshalb leer.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.loader import DataLoader
from src.tools.temporal_tool import get_busiest_station_by_weekday

MAX_VENUES = 3
MAX_HUBS = 3


def _short(name: str) -> str:
    return re.sub(r"\s*\(Berlin\)$", "", str(name))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _concert_hotspots(events) -> list[dict]:
    music = events[(events["segment"] == "Music") & events["venue_name"].notna()]
    venues = music["venue_name"].value_counts().head(MAX_VENUES).index
    out = []
    for venue in venues:
        rows = music[music["venue_name"] == venue]
        top = rows.sort_values("estimated_attendance", ascending=False, na_position="last").iloc[0]
        day = top["began_local"]
        out.append({
            "id": f"konzert-{_slug(venue)}",
            "label": venue,
            "category": "konzert",
            "auto_query": (
                f"{top['event_name']} event on {day:%B} {day.day}, {day:%Y} at the {venue}: "
                "how did the passenger flow develop at the neighbouring stations?"
            ),
            "source": f"{len(rows)} music events in the event data",
        })
    return out


def _fair_hotspots(events) -> list[dict]:
    fair = events[events["event_name"].str.contains("innotrans", case=False, na=False)]
    if fair.empty:
        return []
    first, last = fair["began_local"].min(), fair["began_local"].max()
    return [{
        "id": "messe-innotrans",
        "label": f"InnoTrans · Messe Berlin ({first:%d.%m.}–{last:%d.%m.})",
        "category": "messe",
        "auto_query": "What happened at the stations near Messe Berlin during the InnoTrans event?",
        "source": f"{len(fair)} InnoTrans days in the event data",
    }]


def _hub_hotspots() -> list[dict]:
    ranking = get_busiest_station_by_weekday(weekday=None, top_n=MAX_HUBS)
    out = []
    for station in ranking.get("top_5_stations", [])[:MAX_HUBS]:
        name = _short(station["station_name"])
        out.append({
            "id": f"hub-{_slug(name)}",
            "label": f"{name} ({station['lines']})" if station.get("lines") else name,
            "category": "transit_hub",
            "auto_query": (
                f"When does the commute flow peak at {name} usually take place, and does "
                "it exceed the mean commute peak across all stations?"
            ),
            "source": f"rank {station['rank']} by mean daily flow",
        })
    out.append({
        "id": "hub-critical-stations",
        "label": "Kritische Netzknoten",
        "category": "transit_hub",
        "auto_query": "Which stations are the most critical in the network?",
        "source": "articulation points of the U-Bahn graph",
    })
    return out


@lru_cache(maxsize=4)
def _build(signature: tuple) -> dict:
    """Aufwendiger Teil, gecacht je Datenstand (Datumsgrenzen der Flow-Dateien)."""
    events = DataLoader().load_events()
    hotspots = _concert_hotspots(events) + _fair_hotspots(events) + _hub_hotspots()
    return {"hotspots": hotspots, "data_period": dict(signature)}


def build_hotspots() -> dict:
    """Hotspot-Liste für /api/hotspots; bei Datenfehlern eine leere Liste."""
    try:
        dates = DataLoader().data_dates()
        return _build(tuple(sorted(dates.items())))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"hotspots": [], "error": f"{type(exc).__name__}: {exc}"}


__all__ = ["build_hotspots"]
