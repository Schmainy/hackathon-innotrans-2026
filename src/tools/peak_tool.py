"""Tool 7 – typisches Tagesprofil und Netzvergleich.

get_peak_profile            Durchschnittsprofil einer Station über alle Tage
compare_station_to_network  Peak der Station gegen den Netzschnitt

Beantwortet Fragen der Form "Wann ist der Pendler-Peak an Station X?" und
"Liegt dieser Peak über dem Mittel aller Stationen?" – also Fragen nach dem
Normalfall statt nach einem einzelnen Tag. Ein einzelner Tag wäre hier eine
schlechte Grundlage: die 15-Minuten-Werte rauschen stark (Lag-1-Autokorrelation
einer Einzelstation nur ~0,38).
"""

from __future__ import annotations

import pandas as pd

from src.loader import DataLoader
from src.tools._common import (
    as_float,
    as_int,
    flow_column_for,
    full_day_index,
    resolve_station,
)

# Morgens- und Abendfenster für die Peak-Suche.
MORNING = (5, 12)
EVENING = (12, 24)

DATA_LIMITATION = (
    "Mittelwert über gleiche Tageszeit-Slots aller einbezogenen Tage; "
    "unvollständige Randtage (weniger als 80 Slots) sind ausgeschlossen. "
    "Flow-Werte sind simulierte Stationszählungen ohne Richtungsangabe – "
    "Ein- und Ausstieg lassen sich nicht trennen, eine OD-Matrix gibt es "
    "nicht. Einzelwerte sind bei 3.000 pro 15-Minuten-Slot gedeckelt."
)


def _profile_frame(weekday_only: bool, weekend_only: bool = False):
    """Liefert (mean, max, min, days) je Tageszeit für ALLE Stationen.

    Tagesauswahl:
        weekend_only=True   nur Sa+So (dayofweek 5, 6) – hat Vorrang
        weekday_only=True   nur Mo-Fr (dayofweek 0-4)
        beide False         alle sieben Tage

    Eine einzige groupby-Operation über den gesamten Flow-Frame – dadurch ist
    auch der Netzvergleich über 168 Stationen billig.
    """
    loader = DataLoader()
    flows = loader.load_flows()
    full = flows.loc[full_day_index(flows)]
    if weekend_only:
        full = full[full.index.dayofweek >= 5]
    elif weekday_only:
        full = full[full.index.dayofweek < 5]
    if full.empty:
        return loader, None, None, None, 0

    grouped = full.groupby(full.index.time)
    days = int(full.index.normalize().nunique())
    return loader, grouped.mean(), grouped.max(), grouped.min(), days


def _peak_in_window(means: pd.Series, lo: int, hi: int) -> tuple[str, float] | None:
    """Zeit und Wert des Maximums innerhalb eines Stundenfensters."""
    window = means[[t for t in means.index if lo <= t.hour < hi]]
    if window.empty:
        return None
    peak_time = window.idxmax()
    return peak_time.strftime("%H:%M"), float(window.loc[peak_time])


def get_peak_profile(
    station_name: str,
    weekday_only: bool = True,
    weekend_only: bool = False,
) -> dict:
    """Typisches Tagesprofil einer Station, gemittelt über alle Betriebstage.

    weekend_only=True wertet ausschliesslich Samstag und Sonntag aus und hat
    Vorrang vor weekday_only. Der Netzvergleich (peak_vs_network_mean) nutzt
    dieselbe Tagesauswahl, vergleicht also Wochenende mit Wochenende.
    """
    try:
        loader, means, maxes, mins, days = _profile_frame(
            weekday_only, weekend_only
        )
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    if means is None:
        return {"error": "No data", "station_name": station_name}

    try:
        stations = loader.load_stations()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    matches = resolve_station(stations, station_name)
    if not matches:
        return {"error": "Station not found", "input": station_name}

    match = matches[0]
    col = flow_column_for(loader.col_to_station_id, match["station_id"])
    if col is None:
        return {"error": "Station not in flow data", "input": station_name}

    station_means = means[col]
    profile = [
        {
            "time": t.strftime("%H:%M"),
            "mean_flow": as_float(station_means.loc[t]),
            "max_flow": as_int(maxes.loc[t, col]),
            "min_flow": as_int(mins.loc[t, col]),
        }
        for t in station_means.index
    ]

    morning = _peak_in_window(station_means, *MORNING)
    evening = _peak_in_window(station_means, *EVENING)

    # Netzvergleich: Morgenspitze dieser Station gegen den Mittelwert der
    # Morgenspitzen aller Stationen.
    morning_rows = [t for t in means.index if MORNING[0] <= t.hour < MORNING[1]]
    network_peaks = means.loc[morning_rows].max(axis=0)
    network_mean = float(network_peaks.mean())
    station_peak = morning[1] if morning else 0.0
    delta = ((station_peak - network_mean) / network_mean * 100.0) if network_mean else 0.0

    result = {
        "station_name": match["station_name"],
        "station_id": match["station_id"],
        "weekday_only": bool(weekday_only) and not weekend_only,
        "weekend_only": bool(weekend_only),
        "day_selection": (
            "Samstag+Sonntag" if weekend_only
            else "Montag-Freitag" if weekday_only
            else "alle sieben Tage"
        ),
        "days_included": days,
        "profile": profile,
        "morning_peak": None,
        "evening_peak": None,
        "peak_vs_network_mean": as_float(delta),
        "lines": match["u_bahn_lines"],
        "daily_mean_total": as_float(station_means.sum()),
        "data_limitation": DATA_LIMITATION,
    }
    if morning:
        result["morning_peak"] = {
            "time": morning[0],
            "mean_flow": as_float(morning[1]),
            "hour_range": "05:00-12:00",
        }
    if evening:
        result["evening_peak"] = {
            "time": evening[0],
            "mean_flow": as_float(evening[1]),
            "hour_range": "12:00-23:59",
        }
    if len(matches) > 1:
        result["ambiguity"] = {
            "note": ("Stationsname ist im Datensatz nicht eindeutig. "
                     "Ausgewertet wurde die erste Bahnsteigebene."),
            "alternatives": matches[1:],
        }
    return result


def compare_station_to_network(station_name: str) -> dict:
    """Vergleicht die Morgenspitze einer Station mit dem Netzdurchschnitt.

    Beantwortet Trainingsfrage 4: "Liegt der Peak über dem mittleren
    Pendler-Peak aller Stationen?"
    """
    try:
        loader, means, _maxes, _mins, days = _profile_frame(weekday_only=True)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    if means is None:
        return {"error": "No data", "station_name": station_name}

    try:
        stations = loader.load_stations()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    matches = resolve_station(stations, station_name)
    if not matches:
        return {"error": "Station not found", "input": station_name}

    match = matches[0]
    col = flow_column_for(loader.col_to_station_id, match["station_id"])
    if col is None:
        return {"error": "Station not in flow data", "input": station_name}

    morning_rows = [t for t in means.index if MORNING[0] <= t.hour < MORNING[1]]
    peaks = means.loc[morning_rows].max(axis=0).sort_values(ascending=False)

    station_peak = float(peaks[col])
    network_mean = float(peaks.mean())
    rank = int(peaks.index.get_loc(col)) + 1
    delta = ((station_peak - network_mean) / network_mean * 100.0) if network_mean else 0.0

    peak_time = _peak_in_window(means[col], *MORNING)

    # Abendspitze ebenso: an Wohnstationen (Rudow) ist sie oft die hoehere –
    # "the commute peak" darf nicht nur morgens verglichen werden.
    evening_rows = [t for t in means.index if EVENING[0] <= t.hour < EVENING[1]]
    evening_peaks = means.loc[evening_rows].max(axis=0)
    evening_station = float(evening_peaks[col])
    evening_network = float(evening_peaks.mean())
    evening_time = _peak_in_window(means[col], *EVENING)
    evening_rank = int(evening_peaks.sort_values(ascending=False).index.get_loc(col)) + 1

    return {
        "station_name": match["station_name"],
        "station_id": match["station_id"],
        "station_morning_peak_mean": as_float(station_peak),
        "station_morning_peak_time": peak_time[0] if peak_time else None,
        "network_morning_peak_mean": as_float(network_mean),
        "network_morning_peak_median": as_float(float(peaks.median())),
        "is_above_network_mean": bool(station_peak > network_mean),
        "difference_pct": as_float(delta),
        "rank_among_stations": rank,
        "evening": {
            "station_peak_mean": as_float(evening_station),
            "station_peak_time": evening_time[0] if evening_time else None,
            "network_peak_mean": as_float(evening_network),
            "is_above_network_mean": bool(evening_station > evening_network),
            "difference_pct": as_float((evening_station - evening_network) / evening_network * 100.0)
            if evening_network else None,
            "rank_among_stations": evening_rank,
        },
        "higher_peak_of_day": "evening" if evening_station > station_peak else "morning",
        "total_stations": int(len(peaks)),
        "days_included": days,
        "weekday_only": True,
        "data_limitation": (
            "Verglichen wird die mittlere Morgenspitze (05:00-12:00) über "
            f"{days} Werktage. Umsteigebahnhöfe erscheinen einmal je "
            "Bahnsteigebene. " + DATA_LIMITATION
        ),
    }
