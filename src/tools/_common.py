"""Gemeinsame Helfer für alle Query-Tools.

Enthält nur zustandslose Hilfsfunktionen. Kein Tool exportiert von hier direkt
nach aussen – die öffentliche API sind die Funktionen in den Tool-Modulen.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

TS_OUT = "%Y-%m-%d %H:%M"
DATE_OUT = "%Y-%m-%d"

# Slots pro vollem Betriebstag – zentral im Loader definiert.
from src.loader import FULL_DAY_SLOTS  # noqa: E402

# Ursachen-Mapping. Die Rohbeschreibungen sind englisch ("due to track
# maintenance."); die Challenge-Antworten sind deutsch.
CAUSE_MAP = {
    "track maintenance": "Gleisbau",
    "signal upgrades": "Signaltechnik",
    "switch replacement": "Weichentausch",
    "safety inspection": "Sicherheitsprüfung",
    "power system maintenance": "Energieversorgung",
}


def normalize_station(name: str) -> str:
    """Normalisiert einen Stationsnamen auf einen vergleichbaren Schlüssel.

    Entfernt Präfixe ("U ", "S+U "), das Suffix " (Berlin)", vereinheitlicht
    Strasse/Straße/Str. und wirft alle Sonderzeichen weg. Dadurch matcht die
    Operator-Schreibweise "Kaiserin-Augusta-Strasse" auf den Datensatz-Namen
    "U Kaiserin-Augusta-Str. (Berlin)".
    """
    s = str(name).strip().lower()
    s = re.sub(r"^(s\+u|u|s)\s+", "", s)
    s = re.sub(r"\s*\(berlin\)\s*$", "", s)
    # "Bahnhof"/"Bhf" traegt keine Unterscheidung: "Bahnhof Zoologischer
    # Garten" = "S+U Zoologischer Garten Bhf". Danach erneut Praefix weg
    # ("S+U Bahnhof Alexanderplatz").
    s = re.sub(r"\b(bahnhof|bhf)\b\.?", " ", s).strip()
    s = re.sub(r"^(s\+u|u|s)\s+", "", s)
    s = (
        s.replace("ß", "ss")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
    )
    s = re.sub(r"(strasse|str\.?)(?![a-z])", "str", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# Wortbestandteile ohne Unterscheidungskraft. Bewusst NICHT enthalten sind
# "platz", "damm" und "str": wuerde man die entfernen, fielen
# Hermannplatz/Hermannstr. und Kurfuerstendamm/Kurfuerstenstr. jeweils auf
# denselben Schluessel zusammen - beides stark frequentierte Stationen, die
# dann stillschweigend verwechselt wuerden.
GENERIC_STATION_TOKENS = {
    "u", "s", "su", "bhf", "bahnhof", "berlin", "str", "strasse", "bf",
    "station", "haltestelle", "stop",
}

MIN_TOKEN_LEN = 5


def station_tokens(name: str) -> set[str]:
    """Unterscheidungskräftige Wortbestandteile eines Stationsnamens.

    Dient als Rückfallebene, wenn der Operator eine Station anders benennt
    als der Datensatz ("S+U Bahnhof Spandau" statt "S+U Rathaus Spandau").
    """
    s = str(name).lower()
    s = (
        s.replace("ß", "ss").replace("ä", "ae")
        .replace("ö", "oe").replace("ü", "ue")
    )
    s = re.sub(r"\(berlin\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return {
        t for t in s.split()
        if len(t) >= MIN_TOKEN_LEN and t not in GENERIC_STATION_TOKENS
    }


def resolve_station(stations: pd.DataFrame, name: str) -> list[dict[str, str]]:
    """Löst einen Stationsnamen zu Treffern auf (case- und schreibweisen-tolerant).

    Rückgabe ist eine Liste, weil "U Stadtmitte (Berlin)" auf zwei station_id
    zeigt (U2- und U6-Bahnsteigebene). Leere Liste = kein Treffer.
    """
    key = normalize_station(name)
    if not key:
        return []

    norm = stations["station_name"].map(normalize_station)

    hits = stations[norm == key]
    if hits.empty:
        # Fallback: eindeutiger Teilstring-Treffer (z. B. "Hermannplatz" in
        # einem laengeren Namen). Nur akzeptieren, wenn er nicht mehrdeutig ist.
        contains = stations[norm.str.contains(key, regex=False)]
        if len(contains["station_name"].unique()) == 1:
            hits = contains

    return [
        {"station_id": r.station_id, "station_name": r.station_name,
         "u_bahn_lines": r.u_bahn_lines}
        for r in hits.itertuples()
    ]


def flow_column_for(col_to_station_id: dict[str, str], station_id: str) -> str | None:
    """Findet die Flow-Spalte zu einer station_id über das Positions-Mapping."""
    for col, sid in col_to_station_id.items():
        if sid == station_id:
            return col
    return None


def parse_date(date_str: str) -> pd.Timestamp | None:
    """Parst "YYYY-MM-DD"; gibt None zurück statt zu werfen.

    Auch leere Eingaben ergeben None: pd.Timestamp("") liefert NaT, das erst
    später bei .normalize() mit AttributeError abstürzen würde.
    """
    if date_str is None:
        return None
    try:
        ts = pd.Timestamp(str(date_str).strip())
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts


def validate_hours(hour_from, hour_to) -> tuple[int, int] | dict:
    """Prüft ein Stundenfenster [hour_from, hour_to) innerhalb eines Tages.

    Rückgabe (von, bis) oder ein Fehler-dict. Ohne diese Prüfung liest
    slice_window über die Tagesgrenze: hour_from=-1 holt 23:00 des Vortags,
    hour_to=30 den Folgetag bis 06:00 – falsche Summen ohne jede Warnung.
    """
    try:
        start, end = int(hour_from), int(hour_to)
    except (TypeError, ValueError):
        return {"error": "Invalid hour window", "input": [hour_from, hour_to],
                "expected": "ganze Stunden 0–24, Start < Ende"}
    if not 0 <= start < end <= 24:
        return {"error": "Invalid hour window", "input": [hour_from, hour_to],
                "expected": "0 <= hour_from < hour_to <= 24"}
    return start, end


def hour_window(day: pd.Timestamp, hour_from: int, hour_to: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Liefert [start, ende) des Stundenfensters eines Tages."""
    start = day.normalize() + pd.Timedelta(hours=int(hour_from))
    end = day.normalize() + pd.Timedelta(hours=int(hour_to))
    return start, end


def slice_window(df: pd.DataFrame, day: pd.Timestamp, hour_from: int, hour_to: int) -> pd.DataFrame:
    """Schneidet einen DatetimeIndex-Frame auf Tag + Stundenfenster zu."""
    start, end = hour_window(day, hour_from, hour_to)
    return df[(df.index >= start) & (df.index < end)]


def full_day_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Zeitstempel der Tage mit mindestens 80 Slots (voller Betriebstag).

    Randtage (z. B. 2026-06-10 mit 76, 2026-09-22 mit 4 Slots im
    Trainingsdatensatz) sind unvollstaendig und wuerden jede Tages-Baseline
    verzerren.
    """
    counts = df.groupby(df.index.normalize()).size()
    full_days = set(counts[counts >= FULL_DAY_SLOTS].index)
    return df.index[pd.Series(df.index.normalize(), index=df.index).isin(full_days).to_numpy()]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Grosskreisdistanz zweier WGS84-Punkte in Metern."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def extract_cause(description: str) -> str:
    """Zieht die Störungsursache aus "... due to <cause>." und übersetzt sie.

    Abweichung von der urspruenglichen Spezifikation: die Beschreibungen
    enthalten keine Klammer-Ursache am Ende, sondern eine englische
    "due to"-Phrase. Unbekannte Phrasen werden zu "Unbekannt".
    """
    match = re.search(r"due to\s+(.+?)\s*\.?\s*$", str(description), flags=re.IGNORECASE)
    if not match:
        return "Unbekannt"
    return CAUSE_MAP.get(match.group(1).strip().lower(), "Unbekannt")


def extract_lines(description: str) -> list[str]:
    """Alle Linienkürzel (U1..U9) aus einer Störungsbeschreibung."""
    return sorted(set(re.findall(r"\bU\d\b", str(description))))


def extract_stations(description: str) -> list[str]:
    """Stationsnamen aus einer Störungsbeschreibung.

    Zwei Muster im Datensatz:
      "Line U3 suspended [on a section] between A and B due to <cause>."
      "Station X closed due to <cause>."
    """
    text = str(description)

    seg = re.search(
        r"between\s+(.+?)\s+and\s+(.+?)\s+due to", text, flags=re.IGNORECASE
    )
    if seg:
        return [seg.group(1).strip(), seg.group(2).strip()]

    single = re.search(r"Station\s+(.+?)\s+closed", text, flags=re.IGNORECASE)
    if single:
        return [single.group(1).strip()]

    return []


def as_int(value: Any) -> int:
    """Robuste int-Konvertierung für Rückgabewerte (numpy -> python)."""
    return int(round(float(value)))


def as_float(value: Any, digits: int = 2) -> float:
    """Gerundeter float für Rückgabewerte (numpy -> python)."""
    return round(float(value), digits)
