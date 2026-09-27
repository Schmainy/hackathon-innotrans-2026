"""Tool 3 – Events in Berlin.

get_events  gefilterte Eventliste inkl. nächstgelegener U-Bahn-Station

Die Events tragen keinen Stationsschlüssel und keine Koordinaten. Ohne
Geocoding-API wird die Zuordnung über kuratierte Koordinatentabellen gemacht
(Venue-Namen und Adressen); alles andere bleibt bewusst null statt geraten
zu werden. Die nächste Station wird immer per Haversine über alle 168
Stationen bestimmt, nie fest verdrahtet.

Liegt das angefragte Datum hinter dem letzten Datentag, gibt es keine
Eventzeilen – stattdessen liefert get_events historische Muster vergangener
Events am selben Ort (Anzahl, Attendance, Peak an der nächsten Station an
Eventtagen gegenüber normalen Tagen).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.loader import DataLoader
from src.tools._common import (
    DATE_OUT,
    TS_OUT,
    as_float,
    as_int,
    flow_column_for,
    full_day_index,
    haversine_m,
    parse_date,
)

# Kuratierte Venue-Koordinaten (WGS84). Schluessel werden normalisiert
# (Kleinschreibung, ohne Umlaute/Akzente, "str." -> "strasse") als Teilstring
# gegen venue_name UND address geprueft; laengster Schluessel gewinnt.
VENUE_COORDS: dict[str, tuple[float, float]] = {
    # === ARENAS & CONCERT VENUES ===
    "uber arena":               (52.5053, 13.4435),   # Warschauer Str. (U1/U3)
    "mercedes-benz arena":      (52.5053, 13.4435),   # same venue, old name
    "uber-platz":               (52.5053, 13.4435),   # address of Uber Arena
    "uber eats music hall":     (52.5046, 13.4418),   # next to Uber Arena
    "velodrom":                 (52.5313, 13.4486),   # Paul-Heyse-Str. 26 (S Landsberger Allee)
    "max-schmeling-halle":      (52.5431, 13.4141),   # S+U Prenzlauer Allee
    "columbiahalle":            (52.4766, 13.3882),   # U Platz der Luftbrücke
    "tempodrom":                (52.5006, 13.3817),   # U Möckernbrücke / U Hallesches Tor
    "admiralspalast":           (52.5238, 13.3893),   # U/S Friedrichstr.
    "zitadelle spandau":        (52.5353, 13.2081),   # U Zitadelle (U7)
    "waldbühne":                (52.5142, 13.2233),   # S Pichelsberg
    "huxleys neue welt":        (52.4875, 13.4357),   # U Hermannplatz (U7/U8)
    "lido":                     (52.5025, 13.4473),   # U Schlesisches Tor (U1)
    "arena berlin":             (52.4967, 13.4603),   # S Treptower Park

    # === STADIUMS ===
    "olympiastadion":           (52.5147, 13.2396),   # U Olympia-Stadion (U2)
    "olympia-stadion":          (52.5147, 13.2396),
    "olympiapark":              (52.5147, 13.2396),   # Lollapalooza site, same campus as stadium
    "fc union berlin":          (52.4573, 13.5683),   # S Köpenick / Alte Försterei
    "an der alten försterei":   (52.4573, 13.5683),
    "friedrich-ludwig-jahn-sportpark": (52.5389, 13.4135),  # S Prenzlauer Allee
    "jahn-sportpark":           (52.5389, 13.4135),

    # === FAIRGROUNDS & CONGRESS ===
    "messe berlin":             (52.5072, 13.2821),   # U Kaiserdamm (U2) / S Messe Nord/ICC
    "berlin fairground":        (52.5072, 13.2821),   # InnoTrans-Eintrag im Testdatensatz
    "icc berlin":               (52.5072, 13.2821),   # same campus
    "innotrans":                (52.5072, 13.2821),   # InnoTrans is at Messe Berlin
    "station berlin":           (52.5025, 13.3640),   # U Gleisdreieck (U1/U2/U3)
    "kraftwerk berlin":         (52.5025, 13.4547),   # Bus/Tram Köpenicker Str.
    "citycube":                 (52.5072, 13.2821),   # Messe Berlin campus

    # === PARKS & OUTDOOR ===
    "tempelhofer feld":         (52.4730, 13.4030),   # U Boddinstr. / Paradestr.
    "volkspark friedrichshain": (52.5264, 13.4373),   # S Landsberger Allee / Tram
    "treptower park":           (52.4951, 13.4721),   # S Treptower Park
    "botanischer garten":       (52.4483, 13.3054),   # S Botanischer Garten
    "tierpark":                 (52.5083, 13.5253),   # U Tierpark (U5)

    # === CULTURAL & OTHER ===
    "berghain":                 (52.5112, 13.4426),   # S Ostbahnhof / Bus
    "east side gallery":        (52.5057, 13.4469),   # S Ostbahnhof / U Warschauer Str.
    "museumsinsel":             (52.5169, 13.3985),   # U/S Hackescher Markt
    "brandenburger tor":        (52.5163, 13.3777),   # S+U Brandenburger Tor (S1/S2/S25/U55)
    "reichstag":                (52.5186, 13.3761),   # S+U Brandenburger Tor
    "charlottenburg":           (52.5200, 13.2955),   # U Sophie-Charlotte-Platz (U2)
    "spandau":                  (52.5353, 13.2010),   # U Rathaus Spandau (U7)
}

# Adressen aus berlin_events_*.csv, von Hand geokodiert (Genauigkeit ca.
# +-300 m). Die meisten Eventzeilen haben keinen venue_name, nur eine Adresse
# – ohne diese Tabelle blieben z. B. alle 113 Events am Marlene-Dietrich-Platz
# ohne Station. Bewusst NICHT enthalten: "Invalidenstraße" ohne Hausnummer
# (2,5 km lange Strasse, keine eindeutige Station).
ADDRESS_COORDS: dict[str, tuple[float, float]] = {
    "marlene-dietrich-platz":   (52.5076, 13.3727),   # Stage Theater am Potsdamer Platz
    "metzer strasse 2":         (52.5316, 13.4118),   # Tati Goes Underground
    "sommeringstrasse 15":      (52.5253, 13.3110),   # Charlottenburg, Mierendorff-Kiez
    "gendarmenmarkt":           (52.5136, 13.3925),
    "lenaustrasse 7":           (52.4935, 13.4264),   # Oblomov Kreuzkoelln
    "budapester strasse 45":    (52.5048, 13.3381),   # Europa-Center
    "olympischer platz":        (52.5147, 13.2396),   # Olympiastadion
    "friedrich-friesen-allee":  (52.5147, 13.2396),   # Olympiapark
    "hasenheide 107":           (52.4868, 13.4213),   # Huxleys Neue Welt
    "schonhauser allee 36":     (52.5390, 13.4130),   # Kulturbrauerei
    "markisches ufer 48":       (52.5119, 13.4095),
    "nollendorfplatz 5":        (52.4995, 13.3537),   # Metropol / Mikropol
    "columbiadamm 13":          (52.4838, 13.3895),   # Columbiahalle
    "columbiadamm 9":           (52.4840, 13.3880),   # Columbia Theater
    "hermannstrasse 146":       (52.4775, 13.4265),
    "skalitzer strasse 85":     (52.5005, 13.4388),
    "skalitzer strasse 134":    (52.4990, 13.4190),
    "schnellerstrasse 137":     (52.4577, 13.5140),   # RSO.BERLIN
    "mockernstrasse 10":        (52.5006, 13.3817),   # Tempodrom (beide Schreibweisen)
    "moeckernstrasse 10":       (52.5006, 13.3817),
    "eichenstrasse 4":          (52.4950, 13.4590),   # Haus der Visionäre
    "oudenarder strasse 16":    (52.5518, 13.3575),   # Vagabund Brauerei
    "obentrautstrasse 19":      (52.4958, 13.3800),
    "strasse am fez":           (52.4608, 13.5393),   # FEZ Wuhlheide
    "lilli-henoch-strasse 10":  (52.5337, 13.4396),
    "am juliusturm":            (52.5405, 13.2130),   # Zitadelle Spandau
    "paul-heyse-strasse 26":    (52.5313, 13.4486),   # Velodrom
    "treptower strasse 39":     (52.4865, 13.4520),   # Beach Neukölln
    "wilhelm-kabus-strasse 24": (52.4835, 13.3605),
    "friedrichstrasse 101":     (52.5213, 13.3890),   # Admiralspalast
    "kantstrasse 12":           (52.5057, 13.3297),   # Theater des Westens
    "am wriezener bahnhof":     (52.5111, 13.4431),   # Berghain / Kantine
    "am glockenturm":           (52.5165, 13.2285),   # Waldbühne
    "holzmarktstrasse 15":      (52.5130, 13.4200),
    "holzmarktstrasse 25":      (52.5115, 13.4260),
    "cuvrystrasse 7":           (52.4999, 13.4447),   # Lido
    "schlesisches tor":         (52.5010, 13.4415),   # "Im U-Bhf. Schlesisches Tor"
    "karl-marx-strasse 141":    (52.4770, 13.4400),
    "kurt-schumacher-damm 207": (52.5520, 13.2990),
    "platz der luftbrucke":     (52.4850, 13.3860),
    "revaler strasse 99":       (52.5072, 13.4545),   # RAW-Gelände
    # Testdatensatz 22.09.–01.10.2026
    "messedamm 22":             (52.5072, 13.2821),   # Messe Berlin / InnoTrans
    "franklinstrasse 10":       (52.5186, 13.3290),   # Anna Spree, Charlottenburg
    "herbert-von-karajan-strasse 1": (52.5100, 13.3700),  # Philharmonie
}

# Zeitfenster um ein Event fuer den Peak-Vergleich: Anreise vor Beginn,
# Abreise nach Ende. Fehlt das Ende, gilt Beginn + DEFAULT_EVENT_HOURS.
ARRIVAL_BEFORE = pd.Timedelta(hours=1)
DEPARTURE_AFTER = pd.Timedelta(hours=1, minutes=30)
DEFAULT_EVENT_HOURS = pd.Timedelta(hours=3)

MAX_PATTERNS = 10

ATTENDANCE_NOTE = (
    "estimated_attendance ist skaliert (Wertebereich 165-2394 über alle "
    "Venues), nicht die tatsächliche Besucherzahl – nur als relatives Ranking "
    "verwendbar."
)


def _norm(text: str) -> str:
    """Normalisiert Venue-/Adresstext für den Teilstring-Vergleich.

    "Möckernstraße 10" und "Mockernstrasse 10" ergeben denselben Schlüssel.
    """
    s = str(text).lower().replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"str\.", "strasse", s)
    return re.sub(r"\s+", " ", s).strip()


# Laengster Schluessel zuerst, damit "uber eats music hall" nicht an
# "uber-platz" und "olympia-stadion" nicht an kuerzeren Treffern haengt.
_LOOKUP: list[tuple[str, str, tuple[float, float]]] = sorted(
    [(_norm(k), k, v) for k, v in VENUE_COORDS.items()]
    + [(_norm(k), k, v) for k, v in ADDRESS_COORDS.items()],
    key=lambda item: len(item[0]),
    reverse=True,
)


def _venue_coords(
    venue: str | None, address: str | None
) -> tuple[str, tuple[float, float]] | None:
    """(Schlüssel, Koordinaten) über Venue-Name oder Adresse; None wenn unbekannt."""
    haystack = " ".join(
        _norm(part) for part in (venue, address) if isinstance(part, str)
    )
    if not haystack:
        return None
    for key, label, coords in _LOOKUP:
        if key in haystack:
            return label, coords
    return None


def _keyword_lookup(keyword: str) -> tuple[str, tuple[float, float]] | None:
    """Kurzes Stichwort ("messe", "arena") gegen die Venue-Schlüssel.

    Umkehrung von _venue_coords: hier steckt das Stichwort im Schlüssel.
    Reihenfolge der Tabelle entscheidet, "arena" landet also bei der Uber Arena.
    """
    needle = _norm(keyword)
    if len(needle) < 4:
        return None
    for key, coords in VENUE_COORDS.items():
        if needle in _norm(key):
            return key, coords
    return None


def _nearest_station(stations: pd.DataFrame, lat: float, lon: float) -> dict:
    """Nächste U-Bahn-Station per Haversine über alle Stationen."""
    dists = stations.apply(
        lambda r: haversine_m(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    best = dists.idxmin()
    return {
        "station_name": stations.loc[best, "station_name"],
        "station_id": stations.loc[best, "station_id"],
        "distance_m": as_float(dists.loc[best], 1),
    }


MAX_PATTERN_EVENTS = 30


NEARBY_RADIUS_M = 1500
MAX_NEARBY_STATIONS = 3
MAX_EVENTS_WITH_NEARBY = 20


def _nearby_stations(stations: pd.DataFrame, lat: float, lon: float) -> list[dict]:
    """Bis zu 3 Stationen im Umkreis von 1,5 km, nächste zuerst (eindeutige Namen)."""
    dists = stations.apply(
        lambda r: haversine_m(lat, lon, r["latitude"], r["longitude"]), axis=1
    ).sort_values()
    out, seen = [], set()
    for idx, dist in dists.items():
        if dist > NEARBY_RADIUS_M or len(out) >= MAX_NEARBY_STATIONS:
            break
        name = stations.loc[idx, "station_name"]
        if name in seen:
            continue
        seen.add(name)
        out.append({"station_name": name, "lines": stations.loc[idx, "u_bahn_lines"],
                    "distance_m": as_float(dist, 1)})
    return out


def get_events(
    date_str: str | None = None,
    venue_keyword: str | None = None,
    min_attendance: int = 0,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Events, gefiltert nach Datum, Venue-Stichwort und Mindest-Attendance.

    date_from/date_to (inklusiv) filtern einen Zeitraum, z. B. eine Woche.
    Liegt date_str hinter dem letzten vollständigen Datentag, kommen statt
    Eventzeilen historische Muster zurück (is_future_event=True). Mit
    venue_keyword kommen zusätzlich venue_patterns: Peak an der nächsten
    Station an den Eventtagen gegenüber gleichen Wochentagen ohne Event.
    """
    try:
        min_attendance = int(min_attendance or 0)
    except (TypeError, ValueError):
        return {"error": "Invalid min_attendance", "input": min_attendance}

    loader = DataLoader()
    try:
        events = loader.load_events()
        stations = loader.load_stations()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    filters = {
        "date": date_str,
        "date_from": date_from,
        "date_to": date_to,
        "venue_keyword": venue_keyword,
        "min_attendance": min_attendance,
    }
    df = events

    for bound, keep in ((date_from, lambda d, b: d >= b), (date_to, lambda d, b: d <= b)):
        if bound is None:
            continue
        parsed = parse_date(bound)
        if parsed is None:
            return {"error": "Invalid date", "input": bound}
        df = df[keep(df["began_local"].dt.date, parsed.date())]

    if date_str is not None:
        day = parse_date(date_str)
        if day is None:
            return {"error": "Invalid date", "input": date_str}

        try:
            flows = loader.load_flows()
        except (FileNotFoundError, ValueError) as exc:
            return {"error": "Data load failed", "detail": str(exc)}
        last_day = full_day_index(flows).max().normalize()
        if day.normalize() > last_day:
            return _future_events(
                day, last_day, venue_keyword, events, stations, flows, loader, filters
            )

        df = df[df["began_local"].dt.date == day.date()]

    if venue_keyword is not None:
        needle = str(venue_keyword).strip().lower()
        in_venue = df["venue_name"].fillna("").str.lower().str.contains(needle, regex=False)
        in_addr = df["address"].fillna("").str.lower().str.contains(needle, regex=False)
        in_name = df["event_name"].fillna("").str.lower().str.contains(needle, regex=False)
        df = df[in_venue | in_addr | in_name]

    if min_attendance:
        df = df[df["estimated_attendance"].fillna(0) >= int(min_attendance)]

    out = []
    with_nearby = len(df) <= MAX_EVENTS_WITH_NEARBY
    for row in df.sort_values("began_local").itertuples():
        venue = row.venue_name if isinstance(row.venue_name, str) else None
        address = row.address if isinstance(row.address, str) else None
        match = _venue_coords(venue, address)
        nearest = _nearest_station(stations, *match[1]) if match else {}
        nearby = _nearby_stations(stations, *match[1]) if (match and with_nearby) else []

        end = row.estimated_end_local
        out.append(
            {
                "event_name": row.event_name,
                "began_local": row.began_local.strftime(TS_OUT),
                "end_local": end.strftime(TS_OUT) if pd.notna(end) else None,
                "venue": venue,
                "address": address,
                "segment": row.segment,
                "genre": row.genre if isinstance(row.genre, str) else None,
                "estimated_attendance": (
                    as_int(row.estimated_attendance)
                    if pd.notna(row.estimated_attendance) else None
                ),
                "nearest_station": nearest.get("station_name"),
                "nearest_station_id": nearest.get("station_id"),
                "distance_m": nearest.get("distance_m"),
                "nearby_stations": nearby,
            }
        )

    located = sum(1 for e in out if e["nearest_station"] is not None)
    patterns = _patterns_for(df, stations, loader) if venue_keyword else []
    return {
        "filters": filters,
        "count": len(out),
        "events": out,
        "venue_patterns": patterns,
        "geocoded_count": located,
        "is_future_event": False,
        "data_limitation": (
            ATTENDANCE_NOTE + " nearest_station stammt aus kuratierten "
            "Koordinatentabellen (Venue-Namen und Adressen, Genauigkeit ca. "
            "+-300 m) und ist die per Luftlinie nächste U-Bahn-Station; "
            f"ohne Tabelleneintrag ist sie null ({len(out) - located} von "
            f"{len(out)} Treffern ohne Zuordnung). Kein Geocoding-Dienst im Einsatz."
        ),
    }


def _patterns_for(df: pd.DataFrame, stations: pd.DataFrame, loader: DataLoader) -> list[dict]:
    """Venue-Muster für gefilterte Events (z. B. alle InnoTrans-Tage).

    Nur bei überschaubaren Treffern – eine ungefilterte Liste hätte hunderte
    Orte und würde den Kontext sprengen.
    """
    if df.empty or len(df) > MAX_PATTERN_EVENTS:
        return []
    try:
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError):
        return []
    groups: dict[tuple[float, float], list] = {}
    labels: dict[tuple[float, float], str] = {}
    for row in df.itertuples():
        venue = row.venue_name if isinstance(row.venue_name, str) else None
        address = row.address if isinstance(row.address, str) else None
        match = _venue_coords(venue, address)
        if match is None:
            continue
        groups.setdefault(match[1], []).append(row)
        labels.setdefault(match[1], match[0])
    full_days = set(full_day_index(flows).normalize())
    patterns = [
        _venue_pattern(labels[c], rows, c, stations, flows, loader, full_days)
        for c, rows in groups.items()
    ]
    return sorted(patterns, key=lambda p: p["past_events"], reverse=True)[:MAX_PATTERNS]


def _future_events(
    day: pd.Timestamp,
    last_day: pd.Timestamp,
    venue_keyword: str | None,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    flows: pd.DataFrame,
    loader: DataLoader,
    filters: dict,
) -> dict:
    """Historische Event-Muster für ein Datum hinter dem Datensatzende.

    Gruppiert alle vergangenen Events nach Ort (Tabellenschlüssel) und
    vergleicht je Ort den Peak an der nächsten Station an Eventtagen mit
    demselben Tageszeitfenster an gleichen Wochentagen ohne Event dort.
    """
    target = None
    if venue_keyword is not None:
        target = _venue_coords(venue_keyword, None) or _keyword_lookup(venue_keyword)

    # Gruppiert wird nach Koordinaten, nicht nach Schluessel: "Olympiapark"
    # und "Olympischer Platz 3" sind derselbe Ort und dieselbe Station.
    groups: dict[tuple[float, float], list] = {}
    label_by_coords: dict[tuple[float, float], str] = {}
    for row in events.itertuples():
        venue = row.venue_name if isinstance(row.venue_name, str) else None
        address = row.address if isinstance(row.address, str) else None
        match = _venue_coords(venue, address)
        if match is None:
            continue
        key, coords = match
        if venue_keyword is not None:
            # Unbekanntes Stichwort: nur Textreffer zaehlen, sonst liefe die
            # Antwort mit den Mustern fremder Venues.
            needle = _norm(venue_keyword)
            haystack = " ".join(_norm(p) for p in (venue, address) if p)
            same_place = target is not None and coords == target[1]
            if not same_place and needle not in haystack:
                continue
        groups.setdefault(coords, []).append(row)
        label_by_coords.setdefault(coords, key)

    full_days = set(full_day_index(flows).normalize())
    patterns = [
        _venue_pattern(label_by_coords[coords], rows, coords, stations, flows, loader, full_days)
        for coords, rows in groups.items()
    ]
    patterns.sort(key=lambda p: p["past_events"], reverse=True)
    patterns = patterns[:MAX_PATTERNS]
    n_events = sum(p["past_events"] for p in patterns)

    result = {
        "filters": filters,
        "count": 0,
        "events": [],
        "is_future_event": True,
        "requested_date": day.strftime(DATE_OUT),
        "last_data_day": last_day.strftime(DATE_OUT),
        "historical_patterns": patterns,
        "historical_event_count": n_events,
        "data_limitation": (
            "Event date is outside the training data range. Recommendation "
            f"based on historical patterns from {n_events} similar past events. "
            "The events file contains no rows for this date; peak values are "
            "15-minute station flows averaged over past event days, compared "
            "with the same time window on the same weekdays without an event "
            "at that venue. " + ATTENDANCE_NOTE
        ),
    }
    if target is not None:
        result["venue_location"] = {
            "matched_key": target[0],
            "latitude": target[1][0],
            "longitude": target[1][1],
            **_nearest_station(stations, *target[1]),
        }
    elif venue_keyword is not None:
        result["venue_location"] = None
        result["venue_note"] = (
            f"Venue '{venue_keyword}' ist in keiner Koordinatentabelle – "
            "keine Stationszuordnung möglich."
        )
    return result


def _venue_pattern(
    key: str,
    rows: list,
    coords: tuple[float, float],
    stations: pd.DataFrame,
    flows: pd.DataFrame,
    loader: DataLoader,
    full_days: set,
) -> dict:
    """Kennzahlen vergangener Events an einem Ort."""
    nearest = _nearest_station(stations, *coords)
    attendance = [
        float(r.estimated_attendance) for r in rows if pd.notna(r.estimated_attendance)
    ]
    pattern = {
        "venue_key": key,
        "past_events": len(rows),
        "event_dates": sorted({r.began_local.strftime(DATE_OUT) for r in rows}),
        "avg_estimated_attendance": as_float(sum(attendance) / len(attendance)) if attendance else None,
        "nearest_station": nearest["station_name"],
        "nearest_station_id": nearest["station_id"],
        "distance_m": nearest["distance_m"],
        "mean_peak_on_event_days": None,
        "mean_peak_normal_days": None,
        "peak_uplift_pct": None,
        "event_days_compared": 0,
    }

    col = flow_column_for(loader.col_to_station_id, nearest["station_id"])
    if col is None:
        return pattern
    series = flows[col]

    # Ein Fenster je Eventtag: frühester Beginn bis spätestes Ende an dem Tag.
    windows: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for r in rows:
        start = r.began_local.tz_localize(None)
        end = r.estimated_end_local
        end = end.tz_localize(None) if pd.notna(end) else start + DEFAULT_EVENT_HOURS
        lo, hi = start - ARRIVAL_BEFORE, end + DEPARTURE_AFTER
        d = start.normalize()
        if d in windows:
            lo, hi = min(lo, windows[d][0]), max(hi, windows[d][1])
        windows[d] = (lo, hi)

    event_days = set(windows)
    event_peaks, normal_peaks = [], []
    for d, (lo, hi) in windows.items():
        if d not in full_days:
            continue
        window = series[(series.index >= lo) & (series.index < hi)]
        if window.empty:
            continue
        event_peaks.append(float(window.max()))

        # Gleiches Tageszeitfenster an gleichen Wochentagen ohne Event hier.
        offset_lo, offset_hi = lo - d, hi - d
        for other in full_days:
            if other in event_days or other.dayofweek != d.dayofweek:
                continue
            ref = series[(series.index >= other + offset_lo) & (series.index < other + offset_hi)]
            if not ref.empty:
                normal_peaks.append(float(ref.max()))

    if event_peaks:
        ev = sum(event_peaks) / len(event_peaks)
        pattern["mean_peak_on_event_days"] = as_float(ev)
        pattern["event_days_compared"] = len(event_peaks)
        if normal_peaks:
            base = sum(normal_peaks) / len(normal_peaks)
            pattern["mean_peak_normal_days"] = as_float(base)
            if base > 0:
                pattern["peak_uplift_pct"] = as_float((ev - base) / base * 100.0)
    return pattern
