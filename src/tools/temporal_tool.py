"""Tool 8 – Wochentags- und Wochenmuster.

get_weekday_profile             Flow je Wochentag (netzweit/Station) oder
                                Tagesgang eines bestimmten Wochentags
get_busiest_station_by_weekday  stärkste Stationen an einem Wochentag,
                                optional in einem Stundenfenster
get_station_weekday_pattern     Wochenprofil einer Station plus Klassifikation
get_weekly_summary              Tagessummen einer Kalenderwoche ("last week")
get_temporal_context            Wochentag, Tageszeit-Slot, Rushhour/Wochenende
                                und Betriebsstatus zu einem Zeitpunkt
get_peak_hours                  Betriebliche Definition der Stoßzeiten
is_service_running              False in der Betriebspause 01:00–04:45

Beantwortet Fragen der Form "An welchem Wochentag ist X am vollsten?",
"Welche Station ist montags 7–9 Uhr am stärksten belastet?" und "Wie viele
Fahrgäste waren es letzte Woche?". Unvollständige Randtage (weniger als 80
Slots) bleiben durchgehend ausgeschlossen, sonst verzerren 4- bzw.
76-Slot-Tage jeden Wochentagsmittelwert.
"""

from __future__ import annotations

import datetime

import pandas as pd

from src.loader import DataLoader
from src.tools._common import (
    as_float,
    as_int,
    flow_column_for,
    full_day_index,
    resolve_station,
    validate_hours,
)

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]

WEEKDAY_DE = {
    "Monday": "Montag",
    "Tuesday": "Dienstag",
    "Wednesday": "Mittwoch",
    "Thursday": "Donnerstag",
    "Friday": "Freitag",
    "Saturday": "Samstag",
    "Sunday": "Sonntag",
}

# Akzeptierte Eingaben fuer Wochentags-Parameter.
WEEKDAY_ALIASES = {
    **{d.lower(): d for d in WEEKDAY_ORDER},
    **{de.lower(): en for en, de in WEEKDAY_DE.items()},
}

TOP_N = 5

# Tageszeitfenster fuer Morgen- und Abendspitze im Tagesgang.
MORNING = (5, 12)
EVENING = (12, 24)

DATA_LIMITATION = (
    "Mittelwerte über alle vollständigen Betriebstage des Datensatzes "
    "(siehe days_included); unvollständige Randtage (weniger als 80 Slots) "
    "sind ausgeschlossen. Flow-Werte sind simulierte "
    "Stationszählungen ohne Richtung und pro Station gedeckelt – eine "
    "OD-Matrix gibt es nicht."
)


def _frames():
    """Lädt Flows und liefert (loader, volle Tage, Tagessummen je Station)."""
    loader = DataLoader()
    flows = loader.load_flows()
    full = flows.loc[full_day_index(flows)]
    daily = full.groupby(full.index.normalize()).sum()
    return loader, full, daily


def _weekday_number(weekday: int | str | None) -> int | None:
    """0–6 aus Zahl, englischem oder deutschem Namen (auch Plural "Mondays")."""
    if weekday is None:
        return None
    if isinstance(weekday, int):
        return weekday if 0 <= weekday <= 6 else None
    key = str(weekday).strip().lower()
    name = WEEKDAY_ALIASES.get(key) or WEEKDAY_ALIASES.get(key.rstrip("s"))
    return WEEKDAY_ORDER.index(name) if name else None


def _peak_slot(full: pd.DataFrame, weekday: int, columns=None):
    """Mittlere Tageszeit-Spitze eines Wochentags (Zeit, Wert)."""
    subset = full[full.index.dayofweek == weekday]
    if subset.empty:
        return None, 0.0
    series = subset[columns] if columns is not None else subset
    per_time = series.sum(axis=1).groupby(subset.index.time).mean()
    if per_time.empty:
        return None, 0.0
    peak = per_time.idxmax()
    return peak.strftime("%H:%M"), float(per_time.loc[peak])


def _resolve_columns(loader: DataLoader, station_name: str | None):
    """Flow-Spalten für eine Station bzw. alle Spalten; (Spalten, Treffer, Fehler)."""
    if station_name is None:
        return None, None, None
    matches = resolve_station(loader.load_stations(), station_name)
    if not matches:
        return None, None, {"error": "Station not found", "input": station_name}
    col = flow_column_for(loader.col_to_station_id, matches[0]["station_id"])
    if col is None:
        return None, None, {"error": "Station not in flow data", "input": station_name}
    return [col], matches[0], None


NETWORK_COUNT_NOTE = (
    "Netzwerte sind die Summe der Stationszählungen aller Stationen – ein "
    "Fahrgast mit Umstieg wird an mehreren Stationen gezählt. Das ist keine "
    "Zahl eindeutiger Fahrgäste."
)


def get_weekday_profile(
    station_name: str | None = None,
    weekday: int | str | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
) -> dict:
    """Flow je Wochentag – netzweit oder für eine einzelne Station.

    Mit weekday (0=Montag … 6=Sonntag, oder Name) stattdessen der Tagesgang
    dieses Wochentags: Morgen-/Abendspitze, ruhigster Slot, Stundenprofil.
    Mit hour_start/hour_end zusätzlich die mittlere Summe in diesem Fenster
    ("Monday morning" = 06–10 Uhr).
    """
    window = None
    if hour_start is not None and hour_end is not None:
        window = validate_hours(hour_start, hour_end)
        if isinstance(window, dict):
            return window
    try:
        loader, full, daily = _frames()
        columns, resolved, error = _resolve_columns(loader, station_name)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}
    if error:
        return error

    columns = columns or list(daily.columns)
    scope = resolved["station_name"] if resolved else "network"

    if weekday is not None:
        num = _weekday_number(weekday)
        if num is None:
            return {"error": "Unknown weekday", "input": weekday,
                    "allowed": WEEKDAY_ORDER + list(WEEKDAY_DE.values())}
        return _day_profile(full, daily, columns, num, scope, resolved, window)

    totals = daily[columns].sum(axis=1)
    by_weekday = totals.groupby(totals.index.dayofweek).mean()

    ranking = []
    for num in sorted(by_weekday.index):
        name = WEEKDAY_ORDER[int(num)]
        peak_time, peak_value = _peak_slot(full, int(num), columns)
        ranking.append({
            "weekday": name,
            "weekday_de": WEEKDAY_DE[name],
            "weekday_num": int(num),
            "mean_daily_total": as_float(by_weekday.loc[num]),
            "peak_slot_mean": as_float(peak_value),
            "peak_slot_time": peak_time,
        })
    ranking.sort(key=lambda r: r["mean_daily_total"], reverse=True)

    weekday_mean = float(by_weekday[by_weekday.index < 5].mean())
    weekend_mean = float(by_weekday[by_weekday.index >= 5].mean())
    ratio = (weekend_mean / weekday_mean) if weekday_mean else 0.0

    result = {
        "scope": scope,
        "weekday_ranking": ranking,
        "busiest_weekday": ranking[0]["weekday"] if ranking else None,
        "quietest_weekday": ranking[-1]["weekday"] if ranking else None,
        "weekend_vs_weekday_ratio": as_float(ratio, 3),
        "days_included": int(len(totals)),
        "data_limitation": DATA_LIMITATION,
    }
    if resolved is not None:
        result["station_id"] = resolved["station_id"]
        result["lines"] = resolved["u_bahn_lines"]
    else:
        result["count_note"] = NETWORK_COUNT_NOTE
    return result


def _day_profile(full: pd.DataFrame, daily: pd.DataFrame, columns: list[str],
                 weekday: int, scope: str, resolved: dict | None,
                 window: tuple[int, int] | None = None) -> dict:
    """Tagesgang eines Wochentags, gemittelt über alle vollen Tage dieses Wochentags."""
    subset = full[full.index.dayofweek == weekday]
    if subset.empty:
        return {"error": "No data for weekday", "weekday": WEEKDAY_ORDER[weekday]}

    per_slot = subset[columns].sum(axis=1)
    by_time = per_slot.groupby(subset.index.time).mean()
    day_totals = daily.loc[daily.index.dayofweek == weekday, columns].sum(axis=1)

    def window_peak(lo: int, hi: int) -> dict | None:
        window = by_time[[t for t in by_time.index if lo <= t.hour < hi]]
        if window.empty:
            return None
        t = window.idxmax()
        return {"time": t.strftime("%H:%M"), "mean_flow": as_float(window.loc[t])}

    # Ruhigster Slot nur innerhalb des Betriebs (nicht die Betriebspause).
    operating = by_time[by_time > 0]
    quiet = operating.idxmin() if not operating.empty else None
    peak = by_time.idxmax()
    hourly = per_slot.groupby([subset.index.normalize(), subset.index.hour]).sum()
    hourly = hourly.groupby(level=1).mean()

    name = WEEKDAY_ORDER[weekday]
    result = {
        "scope": scope,
        "weekday": name,
        "weekday_de": WEEKDAY_DE[name],
        "weekday_num": weekday,
        "days_included": int(len(day_totals)),
        "mean_daily_total": as_float(day_totals.mean()),
        "peak_slot": {"time": peak.strftime("%H:%M"), "mean_flow": as_float(by_time.loc[peak])},
        "quietest_operating_slot": (
            {"time": quiet.strftime("%H:%M"), "mean_flow": as_float(operating.loc[quiet])}
            if quiet is not None else None
        ),
        "morning_peak": window_peak(*MORNING),
        "evening_peak": window_peak(*EVENING),
        "hourly_profile": {f"{int(h):02d}:00": as_float(v, 1) for h, v in hourly.items()},
        "data_limitation": DATA_LIMITATION,
    }
    if window is not None:
        lo, hi = window
        in_window = per_slot[(subset.index.hour >= lo) & (subset.index.hour < hi)]
        per_day = in_window.groupby(in_window.index.normalize()).sum()
        if not per_day.empty:
            result["window_total"] = {
                "hours": f"{lo:02d}:00–{hi:02d}:00",
                "mean_total": as_float(per_day.mean()),
                "min_day_total": as_int(per_day.min()),
                "max_day_total": as_int(per_day.max()),
                "days": int(len(per_day)),
            }
    if resolved is not None:
        result["station_id"] = resolved["station_id"]
        result["lines"] = resolved["u_bahn_lines"]
    else:
        result["count_note"] = NETWORK_COUNT_NOTE
    return result


def _lines_by_station_id(loader: DataLoader) -> dict[str, str]:
    """station_id -> U-Bahn-Linien laut Stammdaten ("U2,U5,U8")."""
    stations = loader.load_stations()
    return dict(zip(stations["station_id"], stations["u_bahn_lines"]))


def get_busiest_station_by_weekday(
    weekday: int | str | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
    top_n: int = TOP_N,
) -> dict:
    """Die am stärksten belasteten Stationen an einem Wochentag.

    weekday=None wertet alle Tage aus ("busiest station in the network").
    Mit hour_start/hour_end (0–24, Ende exklusiv) nur in diesem Fenster:
    Summe im Fenster je Tag, gemittelt über alle Tage dieses Wochentags.
    """
    num = None
    if weekday is not None:
        num = _weekday_number(weekday)
        if num is None:
            return {
                "error": "Unknown weekday",
                "input": weekday,
                "allowed": WEEKDAY_ORDER + list(WEEKDAY_DE.values()),
            }
    target = WEEKDAY_ORDER[num] if num is not None else "all days"

    windowed = hour_start is not None and hour_end is not None
    if windowed:
        hours = validate_hours(hour_start, hour_end)
        if isinstance(hours, dict):
            return hours
        hour_start, hour_end = hours
    try:
        top_n = max(1, min(int(top_n), 50))
    except (TypeError, ValueError):
        top_n = TOP_N

    try:
        loader, full, daily = _frames()
        lines = _lines_by_station_id(loader)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    subset = full if num is None else full[full.index.dayofweek == num]
    if subset.empty:
        return {"error": "No data for weekday", "weekday": target}

    if windowed:
        subset = subset[(subset.index.hour >= hour_start) & (subset.index.hour < hour_end)]
        window_str = f"{hour_start:02d}:00–{hour_end:02d}:00"
    else:
        window_str = "full day"

    per_day = subset.groupby(subset.index.normalize()).sum()
    means = per_day.mean().sort_values(ascending=False)

    top = []
    for rank, (col, value) in enumerate(means.head(top_n).items(), start=1):
        per_time = subset[col].groupby(subset.index.time).mean()
        peak = per_time.idxmax()
        sid = loader.col_to_station_id.get(col)
        top.append({
            "rank": rank,
            "station_name": col.removesuffix(".1"),
            "station_id": sid,
            "lines": lines.get(sid),
            "mean_daily_total" if not windowed else "mean_window_total": as_float(value),
            "peak_time": peak.strftime("%H:%M"),
            "peak_flow": as_float(per_time.loc[peak]),
        })

    return {
        "weekday": target,
        "weekday_de": WEEKDAY_DE.get(target, "alle Tage"),
        "time_window": window_str,
        "top_5_stations": top,
        # Mittel je Station, nicht Netzsumme – der alte Name
        # "network_mean_that_day" wurde als Netzwert gelesen.
        "mean_per_station": as_float(float(means.mean())),
        "network_total_all_stations": as_float(float(means.sum())),
        "days_included": int(len(per_day)),
        "data_limitation": DATA_LIMITATION,
    }


def find_stations_above_threshold(
    threshold: float,
    per: str = "hour",
    weekday: int | str | None = None,
) -> dict:
    """Stationen, deren mittlere Belastung einen Schwellwert überschreitet.

    per="hour": Stundensummen (4 Slots), per="slot": 15-Minuten-Werte.
    Gemittelt wird je Tageszeit über alle vollen Tage (bzw. einen Wochentag);
    eine Station zählt, wenn mindestens eine mittlere Stunde/Slot darüber liegt.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return {"error": "Invalid threshold", "input": threshold}
    if per not in ("hour", "slot"):
        return {"error": "Invalid unit", "input": per, "allowed": ["hour", "slot"]}
    num = None
    if weekday is not None:
        num = _weekday_number(weekday)
        if num is None:
            return {"error": "Unknown weekday", "input": weekday,
                    "allowed": WEEKDAY_ORDER + list(WEEKDAY_DE.values())}

    try:
        loader, full, _daily = _frames()
        lines = _lines_by_station_id(loader)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    subset = full if num is None else full[full.index.dayofweek == num]
    if subset.empty:
        return {"error": "No data for weekday", "weekday": weekday}

    if per == "hour":
        per_unit = subset.groupby([subset.index.normalize(), subset.index.hour]).sum()
        profile = per_unit.groupby(level=1).mean()
        label = lambda h: f"{int(h):02d}:00"  # noqa: E731
    else:
        profile = subset.groupby(subset.index.time).mean()
        label = lambda t: t.strftime("%H:%M")  # noqa: E731

    above = []
    for col in profile.columns:
        series = profile[col]
        hits = series[series > threshold]
        if hits.empty:
            continue
        peak = series.idxmax()
        sid = loader.col_to_station_id.get(col)
        above.append({
            "station_name": col.removesuffix(".1"),
            "station_id": sid,
            "lines": lines.get(sid),
            "peak_time": label(peak),
            f"peak_mean_per_{per}": as_float(series.loc[peak]),
            f"{per}s_above_threshold": int(len(hits)),
            "first_time_above": label(hits.index.min()),
            "last_time_above": label(hits.index.max()),
        })
    above.sort(key=lambda s: s[f"peak_mean_per_{per}"], reverse=True)

    return {
        "threshold": threshold,
        "unit": f"passengers per {per} (station count, mean over days)",
        "day_selection": WEEKDAY_ORDER[num] if num is not None else "all full days",
        "count": len(above),
        "total_stations": int(len(profile.columns)),
        "stations": above,
        "days_included": int(subset.index.normalize().nunique()),
        "data_limitation": (
            "Mittelwert je Tageszeit über alle einbezogenen Tage – einzelne Tage "
            "können darüber oder darunter liegen. " + DATA_LIMITATION
        ),
    }


def get_station_weekday_pattern(station_name: str) -> dict:
    """Wochenprofil einer Station plus Einordnung Pendler/Freizeit/gemischt."""
    try:
        loader, full, daily = _frames()
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

    series = daily[col]
    by_weekday = series.groupby(series.index.dayofweek).mean()
    totals = {
        WEEKDAY_ORDER[int(num)]: as_float(value)
        for num, value in by_weekday.items()
    }
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

    per_time = full[col].groupby(full.index.time).mean()
    busiest_time = per_time.idxmax().strftime("%H:%M")

    weekday_mean = float(by_weekday[by_weekday.index < 5].mean())
    weekend_mean = float(by_weekday[by_weekday.index >= 5].mean())
    ratio = (weekend_mean / weekday_mean) if weekday_mean else 0.0

    # Schwellen: unter 0.8 klar werktagsgetrieben, ab 1.0 mindestens
    # gleichauf am Wochenende, dazwischen kein eindeutiges Muster.
    if ratio <= 0.8:
        pattern = "commuter"
    elif ratio >= 1.0:
        pattern = "leisure"
    else:
        pattern = "mixed"

    result = {
        "station_name": match["station_name"],
        "station_id": match["station_id"],
        "lines": match["u_bahn_lines"],
        "weekday_totals": totals,
        "busiest_weekday": ordered[0][0],
        "quietest_weekday": ordered[-1][0],
        "busiest_time_overall": busiest_time,
        "pattern": pattern,
        "weekend_vs_weekday_ratio": as_float(ratio, 3),
        "weekday_mean": as_float(weekday_mean),
        "weekend_mean": as_float(weekend_mean),
        "days_included": int(len(series)),
        "pattern_rule": (
            "commuter: Wochenende/Werktag <= 0.8 – leisure: >= 1.0 – "
            "mixed: dazwischen."
        ),
        "data_limitation": DATA_LIMITATION,
    }
    if len(matches) > 1:
        result["ambiguity"] = {
            "note": ("Stationsname ist im Datensatz nicht eindeutig. "
                     "Ausgewertet wurde die erste Bahnsteigebene."),
            "alternatives": matches[1:],
        }
    return result


def get_weekly_summary(week_offset: int = 0, current_week: bool = False) -> dict:
    """Tagessummen des Netzes für eine Kalenderwoche (Montag–Sonntag).

    week_offset=0     letzte vollständige Woche bis zum letzten Datentag
                      ("last week"); endet der Datensatz an einem Sonntag,
                      ist das genau diese Woche
    week_offset=1     die Woche davor, usw.
    current_week=True laufende Woche bis zum letzten Datentag ("this week")
    """
    try:
        loader = DataLoader()
        flows = loader.load_flows()
        last_day = datetime.date.fromisoformat(loader.data_last_day)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    if current_week:
        week_start = last_day - datetime.timedelta(days=last_day.weekday())
        week_end = last_day
    else:
        # Letzter Sonntag bis einschliesslich last_day.
        last_sunday = last_day - datetime.timedelta(days=(last_day.weekday() + 1) % 7)
        week_end = last_sunday - datetime.timedelta(days=7 * int(week_offset))
        week_start = week_end - datetime.timedelta(days=6)

    full = flows.loc[full_day_index(flows)]
    days = full.index.normalize()
    mask = (days >= pd.Timestamp(week_start)) & (days <= pd.Timestamp(week_end))
    week = full[mask]
    if week.empty:
        return {"error": "No data for week", "week": f"{week_start} to {week_end}"}

    per_day = week.sum(axis=1).groupby(week.index.normalize()).sum()
    daily_totals = {
        d.strftime("%Y-%m-%d"): {"weekday": WEEKDAY_ORDER[d.dayofweek],
                                 "total_pax": as_int(v)}
        for d, v in per_day.items()
    }
    peak_day = per_day.idxmax()
    low_day = per_day.idxmin()
    expected_days = (week_end - week_start).days + 1

    return {
        "week": f"{week_start} to {week_end}",
        "week_type": "current (partial) week" if current_week else "last complete week",
        "days_with_data": int(len(per_day)),
        "days_expected": expected_days,
        "total_weekly_pax": as_int(per_day.sum()),
        "mean_daily_pax": as_float(per_day.mean()),
        "peak_day": {"date": peak_day.strftime("%Y-%m-%d"),
                     "weekday": WEEKDAY_ORDER[peak_day.dayofweek],
                     "pax": as_int(per_day.loc[peak_day])},
        "lowest_day": {"date": low_day.strftime("%Y-%m-%d"),
                       "weekday": WEEKDAY_ORDER[low_day.dayofweek],
                       "pax": as_int(per_day.loc[low_day])},
        "daily_totals": daily_totals,
        "data_limitation": (
            "Summe der Stationszählungen aller 168 Stationen je Tag – ein "
            "Fahrgast mit Umstieg wird an mehreren Stationen gezählt, das ist "
            "keine Zahl eindeutiger Fahrgäste. Nur vollständige Betriebstage. "
            + DATA_LIMITATION
        ),
    }


# --------------------------------------------------------------------- #
# Zeitliche Einordnung eines Zeitpunkts (regelbasiert, ohne Flow-Daten)
# --------------------------------------------------------------------- #

# Betriebliche Stoßzeiten, Mo–Fr, Ende exklusiv.
PEAK_WINDOWS = {
    "morning_peak": (datetime.time(7, 0), datetime.time(9, 0)),
    "evening_peak": (datetime.time(16, 0), datetime.time(19, 0)),
}
# Betriebspause laut Datensatz (CLAUDE.md, Punkt 6): keine Datenlücke.
SERVICE_PAUSE = (datetime.time(1, 0), datetime.time(4, 45))
# "night" = Spätverkehr ab 22:00 plus Betriebspause bis Betriebsbeginn.
NIGHT_START = datetime.time(22, 0)

TEMPORAL_RULE_NOTE = (
    "Regelbasierte Einordnung: Stoßzeiten Mo–Fr 07:00–09:00 und 16:00–19:00, "
    "Betriebspause täglich 01:00–04:45 (wie im Datensatz). Die gemessenen "
    "Spitzen einer Station liefert get_peak_profile."
)


def _to_timestamp(timestamp) -> pd.Timestamp | None:
    """Akzeptiert datetime, pd.Timestamp oder String ("2026-09-21 08:15")."""
    try:
        ts = pd.Timestamp(timestamp)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts


def is_service_running(timestamp) -> bool:
    """False in der Betriebspause 01:00–04:45, sonst True.

    Nicht parsebare Zeitpunkte gelten als nicht im Betrieb – lieber keine
    Aussage als eine falsche.
    """
    ts = _to_timestamp(timestamp)
    if ts is None:
        return False
    start, end = SERVICE_PAUSE
    return not start <= ts.time() < end


def _measured_weekday_peaks() -> dict | None:
    """Gemessener Werktags-Tagesgang des Netzes als Beleg für die Definition.

    Ohne diesen Block stellte das LLM die Regel 07–09/16–19 als Messwert dar.
    """
    try:
        _loader, full, _daily = _frames()
    except (FileNotFoundError, ValueError):
        return None
    weekdays = full[full.index.dayofweek < 5]
    if weekdays.empty:
        return None
    per_hour = weekdays.sum(axis=1).groupby(
        [weekdays.index.normalize(), weekdays.index.hour]).sum()
    hourly = per_hour.groupby(level=1).mean()
    in_peak = hourly[[h for h in hourly.index if 7 <= h < 9 or 16 <= h < 19]].sum()
    top = hourly.sort_values(ascending=False).head(5)
    return {
        "busiest_hours_mon_fri": [
            {"hour": f"{int(h):02d}:00", "mean_network_count": as_int(v)} for h, v in top.items()
        ],
        "share_of_daily_count_in_peak_windows_pct": as_float(in_peak / hourly.sum() * 100.0, 1),
        "peak_window_hours_share_of_operating_hours_pct": as_float(
            5 / int((hourly > 0).sum()) * 100.0, 1),
        "weekdays_included": int(weekdays.index.normalize().nunique()),
        "count_note": NETWORK_COUNT_NOTE,
    }


def get_peak_hours() -> dict:
    """Definition der Stoßzeiten (Mo–Fr 07:00–09:00 und 16:00–19:00).

    Enthält zusätzlich den gemessenen Werktags-Tagesgang (measured), damit die
    Regel mit Daten belegt oder widerlegt werden kann.
    """
    return {
        "definition_source": "operational rule (not derived from the data)",
        "measured": _measured_weekday_peaks(),
        "peak_days": WEEKDAY_ORDER[:5],
        "peak_days_de": [WEEKDAY_DE[d] for d in WEEKDAY_ORDER[:5]],
        "morning_peak": {"start": "07:00", "end": "09:00"},
        "evening_peak": {"start": "16:00", "end": "19:00"},
        "weekend": "keine Stoßzeiten (Sa/So gelten ganztägig als off_peak)",
        "service_pause": {"start": "01:00", "end": "04:45"},
        "slots": {
            "morning_peak": "Mo–Fr 07:00–09:00",
            "evening_peak": "Mo–Fr 16:00–19:00",
            "night": "22:00–04:45 (inkl. Betriebspause 01:00–04:45)",
            "off_peak": "alle übrigen Zeiten, am Wochenende auch 07–09/16–19 Uhr",
        },
        "note": TEMPORAL_RULE_NOTE,
    }


def get_temporal_context(timestamp) -> dict:
    """Wochentag, Tageszeit-Slot, Rushhour und Wochenende zu einem Zeitpunkt.

    Slots: morning_peak / evening_peak (nur Mo–Fr), night (22:00–04:45),
    sonst off_peak.
    """
    ts = _to_timestamp(timestamp)
    if ts is None:
        return {"error": "Invalid timestamp", "input": str(timestamp),
                "expected": "ISO-Format, z. B. 2026-09-21 08:15"}

    weekday = WEEKDAY_ORDER[ts.dayofweek]
    is_weekend = ts.dayofweek >= 5
    t = ts.time()

    slot = "off_peak"
    if t >= NIGHT_START or t < SERVICE_PAUSE[1]:
        slot = "night"
    elif not is_weekend:
        for name, (start, end) in PEAK_WINDOWS.items():
            if start <= t < end:
                slot = name
                break

    return {
        "timestamp": ts.isoformat(),
        "weekday": weekday,
        "weekday_de": WEEKDAY_DE[weekday],
        "time_slot": slot,
        "ist_rushhour": slot in PEAK_WINDOWS,
        "ist_wochenende": bool(is_weekend),
        "service_running": is_service_running(ts),
        "note": TEMPORAL_RULE_NOTE,
    }
