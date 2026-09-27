"""Tool 1 – Fahrgastströme.

get_station_flow              Flow einer Station für Tag/Zeitfenster
get_station_peak_15min        exakter 15-Min-Peak + Anteil am Stations-Allzeitmaximum
get_station_peak_vs_baseline  Peak-Slot gegen trockenen Wochentags-Median
get_network_flow_summary      netzweite Zusammenfassung
detect_anomalies              Wochentag-Baseline-Abweichungen

Alle Funktionen sind pure Funktionen und geben dicts zurück. Fehler kommen
als {"error": ...} zurück, nie als Exception.
"""

from __future__ import annotations

import pandas as pd

from src.loader import DataLoader
from src.tools._common import (
    DATE_OUT,
    TS_OUT,
    as_float,
    as_int,
    extract_lines,
    extract_stations,
    flow_column_for,
    full_day_index,
    hour_window,
    normalize_station,
    parse_date,
    resolve_station,
    slice_window,
    validate_hours,
)

# Ein Event wirkt vor Beginn (Anreise) und nach Ende (Abreise) auf die
# Station. Nur Events in Fusswegnaehe zaehlen als Confounder.
EVENT_LEAD = pd.Timedelta(minutes=90)
EVENT_TAIL = pd.Timedelta(minutes=60)
CONFOUNDER_EVENT_M = 1500

# estimated_attendance ist skaliert (165–2394) – nur relative Stufen.
ATTENDANCE_HIGH = 1500
ATTENDANCE_MEDIUM = 800


def get_station_flow(
    station_name: str,
    date_str: str,
    hour_from: int = 0,
    hour_to: int = 24,
    compare_baseline: bool = False,
) -> dict:
    """Fahrgastfluss einer Station für einen Tag bzw. ein Stundenfenster.

    compare_baseline=True ergänzt den Normalfall: dieselben Slots an allen
    anderen vollen Tagen desselben Wochentags (Mittel je Slot). Erst damit
    lässt sich sagen, ob ein Event- oder Wettertag ungewöhnlich war.

    is_closed ist True, wenn eine Sperrung das angefragte Fenster überlappt
    UND alle Slots innerhalb dieser Überlappung exakt 0 sind. Bewusst nicht
    "alle Slots im Fenster == 0": eine Sperrung beginnt selten auf einer
    vollen Stunde, ein Fenster 06:00-09:00 enthält bei einer Sperrung ab
    06:15 einen regulären Slot um 06:00.
    """
    hours = validate_hours(hour_from, hour_to)
    if isinstance(hours, dict):
        return hours
    hour_from, hour_to = hours

    loader = DataLoader()
    try:
        stations = loader.load_stations()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    matches = resolve_station(stations, station_name)
    if not matches:
        return {"error": "Station not found", "input": station_name}

    day = parse_date(date_str)
    if day is None:
        return {"error": "Invalid date", "input": date_str}

    try:
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}
    match = matches[0]
    col = flow_column_for(loader.col_to_station_id, match["station_id"])
    if col is None:
        return {"error": "Station not in flow data", "input": station_name}

    window = slice_window(flows[[col]], day, hour_from, hour_to)
    if window.empty:
        return {"error": "No data for date", "date": date_str}

    series = window[col]
    slots = [
        {"timestamp": ts.strftime(TS_OUT), "flow": as_int(v)}
        for ts, v in series.items()
    ]
    peak_ts = series.idxmax()

    win_start, win_end = hour_window(day, hour_from, hour_to)
    closed, closure_info = _closure_state(
        loader, match["station_name"], series, win_start, win_end
    )

    result = {
        "station_id": match["station_id"],
        "station_name": match["station_name"],
        "date": day.strftime(DATE_OUT),
        "hour_from": int(hour_from),
        "hour_to": int(hour_to),
        "slots": slots,
        "total": as_int(series.sum()),
        "mean_per_slot": as_float(series.mean()),
        "peak_slot": {
            "timestamp": peak_ts.strftime(TS_OUT),
            "flow": as_int(series.loc[peak_ts]),
        },
        "is_closed": closed,
        "data_points": int(len(series)),
        "lines": match["u_bahn_lines"],
        "zero_slots": int((series == 0).sum()),
        # Rohwerte sind bei 3000 gecappt (Simulationsartefakt): markiert nur
        # nach oben zensierte Slots, ist keine Kapazitaetsgrenze.
        "capped_slots": int((series >= 3000).sum()),
        "matching_closure": closure_info,
    }

    if compare_baseline:
        result["baseline"] = _baseline(flows[col], day, hour_from, hour_to, series)

    if len(matches) > 1:
        result["ambiguity"] = {
            "note": (
                "Station name is ambiguous in the dataset. The first "
                "platform level was analysed."
            ),
            "alternatives": matches[1:],
        }
    return result


def _baseline(full_series: pd.Series, day: pd.Timestamp, hour_from: int,
              hour_to: int, observed: pd.Series) -> dict | None:
    """Gleiche Slots an allen anderen vollen Tagen desselben Wochentags."""
    frame = full_series.to_frame()
    full = full_series.loc[full_day_index(frame)]
    same = full[(full.index.dayofweek == day.dayofweek)
                & (full.index.normalize() != day.normalize())
                & (full.index.hour >= hour_from) & (full.index.hour < hour_to)]
    if same.empty:
        return None
    per_slot = same.groupby(same.index.time).mean()
    base_total = float(per_slot.sum())
    obs_peak_time = observed.idxmax().time()
    base_at_peak = float(per_slot.get(obs_peak_time, 0.0))
    return {
        "method": f"mean of the same slots on the other {day.day_name()}s",
        "days": int(same.index.normalize().nunique()),
        "total": as_float(base_total),
        "change_pct": as_float((observed.sum() - base_total) / base_total * 100.0)
        if base_total else None,
        "at_observed_peak_slot": as_float(base_at_peak),
        "peak_ratio": as_float(float(observed.max()) / base_at_peak) if base_at_peak else None,
    }


def _closure_state(
    loader: DataLoader,
    station_name: str,
    series: pd.Series,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
) -> tuple[bool, dict | None]:
    """Prüft, ob eine Sperrung das Fenster erklärt (Nullwerte im Überlapp)."""
    try:
        closures = loader.load_closures()
    except (FileNotFoundError, ValueError):
        return False, None

    key = normalize_station(station_name)
    for row in closures.itertuples():
        named = [normalize_station(s) for s in extract_stations(row.description)]
        if key not in named:
            continue
        overlap_start = max(win_start, row.when)
        overlap_end = min(win_end, row.end_time)
        if overlap_start >= overlap_end:
            continue
        inside = series[(series.index >= overlap_start) & (series.index < overlap_end)]
        if len(inside) and (inside == 0).all():
            return True, {
                "when": row.when.strftime(TS_OUT),
                "end_time": row.end_time.strftime(TS_OUT),
                "description": row.description,
                "zero_slots_in_overlap": int(len(inside)),
            }
    return False, None


def _station_series(loader: DataLoader, station_name: str) -> tuple[dict, pd.Series] | dict:
    """Station aufloesen und ihre komplette Flow-Zeitreihe liefern."""
    try:
        stations = loader.load_stations()
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}
    matches = resolve_station(stations, station_name)
    if not matches:
        return {"error": "Station not found", "input": station_name}
    col = flow_column_for(loader.col_to_station_id, matches[0]["station_id"])
    if col is None:
        return {"error": "Station not in flow data", "input": station_name}
    return matches[0], flows[col]


def _station_historical_max(series: pd.Series) -> int:
    """Hoechster je gemessene 15-Min-Wert der Station (alle Tage, alle Stunden)."""
    peak = series.max()
    return 0 if pd.isna(peak) else as_int(peak)


def get_station_peak_15min(
    station_name: str,
    date: str | None = None,
    hour_from: int = 0,
    hour_to: int = 24,
) -> dict:
    """Exakter 15-Minuten-Peak einer Station – kein Stunden- oder 3h-Aggregat.

    Mit date: Peak dieses Tages im Stundenfenster. Ohne date: hoechster Slot
    im Stundenfenster ueber den gesamten Datensatz (erster Treffer bei
    Gleichstand). Referenz fuer die Ampel ist das Allzeit-Maximum der Station
    im Datensatz – der 3.000er-Cap der Simulation ist keine Kapazitaet.
    """
    hours = validate_hours(hour_from, hour_to)
    if isinstance(hours, dict):
        return hours
    hour_from, hour_to = hours

    loader = DataLoader()
    resolved = _station_series(loader, station_name)
    if isinstance(resolved, dict):
        return resolved
    match, series = resolved

    if date is not None:
        day = parse_date(date)
        if day is None:
            return {"error": "Invalid date", "input": date}
        window = slice_window(series.to_frame(), day, hour_from, hour_to)[series.name]
    else:
        window = series[(series.index.hour >= hour_from) & (series.index.hour < hour_to)]
    if window.empty:
        return {"error": "No data for date", "date": date}

    peak_ts = window.idxmax()
    peak_value = as_int(window.loc[peak_ts])
    station_max = _station_historical_max(series)
    pct_of_own_max = round(peak_value / station_max * 100.0, 1) if station_max else 0.0
    if pct_of_own_max >= 90.0:
        status = "CRITICAL"
    elif pct_of_own_max >= 70.0:
        status = "ELEVATED"
    else:
        status = "NORMAL"
    return {
        "station_name": match["station_name"],
        "lines": match["u_bahn_lines"],
        "date": peak_ts.strftime(DATE_OUT),
        "scope": "single day" if date is not None else "all days in dataset",
        "hour_from": int(hour_from),
        "hour_to": int(hour_to),
        "peak_time": peak_ts.strftime("%H:%M"),
        "peak_timestamp": peak_ts.strftime(TS_OUT),
        "peak_value": peak_value,
        "station_max_ever": station_max,
        "pct_of_own_max": pct_of_own_max,
        "capacity_status": status,
        "capacity_note": (
            f"{pct_of_own_max:.1f}% of station historical maximum ({station_max:,} pax)"
        ),
        "slots_at_own_max": int((window >= station_max).sum()) if station_max else 0,
        "data_points": int(len(window)),
    }


def _attendance_level(value) -> str:
    if pd.isna(value):
        return "unknown"
    if value >= ATTENDANCE_HIGH:
        return "high"
    if value >= ATTENDANCE_MEDIUM:
        return "medium"
    return "low"


def _event_confounders(date_str: str, station_name: str, peak_ts: pd.Timestamp) -> list[str]:
    """Events in Fusswegnaehe der Station, deren An-/Abreise den Peak trifft."""
    from src.tools.event_tool import get_events  # lokal: haelt den Importgraph flach

    result = get_events(date_str=date_str)
    if "error" in result:
        return []
    key = normalize_station(station_name)
    found = []
    for e in result.get("events") or []:
        near = [s for s in (e.get("nearby_stations") or [])
                if s.get("distance_m", CONFOUNDER_EVENT_M + 1) <= CONFOUNDER_EVENT_M]
        if e.get("nearest_station") and (e.get("distance_m") or 0) <= CONFOUNDER_EVENT_M:
            near.append({"station_name": e["nearest_station"]})
        if key not in {normalize_station(s["station_name"]) for s in near}:
            continue
        begin = pd.Timestamp(e["began_local"])
        end = pd.Timestamp(e["end_local"]) if e.get("end_local") else begin + pd.Timedelta(hours=3)
        if not (begin - EVENT_LEAD <= peak_ts <= end + EVENT_TAIL):
            continue
        venue = e.get("venue") or e.get("address") or "?"
        found.append(
            f"{e['event_name']} ({venue}) {begin.strftime('%H:%M')} "
            f"(Attendance: {_attendance_level(e.get('estimated_attendance'))})"
        )
    return found


def _closure_confounders(loader: DataLoader, match: dict, peak_ts: pd.Timestamp) -> list[str]:
    """Sperrungen zum Peak-Zeitpunkt an der Station oder auf ihren Linien."""
    try:
        closures = loader.load_closures()
    except (FileNotFoundError, ValueError):
        return []
    key = normalize_station(match["station_name"])
    own_lines = {x.strip() for x in str(match["u_bahn_lines"]).split(",") if x.strip()}
    found = []
    for row in closures.itertuples():
        if not (row.when <= peak_ts < row.end_time):
            continue
        named = {normalize_station(s) for s in extract_stations(row.description)}
        if key in named or own_lines & set(extract_lines(row.description)):
            found.append(
                f"Closure {row.when.strftime('%H:%M')}–{row.end_time.strftime('%H:%M')}: "
                f"{row.description}"
            )
    return found


def get_station_peak_vs_baseline(
    station_name: str,
    date: str,
    hour_from: int = 0,
    hour_to: int = 24,
) -> dict:
    """Peak-Slot eines Tages gegen den trockenen Normalfall.

    Baseline: Median desselben 15-Min-Slots (Uhrzeit des Peaks) an allen
    anderen vollen Tagen desselben Wochentags, an denen in dieser Stunde kein
    Niederschlag fiel. Gibt es keinen trockenen Vergleichstag, gilt der Median
    aller Vergleichstage (baseline_dry_only=False).

    confounders_active: zum Peak-Zeitpunkt lief ein Event in Fusswegnaehe
    (An-/Abreisefenster) oder eine Sperrung an der Station bzw. auf einer
    ihrer Linien. Dann ist die Abweichung nicht allein Wetter/Nachfrage.
    """
    peak = get_station_peak_15min(station_name, date, hour_from, hour_to)
    if "error" in peak:
        return peak

    loader = DataLoader()
    resolved = _station_series(loader, station_name)
    if isinstance(resolved, dict):
        return resolved
    match, series = resolved

    peak_ts = pd.Timestamp(peak["peak_timestamp"])
    day = peak_ts.normalize()
    full = series.loc[full_day_index(series.to_frame())]
    same = full[(full.index.dayofweek == day.dayofweek)
                & (full.index.normalize() != day)
                & (full.index.time == peak_ts.time())]

    try:
        prcp = loader.load_weather()["prcp"]
    except (FileNotFoundError, ValueError, KeyError):
        prcp = pd.Series(dtype=float)
    hourly_rain = prcp.groupby(prcp.index.floor("h")).sum() if not prcp.empty else prcp
    peak_hour = peak_ts.floor("h")

    dry_only = False
    if not hourly_rain.empty:
        rain_then = same.index.floor("h").map(lambda h: hourly_rain.get(h))
        dry = same[[r is not None and not pd.isna(r) and r == 0 for r in rain_then]]
        if not dry.empty:
            same, dry_only = dry, True
    if same.empty:
        return {"error": "Insufficient baseline", "station_name": match["station_name"],
                "detail": f"No comparison days for {day.day_name()} {peak['peak_time']}."}

    baseline = float(same.median())
    increase = ((peak["peak_value"] - baseline) / baseline * 100.0) if baseline else None
    rain_at_peak = hourly_rain.get(peak_hour) if not hourly_rain.empty else None

    confounders = (_event_confounders(day.strftime(DATE_OUT), match["station_name"], peak_ts)
                   + _closure_confounders(loader, match, peak_ts))
    if confounders:
        detail = "; ".join(confounders)
    else:
        detail = "No events/closures active"
        if rain_at_peak is not None and not pd.isna(rain_at_peak) and rain_at_peak > 0:
            detail += f" → likely cause: rain ({as_float(rain_at_peak, 1)} mm in the peak hour)"

    return {
        "station_name": match["station_name"],
        "date": day.strftime(DATE_OUT),
        "weekday": day.day_name(),
        "peak_time": peak["peak_time"],
        "peak_value": peak["peak_value"],
        "station_max_ever": peak["station_max_ever"],
        "pct_of_own_max": peak["pct_of_own_max"],
        "capacity_status": peak["capacity_status"],
        "capacity_note": peak["capacity_note"],
        "baseline_value": as_int(baseline),
        "baseline_days": int(same.index.normalize().nunique()),
        "baseline_dry_only": dry_only,
        "baseline_method": (
            f"Median of the {peak['peak_time']} slot on the other {day.day_name()}s"
            + (" without rain in that hour" if dry_only else " (all weather)")
        ),
        "increase_pct": as_float(increase, 1) if increase is not None else None,
        "rain_at_peak_mm": as_float(rain_at_peak, 1)
        if rain_at_peak is not None and not pd.isna(rain_at_peak) else None,
        "confounders_active": bool(confounders),
        "confounder_detail": detail,
    }


def get_network_flow_summary(
    date_str: str,
    hour_from: int = 0,
    hour_to: int = 24,
) -> dict:
    """Netzweite Flow-Zusammenfassung für einen Tag bzw. ein Stundenfenster."""
    hours = validate_hours(hour_from, hour_to)
    if isinstance(hours, dict):
        return hours
    hour_from, hour_to = hours

    loader = DataLoader()
    day = parse_date(date_str)
    if day is None:
        return {"error": "Invalid date", "input": date_str}

    try:
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    window = slice_window(flows, day, hour_from, hour_to)
    if window.empty:
        return {"error": "No data for date", "date": date_str}

    per_slot = window.sum(axis=1)
    per_station = window.sum(axis=0).sort_values(ascending=False)
    peak_ts = per_slot.idxmax()

    return {
        "date": day.strftime(DATE_OUT),
        "hour_from": int(hour_from),
        "hour_to": int(hour_to),
        "total_network_flow": as_int(per_slot.sum()),
        "mean_per_slot": as_float(per_slot.mean()),
        "peak_slot": {
            "timestamp": peak_ts.strftime(TS_OUT),
            "total_flow": as_int(per_slot.loc[peak_ts]),
        },
        "top_5_stations": [
            {"station_name": name, "total": as_int(total)}
            for name, total in per_station.head(5).items()
        ],
        "bottom_5_stations": [
            {"station_name": name, "total": as_int(total)}
            for name, total in per_station.tail(5).items()
        ],
        "data_points": int(len(per_slot)),
        "stations_at_zero": int((per_station == 0).sum()),
    }


def detect_anomalies(date_str: str, z_threshold: float = 2.0) -> dict:
    """Stationen, die am Stichtag stark vom Wochentagsmittel abweichen.

    Baseline: Tagessummen aller gleichen Wochentage im Datensatz, ohne den
    Stichtag selbst und ohne unvollständige Randtage.
    """
    loader = DataLoader()
    day = parse_date(date_str)
    if day is None:
        return {"error": "Invalid date", "input": date_str}

    try:
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    full = flows.loc[full_day_index(flows)]
    daily = full.groupby(full.index.normalize()).sum()

    target = day.normalize()
    if target not in daily.index:
        return {
            "error": "No data for date",
            "date": date_str,
            "detail": "Day missing or incomplete (edge day of the dataset).",
        }

    weekday = target.day_name()
    same_weekday = daily[(daily.index.day_name() == weekday) & (daily.index != target)]
    if len(same_weekday) < 2:
        return {
            "error": "Insufficient baseline",
            "date": date_str,
            "detail": f"Only {len(same_weekday)} comparison days for {weekday}.",
        }

    observed = daily.loc[target]
    mean = same_weekday.mean()
    std = same_weekday.std(ddof=1)

    anomalies = []
    for station in daily.columns:
        sigma = float(std[station])
        if sigma <= 0:
            continue
        z = (float(observed[station]) - float(mean[station])) / sigma
        if abs(z) >= float(z_threshold):
            anomalies.append(
                {
                    "station_name": station,
                    "observed_total": as_int(observed[station]),
                    "baseline_mean": as_float(mean[station]),
                    "z_score": as_float(z),
                    "direction": "above" if z > 0 else "below",
                }
            )
    anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)

    return {
        "date": target.strftime(DATE_OUT),
        "weekday": weekday,
        "z_threshold": float(z_threshold),
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "baseline_days": int(len(same_weekday)),
        "method": (
            "Daily total per station vs. mean/std of all other "
            f"{weekday}s; incomplete edge days excluded."
        ),
    }
