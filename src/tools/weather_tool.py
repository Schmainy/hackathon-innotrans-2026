"""Tool 4 – Wetter.

get_weather             Tages-/Fensterauswertung inkl. lesbarer Bedingung
find_weather_anomalies  stärkster Ausreisser einer Kennzahl in einer Woche

coco und cldc sind vom Stunden- auf das 15-Minuten-Raster interpoliert und
enthalten deshalb Nachkommawerte. Vor jeder kategorialen Auswertung wird
gerundet.
"""

from __future__ import annotations

import pandas as pd

from src.loader import DataLoader
from src.tools._common import (
    DATE_OUT,
    TS_OUT,
    as_float,
    full_day_index,
    parse_date,
    slice_window,
    validate_hours,
)

# Meteostat-Wettercodes (coco). Achtung: 9 ist Starkregen, nicht Gewitter,
# und 17 ist ein Regenschauer, kein Hagel – Gewitter sind 25/26.
COCO_LABELS = {
    1:  "Clear sky",
    2:  "Partly cloudy",
    3:  "Cloudy",
    4:  "Overcast",
    5:  "Fog",
    6:  "Freezing fog",
    7:  "Light rain",
    8:  "Rain",
    9:  "Heavy rain",
    10: "Freezing rain",
    11: "Heavy freezing rain",
    12: "Sleet",
    13: "Heavy sleet",
    14: "Light snowfall",
    15: "Snowfall",
    16: "Heavy snowfall",
    17: "Rain shower",
    18: "Heavy rain shower",
    19: "Sleet shower",
    20: "Heavy sleet shower",
    21: "Snow shower",
    22: "Heavy snow shower",
    23: "Lightning",
    24: "Hail",
    25: "Thunderstorm",
    26: "Heavy thunderstorm",
    27: "Storm",
}

METRICS = {"prcp", "temp", "wspd"}

# 15-Minuten-Slots je Stunde: prcp-Werte sind Stundenraten (siehe get_weather).
SLOTS_PER_HOUR = 4

METRIC_UNITS = {"prcp": "mm/h", "temp": "°C", "wspd": "km/h"}


def get_condition_label(coco_value) -> str:
    """Wandelt einen (interpolierten) coco-Wert in eine lesbare Bedingung."""
    code = int(round(float(coco_value)))
    return COCO_LABELS.get(code, f"Unknown (code {code})")


def get_weather(date_str: str, hour_from: int = 0, hour_to: int = 24) -> dict:
    """Wetterdaten für einen Tag bzw. ein Stundenfenster."""
    hours = validate_hours(hour_from, hour_to)
    if isinstance(hours, dict):
        return hours
    hour_from, hour_to = hours

    loader = DataLoader()
    day = parse_date(date_str)
    if day is None:
        return {"error": "Invalid date", "input": date_str}

    try:
        weather = loader.load_weather()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    window = slice_window(weather, day, hour_from, hour_to)
    if window.empty:
        return {"error": "No data for date", "date": date_str}

    coco_rounded = window["coco"].round().astype(int)
    mode = coco_rounded.mode()
    coco_mode = int(mode.iloc[0]) if not mode.empty else 0

    slots = [
        {
            "timestamp": ts.strftime(TS_OUT),
            "temp": as_float(row["temp"]),
            "prcp": as_float(row["prcp"]),
            "coco": int(coco_rounded.loc[ts]),
            "wspd": as_float(row["wspd"]),
        }
        for ts, row in window.iterrows()
    ]

    return {
        "date": day.strftime(DATE_OUT),
        "hour_from": int(hour_from),
        "hour_to": int(hour_to),
        "summary": {
            "temp_mean": as_float(window["temp"].mean()),
            "temp_max": as_float(window["temp"].max()),
            "temp_min": as_float(window["temp"].min()),
            # prcp sind stuendliche Mengen (mm/h), linear auf 15 Minuten
            # interpoliert. Die Summe der 15-Minuten-Werte zaehlte jede
            # Stunde viermal (13.07.: 99,3 statt 24,8 mm) – deshalb / 4.
            "prcp_total": as_float(window["prcp"].sum() / SLOTS_PER_HOUR),
            "max_prcp_rate_mm_per_h": as_float(window["prcp"].max()),
            "wspd_mean": as_float(window["wspd"].mean()),
            "coco_mode": coco_mode,
            "conditions": get_condition_label(coco_mode),
        },
        "slots": slots,
        "data_points": int(len(window)),
        "data_limitation": (
            "coco und cldc sind vom Stunden- auf das 15-Minuten-Raster "
            "interpoliert; die Kategorien sind hier gerundet. prcp je Slot ist "
            "eine Stundenrate in mm/h; prcp_total ist die Menge in mm im "
            "Fenster. Die Betriebspause 01:00–04:45 fehlt in den Daten, "
            "Nachtregen ist deshalb nicht enthalten. Wetter ist netzweit, "
            "nicht stationsbezogen."
        ),
    }


def find_weather_anomalies(week_start: str, metric: str = "prcp") -> dict:
    """Stärkster Ausreisser einer Wetterkennzahl innerhalb einer Woche.

    Das Fenster umfasst week_start bis week_start + 7 Tage (exklusiv).
    """
    loader = DataLoader()
    if metric not in METRICS:
        return {
            "error": "Invalid metric",
            "input": metric,
            "allowed": sorted(METRICS),
        }

    start = parse_date(week_start)
    if start is None:
        return {"error": "Invalid date", "input": week_start}

    try:
        weather = loader.load_weather()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    end = start.normalize() + pd.Timedelta(days=7)
    week = weather[(weather.index >= start.normalize()) & (weather.index < end)]
    if week.empty:
        return {"error": "No data for week", "week_start": week_start}

    series = week[metric]
    peak_ts = series.idxmax()
    overall_mean = float(weather[metric].mean())
    overall_std = float(weather[metric].std(ddof=1))
    peak_value = float(series.loc[peak_ts])
    z = (peak_value - overall_mean) / overall_std if overall_std > 0 else 0.0

    return {
        "week_start": start.strftime(DATE_OUT),
        "week_end": (end - pd.Timedelta(minutes=15)).strftime(DATE_OUT),
        "metric": metric,
        "peak": {
            "timestamp": peak_ts.strftime(TS_OUT),
            "value": as_float(peak_value),
            "unit": METRIC_UNITS.get(metric),
            "station": None,
        },
        "week_mean": as_float(series.mean()),
        "overall_mean": as_float(overall_mean),
        "z_score": as_float(z),
        "conditions_at_peak": get_condition_label(week.loc[peak_ts, "coco"]),
        "data_points": int(len(week)),
        "data_limitation": (
            "Wetter ist netzweit erfasst – station ist deshalb immer null. "
            "z_score bezieht sich auf den Gesamtdatensatz, nicht auf die Woche."
        ),
    }


# Niederschlagsklassen in mm/h (DWD-Konvention: leicht < 2,5, maessig bis
# 10, stark darueber). "dry" ist die Vergleichsbasis.
RAIN_CLASSES = (("light", 0.1, 2.5), ("moderate", 2.5, 10.0), ("heavy", 10.0, float("inf")))
DRY_MAX = 0.1
MIN_BASELINE_HOURS = 3
TOP_RAIN_HOURS = 5


def get_rain_impact(top_n: int = TOP_RAIN_HOURS) -> dict:
    """Wie verändert Regen die Fahrgastzahlen? Auswertung über den ganzen Datensatz.

    Je Betriebsstunde: Netzsumme der Stationszählungen gegen den Mittelwert
    TROCKENER Stunden mit derselben Uhrzeit und demselben Tagestyp
    (Werktag/Wochenende). Tageszeit und Wochentag sind damit herausgerechnet –
    übrig bleibt der Unterschied, der mit dem Regen zusammenfällt.
    """
    loader = DataLoader()
    try:
        flows = loader.load_flows()
        weather = loader.load_weather()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    full = flows.loc[full_day_index(flows)]
    hour_key = full.index.floor("h")
    network = full.sum(axis=1).groupby(hour_key).sum()
    rate = weather["prcp"].groupby(weather.index.floor("h")).mean()
    df = pd.DataFrame({"flow": network, "prcp": rate}).dropna()
    df = df[df["flow"] > 0]  # Betriebspause ausschliessen
    if df.empty:
        return {"error": "No data", "detail": "Keine gemeinsamen Stunden von Flow und Wetter."}

    df["hour"] = df.index.hour
    df["weekend"] = df.index.dayofweek >= 5
    dry = df[df["prcp"] < DRY_MAX]
    baseline = dry.groupby(["weekend", "hour"])["flow"].agg(["mean", "count"])
    baseline = baseline[baseline["count"] >= MIN_BASELINE_HOURS]["mean"]
    df["expected"] = [baseline.get((w, h)) for w, h in zip(df["weekend"], df["hour"])]
    df = df.dropna(subset=["expected"])
    df["diff_pct"] = (df["flow"] - df["expected"]) / df["expected"] * 100.0

    classes = {}
    for name, lo, hi in RAIN_CLASSES:
        part = df[(df["prcp"] >= lo) & (df["prcp"] < hi)]
        classes[name] = {
            "mm_per_h": f"{lo}–{hi if hi != float('inf') else '∞'}",
            "hours": int(len(part)),
            "mean_change_pct": as_float(part["diff_pct"].mean()) if len(part) else None,
            "median_change_pct": as_float(part["diff_pct"].median()) if len(part) else None,
        }
    rainy = df[df["prcp"] >= RAIN_CLASSES[0][1]]
    dry_part = df[df["prcp"] < DRY_MAX]

    wettest = rainy.sort_values("prcp", ascending=False).head(max(1, int(top_n)))
    examples = [{
        "hour": ts.strftime(TS_OUT),
        "weekday": ts.day_name(),
        "rain_mm_per_h": as_float(row["prcp"]),
        "network_flow": int(round(row["flow"])),
        "expected_dry_flow": int(round(row["expected"])),
        "change_pct": as_float(row["diff_pct"]),
    } for ts, row in wettest.iterrows()]

    return {
        "summary": {
            "rain_hours": int(len(rainy)),
            "dry_hours": int(len(dry_part)),
            "mean_change_rain_pct": as_float(rainy["diff_pct"].mean()) if len(rainy) else None,
            "mean_change_dry_pct": as_float(dry_part["diff_pct"].mean()) if len(dry_part) else None,
        },
        "by_intensity": classes,
        "wettest_hours": examples,
        "period": f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}",
        "method": (
            "Netzsumme je Betriebsstunde gegen den Mittelwert trockener Stunden "
            f"(< {DRY_MAX} mm/h) mit gleicher Uhrzeit und gleichem Tagestyp "
            "(Werktag/Wochenende)."
        ),
        "data_limitation": (
            "Korrelation, keine Kausalität: Regen fällt mit anderen Einflüssen "
            "(Events, Ferien, Sperrungen) zusammen. Wetter ist netzweit, nicht "
            "stationsbezogen. prcp ist eine auf 15 Minuten interpolierte "
            "Stundenrate; Flow-Werte sind simulierte, bei 3.000 je Slot "
            "gedeckelte Stationszählungen."
        ),
    }
