"""Tool 5 – Energieverbrauch und Effizienz.

get_energy  Verbrauch je Linie plus MWh-pro-Fahrgast-Ranking

Die Effizienzrechnung braucht Fahrgastzahlen je Linie. Der Datensatz liefert
aber nur stationsbezogene Flows, und Umsteigebahnhöfe gehören zu mehreren
Linien. Die hier verwendete Vereinfachung steht als "allocation_rule" im
Rückgabe-Dict und muss in jeder Agent-Antwort mitgenannt werden.
"""

from __future__ import annotations

import pandas as pd

from src.loader import DataLoader
from src.loader import FULL_DAY_SLOTS
from src.tools._common import DATE_OUT, as_float, as_int, parse_date

# Mindestzahl Flow-Slots, damit ein Energietag in die Effizienz eingeht:
# 95 % eines vollen Betriebstags. Fehlen nur die Nachtslots nach Mitternacht
# (erster Datentag), ist der Tag praktisch vollstaendig.
ENERGY_MIN_FLOW_SLOTS = int(FULL_DAY_SLOTS * 0.95)

ALLOCATION_RULE = (
    "Der Flow eines Umsteigebahnhofs wird zu gleichen Teilen auf alle dort "
    "haltenden Linien verteilt (Alexanderplatz: je 1/3 an U2, U5, U8). "
    "Umsteiger werden damit nicht doppelt gezählt."
)

# Alternative Zuordnungen fuer den Sensitivitaetscheck. Die frueher genutzte
# Regel "erste Linie gewinnt" haengt an der Reihenfolge in u_bahn_lines:
# alle gemeinsamen U1/U3-Stationen stehen als "U1,U3" im Datensatz und
# fielen komplett an U1 – U3 wirkte dadurch kuenstlich ineffizient.
ALLOCATIONS = {
    "equal_split": "Umsteigebahnhöfe zu gleichen Teilen auf ihre Linien (Standard)",
    "first_line": "Umsteigebahnhof komplett an die erste Linie in u_bahn_lines",
    "full_count": "Umsteigebahnhof zählt für jede Linie voll (Doppelzählung)",
}


def _line_weights(stations: pd.DataFrame, col_to_station_id: dict, rule: str) -> dict:
    """{Flow-Spalte: {Linie: Anteil}} je Zuordnungsregel."""
    lines_by_id = {
        sid: [x.strip() for x in str(lines).split(",") if x.strip()]
        for sid, lines in zip(stations["station_id"], stations["u_bahn_lines"])
    }
    weights = {}
    for col, sid in col_to_station_id.items():
        lines = lines_by_id.get(sid, [])
        if not lines:
            continue
        if rule == "first_line":
            weights[col] = {lines[0]: 1.0}
        elif rule == "full_count":
            weights[col] = {ln: 1.0 for ln in lines}
        else:
            weights[col] = {ln: 1.0 / len(lines) for ln in lines}
    return weights


def _line_pax(flow_totals: pd.Series, weights: dict) -> dict:
    """Fahrgäste je Linie aus Stationssummen und Zuordnungsgewichten."""
    pax: dict[str, float] = {}
    for col, shares in weights.items():
        for ln, share in shares.items():
            pax[ln] = pax.get(ln, 0.0) + float(flow_totals.get(col, 0.0)) * share
    return pax


def get_energy(
    line: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Energieverbrauch je Linie, gefiltert nach Linie und Zeitraum."""
    loader = DataLoader()
    try:
        energy = loader.load_energy()
        stations = loader.load_stations()
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    filters = {"line": line, "date_from": date_from, "date_to": date_to}

    if date_from is not None:
        start = parse_date(date_from)
        if start is None:
            return {"error": "Invalid date", "input": date_from}
        energy = energy[energy.index >= start.normalize()]
    if date_to is not None:
        stop = parse_date(date_to)
        if stop is None:
            return {"error": "Invalid date", "input": date_to}
        energy = energy[energy.index <= stop.normalize()]

    if energy.empty:
        return {"error": "No data for range", "filters": filters}

    # Beide Auswertungen muessen denselben Zeitraum abdecken, sonst widersprechen
    # sich lines-Block und efficiency_ranking – Energie und Fahrgaeste immer
    # ueber dieselben Tage. Massgeblich sind die Energietage mit (nahezu)
    # vollstaendigem Flow: 2026-06-10 fehlen nur die Slots 00:00–00:45
    # (76 von 80, ca. 0,1 % des Tagesvolumens) – der Tag zaehlt mit. Randtage
    # wie 2026-09-22 mit 4 Slots fallen weiter heraus.
    slots = flows.groupby(flows.index.normalize()).size()
    usable = slots[slots >= ENERGY_MIN_FLOW_SLOTS].index
    daily_flow = flows[flows.index.normalize().isin(usable)]
    daily_flow = daily_flow.groupby(daily_flow.index.normalize()).sum()
    shared_days = daily_flow.index.intersection(energy.index)
    if len(shared_days) == 0:
        return {
            "error": "No overlapping full days",
            "filters": filters,
            "detail": "Flow- und Energiedaten haben keinen gemeinsamen vollen Tag.",
        }

    energy = energy.loc[shared_days]
    period = {
        "from": min(shared_days).strftime(DATE_OUT),
        "to": max(shared_days).strftime(DATE_OUT),
        "days": int(len(shared_days)),
    }

    available = list(energy.columns)
    if line is not None:
        want = str(line).strip().upper()
        if want not in available:
            return {
                "error": "Line not found",
                "input": line,
                "available_lines": available,
                "note": "U4 ist im Datensatz nicht enthalten.",
            }
        selected = [want]
    else:
        selected = available

    lines_block = {
        col: {
            "mean_mwh": as_float(energy[col].mean()),
            "total_mwh": as_int(energy[col].sum()),
            "min": as_int(energy[col].min()),
            "max": as_int(energy[col].max()),
        }
        for col in selected
    }

    # --- Effizienz: MWh pro 1000 Fahrgaeste -------------------------------
    # Identischer Zeitraum wie der lines-Block (shared_days, siehe oben).
    flow_totals = daily_flow.loc[shared_days].sum()
    energy_window = energy  # bereits auf shared_days eingegrenzt

    def rank(rule: str) -> list[dict]:
        weights = _line_weights(stations, loader.col_to_station_id, rule)
        pax_by_line = _line_pax(flow_totals, weights)
        out = []
        for ln in available:
            pax = pax_by_line.get(ln, 0.0)
            if pax <= 0:
                continue
            mwh = float(energy_window[ln].sum())
            out.append(
                {
                    "line": ln,
                    "mwh_per_1000_pax": as_float(mwh / (pax / 1000.0), 4),
                    "total_mwh": as_int(mwh),
                    "total_pax": as_int(pax),
                    "stations_served": sum(1 for w in weights.values() if ln in w),
                }
            )
        out.sort(key=lambda r: r["mwh_per_1000_pax"], reverse=True)
        return out

    ranking = rank("equal_split")
    # Wie stabil ist "schlechteste Linie"? Liegen die ersten beiden knapp
    # beieinander oder kippt das Ergebnis mit der Zuordnungsregel, muss die
    # Antwort das sagen.
    sensitivity = {}
    for rule in ALLOCATIONS:
        alt = ranking if rule == "equal_split" else rank(rule)
        if alt:
            sensitivity[rule] = {"worst": alt[0]["line"], "best": alt[-1]["line"]}
    gap_pct = (
        (ranking[0]["mwh_per_1000_pax"] - ranking[1]["mwh_per_1000_pax"])
        / ranking[1]["mwh_per_1000_pax"] * 100.0
        if len(ranking) > 1 and ranking[1]["mwh_per_1000_pax"] else None
    )

    # Ranking vor den Linien-Rohwerten: bei gekuerztem LLM-Kontext muss die
    # Effizienzaussage ueberleben.
    return {
        "filters": filters,
        "worst_efficiency_line": ranking[0]["line"] if ranking else None,
        "best_efficiency_line": ranking[-1]["line"] if ranking else None,
        "efficiency_ranking": ranking,
        "worst_vs_second_gap_pct": as_float(gap_pct, 2) if gap_pct is not None else None,
        "allocation_sensitivity": sensitivity,
        "period": period,
        "allocation_rule": ALLOCATION_RULE,
        "lines": lines_block,
        "data_limitation": (
            "Alle Werte (lines UND efficiency_ranking) beziehen sich auf "
            f"denselben Zeitraum {period['from']} bis {period['to']} "
            f"({period['days']} volle Betriebstage). "
            "Energiedaten sind tagesgranular, Flows 15-minütig – die "
            "Effizienz ist nur auf Tagesebene bestimmbar. U4 fehlt im "
            "Datensatz. Flows sind simulierte Stationszählungen, keine "
            "linienbezogenen Fahrgastkilometer; das Verhältnis ist ein "
            "Proxy, kein betriebswirtschaftlicher Kennwert."
        ),
    }
