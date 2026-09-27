"""Tool 10 – Abhängigkeiten zwischen Stationen und Umleitungsverhalten.

find_station_dependencies    Stationspaare ohne direkte Verbindung, deren
                             Nachfrage gemeinsam schwankt
analyze_disruption_routing   wohin Fahrgäste bei Streckensperrungen ausweichen
                             – verglichen mit dem kürzesten Umweg im Graphen

Beide Analysen arbeiten mit Abweichungen vom Normalfall, nicht mit Rohwerten:
Rohwerte korrelieren an jedem Stationspaar hoch, weil alle Stationen morgens
und abends voll sind. Es gibt keine OD-Matrix – beide Ergebnisse sind
Korrelationen bzw. Mengenverschiebungen, keine beobachteten Einzelreisen.
"""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
import pandas as pd

from src.loader import DataLoader
from src.tools.network_tool import find_transit_route
from src.tools._common import (
    TS_OUT,
    as_float,
    extract_lines,
    extract_stations,
    full_day_index,
    haversine_m,
    resolve_station,
)

MIN_HOPS = 3
TOP_PAIRS = 5
WALKING_DISTANCE_M = 800
MAX_LAG_SLOTS = 2  # ±30 Minuten


NO_OD_NOTE = (
    "Keine OD-Matrix: Ergebnisse sind Korrelationen bzw. Mengenverschiebungen "
    "an Stationen, keine beobachteten Einzelreisen. Flow-Werte sind simulierte, "
    "bei 3.000 je 15-Minuten-Slot gedeckelte Stationszählungen."
)


def _day_type(index: pd.DatetimeIndex) -> np.ndarray:
    """0 = Werktag, 1 = Samstag, 2 = Sonntag."""
    dow = index.dayofweek
    return np.where(dow < 5, 0, np.where(dow == 5, 1, 2))


def _residuals(flows: pd.DataFrame) -> pd.DataFrame:
    """Abweichung je Slot vom mittleren Tagesgang (Tagestyp × Uhrzeit)."""
    keys = [_day_type(flows.index), flows.index.time]
    profile = flows.groupby(keys).transform("mean")
    return flows - profile


def find_station_dependencies(min_hops: int = MIN_HOPS, top_n: int = TOP_PAIRS) -> dict:
    """Stationspaare ohne direkte Verbindung, deren Nachfrage gemeinsam schwankt.

    Methode: Abweichung vom mittleren Tagesgang je Station, davon der
    netzweite Gleichlauf (Mittel der standardisierten Abweichungen aller
    Stationen, z. B. Regen oder Ferien) abgezogen. Korreliert werden die
    Reste von Paaren mit mindestens min_hops Kanten Abstand im U-Bahn-Graph,
    auch zeitversetzt bis ±30 Minuten.
    """
    try:
        min_hops = max(2, int(min_hops))
        top_n = max(1, min(int(top_n), 20))
    except (TypeError, ValueError):
        return {"error": "Invalid parameters", "input": [min_hops, top_n]}

    loader = DataLoader()
    try:
        flows = loader.load_flows()
        stations = loader.load_stations()
        graph = loader.load_graph()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    full = flows.loc[full_day_index(flows)]
    full = full[(full.sum(axis=1) > 0)]  # Betriebspause
    if full.empty:
        return {"error": "No data", "detail": "Keine vollständigen Betriebstage."}

    resid = _residuals(full)
    std = resid.std(ddof=0).replace(0, np.nan)
    z = (resid / std).fillna(0.0)
    common = z.mean(axis=1)
    beta = z.apply(lambda col: np.dot(col, common) / np.dot(common, common))
    specific = z - np.outer(common, beta)

    corr0 = np.corrcoef(specific.to_numpy().T)
    upper = np.triu_indices_from(corr0, 1)
    raw_median = float(np.median(np.corrcoef(full.to_numpy().T)[upper]))
    profile_median = float(np.median(np.corrcoef(resid.to_numpy().T)[upper]))
    cols = list(specific.columns)
    col_to_id = loader.col_to_station_id
    info = stations.set_index("station_id")
    hops = dict(nx.all_pairs_shortest_path_length(graph))

    candidates = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        a_id, b_id = col_to_id.get(cols[i]), col_to_id.get(cols[j])
        if a_id is None or b_id is None:
            continue
        a, b = info.loc[a_id], info.loc[b_id]
        if a["station_name"] == b["station_name"]:
            continue  # Stadtmitte U2/U6: dieselbe Station
        dist_hops = hops.get(a_id, {}).get(b_id)
        if dist_hops is None or dist_hops < min_hops:
            continue
        candidates.append((corr0[i, j], i, j, a_id, b_id, dist_hops))
    candidates.sort(reverse=True)

    pairs = []
    arr = specific.to_numpy()
    for r0, i, j, a_id, b_id, dist_hops in candidates[: top_n * 3]:
        best_lag, best_r = 0, r0
        for lag in range(-MAX_LAG_SLOTS, MAX_LAG_SLOTS + 1):
            if lag == 0:
                continue
            x, y = (arr[:-lag, i], arr[lag:, j]) if lag > 0 else (arr[-lag:, i], arr[:lag, j])
            r = float(np.corrcoef(x, y)[0, 1])
            if r > best_r:
                best_lag, best_r = lag, r
        a, b = info.loc[a_id], info.loc[b_id]
        lines_a = {x.strip() for x in str(a["u_bahn_lines"]).split(",")}
        lines_b = {x.strip() for x in str(b["u_bahn_lines"]).split(",")}
        distance = haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        hints = []
        if distance <= WALKING_DISTANCE_M:
            hints.append(f"walking distance ({distance:.0f} m) – same catchment area")
        if lines_a & lines_b:
            hints.append(f"same line {', '.join(sorted(lines_a & lines_b))} – shared line-level effects")
        if best_lag:
            hints.append(f"strongest at a {abs(best_lag) * 15}-min offset – consistent with travel time")
        pairs.append({
            "station_a": a["station_name"],
            "station_b": b["station_name"],
            "lines_a": a["u_bahn_lines"],
            "lines_b": b["u_bahn_lines"],
            "correlation": as_float(r0, 3),
            "best_lag_minutes": best_lag * 15,
            "correlation_at_best_lag": as_float(best_r, 3),
            "graph_hops": int(dist_hops),
            "distance_m": as_float(distance, 0),
            "shared_lines": sorted(lines_a & lines_b),
            "mechanism_hints": hints,
        })
        if len(pairs) >= top_n:
            break

    all_r = [c[0] for c in candidates]
    strongest = max(all_r) if all_r else None
    return {
        "finding": (
            "No strong dependency: after removing the shared daily rhythm, the "
            f"strongest pair reaches r = {strongest:.3f}. Raw flows correlate at "
            f"r = {raw_median:.2f} (median), but that is only the common daily "
            "pattern, not a station-to-station dependency."
            if strongest is not None and strongest < 0.3 else
            "Pairs with a noticeable co-movement beyond the daily rhythm exist – see pairs."
        ),
        "median_correlation_raw_flows": as_float(raw_median, 3),
        "median_correlation_after_daily_pattern": as_float(profile_median, 3),
        "pairs": pairs,
        "pairs_evaluated": len(candidates),
        "median_correlation_all_pairs": as_float(float(np.median(all_r)), 3) if all_r else None,
        "min_graph_hops": min_hops,
        "slots_used": int(len(specific)),
        "method": (
            "Correlation of flow deviations from each station's typical day "
            "(weekday/Saturday/Sunday × time of day), after removing the "
            "network-wide common movement (weather, holidays). Pairs at least "
            f"{min_hops} stops apart in the U-Bahn graph; lag up to ±30 min."
        ),
        "data_limitation": NO_OD_NOTE,
    }


MAX_CASES = 6


def _sign_test(successes: int, n: int) -> float:
    """Zweiseitiger Vorzeichentest (Binomial, p = 0,5)."""
    if n == 0:
        return 1.0
    k = min(successes, n - successes)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _segment_closures(closures: pd.DataFrame) -> list:
    return [r for r in closures.itertuples()
            if "suspended" in r.description.lower() and len(extract_stations(r.description)) == 2]


def analyze_disruption_routing(max_cases: int = 6) -> dict:
    """Weichen Fahrgäste bei Streckensperrungen erkennbar aus – und wohin?

    Je Streckensperrung: Zuwachs an Stationen nahe dem gesperrten Abschnitt
    (1–2 Stationen) gegen weit entfernte (ab 5), dazu der kürzeste Umweg im
    U-Bahn-Graph und im Gesamtnetz. Basis sind dieselben Slots an den
    übrigen gleichen Wochentagen. Ein Vorzeichentest prüft, ob die
    Nachbarschaft systematisch stärker zulegt als der Rest des Netzes.
    """
    try:
        max_cases = max(1, min(int(max_cases), 18))
    except (TypeError, ValueError):
        max_cases = MAX_CASES

    loader = DataLoader()
    try:
        flows = loader.load_flows()
        stations = loader.load_stations()
        closures = loader.load_closures()
        graph = loader.load_graph()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    full_mask = flows.index.isin(full_day_index(flows))
    id_to_name = dict(zip(stations["station_id"], stations["station_name"]))
    name_col = {sid: col for col, sid in loader.col_to_station_id.items()}
    lines_of = {
        sid: {x.strip() for x in str(v).split(",")}
        for sid, v in zip(stations["station_id"], stations["u_bahn_lines"])
    }

    cases = []
    for row in _segment_closures(closures):
        ends = [resolve_station(stations, n) for n in extract_stations(row.description)]
        if not (ends[0] and ends[1]):
            continue
        a_id, b_id = ends[0][0]["station_id"], ends[1][0]["station_id"]
        line = (extract_lines(row.description) or [None])[0]
        on_line = {sid for sid, ls in lines_of.items() if line in ls} if line else set(graph)
        search = graph.subgraph(on_line) if a_id in on_line and b_id in on_line else graph
        try:
            section = nx.shortest_path(search, a_id, b_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        working = graph.copy()
        working.remove_nodes_from(section[1:-1])
        if len(section) == 2 and working.has_edge(a_id, b_id):
            working.remove_edge(a_id, b_id)
        try:
            detour = nx.shortest_path(working, a_id, b_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            detour = []

        window = (flows.index >= row.when) & (flows.index < row.end_time)
        during = flows[window]
        if during.empty:
            continue
        times = set(during.index.time)
        base_mask = (
            (flows.index.dayofweek == row.when.dayofweek)
            & pd.Index(flows.index.time).isin(times) & ~window & full_mask
        )
        baseline = flows[base_mask]
        if baseline.empty:
            continue
        observed = during.sum()
        expected = baseline.groupby(baseline.index.time).mean().sum()
        uplift = ((observed - expected) / expected.replace(0, np.nan) * 100.0).dropna()
        closed = set(section[1:-1])
        uplift = uplift[[c for c in uplift.index
                         if loader.col_to_station_id.get(c) not in closed]]
        # Entfernung zum gesperrten Abschnitt: eine Mehrquellen-Suche statt
        # einer Pfadsuche je Station.
        dist_to_section = nx.multi_source_dijkstra_path_length(graph, set(section))
        near = [uplift[c] for c in uplift.index
                if 1 <= dist_to_section.get(loader.col_to_station_id.get(c), 99) <= 2]
        far = [uplift[c] for c in uplift.index
               if dist_to_section.get(loader.col_to_station_id.get(c), 0) >= 5]

        # Theoretisch kuerzester Weg im Gesamtnetz (S-Bahn/Bus/Tram erlaubt):
        # die meisten Sperrungen liegen auf Aesten ohne U-Bahn-Umweg.
        transit = find_transit_route(
            id_to_name[a_id], id_to_name[b_id],
            avoid_lines=[line] if line else None,
            avoid_stations=[id_to_name[s] for s in closed],
        )
        multimodal = None
        if "error" not in transit:
            legs = transit["routes"][0]["legs"]
            passed = ([leg["from"] for leg in legs]
                      + [n for leg in legs for n in leg["via"]]
                      + [legs[-1]["to"]])
            u_ids = {m[0]["station_id"] for m in (resolve_station(stations, n) for n in passed) if m}
            u_ids -= {a_id, b_id}
            on_route = [as_float(uplift[name_col[s]]) for s in u_ids
                        if s in name_col and name_col[s] in uplift.index]
            multimodal = {
                "summary": transit["routes"][0]["summary"],
                "minutes": transit["total_minutes"],
                "modes": transit["modes_used"],
                "u_bahn_stations_on_route": sorted(id_to_name[s] for s in u_ids),
                "mean_uplift_at_those_stations_pct": as_float(np.mean(on_route)) if on_route else None,
            }
        cases.append({
            "closure": row.description,
            "when": row.when.strftime(TS_OUT),
            "end_time": row.end_time.strftime(TS_OUT),
            "suspended_section": [id_to_name.get(s, s) for s in section],
            "shortest_detour": [id_to_name.get(s, s) for s in detour],
            "shortest_multimodal_detour": multimodal,
            "mean_uplift_near_section_pct": as_float(np.mean(near)) if near else None,
            "mean_uplift_far_away_pct": as_float(np.mean(far)) if far else None,
        })

    if not cases:
        return {"error": "No data", "detail": "Keine auswertbaren Streckensperrungen."}

    compared = [c for c in cases
                if c["mean_uplift_near_section_pct"] is not None and c["mean_uplift_far_away_pct"] is not None]
    near_wins = sum(c["mean_uplift_near_section_pct"] > c["mean_uplift_far_away_pct"] for c in compared)
    p_value = _sign_test(near_wins, len(compared))
    strongest_local = max(compared, key=lambda c: c["mean_uplift_near_section_pct"] - c["mean_uplift_far_away_pct"],
                          default=None)
    with_mm = [c for c in cases if (c["shortest_multimodal_detour"] or {}).get("mean_uplift_at_those_stations_pct") is not None]
    mm_wins = sum(c["shortest_multimodal_detour"]["mean_uplift_at_those_stations_pct"]
                  > (c["mean_uplift_far_away_pct"] or 0) for c in with_mm)
    consistent = p_value < 0.05 and near_wins > len(compared) / 2

    # Die Faelle nach Staerke der lokalen Verschiebung, ohne Stationslisten:
    # die volle Liste sprengte das Kontextbudget und verleitete das LLM zu
    # Aussagen ueber Einzelstationen, die im Rauschen liegen.
    ranked = sorted(compared, key=lambda c: c["mean_uplift_near_section_pct"] - c["mean_uplift_far_away_pct"],
                    reverse=True)
    compact_cases = [{
        "closure": c["closure"],
        "when": c["when"],
        "uplift_near_section_pct": c["mean_uplift_near_section_pct"],
        "uplift_far_away_pct": c["mean_uplift_far_away_pct"],
        "u_bahn_detour_exists": bool(c["shortest_detour"]),
        "shortest_multimodal_detour": (c["shortest_multimodal_detour"] or {}).get("summary"),
        "uplift_on_multimodal_detour_pct": (c["shortest_multimodal_detour"] or {}).get("mean_uplift_at_those_stations_pct"),
    } for c in ranked[:max_cases]]

    return {
        "finding": (
            f"No consistent rerouting pattern: stations next to the closed section rose more "
            f"than distant stations in only {near_wins} of {len(compared)} closures "
            f"(sign test p = {p_value:.2f}, i.e. chance level). Which routes passengers "
            "prefer cannot be read from station counts; single closures show strong local "
            "shifts, others none."
            if not consistent else
            f"Consistent local shift: stations next to the closed section rose more than "
            f"distant stations in {near_wins} of {len(compared)} closures (p = {p_value:.2f})."
        ),
        "summary": {
            "segment_closures_analysed": len(cases),
            "closures_where_nearby_stations_rose_more_than_distant": f"{near_wins} of {len(compared)}",
            "sign_test_p_value": as_float(p_value, 3),
            "mean_uplift_near_sections_pct": as_float(np.mean([c["mean_uplift_near_section_pct"] for c in compared])) if compared else None,
            "mean_uplift_far_away_pct": as_float(np.mean([c["mean_uplift_far_away_pct"] for c in compared])) if compared else None,
            "closures_with_a_u_bahn_detour": sum(1 for c in cases if c["shortest_detour"]),
            "closures_with_multimodal_detour_evaluated": len(with_mm),
            "multimodal_detour_stations_rose_more_than_distant": f"{mm_wins} of {len(with_mm)}",
            "strongest_local_shift": {
                "closure": strongest_local["closure"],
                "near_pct": strongest_local["mean_uplift_near_section_pct"],
                "far_pct": strongest_local["mean_uplift_far_away_pct"],
            } if strongest_local else None,
        },
        "cases_by_local_shift": compact_cases,
        "method": (
            "Per line-section closure: flow change during the closure versus the "
            "same slots on the other same weekdays, at stations 1–2 stops from "
            "the closed section versus stations 5+ stops away; sign test across "
            "closures. Detours: shortest U-Bahn path with the section removed and "
            "shortest multimodal path (bus/tram/S-Bahn) from the transit graph."
        ),
        "data_limitation": NO_OD_NOTE + (
            " Zuwächse können auch andere Ursachen haben (Events, Wetter) – die "
            "Zuordnung zu Umleitungen ist eine Plausibilitätsaussage."
        ),
    }
