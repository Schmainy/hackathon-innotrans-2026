"""Tool 6 – Netztopologie.

get_critical_stations    Artikulationspunkte, nach abgeschnittenen Stationen gerankt
find_alternative_routes  bis zu 3 kürzeste Wege, optional mit Sperrungen;
                         Fallback auf das Gesamtnetz, wenn die U-Bahn nicht reicht
find_transit_route       schnellste Verbindung im Gesamtnetz (Bus, Tram,
                         S-Bahn, U-Bahn) aus src/graph_db
"""

from __future__ import annotations

import heapq
import itertools
import json
from functools import lru_cache
from pathlib import Path

import networkx as nx

from src.loader import DataLoader
from src.tools._common import as_float, full_day_index, normalize_station, resolve_station

TOP_N = 5

# Gesamtnetz-Graph (build_graph.py + merge_transit.py): Buslinien aus den
# BVG-Linienbändern, U-/S-Bahn und Tram aus dem VBB-GTFS.
TRANSIT_DIR = Path(__file__).resolve().parent.parent / "graph_db"
TRANSFER_PENALTY_MIN = 3.0
TRANSIT_LIMITATION = (
    "All lines, stops and times in this result are timetable data [VBB timetable], "
    "not general knowledge. "
    "Gesamtnetz aus BVG-Linienbändern (Bus, Ø-Fahrzeiten) und VBB-GTFS "
    "(U-/S-Bahn, Tram, Median-Fahrzeiten), Fahrplan statt Echtzeit. Jeder "
    "Linienwechsel wird mit 3 min Umstiegszeit bewertet; Wartezeiten sind "
    "nicht enthalten. Linien mit Suffix 'SEV' sind Schienenersatzverkehr. "
    "Nicht Teil des Hackathon-Datensatzes."
)


def _build_graph(loader: DataLoader):
    """Stationsgraph inkl. Umsteigekanten (siehe DataLoader.load_graph)."""
    return loader.load_graph(), loader.load_stations()


def get_critical_stations(top_n: int = TOP_N) -> dict:
    """Die Stationen, deren Sperrung das Netz am stärksten fragmentiert.

    Kennzahl ist stations_cut_off: wie viele Stationen nach dem Entfernen
    des Knotens nicht mehr mit dem Hauptnetz (größte verbleibende
    Komponente) verbunden sind. Bewusst NICHT die kleinste Komponente –
    Alexanderplatz zerfällt in 142 / 19 / 6 Stationen, abgeschnitten sind
    also 25, nicht 6. Gleichstand wird über den mittleren Tagesflow der
    abgeschnittenen Stationen aufgelöst.
    """
    loader = DataLoader()
    try:
        graph, stations = _build_graph(loader)
        flows = loader.load_flows()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    articulation = set(nx.articulation_points(graph))

    # Mittlerer Tagesflow je Station ueber alle vollstaendigen Betriebstage.
    full = flows.loc[full_day_index(flows)]
    mean_daily = full.groupby(full.index.normalize()).sum().mean()
    daily_by_id = {
        sid: float(mean_daily[col])
        for col, sid in loader.col_to_station_id.items()
        if col in mean_daily.index
    }

    id_to_name = dict(zip(stations["station_id"], stations["station_name"]))
    id_to_lines = dict(zip(stations["station_id"], stations["u_bahn_lines"]))

    candidates = []
    for sid in articulation:
        reduced = graph.copy()
        reduced.remove_node(sid)
        components = sorted(nx.connected_components(reduced), key=len, reverse=True)
        cut_off = [n for comp in components[1:] for n in comp]
        cut_off_flow = sum(daily_by_id.get(n, 0.0) for n in cut_off)
        candidates.append(
            {
                "station_id": sid,
                "station_name": id_to_name.get(sid, sid),
                "lines": id_to_lines.get(sid, ""),
                "stations_cut_off": len(cut_off),
                "components_after_removal": len(components),
                "fragment_sizes": [len(c) for c in components],
                "mean_daily_flow": as_float(daily_by_id.get(sid, 0.0)),
                "cut_off_stations_daily_flow": as_float(cut_off_flow),
                "cut_off_examples": sorted(id_to_name.get(n, n) for n in cut_off)[:3],
            }
        )

    candidates.sort(
        key=lambda c: (c["stations_cut_off"], c["cut_off_stations_daily_flow"]),
        reverse=True,
    )

    top = []
    for rank, cand in enumerate(candidates[:top_n], start=1):
        # Direkt betroffen: Fahrgaeste der Station selbst plus aller Stationen,
        # die ohne sie vom Hauptnetz abgeschnitten sind. Steht vorn, damit es
        # eine Kuerzung des LLM-Kontexts am Ende des Eintrags ueberlebt.
        affected = as_float(cand["mean_daily_flow"] + cand["cut_off_stations_daily_flow"])
        top.append({"rank": rank, "station_name": cand["station_name"],
                    "stations_cut_off": cand["stations_cut_off"],
                    "affected_passengers_estimate": affected, **cand})

    # Rangliste zuerst: bei gekuerztem LLM-Kontext muss sie vollstaendig bleiben.
    return {
        "top_critical_stations": top,
        "method": "articulation_points_ranked_by_stations_cut_off",
        "metric": (
            "stations_cut_off = Stationen, die nach Entfernen des Knotens nicht "
            "mehr mit der größten verbleibenden Komponente verbunden sind"
        ),
        "total_articulation_points": len(articulation),
        "network_nodes": graph.number_of_nodes(),
        "network_edges": graph.number_of_edges(),
        "data_limitation": (
            "Flow-Werte sind stationsbezogen, keine OD-Matrix verfügbar. "
            "affected_passengers_estimate = mittlerer Tagesflow der Station "
            "plus der abgeschnittenen Stationen; Durchfahrende sind nicht "
            "enthalten. Flow-Werte sind bei 3000 pro 15-Minuten-Slot gedeckelt. "
            "Der Graph enthält U4 nicht. Die beiden Stadtmitte-Knoten (U2/U6) "
            "sind über eine ergänzte Umsteigekante verbunden."
        ),
    }


def find_alternative_routes(
    from_station: str,
    to_station: str,
    avoid_stations: list[str] | None = None,
    avoid_direct: bool = False,
    line: str | None = None,
    closed_line: str | None = None,
) -> dict:
    """Bis zu 3 kürzeste Wege zwischen zwei Stationen, optional mit Sperrungen.

    closed_line sperrt eine ganze Linie ("if U2 is closed"): entfernt werden
    Kanten, deren Endpunkte beide an der Linie liegen und keine weitere Linie
    teilen. Gemeinsame Abschnitte (U1/U3) bleiben befahrbar.

    avoid_direct=True behandelt from/to als Endpunkte eines gesperrten
    Abschnitts ("U2 suspended between Pankow and Alexanderplatz"): die
    Zwischenstationen des direkten Wegs – bei angegebener Linie entlang
    dieser Linie – werden entfernt, bei direkt benachbarten Endpunkten die
    Kante. Nur den kürzesten Pfad auszuschliessen reicht nicht: der
    zweitkürzeste führt meist trotzdem durch den gesperrten Abschnitt.
    """
    loader = DataLoader()
    try:
        graph, stations = _build_graph(loader)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    src = resolve_station(stations, from_station)
    dst = resolve_station(stations, to_station)
    if not src or not dst:
        # Station ausserhalb des U-Bahn-Datensatzes (Ostkreuz, Bushalt):
        # Antwort aus dem Gesamtnetz statt "nicht gefunden".
        missing = from_station if not src else to_station
        transit = find_transit_route(from_station, to_station,
                                     avoid_stations=avoid_stations)
        if "error" not in transit:
            transit["fallback_reason"] = (
                f"'{missing}' is not in the U-Bahn dataset – route taken from the "
                "full Berlin transit graph (bus, tram, S-Bahn, U-Bahn)."
            )
            return transit
        return {"error": "Station not found", "input": missing}

    avoid_names = list(avoid_stations or [])
    avoid_ids: list[str] = []
    unresolved: list[str] = []
    for name in avoid_names:
        hits = resolve_station(stations, name)
        if hits:
            avoid_ids.extend(h["station_id"] for h in hits)
        else:
            unresolved.append(name)

    src_id, dst_id = src[0]["station_id"], dst[0]["station_id"]
    if src_id == dst_id:
        return {"error": "Identical stations", "input": [from_station, to_station]}

    id_to_name = dict(zip(stations["station_id"], stations["station_name"]))
    working = graph.copy()

    suspended: list[str] = []
    premise_warning = None
    if line and avoid_direct:
        off_line = [
            m["station_name"] for m in (src[0], dst[0])
            if str(line).strip().upper()
            not in [x.strip() for x in str(m["u_bahn_lines"]).split(",")]
        ]
        if off_line:
            # z. B. "U8 zwischen Hermannplatz und Neukölln": Neukölln liegt
            # an U7, nicht an U8. Die Frage enthält dann eine falsche Annahme.
            premise_warning = (
                f"{', '.join(off_line)} is not served by {line} in the dataset "
                "– the section was taken from the whole network instead."
            )
    if avoid_direct:
        section = _direct_section(graph, stations, src_id, dst_id, line)
        if section:
            inner = section[1:-1]
            if inner:
                avoid_ids.extend(inner)
            elif working.has_edge(src_id, dst_id):
                working.remove_edge(src_id, dst_id)
            suspended = [id_to_name.get(n, n) for n in section]

    removed = [s for s in set(avoid_ids) if s not in (src_id, dst_id) and s in working]
    working.remove_nodes_from(removed)

    closed = str(closed_line).strip().upper() if closed_line else None
    closed_edges = 0
    if closed:
        lines_of = {
            sid: {x.strip() for x in str(v).split(",") if x.strip()}
            for sid, v in zip(stations["station_id"], stations["u_bahn_lines"])
        }
        drop = [
            (a, b) for a, b, data in working.edges(data=True)
            if not data.get("transfer")
            and closed in lines_of.get(a, set()) and closed in lines_of.get(b, set())
            and not (lines_of[a] & lines_of[b]) - {closed}
        ]
        working.remove_edges_from(drop)
        closed_edges = len(drop)

    routes = []
    try:
        generator = nx.shortest_simple_paths(working, src_id, dst_id)
        for rank, path in enumerate(generator, start=1):
            names = [id_to_name.get(p, p) for p in path]
            routes.append(
                {
                    "rank": rank,
                    "stops": len(path) - 1,
                    "path": names,
                    "via": names[1:-1],
                    "path_ids": list(path),
                }
            )
            if rank == 3:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        routes = []

    result = {
        "from": src[0]["station_name"],
        "to": dst[0]["station_name"],
        "avoided": avoid_names,
        "routes": routes,
        "data_limitation": (
            "Keine Reisezeiten verfügbar. Stops als Proxy für Reisezeit. "
            "Der Graph kennt keine Linienführung – ein Weg mit wenigen Stops "
            "kann mehrere Umstiege enthalten. Umsteigezeiten sind nicht "
            "enthalten. U4 fehlt im Netz."
        ),
    }
    if suspended:
        result["suspended_section"] = suspended
        result["suspended_line"] = line
    if closed:
        result["closed_line"] = closed
        result["closed_line_edges_removed"] = closed_edges
    if premise_warning:
        result["premise_warning"] = premise_warning
    if not routes:
        result["error"] = "No route found"
        result["detail"] = (
            "No alternative U-Bahn route found after removing the avoided "
            "stations – the destination is cut off from the rest of the "
            "U-Bahn network."
        )
        result["tip"] = (
            "Consider S-Bahn or bus replacement service (SEV) "
            "[General knowledge – not in the dataset]."
        )
    if unresolved:
        result["unresolved_avoid_stations"] = unresolved
    if removed:
        result["avoided_resolved_ids"] = sorted(removed)

    # Sperrung oder kein U-Bahn-Weg: multimodale Alternative aus dem
    # Gesamtnetz (Bus, Tram, S-Bahn), gesperrte Linie/Halte ausgeschlossen.
    if not routes or avoid_direct or avoid_names or closed:
        blocked = [id_to_name.get(s, s) for s in removed]
        banned = [line] if (line and avoid_direct) else []
        if closed:
            banned.append(closed)
        transit = find_transit_route(
            src[0]["station_name"], dst[0]["station_name"],
            avoid_lines=banned or None,
            avoid_stations=blocked or None,
        )
        if "error" not in transit:
            result["transit_alternative"] = {
                key: transit[key] for key in
                ("routes", "total_minutes", "transfers", "modes_used",
                 "avoided_lines", "source", "data_limitation")
            }
            if not routes:
                # Kein U-Bahn-Weg, aber ein multimodaler: das ist eine Antwort,
                # kein Fehler – sonst verwirft der Agent die Alternative.
                for key in ("error", "detail", "tip"):
                    result.pop(key, None)
                result["status"] = ("No U-Bahn route left – a multimodal "
                                    "alternative via bus/tram/S-Bahn exists.")
    return result


def _direct_section(graph, stations, src_id: str, dst_id: str,
                    line: str | None) -> list[str]:
    """Stationsfolge des direkten Abschnitts zwischen zwei Stationen.

    Mit Linie: kürzester Weg nur über Stationen dieser Linie – so wird der
    tatsächlich gesperrte Streckenabschnitt getroffen, nicht eine kürzere
    Verbindung über andere Linien.
    """
    search = graph
    if line:
        want = str(line).strip().upper()
        on_line = {
            sid for sid, lines in zip(stations["station_id"], stations["u_bahn_lines"])
            if want in [x.strip() for x in str(lines).split(",")]
        }
        if src_id in on_line and dst_id in on_line:
            search = graph.subgraph(on_line)
    try:
        return nx.shortest_path(search, src_id, dst_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# --------------------------------------------------------------------- #
# Gesamtnetz (Bus, Tram, S-Bahn, U-Bahn) aus src/graph_db
# --------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_transit() -> tuple[dict, dict]:
    """adjacency.json und Stop-Namen; wirft FileNotFoundError ohne Graph."""
    with open(TRANSIT_DIR / "adjacency.json", encoding="utf-8") as f:
        adjacency = json.load(f)
    names: dict[str, str] = {}
    index_path = TRANSIT_DIR / "stops_index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            names = json.load(f).get("by_id", {})
    for entries in adjacency.values():
        for e in entries:
            names.setdefault(e["neighbor"], e["name"])
    return adjacency, names


def _resolve_transit_stop(name: str, adjacency: dict, names: dict) -> str | None:
    """Stop-ID zu einem Namen: exakt (normalisiert) vor Teilstring, bei
    mehreren Treffern der am besten angebundene Knoten ("Alexanderplatz" ->
    S+U-Bahnhof statt einer Bushaltestelle am Platz)."""
    if name in adjacency:
        return name
    key = normalize_station(name)
    if not key:
        return None
    norm = {sid: normalize_station(n) for sid, n in names.items() if sid in adjacency}
    exact = [sid for sid, n in norm.items() if n == key]
    candidates = exact or [sid for sid, n in norm.items() if n.startswith(key)] \
        or [sid for sid, n in norm.items() if key in n]
    if not candidates and "/" in str(name):
        # "S Messe Nord/ICC" -> "S Messe Nord": Zusatz nach "/" weglassen.
        return _resolve_transit_stop(str(name).split("/")[0], adjacency, names)
    if not candidates:
        return None
    return max(candidates, key=lambda sid: (len(adjacency[sid]), -len(norm[sid])))


def find_transit_route(
    from_station: str,
    to_station: str,
    avoid_lines: list[str] | None = None,
    avoid_stations: list[str] | None = None,
    transfer_penalty_min: float = TRANSFER_PENALTY_MIN,
) -> dict:
    """Schnellste Verbindung im Gesamtnetz inkl. Bus, Tram und S-Bahn.

    Dijkstra über (Haltestelle, Linie): Fahrzeit laut Graph plus
    transfer_penalty_min je Linienwechsel. avoid_lines (z. B. ["U2"]) und
    avoid_stations blenden gesperrte Linien bzw. Halte aus.
    """
    try:
        adjacency, names = _load_transit()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"error": "Transit graph not available",
                "detail": f"{TRANSIT_DIR} ({exc})"}

    src = _resolve_transit_stop(from_station, adjacency, names)
    if src is None:
        return {"error": "Station not found", "input": from_station, "network": "transit"}
    dst = _resolve_transit_stop(to_station, adjacency, names)
    if dst is None:
        return {"error": "Station not found", "input": to_station, "network": "transit"}
    if src == dst:
        return {"error": "Identical stations", "input": [from_station, to_station]}

    banned_lines = {str(l).strip().upper() for l in (avoid_lines or [])}
    banned_stops = {
        sid for sid in (_resolve_transit_stop(s, adjacency, names) for s in avoid_stations or [])
        if sid and sid not in (src, dst)
    }
    penalty = max(0.0, float(transfer_penalty_min)) * 60

    counter = itertools.count()
    heap = [(0.0, next(counter), src, None)]
    best = {(src, None): 0.0}
    prev: dict[tuple, tuple] = {}
    goal = None
    while heap:
        cost, _, node, line = heapq.heappop(heap)
        if cost > best.get((node, line), float("inf")):
            continue
        if node == dst:
            goal = (node, line)
            break
        for e in adjacency.get(node, []):
            nbr, eline = e["neighbor"], e["line"]
            # "U2" gesperrt sperrt nicht "U2 SEV" – der Ersatzverkehr ist die Alternative.
            if nbr in banned_stops or eline.upper() in banned_lines:
                continue
            new = cost + e["travel_time"] + (penalty if line is not None and eline != line else 0)
            state = (nbr, eline)
            if new < best.get(state, float("inf")):
                best[state] = new
                prev[state] = ((node, line), e)
                heapq.heappush(heap, (new, next(counter), nbr, eline))

    base = {"from": names.get(src, src), "to": names.get(dst, dst),
            "avoided_lines": sorted(banned_lines),
            "avoided_stations": sorted(names.get(s, s) for s in banned_stops)}
    if goal is None:
        return {**base, "error": "No route found",
                "detail": "Keine Verbindung im Gesamtnetz nach Ausschluss der Sperrungen."}

    hops = []
    state = goal
    while state in prev:
        before, edge = prev[state]
        hops.append((before[0], state[0], edge))
        state = before
    hops.reverse()

    legs = []
    for line, group in itertools.groupby(hops, key=lambda h: h[2]["line"]):
        group = list(group)
        legs.append({
            "line": line,
            "mode": group[0][2]["mode"],
            "from": names.get(group[0][0], group[0][0]),
            "to": names.get(group[-1][1], group[-1][1]),
            "stops": len(group),
            "minutes": as_float(sum(h[2]["travel_time"] for h in group) / 60, 1),
            "via": [names.get(h[1], h[1]) for h in group[:-1]],
        })
    ride = sum(leg["minutes"] for leg in legs)
    transfers = max(0, len(legs) - 1)
    summary = " → ".join(f"{leg['line']} ({leg['from']} – {leg['to']})" for leg in legs)
    return {
        **base,
        "routes": [{"rank": 1, "summary": summary, "legs": legs}],
        "total_minutes": as_float(ride, 1),
        "transfers": transfers,
        "total_with_transfers_minutes": as_float(ride + transfers * penalty / 60, 1),
        "modes_used": sorted({leg["mode"] for leg in legs}),
        "source": "src/graph_db/adjacency.json",
        "data_limitation": TRANSIT_LIMITATION,
    }


def find_diverse_transit_routes(
    from_station: str,
    to_station: str,
    n_routes: int = 3,
    avoid_lines: list[str] | None = None,
) -> dict:
    """Mehrere bewusst verschiedene Verbindungen statt nur der schnellsten.

    Nach jeder Route werden die Zwischenhalte ihres Hauptabschnitts (längster
    Abschnitt) gesperrt und neu gesucht. Nur die Linie zu sperren reicht
    nicht: S3/S5/S7 fahren auf demselben Korridor. So entstehen Alternativen
    auf anderen Korridoren – z. B. um einen Andrang zu verteilen.
    """
    try:
        n_routes = max(1, min(int(n_routes), 5))
    except (TypeError, ValueError):
        n_routes = 3
    banned = [str(line) for line in (avoid_lines or [])]
    banned_stops: list[str] = []
    routes, fastest = [], None
    for _ in range(n_routes):
        result = find_transit_route(from_station, to_station, avoid_lines=banned or None,
                                    avoid_stations=banned_stops or None)
        if "error" in result:
            if not routes:
                return result
            break
        legs = result["routes"][0]["legs"]
        main = max(legs, key=lambda leg: leg["minutes"])
        if fastest is None:
            fastest = result["total_with_transfers_minutes"]
        routes.append({
            "rank": len(routes) + 1,
            "summary": result["routes"][0]["summary"],
            "total_minutes": result["total_minutes"],
            "total_with_transfers_minutes": result["total_with_transfers_minutes"],
            "extra_minutes_vs_fastest": as_float(result["total_with_transfers_minutes"] - fastest, 1),
            "transfers": result["transfers"],
            "modes_used": result["modes_used"],
            "main_line": main["line"],
            "legs": legs,
        })
        banned_stops.extend(main["via"])
        if not main["via"]:  # Hauptabschnitt ohne Zwischenhalt: Linie sperren
            banned.append(main["line"])
    return {
        "from": routes[0]["legs"][0]["from"] if routes else from_station,
        "to": routes[0]["legs"][-1]["to"] if routes else to_station,
        "routes": routes,
        "method": (
            "Route 1 is the fastest; each further route excludes the intermediate "
            "stops of the previous routes' main section, so the alternatives use "
            "different corridors."
        ),
        "source": "src/graph_db/adjacency.json",
        "data_limitation": TRANSIT_LIMITATION,
    }
