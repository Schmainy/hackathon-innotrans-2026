"""
merge_transit.py – Bus-Graph (aus build_graph.py) um U-Bahn, S-Bahn und Tram erweitern
=======================================================================================

Quellen:
  * data/derived/gtfs_route_stops.csv          VBB-GTFS-Extrakt: repräsentative Fahrten je Linie/Richtung
                                               (line, mode, direction_id, trip_id, stop_sequence,
                                               departure_time, station_key, stop_name, stop_lat, stop_lon)
  * data/training dataset/berlin_ubahn_connections.csv   U-Bahn-Kanten (station_id_1, station_id_2)
  * data/training dataset/stations_with_ubahn.csv        U-Bahnhöfe mit Koordinaten und Linien

Vorgehen:
  1. graph.json laden; beim ersten Lauf als graph_bus.json gesichert, spätere Läufe starten von dort.
  2. Je GTFS-Station (station_key) einen Knoten bestimmen:
       a) exakter Name existiert schon als Bus-Knoten              -> zusammenführen
       b) gleicher Name ohne S/U-Präfix ("U Alexanderplatz" = "S+U Alexanderplatz")
       c) sehr ähnlicher Name (difflib >= 0.92, gleicher Anfangsbuchstabe)
       d) sonst neuer Knoten; Fußwege zu Bus-Haltepunkten derselben Station
          ("S+U Alexanderplatz" <-> "S+U Alexanderplatz/Memhardstr.")
  3. Kanten aus aufeinanderfolgenden Halten jeder Fahrt, Fahrzeit = Median der Abfahrtsdifferenzen.
  4. U-Bahn-Kanten aus berlin_ubahn_connections.csv ergänzen, falls im GTFS-Extrakt keine Kante existiert
     (Fahrzeit aus Luftlinie geschätzt, estimated=True).
  5. Koordinaten: stations_with_ubahn.csv > GTFS-Station; Bus-Knoten mit exakt gleichem GTFS-Namen
     erhalten ebenfalls GTFS-Koordinaten.
  6. graph.json, adjacency.json, stops_index.json neu schreiben (Export aus build_graph.py).

Ausführen:
  python merge_transit.py
  python merge_transit.py --modes U-Bahn,S-Bahn,Tram,Regional,Ferry
"""

import re
import sys
import json
import math
import difflib
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

from build_graph import stop_key, station_of, export, MODE_COLORS, WALK_TIME_SEC, MIN_EDGE_SEC

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"  # <repo>/data

GTFS_MODES = {"U-Bahn": "ubahn", "S-Bahn": "sbahn", "Tram": "tram", "Regional": "regional", "Ferry": "ferry",
              "SEV": "bus",  # Schienenersatzverkehr: Bus-Fahrten unter einer Schienen-Liniennummer
              "Bus": "bus"}  # nur Buslinien, die in den BVG-Linienbändern fehlen
UBAHN_KMH = 30  # Reisegeschwindigkeit inkl. Halt für geschätzte Kanten
PROXIMITY_WALK_M = 250       # Haltestellen in diesem Umkreis per Fußweg verbinden
WALK_SPEED_MPS = 1.2
WALK_TRANSFER_SEC = 60       # Umsteigezuschlag (Treppen, Orientierung)


def log(msg: str):
    print(msg, flush=True)


# ── Namen ────────────────────────────────────────────────────────────────────

def clean_gtfs_name(name: str) -> str:
    """'S+U Zoologischer Garten Bhf (Berlin)' -> 'S+U Zoologischer Garten'"""
    s = re.sub(r"\s*\[[^\]]*\]", "", name)
    s = re.sub(r"\s*\(Berlin\)$", "", s)
    s = re.sub(r"^Berlin,\s*", "", s)
    s = re.sub(r"\s+Bhf\.?(?=/|$)", "", s)
    return " ".join(s.split())


def unprefixed_key(name: str) -> str:
    return stop_key(re.sub(r"^(S\+U|S|U)\s+", "", name))


def parse_time(t: str) -> int | None:
    try:
        h, m, s = (int(x) for x in t.split(":"))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def add_proximity_walks(G: nx.MultiGraph, radius_m: float = PROXIMITY_WALK_M) -> int:
    """Fußweg-Kanten zwischen allen Haltestellen im Umkreis radius_m.

    Raster mit Zellgröße radius_m: nur Nachbarzellen werden verglichen.
    Gehzeit = Luftlinie / 1,2 m/s + 60 s Umstieg.
    """
    cells: dict[tuple[int, int], list[tuple[str, float, float]]] = defaultdict(list)
    for n, d in G.nodes(data=True):
        if d.get("lat") is None:
            continue
        lat, lon = float(d["lat"]), float(d["lon"])
        key = (int(lat * 111_320 // radius_m), int(lon * 111_320 * math.cos(math.radians(lat)) // radius_m))
        cells[key].append((n, lat, lon))
    added = 0
    for (cy, cx), members in cells.items():
        neighbours = [m for dy in (-1, 0, 1) for dx in (-1, 0, 1) for m in cells.get((cy + dy, cx + dx), [])]
        for a, lat_a, lon_a in members:
            for b, lat_b, lon_b in neighbours:
                if a >= b or G.has_edge(a, b):
                    continue
                dist = haversine_km(lat_a, lon_a, lat_b, lon_b) * 1000
                if dist <= radius_m:
                    G.add_edge(a, b, key="walk", line="Fußweg", mode="walk",
                               travel_time=int(dist / WALK_SPEED_MPS + WALK_TRANSFER_SEC),
                               color=MODE_COLORS["walk"], origin="proximity_walk",
                               distance_m=int(dist))
                    added += 1
    return added


# ── Graph vorbereiten ────────────────────────────────────────────────────────

def load_graph(path: Path) -> nx.MultiGraph:
    with open(path, encoding="utf-8") as f:
        return json_graph.node_link_graph(json.load(f), edges="edges")


def load_bus_base(graph_path: Path) -> nx.MultiGraph:
    """Bus-Graph laden. graph.json ohne Merge-Markierung (frisch aus build_graph.py) wird als
    graph_bus.json gesichert; sonst wird die Sicherung verwendet -> Skript ist wiederholbar."""
    base_path = graph_path.with_name("graph_bus.json")
    G = load_graph(graph_path)
    if "merged" not in G.graph:
        base_path.write_text(graph_path.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"Bus-Basis gesichert: {base_path.name}")
        return G
    if not base_path.exists():
        sys.exit(f"{graph_path.name} ist bereits erweitert und {base_path.name} fehlt – build_graph.py erneut ausführen.")
    log(f"{graph_path.name} ist bereits erweitert -> starte von {base_path.name}")
    return load_graph(base_path)


class StationMatcher:
    """Ordnet GTFS-Stationen bestehenden Bus-Knoten zu."""

    def __init__(self, G: nx.MultiGraph):
        self.G = G
        self.by_key = {n: n for n in G}
        self.by_unprefixed = defaultdict(list)
        self.by_station = defaultdict(list)
        for n, d in G.nodes(data=True):
            self.by_unprefixed[unprefixed_key(d["name"])].append(n)
            if d.get("station"):
                self.by_station[d["station"]].append(n)
        self.keys = list(self.by_unprefixed)
        self.stats = defaultdict(int)

    def match(self, name: str) -> tuple[str | None, str]:
        k = stop_key(name)
        if k in self.by_key:
            return k, "exact"
        cands = self.by_unprefixed.get(unprefixed_key(name), [])
        if len(cands) == 1:
            return cands[0], "prefix"
        uk = unprefixed_key(name)
        close = [c for c in difflib.get_close_matches(uk, self.keys, n=2, cutoff=0.92) if c[:1] == uk[:1]]
        if len(close) == 1 and len(self.by_unprefixed[close[0]]) == 1:
            return self.by_unprefixed[close[0]][0], "fuzzy"
        return None, "new"

    def register(self, node_id: str, name: str):
        self.by_key[node_id] = node_id
        self.by_unprefixed[unprefixed_key(name)].append(node_id)
        st = station_of(name)
        if st:
            self.by_station[st].append(node_id)


# ── Merge ────────────────────────────────────────────────────────────────────

def merge(G: nx.MultiGraph, gtfs_csv: Path, ubahn_conn_csv: Path, ubahn_stations_csv: Path,
          modes: list[str], meta: dict):
    df = pd.read_csv(gtfs_csv, dtype=str)
    df["stop_sequence"] = df["stop_sequence"].astype(int)
    df["t"] = df["departure_time"].map(parse_time)
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    rail = df[df["mode"].isin([m for m in modes if m != "Bus"])].copy()
    # Buslinien ohne Linienband-PDF (z. B. an der Hermannstraße) aus dem GTFS
    # ergaenzen – Linien mit Linienband bleiben unangetastet (keine Dubletten).
    if "Bus" in modes:
        pdf_lines = set(G.graph.get("lines", {}))
        rail_names = set(df.loc[df["mode"] != "Bus", "line"])
        # Nachtlinien (N1, N7X ...) nicht: der Graph kennt keine Tageszeiten,
        # das Routing schlug sonst tagsueber den N9 vor.
        missing_bus = df[(df["mode"] == "Bus") & ~df["line"].isin(pdf_lines)
                         & ~df["line"].isin(rail_names)
                         & ~df["line"].str.match(r"^N\d")]
        rail = pd.concat([rail, missing_bus], ignore_index=True)
        log(f"      +{missing_bus['line'].nunique()} Buslinien aus GTFS (ohne Linienband-PDF)")
    # z. B. U6 Kurt-Schumacher-Platz–Alt-Tegel: im GTFS als Bus unter Linie "U6" -> eigene Linie "U6 SEV"
    sev = df[(df["mode"] == "Bus") & df["line"].isin(set(rail.loc[rail["mode"] != "Bus", "line"]))].copy()
    sev["line"] = sev["line"] + " SEV"
    sev["mode"] = "SEV"
    rail = pd.concat([rail, sev], ignore_index=True)
    log(f"[1/5] GTFS: {len(rail)} Zeilen, {rail['line'].nunique()} Linien ({', '.join(modes)}"
        f"{', SEV: ' + ', '.join(sorted(sev['line'].unique())) if len(sev) else ''})")

    ubahn_st = pd.read_csv(ubahn_stations_csv, dtype={"station_id": str})
    ubahn_coords = {r.station_id: (float(r.latitude), float(r.longitude)) for r in ubahn_st.itertuples()}

    # ── Stationen -> Knoten
    matcher = StationMatcher(G)
    station_node: dict[str, str] = {}
    stations = (rail.groupby("station_key")
                .agg(stop_name=("stop_name", "first"), lat=("stop_lat", "mean"), lon=("stop_lon", "mean"))
                .reset_index())
    match_stats = defaultdict(int)
    for st in stations.itertuples():
        name = clean_gtfs_name(st.stop_name)
        node, how = matcher.match(name)
        if node is None:
            node = stop_key(name)
            if node not in G:
                G.add_node(node, name=name, station=station_of(name), lat=None, lon=None,
                           lines=[], modes=[], transfers={}, origin="gtfs")
                matcher.register(node, name)
        match_stats[how] += 1
        station_node[st.station_key] = node
        d = G.nodes[node]
        d.setdefault("gtfs_station_ids", [])
        if st.station_key not in d["gtfs_station_ids"]:
            d["gtfs_station_ids"].append(st.station_key)
        lat, lon = ubahn_coords.get(st.station_key, (st.lat, st.lon))
        if d.get("lat") is None:
            d["lat"], d["lon"] = round(lat, 6), round(lon, 6)
            d["coord_source"] = "stations_with_ubahn.csv" if st.station_key in ubahn_coords else "gtfs"
    log(f"[2/5] {len(stations)} Stationen: " + ", ".join(f"{k}={v}" for k, v in sorted(match_stats.items())))

    # ── Kanten aus Fahrten
    seg_times: dict[tuple, list[int]] = defaultdict(list)
    seg_mode: dict[tuple, str] = {}
    line_variants: dict[str, dict] = {}
    for (line, mode, direction, trip), trip_df in rail.groupby(["line", "mode", "direction_id", "trip_id"]):
        trip_df = trip_df.sort_values("stop_sequence")
        nodes = [station_node[k] for k in trip_df["station_key"]]
        times = trip_df["t"].tolist()
        for a, b, ta, tb in zip(nodes, nodes[1:], times, times[1:]):
            if a == b:
                continue
            seg = (line, *sorted((a, b)))
            seg_mode[seg] = GTFS_MODES[mode]
            if ta is not None and tb is not None and tb >= ta:
                seg_times[seg].append(tb - ta)
            else:
                seg_times.setdefault(seg, [])
        # längste Fahrt je Richtung als Linienverlauf
        variant = line_variants.setdefault(line, {"mode": GTFS_MODES[mode], "dirs": {}})
        if len(nodes) > len(variant["dirs"].get(direction, ([], []))[0]):
            t0 = times[0] or 0
            variant["dirs"][direction] = (nodes, [round(((t or t0) - t0) / 60, 1) for t in times])

    line_key = {}
    for line, v in line_variants.items():
        existing = G.graph["lines"].get(line)
        line_key[line] = line if not existing or existing.get("mode") == v["mode"] else f"{line} ({v['mode']})"

    for seg, times in seg_times.items():
        line, a, b = seg
        tt = int(statistics.median(times)) if times else MIN_EDGE_SEC * 4
        mode = seg_mode[seg]
        G.add_edge(a, b, key=line_key[line], line=line_key[line], mode=mode, travel_time=max(MIN_EDGE_SEC, tt),
                   color=MODE_COLORS.get(mode, "#666666"), origin="gtfs")
    log(f"[3/5] {len(seg_times)} Schienen-Kanten aus GTFS")

    # ── U-Bahn-CSV: fehlende Kanten ergänzen
    added_csv, skipped_csv = 0, []
    if "U-Bahn" in modes:
        has_ubahn = {n for u, v, d in G.edges(data=True) if d["mode"] == "ubahn" for n in (u, v)}
        conn = pd.read_csv(ubahn_conn_csv, dtype=str)
        st_lines = {r.station_id: set(str(r.u_bahn_lines).split(",")) for r in ubahn_st.itertuples()}
        for r in conn.itertuples():
            a, b = station_node.get(r.station_id_1), station_node.get(r.station_id_2)
            if not a or not b or a == b:
                continue
            if any(d["mode"] == "ubahn" for d in G.get_edge_data(a, b, default={}).values()):
                continue
            if a in has_ubahn and b in has_ubahn:
                # beide Bahnhöfe schon im GTFS-Linienverlauf: CSV-Kante überspringt einen Halt
                skipped_csv.append(f"{G.nodes[a]['name']} – {G.nodes[b]['name']}")
                continue
            shared = sorted(st_lines.get(r.station_id_1, set()) & st_lines.get(r.station_id_2, set())) or ["U?"]
            (la, oa), (lb, ob) = ubahn_coords[r.station_id_1], ubahn_coords[r.station_id_2]
            tt = int(30 + haversine_km(la, oa, lb, ob) / UBAHN_KMH * 3600)
            for line in shared:
                G.add_edge(a, b, key=line, line=line, mode="ubahn", travel_time=tt, color=MODE_COLORS["ubahn"],
                           origin="ubahn_csv", estimated=True)
                added_csv += 1
    log(f"[4/5] U-Bahn-CSV: {added_csv} Kanten ergänzt, {len(skipped_csv)} übersprungen (GTFS hat Zwischenhalt): "
        + "; ".join(skipped_csv))

    # ── Fußwege Station <-> Bus-Haltepunkte derselben Station
    walk = 0
    rail_nodes = set(station_node.values())
    for node in rail_nodes:
        st = G.nodes[node].get("station")
        for other in matcher.by_station.get(st, []) if st else []:
            if other != node and not G.has_edge(node, other, key="walk"):
                G.add_edge(node, other, key="walk", line="Fußweg", mode="walk", travel_time=WALK_TIME_SEC,
                           color=MODE_COLORS["walk"], origin="gtfs_walk")
                walk += 1

    # ── Knotenattribute + Linienmetadaten
    for u, v, d in G.edges(data=True):
        if d.get("origin") in ("gtfs", "ubahn_csv"):
            for n in (u, v):
                nd = G.nodes[n]
                if d["line"] not in nd["lines"]:
                    nd["lines"] = sorted(nd["lines"] + [d["line"]])
                if d["mode"] not in nd["modes"]:
                    nd["modes"] = sorted(nd["modes"] + [d["mode"]])

    for line, v in line_variants.items():
        dirs = [v["dirs"][k] for k in sorted(v["dirs"])]
        G.graph["lines"][line_key[line]] = {
            "mode": v["mode"],
            "title": f"{G.nodes[dirs[0][0][0]]['name']} ◂▸ {G.nodes[dirs[0][0][-1]]['name']}",
            "valid_from": meta.get("valid_from"),
            "source": gtfs_csv.name,
            "source_type": "gtfs",
            "stops": list(dict.fromkeys(n for nodes, _ in dirs for n in nodes)),
            "variants": [nodes for nodes, _ in dirs],
            "variant_minutes": [mins for _, mins in dirs],
            "total_minutes": max(m[-1] for _, m in dirs),
        }

    # Bus-Knoten ohne Koordinaten: GTFS-Haltestelle mit exakt gleichem Namen
    bus_coords = (df[df["mode"] == "Bus"].assign(key=lambda x: x["stop_name"].map(lambda s: stop_key(clean_gtfs_name(s))))
                  .groupby("key")[["stop_lat", "stop_lon"]].mean())
    filled = 0
    for n, d in G.nodes(data=True):
        if d.get("lat") is None and n in bus_coords.index:
            d["lat"], d["lon"] = round(bus_coords.at[n, "stop_lat"], 6), round(bus_coords.at[n, "stop_lon"], 6)
            d["coord_source"] = "gtfs_bus"
            filled += 1
    with_coords = sum(1 for _, d in G.nodes(data=True) if d.get("lat") is not None)
    log(f"[5/5] {walk} Fußwege Station<->Bushalt, Koordinaten: {with_coords}/{G.number_of_nodes()} Knoten "
        f"({filled} Bus-Knoten per Namensgleichheit)")

    # Fußwege nach Entfernung: Namensgleichheit allein verbindet "U Leinestr."
    # nicht mit der Bushaltestelle "Hermannstr./Leinestr." – bei gesperrter
    # U8 war der Bahnhof dann vom Netz abgeschnitten.
    near = add_proximity_walks(G)
    log(f"      {near} Fußwege zwischen Haltestellen im Umkreis von {PROXIMITY_WALK_M} m")

    G.graph["merged"] = {"gtfs": str(gtfs_csv), "gtfs_valid": [meta.get("valid_from"), meta.get("valid_to")],
                         "modes": modes, "station_matches": dict(match_stats), "ubahn_csv_edges": added_csv,
                         "ubahn_csv_skipped": skipped_csv}


def main():
    ap = argparse.ArgumentParser(description="Bus-Graph um U-Bahn/S-Bahn/Tram aus GTFS erweitern")
    ap.add_argument("--graph", type=Path, default=HERE / "graph.json")
    ap.add_argument("--gtfs", type=Path, default=DATA / "derived" / "gtfs_route_stops.csv")
    ap.add_argument("--ubahn-connections", type=Path, default=DATA / "training dataset" / "berlin_ubahn_connections.csv")
    ap.add_argument("--ubahn-stations", type=Path, default=DATA / "training dataset" / "stations_with_ubahn.csv")
    ap.add_argument("--modes", default="U-Bahn,S-Bahn,Tram,Bus",
                    help="GTFS-Modes, kommagetrennt (Bus = nur Linien ohne Linienband-PDF)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(GTFS_MODES)
    if unknown:
        sys.exit(f"Unbekannte Modes: {unknown}. Erlaubt: {list(GTFS_MODES)}")

    G = load_bus_base(args.graph)
    before = (G.number_of_nodes(), G.number_of_edges())
    meta_path = args.gtfs.with_name("gtfs_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    merge(G, args.gtfs, args.ubahn_connections, args.ubahn_stations, modes, meta)
    export(G, args.graph.parent)
    log(f"\nVorher: {before[0]} Nodes, {before[1]} Edges -> Nachher: {G.number_of_nodes()} Nodes, "
        f"{G.number_of_edges()} Edges, {nx.number_connected_components(G)} Zusammenhangskomponenten")


if __name__ == "__main__":
    main()
