"""
network_mcp_tool.py – MCP-Server (stdio) für den Berliner ÖPNV-Graphen
======================================================================

Lädt graph.json (build_graph.py: Busse aus BVG-Linienband-PDFs; merge_transit.py: U-Bahn,
S-Bahn, Tram aus VBB-GTFS) und stellt
folgende Tools bereit:
  search_stop, get_stop_info, get_neighbors, find_path, find_path_by_time,
  get_line_stops, get_network_stats

Datengrenzen (werden auch dem Client als Server-Instructions mitgegeben):
  * Kanten: Bus/MetroBus/ExpressBus (Linienbänder), U-Bahn/S-Bahn/Tram (GTFS), Linien mit
    Suffix "SEV" = Schienenersatzverkehr, Fußwege zwischen Haltepunkten einer Station (3 min).
  * Regionalverkehr und Fähren nur als Umsteigeattribute (node["transfers"]).
  * travel_time: Bus = Durchschnitt laut Linienband (Minutenraster), Schiene = Median GTFS.
  * lat/lon für die meisten Knoten (U-Bahn-CSV, GTFS; Bushalte per Namensgleichheit).

Start:  python network_mcp_tool.py
"""

import sys
import json
import heapq
import difflib
import itertools
import unicodedata
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

GRAPH_PATH = Path(__file__).resolve().parent / "graph.json"

INSTRUCTIONS = """Berliner ÖPNV-Netz: Buslinien aus BVG-Linienbändern, U-Bahn/S-Bahn/Tram aus VBB-GTFS
(nach merge_transit.py), plus Fußwege zwischen Haltepunkten derselben Station (pauschal 3 min).
Regionalverkehr und Fähren stehen nur als Umsteigemöglichkeiten an Haltestellen (transfers).
Fahrzeiten: Bus = Durchschnitt laut Linienband, Schiene = Median der GTFS-Fahrten. Linien mit Suffix
"SEV" sind Schienenersatzverkehr. Haltestellen per Name oder stop_id angeben; bei Mehrdeutigkeit zuerst
search_stop verwenden. Koordinaten (lat/lon) sind für die meisten Knoten vorhanden."""

mcp = _Server("berlin_transit", instructions=INSTRUCTIONS)

_G: nx.MultiGraph | None = None


def graph() -> nx.MultiGraph:
    global _G
    if _G is None:
        if not GRAPH_PATH.exists():
            raise FileNotFoundError(f"{GRAPH_PATH} fehlt – zuerst build_graph.py ausführen.")
        with open(GRAPH_PATH, encoding="utf-8") as f:
            _G = json_graph.node_link_graph(json.load(f), edges="edges")
    return _G


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower()).replace("ß", "ss")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\bstrasse\b|\bstr\b", "str", s)
    return re.sub(r"[^a-z0-9+]+", " ", s).strip()


def _search(query: str, limit: int) -> list[tuple[float, str]]:
    G = graph()
    q = _norm(query)
    scored = []
    for sid, d in G.nodes(data=True):
        n = _norm(d["name"])
        if n == q:
            score = 1.0
        elif re.sub(r"^(s\+u|s|u) ", "", n) == q:  # "Alexanderplatz" -> "S+U Alexanderplatz"
            score = 0.95
        elif n.startswith(q):
            score = 0.9
        elif q in n:
            score = 0.8
        elif all(tok in n for tok in q.split()):
            score = 0.7
        else:
            score = difflib.SequenceMatcher(None, q, n).ratio() * 0.7
        if score >= 0.45:
            scored.append((score + min(G.degree(sid), 20) / 1000, sid))
    return sorted(scored, reverse=True)[:limit]


def _resolve(stop: str) -> str:
    """stop_id oder Name -> stop_id. Wirft ValueError mit Vorschlägen bei Unklarheit."""
    G = graph()
    if stop in G:
        return stop
    hits = _search(stop, 5)
    if hits and hits[0][0] >= 1.0:
        return hits[0][1]
    if len(hits) == 1 or (hits and hits[0][0] >= 0.8 and (len(hits) == 1 or hits[1][0] < 0.8)):
        return hits[0][1]
    suggestions = [f"{G.nodes[s]['name']} ({s})" for _, s in hits]
    raise ValueError(f"Haltestelle '{stop}' nicht eindeutig/gefunden. Vorschläge: {suggestions}")


def _stop_summary(sid: str) -> dict:
    d = graph().nodes[sid]
    return {"stop_id": sid, "name": d["name"], "lines": d["lines"], "modes": d["modes"]}


def _edges_between(u: str, v: str) -> list[dict]:
    return list(graph().get_edge_data(u, v, default={}).values())


def _legs(path: list[str], lines_used: list[str] | None = None) -> list[dict]:
    """Pfad in Fahrtabschnitte (gleiche Linie) zusammenfassen."""
    G = graph()
    hops = []
    prev_line = None
    for i, (u, v) in enumerate(zip(path, path[1:])):
        edges = _edges_between(u, v)
        if lines_used:
            e = next(e for e in edges if e["line"] == lines_used[i])
        else:
            same = [e for e in edges if e["line"] == prev_line]
            e = same[0] if same else min(edges, key=lambda e: e["travel_time"])
        hops.append((u, v, e))
        prev_line = e["line"]

    legs = []
    for line, group in itertools.groupby(hops, key=lambda h: h[2]["line"]):
        group = list(group)
        legs.append({
            "line": line,
            "mode": group[0][2]["mode"],
            "from": G.nodes[group[0][0]]["name"],
            "to": G.nodes[group[-1][1]]["name"],
            "stops": [G.nodes[group[0][0]]["name"]] + [G.nodes[h[1]]["name"] for h in group],
            "travel_time_min": round(sum(h[2]["travel_time"] for h in group) / 60, 1),
        })
    return legs


# ── MCP-Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def search_stop(query: str, limit: int = 10) -> dict:
    """Haltestellen per (Teil-)Name suchen, tolerant gegenüber Tippfehlern und 'Str.'/'Straße'.
    Liefert stop_id, Name, bediente Linien und Verkehrsmittel."""
    hits = _search(query, max(1, min(limit, 50)))
    return {"query": query, "results": [dict(_stop_summary(s), score=round(sc, 3)) for sc, s in hits]}


@mcp.tool()
def get_stop_info(stop: str) -> dict:
    """Details einer Haltestelle (stop_id oder Name): Linien mit Kanten im Graph, Umsteigemöglichkeiten
    laut Linienband (U/S/Tram/Bus/Regional/Fähre), Station, Grad und direkte Nachbarn."""
    G = graph()
    sid = _resolve(stop)
    d = G.nodes[sid]
    return {
        "stop_id": sid,
        "name": d["name"],
        "station": d.get("station"),
        "lines": d["lines"],
        "modes": d["modes"],
        "transfers": d["transfers"],
        "lat": d.get("lat"),
        "lon": d.get("lon"),
        "degree": G.degree(sid),
        "neighbors": sorted({G.nodes[n]["name"] for n in G.neighbors(sid)}),
    }


@mcp.tool()
def get_neighbors(stop: str) -> dict:
    """Direkte Nachbarhaltestellen (Adjazenzliste) mit Linie, Verkehrsmittel und Fahrzeit in Sekunden."""
    G = graph()
    sid = _resolve(stop)
    nbrs = [{"stop_id": v, "name": G.nodes[v]["name"], "line": d["line"], "mode": d["mode"],
             "travel_time": d["travel_time"]}
            for _, v, d in G.edges(sid, data=True)]
    return {"stop_id": sid, "name": G.nodes[sid]["name"],
            "neighbors": sorted(nbrs, key=lambda n: (n["line"], n["name"]))}


def _route(a: str, b: str, edge_cost, penalty: float) -> list[tuple[str, str | None]] | None:
    """Dijkstra über Zustände (Haltestelle, aktuelle Linie); ein Linienwechsel kostet `penalty`."""
    G = graph()
    counter = itertools.count()
    heap = [(0.0, next(counter), a, None)]
    best = {(a, None): 0.0}
    prev: dict[tuple, tuple] = {}
    goal = None
    while heap:
        cost, _, u, line = heapq.heappop(heap)
        if cost > best.get((u, line), float("inf")):
            continue
        if u == b:
            goal = (u, line)
            break
        for _, v, d in G.edges(u, data=True):
            nc = cost + edge_cost(d) + (penalty if line is not None and d["line"] != line else 0)
            state = (v, d["line"])
            if nc < best.get(state, float("inf")):
                best[state] = nc
                prev[state] = (u, line)
                heapq.heappush(heap, (nc, next(counter), v, d["line"]))

    if goal is None:
        return None
    states = [goal]
    while states[-1] in prev:
        states.append(prev[states[-1]])
    return states[::-1]


def _route_result(a: str, b: str, states) -> dict:
    G = graph()
    if states is None:
        return {"from": G.nodes[a]["name"], "to": G.nodes[b]["name"], "error": "Keine Verbindung im Netz."}
    path = [s[0] for s in states]
    legs = _legs(path, [s[1] for s in states[1:]])
    return {
        "from": G.nodes[a]["name"], "to": G.nodes[b]["name"],
        "travel_time_min": round(sum(l["travel_time_min"] for l in legs), 1),
        "transfers": max(0, len(legs) - 1),
        "hops": len(path) - 1,
        "legs": legs,
    }


@mcp.tool()
def find_path(from_stop: str, to_stop: str, transfer_penalty_hops: int = 3) -> dict:
    """Pfad mit den wenigsten Haltestellen-Abschnitten (Hops). Jeder Linienwechsel zählt zusätzlich
    transfer_penalty_hops Hops, damit nicht für einzelne Halte umgestiegen wird (0 = reine Hop-Zahl)."""
    a, b = _resolve(from_stop), _resolve(to_stop)
    return _route_result(a, b, _route(a, b, lambda d: 1, max(0, transfer_penalty_hops)))


@mcp.tool()
def find_path_by_time(from_stop: str, to_stop: str, transfer_penalty_min: float = 3.0) -> dict:
    """Schnellste Verbindung nach Fahrzeit. Jeder Linienwechsel kostet zusätzlich
    transfer_penalty_min Minuten (Wartezeit/Umstieg), um unrealistisches Hin- und Herspringen zu vermeiden."""
    a, b = _resolve(from_stop), _resolve(to_stop)
    penalty = max(0.0, transfer_penalty_min)
    result = _route_result(a, b, _route(a, b, lambda d: d["travel_time"], penalty * 60))
    if "legs" in result:
        result["total_with_transfer_penalty_min"] = round(result["travel_time_min"] + result["transfers"] * penalty, 1)
    return result


@mcp.tool()
def get_line_stops(line: str) -> dict:
    """Alle Haltestellen einer Linie in Fahrtreihenfolge (z. B. '100', 'M11', 'U7', 'S41', 'M4'), je
    Fahrtvariante bzw. Richtung, mit Fahrzeit ab Starthaltestelle in Minuten."""
    G = graph()
    lines = G.graph["lines"]
    key = line.strip().upper()
    if key not in lines:
        serving = sorted({G.nodes[n]["name"] for n, d in G.nodes(data=True)
                          if any(key in v for v in d["transfers"].values())})
        return {"line": key, "error": "Linie nicht als Linienband in den Daten.",
                "known_as_transfer_at": serving[:50], "available_lines": sorted(lines)}
    meta = lines[key]
    variants = [[{"stop_id": sid, "name": G.nodes[sid]["name"], "minute": minute}
                 for sid, minute in zip(ids, minutes)]
                for ids, minutes in zip(meta["variants"], meta["variant_minutes"])]
    return {"line": key, "mode": meta["mode"], "title": meta["title"], "valid_from": meta["valid_from"],
            "source": meta["source"], "n_stops": len(set(meta["stops"])), "variants": variants}


@mcp.tool()
def get_network_stats(top_n: int = 10) -> dict:
    """Kennzahlen des Netzes: Knoten, Kanten, Linien je Verkehrsmittel, Zusammenhangskomponenten
    und die wichtigsten Knoten (höchster Grad = meiste Kanten)."""
    G = graph()
    lines = G.graph["lines"]
    edge_modes, node_modes, lines_by_mode = {}, {}, {}
    for *_, d in G.edges(data=True):
        edge_modes[d["mode"]] = edge_modes.get(d["mode"], 0) + 1
    for _, d in G.nodes(data=True):
        for m in d["modes"]:
            node_modes[m] = node_modes.get(m, 0) + 1
    for meta in lines.values():
        lines_by_mode[meta["mode"]] = lines_by_mode.get(meta["mode"], 0) + 1
    hubs = sorted(G.degree, key=lambda x: (-x[1], x[0]))[:max(1, min(top_n, 100))]
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "lines": len(lines),
        "lines_by_mode": lines_by_mode,
        "edges_by_mode": edge_modes,
        "stops_by_mode_incl_transfers": node_modes,
        "connected_components": comps,
        "top_hubs": [dict(_stop_summary(s), degree=deg, distinct_neighbors=len(set(G.neighbors(s))))
                     for s, deg in hubs],
    }


if __name__ == "__main__":
    G = graph()  # früh scheitern, falls graph.json fehlt
    print(f"Graph geladen: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges, "
          f"{nx.number_connected_components(G)} Komponenten ({GRAPH_PATH.name})", file=sys.stderr)
    print("berlin_transit MCP-Server bereit, stdio ...", file=sys.stderr)
    mcp.run(transport="stdio")
