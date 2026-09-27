"""
build_graph.py – Berliner ÖPNV-Netz aus BVG-Linienband-PDFs -> NetworkX MultiGraph
===================================================================================

Quelldaten (data/Straßennetz/):
  * <Linie>_<Datum>.pdf   BVG-Linienbänder (je Linie 1 Seite, z. B. "100_2026-08-23.pdf",
                          "M11_2026-08-23.pdf", "X7_2025-12-14.pdf"). Enthalten:
                            - Haltestellenfolge (Namen schräg über der Linie)
                            - Marker je Halt: "·" (bedient), "▸"/"◂" (nur eine Richtung),
                              "··" (mehrere Fahrtvarianten = mehrere Zeitspalten)
                            - durchschnittliche Fahrzeit ab Start in Minuten (je Variante)
                            - Umsteigesymbole unter jedem Halt (Symbolfont "TransitFpla"):
                                "0b" Bus, "4t" Tram, "0f" Fähre, ":" S-Bahn,
                                ";" + RE/RB/FEX Regionalbahn, Font "PSUBahn" Ziffer -> U-Bahn
  * Netzkarten (Busliniennetz.pdf, S_U-Bahn-Liniennetz.pdf, ...) sind reine Grafik
    ohne Haltestellenfolge und werden übersprungen.

Das Layout wird koordinatenbasiert geparst (pypdf visitor_text), nicht über den Fließtext.

Output (neben diesem Skript):
  * graph.json        NetworkX node-link-Graph (edges="edges"), Linien in graph["lines"]
  * adjacency.json    stop_id -> [{neighbor, name, line, mode, travel_time}]
  * stops_index.json  normalisierter Name -> [stop_id], plus stop_id -> Name

Ausführen:
  python src/graph_db/build_graph.py --data-dir data/Straßennetz
"""

import re
import sys
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

import networkx as nx
from networkx.readwrite import json_graph
from pypdf import PdfReader

logging.disable(logging.CRITICAL)  # pypdf-Warnungen zu CFF-Fonts unterdrücken

# ── Konfiguration ────────────────────────────────────────────────────────────

LINE_FILE_RE = re.compile(r"^([A-Z]?\d+)_(\d{4}-\d{2}-\d{2})\d?\.pdf$", re.IGNORECASE)
MARKER_RE = re.compile(r"^[·▸◂]+$")
REGIONAL_RE = re.compile(r"^(RE|RB|FEX|IC|EC|ODEG)\d*", re.IGNORECASE)

MODE_COLORS = {
    "bus": "#A5027D",
    "metrobus": "#F39200",
    "express": "#A5027D",
    "tram": "#CC0000",
    "ubahn": "#115D91",
    "sbahn": "#008D4F",
    "regional": "#E3000F",
    "ferry": "#0098D4",
    "walk": "#999999",
}
HEADER_SYMBOL_MODES = {"0b": "bus", "4t": "tram", "0f": "ferry"}
WALK_TIME_SEC = 180          # Umstieg zwischen Haltepunkten derselben S/U-Station
MIN_EDGE_SEC = 30            # Linienband zeigt ganze Minuten; 0-Minuten-Abschnitte -> 30 s


def log(msg: str):
    print(msg, flush=True)


# ── Text-Normalisierung ──────────────────────────────────────────────────────

def clean_name(raw: str) -> str:
    s = " ".join(raw.replace("\n", " ").split())
    s = re.sub(r"\s+([.,/)])", r"\1", s)                                 # "Str ." -> "Str."
    s = re.sub(r"(?<=\w)-\s+(?=\w)", "-", s)                             # "Alt- Tempelhof" (Umbruch)
    s = re.sub(r"(?<![\wÄÖÜäöü])([A-ZÄÖÜ]) (?=[a-zäöüß])", r"\1", s)     # "T eltow" -> "Teltow"
    return s.strip()


def stop_key(name: str) -> str:
    """Stabile ID aus dem Haltestellennamen."""
    s = unicodedata.normalize("NFKD", name.lower())
    s = s.replace("ß", "ss")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def station_of(name: str) -> str | None:
    """'S+U Alexanderplatz/Memhardstr.' -> 'S+U Alexanderplatz' (nur Bahnhofs-Halte)."""
    m = re.match(r"^(S\+U|S|U)\s+(.+)$", name)
    if not m:
        return None
    base = m.group(2).split("/")[0].strip()
    base = re.sub(r"\s+Bhf\.?$", "", base)
    return base or None


def line_mode(line: str, header_mode: str) -> str:
    if header_mode != "bus":
        return header_mode
    if line.startswith("M"):
        return "metrobus"
    if line.startswith("X"):
        return "express"
    return "bus"


# ── PDF-Parsing ──────────────────────────────────────────────────────────────

class Frag:
    __slots__ = ("x", "y", "text", "font")

    def __init__(self, x, y, text, font):
        self.x, self.y, self.text, self.font = x, y, text, font

    @property
    def is_symbol(self):
        return "Fpla" in self.font

    @property
    def is_ubahn_symbol(self):
        return "PSUBahn" in self.font

    @property
    def is_bold(self):
        return self.font.endswith("-Bld")

    @property
    def is_regular(self):
        return not (self.is_symbol or self.is_ubahn_symbol or "-Bl" in self.font or "-Ita" in self.font)


def extract_fragments(page) -> list[Frag]:
    frags: list[Frag] = []

    def visitor(text, cm, tm, font_dict, _size):
        t = text.strip()
        if not t or not font_dict:
            return
        x = cm[4] + tm[4] * cm[0] + tm[5] * cm[2]
        y = cm[5] + tm[4] * cm[1] + tm[5] * cm[3]
        frags.append(Frag(x, y, t, str(font_dict.get("/BaseFont", ""))))

    page.extract_text(visitor_text=visitor)
    return frags


def parse_transfers(frags: list[Frag]) -> dict[str, set]:
    """Umsteige-Block unter einem Halt in {mode: {linien}} übersetzen (Stream-Reihenfolge)."""
    out: dict[str, set] = defaultdict(set)
    mode = None
    for f in frags:
        t = f.text
        if f.is_ubahn_symbol:
            if t.isdigit():
                out["ubahn"].add(f"U{t}")
            mode = None
        elif f.is_symbol:
            mode = {"0b": "bus", "4t": "tram", "0f": "ferry", ":": "sbahn", ";": "rail"}.get(t)
        elif f.is_bold:
            t = t.replace(" ", "")
            if REGIONAL_RE.match(t):
                out["regional"].add(t)
            elif re.fullmatch(r"S\d+", t):
                out["sbahn"].add(t)
            elif mode == "rail" and t.isdigit():
                out["ubahn"].add(f"U{t}")
        elif f.is_regular and mode in ("bus", "tram", "ferry"):
            for tok in t.split():
                if re.fullmatch(r"[A-Z]{0,2}\d{1,3}[A-Z]?", tok):
                    out[mode].add(tok)
    return out


def drop_duplicate_layer(frags: list[Frag]) -> list[Frag]:
    """Manche PDFs (z. B. X34) enthalten das Linienband zweimal, vertikal versetzt.
    Erkennung: viele identische Fragmente (Text, Font, x) mit demselben y-Versatz."""
    by_key = defaultdict(list)
    for f in frags:
        by_key[(f.text, f.font, round(f.x, 1))].append(f.y)
    offsets = defaultdict(int)
    for ys in by_key.values():
        for a in ys:
            for b in ys:
                if 15 < a - b < 60:
                    offsets[round(a - b, 1)] += 1
    if not offsets:
        return frags
    d, count = max(offsets.items(), key=lambda kv: kv[1])
    n_markers = sum(1 for f in frags if MARKER_RE.match(f.text))
    if count < max(20, 0.4 * n_markers):
        return frags
    return [f for f in frags
            if not any(abs(a - f.y - d) <= 0.3 for a in by_key[(f.text, f.font, round(f.x, 1))])]


def cluster_levels(values, gap: float = 2) -> list[list[int]]:
    """Ganzzahlige y-Versätze zu Ebenen gruppieren, oberste Ebene zuerst."""
    clusters: list[list[int]] = []
    for v in sorted(set(values), reverse=True):
        if clusters and clusters[-1][-1] - v <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def parse_line_pdf(path: Path) -> dict | None:
    page = PdfReader(str(path)).pages[0]
    frags = extract_fragments(page)
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    top = height - 55  # Kopfzeile (Liniennummer, Titel); Seitenformate variieren (A4/A3)

    # Kopf: Verkehrsmittel-Symbol + Liniennummer (oben links) + Titel
    header_mode = "bus"
    line_label = None
    title = ""
    for f in frags:
        if f.y > top and f.x < 60 and f.is_symbol and f.text in HEADER_SYMBOL_MODES:
            header_mode = HEADER_SYMBOL_MODES[f.text]
        if f.y > top and "-Blk" in f.font and line_label is None:
            line_label = f.text
        if f.y > top + 15 and f.is_bold and ("◂" in f.text or "▸" in f.text) and len(f.text) > 3:
            title = clean_name(f.text)

    full_text = page.extract_text() or ""
    m = re.search(r"Gültig ab/Valid as of:\s*([^\n]+)", full_text)
    valid_from = m.group(1).strip() if m else None

    frags = drop_duplicate_layer(frags)

    # Marker = ein Halt bzw. eine Fahrtvariante an einem Halt; rechter Rand = Legende
    markers = [f for f in frags if f.is_bold and MARKER_RE.match(f.text) and 60 < f.y < top and f.x < width - 40]
    if not markers:
        return None

    # Bänder: Marker per y-Lücke clustern (Varianten-Marker liegen ~11 pt versetzt im selben Band)
    bands: list[list[Frag]] = []
    for mk in sorted(markers, key=lambda f: -f.y):
        if bands and bands[-1][-1].y - mk.y <= 15:
            bands[-1].append(mk)
        else:
            bands.append([mk])

    raw_stops = []
    for band in bands:
        band_top = max(m.y for m in band)
        # Halt = alle Marker eines Bands an derselben x-Position
        columns: list[list[Frag]] = []
        for mk in sorted(band, key=lambda f: f.x):
            if columns and abs(columns[-1][0].x - mk.x) <= 3:
                columns[-1].append(mk)
            else:
                columns.append([mk])

        for i, col in enumerate(columns):
            sx = col[0].x
            low = min(m.y for m in col)
            next_x = columns[i + 1][0].x if i + 1 < len(columns) else sx + 45
            name_parts = [f for f in frags if f.is_regular and band_top + 9 <= f.y <= band_top + 20
                          and sx - 1 <= f.x <= sx + 7 and not f.text.isdigit()]
            times = [f for f in frags if f.is_regular and f.text.isdigit()
                     and sx - 2 <= f.x <= sx + 4 and low - 4 <= f.y <= band_top + 11]
            transfer_frags = [f for f in frags if band_top - 90 < f.y < low - 2
                              and sx - 3 <= f.x < next_x - 3 and len(f.text) <= 8]
            if not name_parts:
                continue
            name = clean_name(" ".join(p.text for p in sorted(name_parts, key=lambda f: (f.x, -f.y))))
            if len(name) > 60 or "siehe" in name:  # Fußnotentext, kein Haltestellenname
                continue
            raw_stops.append({
                "name": name,
                "marker": "".join(m.text for m in sorted(col, key=lambda m: -m.y)),
                "glyphs": [(round(m.y - band_top), m.text) for m in col],
                "times": [(round(t.y - band_top), int(t.text)) for t in times],
                "transfers": parse_transfers(transfer_frags),
            })

    if len(raw_stops) < 2:
        return None

    # Fahrtvarianten = Zeitspalten, erkannt am y-Versatz der Minutenwerte zur Bandoberkante
    clusters = cluster_levels(dy for s in raw_stops for dy, _ in s["times"])
    n_cols = max(1, len(clusters))
    # Marker-Ebene (y-Versatz des Glyphs) = erste bediente Variante, Punktanzahl = Anzahl
    # bedienter Varianten ab dort ("···" auf dem Stamm = alle drei, auch wenn nur zwei
    # Zeitspalten gedruckt sind)
    marker_levels = cluster_levels(dy for s in raw_stops for dy, _ in s["glyphs"])

    def col_of(dy, levels):
        for ci, cl in enumerate(levels):
            if min(cl) - 1 <= dy <= max(cl) + 1:
                return ci
        return 0

    variants: list[list[tuple[str, int | None]]] = [[] for _ in range(n_cols)]
    for idx, s in enumerate(raw_stops):
        tmap = {col_of(dy, clusters): v for dy, v in s["times"]}
        served = set(tmap)
        for dy, glyph in s["glyphs"]:
            first = col_of(dy, marker_levels)
            served |= set(range(first, first + len(glyph)))
        for c in sorted(c for c in served if c < n_cols):
            variants[c].append((s["name"], tmap.get(c, 0 if idx == 0 else None)))

    variants = [v for v in variants if len(v) >= 2]
    return {
        "line": path.name.split("_")[0],
        "line_label": line_label,
        "mode": line_mode(path.name.split("_")[0], header_mode),
        "title": title,
        "valid_from": valid_from,
        "source": path.name,
        "stops": raw_stops,
        "variants": variants,
    }


def interpolate(seq: list[tuple[str, int | None]]) -> list[tuple[str, float]]:
    """Fehlende Minutenwerte (Halte nur in Gegenrichtung) linear interpolieren."""
    vals = [t for _, t in seq]
    out = []
    for i, (name, t) in enumerate(seq):
        if t is None:
            prev_i = next((j for j in range(i - 1, -1, -1) if vals[j] is not None), None)
            next_i = next((j for j in range(i + 1, len(vals)) if vals[j] is not None), None)
            if prev_i is not None and next_i is not None:
                t = vals[prev_i] + (vals[next_i] - vals[prev_i]) * (i - prev_i) / (next_i - prev_i)
            elif prev_i is not None:
                t = vals[prev_i] + (i - prev_i)
            elif next_i is not None:  # Zweig beginnt auf dem Stamm: ~1 min je Halt rückwärts
                t = max(0, vals[next_i] - (next_i - i))
            else:
                t = 0
        out.append((name, float(t)))
    return out


# ── Graph-Aufbau ─────────────────────────────────────────────────────────────

def select_line_files(data_dir: Path) -> list[Path]:
    """Pro Linie die aktuellste Datei; Duplikate wie 'xyz (1).pdf' werden ignoriert."""
    best: dict[str, tuple[str, Path]] = {}
    for p in sorted(data_dir.glob("*.pdf")):
        m = LINE_FILE_RE.match(p.name)
        if not m:
            continue
        line, date = m.group(1).upper(), m.group(2)
        if line not in best or date > best[line][0]:
            best[line] = (date, p)
    return [p for _, p in sorted(best.values(), key=lambda v: v[1].name)]


def build_graph(data_dir: Path) -> nx.MultiGraph:
    G = nx.MultiGraph(name="Berlin ÖPNV (BVG-Linienbänder)", source=str(data_dir))
    lines_meta = {}
    skipped = []

    files = select_line_files(data_dir)
    log(f"[1/3] {len(files)} Linienband-PDFs gefunden, parse ...")

    for p in files:
        try:
            parsed = parse_line_pdf(p)
        except Exception as e:  # defekte PDF nicht den ganzen Build abbrechen lassen
            skipped.append(f"{p.name}: {e}")
            continue
        if not parsed or not parsed["variants"]:
            skipped.append(f"{p.name}: kein Linienband erkannt")
            continue

        line, mode = parsed["line"], parsed["mode"]

        for s in parsed["stops"]:
            sid = stop_key(s["name"])
            if sid not in G:
                G.add_node(sid, name=s["name"], station=station_of(s["name"]), lat=None, lon=None,
                           lines=set(), modes=set(), transfers=defaultdict(set))
            nd = G.nodes[sid]
            nd["lines"].add(line)
            nd["modes"].add(mode)
            for tmode, tlines in s["transfers"].items():
                nd["transfers"][tmode] |= tlines
                nd["modes"].add(tmode)

        seen_pairs = set()
        variant_ids, variant_minutes = [], []
        for var in parsed["variants"]:
            seq = interpolate(var)
            ids = [stop_key(n) for n, _ in seq]
            variant_ids.append(ids)
            variant_minutes.append([round(t, 1) for _, t in seq])
            for (a, ta), (b, tb) in zip(zip(ids, [t for _, t in seq]), zip(ids[1:], [t for _, t in seq[1:]])):
                if a == b:
                    continue
                pair = tuple(sorted((a, b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                G.add_edge(a, b, key=line, line=line, mode=mode,
                           travel_time=int(max(MIN_EDGE_SEC, round((tb - ta) * 60))),
                           color=MODE_COLORS.get(mode, "#666666"))

        lines_meta[line] = {
            "mode": mode,
            "title": parsed["title"],
            "valid_from": parsed["valid_from"],
            "source": parsed["source"],
            "stops": [stop_key(s["name"]) for s in parsed["stops"]],
            "variants": variant_ids,
            "variant_minutes": variant_minutes,
            "total_minutes": max((v[-1][1] or 0) for v in parsed["variants"]),
        }

    # Fußwege zwischen Haltepunkten derselben S/U-Station
    log("[2/3] Umstiegskanten zwischen Haltepunkten gleicher S/U-Stationen ...")
    by_station = defaultdict(list)
    for sid, d in G.nodes(data=True):
        if d["station"]:
            by_station[d["station"]].append(sid)
    walk_edges = 0
    for members in by_station.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                G.add_edge(a, b, key="walk", line="Fußweg", mode="walk",
                           travel_time=WALK_TIME_SEC, color=MODE_COLORS["walk"])
                walk_edges += 1

    # Sets -> sortierte Listen (JSON-fähig)
    for _, d in G.nodes(data=True):
        d["lines"] = sorted(d["lines"])
        d["transfers"] = {k: sorted(v) for k, v in sorted(d["transfers"].items())}
        d["modes"] = sorted(d["modes"])

    G.graph["lines"] = lines_meta
    G.graph["skipped_files"] = skipped
    G.graph["walk_edges"] = walk_edges
    log(f"      {len(lines_meta)} Linien geparst, {len(skipped)} übersprungen, {walk_edges} Fußwege")
    for s in skipped:
        log(f"      - übersprungen: {s}")
    return G


def export(G: nx.MultiGraph, out_dir: Path):
    log(f"[3/3] Exportiere nach {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json_graph.node_link_data(G, edges="edges")
    (out_dir / "graph.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    adjacency = {}
    for sid in G.nodes:
        entries = []
        for _, nbr, d in G.edges(sid, data=True):
            entries.append({"neighbor": nbr, "name": G.nodes[nbr]["name"], "line": d["line"],
                            "mode": d["mode"], "travel_time": d["travel_time"]})
        adjacency[sid] = sorted(entries, key=lambda e: (e["line"], e["neighbor"]))
    (out_dir / "adjacency.json").write_text(json.dumps(adjacency, ensure_ascii=False, indent=1), encoding="utf-8")

    by_name = defaultdict(list)
    for sid, d in G.nodes(data=True):
        by_name[d["name"].lower()].append(sid)
    index = {"by_name": dict(sorted(by_name.items())),
             "by_id": {sid: d["name"] for sid, d in sorted(G.nodes(data=True))}}
    (out_dir / "stops_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    for f in ("graph.json", "adjacency.json", "stops_index.json"):
        log(f"      {f:18s} {(out_dir / f).stat().st_size / 1024:8.1f} KB")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Berliner ÖPNV-Graph aus BVG-Linienband-PDFs bauen")
    ap.add_argument("--data-dir", type=Path, default=here.parent.parent / "data" / "Straßennetz",
                    help="Ordner mit den Linien-PDFs")
    ap.add_argument("--out-dir", type=Path, default=here, help="Zielordner (Default: graph_db/)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    if not args.data_dir.is_dir():
        sys.exit(f"Datenordner nicht gefunden: {args.data_dir}")

    G = build_graph(args.data_dir)
    export(G, args.out_dir)
    log(f"\nFertig: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges, "
        f"{nx.number_connected_components(G)} Zusammenhangskomponenten")


if __name__ == "__main__":
    main()
