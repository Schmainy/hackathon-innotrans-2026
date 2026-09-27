#!/usr/bin/env python3
"""
ubahn_engine.py - deterministic core for the InnoTrans 2026 U-Bahn operator agent.

Reverse-engineered generative model of the hackathon data (fitted on 10 Jun - 21 Sep 2026):

    flow[s, t] = min(ceiling[s], floor(lambda[s, t] * Exp(1)))          (independent per cell)
    lambda[s, t] = w[s] * P[daytype, time_of_day] * heat(T_day) * rain(prcp_t) + event_load[s, t]

    heat(T)    = exp(-0.0185 * (T_day_mean - 18))        daily mean temperature, degC
    rain(p)    = exp(0.29 * (1 - exp(-p / 0.69)))         15-min precipitation (mm/h), up to about +34 %
    events     = ~57 % of attendance arrives by U-Bahn over the 90 min up to the start slot (ramping up),
                 ~55 % leaves over ~2 h after the listed end, most of it in the first 45 min,
                 spread over stations within ~2 km of the venue (Gaussian weights, sigma 0.6 km)
    station closure  -> flow = 0 for slots in [start, end); ~56 % of the lost volume re-appears in the
                        first 45 min after reopening (39 % / 12 % / 5 % per slot)
    line suspension  -> no measurable effect on any station flow or on energy

The LLM agent should call these methods as tools and do the talking; every number it quotes
should come from here or from the raw data. Full documentation: README.md next to this file.

Usage:
    eng = UBahnEngine('/path/to/data')          # loads every flows*/weather*/events*/closures*/energy* csv
    eng.explain('Warschauer Str.', '2026-06-17 22:45', '2026-06-18 00:30')
    eng.event_impact('Uber Arena', '2026-09-30 21:00', end='2026-09-30 23:15', attendance=17000)
    eng.capacity_risk('2026-09-22 05:00', '2026-09-23 00:45', weather={'tmean': 14, 'prcp': 1.5}, events=[...])
    eng.reroute_options('U8', 'Hermannplatz', 'Leinestr.')
"""
from __future__ import annotations

import difflib
import glob
import os
import csv
import re
import unicodedata

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.loader import EVENT_COLUMNS     # one definition of the events schema (headerless test file)

__version__ = '0.9.4'

LINES = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9']
SERVICE_START_HOUR = 5          # service day runs 05:00 -> 00:45 next calendar day
TOD_ORDER = [f'{h:02d}:{m:02d}' for h in list(range(5, 24)) + [0] for m in (0, 15, 30, 45)
             if not (h == 0 and m > 45)]                        # 80 slots, 05:00 ... 00:45
BERLIN_HOLIDAYS_2026 = {'2026-01-01', '2026-03-08', '2026-04-03', '2026-04-06', '2026-05-01',
                        '2026-05-14', '2026-05-25', '2026-10-03', '2026-12-25', '2026-12-26'}

# Event kernels (share of attendance per 15-min slot, summed over the venue's stations).
# Offsets are in slots relative to the listed start / end (slot 0 = the slot starting at that time).
# (censored-exponential MLE over all 2026 summer events; ~57 % arrive, ~55 % leave by U-Bahn)
ARRIVAL_KERNEL = {-6: 0.015, -5: 0.025, -4: 0.057, -3: 0.087, -2: 0.115, -1: 0.128, 0: 0.146, 1: 0.010}
DEPARTURE_KERNEL = {0: 0.130, 1: 0.140, 2: 0.109, 3: 0.080, 4: 0.049, 5: 0.025, 6: 0.011, 7: 0.004}
VENUE_SIGMA_KM, VENUE_RADIUS_KM = 0.6, 2.0
# Pent-up demand after a closed station reopens: share of the volume lost during the closure that
# re-appears in the 1st, 2nd, 3rd slot after reopening (fitted, censored MLE; ~56 % comes back).
REBOUND_FRACTIONS = [0.39, 0.12, 0.05]

# Venue coordinates (lat, lon) + aliases. Positions for the less-known addresses were placed
# where the flow data says the simulator put them.
VENUES = {
    'Uber Arena (ex Mercedes-Benz Arena)': (52.50535, 13.44385, ['uber-platz 1', 'uber arena', 'mercedes-benz arena', 'mercedes benz arena', 'o2 world']),
    'Uber Eats Music Hall': (52.5046, 13.4418, ['uber-platz 2', 'uber eats music hall', 'verti music hall']),
    'Olympiastadion': (52.5146, 13.2395, ['olympischer platz 3', 'olympiastadion', 'olympic stadium']),
    'Olympiapark': (52.5160, 13.2330, ['friedrich-friesen-allee', 'olympiapark', 'lollapalooza']),
    'Waldbuehne': (52.5173, 13.2286, ['am glockenturm 1', 'waldbuhne', 'waldbuehne']),
    'Zitadelle Spandau': (52.5410, 13.2128, ['am juliusturm', 'zitadelle spandau', 'citadel spandau']),
    'Kulturbrauerei': (52.5389, 13.4128, ['schonhauser allee 36', 'kulturbrauerei']),
    'Huxleys Neue Welt': (52.4868, 13.4211, ['hasenheide 107', 'huxleys']),
    'Columbiahalle': (52.4839, 13.3930, ['columbiadamm 13', 'columbiahalle']),
    'Columbia Theater': (52.4842, 13.3915, ['columbiadamm 9', 'columbia theater']),
    'Tempodrom': (52.5009, 13.3805, ['mockernstrasse 10', 'tempodrom']),
    'Metropol / Mikropol': (52.4995, 13.3530, ['nollendorfplatz 5', 'metropol', 'mikropol']),
    'Admiralspalast': (52.5205, 13.3886, ['friedrichstrasse 101', 'admiralspalast']),
    'Lido': (52.4991, 13.4448, ['cuvrystrasse 7', 'lido']),
    'Berghain': (52.5111, 13.4430, ['am wriezener bahnhof', 'berghain']),
    'Astra Kulturhaus': (52.5073, 13.4550, ['revaler strasse 99', 'astra kulturhaus']),
    'Heimathafen Neukoelln': (52.4766, 13.4404, ['karl-marx-strasse 141', 'heimathafen']),
    'Theater des Westens': (52.5059, 13.3290, ['kantstrasse 12', 'theater des westens']),
    'Konzerthaus / Gendarmenmarkt': (52.5137, 13.3918, ['gendarmenmarkt', 'konzerthaus', 'classic open air']),
    'Stage Theater am Potsdamer Platz': (52.5077, 13.3727, ['marlene-dietrich-platz 1', 'punch line', 'stage theater']),
    'Velodrom': (52.5318, 13.4472, ['paul-heyse-strasse 26', 'velodrom']),
    'Holzmarkt 25': (52.5114, 13.4265, ['holzmarktstrasse 25', 'holzmarkt 25']),
    'Holzmarktstrasse 15': (52.5145, 13.4200, ['holzmarktstrasse 15']),
    'Privatclub (Skalitzer Str. 85-86)': (52.4998, 13.4330, ['skalitzer strasse 85', 'privatclub']),
    'Skalitzer Str. 134': (52.4990, 13.4175, ['skalitzer str. 134', 'skalitzer strasse 134']),
    'Schlesisches Tor (in station)': (52.5011, 13.4418, ['im u-bhf. schlesisches tor']),
    'Obentrautstrasse 19-21': (52.4962, 13.3840, ['obentrautstrasse 19']),
    'Hermannstrasse 146': (52.4665, 13.4395, ['hermannstrasse 146']),
    'Soemmeringstrasse 15': (52.5230, 13.3085, ['sommeringstrasse 15']),
    'Metzer Strasse 2': (52.5295, 13.4100, ['metzer strasse 2']),
    'Lenaustrasse 7': (52.4913, 13.4270, ['lenaustrasse 7', 'oblomov']),
    'Budapester Strasse 45': (52.5055, 13.3380, ['budapester strasse 45']),
    'Tempelhofer Damm 85': (52.4720, 13.3870, ['tempelhofer damm 85', 'luftschloss tempelhofer feld']),
    'Maerkisches Ufer 48': (52.5125, 13.4100, ['markisches ufer 48']),
    'RSO Valley (Schoeneweide)': (52.4555, 13.5120, ['schnellerstrasse 137', 'rso valley']),
    'Kesselhaus Vagabund': (52.5537, 13.3519, ['oudenarder str. 16', 'vagabund']),
    'Haus der Visionaere': (52.4945, 13.4622, ['eichenstrasse 4', 'haus der visionare']),
    'Invalidenstrasse': (52.5300, 13.3790, ['invalidenstrasse']),
    'FEZ Wuhlheide': (52.4598, 13.5372, ['strasse am fez 4', 'fez wuhlheide']),
    'Lilli-Henoch-Strasse 10': (52.5357, 13.4336, ['lilli-henoch-strasse 10', 'zirkus mond']),
    'Kurt-Schumacher-Damm 207': (52.5580, 13.3100, ['kurt-schumacher-damm 207']),
    'Tempelhofer Feld (Platz der Luftbruecke)': (52.4841, 13.3872, ['platz der luftbrucke', 'tempelhofer feld']),
    'Treptower Str. 39': (52.4822, 13.4480, ['treptower str. 39', 'beach neukolln']),
    'Wilhelm-Kabus-Strasse 24': (52.4780, 13.3600, ['wilhelm-kabus-strasse 24']),
    'Messe Berlin / InnoTrans': (52.5031, 13.2757, ['messedamm 22', 'messe berlin', 'innotrans', 'expocenter city', 'icc berlin', 'messe']),
    'Max-Schmeling-Halle': (52.5433, 13.4031, ['max-schmeling-halle', 'am falkplatz']),
    'Stadion An der Alten Foersterei': (52.4573, 13.5681, ['alte forsterei', 'an der wuhlheide 263']),
    'Brandenburger Tor': (52.5163, 13.3777, ['brandenburger tor', 'pariser platz', 'strasse des 17. juni', 'fanmeile']),
}

MOJIBAKE = re.compile('[\u00c3\u00c2][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc\u2013-\u2122]|\u00e2\u20ac')


def repair_text(s):
    """Undo a common encoding accident: UTF-8 text read as Windows-1252 and saved again
    ('JannowitzbrÃ¼cke' -> 'Jannowitzbrücke'). Text without the tell-tale characters is returned as is."""
    if not isinstance(s, str) or not MOJIBAKE.search(s):
        return s
    for enc in ('cp1252', 'latin-1'):
        try:
            return s.encode(enc).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def _rescue_rows(df, sep):
    """A stray quote mark at the start of a line makes the parser put the whole line in the first field and
    leave the others empty. Such lines are split again when that gives exactly one value per column."""
    if df.shape[1] < 3 or len(df) == 0:
        return
    first, rest = df.columns[0], list(df.columns[1:])
    text = df[first].astype(str)
    mask = df[rest].isna().all(axis=1) & (text.str.count(re.escape(sep)) >= len(rest))
    for i in df.index[mask]:
        line = str(df.at[i, first]).replace('\t', ' ').strip().strip('"')
        fields = next(csv.reader([line], delimiter=sep))
        if len(fields) != df.shape[1]:
            continue
        for c, v in zip(df.columns, fields):
            v = v.strip().strip('"').strip()
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                df.at[i, c] = pd.to_numeric(v, errors='coerce') if v else np.nan
            else:
                df.at[i, c] = v if v else np.nan


def read_csv_any(path, **kw):
    """Read a CSV the way people actually deliver them: UTF-8 with or without a byte-order mark, or
    Windows-1252; comma, semicolon or tab separated (semicolon files use a decimal comma); garbled
    accents repaired in column names and text cells."""
    with open(path, 'rb') as fh:
        first = fh.readline()
    enc = 'utf-8-sig'
    try:
        line = first.decode('utf-8-sig')
    except UnicodeDecodeError:
        enc, line = 'cp1252', first.decode('cp1252', errors='replace')
    counts = {d: line.count(d) for d in (',', ';', '\t')}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ','
    opts = dict(sep=sep, encoding=enc)
    if sep == ';':
        opts['decimal'] = ','
    try:
        df = pd.read_csv(path, **opts, **kw)
    except UnicodeDecodeError:
        opts['encoding'] = 'cp1252'
        df = pd.read_csv(path, **opts, **kw)
    df.columns = [repair_text(str(c)).replace('\ufeff', '').replace('\u00ef\u00bb\u00bf', '').strip() for c in df.columns]
    _rescue_rows(df, sep)
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype):
            df[c] = df[c].map(repair_text)
    return df


def read_events_csv(path, **kw):
    """Events file. The official test file (berlin_events_summer_2026_rest.csv) has NO header row: its first
    line is already an event (InnoTrans, 22 September). Same rule as src/loader.py's load_events: if the
    header lacks 'event_name', re-read with header=None and the loader's column names."""
    df = read_csv_any(path, **kw)
    if 'event_name' not in df.columns:
        df = read_csv_any(path, header=None, names=EVENT_COLUMNS, **kw)
    return df


STATION_ALIASES = {'kudamm': 'Kurfürstendamm', 'ku damm': 'Kurfürstendamm', "ku'damm": 'Kurfürstendamm',
                   'zoologischer garten': 'Zoologischer Garten Bhf', 'zoo': 'Zoologischer Garten Bhf', 'hbf': 'Berlin Hauptbahnhof', 'hauptbahnhof': 'Berlin Hauptbahnhof',
                   'alex': 'Alexanderplatz Bhf', 'alexanderplatz': 'Alexanderplatz Bhf', 'kotti': 'Kottbusser Tor',
                   'friedrichstrasse': 'Friedrichstr. Bhf', 'lichtenberg': 'Lichtenberg Bhf',
                   'gesundbrunnen': 'Gesundbrunnen Bhf', 'potsdamer platz': 'Potsdamer Platz Bhf',
                   'checkpoint charlie': 'Kochstr. (Checkpoint Charlie)', 'thielplatz': 'Freie Universität (Thielplatz)'}


def _ascii(s: str) -> str:
    s = s.replace('ß', 'ss')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()


def _norm_station(s: str) -> str:
    s = _ascii(s)
    s = re.sub(r'\(berlin\)', '', s)
    s = re.sub(r'^\s*(s\+u|u)\s+', '', s)
    s = re.sub(r'strasse|str\.', 'str', s)
    s = re.sub(r'\bbhf\b|\bbahnhof\b', '', s) if 'gorlitzer' not in s else s
    s = re.sub(r'[^a-z0-9()]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _parse_duration(s: str) -> pd.Timedelta:
    h = re.search(r'(\d+)\s*h', str(s))
    m = re.search(r'(\d+)\s*min', str(s))
    return pd.Timedelta(hours=int(h.group(1)) if h else 0, minutes=int(m.group(1)) if m else 0)


def _to_dt(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize(None) if t.tzinfo is not None else t


class UBahnEngine:
    # ------------------------------------------------------------------ loading
    def __init__(self, data_dir: str = '/mnt/user-data/uploads', fit: bool = True):
        """Load every dataset in `data_dir`, build the network graph and fit the model (~5 s).

        Args:
            data_dir: folder holding stations_with_ubahn.csv, berlin_ubahn_connections.csv and any
                number of flows*, weather*, berlin_events*, closures* and energy* CSV files, plus an
                optional venues_extra.csv. Training and evaluation files are merged; on duplicate
                timestamps the most recently added file wins.
            fit: False loads the data without fitting (call fit() yourself).
        """
        self.data_dir = data_dir
        self._load()
        self._build_network()
        if fit:
            self.fit()

    def _csvs(self, stem):
        # oldest first: where files overlap, the most recently added file wins
        return sorted(glob.glob(os.path.join(self.data_dir, f'{stem}*.csv')), key=lambda f: (os.path.getmtime(f), f))

    @staticmethod
    def _read_ts(series, fmt='%m/%d/%Y %H:%M'):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            return pd.to_datetime(series, format='mixed')

    def _load_extra_venues(self):
        """Optional venues_extra.csv in the data folder (columns: name, lat, lon, aliases separated by ';')
        adds or overrides venue coordinates, e.g. for venues that appear only in new data."""
        path = os.path.join(self.data_dir, 'venues_extra.csv')
        self.extra_venues = []
        if os.path.exists(path):
            for r in pd.read_csv(path).itertuples():
                aliases = [_ascii(a.strip()) for a in str(getattr(r, 'aliases', '') or '').split(';') if a.strip() and a.strip() != 'nan']
                VENUES[str(r.name)] = (float(r.lat), float(r.lon), aliases + [_ascii(str(r.name))])
                self.extra_venues.append(str(r.name))

    def _load(self):
        self.loaded_files = {stem: [os.path.basename(f) for f in self._csvs(stem)]
                             for stem in ('flows', 'weather', 'berlin_events', 'closures', 'energy')}
        st = pd.read_csv(os.path.join(self.data_dir, 'stations_with_ubahn.csv'))
        base = st.station_name.str.replace(r'^(S\+U|U)\s+', '', regex=True).str.replace(' (Berlin)', '', regex=False).str.strip()
        dup = base.duplicated(keep=False)
        st['key'] = [f'{b} ({l})' if d else b for b, l, d in zip(base, st.u_bahn_lines, dup)]
        st['lines'] = st.u_bahn_lines.str.split(',')
        self.stations = st.set_index('key')
        self.keys = list(st.key)
        self._norm_index = {}
        for k in self.keys:
            self._norm_index.setdefault(_norm_station(k), []).append(k)
        # flows: map columns to keys by name, duplicates by order of appearance
        frames = []
        for f in self._csvs('flows'):
            x = read_csv_any(f)
            ts = self._read_ts(x.iloc[:, 0])
            used, cols = {}, []
            for c in x.columns[1:]:
                nm = re.sub(r'\.\d+$', '', c)
                cand = list(st.key[st.station_name == nm])
                if not cand:                                   # tolerate renamed headers
                    cand = [k for k in st.key if _norm_station(k) == _norm_station(nm)]
                i = used.get(nm, 0)
                used[nm] = i + 1
                cols.append(cand[i] if i < len(cand) else nm)
            x = x.iloc[:, 1:]
            x.columns = cols
            x.index = ts
            frames.append(x)
        fl = pd.concat(frames)
        fl = fl[~fl.index.duplicated(keep='last')].sort_index()
        self.flows = fl[self.keys].astype(float)
        # weather
        frames = []
        for f in self._csvs('weather'):
            w = read_csv_any(f)
            w.index = self._read_ts(w.iloc[:, 0])
            frames.append(w.iloc[:, 1:])
        we = pd.concat(frames)
        self.weather = we[~we.index.duplicated(keep='last')].sort_index()
        # events
        frames = []
        for f in self._csvs('berlin_events'):
            frames.append(read_events_csv(f))
        ev = pd.concat(frames, ignore_index=True).drop_duplicates()
        ev['start'] = pd.to_datetime(ev.began_local, utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        ev['end'] = pd.to_datetime(ev.estimated_end_local, utc=True, errors='coerce').dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        ev['end'] = ev['end'].fillna(ev['start'] + pd.to_timedelta(np.where(ev.segment.eq('Music'), 3, 2), unit='h'))
        ev['address'] = ev.address.astype(str).str.strip()
        self._load_extra_venues()
        ev['venue_key'] = [self.geocode(a, v)[0] for a, v in zip(ev.address, ev.venue_name.fillna(''))]
        self.events = ev
        # closures
        frames = []
        for f in self._csvs('closures'):
            frames.append(read_csv_any(f))
        cl = pd.concat(frames, ignore_index=True).drop_duplicates()
        cl['start'] = self._read_ts(cl['when'])
        cl['end'] = cl['start'] + cl['duration'].map(_parse_duration)
        self.closures = cl.reset_index(drop=True)
        # energy
        frames = []
        for f in self._csvs('energy'):
            e = read_csv_any(f)
            e.index = pd.to_datetime(e.iloc[:, 0], format='mixed')
            frames.append(e.iloc[:, 1:])
        en = pd.concat(frames)
        self.energy = en[~en.index.duplicated(keep='last')].sort_index()

    # ------------------------------------------------------------------ network
    def _build_network(self):
        st = self.stations
        id2key = dict(zip(st.station_id, st.index))
        cx = pd.read_csv(os.path.join(self.data_dir, 'berlin_ubahn_connections.csv'))
        G = nx.Graph()
        G.add_nodes_from(self.keys)
        for a, b in cx.values:
            G.add_edge(id2key[a], id2key[b])
        self.graph = G
        self.line_seq = {}
        for L in LINES:
            H = nx.Graph()
            for a, b in G.edges():
                if L in st.at[a, 'lines'] and L in st.at[b, 'lines']:
                    H.add_edge(a, b)
            if H.number_of_nodes() == 0:
                continue
            # remove spurious chords of triangles (route-variant artefacts), keep the middle station
            for a, b in list(H.edges()):
                common = set(H.neighbors(a)) & set(H.neighbors(b))
                for c in common:
                    outside = lambda n: len(set(H.neighbors(n)) - {a, b, c}) > 0
                    if outside(a) and outside(b) and not outside(c) and H.has_edge(a, b):
                        H.remove_edge(a, b)
            ends = [n for n, d in H.degree() if d == 1]
            self.line_seq[L] = nx.shortest_path(H, ends[0], ends[1]) if len(ends) == 2 else list(H.nodes())
        self.st_lat = st.latitude.values
        self.st_lon = st.longitude.values

    # ------------------------------------------------------------------ lookups
    def resolve_station(self, q: str, multi: bool = False):
        """Map free text to station key(s).

        Accepts official names ('U Kottbusser Tor'), short keys ('Kottbusser Tor'), spelling variants
        ('Hermannstraße', 'Neukoelln'), aliases ('Zoo', 'Alex', 'Kotti', 'Hbf') and small typos.
        'Stadtmitte' matches both 'Stadtmitte (U2)' and 'Stadtmitte (U6)'.

        Args:
            q: free-text station name.
            multi: return the list of all matches instead of a single key.

        Returns:
            A station key, or a list of keys when multi=True.

        Raises:
            ValueError: unknown or ambiguous name; the message includes suggestions.
        """
        if q in self.stations.index:
            return [q] if multi else q
        n = _norm_station(q)
        n = _norm_station(STATION_ALIASES.get(n, n))
        hits = self._norm_index.get(n) or [k for nk, ks in self._norm_index.items() for k in ks if nk.split(' (')[0] == n]
        if not hits:
            close = difflib.get_close_matches(n, list(self._norm_index), n=3, cutoff=0.75)
            if len(close) == 1 or (close and difflib.SequenceMatcher(None, n, close[0]).ratio() > 0.9):
                hits = self._norm_index[close[0]]
            else:
                # part of a name, e.g. 'Spandau' -> Altstadt Spandau, Rathaus Spandau
                part = [k for nk, ks in self._norm_index.items() for k in ks if re.search(rf'\b{re.escape(n)}\b', nk)]
                if len(set(part)) == 1:
                    hits = part
                elif part:
                    raise ValueError(f'"{q}" matches several stations: {sorted(set(part))}. Name one, or list them all.')
                else:
                    sugg = [self._norm_index[c][0] for c in close]
                    raise ValueError(f'Unknown station "{q}".' + (f' Did you mean: {sugg}?' if sugg else
                                     f' It is not one of the {len(self.keys)} U-Bahn stations in the data '
                                     '(S-Bahn-only stations are not included, U4 is absent and U6 ends at Kurt-Schumacher-Platz).'))
        if multi:
            return hits
        if len(hits) > 1:
            raise ValueError(f'"{q}" is ambiguous: {hits}')
        return hits[0]

    def geocode(self, address: str, venue_name: str = ''):
        """Locate an event venue with the built-in VENUES gazetteer.

        Args:
            address: address or venue name ('Uber-Platz 1', 'Mercedes-Benz Arena', 'Messedamm 22'),
                or a (lat, lon) tuple for a venue that is not in the gazetteer.
            venue_name: optional venue name, matched together with the address.

        Returns:
            (venue_key, lat, lon), or (None, nan, nan) when nothing matches.
        """
        if isinstance(address, (tuple, list)) and len(address) == 2:
            return (f'point {address[0]:.4f},{address[1]:.4f}', float(address[0]), float(address[1]))
        q = _ascii(f'{address} {venue_name}')
        best, best_len = None, 0
        for key, (lat, lon, aliases) in VENUES.items():
            for a in aliases + [_ascii(key)]:
                if a in q and len(a) > best_len:
                    best, best_len = key, len(a)
        if best is None:
            return (None, np.nan, np.nan)
        lat, lon, _ = VENUES[best]
        return (best, lat, lon)

    def stations_near(self, lat, lon, radius_km=1.5):
        """Stations within `radius_km` (straight line) of a point, nearest first: [(key, km), ...]."""
        d = _hav(lat, lon, self.st_lat, self.st_lon)
        o = np.argsort(d)
        return [(self.keys[i], round(float(d[i]), 2)) for i in o if d[i] <= radius_km]

    # ------------------------------------------------------------------ time helpers
    @staticmethod
    def service_day(ts):
        """Service day of each timestamp: 05:00 to 00:45 the next morning count as one day."""
        return (pd.DatetimeIndex(ts) - pd.Timedelta(hours=SERVICE_START_HOUR)).normalize()

    @staticmethod
    def daytype(days):
        """Day type per service day: 0 weekday, 1 Saturday, 2 Sunday or Berlin public holiday."""
        days = pd.DatetimeIndex(days)
        dow = days.dayofweek
        hol = np.isin(days.strftime('%Y-%m-%d'), list(BERLIN_HOLIDAYS_2026))
        return np.where(hol | (dow == 6), 2, np.where(dow == 5, 1, 0))      # 0 wd, 1 sat, 2 sun/holiday

    @staticmethod
    def slots(start, end):
        """Service slots (15 min) in [start, end]; drops the 01:00-04:45 non-service window."""
        idx = pd.date_range(_to_dt(start).floor('15min'), _to_dt(end), freq='15min')
        keep = np.isin(idx.strftime('%H:%M'), TOD_ORDER)
        return idx[keep].rename('slot')

    # ------------------------------------------------------------------ fitting
    def _closure_list(self, closures=None):
        """Parse closure descriptions into dicts (kind, line, stations, start, end, description)."""
        cl = self.closures if closures is None else closures
        out = []
        for _, r in cl.iterrows():
            out.append(self.parse_closure(r['description'], r['start'], r['end']))
        return out

    def parse_closure(self, desc, start=None, end=None):
        """Turn a closure description into a structured record.

        Understands line suspensions ('Line U6 suspended (on a section) between A and B ...',
        'Line U2 suspended ...' for the whole line), station closures ('Station X closed ...'),
        platform closures ('Platform 2 at X closed ...') and segment closures ('Segment between A and B
        (on U1) closed ...'). Other wordings fall back to matching station names.

        Returns:
            dict with kind ('line', 'station', 'platform', 'segment' or 'unknown'), line, stations (the
            ordered section or the closed station), start, end, description and issues.
        """
        d = {'description': desc, 'start': _to_dt(start) if start is not None else None,
             'end': _to_dt(end) if end is not None else None, 'kind': 'unknown', 'line': None, 'stations': [], 'issues': []}
        text = str(desc)
        stop = r'(?=\s+(?:due to|because|for|on line|on the|of line)\b|\s*[.,;]|$)'
        m = re.search(r'Line\s+(U\d+)\s+(?:is\s+)?(?:suspended|closed|interrupted)\s+(?:on\s+(?:a|the)\s+section\s+)?'
                      r'between\s+(.+?)\s+and\s+(.+?)' + stop, text, re.I)
        if m:
            L, a, b = m.groups()
            d.update(kind='line', line=L.upper())
            try:
                d['stations'] = self.section(L, a, b)
            except ValueError as e:
                d['issues'].append(str(e))
            return d
        m = re.search(r'(?:segment|section|track)s?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:on|of)\s+(?:line\s+)?(U\d+))?'
                      r'\s+(?:is\s+|are\s+)?(?:closed|suspended|blocked|interrupted)', text, re.I)
        if m:
            a, b, L = m.groups()
            try:
                if L:
                    d.update(kind='line', line=L.upper(), stations=self.section(L, a, b))
                else:
                    ka, kb = self.resolve_station(a), self.resolve_station(b)
                    d.update(kind='segment', stations=nx.shortest_path(self.graph, ka, kb))
            except (ValueError, nx.NetworkXNoPath) as e:
                d['issues'].append(str(e))
            return d
        m = re.search(r'Line\s+(U\d+)\s+(?:is\s+)?(?:suspended|closed|interrupted)' + stop, text, re.I)
        if m and m.group(1).upper() in self.line_seq:
            d.update(kind='line', line=m.group(1).upper(), stations=list(self.line_seq[m.group(1).upper()]))
            return d
        m = re.search(r'platform.*?\bat\s+(.+?)\s+(?:is\s+)?closed', text, re.I) or \
            re.search(r'^(?:Station\s+)?(.+?)\s*,?\s+platform\b.*?closed', text, re.I)
        if m:
            d['kind'] = 'platform'
            try:
                d['stations'] = self.resolve_station(m.group(1).strip(), multi=True)
            except ValueError as e:
                d['issues'].append(str(e))
            lm = re.search(r'\b(U\d)\b', text)
            d['line'] = lm.group(1) if lm else None
            return d
        m = re.search(r'Station\s+(.+?)\s+(?:is\s+)?closed', text, re.I)
        if m:
            d['kind'] = 'station'
            try:
                d['stations'] = self.resolve_station(m.group(1), multi=True)
            except ValueError as e:
                d['issues'].append(str(e))
            return d
        # fallback: any known station names mentioned
        found = [k for k in self.keys if _norm_station(k).split(' (')[0] in _norm_station(text)]
        lm = re.search(r'\b(U\d)\b', text)
        d.update(stations=found, line=lm.group(1) if lm else None)
        d['issues'].append('description not in a known pattern; stations matched by name only')
        return d

    def section(self, line, a, b):
        """Ordered stations of `line` between a and b, inclusive.

        Doubles as the premise check: raises ValueError when a station is not served by the line,
        with the line's full station list in the message (e.g. U8 does not serve Neukölln).
        """
        line = line.upper().replace(' ', '')
        if line not in self.line_seq:
            raise ValueError(f'{line} is not in this network (available: {sorted(self.line_seq)})')
        seq = self.line_seq[line]
        ka, kb = self.resolve_station(a, multi=True), self.resolve_station(b, multi=True)
        ka = [k for k in ka if k in seq]
        kb = [k for k in kb if k in seq]
        bad = [x for x, k in ((a, ka), (b, kb)) if not k]
        if bad:
            raise ValueError(f'{", ".join(bad)} is not served by {line}. {line} runs {seq[0]} - {seq[-1]}: {", ".join(seq)}')
        i, j = seq.index(ka[0]), seq.index(kb[0])
        return seq[min(i, j):max(i, j) + 1] if i <= j else seq[j:i + 1][::-1]

    def _mask_station_closures(self, index, closures, rebound_slots=4):
        T, S = len(index), len(self.keys)
        closed = np.zeros((T, S), bool)
        after = np.zeros((T, S), int) - 1            # slot number after reopening (0..), -1 otherwise
        col = {k: i for i, k in enumerate(self.keys)}
        for c in closures:
            if c['kind'] != 'station' or c['start'] is None:
                continue
            for k in c['stations']:
                j = col[k]
                m = (index >= c['start']) & (index < c['end'])
                closed[m, j] = True
                first = index.searchsorted(c['end'])
                for r in range(rebound_slots):
                    if first + r < T:
                        after[first + r, j] = r
        return closed, after

    def fit(self, iters: int = 25):
        self.fitted_at = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        """(Re)fit every parameter on the loaded data. Called by __init__.

        Censored-exponential maximum likelihood, alternating between station levels w (from
        05:00-13:00 slots only), time profiles P and per-slot multipliers M; a second pass drops
        cells carrying event load. Then fits the weather rule on M, learns venue -> station shares
        from post-event departure windows, and fits the per-line energy model.
        """
        F = self.flows
        X = F.values
        T, S = X.shape
        cap = X.max(axis=0)
        self.ceiling = pd.Series(cap, index=self.keys)
        sd = self.service_day(F.index)
        tod = pd.Index(TOD_ORDER).get_indexer(F.index.strftime('%H:%M'))
        dt = self.daytype(sd)
        self._clist = self._closure_list()
        closed, after = self._mask_station_closures(F.index, self._clist)
        use0 = ~closed & (after < 0)
        cens = X >= cap[None, :]
        Xc = np.minimum(X + 0.5, cap[None, :])
        morning = (F.index.hour >= 5) & (F.index.hour < 13)
        self.venue_learned = {}
        ev_cells = np.zeros_like(use0)
        for rnd in range(2):                      # pass 2 drops cells carrying event load
            use = use0 & ~ev_cells
            unc = use & ~cens
            w, P, M = self._fit_core(Xc, use, unc, dt, tod, morning, iters)
            self._set_params(F, w, P, M)
            self._learn_venue_shares()
            evl = self._event_lambda(F.index, self.keys, self._data_events(F.index.min(), F.index.max()))
            ev_cells = evl > 1.0
        self._fit_energy()
        return self

    def _fit_core(self, Xc, use, unc, dt, tod, morning, iters):
        T, S = Xc.shape
        w = np.nanmean(np.where(use, Xc, np.nan), axis=0)
        P = np.ones((3, len(TOD_ORDER)))
        M = np.ones(T)
        for _ in range(iters):
            a = w[None, :] * M[:, None]
            num = np.where(use, Xc / a, 0).sum(axis=1)
            den = unc.sum(axis=1)
            for d in range(3):
                for t in range(len(TOD_ORDER)):
                    m = (dt == d) & (tod == t)
                    if m.any():
                        P[d, t] = num[m].sum() / max(den[m].sum(), 1)
            a = w[None, :] * P[dt, tod][:, None]
            M = np.where(use, Xc / a, 0).sum(axis=1) / np.maximum(unc.sum(axis=1), 1)
            a = (P[dt, tod] * M)[:, None]
            mm = use & morning[:, None]
            w = np.where(mm, Xc / a, 0).sum(axis=0) / (unc & morning[:, None]).sum(axis=0)
        return w, P, M

    def _set_params(self, F, w, P, M):
        # parametric weather on the per-slot multiplier M
        wx = self.weather.reindex(F.index)
        tday = self._daily_temp(F.index)
        sel = (F.index.hour >= 6) & (F.index.hour <= 21) & np.isfinite(tday) & wx.prcp.notna().values
        y = np.log(M[sel])
        Tm, Pp = tday[sel], wx.prcp.values[sel]
        r = least_squares(lambda p: p[0] + p[1] * (Tm - 18) + p[2] * (1 - np.exp(-Pp / p[3])) - y,
                          [0, -0.018, 0.28, 0.6], bounds=([-1, -0.2, 0, 0.05], [1, 0.2, 1, 5]))
        a0, b, c, dsc = r.x
        self.weather_params = {'heat_per_degC': b, 'rain_log_max': c, 'rain_scale_mm_h': dsc, 'T_ref': 18.0}
        P = P * np.exp(a0)                                   # reference: T_day = 18 degC, dry
        self.w = pd.Series(w, index=self.keys)
        self.profile = pd.DataFrame(P.T, index=TOD_ORDER, columns=['weekday', 'saturday', 'sunday_holiday'])
        self.slot_multiplier = pd.Series(M / np.exp(a0), index=F.index)
        self.fit_resid_sd = float(np.std(y - (a0 + b * (Tm - 18) + c * (1 - np.exp(-Pp / dsc)))))

    def _kernel_load(self, index, events):
        """Attendance x kernel per slot for a set of events, as if a single station took the whole crowd."""
        pos = {t: i for i, t in enumerate(index)}
        u = np.zeros(len(index))
        for _, r in events.iterrows():
            s0, e0 = r['start'].floor('15min'), r['end'].floor('15min')
            for kern, anchor in ((ARRIVAL_KERNEL, s0), (DEPARTURE_KERNEL, e0)):
                for k, f in kern.items():
                    i = pos.get(anchor + pd.Timedelta(minutes=15 * k))
                    if i is not None:
                        u[i] += r['estimated_attendance'] * f
        return u

    @staticmethod
    def _share_mle(x, base, u, cap):
        """Censored-exponential maximum likelihood of the share of an event crowd taken by one station, with the
        likelihood-ratio statistic against 'no share' (readings at the ceiling count as 'at least the ceiling')."""
        from scipy.optimize import minimize_scalar
        m = u > 0
        x, base, u = x[m], np.maximum(base[m], 0.5), u[m]
        cens = x >= cap
        xc = np.minimum(x + 0.5, cap)

        def nll(sh):
            lam = base + sh * u
            return np.sum(np.log(lam[~cens]) + xc[~cens] / lam[~cens]) + np.sum(cap / lam[cens])
        r = minimize_scalar(nll, bounds=(0, 1.5), method='bounded')
        return float(r.x), float(2 * (nll(0.0) - r.fun)), float(cens.mean())

    def _guard_split(self, vkey, shares, min_rule=0.2, keep=0.5):
        """A station the distance rule expects to take min_rule or more of a venue's crowd keeps at least keep x its
        rule share when the data could not confirm it: weak evidence for a station is not evidence that the crowd
        avoids it. Returns the renormalised split and the stations kept this way."""
        rule = self.rule_shares(vkey)
        missing = {k: keep * v for k, v in rule.items() if v >= min_rule and k not in shares}
        if not missing:
            return shares, []
        out = {**shares, **missing}
        tot = sum(out.values())
        return {k: v / tot for k, v in out.items()}, sorted(missing)

    def _learn_venue_shares(self, min_z=6.0, keep_z=3.0, radius_km=1.8, min_slots=2, capped_mle=0.15, min_lr=4.0):
        """Venue -> station shares learned from the data (captures where the simulator actually geocoded each
        venue). Usually from post-event excess; when readings around a venue are often at the ceiling (large
        crowds), a censored maximum-likelihood estimate per station is used instead, since excess counts would
        be truncated by the ceiling."""
        self.venue_learned, self.venue_learned_diag = {}, {}
        F = self.flows
        idx = F.index
        lam = pd.DataFrame(self._base_lambda(idx, self.keys), index=idx, columns=self.keys)
        lam = self._apply_closures(lam, self._clist, weather=None, events=[]).values
        er = self.expected_reading(lam, self.ceiling.values[None, :])
        EX, V = F.values - er, er ** 2
        X = F.values
        for vkey, g in self.events.dropna(subset=['venue_key']).groupby('venue_key'):
            lat, lon = VENUES[vkey][:2]
            cand = _hav(lat, lon, self.st_lat, self.st_lon) <= radius_km
            if not cand.any() or g.groupby(['start', 'end']).ngroups < min_slots:
                continue
            exs, var, att = np.zeros(len(self.keys)), np.zeros(len(self.keys)), 0.0
            win = np.zeros(len(idx), bool)
            for (s0, e0), gg in g.groupby(['start', 'end']):
                m = (idx >= e0) & (idx < e0 + pd.Timedelta('2h'))
                if not m.any():
                    continue
                win |= m
                exs += EX[m].sum(axis=0)
                var += V[m].sum(axis=0) + 1
                att += gg.estimated_attendance.sum()
            if att == 0:
                continue
            z = exs / np.sqrt(np.maximum(var, 1))
            cj = np.where(cand)[0]
            capped = float((X[np.ix_(win, cj)] >= self.ceiling.values[cj][None, :]).mean()) if win.any() else 0.0
            if capped > capped_mle:
                u = self._kernel_load(idx, g)
                est = {}
                for j in cj:
                    sh, lr, _ = self._share_mle(X[:, j], lam[:, j], u, self.ceiling.values[j])
                    if lr >= min_lr and sh >= 0.03:
                        est[self.keys[j]] = (sh, lr)
                if est:
                    tot = sum(v[0] for v in est.values())
                    self.venue_learned[vkey], kept = self._guard_split(vkey, {k: v[0] / tot for k, v in est.items()})
                    self.venue_learned_diag[vkey] = {'attendance': att, 'method': 'censored MLE (crowds often at the ceiling)',
                                                     'capped_share': capped, 'share_sum': tot,
                                                     'lr': {k: round(v[1], 1) for k, v in est.items()},
                                                     'kept_by_distance_rule': kept}
                continue
            keep = cand & (z >= keep_z) & (exs > 0)
            ratio = exs[keep].sum() / max(att, 1)
            if z[cand].max() >= min_z and 0.25 <= ratio <= 1.0:
                sh = exs[keep] / exs[keep].sum()
                self.venue_learned[vkey], kept = self._guard_split(vkey, {self.keys[i]: float(s) for i, s in zip(np.where(keep)[0], sh)})
                self.venue_learned_diag[vkey] = {'attendance': att, 'departure_share_observed': float(exs[keep].sum() / max(att, 1)),
                                                 'max_z': float(z[cand].max()), 'method': 'post-event excess', 'capped_share': capped,
                                                 'kept_by_distance_rule': kept}

    def _daily_temp(self, index):
        """Mean of hourly temperatures over each slot's service day (nan if missing)."""
        wx = self.weather
        hourly = wx[wx.index.minute == 0]
        daymean = hourly.temp.groupby(self.service_day(hourly.index)).mean()
        return daymean.reindex(self.service_day(index)).values

    def weather_multiplier(self, tday, prcp):
        """Flow multiplier for a service day's mean temperature `tday` (degC) and a 15-min
        precipitation rate `prcp` (mm/h). Equals 1.0 at 18 degC and dry; arrays broadcast."""
        p = self.weather_params
        return np.exp(p['heat_per_degC'] * (np.asarray(tday, float) - p['T_ref'])
                      + p['rain_log_max'] * (1 - np.exp(-np.asarray(prcp, float) / p['rain_scale_mm_h'])))

    def _fit_energy(self):
        sd = self.service_day(self.flows.index)
        daily = self.flows.groupby(sd).sum()
        self.energy_model = {}
        for L in self.energy.columns:
            ks = [k for k in self.keys if L in self.stations.at[k, 'lines']]
            f = daily[ks].sum(axis=1)
            e = self.energy[L].reindex(f.index)
            ok = e.notna()
            if ok.sum() > 5:
                b1, b0 = np.polyfit(f[ok], e[ok], 1)
                self.energy_model[L] = (b0, b1, float(np.std(e[ok] - (b0 + b1 * f[ok]))))

    # ------------------------------------------------------------------ expectation
    def _base_lambda(self, index, stations, weather=None):
        """w * P * weather multiplier, shape (len(index), len(stations))."""
        index = pd.DatetimeIndex(index)
        tod = pd.Index(TOD_ORDER).get_indexer(index.strftime('%H:%M'))
        dt = self.daytype(self.service_day(index))
        prof = np.where(tod >= 0, self.profile.values[np.maximum(tod, 0), dt], 0.0)
        if weather is None:
            tday = self._daily_temp(index)
            prcp = self.weather.prcp.reindex(index).values
            miss = ~np.isfinite(tday) | ~np.isfinite(prcp)
            if miss.any():
                raise ValueError('No weather data for some slots: pass weather={"tmean": .., "prcp": ..}')
        else:
            tday = np.full(len(index), float(weather.get('tmean', 18.0)))
            prcp = np.full(len(index), float(weather.get('prcp', 0.0)))
            if 'prcp_by_slot' in weather:                          # optional {timestamp: mm/h}
                for t, v in weather['prcp_by_slot'].items():
                    prcp[index == _to_dt(t)] = v
        mult = self.weather_multiplier(tday, prcp)
        wv = self.w[stations].values
        return (prof * mult)[:, None] * wv[None, :]

    def venue_shares(self, venue):
        """How an event crowd splits across stations.

        Uses the split learned from the data when the venue had enough events (15 venues in the
        training set); otherwise a distance rule: Gaussian weights (sigma 0.6 km) over stations
        within 2 km, dropping shares under 1 %.

        Returns:
            (venue_key, {station_key: share}), or (None, {}) if the venue cannot be located.
        """
        key, lat, lon = self.geocode(venue) if not isinstance(venue, dict) else (venue.get('key'), venue['lat'], venue['lon'])
        if key in getattr(self, 'venue_learned', {}):
            return key, dict(self.venue_learned[key])
        if key is None or not np.isfinite(lat):
            return None, {}
        d = _hav(lat, lon, self.st_lat, self.st_lon)
        m = d <= VENUE_RADIUS_KM
        if m.sum() == 0:
            m = d <= np.sort(d)[1]
        wgt = np.where(m, np.exp(-d ** 2 / (2 * VENUE_SIGMA_KM ** 2)), 0)
        wgt = wgt / wgt.sum()
        return key, {self.keys[i]: float(wgt[i]) for i in np.argsort(-wgt) if wgt[i] > 0.01}

    def rule_shares(self, venue):
        """Distance-rule split for a venue, ignoring anything learned from the data."""
        key, lat, lon = self.geocode(venue) if not isinstance(venue, dict) else (venue.get('key'), venue['lat'], venue['lon'])
        if key is None or not np.isfinite(lat):
            return {}
        d = _hav(lat, lon, self.st_lat, self.st_lon)
        m = d <= VENUE_RADIUS_KM
        if m.sum() == 0:
            m = d <= np.sort(d)[1]
        wgt = np.where(m, np.exp(-d ** 2 / (2 * VENUE_SIGMA_KM ** 2)), 0)
        wgt = wgt / wgt.sum()
        return {self.keys[i]: float(wgt[i]) for i in np.argsort(-wgt) if wgt[i] > 0.01}

    def _event_lambda(self, index, stations, events):
        """Additive event load for the given slots/stations. events: iterable of dicts
        {venue|address, start, end, attendance}."""
        index = pd.DatetimeIndex(index)
        out = np.zeros((len(index), len(stations)))
        col = {k: i for i, k in enumerate(stations)}
        pos = {t: i for i, t in enumerate(index)}
        for e in events:
            _, shares = self.venue_shares(e.get('venue') or e.get('address'))
            if not shares:
                continue
            att = float(e['attendance'])
            s0, e0 = _to_dt(e['start']).floor('15min'), _to_dt(e['end']).floor('15min')
            for kern, anchor in ((ARRIVAL_KERNEL, s0), (DEPARTURE_KERNEL, e0)):
                for k, f in kern.items():
                    i = pos.get(anchor + pd.Timedelta(minutes=15 * k))
                    if i is None:
                        continue
                    for stn, sh in shares.items():
                        if stn in col:
                            out[i, col[stn]] += att * f * sh
        return out

    def _data_events(self, start, end):
        ev = self.events
        m = (ev.end + pd.Timedelta('2h') >= start) & (ev.start - pd.Timedelta('2h') <= end)
        return [{'address': r.address, 'venue': r.address + ' ' + str(r.venue_name if isinstance(r.venue_name, str) else ''),
                 'start': r.start, 'end': r.end, 'attendance': r.estimated_attendance, 'name': r.event_name}
                for r in ev[m].itertuples()]

    def expected(self, start, end, stations=None, weather=None, events='data', closures='data', extra_events=None):
        """Expected intensity lambda per 15-min slot and station, before noise and ceiling.

        Args:
            start, end: window, inclusive, local Berlin time (e.g. '2026-09-22 05:00').
            stations: station names in any spelling resolve_station() accepts; None = all 168.
            weather: None uses the weather file (ValueError if the window is not covered), or a
                scenario {'tmean': daily mean degC, 'prcp': mm/h, 'prcp_by_slot': {timestamp: mm/h}}.
            events: 'data' uses the events file, or a list of event dicts
                {'venue' or 'address', 'start', 'end', 'attendance'}; [] for none.
            closures: 'data' uses the closures file, or a list of parse_closure() dicts or
                (description, start, end) tuples; [] for none.
            extra_events: scenario events added on top of `events`.

        Returns:
            DataFrame (slots x stations) of lambda. expected_reading() turns it into the mean
            counter value; exp(-ceiling / lambda) is the chance a reading hits the ceiling.
        """
        idx = self.slots(start, end)
        stations = self.keys if stations is None else [self.resolve_station(s) for s in stations]
        lam = self._base_lambda(idx, stations, weather)
        evs = self._data_events(idx.min(), idx.max()) if isinstance(events, str) and events == 'data' else list(events or [])
        evs = evs + list(extra_events or [])
        lam = lam + self._event_lambda(idx, stations, evs)
        cls = self._clist if isinstance(closures, str) and closures == 'data' else [
            c if isinstance(c, dict) else self.parse_closure(*c) for c in (closures or [])]
        lam = pd.DataFrame(lam, index=idx, columns=stations)
        return self._apply_closures(lam, cls, weather=weather, events=evs)

    def _apply_closures(self, lam, closures, weather=None, events=()):
        """Zero closed stations and add the pent-up rebound after reopening."""
        lam = lam.copy()
        idx = lam.index
        for c in closures:
            if c['kind'] != 'station' or c['start'] is None:
                continue
            for k in c['stations']:
                if k not in lam.columns:
                    continue
                if c['end'] < idx.min() or c['start'] > idx.max() + pd.Timedelta('1h'):
                    continue
                cidx = self.slots(c['start'], c['end'] - pd.Timedelta('1min'))
                cidx = cidx[cidx >= c['start']]
                try:
                    lost = self._base_lambda(cidx, [k], weather)[:, 0] + self._event_lambda(cidx, [k], events)[:, 0]
                except ValueError:
                    lost = self._base_lambda(cidx, [k], {'tmean': 18, 'prcp': 0})[:, 0]
                inside = (idx >= c['start']) & (idx < c['end'])
                lam.loc[inside, k] = 0.0
                for r, f in enumerate(REBOUND_FRACTIONS):
                    t = c['end'].ceil('15min') + pd.Timedelta(minutes=15 * r)
                    if t in lam.index:
                        lam.at[t, k] += f * lost.sum()
        return lam

    def _cap(self, stations):
        return self.ceiling[stations].values

    @staticmethod
    def expected_reading(lam, cap):
        """Mean counter value for intensity `lam` and ceiling `cap`:
        lam * (1 - exp(-cap / lam)) - 0.5, floored at 0 (the -0.5 accounts for integer rounding)."""
        lam = np.asarray(lam, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            v = lam * (1 - np.exp(-cap / np.where(lam > 0, lam, 1))) - 0.5
        return np.where(lam > 0, np.maximum(v, 0), 0)

    # ------------------------------------------------------------------ tools
    def observed(self, station, start, end):
        """Recorded 15-min flows at one station between start and end (inclusive), as a Series."""
        k = self.resolve_station(station)
        return self.flows.loc[_to_dt(start):_to_dt(end), k]

    def explain(self, station, start, end):
        """Observed vs expected flow at one station over a window, with the active drivers.

        Returns:
            (table, drivers). table has one row per slot: observed, expected_reading, lambda_base
            (weather only), lambda_total (with events and closures) and p_at_ceiling. drivers gives
            the ceiling, the mean weather multiplier, the events loading this station, the closures
            affecting it, observed and expected totals, and the number of slots at the ceiling.
        """
        k = self.resolve_station(station)
        idx = self.slots(start, end)
        obs = self.flows[k].reindex(idx)
        base = self.expected(start, end, [k], events=[], closures=[])[k]
        full = self.expected(start, end, [k])[k]
        cap = self.ceiling[k]
        df = pd.DataFrame({'observed': obs, 'expected_reading': self.expected_reading(full.values, cap).round(1),
                           'lambda_base': base.round(1), 'lambda_total': full.round(1),
                           'p_at_ceiling': np.where(full > 0, np.exp(-cap / full.clip(lower=1e-9)), 0).round(3)})
        tday = self._daily_temp(idx)
        drivers = {
            'ceiling': float(cap),
            'weather_multiplier_mean': float(np.nanmean(self.weather_multiplier(tday, self.weather.prcp.reindex(idx).values))),
            'events_affecting_station': [e['name'] + f" ({e['start']:%H:%M}-{e['end']:%H:%M}, {int(e['attendance'])})"
                                         for e in self._data_events(idx.min(), idx.max())
                                         if k in self.venue_shares(e['venue'])[1]],
            'closures': [c['description'] for c in self._clist
                         if c['start'] <= idx.max() and c['end'] >= idx.min() and k in c['stations']],
            'observed_total': float(obs.sum()), 'expected_total': float(df.expected_reading.sum()),
            'slots_at_ceiling': int((obs >= cap).sum()),
        }
        return df, drivers

    def event_impact(self, venue, start, end=None, attendance=2000, segment='Music', weather=None):
        """Scenario for one event: extra passengers per station and slot on top of the baseline.

        Args:
            venue: address, venue name or (lat, lon); 'Mercedes-Benz Arena' resolves to the Uber Arena.
            start, end: event times; end defaults to start + 3 h for Music and + 2 h otherwise,
                the dataset's own convention.
            attendance: expected attendance.
            segment: event segment, only used for the default end time.
            weather: scenario dict as in expected(); defaults to the weather file when it covers
                the window, else 18 degC and dry.

        Returns:
            dict with venue, station_shares, ceiling, and three DataFrames covering 90 min before
            the start to 2 h 15 after the end: extra_lambda, baseline_lambda and p_at_ceiling.
        """
        s0 = _to_dt(start)
        e0 = _to_dt(end) if end is not None else s0 + (pd.Timedelta('3h') if segment == 'Music' else pd.Timedelta('2h'))
        key, shares = self.venue_shares(venue)
        if not shares:
            raise ValueError(f'Could not geocode "{venue}". Pass (lat, lon) instead.')
        stations = list(shares)
        w0, w1 = s0 - pd.Timedelta('90min'), e0 + pd.Timedelta('2h15min')
        idx = self.slots(w0, w1)
        if weather is None and not np.isfinite(self._daily_temp(idx)).all():
            weather = {'tmean': 18, 'prcp': 0}
        base = self.expected(w0, w1, stations, weather=weather, events=[], closures=[])
        ev = self._event_lambda(base.index, stations, [{'venue': venue, 'start': s0, 'end': e0, 'attendance': attendance}])
        tot = base.values + ev
        cap = self._cap(stations)
        res = {
            'venue': key, 'station_shares': {k: round(v, 3) for k, v in shares.items()},
            'extra_lambda': pd.DataFrame(ev.round(0), index=base.index, columns=stations),
            'baseline_lambda': base.round(0),
            'p_at_ceiling': pd.DataFrame(np.exp(-cap / np.maximum(tot, 1e-9)).round(3), index=base.index, columns=stations),
            'ceiling': dict(zip(stations, cap)),
        }
        return res

    def capacity_risk(self, start, end, stations=None, weather=None, events='data', closures='data', top=10,
                      extra_events=None, sort_by='exp_slots_at_ceiling'):
        """Rank stations by the risk that 15-min flow reaches the station ceiling in [start, end].

        Args:
            start, end, stations, weather, events, closures, extra_events: as in expected().
            top: number of stations returned.
            sort_by: 'exp_slots_at_ceiling' (default, likelihood), 'p_any_slot_at_ceiling',
                'exp_pax_above_ceiling' (volume of unmet demand) or 'peak_lambda_over_ceiling'.

        Returns:
            DataFrame per station: exp_slots_at_ceiling, p_any_slot_at_ceiling,
            peak_lambda_over_ceiling, peak_slot, exp_pax_above_ceiling (lambda * exp(-ceiling/lambda)
            summed over slots) and ceiling.
        """
        lam = self.expected(start, end, stations, weather=weather, events=events, closures=closures,
                            extra_events=extra_events)
        cap = self._cap(list(lam.columns))
        with np.errstate(divide='ignore', over='ignore'):
            p = np.where(lam.values > 0, np.exp(-cap / np.maximum(lam.values, 1e-9)), 0)
        over = lam.values * p
        res = pd.DataFrame({
            'exp_slots_at_ceiling': p.sum(axis=0), 'p_any_slot_at_ceiling': 1 - np.prod(1 - p, axis=0),
            'peak_lambda_over_ceiling': (lam.values / cap).max(axis=0), 'peak_slot': lam.idxmax().dt.strftime('%H:%M').values,
            'exp_pax_above_ceiling': over.sum(axis=0), 'ceiling': cap}, index=lam.columns)
        res.index.name = 'station'
        return res.sort_values(sort_by, ascending=False).head(top).round(3)

    def reroute_options(self, line, a, b, walk_km=1.2):
        """Topology-based rerouting for a suspended section of `line` between stations a and b.

        In the training data a suspension leaves no measurable trace in station flows, so this is
        planning logic, not a measured response. S-Bahn, trams and buses are not in the dataset.

        Returns:
            dict with the ordered section, interior stations left without service, the two remaining
            parts of the line, the U-Bahn detour between the section ends (None if there is none)
            and, per station, the other lines serving it and the stations with another line within
            `walk_km` (straight-line km).
        """
        L = line.upper().replace(' ', '')
        sec = self.section(L, a, b)
        seq = self.line_seq[L]
        lo, hi = sorted((seq.index(sec[0]), seq.index(sec[-1])))
        interior = seq[lo + 1:hi]
        part_a, part_b = seq[:lo + 1], seq[hi:]
        H = self.graph.copy()
        for x, y in zip(seq[lo:hi], seq[lo + 1:hi + 1]):
            if H.has_edge(x, y):
                H.remove_edge(x, y)
        try:
            detour = nx.shortest_path(H, seq[lo], seq[hi])
        except nx.NetworkXNoPath:
            detour = None
        alt = {}
        for s in interior + [seq[lo], seq[hi]]:
            others = [l for l in self.stations.at[s, 'lines'] if l != L]
            lat, lon = self.stations.at[s, 'latitude'], self.stations.at[s, 'longitude']
            # any other station that keeps a service during the suspension (i.e. has a line other than L)
            near = [(k, d) for k, d in self.stations_near(lat, lon, walk_km)
                    if k != s and any(l != L for l in self.stations.at[k, 'lines'])]
            alt[s] = {'other_lines_here': others, 'walkable_alternatives_km': near[:4]}
        return {'line': L, 'section': seq[lo:hi + 1], 'interior_without_service': interior,
                'line_split_into': [(f'{p[0]} - {p[-1]}' if len(p) > 1 else f'{p[0]} (cut off from the rest of the line)')
                                    for p in (part_a, part_b)],
                'network_detour_between_section_ends': detour,
                'detour_by_line': self._path_lines(detour) if detour else None,
                'per_station_alternatives': alt}

    def suspension_scenario(self, line, a, b, at, minutes=20, displaced_share=1.0, weather=None, walk_km=1.2):
        """Risk at the stations likely to absorb passengers of a suspended section, for the next
        `minutes` from `at`.

        Two views: as_in_data (suspensions do not move flows in the training set) and planning_case,
        where `displaced_share` of the demand of interior stations left with no line at all walks to
        the nearest stations with another line (inverse-distance weights). Interchanges inside the
        section keep their passengers, who change line on the spot.

        Returns:
            dict with reroute (reroute_options output), slots, as_in_data and planning_case
            (DataFrames: mean lambda, ceiling, max per-slot and any-slot chance of hitting the
            ceiling), displaced_from and no_u_bahn_alternative.
        """
        ro = self.reroute_options(line, a, b, walk_km)
        t0 = _to_dt(at)
        t1 = t0 + pd.Timedelta(minutes=minutes) - pd.Timedelta('1min')
        idx = self.slots(t0, t1)
        # displaced demand: interior stations left with no line at all; interior stations served by another
        # line (e.g. an interchange) keep their passengers, who change line on the spot
        donors = [s for s in ro['interior_without_service'] if not ro['per_station_alternatives'][s]['other_lines_here']]
        stay = [s for s in ro['interior_without_service'] if s not in donors]
        recv = {}
        for s in donors:
            alts = ro['per_station_alternatives'][s]['walkable_alternatives_km']
            if not alts:
                continue
            wts = np.array([1 / max(d, 0.2) for _, d in alts])
            for (k, _), wv in zip(alts, wts / wts.sum()):
                recv.setdefault(k, []).append((s, wv))
        ends = [ro['section'][0], ro['section'][-1]]
        watch = sorted(set(list(recv) + ends + stay + [k for k in (ro['network_detour_between_section_ends'] or [])]))
        allst = sorted(set(watch + donors))
        lam = self.expected(t0, t1 + pd.Timedelta('1min'), allst, weather=weather)
        plan = lam.copy()
        for k, lst in recv.items():
            for s, wv in lst:
                plan[k] += displaced_share * wv * lam[s]
        cap = self.ceiling[watch].values
        def summary(df):
            p = np.exp(-cap / np.maximum(df[watch].values, 1e-9))
            return pd.DataFrame({'lambda_mean': df[watch].mean().values, 'ceiling': cap,
                                 'p_at_ceiling_max_slot': p.max(axis=0), 'p_any_slot': 1 - np.prod(1 - p, axis=0)},
                                index=pd.Index(watch, name='station')).round(3)
        return {'reroute': ro, 'slots': [t.strftime('%Y-%m-%d %H:%M') for t in idx],
                'as_in_data': summary(lam).sort_values('p_any_slot', ascending=False),
                'planning_case': summary(plan).sort_values('p_any_slot', ascending=False),
                'displaced_from': {s: round(float(lam[s].sum()), 0) for s in donors},
                'no_u_bahn_alternative': [s for s in donors if s not in {x for lst in recv.values() for x, _ in lst}]}

    def _path_lines(self, path):
        out = []
        for x, y in zip(path[:-1], path[1:]):
            common = set(self.stations.at[x, 'lines']) & set(self.stations.at[y, 'lines'])
            out.append(sorted(common)[0] if common else '?')
        segs = []
        for (x, y), l in zip(zip(path[:-1], path[1:]), out):
            if segs and segs[-1][0] == l:
                segs[-1][2] = y
            else:
                segs.append([l, x, y])
        return [f'{l}: {x} -> {y}' for l, x, y in segs]

    def energy_estimate(self, line, daily_flow):
        """Daily energy estimate (MWh) for `line` from its daily ridership (sum of its stations'
        flows over a service day). Returns (estimate, residual standard deviation)."""
        b0, b1, sd = self.energy_model[line]
        return b0 + b1 * daily_flow, sd

    def network_status(self, at):
        """Closures in force at time `at`, as parse_closure() records."""
        t = _to_dt(at)
        return [c for c in self._clist if c['start'] <= t < c['end']]

    def data_status(self):
        """What is loaded: files, date ranges, counts, and anything the engine could not interpret
        (event addresses it cannot locate, closure descriptions it cannot parse)."""
        ev, cl = self.events, self._clist
        unparsed = [c['description'] for c in cl if c['kind'] == 'unknown' or c['issues']]
        return {
            'data_dir': self.data_dir, 'files': self.loaded_files, 'fitted_at': getattr(self, 'fitted_at', None),
            'flows': f'{self.flows.index.min():%Y-%m-%d %H:%M} -> {self.flows.index.max():%Y-%m-%d %H:%M} '
                     f'({self.flows.shape[0]} slots x {self.flows.shape[1]} stations)',
            'weather': f'{self.weather.index.min():%Y-%m-%d} -> {self.weather.index.max():%Y-%m-%d}',
            'events': f'{len(ev)} listings, {ev.start.min():%Y-%m-%d} -> {ev.start.max():%Y-%m-%d}',
            'closures': f'{len(cl)} ({sum(c["kind"] == "station" for c in cl)} station closures, '
                        f'{sum(c["kind"] == "line" for c in cl)} line suspensions, {len(unparsed)} not understood)',
            'energy': f'{self.energy.index.min():%Y-%m-%d} -> {self.energy.index.max():%Y-%m-%d}',
            'events_not_located': sorted(ev[ev.venue_key.isna()].address.unique().tolist()),
            'closures_not_understood': unparsed,
            'extra_venues': getattr(self, 'extra_venues', []),
            'venues_with_learned_station_split': len(getattr(self, 'venue_learned', {})),
        }

    def calibration(self, start, end):
        """Daily observed / expected for the whole network over [start, end]: a quick check that new data
        behaves like the training data (values far from 1 point to something the model does not know)."""
        lam = self.expected(start, end)
        er = self.expected_reading(lam.values, self.ceiling[list(lam.columns)].values[None, :])
        obs = self.flows.reindex(lam.index).values
        sd = self.service_day(lam.index)
        df = pd.DataFrame({'observed': pd.Series(np.nansum(obs, axis=1), index=lam.index).groupby(sd).sum(),
                           'expected': pd.Series(er.sum(axis=1), index=lam.index).groupby(sd).sum()})
        df['ratio'] = (df.observed / df.expected).round(3)
        per_station = pd.Series(np.nansum(obs, axis=0) / er.sum(axis=0), index=lam.columns)
        o_sd = pd.DataFrame(obs, index=lam.index, columns=lam.columns).groupby(sd).sum()
        e_sd = pd.DataFrame(er, index=lam.index, columns=lam.columns).groupby(sd).sum()
        r_sd = (o_sd / e_sd).where(e_sd > 2000).stack()
        far = r_sd.loc[np.log(r_sd).abs().sort_values(ascending=False).index[:6]]
        return {'by_day': {f'{d:%Y-%m-%d}': r for d, r in df.ratio.items()},
                'station_days_furthest_from_model': [{'day': f'{d:%Y-%m-%d}', 'station': k, 'observed_over_expected': round(float(v), 2)}
                                                     for (d, k), v in far.items()],
                'stations_most_above_model': per_station.sort_values(ascending=False).head(5).round(2).to_dict(),
                'stations_most_below_model': per_station.sort_values().head(5).round(2).to_dict()}

    def model_card(self):
        """Plain-language summary of the fitted rules, for the agent to cite when explaining itself."""
        p = self.weather_params
        big = self.w.sort_values(ascending=False)
        return {
            'data_range': f'{self.flows.index.min():%Y-%m-%d %H:%M} -> {self.flows.index.max():%Y-%m-%d %H:%M}',
            'service_day': '05:00 -> 00:45 (slots after midnight belong to the previous day)',
            'noise': 'each 15-min value = expected level x random exponential factor (single readings are very noisy)',
            'profile': 'weekday peaks 08:00 and 18:00, trough ~12:30; weekends ~60 % of weekday',
            'typical_station_level': float(round(self.w.median(), 1)),
            'busiest_stations': [k for k in big.index[:7]],
            'ceiling': 'hard per-station ceiling on 15-min flow: 500-527 for most, up to %d (%s)' % (self.ceiling.max(), self.ceiling.idxmax()),
            'heat_effect_per_degC_of_daily_mean': f"{(np.exp(p['heat_per_degC']) - 1) * 100:.1f} %",
            'rain_effect_max': f"+{(np.exp(p['rain_log_max']) - 1) * 100:.0f} % (half reached at ~{p['rain_scale_mm_h'] * np.log(2):.1f} mm/h)",
            'events': 'arrivals ~57 % of attendance over the 90 min to the start; departures ~55 % over 2 h after the end',
            'station_closure': 'zero during closure; ~56 % of lost volume returns in the 45 min after reopening',
            'line_suspension': 'no measurable effect on station flows or energy in the training data',
            'energy': 'daily MWh per line = a + b x daily ridership of the line (residual < 1 MWh)',
        }


if __name__ == '__main__':
    import sys
    pd.set_option('display.width', 200)
    eng = UBahnEngine(sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads')
    for k, v in eng.model_card().items():
        print(f'{k:>36}: {v}')
    # Example question 1 - premise check, then both plausible readings
    try:
        eng.section('U8', 'Hermannplatz', 'Neukölln')
    except ValueError as e:
        print('\nQ1 premise check:', str(e).split('.')[0])
    r = eng.suspension_scenario('U8', 'Hermannplatz', 'Hermannstr.', '2026-09-21 17:45', minutes=20)
    print(r['reroute']['per_station_alternatives'])
    print(r['as_in_data'], r['planning_case'], sep='\n')
    # Example question 2 - Uber Arena (ex Mercedes-Benz Arena), 21:00-23:15
    q2 = eng.event_impact('Mercedes-Benz Arena', '2026-09-23 21:00', end='2026-09-23 23:15', attendance=17000,
                          weather={'tmean': 15, 'prcp': 0})
    print('\nQ2 shares', q2['station_shares'])
    print((q2['baseline_lambda'] + q2['extra_lambda']).loc['2026-09-23 23:00':].round(0))
    # Example question 3 - InnoTrans day 1, bad weather
    innotrans = [{'venue': 'Messedamm 22', 'start': '2026-09-22 09:00', 'end': '2026-09-22 18:00', 'attendance': 40000}]
    print('\nQ3', eng.capacity_risk('2026-09-22 05:00', '2026-09-23 00:45', weather={'tmean': 13, 'prcp': 1.5},
                                    events=[], extra_events=innotrans, closures=[], top=5))
