"""Zentraler Daten-Loader für das Talk-To-My-Train Projekt.

Kapselt sämtliche Eigenheiten des Hackathon-Rohdatensatzes an genau einer
Stelle, damit Tools und Agent mit sauberen, typisierten DataFrames arbeiten.
Liest ausschließlich lokale Dateien – kein Netzwerkzugriff.

Behandelte Fallstricke (siehe CLAUDE.md, "KRITISCHE Dateneigenheiten"):
  * Doppelter Spaltenname "U Stadtmitte (Berlin)" in flows -> positions-basiertes
    Mapping auf station_id statt name-basiertem Lookup.
  * Vier verschiedene Datumsformate -> ein zentraler Parser je Format.
  * Unbenannte erste Spalten ("Unnamed: 0") in weather und energy.
  * closures.duration als String ("3h30min") -> pd.Timedelta.

Zeitreihen (flows, weather, events, closures, energy) werden per Glob unter
data/ gefunden – auch in Unterordnern – und zusammengefügt. Neue Dateien für
22.–30.09. (z. B. data/test dataset/flows_post_innotrans.csv) werden damit
ohne Codeänderung mitgeladen; überlappende Zeitstempel gewinnt die zuletzt
sortierte Datei. Der Loader hält keinen Zustand über Instanzen hinweg, jede
neue Instanz sieht den aktuellen Dateistand.
"""

from __future__ import annotations

import datetime
import itertools
import re
from pathlib import Path

import networkx as nx
import pandas as pd

# Repo-Root aus der Modulposition ableiten: src/loader.py -> Repo-Root.
# So funktionieren die Pfade unabhängig vom aktuellen Arbeitsverzeichnis,
# ohne dass ein absoluter Rechnerpfad im Code steht.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DATA_DIR = DATA_ROOT / "training dataset"

# utf-8-sig entfernt ein BOM am Dateianfang: die Testdateien (*_rest.csv)
# beginnen mit einem BOM, sonst hiesse die erste Spalte "\ufeffwhen".
ENCODING = "utf-8-sig"

# Spalten der Eventdatei. Die Testdatei kommt ohne Kopfzeile.
EVENT_COLUMNS = [
    "event_name", "began_local", "estimated_end_local", "venue_name", "address",
    "city", "country", "segment", "genre", "description",
    "estimated_attendance", "event_url",
]

# Datumsformate der einzelnen Rohdateien
TS_FORMAT = "%m/%d/%Y %H:%M"   # flows, weather, closures
DATE_FORMAT = "%m/%d/%Y"       # energy

# Stammdaten: feste Namen. Bevorzugt die Kopie im Trainingsordner.
STATIC_FILES = {
    "stations": "stations_with_ubahn.csv",
    "connections": "berlin_ubahn_connections.csv",
    "lines": "berlin_ubahn_lines_used.csv",
}

# Zeitreihen: alle passenden Dateien unter data/ (rekursiv) werden verbunden.
SERIES_PATTERNS = {
    "flows": "*flow*.csv",
    "weather": "*weather*.csv",
    "events": "*event*.csv",
    "closures": "*closure*.csv",
    "energy": "*energy*.csv",
}

# Slots pro vollem Betriebstag: 00:00-00:45 (4) + 05:00-23:45 (76).
# Die Betriebspause 01:00-04:45 ist keine Datenluecke. ">=" statt "==",
# falls neue Daten den Tag lueckenlos (96 Slots) liefern.
FULL_DAY_SLOTS = 80

_DURATION_RE = re.compile(r"(\d+)h(?:(\d+)min)?")

# Datumsgrenzen je Flow-Dateistand (Pfad, Aenderungszeit, Groesse). Eine neue
# oder geaenderte Datei erzeugt einen neuen Schluessel – /reload braucht
# deshalb keinen Cache zu leeren.
_DATES_CACHE: dict[tuple, dict[str, str]] = {}


def parse_duration(value: str) -> pd.Timedelta:
    """Wandelt einen Dauer-String der Form "3h30min" / "4h" in ein Timedelta."""
    match = _DURATION_RE.match(str(value).strip())
    if match is None:
        raise ValueError(f"Nicht parsebare Dauer: {value!r}")
    hours, minutes = int(match.group(1)), int(match.group(2) or 0)
    return pd.Timedelta(hours=hours, minutes=minutes)


def repair_mojibake(text):
    """Repariert doppelt kodierte Umlaute: "KurfÃ¼rstenstr." -> "Kurfürstenstr.".

    Die Testdateien wurden als UTF-8 gespeichert, nachdem sie schon einmal
    als cp1252 gelesen worden waren. Nur Texte mit den typischen Mustern
    werden angefasst; alles andere bleibt unverändert.
    """
    if not isinstance(text, str) or ("Ã" not in text and "â€" not in text):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _parse_datetimes(values: pd.Series, fmt: str) -> pd.Series:
    """Parst mit dem bekannten Format; neue Dateien dürfen davon abweichen."""
    try:
        return pd.to_datetime(values, format=fmt)
    except (ValueError, TypeError):
        return pd.to_datetime(values, format="mixed")


class DataLoader:
    """Lädt die Rohdateien des Hackathon-Datensatzes.

    Attribute:
        data_dir: Wurzel der Dateisuche (Default: data/, rekursiv).
        col_to_station_id: Mapping Flow-Spaltenname -> station_id. Wird von
            load_flows() positions-basiert befüllt und löst damit die
            Stadtmitte-Doppeldeutigkeit auf.
        files_used: je Datensatz die tatsächlich gelesenen Dateien.
    """

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_ROOT
        self.col_to_station_id: dict[str, str] = {}
        self.files_used: dict[str, list[str]] = {}
        self._dates: dict[str, str] | None = None

    # ------------------------------------------------------------------ #
    # Dateisuche
    # ------------------------------------------------------------------ #

    def _static_path(self, key: str) -> Path:
        """Pfad einer Stammdatei; die Kopie im Trainingsordner hat Vorrang."""
        name = STATIC_FILES[key]
        candidates = sorted(self.data_dir.rglob(name))
        if not candidates:
            raise FileNotFoundError(f"Rohdatei nicht gefunden: {name} unter {self.data_dir}")
        preferred = [p for p in candidates if p.parent == DATA_DIR]
        return (preferred or candidates)[0]

    def series_files(self, key: str) -> list[Path]:
        """Alle Dateien eines Zeitreihen-Datensatzes, Trainingsdaten zuerst.

        Sortierung: Trainingsordner vor allen anderen, danach nach Pfad. Bei
        überlappenden Zeitstempeln gewinnt so die neuere (Nicht-Trainings-)Datei.
        """
        files = [p for p in self.data_dir.rglob(SERIES_PATTERNS[key]) if p.is_file()]
        if not files:
            raise FileNotFoundError(
                f"Keine Datei für {key} ({SERIES_PATTERNS[key]}) unter {self.data_dir}"
            )
        return sorted(files, key=lambda p: (p.parent != DATA_DIR, str(p)))

    def _read_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, encoding=ENCODING, **kwargs)

    def _read_series(self, key: str) -> list[pd.DataFrame]:
        files = self.series_files(key)
        self.files_used[key] = [str(p.relative_to(self.data_dir)) for p in files]
        return [self._read_csv(p) for p in files]

    @staticmethod
    def _rename_first_column(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
        """Benennt eine unbenannte erste Spalte ("Unnamed: 0") um.

        Trägt die Spalte bereits den Zielnamen, bleibt der Frame unverändert.
        """
        first = df.columns[0]
        if first != new_name:
            df = df.rename(columns={first: new_name})
        return df

    # ------------------------------------------------------------------ #
    # Netz-Stammdaten
    # ------------------------------------------------------------------ #

    def load_stations(self) -> pd.DataFrame:
        """168 U-Bahn-Stationen mit VBB-ID, Name, Koordinaten und Linien.

        Achtung: station_name ist NICHT eindeutig ("U Stadtmitte (Berlin)"
        existiert zweimal). Eindeutiger Schlüssel ist station_id.
        """
        df = self._read_csv(self._static_path("stations"))
        df["station_id"] = df["station_id"].astype("string").str.strip()
        df["station_name"] = df["station_name"].astype("string").str.strip()
        df["u_bahn_lines"] = df["u_bahn_lines"].astype("string").str.strip()
        df["longitude"] = df["longitude"].astype(float)
        df["latitude"] = df["latitude"].astype(float)

        if df["station_id"].isna().any():
            raise ValueError("stations_with_ubahn.csv enthält leere station_id")
        if df["station_id"].duplicated().any():
            raise ValueError("stations_with_ubahn.csv enthält doppelte station_id")
        return df

    def load_connections(self) -> pd.DataFrame:
        """182 bidirektionale Kanten zwischen Stationen (station_id-Paare)."""
        df = self._read_csv(self._static_path("connections"))
        for col in ("station_id_1", "station_id_2"):
            df[col] = df[col].astype("string").str.strip()
        return df

    def load_lines(self) -> pd.DataFrame:
        """Linien-Metadaten der 8 enthaltenen Linien (U4 fehlt im Datensatz)."""
        df = self._read_csv(self._static_path("lines"))
        df["line_id"] = df["line_id"].astype("string").str.strip()
        df["line_name"] = df["line_name"].astype("string").str.strip()
        return df

    def load_graph(self) -> nx.Graph:
        """Ungerichteter Stationsgraph inklusive Umsteigekanten.

        Der Datensatz führt "U Stadtmitte (Berlin)" als zwei Knoten (U6
        900100011, U2 900100701) ohne Kante dazwischen – der reale
        Bahnsteigwechsel fehlt. Stationen gleichen Namens werden deshalb mit
        einer Umsteigekante (transfer=True) verbunden.
        """
        stations = self.load_stations()
        connections = self.load_connections()
        graph = nx.Graph()
        graph.add_nodes_from(stations["station_id"].tolist())
        graph.add_edges_from(
            connections[["station_id_1", "station_id_2"]].itertuples(index=False, name=None)
        )
        for name, group in stations.groupby("station_name"):
            ids = group["station_id"].tolist()
            for a, b in itertools.combinations(ids, 2):
                if not graph.has_edge(a, b):
                    graph.add_edge(
                        a, b, transfer=True, line="transfer", weight=1,
                        description=f"In-station transfer: {name}",
                    )
        return graph

    # ------------------------------------------------------------------ #
    # Zeitreihen
    # ------------------------------------------------------------------ #

    def load_flows(self) -> pd.DataFrame:
        """Fahrgastströme, 15-Minuten-Raster, 168 Stationsspalten.

        Setzt zusätzlich self.col_to_station_id. Das Mapping ist bewusst
        positions-basiert: die Spaltenreihenfolge in flows entspricht der
        Zeilenreihenfolge in stations_with_ubahn.csv. Ein name-basiertes
        Mapping würde an "U Stadtmitte (Berlin)" / "... (Berlin).1"
        stillschweigend eine Station verlieren. Alle Flow-Dateien müssen
        dieselben Spalten in derselben Reihenfolge haben.
        """
        frames = []
        reference_cols: list[str] | None = None
        for path, df in zip(self.series_files("flows"), self._read_series("flows")):
            df = self._rename_first_column(df, "timestamp")
            df.columns = [repair_mojibake(c) for c in df.columns]
            cols = [c for c in df.columns if c != "timestamp"]
            if reference_cols is None:
                reference_cols = cols
            elif cols != reference_cols:
                raise ValueError(
                    f"Spalten in {path.name} weichen von der ersten Flow-Datei ab "
                    "– positions-basiertes Mapping wäre falsch."
                )
            df["timestamp"] = _parse_datetimes(df["timestamp"], TS_FORMAT)
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        flow_cols = [c for c in df.columns if c != "timestamp"]

        station_ids = self.load_stations()["station_id"].tolist()
        if len(flow_cols) != len(station_ids):
            raise ValueError(
                f"Spaltenzahl in flows ({len(flow_cols)}) passt nicht zur "
                f"Stationszahl ({len(station_ids)}) – positions-basiertes "
                "Mapping waere falsch."
            )
        self.col_to_station_id = dict(zip(flow_cols, station_ids))

        df = df.drop_duplicates(subset="timestamp", keep="last")
        return df.set_index("timestamp").sort_index()

    def load_weather(self) -> pd.DataFrame:
        """Wetter im selben 15-Minuten-Raster wie flows (1:1-Join möglich).

        coco und cldc sind auf 15 Minuten interpoliert und enthalten
        Nachkommawerte – vor kategorialer Auswertung runden.
        """
        frames = []
        for df in self._read_series("weather"):
            df = self._rename_first_column(df, "timestamp")
            df["timestamp"] = _parse_datetimes(df["timestamp"], TS_FORMAT)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="timestamp", keep="last")
        return df.set_index("timestamp").sort_index()

    def load_events(self) -> pd.DataFrame:
        """Berliner Events. Zeitstempel sind ISO 8601 mit +02:00-Offset.

        Beide Zeitspalten werden nach Europe/Berlin konvertiert, damit sie
        mit den naiven Ortszeit-Stempeln der übrigen Dateien vergleichbar sind.
        """
        frames = []
        for path in self.series_files("events"):
            df = self._read_csv(path)
            if "event_name" not in df.columns:
                # Testdatei ohne Kopfzeile: erste Zeile ist ein Event.
                df = self._read_csv(path, header=None, names=EVENT_COLUMNS)
            frames.append(df)
        self.files_used["events"] = [
            str(p.relative_to(self.data_dir)) for p in self.series_files("events")
        ]
        df = pd.concat(frames, ignore_index=True)
        for col in ("event_name", "venue_name", "address", "genre", "description"):
            df[col] = df[col].map(repair_mojibake)
            if df[col].dtype == object:
                df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
        for col in ("began_local", "estimated_end_local"):
            df[col] = pd.to_datetime(
                df[col], format="ISO8601", utc=True
            ).dt.tz_convert("Europe/Berlin")
        df["estimated_attendance"] = df["estimated_attendance"].astype("Int64")
        df = df.drop_duplicates(
            subset=["event_name", "began_local", "venue_name", "address"], keep="last"
        )
        return df.reset_index(drop=True)

    def load_closures(self) -> pd.DataFrame:
        """Störungen mit geparster Dauer und berechnetem end_time."""
        df = pd.concat(self._read_series("closures"), ignore_index=True)
        df["when"] = _parse_datetimes(df["when"], TS_FORMAT)
        df["duration"] = df["duration"].map(parse_duration)
        df["end_time"] = df["when"] + df["duration"]
        df = df.drop_duplicates(subset=["when", "description"], keep="last")
        return df.sort_values("when").reset_index(drop=True)

    def load_energy(self) -> pd.DataFrame:
        """Täglicher Energieverbrauch je Linie in MWh, date als Index.

        Tagesgranular – nicht mit dem 15-Minuten-Raster der flows joinbar.
        """
        frames = []
        for df in self._read_series("energy"):
            df = self._rename_first_column(df, "date")
            df["date"] = _parse_datetimes(df["date"], DATE_FORMAT)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="date", keep="last")
        return df.set_index("date").sort_index()

    # ------------------------------------------------------------------ #
    # Datumsgrenzen – aus den Daten abgeleitet, nie fest verdrahtet
    # ------------------------------------------------------------------ #

    def data_dates(self) -> dict[str, str]:
        """Datumsgrenzen des aktuell geladenen Flow-Bestands (YYYY-MM-DD).

        first_calendar_day  erster Tag mit überhaupt einem Slot
        first_day           erster vollständiger Betriebstag
        last_day            letzter vollständiger Betriebstag
        last_weekday / last_saturday / last_sunday  bis einschließlich last_day
        """
        if self._dates is not None:
            return self._dates

        signature = tuple(
            (str(p), p.stat().st_mtime_ns, p.stat().st_size)
            for p in self.series_files("flows")
        )
        if signature not in _DATES_CACHE:
            flows = self.load_flows()
            counts = flows.groupby(flows.index.normalize()).size()
            full = counts[counts >= FULL_DAY_SLOTS].index
            first_cal = flows.index.min().normalize()
            first = full[0] if len(full) else first_cal
            last = full[-1] if len(full) else flows.index.max().normalize()

            def back_to(pred) -> str:
                day = last.date()
                while not pred(day):
                    day -= datetime.timedelta(days=1)
                return day.isoformat()

            _DATES_CACHE.clear()
            _DATES_CACHE[signature] = {
                "first_calendar_day": first_cal.date().isoformat(),
                "first_day": first.date().isoformat(),
                "last_day": last.date().isoformat(),
                "last_weekday": back_to(lambda d: d.weekday() < 5),
                "last_saturday": back_to(lambda d: d.weekday() == 5),
                "last_sunday": back_to(lambda d: d.weekday() == 6),
            }
        self._dates = _DATES_CACHE[signature]
        return self._dates

    @property
    def data_first_day(self) -> str:
        """Erster vollständiger Betriebstag der geladenen Flow-Daten."""
        return self.data_dates()["first_day"]

    @property
    def data_last_day(self) -> str:
        """Letzter vollständiger Betriebstag der geladenen Flow-Daten."""
        return self.data_dates()["last_day"]

    @property
    def data_last_weekday(self) -> str:
        """Jüngster Montag–Freitag bis einschließlich data_last_day."""
        return self.data_dates()["last_weekday"]

    @property
    def data_last_saturday(self) -> str:
        return self.data_dates()["last_saturday"]

    @property
    def data_last_sunday(self) -> str:
        return self.data_dates()["last_sunday"]

    # ------------------------------------------------------------------ #
    # Sammel-Loader
    # ------------------------------------------------------------------ #

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Lädt alle acht Datensätze und füllt dabei col_to_station_id.

        load_flows() läuft vor der Rückgabe, das Mapping steht danach also
        auf der Instanz bereit.
        """
        data = {
            "stations": self.load_stations(),
            "connections": self.load_connections(),
            "lines": self.load_lines(),
            "flows": self.load_flows(),
            "weather": self.load_weather(),
            "events": self.load_events(),
            "closures": self.load_closures(),
            "energy": self.load_energy(),
        }
        if not self.col_to_station_id:
            raise RuntimeError("col_to_station_id wurde nicht befüllt")
        return data


__all__ = ["DataLoader", "DATA_DIR", "DATA_ROOT", "REPO_ROOT", "FULL_DAY_SLOTS",
           "parse_duration"]
