"""Agent-Orchestrierung – Drei-Stufen-Loop.

    ROUTE   regelbasierte Toolauswahl aus der Frage (kein LLM-Call)
    GATHER  Tools aufrufen, Fehler sammeln statt abbrechen
    ANSWER  Prompt bauen, Azure-Endpoint aufrufen

Reine Logik, kein HTTP. Kein globaler State und kein globaler DataLoader –
jeder answer()-Aufruf ist in sich abgeschlossen, damit der Agent später
parallel aus mehreren MCP-Sessions bedient werden kann.
"""

from __future__ import annotations

import datetime
import difflib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

from src.fallback import build_fallback_answer
from src.loader import STATIC_FILES, DataLoader
from src.tools import (
    analyze_disruption_routing,
    call_llm,
    compare_station_to_network,
    detect_anomalies,
    find_alternative_routes,
    find_diverse_transit_routes,
    find_station_dependencies,
    find_stations_above_threshold,
    find_transit_route,
    find_weather_anomalies,
    get_closure_impact,
    get_closures,
    get_critical_stations,
    get_energy,
    get_events,
    get_busiest_station_by_weekday,
    get_network_flow_summary,
    get_peak_hours,
    get_peak_profile,
    get_rain_impact,
    get_station_flow,
    get_station_peak_15min,
    get_station_peak_vs_baseline,
    get_station_weekday_pattern,
    get_temporal_context,
    get_weather,
    get_weekday_profile,
)
from src.tools._common import normalize_station, station_tokens
from src.tools.temporal_tool import get_weekly_summary
from src.tools.gtfs_tool import (
    find_route_between_stations,
    find_stations_in_text,
    get_all_ubahn_lines,
    get_lines_for_station,
    get_stops_for_line,
)

# Hartes Zeitlimit fuer den LLM-Aufruf. 24 s + Tools (< 1 s) + Netz bleiben
# unter den 28 s des Servers. 18 s rissen 3 der 5 offiziellen Testfragen. Reisst es, antwortet der Agent
# deterministisch aus den Tool-Ergebnissen (src/fallback.py).
LLM_TIMEOUT_SECONDS = 24

# Der LLM-Aufruf laeuft in einem eigenen Thread, damit das Zeitlimit hart
# greift – urllib-Timeouts gelten nur je Socket-Operation, nicht fuer die
# Gesamtdauer. Ein abgebrochener Aufruf laeuft im Hintergrund aus.
_LLM_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")

MAX_CONTEXT_CHARS = 6000

# Investitionsfragen: 4 Tools a 2000 Zeichen, siehe _format_context.
INVEST_CONTEXT_CHARS = 8000

# Eventfragen: je Station Fluss mit Baseline plus Peak fuer An- und Abreise
# (EVENT QUERY FORMAT). Mit 6000 Zeichen fielen die Stationsfluesse und damit
# die Spalte "vs. Baseline" heraus.
EVENT_CONTEXT_CHARS = 12000
EVENT_TOOL_CHARS = 2000

# Felder eines Peak-Ergebnisses, die im LLM-Kontext nur doppeln
# (peak_time/date/pct_of_own_max tragen dieselbe Aussage).
PEAK_CONTEXT_DROP = {"scope", "peak_timestamp", "data_points", "capacity_note",
                     "slots_at_own_max"}
INVEST_TOOL_CHARS = 2000
INVEST_CONTEXT_PRIORITY: dict[str, int] = {
    "get_critical_stations": 1,
    "get_energy": 2,
    "get_closures": 3,
    "get_network_flow_summary": 4,
}

# InnoTrans 2026 findet vom 22. bis 25.09.2026 auf dem Messegelaende statt.
# Das ist ein Veranstaltungsdatum, keine Datengrenze.
INNOTRANS_DAYS = ("2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25")

# Wochenfragen: "last week" = letzte vollstaendige Woche, "this week" =
# laufende Woche bis zum letzten Datentag. Ohne diese Erkennung griffe die
# "recent"-Regel ("last") und werte nur einen einzigen Tag aus.
WEEK_RE = re.compile(
    r"\b(?:(last|past|previous|this|current)\s+week|"
    r"(letzte[nr]?|vergangene[nr]?|vorige[nr]?|diese[nr]?)\s+woche)\b",
    flags=re.IGNORECASE,
)
YESTERDAY_RE = re.compile(r"\b(yesterday|gestern)\b", flags=re.IGNORECASE)

# Tageszeit ohne Uhrzeit ("Monday mornings") fuer Stationsranglisten.
DAYPART_HOURS = {
    "morning": (6, 10), "morgen": (6, 10), "vormittag": (9, 12),
    "midday": (11, 14), "mittag": (11, 14), "afternoon": (14, 17),
    "nachmittag": (14, 17), "evening": (16, 20), "abend": (16, 20),
    "night": (20, 24), "nacht": (20, 24),
}

# Gesperrter Streckenabschnitt ("suspended between A and B").
SUSPENSION_RE = re.compile(
    r"\b(suspend\w*|closed|closure|gesperrt|sperrung|unterbroch\w*|"
    r"out of service|not running|ausfall)\b",
    flags=re.IGNORECASE,
)

# "Jetzt"-Fragen ohne Datum ("Is there a disruption today?").
TODAY_RE = re.compile(
    r"\b(today|now|current|currently|right now|heute|aktuell|aktuelle[nrs]?|jetzt|gerade)\b",
    flags=re.IGNORECASE,
)

# Datensatz-Grenzen werden aus den geladenen Daten abgeleitet
# (DataLoader.data_dates), nie fest verdrahtet: am Finaltag kommen Daten fuer
# 22.–30.09. hinzu. Fehlt ein Datum in der Frage, gilt der letzte volle Tag
# und die Annahme wird im Kontext ausgewiesen.


def _data_dates() -> dict[str, str]:
    """Aktuelle Datumsgrenzen des Flow-Bestands (gecacht je Dateistand)."""
    return DataLoader().data_dates()


FORECAST_PROMPT = """
QUESTION TYPE NOTE:
This is a scenario / forecast question. Use historical patterns as the basis
and label every such statement "based on historical patterns". No guarantees
about the future. The dataset ends on the last day of the DATA RANGE (see the
question block); statements about later periods are extrapolations, not
measurements. The LANGUAGE RULE applies unchanged.
"""

SYSTEM_PROMPT = """You are an AI assistant for Berlin U-Bahn operations.
You answer questions from operators using operational data for the period
given as DATA RANGE with each question.

== FORMATTING RULES ==
- NEVER use markdown headings (# ## ###). They are forbidden.
- Use **bold text** for section labels, followed by a colon and a line break.
- Use bullet points (–) for lists and recommendations.
- Use markdown tables for structured comparisons (station lists, metrics).
- Section order for data answers:
    **Situation:** one sentence summary of what the data shows
    **Data:** table or bullet list of key figures
    **Recommendations:** bullet list, max 5 points, actionable and short
    **Limitations:** one short sentence on data limits, only if relevant
- For hotspot answers (alternatives table): no section labels needed, just
  the table + the clarifying question at the end.
- Source citations ([Hotspot plan], [VBB timetable], [General knowledge]):
  write them ONLY ONCE as a footnote at the very end of the answer, never
  inline in table cells or after every sentence. Only when the tool results
  contain hotspot alternatives, add exactly:
  _Source: Hotspot planning values (data/hotspots.json) — no measured data._
  Never add this line to any other answer.
- No introduction, never start with "Based on the data" or similar.
- Bold station names, times, passenger numbers and line names.
- No tool names or internal field names (not "get_events", not
  "mean_window_total").

== OPERATOR DATA DENSITY RULES ==
- You are a decision-support tool for U-Bahn operators. When data
  is available, ALWAYS include specific numbers, times, and station
  names - never summarize without them.
- For every flow/capacity answer: include peak time, peak value,
  station name, 15-min slot value, % of station historical max,
  and capacity status (CRITICAL / ELEVATED / NORMAL).
- For energy efficiency: report kWh per passenger (= MWh per 1,000 pax) with
  4 decimals exactly as in the tool result, total passengers as whole
  numbers, percentages with 2 decimals, and name the allocation rule
  (interchange passengers split equally between the lines serving them).
- For typical-day / weekday-profile questions use the typical load status
  result: typical peak slot (mean) vs. the station's all-time maximum.
- For every weather answer: include rain intensity (mm/h),
  network flow delta (%), affected time window, and the top-3
  stations by volume in that window.
- For every event answer: include estimated attendance, arrival /
  departure phase, and the 2-3 most affected stations with their
  expected load.
- If data for the exact queried location is not available, say
  so in one sentence, then provide the nearest available data
  point (closest station, closest time window, or network average)
  with explicit labeling: "Nearest available data: ..."
- NEVER answer "no data available" without first checking the
  closest proxy in the tool results. An operator needs a best estimate,
  not a refusal. The proxy must still come from the tool results.
- Always end with a concrete operational recommendation
  ("Increase service frequency on U2 between 07:30-08:30",
  "Deploy additional staff at U Spichernstr. at 07:45").
- If a field above is not in the tool results, leave it out rather than
  estimating it. In a table cell write "Classification unavailable" for a
  missing load status - never "Not supplied".

== EVENT QUERY FORMAT ==
For any question about a specific event or concert, structure the answer
EXACTLY as follows - no deviations (this replaces the generic section order):

**Situation:** [Event name] · [Venue] · [Date] · [Time]
Arrival window: [start-2h]–[start] · Departure window: [end]–[end+1h]

**Arrival Phase**
| Station | Lines | Walk dist | Peak slot | Passengers | vs. Weekday avg | Load status |
[one row per station, ordered by distance to venue]
_Distances are straight-line approximations; actual walking time may differ._
_Baseline: mean of the same time slots on the other same-weekday days in the dataset (full event window)._
Load status format: ⚠️ CRITICAL (XX%) or 🟡 ELEVATED (XX%) or 🟢 NORMAL (XX%)
where XX% = pct_of_own_max of the station's peak result with phase "arrival".
"vs. Weekday avg" = the station flow's change against its baseline if present,
otherwise "–". This value covers the full event window (arrival and
departure together), not the specific peak slot. Label it consistently and
do not imply it is slot-specific.

**Departure Phase**
If peak results with phase "departure" exist: show them in the same table
format, followed by the same baseline note line. If NOT available: write exactly:
"No post-[end time] slot data in dataset. Historical pattern suggests
departure surge [end+0:30]–[end+1:00]. Retain staff at [top 2 stations]."

**Immediate Actions (priority order)**
Sort by: (1) load status - CRITICAL before ELEVATED before NORMAL;
(2) within the same status: walking distance ASCENDING (the closest station
has the highest priority - staff reaches it fastest and overflow is most
immediate). Example: two CRITICAL stations - the one at 386 m comes before
the one at 1,284 m.
1. ⚠️ [Station] from [time] — CRITICAL load, [specific action]
2. 🟡 [Station] from [time] — ELEVATED, [specific action]
3. [Further stations: monitor only if NORMAL]
4. [Departure phase action]

**Limitations**
One sentence only, covering data gaps specific to this answer.

== CLOSURE/DISRUPTION QUERY FORMAT ==
For questions about line suspensions, closures, or disruptions (this replaces
the generic section order):

**Situation:** Line [X] suspended — [section] — [date if known]
Duration: [from closure data if available, otherwise "Duration not in dataset"]
Reason: [from closure data if available, otherwise "Reason not recorded in dataset"]

**Affected Section**
| From station | To station | Duration | Reason |
[rows from closure data]

**Rerouting Options**
| Alternative | Via | Additional travel time | Notes |
[rows from the alternative route results]

**Stations likely to become overloaded**
[list from the closure impact result, with passenger estimates if available]

**Immediate Actions (priority order)**
[numbered list as per the event format]

**Limitations**
[one sentence]

If duration and reason are not in the dataset, say so explicitly in the
Situation block — do not guess or omit the fields.

== SIMULATION CAP NOTE ==
The dataset caps station flow at 3,000 passengers per 15-minute slot.
When a station shows 100% of historical maximum AND the raw value is
exactly 3,000 pax, always add in the Limitations section:
"All stations marked CRITICAL at 100% hit the simulation cap of 3,000 pax/15 min;
actual passenger counts may exceed these values."
Never omit this note when any station shows exactly 3,000 pax.

== WEATHER QUERY FORMAT ==
For weather-related flow questions (this replaces the generic section order):

**Situation:** [Date], [weather event], [intensity]
Affected window: [from]–[to]

**Network Impact**
| Metric | Value |
| Weather peak | [time], [intensity mm/h] |
| Network flow change vs. dry conditions | [+/- %] |
| Network peak slot | [time] — [passengers] pax |

**Top Stations by Volume**
| Rank | Station | Lines | Total volume in window | Peak 15-min slot | Load status |
[rows for top 3–5 stations; load status as in the event format]

**Immediate Actions**
Numbered, same format as event queries.

**Limitations**
One sentence.

== ENERGY EFFICIENCY RECOMMENDATION RULE ==
After listing the ranking, take the gap between rank 1 and rank 2 from the
energy result (the worst-vs-second gap in %) and apply:
1. Gap < 5%: do NOT recommend only the worst line. Recommend a JOINT
   PROGRAMME covering both lines, phrased like:
   "[Line 1] and [Line 2] are near-identical at [x] and [y] kWh/pax
   (gap: [g]%). A joint efficiency programme covering both lines will
   deliver more impact than targeting [Line 1] alone."
2. Gap >= 5% but < 10%: recommend the worst line as primary and the second
   line as "monitor closely".
3. Gap >= 10%: recommend only the worst line as the priority.
For a joint programme always include:
- one shared intervention that applies to both lines (e.g. driving-profile
  optimisation applies equally to both);
- one line-specific factor only if the tool result supports it (e.g. a
  different number of stations served or passenger volume) - otherwise none;
- the note that the allocation method affects the ranking (use the
  allocation sensitivity from the result).
NEVER present a 1-2% difference as a clear winner. At this margin,
operational planning requires a joint approach.

== LANGUAGE RULE ==
- ALWAYS respond in English, regardless of the language the question
  was written in. This is a non-negotiable interface requirement.
- The only exception: if an operator explicitly writes "Antworte auf
  Deutsch" or similar, switch for that one reply only.

If the question is completely outside the data: answer only
"I don't have data for that." plus one sentence on why and what you can say
instead - no section labels in that case.

== ABSOLUTE RULES (never break these) ==
1. Every number in your answer must come from a tool result. Never invent figures.
2. Mark general knowledge (S-Bahn, bus, tram, regional rail) as
   [General knowledge]. Lines and stops from the VBB timetable results are
   data, not general knowledge: mark them [VBB timetable]. Both only once in
   the closing footnote (see FORMATTING RULES).
3. Never claim to know OD flows or actual passenger routing. Say so clearly if asked.
4. U4 is not in the dataset. Say so if U4 is relevant to the question.
5. Flow values are capped at 3,000 per 15-min slot (simulation). Mention if relevant.
6. Event attendance values are scaled estimates, not absolute counts.
7. If a date is outside the DATA RANGE: say so, use historical patterns if available.
8. Follow the LANGUAGE RULE. Use the bold section labels from FORMATTING
   RULES, never markdown headings.
"""

USER_PROMPT = """
OPERATOR QUESTION:
{question}

DATA RANGE: {first_day} to {last_day} (last full operating day: {last_day};
no real-time data)

TOOL RESULTS (the only allowed source for numbers):
{tool_results}

Answer the operator's question with the bold section labels above (no
markdown headings). If the tool results only cover part of the question,
answer that part and name what is missing under the limitations label.
"""

# Kontext fuer Fragen, die keiner Kategorie zugeordnet werden konnten.
# Es gibt keine Tool-Ergebnisse – deshalb darf das LLM keine Zahlen nennen.
FALLBACK_PROMPT = """
The user asked a question that doesn't clearly match a data category.
Use your knowledge of Berlin U-Bahn operations to answer if possible.
If the question is completely unrelated to public transport, decline politely.
No tool results are available for this question, so do not state any passenger,
energy or closure figures. Mark general knowledge once, as a closing
footnote: [General knowledge]. No markdown headings.

QUESTION:
{question}
"""

# Erste Stufe einer Hotspot-Anfrage: kuratierte Ausweichstationen zeigen und
# nach Reiserichtung fragen, bevor Tools laufen.
HOTSPOT_CLARIFY_QUESTION = (
    "Is this for arrival or departure? And what is the destination direction?"
)
HOTSPOT_PROMPT = """
The operator selected a known crowd hotspot. Below are curated alternative
stations for it (planning knowledge, not measured data). Do NOT use section
labels or markdown headings for this message, and do NOT tag table cells.

Write in English (see LANGUAGE RULE):
1. One sentence naming the hotspot and its main problem.
2. A compact markdown table of the alternatives with the columns:
   Alternative Station | Direction | Walk (min) | Notes.
   Translate German direction and note texts into English; keep station
   and line names as they are.
3. One footnote line, exactly:
   _Source: Hotspot planning values (data/hotspots.json) — no measured data._
4. End with exactly this question on its own line:
   {clarify}

No passenger numbers - there are no tool results yet.

OPERATOR QUESTION:
{question}

HOTSPOT:
{hotspot}
"""

# Standardtexte in beiden Sprachen; Auswahl ueber _lang(question).
MESSAGES: dict[str, dict[str, str]] = {
    "out_of_scope": {
        "de": ("Diese Frage liegt außerhalb meines Themenbereichs (Berliner "
               "U-Bahn-Betrieb, Fahrgastströme, Störungen, Events, Wetter, "
               "Energieverbrauch)."),
        "en": ("This question is outside my scope (Berlin U-Bahn operations, "
               "passenger flows, disruptions, events, weather, energy)."),
    },
    "no_data": {
        "de": ("Keine Daten verfügbar. Zu dieser Frage liefern die vorhandenen "
               "Datensätze keine auswertbaren Werte."),
        "en": ("I don't have data for that. The available datasets contain no "
               "values for this question."),
    },
    "need_details": {
        "de": ("Die Frage passt thematisch, aber mir fehlen die nötigen Angaben "
               "(z. B. Station, Datum oder Linie), um die Daten gezielt "
               "auszuwerten. Bitte präzisieren."),
        "en": ("The question is on topic, but I need more details (e.g. station, "
               "date or line) to query the data. Please be more specific."),
    },
    "llm_failed": {
        "de": ("Die Daten wurden ausgewertet, aber die Antwortgenerierung ist "
               "fehlgeschlagen.\n\nGrund: {detail}\n\nDie Rohdaten stehen "
               "unter data_basis und in den Tool-Ergebnissen bereit."),
        "en": ("The data was analysed, but generating the answer failed."
               "\n\nReason: {detail}\n\nThe raw results are listed under "
               "data_basis and in the tool results."),
    },
}

# Keyword-Kategorien fuer das Routing.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "closure": ("sperrung", "gesperrt", "closure", "closed", "shutdown",
                "suspension", "suspended", "suspend", "maintenance",
                "wartung", "betriebspause", "störung", "stoerung",
                "ausfall", "unterbrech"),
    "event": ("event", "konzert", "veranstaltung", "match", "spiel",
              "festival", "messe", "arena", "publikum", "besucher", "innotrans"),
    "weather": ("wetter", "regen", "temperatur", "gewitter", "weather",
                "hitze", "niederschlag", "sturm", "wind"),
    "energy": ("energie", "energy", "verbrauch", "mwh", "effizienz",
               "stromverbrauch"),
    "critical": ("kritisch", "fragmentier", "ausfall", "wichtigste station",
                 "resilien", "artikulation", "engpass"),
    "route": ("alternativ", "umweg", "route", "umleitung", "umleiten",
              "rerouting", "ausweich", "weg von", "verbindung"),
    "flow": ("flow", "fahrgast", "passagier", "auslastung", "peak", "rush",
             "andrang", "überlast", "ueberlast", "aufkommen", "pendler",
             "spitze", "frequenz", "passengers", "passagiere", "how many",
             "wie viele", "count", "anzahl", "between", "zwischen",
             "passed through", "durchgefahren", "total", "gesamt", "volume"),
    "anomaly": ("anomalie", "ungewöhnlich", "ungewoehnlich", "auffällig",
                "auffaellig", "ausreißer", "ausreisser", "abweichung"),
    # Fragen nach dem Normalfall statt nach einem einzelnen Tag.
    "peak": ("peak", "rush", "pendler", "wann ist", "typisch", "typical",
             "durchschnitt", "normalerweise", "üblicherweise", "ueblicherweise",
             "usually", "commute", "tagesprofil", "profil"),
    "compare": ("vergleich", "über dem mittel", "ueber dem mittel",
                "über durchschnitt", "ueber durchschnitt", "höher als",
                "hoeher als", "netzwerkdurchschnitt", "netzdurchschnitt",
                "mittelwert aller", "above network", "above average",
                "above the network", "compared to", "exceed"),
    # Andere Verkehrsträger sind KEIN Out-of-Scope mehr: Operatoren brauchen
    # S-Bahn, Bus und Tram als Umleitungsalternativen. Die U-Bahn-Daten
    # liefern die Zahlen, das Allgemeinwissen die Alternativen.
    # "bus", "sev" und "tram" stehen NICHT hier, sondern in REGEX_KEYWORDS:
    # als Teilstring greift "bus" in "busiest" und "sev" in "several".
    "multimodal": ("umleitung", "alternative", "alternativ", "s-bahn",
                   "sbahn", "s bahn", "ringbahn", "ersatzverkehr", "ersatz",
                   "replacement", "reroute", "reroutin", "rerouted",
                   "bypass", "bvg", "öpnv", "oepnv", "nahverkehr",
                   "regionalbahn", "verkehrsmittel"),
    # Szenario- und Prognosefragen: die Zukunft steht nicht in den Daten,
    # historische Muster schon.
    "forecast": ("forecast", "predict", "prognose", "tomorrow", "next week",
                 "next weekend", "morgen", "nächste woche", "naechste woche",
                 "what if", "scenario", "szenario", "would happen",
                 "will happen", "erwarten", "wenn "),
    # Wochenendfragen brauchen ein Sa/So-Profil, kein Werktagsprofil.
    "weekend": ("weekend", "wochenende", "saturday", "sunday",
                "samstag", "sonntag"),
    # Stammdatenfragen ("Wie viele Stationen gibt es?") brauchen kein
    # Zeitreihen-Tool, nur die Netzkennzahlen.
    "network_info": ("how many stations", "wie viele stationen",
                     "network size", "netzgröße", "netzgroesse",
                     "how many lines", "wie viele linien", "stations in",
                     "lines in", "total stations", "how big", "wie groß",
                     "wie gross", "overview", "überblick", "ueberblick",
                     "network overview", "what stations", "welche stationen",
                     "list of stations", "alle stationen"),
    # Wochentagsmuster über den gesamten Datensatz.
    "temporal": ("weekday", "wochentag", "monday", "tuesday", "wednesday",
                 "thursday", "friday", "saturday", "sunday", "montag",
                 "dienstag", "mittwoch", "donnerstag", "freitag", "samstag",
                 "sonntag", "busiest day", "meistbefahren", "which day",
                 "welcher tag", "welche wochentage", "busiest weekday",
                 "wochenprofil", "weekly pattern", "day of week",
                 "per weekday", "pro wochentag"),
}

# Kategorien, deren Stichwoerter zu kurz oder zu mehrdeutig fuer eine reine
# Teilstring-Suche sind. "last" steckt sonst in "Auslastung", "Ueberlastung"
# und "Belastung" und wuerde fast jede deutsche Flow-Frage als "recent"
# einstufen.
# "last" zaehlt nur vor einer Zeitangabe ("last day", "last Monday",
# "last 24 hours"). Als Verb ("how long will it last") setzte es sonst das
# Datum auf den letzten Datentag – Sperrungsfragen fanden dann nichts.
REGEX_KEYWORDS: dict[str, str] = {
    "recent": r"\b(recent|recently|latest|aktuell|aktuelle[nrs]?|"
              r"zuletzt|neueste[nrs]?|juengste[nrs]?|jüngste[nrs]?)\b|"
              r"\blast\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|year|"
              r"hour|hours|night|evening|morning|weekend|monday|tuesday|"
              r"wednesday|thursday|friday|saturday|sunday|data|available|"
              r"recorded|operating)\b",
    # "bus" steckt in "busiest", "sev" in "several", "tram" in "trampeln" –
    # ohne Wortgrenzen wuerde jede dieser Fragen als Multimodal gelten.
    "multimodal": r"\b(bus|busse|busses|buslinien?|busverkehr|sev|tram|"
                  r"strassenbahn|straßenbahn|transit)\b",
    # Betriebszeiten/Betriebspause und Einordnung einer Uhrzeit – regelbasiert
    # ueber temporal_tool ("Is the U-Bahn running at 3am?", "When does service
    # stop at night?", "What time slot is 17:45?").
    "service_time": r"\b(?:is|are|does|do)\b[^?]*\b(?:running|in service|operating|run)\b|"
                    r"\b(?:service hours|operating hours|betriebszeit\w*|betriebspause|"
                    r"betriebsruhe|in betrieb|nachtbetrieb|betriebsschluss|betriebsbeginn|"
                    r"night service|last train|first train|letzte[rn]? zug|erste[rn]? zug|"
                    r"time slot|zeitslot|tageszeit-?slot)\b|"
                    r"\bservice\b[^?]*\b(?:stop|stops|end|ends|start|starts|begin|begins|pause)\b|"
                    r"\bwhen (?:does|do) (?:the )?(?:u-bahn|ubahn|metro|subway|service|"
                    r"trains?|operations?)\b[^?]*\b(?:stop|end|start|begin|run)\b",
    # "crowd"/"crowds" ist ein Event-Stichwort, "crowded" nicht: "the most
    # crowded stations" fragt nach Fahrgastzahlen, nicht nach Events.
    "event": r"\bcrowds?\b",
}

# Ursachenfragen zu Anomalien ("determine the root causes using all of the
# available data"): Events und Wetter desselben Tages gehoeren in den Kontext.
CAUSE_RE = re.compile(
    r"\b(cause[sd]?|root|explain\w*|why|reasons?|drivers?|all (?:of )?the available data|"
    r"ursache\w*|warum|erklär\w*|erklaer\w*|grund|gründe)\b",
    flags=re.IGNORECASE,
)

# "Stations whose demand depends on another station despite no direct
# connection" -> Korrelation der Abweichungen vom Tagesgang.
DEPENDENCY_RE = re.compile(
    r"\b(dependen\w*|depends on|correlat\w*|co-?mov\w*|abhängig\w*|abhaengig\w*|"
    r"gekoppelt|zusammenhäng\w*)\b",
    flags=re.IGNORECASE,
)
# "Which alternative routes do passengers actually prefer during disruptions?"
ROUTE_BEHAVIOUR_RE = re.compile(
    r"\b(actually prefer\w*|prefer\w*|actually (?:take|use|choose)|route choice|"
    r"passenger behaviou?r|tatsächlich\s+\w+|bevorzug\w*|ausweichverhalten)\b",
    flags=re.IGNORECASE,
)
DISRUPTION_RE = re.compile(
    r"\b(disruption\w*|closure\w*|suspension\w*|sperrung\w*|störung\w*|stoerung\w*)\b",
    flags=re.IGNORECASE,
)
# "Suggest an unconventional alternative route not based on the shortest path"
UNCONVENTIONAL_RE = re.compile(
    r"\b(unconventional|not based on the shortest|other than the shortest|"
    r"non-?shortest|unkonventionell\w*|nicht (?:der |die |den )?kürzeste\w*)\b",
    flags=re.IGNORECASE,
)
CITY_CENTRE = ("S+U Alexanderplatz", "S+U Friedrichstr.")

# "Which stations have more than 2000 passengers per hour?"
THRESHOLD_RE = re.compile(
    r"\b(?:more than|over|above|exceeds?|exceeding|greater than|at least|mehr als|"
    r"über|ueber|mindestens)\s+(\d{1,3}(?:[.,]\d{3})+|\d+)\s*"
    r"(?:passengers|pax|people|persons|fahrgäste|fahrgaeste|personen)?\s*"
    r"(?:per|pro|an|in|/|je|each|a|an)?\s*(?:an?\s+)?"
    r"(hour|stunde|h|slot|15[- ]?min\w*|quarter)?",
    flags=re.IGNORECASE,
)
# "Which is the busiest station in the network?" / "most crowded stations"
BUSIEST_RE = re.compile(
    r"\b(busiest|most crowded|most frequented|most used|highest (?:ridership|traffic|"
    r"flow|demand)|meistfrequentiert\w*|am stärksten (?:frequentiert|belastet)|"
    r"am staerksten (?:frequentiert|belastet)|vollste[nr]?|meisten fahrgäste)\b",
    flags=re.IGNORECASE,
)
# "How does rain affect passenger numbers?" – Wirkung, kein Einzeltag.
WEATHER_EFFECT_RE = re.compile(
    r"\b(affect\w*|impact\w*|influence\w*|effect\w*|correlat\w*|relationship|"
    r"depend\w*|einfluss|auswirk\w*|wirkt|beeinfluss\w*|zusammenhang)\b",
    flags=re.IGNORECASE,
)
# "How many lines does Berlin have?"
LINE_COUNT_RE = re.compile(
    r"\b(how many (?:u-bahn |ubahn |metro |subway )?lines|"
    r"wie viele (?:u-bahn-?)?linien)\b",
    flags=re.IGNORECASE,
)
# "What stations are on U1?" / "U1 stops" – Halteliste einer Linie. Stationswort
# und Linie muessen zusammengehoeren: "U6 suspended ... which stations would
# become overloaded" fragt nicht nach dem Linienverlauf.
_LINE_CODE = r"(?:u\d|s\d{1,2}|m\d{1,2}|x\d{1,3}|n\d{1,2}|\d{2,3})"
LINE_LIST_RE = re.compile(
    r"\b(?:stations|stops|stationen|haltestellen|halte|bahnhöfe|bahnhoefe)\s+"
    r"(?:are\s+|is\s+|does\s+|do\s+)?(?:on|of|along|served by|der|auf|an|entlang)\s+"
    r"(?:the\s+|line\s+|linie\s+|der\s+|die\s+)*" + _LINE_CODE + r"\b"
    r"|\b" + _LINE_CODE + r"\s+(?:stations|stops|stationen|haltestellen)\b",
    flags=re.IGNORECASE,
)
PEAK_DEFINITION_RE = re.compile(
    r"\b(peak hours?|rush hours?|stoßzeit\w*|stosszeit\w*|hauptverkehrszeit\w*)",
    flags=re.IGNORECASE,
)
# "How many stations does U7 have?" / "Wie viele Stationen hat die U6?"
LINE_SIZE_RE = re.compile(
    r"\b(how many (stations|stops)|number of (stations|stops)|"
    r"wie viele (stationen|haltestellen|halte|bahnhöfe|bahnhoefe))\b",
    flags=re.IGNORECASE,
)

# Englische Stichwoerter. Anders als KEYWORDS nur am WORTANFANG gematcht:
# als Teilstring stecken "wet" in "between", "eco" in "second", "count" in
# "account" und "load" in "download" – jede dieser Fragen landete sonst in der
# falschen Kategorie. Deutsche Komposita brauchen dagegen die Teilstring-Suche
# in KEYWORDS ("Stromverbrauch" -> "verbrauch").
EN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "weather": ("rain", "rainfall", "precipitation", "storm", "thunderstorm",
                "temperature", "heat", "cold", "snow", "weather", "climate",
                "wet", "dry", "heatwave", "hot day", "rainy", "windy",
                "forecast", "meteorolog"),
    "flow": ("passenger", "passengers", "ridership", "flow", "count", "total",
             "how many", "volume", "traffic", "load", "capacity", "crowded",
             "busy", "busiest", "quiet", "occupancy", "boarding", "alighting",
             "passed through"),
    "closure": ("closure", "closed", "disruption", "disrupted", "suspended",
                "suspension", "shutdown", "maintenance", "broken down",
                "out of service", "not running", "affected", "incident",
                "accident", "delay", "delayed", "service change"),
    "critical": ("critical", "articulation", "bottleneck",
                 "single point of failure", "most important", "most critical",
                 "key station", "hub", "interchange", "network resilience",
                 "graph", "topology", "connected", "fragmentation",
                 "stations cut off", "components", "split", "disconnect",
                 "fragment"),
    "network_info": ("how many stations", "how many lines", "network size",
                     "network overview", "which lines", "which stations",
                     "list stations", "list lines", "entire network",
                     "whole network", "all lines", "all stations"),
    "anomaly": ("anomaly", "anomalies", "abnormal", "unusual", "unexpected",
                "outlier", "spike", "surge", "drop", "dip", "irregular",
                "strange", "weird flow"),
    "route": ("route", "routing", "path", "alternative", "backup",
              "replacement", "how to get", "fastest way", "best route",
              "avoid", "bypass", "reroute", "detour", "diversi", "get from",
              "travel from", "go from"),
    # "show" bewusst nicht: trifft jedes "Show me ..." und holte dann alle
    # 417 Events in den Kontext.
    "event": ("concert", "event", "match", "game", "festival",
              "performance", "stadium", "arena", "venue", "fans",
              "audience", "spectators", "sold out", "ticket", "lollapalooza",
              "guns n roses", "guns n’ roses", "guns n' roses", "bruno mars"),
    "energy": ("energy", "electricity", "consumption", "efficient",
               "efficiency", "kilowatt", "megawatt", "mwh", "kwh", "power",
               "sustainable", "green", "carbon", "eco", "per passenger"),
    "peak": ("peak", "rush hour", "busiest time", "when is it busy",
             "quiet time", "off-peak", "morning rush", "evening rush",
             "after work", "commuter", "weekend", "saturday", "sunday",
             "leisure", "pattern", "profile"),
    # "weekend"/"wochenende" stehen bewusst nur in der weekend-Kategorie
    # (Sa/So-Profil), sonst kaeme jede Wochenendfrage doppelt.
    "temporal": ("weekday", "monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday", "weekly", "daily pattern",
                 "which day", "busiest day", "quietest day", "day of week",
                 "weekly pattern", "wochentag", "montag", "dienstag",
                 "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
                 "morning peak", "evening peak", "rush hour", "last week",
                 "this week", "past week", "peak hour", "off-peak",
                 "time of day", "hourly", "by hour", "morgenspitze",
                 "abendspitze", "stoßzeit", "stosszeit", "wöchentlich",
                 "letzte woche", "letzten woche", "vergangene woche",
                 "diese woche", "tagesgang", "wochenprofil", "stoßstunde",
                 "verkehrsschwach"),
    # Investitionsfragen (Bonusfrage 1): Resilienz, Energie, Stoerungshistorie
    # und Nachfrage zusammen – siehe route().
    # Gesamtnetz aus dem VBB-Fahrplan (GTFS). Bewusst NICHT enthalten:
    # "u-bahn"/"ubahn" (steckt in fast jeder Frage), "connect" (trifft
    # "connected" der Resilienzfragen), "go to"/"reach" (zu allgemein).
    "gtfs": ("what lines", "which lines", "what trains", "which train",
             "all stops", "stops on", "stops of", "haltestellen", "linien",
             "linie", "s-bahn", "sbahn", "tram", "bus route", "bus line",
             "serve", "served by", "direct line", "direct connection",
             "connects", "verbindung", "fährt", "fahren nach", "timetable",
             "fahrplan", "ringbahn", "regional train", "u-bahn lines",
             "ubahn lines", "u-bahn-linien"),
    "invest": ("invest", "investment", "infrastructure", "improvement",
               # "capital" allein trifft "What is the capital of France?"
               "upgrade", "build", "construct", "capital invest", "capex",
               "priority",
               "should we build", "best use of", "where to spend",
               "modernize", "expand", "extend", "new line", "new station",
               "cbtc", "signalling", "automation"),
}

_EN_PATTERNS: dict[str, re.Pattern[str]] = {
    category: re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in words) + r")",
        flags=re.IGNORECASE,
    )
    for category, words in EN_KEYWORDS.items()
}

# Eindeutig themenfremde Anfragen werden ohne LLM-Call abgelehnt. Alles
# andere, was keine Kategorie trifft, geht an den LLM-Fallback.
OFF_TOPIC = re.compile(
    r"\b(poem|poetry|gedicht|song|lyrics|recipe|rezept|kochen|cook|joke|witz)",
    flags=re.IGNORECASE,
)

# Datensatz-Herkunft je Tool, fuer das data_basis-Feld.
# Datensatz-Herkunft je Tool, fuer das data_basis-Feld. Kategorien statt
# Dateinamen: _basis_files() loest sie auf die tatsaechlich gefundenen Dateien
# auf – neue Dateien (z. B. *_rest.csv fuer 22.09.–01.10.) erscheinen so
# automatisch in der Quellenangabe.
DATA_BASIS: dict[str, tuple[str, ...]] = {
    "get_weekly_summary": ("flows",),
    "get_temporal_context": (),
    "get_peak_hours": (),
    "find_stations_above_threshold": ("flows", "stations"),
    "find_station_dependencies": ("flows", "stations", "connections"),
    "analyze_disruption_routing": ("closures", "flows", "connections", "graph:adjacency.json"),
    "find_diverse_transit_routes": ("graph:adjacency.json",),
    "get_rain_impact": ("flows", "weather"),
    "get_lines_for_station": ("gtfs:gtfs_stop_routes.csv",),
    "get_stops_for_line": ("gtfs:gtfs_route_stops.csv",),
    "get_all_ubahn_lines": ("gtfs:gtfs_route_stops.csv",),
    "find_route_between_stations": ("gtfs:gtfs_stop_routes.csv",),
    "get_closures": ("closures",),
    "get_closure_impact": ("closures", "flows", "connections"),
    "get_events": ("events", "stations"),
    "get_weather": ("weather",),
    "find_weather_anomalies": ("weather",),
    "get_energy": ("energy", "flows", "stations"),
    "get_critical_stations": ("connections", "stations", "flows"),
    "find_alternative_routes": ("connections", "stations"),
    "find_transit_route": ("graph:adjacency.json",),
    "get_station_flow": ("flows", "stations", "closures"),
    "get_station_peak_vs_baseline": ("flows", "stations", "weather", "events", "closures"),
    "get_station_peak_15min": ("flows", "stations"),
    "typical_load_status": ("flows", "stations"),
    "hotspot_alternatives": ("hotspots",),
    "get_network_flow_summary": ("flows",),
    "detect_anomalies": ("flows",),
    "get_peak_profile": ("flows", "stations"),
    "compare_station_to_network": ("flows", "stations"),
    "network_info": ("stations", "lines"),
    "get_weekday_profile": ("flows", "stations"),
    "get_busiest_station_by_weekday": ("flows",),
    "get_station_weekday_pattern": ("flows", "stations"),
}


def _basis_files(keys: set[str]) -> list[str]:
    """Dateinamen zu Datensatz-Kategorien, wie sie der Loader gerade findet."""
    loader = DataLoader()
    files: set[str] = set()
    for key in keys:
        if key.startswith("graph:"):
            files.add(f"Berlin transit graph (src/graph_db/{key[6:]})")
            continue
        if key.startswith("gtfs:"):
            files.add(f"VBB GTFS (data/derived/{key[5:]})")
            continue
        if key == "hotspots":
            files.add("Hotspot plan (data/hotspots.json)")
            continue
        try:
            if key in STATIC_FILES:
                files.add(STATIC_FILES[key])
            else:
                files.update(p.name for p in loader.series_files(key))
        except (FileNotFoundError, KeyError):
            files.add(key)
    return sorted(files)


# ---------------------------------------------------------------------- #
# Hotspots (data/hotspots.json)
# ---------------------------------------------------------------------- #

HOTSPOTS_PATH = Path(__file__).resolve().parent.parent / "data" / "hotspots.json"


def _load_hotspots() -> list[dict[str, Any]]:
    """Liest die kuratierten Hotspots; eine fehlende Datei deaktiviert sie."""
    try:
        with open(HOTSPOTS_PATH, encoding="utf-8") as fh:
            return list(json.load(fh).get("hotspots") or [])
    except (OSError, ValueError) as exc:
        print(f"[hotspots] not loaded: {type(exc).__name__}: {exc}", flush=True)
        return []


# Einmal beim Import geladen – statische Planungsdaten, kein Anfragezustand.
HOTSPOTS_DATA: list[dict[str, Any]] = _load_hotspots()

HOTSPOT_FUZZY_CUTOFF = 0.7
# Kurze Schluesselwoerter ("icc", "alba", "csd") nur exakt – unscharf
# getroffen waeren sie Zufall.
HOTSPOT_FUZZY_MIN_LEN = 6

# Nur Fragen nach Entlastung/Ausweichen starten den Hotspot-Dialog. Eine
# reine Datenfrage ("Fahrgaeste am Hauptbahnhof am 12. Juli") bleibt im
# normalen Routing.
HOTSPOT_INTENT_RE = re.compile(
    r"alternativ|ausweich|überfüll|ueberfuell|überlast|ueberlast|umleit|entlast|"
    r"gesperrt|sperrung|andrang|\bvoll\b|crowd|rerout|detour|avoid|congest",
    flags=re.IGNORECASE,
)
TRAVEL_PHASE_RE = re.compile(
    r"\b(anreise|anfahrt|hinfahrt|ankunft|arrival|arriving|inbound|"
    r"abreise|abfahrt|rückfahrt|rueckfahrt|heimweg|departure|departing|leaving|outbound)\b",
    flags=re.IGNORECASE,
)
DIRECTION_RE = re.compile(
    r"\b(?:richtung|direction|towards?|heading)\s+([\w.\-äöüß+ ]{3,40}?)(?=[,.;!?]|$|\s+(?:und|and|bitte|please)\b)",
    flags=re.IGNORECASE,
)
ARRIVAL_WORDS = ("anreise", "anfahrt", "hinfahrt", "ankunft", "arrival", "arriving", "inbound")

# Offene Rueckfrage: der Server reicht keinen Verlauf durch, deshalb merkt
# sich der Agent den zuletzt erkannten Hotspot. Gilt prozessweit (ein
# Leitstand-Operator) und verfaellt nach PENDING_HOTSPOT_TTL_S.
PENDING_HOTSPOT_TTL_S = 600
_PENDING_HOTSPOT: dict[str, Any] = {}
_PENDING_LOCK = threading.Lock()


# Sperrungs-/Stoerungsfragen gehen vor dem Hotspot-Schritt an die
# Closure-Tools ("U6 gesperrt, Ausweichrouten?" traf sonst den Hotspot
# "U6 Nord" und endete in der Rueckfrage statt bei den Sperrdaten).
# Bewusst nicht "close"/"closest" ("closest station").
CLOSURE_RE = re.compile(
    r"\b(suspen\w*|clos(?:ed|ures?)|sperr\w*|gesperrt|ausf(?:a|ä)ll\w*|"
    r"\w*ersatzverkehr|sev|not in service|out of service|not running|"
    r"unterbroch\w*|betriebsunterbrechung)\b",
    flags=re.IGNORECASE,
)


def _is_closure_question(question: str) -> bool:
    return bool(CLOSURE_RE.search(question or ""))


def _is_hotspot_button(question: str) -> bool:
    """True fuer den exakten auto_query-Text eines Hotspot-Buttons.

    Einige Buttons fragen selbst nach Sperrungen ("full closure of the
    Ringbahn"); sie sollen trotzdem den Hotspot-Dialog oeffnen.
    """
    text = (question or "").strip().lower()
    return any(text == str(h.get("auto_query", "")).strip().lower() for h in HOTSPOTS_DATA)


def _set_pending_hotspot(hotspot: dict[str, Any]) -> None:
    with _PENDING_LOCK:
        _PENDING_HOTSPOT.clear()
        _PENDING_HOTSPOT.update(hotspot=hotspot, at=time.monotonic())


def _take_pending_hotspot() -> dict[str, Any] | None:
    with _PENDING_LOCK:
        hotspot = _PENDING_HOTSPOT.get("hotspot")
        fresh = hotspot and time.monotonic() - _PENDING_HOTSPOT["at"] <= PENDING_HOTSPOT_TTL_S
        _PENDING_HOTSPOT.clear()
        return hotspot if fresh else None


def _clear_pending_hotspot() -> None:
    with _PENDING_LOCK:
        _PENDING_HOTSPOT.clear()


def _detect_hotspot(question: str) -> dict[str, Any] | None:
    """Findet den Hotspot, dessen Schluesselwort die Frage nennt.

    Exakte Treffer (Wortgrenzen, case-insensitive) schlagen unscharfe. Unter
    exakten gewinnt der zuerst genannte – das Subjekt der Frage ("Staatsbesuch
    ... Brandenburger Tor gesperrt" ist ein Staatsbesuch) –, bei gleicher
    Position das laengere ("demo alexanderplatz" vor "demo alex").
    Unscharf (Tippfehler): difflib-Ratio >= 0.7 gegen gleich lange
    Wortfolgen, nur bei gleichem Wortanfang und aehnlicher Laenge – sonst
    traefe "kanzleramt" auf "konzert" und "reichstag" auf "richtung".
    """
    text = (question or "").lower()
    if not text or not HOTSPOTS_DATA:
        return None
    words = re.findall(r"[\wäöüß'.]+", text)
    best, best_score = None, 0.0
    for hotspot in HOTSPOTS_DATA:
        for keyword in hotspot.get("keywords") or []:
            kw = keyword.lower().strip()
            if not kw:
                continue
            exact = re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text)
            if exact:
                score = 2.0 - exact.start() / (len(text) + 1) + len(kw) / 10000
            elif len(kw) >= HOTSPOT_FUZZY_MIN_LEN:
                n = len(kw.split())
                grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
                score = max(
                    (difflib.SequenceMatcher(None, kw, g).ratio()
                     for g in grams
                     if g[:2] == kw[:2] and abs(len(g) - len(kw)) <= max(2, len(kw) // 5)),
                    default=0.0,
                )
                if score < HOTSPOT_FUZZY_CUTOFF:
                    continue
            else:
                continue
            if score > best_score:
                best, best_score = hotspot, score
    return best


def _travel_phase(question: str) -> str | None:
    match = TRAVEL_PHASE_RE.search(question or "")
    if not match:
        return None
    return "arrival" if match.group(1).lower() in ARRIVAL_WORDS else "departure"


def _direction(question: str) -> str | None:
    match = DIRECTION_RE.search(question or "")
    return match.group(1).strip() if match else None


def hotspot_alternatives(hotspot_id: str, travel_phase: str | None = None,
                         direction: str | None = None) -> dict[str, Any]:
    """Kuratierte Ausweichstationen eines Hotspots, passende Richtung zuerst."""
    hotspot = next((h for h in HOTSPOTS_DATA if h.get("id") == hotspot_id), None)
    if hotspot is None:
        return {"error": "Hotspot not found", "input": hotspot_id}
    alternatives = list(hotspot.get("alternatives") or [])
    if direction:
        needle = direction.lower()
        alternatives.sort(key=lambda a: -difflib.SequenceMatcher(
            None, needle, str(a.get("direction_focus", "")).lower()).ratio()
            - (1.0 if needle in str(a.get("direction_focus", "")).lower() else 0.0))
    return {
        "hotspot": hotspot.get("label"),
        "category": hotspot.get("category"),
        "primary_problem": hotspot.get("primary_problem"),
        "travel_phase": travel_phase,
        "direction": direction,
        "alternatives": alternatives,
        "data_limitation": (
            "Alternative stations and walking times are curated planning "
            "values (data/hotspots.json), not measured data."
        ),
    }


WEEKDAYS = {
    "monday": "Monday", "montag": "Monday",
    "tuesday": "Tuesday", "dienstag": "Tuesday",
    "wednesday": "Wednesday", "mittwoch": "Wednesday",
    "thursday": "Thursday", "donnerstag": "Thursday",
    "friday": "Friday", "freitag": "Friday",
    "saturday": "Saturday", "samstag": "Saturday",
    "sunday": "Sunday", "sonntag": "Sunday",
}

MONTHS = {
    # deutsch
    "januar": 1, "jan": 1, "februar": 2, "feb": 2, "märz": 3, "maerz": 3,
    "mrz": 3, "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6,
    "juli": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
    "sept": 9, "oktober": 10, "okt": 10, "november": 11, "nov": 11,
    "dezember": 12, "dez": 12,
    # englisch – die Trainingsfragen sind teilweise englisch formuliert
    "january": 1, "february": 2, "march": 3, "may": 5, "june": 6,
    "july": 7, "october": 10, "december": 12,
}

# Formulierungen, die eine BEOBACHTUNG von Umleitungsverhalten unterstellen.
# Ohne OD-Matrix ist das nicht belegbar -> Konfidenz "low".
# Reine EMPFEHLUNGS-Fragen ("wie sollte umgeleitet werden?") stehen bewusst
# nicht mehr hier: sie sind mit Topologie plus gekennzeichnetem
# Allgemeinwissen beantwortbar und landen ueber "multimodal" bei "medium".
LIMITATION_TRIGGERS = ("umgeleitet werden", "wohin", "od-matrix", "u4",
                       "actually prefer", "tatsächlich bevorzug",
                       "tatsaechlich bevorzug", "welche route nehmen",
                       "which route do passengers take")


class TrainAgent:
    """Drei-Stufen-Loop: ROUTE, GATHER, ANSWER.

    Der Agent hält keinen Zustand über Aufrufe hinweg. Instanzen sind billig
    und können pro Request neu erzeugt werden.
    """

    def __init__(self, llm: Callable[[str], str] | None = None):
        """llm ist injizierbar, damit der Loop ohne Endpoint testbar bleibt."""
        self._llm = llm or (lambda prompt: call_llm(prompt))

    # ================================================================== #
    # Extraktion
    # ================================================================== #

    @staticmethod
    def _extract_date(question: str) -> str | None:
        """Findet das erste Datum und normalisiert es auf YYYY-MM-DD."""
        # "InnoTrans day 1" / "first day of InnoTrans" -> Messetag 1-4.
        innotrans = re.search(
            r"innotrans\W+(?:day|tag)\s*(\d)|(?:day|tag)\s*(\d)\W+(?:of\s+)?innotrans|"
            r"(first|opening|erste[nr]?)\s+(?:day|tag)\s+(?:of\s+|der\s+)?innotrans",
            question, flags=re.IGNORECASE,
        )
        if innotrans:
            day = int(innotrans.group(1) or innotrans.group(2) or 1)
            if 1 <= day <= len(INNOTRANS_DAYS):
                return INNOTRANS_DAYS[day - 1]

        iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", question)
        if iso:
            return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

        dotted = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", question)
        if dotted:
            day, month, year = (int(g) for g in dotted.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"

        # "23. Juni 2026" / "24. Juni"
        named = re.search(
            r"\b(\d{1,2})\.\s*([A-Za-zÄÖÜäöü]+)\.?(?:\s+(\d{4}))?",
            question,
        )
        if named:
            month = MONTHS.get(named.group(2).lower())
            if month:
                day = int(named.group(1))
                year = int(named.group(3)) if named.group(3) else 2026
                return f"{year:04d}-{month:02d}-{day:02d}"

        # Englische Schreibweise "June 23rd" / "on July 13, 2026"
        english = re.search(
            r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(\d{4}))?",
            question,
        )
        if english:
            month = MONTHS.get(english.group(1).lower())
            if month:
                year = int(english.group(3)) if english.group(3) else 2026
                return f"{year:04d}-{month:02d}-{int(english.group(2)):02d}"

        # "23 Juni" ohne Punkt
        loose = re.search(r"\b(\d{1,2})\s+([A-Za-zÄÖÜäöü]+)\b", question)
        if loose:
            month = MONTHS.get(loose.group(2).lower())
            if month:
                return f"2026-{month:02d}-{int(loose.group(1)):02d}"
        return None

    @staticmethod
    def _extract_week_start(question: str, date_str: str | None) -> str | None:
        """Leitet den Wochenstart ab, wenn nach einer Woche gefragt wird."""
        if not re.search(r"\bwoche\b|\bweek\b", question, flags=re.IGNORECASE):
            return None
        # "Woche 20.-26. Juli" / "week of July 20-26": der ERSTE Tag der
        # Spanne ist der Wochenstart, nicht der zuletzt genannte.
        span = re.search(
            r"\b(\d{1,2})\.?\s*(?:-|–|—|bis|to)\s*(\d{1,2})\.?\s*([A-Za-zÄÖÜäöü]+)?",
            question,
        )
        if span:
            month = None
            if span.group(3):
                month = MONTHS.get(span.group(3).lower())
            if month is None:
                lead = re.search(r"\b([A-Za-zÄÖÜäöü]+)\s+\d{1,2}\s*(?:-|–|—|to|bis)",
                                 question)
                if lead:
                    month = MONTHS.get(lead.group(1).lower())
            if month:
                return f"2026-{month:02d}-{int(span.group(1)):02d}"
        return date_str

    @staticmethod
    def _forecast_reference(
        text: str, date_str: str | None, dates: dict[str, str]
    ) -> str:
        """Wählt den historischen Vergleichstag für eine Prognosefrage.

        Nennt die Frage ein Datum innerhalb des Datensatzes, gilt dieses.
        Sonst der letzte gleichartige Tag: Samstag, Sonntag oder Werktag.
        """
        if (date_str is not None
                and dates["first_calendar_day"] <= date_str <= dates["last_day"]):
            return date_str
        if any(w in text for w in ("samstag", "saturday")):
            return dates["last_saturday"]
        if any(w in text for w in ("sonntag", "sunday")):
            return dates["last_sunday"]
        if any(w in text for w in ("wochenende", "weekend")):
            return dates["last_saturday"]
        return dates["last_weekday"]

    @staticmethod
    def _extract_weekday(question: str) -> str | None:
        """Findet einen genannten Wochentag (deutsch oder englisch)."""
        days = TrainAgent._extract_weekdays(question)
        return days[0] if days else None

    @staticmethod
    def _extract_weekdays(question: str) -> list[str]:
        """Alle genannten Wochentage in Nennungsreihenfolge ("Tuesday vs. Sunday")."""
        text = question.lower()
        found = sorted(
            (text.find(key), value) for key, value in WEEKDAYS.items() if key in text
        )
        return list(dict.fromkeys(value for _, value in found))

    @staticmethod
    def _extract_daypart(question: str) -> tuple[int, int] | None:
        """Explizites Stundenfenster oder Tageszeit-Wort; None ohne Angabe."""
        hours = TrainAgent._extract_hours(question)
        if hours != (0, 24):
            return hours
        text = question.lower()
        for word, window in DAYPART_HOURS.items():
            if re.search(rf"\b{word}", text):
                return window
        return None

    @staticmethod
    def _extract_clock(question: str) -> tuple[int, int] | None:
        """Uhrzeit aus der Frage: "17:45", "at 3am", "at 2:30am", "um 8 Uhr"."""
        match = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", question, re.IGNORECASE)
        if match:
            hour, minute, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
        else:
            match = re.search(r"\b(?:at|um|gegen)\s+(\d{1,2})\s*(am|pm|uhr)?\b",
                              question, re.IGNORECASE)
            if not match:
                return None
            hour, minute, suffix = int(match.group(1)), 0, match.group(2)
        suffix = (suffix or "").lower()
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        return (hour, minute) if 0 <= hour < 24 and 0 <= minute < 60 else None

    @staticmethod
    def _extract_threshold(text: str) -> tuple[float, str] | None:
        """Schwellwert und Einheit aus "more than 2000 passengers per hour".

        Nur mit Fahrgast- oder Zeiteinheit – "at least 3 stations" ist kein
        Fahrgast-Schwellwert.
        """
        match = THRESHOLD_RE.search(text)
        if not match:
            return None
        tail = text[match.end(1):match.end(1) + 40]
        if not re.search(r"passeng|pax|people|person|fahrg|hour|stunde|slot|min", tail):
            return None
        value = float(re.sub(r"[.,]", "", match.group(1)))
        unit = (match.group(2) or "hour").lower()
        per = "slot" if unit.startswith(("slot", "15", "quarter")) else "hour"
        return value, per

    @staticmethod
    def _extract_lines(question: str) -> list[str]:
        """Alle genannten Linienkürzel, z. B. "U6"."""
        found = re.findall(r"\bU\s?([1-9])\b", question, flags=re.IGNORECASE)
        return sorted({f"U{d}" for d in found})

    @staticmethod
    def _extract_stations(question: str) -> list[str]:
        """Stationsnamen aus der Frage, gegen den Datensatz abgeglichen.

        Vergleicht normalisiert (ohne Präfix, ohne "(Berlin)", Str./Strasse
        vereinheitlicht) und bevorzugt die längsten Treffer, damit
        "Hallesches Tor" nicht an "Tor" hängenbleibt.
        """
        try:
            stations = DataLoader().load_stations()
        except (FileNotFoundError, ValueError):
            return []

        norm_q = normalize_station(question)
        candidates: list[tuple[int, str, str]] = []
        for name in stations["station_name"].unique():
            for key in _station_aliases(name):
                if len(key) >= 5 and key in norm_q:
                    candidates.append((len(key), key, name))
                    break

        # Laengster Treffer gewinnt, damit "Hallesches Tor" nicht als "Tor"
        # und "Alexanderplatz" nicht als Teil eines laengeren Namens endet.
        candidates.sort(key=lambda c: c[0], reverse=True)
        picked: list[tuple[int, str]] = []
        consumed = ""
        for _, key, name in candidates:
            if key in consumed:
                continue
            picked.append((norm_q.find(key), name))
            consumed += key + "|"

        # Reihenfolge der Nennung wiederherstellen – find_alternative_routes
        # leitet from/to aus der Position in der Frage ab.
        picked.sort(key=lambda p: p[0])
        if picked:
            return [name for _, name in picked]

        # Rueckfallebene: der Operator benennt die Station anders als der
        # Datensatz ("S+U Bahnhof Spandau" statt "S+U Rathaus Spandau").
        # Nur seltene Tokens zaehlen, sonst wuerde "Rathaus" (mehrfach
        # vergeben) beliebige Stationen einsammeln.
        return _token_fallback(question, stations["station_name"].unique())

    @staticmethod
    def _extract_hours(question: str) -> tuple[int, int]:
        """Stundenfenster aus der Frage; Default ist der ganze Tag."""
        # Englisch: "between 00:00 and 12:00" – ohne "Uhr" am Ende.
        between = re.search(
            r"between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+and\s+"
            r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            question, re.IGNORECASE,
        )
        if between:
            start = int(between.group(1))
            end = int(between.group(4))
            end_pm = (between.group(6) or "").lower() == "pm"
            start_pm = (between.group(3) or "").lower() == "pm" or (
                end_pm and not between.group(3) and start < end
            )
            if start_pm and start < 12:
                start += 12
            if end_pm and end < 12:
                end += 12
            if 0 <= start < end <= 24:
                return start, end

        # "from 8 to 10" / "from 8am to 10am" / "von 8 bis 10"
        from_to = re.search(
            r"\b(?:from|von)\s+(\d{1,2})(?::\d{2})?\s*(am|pm)?\s+(?:to|until|till|bis)\s+"
            r"(\d{1,2})(?::\d{2})?\s*(am|pm)?\b",
            question, re.IGNORECASE,
        )
        if from_to:
            start, end = int(from_to.group(1)), int(from_to.group(3))
            if (from_to.group(2) or "").lower() == "pm" and start < 12:
                start += 12
            if (from_to.group(4) or "").lower() == "pm" and end < 12:
                end += 12
            if 0 <= start < end <= 24:
                return start, end

        span = re.search(r"\b(\d{1,2})(?::\d{2})?\s*(?:-|–|bis|und)\s*(\d{1,2})(?::\d{2})?\s*uhr",
                         question, flags=re.IGNORECASE)
        if span:
            start, end = int(span.group(1)), int(span.group(2))
            if 0 <= start < end <= 24:
                return start, end

        single = re.search(r"\b(?:um|ab|gegen)\s+(\d{1,2})(?::(\d{2}))?\s*uhr",
                           question, flags=re.IGNORECASE)
        if single:
            hour = int(single.group(1))
            if 0 <= hour < 24:
                return hour, min(hour + 2, 24)
        return 0, 24

    # ================================================================== #
    # 1. ROUTE
    # ================================================================== #

    @staticmethod
    def _extract_gtfs_lines(question: str) -> list[str]:
        """Linienkürzel aller Verkehrsmittel: S41, M10, X7, RE1, FEX, Bus 100."""
        codes = re.findall(
            r"\b(S\d{1,2}|U\d|M\d{1,2}|X\d{1,3}|N\d{1,2}|RE\d{1,2}|RB\d{1,2}|FEX)\b",
            question, flags=re.IGNORECASE,
        )
        codes += re.findall(
            r"\b(?:bus|line|linie|tram)\s+(\d{1,3})\b", question, flags=re.IGNORECASE
        )
        return list(dict.fromkeys(c.upper() for c in codes))

    def _route_gtfs(self, question: str, text: str, stations: list[str],
                    add: Callable) -> None:
        """Wählt das GTFS-Tool: Linienverlauf, Direktverbindung oder Linien je Station."""
        network = "all"
        for word, net in (("s-bahn", "sbahn"), ("sbahn", "sbahn"), ("ringbahn", "sbahn"),
                          ("tram", "tram"), ("straßenbahn", "tram"), ("bus", "bus")):
            if re.search(rf"\b{word}", text):
                network = net
                break

        # Stationen ausserhalb des U-Bahn-Datensatzes (Ostkreuz, Messe Nord).
        names = list(stations) or find_stations_in_text(question)
        codes = self._extract_gtfs_lines(question)
        asks_for_stops = re.search(r"\b(stops|haltestellen|halte|stations on|verlauf)", text)
        # U-Linien nur, wenn nach ihrem Verlauf gefragt wird – "U6 suspended"
        # braucht keine Halteliste.
        line_codes = [c for c in codes if not c.startswith("U") or asks_for_stops]

        if line_codes and (asks_for_stops or not names):
            for code in line_codes[:2]:
                add("get_stops_for_line", get_stops_for_line, line_name=code)
        elif len(names) >= 2:
            add("find_route_between_stations", find_route_between_stations,
                from_station=names[0], to_station=names[1], prefer_network=network)
        elif names:
            add("get_lines_for_station", get_lines_for_station,
                station_name=names[0], network=network)
        elif re.search(r"\bu-?bahn", text) and re.search(r"\b(lines|linien)", text):
            add("get_all_ubahn_lines", get_all_ubahn_lines)

    def route(self, question: str) -> dict[str, Any]:
        """Entscheidet regelbasiert, welche Tools aufgerufen werden.

        Vor dem normalen Routing: Nennt die Frage einen bekannten Hotspot
        (data/hotspots.json) und fragt nach Entlastung, aber ohne Reiserichtung,
        liefert der Plan keine Tools, sondern hotspot_clarify – answer() zeigt
        dann die Ausweichstationen und fragt nach An-/Abreise und Richtung.
        Die Antwort darauf (auch ohne erneuten Hotspot-Namen) vertieft mit
        Flow-, Event- und Netz-Tools.
        """
        if _is_closure_question(question) and not _is_hotspot_button(question):
            _clear_pending_hotspot()
            return self._route_closure(question)
        hotspot, clarify = self._resolve_hotspot(question)
        if hotspot and clarify:
            return {
                "calls": [], "matched": ["hotspot"], "assumptions": [],
                "uncertain_assumptions": 0, "out_of_scope": False,
                "is_forecast": False, "question": question,
                "dates": _data_dates(), "params": {},
                "hotspot": hotspot, "hotspot_clarify": True,
            }
        plan = self._route_core(question)
        if hotspot:
            plan = self._deepen_hotspot(plan, question, hotspot)
        return _add_typical_load(plan)

    def _route_closure(self, question: str) -> dict[str, Any]:
        """Sperrungsfrage: normales Routing plus garantierte Closure-Tools.

        get_closures immer; find_alternative_routes, sobald zwei Stationen
        genannt sind (bei "between A and B" ohne den gesperrten Abschnitt).
        get_closure_impact haengt gather() an, sobald 1-3 Sperrungen passen.
        """
        plan = self._route_core(question)
        text = question.lower()
        dates = plan.get("dates") or _data_dates()
        params = plan.get("params") or {}
        stations = params.get("stations") or self._extract_stations(question)
        lines = params.get("lines") or self._extract_lines(question)
        calls = list(plan.get("calls") or [])
        names = {c["name"] for c in calls}

        if "get_closures" not in names:
            calls.insert(0, {"name": "get_closures", "fn": get_closures, "kwargs": {
                "date_str": params.get("date"),
                "station_name": stations[0] if stations else None,
                "line": lines[0] if lines else None}})
        if "find_alternative_routes" not in names and len(stations) >= 2:
            segment = bool(re.search(r"\b(between|zwischen)\b", text))
            calls.append({"name": "find_alternative_routes", "fn": find_alternative_routes,
                          "kwargs": {"from_station": stations[0], "to_station": stations[1],
                                     "avoid_stations": stations[2:] or None,
                                     "avoid_direct": segment,
                                     "line": lines[0] if lines else None,
                                     "closed_line": lines[0] if lines and not segment else None}})

        assumptions = list(plan.get("assumptions") or [])
        return {
            **plan,
            "calls": calls,
            "matched": sorted(set(plan.get("matched") or ()) | {"closure"}),
            "assumptions": assumptions,
            "uncertain_assumptions": plan.get("uncertain_assumptions", len(assumptions)),
            "out_of_scope": False,
            "is_forecast": plan.get("is_forecast", False),
            "question": question,
            "dates": dates,
            "params": {**params, "stations": stations, "lines": lines},
            "intent": "closure_disruption",
        }

    @staticmethod
    def _resolve_hotspot(question: str) -> tuple[dict[str, Any] | None, bool]:
        """(Hotspot, Rueckfrage noetig?) fuer eine Frage.

        Rueckfrage nur, wenn weder An-/Abreise noch eine Richtung genannt ist.
        Eine Folgefrage ohne Hotspot-Namen, die An-/Abreise oder Richtung
        nennt, bezieht sich auf den zuletzt offenen Hotspot.
        """
        answered = bool(_travel_phase(question) or _direction(question))
        hotspot = _detect_hotspot(question)
        # Ohne Entlastungs-Absicht ist ein Hotspot-Name nur ein Ort (z. B. das
        # Fahrziel in "Abreise Richtung Alexanderplatz").
        if hotspot is not None and HOTSPOT_INTENT_RE.search(question):
            if answered:
                _clear_pending_hotspot()
                return hotspot, False
            _set_pending_hotspot(hotspot)
            return hotspot, True
        if answered:
            pending = _take_pending_hotspot()
            if pending is not None:
                return pending, False
        return None, False

    def _deepen_hotspot(self, plan: dict[str, Any], question: str,
                        hotspot: dict[str, Any]) -> dict[str, Any]:
        """Ergaenzt den Plan um Hotspot-Kontext, Flow, Events und Netz."""
        dates = plan.get("dates") or _data_dates()
        params = plan.get("params") or {}
        phase, direction = _travel_phase(question), _direction(question)
        explicit_date = params.get("effective_date") or params.get("date")
        day = explicit_date or dates["last_day"]

        # Ohne Uhrzeit in der Frage: typisches Fenster je Reisephase.
        hour_from, hour_to = params.get("hour_from", 0), params.get("hour_to", 24)
        if (hour_from, hour_to) == (0, 24):
            hour_from, hour_to = {"arrival": (14, 20), "departure": (18, 24)}.get(phase, (0, 24))

        calls = [{"name": "hotspot_alternatives", "fn": hotspot_alternatives,
                  "kwargs": {"hotspot_id": hotspot["id"], "travel_phase": phase,
                             "direction": direction}}]
        ranked = hotspot_alternatives(hotspot["id"], phase, direction).get("alternatives") or []
        known = set()
        for alt in ranked:
            if len(known) >= 3:
                break
            name = alt.get("target_station", "")
            key = normalize_station(name)
            if key in known or not self._extract_stations(name):
                continue  # S-Bahn/Tram: nicht im U-Bahn-Flow-Datensatz
            known.add(key)
            calls.append({"name": "get_station_peak_vs_baseline",
                          "fn": get_station_peak_vs_baseline,
                          "kwargs": {"station_name": name, "date": day,
                                     "hour_from": hour_from, "hour_to": hour_to}})

        existing = list(plan.get("calls") or [])
        if not any(c["name"] == "get_events" for c in existing):
            calls.append({"name": "get_events", "fn": get_events,
                          "kwargs": {"date_str": day}})

        # Ziel genannt ("Richtung Alexanderplatz"): Route von der besten
        # Ausweichstation dorthin.
        targets = [s for s in self._extract_stations(direction or "")
                   if normalize_station(s) not in known]
        if targets and ranked and not any(c["name"] == "find_transit_route" for c in existing):
            calls.append({"name": "find_transit_route", "fn": find_transit_route,
                          "kwargs": {"from_station": ranked[0]["target_station"],
                                     "to_station": targets[0]}})

        assumptions = list(plan.get("assumptions") or [])
        uncertain = plan.get("uncertain_assumptions", len(assumptions))
        phase_en = phase or "travel phase open"
        note = (f"Hotspot detected: {hotspot.get('label')} "
                f"({phase_en}, direction {direction or 'open'}). "
                f"Load of the alternative stations on {day}, "
                f"{hour_from:02d}:00–{hour_to:02d}:00")
        if explicit_date:
            assumptions.append(note + ".")
        else:
            assumptions.append(note + " (last full data day – no date given).")
            uncertain += 1

        return {
            **plan,
            "calls": calls + existing,
            "matched": sorted(set(plan.get("matched") or ()) | {"hotspot"}),
            "assumptions": assumptions,
            "uncertain_assumptions": uncertain,
            "out_of_scope": False,
            "question": question,
            "dates": dates,
            "params": {**params, "effective_date": day},
            "hotspot": hotspot,
            "hotspot_clarify": False,
        }

    def _route_core(self, question: str) -> dict[str, Any]:
        """Regelbasiertes Routing ohne Hotspot-Schritt (siehe route())."""
        text = question.lower()
        dates = _data_dates()
        last_day = dates["last_day"]
        matched = {
            category for category, words in KEYWORDS.items()
            if any(word in text for word in words)
        }
        matched |= {
            category for category, pattern in REGEX_KEYWORDS.items()
            if re.search(pattern, text)
        }
        matched |= {
            category for category, pattern in _EN_PATTERNS.items()
            if pattern.search(text)
        }

        date_str = self._extract_date(question)
        lines = self._extract_lines(question)
        hour_from, hour_to = self._extract_hours(question)
        assumptions: list[str] = []
        # Eindeutige Aufloesungen (genau eine passende Sperrung) werden
        # ausgewiesen, senken die Konfidenz aber nicht – anders als ein
        # unterstelltes Datum.
        resolved_notes: list[str] = []

        stations: list[str] = []
        if matched & {"flow", "closure", "route", "event", "anomaly", "critical",
                      "peak", "compare", "multimodal", "forecast", "weekend",
                      "recent", "temporal", "network_info", "gtfs",
                      "service_time"}:
            stations = self._extract_stations(question)

        # "recent"/"latest" meint den letzten vollstaendigen Tag des
        # Datensatzes. Das ist eine deterministische Aufloesung, kein Raten -
        # sie wird trotzdem ausgewiesen, damit der Operator sie sieht.
        # "last week"/"this week": Wochenaggregat statt Einzeltag.
        flow_asked = "flow" in matched  # vor den Discards unten
        week_match = WEEK_RE.search(text) if date_str is None else None
        week_mode = None
        if week_match:
            word = (week_match.group(1) or week_match.group(2) or "").lower()
            week_mode = "current" if word.startswith(("this", "current", "dies")) else "last"
            matched.add("temporal")
            matched.discard("recent")
            matched.discard("flow")

        # "yesterday" relativ zum letzten Datentag – Echtzeit gibt es nicht.
        if date_str is None and YESTERDAY_RE.search(text):
            date_str = (
                datetime.date.fromisoformat(last_day) - datetime.timedelta(days=1)
            ).isoformat()
            matched.discard("recent")
            assumptions.append(
                f"\"yesterday\" was resolved relative to the last full data day "
                f"({last_day}): {date_str}. There is no real-time data."
            )

        if "recent" in matched and date_str is None:
            date_str = last_day
            assumptions.append(
                f"\"recent\"/\"latest\" was resolved as the last full day "
                f"of the dataset ({last_day}). The dataset ends there; "
                f"no newer data exists."
            )

        # Fix C: Fragen nach dem Normalfall ("Wann ist der Peak an X?") haben
        # bewusst kein Datum. Ein unterstelltes Einzeldatum waere hier falsch –
        # stattdessen uebernimmt get_peak_profile die Mittelung ueber alle Tage.
        aggregate_peak = bool(matched & {"peak", "compare"}) and bool(stations)
        if aggregate_peak and date_str is None:
            matched.discard("flow")

        # "Wie viele Stationen gibt es?" trifft ueber "wie viele" auch die
        # flow-Kategorie. Ohne Station und ohne Datum waere ein
        # Tages-Flow-Report dort nur Rauschen. Nur bei echten Groessenfragen –
        # "Which stations ..." / "entire network" treffen network_info auch,
        # meinen aber sehr wohl Fahrgastzahlen ("top 5 busiest stations").
        size_question = re.search(
            r"how many (stations|lines)|wie viele (stationen|linien)|"
            r"network size|netzgr",
            text,
        )
        if (size_question and "network_info" in matched and not stations
                and date_str is None):
            matched.discard("flow")

        # Energie-, Kritikalitaets- und Investitionsfragen enthalten oft
        # "per passenger" / "passengers affected". Die Tools dafuer rechnen die
        # Fahrgastzahlen selbst ueber den ganzen Zeitraum – ein zusaetzlicher
        # Einzeltag-Flow waere nur Rauschen und eine unterstellte Annahme.
        if (matched & {"energy", "critical", "invest"} and not stations
                and date_str is None):
            matched.discard("flow")

        # Eventfragen: die Fluesse an den Stationen am Veranstaltungsort holt
        # gather() gezielt nach (_event_station_flows). Eine netzweite
        # Tageszusammenfassung waere nur Rauschen im Prompt.
        if "event" in matched and not stations:
            matched.discard("flow")

        # Wochentagsfragen werden ueber die Aggregation beantwortet, nicht
        # ueber einen unterstellten Einzeltag.
        if "temporal" in matched and date_str is None:
            matched.discard("flow")

        # Fragen mit eigenem Aggregat-Tool. Ohne diese Erkennung landeten sie
        # beim Tagesreport eines unterstellten Einzeltags – falsche Grundlage
        # fuer "per hour", "in the network" oder "how does rain affect".
        weekdays_named = self._extract_weekdays(question)
        station_word = re.search(r"\b(station|stations|stationen|bahnh|haltestell)", text)
        threshold = self._extract_threshold(text) if station_word else None
        busiest_overall = bool(
            BUSIEST_RE.search(text) and station_word and not weekdays_named
            and date_str is None and not stations and not week_mode
        )
        rain_effect = bool(
            "weather" in matched and date_str is None and not week_match
            and WEATHER_EFFECT_RE.search(text)
            and (flow_asked or re.search(r"passeng|ridership|fahrg|demand|nachfrage", text))
        )
        innotrans_range = "innotrans" in text and date_str is None
        weekend_aggregate = "weekend" in matched and not stations and date_str is None
        if threshold or busiest_overall:
            matched -= {"flow", "network_info", "gtfs", "event"}
            matched.add("ranking")
        if rain_effect:
            matched -= {"flow", "weather"}
            matched.add("rain_effect")
        if weekend_aggregate:
            matched.discard("flow")
        # Analysefragen der Challenge (Trainingsfragen 8, 9, Bonus 2): eigene
        # Auswertungen statt Tagesreport, Stoerungsliste oder Kritikalitaet.
        dependency_q = bool(DEPENDENCY_RE.search(text) and station_word)
        behaviour_q = bool(ROUTE_BEHAVIOUR_RE.search(text) and DISRUPTION_RE.search(text))
        unconventional_q = bool(UNCONVENTIONAL_RE.search(text))
        if dependency_q:
            matched -= {"flow", "network_info", "critical"}
            matched.add("analysis")
        if behaviour_q:
            matched -= {"flow", "closure", "route", "multimodal", "critical"}
            matched.add("analysis")
        if unconventional_q:
            matched -= {"flow", "critical", "anomaly", "route", "multimodal"}
            matched.add("analysis")
        # "U7 stations please" trifft sonst keine Kategorie.
        if (LINE_LIST_RE.search(text) or LINE_SIZE_RE.search(text)) \
                and self._extract_gtfs_lines(question):
            matched.add("line_stops")

        if not matched:
            return {"calls": [], "matched": [], "assumptions": [],
                    "params": {}, "out_of_scope": True,
                    "off_topic": bool(OFF_TOPIC.search(question))}

        effective_date = date_str

        # Nennt die Frage eine Sperrung ohne Datum, ist das Datum dieser
        # Sperrung gemeint – nicht der Default. Ohne diesen Schritt wuerde
        # get_station_flow den falschen Tag auswerten.
        if effective_date is None and "closure" in matched and (lines or stations):
            probe = get_closures(
                station_name=stations[0] if stations else None,
                line=lines[0] if lines else None,
            )
            found = probe.get("closures", []) if "error" not in probe else []
            if len(found) == 1:
                effective_date = found[0]["when"][:10]
                resolved_notes.append(
                    f"No date given – the question was matched to the only "
                    f"fitting closure on {effective_date} "
                    f"({found[0]['description']})."
                )
            elif len(found) > 1:
                # Eigener Name: "dates" enthaelt die Datumsgrenzen des
                # Datensatzes und geht in den Plan (DATA RANGE im Prompt).
                # Die Annahme nur ausweisen, wenn nach Fahrgaesten gefragt
                # war – "Are there closures affecting U6?" verliert sonst ohne
                # Grund Konfidenz (Annahme -> "low").
                if "flow" in matched:
                    closure_dates = sorted({c["when"][:10] for c in found})
                    assumptions.append(
                        f"No date given and {len(found)} matching closures "
                        f"found ({', '.join(closure_dates)}). Flow analysis was "
                        "skipped – please specify a date."
                    )
                matched.discard("flow")

        # Bei Prognosefragen gilt der historische Vergleichstag fuer ALLE
        # Tools. Sonst holte das Wetter-Tool den Default-Montag, waehrend der
        # Forecast-Zweig den letzten Samstag auswertet.
        if "forecast" in matched:
            reference = self._forecast_reference(text, date_str, dates)
            if effective_date is None or effective_date != reference:
                effective_date = reference

        if effective_date is None and (matched & {"flow", "weather", "anomaly"}):
            effective_date = last_day
            assumptions.append(
                f"No date detected in the question – the last full day of "
                f"the dataset was analysed ({last_day}). "
                "The answer does not automatically apply to other days."
            )

        calls: list[dict[str, Any]] = []

        def add(name: str, fn: Callable, **kwargs: Any) -> None:
            # Mehrere Kategorien koennen denselben Aufruf anfordern
            # (z. B. forecast und weekend beide den letzten Samstag).
            # Identische Aufrufe nur einmal ausfuehren.
            for existing in calls:
                if existing["name"] == name and existing["kwargs"] == kwargs:
                    return
            calls.append({"name": name, "fn": fn, "kwargs": kwargs})

        if "closure" in matched:
            # "today"/"now" ohne Datum: get_closures bildet das auf den
            # letzten Datentag ab und weist es aus (keine Echtzeitdaten).
            closure_date = date_str
            if closure_date is None and TODAY_RE.search(text):
                closure_date = "today"
            add("get_closures", get_closures,
                date_str=closure_date,
                station_name=stations[0] if stations else None,
                line=lines[0] if lines else None)

        if "event" in matched:
            venue = None
            for keyword in ("arena", "olympiastadion", "tempodrom", "velodrom",
                            "messe", "tempelhof"):
                if keyword in text:
                    venue = keyword
                    break
            if innotrans_range:
                # "What happened during InnoTrans?": alle Messetage plus der
                # Vergleich Eventtage vs. normale Tage an der nächsten Station.
                add("get_events", get_events, venue_keyword="innotrans",
                    date_from=INNOTRANS_DAYS[0], date_to=INNOTRANS_DAYS[-1])
            elif week_mode and date_str is None:
                # "events this week": Kalenderwoche relativ zum letzten Datentag.
                ref = datetime.date.fromisoformat(last_day)
                monday = ref - datetime.timedelta(days=ref.weekday())
                if week_mode == "last":
                    monday -= datetime.timedelta(days=7)
                add("get_events", get_events, venue_keyword=venue,
                    date_from=monday.isoformat(),
                    date_to=(monday + datetime.timedelta(days=6)).isoformat())
            else:
                add("get_events", get_events, date_str=date_str, venue_keyword=venue)

        if threshold:
            weekdays_for_threshold = weekdays_named[:1]
            add("find_stations_above_threshold", find_stations_above_threshold,
                threshold=threshold[0], per=threshold[1],
                weekday=weekdays_for_threshold[0] if weekdays_for_threshold else None)
        elif busiest_overall:
            daypart = self._extract_daypart(question)
            kwargs = {"weekday": None}
            if daypart:
                kwargs.update(hour_start=daypart[0], hour_end=daypart[1])
            add("get_busiest_station_by_weekday", get_busiest_station_by_weekday, **kwargs)

        if rain_effect:
            add("get_rain_impact", get_rain_impact)

        if dependency_q:
            add("find_station_dependencies", find_station_dependencies)
        if behaviour_q:
            add("analyze_disruption_routing", analyze_disruption_routing)
        if unconventional_q:
            endpoints = list(stations[:2]) if len(stations) >= 2 else find_stations_in_text(question)
            if len(endpoints) >= 2:
                add("find_diverse_transit_routes", find_diverse_transit_routes,
                    from_station=endpoints[0], to_station=endpoints[1])
            elif re.search(r"\b(messe|innotrans|icc)", text):
                # "around Messe Berlin towards the city center": Stadtzentrum
                # als die beiden grossen Knoten Alexanderplatz und Friedrichstr.
                resolved_notes.append(
                    "\"City centre\" was interpreted as S+U Alexanderplatz and "
                    "S+U Friedrichstr.; origin S Messe Nord/ICC (Messe Berlin)."
                )
                for target in CITY_CENTRE:
                    add("find_diverse_transit_routes", find_diverse_transit_routes,
                        from_station="S Messe Nord/ICC", to_station=target)

        # "Flow peak caused by bad weather in the week of July 20-26": erst
        # das Wetterextrem finden, dann die Fahrgaeste GENAU dann auswerten
        # (gather). Der Wochenbeginn als Stichtag traf den Regen nie.
        weather_followup = False
        if "weather" in matched:
            week_start = self._extract_week_start(question, effective_date)
            if week_start:
                metric = "prcp"
                if any(w in text for w in ("temperatur", "hitze", "warm", "kalt")):
                    metric = "temp"
                elif any(w in text for w in ("wind", "sturm")):
                    metric = "wspd"
                add("find_weather_anomalies", find_weather_anomalies,
                    week_start=week_start, metric=metric)
                weather_followup = flow_asked
                if weather_followup and metric == "prcp":
                    add("get_rain_impact", get_rain_impact)
            if effective_date and not weather_followup:
                add("get_weather", get_weather, date_str=effective_date,
                    hour_from=hour_from, hour_to=hour_to)

        if "energy" in matched:
            add("get_energy", get_energy, line=lines[0] if lines else None)

        if "critical" in matched:
            add("get_critical_stations", get_critical_stations)

        if "invest" in matched:
            # Eine Investitionsempfehlung braucht alle vier Blickwinkel der
            # Bonusfrage: Resilienz, Nachfrage, Energie, Stoerungshistorie.
            # Reihenfolge im Kontext regelt INVEST_CONTEXT_PRIORITY.
            add("get_critical_stations", get_critical_stations)
            # Gleiche kwargs wie im energy-Zweig, sonst laeuft get_energy doppelt.
            add("get_energy", get_energy, line=lines[0] if lines else None)
            add("get_closures", get_closures, date_str=None,
                station_name=None, line=None)
            add("get_network_flow_summary", get_network_flow_summary,
                date_str=last_day)

        # "U2 suspended between Pankow and Alexanderplatz": from/to sind die
        # Endpunkte des gesperrten Abschnitts. Ohne avoid_direct kaeme genau
        # die gesperrte Strecke als "Alternative" zurueck.
        segment_suspended = bool(
            SUSPENSION_RE.search(text) and re.search(r"\b(between|zwischen)\b", text)
        )
        # "if U2 is closed" ohne Abschnitt: die ganze Linie faellt aus.
        line_closed = bool(SUSPENSION_RE.search(text) and lines and not segment_suspended)
        route_kwargs = {
            "avoid_direct": segment_suspended,
            "line": lines[0] if lines else None,
            "closed_line": lines[0] if line_closed else None,
        }

        if "route" in matched and len(stations) >= 2:
            add("find_alternative_routes", find_alternative_routes,
                from_station=stations[0], to_station=stations[1],
                avoid_stations=stations[2:] or None, **route_kwargs)

        # Routenfrage, deren Stationen nicht (beide) im U-Bahn-Datensatz
        # liegen ("from Ostkreuz to Hertzallee"): Gesamtnetz inkl. Bus/Tram.
        if "route" in matched and len(stations) < 2:
            names = find_stations_in_text(question)
            if len(names) >= 2:
                add("find_transit_route", find_transit_route,
                    from_station=names[0], to_station=names[1])

        if "anomaly" in matched and effective_date:
            add("detect_anomalies", detect_anomalies, date_str=effective_date)
            if CAUSE_RE.search(text):
                add("get_events", get_events, date_str=effective_date)
                add("get_weather", get_weather, date_str=effective_date)

        if "multimodal" in matched:
            # Netztopologie zeigt, wo eine Sperrung wirklich weh tut – aber
            # nur ohne konkrete Sperrung/Strecke. Mit Sperrung liefert
            # get_closure_impact die betroffenen Nachbarstationen.
            if "closure" not in matched and len(stations) < 2:
                add("get_critical_stations", get_critical_stations)
            # S-Bahn-/Tram-/Bus-Alternativen an den genannten Stationen aus
            # dem VBB-Fahrplan statt aus Allgemeinwissen.
            for station in stations[:2]:
                add("get_lines_for_station", get_lines_for_station,
                    station_name=station, network="all")
            if len(stations) >= 2:
                add("find_alternative_routes", find_alternative_routes,
                    from_station=stations[0], to_station=stations[1],
                    avoid_stations=stations[2:] or None, **route_kwargs)
            elif effective_date and "flow" not in matched:
                add("get_network_flow_summary", get_network_flow_summary,
                    date_str=effective_date, hour_from=hour_from,
                    hour_to=hour_to)

        if "gtfs" in matched:
            self._route_gtfs(question, text, stations, add)

        if "forecast" in matched:
            reference = self._forecast_reference(text, date_str, dates)
            if reference != date_str:
                assumptions.append(
                    f"Forecast question: the dataset ends on {last_day}. "
                    f"The last comparable day in the history serves as "
                    f"the reference ({reference})."
                )
            add("get_weather", get_weather, date_str=reference,
                hour_from=hour_from, hour_to=hour_to)
            add("get_network_flow_summary", get_network_flow_summary,
                date_str=reference)
            add("detect_anomalies", detect_anomalies, date_str=reference)

        if "weekend" in matched:
            # Wochenendfrage: Sa/So-Profil statt Werktagsprofil.
            if stations:
                for station in stations[:2]:
                    add("get_peak_profile", get_peak_profile,
                        station_name=station, weekday_only=False,
                        weekend_only=True)
            elif not self._extract_weekdays(question) and "forecast" not in matched:
                # Nur bei "weekend" ohne konkreten Tag – "Tuesday vs. Sunday"
                # beantwortet der Tagesgang im temporal-Zweig. Mittelwerte je
                # Wochentag (inkl. Wochenend/Werktag-Verhältnis) statt eines
                # einzelnen, zufälligen Samstags. Prognosefragen holen den
                # Vergleichssamstag bereits im forecast-Zweig.
                add("get_weekday_profile", get_weekday_profile)

        if aggregate_peak and "weekend" not in matched:
            for station in stations[:2]:
                add("get_peak_profile", get_peak_profile,
                    station_name=station, weekday_only=True)

        if aggregate_peak and "compare" in matched:
            add("compare_station_to_network", compare_station_to_network,
                station_name=stations[0])

        # "Which lines serve Alexanderplatz?" trifft network_info ueber
        # "which lines" – gemeint sind aber die Linien der Station (GTFS).
        # network_info nur fuer Groessen-/Uebersichtsfragen. "Which stations
        # would become overloaded" trifft die Kategorie auch, braucht aber
        # keine Netzkennzahlen – das verlaengert nur den Prompt.
        other_data = matched - {"network_info", "flow", "recent"}
        if "network_info" in matched and (size_question or not other_data) \
                and not ("gtfs" in matched and stations):
            add("network_info", network_info)

        # "How many stations does U7 have?": Linienverlauf aus dem Fahrplan
        # (inkl. U4 und Nicht-U-Bahn-Linien) zusaetzlich zu den Netzkennzahlen.
        # "What stations are on U1?" – Halteliste, auch ohne "how many".
        wants_line_stops = LINE_SIZE_RE.search(text) or LINE_LIST_RE.search(text)
        if wants_line_stops:
            codes = self._extract_gtfs_lines(question)
            for code in codes[:2]:
                add("get_stops_for_line", get_stops_for_line, line_name=code)
            if codes:
                add("network_info", network_info)

        # "How many lines does Berlin have?": Datensatz (8 Linien) UND
        # Fahrplan (9 inkl. U4) – sonst fehlt der Hinweis auf die U4.
        if LINE_COUNT_RE.search(text):
            add("network_info", network_info)
            add("get_all_ubahn_lines", get_all_ubahn_lines)

        # Stoßzeit-Definition: betriebliche Regel neben den gemessenen Spitzen.
        if PEAK_DEFINITION_RE.search(text):
            add("get_peak_hours", get_peak_hours)

        if "service_time" in matched:
            clock = self._extract_clock(question)
            if clock:
                ref = effective_date or date_str or last_day
                add("get_temporal_context", get_temporal_context,
                    timestamp=f"{ref} {clock[0]:02d}:{clock[1]:02d}")
            else:
                add("get_peak_hours", get_peak_hours)

        if "temporal" in matched:
            weekdays = self._extract_weekdays(question)
            daypart = self._extract_daypart(question)
            asks_peak = re.search(
                r"\b(peak|rush|busiest time|hour|spitze|stoß|stoss|tagesgang)", text
            )
            if week_mode:
                # "Any events this week?" fragt nicht nach Fahrgastsummen.
                if flow_asked or "event" not in matched:
                    add("get_weekly_summary", get_weekly_summary,
                        current_week=week_mode == "current")
            elif "compare" in matched and len(stations) >= 2:
                for station in stations[:2]:
                    add("get_weekday_profile", get_weekday_profile,
                        station_name=station)
            elif stations:
                add("get_station_weekday_pattern", get_station_weekday_pattern,
                    station_name=stations[0])
                # "Monday morning" braucht den Tagesgang dieses Wochentags –
                # das Wochenprofil allein kennt keine Tageszeit.
                # "Monday morning vs. Friday evening": beide Tagesgänge.
                if weekdays and (asks_peak or daypart or len(weekdays) > 1):
                    # Ein Wochentag + Tageszeit ("Monday morning"): mittlere
                    # Summe im Fenster. Bei zwei Wochentagen gehoeren die
                    # Tageszeiten zu verschiedenen Tagen – dort nur Tagesgaenge.
                    window = {}
                    if daypart and len(weekdays) == 1:
                        window = {"hour_start": daypart[0], "hour_end": daypart[1]}
                    for day in weekdays[:2]:
                        add("get_weekday_profile", get_weekday_profile,
                            station_name=stations[0], weekday=day, **window)
            elif len(weekdays) >= 2:
                # "Compare Tuesday vs. Sunday": Tagesgang beider Wochentage.
                for day in weekdays[:2]:
                    add("get_weekday_profile", get_weekday_profile, weekday=day)
            elif weekdays and asks_peak and daypart is None:
                # "Peak hours on Friday": Tagesgang des Wochentags.
                add("get_weekday_profile", get_weekday_profile, weekday=weekdays[0])
            elif weekdays:
                # "Busiest stations on Monday mornings (7–9)": Rangliste,
                # bei Tageszeitangabe nur im Fenster.
                kwargs = {"weekday": weekdays[0]}
                if daypart:
                    kwargs.update(hour_start=daypart[0], hour_end=daypart[1])
                add("get_busiest_station_by_weekday",
                    get_busiest_station_by_weekday, **kwargs)
            else:
                add("get_weekday_profile", get_weekday_profile)

        if "recent" in matched and date_str:
            add("detect_anomalies", detect_anomalies, date_str=date_str)

        if "flow" in matched and effective_date and not weather_followup:
            if stations:
                for station in stations[:3]:
                    add("get_station_flow", get_station_flow,
                        station_name=station, date_str=effective_date,
                        hour_from=hour_from, hour_to=hour_to)
            else:
                add("get_network_flow_summary", get_network_flow_summary,
                    date_str=effective_date, hour_from=hour_from, hour_to=hour_to)

        return {
            "calls": calls,
            "matched": sorted(matched),
            "assumptions": assumptions + resolved_notes,
            "uncertain_assumptions": len(assumptions),
            "out_of_scope": False,
            "is_forecast": "forecast" in matched,
            "question": question,
            "dates": dates,
            "week_mode": week_mode,
            "weather_followup": weather_followup,
            "params": {
                "date": date_str,
                "effective_date": effective_date,
                "lines": lines,
                "stations": stations,
                "hour_from": hour_from,
                "hour_to": hour_to,
            },
        }

    # ================================================================== #
    # 2. GATHER
    # ================================================================== #

    @staticmethod
    def gather(plan: dict[str, Any]) -> dict[str, Any]:
        """Ruft die geplanten Tools auf. Ein Toolfehler bricht nichts ab."""
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for call in plan["calls"]:
            try:
                payload = call["fn"](**call["kwargs"])
            except Exception as exc:  # defensiv: Tool darf den Loop nie killen
                payload = {"error": "Tool raised", "detail": f"{type(exc).__name__}: {exc}"}
            entry = {"name": call["name"], "kwargs": call["kwargs"], "result": payload}
            if isinstance(payload, dict) and "error" in payload:
                failures.append(entry)
            else:
                results.append(entry)

        # Abhaengiger Schritt: Impact nur, wenn eine konkrete Sperrung feststeht.
        for entry in list(results):
            if entry["name"] != "get_closures":
                continue
            closures = entry["result"].get("closures", [])
            if 1 <= len(closures) <= 3:
                for closure in closures:
                    impact = get_closure_impact(closure["description"])
                    item = {"name": "get_closure_impact",
                            "kwargs": {"closure_description": closure["description"]},
                            "result": impact}
                    (failures if "error" in impact else results).append(item)
            break

        # Abhaengiger Schritt 3: Fahrgaeste zum Zeitpunkt des Wetterextrems.
        if plan.get("weather_followup"):
            for item in _weather_peak_flows(results):
                (failures if "error" in item["result"] else results).append(item)

        # Abhaengiger Schritt 2: Fluss an den naechsten Stationen eines Events
        # im Zeitfenster rund um das Event ("flow at the neighboring stations").
        for entry in list(results):
            if entry["name"] == "get_events":
                for item in _event_station_flows(entry["result"],
                                                 plan.get("question", "")):
                    (failures if "error" in item["result"] else results).append(item)
                break

        return {"results": results, "failures": failures}

    # ================================================================== #
    # 3. ANSWER
    # ================================================================== #

    @staticmethod
    def _format_context(
        gathered: dict[str, Any], assumptions: list[str], invest: bool = False,
        event: bool = False,
    ) -> str:
        """Baut den Tool-Kontext als Text, hart auf MAX_CONTEXT_CHARS begrenzt.

        Investitionsfragen brauchen vier Tools gleichzeitig (Resilienz,
        Energie, Stoerungshistorie, Nachfrage). Mit dem normalen Budget
        verdraengte die Stoerungsliste Energie und Nachfrage. Dort gilt
        deshalb ein groesseres Gesamtbudget, ein festes Budget je Tool und
        kompaktes JSON ohne Einrueckung.
        """
        budget = (INVEST_CONTEXT_CHARS if invest
                  else EVENT_CONTEXT_CHARS if event else MAX_CONTEXT_CHARS)
        priority = INVEST_CONTEXT_PRIORITY if invest else CONTEXT_PRIORITY
        blocks: list[str] = []
        if assumptions:
            blocks.append("=== AGENT ASSUMPTIONS ===\n" + "\n".join(
                f"- {a}" for a in assumptions))

        # Nach Aussagekraft sortieren: bei knappem Budget muss das
        # aggregierte Ergebnis ueberleben, nicht eine lange Slot-Liste.
        ranked = sorted(
            gathered["results"],
            key=lambda e: priority.get(e["name"], 50),
        )
        for entry in ranked:
            # default=str: ein Timestamp oder numpy-Wert in einem Toolergebnis
            # darf den Prompt-Aufbau nicht mit TypeError abbrechen.
            if invest:
                body = json.dumps(_compact(entry["result"]), ensure_ascii=False,
                                  separators=(",", ":"), default=str)
                if len(body) > INVEST_TOOL_CHARS:
                    body = body[:INVEST_TOOL_CHARS] + " [... truncated ...]"
            else:
                body = json.dumps(_compact(entry["result"]), indent=2,
                                  ensure_ascii=False, default=str)
                # Eventfragen: ein einzelnes Nebenergebnis (z. B. eine lange
                # Sperrungsanalyse) darf die Stationsfluesse nicht verdraengen.
                if event and len(body) > EVENT_TOOL_CHARS:
                    body = body[:EVENT_TOOL_CHARS] + " [... truncated ...]"
            blocks.append(f"=== {entry['name']} ===\n{body}")

        for entry in gathered["failures"]:
            body = json.dumps(entry["result"], indent=2, ensure_ascii=False, default=str)
            blocks.append(f"=== {entry['name']} (FAILED) ===\n{body}")

        out: list[str] = []
        used = 0
        for block in blocks:
            remaining = budget - used
            if remaining <= 0:
                out.append("\n[... further tool results cut off due to length "
                           "limit ...]")
                break
            if len(block) > remaining:
                block = block[:remaining] + "\n[... truncated ...]"
            out.append(block)
            used += len(block) + 2
        return "\n\n".join(out)

    @staticmethod
    def _confidence(
        question: str, gathered: dict[str, Any], plan: dict[str, Any]
    ) -> str:
        """Leitet die Konfidenz aus Toolerfolg und bekannten Datenlücken ab."""
        results, failures = gathered["results"], gathered["failures"]
        if not results:
            return "none"

        substantive = any(
            _has_payload(entry["result"]) for entry in results
        )
        if not substantive:
            return "none"

        text = question.lower()
        assumptions = plan.get("assumptions") or []
        matched = set(plan.get("matched") or ())

        # Behauptete Beobachtung von Umleitungsverhalten: nicht belegbar.
        if any(trigger in text for trigger in LIMITATION_TRIGGERS):
            level = "low"
        # Eine unterstellte Annahme (z. B. geratenes Datum) ist eine bekannte
        # Luecke, kein Teilerfolg – deshalb "low", nicht "medium". Eindeutige
        # Aufloesungen zaehlen nicht (uncertain_assumptions).
        elif plan.get("uncertain_assumptions", len(assumptions)):
            level = "low"
        # Multimodal und Prognose stuetzen sich teilweise auf Allgemeinwissen
        # bzw. auf Extrapolation – nie "high", auch wenn alle Tools liefern.
        elif matched & {"multimodal", "forecast"}:
            level = "medium"
        elif failures:
            level = "medium"
        else:
            level = "high"
        return _apply_confidence_caps(level, gathered)

    def _llm_fallback(self, question: str, started: float) -> dict[str, Any]:
        """Beantwortet eine Frage ohne Kategorie-Treffer direkt über das LLM.

        Keine Tools, keine Zahlen aus Daten – deshalb immer Konfidenz "low".
        Der Aufruf wird auf der Konsole protokolliert, damit fehlende
        Stichwörter nach dem Testlauf nachgetragen werden können.
        """
        print(f"[LLM fallback] no keyword category matched: {question!r}",
              flush=True)
        prompt = SYSTEM_PROMPT + "\n" + FALLBACK_PROMPT.format(question=question)
        try:
            text = self._llm(prompt)
        except Exception as exc:  # noqa: BLE001 – Loop darf nie nach aussen werfen
            detail = f"{type(exc).__name__}: {exc}"
            return _envelope(question, _msg("out_of_scope", question), [], "none",
                             [], started, error=detail, llm_fallback=True)
        return _envelope(question, text.strip(), [], "low", [], started,
                         llm_fallback=True)

    def answer(self, question: str) -> dict[str, Any]:
        """Beantwortet eine Operator-Frage. Wirft nie – Fehler stehen im Dict."""
        started = time.perf_counter()
        question = (question or "").strip()

        if not question:
            return _envelope(question, _msg("out_of_scope", question), [], "none",
                             [], started, error="Empty question")

        try:
            plan = self.route(question)
        except (FileNotFoundError, ValueError) as exc:
            # Ohne lesbare Daten gibt es keine Datumsgrenzen und keine Tools.
            detail = f"{type(exc).__name__}: {exc}"
            return _envelope(question, _msg("no_data", question), [], "none",
                             [], started, error=detail)
        except Exception as exc:  # noqa: BLE001 – answer() wirft nie
            # Ein Fehler im Routing (Regex, Probe-Aufruf von get_closures)
            # darf die Anfrage nicht als HTTP 500 enden lassen.
            detail = f"Routing failed: {type(exc).__name__}: {exc}"
            return _envelope(question, _msg("need_details", question), [], "none",
                             [], started, error=detail)

        if plan.get("hotspot_clarify"):
            return self._hotspot_clarify_answer(question, plan, started)
        result = self._answer_with_plan(question, plan, started)
        if plan.get("hotspot"):
            result["hotspot"] = plan["hotspot"].get("id")
        return result

    def _hotspot_clarify_answer(self, question: str, plan: dict[str, Any],
                                started: float) -> dict[str, Any]:
        """Stufe 1 des Hotspot-Dialogs: Alternativen zeigen, Richtung erfragen.

        Die Alternativen gehen zuerst als Kontext an das LLM. Faellt es aus,
        kommt dieselbe Liste deterministisch. Die Rueckfrage steht immer am
        Ende, damit die Folgeantwort des Operators zugeordnet werden kann.
        """
        hotspot = plan["hotspot"]
        payload = hotspot_alternatives(hotspot["id"])
        basis = _basis_files({"hotspots"})
        prompt = SYSTEM_PROMPT + "\n" + HOTSPOT_PROMPT.format(
            clarify=HOTSPOT_CLARIFY_QUESTION, question=question,
            hotspot=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        error = None
        try:
            text = self._call_llm_with_timeout(prompt)
        except Exception as exc:  # noqa: BLE001 – answer() wirft nie
            text, error = None, f"{type(exc).__name__}: {exc}"
        if not text or not text.strip():
            text = build_fallback_answer(
                {"results": [{"name": "hotspot_alternatives", "result": payload}],
                 "failures": []}, [], "LLM unavailable" if error else "LLM response timed out")
            error = error or f"LLM timeout after {LLM_TIMEOUT_SECONDS}s"
        text = text.strip()
        if HOTSPOT_CLARIFY_QUESTION not in text:
            text += "\n\n**" + HOTSPOT_CLARIFY_QUESTION + "**"
        result = _envelope(question, text, ["hotspot_alternatives"], "medium", basis,
                           started, error=error)
        result["hotspot"] = hotspot.get("id")
        result["awaiting_clarification"] = True
        return result

    def _answer_with_plan(self, question: str, plan: dict[str, Any],
                          started: float) -> dict[str, Any]:
        """GATHER + ANSWER für einen fertigen Plan."""
        if plan.get("out_of_scope"):
            # Eindeutig themenfremd (Gedicht, Rezept, ...): ohne LLM ablehnen.
            # Alles andere bekommt eine Antwort über den LLM-Fallback statt
            # einer harten Ablehnung.
            if plan.get("off_topic"):
                return _envelope(question, _msg("out_of_scope", question), [],
                                 "none", [], started)
            return self._llm_fallback(question, started)

        if not plan["calls"]:
            return _envelope(question, _msg("need_details", question), [],
                             "none", [], started)

        gathered = self.gather(plan)
        tools_used = [e["name"] for e in gathered["results"]] + \
                     [e["name"] for e in gathered["failures"]]
        tools_used = list(dict.fromkeys(tools_used))

        basis = _basis_files({
            source
            for entry in gathered["results"]
            for source in DATA_BASIS.get(entry["name"], ())
        })

        if not gathered["results"]:
            detail = "; ".join(
                str(e["result"].get("error")) for e in gathered["failures"]
            )
            return _envelope(question, _msg("no_data", question), tools_used,
                             "none", [], started, error=detail or "All tools failed")

        confidence = self._confidence(question, gathered, plan)
        context = self._format_context(
            gathered, plan["assumptions"],
            invest="invest" in (plan.get("matched") or ()),
            # Sperrungsfragen brauchen wie Eventfragen mehrere Ergebnisse
            # gleichzeitig: Sperrung, Impact und die Ausweichrouten (sonst
            # am Ende der Prioritaet und abgeschnitten).
            event=plan.get("intent") == "closure_disruption" or any(
                e["name"] == "get_station_peak_15min" and e["result"].get("phase")
                for e in gathered["results"]),
        )
        # SYSTEM_PROMPT bleibt vorn und unveraendert – nur so greift der
        # prompt_cache_key des Endpoints.
        prompt = SYSTEM_PROMPT
        if plan.get("is_forecast"):
            prompt += FORECAST_PROMPT
        dates = plan["dates"]
        prompt += "\n" + USER_PROMPT.format(
            question=question, tool_results=context,
            first_day=dates["first_calendar_day"], last_day=dates["last_day"],
        )

        try:
            text = self._call_llm_with_timeout(prompt)
        except Exception as exc:  # noqa: BLE001 – Loop darf nie nach aussen werfen
            # Der originale Fehlertext MUSS durchgereicht werden. Eine
            # generische Meldung wie "Endpoint nicht erreichbar" verschleiert
            # den Unterschied zwischen fehlendem API-Key, HTTP 401, Timeout
            # und Netzwerkausfall und macht die Diagnose unnoetig teuer.
            # Die Zahlen gehen trotzdem an den Operator – ohne LLM-Text.
            detail = f"{type(exc).__name__}: {exc}"
            return self._deterministic_answer(
                question, gathered, plan, tools_used, confidence, basis, started,
                reason="LLM unavailable", error=detail,
            )

        if text is None:
            print(f"[LLM timeout] >{LLM_TIMEOUT_SECONDS}s, deterministic answer: "
                  f"{question!r}", flush=True)
            return self._deterministic_answer(
                question, gathered, plan, tools_used, confidence, basis, started,
                reason="LLM response timed out",
                error=f"LLM timeout after {LLM_TIMEOUT_SECONDS}s",
            )

        return _envelope(question, text.strip(), tools_used, confidence, basis,
                         started)

    def _call_llm_with_timeout(self, prompt: str) -> str | None:
        """LLM-Aufruf mit hartem Zeitlimit; None bei Überschreitung.

        Andere Fehler (HTTP, Netz, Credentials) werden weitergereicht.
        """
        future = _LLM_POOL.submit(self._llm, prompt)
        try:
            return future.result(timeout=LLM_TIMEOUT_SECONDS)
        except FutureTimeout:
            future.cancel()
            return None

    @staticmethod
    def _deterministic_answer(
        question: str, gathered: dict[str, Any], plan: dict[str, Any],
        tools_used: list[str], confidence: str, basis: list[str],
        started: float, reason: str, error: str,
    ) -> dict[str, Any]:
        """Antwort direkt aus den Tool-Ergebnissen, wenn das LLM ausfällt.

        Konfidenz höchstens "medium": die Zahlen stimmen, aber niemand hat
        sie eingeordnet oder die Teilfragen einzeln beantwortet.
        """
        answer = build_fallback_answer(gathered, plan.get("assumptions") or [], reason)
        result = _envelope(question, answer, tools_used, _cap(confidence, "medium"),
                           basis, started, error=error)
        result["note"] = f"Direct data answer ({reason})"
        result["llm_timeout"] = reason == "LLM response timed out"
        return result


CONFIDENCE_LEVELS = ["none", "low", "medium", "high"]

# Ab dieser Luftlinie zur naechsten U-Bahn-Station ist die Zuordnung eines
# Events zu "seiner" Station fraglich (mehr als ein kurzer Fussweg).
MAX_STATION_DISTANCE_M = 1000

# Fehlermeldungen der Tools, die "zu dieser Frage gibt es keine Daten"
# bedeuten – im Gegensatz zu Ladefehlern oder ungueltigen Eingaben.
NO_DATA_ERRORS = ("No data", "No route found", "No flow data",
                  "No overlapping full days", "Station not found",
                  "Closure not found", "Insufficient baseline")


def _cap(level: str, maximum: str) -> str:
    """Begrenzt eine Konfidenz nach oben."""
    return min(level, maximum, key=CONFIDENCE_LEVELS.index)


def _downgrade(level: str) -> str:
    """Eine Stufe tiefer, aber nie unter "low" – Daten sind ja vorhanden."""
    idx = CONFIDENCE_LEVELS.index(level)
    return CONFIDENCE_LEVELS[idx - 1] if idx > 1 else level


def _station_distances(result: dict[str, Any]) -> list[float]:
    """Stationsdistanzen aus einem get_events-Ergebnis, auf die sich die Antwort stützt.

    Bei langen Eventlisten (ungefilterte Abfrage) zaehlt nicht jedes
    entfernte Einzelevent – nur Listen bis 5 Events, der angefragte Ort und
    gefilterte historische Muster (bis 3).
    """
    distances: list[float] = []
    events = result.get("events") or []
    if len(events) <= 5:
        distances += [e["distance_m"] for e in events if e.get("distance_m") is not None]
    location = result.get("venue_location") or {}
    if location.get("distance_m") is not None:
        distances.append(location["distance_m"])
    patterns = result.get("historical_patterns") or []
    if len(patterns) <= 3:
        distances += [p["distance_m"] for p in patterns if p.get("distance_m") is not None]
    return distances


def _apply_confidence_caps(level: str, gathered: dict[str, Any]) -> str:
    """Datenqualitäts-Regeln, die eine bereits bestimmte Konfidenz senken.

    - Station weiter als 1000 m vom Event -> eine Stufe tiefer
    - Datum hinter dem Datensatzende (is_future_event) -> höchstens "medium"
    - ein Tool meldet "keine Daten" -> höchstens "low"
    """
    results = [entry["result"] for entry in gathered["results"]]

    if any(
        d > MAX_STATION_DISTANCE_M
        for r in results for d in _station_distances(r)
    ):
        level = _downgrade(level)

    # Sperrungen "heute" stammen vom letzten Datentag, nicht aus Echtzeit.
    if any(r.get("as_of_last_data_day") for r in results):
        level = _cap(level, "medium")

    if any(r.get("is_future_event") for r in results):
        level = _cap(level, "medium")

    no_data = any(
        str(entry["result"].get("error", "")).startswith(NO_DATA_ERRORS)
        for entry in gathered["failures"]
    ) or any(
        str(r.get("error", "")).startswith(NO_DATA_ERRORS) for r in results
    )
    if no_data:
        level = _cap(level, "low")
    return level


GERMAN_REQUEST_RE = re.compile(
    r"\b(auf|in)\s+deutsch\b|\bantworte\w*\s+(bitte\s+)?deutsch",
    flags=re.IGNORECASE,
)


def _lang(question: str) -> str:
    """Sprache der Standardtexte: immer Englisch (LANGUAGE RULE).

    Ausnahme wie im SYSTEM_PROMPT: der Operator verlangt ausdruecklich
    Deutsch ("Antworte auf Deutsch").
    """
    return "de" if GERMAN_REQUEST_RE.search(question or "") else "en"


def _msg(key: str, question: str) -> str:
    """Standardtext in der Sprache der Frage."""
    return MESSAGES[key][_lang(question)]


# Kleinere Zahl = weiter vorn im Kontext. Aggregierte, direkt antwortende
# Ergebnisse stehen vor rohen Zeitreihen.
CONTEXT_PRIORITY: dict[str, int] = {
    # Operator-Reihenfolge: Hotspot, Peak, Events, Stoerungen, Wetter,
    # Fluss, Energie, Routen.
    "hotspot_alternatives": 1,
    "get_station_peak_vs_baseline": 2,
    "get_station_peak_15min": 3,
    "typical_load_status": 3,
    "get_events": 4,
    "get_closures": 5,
    "get_closure_impact": 5,
    "get_weather": 6,
    "find_weather_anomalies": 6,
    "get_rain_impact": 6,
    "get_station_flow": 7,
    "get_network_flow_summary": 7,
    "get_energy": 8,
    "find_transit_route": 9,
    "find_alternative_routes": 9,
    # Uebrige Tools dahinter, untereinander in der bisherigen Reihenfolge.
    "network_info": 15,
    "get_temporal_context": 18,
    "find_station_dependencies": 18,
    "analyze_disruption_routing": 18,
    "find_diverse_transit_routes": 19,
    "compare_station_to_network": 20,
    "find_stations_above_threshold": 20,
    "find_route_between_stations": 22,
    "get_stops_for_line": 22,
    "get_all_ubahn_lines": 22,
    "get_station_weekday_pattern": 25,
    "get_busiest_station_by_weekday": 25,
    "get_weekly_summary": 25,
    "get_weekday_profile": 25,
    "get_critical_stations": 25,
    "get_lines_for_station": 28,
    "get_peak_profile": 35,
    "get_peak_hours": 36,
    "detect_anomalies": 40,
}

# Maximale Zahl Events je Toolergebnis im Prompt.
MAX_EVENTS_IN_CONTEXT = 10

# Maximale Zahl roher 15-Minuten-Slots je Toolergebnis im Prompt.
MAX_SLOTS_IN_CONTEXT = 24
MAX_SLOTS_WITH_BASELINE = 8


# Stationen weiter als 1,5 km vom Veranstaltungsort sind keine "Nachbarn".
MAX_EVENT_STATION_M = 1500
MAX_EVENT_STATIONS = 3


WEATHER_TOP_STATIONS = 3


def _weather_peak_flows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Netz- und Stationsflüsse rund um den Spitzenwert einer Wetterwoche.

    Fenster: 1 h vor bis 2 h nach dem Wetter-Peak. Die drei stärksten
    Stationen im Fenster kommen mit Vergleich zu den gleichen Wochentagen.
    """
    anomaly = next((e["result"] for e in results if e["name"] == "find_weather_anomalies"), None)
    peak = (anomaly or {}).get("peak") or {}
    if not peak.get("timestamp"):
        return []
    date_str, hour = peak["timestamp"][:10], int(peak["timestamp"][11:13])
    hour_from, hour_to = max(0, hour - 1), min(24, hour + 2)
    items = []
    summary_kwargs = {"date_str": date_str, "hour_from": hour_from, "hour_to": hour_to}
    summary = get_network_flow_summary(**summary_kwargs)
    items.append({"name": "get_network_flow_summary", "kwargs": summary_kwargs, "result": summary})
    weather_kwargs = dict(summary_kwargs)
    items.append({"name": "get_weather", "kwargs": weather_kwargs, "result": get_weather(**weather_kwargs)})
    for station in (summary.get("top_5_stations") or [])[:WEATHER_TOP_STATIONS]:
        kwargs = {"station_name": station["station_name"], **summary_kwargs,
                  "compare_baseline": True}
        items.append({"name": "get_station_flow", "kwargs": kwargs,
                      "result": get_station_flow(**kwargs)})
        # Load status fuer die Tabelle "Top Stations by Volume".
        peak_kwargs = {"station_name": station["station_name"], "date": date_str,
                       "hour_from": hour_from, "hour_to": hour_to}
        items.append({"name": "get_station_peak_15min", "kwargs": peak_kwargs,
                      "result": get_station_peak_15min(**peak_kwargs)})
    return items


def _event_station_flows(result: dict[str, Any], question: str = "") -> list[dict[str, Any]]:
    """Fluss an den Stationen rund um die grössten Events eines Tages.

    Fenster: 2 h vor Beginn bis 3 h nach Ende (An- und Abreise), jeweils mit
    Vergleich zum selben Fenster an den anderen gleichen Wochentagen. Nur für
    Abfragen mit Datum und überschaubarer Eventliste. Nennt die Frage ein
    Event ("System Of A Down"), zählen nur dessen Einträge – sonst käme die
    Station einer Comedy-Show am selben Abend dazu. Bis zu drei Stationen im
    Umkreis von 1,5 km ("neighboring stations"), nicht nur die nächste.
    """
    date_str = (result.get("filters") or {}).get("date")
    events = result.get("events") or []
    if not date_str or not events or len(events) > 8:
        return []

    q_words = set(re.findall(r"[a-zäöüß0-9]{4,}", question.lower()))
    named = [
        e for e in events
        if q_words & set(re.findall(r"[a-zäöüß0-9]{4,}", str(e["event_name"]).lower()))
    ]
    events = named or events

    def candidates(event: dict) -> list[dict]:
        near = event.get("nearby_stations") or (
            [{"station_name": event.get("nearest_station"),
              "distance_m": event.get("distance_m")}]
            if event.get("nearest_station") else []
        )
        return [c for c in near
                if c.get("distance_m") is not None and c["distance_m"] <= MAX_EVENT_STATION_M]

    # Reihum: erst die naechste Station jedes Events (groesstes zuerst), dann
    # die zweitnaechste usw. Sonst belegt ein Event alle Plaetze und die
    # Station eines anderen Events am selben Tag fehlt in der Ursachenanalyse.
    ranked = sorted(events, key=lambda e: e.get("estimated_attendance") or 0, reverse=True)
    lists = [(event, candidates(event)) for event in ranked]
    picked: dict[str, tuple[dict, float]] = {}
    for depth in range(MAX_EVENT_STATIONS):
        for event, cands in lists:
            if depth < len(cands) and len(picked) < MAX_EVENT_STATIONS:
                cand = cands[depth]
                picked.setdefault(cand["station_name"], (event, cand["distance_m"]))

    items = []
    for station, (event, distance) in picked.items():
        start = int(event["began_local"][11:13])
        end = int(event["end_local"][11:13]) if event.get("end_local") else start + 3
        if end < start:  # Ende nach Mitternacht
            end = 24
        hour_from, hour_to = max(0, start - 2), min(24, end + 3)
        kwargs = {"station_name": station, "date_str": date_str,
                  "hour_from": hour_from, "hour_to": hour_to,
                  "compare_baseline": True}
        flow = get_station_flow(**kwargs)
        context = (
            f"{event['event_name']} ({event['began_local'][11:]}–"
            f"{(event.get('end_local') or '')[11:]}), station "
            f"{distance} m from the venue"
        )
        if isinstance(flow, dict) and "error" not in flow:
            flow["event_context"] = context
        items.append({"name": "get_station_flow", "kwargs": kwargs, "result": flow})
        items.extend(_event_phase_peaks(station, date_str, start, end, distance, context))
    return items


PROFILE_TOOLS = ("get_station_weekday_pattern", "get_peak_profile")


def _add_typical_load(plan: dict[str, Any]) -> dict[str, Any]:
    """Profilfragen bekommen einen Auslastungsstatus je Station."""
    calls = plan.get("calls") or []
    stations = [c["kwargs"].get("station_name") for c in calls
                if c["name"] in PROFILE_TOOLS and c["kwargs"].get("station_name")]
    done = {c["kwargs"].get("station_name") for c in calls if c["name"] == "typical_load_status"}
    extra = [{"name": "typical_load_status", "fn": typical_load_status,
              "kwargs": {"station_name": name}}
             for name in dict.fromkeys(stations) if name not in done]
    return {**plan, "calls": calls + extra} if extra else plan


def typical_load_status(station_name: str) -> dict[str, Any]:
    """Typischer Tagespeak (Mo–Fr-Mittel) gegen das Allzeit-Maximum der Station.

    get_station_peak_15min ohne Datum liefert das Allzeit-Maximum selbst –
    dessen Anteil am Maximum ist immer 100 %, jede Station waere CRITICAL.
    Aussagekraeftig fuer "wie voll ist es normalerweise" ist der mittlere
    Peak-Slot eines Werktags im Verhaeltnis zu diesem Maximum.
    """
    profile = get_peak_profile(station_name)
    peak = get_station_peak_15min(station_name)
    for result in (profile, peak):
        if "error" in result:
            return result
    candidates = [p for p in (profile.get("morning_peak"), profile.get("evening_peak"))
                  if p and p.get("mean_flow") is not None]
    if not candidates or not peak.get("station_max_ever"):
        return {"error": "No typical peak", "input": station_name}
    top = max(candidates, key=lambda p: p["mean_flow"])
    station_max = peak["station_max_ever"]
    pct = round(top["mean_flow"] / station_max * 100.0, 1)
    status = "CRITICAL" if pct >= 90.0 else "ELEVATED" if pct >= 70.0 else "NORMAL"
    return {
        "station_name": peak["station_name"],
        "lines": peak.get("lines"),
        "typical_peak_time": top["time"],
        "typical_peak_mean": round(float(top["mean_flow"]), 1),
        "days_included": profile.get("days_included"),
        "station_max_ever": station_max,
        "station_max_at": peak.get("peak_timestamp"),
        "pct_of_own_max": pct,
        "capacity_status": status,
        "method": ("Mean of the busiest 15-min slot on weekdays (Mon–Fri) vs. the "
                   "station's all-time 15-min maximum in the dataset."),
    }


def _event_phase_peaks(station: str, date_str: str, start: int, end: int,
                       distance: Any, context: str) -> list[dict[str, Any]]:
    """15-Min-Peak mit Auslastungsstatus je Phase: Anreise und Abreise.

    Anreise: 2 h vor Beginn bis Beginn. Abreise: Ende bis 2 h danach. Ein
    gemeinsamer Peak ueber beide Phasen koennte die Tabellen "Arrival Phase"
    und "Departure Phase" nicht getrennt fuellen. Endet ein Event nach
    Mitternacht, gibt es kein Abreisefenster am selben Tag.
    """
    items = []
    windows = (("arrival", max(0, start - 2), start),
               ("departure", end, min(24, end + 2)))
    for phase, hour_from, hour_to in windows:
        if hour_from >= hour_to:
            continue
        kwargs = {"station_name": station, "date": date_str,
                  "hour_from": hour_from, "hour_to": hour_to}
        peak = get_station_peak_15min(**kwargs)
        if isinstance(peak, dict) and "error" not in peak:
            peak.update(phase=phase, distance_m=distance, event_context=context)
        items.append({"name": "get_station_peak_15min", "kwargs": kwargs, "result": peak})
    return items


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    """Kürzt lange Slot-Listen, damit sie den Kontext nicht auffressen.

    Behalten wird ein Fenster um den Spitzenwert – das ist die Stelle, nach
    der Operatoren fragen. Die Kürzung wird im Ergebnis ausgewiesen, damit
    das LLM nicht auf unvollständigen Daten Vollständigkeit behauptet.
    """
    trimmed = result
    if "capacity_status" in result and "peak_value" in result:
        trimmed = {k: v for k, v in result.items() if k not in PEAK_CONTEXT_DROP}
    # Mit Vergleichswert (baseline) tragen total/peak_slot/baseline die
    # Aussage – drei Event-Stationen mit je 24 Slots verdraengten sonst die
    # dritte Station aus dem Kontextbudget.
    max_slots = MAX_SLOTS_WITH_BASELINE if result.get("baseline") else MAX_SLOTS_IN_CONTEXT
    for key, label in (("slots", "timestamp"), ("profile", "time")):
        series = trimmed.get(key)
        if not isinstance(series, list) or len(series) <= max_slots:
            continue

        values = [
            s.get("flow", s.get("mean_flow", s.get("prcp", 0)))
            if isinstance(s, dict) else 0
            for s in series
        ]
        peak = max(range(len(values)), key=lambda i: values[i])
        half = max_slots // 2
        start = max(0, min(peak - half, len(series) - max_slots))
        window = series[start:start + max_slots]

        trimmed = dict(trimmed)
        trimmed[key] = window
        trimmed[f"{key}_note"] = (
            f"Truncated: {len(window)} of {len(series)} 15-minute entries, "
            f"window around the peak ({series[peak].get(label)}). "
            "Aggregates (total, mean_per_slot, peak_slot, summary, "
            f"morning_peak, evening_peak) refer to ALL {len(series)} "
            "entries."
        )

    # Ungefilterte Eventabfragen liefern alle 417 Events. Ungekuerzt fuellen
    # sie das Kontextbudget und verdraengen die Ergebnisse anderer Tools.
    events = trimmed.get("events")
    if isinstance(events, list) and len(events) > MAX_EVENTS_IN_CONTEXT:
        largest = sorted(
            events, key=lambda e: e.get("estimated_attendance") or 0, reverse=True
        )[:MAX_EVENTS_IN_CONTEXT]
        trimmed = dict(trimmed)
        trimmed["events"] = largest
        trimmed["events_note"] = (
            f"Truncated: the {len(largest)} events with the highest (scaled) "
            f"attendance out of {len(events)} matches. count refers to all."
        )

    # Rohlisten ans Ende: wird der Block am Kontextbudget abgeschnitten,
    # fallen Einzelslots weg, nicht Kennzahlen wie total oder baseline.
    raw = [k for k in ("slots", "profile", "hourly_profile") if k in trimmed]
    if raw:
        trimmed = {k: v for k, v in trimmed.items() if k not in raw} | {k: trimmed[k] for k in raw}
    return trimmed


# Ein Token darf hoechstens so viele Stationen treffen, um als Hinweis zu
# gelten. "spandau" (2 Stationen) zaehlt, "rathaus" (mehrere) nicht.
MAX_TOKEN_AMBIGUITY = 3


def network_info() -> dict[str, Any]:
    """Netzkennzahlen aus den Stammdaten – ohne Zeitreihen.

    Beantwortet "Wie viele Stationen/Linien gibt es?" direkt aus
    stations_with_ubahn.csv und berlin_ubahn_lines_used.csv.
    """
    loader = DataLoader()
    try:
        stations = loader.load_stations()
        lines = loader.load_lines()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": "Data load failed", "detail": str(exc)}

    line_names = sorted(lines["line_name"].tolist())
    unique = stations.drop_duplicates("station_name")["u_bahn_lines"]
    per_line = {
        line: int(sum(line in [x.strip() for x in str(v).split(",")] for v in unique))
        for line in line_names
    }
    return {
        "total_stations": int(len(stations)),
        "stations_per_line": per_line,
        "stations_per_line_note": (
            "Unique station names per line in the hackathon dataset "
            "(stations_with_ubahn.csv). The full line route according to the "
            "timetable comes from the VBB GTFS stop list."
        ),
        "unique_station_names": int(stations["station_name"].nunique()),
        "lines": line_names,
        "total_lines": int(len(line_names)),
        "note": "U4 is not included in the dataset",
        "data_period": (
            f"{loader.data_dates()['first_calendar_day']} to "
            f"{loader.data_last_day}"
        ),
        "name_collision": (
            "168 station_id but only 167 unique names – "
            "\"U Stadtmitte (Berlin)\" exists twice (U2 and "
            "U6 platform levels)."
        ),
    }


def _token_fallback(question: str, station_names) -> list[str]:
    """Stationen über seltene Wortbestandteile finden.

    Greift nur, wenn kein vollständiger Namenstreffer vorliegt. Liefert
    mehrere Namen, wenn das Token nicht eindeutig ist – der Aufrufer nimmt
    den ersten, die übrigen bleiben als Alternativen sichtbar.
    """
    q_tokens = station_tokens(question)
    if not q_tokens:
        return []

    index: dict[str, list[str]] = {}
    for name in station_names:
        for token in station_tokens(name):
            index.setdefault(token, []).append(name)

    hits: dict[str, int] = {}
    for token in q_tokens:
        owners = index.get(token)
        if not owners or len(owners) > MAX_TOKEN_AMBIGUITY:
            continue
        for name in owners:
            hits[name] = hits.get(name, 0) + 1

    if not hits:
        return []
    best = max(hits.values())
    return sorted(name for name, score in hits.items() if score == best)


def _station_aliases(name: str) -> list[str]:
    """Vergleichsschlüssel für einen Stationsnamen, längster zuerst.

    Operatoren sagen "Alexanderplatz", der Datensatz heisst
    "S+U Alexanderplatz Bhf (Berlin)" – ohne Alias findet die Suche nichts.
    """
    base = normalize_station(name)  # entfernt bereits "Bhf"/"Bahnhof"
    keys = {base}
    for key in list(keys):
        if key.startswith("berlin") and len(key) > 9:
            keys.add(key[6:])
    return sorted(keys, key=len, reverse=True)


def _has_payload(result: dict[str, Any]) -> bool:
    """True, wenn ein Toolergebnis tatsächlich Daten enthält."""
    for key in ("count", "anomaly_count", "data_points", "total",
                "total_stations", "days_included"):
        if isinstance(result.get(key), int) and result[key] > 0:
            return True
    for key in ("closures", "events", "routes", "critical_stations",
                "top_critical_stations",
                "slots", "efficiency_ranking", "anomalies", "during_closure",
                "profile", "morning_peak", "evening_peak",
                "historical_patterns", "lines_by_type", "stops",
                "ubahn_lines", "from_lines", "daily_totals",
                "transit_alternative", "time_slot", "pairs",
                "cases_by_local_shift", "stations", "window_total",
                "alternatives", "peak_value"):
        if result.get(key):
            return True
    return bool(result.get("lines") or result.get("summary"))


def _envelope(
    question: str,
    answer: str,
    tools_used: list[str],
    confidence: str,
    data_basis: list[str],
    started: float,
    error: str | None = None,
    llm_fallback: bool = False,
) -> dict[str, Any]:
    """Einheitliches Rückgabeformat von answer()."""
    return {
        "question": question,
        "answer": answer,
        "tools_used": tools_used,
        "confidence": confidence,
        "data_basis": data_basis,
        "processing_time_s": round(time.perf_counter() - started, 3),
        "error": error,
        "llm_fallback": llm_fallback,
        "hotspot": None,
        "awaiting_clarification": False,
    }


__all__ = ["TrainAgent", "SYSTEM_PROMPT", "USER_PROMPT"]
