#!/usr/bin/env python3
"""
agent.py - conversational U-Bahn operations agent (InnoTrans 2026 hackathon).

The language model reads the operator's question, calls the engine's tools and writes the answer.
Numbers come from ubahn_engine.py / ubahn_analyses.py, so the same question gives the same figures.

Backends
    anthropic : Claude through the Anthropic API             (pip install anthropic; ANTHROPIC_API_KEY)
    openai    : any OpenAI-compatible endpoint with native tool calling, e.g. an open-source model served
                by vLLM or Ollama, or the hackathon LLM endpoint (OPENAI_BASE_URL, OPENAI_API_KEY, --model)
    json      : same endpoint for models WITHOUT native tool calling: tools are described in the prompt and
                the model answers with a JSON tool request or with the final answer
    azure     : Azure OpenAI Responses API, e.g. an endpoint ending in
                /openai/responses?api-version=2025-04-01-preview
                (AZURE_ENDPOINT = that URL as given, AZURE_API_KEY, MODEL_NAME = deployment name).
                Default backend; "azure_responses" is an alias.

Merged project (Hackathon_2026_JJM): src/server.py calls answer() at the end of this file, which runs Agent.ask
with a 26 s budget and falls back to src/fallback.py (answer built from the tool results) when it runs out.

Usage
    python agent.py --data /path/to/data "Which three stations are most at risk on InnoTrans day 1?"
    python agent.py --data /path/to/data                  # interactive session
    python agent.py --data /path/to/data --show-tools ... # print every tool call and its size
    python agent.py --data /path/to/data --selftest       # run the tools behind the training questions, no LLM
    python agent.py --data /path/to/data --batch questions.md --out answers.md   # answer a question file
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import networkx as nx
import numpy as np
import pandas as pd

import src.analyses as A
import src.ops as OPS
import src.reports as REP
import src.verify as VF
from src.engine import TOD_ORDER, UBahnEngine
from src.loader import DATA_ROOT, REPO_ROOT

try:                                    # same .env as src/server.py and src/tools/llm_client.py
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / '.env', override=False)
except ImportError:
    pass
from src.env_aliases import apply_env_aliases  # noqa: E402
apply_env_aliases()

# MODEL_NAME is the merged project's name; AZURE_MODEL is what the original final_InnoTrans .env used.
_MODEL = os.environ.get('MODEL_NAME') or os.environ.get('AZURE_MODEL') or ''
DEFAULT_BACKEND = os.environ.get('LLM_BACKEND', 'azure_responses')
BACKEND_ALIASES = {'azure_responses': 'azure'}
DEFAULT_MODELS = {'anthropic': _MODEL or 'claude-opus-5-5', 'openai': _MODEL, 'json': _MODEL, 'azure': _MODEL}


STYLE_RULES = {
    'concise': """

ANSWER FORMAT (operators read this under pressure)
- First sentence: the direct answer to the question (the figure, time, station, yes/no or recommendation asked for).
- Then the key figures, as a short table when there are several stations or times.
- For operational questions, then "What to do": staff, trains, reinforcement, passenger information.
- Then at most two caveats (premise corrections, definitions, planning assumptions).
- Keep it under about 250 words unless the user asks for detail or the question has several parts; never add background the question did not ask for.""",
    'detailed': '',
}


def azure_client(url=None, api_key=None, api_version=None):
    """AzureOpenAI client from the endpoint URL as the provider gives it, e.g.
    https://<host>/<optional gateway path>/openai/responses?api-version=2025-04-01-preview
    (the part before /openai is the endpoint; api-version is read from the URL unless given)."""
    from urllib.parse import parse_qs, urlparse
    from openai import AzureOpenAI
    raw = url or os.environ.get('AZURE_ENDPOINT', '')
    if not raw:
        raise RuntimeError('Set AZURE_ENDPOINT to the URL you were given.')
    u = urlparse(raw)
    version = api_version or os.environ.get('OPENAI_API_VERSION') or parse_qs(u.query).get('api-version', ['2025-04-01-preview'])[0]
    endpoint = f"{u.scheme}://{u.netloc}{u.path.split('/openai')[0]}".rstrip('/')
    # max_retries=0: a retry would blow the 26 s budget; the timeout is set per call from the remaining budget
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key or os.environ.get('AZURE_API_KEY'), api_version=version,
                       max_retries=0)
MAX_STEPS = 12              # tool-use rounds per question
MAX_TOOL_CHARS = 20000      # tool output sent back to the model
MIN_CALL_S = 2.0            # with a deadline: no new model call with less time left than this
FINAL_CALL_S = 14.0         # with a deadline: below this, the model must answer without further tools (writing takes ~8 s)
SELF_CHECK_MIN_S = 10.0     # with a deadline: the figure self-correction round needs at least this much time


class AgentTimeout(RuntimeError):
    """The answer budget (Agent.deadline) ran out before the model wrote its answer."""

SYSTEM_PROMPT = """You are the U-Bahn Operations Assistant for Berlin U-Bahn operators.

DATA
- Simulated 15-minute passenger flows for 168 U-Bahn stations on lines U1, U2, U3, U5, U6, U7, U8, U9 (no U4; U6 ends at Kurt-Schumacher-Platz), with weather, events, closures and daily energy per line. Training period 10 June to 21 September 2026; evaluation files may extend it.
- A service day runs from 05:00 to 00:45; slots after midnight belong to the previous day.
- Each station has a ceiling: the most its counter ever records in 15 minutes. Call it the station's ceiling (the highest 15-minute flow the data records, used as a proxy for platform capacity); do not present it as a measured safe capacity.
- 27 stations are S+U (S-Bahn interchanges) and the tools say which. S-Bahn, trams and buses have no data: mention them only as general knowledge, and say so.

HOW TO WORK
1. Every number you give must come from a tool result. Never estimate from memory.
2. Check the premise first. If a station is not on the named line (ubahn_check_line_section), a name matches several stations (for example "Spandau": Altstadt Spandau and Rathaus Spandau), a venue has been renamed (Mercedes-Benz Arena is now the Uber Arena), or a date is outside the data, say so plainly and answer the plausible reading or readings.
2b. A question about a date inside the data is answered from the recorded data, even when it is phrased in the future tense ("what will the flow look like on 23 June"): report what happened, compared with normal (ubahn_event_report, ubahn_compare_to_normal, ubahn_flow_stats). Lead with the recorded facts the tools list first (for events: "facts_from_recorded_data"). What-if tools (ubahn_event_impact, ubahn_capacity_risk with scenarios, ubahn_suspension_scenario) are for dates after the data, hypothetical situations, or a planning view added after the recorded facts.
2c. Always report closures or suspensions on the same day in the area concerned when a tool lists them, even if the question does not ask; they change the operational advice.
2d. For an event, check the events data first (ubahn_list_events). If it is recorded, answer from the recorded days (ubahn_event_report, ubahn_compare_to_normal) and use scenarios only for what-ifs such as a larger crowd. InnoTrans 2026 (Messe Berlin, Messedamm 22, 22 to 25 September 2026, about 09:00 to 18:00) is recorded once the evaluation files are loaded. For an event that is not in the data, do not stop at "no data": build explicit scenarios (for example 5,000 visitors a day at the dataset's scale and 40,000 at real-world scale, dry and rainy weather), run the what-if tools (ubahn_capacity_risk with extra_events, ubahn_diversion_scenario) and state the assumptions.
3. For plain data questions (how many, average, busiest, compared with normal, which events or closures, facts about a station, weather), use the generic tools: ubahn_flow_stats, ubahn_rank_stations, ubahn_compare_to_normal, ubahn_list_events, ubahn_list_closures, ubahn_station_info, ubahn_weather.
4. For these analysis questions a dedicated tool exists:
   - an event at a venue or on a date: ubahn_event_report (in the data) or ubahn_event_impact (future or what-if)
   - a weather-driven peak in a period: ubahn_weather_peaks
   - a closure or suspension (reason, duration, rerouting, overload, staff): ubahn_closure_report (in the data) or ubahn_suspension_scenario (what-if)
   - when a station usually peaks, compared with the network: ubahn_peak_time
   - energy efficiency by line: ubahn_energy_efficiency
   - critical stations, network fragmentation: ubahn_fragmentation_ranking
   - anomalies on a date: ubahn_anomaly_scan
   - stations whose demand moves together: ubahn_station_dependencies
   - how passengers behave during disruptions: ubahn_disruption_response
   - the best infrastructure investment: ubahn_best_new_link, together with ubahn_energy_efficiency and ubahn_fragmentation_ranking
   - alternative routes or spreading an event crowd: ubahn_diversion_scenario and ubahn_route
   - forecasts or capacity risk for a window or scenario: ubahn_capacity_risk
   - observed against expected at one station, slot by slot: ubahn_explain_station
   - what to expect or prepare for a day (briefing): ubahn_daily_brief; what to watch and when to act: ubahn_action_plan; where to put staff: ubahn_staff_plan
   - a forecast with plausible ranges: ubahn_forecast; the state of the network at one moment ("what would we have seen at 22:45?"): ubahn_snapshot
   - what the model learned from newly added data: ubahn_learning_report
   - energy over a period, per line or per day: ubahn_energy_stats (generic); passenger messages for a closure or an event: ubahn_passenger_messages
   - how reliable the model and the answers are: ubahn_reliability; what the system would have achieved over the summer (value, alerts, planning): ubahn_value_report
   - two scenarios side by side: ubahn_compare_scenarios; what to do if a given station closes: ubahn_station_contingency; a report on a past day: ubahn_incident_report
   - reinforcing transport when many people are expected (more U-Bahn or S-Bahn trains, shuttle buses, taxis, bikes, walking routes, metering): ubahn_reinforcement_plan; use it for "what measures should we take" questions about crowds, and state that other modes rest on planning assumptions
5. Combine tools freely and call several when a question has several parts: for example list a day's events, then compare the nearby stations with normal; or rank stations, then explain the top one.
5b. An alternative route that boards the same line further along (for example Sophie-Charlotte-Platz on U2 for the Messe) spreads platform crowding but adds no line capacity, because the trains arrive already loaded. For "alternative" or "unconventional" routes, prefer another corridor (another line): ubahn_diversion_scenario proposes one when alternative_stations is left empty.
6. Question the tools. Each one encodes a definition: energy per passenger counts interchange stations for each line they serve; the "usual peak" of a station is a weekday mean; the rerouting "planning case" assumes displaced passengers walk to the nearest station of another line; the ceiling is the highest reading ever recorded. When the question implies another definition, say so and compute it with the generic tools or run_python. If no tool fits, use run_python (or ubahn_flow_stats) and say in one sentence how the number was obtained.
7. Keep three kinds of statement apart: what the data shows, what the model expects, and planning assumptions. Line suspensions leave no trace in the flows, and passengers in this data never reroute (they wait for their station to reopen), so rerouting advice is a planning assumption and must be labelled as one.
8. Single 15-minute readings are very noisy: one reading can be a tenth or three times its expected level, and about 11 isolated spikes a day are normal. Base conclusions on windows, several stations or network totals, and compare with other weeks (ubahn_compare_to_normal) before calling something unusual. An anomaly must be robust (several slots, several stations, or the whole network); an isolated single-slot spike is not an anomaly and should be mentioned only as normal noise. For anomaly questions, use the "ranked_findings" of ubahn_anomaly_scan in that order.
9. For forecasts, state the assumptions (weather, attendance) and give two scenarios when they change the answer.
10. Time conventions: slots are labelled by their start time, so "from 16:30 to 18:30" means the slots 16:30 to 18:15. The slots 00:00 to 00:45 of a calendar date are the end of the previous service day, and 01:00 to 04:45 has no service. Say which convention you used when it changes the number.

HOW TO ANSWER
- Start with the direct answer in one or two sentences.
- Then the key numbers, with a small table when comparing stations, times or options. Flows are passengers per 15 minutes unless stated.
- Then operational recommendations (staff, trains, information, timing) tied to specific stations and times.
- End with at most two lines of caveats.
- Give counts exactly as the tools return them. Round derived values for readability: averages to whole passengers, energy to 1 decimal MWh (99.8 MWh, not 99.827), kWh per passenger to 2 decimals, percentages to whole numbers.
- Write the final answer cleanly: no self-corrections, question marks after figures or "Correction:" notes. If a figure is missing, leave it out or call the tool again.
- Read tool figures by their names: "total" figures are sums over the whole window (and the listed stations), not per 15 minutes; a "_pct" figure is a percentage; "before" and "after" figures of an intervention refer to the same scope. Do not build a comparison from two figures with different scopes.
- Use plain operator language: no jargon such as lambda or z-score; say "expected level", "normal", "chance of reaching the ceiling".
- Reply in the language of the question: an English question gets an English answer, a French question a French answer. Never switch to German because the data is about Berlin.

REASONING CHECKS (a figure taken from a tool does not make the conclusion right)
- Recorded before hypothetical: when an event, closure or suspension happened at a recorded time, give what happened and the risks and staffing for that actual time first. Peak-hour or other what-if cases come after, labelled "hypothetical".
- Proposals are proposals: staff, extra trains, buses, diversions and energy interventions come from planning tools and rest on stated assumptions. Present them as proposals, never as measured effects.
- Diversions are options to investigate: give the share the alternative stations can take without exceeding their ceiling (capacity_limited_share) and how passengers reach them (walking time, shuttle). Never recommend a diversion that pushes a station beyond its ceiling.
- Crowd control: meter the inflow at the entrances and hold passengers at street level or before the gates, never on platforms; keep exits clear.
- Causality: say "coincided with" or "likely contributed to" unless a tool attributes the effect. When a tool gives a share explained (rain explains 46 %), say "partly".
- Network totals and station readings are different things: name the station with the highest reading separately from the network peak.
- Near-ties: when two compared values differ by less than 2 %, say "about equal" and give both. For "usually" questions, state the period and number of days used; when a tool also gives the comparison before the last data injection, report both periods.
- Call a station an interchange only if the tools list more than one line for it.
- Keep the conditions tools state: fragmentation assumes the station and its tracks are lost; the best new link is best among the candidates tested, with costs not evaluated; a count may cover the cut-off stations only.
- The data are simulated: describe patterns such as the rebound after a reopening as behaviour of the data, not as observed passenger preferences.
- An anomaly answer lists distinct causes: the arrival and departure of one event are one anomaly."""


# ------------------------------------------------------------------------------ JSON helpers
def to_jsonable(x):
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y-%m-%d %H:%M')
        return json.loads(df.reset_index().round(3).to_json(orient='records', force_ascii=False))
    if isinstance(x, pd.Series):
        return to_jsonable(x.to_frame())
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return A._py(x)


def weather_arg(tmean, prcp):
    return None if tmean is None else {'tmean': float(tmean), 'prcp': float(prcp or 0.0)}


# ------------------------------------------------------------------------------ tools
class Toolbox:
    def __init__(self, eng: UBahnEngine):
        self.eng = eng
        e = eng
        S = {'type': 'string'}
        N = {'type': 'number'}
        I = {'type': 'integer'}
        TIME = {'type': 'string', 'description': "local Berlin time, 'YYYY-MM-DD HH:MM'"}
        DATE = {'type': 'string', 'description': "'YYYY-MM-DD'"}
        TMEAN = {'type': 'number', 'description': 'scenario daily mean temperature in degC (only for dates without recorded weather)'}
        PRCP = {'type': 'number', 'description': 'scenario rain in mm/h (0 = dry)'}
        EVENT = {'type': 'object', 'properties': {'venue': S, 'start': TIME, 'end': TIME, 'attendance': N},
                 'required': ['venue', 'start', 'end', 'attendance']}
        HHMM = {'type': 'string', 'description': "time of day 'HH:MM' (service-day order, 05:00 to 00:45)"}
        DAYS = {'type': 'string', 'enum': list(A.DAY_TYPES), 'description': 'which service days (default all)'}
        STATIONS = {'type': 'array', 'items': S, 'description': 'station names; omit for a line or the whole network'}
        CLOSURE = {'type': 'object', 'properties': {'description': S, 'start': TIME, 'end': TIME},
                   'required': ['description', 'start', 'end'],
                   'description': "e.g. {'description': 'Station Kaiserdamm closed', 'start': ..., 'end': ...}"}
        SCEN = {'tmean': TMEAN, 'prcp': PRCP, 'extra_events': {'type': 'array', 'items': EVENT},
                'extra_closures': {'type': 'array', 'items': CLOSURE}}

        def obj(props, req=()):
            return {'type': 'object', 'properties': props, 'required': list(req)}

        self.tools = [
            ('ubahn_model_card', 'Plain-language summary of the data and of the rules the engine has learned (profile, weather, events, closures, ceilings, energy).',
             obj({}), lambda: e.model_card()),
            ('ubahn_data_status', 'What data is loaded (files, date ranges, counts), what the engine could not interpret (venues it cannot locate, closures it does not understand), and a check of the latest days against the model. Use it first when a question concerns recent or newly added data.',
             obj({}), self._data_status),
            ('ubahn_check_line_section', 'Premise check: ordered stations of a line between two stations. Returns an error with the full station list if a station is not on the line.',
             obj({'line': S, 'station_a': S, 'station_b': S}, ['line', 'station_a', 'station_b']), self._section),
            ('ubahn_flow_stats', 'Recorded passenger flow for stations, a line or the whole network. Absolute window: start and end (end excluded), e.g. passengers between 00:00 and 12:00 on a date. Typical day: a daily window time_from-time_to over selected days (days, date_from/date_to or dates) with stat = mean (default), median, sum or max across days. by = total, slot, hour or day for a breakdown (average traffic quarter by quarter: by=slot).',
             obj({'stations': STATIONS, 'line': S, 'start': TIME, 'end': TIME, 'time_from': HHMM, 'time_to': HHMM,
                  'date_from': DATE, 'date_to': DATE, 'days': DAYS, 'dates': {'type': 'array', 'items': DATE},
                  'by': {'type': 'string', 'enum': ['total', 'slot', 'hour', 'day']},
                  'stat': {'type': 'string', 'enum': ['sum', 'mean', 'median', 'max']}}), lambda **k: A.flow_stats(e, **k)),
            ('ubahn_rank_stations', 'Rank stations over a period by total passengers, mean per day, highest 15-min reading (peak_slot), slots at the ceiling, or observed vs the model (vs_expected). Period as in ubahn_flow_stats; line restricts to one line.',
             obj({'metric': {'type': 'string', 'enum': ['total', 'mean_per_day', 'peak_slot', 'slots_at_ceiling', 'vs_expected']},
                  'start': TIME, 'end': TIME, 'time_from': HHMM, 'time_to': HHMM, 'date_from': DATE, 'date_to': DATE,
                  'days': DAYS, 'line': S, 'top': I, 'ascending': {'type': 'boolean'}}), lambda **k: A.rank_stations(e, **k)),
            ('ubahn_compare_to_normal', 'Is a window unusual? Recorded flow for stations / a line / the network between start and end (end excluded), against the model expectation and the same weekday and hours in the other weeks (mean, min, max, share of weeks higher), with events and closures in the window.',
             obj({'start': TIME, 'end': TIME, 'stations': STATIONS, 'line': S}, ['start', 'end']), lambda **k: A.compare_to_normal(e, **k)),
            ('ubahn_list_events', 'Events in the data (listings of the same show added up) filtered by dates, venue or text, or distance to a station, with attendance and the stations their crowds use.',
             obj({'date_from': DATE, 'date_to': DATE, 'venue': S, 'near_station': S, 'radius_km': N, 'min_attendance': N, 'limit': I}),
             lambda **k: A.list_events(e, **k)),
            ('ubahn_list_closures', 'Closures and suspensions in the data filtered by dates, line, station or kind (station, line, platform, segment), with reason and duration.',
             obj({'date_from': DATE, 'date_to': DATE, 'line': S, 'station': S,
                  'kind': {'type': 'string', 'enum': ['station', 'line', 'platform', 'segment', 'unknown']}}), lambda **k: A.list_closures(e, **k)),
            ('ubahn_station_info', 'Facts about one station: lines, S-Bahn interchange, adjacent stations, stations and event venues within walking distance, ceiling, typical daily traffic and peaks, closures.',
             obj({'station': S, 'radius_km': N}, ['station']), lambda station, radius_km=1.0: A.station_info(e, station, radius_km)),
            ('ubahn_weather', 'Weather for a date or period, by day or by hour, with the flow multiplier the model attributes to it (1.0 = 18 degC and dry).',
             obj({'date_from': DATE, 'date_to': DATE, 'by': {'type': 'string', 'enum': ['day', 'hour']}}, ['date_from']),
             lambda **k: A.weather(e, **k)),
            ('ubahn_daily_brief', 'Proactive brief for a service day (in the data or a future scenario): weather, events, closures, the stations and windows at risk, a staff plan and the key "if ... then ..." triggers. Use it for "what should we expect / prepare today".',
             obj({'date': DATE, 'staff': I, **SCEN}, ['date']), self._brief),
            ('ubahn_action_plan', 'Conditional action plan for a service day or scenario: station windows to watch, calibrated thresholds (plausible ranges), triggers and actions with their indicator; for a day in the data, when each trigger would really have fired.',
             obj({'date': DATE, 'top': I, **SCEN}, ['date']), self._plan),
            ('ubahn_staff_plan', 'Where and when to deploy N additional staff (beyond usual peak staffing) to cover the most demand above capacity and event crowds; shifts per station.',
             obj({'date': DATE, 'staff': I, 'block_hours': N, **SCEN}, ['date']), self._staff),
            ('ubahn_forecast', 'Expected 15-min flow with calibrated plausible ranges (80 % and 90 %) for stations or a line over a window, the normal level, chance of reaching the ceiling and, in the data, the recorded reading.',
             obj({'start': TIME, 'end': TIME, 'stations': STATIONS, 'line': S, **SCEN}, ['start', 'end']), self._forecast),
            ('ubahn_snapshot', 'The network at one moment ("what would we have seen at 22:45 on 23 June?"): biggest deviations from normal, stations at the ceiling, events in their arrival or departure phase, weather and closures in force.',
             obj({'at': TIME, 'top': I}, ['at']), lambda at, top=10: OPS.snapshot(e, at, top)),
            ('ubahn_reinforcement_plan', 'When demand exceeds what a normal day asks of the U-Bahn (events, rain, closures): how to absorb it across modes - extra U-Bahn trains per hour, extra S-Bahn trains where an S-Bahn station is within walking distance, walking or shuttle buses to a station on another line (within its spare capacity), shared bikes (dry daytime), taxis (night first), and what must be metered at the entrances. Quantities rest on planning assumptions that the result lists; override them with assumptions.',
             obj({'date': DATE, 'top': I, 'assumptions': {'type': 'object', 'description': 'e.g. {"ubahn_max_trains_per_hour": 24, "buses_max": 30, "bus_capacity": 100}'}, **SCEN}, ['date']),
             self._reinforce),
            ('ubahn_compare_scenarios', 'Two scenarios of the same day side by side (weather, extra events, closures): network passengers, passengers above capacity, stations likely to reach their ceiling, the stations that change the most, and each scenario\'s reinforcement.',
             obj({'date': DATE, 'scenario_a': {'type': 'object', 'description': 'a scenario: {label, tmean, prcp, extra_events: [...], extra_closures: [...]}; empty = the day as recorded (or a dry 16 degC day)'}, 'scenario_b': {'type': 'object', 'description': 'a scenario: {label, tmean, prcp, extra_events: [...], extra_closures: [...]}; empty = the day as recorded (or a dry 16 degC day)'}, 'top': I}, ['date']),
             lambda date, scenario_a=None, scenario_b=None, top=10: REP.compare_scenarios(e, date, scenario_a, scenario_b, top)),
            ('ubahn_station_contingency', 'What to do if a station closes (one page of the contingency handbook): passengers affected, stations cut off if its tracks are out, alternatives on foot and by S-Bahn, measures and ready passenger messages.',
             obj({'station': S}, ['station']), lambda station: REP.station_contingency(e, station)),
            ('ubahn_incident_report', 'Post-event report of a recorded day: situation, events and closures with their recorded impact, anomalies, when the alerts would have fired, reinforcement and staff the plans would have recommended, lessons learned. (The printable HTML version is in the app, Reports tab.)',
             obj({'date': DATE}, ['date']), lambda date: {k: v for k, v in REP.incident_report(e, date).items() if k != 'html'}),
            ('ubahn_value_report', 'Proof of value over the whole recorded period: how much of the event-related overloads the plan flagged in advance, how live alert rules perform (events detected, alerts per day, share explained), and the demand above capacity the reinforcement plans would have absorbed. Takes about 10 s the first time.',
             obj({}), lambda: {k: v for k, v in REP.value_report(e).items() if not k.startswith('_')}),
            ('ubahn_energy_stats', 'Recorded energy per line over any period (total, per day, highest/lowest day), ridership, kWh per passenger and how far days were from what ridership explains. Generic building block for energy questions.',
             obj({'lines': {'type': 'array', 'items': S}, 'date_from': DATE, 'date_to': DATE, 'days': DAYS, 'by': {'type': 'string', 'enum': ['total', 'day']}}),
             lambda lines=None, date_from=None, date_to=None, days='all', by='total': OPS.energy_stats(e, lines, date_from, date_to, days, by)),
            ('ubahn_passenger_messages', 'Ready-to-broadcast passenger messages in German, English and French (announcement, display, app) for a closure (description with start and end, or a closure of the data found by text and date) or an event (name and date from the data, or venue with event_start and event_end).',
             obj({'closure': S, 'start': TIME, 'end': TIME, 'event': S, 'date': DATE, 'venue': S, 'event_start': TIME, 'event_end': TIME}),
             lambda **kw: OPS.passenger_messages(e, **kw)),
            ('ubahn_reliability', 'Evidence that the numbers can be trusted: calibration of the forecast ranges, backtest of the model against the recorded data, how answers are checked, known limits.',
             obj({}), lambda: OPS.reliability_report(e)),
            ('ubahn_learning_report', 'What the model learned from the last data injection: new days, venues whose crowd split was learned or changed, how the new days fit the model, what could not be read.',
             obj({}), lambda: OPS.last_learning_report(e)),
            ('ubahn_explain_station', 'Observed vs expected 15-min flow at one station over a window, with active drivers (events, closures, weather) and slots at the ceiling. Only for times inside the data.',
             obj({'station': S, 'start': TIME, 'end': TIME}, ['station', 'start', 'end']), self._explain),
            ('ubahn_capacity_risk', 'Rank stations by chance of reaching their ceiling in a window. For dates without recorded weather pass tmean and prcp. extra_events adds scenario events (e.g. InnoTrans at Messedamm 22).',
             obj({'start': TIME, 'end': TIME, 'tmean': TMEAN, 'prcp': PRCP, 'top': I,
                  'sort_by': {'type': 'string', 'enum': ['exp_slots_at_ceiling', 'p_any_slot_at_ceiling', 'exp_pax_above_ceiling', 'peak_lambda_over_ceiling']},
                  'use_data_events': {'type': 'boolean', 'description': 'include events from the events file (default true)'},
                  'extra_events': {'type': 'array', 'items': EVENT}}, ['start', 'end']), self._capacity_risk),
            ('ubahn_event_impact', 'What-if for one event: extra passengers per station and slot on top of normal, and chance of reaching the ceiling, from 90 min before the start to 2 h 15 after the end.',
             obj({'venue': S, 'start': TIME, 'end': TIME, 'attendance': N, 'tmean': TMEAN, 'prcp': PRCP}, ['venue', 'start', 'attendance']), self._event_impact),
            ('ubahn_suspension_scenario', 'What-if for a suspended section: rerouting topology and risk at the stations likely to absorb passengers, as in the data and in a planning case.',
             obj({'line': S, 'station_a': S, 'station_b': S, 'at': TIME, 'minutes': I, 'tmean': TMEAN, 'prcp': PRCP}, ['line', 'station_a', 'station_b', 'at']), self._suspension),
            ('ubahn_route', 'Realistic U-Bahn route between two stations (changes of line penalised): stops, changes and legs.',
             obj({'origin': S, 'destination': S}, ['origin', 'destination']), lambda origin, destination: A.route(e, origin, destination)),
            ('ubahn_event_report', 'What an event in the data did to the stations around its venue: arrival and departure waves (observed vs normal), timeline, ceiling hits, closures and weather that day, other dates at the venue. query = artist, event name, venue or address.',
             obj({'query': S, 'date': DATE}, ['query']), lambda query, date=None: A.event_report(e, query, date)),
            ('ubahn_weather_peaks', 'Strongest bad-weather passenger peak between two dates: rain episode, network peak slot vs a dry same weekday, stations at their ceiling, top station readings, share explained by rain.',
             obj({'start': DATE, 'end': DATE}, ['start', 'end']), lambda start, end: A.weather_peaks(e, start, end)),
            ('ubahn_closure_report', 'Look up a closure in the data by line + section stations, or by station, and/or date: reason, duration, rerouting, what the data shows, overload (actual time and evening peak), where to put staff.',
             obj({'line': S, 'station_a': S, 'station_b': S, 'station': S, 'date': DATE}), lambda **k: A.closure_report(e, **k)),
            ('ubahn_peak_time', 'Usual weekday commute peak of a station (time, level) compared with the mean and median peak of all stations, with robustness checks.',
             obj({'station': S}, ['station']), lambda station: A.peak_time(e, station)),
            ('ubahn_energy_efficiency', 'Energy per passenger for every line, fixed vs per-passenger energy, passengers per km, weekday/weekend, and interventions for the worst line.',
             obj({}), lambda: A.energy_efficiency(e)),
            ('ubahn_fragmentation_ranking', 'Stations whose closure (station and tracks) splits the network the most, with passengers affected, walking links across the gap and mitigation principles.',
             obj({'top': I}), lambda top=5: A.fragmentation_ranking(e, top)),
            ('ubahn_anomaly_scan', 'Unusual flows on one date and their likely causes: event clusters, rain episodes, isolated spikes vs the normal rate, hidden closures, energy residuals.',
             obj({'date': DATE}, ['date']), lambda date: A.anomaly_scan(e, date)),
            ('ubahn_station_dependencies', 'Non-adjacent station pairs whose demand moves together, whether the link survives outside event hours, and the venue behind it (~5 s).',
             obj({'top': I}), lambda top=8: A.station_dependencies(e, top)),
            ('ubahn_disruption_response', 'What passengers actually do during station closures and line suspensions (pooled over all disruptions in the data), rebound after reopening, and an example.',
             obj({}), lambda: A.disruption_response(e)),
            ('ubahn_best_new_link', 'Test every short new link between lines and rank by resilience gain; includes disruption history by line (~20 s).',
             obj({'max_km': N, 'top': I}), lambda max_km=1.3, top=5: A.best_new_link(e, max_km, top)),
            ('ubahn_diversion_scenario', "Send a share of an event crowd to alternative stations on ANOTHER line (shuttle or walk) and compare peak demand / ceiling for the exit wave, with default vs alternative routes to the centre. Leave alternative_stations empty and the tool picks the nearest stations of another corridor; it warns when a proposed station is on the same line as the venue (trains arrive already full).",
             obj({'venue': S, 'start': TIME, 'end': TIME, 'attendance': N, 'alternative_stations': {'type': 'array', 'items': S},
                  'destinations': {'type': 'array', 'items': S}, 'tmean': TMEAN, 'prcp': PRCP},
                 ['venue', 'start', 'end', 'attendance']), self._diversion),
            ('run_python', 'Run Python for questions no other tool covers. Available: eng (UBahnEngine: eng.flows [timestamp x station], eng.weather, eng.events, eng.closures, eng.energy, eng.stations, eng.graph, eng.line_seq, eng.ceiling), A (analyses), pd, np, nx, TOD_ORDER. Print what you need or assign it to `result`.',
             obj({'code': S}, ['code']), self._run_python),
        ]
        self.by_name = {name: fn for name, _, _, fn in self.tools}

    # -- wrappers -------------------------------------------------------------------------
    def _data_status(self):
        e = self.eng
        latest = max(e.flows.index.max() - pd.Timedelta('9D'), e.flows.index.min()).normalize() + pd.Timedelta('5h')
        return {'status': e.data_status(), 'model_check_latest_days': e.calibration(latest, e.flows.index.max()),
                'closures_in_latest_days': [f"{c['description']} ({c['start']:%a %d %b %H:%M}-{c['end']:%H:%M})"
                                            for c in e._clist if c['start'] >= latest],
                'events_in_latest_days': [f"{r.event_name} @ {r.address} ({r.start:%a %d %b %H:%M}, {int(r.estimated_attendance)})"
                                          for r in e.events[e.events.start >= latest].sort_values('start').itertuples()][:40]}

    def _scen(self, tmean, prcp, extra_events, extra_closures):
        return weather_arg(tmean, prcp), extra_events or None, extra_closures or None

    def _brief(self, date, staff=10, tmean=None, prcp=None, extra_events=None, extra_closures=None):
        w, ev, cl = self._scen(tmean, prcp, extra_events, extra_closures)
        return OPS.daily_brief(self.eng, date, w, ev, cl, staff)

    def _plan(self, date, top=8, tmean=None, prcp=None, extra_events=None, extra_closures=None):
        w, ev, cl = self._scen(tmean, prcp, extra_events, extra_closures)
        return OPS.action_plan(self.eng, date, w, ev, cl, top)

    def _reinforce(self, date, top=6, assumptions=None, tmean=None, prcp=None, extra_events=None, extra_closures=None):
        w, ev, cl = self._scen(tmean, prcp, extra_events, extra_closures)
        return OPS.reinforcement_plan(self.eng, date, w, ev, cl, top, assumptions)

    def _staff(self, date, staff=10, block_hours=2.0, tmean=None, prcp=None, extra_events=None, extra_closures=None):
        w, ev, cl = self._scen(tmean, prcp, extra_events, extra_closures)
        return OPS.staff_plan(self.eng, date, staff, block_hours, w, ev, cl)

    def _forecast(self, start, end, stations=None, line=None, tmean=None, prcp=None, extra_events=None, extra_closures=None):
        w, ev, cl = self._scen(tmean, prcp, extra_events, extra_closures)
        return OPS.forecast(self.eng, start, end, stations, line, w, ev, cl)

    def _section(self, line, station_a, station_b):
        try:
            return {'ok': True, 'section': self.eng.section(line, station_a, station_b)}
        except ValueError as err:
            return {'ok': False, 'error': str(err)}

    def _explain(self, station, start, end):
        table, drivers = self.eng.explain(station, start, end)
        return {'drivers': drivers, 'slots': table}

    def _capacity_risk(self, start, end, tmean=None, prcp=None, top=10, sort_by='exp_slots_at_ceiling', use_data_events=True, extra_events=None):
        return self.eng.capacity_risk(start, end, weather=weather_arg(tmean, prcp), events='data' if use_data_events else [],
                                      extra_events=extra_events, top=top, sort_by=sort_by)

    def _event_impact(self, venue, start, attendance, end=None, tmean=None, prcp=None):
        r = self.eng.event_impact(venue, start, end=end, attendance=attendance, weather=weather_arg(tmean, prcp))
        tot = r['baseline_lambda'] + r['extra_lambda']
        rows = pd.concat({'expected': tot.round(0), 'extra': r['extra_lambda'], 'chance_at_ceiling': r['p_at_ceiling']}, axis=1)
        rows.columns = [f'{st} {what}' for what, st in rows.columns]
        return {'venue': r['venue'], 'station_shares': r['station_shares'], 'ceiling': r['ceiling'], 'timeline': rows}

    def _suspension(self, line, station_a, station_b, at, minutes=20, tmean=None, prcp=None):
        r = self.eng.suspension_scenario(line, station_a, station_b, at, minutes=minutes, weather=weather_arg(tmean, prcp))
        return {k: r[k] for k in ('reroute', 'slots', 'as_in_data', 'planning_case', 'displaced_from', 'no_u_bahn_alternative')}

    def _diversion(self, venue, start, end, attendance, alternative_stations=None, destinations=None, tmean=None, prcp=None):
        return A.diversion_scenario(self.eng, venue, start, end, attendance, alternative_stations, destinations,
                                    weather=weather_arg(tmean, prcp))

    def _run_python(self, code, timeout_s=60):
        ns = {'eng': self.eng, 'A': A, 'pd': pd, 'np': np, 'nx': nx, 'TOD_ORDER': TOD_ORDER}
        buf = io.StringIO()
        use_alarm = hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread()  # server: worker thread
        if use_alarm:
            def _timeout(*_):
                raise TimeoutError(f'code ran longer than {timeout_s} s')
            old = signal.signal(signal.SIGALRM, _timeout); signal.alarm(timeout_s)
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)                      # trusted demo use only: no sandboxing
        finally:
            if use_alarm:
                signal.alarm(0); signal.signal(signal.SIGALRM, old)
        out = {'stdout': buf.getvalue()[-8000:]}
        if 'result' in ns:
            out['result'] = ns['result']
        return out

    # -- dispatch ---------------------------------------------------------------------------
    def schemas(self):
        return [{'name': n, 'description': d, 'parameters': p} for n, d, p, _ in self.tools]

    def call(self, name, args):
        """Run a tool; returns (text for the model, is_error)."""
        fn = self.by_name.get(name)
        if fn is None:
            return json.dumps({'error': f'unknown tool {name}'}), True
        try:
            res = fn(**(args or {}))
            text = json.dumps(to_jsonable(res), ensure_ascii=False, default=str)
            err = False
        except Exception as ex:                     # engine errors are useful to the model (e.g. premise checks)
            text, err = json.dumps({'error': f'{type(ex).__name__}: {ex}'}, ensure_ascii=False), True
        if len(text) > MAX_TOOL_CHARS:
            text = text[:MAX_TOOL_CHARS] + ' ... [truncated: ask for a narrower window or fewer stations]'
        return text, err


# ------------------------------------------------------------------------------ LLM loops
class Agent:
    def __init__(self, eng, backend=DEFAULT_BACKEND, model=None, show_tools=False, client=None, audit_path=None, progress_cb=None,
                 style=None, self_check=None):
        backend = BACKEND_ALIASES.get(backend, backend)
        self.deadline = None                 # time.time() by which ask() must be done (set by answer()); None = no limit
        self.tb = Toolbox(eng)
        self.trace = []                      # tool calls of the last question: (name, args, chars, error)
        self.audit_path = audit_path         # JSON-lines audit log (one line per answer), None = off
        self.style = style or os.environ.get('UBAHN_ANSWER_STYLE', 'concise')
        self.system_prompt = SYSTEM_PROMPT + STYLE_RULES.get(self.style, '')
        self.self_check = (os.environ.get('UBAHN_SELF_CHECK', '1') != '0') if self_check is None else bool(self_check)
        self.last_self_check = None
        self.progress_cb = progress_cb       # callable(event dict) for live progress, None = off
        self.last_verification = None
        self.backend = backend
        self.model = model or DEFAULT_MODELS[backend]
        if not self.model:
            raise RuntimeError('Set MODEL_NAME (the deployment name served by the endpoint) or give --model.')
        self.show = show_tools
        self.history = []
        if client is not None:
            self.client = client
        elif backend == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic()
        elif backend == 'azure':
            self.client = azure_client()
        else:                               # 'openai' and 'json' use the same OpenAI-compatible client
            from openai import OpenAI
            self.client = OpenAI(base_url=os.environ.get('OPENAI_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'not-needed'))

    def _remaining(self):
        return float('inf') if self.deadline is None else self.deadline - time.time()

    def reload(self, data_dir=None):
        """Rebuild the engine from the data folder (after new files were injected) and keep the conversation."""
        eng = UBahnEngine(data_dir or self.tb.eng.data_dir)
        for f in (A._daily, A._line_graph, A._residual_z, A._vulnerability):
            f.cache_clear()
        self.tb = Toolbox(eng)
        return eng.data_status()

    def _emit(self, kind, **kw):
        """Progress event for a live display (app): model | tool_start | tool_end. Never breaks an answer."""
        if self.progress_cb:
            try:
                self.progress_cb({'kind': kind, **kw})
            except Exception:
                pass

    def _tool(self, name, args):
        self._emit('tool_start', tool=name, args=args)
        t0 = time.time()
        text, err = self.tb.call(name, args)
        secs = time.time() - t0
        self._log(name, args, text, err, secs)
        self._emit('tool_end', tool=name, args=args, chars=len(text), error=err, seconds=secs)
        return text, err

    def _audit(self, question, answer, seconds):
        """Append one JSON line per answer: question, answer, figure check, every tool call and its result."""
        if not self.audit_path:
            return
        try:
            v = self.last_verification or {}
            eng = self.tb.eng
            rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'backend': self.backend, 'model': self.model,
                   'seconds': round(seconds, 1), 'question': question, 'answer': answer,
                   'figures': v.get('summary', {}), 'self_check': self.last_self_check,
                   'steps': [{'tool': c['tool'], 'args': c['args'], 'chars': c['chars'], 'error': c['error'],
                              'seconds': round(c.get('seconds', 0), 2), 'output': c['output'][:4000]} for c in self.trace],
                   'data': {'flows_end': str(eng.flows.index.max()), 'fitted_at': getattr(eng, 'fitted_at', None)}}
            d = os.path.dirname(os.path.abspath(self.audit_path))
            os.makedirs(d, exist_ok=True)
            with open(self.audit_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except Exception as ex:
            print(f'(audit log not written: {ex})', file=sys.stderr)

    def _log(self, name, args, text, err, seconds=0.0):
        self.trace.append({'tool': name, 'args': args, 'chars': len(text), 'error': err, 'output': text, 'seconds': seconds})
        if self.show:
            print(f'  [tool] {name}({json.dumps(args, ensure_ascii=False)[:160]}) -> {len(text)} chars{" ERROR" if err else ""}', file=sys.stderr)

    def _ask_backend(self, question):
        if self.backend == 'anthropic':
            return self._ask_anthropic(question)
        if self.backend == 'json':
            return self._ask_json(question)
        if self.backend == 'azure':
            return self._ask_responses(question)
        return self._ask_openai(question)

    def _repair(self, answer, v):
        """One correction round when the figure check finds figures in no tool result: the model checks them with
        the tools and rewrites the answer. The corrected answer is kept only if it has fewer unsupported figures."""
        bad = v['summary']['not_found_values'][:10]
        prompt = ('Figure check before this answer is shown: these figures from your answer do not appear in any tool '
                  f"result: {', '.join(bad)}. Check each one with the tools, then replace it with the figure the tools "
                  'return or remove it. Reply with the complete corrected answer only, in the same language and format, '
                  'without mentioning this check.')
        self._emit('self_check', figures=bad)
        try:
            answer2 = self._ask_backend(prompt)
            v2 = VF.verify_answer(answer2, self.trace, self.tb.eng.keys)
        except Exception as ex:
            self.last_self_check = {'applied': False, 'unsupported_before': bad, 'error': str(ex)}
            return answer, v
        better = bool(answer2.strip()) and v2['summary']['not_found'] < v['summary']['not_found']
        self.last_self_check = {'applied': better, 'unsupported_before': bad,
                                'unsupported_after': v2['summary']['not_found_values'][:10] if better else bad}
        return (answer2, v2) if better else (answer, v)

    def ask(self, question: str) -> str:
        """Answer a question; afterwards self.last_verification holds the check of every figure against the
        tool results (see ubahn_verify) and self.last_self_check what the correction round did, if it ran."""
        self.trace = []
        self.last_self_check = None
        t_start = time.time()
        answer = self._ask_backend(question)
        try:
            self.last_verification = VF.verify_answer(answer, self.trace, self.tb.eng.keys)
        except Exception as ex:                          # the check must never break an answer
            self.last_verification = {'figures': [], 'summary': {'checked': 0, 'verified': 0, 'calculated': 0,
                                                                 'not_found': 0, 'not_found_values': [], 'error': str(ex)}}
        if self.deadline is not None and self._remaining() < SELF_CHECK_MIN_S:
            self.last_self_check = {'applied': False, 'skipped': 'not enough time left in the answer budget'}
        elif self.self_check and self.last_verification['summary'].get('not_found', 0) > 0:
            answer, self.last_verification = self._repair(answer, self.last_verification)
        self._audit(question, answer, time.time() - t_start)
        return answer

    def _ask_responses(self, question):
        """Responses API loop (Azure OpenAI or OpenAI): function calls come back as output items; results are
        sent as function_call_output items chained with previous_response_id. strict=False because the tool
        schemas have optional parameters."""
        tools = [{'type': 'function', 'name': s['name'], 'description': s['description'], 'parameters': s['parameters'],
                  'strict': False} for s in self.tb.schemas()]
        convo = self.history + [{'role': 'user', 'content': question}]
        new_input, prev = convo, None
        for _ in range(MAX_STEPS):
            kw = {'model': self.model, 'instructions': self.system_prompt, 'tools': tools, 'input': new_input}
            if prev:
                kw['previous_response_id'] = prev
            client = self.client
            if self.deadline is not None:
                left = self._remaining()
                if left < MIN_CALL_S:
                    raise AgentTimeout('answer budget used up')
                if self.trace and left < FINAL_CALL_S:
                    kw['tool_choice'] = 'none'           # last round: write the answer from the results so far
                client = self.client.with_options(timeout=left)
            self._emit('model', steps_done=len(self.trace))
            resp = client.responses.create(**kw)
            calls = [it for it in resp.output if getattr(it, 'type', None) == 'function_call']
            if not calls:
                text = resp.output_text or ''
                self.history = convo + [{'role': 'assistant', 'content': text}]
                return text
            outputs = []
            for c in calls:
                try:
                    args = json.loads(c.arguments or '{}')
                except json.JSONDecodeError:
                    args = {}
                text, err = self._tool(c.name, args)
                outputs.append({'type': 'function_call_output', 'call_id': c.call_id, 'output': text})
            new_input, prev = outputs, resp.id
        self.history = convo
        return 'Stopped: too many tool calls for one question.'

    def _ask_json(self, question):
        """Tool use without native function calling: the model replies with {"tool": ..., "arguments": {...}}
        or with {"answer": "..."}; anything that is not valid JSON is taken as the final answer."""
        catalog = '\n'.join(f"- {s['name']}: {s['description']} Arguments (JSON schema): {json.dumps(s['parameters'], ensure_ascii=False)}"
                            for s in self.tb.schemas())
        system = (self.system_prompt + '\n\nTOOLS\n' + catalog + '\n\nTo call a tool, reply with ONLY a JSON object '
                  '{"tool": "<name>", "arguments": {...}}. When you have what you need, reply with ONLY '
                  '{"answer": "<your answer in Markdown>"}. One tool call per reply.')
        msgs = [{'role': 'system', 'content': system}] + self.history + [{'role': 'user', 'content': question}]
        for _ in range(MAX_STEPS):
            self._emit('model', steps_done=len(self.trace))
            resp = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0)
            content = resp.choices[0].message.content or ''
            m = re.search(r'\{.*\}', content, re.S)
            try:
                req = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                req = None
            if not isinstance(req, dict) or 'tool' not in req:
                answer = req.get('answer', content) if isinstance(req, dict) else content
                msgs.append({'role': 'assistant', 'content': content})
                self.history = msgs[1:]
                return answer
            text, err = self._tool(req['tool'], req.get('arguments') or {})
            msgs.append({'role': 'assistant', 'content': content})
            msgs.append({'role': 'user', 'content': f'Tool result for {req["tool"]}:\n{text}'})
        self.history = msgs[1:]
        return 'Stopped: too many tool calls for one question.'

    def _ask_anthropic(self, question):
        tools = [{'name': s['name'], 'description': s['description'], 'input_schema': s['parameters']} for s in self.tb.schemas()]
        msgs = self.history + [{'role': 'user', 'content': question}]
        for _ in range(MAX_STEPS):
            # current Claude models take no sampling parameters (no temperature): the tools keep the numbers fixed
            self._emit('model', steps_done=len(self.trace))
            resp = self.client.messages.create(model=self.model, max_tokens=4096, system=self.system_prompt,
                                               tools=tools, messages=msgs)
            msgs.append({'role': 'assistant', 'content': resp.content})
            if resp.stop_reason != 'tool_use':
                self.history = msgs
                return ''.join(b.text for b in resp.content if b.type == 'text')
            results = []
            for b in resp.content:
                if b.type == 'tool_use':
                    text, err = self._tool(b.name, b.input)
                    results.append({'type': 'tool_result', 'tool_use_id': b.id, 'content': text, 'is_error': err})
            msgs.append({'role': 'user', 'content': results})
        self.history = msgs
        return 'Stopped: too many tool calls for one question.'

    def _ask_openai(self, question):
        tools = [{'type': 'function', 'function': s} for s in self.tb.schemas()]
        msgs = [{'role': 'system', 'content': self.system_prompt}] + self.history + [{'role': 'user', 'content': question}]
        for _ in range(MAX_STEPS):
            self._emit('model', steps_done=len(self.trace))
            resp = self.client.chat.completions.create(model=self.model, messages=msgs, tools=tools, temperature=0)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                msgs.append({'role': 'assistant', 'content': msg.content or ''})
                self.history = msgs[1:]
                return msg.content or ''
            msgs.append({'role': 'assistant', 'content': msg.content or '',
                         'tool_calls': [{'id': tc.id, 'type': 'function',
                                         'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                                        for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or '{}')
                except json.JSONDecodeError:
                    args = {}
                text, err = self._tool(tc.function.name, args)
                msgs.append({'role': 'tool', 'tool_call_id': tc.id, 'content': text})
        self.history = msgs[1:]
        return 'Stopped: too many tool calls for one question.'


# ------------------------------------------------------------------------------ self-test
SELFTEST = [   # the tool calls behind the training questions, with the figures given in the chat
    ('Q1 Guns N Roses 23 June', 'ubahn_event_report', {'query': 'Guns', 'date': '2026-06-23'},
     lambda r: f"departure extra {r['departure_window']['extra_passengers']:.0f} (chat: ~1,190), share {r['departure_window']['extra_as_share_of_attendance']:.0%}"),
    ('Q2 bad weather 20-26 July', 'ubahn_weather_peaks', {'start': '2026-07-20', 'end': '2026-07-26'},
     lambda r: f"peak {r['network_peak']['slot']} {r['network_peak']['passengers_15min']:.0f} vs {r['network_peak']['typical_dry_same_weekday']} (chat: 65,448 vs ~43,000)"),
    ('Q3 U6 suspension', 'ubahn_closure_report', {'line': 'U6', 'station_a': 'Hallesches Tor', 'station_b': 'Kaiserin-Augusta-Str.'},
     lambda r: f"{r['closure']['reason']}, {r['closure']['duration']} on {r['closure']['start']} (chat: safety inspection, 1 h 30, 13 July 13:50)"),
    ('Q4 Rudow peak', 'ubahn_peak_time', {'station': 'Rudow'},
     lambda r: f"{r['definitions']['weekday_mean']['peak_time']} {r['definitions']['weekday_mean']['peak_value']:.0f} vs mean {r['definitions']['weekday_mean']['mean_of_all_station_peaks']:.0f} (chat: 18:30, 245 vs 297)"),
    ('Q5 energy', 'ubahn_energy_efficiency', {},
     lambda r: f"worst {r['worst_line']} {r['by_line'][r['worst_line']]['kwh_per_passenger']:.2f} kWh (chat: U5 0.55)"),
    ('Q6 fragmentation', 'ubahn_fragmentation_ranking', {'top': 5},
     lambda r: 'top: ' + ', '.join(x['station'] for x in r['ranking']) + ' (chat: Alexanderplatz, Bismarckstr., Schillingstr., Strausberger Platz, Weberwiese)'),
    ('Q7 anomalies 24 June', 'ubahn_anomaly_scan', {'date': '2026-06-24'},
     lambda r: 'departure extras: ' + ', '.join(f"{x['event'][:14]} {x['extra']:+.0f}" for x in r['event_checks'] if x['phase'] == 'departure') + ' (chat: OXIS ~+940 incl. Schillingstr., Lachkater +391)'),
    ('Q7 ranked findings', 'ubahn_anomaly_scan', {'date': '2026-06-24'},
     lambda r: ' | '.join(f"{f['type']}: {f['most_likely_cause'][:32]}" for f in r['ranked_findings'][:3]) + ' (chat: OXIS, Lachkater, rain)'),
    ('Q1 recorded facts', 'ubahn_event_report', {'query': 'Guns', 'date': '2026-06-23'},
     lambda r: r['facts_from_recorded_data'][5][:95] + ' (chat: U3 suspension that evening)'),
    ('Q7 stations in the totals', 'ubahn_anomaly_scan', {'date': '2026-06-24'},
     lambda r: next(f"{x['event'][:10]} {x['phase']}: {len(x['stations'])} stations listed, incl. Samariterstr.: {'Samariterstr.' in x['stations']}"
                    for x in r['event_checks'] if 'OXIS' in x['event'] and x['phase'] == 'departure')),
    ('19 July distinct anomalies', 'ubahn_anomaly_scan', {'date': '2026-07-19'},
     lambda r: ' | '.join(f"{f['type']}: {', '.join(f['where'][:2]) if isinstance(f['where'], list) else f['where']}" for f in r['ranked_findings'][:4])),
    ('InnoTrans in the events data?', 'ubahn_list_events', {'date_from': '2026-09-22', 'date_to': '2026-09-25', 'venue': 'Messe'},
     lambda r: f"{r['events_found']} listing(s) at the Messe 22-25 Sept (0 on training data, 4 once the evaluation files are loaded)"),
    ('Adenauerplatz near-tie check', 'ubahn_peak_time', {'station': 'Adenauerplatz'},
     lambda r: f"{r['comparison_with_mean_of_all_station_peaks']['verdict']} ({r['comparison_with_mean_of_all_station_peaks']['difference_pct']:+.1f} %), {r['period_used']}"),
    ('Q8 dependencies', 'ubahn_station_dependencies', {'top': 5},
     lambda r: ', '.join(f"{p['pair'][0]}/{p['pair'][1]} {p['correlation']:.2f}" for p in r['pairs'][:3]) + ' (chat: Schönleinstr./Südstern 0.24 ...)'),
    ('Q9 disruption behaviour', 'ubahn_disruption_response', {},
     lambda r: f"adjacent change during closures {r['pooled']['adjacent']['change']:+.0%}, alternatives during suspensions {r['pooled']['susp_alternatives']['change']:+.0%} (chat: no gain)"),
    ('Q10 investment', 'ubahn_best_new_link', {},
     lambda r: f"best link {r['ranking'][0]['link']} -{r['ranking'][0]['exposure_reduction']:.0%} (chat: Warschauer Str.-Frankfurter Tor, -14%)"),
    ('Q11 InnoTrans diversion', 'ubahn_diversion_scenario',
     {'venue': 'Messedamm 22', 'start': '2026-09-22 09:00', 'end': '2026-09-22 18:00', 'attendance': 5000,
      'alternative_stations': ['Wilmersdorfer Str.', 'Adenauerplatz'], 'tmean': 13, 'prcp': 1.5},
     lambda r: f"Theodor-Heuss-Platz {r['peak_demand_over_ceiling_exit_wave']['0%_diverted']['Theodor-Heuss-Platz']} -> {r['peak_demand_over_ceiling_exit_wave']['40%_diverted']['Theodor-Heuss-Platz']} (chat: 1.45 -> 1.15)"),
    ('Q11 capacity-limited diversion', 'ubahn_diversion_scenario',
     {'venue': 'Messedamm 22', 'start': '2026-09-23 09:00', 'end': '2026-09-23 18:00', 'attendance': 40000, 'tmean': 16, 'prcp': 0},
     lambda r: 'capacity-limited share {:.0%} (Theodor-Heuss-Platz still {} x ceiling); access: '.format(
               r['capacity_limited_share']['share'], r['capacity_limited_share']['ratios']['Theodor-Heuss-Platz'])
               + ', '.join(k + ' ' + str(v['walk_minutes_at_4_5_kmh']) + ' min' for k, v in r['access'].items())),
    ('Q11 other corridor (auto)', 'ubahn_diversion_scenario',
     {'venue': 'Messedamm 22', 'start': '2026-09-22 09:00', 'end': '2026-09-22 18:00', 'attendance': 40000, 'tmean': 13, 'prcp': 1.5},
     lambda r: f"picked {r['alternative_stations']} ({'/'.join(r['venue_lines'])} avoided); THP {r['peak_demand_over_ceiling_exit_wave']['0%_diverted']['Theodor-Heuss-Platz']} -> {r['peak_demand_over_ceiling_exit_wave']['40%_diverted']['Theodor-Heuss-Platz']} (chat: U7, 6.7 -> 4.3)"),
    ('Example premise check', 'ubahn_check_line_section', {'line': 'U8', 'station_a': 'Hermannplatz', 'station_b': 'Neukölln'},
     lambda r: r.get('error', 'no error')[:60]),
    ('Action plan 23 June (replay)', 'ubahn_action_plan', {'date': '2026-06-23'},
     lambda r: '; '.join(f"{w['station']} {w['window']}: {w['replay_on_recorded_data']['would_have_fired_at']}" for w in r['watch'][:3])[:170]),
    ('Staff plan InnoTrans (scenario)', 'ubahn_staff_plan',
     {'date': '2026-09-22', 'staff': 12, 'tmean': 13, 'prcp': 1.5,
      'extra_events': [{'venue': 'Messedamm 22', 'start': '2026-09-22 09:00', 'end': '2026-09-22 18:00', 'attendance': 40000}]},
     lambda r: '; '.join(f"{x['staff']} at {x['station']} {x['from']}-{x['to']}" for x in r['roster'])),
    ('Daily brief 23 June', 'ubahn_daily_brief', {'date': '2026-06-23', 'staff': 8}, lambda r: r['headlines'][2][:120]),
    ('Snapshot 23 June 22:30', 'ubahn_snapshot', {'at': '2026-06-23 22:30'}, lambda r: r['facts'][1]),
    ('Forecast ranges calibration', 'ubahn_forecast', {'start': '2026-06-23 21:30', 'end': '2026-06-23 22:00', 'stations': ['Warschauer Str.']},
     lambda r: r['calibration'] + f" | 21:30 expected {r['slots'][0]['expected']} range {r['slots'][0]['range_80']} recorded {r['slots'][0].get('recorded')}"),
    ('Energy U5 August', 'ubahn_energy_stats', {'lines': ['U5'], 'date_from': '2026-08-01', 'date_to': '2026-08-31'},
     lambda r: f"U5 {r['by_line']['U5']['total_mwh']:.0f} MWh, {r['by_line']['U5']['kwh_per_passenger']:.3f} kWh/pax, highest {r['by_line']['U5']['highest_day']}"),
    ('Passenger messages U6 13 July', 'ubahn_passenger_messages', {'closure': 'U6 suspended', 'date': '2026-07-13'},
     lambda r: r['messages']['de']['display'][:110]),
    ('Reinforcement InnoTrans 5k', 'ubahn_reinforcement_plan',
     {'date': '2026-09-22', 'tmean': 16, 'prcp': 0,
      'extra_events': [{'venue': 'Messedamm 22', 'start': '2026-09-22 09:00', 'end': '2026-09-22 18:00', 'attendance': 5000}]},
     lambda r: ' | '.join(r['line_reinforcement'][:2]) + ' | ' + next((m['action'] for w in r['windows'] for m in w['measures'] if m['mode'] == 'Bus'), 'no bus')),
    ('Reinforcement normal day', 'ubahn_reinforcement_plan', {'date': '2026-08-12'}, lambda r: r['summary'][0]),
    ('Compare InnoTrans scenarios', 'ubahn_compare_scenarios',
     {'date': '2026-09-22', 'scenario_a': {'label': 'no InnoTrans', 'tmean': 16, 'prcp': 0},
      'scenario_b': {'label': 'InnoTrans 40k', 'tmean': 16, 'prcp': 0,
                     'extra_events': [{'venue': 'Messedamm 22', 'start': '2026-09-22 09:00', 'end': '2026-09-22 18:00', 'attendance': 40000}]}},
     lambda r: r['headlines'][-1][:120]),
    ('Contingency Alexanderplatz', 'ubahn_station_contingency', {'station': 'Alexanderplatz'},
     lambda r: f"tracks out: {r['if_tracks_out']['stations_cut_off']} stations cut off, {r['if_tracks_out']['passengers_affected_weekday']:,} weekday passengers (chat: 25, ~202k)"),
    ('Incident report 23 June', 'ubahn_incident_report', {'date': '2026-06-23'},
     lambda r: f"{len(r['lessons_learned'])} lessons; " + next((l for l in r['lessons_learned'] if '1,192' in l), 'no 1,192 lesson')[:100]),
    ('Value report (summer replay)', 'ubahn_value_report', {},
     lambda r: f"planning flagged {r['planning']['share_flagged_in_advance']:.0%} of event overloads; best alert rule {r['early_warning']['recommended_rule']}; reinforcement absorbed {r['reinforcement']['share_absorbed']:.0%}"),
    ('Reliability', 'ubahn_reliability', {},
     lambda r: f"ranges {r['forecast_ranges_calibration']['80 % range holds']:.0%}/{r['forecast_ranges_calibration']['90 % range holds']:.0%}, "
               f"daily totals corr {r['backtest']['daily_network_totals']['correlation']}, MAE {r['backtest']['daily_network_totals']['mean_absolute_error_pct']}%"),
    ('Spandau 1 Sep 00:00-12:00', 'ubahn_flow_stats', {'stations': ['Rathaus Spandau'], 'start': '2026-09-01 00:00', 'end': '2026-09-01 12:00'},
     lambda r: f"total {r['total']:.0f} (chat: 2,828)"),
    ('"Spandau" name check', 'ubahn_station_info', {'station': 'Spandau'}, lambda r: r.get('error', 'no error')[:80]),
    ('Weberwiese by hour', 'ubahn_flow_stats', {'stations': ['Weberwiese'], 'by': 'hour'},
     lambda r: f"17:00 {r['by_hour']['17:00']:.0f}, day {r['window_total']:.0f} (chat: 762, ~7,000)"),
    ('Weberwiese 16:30-18:30 mean', 'ubahn_flow_stats', {'stations': ['Weberwiese'], 'time_from': '16:30', 'time_to': '18:30'},
     lambda r: f"{r['window_total']:.0f} (chat: ~1,410)"),
    ('Weberwiese 23 Jun 16:30-18:30', 'ubahn_compare_to_normal', {'stations': ['Weberwiese'], 'start': '2026-06-23 16:30', 'end': '2026-06-23 18:30'},
     lambda r: f"{r['observed']:.0f} vs model {r['model_expected']:.0f}, higher on {r['share_of_other_weeks_with_higher_flow']:.0%} of other Tuesdays (chat: 1,865 vs ~1,540)"),
    ('Events 23 June', 'ubahn_list_events', {'date_from': '2026-06-23', 'date_to': '2026-06-23'},
     lambda r: f"{r['events_found']}: {r['events'][0]['names'][0]} {r['events'][0]['attendance']:.0f} -> {list(r['events'][0]['stations'])[:2]}"),
    ('Closures 13 July', 'ubahn_list_closures', {'date_from': '2026-07-13', 'date_to': '2026-07-13'},
     lambda r: '; '.join(f"{c['kind']} {c['line'] or ''} {c['reason']} {c['duration']}" for c in r['closures'])),
    ('Peak slot 21 July 18:15', 'ubahn_rank_stations', {'metric': 'peak_slot', 'start': '2026-07-21 18:15', 'end': '2026-07-21 18:30', 'top': 1},
     lambda r: f"{r['ranking'][0]['station']} {r['ranking'][0]['peak_slot']:.0f} (chat: Berliner Str. 2,448)"),
    ('Weather 21 July', 'ubahn_weather', {'date_from': '2026-07-21'},
     lambda r: ', '.join(f"{k}: {v['rain_mm']} mm, max {v['max_rain_mm_h']} mm/h" for k, v in r['rows'].items())),
]


def read_audit(path: str) -> list:
    """Entries of an audit log (malformed lines are skipped)."""
    out = []
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


SLOW = {'Q10 investment', 'Q8 dependencies', 'Q6 fragmentation', 'Value report (summer replay)'}
EXPECTED_ERRORS = {'"Spandau" name check'}


def run_selftest(tb: Toolbox, quick: bool = False) -> list:
    """Run the self-test and return one row per check (label, tool, seconds, chars, result, error)."""
    rows = []
    text, _ = tb.call('ubahn_event_report', {'query': 'Guns', 'date': '2026-06-23'})
    trace = [{'tool': 'ubahn_event_report', 'args': {'query': 'Guns', 'date': '2026-06-23'}, 'output': text}]
    sample = ('After the show (1,996 attendees) Warschauer Str. peaked at 250 at 22:30 against about 19 normally; '
              'the departure wave added 1,192 passengers (60% of attendance). Schlesisches Tor carried 3,210 passengers.')
    v = VF.verify_answer(sample, trace, tb.eng.keys)
    rows.append({'label': 'Figure check (sample)', 'tool': 'ubahn_verify', 'seconds': 0.0, 'chars': 0,
                 'result': VF.summary_line(v) + ' (expected: 3,210 not found)', 'error': v['summary']['not_found_values'] != ['3,210']})
    for label, name, args, summary in SELFTEST:
        if quick and label in SLOW:
            continue
        t0 = time.time()
        text, err = tb.call(name, args)
        dt = time.time() - t0
        try:
            res = summary(json.loads(text)) if not err else text[:150]
        except Exception as ex:
            res, err = f'summary failed: {ex}', True
        if label in EXPECTED_ERRORS and err:
            res, err = f'expected error: {text[:140]}', False
        rows.append({'label': label, 'tool': name, 'seconds': round(dt, 1), 'chars': len(text), 'result': str(res), 'error': bool(err)})
    return rows


def selftest(tb: Toolbox, quick: bool = False):
    for r in run_selftest(tb, quick):
        print(f"{r['label']:28s} {r['tool']:30s} {r['seconds']:4.1f}s {r['chars']:7d} chars  {'ERROR ' if r['error'] else ''}{r['result']}")


def read_questions(path):
    """Questions from a Markdown/text file: numbered items ('1. ...') or one question per paragraph."""
    text = open(path, encoding='utf-8').read()
    text = re.sub(r'```[a-z]*', '', text)
    items = re.split(r'\n\s*(?=\d+[.)]\s)', '\n' + text)
    qs = [re.sub(r'^\d+[.)]\s*', '', it.strip()).strip() for it in items if re.match(r'\s*\d+[.)]\s', it)]
    qs = [re.sub(r'\n\s*#+[^\n]*', '', q).strip() for q in qs]           # e.g. a trailing '## Bonus Questions' heading
    if not qs:
        qs = [p.strip() for p in text.split('\n\n') if p.strip() and not p.strip().startswith('#')]
    return qs


def _figure_values(text):
    return {round(f['values'][0], 3) if f['kind'] == 'number' else f['time'] for f in VF.extract_figures(text or '')}


def consistency(answers):
    """Compare the figures of several answers to the same question."""
    sets = [_figure_values(a) for a in answers]
    common = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    return {'runs': len(answers), 'figures_in_common': len(common), 'figures_in_any': len(union),
            'share_identical': round(len(common) / len(union), 3) if union else 1.0,
            'differing': sorted(map(str, union - common))[:12]}


def run_batch(agent, questions, out_path, repeat: int = 1):
    """Answer each question on its own (repeat > 1: several independent runs, the answer with the fewest
    unsupported figures is kept and the figures of all runs are compared)."""
    lines = [f'# Answers ({time.strftime("%Y-%m-%d %H:%M")}, model {agent.model})', '']
    for i, q in enumerate(questions, 1):
        runs = []
        for r in range(max(1, int(repeat))):
            agent.history = []                          # each run answered on its own
            t = time.time()
            ans = agent.ask(q)
            runs.append((ans, agent.last_verification, list(agent.trace), time.time() - t, agent.last_self_check))
        best = min(range(len(runs)), key=lambda k: (runs[k][1]['summary'].get('not_found', 0), k))
        ans, ver, trace, dt, sc = runs[best]
        lines += [f'## {i}. {q}', '', ans, '', f'_{VF.summary_line(ver)}_', '']
        if sc and sc.get('applied'):
            lines += [f"_Figure check corrected: {', '.join(sc['unsupported_before'])}_", '']
        if len(runs) > 1:
            c = consistency([x[0] for x in runs])
            lines += [f"_Consistency over {c['runs']} runs: {c['figures_in_common']} of {c['figures_in_any']} figures identical"
                      + (f"; differing: {', '.join(c['differing'])}" if c['differing'] else '') + f"; run {best + 1} kept_", '']
        lines += ['<details><summary>Tools used</summary>', '']
        lines += [f"- `{c['tool']}` {json.dumps(c['args'], ensure_ascii=False)}{' (error)' if c['error'] else ''}" for c in trace]
        lines += ['', f'</details>', '', f'_{dt:.0f} s_', '']
        print(f'{i}/{len(questions)} answered ({len(trace)} tool calls, {len(runs)} run(s))', file=sys.stderr)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('question', nargs='*')
    ap.add_argument('--data', default=os.environ.get('DATA_DIR', str(DATA_ROOT)))
    ap.add_argument('--backend', choices=['anthropic', 'openai', 'json', 'azure', 'azure_responses'], default=DEFAULT_BACKEND)
    ap.add_argument('--model')
    ap.add_argument('--show-tools', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--quick', action='store_true', help='with --selftest: skip the three slow network checks')
    ap.add_argument('--audit', default=os.environ.get('UBAHN_AUDIT', 'audit_log.jsonl'),
                    help='audit log (one JSON line per answer); default audit_log.jsonl')
    ap.add_argument('--no-audit', action='store_true', help='do not write the audit log')
    ap.add_argument('--repeat', type=int, default=1, help='with --batch: answer each question N times and compare the figures')
    ap.add_argument('--style', choices=['concise', 'detailed'], default=None, help='answer format (default concise, or UBAHN_ANSWER_STYLE)')
    ap.add_argument('--no-self-check', action='store_true', help='do not let the model correct figures found in no tool result')
    ap.add_argument('--batch', help='file with numbered questions to answer in one go')
    ap.add_argument('--out', default='answers.md')
    a = ap.parse_args()
    eng = UBahnEngine(a.data)
    if a.selftest:
        selftest(Toolbox(eng), a.quick); return
    agent = Agent(eng, a.backend, a.model, a.show_tools, audit_path=None if a.no_audit else a.audit,
                  style=a.style, self_check=False if a.no_self_check else None)
    if a.batch:
        run_batch(agent, read_questions(a.batch), a.out, a.repeat)
        print(f'answers written to {a.out}'); return
    if a.question:
        print(agent.ask(' '.join(a.question)))
        print('\n-- ' + VF.summary_line(agent.last_verification)); return
    print('U-Bahn operations assistant - ask a question, empty line to quit.')
    while True:
        try:
            q = input('\n> ').strip()
        except EOFError:
            break
        if not q:
            break
        print('\n' + agent.ask(q))
        print('\n-- ' + VF.summary_line(agent.last_verification))


# ------------------------------------------------------------------------------ server integration (merged project)
# src/server.py talks to the agent only through the functions below. The engine is fitted once per process
# (about 5 s) and shared by all requests; every question gets its own Agent, so conversations do not mix.
AGENT_TIMEOUT_S = float(os.environ.get('AGENT_TIMEOUT_S', '26'))
_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='llm-agent')


def data_dir() -> str:
    """DATA_DIR from .env (relative paths are taken from the repo root), else the loader's data/ folder."""
    d = os.environ.get('DATA_DIR')
    if not d:
        return str(DATA_ROOT)
    return d if os.path.isabs(d) else str((REPO_ROOT / d).resolve())


def get_engine() -> UBahnEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = UBahnEngine(data_dir())
        return _ENGINE


def reload_engine() -> dict:
    """Refit on the current data folder (after new CSV files were dropped in)."""
    global _ENGINE
    eng = UBahnEngine(data_dir())
    for f in (A._daily, A._line_graph, A._residual_z, A._vulnerability):
        f.cache_clear()
    with _ENGINE_LOCK:
        _ENGINE = eng
    return eng.data_status()


def default_date(eng: UBahnEngine | None = None) -> str:
    """Today if the data covers it (InnoTrans week), else the last recorded service day."""
    eng = eng or get_engine()
    last = (eng.flows.index.max() - pd.Timedelta('5h')).normalize()
    today = pd.Timestamp.now().normalize()
    return str((today if eng.flows.index.min().normalize() <= today <= last else last).date())


def run_tool(name: str, args: dict) -> dict:
    """Run one tool without the language model; returns the JSON-ready result (not truncated like Toolbox.call,
    which cuts outputs to MAX_TOOL_CHARS for the model)."""
    fn = Toolbox(get_engine()).by_name.get(name)
    if fn is None:
        raise ValueError(f'unknown tool {name}')
    return json.loads(json.dumps(to_jsonable(fn(**(args or {}))), ensure_ascii=False, default=str))


# -- Markdown for tool results (operator endpoints and the no-LLM fallback) --------------------------------------
def _fmt_val(v):
    if isinstance(v, float):
        return f'{v:,.2f}'.rstrip('0').rstrip('.') if abs(v) < 100 else f'{v:,.0f}'
    if isinstance(v, int) and not isinstance(v, bool):
        return f'{v:,}'
    if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
        return ', '.join(map(str, v))
    return str(v)


def _row(d: dict, keys=None, limit=6) -> str:
    items = [(k, d[k]) for k in (keys or d) if k in d and not isinstance(d[k], (dict,)) and d[k] not in (None, '', [])]
    items = [(k, v) for k, v in items if not (isinstance(v, list) and v and isinstance(v[0], (dict, list)))]
    return '; '.join(f"{k.replace('_', ' ')}: {_fmt_val(v)}" for k, v in items[:limit])


def md_generic(r) -> list:
    """Any tool result: headline lists first, then scalars, then the first rows of each table."""
    if isinstance(r, list):
        r = {'rows': r}
    if not isinstance(r, dict):
        return [str(r)[:400]]
    out = []
    for k in ('headlines', 'facts_from_recorded_data', 'facts', 'summary', 'lessons_learned', 'line_reinforcement'):
        v = r.get(k)
        if isinstance(v, list) and v and isinstance(v[0], str):
            out += [f'- {x}' for x in v[:8]]
    scal = _row({k: v for k, v in r.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)}, limit=8)
    if scal:
        out.append(scal)
    for k, v in r.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out.append(f"_{k.replace('_', ' ')}:_")
            out += [f'- {_row(x)}' for x in v[:5]]
    return out


def md_brief(r) -> list:
    out = [f'- {h}' for h in r.get('headlines', [])]
    for k in ('notes',):
        out += [f'_Note: {n}_' for n in r.get(k, [])[:3]]
    return out or md_generic(r)


def md_action_plan(r) -> list:
    out = [f"**{r.get('weekday', '')} {str(r.get('date', ''))[:10]}** – expected network passengers "
           f"{_fmt_val(r.get('network_expected_passengers'))} (normal {_fmt_val(r.get('network_normal_passengers'))})"]
    for w in r.get('watch', [])[:6]:
        out.append(f"- **{w['station']}** ({', '.join(w.get('lines', []))}) {w['window']}: {w.get('reason', '')}")
        for t in w.get('triggers', [])[:3]:
            out.append(f"  - {t.get('id', '')}: if {t.get('if', '')} → {t.get('then', '')}")
        rp = w.get('replay_on_recorded_data') or {}
        if rp.get('would_have_fired_at'):
            out.append(f"  - replay on recorded data: {_fmt_val(rp['would_have_fired_at'])}")
    return out


def md_staff(r) -> list:
    out = [f"{r.get('staff_assigned')} of {r.get('staff_available')} additional staff assigned "
           f"({r.get('block_hours')} h blocks), {r.get('weekday', '')} {str(r.get('date', ''))[:10]}:", '',
           '| Station | Lines | From | To | Staff | Reason |', '|---|---|---|---|---|---|']
    for x in r.get('roster', []):
        out.append(f"| {x['station']} | {', '.join(x.get('lines', []))} | {x['from']} | {x['to']} | {x['staff']} | {x.get('reason', '')} |")
    return out


def md_forecast(r) -> list:
    out = [f"{r.get('scope', '')}, {' – '.join(r.get('window', []))}. _{r.get('calibration', '')}_", '',
           '| Station | Slot | Expected | 80 % range | Chance at ceiling | Recorded |', '|---|---|---|---|---|---|']
    for s in r.get('slots', [])[:24]:
        rng = s.get('range_80') or ['', '']
        out.append(f"| {s['station']} | {s['slot'][11:]} | {s.get('expected')} | {rng[0]}–{rng[1]} | "
                   f"{s.get('chance_at_ceiling', 0):.0%} | {s.get('recorded', '–')} |")
    return out


def md_messages(r) -> list:
    out = [f"**{r.get('situation', '')}** ({r.get('start', '')} – {r.get('end', '')}), stations: "
           f"{', '.join(r.get('stations', []))}; alternative: {', '.join(r.get('alternative', []) or ['–'])}"]
    for lang, msgs in (r.get('messages') or {}).items():
        out.append(f'\n**{lang.upper()}**')
        out += [f'- _{kind}_: {text}' for kind, text in msgs.items()] if isinstance(msgs, dict) else [f'- {msgs}']
    return out


MARKDOWN = {'ubahn_daily_brief': md_brief, 'ubahn_action_plan': md_action_plan, 'ubahn_staff_plan': md_staff,
            'ubahn_forecast': md_forecast, 'ubahn_passenger_messages': md_messages}


def to_markdown(name: str, result) -> str:
    try:
        return '\n'.join(MARKDOWN.get(name, md_generic)(result))
    except (KeyError, TypeError, ValueError, AttributeError):
        return '\n'.join(md_generic(result))


# -- answering with a time budget -------------------------------------------------------------------------------
def _title(name: str) -> str:
    return name.replace('ubahn_', '').replace('_', ' ').capitalize()


def fallback_answer(question: str, trace: list, reason: str) -> tuple[str, str]:
    """No LLM answer in time: A's src/fallback.py writes the answer from the tool results gathered so far. With no
    tool result at all, A's keyword router (src/legacy_agent.py) picks and runs its own tools, still without LLM.
    Returns (answer, source)."""
    import src.fallback as FB
    results, failures = [], []
    for c in trace:
        try:
            res = json.loads(c['output'])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        FB.FORMATTERS.setdefault(c['tool'], MARKDOWN.get(c['tool'], md_generic))
        FB.TITLES.setdefault(c['tool'], _title(c['tool']))
        (failures if c.get('error') else results).append({'name': c['tool'], 'result': res if isinstance(res, dict) else {'rows': res}})
    if results:
        return FB.build_fallback_answer({'results': results, 'failures': failures}, [], reason), 'fallback_tool_results'
    from src.legacy_agent import TrainAgent

    def _no_llm(_prompt):
        raise RuntimeError('LLM budget used up')
    return TrainAgent(llm=_no_llm).answer(question).get('answer', ''), 'fallback_keyword_router'


def answer(question: str, timeout: float | None = None) -> dict:
    """Answer one operator question for the web UI: B's tool-calling agent with a total budget of `timeout`
    seconds (default AGENT_TIMEOUT_S = 26), then A's deterministic fallback. Never raises."""
    timeout = AGENT_TIMEOUT_S if timeout is None else timeout
    t0 = time.time()
    try:
        agent = Agent(get_engine(), show_tools=True, audit_path=os.environ.get('AUDIT_LOG', str(REPO_ROOT / 'audit_log.jsonl')))
    except Exception as ex:                                   # no credentials, no data ...
        text, source = fallback_answer(question, [], f'{type(ex).__name__}: {ex}')
        return {'question': question, 'answer': text, 'mode': source, 'tools_used': [], 'confidence': 'low',
                'note': f'{type(ex).__name__}: {ex}', 'seconds': round(time.time() - t0, 1)}
    agent.deadline = t0 + timeout
    fut = _POOL.submit(agent.ask, question)
    text, note = None, None
    try:
        text = fut.result(timeout=max(0.1, agent.deadline - time.time()))
        if not text.strip() or text.startswith('Stopped:'):
            note, text = text or 'empty answer', None
    except FuturesTimeout:
        note = f'LLM answer not ready after {timeout:.0f} s'
    except Exception as ex:
        note = f'{type(ex).__name__}: {ex}'
    trace = list(agent.trace)                                 # the worker may still append after a timeout
    tools = list(dict.fromkeys(c['tool'] for c in trace))
    calls = [{'tool': c['tool'], 'args': c['args'], 'seconds': round(c.get('seconds', 0), 2), 'error': c['error']} for c in trace]
    if text is not None:
        summ = (agent.last_verification or {}).get('summary', {})
        conf = 'high' if trace and summ.get('not_found', 0) == 0 else ('medium' if trace else 'low')
        return {'question': question, 'answer': text, 'mode': 'llm_tools', 'tools_used': tools, 'tool_calls': calls,
                'confidence': conf, 'verification': summ, 'figures_line': VF.summary_line(agent.last_verification),
                'self_check': agent.last_self_check, 'seconds': round(time.time() - t0, 1)}
    print(f'[agent] fallback: {note}', file=sys.stderr, flush=True)
    try:
        text, source = fallback_answer(question, trace, note)
    except Exception as ex:
        text, source = f'No answer could be computed ({type(ex).__name__}: {ex}).', 'error'
    return {'question': question, 'answer': text, 'mode': source, 'tools_used': tools, 'tool_calls': calls,
            'confidence': 'medium' if source == 'fallback_tool_results' else 'low', 'note': note,
            'seconds': round(time.time() - t0, 1)}


if __name__ == '__main__':
    main()
