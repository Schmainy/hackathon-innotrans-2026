"""
app.py - operator web app for the U-Bahn agent (Streamlit), Track Knack design.

    streamlit run app.py            # reads .env: AZURE_ENDPOINT, AZURE_API_KEY, MODEL_NAME, DATA_DIR

The backend team's Streamlit app (ubahn_solution v0.9.4) with the frontend team's light theme:
white background, light-gray sidebar, blue accent, Material Symbols instead of emojis, a persistent header and
ALSTOM footer, and a Chat tab laid out like the FastAPI chat (operator tools and hotspots on the left).

Tabs: Start, Chat (answers with every figure checked and linked to its source), Daily brief, Action plan, Replay
(animated map of a past day, "what would we have seen at 22:45?"), Network map (animated risk forecast for a
day or scenario), Compare, Value, Reports, Trust, Audit and Data (status, injection of new files, what the model
learned). The day and scenario in the sidebar drive the brief, the plan and the network map.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from datetime import date as _date, time as _time
from pathlib import Path

import pandas as pd
import streamlit as st

import src.analyses as A
import src.maps as MP
import src.ops as O
import src.reports as RP
import src.verify as VF
from src.agent import DEFAULT_BACKEND, Agent, data_dir, read_audit, run_selftest, to_jsonable, to_markdown
from src.engine import UBahnEngine
from src.ingest import ingest

ROOT = Path(__file__).resolve().parent
DATA = os.environ.get('UBAHN_DATA') or data_dir()
BACKEND = os.environ.get('UBAHN_BACKEND', DEFAULT_BACKEND)
MODEL = os.environ.get('MODEL_NAME') or os.environ.get('AZURE_MODEL') or os.environ.get('UBAHN_MODEL') or None
AUDIT = os.environ.get('UBAHN_AUDIT', 'audit_log.jsonl')
EXPERT_DEFAULT = os.environ.get('UBAHN_EXPERT', '0') == '1'
QUICK_QUESTIONS = ['What should we prepare for on this day?', 'Which stations are most at risk, and when?',
                   'What did the model learn from the latest data?']
FEEDBACK = os.environ.get('UBAHN_FEEDBACK', os.path.join(os.path.dirname(AUDIT), 'feedback_log.jsonl') if os.path.dirname(AUDIT) else 'feedback_log.jsonl')
EXAMPLES = [
    'U8 is suspended between Hermannplatz and Neukölln. Where will passengers reroute, and which stations are at risk in the next 20 minutes?',
    "There's a Guns N' Roses concert on June 23rd at the Uber Arena. What will the passenger flow look like at the neighboring stations and what measures should we take?",
    'During InnoTrans 2026 we expect major passenger flow and bad weather. Which 3 stations are most likely to exceed safe platform capacity on the first day?',
    'What would we have seen on the network at 22:45 on 23 June?',
    'What data is loaded, and what did the model learn from the latest data?',
]
HOTSPOTS_FILE = ROOT / 'data' / 'hotspots.json'
ASSISTANT_AVATAR = ':material/train:'
USER_AVATAR = ':material/person:'


def _b64(name: str) -> str:
    p = ROOT / 'static' / name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ''


def icon(name: str, color: str | None = None, size: int = 18) -> str:
    """Material Symbol for HTML blocks (st.markdown with unsafe_allow_html)."""
    style = f'font-size:{size}px;vertical-align:-4px;' + (f'color:{color};' if color else '')
    return f'<span class="material-symbols-outlined" style="{style}">{name}</span>'


st.set_page_config(page_title='Track Knack · U-Bahn operations assistant', page_icon=':material/subway:', layout='wide')

# ------------------------------------------------------------------------------------ theme (light design)
THEME_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* Base */
.stApp { background-color: #FFFFFF; }
section[data-testid="stSidebar"] { background-color: #F5F5F5; border-right: 1px solid #E5E7EB; }
section[data-testid="stSidebar"] > div { padding-bottom: 96px; }
.stApp, .stMarkdown, p, li, label { color: #1a1a1a; font-family: 'Inter', sans-serif; }
.material-symbols-outlined { font-family: 'Material Symbols Outlined' !important; font-weight: normal; font-style: normal;
  line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; }
header[data-testid="stHeader"] { background: #FFFFFF; }
.block-container { padding-top: 2.2rem; padding-bottom: 88px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #6B7280; font-weight: 500; }
.stTabs [data-baseweb="tab"] p { color: inherit; }
.stTabs [aria-selected="true"] { color: #2563EB; border-bottom: 2px solid #2563EB; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2563EB; }
/* Streamlit >= 1.60 renders tabs with react-aria instead of baseweb */
.stTabs [role="tablist"] { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; }
.stTabs [data-testid="stTab"] { color: #6B7280; font-weight: 500; }
.stTabs [data-testid="stTab"] p { color: inherit; font-weight: 500; }
.stTabs [data-testid="stTab"][aria-selected="true"] { color: #2563EB; box-shadow: inset 0 -2px 0 #2563EB; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  background-color: #2563EB; color: white; border-radius: 8px; border: none; }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
  background-color: #1D4ED8; color: white; }
.stButton > button p, .stFormSubmitButton > button p, .stDownloadButton > button p { color: white; }

/* Inputs */
.stTextInput > div > div > input, .stTextArea textarea, .stNumberInput input, .stDateInput input, .stTimeInput input {
  border: 1px solid #D1D5DB; border-radius: 8px; background: #FFFFFF; }
.stTextInput > div > div, .stNumberInput > div > div, .stDateInput > div > div, .stTimeInput > div > div,
[data-baseweb="select"] > div { background: #FFFFFF; border-radius: 8px; }

/* Tables */
.stDataFrame { border: 1px solid #E5E7EB; border-radius: 8px; }

/* Sidebar text */
section[data-testid="stSidebar"] .stMarkdown { color: #374151; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #111827; }
.side-h { font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #6B7280;
  margin: 18px 0 6px; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; }

/* Metrics */
[data-testid="metric-container"], [data-testid="stMetric"] { background: #F9FAFB; border: 1px solid #E5E7EB;
  border-radius: 8px; padding: 12px; }

/* Expanders and alerts: no heavy card borders */
[data-testid="stExpander"] details { border: 1px solid #E5E7EB; border-radius: 8px; }

/* Remove Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Persistent header / footer */
.tk-header { display:flex; align-items:center; padding:10px 8px; border-bottom:1px solid #E5E7EB; background:#fff; margin-bottom:12px; }
.tk-header .team { font-size:13px; color:#6B7280; font-weight:500; min-width:90px; }
.tk-header .mid { flex:1; display:flex; align-items:center; justify-content:center; gap:10px; }
.tk-header img { height:36px; border-radius:6px; }
.tk-footer { position:fixed; bottom:0; left:0; width:300px; padding:10px 16px; background:#F5F5F5; border-top:1px solid #E5E7EB;
  font-size:11px; color:#6B7280; z-index:999991; display:flex; align-items:center; gap:10px; line-height:1.35; }
.tk-footer img { height:18px; }

/* Chat tab: left panel like the FastAPI sidebar */
.st-key-tk_panel { background: #F5F5F5; border-radius: 10px; padding: 8px 6px 10px; }
.st-key-tk_panel [data-testid="stExpander"] details { border: none; background: transparent; }
.st-key-tk_panel [data-testid="stExpander"] summary p { font-size: 12px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: #6B7280; }
.st-key-tk_panel .stButton > button { background: transparent; color: #1a1a1a; justify-content: flex-start; text-align: left;
  padding: 4px 10px; min-height: 0; font-weight: 400; width: 100%; }
.st-key-tk_panel .stButton > button p { color: #1a1a1a; font-size: 14px; text-align: left; }
.st-key-tk_panel .stButton > button:hover { background: #E5E7EB; }

/* Chat bubbles: white for the assistant, light blue for the user */
[data-testid="stChatMessage"] { background: transparent; padding: 4px 0; gap: 10px; }
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] { background: #FFFFFF; border: 1px solid #E5E7EB;
  border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
[data-testid="stElementContainer"]:has(> div .tk-user), [data-testid="stElementContainer"]:has(> div .tk-assistant),
[data-testid="stElementContainer"]:has(.tk-user):not(:has([data-testid="stChatMessage"])),
[data-testid="stElementContainer"]:has(.tk-assistant):not(:has([data-testid="stChatMessage"])) { display: none; }
[data-testid="stChatMessage"]:has(.tk-user) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has(.tk-user) [data-testid="stChatMessageContent"] { background: #EFF6FF; border-color: #DBEAFE;
  flex: 0 1 auto; max-width: 75%; margin-left: auto; }
[data-testid="stChatMessage"]:has(.tk-assistant) [data-testid^="stChatMessageAvatar"],
[data-testid="stChatMessageAvatarAssistant"] { background: #10B981; color: #FFFFFF; border-radius: 50%; }
[data-testid="stChatMessage"]:has(.tk-user) [data-testid^="stChatMessageAvatar"] { background: #DBEAFE; color: #2563EB; border-radius: 50%; }

/* Chat input: search bar with a blue circular send button */
[data-testid="stChatInput"] { border: 1px solid #93C5FD; border-radius: 24px; background: #FFFFFF; }
[data-testid="stChatInput"] textarea { background: #FFFFFF; }
[data-testid="stChatInputSubmitButton"] { background: #2563EB !important; color: #FFFFFF !important; border-radius: 50% !important; }
[data-testid="stChatInputSubmitButton"]:disabled { background: #93C5FD !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #FFFFFF; color: #FFFFFF; }
</style>
"""
THEME_CSS = '\n'.join(line for line in THEME_CSS.splitlines() if line.strip())   # a blank line ends Markdown's HTML block
st.markdown(THEME_CSS + VF.CSS + """<style>
.vsources{max-height:170px;overflow-y:auto;border-left:3px solid #d1d5db;padding:4px 8px;margin-top:4px}
.vsrc:target{background:#fef08a}
.vbadge{font-size:.82em;opacity:.9}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------ engine and agent
@st.cache_resource(show_spinner='Loading data and fitting the model...')
def load_engine(data_dir: str, version: int):
    eng = UBahnEngine(data_dir)
    O.band_coverage(eng)
    return eng


if 'data_version' not in st.session_state:
    st.session_state.data_version = 0
eng = load_engine(DATA, st.session_state.data_version)
if st.session_state.get('agent_version') != st.session_state.data_version:
    for f in (A._daily, A._line_graph, A._residual_z, A._vulnerability):
        f.cache_clear()
    st.session_state.agent = Agent(eng, BACKEND, MODEL, client=st.session_state.get('agent_client'), audit_path=AUDIT)
    st.session_state.agent_version = st.session_state.data_version
    st.session_state.setdefault('messages', [])
agent = st.session_state.agent
VER = st.session_state.data_version
first_day = eng.service_day(pd.DatetimeIndex([eng.flows.index.min()]))[0].date()
last_day = eng.service_day(pd.DatetimeIndex([eng.flows.index.max()]))[0].date()


def has_weather(d: _date) -> bool:
    day = pd.Timestamp(d)
    import numpy as np
    return bool(np.isfinite(eng._daily_temp(eng.slots(day + pd.Timedelta('5h'), day + pd.Timedelta('23h')))).all())


def side_header(text: str):
    st.markdown(f'<div class="side-h">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    side_header('Data')
    status = eng.data_status()
    st.caption(f"Flows: {status['flows']}")
    st.caption(f"Events: {status['events']}  \nClosures: {status['closures']}")
    if status['events_not_located'] or status['closures_not_understood']:
        st.warning('Some records could not be interpreted: see the Data tab.', icon=':material/warning:')

    side_header('Day and scenario')
    goto = st.session_state.pop('goto_last_day', False)
    if goto or 'day' not in st.session_state:
        st.session_state['day'] = last_day if goto else _date(2026, 9, 24)
    day = st.date_input('Service day', key='day')
    rec_w = has_weather(day)
    use_w = st.checkbox('Weather scenario', value=not rec_w, help='Needed for days without recorded weather')
    weather = None
    if use_w:
        c1, c2 = st.columns(2)
        tmean = c1.number_input('Mean °C', value=16.0, step=1.0)
        prcp = c2.number_input('Rain mm/h', value=0.0, step=0.5, min_value=0.0)
        weather = {'tmean': float(tmean), 'prcp': float(prcp)}
    extra_events = []
    if st.checkbox('InnoTrans at Messe Berlin (09:00-18:00)', value=True):
        att = st.number_input('Visitors that day', value=40000, step=5000, min_value=1000)
        extra_events.append({'name': 'InnoTrans 2026', 'venue': 'Messedamm 22', 'start': f'{day} 09:00',
                             'end': f'{day} 18:00', 'attendance': float(att)})
    with st.expander('Add an event', icon=':material/event:'):
        ev_venue = st.text_input('Venue or address', value='', placeholder='e.g. Uber Arena')
        c1, c2, c3 = st.columns(3)
        ev_s = c1.time_input('Start', value=_time(20, 0))
        ev_e = c2.time_input('End', value=_time(23, 0))
        ev_a = c3.number_input('Attendance', value=5000, step=500, min_value=100)
        if ev_venue.strip():
            extra_events.append({'name': ev_venue.strip(), 'venue': ev_venue.strip(), 'start': f'{day} {ev_s:%H:%M}',
                                 'end': f'{day} {ev_e:%H:%M}', 'attendance': float(ev_a)})
    extra_closures = []
    with st.expander('Add a closure', icon=':material/block:'):
        cl_txt = st.text_input('Description', value='', placeholder='e.g. Station Kaiserdamm closed')
        c1, c2 = st.columns(2)
        cl_s = c1.time_input('From', value=_time(17, 0))
        cl_h = c2.number_input('Hours', value=2.0, step=0.5, min_value=0.25)
        if cl_txt.strip():
            s0 = pd.Timestamp(f'{day} {cl_s:%H:%M}')
            extra_closures.append({'description': cl_txt.strip(), 'start': str(s0), 'end': str(s0 + pd.Timedelta(hours=float(cl_h)))})
    staff = int(st.number_input('Additional staff available', value=10, step=1, min_value=1, max_value=80))
    with st.expander('Reinforcement assumptions (other modes)', icon=':material/tune:'):
        st.caption('Planning assumptions: the data only records U-Bahn flows.')
        c1, c2 = st.columns(2)
        mode_a = {'ubahn_max_trains_per_hour': int(c1.number_input('Max U-Bahn trains/h', value=20, step=1, min_value=6, max_value=40)),
                  'sbahn_extra_trains_max': int(c2.number_input('Extra S-Bahn trains/h', value=4, step=1, min_value=0, max_value=12)),
                  'buses_max': int(c1.number_input('Shuttle buses available', value=20, step=1, min_value=0, max_value=200)),
                  'bus_capacity': int(c2.number_input('Passengers per bus', value=90, step=10, min_value=20, max_value=200)),
                  'taxis_max': int(c1.number_input('Taxis available', value=40, step=5, min_value=0, max_value=500)),
                  'bikes_max': int(c2.number_input('Shared bikes available', value=80, step=10, min_value=0, max_value=1000))}
    side_header('Display')
    expert = st.toggle('Expert mode (all tools)', value=EXPERT_DEFAULT, key='expert',
                       help='Operator mode shows the Start page, the Chat and the Data; expert mode adds the plan, maps, comparison, value, reports, trust and audit tabs.')
    side_header('Answers')
    verify_on = st.toggle('Check every figure of the answers', value=True)
    if st.button('New conversation', icon=':material/add_comment:'):
        st.session_state.messages = []
        agent.history = []
        st.rerun()

SCEN = json.dumps({'day': str(day), 'weather': weather, 'events': extra_events, 'closures': extra_closures, 'staff': staff,
                   'modes': mode_a})


@st.cache_data(show_spinner=False)
def cached(kind: str, version: int, scen: str):
    s = json.loads(scen)
    d, w, ev, cl = s['day'], s['weather'], s['events'] or None, s['closures'] or None
    if kind == 'brief':
        return O.daily_brief(eng, d, w, ev, cl, s['staff'], s.get('modes'))
    if kind == 'plan':
        return O.action_plan(eng, d, w, ev, cl, top=8)
    if kind == 'risk':
        return MP.forecast_frames(eng, d, w, ev, cl)
    if kind == 'reinforce':
        return O.reinforcement_plan(eng, d, w, ev, cl, top=8, assumptions=s.get('modes'))
    raise ValueError(kind)


@st.cache_data(show_spinner=False)
def cached_value(version: int):
    return {k: v for k, v in RP.value_report(eng).items() if not k.startswith('_')}


@st.cache_data(show_spinner=False)
def cached_incident(version: int, d: str):
    return RP.incident_report(eng, d)


@st.cache_data(show_spinner=False)
def cached_handbook(version: int):
    return RP.contingency_handbook(eng)


@st.cache_data(show_spinner=False)
def cached_messages(version: int, kind: str, item_json: str):
    it = json.loads(item_json)
    if kind == 'closure':
        return O.passenger_messages(eng, closure=it['description'], start=str(it['start']), end=str(it['end']))
    return O.passenger_messages(eng, venue=it['venue'], event_start=str(it['start']), event_end=str(it['end']))


@st.cache_data(show_spinner=False)
def cached_replay(version: int, d: str, t_from: str, t_to: str):
    return MP.replay_frames(eng, d, t_from, t_to)


@st.cache_data(show_spinner=False)
def load_hotspots() -> list:
    try:
        return json.loads(HOTSPOTS_FILE.read_text(encoding='utf-8')).get('hotspots', [])
    except (OSError, json.JSONDecodeError):
        return []


def code_block(txt: str):
    try:
        st.code(txt, language=None, wrap_lines=True)
    except TypeError:                                   # older Streamlit versions
        st.code(txt, language=None)


def show_plotly(fig):
    try:
        st.plotly_chart(fig, width='stretch')
    except TypeError:                                   # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def render_answer(text: str, verification: dict | None, trace: list | None, key: str):
    if verify_on and verification and verification.get('figures'):
        st.markdown(VF.annotate_html(text, verification, prefix=key), unsafe_allow_html=True)
        s = verification['summary']
        badge = (f"{icon('check_circle', '#10B981', 16)} {s['verified']} traced to a tool result · "
                 f"{icon('calculate', '#2563EB', 16)} {s['calculated']} calculated · "
                 f"{icon('error', '#EF4444', 16)} {s['not_found']} not found" + (' in the tool results' if s['not_found'] else ''))
        st.markdown(f'<div class="vbadge">{badge} — hover a figure or click its number to see its source. '
                    'Tracing checks where each figure comes from, not the reasoning.</div>'
                    f'<div class="vsources">{VF.sources_html(verification, prefix=key)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)
    with st.expander('Plain text to copy', icon=':material/content_copy:'):
        code_block(text)
    if trace:
        with st.expander(f'How this answer was computed ({len(trace)} steps)', icon=':material/account_tree:'):
            for i, c in enumerate(trace, 1):
                st.markdown(f"**{i}. `{c['tool']}`** {json.dumps(c['args'], ensure_ascii=False)}" + (' — error' if c['error'] else ''))
                st.code(c['output'][:3000] + (' ...' if len(c['output']) > 3000 else ''), language='json')


def feedback_row(key: str, kind: str, about: str, extra: dict | None = None,
                 up_label=':material/thumb_up:', down_label=':material/thumb_down:'):
    """Two buttons to rate an answer or an alert, with an optional comment; ratings go to the feedback log."""
    done = st.session_state.get(f'fbdone_{key}')
    if done:
        st.caption(f'Thanks, recorded: {done}')
        return
    c1, c2, c3 = st.columns([1, 1, 6])
    comment = c3.text_input('Comment (optional)', key=f'fbc_{key}', label_visibility='collapsed', placeholder='Comment (optional)')
    for col, lab, rating in ((c1, up_label, 'up'), (c2, down_label, 'down')):
        if col.button(lab, key=f'fb{rating}_{key}'):
            RP.log_feedback(FEEDBACK, {'kind': kind, 'rating': rating, 'about': about, 'comment': comment.strip(), 'model': MODEL, **(extra or {})})
            st.session_state[f'fbdone_{key}'] = lab
            st.rerun()


def ask(question: str):
    """Ask the agent and show each step live (model requests and tool calls with their duration)."""
    t0 = time.time()
    with st.status('Working through the data...', expanded=True) as status:
        def progress(ev):
            if ev['kind'] == 'model':
                status.write(f":material/smart_toy: asking the model ({ev.get('steps_done', 0)} calculation(s) so far)...")
            elif ev['kind'] == 'tool_start':
                args = json.dumps(ev.get('args') or {}, ensure_ascii=False)
                status.write(f":material/build: `{ev['tool']}` {args[:110]}{'...' if len(args) > 110 else ''}")
            elif ev['kind'] == 'tool_end':
                status.write((':orange[:material/warning:] error' if ev.get('error') else ':green[:material/check:]')
                             + f" {ev['seconds']:.1f} s · {ev['chars']:,} characters of results")
        agent.progress_cb = progress
        try:
            answer = agent.ask(question)
            status.update(label=f'Done in {time.time() - t0:.0f} s · {len(agent.trace)} calculation(s)', state='complete', expanded=False)
        except Exception as ex:             # keep the app usable if the language model cannot be reached
            answer = f'The language model could not be reached ({type(ex).__name__}: {ex}).'
            agent.trace, agent.last_verification = [], None
            status.update(label='The language model could not be reached', state='error', expanded=True)
        finally:
            agent.progress_cb = None
    return {'role': 'assistant', 'content': answer, 'trace': list(agent.trace),
            'verification': getattr(agent, 'last_verification', None), 'seconds': round(time.time() - t0, 1)}


def df(rows, cols=None):
    d = pd.DataFrame(rows)
    return d[cols] if cols and not d.empty else d


# ------------------------------------------------------------------------------------ operator tools (chat panel)
def _op_forecast_args(s: dict) -> dict:
    """Same default as POST /api/forecast: the first watch window of the action plan and its stations."""
    watch = cached('plan', VER, SCEN).get('watch', [])
    if not watch:
        return {'start': f"{s['day']} 07:00", 'end': f"{s['day']} 10:00", 'stations': None, 'line': 'U2'}
    window = watch[0]['window']
    stations = list(dict.fromkeys(w['station'] for w in watch if w['window'] == window))[:4]
    return {'start': f"{s['day']} {window.split('-')[0]}", 'end': f"{s['day']} {window.split('-')[1]}", 'stations': stations, 'line': None}


def run_operator_tool(kind: str) -> tuple[str, str]:
    """The five operator tools of the FastAPI sidebar, on the day and scenario set in the Streamlit sidebar.
    Deterministic, no language model. Returns (tool name, Markdown)."""
    s = json.loads(SCEN)
    d, w, ev, cl = s['day'], s['weather'], s['events'] or None, s['closures'] or None
    if kind == 'brief':
        return 'ubahn_daily_brief', to_markdown('ubahn_daily_brief', to_jsonable(cached('brief', VER, SCEN)))
    if kind == 'plan':
        return 'ubahn_action_plan', to_markdown('ubahn_action_plan', to_jsonable(cached('plan', VER, SCEN)))
    if kind == 'forecast':
        a = _op_forecast_args(s)
        res = O.forecast(eng, a['start'], a['end'], a['stations'], a['line'], w, ev, cl)
        return 'ubahn_forecast', to_markdown('ubahn_forecast', to_jsonable(res))
    if kind == 'staff':
        return 'ubahn_staff_plan', to_markdown('ubahn_staff_plan', to_jsonable(O.staff_plan(eng, d, staff, 2.0, w, ev, cl)))
    if kind == 'messages':
        b = cached('brief', VER, SCEN)
        items = [('closure', c) for c in b['closures']] + [('event', e) for e in sorted(b['events'], key=lambda e: -e.get('attendance', 0))]
        if not items:
            return 'ubahn_passenger_messages', 'No closure or event on this day, so no passenger messages are needed.'
        k, it = items[0]
        res = cached_messages(VER, k, json.dumps(it, default=str))
        if 'error' in res:
            return 'ubahn_passenger_messages', res['error']
        return 'ubahn_passenger_messages', to_markdown('ubahn_passenger_messages', to_jsonable(res))
    raise ValueError(kind)


OPERATOR_TOOLS = [('brief', 'Daily brief', ':material/today:'), ('plan', 'Action plan & triggers', ':material/rule:'),
                  ('forecast', 'Forecast with ranges', ':material/show_chart:'), ('staff', 'Staff plan', ':material/group:'),
                  ('messages', 'Passenger messages (DE/EN/FR)', ':material/campaign:')]


def render_chat_panel():
    """Left panel of the Chat tab, like the FastAPI sidebar: operator tools, then the hotspots in two sections."""
    hs = load_hotspots()
    sections = [('transit', 'Standard Traffic Problems'), ('places', 'Event Places')]
    with st.container(key='tk_panel'):
        with st.expander(f'Operator Tools ({len(OPERATOR_TOOLS)})', expanded=True):
            for kind, label, ic in OPERATOR_TOOLS:
                if st.button(label, key=f'op_{kind}', icon=ic):
                    st.session_state.pending_op = (kind, label)
        for sec, title in sections:
            items = [h for h in hs if (h.get('section') or ('transit' if h.get('category') == 'transit_hub' else 'places')) == sec]
            if not items:
                continue
            with st.expander(f'{title} ({len(items)})', expanded=False):
                for h in items:
                    if st.button(h.get('label') or h['id'], key=f"hs_{h['id']}", help=h.get('auto_query')):
                        st.session_state.pending = h.get('auto_query') or h.get('label')


def chat_message(role: str):
    """Chat bubble with the Track Knack avatars; the marker span lets the CSS style user and assistant bubbles."""
    box = st.chat_message(role, avatar=ASSISTANT_AVATAR if role == 'assistant' else USER_AVATAR)
    box.markdown(f'<span class="tk-{role}"></span>', unsafe_allow_html=True)
    return box


# ------------------------------------------------------------------------------------ header, footer, tabs
st.markdown(f"""
<div class="tk-header">
  <span class="team">Track Knack</span>
  <div class="mid">
    <img src="data:image/jpeg;base64,{_b64('logo.jpg')}" alt="Hackathon 2026 InnoTrans">
    <span style="font-size:18px; font-weight:700; color:#111827;">Track Knack</span>
    <span style="font-size:13px; color:#6B7280;">InnoTrans Hackathon 2026</span>
  </div>
  <span class="team" style="text-align:right;">{icon('subway', '#6B7280', 18)} U-Bahn operations</span>
</div>
<div class="tk-footer">
  <span>Developed in the context of the InnoTrans 2026 Hackathon in collaboration with ALSTOM</span>
</div>
""", unsafe_allow_html=True)


def render_chat():
    # The FastAPI chat (src/server.py on port 8000) is embedded as is: its input stays pinned to the bottom.
    # ?embed=1 hides its own header and footer; the LAN address lets other devices on the network load it too.
    import socket

    import streamlit.components.v1 as components

    local_ip = socket.gethostbyname(socket.gethostname())
    components.html(
        f"""
        <style>
          body {{ margin: 0; padding: 0; overflow: hidden; }}
          iframe {{
            display: block;
            border: none;
            border-radius: 8px;
            width: 100%;
            height: 100vh;
            max-height: 100%;
          }}
        </style>
        <iframe
            src="http://{local_ip}:8000?embed=1"
            width="100%"
            height="100%"
            frameborder="0"
            scrolling="auto"
        ></iframe>
        <script>
          // 100vh inside components.html is this component's own height (750 px), not the browser window.
          // Resize the component to the space left below it in the Streamlit page so the input stays visible.
          (function () {{
            var frame = window.frameElement;
            if (!frame) {{ return; }}
            var last = 0;
            function fit() {{
              var r = frame.getBoundingClientRect();
              if (r.width === 0) {{ return; }}            // tab not shown yet: its position is not known
              var top = r.top + window.parent.scrollY;     // position in the page, not in the scrolled view
              var h = Math.max(420, Math.round(window.parent.innerHeight - top - 16));
              if (h === last) {{ return; }}
              last = h;
              frame.style.height = h + "px";
              frame.setAttribute("height", h);
            }}
            fit();
            window.parent.addEventListener("resize", fit);
            setInterval(fit, 400);                         // also catches the switch to the Chat tab
          }})();
        </script>
        """,
        height=750,
        scrolling=False
    )


def render_brief():
    b = cached('brief', VER, SCEN)
    st.subheader(f"Brief for {pd.Timestamp(b['date']):%A %d %B %Y}")
    for n in b['notes']:
        st.info(n, icon=':material/info:')
    for h in b['headlines']:
        st.markdown(f'- {h}')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Where to put the additional staff**')
        st.dataframe(df(b['staff_plan'], ['station', 'from', 'to', 'staff', 'reason', 'peak_chance_at_ceiling']), hide_index=True)
    with c2:
        st.markdown('**Stations and windows to watch**')
        st.dataframe(df(b['watch']), hide_index=True)
    st.markdown('**Key triggers**')
    st.dataframe(df(b['key_triggers']), hide_index=True)
    if b['replay']:
        st.markdown('**Replayed on the recorded data: when the triggers would have fired**')
        st.dataframe(pd.DataFrame([{'station': r['station'], **{k: v or '—' for k, v in r['would_have_fired_at'].items()}} for r in b['replay']]), hide_index=True)


def render_plan():
    p = cached('plan', VER, SCEN)
    st.subheader(f"Conditional action plan · {pd.Timestamp(p['date']):%A %d %B %Y}")
    st.caption(p['threshold_basis'])
    for n in p['notes']:
        st.info(n, icon=':material/info:')
    for w in p['watch']:
        with st.expander(f"**{w['station']}** ({', '.join(w['lines'])}) · {w['window']} · {w['reason']}", icon=':material/visibility:'):
            st.dataframe(pd.DataFrame([{k: t.get(k, '') for k in ('id', 'when', 'if', 'then', 'indicator')} for t in w['triggers']]), hide_index=True)
            thr = {t['id']: t.get('thresholds') for t in w['triggers'] if t.get('thresholds')}
            table = pd.DataFrame(thr)
            if 'replay_on_recorded_data' in w:
                table['recorded'] = pd.Series(w['replay_on_recorded_data']['readings'])
                fired = {k: v or '—' for k, v in w['replay_on_recorded_data']['would_have_fired_at'].items()}
                st.markdown('Replay on the recorded data, would have fired at: ' + ', '.join(f'**{k}** {v}' for k, v in fired.items()))
            st.dataframe(table.rename(columns={'surge': 'surge threshold', 'above_plan': 'above-plan threshold'}))
            st.caption('Was this alert useful?')
            feedback_row(f"al_{p['date']}_{w['station']}_{w['window']}", 'alert', f"{str(p['date'])[:10]} {w['station']} {w['window']}",
                         {'reason': w['reason']}, up_label=':material/thumb_up: useful', down_label=':material/thumb_down: not useful')
    st.subheader('Reinforcement across modes')
    rp = cached('reinforce', VER, SCEN)
    for line in rp['summary']:
        st.markdown(f'- {line}')
    if rp['windows']:
        st.dataframe(pd.DataFrame([{'station': w['station'], 'window': w['window'], 'reason': w['reason'],
                                    'peak demand / ceiling': w['peak_demand_over_ceiling'],
                                    'extra above capacity per hour': w['extra_passengers_above_capacity_per_hour'],
                                    'measures': ' · '.join(f"{m['mode']}: {m['action']}" for m in w['measures'])}
                                   for w in rp['windows']]), hide_index=True)
    st.caption(rp['caution'] + '. S-Bahn-only stations (e.g. Messe Nord/ICC) can be added in sbahn_extra.csv in the data folder.')
    st.subheader('Passenger messages for the day')
    bq = cached('brief', VER, SCEN)
    situations = [('closure', c) for c in bq['closures']] + [('event', e) for e in bq['events']]
    if not situations:
        st.caption('No closure or event this day.')
    for kind, item in situations:
        msgs = cached_messages(VER, kind, json.dumps(item, default=str))
        title = item['description'] if kind == 'closure' else f"{item['name']} ({item['venue']})"
        with st.expander(title, icon=':material/construction:' if kind == 'closure' else ':material/confirmation_number:'):
            if 'error' in msgs:
                st.warning(msgs['error'], icon=':material/warning:')
                continue
            cols = st.columns(3)
            for col, (lang, label) in zip(cols, [('de', 'Deutsch'), ('en', 'English'), ('fr', 'Français')]):
                with col:
                    st.markdown(f'**{label}**')
                    for ch, txt in msgs['messages'][lang].items():
                        st.caption(ch)
                        code_block(txt)


def render_replay():
    st.subheader('Replay of a recorded day')
    c1, c2, c3 = st.columns([2, 1, 1])
    r_day = c1.date_input('Day', value=day if first_day <= day <= last_day else last_day, min_value=first_day, max_value=last_day, key='rday')
    r_from = c2.selectbox('From', ['05:00', '07:00', '12:00', '16:00', '17:00', '19:00', '21:00'], index=4)
    r_to = c3.selectbox('To', ['12:00', '20:00', '22:00', '23:00', '00:45'], index=4)
    try:
        frames = cached_replay(VER, str(r_day), r_from, r_to)
    except ValueError as ex:
        frames = []
        st.warning(str(ex), icon=':material/warning:')
    if frames:
        show_plotly(MP.network_figure(eng, frames, 'replay'))
        slots = [f['slot'].strftime('%H:%M') for f in frames]
        pick = st.select_slider('Moment to inspect', options=slots, value=slots[min(len(slots) - 1, len(slots) // 2)])
        snap = O.snapshot(eng, f'{r_day} {pick}' if pick >= '05:00' else f'{pd.Timestamp(r_day) + pd.Timedelta("1D"):%Y-%m-%d} {pick}')
        for f_ in snap.get('facts', [snap.get('error', '')]):
            st.markdown(f'- {f_}')
        if snap.get('largest_deviations'):
            st.dataframe(df(snap['largest_deviations'], ['station', 'recorded', 'normal', 'ratio', 'ceiling']), hide_index=True)
        if st.button(f'Ask the agent: what would we have seen at {pick}?', icon=':material/psychology:'):
            with st.spinner('Asking the agent...'):
                st.session_state.replay_answer = ask(f'What would we have seen on the network at {pick} on {r_day:%d %B %Y}? '
                                                     'Explain the unusual stations and their causes.')
        if st.session_state.get('replay_answer'):
            ra = st.session_state.replay_answer
            render_answer(ra['content'], ra['verification'], ra['trace'], key='replay')


def render_map():
    st.subheader(f"Network risk map · {pd.Timestamp(day):%A %d %B %Y}")
    frames_r, notes_r = cached('risk', VER, SCEN)
    for n in notes_r:
        st.info(n, icon=':material/info:')
    st.caption('Colour: chance that a station reaches its ceiling in each 15-minute slot, with the recorded or scenario '
               'weather, the events and the closures of the day. Press :material/play_arrow: Play to play the day.')
    show_plotly(MP.network_figure(eng, frames_r, 'risk'))
    slots = [f['slot'].strftime('%H:%M') for f in frames_r]
    pick = st.select_slider('Slot', options=slots, value='18:00' if '18:00' in slots else slots[0], key='riskslot')
    fr = frames_r[slots.index(pick)]
    st.markdown(f"**{fr['caption']}**")
    st.dataframe(pd.DataFrame([{'station': k, 'chance of reaching the ceiling': f'{p_:.0%}', 'expected': int(e), 'ceiling': int(c)}
                               for k, p_, e, c in fr['top']]), hide_index=True)


def render_cmp():
    st.subheader(f"Compare two scenarios · {pd.Timestamp(day):%A %d %B %Y}")
    st.caption('The day is set in the sidebar. An empty scenario is the day as recorded, or a dry 16 °C day in the future.')
    cols = st.columns(2)
    scen = {}
    for col, tag, default_att in ((cols[0], 'A', 0), (cols[1], 'B', 40000)):
        with col:
            label = st.text_input('Name', value='Without InnoTrans' if tag == 'A' else 'With InnoTrans', key=f'cl{tag}')
            use_w_ = st.checkbox('Set the weather', value=not rec_w, key=f'cw{tag}')
            sc = {'label': label or tag}
            if use_w_:
                c1, c2 = st.columns(2)
                sc['tmean'] = float(c1.number_input('Mean °C', value=16.0, step=1.0, key=f'ct{tag}'))
                sc['prcp'] = float(c2.number_input('Rain mm/h', value=0.0 if tag == 'A' else 1.5, step=0.5, min_value=0.0, key=f'cp{tag}'))
            att_ = st.number_input('InnoTrans visitors (0 = none)', value=default_att, step=5000, min_value=0, key=f'ca{tag}')
            if att_ > 0:
                sc['extra_events'] = [{'name': 'InnoTrans 2026', 'venue': 'Messedamm 22', 'start': f'{day} 09:00', 'end': f'{day} 18:00', 'attendance': float(att_)}]
            clo = st.text_input('Closure (optional)', value='', key=f'cc{tag}', placeholder='e.g. Station Kaiserdamm closed')
            if clo.strip():
                sc['extra_closures'] = [{'description': clo.strip(), 'start': f'{day} 07:00', 'end': f'{day} 20:00'}]
            scen[tag] = sc
    if scen['A']['label'] == scen['B']['label']:
        scen['B']['label'] = scen['B']['label'] + ' (B)'
    if st.button('Compare', type='primary', icon=':material/compare_arrows:'):
        with st.spinner('Computing both scenarios...'):
            st.session_state.compare = RP.compare_scenarios(eng, str(day), scen['A'], scen['B'])
    cr = st.session_state.get('compare')
    if cr:
        for h in cr['headlines']:
            st.markdown(f'- {h}')
        cols = st.columns(2)
        for col, lab in zip(cols, cr['summary']):
            sm = cr['summary'][lab]
            with col:
                st.markdown(f'**{lab}**')
                st.metric('Passengers on the network', f"{sm['network_expected_passengers']:,}")
                st.metric('Passengers above capacity', f"{sm['passengers_above_capacity']:,}")
                st.metric('Stations likely at their ceiling', sm['stations_likely_at_ceiling'])
                st.caption('Reinforcement: ' + ' '.join(cr['reinforcement'][lab]))
        st.markdown('**Stations that change the most**')
        st.dataframe(pd.DataFrame([{'station': c['station'], 'lines': '/'.join(c['lines']), 'passengers change': c['passengers_change'],
                                    'above capacity change': c['above_capacity_change'],
                                    'chance of reaching the ceiling': f"{c['max_chance_at_ceiling'][0]:.0%} -> {c['max_chance_at_ceiling'][1]:.0%}"}
                                   for c in cr['biggest_changes']]), hide_index=True)


def render_value():
    st.subheader('Proof of value: replaying the whole recorded period')
    st.caption('What the plans and alerts would have achieved on the recorded data. About 10 seconds the first time.')
    if st.button('Compute the proof of value', type='primary', icon=':material/insights:') or st.session_state.get('value_ready'):
        with st.spinner('Replaying the recorded period...'):
            vr = cached_value(VER)
        st.session_state.value_ready = True
        for h in vr['headlines']:
            st.markdown(f'- {h}')
        pl = vr['planning']
        c1, c2, c3 = st.columns(3)
        c1.metric('Event overloads flagged in advance', f"{pl['share_flagged_in_advance']:.0%}" if pl['share_flagged_in_advance'] is not None else '—',
                  help=pl['definitions']['flagged in advance'])
        c2.metric('Flagged event windows confirmed', f"{pl['share_confirmed_by_readings']:.0%}" if pl['share_confirmed_by_readings'] is not None else '—')
        best = next(r for r in vr['early_warning']['rules'] if r['rule'] == vr['early_warning']['recommended_rule'])
        c3.metric('Large events detected live', f"{best['events_detected']:.0%}", help=f"rule {best['rule']}: {best['description']}")
        st.markdown('**Live alert rules compared**')
        st.dataframe(pd.DataFrame(vr['early_warning']['rules'])[['rule', 'description', 'alerts_per_day', 'share_explained', 'events_detected',
                                                                  'minutes_after_event_end_median', 'used_by_action_plan']], hide_index=True)
        st.caption(vr['early_warning']['note'] + '. ' + vr['early_warning']['action_plan_rule'] + '.')
        if 'reinforcement' in vr:
            st.markdown('**Most demanding days and what reinforcement would have absorbed**')
            st.dataframe(pd.DataFrame([{'date': d['date'], 'day': d['weekday'], 'why': ', '.join(d['main_reasons']),
                                        'above capacity': d['extra_above_capacity'], 'absorbed': d['absorbed'], 'left to meter': d['residual'],
                                        'worst window': d['worst_window']} for d in vr['reinforcement']['top_days']]), hide_index=True)
            st.caption(vr['reinforcement']['basis'] + '.')


def render_reports():
    st.subheader('Incident report')
    c1, c2 = st.columns([2, 1])
    i_day = c1.date_input('Recorded day', value=day if first_day <= day <= last_day else last_day,
                          min_value=first_day, max_value=last_day, key='iday')
    if c2.button('Build the report', icon=':material/description:'):
        st.session_state.incident_day = str(i_day)
    if st.session_state.get('incident_day'):
        ir = cached_incident(VER, st.session_state.incident_day)
        if 'error' in ir:
            st.warning(ir['error'], icon=':material/warning:')
        else:
            st.download_button('Download the report (HTML, printable to PDF)', data=ir['html'], icon=':material/download:',
                               file_name=f"incident_{st.session_state.incident_day}.html", mime='text/html')
            st.markdown('**Lessons learned**')
            for l in ir['lessons_learned']:
                st.markdown(f'- {l}')
            with st.expander('Alerts replayed on the recorded readings', icon=':material/notifications:'):
                st.dataframe(pd.DataFrame(ir['alerts_replay']), hide_index=True)
    st.subheader('Contingency handbook')
    st.caption('One printable page per station: what to do if it closes, with alternatives, measures and passenger messages.')
    pick_st = st.selectbox('Look at one station', options=sorted(eng.keys), index=sorted(eng.keys).index('Kaiserdamm') if 'Kaiserdamm' in eng.keys else 0)
    cp = RP.station_contingency(eng, pick_st)
    t_ = cp['if_tracks_out']
    st.markdown(f"**{cp['station']}** ({'/'.join(cp['lines'])}): {cp['weekday_passengers']:,} passengers on a typical weekday. "
                + (f"If its tracks are out, {t_['stations_cut_off']} stations are cut off ({t_['passengers_affected_weekday']:,} weekday passengers affected)."
                   if t_['stations_cut_off'] else 'Its closure cuts no station off.'))
    for m_ in cp['measures']:
        st.markdown(f'- {m_}')
    if st.button('Build the full handbook (168 pages)', icon=':material/menu_book:'):
        st.session_state.handbook_ready = True
    if st.session_state.get('handbook_ready'):
        hb = cached_handbook(VER)
        st.download_button(f"Download the handbook ({hb['stations']} stations, HTML)", data=hb['html'], icon=':material/download:',
                           file_name='contingency_handbook.html', mime='text/html')


def render_trust():
    rel = O.reliability_report(eng)
    st.subheader('Can the numbers be trusted?')
    c1, c2, c3 = st.columns(3)
    cal = rel['forecast_ranges_calibration']
    bt = rel['backtest']
    c1.metric('80 % forecast range holds', f"{cal['80 % range holds']:.0%}", help='share of all recorded readings inside the range')
    c2.metric('90 % forecast range holds', f"{cal['90 % range holds']:.0%}")
    c3.metric('Daily network totals', f"±{bt['daily_network_totals']['mean_absolute_error_pct']:.1f} %",
              help=f"mean absolute error; correlation {bt['daily_network_totals']['correlation']}")
    c1, c2, c3 = st.columns(3)
    c1.metric('Recorded / expected, whole period', f"{bt['total_observed_over_expected']:.3f}")
    c2.metric('Hourly recorded / expected', f"{bt['hourly_observed_over_expected_range'][0]:.2f}–{bt['hourly_observed_over_expected_range'][1]:.2f}")
    c3.metric('Readings at the ceiling', f"{bt['readings_at_ceiling']['observed']:,}",
              delta=f"expected {bt['readings_at_ceiling']['expected']:,}", delta_color='off')
    st.subheader('Figures in the answers')
    sess = [m['verification']['summary'] for m in st.session_state.messages if m.get('role') == 'assistant' and m.get('verification')]
    logged = [e['figures'] for e in read_audit(AUDIT) if e.get('figures')]
    for label, lst in (('This session', sess), ('All answers in the audit log', logged)):
        tot = {k: sum(x.get(k, 0) for x in lst) for k in ('checked', 'verified', 'calculated', 'not_found')}
        rate = (tot['verified'] + tot['calculated']) / tot['checked'] if tot['checked'] else None
        st.markdown(f"**{label}**: {len(lst)} answer(s), {tot['checked']} figure(s) checked, "
                    f"{tot['verified']} verified, {tot['calculated']} calculated, {tot['not_found']} not found"
                    + (f" → **{rate:.1%} traced to a tool result**" if rate is not None else ''))
    st.subheader('Self-test')
    st.caption('Re-runs the calculations behind the training questions and checks the figures (quick version skips three slow network checks).')
    if st.button('Run the quick self-test', icon=':material/fact_check:'):
        with st.spinner('Running...'):
            st.session_state.selftest_rows = run_selftest(agent.tb, quick=True)
    if st.session_state.get('selftest_rows'):
        rows = st.session_state.selftest_rows
        bad = [r for r in rows if r['error']]
        (st.error if bad else st.success)(f"{len(rows) - len(bad)} of {len(rows)} checks passed",
                                          icon=':material/error:' if bad else ':material/check_circle:')
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader('Operator feedback')
    fbs = RP.feedback_summary(RP.read_feedback(FEEDBACK))
    c1, c2 = st.columns(2)
    for col, kind, lab in ((c1, 'answer', 'Answers'), (c2, 'alert', 'Alerts')):
        f_ = fbs[kind]
        col.metric(f'{lab} rated useful', f"{f_['share_up']:.0%}" if f_['share_up'] is not None else '—',
                   help=f"{f_['up']} useful and {f_['down']} not useful out of {f_['ratings']} ratings")
    if fbs['comments']:
        st.dataframe(pd.DataFrame(fbs['comments']), hide_index=True)
    st.caption(f'Ratings are stored in `{FEEDBACK}`.')
    with st.expander('Known limits', icon=':material/info:'):
        for l in rel['limits']:
            st.markdown(f'- {l}')


def render_audit():
    st.subheader('Audit log')
    entries = read_audit(AUDIT)
    st.caption(f'{len(entries)} answer(s) recorded in `{AUDIT}`: question, answer, figure check, every calculation and its result.')
    if entries:
        table = pd.DataFrame([{'time': e['time'], 'question': e['question'][:90], 'seconds': e.get('seconds'),
                               'calculations': len(e.get('steps', [])), 'figures checked': e.get('figures', {}).get('checked'),
                               'not found': e.get('figures', {}).get('not_found'), 'model': e.get('model')} for e in entries][::-1])
        st.dataframe(table, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button('Download the full log (JSON lines)', data='\n'.join(json.dumps(e, ensure_ascii=False) for e in entries),
                           file_name='audit_log.jsonl', mime='application/json', icon=':material/download:')
        c2.download_button('Download the summary (CSV)', data=table.to_csv(index=False), file_name='audit_summary.csv', mime='text/csv',
                           icon=':material/download:')
        pick = st.selectbox('Look at one answer', options=list(range(len(entries)))[::-1],
                            format_func=lambda i: f"{entries[i]['time']} · {entries[i]['question'][:80]}")
        e = entries[pick]
        st.markdown(f"**Question:** {e['question']}")
        st.markdown(e['answer'])
        st.caption(f"Figures: {e.get('figures')}")
        for i, stp in enumerate(e.get('steps', []), 1):
            st.markdown(f"{i}. `{stp['tool']}` {json.dumps(stp['args'], ensure_ascii=False)[:200]} · {stp.get('seconds', 0)} s")


def render_data():
    st.subheader('Data status')
    st.json(status, expanded=False)
    st.subheader('What the model learned from the last injection')
    lr = O.last_learning_report(eng)
    if 'summary' in lr:
        for line in lr['summary']:
            st.markdown(f'- {line}')
        if lr.get('venue_splits'):
            st.dataframe(pd.DataFrame([{'venue': v['venue'], 'status': v['status'],
                                        'before': ', '.join(f'{k} {x:.0%}' for k, x in (v.get('before (distance rule)') or v.get('before') or v.get('rule') or {}).items()),
                                        'after': ', '.join(f'{k} {x:.0%}' for k, x in (v.get('after (learned)') or v.get('after') or {}).items())}
                                       for v in lr['venue_splits']]), hide_index=True)
    else:
        st.caption(lr.get('note', ''))
    st.subheader('Add new data')
    uploads = st.file_uploader('CSV files (flows, weather, events, closures, energy, venues_extra)', type='csv', accept_multiple_files=True)
    if uploads:
        tmp = tempfile.mkdtemp()
        paths = []
        for u in uploads:
            pth = os.path.join(tmp, u.name)
            with open(pth, 'wb') as fh:
                fh.write(u.getbuffer())
            paths.append(pth)
        report = ingest(paths, DATA, dry_run=True, ref=eng)
        for r in report['files']:
            mark = ':red[:material/cancel:]' if r['errors'] else (':orange[:material/warning:]' if r['warnings'] else ':green[:material/check_circle:]')
            st.markdown(f"{mark} **{r['file']}** ({r['kind']}, {r['rows']} rows, {r['range']})")
            for m in r['errors']:
                st.error(m, icon=':material/cancel:')
            for m in r['warnings']:
                st.warning(m, icon=':material/warning:')
        if report['ok'] and st.button('Inject and refit', type='primary', icon=':material/upload:'):
            with st.spinner('Adding files, refitting and comparing with the previous model...'):
                done = ingest(paths, DATA, dry_run=False, ref=eng)
                done.pop('_engine', None)
                load_engine.clear()
                cached.clear()
                cached_replay.clear()
                st.session_state.data_version += 1
                st.session_state.goto_last_day = True
                st.session_state.injected = {'files': done['copied'], 'time': time.strftime('%H:%M')}
            st.success(f"Added: {', '.join(done['copied'])}", icon=':material/check_circle:')
            st.rerun()


def render_start():
    """Operator start page: the day at a glance, what to watch, what to reinforce, and a question box."""
    inj = st.session_state.get('injected')
    if inj:
        lr = O.last_learning_report(eng)
        with st.container(border=True):
            st.markdown(f"**New data added at {inj['time']}** ({', '.join(inj['files'])}). The day below is the latest recorded day.")
            for line in lr.get('summary', [])[:5]:
                st.markdown(f'- {line}')
            if st.button('Dismiss', key='dismiss_injected'):
                st.session_state.injected = None
                st.rerun()
    b = cached('brief', VER, SCEN)
    st.subheader(f"{pd.Timestamp(b['date']):%A %d %B %Y}")
    for n in b['notes']:
        st.info(n, icon=':material/info:')
    for h in b['headlines'][:7]:
        st.markdown(f'- {h}')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**:material/visibility: Stations and windows to watch**')
        st.dataframe(df(b['watch'], ['station', 'window', 'reason']), hide_index=True)
    with c2:
        st.markdown('**:material/group: Additional staff**')
        st.dataframe(df(b['staff_plan'], ['station', 'from', 'to', 'staff']), hide_index=True)
    rein = b.get('reinforcement', {}).get('summary', [])
    if rein:
        st.markdown('**:material/directions_subway: Reinforcement**')
        for line in rein[:4]:
            st.markdown(f'- {line}')
    st.markdown('**Ask a question** (the answer also appears in the Chat tab)')
    cols = st.columns(len(QUICK_QUESTIONS))
    for i, (col, qq) in enumerate(zip(cols, QUICK_QUESTIONS)):
        if col.button(qq, key=f'qq{i}'):
            st.session_state.start_q = qq
    with st.form('start_form', clear_on_submit=True):
        typed = st.text_input('Your question', placeholder='e.g. Which stations will be crowded after the concert tonight?')
        if st.form_submit_button('Ask', icon=':material/send:') and typed.strip():
            st.session_state.start_q = typed.strip()
    q = st.session_state.pop('start_q', None)
    if q:
        full = q if q not in QUICK_QUESTIONS[:2] else f"{q} (service day {pd.Timestamp(day):%d %B %Y})"
        st.session_state.messages.append({'role': 'user', 'content': full})
        msg = ask(full)
        st.session_state.messages.append(msg)
        st.session_state.start_answer = len(st.session_state.messages) - 1
    ia = st.session_state.get('start_answer')
    if ia is not None and ia < len(st.session_state.messages):
        m = st.session_state.messages[ia]
        st.markdown(f"**Q: {st.session_state.messages[ia - 1]['content']}**")
        render_answer(m['content'], m.get('verification'), m.get('trace'), key=f'start{ia}')


PAGES = [('start', ':material/traffic: Start', render_start), ('chat', ':material/chat: Chat', render_chat),
         ('brief', ':material/assignment: Daily brief', render_brief), ('plan', ':material/shield: Action plan', render_plan),
         ('replay', ':material/replay: Replay', render_replay), ('map', ':material/map: Network map', render_map),
         ('compare', ':material/balance: Compare', render_cmp), ('value', ':material/trending_up: Value', render_value),
         ('reports', ':material/description: Reports', render_reports), ('trust', ':material/verified: Trust', render_trust),
         ('audit', ':material/folder_open: Audit', render_audit), ('data', ':material/database: Data', render_data)]
OPERATOR_PAGES = {'start', 'chat', 'data'}
visible = PAGES if expert else [pg for pg in PAGES if pg[0] in OPERATOR_PAGES]
for tab, (_, _, fn) in zip(st.tabs([pg[1] for pg in visible]), visible):
    with tab:
        fn()
