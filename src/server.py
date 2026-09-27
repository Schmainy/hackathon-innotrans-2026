"""FastAPI-Server für den Operator-Assistenten "Track Knack".

Start:
    python -m src.server
    uvicorn src.server:app --reload --port 8000

Endpoints:
    GET  /              Chat-Oberfläche für Operatoren (inline, Light Mode),
                        mit Hotspot-Seitenleiste in zwei Akkordeon-Sektionen
    GET  /health        Bereitschaft inkl. Datenverfügbarkeit
    POST /ask           Frage stellen, vollständiges agent.answer()-Dict zurück
    POST /reload        Datenbestand neu prüfen und Engine neu fitten
    GET  /api/hotspots  Hotspots für die Seitenleiste (data/hotspots.json)
    POST /api/forecast, /api/action-plan, /api/staff-plan, /api/brief, /api/messages
                        Operator-Funktionen (ubahn_ops) ohne LLM; mit "question"
                        zusaetzlich eine LLM-Antwort

Merged-Version (Hackathon_2026_JJM): /ask beantwortet Fragen mit Mathews
LLM-Tool-Calling-Agent (src/agent.py, 41 Tools) statt mit dem Keyword-Router;
der Router bleibt als LLM-freie Rueckfallebene in src/legacy_agent.py.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
from pydantic import BaseModel, Field

from src.env_aliases import apply_env_aliases
from src.agent import answer as agent_answer
from src.agent import default_date, get_engine, reload_engine, run_tool, to_markdown
from src.hotspots import build_hotspots
from src.loader import DataLoader

# Immer relativ zur server.py-Datei, egal von wo uvicorn gestartet wird.
# Ein blosses load_dotenv() sucht ab dem Arbeitsverzeichnis aufwaerts und
# findet die .env nicht, wenn der Server aus einem anderen Ordner startet.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)
apply_env_aliases()  # AZURE_OPENAI_* aus .env.example -> Namen, die der Code liest

# Fail fast: fehlende Credentials sollen beim Start auffallen, nicht erst
# bei der ersten Operator-Frage waehrend der Bewertung.
_REQUIRED_ENV = ["AZURE_API_KEY", "AZURE_ENDPOINT", "MODEL_NAME"]
if not os.getenv("MODEL_NAME") and os.getenv("AZURE_MODEL"):
    os.environ["MODEL_NAME"] = os.environ["AZURE_MODEL"]  # Name aus der alten final_InnoTrans-.env
_missing = [key for key in _REQUIRED_ENV if not os.getenv(key)]
if _missing:
    raise RuntimeError(
        f"Fehlende Umgebungsvariablen: {_missing}. "
        f".env gefunden: {_ENV_PATH.exists()} ({_ENV_PATH})"
    )

# 28s: 2s Puffer unter dem 30s-Limit der Challenge. Der Agent selbst hat ein
# Budget von 26s (AGENT_TIMEOUT_S) und antwortet danach per Fallback aus den
# bis dahin gesammelten Tool-Ergebnissen.
REQUEST_TIMEOUT_S = 28

app = FastAPI(
    title="Track Knack – Operator Assistant",
    description="KI-Agent für Berliner U-Bahn-Operatoren (Alstom Challenge, InnoTrans 2026)",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# answer() ist blockierend (pandas/networkx). Der Executor haelt den
# Event-Loop frei, damit /health auch waehrend einer laufenden Frage antwortet.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.on_event("startup")
async def _warm_engine() -> None:
    """Engine einmal beim Start fitten (~5s), nicht bei der ersten Frage."""
    await asyncio.get_running_loop().run_in_executor(_executor, get_engine)


@app.post("/ask")
async def ask(payload: AskRequest) -> dict[str, Any]:
    """Beantwortet eine Operator-Frage; 504 wenn die Bearbeitung zu lange läuft.

    Keine Vorfilterung der Frage: die offiziellen Bewertungsfragen buendeln
    mehrere Teilfragen in einem Satz, eine Mehrfachfragen-Sperre wuerde sie
    abweisen. Gegen ueberlange Laeufe schuetzt allein REQUEST_TIMEOUT_S.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, agent_answer, payload.question),
            timeout=REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Zeitlimit von {REQUEST_TIMEOUT_S}s überschritten. "
                "Bitte die Frage enger fassen (Station, Datum, Zeitfenster)."
            ),
        ) from None


@app.get("/health")
async def health() -> dict[str, Any]:
    """Bereitschaftscheck inklusive Zugriff auf die Rohdaten und Datenzeitraum."""
    loaded = True
    detail = None
    first_day = last_day = None
    try:
        loader = DataLoader()
        stations = loader.load_stations()
        detail = f"{len(stations)} Stationen"
        # Gecacht je Dateistand – nur der erste Aufruf nach neuen Dateien
        # liest die Flows.
        first_day = loader.data_first_day
        last_day = loader.data_last_day
    except Exception as exc:  # noqa: BLE001 – Health darf nie 500 werfen
        loaded = False
        detail = f"{type(exc).__name__}: {exc}"

    engine: dict[str, Any] = {"ready": False}
    try:
        eng = get_engine()
        innotrans = eng.events[eng.events.event_name.astype(str).str.contains("InnoTrans", case=False)]
        engine = {"ready": True, "events": int(len(eng.events)),
                  "innotrans_days": sorted(str(d.date()) for d in innotrans.start),
                  "flows_until": str(eng.flows.index.max())}
    except Exception as exc:  # noqa: BLE001 – Health darf nie 500 werfen
        engine["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok" if loaded and engine["ready"] else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_loaded": loaded,
        "detail": detail,
        "data_first_day": first_day,
        "data_last_day": last_day,
        "engine": engine,
    }


@app.post("/reload")
async def reload_data() -> Any:
    """Prüft und meldet den Datenbestand nach dem Ablegen neuer CSV-Dateien.

    Der Loader hält keinen Zustand – jede Frage liest den aktuellen
    Dateistand. /reload lädt trotzdem alle Datensätze einmal vollständig,
    damit ein kaputtes Dateiformat hier auffällt und nicht erst bei der
    ersten Operator-Frage.
    """
    loader = DataLoader()
    try:
        data = await asyncio.get_running_loop().run_in_executor(
            _executor, loader.load_all
        )
        dates = loader.data_dates()
        await asyncio.get_running_loop().run_in_executor(_executor, reload_engine)
    except Exception as exc:  # noqa: BLE001 – Fehler als JSON, nicht als 500-HTML
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": f"{type(exc).__name__}: {exc}",
                     "files_found": loader.files_used},
        )

    flows = data["flows"]
    return {
        "status": "reloaded",
        "data_first_day": dates["first_calendar_day"],
        "data_first_full_day": dates["first_day"],
        "data_last_day": dates["last_day"],
        "flow_rows": int(flows.shape[0]),
        "flow_columns": int(flows.shape[1]),
        "rows": {key: int(len(df)) for key, df in data.items()},
        "files": loader.files_used,
        "message": (
            f"Data reloaded: {flows.shape[0]} slots x {flows.shape[1]} stations, "
            f"{dates['first_calendar_day']} to {dates['last_day']} "
            "(last full operating day)"
        ),
    }


# ---------------------------------------------------------------------------
# Operator-Funktionen (Mathews ubahn_ops): deterministisch, ohne LLM, < 2s.
# Body: {"question": "...", "context": {...}}. "context" enthaelt die Tool-
# Argumente (z. B. date, stations, start, end, tmean, prcp, extra_events,
# closure, event); fehlende Werte werden sinnvoll vorbelegt (Datum: heute,
# falls in den Daten, sonst letzter Datentag). Mit "question" kommt
# zusaetzlich eine LLM-Antwort ("answer") dazu.
# ---------------------------------------------------------------------------
class OpRequest(BaseModel):
    question: str = Field("", max_length=2000)
    context: dict[str, Any] = Field(default_factory=dict)


def _ctx_date(ctx: dict[str, Any]) -> str:
    return str(ctx.get("date") or default_date())


def _scenario(ctx: dict[str, Any]) -> dict[str, Any]:
    return {k: ctx[k] for k in ("tmean", "prcp", "extra_events", "extra_closures") if ctx.get(k) is not None}


def _args_brief(ctx: dict[str, Any]) -> dict[str, Any]:
    return {"date": _ctx_date(ctx), "staff": int(ctx.get("staff", 10)), **_scenario(ctx)}


def _args_action_plan(ctx: dict[str, Any]) -> dict[str, Any]:
    return {"date": _ctx_date(ctx), "top": int(ctx.get("top", 6)), **_scenario(ctx)}


def _args_staff_plan(ctx: dict[str, Any]) -> dict[str, Any]:
    return {"date": _ctx_date(ctx), "staff": int(ctx.get("staff", 10)),
            "block_hours": float(ctx.get("block_hours", 2.0)), **_scenario(ctx)}


def _args_forecast(ctx: dict[str, Any]) -> dict[str, Any]:
    """Ohne Stationen/Fenster: das erste Beobachtungsfenster des Aktionsplans und alle
    Stationen, die im selben Fenster beobachtet werden."""
    args = {k: ctx[k] for k in ("start", "end", "stations", "line") if ctx.get(k)}
    if not ("start" in args and "end" in args and ("stations" in args or "line" in args)):
        date = _ctx_date(ctx)
        watch = run_tool("ubahn_action_plan", {"date": date, "top": 6, **_scenario(ctx)}).get("watch", [])
        if watch:
            window = watch[0]["window"]
            same = [w["station"] for w in watch if w["window"] == window][:4]
            args.setdefault("stations", list(dict.fromkeys(same)))
            args.setdefault("start", f"{date} {window.split('-')[0]}")
            args.setdefault("end", f"{date} {window.split('-')[1]}")
        else:
            args.setdefault("start", f"{date} 07:00")
            args.setdefault("end", f"{date} 10:00")
            args.setdefault("line", "U2")
    return {**args, **_scenario(ctx)}


def _args_messages(ctx: dict[str, Any]) -> dict[str, Any]:
    """Ohne Angabe: erste Sperrung des Tages, sonst das groesste Event des Tages."""
    keys = ("closure", "start", "end", "event", "date", "venue", "event_start", "event_end")
    args = {k: ctx[k] for k in keys if ctx.get(k)}
    if not any(k in args for k in ("closure", "event", "venue")):
        eng = get_engine()
        day = pd.Timestamp(_ctx_date(ctx))
        args["date"] = str(day.date())
        closures = [c for c in eng._clist if c["start"].normalize() == day]
        events = eng.events[eng.events.start.dt.normalize() == day]
        if closures:
            args["closure"] = closures[0]["description"]
        elif len(events):
            args["event"] = str(events.sort_values("estimated_attendance").iloc[-1].event_name)
    return args


_OPS: dict[str, tuple[str, Any]] = {
    "forecast": ("ubahn_forecast", _args_forecast),
    "action-plan": ("ubahn_action_plan", _args_action_plan),
    "staff-plan": ("ubahn_staff_plan", _args_staff_plan),
    "brief": ("ubahn_daily_brief", _args_brief),
    "messages": ("ubahn_passenger_messages", _args_messages),
}


def _run_op(kind: str, payload: OpRequest) -> dict[str, Any]:
    tool, build = _OPS[kind]
    t0 = datetime.now(timezone.utc)
    ctx = payload.context or {}
    try:
        args = build(ctx)
        result = run_tool(tool, args)
    except Exception as exc:  # noqa: BLE001 – Fehler als JSON
        return {"status": "error", "endpoint": kind, "tool": tool, "error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"status": "ok", "endpoint": kind, "tool": tool, "args": args,
                           "markdown": to_markdown(tool, result), "result": result}
    if payload.question.strip():
        hint = (f"{payload.question.strip()}\n\n(Context from the operator panel: use {tool} with "
                f"{json.dumps(args, ensure_ascii=False)})")
        out["llm"] = agent_answer(hint)
        out["answer"] = out["llm"].get("answer")
    out["seconds"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    return out


async def _op_endpoint(kind: str, payload: OpRequest) -> Any:
    loop = asyncio.get_running_loop()
    try:
        res = await asyncio.wait_for(loop.run_in_executor(_executor, _run_op, kind, payload),
                                     timeout=REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Zeitlimit von {REQUEST_TIMEOUT_S}s überschritten.") from None
    return res if res.get("status") == "ok" else JSONResponse(status_code=422, content=res)


@app.post("/api/forecast")
async def api_forecast(payload: OpRequest) -> Any:
    """Prognose je 15-min-Slot mit kalibrierten 80/90-%-Bereichen."""
    return await _op_endpoint("forecast", payload)


@app.post("/api/action-plan")
async def api_action_plan(payload: OpRequest) -> Any:
    """Beobachtungsfenster mit Triggern (prepare/surge/near capacity/closure)."""
    return await _op_endpoint("action-plan", payload)


@app.post("/api/staff-plan")
async def api_staff_plan(payload: OpRequest) -> Any:
    """Wo und wann N zusaetzliche Mitarbeitende eingesetzt werden."""
    return await _op_endpoint("staff-plan", payload)


@app.post("/api/brief")
async def api_brief(payload: OpRequest) -> Any:
    """Tagesbriefing: Wetter, Events, Sperrungen, Risiken, Personal, Trigger."""
    return await _op_endpoint("brief", payload)


@app.post("/api/messages")
async def api_messages(payload: OpRequest) -> Any:
    """Fahrgastdurchsagen DE/EN/FR fuer eine Sperrung oder ein Event."""
    return await _op_endpoint("messages", payload)


HOTSPOTS_FILE = Path(__file__).resolve().parent.parent / "data" / "hotspots.json"


@app.get("/api/hotspots")
async def get_hotspots() -> dict[str, Any]:
    """Hotspots für die Seitenleiste: {"hotspots": [{id, label, section, auto_query, ...}]}.

    Eine kuratierte data/hotspots.json hat Vorrang. Ohne sie werden die
    Hotspots aus Events und Flow-Daten abgeleitet (src/hotspots.py); die
    Oberfläche leitet dann die Sektion aus der Kategorie ab.
    """
    if HOTSPOTS_FILE.is_file():
        try:
            return json.loads(HOTSPOTS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # defekte Datei: auf die abgeleiteten Hotspots zurueckfallen
    return await asyncio.get_running_loop().run_in_executor(_executor, build_hotspots)


# Hackathon-Logo (docs/pics/b1d2d5cc-...jpg), verkleinert eingebettet: das
# Original hat 133 KB, angezeigt wird es nur mit 36 px (Header) bzw. als
# 28-px-Avatar. LOGO_B64 = ganzes Logo (144x72), AVATAR_B64 = quadratischer
# Ausschnitt der Qualle (96x96) – im Kreis waere sonst nur Schrift zu sehen.
LOGO_B64 = "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"
AVATAR_B64 = "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"

_INDEX_TEMPLATE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Track Knack</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #F7F7F8;
    --panel-bg: #FFFFFF;
    --border: #E5E5E6;
    --text-primary: #0D0D0D;
    --text-muted: #6B6B6B;
    --accent: #0066CC;
    --accent-hover: #0052A3;
    --accent-soft: #F0F4FF;
    --user-bubble: #0066CC;
    --user-text: #FFFFFF;
    --ai-bubble: #FFFFFF;
    --input-bg: #FFFFFF;
    --shadow: 0 1px 3px rgba(0,0,0,0.08);
    --header-h: 56px;
    color-scheme: light;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
  }
  button { font: inherit; }
  [hidden] { display: none !important; }

  /* ---------------- Layout ---------------- */
  .app {
    display: grid;
    grid-template-columns: 260px 1px minmax(0, 1fr);
    height: 100vh; height: 100dvh;
  }
  .divider { background: var(--border); }

  .brand {
    height: var(--header-h); flex: 0 0 var(--header-h);
    display: flex; align-items: center; gap: 10px;
    padding: 0 16px; background: var(--panel-bg);
    border-bottom: 1px solid var(--border);
  }
  /* Das Logo steht nur noch im Chat-Header. */
  .chat > .brand > img { height: 36px; width: auto; border-radius: 6px; display: block; }
  .brand-title { font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
  .brand-sub { font-size: 12px; color: var(--text-muted); }
  .brand-text { display: flex; align-items: baseline; gap: 8px; min-width: 0; }

  /* ---------------- Sidebar ---------------- */
  .sidebar {
    background: var(--panel-bg);
    display: flex; flex-direction: column;
    height: 100%; min-height: 0; overflow: hidden;
  }
  .sidebar-brand {
    padding: 0 16px;
    height: var(--header-h);
    display: flex;
    align-items: center;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-team-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.02em;
  }
  .sidebar-body { flex: 1 1 auto; overflow-y: auto; padding: 8px 0 16px; }

  /* Sidebar-Footer: ALSTOM-Partnerhinweis */
  .sidebar-footer {
    flex: 0 0 auto;
    padding: 12px 16px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .alstom-logo {
    height: 24px;
    width: auto;
    flex: 0 0 auto;
    display: block;
  }
  .alstom-text {
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  .accordion + .accordion { border-top: 1px solid var(--border); }
  .accordion-head {
    width: 100%; background: none; border: 0; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px 8px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    color: var(--text-muted); text-transform: uppercase; text-align: left;
  }
  .accordion-head:hover { color: var(--text-primary); }
  .accordion-head:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .chevron {
    display: inline-block; font-size: 16px; line-height: 1;
    transition: transform 0.15s ease;
  }
  .accordion.open .chevron { transform: rotate(90deg); }
  .accordion-body { padding: 0 8px 8px; display: none; }
  .accordion.open .accordion-body { display: block; }
  .accordion-count { font-weight: 400; margin-left: 6px; letter-spacing: 0; }

  .hs-btn {
    display: block; width: 100%; text-align: left; cursor: pointer;
    padding: 8px 12px; border-radius: 6px; border: 0; box-shadow: none;
    background: transparent; color: var(--text-primary);
    font-size: 13px; font-weight: 400; line-height: 1.35;
    transition: background 0.15s, color 0.15s;
  }
  .hs-btn:hover:not(:disabled) { background: var(--accent-soft); color: var(--accent); }
  .hs-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .hs-btn.flash, .hs-btn.flash:hover { background: var(--accent); color: #fff; }
  .hs-btn:disabled { cursor: default; opacity: 0.55; }
  .hs-empty { padding: 8px 16px; font-size: 13px; color: var(--text-muted); }

  /* ---------------- Chat ---------------- */
  .chat {
    display: grid; grid-template-rows: var(--header-h) minmax(0, 1fr) auto;
    min-width: 0; min-height: 0; background: var(--bg);
  }
  .burger {
    display: none; width: 36px; height: 36px; margin-left: -6px;
    border: 0; border-radius: 8px; background: none; cursor: pointer;
    color: var(--text-primary);
    align-items: center; justify-content: center;
  }
  .burger:hover { background: var(--bg); }

  .messages {
    overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .row { display: flex; max-width: 900px; width: 100%; margin: 0 auto; }
  .row.user { justify-content: flex-end; }
  .row.ai { gap: 10px; align-items: flex-start; }

  .bubble-user {
    max-width: 75%;
    background: var(--user-bubble); color: var(--user-text);
    border-radius: 16px 16px 4px 16px; padding: 10px 14px; font-size: 14px;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .avatar {
    width: 28px; height: 28px; flex: 0 0 28px; border-radius: 50%;
    object-fit: cover; margin-top: 2px; border: 1px solid var(--border);
  }
  .ai-col { display: flex; flex-direction: column; max-width: 85%; min-width: 0; }
  .bubble-ai {
    background: var(--ai-bubble); border: 1px solid var(--border);
    border-radius: 4px 16px 16px 16px; padding: 12px 16px; font-size: 14px;
    box-shadow: var(--shadow); overflow-wrap: anywhere; min-width: 0;
  }
  .bubble-ai.plain { white-space: pre-wrap; }
  .meta {
    font-size: 11px; font-style: italic; color: var(--text-muted);
    margin-top: 4px; padding-left: 2px;
  }

  /* Markdown im AI-Bubble */
  .bubble-ai > :first-child { margin-top: 0; }
  .bubble-ai > :last-child { margin-bottom: 0; }
  .bubble-ai p { margin: 0 0 8px; }
  .bubble-ai ul, .bubble-ai ol { margin: 0 0 8px; padding-left: 20px; }
  .bubble-ai li { margin: 2px 0; }
  .bubble-ai strong { font-weight: 600; }
  .bubble-ai h1, .bubble-ai h2, .bubble-ai h3, .bubble-ai h4 {
    font-size: 14px; font-weight: 600; margin: 12px 0 6px;
  }
  .bubble-ai table {
    border-collapse: collapse; margin: 4px 0 10px; font-size: 13px;
    display: block; max-width: 100%; overflow-x: auto;
  }
  .bubble-ai th, .bubble-ai td {
    border: 1px solid var(--border); padding: 5px 9px; text-align: left;
    vertical-align: top;
  }
  .bubble-ai th { background: var(--bg); font-weight: 600; }
  .bubble-ai pre {
    margin: 4px 0 10px; padding: 10px 12px; overflow-x: auto;
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    font: 12px/1.5 ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  .bubble-ai code { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.5px; }
  .bubble-ai a { color: var(--accent); }
  .bubble-ai hr { border: 0; border-top: 1px solid var(--border); margin: 10px 0; }

  /* Typing indicator */
  .typing { display: inline-flex; gap: 5px; padding: 4px 2px; }
  .typing span {
    width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted);
    opacity: 0.35; animation: pulse 1.2s ease-in-out infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes pulse {
    0%, 80%, 100% { opacity: 0.35; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-2px); }
  }

  /* ---------------- Input ---------------- */
  .composer {
    background: #fff; border-top: 1px solid var(--border); padding: 12px 16px;
  }
  .composer form { position: relative; max-width: 900px; margin: 0 auto; }
  #q {
    display: block; width: 100%; resize: none; overflow-y: hidden;
    min-height: 40px; max-height: 144px; height: 40px;
    padding: 10px 44px 10px 14px;
    font: 14px/20px system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary); background: var(--input-bg);
    border: 1px solid var(--border); border-radius: 12px; outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  #q:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,102,204,0.15); }
  #q::placeholder { color: var(--text-muted); }
  #send {
    position: absolute; right: 4px; bottom: 4px;
    width: 32px; height: 32px; border-radius: 50%; border: 0; cursor: pointer;
    background: var(--accent); color: #fff;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s, opacity 0.15s;
  }
  #send:hover:not(:disabled) { background: var(--accent-hover); }
  #send:disabled { opacity: 0.4; cursor: default; }

  .backdrop { display: none; }

  /* ---------------- Mobile ---------------- */
  @media (max-width: 767.98px) {
    .app { grid-template-columns: minmax(0, 1fr); }
    .divider { display: none; }
    .sidebar {
      position: fixed; top: 0; left: 0; bottom: 0; z-index: 40;
      width: min(280px, 85vw); box-shadow: 4px 0 24px rgba(0,0,0,0.12);
      transform: translateX(-100%); transition: transform 0.25s ease;
    }
    .app.drawer-open .sidebar { transform: none; }
    .backdrop {
      display: block; position: fixed; inset: 0; z-index: 30;
      background: rgba(0,0,0,0.3); opacity: 0; pointer-events: none;
      transition: opacity 0.25s ease;
    }
    .app.drawer-open .backdrop { opacity: 1; pointer-events: auto; }
    .burger { display: inline-flex; }
    .messages { padding: 16px 12px; }
    .bubble-user { max-width: 85%; }
    .ai-col { max-width: calc(100% - 38px); }
  }
  @media (prefers-reduced-motion: reduce) {
    .typing span, .sidebar, .backdrop, .chevron { animation: none; transition: none; }
  }
  /* ?embed=1 (Iframe in der Streamlit-App): Kopfzeile und ALSTOM-Footer
     liefert die umgebende Seite. Mobil bleibt der Burger fuer die Hotspots. */
  .embed .sidebar-brand, .embed .sidebar-footer,
  .embed .brand-logo, .embed .brand-text { display: none; }
  @media (min-width: 768px) {
    .embed .chat > header.brand { display: none; }
    .embed .chat { grid-template-rows: minmax(0, 1fr) auto; }
  }
</style>
</head>
<body>
<div class="app" id="app">

  <aside class="sidebar" id="sidebar" aria-label="Hotspots">
    <div class="brand sidebar-brand">
      <span class="sidebar-team-name">Track Knack</span>
    </div>
    <div class="sidebar-body" id="hotspots">
      <div class="hs-empty">Hotspots werden geladen…</div>
    </div>
    <footer class="sidebar-footer">
      <span class="alstom-text">
        Developed in the context of the<br>
        InnoTrans 2026 Hackathon in<br>
        collaboration with ALSTOM
      </span>
    </footer>
  </aside>
  <div class="divider" aria-hidden="true"></div>
  <div class="backdrop" id="backdrop"></div>

  <main class="chat">
    <header class="brand">
      <button type="button" class="burger" id="burger" aria-label="Hotspots öffnen"
              aria-controls="sidebar" aria-expanded="false">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
      </button>
      <img class="brand-logo" src="data:image/jpeg;base64,__LOGO_B64__" alt="Hackathon 2026 InnoTrans">
      <div class="brand-text">
        <span class="brand-title">Track Knack</span>
        <span class="brand-sub">InnoTrans Hackathon 2026</span>
      </div>
    </header>

    <div class="messages" id="messages" aria-live="polite"></div>

    <div class="composer">
      <form id="form">
        <textarea id="q" rows="1" placeholder="Frage stellen..." aria-label="Frage"></textarea>
        <button type="submit" id="send" aria-label="Senden" disabled>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M8 13V3M8 3L3.5 7.5M8 3l4.5 4.5" stroke="currentColor" stroke-width="1.8"
                  stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </form>
    </div>
  </main>
</div>

<script>
(function () {
  "use strict";

  var AVATAR = "data:image/jpeg;base64,__AVATAR_B64__";
  var MAX_INPUT_H = 144;
  var SECTIONS = [
    { key: "transit", title: "Standard Traffic Problems" },
    { key: "places", title: "Event Places" }
  ];
  // Kurznamen fuer die Metadatenzeile; unbekannte Tools ohne get_/find_-Praefix.
  var TOOL_NAMES = {
    get_station_flow: "flow", get_network_flow_summary: "flow",
    get_station_peak_15min: "peak", get_station_peak_vs_baseline: "peak",
    get_events: "events", get_closures: "closures", get_closure_impact: "closures",
    get_weather: "weather", find_weather_anomalies: "weather", get_rain_impact: "weather",
    get_energy: "energy", get_critical_stations: "criticality",
    find_alternative_routes: "routing", find_transit_route: "routing",
    find_diverse_transit_routes: "routing", hotspot_alternatives: "hotspots",
    detect_anomalies: "anomalies"
  };

  var app = document.getElementById("app");
  var messages = document.getElementById("messages");
  var form = document.getElementById("form");
  var input = document.getElementById("q");
  var send = document.getElementById("send");
  var hsBox = document.getElementById("hotspots");
  var burger = document.getElementById("burger");
  var backdrop = document.getElementById("backdrop");
  var mobile = window.matchMedia("(max-width: 767.98px)");
  var busy = false;

  // ---------------- Rendering ----------------
  function scrollDown() { messages.scrollTop = messages.scrollHeight; }

  function renderMarkdown(el, text) {
    if (window.marked && typeof window.marked.parse === "function") {
      // Rohes HTML aus der Antwort nicht ausfuehren.
      el.innerHTML = window.marked.parse(String(text).replace(/</g, "&lt;"),
                                         { gfm: true, breaks: true });
    } else {
      // CDN nicht erreichbar (Messe-WLAN): Klartext statt leerer Blase.
      el.classList.add("plain");
      el.textContent = text;
    }
  }

  function addUser(text) {
    var row = document.createElement("div");
    row.className = "row user";
    var b = document.createElement("div");
    b.className = "bubble-user";
    b.textContent = text;
    row.appendChild(b);
    messages.appendChild(row);
    scrollDown();
  }

  function addAi() {
    var row = document.createElement("div");
    row.className = "row ai";
    var img = document.createElement("img");
    img.className = "avatar";
    img.src = AVATAR;
    img.alt = "";
    var col = document.createElement("div");
    col.className = "ai-col";
    var b = document.createElement("div");
    b.className = "bubble-ai";
    col.appendChild(b);
    row.appendChild(img);
    row.appendChild(col);
    messages.appendChild(row);
    scrollDown();
    return { bubble: b, col: col };
  }

  function typingBubble() {
    var ai = addAi();
    ai.bubble.innerHTML = '<span class="typing" aria-label="Antwort wird erstellt">' +
                          '<span></span><span></span><span></span></span>';
    return ai;
  }

  function toolLabel(name) {
    return TOOL_NAMES[name] ||
      String(name).replace(/^(get|find|ubahn)_/, "").replace(/_/g, " ").toLowerCase();
  }

  function metaLine(data, secs) {
    var tools = [];
    (data.tools_used || []).forEach(function (t) {
      var label = toolLabel(t);
      if (tools.indexOf(label) === -1) { tools.push(label); }
    });
    var conf = String(data.confidence || "none");
    conf = conf.charAt(0).toUpperCase() + conf.slice(1).toLowerCase();
    return "Tools: " + (tools.length ? tools.join(", ") : "–") +
           " · " + secs + "s · Confidence: " + conf;
  }

  // ---------------- Input ----------------
  function autoGrow() {
    input.style.height = "auto";
    var h = Math.min(Math.max(input.scrollHeight + 2, 40), MAX_INPUT_H);
    input.style.height = h + "px";
    input.style.overflowY = input.scrollHeight + 2 > MAX_INPUT_H ? "auto" : "hidden";
  }

  function updateSend() {
    send.disabled = busy || !input.value.trim();
  }

  function setBusy(state) {
    busy = state;
    updateSend();
    var btns = hsBox.querySelectorAll(".hs-btn");
    for (var i = 0; i < btns.length; i++) { btns[i].disabled = state; }
  }

  async function ask(question) {
    question = (question || "").trim();
    if (!question || busy) { return; }
    addUser(question);
    input.value = "";
    autoGrow();
    setBusy(true);
    var ai = typingBubble();
    var t0 = performance.now();
    try {
      var res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
      });
      var data = await res.json();
      var secs = ((performance.now() - t0) / 1000).toFixed(1);
      if (res.ok) {
        renderMarkdown(ai.bubble, data.answer || "(leere Antwort)");
        var meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = metaLine(data, secs);
        ai.col.appendChild(meta);
      } else {
        ai.bubble.classList.add("plain");
        ai.bubble.textContent = "Anfrage fehlgeschlagen (HTTP " + res.status + "): " +
                                (data.detail || "unbekannter Fehler");
      }
    } catch (err) {
      ai.bubble.classList.add("plain");
      ai.bubble.textContent = "Netzwerkfehler: " + err;
    } finally {
      setBusy(false);
      scrollDown();
      if (!mobile.matches) { input.focus(); }
    }
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    ask(input.value);
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      ask(input.value);
    }
  });
  input.addEventListener("input", function () { autoGrow(); updateSend(); });

  // ---------------- Sidebar ----------------
  function sectionOf(h) {
    if (h.section === "transit" || h.section === "places") { return h.section; }
    return h.category === "transit_hub" ? "transit" : "places";
  }

  function setDrawer(open) {
    app.classList.toggle("drawer-open", open);
    burger.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function hotspotButton(h) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "hs-btn";
    btn.textContent = h.label || h.id;
    btn.title = h.auto_query || "";
    btn.addEventListener("click", function () {
      if (busy) { return; }
      btn.classList.add("flash");
      setTimeout(function () { btn.classList.remove("flash"); }, 150);
      if (mobile.matches) { setDrawer(false); }
      ask(h.auto_query || h.label);
    });
    return btn;
  }

  // ---------------- Operator Tools (Mathews ubahn_ops, ohne LLM) ----------------
  var OPS = [
    { path: "/api/brief", label: "Daily brief" },
    { path: "/api/action-plan", label: "Action plan & triggers" },
    { path: "/api/forecast", label: "Forecast with ranges" },
    { path: "/api/staff-plan", label: "Staff plan" },
    { path: "/api/messages", label: "Passenger messages (DE/EN/FR)" }
  ];

  async function runOp(op) {
    if (busy) { return; }
    addUser(op.label);
    setBusy(true);
    var ai = typingBubble();
    var t0 = performance.now();
    try {
      var res = await fetch(op.path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "", context: {} })
      });
      var data = await res.json();
      var secs = ((performance.now() - t0) / 1000).toFixed(1);
      if (res.ok) {
        renderMarkdown(ai.bubble, data.markdown || "(leere Antwort)");
        var meta = document.createElement("div");
        meta.className = "meta";
        var day = data.args && (data.args.date || String(data.args.start || "").slice(0, 10));
        meta.textContent = "Tool: " + toolLabel(data.tool) + (day ? " · " + day : "") +
                           " · " + secs + "s · computed, no LLM";
        ai.col.appendChild(meta);
      } else {
        ai.bubble.classList.add("plain");
        ai.bubble.textContent = "Anfrage fehlgeschlagen (HTTP " + res.status + "): " +
                                (data.error || data.detail || "unbekannter Fehler");
      }
    } catch (err) {
      ai.bubble.classList.add("plain");
      ai.bubble.textContent = "Netzwerkfehler: " + err;
    } finally {
      setBusy(false);
      scrollDown();
    }
  }

  function renderOps() {
    var acc = document.createElement("section");
    acc.className = "accordion open";         // Operator Tools starten aufgeklappt
    var head = document.createElement("button");
    head.type = "button";
    head.className = "accordion-head";
    head.id = "acc-head-ops";
    head.setAttribute("aria-expanded", "true");
    head.setAttribute("aria-controls", "acc-body-ops");
    head.innerHTML = '<span></span><span class="chevron" aria-hidden="true">›</span>';
    head.firstChild.textContent = "Operator Tools";
    var count = document.createElement("span");
    count.className = "accordion-count";
    count.textContent = "(" + OPS.length + ")";
    head.firstChild.appendChild(count);
    var body = document.createElement("div");
    body.className = "accordion-body";
    body.id = "acc-body-ops";
    body.setAttribute("role", "region");
    body.setAttribute("aria-labelledby", head.id);
    OPS.forEach(function (op) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "hs-btn";
      btn.textContent = op.label;
      btn.title = "POST " + op.path;
      btn.addEventListener("click", function () {
        if (busy) { return; }
        if (mobile.matches) { setDrawer(false); }
        runOp(op);
      });
      body.appendChild(btn);
    });
    head.addEventListener("click", function () {
      var open = acc.classList.toggle("open");
      head.setAttribute("aria-expanded", open ? "true" : "false");
    });
    acc.appendChild(head);
    acc.appendChild(body);
    hsBox.appendChild(acc);
  }

  function renderHotspots(list) {
    hsBox.textContent = "";
    renderOps();
    if (!list.length) {
      var empty = document.createElement("div");
      empty.className = "hs-empty";
      empty.textContent = "Keine Hotspots verfügbar.";
      hsBox.appendChild(empty);
      return;
    }
    SECTIONS.forEach(function (sec, i) {
      var items = list.filter(function (h) { return sectionOf(h) === sec.key; });
      if (!items.length) { return; }
      var acc = document.createElement("section");
      acc.className = "accordion";           // startet eingeklappt
      var head = document.createElement("button");
      head.type = "button";
      head.className = "accordion-head";
      head.id = "acc-head-" + i;
      head.setAttribute("aria-expanded", "false");
      head.setAttribute("aria-controls", "acc-body-" + i);
      head.innerHTML = '<span></span><span class="chevron" aria-hidden="true">›</span>';
      head.firstChild.textContent = sec.title;
      var count = document.createElement("span");
      count.className = "accordion-count";
      count.textContent = "(" + items.length + ")";
      head.firstChild.appendChild(count);
      var body = document.createElement("div");
      body.className = "accordion-body";
      body.id = "acc-body-" + i;
      body.setAttribute("role", "region");
      body.setAttribute("aria-labelledby", head.id);
      items.forEach(function (h) { body.appendChild(hotspotButton(h)); });
      head.addEventListener("click", function () {
        var open = acc.classList.toggle("open");
        head.setAttribute("aria-expanded", open ? "true" : "false");
      });
      acc.appendChild(head);
      acc.appendChild(body);
      hsBox.appendChild(acc);
    });
    setBusy(busy);
  }

  fetch("/api/hotspots")
    .then(function (r) { return r.ok ? r.json() : { hotspots: [] }; })
    .then(function (d) { renderHotspots((d && d.hotspots) || []); })
    .catch(function () { renderHotspots([]); });

  burger.addEventListener("click", function () { setDrawer(!app.classList.contains("drawer-open")); });
  backdrop.addEventListener("click", function () { setDrawer(false); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && app.classList.contains("drawer-open")) { setDrawer(false); burger.focus(); }
  });
  mobile.addEventListener("change", function (e) { if (!e.matches) { setDrawer(false); } });

  // ---------------- Start ----------------
  var welcome = addAi();
  renderMarkdown(welcome.bubble,
    "Hello, I'm **Track Knack** — your AI assistant for Berlin U-Bahn operations.\n\n" +
    "Ask me about passenger flows, disruptions, events, weather impacts, or energy — " +
    "or select a hotspot from the left panel.");
  autoGrow();
  if (!mobile.matches) { input.focus(); }
})();
</script>
</body>
</html>
"""

INDEX_HTML = (
    _INDEX_TEMPLATE
    .replace("__LOGO_B64__", LOGO_B64)
    .replace("__AVATAR_B64__", AVATAR_B64)
)


INDEX_HTML_EMBED = INDEX_HTML.replace('<html lang="de">', '<html lang="de" class="embed">', 1)


@app.get("/", response_class=HTMLResponse)
async def index(embed: str | None = None) -> HTMLResponse:
    """Operator-Chatoberfläche; CSS und JS inline, nur marked.js vom CDN.

    ?embed=1 blendet Kopfzeile und ALSTOM-Footer aus (Iframe in app.py).
    """
    return HTMLResponse(INDEX_HTML_EMBED if embed == "1" else INDEX_HTML)


if __name__ == "__main__":
    # "python -m src.server" – gleichwertig zu "uvicorn src.server:app --port 8000".
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
