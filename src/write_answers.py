"""Beantwortet die Fragen der Abgabedatei mit dem Agenten und trägt sie ein.

    python -m src.write_answers              # TRAINING (9 Kern- + 2 Bonusfragen)
    python -m src.write_answers --stage FINAL_TEST

Jede Antwort kommt live aus TrainAgent.answer() – Tools plus LLM, dieselbe
Pipeline wie im Operator-Chat. Keine handgeschriebenen Texte: frühere
Handantworten (bis Commit e1c820e) enthielten Zahlen aus älteren Datenständen
und nicht belegbare Ursachen.

Angefasst werden nur TEAM_ANSWER (Spalte D) und TEAM_NAME in der ersten
Antwortzeile. STAGE und QUESTION bleiben unverändert. Zusätzlich landet ein
Protokoll mit Konfidenz, Tools und Antwortzeit je Frage in
evaluation/agent_answers_<stage>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment

import src.legacy_agent as agent_module
from src.legacy_agent import TrainAgent
from src.tools.llm_client import call_llm

# Die Abgabedatei entsteht offline: das 24-s-Limit des Live-Servers (30 s
# Challenge-Limit) gilt hier nicht. Ein Latenz-Ausreisser des Endpoints
# soll nicht die deterministische Notantwort in die Abgabe schreiben.
BATCH_LLM_TIMEOUT_S = 60
MAX_ATTEMPTS = 2

# Konsole auf UTF-8 zwingen - sonst bricht die Ausgabe unter Windows (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "evaluation" / "team_answers_template v1.xlsx"
TEAM_NAME = "A2 - ALSTOM"
COL_TEAM, COL_STAGE, COL_QUESTION, COL_ANSWER = 0, 1, 2, 3
EXCEL_CELL_LIMIT = 32_000


def to_cell_text(markdown: str) -> str:
    """Markdown für eine Excel-Zelle: Fettdruck-Sterne weg, Tabellen bleiben lesbar."""
    text = markdown.replace("**", "")
    text = re.sub(r"^\|[-:| ]+\|$\n?", "", text, flags=re.MULTILINE)  # Trennzeile |---|---:|
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:EXCEL_CELL_LIMIT]


def answer_stage(stage: str) -> tuple[list[dict], int]:
    """Beantwortet alle Fragen einer Stage und schreibt sie in die Vorlage."""
    if not TEMPLATE.is_file():
        raise SystemExit(f"Abgabedatei nicht gefunden: {TEMPLATE}")
    wb = openpyxl.load_workbook(TEMPLATE)
    ws = wb["TEAM_ANSWERS"]
    rows = [row for row in ws.iter_rows(min_row=2) if row[COL_STAGE].value == stage]
    questions = [row for row in rows if str(row[COL_QUESTION].value or "").strip()]
    if not questions:
        raise SystemExit(f"Keine Fragen mit Text in Stage {stage}.")

    agent_module.LLM_TIMEOUT_SECONDS = BATCH_LLM_TIMEOUT_S
    agent = TrainAgent(llm=lambda prompt: call_llm(prompt, timeout=BATCH_LLM_TIMEOUT_S))

    log, wrap = [], Alignment(wrap_text=True, vertical="top")
    for number, row in enumerate(questions, start=1):
        question = str(row[COL_QUESTION].value).strip()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            result = agent.answer(question)
            seconds = round(time.perf_counter() - started, 1)
            # Fehler oder Notantwort ohne LLM: einmal wiederholen.
            if not (result.get("error") or result.get("note")):
                break
            print(f"{stage} {number:2d}: Versuch {attempt} ohne LLM-Antwort "
                  f"({result.get('error') or result.get('note')})")
        cell = row[COL_ANSWER]
        cell.value = to_cell_text(result["answer"])
        cell.alignment = wrap
        log.append({
            "stage": stage, "number": number, "row": cell.row, "question": question,
            "answer": result["answer"], "confidence": result["confidence"],
            "tools_used": result["tools_used"], "data_basis": result["data_basis"],
            "seconds": seconds, "error": result["error"], "note": result.get("note"),
        })
        print(f"{stage} {number:2d}: {seconds:5.1f}s  {result['confidence']:6s}  "
              f"{', '.join(result['tools_used'])}")

    # Laut Vorlage nur in der ersten Antwortzeile; darunter wird geerbt.
    if stage == "TRAINING":
        rows[0][COL_TEAM].value = TEAM_NAME
    wb.save(TEMPLATE)

    out = REPO_ROOT / "evaluation" / f"agent_answers_{stage.lower()}.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template": TEMPLATE.name,
        "answers": log,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ {len(log)} Antworten in {TEMPLATE.name}, Protokoll: {out.relative_to(REPO_ROOT)}")
    return log, len(rows) - len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="TRAINING", choices=["TRAINING", "FINAL_TEST"])
    args = parser.parse_args()
    _, without_text = answer_stage(args.stage)
    if without_text:
        print(f"ℹ  {without_text} Zeile(n) in {args.stage} ohne Fragetext – übersprungen.")


if __name__ == "__main__":
    main()
