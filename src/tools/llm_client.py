"""Azure-Endpoint-Client.

call_llm  schickt einen Prompt an den Azure-OpenAI-Endpoint

Bewusst nur mit urllib aus der Standardbibliothek – keine zusätzliche
HTTP-Abhängigkeit im heissen Pfad. Credentials kommen ausschliesslich aus
os.environ (befüllt via python-dotenv aus .env im Repo-Root).
"""

from __future__ import annotations

import json
import os
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.env_aliases import apply_env_aliases

# Wird nur ergaenzt, wenn AZURE_ENDPOINT die reine Ressourcen-URL ist
# (https://<resource>.openai.azure.com/) statt der vollen Responses-URL.
RESPONSES_PATH = "/openai/responses?api-version=2025-04-01-preview"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = REPO_ROOT / ".env"

# 25s statt 30s: 5 Sekunden Puffer vor dem Antwortzeit-Limit der Challenge.
DEFAULT_TIMEOUT = 25

# Stabiler Cache-Schluessel fuer den unveraenderlichen System-Prompt.
# Bei jeder Aenderung am SYSTEM_PROMPT die Version hochzaehlen.
PROMPT_CACHE_KEY = "talk-to-my-train-system-v5"


class LLMError(RuntimeError):
    """Fehler beim Aufruf des LLM-Endpoints."""


class LLMConfigError(LLMError, ValueError):
    """Fehlende oder leere Credentials.

    Erbt bewusst von LLMError UND ValueError: bestehende
    ``except LLMError``-Pfade greifen weiter, und der Aufrufer kann den
    Konfigurationsfehler gezielt als ValueError von einem Transportfehler
    unterscheiden.
    """


def _load_env() -> None:
    """Lädt .env in os.environ, ohne bereits gesetzte Werte zu überschreiben."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_fallback()
    else:
        load_dotenv(ENV_PATH, override=False)
    apply_env_aliases()


def _load_env_fallback() -> None:
    """Minimaler .env-Parser, falls python-dotenv nicht installiert ist."""
    if not ENV_PATH.is_file():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _extract_text(payload: Any) -> str:
    """Zieht den Antworttext aus der Responses-API-Struktur.

    Erwarteter Pfad: output -> content -> output_text -> text.
    Fällt auf eine rekursive Suche zurück, falls der Endpoint eine leicht
    abweichende Hülle liefert.
    """
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for block in item.get("content", []) or []:
            if block.get("type") == "output_text" and block.get("text"):
                parts.append(block["text"])
    if parts:
        return "".join(parts)

    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "output_text" and isinstance(node.get("text"), str):
                found.append(node["text"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    if found:
        return "".join(found)

    raise LLMError(f"Kein Text in der Antwort gefunden: {json.dumps(payload)[:400]}")


def call_llm(prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Schickt prompt an den Azure-Endpoint und gibt den Antworttext zurück.

    Wirft LLMError bei fehlenden Credentials, HTTP-/Netzwerkfehlern oder
    unlesbarer Antwort. Anders als die Query-Tools gibt diese Funktion kein
    Fehler-dict zurück – der Agent soll einen LLM-Ausfall explizit behandeln.
    """
    _load_env()

    # Diagnose vor dem Request: fehlende Credentials sind ein Konfigurations-
    # fehler, kein Netzwerkproblem. Sie duerfen nicht als "Endpoint nicht
    # erreichbar" verschleiert werden - die Meldung nennt den Namen der
    # fehlenden Variable und den geprueften .env-Pfad.
    values = {
        "AZURE_API_KEY": os.environ.get("AZURE_API_KEY") or "",
        "AZURE_ENDPOINT": os.environ.get("AZURE_ENDPOINT") or "",
        "AZURE_MODEL": os.environ.get("AZURE_MODEL") or "",
    }
    values = {k: v.strip() for k, v in values.items()}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise LLMConfigError(
            f"Fehlende oder leere Umgebungsvariablen: {', '.join(missing)}. "
            f".env erwartet unter {ENV_PATH} (vorhanden: {ENV_PATH.is_file()}). "
            "Werte kommen ausschliesslich aus os.environ / .env."
        )

    api_key = values["AZURE_API_KEY"]
    endpoint = values["AZURE_ENDPOINT"]
    if "/openai/" not in endpoint:
        endpoint = endpoint.rstrip("/") + RESPONSES_PATH
    model = values["AZURE_MODEL"]

    # Kein temperature-Parameter: das verwendete Reasoning-Modell lehnt ihn mit HTTP 400 ab
    # ("Unsupported parameter: 'temperature' is not supported with this
    # model") - Reasoning-Modelle der Responses-API erlauben kein Sampling-
    # Tuning. Die Antwort meldet zwar temperature=1.0 zurueck, akzeptiert den
    # Wert aber nicht als Eingabe. Determinismus kommt daher aus der
    # Toolschicht, nicht aus der Dekodierung.
    # prompt_cache_key: der System-Prompt ist bei jeder Frage identisch.
    # Der Endpoint haelt den Cache 24h (prompt_cache_retention), das spart
    # Verarbeitungszeit bei wiederholten Anfragen.
    body = json.dumps({
        "model": model,
        "input": prompt,
        "prompt_cache_key": PROMPT_CACHE_KEY,
    }).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise LLMError(f"HTTP {exc.code} vom Endpoint: {detail}") from exc
    except URLError as exc:
        raise LLMError(f"Endpoint nicht erreichbar: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Timeout nach {timeout}s") from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"Antwort ist kein gültiges JSON: {exc}") from exc
    except (HTTPException, OSError) as exc:
        # Abbruch waehrend des Lesens (RemoteDisconnected, IncompleteRead,
        # ConnectionResetError) ist kein URLError – ohne diesen Zweig kaeme er
        # als unbehandelte Exception statt als LLMError beim Agenten an.
        raise LLMError(f"Verbindung abgebrochen: {type(exc).__name__}: {exc}") from exc

    return _extract_text(payload)
