# InnoTrans Hackathon 2026 – U-Bahn Passenger Flow Optimizer

An AI-assisted tool for analysing and optimising passenger flows in the Berlin U-Bahn network, built during the InnoTrans Hackathon 2026 (Alstom challenge) with data provided by Alstom.

Operators ask questions in plain language – *"Which stations are most at risk on the first InnoTrans day?"*, *"U8 is suspended between two stations – where will passengers reroute?"* – and an LLM agent answers by calling analysis tools on a fitted passenger-flow model. Every figure in an answer comes from a tool result and is checked after the answer is written.

## Features

- **FastAPI backend** (`src/server.py`) – chat endpoint (`/ask`), a built-in chat UI with a hotspot sidebar, and LLM-free operator endpoints (daily brief, action plan, forecast, staff plan, passenger messages)
- **Streamlit frontend** (`app.py`) – operator web app with chat, daily brief, action plan, replay and animated network maps, comparisons, reports and data management
- **Azure OpenAI integration** – tool-calling agent on the Azure OpenAI Responses API with around 40 analysis tools, a time budget per question, figure verification and a fallback answer built from tool results when the model is unavailable
- **MCP server** (`mcp_server.py`) – exposes the analysis tools via the Model Context Protocol
- **Network graph analysis** – networkx-based routing, rerouting under closures and station dependencies
- **Data ingestion** – validate and inject new CSV files at runtime

## Tech stack

Python · FastAPI · Streamlit · Azure OpenAI · networkx · plotly · pandas · numpy · scipy

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your values
```

## Configuration

All credentials are read from `.env` (see [`.env.example`](.env.example)):

| Variable | Meaning |
|---|---|
| `AZURE_OPENAI_API_KEY` | API key of your Azure OpenAI resource |
| `AZURE_OPENAI_ENDPOINT` | Endpoint of your resource – either the base URL or the full Responses URL (`…/openai/responses?api-version=…`) |
| `AZURE_OPENAI_DEPLOYMENT` | Name of your model deployment |

Optionally set `DATA_DIR` to point to a data folder other than `data/`.

## Data

**The original dataset is not included.** It was provided by Alstom for the hackathon and may not be redistributed – see [`data/README.md`](data/README.md). Without data files in `data/`, the engine and most tools cannot run.

## Usage

```bash
# FastAPI backend with chat UI on http://127.0.0.1:8000
uvicorn src.server:app --port 8000

# Streamlit operator app (embeds the FastAPI chat)
streamlit run app.py

# Agent on the command line
python -m src.agent --show-tools "Which stations are most at risk tomorrow?"
python -m src.agent --selftest --quick      # tool self-test without LLM

# MCP server (stdio)
python mcp_server.py --data data

# Validate and inject new data files, then POST /reload
python -m src.ingest --data data new/*.csv
```

## Tests

```bash
pytest tests/ -v
```

Most tests need the original dataset in `data/`.

## License

[MIT](LICENSE)
