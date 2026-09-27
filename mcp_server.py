#!/usr/bin/env python3
"""
mcp_server.py - serve the U-Bahn tools over MCP, so any MCP-capable agent or client can use them
(an open-source agent framework, Claude Desktop, the hackathon LLM through an MCP-aware client...).

Every tool of agent.Toolbox is exposed with its JSON schema, plus two data tools:
    ubahn_ingest_files   validate new CSV files on the server and add them to the data folder, then reload
    ubahn_reload_data    rebuild the engine from the data folder (after files were copied in by hand)

Run
    python mcp_server.py --data /path/to/data                       # stdio (local clients)
    python mcp_server.py --data /path/to/data --http --port 8000     # streamable HTTP (remote clients)
Works with the MCP Python SDK 2.x (MCPServer) and 1.x (FastMCP).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Optional

try:                                               # MCP Python SDK 2.x
    from mcp.server import MCPServer
except ImportError:                                # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

import src.analyses as A
from src.agent import Toolbox
from src.engine import UBahnEngine
from src.ingest import ingest

PY_TYPES = {'string': str, 'number': float, 'integer': int, 'boolean': bool, 'array': list, 'object': dict}


class Holder:
    """Keeps the current engine and toolbox, so a reload swaps them for every tool at once."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tb = Toolbox(UBahnEngine(data_dir))

    def reload(self):
        for f in (A._daily, A._line_graph, A._residual_z, A._vulnerability):
            f.cache_clear()
        self.tb = Toolbox(UBahnEngine(self.data_dir))
        return self.tb.eng.data_status()


def _tool_function(holder, name, schema):
    """A Python function with a real signature built from the JSON schema, so the MCP SDK can
    derive the same input schema; calls go to the holder's current toolbox."""
    props, req = schema.get('properties', {}), set(schema.get('required', []))
    params = []
    for p, spec in props.items():
        t = PY_TYPES.get(spec.get('type'), str)
        if p in req:
            params.append(inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY, annotation=t))
        else:
            params.append(inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY, annotation=Optional[t], default=None))

    def fn(**kwargs):
        text, _ = holder.tb.call(name, {k: v for k, v in kwargs.items() if v is not None})
        return text
    fn.__signature__ = inspect.Signature(params, return_annotation=str)
    fn.__annotations__ = {**{p.name: p.annotation for p in params}, 'return': str}
    fn.__name__ = name
    return fn


def build_server(data_dir: str) -> tuple[MCPServer, Holder]:
    holder = Holder(data_dir)
    server = MCPServer('ubahn_mcp')
    for s in holder.tb.schemas():
        server.add_tool(_tool_function(holder, s['name'], s['parameters']), name=s['name'], description=s['description'])

    def ubahn_ingest_files(paths: list, dry_run: Optional[bool] = False) -> str:
        """Validate new data files (paths on the server) against the dataset schema, add them to the data
        folder, refit the engine and report what the model learned. Nothing is added if a file has errors."""
        rep = ingest(paths, holder.data_dir, bool(dry_run), ref=holder.tb.eng)
        new_eng = rep.pop('_engine', None)
        if rep['ok'] and not dry_run:
            for f in (A._daily, A._line_graph, A._residual_z, A._vulnerability):
                f.cache_clear()
            holder.tb = Toolbox(new_eng) if new_eng is not None else Toolbox(UBahnEngine(holder.data_dir))
            rep['status_after_reload'] = holder.tb.eng.data_status()
        return json.dumps(rep, ensure_ascii=False, default=str)

    def ubahn_reload_data() -> str:
        """Rebuild the engine from the data folder (e.g. after files were copied in by hand)."""
        return json.dumps(holder.reload(), ensure_ascii=False, default=str)

    server.add_tool(ubahn_ingest_files, name='ubahn_ingest_files', description=ubahn_ingest_files.__doc__)
    server.add_tool(ubahn_reload_data, name='ubahn_reload_data', description=ubahn_reload_data.__doc__)
    return server, holder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.environ.get('UBAHN_DATA', '.'))
    ap.add_argument('--http', action='store_true', help='serve over streamable HTTP instead of stdio')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    server, _ = build_server(a.data)
    if a.http:
        try:
            server.run('streamable-http', port=a.port)
        except TypeError:                          # SDK versions that take the port from settings
            server.settings.port = a.port
            server.run('streamable-http')
    else:
        server.run()


if __name__ == '__main__':
    main()
