"""Optional read-only web dashboard for browsing what's in the memory store.

DISABLED by default. The user turns it on/off and picks the port purely with
env vars (set them where Claude Code launches this MCP server, or in a manual
shell):

    WEB_UI=on            # 1/on/true/yes enables it; anything else = off
    WEB_PORT=8765        # port to serve on (default 8765)
    WEB_HOST=127.0.0.1   # bind address (default localhost-only; see note)

When enabled it autostarts together with the MCP server (the server boots when
Claude Code opens it) and runs in a daemon thread, so it dies with the process
and never blocks the MCP stdio loop. It NEVER writes to stdout — stdout is the
MCP channel — so all logging goes to stderr.

The page itself is a plain template at web/index.html that you can edit freely;
this module just serves that file plus two read-only JSON endpoints it calls.

Security note: it binds to 127.0.0.1 by default, so only this machine can reach
it. Set WEB_HOST=0.0.0.0 only if you knowingly want it on your network.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_TEMPLATE = os.path.join(os.path.dirname(__file__), "web", "index.html")


def _truthy(val) -> bool:
    return (val or "").strip().lower() in ("1", "on", "true", "yes")


def enabled() -> bool:
    """Whether the user turned the dashboard on via WEB_UI."""
    return _truthy(os.environ.get("WEB_UI"))


def _log(msg: str) -> None:
    print(f"[db-memory] {msg}", file=sys.stderr, flush=True)


def start(fetch_memories, fetch_stats):
    """Start the dashboard in a daemon thread if WEB_UI is enabled.

    fetch_memories() -> list[dict(id, problem, solution)]
    fetch_stats()    -> dict
    Returns the server thread, or None if disabled / it couldn't start. Any
    failure is logged and swallowed so the MCP server still runs normally.
    """
    if not enabled():
        return None

    host = os.environ.get("WEB_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("WEB_PORT", "8765"))
    except ValueError:
        _log("WEB_PORT is not a number; web UI not started.")
        return None

    try:
        with open(_TEMPLATE, "rb") as f:
            template = f.read()
    except OSError as e:
        _log(f"web template missing ({e}); web UI not started.")
        return None

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    self._send(200, template, "text/html; charset=utf-8")
                elif path == "/api/memories":
                    body = json.dumps(fetch_memories()).encode()
                    self._send(200, body, "application/json")
                elif path == "/api/stats":
                    body = json.dumps(fetch_stats()).encode()
                    self._send(200, body, "application/json")
                else:
                    self._send(404, b'{"error":"not found"}', "application/json")
            except Exception as e:  # noqa: BLE001 — a bad request must not kill the thread
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, *args):  # silence default stderr access logs
            pass

    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        _log(f"web UI could not bind {host}:{port} ({e}); not started.")
        return None

    t = threading.Thread(target=httpd.serve_forever, name="db-memory-web", daemon=True)
    t.start()
    _log(f"web dashboard on http://{host}:{port} (WEB_UI=on)")
    return t
