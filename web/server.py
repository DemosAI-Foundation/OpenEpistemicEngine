"""Local web server for the Epistemic Engine. Stdlib only; no external services."""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from web import uidata
from web.jobs import Manager
from web.store import Store

STATIC_DIR = Path(__file__).parent / "static"
FORCED_MODES = {"user-forced", "hypothetical", "speculative", "preference-based"}


class Handler(BaseHTTPRequestHandler):
    server_version = "EpistemicEngine/1.0"
    store: Store
    manager: Manager

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            return self._get_api(path, parsed.query)
        return self._static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json()
        if path == "/api/cases":
            prompt = str(body.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "prompt is required"}, 400)
            overrides = {k: v for k, v in (body.get("overrides") or {}).items()}
            if body.get("mode"):
                overrides["mode"] = body["mode"]
            job = self.manager.start(prompt, overrides)
            return self._json({"id": job.case_id, "status": "started"})
        if path.endswith("/continue"):
            case_id = unquote(path.split("/")[3])
            prompt = str(body.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "prompt is required"}, 400)
            overrides = {k: v for k, v in (body.get("overrides") or {}).items()}
            if body.get("mode"):
                overrides["mode"] = body["mode"]
            try:
                job = self.manager.continue_case(case_id, prompt, overrides)
            except KeyError:
                return self._json({"error": "case not found"}, 404)
            return self._json({"id": job.case_id, "status": "started"})
        if path.endswith("/checkpoint"):
            case_id = unquote(path.split("/")[3])
            action = str(body.get("action") or "confirm")
            mode = str(body.get("mode") or "")
            if action in {"force-original", "override", "continue-anyway"} and mode not in FORCED_MODES:
                mode = "user-forced"
            if action == "confirm":
                mode = mode if mode in FORCED_MODES else ""
            decision = {
                "action": {"continue-anyway": "force-original", "force": "force-original"}.get(action, action),
                "mode": mode,
                "edited_assumptions": list(body.get("edited_assumptions") or []),
                "note": str(body.get("note") or ""),
                "auto_approved": action in {"auto-continue"},
            }
            if not decision["note"]:
                decision["note"] = {
                    "confirm": "user confirmed neutral analysis",
                    "edit": "user edited assumptions",
                    "force-original": "user forced original framing",
                    "auto-continue": "auto-approved by the user",
                }.get(decision["action"], decision["action"])
            ok = self.manager.resolve_checkpoint(case_id, decision)
            return self._json({"resolved": ok}, 200 if ok else 409)
        if path.endswith("/stop"):
            case_id = unquote(path.split("/")[3])
            return self._json({"stopped": self.manager.stop(case_id)})
        if path.endswith("/retry"):
            case_id = unquote(path.split("/")[3])
            try:
                job = self.manager.retry(case_id)
            except KeyError:
                return self._json({"error": "case not found"}, 404)
            return self._json({"id": job.case_id, "status": "restarted"})
        if path == "/api/settings":
            return self._json(self.store.save_settings(body))
        if path == "/api/documents":
            name = str(body.get("name") or "document.txt")
            text = str(body.get("text") or "")
            if not text.strip():
                return self._json({"error": "text is required"}, 400)
            record = self.store.add_document(name, text)
            self.manager.retriever = None
            return self._json(record)
        if path == "/api/documents/delete":
            removed = self.store.delete_document(str(body.get("name") or ""))
            self.manager.retriever = None
            return self._json({"removed": removed})
        return self._json({"error": "not found"}, 404)

    def _get_api(self, path: str, query: str) -> None:
        if path == "/api/health":
            settings = self.store.settings()
            return self._json(
                {
                    "health": self.manager.health(),
                    "modules": self.manager.module_status(),
                    "constraint": "All LLM inference is routed to the single local model on port 8082. No external LLM APIs are used.",
                    "model_locked": True,
                    "settings": {
                        k: settings[k]
                        for k in ("mode", "analysis_depth", "checkpoint_policy", "max_tokens", "max_context")
                    },
                }
            )
        if path == "/api/settings":
            return self._json(self.store.settings())
        if path == "/api/cases":
            return self._json({"cases": self.store.case_summaries()})
        if path == "/api/documents":
            return self._json({"documents": self.store.documents(), "corpus_dir": str(self.store.corpus_dir)})
        parts = [unquote(p) for p in path.split("/")]
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "cases" and parts[3] != "events":
            case = self.store.load_case(parts[3])
            if case is None:
                return self._json({"error": "case not found"}, 404)
            payload = uidata.build_ui(case)
            job = self.manager.get(parts[3])
            if job:
                payload["live"] = job.state
            return self._json(payload)
        if len(parts) == 5 and parts[1] == "api" and parts[2] == "cases" and parts[4] == "events":
            return self._events(parts[3])
        return self._json({"error": "not found"}, 404)

    def _events(self, case_id: str) -> None:
        job = self.manager.get(case_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        case = self.store.load_case(case_id)
        bootstrap = {"type": "snapshot", "case_id": case_id, "ui": uidata.build_ui(case) if case else None}
        try:
            self.wfile.write(f"data: {json.dumps(bootstrap)}\n\n".encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            return
        if job is None:
            try:
                self.wfile.write(b'data: {"type":"done"}\n\n')
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        listener = job.subscribe()
        last_ping = time.time()
        try:
            while True:
                try:
                    event = listener.get(timeout=1)
                except queue.Empty:
                    if job.finished and listener.empty():
                        self.wfile.write(b'data: {"type":"done"}\n\n')
                        break
                    if time.time() - last_ping > 15:
                        self.wfile.write(b": ping\n\n")
                        last_ping = time.time()
                    continue
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                if event.get("type") in {"final", "error", "cancelled"}:
                    self.wfile.write(b'data: {"type":"done"}\n\n')
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            job.unsubscribe(listener)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "", "/index.html") else path.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript",):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)


def serve(
    host: str = "127.0.0.1",
    port: int = 8085,
    store_dir: str = ".epistemic",
    url: str = "http://127.0.0.1:8082",
    verbose: bool = False,
) -> None:
    store = Store(store_dir)
    manager = Manager(store, url=url)
    Handler.store = store
    Handler.manager = manager
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.verbose = verbose
    health = manager.health(max_age=0)
    status = "online" if health.get("online") else "offline"
    print(f"Epistemic Engine UI: http://{host}:{port}")
    print(f"Local LLM {url} -> {status} ({health.get('model') or health.get('error')})")
    print("No external LLM APIs. Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="epistemic-web", description="Epistemic Engine web UI (local LLM on port 8082 only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--store", default=".epistemic")
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    serve(args.host, args.port, args.store, args.url, args.verbose)


if __name__ == "__main__":
    main()
