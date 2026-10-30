"""FastAPI server: transparent API surface for the three-pane UI."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import prompts as P
from .config import Settings, load_settings
from .engine import Engine

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class AskBody(BaseModel):
    text: str = Field(min_length=1, max_length=200000)
    session: str = "default"
    data: str = Field(default="", max_length=400000)


class DocBody(BaseModel):
    text: str = Field(min_length=1, max_length=400000)
    label: str = ""


class SessionBody(BaseModel):
    session: str = "default"


class SimplifyBody(BaseModel):
    text: str = Field(min_length=1, max_length=200000)
    whole: bool = False


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    engine = Engine(settings)
    print(f"[oee] {settings.wiring or f'provider={settings.provider}'} | "
          f"model={settings.model} | base_url={settings.base_url} | window={settings.context_window}")
    app = FastAPI(title="Open Epistemic Engine", version="0.1.0")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/api/status")
    def status():
        return engine.status()

    @app.get("/api/prompts")
    def prompts():
        return {
            "pass1_system": P.PASS1_SYSTEM,
            "pass1_schema": P.PASS1_SCHEMA,
            "pass2_system": P.PASS2_SYSTEM,
            "pass2_schema": P.PASS2_SCHEMA,
            "section_system": P.SECTION_SUMMARY_SYSTEM,
            "doc_card_system": P.DOC_CARD_SYSTEM,
            "case_state_system": P.CASE_STATE_SYSTEM,
            "modal_tags": P.MODAL_TAGS,
            "confidence_tiers": P.CONFIDENCE_TIERS,
        }

    @app.post("/api/ask")
    def ask(body: AskBody):
        if (body.data or "").strip() and not (body.text or "").strip():
            raise HTTPException(422, "Data requires a question. Add a question to the chat box.")
        result = engine.ask(body.text, session=body.session, data=body.data)
        return result

    @app.post("/api/ask/stream")
    def ask_stream(body: AskBody):
        """Server-sent events: pipeline stage events, Pass 2 token deltas,
        then the complete turn under `result`. The UI renders live."""
        if (body.data or "").strip() and not (body.text or "").strip():
            raise HTTPException(422, "Data requires a question. Add a question to the chat box.")
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                result = engine.ask(body.text, session=body.session,
                                    on_event=events.put, data=body.data)
                events.put({"type": "result", "data": result})
            except Exception as exc:  # never leave the stream hanging
                events.put({"type": "error",
                            "message": f"{type(exc).__name__}: {exc}"})
            events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def pump():
            while True:
                ev = events.get()
                if ev is None:
                    break
                yield "data: " + json.dumps(ev, default=str) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            pump(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/turns")
    def turns(session: str = "default", limit: int = 50):
        return {"turns": engine.history(session, limit)}

    @app.post("/api/simplify")
    def simplify(body: SimplifyBody):
        """Display-only rewrite. Writes nothing to the store, so the output can
        never become context for a later turn."""
        return engine.simplify(body.text, whole=body.whole)

    @app.post("/api/session/reset")
    def reset(body: SessionBody):
        engine.store.set_case_state(body.session, {
            "rolling_summary": "", "established_claims": [], "open_questions": [],
            "user_constraints": [], "user_pressure": [], "turn_count": 0,
        })
        return {"ok": True, "session": body.session}

    @app.post("/api/session/delete")
    def delete_case(body: SessionBody):
        """Delete the conversation case: turn history + case state, and clear the
        in-memory job cache. Wipes every case (session) when session is empty.
        Ingested reference documents (Attach Doc) are left intact."""
        session = (body.session or "").strip()
        counts = engine.store.wipe_conversation(session or None)
        with JOBS_LOCK:
            JOBS.clear()
        return {"ok": True, "session": session or "ALL",
                "scope": "all" if not session else "session", **counts}

    @app.post("/api/store/delete_all")
    def delete_all():
        """Delete the whole memory: every document, chunk, vector, card, turn, and
        case state, plus the job cache. Irreversible."""
        counts = engine.wipe_all()
        with JOBS_LOCK:
            JOBS.clear()
        return {"ok": True, "scope": "everything", "deleted": counts}

    @app.post("/api/documents")
    def ingest(body: DocBody):
        text = body.text
        job_id = uuid.uuid4().hex[:10]
        job = {"job_id": job_id, "status": "queued", "percent": 0,
               "log": [], "created_at": time.time(), "label": body.label,
               "chars": len(text), "result": None, "error": None}
        with JOBS_LOCK:
            JOBS[job_id] = job

        def worker():
            with JOBS_LOCK:
                job["status"] = "running"
                job["percent"] = 5
            try:
                result = engine.ingest(text, label=body.label)
                with JOBS_LOCK:
                    job.update(status="done", percent=100,
                               log=[t["step"] for t in result.trace],
                               result={"doc_id": result.doc_id, "title": result.title,
                                       "leaves": result.leaves, "sections": result.sections,
                                       "trace": result.trace, "warnings": result.warnings})
            except Exception as exc:  # surfaced to the UI, never swallowed
                with JOBS_LOCK:
                    job.update(status="error", percent=100, error=f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "queued", "async": True}

    @app.get("/api/jobs")
    def jobs():
        with JOBS_LOCK:
            return {"jobs": [dict(v) for v in JOBS.values()]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with JOBS_LOCK:
            if job_id not in JOBS:
                raise HTTPException(404, "unknown job")
            return dict(JOBS[job_id])

    @app.get("/api/documents")
    def documents():
        return {"documents": engine.store.documents()}

    @app.get("/api/documents/{doc_id}/tree")
    def tree(doc_id: str):
        doc = engine.store.doc(doc_id)
        if not doc:
            raise HTTPException(404, "unknown document")
        nodes = engine.store.nodes(("leaf", "section", "card"), doc_id=doc_id)
        return {"document": doc,
                "nodes": [{k: v for k, v in n.items() if k != "text"} for n in nodes],
                "leaves": [n for n in nodes if n["kind"] == "leaf"]}

    @app.delete("/api/documents/{doc_id}")
    def delete(doc_id: str):
        if not engine.store.doc(doc_id):
            raise HTTPException(404, "unknown document")
        engine.delete_document(doc_id)
        return {"ok": True, "deleted": doc_id}

    @app.get("/api/nodes/{node_id}")
    def node(node_id: str):
        node_row = engine.store.node(node_id)
        if not node_row:
            raise HTTPException(404, "unknown node")
        return node_row

    @app.post("/api/eval")
    def run_eval():
        from evals.harness import run_suite

        return run_suite(settings)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def serve() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port,
                log_level="info")


app = create_app(load_settings())

if __name__ == "__main__":
    serve()
