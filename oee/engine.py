"""The engine: phases 1 -> 5 in one trace-per-turn pipeline, fully inspectable."""

from __future__ import annotations

import re
import time

from . import assembly as A
from . import backstop as BS
from . import jsonx
from . import prompts as P
from . import s2a
from . import semantic
from .config import Settings, autoconfigure
from .embed import Embedder
from .jsonx import parse_and_repair
from .llm import LLMResponse, ProviderError, build_provider
from .raptor import Ingestor
from .retrieval import HybridIndex
from .security import scan_injection, secure_frontend, strip_reference_delimiter_collisions
from .store import Store
from .textutil import quote_span

CITATION_RE = re.compile(r"\[(R\d+-C\d+)\]")

BOTTOM_STRIP_RE = re.compile(
    r"(?m)\[(?:FACT|PREDICTION|HYPOTHETICAL|POSSIBILITY|CONSTRAINT|AUTHOR_CLAIM|"
    r"INFERENCE|SPECULATION|R\d+-[A-Z0-9-]+)\]|^#{1,6}\s*|^\s*[-*]\s+|\*\*|__")
BOTTOM_SENT_RE = re.compile(r"(?<=[.!?])\s+")
BOTTOM_MAX_CHARS = 560


class Engine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        autoconfigure(self.settings)
        self.settings.validate()
        self.provider = build_provider(self.settings)
        self.embedder = Embedder(self.provider)
        self.store = Store(self.settings.db_path)
        self.index = HybridIndex(self.store, self.embedder)
        self.ingestor = Ingestor(self.provider, self.store, self.embedder, self.settings)
        self._ensure_vector_space()

    # ---------------------------------------------------------------- reindex
    def reindex(self) -> int:
        """Re-embed every leaf/section with the currently active embedder. A
        provider swap, a semantics toggle, or an OEE_EMBED_MODEL change moves
        the dense channel to a new vector space; stored vectors from the old
        one cannot be compared against new queries, so they are rebuilt."""
        nodes = self.store.nodes(("leaf", "section"))
        if not nodes:
            self.index.refresh(force=True)
            return 0
        mat, mode = self.embedder.encode([n["text"] for n in nodes])
        vectors = {n["node_id"]: [float(x) for x in mat[i]] for i, n in enumerate(nodes)}
        modes = {n["node_id"]: mode for n in nodes}
        self.store.save_vectors(vectors, modes)
        self.index.refresh(force=True)
        return len(vectors)

    def _ensure_vector_space(self) -> None:
        active_dim = self.index.target_dim
        stored_dims = self.store.vector_dims()
        if active_dim is None or not stored_dims:
            return
        if any(d != active_dim for d in stored_dims):
            self.reindex()

    # ---------------------------------------------------------------- ingest
    def ingest(self, text: str, doc_id: str | None = None, label: str = ""):
        result = self.ingestor.ingest(text, doc_id=doc_id, label=label)
        self.index.refresh(force=True)
        return result

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        self.index.refresh(force=True)

    def wipe_all(self) -> dict:
        """Nuclear delete: every document, node, vector, card, turn, and case
        state, then rebuild the (now empty) index. Nothing survives."""
        counts = self.store.wipe_everything()
        self.index.refresh(force=True)
        return counts

    # ------------------------------------------------------------------ ask
    def ask(self, raw_text: str, session: str = "default",
            on_event=None, data: str = "") -> dict:
        """on_event(event: dict) streams pipeline stages and Pass 2 token
        deltas to the caller while the turn is still running. `data` is a separate,
        optional channel: inline single-turn text that bypasses S2A, is screened for
        injection, and is passed to Pass 2 as quarantined inert <user_data>; it is
        never ingested, embedded, retrieved, or persisted."""
        trace: list[dict] = []
        t_start = time.perf_counter()

        def emit() -> None:
            if on_event is None:
                return
            step = trace[-1]
            try:
                event = {"type": "stage"}
                event.update(step)
                on_event(event)
            except Exception:
                pass

        def delta_cb(piece: str) -> None:
            if on_event is not None:
                try:
                    on_event({"type": "delta", "text": piece})
                except Exception:
                    pass

        clean, sec_flags = secure_frontend(raw_text)
        inj_flags = scan_injection(clean)
        trace.append({
            "step": "1.secure_frontend",
            "status": "flagged" if (sec_flags or inj_flags) else "ok",
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
            "flags": sec_flags + inj_flags,
            "detail": f"{len(raw_text)} chars screened; control tokens removed before any model call",
            "input_chars": len(raw_text),
            "output_chars": len(clean),
            "output_preview": clean[:400],
        })
        emit()

        clean_data = ""
        data = (data or "").strip()
        if data:
            t_data = time.perf_counter()
            data_clean, data_sec = secure_frontend(data)
            data_inj = scan_injection(data_clean)
            data_clean, data_delim = strip_reference_delimiter_collisions(data_clean)
            clean_data = data_clean
            data_flags = data_sec + data_inj + data_delim
            trace.append({
                "step": "1b.data_frontend",
                "status": "flagged" if data_flags else "ok",
                "latency_ms": round((time.perf_counter() - t_data) * 1000, 3),
                "flags": data_flags,
                "detail": (f"{len(data)} chars of inline data screened; control tokens "
                           "stripped, content preserved; not sanitized, not ingested, "
                           "not persisted, not sent to S2A"),
                "input_chars": len(data),
                "output_chars": len(clean_data),
                "output_preview": clean_data[:400],
            })
            emit()

        pass1 = s2a.sanitize(self.provider, clean, self._doc_ids())
        trace.append({
            "step": "2.s2a_pass1",
            "status": "flagged" if pass1["flags"] else "ok",
            "latency_ms": pass1["latency_ms"],
            "flags": pass1["flags"],
            "detail": f"model={pass1['model']} schema=PASS1 repairs={len(pass1['repairs'])}",
            "system_prompt": pass1["system_prompt"],
            "user_prompt": pass1["user_prompt"],
            "raw_output": pass1["raw_output"],
            "sanitized_request": pass1["sanitized_request"],
            "bias_removed": pass1["bias_removed"],
            "deterministic_bias": pass1["deterministic_bias"],
            "intent_preserved": pass1["intent_preserved"],
        })
        emit()

        query = f"{pass1['sanitized_request']}\n{clean}"
        search = self.index.search(
            query,
            leaf_k=self.settings.leaf_k,
            section_k=self.settings.section_k,
            rrf_k=self.settings.rrf_k,
        )
        cards = self.index.cards()
        fallback_hit = search["channels"].get("attached_document_fallback", False)
        no_hit = not (search["leaves"] or search["sections"])
        top_cosine = max([r["cosine"] for r in search["leaves"] + search["sections"]
                          if r.get("cosine") is not None] or [0.0])
        weak_grounding = (not no_hit and not fallback_hit
                          and search["channels"].get("dense_hits", 0) > 0
                          and top_cosine < self.settings.weak_grounding_cosine)
        retrieval_flags = []
        if no_hit:
            retrieval_flags.append(
                "NO HITS: no references attached, or nothing matched. Answer is parametric only."
            )
        elif fallback_hit:
            retrieval_flags.append(
                "FALLBACK: no BM25/dense match; the attached document tree was supplied as "
                "grounding because attaching a document is an explicit intent signal."
            )
        elif weak_grounding:
            retrieval_flags.append(
                f"FALLBACK: best dense cosine {top_cosine:.4f} is below "
                f"{self.settings.weak_grounding_cosine}; the retrieved chunks are topically "
                "unrelated to the question. The corpus is not a source on this topic: the "
                "grounding supports an absence verdict only, not answer content."
            )
        trace.append({
            "step": "3.hybrid_retrieval",
            "status": "ok" if (not no_hit and not fallback_hit and not weak_grounding)
                     else ("flagged" if (fallback_hit or weak_grounding) else "empty"),
            "latency_ms": search["latency_ms"],
            "flags": retrieval_flags,
            "detail": (f"BM25 hits={search['channels']['bm25_hits']} "
                       f"dense hits={search['channels']['dense_hits']} "
                       f"rrf_k={search['channels']['rrf_k']} pool={search['channels']['fused_pool']} "
                       f"-> leaves={[r['node_id'] for r in search['leaves']]} "
                       f"sections={[r['node_id'] for r in search['sections']]}"),
            "channels": search["channels"],
            "explain_top": search["explain_top"],
            "ranked": [
                {"node_id": r["node_id"], "rrf": r["rrf"], "bm25": r["bm25"],
                 "cosine": r["cosine"], "rank_bm25": r["rank_bm25"], "rank_dense": r["rank_dense"]}
                for r in search["leaves"] + search["sections"]
            ],
        })
        emit()

        pack = A.assemble(
            self.store.get_case_state(session), search, cards,
            pass1["sanitized_request"], clean, self.settings.max_context_tokens,
            clean_data,
        )
        trace.append({
            "step": "4.context_assembly",
            "status": "flagged" if pack["truncated"] else "ok",
            "latency_ms": 0.0,
            "flags": ([f"TRUNCATED: dropped {', '.join(pack['dropped_nodes'])} to fit "
                       f"{self.settings.max_context_tokens} token budget"] if pack["truncated"] else [])
                     + ([f"TRUNCATED: the user request was clipped to fit the "
                         f"{self.settings.max_context_tokens} token budget; only the first "
                         f"{pack['raw_request_budget']} chars reached the model"
                         ] if pack.get("raw_request_truncated") else [])
                     + ([f"OVER BUDGET: assembled prompt exceeds the "
                         f"{self.settings.max_context_tokens} token budget by "
                         f"{pack['budget_exceeded_tokens']} tokens after dropping leaves"
                         ] if pack["budget_exceeded_tokens"] else [])
                     + pack["delimiter_flags"],
            "detail": (f"system+user={pack['tokens']} tokens / budget "
                       f"{self.settings.max_context_tokens}; "
                       f"{len(pack['leaf_ids'])} leaves + {len(pack['section_ids'])} sections + "
                       f"{len(cards)} doc card(s)"),
            "system_prompt": pack["system_prompt"],
            "assembled_context": pack["preview"],
            "context_truncated_for_view": len(pack["user_prompt"]) > pack["full_view_chars"],
            "context_chars": len(pack["user_prompt"]),
            "tokens": pack["tokens"],
            "budget_tokens": pack["budget_tokens"],
        })
        emit()

        has_references = bool(search["leaves"] or search["sections"])
        retrieved_ids = {r["node_id"] for r in search["leaves"] + search["sections"]}
        chunk_texts = {r["node_id"]: r["text"] for r in search["leaves"] + search["sections"]}
        card_ids = {c["node_id"] for c in cards}
        for card in cards:
            retrieved_ids.add(card["node_id"])
            chunk_texts[card["node_id"]] = card.get("text", "")

        t0 = time.perf_counter()
        cand = self._candidate(pack, on_delta=(delta_cb if on_event else None))
        self._finish_candidate(cand, search, has_references, retrieved_ids, chunk_texts,
                               pack, card_ids, weak_grounding)
        gen_notes, raw_output, sidecar = cand["notes"], cand["raw"], cand["sidecar"]
        gen_tokens, model = cand["tokens"], cand["model"]
        generation_error, resp_meta = cand["error"], cand["meta"]
        bs_flags, sidecar_flags, sidecar_full = (cand["bs_flags"], cand["sidecar_flags"],
                                                 cand["full"])
        extra_flags: list[str] = []

        best_of = None
        reason = self._best_of_trigger(cand, sidecar_full, bs_flags + sidecar_flags,
                                       has_references)
        if reason:
            alt = self._candidate(pack)
            self._finish_candidate(alt, search, has_references, retrieved_ids, chunk_texts,
                                   pack, card_ids, weak_grounding)
            sa, parts_a = BS.score_candidate(
                {**sidecar, "citations": sidecar_full["citations"]},
                bs_flags + sidecar_flags)
            sb, parts_b = BS.score_candidate(
                {**alt["sidecar"], "citations": alt["full"]["citations"]},
                alt["bs_flags"] + alt["sidecar_flags"])
            kept = "A" if sa >= sb else "B"
            best_of = {"trigger": reason, "score_a": sa, "score_b": sb, "kept": kept,
                       "parts_a": parts_a, "parts_b": parts_b}
            extra_flags.append(
                f"BEST_OF: two candidates ({reason}); kept {kept} "
                f"(score {max(sa, sb):.1f} vs {min(sa, sb):.1f})")
            if kept == "B":
                cand = alt
                gen_notes, raw_output, sidecar = cand["notes"], cand["raw"], cand["sidecar"]
                gen_tokens, model = cand["tokens"], cand["model"]
                generation_error, resp_meta = cand["error"], cand["meta"]
                bs_flags, sidecar_flags = cand["bs_flags"], cand["sidecar_flags"]
                sidecar_full = cand["full"]

        critic_res = BS.critic(sidecar_full)
        critic_report = dict(critic_res)
        if (self.settings.critic != "off" and not critic_res["passed"]
                and critic_res["severity"] == "high" and not generation_error):
            instruction = (
                "REPAIR PASS: the payload you just produced violated the epistemic "
                "contract. Fix ONLY these violations, keep everything else, and emit "
                "the complete object again:\n- " + "\n- ".join(critic_res["violations"]))
            rep = self._candidate(pack, extra_instruction=instruction, temperature=0.05)
            self._finish_candidate(rep, search, has_references, retrieved_ids, chunk_texts,
                                   pack, card_ids, weak_grounding)
            sr, _ = BS.score_candidate(
                {**rep["sidecar"], "citations": rep["full"]["citations"]},
                rep["bs_flags"] + rep["sidecar_flags"])
            sc, _ = BS.score_candidate(
                {**sidecar, "citations": sidecar_full["citations"]},
                bs_flags + sidecar_flags)
            critic_report["repair_score"] = sr
            critic_report["original_score"] = sc
            critic_report["kept"] = "repair" if sr > sc else "original"
            if sr > sc:
                cand = rep
                gen_notes, raw_output, sidecar = cand["notes"], cand["raw"], cand["sidecar"]
                gen_tokens, model = cand["tokens"], cand["model"]
                resp_meta = cand["meta"]
                bs_flags, sidecar_flags = cand["bs_flags"], cand["sidecar_flags"]
                sidecar_full = cand["full"]
                extra_flags.append(
                    "CRITIC: high-severity violations repaired in one targeted pass ("
                    + "; ".join(critic_res["violations"])[:200] + ")")
            else:
                extra_flags.append(
                    "CRITIC: repair scored lower than the original; kept the original ("
                    + "; ".join(critic_res["violations"])[:200] + ")")
        elif not critic_res["passed"]:
            extra_flags.append("CRITIC: violations noted ("
                               + "; ".join(critic_res["violations"])[:200] + ")")
        reply = sidecar_full.get("chat_reply", sidecar.get("chat_reply", ""))
        trace.append({
            "step": "5.pass2_generation",
            "status": "error" if generation_error else "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "flags": [f"ERROR: {generation_error}"] if generation_error else [],
            "detail": (f"model={model} completion_tokens={gen_tokens} "
                       f"tier={sidecar.get('confidence_tier')} basis={sidecar.get('basis')} "
                       f"schema_repairs={len(gen_notes)}"
                       + (f" response_format={resp_meta.get('response_format')}"
                          if resp_meta.get("response_format") else "")
                       + (f" best_of={best_of['kept']}" if best_of else "")
                       + (f" critic={'passed' if critic_res['passed'] else critic_res['severity']}")),
            "best_of": best_of,
            "critic": critic_report,
            "response_format": resp_meta.get("response_format"),
            "response_format_fallbacks": resp_meta.get("response_format_fallbacks", []),
            "system_prompt": pack["system_prompt"],
            "raw_output": raw_output,
            "tier": sidecar.get("confidence_tier"),
            "basis": sidecar.get("basis"),
            "repairs": gen_notes,
        })
        emit()
        trace.append({
            "step": "6.epistemic_backstop",
            "status": "flagged" if (bs_flags + sidecar_flags + extra_flags) else "ok",
            "latency_ms": 0.0,
            "flags": bs_flags + sidecar_flags + extra_flags,
            "detail": (f"provenance checks over chat_reply + sidecar; "
                       f"{len(retrieved_ids)} citable node ids"
                       + ("; best-of-2 triggered" if best_of else "")
                       + ("; critic " + ("clean" if critic_res["passed"]
                                          else critic_res["severity"]))),
            "best_of": best_of,
            "critic": critic_report,
            "reply_before_scrub": (raw_output and _extract_reply(raw_output)) or "",
            "reply_after_scrub": reply,
        })
        emit()

        case_state = self._update_case_state(session, pass1, sidecar_full)
        trace.append({
            "step": "7.case_state_update",
            "status": "ok" if not case_state["flags"] else "flagged",
            "latency_ms": case_state["latency_ms"],
            "flags": case_state["flags"],
            "detail": f"turn_count={case_state['state'].get('turn_count')} fields="
                      f"{sorted(k for k in case_state['state'] if k != 'turn_count')}",
            "case_state_new": case_state["state"],
            "case_state_rendered": A.render_case_state(case_state["state"]),
        })
        emit()

        turn_id = self.store.add_turn({
            "session": session, "raw": raw_text, "clean": clean,
            "sanitized": pass1["sanitized_request"], "bias": pass1["bias_removed"],
            "chat_reply": reply, "sidecar": sidecar_full, "trace": trace,
            "context_tokens": pack["tokens"], "model": model,
        })
        flags_all = [f"{t['step']}: {f}" for t in trace for f in t["flags"]]
        return {
            "turn_id": turn_id,
            "data_chars": pack.get("data_chars", 0),
            "chat_reply": reply,
            "sidecar": sidecar_full,
            "trace": trace,
            "flags": flags_all,
            "prompts": {
                "pass1_system": pass1["system_prompt"],
                "pass1_user": pass1["user_prompt"],
                "pass1_output": pass1["raw_output"],
                "pass2_system": pack["system_prompt"],
                "assembled_context": pack["user_prompt"],
                "pass2_output": raw_output,
            },
            "usage": {
                "context_tokens": pack["tokens"],
                "completion_tokens": gen_tokens,
                "window_tokens": self.settings.context_window,
                "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "model": model,
                "provider": self.provider.name,
            },
        }

    def _candidate(self, pack: dict, extra_instruction: str | None = None,
                   temperature: float | None = None, on_delta=None) -> dict:
        """One Pass 2 generation: tool call in, repaired payload out."""
        user = pack["user_prompt"]
        if extra_instruction:
            user = user + "\n\n" + extra_instruction
        cand = {"sidecar": {}, "raw": "", "meta": {}, "tokens": 0, "latency": 0.0,
                "model": self.provider.settings.model, "notes": [], "error": None}
        try:
            temp = (temperature if temperature is not None
                    else min(self.settings.temperature, 0.1))
            if on_delta is not None and hasattr(self.provider, "stream_chat"):
                pieces = []
                for piece in self.provider.stream_chat(
                        pack["system_prompt"], user, schema=P.PASS2_SCHEMA,
                        kind="pass2", temperature=temp,
                        max_tokens=self.settings.pass2_max_tokens):
                    pieces.append(piece)
                    on_delta(piece)
                text = "".join(pieces)
                resp = LLMResponse(text=text, model=self.provider.settings.model,
                                   endpoint="stream", latency_ms=0.0,
                                   prompt_tokens=0, completion_tokens=len(text) // 4,
                                   kind="pass2", system=pack["system_prompt"], user=user,
                                   meta={"streamed": True,
                                         "completion_tokens_est": len(text) // 4})
            else:
                resp = self.provider.chat(pack["system_prompt"], user, schema=P.PASS2_SCHEMA,
                                          kind="pass2", temperature=temp,
                                          max_tokens=self.settings.pass2_max_tokens)
            cand.update({"raw": resp.text, "meta": resp.meta or {},
                         "tokens": resp.completion_tokens, "latency": resp.latency_ms,
                         "model": resp.model})
            cand["sidecar"] = parse_and_repair(resp.text, P.PASS2_SCHEMA, cand["notes"])
        except ProviderError as exc:
            cand["error"] = str(exc)
            cand["notes"].append("generation failed; degraded reply emitted")
            cand["sidecar"] = {
                "chat_reply": (
                    "Generation is unavailable: the configured local model did not answer "
                    "(`" + str(exc)[:160] + "`). Nothing in this turn is parametric "
                    "output [CONSTRAINT]."
                ),
                "bottom_line": "Generation failed; nothing was answered.",
                "confidence_tier": "Conceptually Open",
                "basis": ["inference"],
                "citations": [],
                "assumptions_used": ["No generation backend available."],
                "unknowns": ["Model availability"],
                "what_would_change": ["Start the configured local LLM server"],
                "warnings": [f"generation failed: {str(exc)[:200]}"],
            }
        return cand

    def _finish_candidate(self, cand: dict, search: dict, has_references: bool,
                          retrieved_ids: set, chunk_texts: dict, pack: dict,
                          card_ids: set, weak_grounding: bool = False) -> None:
        """Backstop + sidecar assembly for one candidate, so Best-of-2 can score
        two fully-checked payloads."""
        sidecar = cand["sidecar"]
        reply, bs_flags, bl_scrubbed = BS.epistemic_backstop(
            sidecar.get("chat_reply", ""), has_references, retrieved_ids,
            sidecar.get("citations", []), chunk_texts,
            bottom_line=sidecar.get("bottom_line", ""))
        sidecar["chat_reply"] = reply
        sidecar["bottom_line"] = bl_scrubbed
        full = self._sidecar(sidecar, search, has_references, cand["model"],
                             cand["tokens"], pack, cand["notes"], card_ids,
                             weak_grounding)
        cand["bs_flags"] = bs_flags
        cand["full"] = full
        # propagate the healed bottom line into the scored candidate so Best-of-2
        # and the critic see the payload the user will actually get.
        sidecar["bottom_line"] = full.get("bottom_line", "")
        cand["sidecar_flags"] = BS.sidecar_backstop(
            {**sidecar, "citations": full["citations"]}, has_references, retrieved_ids)

    def _best_of_trigger(self, cand: dict, full: dict, flags: list,
                         has_references: bool) -> str | None:
        """High stakes only: guardrail flags, a contested tier, declared stakes,
        or explicit configuration. Simple chat stays a single pass."""
        mode = self.settings.best_of
        if mode == "off" or cand["error"]:
            return None
        if mode == "on":
            return "configured (OEE_BEST_OF=on)"
        hard = [f for f in flags if any(h in f for h in
                ("SCRUBBED", "UNVERIFIED", "QUOTE MISMATCH", "INCONSISTENT", "INCOMPLETE",
                 "WEAK", "CONTRADICT", "UNSUPPORTED", "WARNING"))]
        stakes = bool((full.get("stakes_and_policy") or "").strip())
        reasons = []
        if hard:
            reasons.append(f"{len(hard)} guardrail flag(s)")
        if stakes:
            reasons.append("stakes declared")
        if not (has_references and reasons):
            return None
        return "; ".join(reasons)

    # ------------------------------------------------------------- internals
    def _doc_ids(self) -> list[str]:
        return [d["doc_id"] for d in self.store.documents()]

    def _sidecar(self, sidecar: dict, search: dict, has_references: bool, model: str,
                 gen_tokens: int, pack: dict, gen_notes: list,
                 card_ids: set[str] | None = None,
                 weak_grounding: bool = False) -> dict:
        card_ids = card_ids or set()
        by_id = {r["node_id"]: r for r in search["leaves"] + search["sections"]}
        omitted = sorted({n.split("'")[1] for n in gen_notes
                          if n.startswith("missing required field")})
        derived_notes: list[str] = []

        citations = []
        for row in sidecar.get("citations", []) or []:
            cid = str(row.get("chunk_id", "")).strip()
            node = by_id.get(cid) or self.store.node(cid)
            quote = (row.get("quote") or "").strip()
            auto = False
            if not quote and node:
                quote = quote_span(node["text"])
                auto = True
            row_out = {
                "chunk_id": cid,
                "quote": quote,
                "verified": bool(node and _quote_present(quote, node["text"])),
                "in_retrieved_set": cid in by_id or cid in card_ids,
                "page_hint": (node or {}).get("page_hint", ""),
                "doc_id": (node or {}).get("doc_id", cid.split("-")[0] if "-" in cid else ""),
                "text": (node or {}).get("text", ""),
                "chunk_excerpt": quote_span((node or {}).get("text", "")),
                "auto_derived": auto,
            }
            for k in ("nli_label", "nli_score", "nli_hypothesis", "nli_path"):
                if row.get(k) is not None:
                    row_out[k] = row[k]
            citations.append(row_out)

        if not citations and has_references:
            for cid in dict.fromkeys(CITATION_RE.findall(sidecar.get("chat_reply", ""))):
                node = by_id.get(cid)
                if not node:
                    continue
                citations.append({
                    "chunk_id": cid,
                    "quote": quote_span(node["text"]),
                    "verified": True,
                    "in_retrieved_set": True,
                    "page_hint": node.get("page_hint", ""),
                    "doc_id": node.get("doc_id", cid.split("-")[0]),
                    "text": node["text"],
                    "chunk_excerpt": quote_span(node["text"]),
                    "auto_derived": True,
                })
            if citations:
                derived_notes.append(
                    "citation rows auto-derived from the inline [R#-CX] citations in "
                    "chat_reply; the model omitted the citations array. Quotes are the "
                    "stored chunk text, not model prose."
                )

        basis = list(sidecar.get("basis") or [])
        if not basis:
            basis = ["reference"] if (has_references and citations) else []
            derived_notes.append(
                "basis derived by the engine (model omitted 'basis'): "
                + ("citation grounding exists, so 'reference'" if basis
                   else "no grounding exists, so it stays empty")
            )
        elif has_references and citations and "reference" not in basis:
            basis = ["reference"] + basis
            derived_notes.append(
                "basis auto-corrected by the backstop: the reply cites "
                f"{sorted({c['chunk_id'] for c in citations})[:4]} but the model "
                "omitted 'reference' from basis"
            )
        if has_references and (weak_grounding or
                               search["channels"].get("attached_document_fallback")):
            weak_cos = max([r["cosine"] for r in search["leaves"] + search["sections"]
                            if r.get("cosine") is not None] or [0.0])
            if "parametric" not in basis:
                basis = basis + ["parametric"]
            derived_notes.append(
                "FALLBACK: the retrieved chunks are topically unrelated to the question "
                f"(best dense cosine {weak_cos:.4f} < {self.settings.weak_grounding_cosine}); "
                "the corpus is not a source on this topic. basis corrected by the engine to "
                "include 'parametric': 'reference' grounds the absence verdict only, the "
                "answer's substantive content has no reference grounding."
            )
        if not has_references and "reference" in basis:
            valid_cits = [c for c in citations
                          if c.get("verified") or c.get("in_retrieved_set")]
            if not valid_cits:
                basis = [b for b in basis if b != "reference"]
                derived_notes.append(
                    "basis auto-corrected by the engine: 'reference' removed - every "
                    "cited chunk ID was invented for inline prompt text (no reference "
                    "document was ingested), so nothing in this turn is reference-grounded."
                )
        tiers = P.CONFIDENCE_TIERS
        tier = sidecar.get("confidence_tier", tiers[2])
        tier_reported = "confidence_tier" not in omitted
        if not tier_reported:
            derived_notes.append(
                f"confidence tier not reported by the model; shown as the neutral "
                f"default {tier!r}"
            )
        reply = str(sidecar.get("chat_reply", "") or "")
        if not reply.strip():
            promoted = str(sidecar.get("bottom_line", "") or "").strip()
            if promoted:
                derived_notes.append(
                    "chat_reply came back empty: the bottom line was promoted to the "
                    "reply body so the turn is not blank; the analysis fields carry the "
                    "rest of the reasoning"
                )
                reply = promoted
            else:
                reply = ("The model produced no reply text on this turn. The structured "
                         "fields below are all that exists [CONSTRAINT].")
                derived_notes.append("chat_reply and bottom_line both empty; "
                                     "placeholder reply emitted")
        bottom_line = str(sidecar.get("bottom_line", "") or "").strip()
        if not bottom_line and reply.strip():
            bottom_line = _derive_bottom_line(reply)
            derived_notes.append(
                "bottom_line auto-derived by the engine (the model omitted it): the first "
                "direct sentence of chat_reply, stripped of modal and citation markers. It "
                "is the model's own words, not an invented summary."
            )
        has_data = bool((pack or {}).get("has_data"))
        if has_references:
            provenance_note = ("Grounded in user-provided reference chunks (cited by id) "
                               "plus parametric synthesis. No web access exists in this system.")
        elif has_data:
            provenance_note = (
                "No ingested references provided. The analysis is based on user-supplied "
                "inline data (not chunked, not embedded, not persisted, not cited by id) "
                "plus parametric synthesis. No web access exists in this system.")
        else:
            provenance_note = (
                "No references provided: this is parametric synthesis only. My parametric "
                "knowledge has a training cutoff and there is no web access in this system.")
        warnings = list(sidecar.get("warnings", [])) + derived_notes
        return {
            "chat_reply": reply,
            "bottom_line": bottom_line,
            "dialectic": sidecar.get("dialectic", {}),
            "question_decomposition": sidecar.get("question_decomposition", []),
            "presupposition_audit": sidecar.get("presupposition_audit", []),
            "vocabulary_contract": sidecar.get("vocabulary_contract", []),
            "theory_matrix": sidecar.get("theory_matrix", []),
            "claim_typology": sidecar.get("claim_typology", []),
            "methodological_constraints": sidecar.get("methodological_constraints", []),
            "stakes_and_policy": sidecar.get("stakes_and_policy", ""),
            "fields_omitted_by_model": omitted,
            "tier_reported": tier_reported,
            "confidence_tier_display": tier if tier_reported else "UNDECLARED",
            "field_sources": {
                "confidence_tier": "model_declared" if tier_reported else "engine_default",
                "basis": "model_declared" if list(sidecar.get("basis") or [])
                         else "engine_derived",
                "citations": "model_declared" if sidecar.get("citations")
                             else ("engine_derived" if citations else "absent"),
            },
            "derived_notes": derived_notes,
            "pushback": _as_text(sidecar.get("pushback", "")),
            "epistemic_analysis": _as_text(
                sidecar.get("epistemic_analysis") or sidecar.get("epistemic_status")
                or sidecar.get("analysis") or ""),
            "model_status": _as_lines(sidecar.get("epistemic_status")
                                       if isinstance(sidecar.get("epistemic_status"), (dict, list))
                                       else None),
            "basis_derived": not list(sidecar.get("basis") or []),
            "confidence_tier": tier,
            "tier_index": tiers.index(sidecar.get("confidence_tier", tiers[2]))
                          if sidecar.get("confidence_tier") in tiers else None,
            "confidence_ladder": tiers,
            "basis": basis,
            "citations": citations,
            "assumptions_used": sidecar.get("assumptions_used", []),
            "unknowns": sidecar.get("unknowns", []),
            "what_would_change": sidecar.get("what_would_change", []),
            "warnings": warnings,
            "retrieval": {
                "leaves": [r["node_id"] for r in search["leaves"]],
                "sections": [r["node_id"] for r in search["sections"]],
                "channels": search["channels"],
                "has_references": has_references,
            },
            "provenance_note": provenance_note,
            "data_chars": (pack or {}).get("data_chars", 0),
            "model": model,
            "completion_tokens": gen_tokens,
            "context_tokens": pack["tokens"],
        }

    def _update_case_state(self, session: str, pass1: dict, sidecar: dict) -> dict:
        current = self.store.get_case_state(session)
        rendered = A.render_case_state(current)
        user = P.CASE_STATE_USER_TEMPLATE.format(
            current_case_state=rendered,
            sanitized_request=pass1["sanitized_request"].replace("\n", " "),
            confidence_tier=sidecar.get("confidence_tier", ""),
            chat_reply=sidecar.get("chat_reply", "").replace("\n", " ")[:900],
        )
        notes: list[str] = []
        flags: list[str] = []
        state = dict(current)
        t0 = time.perf_counter()
        latency = 0.0
        try:
            resp = self.provider.chat(P.CASE_STATE_SYSTEM, user, schema=P.CASE_STATE_SCHEMA,
                                      kind="casestate", temperature=0.0)
            latency = resp.latency_ms
            obj = parse_and_repair(resp.text, P.CASE_STATE_SCHEMA, notes)
            for key in ("rolling_summary", "established_claims", "open_questions",
                        "user_constraints", "user_pressure"):
                if key in obj:
                    state[key] = obj[key]
        except ProviderError as exc:
            flags.append(f"DEGRADED: case-state call failed ({exc}); deterministic update used")
        prior_bias = current.get("user_pressure", [])
        state["user_pressure"] = _merge(prior_bias, state.get("user_pressure", []), 4)
        if pass1["bias_removed"]:
            state["user_pressure"] = _merge(
                state["user_pressure"],
                [f"loaded framing: {', '.join(pass1['bias_removed'][:4])}"], 4,
            )
        state["turn_count"] = int(current.get("turn_count", 0)) + 1
        if not state.get("rolling_summary"):
            state["rolling_summary"] = pass1["sanitized_request"][:300]
        self.store.set_case_state(session, state)
        return {"state": state, "latency_ms": round(latency, 2),
                "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
                "flags": flags + notes[:2], "user_prompt": user}

    def history(self, session: str = "default", limit: int = 50) -> list[dict]:
        return self.store.turns(session, limit)

    def simplify(self, text: str, whole: bool = False) -> dict:
        """Display-only plain-language rewrite. It performs NO write: the result is
        never stored, never enters case state, and is never sent back as a turn, so
        it can never be used as context. The guarantee is structural - this path
        touches no store."""
        text = (text or "").strip()
        if not text:
            return {"simplified": "", "whole": whole, "source_chars": 0,
                    "note": "nothing to simplify"}
        scope = ("Rewrite the whole answer below."
                 if whole else "Rewrite the direct bottom-line verdict below.")
        system = P.SIMPLIFY_WHOLE_SYSTEM if whole else P.SIMPLIFY_BOTTOM_SYSTEM
        resp = self.provider.chat(system, f"{scope}\n\n{text}",
                                  schema=None, kind="simplify", temperature=0.3)
        return {"simplified": (resp.text or "").strip(), "whole": whole,
                "source_chars": len(text)}

    def status(self) -> dict:
        health = self.provider.health()
        return {
            "settings": self.settings.as_public_dict(),
            "provider_health": health,
            "index": self.index.stats(),
            "counts": self.store.stats(),
            "semantic": semantic.status(),
            "documents": [
                {"doc_id": d["doc_id"], "title": d["title"], "n_leaves": d["n_leaves"],
                 "n_sections": d["n_sections"], "stance": d["stance"],
                 "limitations": d["limitations"], "chars": d["chars"]}
                for d in self.store.documents()
            ],
        }


def _extract_reply(raw_output: str) -> str:
    try:
        obj = jsonx.extract_json(raw_output)
        return str(obj.get("chat_reply", ""))
    except Exception:
        return ""


def _derive_bottom_line(reply: str) -> str:
    """The model omitted bottom_line: recover it from its own direct answer, not
    invent one. Strip modal/citation/markdown markers, take leading sentences up
    to the concise-answer size. A longer answer is left to finish in peace - no
    ellipsis is appended; the UI shows the full text."""
    clean = " ".join(BOTTOM_STRIP_RE.sub(" ", reply or "").split())
    out = ""
    for sent in BOTTOM_SENT_RE.split(clean):
        sent = sent.strip()
        if not sent:
            continue
        cand = (out + " " + sent).strip()
        if len(cand) > BOTTOM_MAX_CHARS:
            break
        out = cand
        if len(out) >= 80:
            break
    if not out:
        out = clean[:BOTTOM_MAX_CHARS].strip()
    return out


def _as_text(value) -> str:
    """Models return structured metadata where prose is expected. Render it
    readably instead of printing Python repr."""
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(_as_text(v) for v in value if _as_text(v))
    if isinstance(value, dict):
        for probe in ("response", "note", "summary", "text", "answer", "status"):
            if probe in value and _as_text(value[probe]):
                head = _as_text(value[probe])
                rest = [f"{k}: {_as_text(v)}" for k, v in value.items()
                        if k != probe and _as_text(v)]
                return (head + (" | " + "; ".join(rest) if rest else ""))[:1200]
        return "; ".join(f"{k}: {_as_text(v)}" for k, v in value.items() if _as_text(v))[:1200]
    return str(value)


def _as_lines(value) -> list:
    if isinstance(value, dict):
        return [f"{k}: {_as_text(v)}" for k, v in value.items() if _as_text(v)]
    if isinstance(value, list):
        return [_as_text(v) for v in value if _as_text(v)]
    return []


def _merge(first: list, second: list, cap: int) -> list:
    out: list[str] = []
    for item in list(first) + list(second):
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out[:cap]


def _quote_present(quote: str, source: str) -> bool:
    return BS._quote_in(quote, source) if quote else False
