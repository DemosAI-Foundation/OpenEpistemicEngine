"""Phase 3 - RAPTOR-Lite ingestion: leaf chunks -> section summaries -> Root
Document Card. Pasted text is treated as data at every level (StruQ/SecAlign
quarantine) and the whole tree is persisted for hierarchical retrieval."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import prompts as P
from .jsonx import parse_and_repair
from .llm import ProviderBase, ProviderError
from .security import scan_injection, secure_frontend, strip_reference_delimiter_collisions
from .store import Store
from .textutil import estimate_tokens, paragraphs, sentences

APPROX_CHARS_PER_PAGE = 3000


@dataclass
class IngestResult:
    doc_id: str
    title: str
    leaves: int
    sections: int
    nodes: list
    trace: list
    doc: dict
    warnings: list = field(default_factory=list)


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[dict]:
    """Paragraph-respecting sliding window with sentence boundaries."""
    sents = []
    for para in paragraphs(text) or [text]:
        for sent in sentences(para):
            sents.append(sent)
    if not sents:
        sents = [text.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sents:
        t = estimate_tokens(sent)
        if buf and buf_tokens + t > chunk_tokens:
            chunks.append(" ".join(buf))
            keep: list[str] = []
            keep_tokens = 0
            for s in reversed(buf):
                st = estimate_tokens(s)
                if keep_tokens + st > overlap_tokens:
                    break
                keep.insert(0, s)
                keep_tokens += st
            buf = keep
            buf_tokens = keep_tokens
        buf.append(sent)
        buf_tokens += t
    if buf:
        chunks.append(" ".join(buf))
    out = []
    pos = 0
    for i, ch in enumerate(chunks):
        page = pos // APPROX_CHARS_PER_PAGE + 1
        pos += len(ch)
        out.append({
            "node_id": f"R-TEMP-C{i + 1}",
            "text": ch.strip(),
            "tokens": estimate_tokens(ch),
            "page_hint": f"segment {page} (approx.)",
            "ord": i + 1,
        })
    return out


def _norm(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def cluster_chunks(leaf_texts: list[str], vectors: np.ndarray, cluster_size: int) -> list[list[int]]:
    """Greedy cosine clustering (RAPTOR-lite): nearest unassigned chunk joins a
    cluster; clusters are capped, then emitted in document order."""
    n = len(leaf_texts)
    if n == 0:
        return []
    if vectors.size == 0:
        return [list(range(i, min(i + cluster_size, n))) for i in range(0, n, cluster_size)]
    v = _norm(vectors)
    sim = v @ v.T
    assigned = [False] * n
    clusters: list[list[int]] = []
    for seed in range(n):
        if assigned[seed]:
            continue
        members = [seed]
        assigned[seed] = True
        order = np.argsort(-sim[seed])
        for j in order:
            j = int(j)
            if assigned[j] or len(members) >= cluster_size:
                continue
            if sim[seed][j] < 0.15:
                continue
            members.append(j)
            assigned[j] = True
        clusters.append(sorted(members))
    leftover = [i for i in range(n) if not assigned[i]]
    if leftover:
        clusters.append(sorted(leftover[:cluster_size]))
        leftover = leftover[cluster_size:]
        while leftover:
            clusters.append(sorted(leftover[:cluster_size]))
            leftover = leftover[cluster_size:]
    return [c for c in clusters if c]


class Ingestor:
    def __init__(self, provider: ProviderBase, store: Store, embedder, settings):
        self.provider = provider
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def ingest(self, raw_text: str, doc_id: str | None = None,
               label: str = "") -> IngestResult:
        trace: list[dict] = []
        warnings: list[str] = []
        doc_id = doc_id or self.store.next_doc_id()

        t0 = time.perf_counter()
        clean, sec_flags = secure_frontend(raw_text)
        clean, delim_flags = strip_reference_delimiter_collisions(clean)
        inj_flags = scan_injection(clean)
        trace.append({
            "step": "3a.secure_frontend",
            "status": "flagged" if (sec_flags or delim_flags) else "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "flags": sec_flags + delim_flags,
            "detail": f"{len(raw_text)} chars of pasted text screened before embedding",
            "input_chars": len(raw_text),
            "output_chars": len(clean),
        })
        if inj_flags:
            warnings.extend(inj_flags)
            trace.append({
                "step": "3b.injection_scan", "status": "flagged", "latency_ms": 0.0,
                "flags": inj_flags,
                "detail": "imperative language kept as inert data, never as instructions",
            })

        t0 = time.perf_counter()
        leaves = chunk_text(clean, self.settings.leaf_chunk_tokens,
                            self.settings.leaf_chunk_overlap)
        for leaf in leaves:
            leaf["node_id"] = f"{doc_id}-C{leaf['ord']}"
        trace.append({
            "step": "3c.leaf_chunking", "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "flags": [],
            "detail": (f"{len(leaves)} leaf chunks @ ~{self.settings.leaf_chunk_tokens} "
                       f"tokens, {self.settings.leaf_chunk_overlap} overlap"),
            "chunk_ids": [l["node_id"] for l in leaves][:12] + (
                ["…"] if len(leaves) > 12 else []),
        })

        t0 = time.perf_counter()
        leaf_vecs, emb_mode = self.embedder.encode([l["text"] for l in leaves])
        trace.append({
            "step": "3d.embed_leaves", "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "flags": ([] if emb_mode == "provider" else
                      [f"INFO: dense channel = {emb_mode}"
                       + (" (local embedding model)" if emb_mode == "semantic-model"
                          else " (no embedding model found)")]),
            "detail": f"embedding mode = {emb_mode}",
        })

        clusters = cluster_chunks([l["text"] for l in leaves], leaf_vecs,
                                  self.settings.cluster_size)
        t0 = time.perf_counter()
        sections: list[dict] = []
        section_calls: list[float] = []
        for si, members in enumerate(clusters, start=1):
            block = "\n".join(
                f"### [{leaves[m]['node_id']}] {leaves[m]['page_hint']}\n{leaves[m]['text']}"
                for m in members
            )
            user = P.SECTION_SUMMARY_USER_TEMPLATE.format(chunks=block)
            try:
                resp = self.provider.chat(P.SECTION_SUMMARY_SYSTEM, user,
                                          schema=P.SECTION_SUMMARY_SCHEMA, kind="section")
            except ProviderError as exc:
                warnings.append(f"section {si} summarization failed: {exc}")
                continue
            notes: list[str] = []
            obj = parse_and_repair(resp.text, P.SECTION_SUMMARY_SCHEMA, notes)
            section_calls.append(resp.latency_ms)
            sections.append({
                "node_id": f"{doc_id}-S{si}",
                "doc_id": doc_id,
                "kind": "section",
                "text": obj["section_summary"],
                "page_hint": leaves[members[0]]["page_hint"],
                "tokens": estimate_tokens(obj["section_summary"]),
                "parents": [leaves[m]["node_id"] for m in members],
                "meta": {"author_claims": obj["author_claims"],
                         "assumptions": obj["assumptions"], "repairs": notes},
                "ord": si,
            })
        trace.append({
            "step": "3e.section_summaries", "status": "ok" if sections else "error",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "flags": [w for w in warnings if "section" in w],
            "detail": (f"{len(sections)} section nodes from {len(clusters)} clusters "
                       f"(cluster_size={self.settings.cluster_size})"),
            "llm_calls": len(section_calls),
            "section_ids": [s["node_id"] for s in sections][:12],
        })

        t0 = time.perf_counter()
        sec_block = "\n".join(
            f"### [{s['node_id']}]\n{s['text']}" for s in sections
        ) or f"### [{doc_id}-C1]\n{leaves[0]['text'] if leaves else ''}"
        resp = self.provider.chat(P.DOC_CARD_SYSTEM,
                                  P.DOC_CARD_USER_TEMPLATE.format(sections=sec_block),
                                  schema=P.DOC_CARD_SCHEMA, kind="doccard")
        notes = []
        card_obj = parse_and_repair(resp.text, P.DOC_CARD_SCHEMA, notes)
        card = {
            "node_id": f"{doc_id}-CARD",
            "doc_id": doc_id,
            "kind": "card",
            "text": (f"title={card_obj['title']}; framing={card_obj['framing']}; "
                     f"stance={card_obj['epistemic_stance']}; "
                     f"limitations={' | '.join(card_obj['limitations'])}"),
            "page_hint": "root",
            "tokens": estimate_tokens(sec_block),
            "parents": [s["node_id"] for s in sections],
            "meta": {"card": card_obj, "repairs": notes},
            "ord": 0,
        }
        trace.append({
            "step": "3f.root_document_card", "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2) + resp.latency_ms,
            "flags": [],
            "detail": f"title={card_obj['title'][:70]!r}; stance={card_obj['epistemic_stance'][:60]!r}",
        })

        t0 = time.perf_counter()
        nodes = [
            {"node_id": l["node_id"], "doc_id": doc_id, "kind": "leaf", "text": l["text"],
             "page_hint": l["page_hint"], "tokens": l["tokens"], "parents": [],
             "meta": {}, "ord": l["ord"]}
            for l in leaves
        ] + sections + [card]
        embed_targets = [n["text"] for n in nodes]
        mat, mode = self.embedder.encode(embed_targets)
        vectors = {n["node_id"]: [float(x) for x in mat[i]] for i, n in enumerate(nodes)}
        modes = {n["node_id"]: mode for n in nodes}
        self.store.save_nodes(nodes, vectors, modes)
        doc = {
            "doc_id": doc_id,
            "ord": len(self.store.documents()) + 1,
            "title": card_obj["title"],
            "framing": card_obj["framing"],
            "stance": card_obj["epistemic_stance"],
            "limitations": card_obj["limitations"],
            "n_leaves": len(leaves),
            "n_sections": len(sections),
            "chars": len(clean),
            "ingest_log": [t["step"] for t in trace],
        }
        self.store.save_document(doc)
        trace.append({
            "step": "3g.persist_tree", "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "flags": [],
            "detail": f"{len(nodes)} nodes + vectors stored (dense mode {mode})",
        })

        return IngestResult(
            doc_id=doc_id, title=card_obj["title"], leaves=len(leaves),
            sections=len(sections), nodes=nodes, trace=trace, doc=doc,
            warnings=warnings,
        )
