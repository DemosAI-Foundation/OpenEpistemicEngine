"""Phase 4a - Hybrid retrieval: BM25 + dense + Reciprocal Rank Fusion.

Every returned node carries its per-channel rank and fused score so the trace
pane can show exactly why a chunk entered the prompt.
"""

from __future__ import annotations

import time

import numpy as np

from .bm25 import LexicalIndex
from .embed import Embedder
from .store import Store

POOL_MULTIPLIER = 6


class HybridIndex:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.lexical = LexicalIndex()
        self._nodes: dict[str, dict] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        self._generation = -1
        self.generation = 0
        self.refresh()

    def refresh(self, force: bool = False) -> bool:
        generation = len(self.store.nodes(("leaf", "section")))
        if self._generation == generation and not force:
            return False
        self.lexical = LexicalIndex()
        self._nodes = {}
        self._ids = []
        rows = []
        vectors = self.store.vectors()
        for node in self.store.nodes(("leaf", "section")):
            self._nodes[node["node_id"]] = node
            self.lexical.add(node["node_id"], node["text"])
            vec = vectors.get(node["node_id"])
            if vec:
                self._ids.append(node["node_id"])
                rows.append(vec[0])
        self._matrix = np.asarray(rows, dtype=np.float32) if rows else None
        if self._matrix is not None:
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            self._matrix = self._matrix / np.where(norms == 0, 1.0, norms)
        self._generation = generation
        self.generation += 1
        return True

    def search(self, query: str, leaf_k: int = 8, section_k: int = 3,
               rrf_k: int = 60) -> dict:
        self.refresh()
        t0 = time.perf_counter()
        pool = max(leaf_k, section_k) * POOL_MULTIPLIER
        lexical_rank = self.lexical.search(query, k=pool)
        dense_rank, dense_scores = self._dense(query, pool)
        fused = _rrf({"bm25": lexical_rank, "dense": dense_rank}, rrf_k)
        lexical_pos = {d: i for i, (d, _) in enumerate(lexical_rank)}
        dense_pos = {d: i for i, (d, _) in enumerate(dense_rank)}
        dense_val = dict(dense_rank)
        bm25_val = dict(lexical_rank)

        leaves: list[dict] = []
        sections: list[dict] = []
        for node_id, score in fused:
            node = self._nodes.get(node_id)
            if not node:
                continue
            row = self._row(node, score, lexical_pos, dense_pos, dense_val, bm25_val)
            if node["kind"] == "leaf" and len(leaves) < leaf_k:
                leaves.append(row)
            elif node["kind"] == "section" and len(sections) < section_k:
                sections.append(row)
            if len(leaves) >= leaf_k and len(sections) >= section_k:
                break
        leaves, children_added = self._expand_section_children(leaves, sections, leaf_k)
        fallback_used = False
        if not leaves and not sections and self._nodes:
            sections, leaves, fallback_used = self._attached_document_fallback(sections,
                                                                             leaves,
                                                                             section_k, leaf_k)
        top = (leaves + sections)[:1]
        explain = self.lexical.explain(query, top[0]["node_id"]) if top else {}
        return {
            "leaves": leaves,
            "sections": sections,
            "channels": {
                "bm25_hits": len(lexical_rank),
                "dense_hits": len(dense_rank),
                "rrf_k": rrf_k,
                "fused_pool": len(fused),
                "section_children_added": children_added,
                "attached_document_fallback": fallback_used,
            },
            "explain_top": explain,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "index_generation": self.generation,
        }

    def _attached_document_fallback(self, sections, leaves, section_k, leaf_k):
        """Attaching a document is an explicit intent signal. When nothing
        matches lexically or densely, the tree is supplied as grounding anyway,
        in document order, and the trace says so."""
        by_doc: dict[str, dict] = {}
        for node in self._nodes.values():
            by_doc.setdefault(node["doc_id"], {"leaves": [], "sections": []})
            if node["kind"] == "leaf":
                by_doc[node["doc_id"]]["leaves"].append(node)
            elif node["kind"] == "section":
                by_doc[node["doc_id"]]["sections"].append(node)
        for bundle in by_doc.values():
            bundle["leaves"].sort(key=lambda n: n["ord"])
            bundle["sections"].sort(key=lambda n: n["ord"])
        ordered = sorted(by_doc.items(), key=lambda kv: -len(kv[1]["leaves"]))
        used = 0
        for doc_id, bundle in ordered:
            for node in bundle["leaves"]:
                if used >= leaf_k:
                    break
                leaves.append(self._fallback_row(node, doc_id))
                used += 1
            if used >= leaf_k:
                break
        sec_used = 0
        for doc_id, bundle in ordered:
            for node in bundle["sections"]:
                if sec_used >= section_k:
                    break
                sections.append(self._fallback_row(node, doc_id))
                sec_used += 1
            if sec_used >= section_k:
                break
        return sections, leaves, bool(leaves)

    def _fallback_row(self, node: dict, doc_id: str) -> dict:
        return {
            "node_id": node["node_id"], "doc_id": doc_id, "kind": node["kind"],
            "text": node["text"], "page_hint": node["page_hint"], "rrf": 0.0,
            "rank_bm25": None, "rank_dense": None, "cosine": 0.0, "bm25": 0.0,
            "parents": node["parents"], "via": "attached_document_fallback",
        }

    def _row(self, node: dict, score: float, lexical_pos: dict, dense_pos: dict,
             dense_val: dict, bm25_val: dict) -> dict:
        node_id = node["node_id"]
        return {
            "node_id": node_id,
            "doc_id": node["doc_id"],
            "kind": node["kind"],
            "text": node["text"],
            "page_hint": node["page_hint"],
            "rrf": round(score, 6),
            "rank_bm25": lexical_pos.get(node_id),
            "rank_dense": dense_pos.get(node_id),
            "cosine": round(float(dense_val.get(node_id, 0.0)), 4),
            "bm25": round(float(bm25_val.get(node_id, 0.0)), 4),
            "parents": node["parents"],
            "via": "rrf",
        }

    def _expand_section_children(self, leaves: list[dict], sections: list[dict],
                                 leaf_k: int) -> tuple[list[dict], int]:
        """RAPTOR tree traversal: a matched section pulls its verbatim children,
        so section-level matches stay quotable at leaf level."""
        floor = max(2, leaf_k // 3)
        added = 0
        if len(leaves) >= floor or not sections:
            return leaves, added
        have = {row["node_id"] for row in leaves}
        for sec in sections:
            for child_id in sec["parents"]:
                if child_id in have:
                    continue
                child = self._nodes.get(child_id)
                if not child:
                    continue
                row = dict(
                    node_id=child_id, doc_id=child["doc_id"], kind=child["kind"],
                    text=child["text"], page_hint=child["page_hint"], rrf=sec["rrf"],
                    rank_bm25=None, rank_dense=None, cosine=0.0, bm25=0.0,
                    parents=child["parents"], via=f"section_children:{sec['node_id']}",
                )
                leaves.append(row)
                have.add(child_id)
                added += 1
                if len(leaves) >= leaf_k:
                    return leaves, added
        return leaves, added

    def _dense(self, query: str, pool: int) -> tuple[list[tuple[str, float]], dict]:
        if self._matrix is None or not self._ids:
            return [], {}
        qvec, _mode = self.embedder.encode_one(query)
        q = np.asarray(qvec, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n == 0:
            return [], {}
        q = q / n
        sims = self._matrix @ q
        order = np.argsort(-sims)[:pool]
        out = []
        for i in order:
            s = float(sims[i])
            if s <= 0.0:
                continue
            out.append((self._ids[int(i)], s))
        return out, dict(out)

    def cards(self) -> list[dict]:
        return self.store.nodes(("card",))

    def stats(self) -> dict:
        return {
            "nodes_indexed": len(self._nodes),
            "dense_vectors": len(self._ids),
            "vector_modes": self.store.vector_mode_counts(),
            "generation": self.generation,
            "embed_last_mode": self.embedder.last_mode,
        }


def _rrf(rankings: dict[str, list[tuple[str, float]]], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for _channel, ranked in rankings.items():
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
