import numpy as np
import pytest

from oee.engine import Engine
from oee.retrieval import _rrf
from tests.conftest import DOC, stub_settings

QUERY = ("Combining the 1998 drift measurement at 85C with the field failure rate "
         "of 11 percent per year: does the logbook verify the self-calibration claim?")


@pytest.fixture
def fine_engine(tmp_path):
    s = stub_settings(str(tmp_path / "fine.db"))
    s.leaf_chunk_tokens = 45
    s.leaf_chunk_overlap = 8
    s.cluster_size = 2
    eng = Engine(s)
    yield eng
    eng.store.close()


def test_hybrid_search_returns_both_levels(engine):
    engine.ingest(DOC)
    res = engine.index.search(QUERY, leaf_k=3, section_k=2)
    assert res["leaves"] and res["sections"]
    assert [r["rrf"] for r in res["leaves"]] == sorted([r["rrf"] for r in res["leaves"]], reverse=True)
    top = res["leaves"][0]
    for field in ("node_id", "rank_bm25", "rank_dense", "cosine", "bm25", "via"):
        assert field in top
    assert res["channels"]["bm25_hits"] > 0 and res["channels"]["dense_hits"] > 0
    assert res["explain_top"]["matched"]


def test_multi_hop_query_pulls_far_apart_chunks(fine_engine):
    fine_engine.ingest(DOC)
    assert len(fine_engine.store.nodes(("leaf",))) >= 3
    res = fine_engine.index.search(QUERY, leaf_k=6, section_k=2)
    ids = [r["node_id"] for r in res["leaves"]]
    assert "R1-C1" in ids
    assert len(set(ids)) >= 2
    assert any(i != "R1-C1" for i in ids)


def test_section_children_expansion_keeps_grounding(fine_engine):
    fine_engine.ingest(DOC)
    sections = [n["node_id"] for n in fine_engine.store.nodes(("section",))]
    first = fine_engine.store.nodes(("section",))[0]
    leaves, added = fine_engine.index._expand_section_children(
        [], [{"node_id": first["node_id"], "rrf": 0.03, "parents": first["parents"],
              "doc_id": "R1", "text": "x", "page_hint": "p", "rank_bm25": 0,
              "rank_dense": 0, "cosine": 0.0, "bm25": 0.0, "via": "rrf"}], leaf_k=8)
    assert added == len(first["parents"])
    assert [l["node_id"] for l in leaves] == first["parents"]
    assert leaves[0]["via"] == f"section_children:{first['node_id']}"
    assert sections


def test_rrf_favour_cross_channel_consensus():
    fused = dict(_rrf({"bm25": [("a", 9.0), ("b", 8.0)], "dense": [("b", 0.5), ("c", 0.4)]}, 60))
    assert fused["b"] > fused["a"] > fused["c"]


def test_dense_channel_degrades_to_lexical_only(engine):
    engine.ingest(DOC)
    engine.embedder.fallback.encode = lambda texts: np.zeros((len(texts), 256), dtype=np.float32)
    engine.index.refresh(force=True)
    res = engine.index.search(QUERY, leaf_k=3, section_k=1)
    assert res["channels"]["dense_hits"] == 0
    assert res["leaves"]


def test_empty_corpus_search_is_safe(engine):
    res = engine.index.search("anything at all", leaf_k=4, section_k=2)
    assert res["leaves"] == [] and res["sections"] == []


def test_index_is_current_after_ingest(engine):
    engine.ingest(DOC)
    generation = engine.index.generation
    assert engine.index.refresh() is False
    engine.ingest("A second reference. It reports nothing relevant at all.")
    assert engine.index.generation > generation
    assert engine.index.refresh() is False
    assert len(engine.index.cards()) == 2


def test_deleting_a_document_leaves_no_orphan_vectors(tmp_path):
    from oee.store import Store
    st = Store(str(tmp_path / "del.db"))
    st.save_document({"doc_id": "R1", "title": "t", "framing": "", "stance": "",
                      "limitations": [], "n_leaves": 1, "n_sections": 0,
                      "chars": 5, "created_at": 0.0, "ingest_log": []})
    st.save_nodes([{"node_id": "R1-C1", "doc_id": "R1", "kind": "leaf",
                    "text": "alpha beta gamma", "page_hint": "p1", "tokens": 3,
                    "parents": [], "meta": {}, "ord": 0}],
                  {"R1-C1": [0.1, 0.2, 0.3]}, {"R1-C1": "hash-local"})
    assert len(st.vectors()) == 1
    st.delete_document("R1")
    assert st.vectors() == {}
    assert st.nodes() == []
    st.close()


def test_store_startup_prunes_orphan_vectors(tmp_path):
    import struct
    from oee.store import Store
    path = str(tmp_path / "orphan.db")
    st = Store(path)
    st._exec("INSERT INTO vectors (node_id, dim, mode, vec) VALUES (?,?,?,?)",
             ("R9-C9", 3, "hash-local",
              b"".join(struct.pack("<f", x) for x in (0.1, 0.2, 0.3))))
    assert "R9-C9" in st.vectors()
    st.close()
    reopened = Store(path)
    assert reopened.vectors() == {}
    reopened.close()
