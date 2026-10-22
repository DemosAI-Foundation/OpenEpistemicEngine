from epistemic.retrieval import Retriever


def write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_local_index_ranks_relevant_document_first(tmp_path):
    write(
        tmp_path,
        "productivity.md",
        "Productivity baseline report. Cycle time per ticket was 4.2 days before the project management software "
        "migration. Throughput per sprint: 31 items. Rework rate 12%. Measured from ticket logs.",
    )
    write(tmp_path, "logo.md", "Customer survey about the new logo. 61% of respondents recognized the old logo. "
    "Brand trust score unchanged.")
    retriever = Retriever(corpus_dir=str(tmp_path)).build()
    assert retriever.ready
    hits = retriever.search("project management software productivity baseline cycle time")
    assert hits
    assert "productivity.md" in hits[0].source
    block = retriever.context_block("productivity baseline")
    assert "4.2 days" in block


def test_missing_corpus_is_reported_as_unavailable(tmp_path):
    retriever = Retriever(corpus_dir=str(tmp_path / "absent")).build()
    assert retriever.ready is False
    assert retriever.status()["available"] is False
    assert "none available" in retriever.context_block("anything")
    assert retriever.search("anything") == []


def test_retriever_feeds_the_researcher_packet(tmp_path):
    from epistemic.pipeline import Checkpoint, Engine
    from tests.conftest import CRITIC_PASS, RESEARCHER_B, SKEPTIC_B, TRIAGE_ENGINE, WRITER_B, FakeLLM

    write(tmp_path, "notes.md", "Baseline productivity metrics after the software change: cycle time 6.1 days, throughput 19 items per sprint.")
    client = FakeLLM(
        script={
            "Triage": [TRIAGE_ENGINE],
            "Skeptic": [SKEPTIC_B],
            "Researcher": [RESEARCHER_B],
            "Writer": [WRITER_B],
            "Critic": [CRITIC_PASS],
        }
    )
    retriever = Retriever(corpus_dir=str(tmp_path)).build()
    case = Engine(client=client, retriever=retriever).run("Productivity is down because of the new software.", Checkpoint(action="confirm"))
    text = client.user_text("Researcher")[0]
    assert "RETRIEVAL (local index, non-LLM)" in text
    assert "6.1 days" in text
    assert case.retrieval["status"]["available"] is True
    assert case.retrieval["hits"]
