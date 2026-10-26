from oee.bm25 import LexicalIndex
from oee.textutil import estimate_tokens, paragraphs, sentences, stem, tokenize


def test_stemming_is_shallow_but_stable():
    assert stem("transformers") == stem("transformer")
    assert stem("measurements") == stem("measurement")
    assert stem("the") == "the"


def test_tokenize_drops_stopwords():
    toks = tokenize("The grid will collapse by Friday")
    assert "grid" in toks and "will" not in toks


def test_estimate_tokens_scales():
    short = estimate_tokens("one two three")
    long = estimate_tokens(" ".join(["word"] * 100))
    assert 0 < short < long


def test_sentence_and_paragraph_split():
    assert len(sentences("A. B. C.")) == 3
    assert paragraphs("a\n\nb") == ["a", "b"]


def test_bm25_prefers_matching_doc():
    idx = LexicalIndex()
    idx.add("a", "the reader panels failed at 11 percent per year in the field array")
    idx.add("b", "the budget committee reviewed procurement and travel expenses")
    hits = idx.search("reader panels failure rate", k=2)
    assert hits and hits[0][0] == "a"
    assert hits[0][1] > 0


def test_bm25_explain_lists_contributing_terms():
    idx = LexicalIndex()
    idx.add("a", "gearbox replacement backlog tickets fleet utilisation")
    exp = idx.explain("gearbox backlog", "a")
    assert exp["score"] > 0
    assert {m["term"] for m in exp["matched"]} & {"gearbox", "backlog"}
