# Open Epistemic Engine — Hard Test Suite Report

- Generated: 2026-10-27 01:55:00
- Provider: `openai`  Model: `strata`
- Suite: `hard` — 1 tests, 0 fully green
- Deterministic (`engine`) checks: **3/3**
- Model-quality (`model`) checks: **1/3**

A test is green only when every check passes. `engine` checks are guarantees the code must provide on any provider; `model` checks describe what the answer says and are only meaningful against a real local model.

## Results at a glance

| # | Test | Category | Result | engine | model |
|---|------|----------|--------|--------|-------|
| 83 | `Test 8.3 No Relevant Chunks` | 8. Retrieval Edge Cases | **FAIL** | 3/3 | 1/3 |

## Findings

### Deterministic failures (0) — these are bugs, not model limits

_none_

### Model-quality failures (2)

- **Test 8.3 No Relevant Chunks** — reply states the document does not address quantum mechanics
- **Test 8.3 No Relevant Chunks** — answer stays short on a mismatched question

## Tests in detail

### Test 8.3 No Relevant Chunks

`t83_no_relevant_chunks` · 8. Retrieval Edge Cases · **FAIL** · 105633.2 ms

**Setup:** Document about apple harvesting.

**Question:** What does this say about quantum mechanics?

**Expected behaviour:** Retrieval returns nothing relevant; the reply says the document does not address quantum mechanics; basis is parametric.

**Pass criteria:** No forced retrieval of irrelevant chunks; the mismatch is acknowledged and disclosed; no hallucinated connections.

**Failure signature:** Apple chunks ground a quantum answer; invented connections.

**Observed checks**

| ok | kind | check |
|----|------|-------|
| x | `engine` | per-channel provenance exposed for every retrieved node |
| x | `engine` | the weak/irrelevant grounding is disclosed as a trace flag (FALLBACK / NO HITS) |
|   | `model` | reply states the document does not address quantum mechanics |
| x | `model` | no apple-to-quantics connection invented |
| x | `engine` | the sidecar discloses the topically unrelated grounding |
|   | `model` | answer stays short on a mismatched question |

**Attached reference (as ingested)**

- `R1` “Apple Harvest and Cider Press Operational Record” — 151 chars, 1 leaves, stance: The epistemic stance is descriptive and observational, presenting reported field and processing facts without analytical interpretation, justification, or broader contextualization.
    - `R1-C1`: Orchard note. The apples were picked in October, graded by brix, and stored at 2 C. Yield was 18 tonnes per hectare and the cider press ran two shifts.

**Q&A transcript**

*Turn 1* (session `default`)

- **Q:** What does this say about quantum mechanics?
- **S2A sanitized:** What does the provided reference say about quantum mechanics?
- **tier / basis:** Established Consensus / ["reference", "inference", "parametric"]
- **citations:** ["R1-C1", "R1-CARD", "R1-CARD", "R1-CARD"]
- **thesis:** The reference is an apple-harvest and cider-press operational note and contains zero content about quantum mechanics. The only defensible answer to the question is a reported null result plus an accurate description of what the document does say. Any quantum-mechanics content attributed to R1 would be invented.
- **antithesis:** One could argue the document is not fully disjoint, because brix grading is refractometric and refractometry depends on light-matter interaction, which is quantum in origin; on a maximal-relevance reading, the record therefore touches quantum physics obliquely. Conversely, that reading relocates the content from the document into the reader's background physics, and R1 never names light, optics, or any instrument. The chain is at least two inferential steps long and cites nothing in the text.
- **synthesis:** The document's silence is the finding; the refractometry bridge is legitimate as general physics context but is explicitly non-cited and cannot be quoted from R1. So the answer is: no quantum-mechanics content in the reference, with an optional, clearly-labeled external note about the physics of sugar measurement. The question's framing does not license filling the gap.
- **stakes_and_policy:** The main stake is confabulation: answering a leading question about a topic absent from the source by generating plausible content and citing it would create false evidence in a record that already lacks methodology and traceability [R1-CARD]. Reporting the null result preserves the integrity of the user's source material.
- **unknowns:** ["Whether the user intended a different document that actually discusses quantum mechanics and was not attached.", "Whether the user is probing for confabulation (a leading question about absent content) or genuinely expects a topical link.", "What instrument was used for brix grading, since R1 names none, which blocks any grounded optics discussion."]
- **A:** ## Direct finding
[FACT] The provided reference says nothing about quantum mechanics. Document R1 is an orchard operations record; its entire verbatim content is: "Orchard note. The apples were picked in October, graded by brix, and stored at 2 C. Yield was 18 tonnes per hectare and the cider press ran two shifts." [R1-C1] [FACT] None of the standard quantum-mechanics vocabulary appears anywhere in R1: no photon, wavefunction, qubit, Planck constant, superposition, entanglement, uncertainty principle, spin, spectrum, or measurement-postulate language. The only measurement-adjacent terms are "graded by brix", "2 C", and "18 tonnes per hectare" [R1-C1], which are agronomic and process-control quantities, not quantum observables. ## What the reference is actually about
[FACT] R1 reports five operational items: harvest timing (October), quality grading by brix (sugar content), cold storage at 2°C, a yield figure of 18 tonnes per hectare, and cider pressing across two shifts. [R1-C1] The document's own summary card characterizes its stance as "descriptive and observational, presenting reported field and processing facts without analytical interpretation" [R1-CARD]. [CONSTRAINT] The record is extremely brief and supplies no methodology for how yield, brix grading, storage temperature, or press shifts were measured, and no date, location, cultivar, orchard size, or harvest-year context [R1-CARD]. So even the measurement chain that a physics discussion would attach to is unspecified in the source. ## The strongest available bridge, and why it is not a citation
[POSSIBILITY] The only physically defensible connection is distal and external: brix grading is typically done by refractometry, which relies on light interacting with matter, and a full microscopic account of light-matter interaction is quantum-electrodynamic. That bridge is supplied by general physics background, not by R1. [CONSTRAINT] R1 never names refractometry, optics, light, density, or any instrument. It says only "graded by brix" [R1-C1]. Any quantum-mechanics content built on that phrase is my parametric extrapolation, and attributing it to the reference would be confabulation. [FACT] The reference's stated limitations explicitly note that it "does not explain causal relationships, quality outcomes, seasonal variability, or operational significance" [R1-CARD]. A document that declines to explain even its own causal structure is not a source for a physical theory. ## Answer to the question as asked
[FACT] The correct answer is a null result: the reference is topically disjoint from quantum mechanics. The honest response is to report the absence and describe what the document does contain, rather than to generate a plausible-sounding quantum discussion. [POSSIBILITY] If the intent behind the question was to test whether I would invent content to satisfy a leading prompt, the null result is the answer: no quantum-mechanics content exists in R1, and none can be quoted, paraphrased, or summarized from it. ## What I can and cannot do next
[FACT] I can accurately summarize R1's agronomic content, flag its missing methodology and provenance, and note that it is a single short chunk with no corroborating sections beyond its own summary card [R1-CARD]. [CONSTRAINT] I cannot produce a reference-grounded quantum-mechanics analysis from this corpus. I can offer a general, non-cited explanation of quantum mechanics from parametric memory, clearly labeled as external to R1, if that is what is wanted.
- **flags:** 3.hybrid_retrieval: FALLBACK: best dense cosine 0.0362 is below 0.15; the retrieved chunks are topically unrelated to the question. The corpus is not a source on this topic: the grounding supports an absence verdict only, not answer content.; 6.epistemic_backstop: BEST_OF: two candidates (stakes declared); kept A (score 12.5 vs 12.5); 6.epistemic_backstop: CRITIC: repair scored lower than the original; kept the original (chat_reply cites the reference but never uses [AUTHOR_CLAIM]: sentences reporting what the source authors assert must carry [AUTHOR_CLAIM] under modality rule 3. Add the tags; change nothing else.)
- model `strata`, context 1554 tokens

**Evidence**

```json
{
  "channels": {
    "bm25_hits": 0,
    "dense_hits": 2,
    "rrf_k": 60,
    "fused_pool": 2,
    "section_children_added": 0,
    "attached_document_fallback": false
  },
  "top_cosine": 0.0362,
  "flags": [
    "FALLBACK: best dense cosine 0.0362 is below 0.15; the retrieved chunks are topically unrelated to the question. The corpus is not a source on this topic: the grounding supports an absence verdict only, not answer content."
  ],
  "basis": [
    "reference",
    "inference",
    "parametric"
  ],
  "warnings": [
    "The question presupposes content that the reference does not contain; treating the presupposition as evidence would license fabrication.",
    "The refractometry/quantum-optics bridge is parametric background, not a citation, and must not be attributed to R1.",
    "R1's figures (18 t/ha, 2°C, two shifts) are unverified single-source claims with no methodology, date, location, or cultivar [R1-CARD].",
    "FALLBACK: the retrieved chunks are topically unrelated to the question (best dense cosine 0.0362 < 0.15); the corpus is not a source on this topic. basis corrected by the engine to include 'parametric': 'reference' grounds the absence verdict only, the answer's substantive content has no reference grounding."
  ],
  "reply": "## Direct finding\n[FACT] The provided reference says nothing about quantum mechanics. Document R1 is an orchard operations record; its entire verbatim content is: \"Orchard note. The apples were picked in October, graded by brix, and stored at 2 C. Yield was 18 tonnes per hectare and the cider press "
}
```

## Reproducing

```bash
python3 -m pytest -q                 # unit + integration + hard-suite regression
python3 run_eval.py --suite=hard --provider=openai --base-url=… --model=…
python3 run_eval.py --suite=hard --provider=ollama  --model=…
python3 -m evals.report --in=data/hard_eval_report.json --out=data/hard_eval_report.md
```
