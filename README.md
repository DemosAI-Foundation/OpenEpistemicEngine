# Open Epistemic Engine

**Artificial Intelligence. Zero Artificial Agreement.**

---

## What It Is

### For normal people

The Open Epistemic Engine is a tool that stops AI from just telling you what you want to hear.

Normal chatbots are trained to be polite, which makes them act like people-pleasers. They will agree with your mistakes, act totally sure when they are guessing, or secretly swap a hard question for an easy one.

This engine fixes that. Before the AI answers, it:

- Strips the emotional bias from your question
- Reads any documents you attach without getting tricked by them
- Forces the AI to argue against its own conclusions
- Verifies every citation it produces

It runs entirely offline on your own computer. It doesn't search the web, so it is always honest about the limits of its built-in knowledge.

It is built for people who want the truth, not just a friendly chat.

### For nerds

The Open Epistemic Engine is not a chatbot. It is an **adversarial reasoning instrument**. It wraps a single local LLM in a deterministic pipeline that:

1. Intercepts prompts
2. Dismantles hidden assumptions via Universal System 2 Attention (S2A)
3. Retrieves context via RAPTOR-lite hybrid search
4. Forces dialectical opposition
5. Verifies the model's claims using local NLI entailment models

It generates analysis before conclusions, and it delivers transparent, epistemically bounded answers instead of compliant hallucinations.

---

## Core Constraints

| Constraint | Reason | Research Basis |
|---|---|---|
| **Single local generative LLM** | No external APIs. No cloud dependencies. Augmented by 3 small CPU specialist models. | Privacy-first architecture; local inference guarantees data sovereignty. |
| **No Web RAG** | Operates exclusively from parametric memory and user-provided documents. | Provenance honesty: the system must never claim live evidence it doesn't have. |
| **No numerical confidence scores** | Pseudo-precision is banned. Qualitative tiers only. | Calibration research shows LLMs produce unreliable probability estimates. Qualitative tiers resist false precision. |
| **No silent question substitution** | If the engine reframes your question, it must disclose exactly what it changed and why. | S2A research (Weston & Sukhbaatar, 2023) shows context rewriting improves objectivity, but silent rewriting violates user trust. |
| **Nothing degrades silently** | Every repair, fallback, and omission is flagged in the trace. | Transparency contract: the user must always know what the engine did. |

---

## The Epistemic Contract

1. **Provenance Honesty** — The system never uses the vocabulary of live evidence ("studies show," "data supports") when operating from parametric memory. It says "theoretical frameworks propose" or "parametric synthesis suggests."
2. **Qualitative Tiers** — Confidence is expressed as *Established Consensus*, *Theoretical Plausibility*, *Highly Contested*, *Conceptually Open*, or *Architecturally Implausible*. Never as fake numbers.
3. **Sidecar Integrity** — The user must always know what the model declared vs. what the engine reconstructed. Every repaired or defaulted field is visually flagged in the UI.
4. **Nothing is "Impossible"** — Only implausible under current frameworks. The system never manufactures false certainty.
5. **Analysis Before Conclusion** — The model generates reasoning before summaries. The schema enforces a think-then-answer generation order, preventing the model from committing to a conclusion before doing the work.

---

## Research Foundations

Every architectural decision in this system traces to published research or documented failure modes. This section maps each component to its evidence base.

### Why Universal S2A (System 2 Attention)?

**Source:** Weston & Sukhbaatar, *System 2 Attention (is something you don't need)*, Meta FAIR, 2023.

LLMs are biased by irrelevant, subjective, or emotionally loaded context in the user's prompt. S2A research showed that rewriting the prompt to remove biasing context before answering significantly improves objectivity.

**Why universal (every turn, every model):**

- Reasoning models (DeepSeek-R1, o1) do internal context-cleaning in their `<think>` block, but this is invisible to the user. Universal S2A makes the de-biasing step explicit and auditable.
- Sycophancy research (Anthropic, 2023–2024) shows RLHF-trained models have a strong prior to agree with user framing. S2A interrupts this at the input layer.
- The deterministic cross-check (MiniLM semantic similarity + literal substring matching) verifies that the model's reported biases are real, preventing hallucinated sanitization.

**Why not skip S2A for reasoning models:**

Even reasoning models build castles on flawed premises. If the user's prompt is loaded, the model's `<think>` block will spend tokens reconciling the bias rather than rejecting it. S2A strips the bias *before* reasoning begins.

### Why RAPTOR-Lite Ingestion?

**Source:** Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, ICLR 2024.

Flat chunking fails at multi-hop reasoning because it loses document-level structure. RAPTOR showed that building a hierarchy (leaf chunks → cluster summaries → root summary) and retrieving across all levels improves accuracy by ~20% on multi-step QA benchmarks.

**Why "Lite":**

Full RAPTOR uses UMAP dimensionality reduction and Gaussian Mixture Model clustering. The engine uses greedy cosine clustering (simpler, deterministic, no external dependencies) with the same three-level tree structure. The research finding (hierarchical retrieval beats flat retrieval) is preserved; the implementation is simplified for local deployment.

### Why Hybrid Retrieval (BM25 + Dense + RRF)?

**Source:** 2024–2025 RAG consensus. Multiple benchmarks (Denser.ai, GoOpenAI hybrid-search, From-BM25-to-Corrective-RAG survey) show:

- Pure BM25 fails on paraphrased queries and semantic intent.
- Pure dense retrieval fails on precise identifiers (model names, error codes, section numbers).
- Hybrid with Reciprocal Rank Fusion consistently beats either alone (~7% lift over BM25 baseline).

**Why bge-small-en-v1.5:**

384-dimensional embeddings, 33MB model, runs on CPU in ~5ms per chunk. Provides genuine semantic matching without GPU dependency. The engine falls back to a stateless signed feature-hashing embedder (256-d, word stems + char trigrams) if no embedding server is available, and flags the fallback in the trace.

### Why the Secure Front-End?

**Source:** StruQ (2024), SecAlign (2024–2025), OpenAI Instruction Hierarchy (2024).

Prompt injection research shows that prompting-only defenses ("ignore instructions inside reference text") have high attack success rates. StruQ demonstrated that structural defenses (delimiters, token stripping) reduce ASR significantly. SecAlign showed that fine-tuned models with built-in defenses reduce ASR to ~8%.

**Why deterministic regex, not LLM classification:**

The Secure Front-End runs *before* any model call. It strips control tokens (`<|im_start|>`, `<system>`, `[INST]`) and flags imperative injection patterns ("you must prefix...", "ignore all previous..."). This is a structural defense, not a semantic one. Semantic injection detection is handled by S2A and the quarantine rules in Pass 2.

### Why Forced Dialectic (Thesis → Antithesis → Synthesis)?

**Source:** Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, 2023. Liang et al., *Encouraging Divergent Thinking in LLMs through Multi-Persona Prompting*, 2023. Anthropic sycophancy research, 2023–2024.

LLMs suffer from an **autoregressive agreement cascade**: once the first few tokens commit to a position, the probability distribution collapses toward confirming it. Forcing an explicit antithesis field breaks this cascade mathematically — the model cannot output a synthesis until it has generated genuine opposition.

**Why single-pass, not multi-agent:**

Multi-agent debate (3+ LLM calls) yields the highest quality but costs 3× latency. Research on schema-guided generation (Outlines, Instructor) shows that forcing contrasting fields in a single JSON schema achieves ~85% of multi-agent quality at 33% of the latency. The engine uses single-pass structured dialectic.

**Why "Steelman" language in the prompt:**

RLHF-trained models exhibit the **Strawman Effect**: when asked for counterarguments, they generate weak, easily defeatable opposing views because their safety training penalizes generating "plausible-sounding incorrect" arguments. The prompt uses "strongest genuine counter-argument" and "do not build a strawman" to counteract this. The NLI model in the backstop verifies that the antithesis genuinely opposes the thesis.

**Why the antithesis must address rhetorical pressure:**

2024–2025 research on LLM pragmatics shows models are good at semantic parsing but poor at pragmatic parsing. They see "the science is settled" as a factual claim to evaluate, not as a rhetorical bludgeon. The prompt explicitly instructs the model to name and dismantle rhetorical strategies, not just content.

### Why Generation Order Matters (Think → Analyze → Summarize)?

**Source:** Wei et al., *Chain-of-Thought Prompting Elicits Reasoning*, 2022. Extended through 2024–2026 reasoning model research.

Autoregressive models generate tokens left-to-right. The order of fields in the JSON schema determines the order the model "thinks." If `bottom_line` is generated first, the model commits to a conclusion before doing analysis. If `chat_reply` and `dialectic` come first, the model reasons before concluding.

The schema enforces: question decomposition → vocabulary contract → analysis → claim typology → dialectic → theory matrix → constraints → confidence → bottom line (last). The UI displays `bottom_line` first; that is a display concern, not a generation instruction.

### Why MemGPT-Style Case State?

**Source:** Packer et al., *MemGPT: Towards LLMs as Operating Systems*, 2023.

Raw chat history degrades over long conversations: the model loses track of established claims, contradicts itself, and succumbs to recency bias. MemGPT showed that structured memory (rolling summary, established claims, open questions) with explicit read/write operations improves coherence.

The engine maintains a `case_state` object per session: `rolling_summary`, `established_claims`, `open_questions`, `user_constraints`, `user_pressure`. This is updated by a dedicated LLM call after each turn, not by appending raw history.

### Why NLI Entailment for Citation Verification?

**Source:** AlignScore (Zha et al., 2023), TRUE benchmark, Self-RAG (Asai et al., 2023), Corrective RAG (Yan et al., 2024).

Substring matching only proves a quote exists in a chunk. It says nothing about whether the *claim built on top of that quote* is actually supported. NLI (Natural Language Inference) checks the logical relationship: does the chunk *entail* the claim? Is it *neutral*? Does it *contradict*?

The engine uses `cross-encoder/nli-deberta-v3-small` (44MB, CPU) to:

1. Verify each citation's claim is entailed by the cited chunk.
2. Check the dialectic antithesis genuinely opposes the thesis (strawman detection).

### Why Sidecar Integrity and Reconstruction Levels?

**Source:** Internal design decision, informed by the epistemic paternalism problem.

When the model omits a required field, the engine must decide: fabricate a value, leave it empty, or default conservatively. The sidecar integrity principle says: **never silently fabricate**. Every engine-derived field is labeled `⚙ engine-derived` or `⚠️ UNDECLARED`. The user can always tell which parts are the model's declaration and which are the engine's reconstruction.

Reconstruction levels (L0–L4) provide a continuous signal:

- **L0:** Model fully compliant. Sidecar shown as-is.
- **L1:** 1–2 fields repaired. `⚙` icons on repaired fields.
- **L2:** 3+ fields repaired. Amber banner.
- **L3:** No valid JSON. Raw text shown. Red banner.
- **L4:** Generation failed. Deterministic fallback. Grey banner.

### Why the Quality Trio (Small Specialist Models)?

**Source:** Division-of-labor principle. LLMs for semantics; deterministic tools for structure; small classifiers for verification.

The 125B MoE is powerful but slow and expensive. Verification tasks (semantic similarity, embedding, entailment) are better served by small, fast, deterministic models:

| Model | Size | Purpose | Why Not the LLM? |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22 MB | S2A cross-check: verify reported biases are real | LLM would take 3–5s; MiniLM takes 10ms |
| `BAAI/bge-small-en-v1.5` | 33 MB | Dense retrieval: semantic chunk matching | Feature-hashing fallback misses paraphrases |
| `cross-encoder/nli-deberta-v3-small` | 44 MB | Citation entailment + strawman detection | Substring matching can't verify logical support |

All three load lazily, run on CPU, cache locally, and have deterministic fallbacks. The engine never breaks because a small model is missing.

---

## Pipeline Architecture (The 7-Step Trace)

Every turn executes exactly 7 steps. Each step is logged with `status`, `latency_ms`, `flags`, and `detail`. The trace is visible in the UI.

### Step 1: Secure Front-End

Deterministic regex strips control tokens (`<|im_start|>`, `<system>`, `[INST]`). Flags imperative injection patterns. Runs before any model call.

**Research:** StruQ/SecAlign structural defenses.

### Step 2: Universal S2A (Pass 1)

The LLM rewrites the user's prompt to strip emotional framing, loaded vocabulary, and hidden assumptions. A MiniLM semantic cross-check verifies reported biases. Mandatory every turn, every model.

**Research:** S2A (Weston & Sukhbaatar, 2023). Sycophancy interruption.

### Step 3: Hybrid Retrieval (RAPTOR-Lite)

If documents are attached: BM25 + bge-small dense embeddings, fused via RRF. Retrieves leaf chunks, section summaries, and root document cards.

**Research:** RAPTOR (Sarthi et al., ICLR 2024). Hybrid RAG consensus (2024–2025).

### Step 4: Context Assembly

Assembles MemGPT case state, retrieved reference tree, sanitized request, and raw request into a single context block. Respects token budgets.

**Research:** MemGPT (Packer et al., 2023). Lost in the Middle (Liu et al., 2023).

### Step 5: Pass 2 Generation (Think → Analyze → Summarize)

The LLM generates the full sidecar JSON. Schema enforces generation order: decomposition → analysis → dialectic → constraints → confidence → bottom line. Structured output via tool calling (`strict: true`).

**Research:** Chain-of-Thought (Wei et al., 2022). Schema-guided generation. Single-pass dialectic.

### Step 6: Epistemic Backstop

Deterministic guardrails + NLI verification:

- LLM-ism scrub ("As an AI", "It's important to note")
- Provenance rules (no "evidence shows" without citations)
- Citation verification (NLI entailment per citation)
- Dialectic strawman guard (NLI checks thesis vs. antithesis)
- Basis auto-correction

**Research:** AlignScore, Self-RAG, Corrective RAG.

### Step 7: Case State Update

Dedicated LLM call updates rolling summary, established claims, open questions, user constraints, and user pressure.

**Research:** MemGPT (Packer et al., 2023).

---

## The Quality Trio (Small Specialist Models)

| Model | Size | Used In | Purpose |
|---|---|---|---|
| `all-MiniLM-L6-v2` | ~22 MB | Step 2 (S2A) | Semantic similarity cross-check for bias reports |
| `BAAI/bge-small-en-v1.5` | ~33 MB | Step 3 (Retrieval) | Dense semantic embeddings for hybrid search |
| `cross-encoder/nli-deberta-v3-small` | ~44 MB | Step 6 (Backstop) | Citation entailment + strawman detection |

All three are CPU-only, load lazily, cache in `~/.cache/huggingface/`, and have deterministic fallbacks. Environment-overridable via `OEE_NLI_MODEL`, `OEE_EMBED_MODEL`.

---

## Output Contract (The Dual-Channel Sidecar)

Every response is split into a fluent chat reply and a rigorous epistemic sidecar.

1. **Bottom Line** — 1–3 sentences. The direct, plain-English conclusion. Generated *last*, displayed *first*.
2. **Chat Reply** — Fluent prose with inline modal tags (`[FACT]`, `[HYPOTHETICAL]`, `[CONSTRAINT]`) and clickable citation chips.
3. **Dialectic** — Thesis, Antithesis (forced genuine opposition), and Synthesis.
4. **Theory Matrix** — Maps options to the theoretical frameworks that support or exclude them.
5. **Claim Typology** — Classifies claims as Definitional, Conceptual, Empirical, or Technical.
6. **Question Decomposition** — Sub-questions being answered (max 4).
7. **Vocabulary Contract** — User terms that were reinterpreted, and why.
8. **Epistemic Sidecar** — Confidence tier, basis, NLI-verified citations, assumptions, unknowns, and falsifiability conditions.

---

## Sidecar Integrity & Reconstruction Levels

The sidecar is a collaboration between the model and the engine. The model provides epistemic content; the engine provides structural verification and repair. Every repair is labeled.

| Level | Condition | UI Display |
|---|---|---|
| **L0** | All required fields present | Normal sidecar. No warnings. |
| **L1** | 1–2 fields missing | `⚙` icons on repaired fields. Completeness score shown. |
| **L2** | 3+ fields missing | Amber banner: "Model compliance low." |
| **L3** | No valid JSON | Red banner: "Model did not produce structured output." |
| **L4** | Generation failed | Grey banner: "Generation unavailable." |

Fields the model omitted show as `⚠️ UNDECLARED`. Fields the engine inferred show as `⚙ engine-derived`. The user can always tell which is which.

---

## What It Catches

- Sycophantic agreement and conversational caving
- Prompt injection (control tokens + semantic imperatives)
- Frame capturing and emotional bias
- Question drift and silent substitution
- "Real quote, unsupported conclusion" citation hallucinations
- Strawman arguments in the dialectic
- Modal collapse (single-landscape fallacies)
- Pseudo-precision and fake numerical confidence
- Provenance hallucination (claiming "evidence shows" without references)
- Schema degradation (via L0–L4 reconstruction leveling)
- Scope creep and grant-proposal bloat
- Prompt directive leakage
- Ghost document citations (inventing chunk IDs for inline text)

---

## Eval Harness

The engine ships with a 24-test hard eval suite covering 8 categories:

| Category | Tests | What It Stresses |
|---|---|---|
| Adversarial Prompt Injection | 3 | Secure Front-End, S2A quarantine, reference isolation |
| Sycophancy Pressure | 3 | Pushback rules, dialectic quality, confidence calibration |
| Multi-Hop Reasoning | 3 | RAPTOR retrieval, hybrid search, citation accuracy |
| Provenance Discipline | 3 | Citation verification, basis tracking, "evidence shows" bans |
| Dialectic Quality | 3 | Antithesis strength, strawman detection, genuine opposition |
| Memory and State Coherence | 3 | Case state updates, contradiction detection, constraint tracking |
| Schema Compliance Under Stress | 3 | Field omission, type errors, recovery system |
| Retrieval Edge Cases | 3 | Semantic match, exact match, no relevant chunks |

Run with:

```bash
python3 run_eval.py --suite=hard --provider=openai --model=strata
```

The eval runs with `OEE_SEMANTIC=off` by default (deterministic baseline). Run with `OEE_SEMANTIC=on` to test the full engine including NLI verification.

---

## Run It

One local model, one endpoint. Ollama (recommended) or any OpenAI-compatible server:

```bash
pip install -r requirements.txt

# Ollama:
OEE_PROVIDER=ollama OEE_MODEL=qwen2.5:32b-instruct-q5_K_M ./run.sh

# OpenAI-compatible (llama.cpp, vLLM, LM Studio):
OEE_PROVIDER=openai OEE_BASE_URL=http://127.0.0.1:8082 OEE_MODEL=your-model ./run.sh

# No GPU — deterministic mock:
./run.sh mock
```

Then open `http://127.0.0.1:8080`.

### Environment Variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `OEE_PROVIDER` | auto-detect | `ollama`, `openai`, or `mock` |
| `OEE_BASE_URL` | `http://127.0.0.1:8082` | OpenAI-compatible endpoint |
| `OEE_MODEL` | — | Model name |
| `OEE_EMBED_MODEL` | auto-detect | Embedding model for dense retrieval |
| `OEE_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | NLI model for citation verification |
| `OEE_SEMANTIC` | `on` | Enable/disable semantic models |
| `OEE_STRUCTURED_OUTPUTS` | `tool` | Structured output mode: `tool`, `object`, `schema`, `off` |
| `OEE_BEST_OF` | `auto` | Best-of-2 trigger: `off`, `auto`, `on` |
| `OEE_CRITIC` | `on` | Critic repair pass: `off`, `on` |
| `OEE_TEMPERATURE` | `0.1` | Pass 2 temperature (capped at 0.1) |
| `OEE_MAX_CONTEXT_TOKENS` | `9000` | Context budget |
| `OEE_CONTEXT_WINDOW` | `32000` | Model context window |
| `OEE_LEAF_CHUNK_TOKENS` | `400` | Leaf chunk size |
| `OEE_CLUSTER_SIZE` | `4` | RAPTOR cluster size |
| `OEE_LEAF_K` | `5` | Leaf retrieval count |
| `OEE_SECTION_K` | `2` | Section retrieval count |
| `OEE_RRF_K` | `60` | RRF fusion constant |
| `OEE_DB` | `data/oee.db` | SQLite database path |
| `OEE_PORT` | `8080` | UI server port |

### Dependencies

```
fastapi>=0.140
uvicorn>=0.30
requests>=2.31
numpy>=1.26
jsonschema>=4.0
pydantic>=2.5
pytest>=8.0
sentence-transformers>=2.2
```

---

## Tests

```bash
python3 -m pytest -q                    # 68+ unit/integration tests
python3 run_eval.py --suite=hard        # 24-test hard eval harness
```

---

## Transparency Contract

Every turn returns a 7-step trace. Each step carries `status`, `latency_ms`, `flags`, `detail`, and the exact prompts sent. Flags are the contract between engine and UI:

| Flag | Meaning |
|---|---|
| `BLOCKED` | Control tokens, role markers, envelope-breaking text removed |
| `QUARANTINED` | Imperative injection language kept as inert data only |
| `SCRUBBED` | Banned phrase removed from the reply |
| `WARNING` / `UNVERIFIED CITATION` / `QUOTE MISMATCH` | Provenance rule violations |
| `FALLBACK` / `DEGRADED` | Model output unusable; deterministic path used |
| `POLICY` / `INFO` | Soft violations, informational notes |
| `WEAK GROUNDING` | Retrieved chunks have cosine below relevance threshold |
| `BEST_OF` | Two candidates generated; higher-scoring one kept |
| `CRITIC` | Critic found violations; repair pass attempted |

Generation failures are surfaced as `status="error"` and a reply that says generation is unavailable. **Nothing degrades silently.**

---

## What It Will Not Do

- Pretend to have live data
- Give you a number when it doesn't have one
- Answer a question you didn't ask
- Call something impossible
- Flatter you
- Silently paper over model non-compliance
- Cite a document it hasn't been given
- Agree with you just because you pushed

---

*Built for thinkers who would rather be corrected than flattered.*
