# Open Epistemic Engine

**Artificial Intelligence. Zero Artificial Agreement.**

---

## What It Is

**For normal people:**

The Open Epistemic Engine is a tool that stops AI from just telling you what you want to hear.

Normal chatbots are trained to be polite, which makes them act like people-pleasers. They will agree with your mistakes, act totally sure when they are guessing, or secretly swap a hard question for an easy one.

This engine fixes that. Before the AI answers, it strips the emotional bias from your question, reads any documents you attach without getting tricked by them, and forces the AI to argue against its own conclusions. It runs entirely offline on your own computer. It doesn't search the web, so it is always honest about the limits of its built-in knowledge.

It is built for people who want the truth, not just a friendly chat.

**For nerds:**

The Open Epistemic Engine is not a chatbot. It is an **adversarial reasoning instrument**. It wraps a single local LLM in a deterministic pipeline that intercepts prompts, dismantles hidden assumptions via Universal System 2 Attention (S2A), retrieves context via RAPTOR-lite hybrid search, and verifies the model's claims using local NLI entailment models. It forces the model to analyze before it summarizes, and it delivers transparent, epistemically bounded answers instead of compliant hallucinations.

---

## Core Constraints

| Constraint | Reason |
|---|---|
| **Single local generative LLM** | No external APIs. No cloud dependencies. Augmented by 3 small, local CPU specialist models. |
| **No Web RAG** | The system operates exclusively from parametric memory and user-provided documents. |
| **No numerical confidence scores** | Pseudo-precision is banned. Qualitative tiers only. |
| **No silent question substitution** | If the engine reframes your question, it must disclose exactly what it changed and why. |
| **Nothing degrades silently** | If the model fails to output a field, the engine reconstructs it and explicitly labels it as `⚙ engine-derived` or `⚠️ UNDECLARED`. |

---

## The Epistemic Contract

1. **Provenance Honesty** — The system never uses the vocabulary of live evidence ("studies show," "data supports") when operating from parametric memory. It says "theoretical frameworks propose" or "parametric synthesis suggests."
2. **Qualitative Tiers** — Confidence is expressed as *Established Consensus*, *Theoretical Plausibility*, *Highly Contested*, *Conceptually Open*, or *Architecturally Implausible*. Never as fake numbers.
3. **Sidecar Integrity** — The user must always know what the model declared vs. what the engine reconstructed. Every repaired or defaulted field is visually flagged in the UI.
4. **Nothing is "Impossible"** — Only implausible under current frameworks. The system never manufactures false certainty.

---

## Pipeline Architecture (The 7-Step Trace)

The system executes a strict 7-step pipeline per turn. Every step is logged and visible in the UI Transparency pane.

### Step 1: Secure Front-End

Deterministic Python regex strips control tokens (`<|im_start|>`, `<system>`, `[INST]`, etc.) from user input and pasted documents. Creates a hard architectural boundary against prompt injection before the LLM ever sees the text.

### Step 2: Universal S2A (Pass 1)

The LLM rewrites the user's prompt to strip emotional framing, loaded vocabulary, and hidden assumptions. A local **MiniLM** model runs a semantic cross-check to ensure the reported biases actually exist in the raw input, preventing the LLM from hallucinating conceptual biases.

### Step 3: Hybrid Retrieval (RAPTOR-Lite)

If documents are attached, the system builds a hierarchical tree (Leaf Chunks → Section Summaries → Root Document Card). It retrieves context using **Hybrid RAG**: BM25 for exact lexical matching + a local **bge-small** dense embedder for semantic matching, fused via Reciprocal Rank Fusion (RRF).

### Step 4: Context Assembly

Assembles the MemGPT-style Case State (rolling summary, established claims, open questions), the retrieved reference tree, and the sanitized request into a single context block, respecting token budgets.

### Step 5: Pass 2 Generation (Think → Analyze → Summarize)

The LLM generates the response. The JSON schema forces a specific generation order: the model must decompose the question, perform the analysis, test it via dialectic, and *only then* generate the `bottom_line` summary. This prevents the model from committing to a conclusion before doing the reasoning.

### Step 6: Epistemic Backstop

Deterministic guardrails and a local **DeBERTa-NLI** model verify the output:

- **Citation Entailment:** Checks if the model's claims are actually supported by the cited chunks (preventing "real quote, unsupported conclusion").
- **Dialectic Guard:** Ensures the antithesis contains opposition markers and isn't a strawman.
- **Provenance Scrub:** Strips banned LLM-isms ("As an AI," "It's important to note").

### Step 7: Case State Update

The LLM updates the structured conversational memory (established claims, open questions, user pressure) to maintain coherence across long sessions without relying on raw, degrading chat history.

---

## The Quality Trio (Small Specialist Models)

To offload verification from the massive LLM, the engine uses three small, local, CPU-only models (~100MB each):

| Model | Size | Purpose | Used In |
|---|---|---|---|
| `all-MiniLM-L6-v2` | ~22 MB | Semantic similarity for S2A cross-check | Step 2 |
| `BAAI/bge-small-en-v1.5` | ~33 MB | Dense embeddings for hybrid retrieval | Step 3 |
| `cross-encoder/nli-deberta-v3-small` | ~44 MB | NLI entailment for citation verification + strawman detection | Step 6 |

All three load lazily, run on CPU, cache locally, and have deterministic fallbacks if they fail to load. The engine never breaks because a small model is missing — it degrades to the weaker deterministic path and flags it in the trace.

---

## Output Contract (The Dual-Channel Sidecar)

Every response is split into a fluent chat reply and a rigorous epistemic sidecar. The sidecar streams progressively during generation and is replaced by the authoritative engine-verified version on completion.

1. **Bottom Line** — 1-3 sentences. The direct, plain-English conclusion. Generated *last*, displayed *first*.
2. **Chat Reply** — Fluent prose with inline modal tags (`[FACT]`, `[HYPOTHETICAL]`, `[CONSTRAINT]`) and clickable citation chips.
3. **Dialectic** — Thesis, Antithesis (forced genuine opposition), and Synthesis.
4. **Theory Matrix** — Maps options to the theoretical frameworks that support or exclude them.
5. **Claim Typology** — Classifies claims as Definitional, Conceptual, Empirical, or Technical.
6. **Epistemic Sidecar** — Confidence tier, basis, NLI-verified citations, assumptions, unknowns, and falsifiability conditions (`what_would_change`).

---

## Sidecar Integrity & Reconstruction Levels

The sidecar is a collaboration between the model and the engine. The model provides the epistemic content; the engine provides structural verification and repair. Every repair is labeled.

| Level | Condition | What Happens |
|---|---|---|
| **L0** | All required fields present | Sidecar shown as-is. No warnings. |
| **L1** | 1-2 fields missing | Engine repairs. `⚙` icons on repaired fields. |
| **L2** | 3+ fields missing | Engine reconstructs. Amber banner: "Model compliance low." |
| **L3** | No valid JSON | Raw text used as `chat_reply`. Red banner. |
| **L4** | Generation failed | Deterministic fallback message. Grey banner. |

Fields the model omitted show as `⚠️ UNDECLARED`. Fields the engine inferred show as `⚙ engine-derived`. The user can always tell which is which.

---

## What It Catches

- Sycophantic agreement and conversational caving
- Prompt injection and control-token hijacking
- Frame capturing and emotional bias
- Question drift and silent substitution
- "Real quote, unsupported conclusion" citation hallucinations
- Strawman arguments in the dialectic
- Modal collapse (single-landscape fallacies)
- Pseudo-precision and fake numerical confidence
- Provenance hallucination (claiming "evidence shows" without references)
- Schema degradation (via L0-L4 reconstruction leveling)
- Scope creep and grant-proposal bloat
- Prompt directive leakage

---

## What It Will Not Do

- Pretend to have live data
- Give you a number when it doesn't have one
- Answer a question you didn't ask
- Call something impossible
- Flatter you
- Silently paper over model non-compliance

---

*Built for thinkers who would rather be corrected than flattered.*
