# Open Epistemic Engine

****

## What It Is

**For normal people:**
The Open Epistemic Engine is a tool that stops AI from just telling you what you want to hear. 
Normal chatbots are trained to be polite, which makes them act like people-pleasers. They will agree with your mistakes, act totally sure when they are guessing, or secretly swap a hard question for an easy one. 
This engine fixes that. Before the AI answers, it breaks your question into pieces, checks your hidden assumptions, and forces the AI to use strict logic. 
It runs entirely offline on your own computer. It doesn't search the web for live updates, so it is always honest about the limits of its built-in knowledge. 
It is built for people who want the truth, not just a friendly chat.

**For nerds:**
The Open Epistemic Engine is not a chatbot. It is not an evidence synthesizer. It is an **adversarial reasoning instrument** that intercepts prompts, dismantles hidden assumptions, and forces a single local LLM to deliver transparent, epistemically bounded answers instead of compliant hallucinations.

---

## Core Constraints

| Constraint | Reason |
|---|---|
| **Single local LLM only** (port 8082) | No external APIs. No cloud dependencies. |
| **No retrieval / No RAG** | The system operates exclusively from parametric memory and is architected to be honest about that boundary. |
| **No numerical confidence scores** | Pseudo-precision is banned. Qualitative tiers only. |
| **No silent question substitution** | If the engine reframes your question, it must disclose why. |

---

## The Epistemic Contract

1. **Provenance Honesty** — The system never uses the vocabulary of live evidence ("studies show," "data supports") when operating from parametric memory. It says "theoretical frameworks propose" or "parametric synthesis suggests."
2. **Qualitative Tiers** — Confidence is expressed as *Established Consensus*, *Theoretical Plausibility*, *Highly Contested*, *Conceptually Open*, or *Architecturally Implausible (Current Frameworks)*. Never as fake numbers.
3. **Temporal Honesty** — Every divergent output states its knowledge cutoff and flags active frontiers it cannot see past.
4. **Nothing is "Impossible"** — Only implausible under current frameworks. The system never manufactures false certainty.

---

## Pipeline Architecture

```
Phase 0  Triage, Scoping & Stakes
Phase 1  Skeptic & Question Decomposition
Phase 2  Parametric Ontology Mapper
Phase 3  Dialectical Engine & Modal Logic (RaLU)
Phase 4  Deterministic Editor (Python Guardrails)
Phase 5  Writer & Mechanical Scrubber
Phase 6  Presentation Critic
```

### Phase 0: Triage
Routes prompts as Convergent or Divergent. Classifies intent scope. Flags moral/ontological stakes.

### Phase 1: Skeptic
Decomposes ambiguous questions. Tags claim types (definitional, conceptual, empirical, technical). Enforces vocabulary contracts.

### Phase 2: Ontology Mapper
Extracts the conceptual territory from parametric memory. Flags mandatory landmark concepts (e.g., the Hard Problem) for consideration — engagement or explicit justification of dismissal required.

### Phase 3: Dialectical Engine
Forces Thesis → Antithesis → Synthesis. Tags every claim with modal status: `Actual Fact`, `Theory Prediction`, `Hypothetical Schema`, `Conceptual Possibility`, `Architectural Constraint`.

### Phase 4: Deterministic Editor
Applies non-LLM guardrails:
- **Theory-Indexed Option Matrix** — options are conditional, not global
- **Substrate-Test Compatibility Filter** — blocks architecturally invalid tests
- **Universal Objection Filter** — rejects asymmetric critique
- **Epistemic Labeling Enforcement** — bans evidence vocabulary when no retrieval exists
- **Scope Enforcement** — prevents grant-proposal bloat on feasibility questions

### Phase 5: Writer & Scrubber
Generates the final response in plain English. Post-processing scrubs leaked prompt artifacts and deduplicates content semantically.

### Phase 6: Presentation Critic
20-point checklist covering question fidelity, modal logic, symmetry, epistemic honesty, scope, and stakes. Rejects drafts that fail. Maximum two rewrite cycles.

---

## Output Contract

Every divergent response contains:

1. **Bottom Line** — Approximately two sentences. Up to four if genuinely conditional. Direct answer first.
2. **Epistemic & Temporal Status** — Mode, cutoff, vocabulary contract.
3. **Question Decomposition** — Sub-questions addressed. Dropped ones justified.
4. **Core Analysis** — Theory-indexed. Modal-tagged. Plain English.
5. **Theory-Indexed Feasibility Matrix** — Options conditional on theories.
6. **Methodological Constraints** — Architecturally implausible approaches flagged.
7. **Key Unknowns** — Maximum five. Prioritized.
8. **What Would Change This Answer** — Mandatory. Lists specific falsification conditions.
9. **Stakes & Policy** — Mandatory when moral/ontological stakes are flagged.

---

## What It Catches

- Sycophantic agreement
- Frame capturing
- Question drift and silent substitution
- Causal attribution compliance
- Modal collapse
- Vocabulary compliance
- Termination compliance
- Asymmetric critique
- Pseudo-precision
- Provenance hallucination
- Scope creep
- Prompt directive leakage

---
