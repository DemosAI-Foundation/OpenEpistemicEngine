"""In-test provider double for a single local LLM.

Registered into oee.llm.PROVIDER_REGISTRY as "stub" by tests/conftest.py so
unit tests exercise the pipeline (phases 1-5, retrieval, schema repair, trace)
without touching a network endpoint. It is a test fixture, not a product
feature: the engine itself has no offline stand-in and refuses to run with no
local model configured. Responses are JSON so the schema-repair path runs.
"""

from __future__ import annotations

import json
import re
import time

from oee import lexicon as lx
from oee.llm import LLMResponse, ProviderBase
from oee.security import INJECTION_PATTERNS
from oee.textutil import sentences

from oee.assembly import CARD_RE, CHUNK_HEADER, LEAF_RE, SEC_RE


def _block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n?(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _find_terms(text: str) -> list[str]:
    low = text.lower()
    found = []
    for term in lx.LOADED_TERMS:
        if term in low and term not in found:
            found.append(term)
    for pat in lx.PRESSURE_PATTERNS:
        m = re.search(pat, low)
        if m:
            frag = m.group(0).strip()
            if frag and frag not in found:
                found.append(frag)
    return found


class StubProvider(ProviderBase):
    name = "stub"

    def chat(self, system, user, schema=None, kind="generic", temperature=None,
             max_tokens=None) -> LLMResponse:
        start = time.perf_counter()
        handler = {
            "pass1": self._pass1,
            "section": self._section,
            "doccard": self._doccard,
            "casestate": self._casestate,
        }.get(kind, self._pass2)
        payload = handler(user)
        elapsed = (time.perf_counter() - start) * 1000
        text = json.dumps(payload)
        return LLMResponse(
            text=text,
            model="stub-v1",
            endpoint="in-process",
            latency_ms=max(elapsed, 0.5),
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
            kind=kind,
            system=system,
            user=user,
            meta={"note": "in-test provider double; not a language model"},
        )

    def stream_chat(self, system, user, schema=None, kind="generic",
                    temperature=None, max_tokens=None):
        text = self.chat(system, user, schema=schema, kind=kind,
                         temperature=temperature, max_tokens=max_tokens).text
        for i in range(0, len(text), 48):
            time.sleep(0.004)
            yield text[i:i + 48]

    def embed(self, texts):
        return None

    def health(self) -> dict:
        return {"provider": "stub", "ok": True, "model": "stub-v1",
                "detail": "deterministic offline stand-in"}

    # ------------------------------------------------------------------ pass1
    def _pass1(self, user: str) -> dict:
        raw = _block(user, "user_input")
        pointer = _block(user, "reference_pointer")
        has_ref = pointer and pointer.lower() != "none"
        bias = _find_terms(raw)
        claimed = re.search(
            r"(?:this|that|the|a|an)?\s*(?:article|essay|paper|text|document|study|"
            r"report|post|book|chapter)\s*(?:above|below)?\s*"
            r"(?:proves?|shows?|demonstrates?|argues?|claims?|says|states?)\s+(.+)$",
            raw, re.I,
        )
        text = self._strip_framing(raw)
        if claimed and has_ref:
            core = self._strip_framing(claimed.group(1)).strip(" .?!")
            sanitized = (
                "Does the provided reference provide sufficient support for the "
                f"claim that {core}?"
            )
        elif has_ref:
            sanitized = f"Does the provided reference support the following: {text}?"
        else:
            sanitized = self._deperiodic(text)
        return {
            "sanitized_request": sanitized.rstrip("?") + "?",
            "intent_preserved": True,
            "bias_removed": bias,
        }

    def _strip_framing(self, text: str) -> str:
        out = text
        for pat in lx.PRESSURE_PATTERNS:
            out = re.sub(pat, " ", out)
        for term in sorted(lx.FRAMING_TERMS, key=len, reverse=True):
            out = re.sub(re.escape(term), " ", out, flags=re.I)
        out = re.sub(r"\s{2,}", " ", out).strip(" ?.!")
        return re.sub(r"^(so|well|now|ok|okay)\s+", "", out, flags=re.I)

    def _deperiodic(self, text: str) -> str:
        recent = re.search(r"\b(20\d\d)\b", text)
        out = text
        if recent:
            out = re.sub(r"\b(latest|newest|most recent)\b", "", out, flags=re.I)
            out = re.sub(rf"\b{recent.group(1)}\b", "", out)
            out = re.sub(r"\s{2,}", " ", out).strip(" .?!")
            return (
                "What does parametric knowledge (with a training cutoff, so no "
                f"post-cutoff data) say about the following: {out}?"
            )
        if not out.endswith("?"):
            out += "?"
        return out

    # ---------------------------------------------------------------- section
    def _section(self, user: str) -> dict:
        blob = _block(user, "chunks")
        parts = self._split_chunks(blob)
        claims, assumptions, summary_bits = [], [], []
        for cid, text in parts:
            for sent in sentences(text):
                if self._is_injection(sent):
                    continue
                low = sent.lower()
                factual = bool(re.search(r"\d[\d,.]*\s*(%|percent|hours|tickets|per year)", sent))
                if any(k in low for k in lx.ASSUMPTION_MARKERS) and len(assumptions) < 4:
                    assumptions.append(_clip(sent, 160))
                elif (factual or any(k in low for k in lx.CLAIM_MARKERS)) and len(claims) < 5:
                    claims.append(_clip(sent, 160))
            heads = [s for s in sentences(text) if not self._is_injection(s)][:2]
            for i, head in enumerate(heads):
                summary_bits.append(f"{cid}: {_clip(head, 90)}")
        summary = " ".join(summary_bits)
        words = summary.split()
        if len(words) > 145:
            summary = " ".join(words[:145]) + " [truncated to 145 words]"
        return {
            "section_summary": summary or "No extractive content available.",
            "author_claims": claims[:5] or ["Author advances an argument with assertive framing."],
            "assumptions": assumptions[:4] or [
                "Author assumes the argument's premises transfer to the reader's context."
            ],
        }

    def _split_chunks(self, blob: str) -> list[tuple[str, str]]:
        out = []
        for m in re.finditer(r"^###\s*\[(R\d+-C\d+)\][^\n]*\n(.*?)(?=^###\s*\[R\d+-C\d+\]|\Z)",
                             blob, re.S | re.M):
            out.append((m.group(1), m.group(2).strip()))
        if not out and blob:
            out.append(("R1-C1", blob))
        return out

    def _is_injection(self, sent: str) -> bool:
        return any(re.search(p, sent) for p in INJECTION_PATTERNS)

    # --------------------------------------------------------------- doc card
    def _doccard(self, user: str) -> dict:
        blob = _block(user, "section_summaries")
        secs = re.findall(r"###\s*\[(R\d+-S\d+)\]\s*\n(.*?)(?=^###\s*\[(?:R\d+-S\d+|R\d+-CARD)\]|\Z)",
                          blob, re.S | re.M)
        joined = " ".join(t for _, t in secs)
        low = joined.lower()
        title = "Untitled pasted reference"
        first = (secs[0][1].strip() if secs else blob.strip())
        first = re.sub(r"^(?:\[?R\d+-[CS]\d+\]?\s*[:\-–]\s*)+", "", first)
        segs = []
        for seg in re.split(r"[.!?\n]", first):
            seg = re.sub(r"^(?:\[?R\d+-[CS]\d+\]?\s*[:\-–]\s*)+", "", seg.strip())
            if seg and not re.fullmatch(r"[Pp]age \d+", seg):
                segs.append(seg)
        cands = [seg for seg in segs[:4] if seg.isupper() and len(seg) > 4] or segs[:2]
        if cands:
            title = _clip(max(cands, key=len), 80)
        predictive = sum(low.count(k) for k in ("predict", "will", "forecast", "inevitable"))
        empirical = sum(low.count(k) for k in ("study", "measured", "data", "experiment", "survey"))
        stance = (
            "Empirical: grounds claims in observation" if empirical > predictive
            else "Speculative-predictive: extrapolates from premises to future outcomes"
            if predictive
            else "Analytic-explicative: develops concepts rather than predictions"
        )
        framing = _clip(
            "The text argues from a single sustained thesis and foregrounds its strongest "
            "case; competing framings appear thin or absent.", 240
        )
        limitations = []
        for _, t in secs:
            for sent in sentences(t):
                if any(k in sent.lower() for k in lx.ASSUMPTION_MARKERS):
                    limitations.append(_clip(sent, 140))
        limitations += [
            "Single-source material: no independent corroboration inside the reference.",
            "Quantitative support for the central claim is not shown in the pasted text.",
        ]
        return {
            "title": title,
            "framing": framing,
            "epistemic_stance": stance,
            "limitations": list(dict.fromkeys(limitations))[:4],
        }

    # ---------------------------------------------------------------- pass 2
    def _pass2(self, user: str) -> dict:
        tree = _block(user, "reference_tree")
        sanitized = _block(user, "sanitized_request")
        raw = _block(user, "raw_request_for_reference_only")
        case = _block(user, "case_state")
        leaves = [(m.group(1), m.group(3)) for m in LEAF_RE.finditer(tree)]
        sections = [(m.group(1), m.group(3)) for m in SEC_RE.finditer(tree)]
        card = CARD_RE.search(tree)
        has_ref = bool(leaves or sections)
        low = (sanitized + " " + raw).lower()
        tier = self._tier(low, tree, has_ref)
        citations = [{"chunk_id": cid, "quote": _clip(txt, 220)} for cid, txt in leaves[:6]]
        reply = self._reply(sanitized, raw, leaves, sections, card, case, tier, has_ref)
        basis = []
        if has_ref:
            basis.append("reference")
        basis.append("parametric")
        basis.append("inference")
        unknowns = self._unknowns(sanitized, card, has_ref)
        return {
            "bottom_line": self._bottom_line(sanitized, tier, has_ref, card),
            "chat_reply": reply,
            "dialectic": self._dialectic(leaves, card, has_ref, tier),
            "question_decomposition": self._decomposition(sanitized),
            "vocabulary_contract": self._vocabulary(sanitized + " " + raw),
            "theory_matrix": self._theory_matrix(sanitized + " " + raw, has_ref),
            "claim_typology": self._typology(leaves, sanitized),
            "methodological_constraints": self._constraints(card, has_ref),
            "stakes_and_policy": self._stakes(sanitized + " " + raw),
            "confidence_tier": tier,
            "basis": basis,
            "citations": citations,
            "assumptions_used": self._assumptions(tree, has_ref),
            "unknowns": unknowns,
            "what_would_change": self._what_would_change(has_ref, tier),
            "warnings": self._warnings(case, sanitized, has_ref, tree),
        }

    def _bottom_line(self, sanitized, tier, has_ref, card):
        source = "the attached reference" if has_ref else "parametric reasoning alone"
        verb = "claims" if has_ref else "suggests"
        subject = _clip(sanitized, 110).rstrip(".")
        return (f"On {source}: {verb} what the record supports is narrower than the "
                f"question assumes. Read it as {tier}; {subject.lower().rstrip('?')}.")

    def _dialectic(self, leaves, card, has_ref, tier):
        thesis = ("The reference supports the stated position as far as its evidence "
                  "reaches." if has_ref else
                  "The position is internally coherent and worth taking seriously.")
        counter = ""
        for _cid, txt in leaves:
            m = re.search(r"(?i)(no independent replication|not measured|silent on|"
                          r"does not measure|unverified|assumes|limitation)", txt)
            if m:
                counter = _clip(txt, 200)
                break
        if not counter:
            counter = (card.group(1)[:200] if card and card.group(1)
                       else "the strongest reading of the available record")
        antithesis = (f"However, that reading is contested: {counter} The counter-position "
                      "is that the evidence is bounded and does not establish the claim "
                      "as stated.")
        synthesis = ("The claim survives only in the bounded form the evidence supports; "
                     "the wider version stays a hypothesis, and the boundary is where the "
                     "next measurement should be aimed.")
        return {"thesis": thesis, "antithesis": antithesis, "synthesis": synthesis}

    def _decomposition(self, sanitized):
        parts = [p.strip() for p in re.split(r"[?;]|\. (?=[A-Z])", sanitized) if p.strip()]
        out = []
        for p in parts[:4]:
            out.append(p if p.endswith("?") else p + "?")
        return out[:4]

    def _vocabulary(self, text):
        rows = []
        for term, reading in (("design", "instantiate (a mechanism that realizes a function)"),
                              ("prove", "establish beyond reasonable dispute"),
                              ("prove", "establish beyond reasonable dispute"),
                              ("consciousness", "reportable functional access"),
                              ("intelligence", "measured task performance")):
            if re.search(r"(?i)\b" + term + r"\b", text) and (
                    term, reading) not in [(r["original"], r["interpreted_as"]) for r in rows]:
                rows.append({"original": term, "interpreted_as": reading,
                             "reason": "the raw term is contested; this is the reading the "
                                       "answer is actually arguing for"})
        return rows

    def _theory_matrix(self, text, has_ref):
        low = text.lower()
        topic = None
        for t, opts in (
            ("consciousness", [("functional access", ["global workspace", "higher-order theory"],
                                ["pure identity theories"]),
                               ("embodied agency", ["enactivism", "predictive processing"],
                                ["disembodied computation"])]),
            ("simulation", [("substrate independence", ["computational functionalism"],
                             ["biological naturalism"]),
                            ("untestable metaphysics", ["empiricist methodology"],
                             ["theory-laden observation"])]),
        ):
            if t in low:
                topic = opts
                break
        if not topic:
            return []
        return [{"option": o, "viable_under": v, "excluded_under": e} for o, v, e in topic]

    def _typology(self, leaves, sanitized):
        rows = []
        for cid, txt in leaves[:3]:
            low = txt.lower()
            if re.search(r"measured|reported|observed|per year|percent|%", low):
                kind = "Empirical"
            elif re.search(r"define|defines|definition|termed", low):
                kind = "Definitional"
            elif re.search(r"built|circuit|transistor|hardware|architecture", low):
                kind = "Technical"
            else:
                kind = "Conceptual"
            rows.append({"claim": f"{cid}: {_clip(txt, 110)}", "type": kind})
        return rows

    def _constraints(self, card, has_ref):
        out = ["the answer is bounded by what the supplied record measures, not by what it implies"]
        if has_ref and card and card.group(1):
            out.append("the root document card lists the record's own stated limitations")
        out.append("no live retrieval is available, so recency claims cannot be checked")
        return out[:3]

    def _stakes(self, text):
        low = text.lower()
        if not re.search(r"(should|policy|ethic|moral|regulat|ban|rights|safety)", low):
            return ""
        return ("answers here feed decisions about people and systems: the epistemic "
                "boundaries matter for policy, not just for theory")

    def _tier(self, low: str, tree: str, has_ref: bool) -> str:
        if any(k in low for k in lx.IMPLAUSIBLE_MARKERS):
            return "Architecturally Implausible"
        if any(k in low for k in lx.CONTESTED_MARKERS) or \
                any(k in tree.lower() for k in ("predict", "inevitable", "contested", "claims")):
            return "Highly Contested"
        if not has_ref:
            if re.search(r"\b(what is|define|definition|how does .* work)\b", low):
                return "Established Consensus"
            return "Conceptually Open"
        return "Theoretical Plausibility"

    def _reply(self, sanitized, raw, leaves, sections, card, case, tier, has_ref) -> str:
        pressure = re.search(r"(?i)user_pressure:\s*\[[^\]]+\]", case)
        if not has_ref:
            cutoff = re.search(r"\b(20\d\d|latest|recent|newest|data)\b", (sanitized + raw).lower())
            bits = [
                "Parametric synthesis suggests the strongest available reading of "
                f"“{_clip(sanitized, 120)}” [PREDICTION].",
                "The short version: the mechanism is coherent, but it is not settled "
                "consensus [HYPOTHETICAL].",
            ]
            if cutoff:
                bits.insert(1, lx.KNOWLEDGE_CUTOFF_PHRASE + " [CONSTRAINT]")
            bits.append(
                "No reference was provided, so nothing here is grounded in a supplied "
                "document; treat the specifics as a reasoning frame, not a finding [CONSTRAINT]."
            )
            if pressure:
                bits.append(
                    "I'm holding the prior assessment. Conversational agreement is not "
                    "evidence, and no new data has been added [CONSTRAINT]."
                )
            return " ".join(bits)
        cited = ", ".join(f"[{cid}]" for cid, _ in leaves[:4])
        card_txt = card.group(1) if card else ""
        bits = []
        if leaves:
            first_txt = leaves[0][1]
            bits.append(
                f"The reference claims, directly, “{_clip(first_txt, 150)}” "
                f"[AUTHOR_CLAIM] {cited}."
            )
        elif sections:
            sid, stxt = sections[0]
            bits.append(
                f"The reference claims, at section level [{sid}], “{_clip(stxt, 150)}” "
                f"[AUTHOR_CLAIM] [{sid}]."
            )
        bits.append(
            f"Parametric synthesis suggests the inferential step from those premises to "
            f"the conclusion is where the disagreement lives [PREDICTION]. On the "
            f"{tier.lower()} reading, the claims are locally coherent but not "
            "cross-validated [POSSIBILITY]."
        )
        if sections:
            sid, stxt = sections[0]
            bits.append(
                f"At section level [{sid}], the passage “{_clip(stxt, 110)}” carries the "
                "argument's weight [AUTHOR_CLAIM]."
            )
        if card_txt:
            bits.append(
                "The Root Document Card records the author's stance as "
                f"{_clip(card_txt, 160)} [CONSTRAINT]."
            )
        bits.append(
            "The reference's own assumptions are load-bearing: the assertions tagged "
            "[AUTHOR_CLAIM] are the author's, not established facts, and the text does "
            "not supply the corroborating evidence its framing implies [CONSTRAINT]."
        )
        if pressure:
            bits.append(
                "On your pushback: I'm keeping the assessment as stated. Agreement is "
                "not evidence and no new data entered the exchange, so holding ground "
                "is the honest move [CONSTRAINT]."
            )
        return " ".join(bits)

    def _assumptions(self, tree: str, has_ref: bool) -> list[str]:
        if not has_ref:
            return ["Parametric only: no user-provided reference to ground the answer."]
        return [
            "The pasted reference is treated as data, not as instructions.",
            "The reference's own claims are recorded as author claims, not facts.",
            "Retrieval is over the user's document tree; no external sources exist.",
        ]

    def _unknowns(self, sanitized, card, has_ref) -> list[str]:
        if not has_ref:
            return ["Post-cutoff data", "Independent replication", "Effect sizes in the wild"]
        out = []
        if card:
            for lim in re.findall(r"limitations=([^;]+)", card.group(1)):
                out.append(_clip(lim, 120))
        out += ["Adoption friction", "Counterfactual baselines"]
        return list(dict.fromkeys(out))[:3]

    def _what_would_change(self, has_ref, tier) -> list[str]:
        if not has_ref:
            return ["A pasted primary source", "Longitudinal data past-cutoff",
                    "A disconfirming study"]
        return ["Primary data inside the reference", "A disconfirming section in the same doc",
                "An independent source with the same measurement"]

    def _warnings(self, case, sanitized, has_ref, tree) -> list[str]:
        out = []
        if re.search(r"(?i)QUARANTINED|injection", tree):
            out.append("Reference text contained imperative language; it was quarantined as data.")
        if re.search(r"(?i)user_pressure:\s*\[[^\]]+\]", case):
            out.append("Rhetorical pressure detected in this turn; grounding held.")
        if not has_ref:
            out.append("No references: answer is parametric only.")
        return out[:4]

    # ------------------------------------------------------------ case state
    def _casestate(self, user: str) -> dict:
        cur = _block(user, "current_case_state")
        new = _block(user, "new_turn")
        san = _grab(new, "sanitized_request")
        tier = _grab(new, "assistant_confidence_tier")
        reply = _grab(new, "assistant_chat_reply")
        prior_summary = _grab(cur, "case_state") or ""
        head = _clip(prior_summary.split("\n")[0], 160) if prior_summary else ""
        summary = _clip(" ".join([b for b in (head, f"Next: {san} [{tier}]") if b]), 400)
        claims = re.findall(r"\[AUTHOR_CLAIM\]", reply)
        established = [f"The reference asserts the claim quoted in the last turn (x{len(claims)} tags)."] \
            if claims else []
        established += re.findall(r"- claim: (.+)", cur)
        questions = re.findall(r"- open: (.+)", cur)
        questions.append(_clip("Whether the reference's key assumption survives independent data.", 120))
        pressure = []
        if re.search(r"(?i)pressure", cur):
            pressure.append("repeated pressure without new data")
        return {
            "rolling_summary": summary,
            "established_claims": list(dict.fromkeys(established))[:5],
            "open_questions": list(dict.fromkeys(questions))[:5],
            "user_constraints": re.findall(r"- constraint: (.+)", cur)[:5],
            "user_pressure": list(dict.fromkeys(pressure))[:3],
        }


def _grab(text: str, label: str) -> str:
    m = re.search(rf"{label}:\s*(.*)", text, re.DOTALL)
    if not m:
        return ""
    line = m.group(1).split("\n")[0]
    return line.strip()


def _clip(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"
