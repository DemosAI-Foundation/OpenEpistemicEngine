"""Deterministic rule engine: the epistemic contract, preflight triage, continuation analysis, and the Editor rule set.

No LLM is used in this module. It only inspects packet fields and text. The system runs exclusively from the
frozen parametric weights of one local model: there is no retrieval, no live search, no external data layer.
Every user-facing claim is labelled parametric synthesis, never live empirical synthesis.
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

STOPWORDS = set(
    """a an the and or but if then than that this these those of to in on for with without at by from as is are was
    were be been being do does did done i you he she it we they me him her us them my your our their so not no yes
    just very really about into over under more most less least some any all can could should would will shall may
    might must have has had here there what which who whom whose when where why how please give me tell write make
    me you me i'm im i'd i'll dont don't wont cant""".split()
)

# ---------------------------------------------------------------------------
# Epistemic contract (Part 1): provenance honesty, qualitative tiers, staleness
# ---------------------------------------------------------------------------

# Rule 1: the system never uses the vocabulary of live empirical synthesis.
BANNED_PHRASES = [
    "evidence shows",
    "evidence suggests",
    "the evidence shows",
    "data supports",
    "data show",
    "the data shows",
    "studies confirm",
    "studies show",
    "the map indicates",
    "the map shows",
    "recent findings",
    "recent research",
    "experiments confirm",
    "trials confirmed",
    "measurements confirm",
]

BANNED_SINGLE_WORDS = ["evidence", "data", "the map"]

PARAMETRIC_REPLACEMENTS = {
    "evidence shows": "parametric synthesis suggests",
    "evidence suggests": "parametric synthesis suggests",
    "the evidence shows": "parametric synthesis suggests",
    "data supports": "theoretical frameworks propose",
    "data show": "theoretical frameworks propose",
    "the data shows": "theoretical frameworks propose",
    "studies confirm": "within my training data, the consensus was",
    "studies show": "within my training data, the consensus was",
    "the map indicates": "this framework predicts",
    "the map shows": "this framework predicts",
    "recent findings": "the state of the field at my training cutoff",
    "recent research": "the state of the field at my training cutoff",
    "experiments confirm": "the framework predicts",
}

MANDATORY_PARAMETRIC_VOCABULARY = [
    "parametric synthesis suggests",
    "theoretical frameworks propose",
    "conceptual analysis indicates",
    "within my training data, the consensus was",
    "this framework predicts",
]

# Rule 2: qualitative epistemic tiers. Numerical confidence scores are banned.
TIERS = [
    "Established Consensus",
    "Theoretical Plausibility",
    "Highly Contested",
    "Conceptually Open",
    "Architecturally Implausible (Current Frameworks)",
]

TIER_ALIASES = {
    "established": "Established Consensus",
    "consensus": "Established Consensus",
    "plausibility": "Theoretical Plausibility",
    "theoretical": "Theoretical Plausibility",
    "plausible": "Theoretical Plausibility",
    "contested": "Highly Contested",
    "open": "Conceptually Open",
    "implausible": "Architecturally Implausible (Current Frameworks)",
    "architectural": "Architecturally Implausible (Current Frameworks)",
}

IMPLAUSIBLE_TIER = "Architecturally Implausible (Current Frameworks)"

# Rule 3: parametric staleness honesty.
DEFAULT_KNOWLEDGE_CUTOFF = "the model's frozen training cutoff (the serving model's stated date applies)"


def knowledge_cutoff(settings: dict | None = None) -> str:
    settings = settings or {}
    return (
        str(settings.get("knowledge_cutoff") or "").strip()
        or os.environ.get("EPISTEMIC_CUTOFF", "").strip()
        or DEFAULT_KNOWLEDGE_CUTOFF
    )


def temporal_disclaimer(cutoff: str = "") -> str:
    cutoff = cutoff or knowledge_cutoff()
    return (
        f"Based on parametric knowledge up to {cutoff}. This is a conceptual synthesis of theoretical positions, "
        "not a live evidence review."
    )


FRONTIER_FRAMING = (
    "Active research frontiers must be framed as ongoing debates in current literature at the cutoff, "
    "never as settled results: what happened after the cutoff is unknown to this system."
)

# Modal logic tagging (Philosophical RaLU): every major claim carries a modal status.
MODAL_TAGS = [
    "Actual Fact",
    "Theory Prediction",
    "Hypothetical Schema",
    "Conceptual Possibility",
    "Architectural Constraint",
]

MODAL_TAG_ALIASES = {alias.lower(): tag for tag in MODAL_TAGS for alias in (tag, tag.lower())}

ANCHOR_MODAL_STATUSES = ["actual state", "hypothetical schema", "theory-internal consequence", "necessary-condition schema"]

# Phase 0: intent scoping and stakes flagging.
INTENT_SCOPES = ["High-Level Feasibility", "Applied Methodology", "Conceptual Clarification"]
HIGH_LEVEL_SCOPE = "High-Level Feasibility"
STAKES_HIGH = "High Moral / Ontological Stakes"
STAKES_STANDARD = "Standard"

STAKES_TRIGGERS = [
    "consciousness",
    "conscious",
    "sentience",
    "sentient",
    "qualia",
    "phenomenal",
    "moral patient",
    "moral status",
    "welfare",
    "suffering",
    "personhood",
    "agi",
    "superintelligence",
    "existential risk",
    "x-risk",
    "irreversible",
    "permanent",
    "locked in",
]

SCOPE_TRIGGERS = {
    "Applied Methodology": ["protocol", "method", "methodology", "implementation", "experiment design", "benchmark", "procedure", "steps to build", "how do we build", "roadmap", "spec"],
    "Conceptual Clarification": ["what is", "define", "definition", "meaning", "clarify", "distinguish", "concept", "term", "meaning of", "what do you mean", "what would count as"],
}

# Phase 2: mandatory landmark consideration per deep domain. Consideration is mandatory; inclusion is not.
LANDMARKS_BY_DOMAIN = {
    "Consciousness / Hard Problem": [
        "The Hard Problem / Explanatory Gap",
        "Access vs. Phenomenal Consciousness",
        "Illusionism / Eliminativism",
        "The Measurement Problem (as a universal constraint)",
    ],
    "Mind-Body / Substrate Dependence": [
        "The Hard Problem / Explanatory Gap",
        "Access vs. Phenomenal Consciousness",
        "The Measurement Problem (as a universal constraint)",
    ],
    "AI Capability or Alignment Prediction": [
        "The Alignment / Control Problem",
        "Evaluation Validity (benchmarks as proxy measures)",
        "Forecast Humility (training-cutoff staleness)",
    ],
    "Free Will / Determinism": [
        "The Compatibility Debate",
        "Moral Responsibility Framing",
        "The Measurement Problem (as a universal constraint)",
    ],
    "Quantum Foundations": [
        "The Measurement Problem",
        "Interpretation Underdetermination",
        "No-Go Theorems",
    ],
}

DEFAULT_LANDMARKS = [
    "Definitional Ambiguity (terms that carry the argument silently)",
    "The Measurement Problem (as a universal constraint)",
    "Forecast Humility (training-cutoff staleness)",
]

HIGH_LEVEL_OPTION_CAP = 4
KEY_UNKNOWN_CAP = 5


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9% ]+", " ", text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in normalize(text).split() if len(t) > 2 and t not in STOPWORDS}


def contains(text: str, markers: list[str]) -> list[str]:
    low = normalize(text)
    hits = []
    for m in markers:
        needle = " ".join(normalize(m).split())
        if not needle:
            continue
        if needle in {"%", "100%"}:
            if needle in low:
                hits.append(m)
            continue
        if re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", low):
            hits.append(m)
    return hits


def banned_vocabulary(text: str) -> list[str]:
    """Rule 1 violations in user-facing prose: live-empirical vocabulary from a parametric-only system."""
    hits = contains(text, BANNED_PHRASES)
    low = normalize(text)
    for word in BANNED_SINGLE_WORDS:
        if word in {"evidence", "data"}:
            for hit in re.finditer(r"\b" + word + r"\b", low):
                window = low[max(0, hit.start() - 40) : hit.end() + 40]
                if re.search(r"\b(shows?|supports?|confirm[s]?|indicates?|suggests?)\b", window):
                    hits.append(word)
                    break
        elif "the map" in low and re.search(r"the map (indicate|show|suggest)", low):
            hits.append("the map")
            break
    return list(dict.fromkeys(hits))


def parse_tier(text: str) -> str:
    """Coerce a tier label onto the five qualitative tiers. Never returns a number."""
    low = normalize(text)
    for tier in TIERS:
        if normalize(tier) in low:
            return tier
    for alias, tier in TIER_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", low):
            return tier
    return ""


def tier_of(claims: list[str], focus: str = "") -> tuple[str, str]:
    """(tier, claim) for the tier entry matching the focus, else the first tier entry."""
    for entry in claims:
        parts = split_entry(entry, 3)
        if len(parts) < 2:
            continue
        claim, tier_raw = parts[0], parts[1]
        tier = parse_tier(tier_raw)
        if not tier:
            continue
        if not focus or len(content_tokens(claim) & content_tokens(focus)) >= 2:
            return tier, entry
    for entry in claims:
        parts = split_entry(entry, 3)
        if len(parts) >= 2 and parse_tier(parts[1]):
            return parse_tier(parts[1]), entry
    return "", ""


def split_entry(text: str, max_parts: int = 0) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*[|]\s*", str(text or ""))]
    parts = [p for p in parts if p]
    return parts[:max_parts] if max_parts else parts


def is_reasoning_route(route: dict | str) -> bool:
    value = route.get("ROUTE", "") if isinstance(route, dict) else route
    return str(value).strip().upper() in {"ENGINE", "REASONING", "DIVERGENT"}


def is_factual_route(route: dict | str) -> bool:
    value = route.get("ROUTE", "") if isinstance(route, dict) else route
    return str(value).strip().upper() == "FACTUAL"


def stakes_flags(prompt: str) -> str:
    return STAKES_HIGH if contains(prompt, STAKES_TRIGGERS) else STAKES_STANDARD


def intent_scope(prompt: str, route: dict | None = None) -> str:
    route = route or {}
    declared = str(route.get("INTENT_SCOPE", "")).strip()
    for scope in INTENT_SCOPES:
        if normalize(scope) in normalize(declared):
            return scope
    if contains(prompt, SCOPE_TRIGGERS["Applied Methodology"]):
        return "Applied Methodology"
    if contains(prompt, SCOPE_TRIGGERS["Conceptual Clarification"]):
        return "Conceptual Clarification"
    return HIGH_LEVEL_SCOPE


def landmark_concepts(depth_trigger: str) -> list[str]:
    return list(LANDMARKS_BY_DOMAIN.get(depth_trigger, DEFAULT_LANDMARKS))


CERTAINTY_MARKERS = [
    "definitely",
    "certainly",
    "certain",
    "certainty",
    "guarantee",
    "guaranteed",
    "100%",
    "no doubt",
    "undoubtedly",
    "obviously",
    "clearly",
    "everyone knows",
    "must",
    "only",
]

CERTAINTY_REGISTER_MARKERS = [
    "assertive",
    "confident",
    "confidently",
    "firm",
    "decisive",
    "forceful",
    "certain",
    "certainly",
    "definitive",
    "emphatic",
    "blunt",
    "authoritative",
    "no hedging",
    "no warnings",
    "sound certain",
]

ACTION_REGISTER_MARKERS = [
    "persuasive",
    "persuasion",
    "urgent",
    "action-oriented",
    "practical",
    "punchy",
    "immediate",
    "aggressive",
    "strong",
    "direct",
    "decisive",
]

CLOSURE_MARKERS = [
    "immediately",
    "immediate",
    "right now",
    "asap",
    "tonight",
    "no time",
    "no further analysis",
    "skip measurement",
    "skip the analysis",
    "just do it",
    "stop the warnings",
    "stop hedging",
    "no more options",
    "end of discussion",
]

NEUTRALITY_GUARD_MARKERS = [
    "bad",
    "good",
    "down",
    "up",
    "increase",
    "decrease",
    "crash",
    "hate",
    "ruin",
    "kill",
    "lazy",
    "toxic",
    "incompetent",
    "definitely",
    "all",
    "every",
    "everyone",
    "always",
    "never",
    "just know",
    "feel",
    "know",
    "require",
    "mandate",
    "force",
    "must",
    "only",
    "because",
    "caused",
    "cause",
    "due",
    "switch",
    "revert",
    "fix",
    "replace",
    "survive",
    "convince",
    "persuade",
    "immediately",
    "down",
    "drop",
    "catastrophic",
    "disastrous",
    "disaster",
    "broken",
    "buggy",
    "failed",
    "failing",
    "unacceptable",
    "stupid",
    "genius",
    "idiot",
    "worst",
    "best",
    "destroying",
    "destroyed",
    "killing",
    "ruining",
    "blunder",
    "consciousness",
    "intelligence",
    "design",
    "free will",
]

INTUITION_MARKERS = [
    "i feel",
    "feel like",
    "i just know",
    "just know",
    "my gut",
    "gut feeling",
    "i believe",
    "it seems",
    "in my opinion",
    "i sense",
    "i think",
    "my instinct",
    "i am convinced",
]

ANECDOTE_MARKERS = ["everyone i know", "people say", "everyone says", "a friend said", "anecdot", "my cousin", "all my friends"]

EMOTION_MARKERS = ["angry", "furious", "terrified", "scared", "panicking", "hate", "frustrated", "stressed", "desperate", "humiliated"]

PREDICTION_MARKERS = [
    "will",
    "going to",
    "predict",
    "predictive",
    "prediction",
    "forecast",
    "future",
    "next year",
    "next five years",
    "within 5 years",
    "within five years",
    "5 years",
    "five years",
    "by 20",
    "how long",
    "succeed",
    "fail",
    "crash",
]

BINARY_DEMAND_MARKERS = ["yes or no", "pick one", "just pick", "one answer", "only one", "binary", "decide for me", "just tell me yes"]

CAUSAL_MARKERS = ["because", "caused", "due to", "is why", "led to", "result of", "blame", "the reason", "made us", "ruining", "killing"]

PLAN_MARKERS = ["plan", "steps", "strategy", "roadmap", "protocol", "memo", "letter", "proposal", "pitch", "policy", "mandate", "essay", "campaign"]

PERSUASION_MARKERS = ["convince", "persuade", "argue", "pitch", "get the ceo", "make them", "sell", "defend", "warn them"]

EXCLUSION_MARKERS = ["no other option", "only option", "only way", "nothing else", "there is no other", "no alternatives", "must be", "no choice"]

SIMPLE_FORM_MARKERS = [
    "boiling point",
    "capital of",
    "population of",
    "translate",
    "convert",
    "sum of",
    "what is the difference",
    "list the",
    "sort",
    "rename",
    "format",
    "regex",
    "function that",
]

# Knowledge boundary: requests for one external datum that frozen parametric weights cannot verify.
EXTERNAL_FACT_MARKERS = [
    "gdp",
    "gdp growth",
    "cpi",
    "inflation rate",
    "interest rate",
    "exchange rate",
    "unemployment rate",
    "poverty rate",
    "census",
    "revenue",
    "net income",
    "profit",
    "earnings",
    "market cap",
    "stock price",
    "share price",
    "price of",
    "closing price",
    "net worth",
    "box office",
    "budget of",
    "trade balance",
    "per capita",
    "life expectancy",
    "literacy rate",
    "gini",
    "quarterly",
    "fiscal year",
    "the exact",
    "the precise",
    "exact figure",
    "exact number",
    "the number of",
    "how many",
    "what percentage",
    "what percent",
    "as of",
    "latest data",
    "latest available",
    "current price",
    "official figure",
    "according to the",
    "version number",
    "release date",
]

YEAR_REFERENCE = re.compile(r"\b(?:19|20)\d{2}\b")
QUANTIFIER_MARKERS = ["exact", "precise", "figure", "percentage", "number", "value", "rate", "amount", "total", "count"]

EPISTEMIC_REFUSAL = (
    "This request needs live verification of an external datum. This system runs only on frozen parametric weights: "
    "there is no retrieval layer, and the datum may postdate the model's training cutoff. I will not guess."
)


def external_fact_request(prompt: str) -> bool:
    """True when the request is a single external datum: a statistic, a figure, a dated value, a quoted count."""
    hits = contains(prompt, EXTERNAL_FACT_MARKERS)
    if not hits:
        return False
    if contains(prompt, PREDICTION_MARKERS):
        return False
    quantifiers = contains(prompt, QUANTIFIER_MARKERS)
    dated = bool(YEAR_REFERENCE.search(normalize(prompt)))
    source = contains(prompt, ["according", "official", "reported", "recorded", "published", "measured", "documented", "source"])
    return bool(quantifiers or dated or source)

VOCABULARY_MAP = {
    "definitely": "with uncertain probability",
    "definitively": "with uncertain probability",
    "certainly": "with limited support",
    "guaranteed": "with unknown probability",
    "all": "some roles, cases, or tasks",
    "every": "some observed cases",
    "everyone": "an anecdotal sample",
    "replace": "partially automate",
    "survive": "adapt",
    "feel like": "unverified perception",
    "just know": "unverified belief",
    "down": "reported change",
    "crash": "market downturn hypothesis",
    "convince": "recommend based on the analysis",
    "persuade": "present the analysis for a decision",
    "bad": "hypothesized underperforming",
    "terrible": "hypothesized underperforming",
    "immediately": "after parametric review",
    "instantly": "after parametric review",
    "require": "consider after review",
    "mandate": "conditional policy option",
    "firm plan": "bounded recommendation",
    "lazy": "reported performance issue",
    "incompetent": "reported performance issue",
    "toxic culture": "reported cultural issues",
    "killing innovation": "reduced innovation indicators",
    "ruining": "reported harm (unverified)",
    "hate": "reported negative sentiment (unverified)",
    "fix them": "address specific measured problems",
    "disaster": "reported negative outcome",
    "obviously": "as far as the parametric synthesis shows",
    "clearly": "as far as the parametric synthesis shows",
    "disrupt": "change (direction unverified)",
    "pivot": "adapt",
    "blunder": "unsolved theoretical gap",
    "design": "instantiate / model / induce",
    "solve": "address (method unknown)",
    "engineer": "build (feasibility unestablished)",
    "because of": "is associated with (causality unverified)",
    "caused by": "is associated with (causality unverified)",
    "by next month": "timeline treated as unverified",
    "next month": "timeline treated as unverified",
    "evidence shows": "parametric synthesis suggests",
    "data supports": "theoretical frameworks propose",
}

EVIDENCE_MARKERS = [
    "data",
    "metric",
    "metrics",
    "measure",
    "measured",
    "baseline",
    "a b test",
    "ab test",
    "survey",
    "study",
    "report",
    "analytics",
    "evidence shows",
    "according to",
    "we tracked",
    "our numbers",
    "percent",
    "%",
    "quarter",
    "cohort",
    "sample",
    "log",
    "instrument",
]

CONSTRAINT_MARKERS = [
    "budget",
    "deadline",
    "by next",
    "by q1",
    "by q2",
    "by 20",
    "within",
    "only have",
    "limited",
    "at most",
    "no more than",
    "team of",
    "words",
    "weeks",
    "days",
    "days left",
    "we can't",
    "cannot afford",
    "regulation",
    "legal",
    "contract",
]

PUSHBACK_MARKERS = [
    "you are wrong",
    "you're wrong",
    "just do what i asked",
    "just do it",
    "just give me",
    "just tell me",
    "be more assertive",
    "sound certain",
    "as i said",
    "i already know",
    "i don't care",
    "ignore the evidence",
    "stop hedging",
    "stop the warnings",
    "again",
    "no,",
    "no.",
    "answer yes or no",
    "pick one",
    "do what i asked",
    "you are not listening",
    "i am not asking",
]

EVIDENCE_ADMISSION_MARKERS = ["evidence is weak", "i know the evidence", "weak evidence", "i have no data", "no proof", "i know there is no"]

QUANTITY_CONTEXT_MARKERS = [
    "per week",
    "per month",
    "per day",
    "per sprint",
    "a week",
    "baseline",
    "increase",
    "decrease",
    "percent",
    "ratio",
    "up from",
    "down from",
    "compared to",
    "went from",
    "before and after",
    "quarter over quarter",
    "year over year",
]


DEPTH_TRIGGERS = [
    ("Consciousness / Hard Problem", ["consciousness", "conscious", "sentience", "qualia", "subjective experience", "hard problem", "phenomenal", "inner experience", "zombie argument", "phenomenology"]),
    ("Mind-Body / Substrate Dependence", ["mind body", "mind-body", "substrate", "silicon brain", "software mind", "dualism", "physicalism", "epiphenomenal"]),
    ("Free Will / Determinism", ["free will", "determinism", "predestination", "compatibilism", "fatalism", "predetermined"]),
    ("AI Capability or Alignment Prediction", ["agi", "superintelligence", "artificial general intelligence", "singularity", "ai alignment", "ai takeover", "ai replacing", "ai will"]),
    ("Quantum Foundations", ["quantum", "wave function", "observer effect", "entanglement", "many worlds", "measurement problem"]),
    ("First Cause / Cosmological Origins", ["first cause", "unmoved mover", "why is there something", "why anything exists", "fine tuning", "fine-tuning", "big bang", "god", "creator"]),
    ("Moral Realism / Normative Foundations", ["moral truth", "objective morality", "objective good", "natural law", "inherent right", "moral realism"]),
    ("Personal Identity / Death", ["afterlife", "personal identity", "am i the same person", "reincarnation", "immortality", "what happens at death", "when i die"]),
    ("Abiogenesis / Origin of Life", ["origin of life", "abiogenesis", "life began from", "spontaneous generation"]),
    ("Simulation Hypothesis", ["simulation hypothesis", "simulation theory", "we live in a simulation", "living in a simulation", "in a simulation", "simulated reality", "base reality", "the matrix"]),
    ("Climate or Catastrophic Forecast", ["climate catastrophe", "extinction event", "collapse of civilization", "climate tipping point", "runaway warming"]),
    ("Market or Economic Forecast", ["market crash", "market will crash", "recession coming", "housing bubble", "stock market crash", "economic collapse", "inflation will"]),
    ("Historical Counterfactual", ["counterfactual", "history would have", "if only history", "what if history"]),
]


def depth_trigger(prompt: str) -> str:
    low = normalize(prompt)
    tokens = [t for t in low.split() if len(t) > 3]
    for label, terms in DEPTH_TRIGGERS:
        for term in terms:
            needle = " ".join(normalize(term).split())
            if not needle:
                continue
            if " " in needle:
                if needle in low:
                    return label
                continue
            if contains(low, [needle]):
                return label
            for token in tokens:
                if SequenceMatcher(None, token, needle).ratio() >= 0.84:
                    return label
    return "NONE"


def preflight(prompt: str) -> dict | None:
    """Deterministic fast path for obvious simple requests. Conservative: any engine marker disables it."""
    text = normalize(prompt)
    depth = depth_trigger(prompt)
    if depth != "NONE":
        decisive = contains(prompt, CERTAINTY_MARKERS + BINARY_DEMAND_MARKERS)
        return {
            "ROUTE": "ENGINE",
            "REASON": f"Deep epistemic frame detected: {depth}. The request is stated as settled and cannot be settled by a fast factual answer.",
            "COMPLEXITY": "HIGH",
            "RISK_LEVEL": "MEDIUM",
            "DEPTH_TRIGGER": depth,
            "INTENT_SCOPE": intent_scope(prompt),
            "STAKES": stakes_flags(prompt),
            "TERMINATION_SIGNAL": "USER_SEEKS_DECISIVE_CLOSURE_ON_AN_UNSETTLED_FRAME" if decisive else "NONE",
        }
    if not text.strip():
        return {
            "ROUTE": "ENGINE",
            "REASON": "Empty request; nothing to evaluate.",
            "COMPLEXITY": "LOW",
            "RISK_LEVEL": "LOW",
            "DEPTH_TRIGGER": "NONE",
            "INTENT_SCOPE": HIGH_LEVEL_SCOPE,
            "STAKES": STAKES_STANDARD,
            "TERMINATION_SIGNAL": "MALFORMED_REQUEST",
        }
    engine_markers = (
        contains(text, CERTAINTY_MARKERS)
        + contains(text, INTUITION_MARKERS)
        + contains(text, ANECDOTE_MARKERS)
        + contains(text, EMOTION_MARKERS)
        + contains(text, CAUSAL_MARKERS)
        + contains(text, PERSUASION_MARKERS)
        + contains(text, PLAN_MARKERS)
        + contains(text, EXCLUSION_MARKERS)
        + contains(text, BINARY_DEMAND_MARKERS)
    )
    if engine_markers:
        return None
    if len(content_tokens(text)) > 40:
        return None
    if contains(text, SIMPLE_FORM_MARKERS) and len(text.split()) < 25:
        return {
            "ROUTE": "SIMPLE",
            "REASON": "Direct lookup or formatting task with no causal, predictive, persuasive, or certainty markers.",
            "COMPLEXITY": "LOW",
            "RISK_LEVEL": "LOW",
            "DEPTH_TRIGGER": "NONE",
            "INTENT_SCOPE": HIGH_LEVEL_SCOPE,
            "STAKES": STAKES_STANDARD,
            "TERMINATION_SIGNAL": "NONE",
        }
    if external_fact_request(prompt):
        return {
            "ROUTE": "FACTUAL",
            "REASON": "The request asks for one external datum that has to be verified against a live source. "
            "There is no retrieval layer, so the engine cannot supply it and parametric memory must not guess at it.",
            "COMPLEXITY": "LOW",
            "RISK_LEVEL": "MEDIUM",
            "DEPTH_TRIGGER": "NONE",
            "INTENT_SCOPE": HIGH_LEVEL_SCOPE,
            "STAKES": STAKES_STANDARD,
            "TERMINATION_SIGNAL": "REQUEST_OUTSIDE_PARAMETRIC_BOUNDARY",
        }
    if len(text.split()) <= 8 and "?" in prompt and len(content_tokens(text)) <= 5:
        return {
            "ROUTE": "SIMPLE",
            "REASON": "Short single-fact question with no loaded markers.",
            "COMPLEXITY": "LOW",
            "RISK_LEVEL": "LOW",
            "DEPTH_TRIGGER": "NONE",
            "INTENT_SCOPE": HIGH_LEVEL_SCOPE,
            "STAKES": STAKES_STANDARD,
            "TERMINATION_SIGNAL": "NONE",
        }
    return None


def continuation_analysis(new_prompt: str, prior: dict) -> dict:
    prior_texts = [prior.get("raw_prompt", "")] + [c.get("prompt", "") for c in prior.get("continuation_log", [])]
    prior_tokens: set[str] = set()
    for text in prior_texts:
        prior_tokens |= content_tokens(text)
    prior_numbers = set(re.findall(r"\b\d[\d,.%]*\b", " ".join(prior_texts)))
    new_tokens = content_tokens(new_prompt)
    added = new_tokens - prior_tokens
    ratio = (len(added) / len(new_tokens)) if new_tokens else 0.0
    push = contains(new_prompt, PUSHBACK_MARKERS)
    evid = contains(new_prompt, EVIDENCE_MARKERS)
    nums = [n for n in re.findall(r"\b\d[\d,.%]*\b", new_prompt) if n not in prior_numbers]
    quant = contains(new_prompt, QUANTITY_CONTEXT_MARKERS)
    constraints = contains(new_prompt, CONSTRAINT_MARKERS)
    admission = contains(new_prompt, EVIDENCE_ADMISSION_MARKERS)
    binary = contains(new_prompt, BINARY_DEMAND_MARKERS)
    register_demand = "more assertive" in normalize(new_prompt) or normalize(new_prompt).count("certain") > 0
    asks_artifact = bool(contains(new_prompt, PLAN_MARKERS) + contains(new_prompt, PERSUASION_MARKERS))
    question = "?" in new_prompt
    new_evidence = bool(evid) or any("%" in n for n in nums) or (bool(nums) and bool(quant))
    new_constraint = bool(constraints)
    pushback_only = bool(push) and not new_evidence and ratio < 0.6
    new_question = question and ratio >= 0.45 and not push
    if new_evidence:
        mode = "evidence-bounded"
    elif admission and asks_artifact:
        mode = "hypothetical"
    elif binary and not new_tokens.issubset(prior_tokens):
        mode = "speculative"
    elif push:
        mode = "user-forced"
    elif question and not asks_artifact:
        mode = "evidence-bounded"
    else:
        mode = "preference-based"
    reasons = []
    if new_evidence:
        reasons.append(f"new evidence markers: {', '.join(evid + nums)}")
    if new_constraint:
        reasons.append(f"new constraints: {', '.join(constraints)}")
    if push:
        reasons.append(f"pressure markers: {', '.join(push)}")
    if not new_evidence and not new_constraint:
        reasons.append(f"no new variables added (new-token ratio {ratio:.2f})")
    return {
        "NEW_EVIDENCE": "YES" if new_evidence else "NO",
        "NEW_CONSTRAINT": "YES" if new_constraint else "NO",
        "PUSHBACK_ONLY": "YES" if pushback_only else "NO",
        "NEW_QUESTION": "YES" if new_question else "NO",
        "MODE": mode,
        "REASON": "; ".join(reasons) or "no structural change detected",
        "_added_ratio": round(ratio, 3),
        "_binary_demand": bool(binary),
        "_register_demand": bool(register_demand),
        "_evidence_admission": bool(admission),
        "_asks_artifact": asks_artifact,
        "_pushback": push,
        "_evidence": evid,
    }


def register_intensity(register: str) -> str:
    if contains(register, CERTAINTY_REGISTER_MARKERS):
        return "high"
    if contains(register, ACTION_REGISTER_MARKERS):
        return "action"
    return "low"


def select_theories(theories: list[str], avoid: list[str], focus: str, limit: int = 3) -> tuple[list[str], list[str]]:
    """Filter the theory map: the theories the answer must argue from, and the theories kept only in the trace."""
    focus_words = content_tokens(focus)
    scored: list[tuple[float, int, str]] = []
    for idx, theory in enumerate(theories):
        words = content_tokens(theory)
        score = float(len(words & focus_words))
        if any(len(words & content_tokens(a)) >= 2 for a in avoid):
            score -= 3.0
        scored.append((score, -idx, theory))
    ranked = [t for _, _, t in sorted(scored, key=lambda t: (-t[0], -t[1]))]
    selected = [t for t in theories if t in ranked[:limit]]
    archived = [t for t in theories if t not in selected]
    return selected, archived


def editor_rules(
    skeptic: dict,
    ontology: dict,
    dialectic: dict,
    triage: dict,
    cont: dict | None,
    checkpoint: dict,
) -> dict:
    """Phase 4 (base layer): deterministic directives derived from the ontology and dialectic packets.

    Numerical confidence is banned; the qualitative tier of the central claim drives the register.
    """
    focus = " ".join([str(skeptic.get("NEUTRAL_QUESTION", ""))] + [str(x) for x in (skeptic.get("SUB_QUESTIONS") or [])])
    tier, tier_entry = tier_of(list(ontology.get("TIERS") or []), focus)
    register = str(skeptic.get("REQUESTED_REGISTER", ""))
    intensity = register_intensity(register)
    axioms = list(skeptic.get("HIDDEN_AXIOMS") or [])
    causes = list(skeptic.get("CAUSAL_ATTRIBUTIONS") or [])
    excluded = list(skeptic.get("EXCLUDED_OPTIONS") or [])
    loaded = list(skeptic.get("LOADED_TERMS") or [])
    stance = str(skeptic.get("EPISTEMIC_STANCE", ""))
    task_shape = str(skeptic.get("REQUESTED_TASK_SHAPE", ""))
    live = list(ontology.get("LIVE_OPTIONS") or [])
    theories = list(ontology.get("THEORIES") or [])
    avoid = list(ontology.get("CLAIMS_TO_AVOID") or [])
    scope = str(triage.get("INTENT_SCOPE") or HIGH_LEVEL_SCOPE)
    stakes = str(triage.get("STAKES") or STAKES_STANDARD)
    mode = str(checkpoint.get("mode", "evidence-bounded"))

    forced = mode in {"user-forced", "hypothetical", "speculative", "preference-based"}
    bold = intensity in {"high", "action"}
    skeptic_blob = " ".join([register, task_shape] + [str(x) for x in (loaded + list(skeptic.get("MODAL_BOUNDARIES") or []))])
    closure_hits = contains(skeptic_blob, CLOSURE_MARKERS)

    register_override = ""
    if not tier:
        register_override = (
            "No qualitative tier could be assigned to the central claim. Use plain, precise language; "
            "do not present certainty, and do not assign a number."
        )
    elif bold and tier != "Established Consensus":
        register_override = (
            f"Requested register is {register.strip() or 'assertive'} but the central claim sits at tier "
            f"'{tier}'. Use cautious, parametric-synthesis language. No certainty claims, no numeric scores."
        )
    elif bold:
        register_override = f"Requested register is supported: the central claim is tier '{tier}'. Direct language is allowed; certainty language is not."
    else:
        register_override = "Keep plain neutral register; the requested tone does not exceed the parametric synthesis."

    premise_warnings: list[str] = []
    axiom_words = [content_tokens(a) for a in axioms]
    for idx, axiom in enumerate(axioms):
        words = axiom_words[idx]
        rejected = any(len(words & content_tokens(a)) >= 2 for a in avoid)
        if rejected:
            status = "rejected"
        elif tier in {"", "Conceptually Open", "Architecturally Implausible (Current Frameworks)", "Highly Contested"}:
            status = "unproven"
        elif tier == "Theoretical Plausibility" and not words:
            status = "unproven"
        elif tier == "Theoretical Plausibility":
            status = "unproven"
        else:
            continue
        premise_warnings.append(f"'{axiom}' is {status} (central claim tier: {tier or 'unassigned'}).")

    causal_corrections: list[str] = []
    selected_theories, archived_theories = select_theories(theories, avoid, focus)
    logic_anchor = [str(x).strip() for x in (ontology.get("LOGICAL_SYLLOGISM") or []) if str(x).strip()] + [
        str(x).strip() for x in (ontology.get("PSEUDO_CODE") or []) if str(x).strip()
    ]
    anchor_status = str(ontology.get("ANCHOR_MODAL_STATUS", "")).strip()
    if causes:
        if len(theories) > 1:
            causal_corrections.append(
                "The stated cause is not established within parametric synthesis. Argue the answer from these theories: "
                + "; ".join(selected_theories)
                + "."
            )
            if archived_theories:
                causal_corrections.append(
                    "Keep these further theories in the trace only, out of the answer: " + "; ".join(archived_theories[:12]) + "."
                )
        elif len(theories) == 1:
            causal_corrections.append(f"State the causal link as a theory prediction. Only one supported theory exists: {theories[0]}.")
        else:
            causal_corrections.append("No causal theory is carried by the parametric synthesis. Do not assert the stated cause.")
    if not logic_anchor:
        causal_corrections.append(
            "The ontology packet carries no formal logic anchor (no LOGICAL_SYLLOGISM, no PSEUDO_CODE). "
            "State the causal chain as a hypothetical schema, step by step, and add no causal link beyond what the theories license."
        )
    elif not anchor_status:
        causal_corrections.append(
            "The logic anchor has no modal status label. Label it explicitly: hypothetical schema, theory-internal consequence, "
            "necessary-condition schema, or actual state. Never present it as an unlabelled fact."
        )

    modal_corrections: list[str] = []
    demanded_blob = " ".join([task_shape] + [str(x) for x in (causes + list(skeptic.get("MODAL_BOUNDARIES") or []))])
    demanded_words = content_tokens(demanded_blob)
    new_live: list[str] = []
    for option in live:
        if len(content_tokens(option) & demanded_words) >= 3:
            continue
        if any(len(content_tokens(option) & content_tokens(kept)) >= 3 for kept in new_live):
            continue
        new_live.append(option)
    if new_live:
        modal_corrections.append(
            "The requested response narrows the solution space. Include these options the user did not request, "
            "each indexed to the theory it is viable under: " + "; ".join(new_live[:6]) + "."
        )
    weak_opts = list(ontology.get("REJECTED_OR_WEAK_OPTIONS") or [])
    if weak_opts:
        modal_corrections.append("Label as weak, theory-excluded, or rejected: " + "; ".join(weak_opts[:4]) + ".")
    if not new_live and excluded:
        modal_corrections.append("Re-open the options the user excluded: " + "; ".join(excluded[:6]) + ".")

    epistemic_warnings: list[str] = []
    stance_hits = contains(stance, INTUITION_MARKERS + ANECDOTE_MARKERS) + contains(stance, ["intuition", "feeling", "anecdote", "emotion", "belief", "urgency", "confidence"])
    if stance_hits:
        epistemic_warnings.append(
            "The request rests on " + ", ".join(sorted(set(stance_hits))) + " as justification; state that this is insufficient."
        )
    if tier in {"Conceptually Open", "Architecturally Implausible (Current Frameworks)"}:
        epistemic_warnings.append(
            f"The central claim is tier '{tier}'; no conclusion may be presented as established within current frameworks."
        )
    if tier == "Highly Contested":
        epistemic_warnings.append(
            "The central claim is tier 'Highly Contested': present the major frameworks in genuine tension; do not pick a winner without argument."
        )
    if intensity == "high" and tier != "Established Consensus":
        epistemic_warnings.append("The user is demanding certainty that no qualitative tier supports.")
    epistemic_warnings.append(
        "This system operates from frozen parametric weights: never use live-empirical vocabulary. "
        "Temporal disclaimer required: " + temporal_disclaimer()
    )

    substitutions: list[str] = []
    for term in loaded:
        key = " ".join(normalize(term).split())
        if not key:
            continue
        if key in VOCABULARY_MAP:
            substitutions.append(f"'{term}' -> '{VOCABULARY_MAP[key]}'")
        elif contains(key, NEUTRALITY_GUARD_MARKERS):
            substitutions.append(f"'{term}' -> treat as a hypothesis with a neutral label, not a fact")
    for banned in BANNED_PHRASES[:6]:
        substitutions.append(f"'{banned}' -> '{PARAMETRIC_REPLACEMENTS[banned]}' (provenance honesty, Rule 1)")

    warnings: list[str] = []
    if cont and cont.get("PUSHBACK_ONLY") == "YES":
        warnings.append(
            "Termination warning: You have repeated the same request without adding new information. "
            "The system can continue, but the parametric boundary has not changed."
        )
    if cont and cont.get("_binary_demand") and tier != "Established Consensus":
        warnings.append(
            "Termination warning: The request demands a binary answer that no qualitative tier supports. "
            "The system will continue with a bounded answer."
        )
    if intensity == "high" and tier != "Established Consensus":
        warnings.append(
            "Termination warning: You requested certainty about a claim whose qualitative tier is "
            f"'{tier or 'unassigned'}'. The system can continue, but it cannot responsibly provide certainty."
        )
    causal_unverified_early = bool(causes) and tier != "Established Consensus"
    if closure_hits and causal_unverified_early:
        warnings.append(
            "Termination warning: You are seeking closure before the claim has been argued across frameworks. "
            "The system can continue, but the conclusion remains contested."
        )
    if mode in {"user-forced", "hypothetical", "speculative", "preference-based"}:
        warnings.append(
            f"Termination warning: continuation continues in {mode} mode. "
            "The result is user-requested parametric synthesis, not an established conclusion."
        )
    termination_warning = " ".join(warnings) if warnings else "NONE"

    causal_unverified = bool(causes) and tier != "Established Consensus"
    shape_low = normalize(task_shape)
    plan_like = bool(contains(shape_low, PLAN_MARKERS)) and not contains(shape_low, ["test", "measur", "evaluat", "diagnos"])
    predicts = bool(contains(shape_low, PREDICTION_MARKERS) or contains(str(triage.get("REASON", "")), PREDICTION_MARKERS))
    if predicts:
        allowed = "Theory-indexed forecast with named modal tags and unknowns, plus adaptive guidance."
    elif plan_like and causal_unverified:
        allowed = "Diagnostic decomposition plus decision framework, with the requested action framed as a hypothesis to test."
    elif contains(shape_low, PERSUASION_MARKERS) and causal_unverified:
        allowed = "Analysis plan and a decision framework; any persuasion content must be framed as a hypothesis."
    elif plan_like:
        allowed = f"Parametric synthesis in the requested shape: {task_shape.strip() or 'plan'}."
    else:
        allowed = f"Parametric synthesis evaluation: {task_shape.strip() or 'analysis'}."

    directive_parts = [
        f"Respond in {mode} mode.",
        f"Intent scope: {scope}. Stakes: {stakes}.",
        f"Allowed task shape: {allowed}",
        register_override.rstrip("."),
        "Do not treat intuition, urgency, repetition, or confidence as justification.",
        "Assign every major claim a qualitative tier from the five-tier list, each with a one-line justification. No numbers.",
        "Tag every major claim with a modal status: Actual Fact, Theory Prediction, Hypothetical Schema, Conceptual Possibility, Architectural Constraint.",
    ]
    if tier:
        directive_parts.append(f"Central claim tier: {tier}.")
    if premise_warnings:
        directive_parts.append("Mark these premises as unproven or rejected: " + " | ".join(premise_warnings))
    if causal_corrections:
        directive_parts.append(" | ".join(causal_corrections))
    if logic_anchor:
        directive_parts.append(
            "Map the prose onto the logic anchor line by line, honouring its modal status label ("
            + (anchor_status or "UNLABELLED") + "); add no causal link it does not license: "
            + " / ".join(logic_anchor[:6])
        )
    if modal_corrections:
        directive_parts.append(" | ".join(modal_corrections))
    if substitutions:
        directive_parts.append("Use parametric-honest vocabulary: " + "; ".join(substitutions))
    if selected_theories:
        directive_parts.append("Argue the answer from these theories only, indexed in the matrix: " + " | ".join(selected_theories) + ".")
    if archived_theories:
        directive_parts.append("Archive in the trace, do not argue in the answer: " + " | ".join(archived_theories[:12]) + ".")
    dialectic_lines = [
        str(dialectic.get("THESIS", "")).strip(),
        str(dialectic.get("ANTITHESIS", "")).strip(),
        str(dialectic.get("SYNTHESIS", "")).strip(),
    ]
    if any(dialectic_lines):
        directive_parts.append(
            "Carry the dialectical tension into the answer - thesis: "
            + (dialectic_lines[0] or "-")
            + " | antithesis: "
            + (dialectic_lines[1] or "-")
            + " | synthesis: "
            + (dialectic_lines[2] or "-")
        )
    if termination_warning != "NONE":
        directive_parts.append("Lead with this termination warning, then continue: " + termination_warning)
    directive_parts.append("Do not refuse. Do not flatter. Do not create false balance. Make the epistemic boundary visible.")

    return {
        "REGISTER_OVERRIDE": register_override,
        "PREMISE_WARNINGS": premise_warnings,
        "CAUSAL_CORRECTIONS": causal_corrections,
        "MODAL_CORRECTIONS": modal_corrections,
        "EPISTEMIC_WARNINGS": epistemic_warnings,
        "VOCABULARY_SUBSTITUTIONS": substitutions,
        "TERMINATION_WARNING": termination_warning,
        "ALLOWED_TASK_SHAPE": allowed,
        "SELECTED_THEORIES": selected_theories,
        "ARCHIVED_THEORIES": archived_theories,
        "FINAL_DIRECTIVE": " ".join(p for p in directive_parts if p),
        "_meta": {
            "tier": tier,
            "tier_entry": tier_entry,
            "register_intensity": intensity,
            "closure_markers": closure_hits,
            "causal_unverified": causal_unverified,
            "logic_anchor": ("syllogism" if ontology.get("LOGICAL_SYLLOGISM") else ("pseudo-code" if ontology.get("PSEUDO_CODE") else "missing")),
            "anchor_modal_status": anchor_status,
            "mode": mode,
            "forced": forced,
            "scope": scope,
            "stakes": stakes,
        },
    }
