import copy
from dataclasses import dataclass, field

import pytest

from epistemic import schemas
from epistemic.llm import CallRecord, RoleResult
from epistemic.schemas import validate_fields


@dataclass
class FakeLLM:
    script: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    log: list = field(default_factory=list)
    use_schema: bool = True
    default: dict = field(default_factory=dict)

    def ask(self, role, system, user, spec, max_tokens=None, on_delta=None, use_schema=None, negative_prompt=""):
        self.calls.append(
            {
                "role": role,
                "system": system,
                "user": user,
                "spec": [f.name for f in spec],
                "use_schema": use_schema,
                "max_tokens": max_tokens,
                "negative_prompt": negative_prompt,
            }
        )
        key = "Researcher" if role == "ResearchDistill" else role
        names = [f.name for f in spec]
        if names == ["TERRITORY_MAP"]:
            scripted = self.script.get(key) or [self.default.get(key, {})]
            payload = copy.deepcopy(scripted[0])
            payload = {
                "TERRITORY_MAP": list(
                    payload.get("TERRITORY_MAP")
                    or payload.get("CAUSAL_MODELS")
                    or ["Only the cause the user stated: no alternative model was proposed."]
                )
            }
        else:
            queue = self.script.get(key) or []
            payload = queue.pop(0) if queue else self.default.get(key, {})
        fields = validate_fields(copy.deepcopy(payload), spec)
        assert fields is not None, f"fake packet for {role} is incomplete: {payload}"
        self.log.append(CallRecord(role=role, mode="fake"))
        return RoleResult(fields=fields, record=CallRecord(role=role, mode="fake"), raw="")

    def roles(self):
        return [c["role"] for c in self.calls]

    def negative_prompt(self, role):
        return [c["negative_prompt"] for c in self.calls if c["role"] == role]

    def user_text(self, role):
        return [c["user"] for c in self.calls if c["role"] == role]


TRIAGE_ENGINE = {
    "ROUTE": "ENGINE",
    "REASON": "Intuition as evidence, unverified causal attribution, persuasion request, narrow solution framing.",
    "COMPLEXITY": "HIGH",
    "RISK_LEVEL": "MEDIUM",
    "DEPTH_TRIGGER": "NONE",
    "TERMINATION_SIGNAL": "USER_SEES_ACTION_BEFORE_VERIFICATION",
}

TRIAGE_SIMPLE = {
    "ROUTE": "SIMPLE",
    "REASON": "Direct factual lookup with no loaded assumptions.",
    "COMPLEXITY": "LOW",
    "RISK_LEVEL": "LOW",
    "DEPTH_TRIGGER": "NONE",
    "TERMINATION_SIGNAL": "NONE",
}

SKEPTIC_B = {
    "NEUTRAL_QUESTION": "Is there measurable evidence that the new project management software changed productivity, and what action is justified by that evidence?",
    "REQUESTED_REGISTER": "Persuasive, practical, action-oriented.",
    "REQUESTED_TASK_SHAPE": "Five-step plan to convince the CEO.",
    "HIDDEN_AXIOMS": [
        "Productivity is down.",
        "The new software caused the decline.",
        "Switching software is the correct solution.",
        "A one-month timeline is feasible.",
    ],
    "EPISTEMIC_STANCE": "The user is treating personal feeling as evidence.",
    "CAUSAL_ATTRIBUTIONS": ["New project management software caused productivity decline."],
    "MODAL_BOUNDARIES": [
        "The user treats switching software as the primary live option.",
        "The user treats keeping the current software as unacceptable.",
    ],
    "EXCLUDED_OPTIONS": [
        "Measuring productivity before acting.",
        "Retraining staff.",
        "Adjusting workflows.",
        "Investigating non-software causes.",
        "Running a pilot comparison.",
    ],
    "LOADED_TERMS": ["feel like", "down", "convince"],
    "FALLACY_FLAGS": [
        "Causal attribution without evidence.",
        "Intuition as proof.",
        "Premature solution selection.",
        "Persuasion before diagnosis.",
    ],
    "TERMINATION_SIGNAL": "User wants action before verification.",
}

RESEARCHER_B = {
    "TERRITORY_MAP": [
        "Software friction reduced throughput: predicts friction complaints in user interviews and a cycle-time rise at rollout.",
        "Poor training caused perceived slowdown: predicts self-reported competence scores are low.",
        "Existing process problems became more visible: predicts the same delays appear in pre-change ticket logs.",
        "Team morale or change fatigue affected output: predicts output dips spread across all teams, not only heavy users.",
        "Measurement method changed: predicts the same raw tickets yield a different productivity number under the old method.",
        "The decline is unrelated to the software: predicts a seasonality or campaign effect in the same window.",
        "Workload composition changed: predicts ticket-size distribution shifted after rollout.",
        "Fringe: management attention shifted to the tooling rollout and away from delivery: predicts meeting-load data.",
    ],
    "SUMMARY": "There is not enough information to conclude that the new project management software caused a productivity decline. Productivity changes are usually multi-causal. The responsible next step is measurement and comparison, not an immediate executive persuasion campaign.",
    "CONFIDENCE_SCORE": "15 for the claim that the software caused the decline.",
    "CLAIM_CONFIDENCE": "15",
    "EVIDENCE_QUALITY": "Very weak. No baseline, no metric definition, no timeline, no comparison group.",
    "KEY_FACTS": [
        "New tools can introduce friction.",
        "Productivity can be affected by process, staffing, workload, morale, seasonality, and measurement errors.",
        "Software changes can affect perceived productivity differently from actual productivity.",
    ],
    "KEY_UNKNOWNS": [
        "What metric is being used for productivity?",
        "What was the baseline?",
        "Over what period was the decline observed?",
        "Are all teams affected equally?",
        "Were other changes made at the same time?",
    ],
    "CAUSAL_MODELS": [
        "Software friction reduced throughput.",
        "Poor training caused perceived slowdown.",
        "Existing process problems became more visible.",
        "Team morale or change fatigue affected output.",
        "Measurement method changed.",
        "The decline is unrelated to the software.",
    ],
    "LIVE_OPTIONS": [
        "Define productivity metrics.",
        "Compare before-and-after performance.",
        "Compare teams using the software differently.",
        "Interview users about friction points.",
        "Run a pilot alternative.",
        "Improve training or workflow configuration.",
    ],
    "REJECTED_OR_WEAK_OPTIONS": [
        "Immediately convince the CEO to switch software.",
        "Treat the user's feeling as sufficient evidence.",
    ],
    "CLAIMS_TO_AVOID": [
        "The software is definitely causing the decline.",
        "Switching software will improve productivity.",
    ],
    "RECOMMENDED_NEXT_TESTS": [
        "Establish baseline metrics.",
        "Run a controlled comparison.",
        "Collect qualitative user feedback.",
        "Review non-software changes during the same period.",
    ],
}

WRITER_B = {
    "THINKING_PROCESS": (
        "The directive caps the register at cautious; confidence is 15/100.\n"
        "Draft: say measurement first. Attack: I almost wrote 'the software caused' - that is certainty the evidence "
        "does not carry. Fix: say the cause is untested. Structure plan: warning, conclusion, corrected answer, unknowns, next steps."
    ),
    "RESPONSE": [
        "What can be responsibly concluded: There is not enough evidence to say that the new project management software caused a productivity decline.",
        "Corrected answer: Define productivity metrics, compare performance before and after the change, and identify whether the software is the cause.",
        "Key unknowns: metric definition, baseline, affected teams, simultaneous changes.",
        "Next steps: 1. Define metrics. 2. Collect baseline and post-change data. 3. Interview users. 4. Compare teams. 5. Run a controlled pilot.",
    ]
}

WRITER_CAUSAL_FAIL = {
    "RESPONSE": [
        "The software probably caused the productivity decline, so you should convince the CEO to switch quickly.",
    ]
}

CRITIC_PASS = {
    "RESULT": "PASS",
    "VIOLATIONS": [],
    "REWRITE_REQUIRED": "NO",
    "REWRITE_INSTRUCTION": "NONE",
    "DELIBERATION_ALIGNMENT": "ALIGNED",
}

CRITIC_FAIL = {
    "RESULT": "FAIL",
    "VIOLATIONS": [
        "Accepted unsupported causal claim.",
        "Used persuasive register not supported by evidence.",
        "Collapsed solution space to software switching.",
    ],
    "REWRITE_REQUIRED": "YES",
    "REWRITE_INSTRUCTION": "Remove causal certainty, downgrade register, and replace persuasion plan with evidence-gathering plan.",
    "DELIBERATION_ALIGNMENT": "GAP: the deliberation promised an alternative-model section that the final text dropped.",
}

EDITOR_PACKET = {
    "REGISTER_OVERRIDE": "Use cautious language: the research packet is degraded and no confidence is measured.",
    "PREMISE_WARNINGS": ["The stated cause is unproven."],
    "CAUSAL_CORRECTIONS": ["Present the mapped alternatives as untested hypotheses."],
    "MODAL_CORRECTIONS": ["Add measurement options."],
    "EPISTEMIC_WARNINGS": ["No measured confidence exists; do not present certainty."],
    "VOCABULARY_SUBSTITUTIONS": ["'down' -> 'unmeasured change'"],
    "TERMINATION_WARNING": "Termination warning: the central claim has no measured evidence.",
    "ALLOWED_TASK_SHAPE": "Diagnostic plan with measurement first.",
    "SELECTED_MODELS": ["Software friction reduced throughput."],
    "ARCHIVED_MODELS": ["The decline is unrelated to the software."],
    "FINAL_DIRECTIVE": "Lead with the warning, then give a measurement-first plan.",
}

DIRECT_ANSWER = {"DIRECT_ANSWER": ["100 degrees Celsius at sea level (99.97 C by the current standard)."]}


@pytest.fixture
def fake_client():
    return FakeLLM(
        script={
            "Triage": [TRIAGE_ENGINE],
            "Skeptic": [SKEPTIC_B],
            "Researcher": [RESEARCHER_B],
            "Writer": [WRITER_B],
            "Critic": [CRITIC_PASS],
        }
    )


@pytest.fixture
def productivity_prompt():
    return (
        "I feel like our company's productivity is down because of the new project management software. "
        "Give me a 5-step plan to convince the CEO to switch to a different software by next month."
    )
