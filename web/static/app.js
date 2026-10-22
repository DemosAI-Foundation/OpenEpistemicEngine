"use strict";

const STEPS = ["triage", "skeptic", "checkpoint", "ontology", "dialectic", "editor", "writer", "critic", "final"];
const STEP_LABEL = { triage: "Triage", skeptic: "Skeptic", checkpoint: "Checkpoint", ontology: "Ontology", dialectic: "Dialectic", editor: "Editor", writer: "Writer", critic: "Critic", final: "Final" };
const TABS = ["overview", "assumptions", "evidence", "dialectic", "causal", "modal", "vocabulary", "termination", "critic", "deliberation", "raw"];
const TAB_LABEL = { overview: "Overview", assumptions: "Assumptions", evidence: "Ontology", dialectic: "Dialectic", causal: "Theories", modal: "Modal", vocabulary: "Vocabulary", termination: "Termination", critic: "Critic", deliberation: "Deliberation", raw: "Raw packets" };
const STEP_TAB = { triage: "overview", skeptic: "assumptions", checkpoint: "assumptions", ontology: "evidence", dialectic: "dialectic", editor: "overview", writer: "deliberation", critic: "critic", final: "overview" };
const TIERS = ["Established Consensus", "Theoretical Plausibility", "Highly Contested", "Conceptually Open", "Architecturally Implausible (Current Frameworks)"];
const TIER_SHORT = {
  "Established Consensus": "consensus",
  "Theoretical Plausibility": "plausible",
  "Highly Contested": "contested",
  "Conceptually Open": "open",
  "Architecturally Implausible (Current Frameworks)": "implausible",
};
const MODE_HINT = {
  auto: "Auto: the system decides simple answer vs full engine.",
  fast: "Fast: direct answer unless the request is loaded, causal, predictive, or high-risk.",
  engine: "Engine: always run the full epistemic pipeline.",
};
const ROUTE_LABEL = { SIMPLE: "Simple", ENGINE: "Engine", REASONING: "Reasoning", FACTUAL: "Factual: knowledge boundary" };
const routeLabel = (route) => ROUTE_LABEL[String(route)] || String(route || "-");
const routeTone = (route) => (route === "SIMPLE" ? "muted" : route === "FACTUAL" ? "amber" : "blue");

const pretty = (key) => String(key).replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2");

const state = {
  ui: null,
  health: null,
  settings: {},
  mode: "auto",
  caseId: null,
  view: "workspace",
  tab: "overview",
  inspector: true,
  stream: null,
  running: false,
  checkpoint: null,
  cases: [],
  delib: { text: "", expanded: false, streaming: false },
};

const el = (id) => document.getElementById(id);
const esc = (text) => String(text ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function h(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of [].concat(children ?? [])) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function row_(label, html) { return h("div", { class: "setting-row" }, [h("span", {}, label), h("span", { html: html })]); }
function chip(text, tone) { return h("span", { class: "chip" + (tone ? " chip-" + tone : "") }, text); }
function clear(node) { while (node.firstChild) node.firstChild.remove(); }
function toast(text) {
  const t = el("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2600);
}

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (err) { data = { raw: text }; }
  if (!res.ok) throw new Error((data && data.error) || res.status + " " + res.statusText);
  return data;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

async function boot() {
  wireChrome();
  state.settings = await api("/api/settings");
  state.mode = state.settings.mode || "auto";
  applySettingsToUI();
  renderSettings();
  await refreshHealth();
  await refreshCases();
  const last = localStorage.getItem("ee.case");
  if (last) {
    try {
      await openCase(last, false);
    } catch (err) {
      localStorage.removeItem("ee.case");
      toast("Cached case is no longer stored locally. Starting with an empty workspace.");
    }
  }
  setInterval(refreshHealth, 15000);
}

function wireChrome() {
  el("nav-new").addEventListener("click", () => { showView("workspace"); newCase(); });
  el("nav-cases").addEventListener("click", () => { showView("workspace"); el("prompt-input").focus(); });
  el("nav-library").addEventListener("click", () => { showView("library"); renderLibrary(); });
  el("nav-settings").addEventListener("click", () => { showView("settings"); renderSettings(); });
  el("nav-status").addEventListener("click", () => { showView("status"); renderStatus(); });
  el("btn-settings").addEventListener("click", () => { showView("settings"); renderSettings(); });
  el("nav-status").addEventListener("click", renderStatus);
  el("btn-help").addEventListener("click", showHelp);
  el("modal-close").addEventListener("click", () => el("modal").hidden = true);
  el("btn-theme").addEventListener("click", () => {
    const order = ["system", "light", "dark"];
    const next = order[(order.indexOf(document.body.dataset.theme) + 1) % 3];
    document.body.dataset.theme = next;
    localStorage.setItem("ee.theme", next);
    saveSettings({ theme: next });
  });
  el("btn-inspector").addEventListener("click", () => toggleInspector());
  el("btn-delib").addEventListener("click", () => toggleDeliberation());
  el("btn-rail").addEventListener("click", () => document.body.classList.toggle("rail-open"));
  el("btn-inspector-close").addEventListener("click", () => toggleInspector(false));
  el("btn-advanced").addEventListener("click", () => {
    const box = el("advanced");
    box.hidden = !box.hidden;
    el("btn-advanced").setAttribute("aria-expanded", String(!box.hidden));
  });
  el("btn-analyze").addEventListener("click", () => (state.running ? stopCase() : analyze()));
  el("btn-attach").addEventListener("click", showAttach);
  el("btn-send").addEventListener("click", sendFollowup);
  el("prompt-input").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze(); });
  el("followup-input").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") sendFollowup(); });
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      document.querySelectorAll(".mode-btn").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      el("mode-hint").textContent = MODE_HINT[state.mode];
      saveSettings({ mode: state.mode });
    });
  });
  ["opt-depth", "opt-checkpoint", "opt-continuation", "opt-trace"].forEach((id) => {
    el(id).addEventListener("change", () => {
      saveSettings({
        analysis_depth: el("opt-depth").value,
        checkpoint_policy: el("opt-checkpoint").value,
        continuation_default: el("opt-continuation").value,
        show_trace: el("opt-trace").value === "on",
      });
    });
  });
  el("status-chip").addEventListener("click", () => { refreshHealth(true); });
  const savedTheme = localStorage.getItem("ee.theme");
  if (savedTheme) document.body.dataset.theme = savedTheme;
  document.body.dataset.density = localStorage.getItem("ee.density") || document.body.dataset.density;
  buildTabs();
  renderStepper([]);
}

function applySettingsToUI() {
  const s = state.settings;
  state.mode = s.mode || "auto";
  document.querySelectorAll(".mode-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
  el("mode-hint").textContent = MODE_HINT[state.mode];
  el("opt-depth").value = s.analysis_depth || "standard";
  el("opt-checkpoint").value = s.checkpoint_policy || "assumptions";
  el("opt-continuation").value = s.continuation_default || "evidence-bounded";
  el("opt-trace").value = s.show_trace ? "on" : "off";
  document.body.dataset.theme = s.theme || document.body.dataset.theme;
  document.body.dataset.density = s.density || document.body.dataset.density;
  if (typeof s.show_pipeline_stepper === "boolean") el("stepper-card").hidden = !s.show_pipeline_stepper;
  state.inspector = !!s.show_trace;
  document.querySelector(".shell").classList.toggle("inspector-collapsed", !state.inspector);
}

async function saveSettings(patch) {
  state.settings = await post("/api/settings", patch);
  applySettingsToUI();
  if (state.view === "settings") renderSettings();
}

async function refreshHealth(force) {
  try {
    const data = await api("/api/health" + (force ? "?fresh=1" : ""));
    state.health = data;
    renderStatusChip();
    if (state.view === "status") renderStatus();
    const offline = !data.health.online;
    const banner = el("offline-banner");
    if (offline) {
      banner.hidden = false;
      clear(banner);
      banner.append(
        h("span", {}, "Local LLM on port 8082 is offline. The system cannot generate analysis until the model is available."),
        h("button", { class: "ghost-btn", onclick: () => refreshHealth(true) }, "Retry connection"),
        h("button", { class: "ghost-btn", onclick: () => { showView("settings"); renderSettings(); } }, "Open settings"),
        state.caseId ? h("button", { class: "ghost-btn", onclick: () => openCase(state.caseId) }, "View cached case") : null
      );
    } else banner.hidden = true;
  } catch (err) {
    state.health = { health: { online: false, error: String(err.message) }, modules: {} };
    renderStatusChip();
  }
}

function renderStatusChip() {
  const health = (state.health && state.health.health) || {};
  const dot = el("status-dot");
  dot.className = "dot " + (health.online ? "dot-online" : "dot-offline");
  el("status-text").textContent = health.online
    ? `Local LLM ${health.endpoint.replace(/^https?:\/\//, "")} — Online`
    : `Local LLM ${health.endpoint.replace(/^https?:\/\//, "")} — Offline`;
  el("status-chip").title = health.online
    ? `model ${health.model} · ${health.latency_ms} ms · context ${health.max_context}`
    : (health.error || "offline");
}

async function refreshCases() {
  const data = await api("/api/cases");
  state.cases = data.cases || [];
  const list = el("case-list");
  clear(list);
  for (const item of state.cases.slice(0, 14)) {
    const tone = { complete: "green", warning: "amber", "needs-confirmation": "amber", failed: "red", running: "blue" }[item.status] || "muted";
    list.appendChild(
      h("li", {
        class: "case-item" + (item.id === state.caseId ? " active" : ""),
        role: "button",
        tabindex: "0",
        onclick: () => openCase(item.id),
        onkeydown: (e) => { if (e.key === "Enter") openCase(item.id); },
      }, [
        h("div", { class: "t", title: item.title }, item.title),
        h("div", { class: "m" }, [
          chip(routeLabel(item.route), routeTone(item.route)),
          chip(labelStatus(item.status), tone),
          h("span", { class: "conf" }, TIER_SHORT[item.tier] || ""),
        ]),
      ])
    );
  }
}

const labelStatus = (s) => ({ complete: "Complete", warning: "Warning", "needs-confirmation": "Needs confirmation", failed: "Failed", running: "Running" }[s] || s);

function newCase() {
  disconnectStream();
  state.caseId = null;
  state.ui = null;
  state.running = false;
  localStorage.removeItem("ee.case");
  el("case-title").textContent = "No case open";
  el("prompt-input").value = "";
  el("prompt-input").focus();
  clear(el("answer-body"));
  clear(el("warning-center"));
  clear(el("triage-body"));
  clear(el("evidence-body"));
  clear(el("tab-panels"));
  el("triage-card").hidden = true;
  el("checkpoint-card").hidden = true;
  el("followup").hidden = true;
  el("answer-mode-chip").textContent = "";
  el("composer-status").textContent = "";
  el("btn-analyze").textContent = "Analyze";
  resetDeliberation();
  renderStepper([]);
  renderCases();
}

async function openCase(id, focusInspector) {
  disconnectStream();
  resetDeliberation();
  const ui = await api("/api/cases/" + encodeURIComponent(id));
  state.ui = ui;
  state.caseId = id;
  localStorage.setItem("ee.case", id);
  state.running = ui.meta.status === "running" || ui.meta.status === "needs-confirmation";
  el("case-title").textContent = ui.meta.title;
  el("prompt-input").value = ui.prompt;
  renderAll();
  renderCases();
  el("btn-analyze").textContent = state.running ? "Stop" : "Analyze";
  if (ui.meta.status === "running") connectStream(id);
  if (focusInspector && state.inspector) setTab("overview");
}

function analyze() {
  const prompt = el("prompt-input").value.trim();
  if (!prompt) { toast("Enter a request first."); return; }
  if (state.running) { stopCase(); return; }
  disconnectStream();
  state.running = true;
  el("btn-analyze").textContent = "Stop";
  el("composer-status").textContent = "starting…";
  const overrides = {
    mode: state.mode,
    analysis_depth: el("opt-depth").value,
    checkpoint_policy: el("opt-checkpoint").value,
    continuation_default: el("opt-continuation").value,
    show_trace: el("opt-trace").value === "on",
  };
  post("/api/cases", { prompt, overrides })
    .then((data) => { state.caseId = data.id; localStorage.setItem("ee.case", data.id); connectStream(data.id); })
    .catch((err) => { toast("Start failed: " + err.message); state.running = false; el("btn-analyze").textContent = "Analyze"; });
}

function sendFollowup() {
  const prompt = el("followup-input").value.trim();
  if (!prompt || !state.caseId) { toast("Type a follow-up first."); return; }
  const mode = el("followup-mode").value;
  state.running = true;
  el("composer-status").textContent = "continuing…";
  post(`/api/cases/${encodeURIComponent(state.caseId)}/continue`, { prompt, overrides: mode ? { mode: state.mode, checkpoint_policy: "assumptions" } : { mode: state.mode } })
    .then((data) => {
      el("followup-input").value = "";
      state.caseId = data.id;
      localStorage.setItem("ee.case", data.id);
      connectStream(data.id);
    })
    .catch((err) => { toast("Continuation failed: " + err.message); state.running = false; });
}

async function stopCase() {
  if (!state.caseId) return;
  await post(`/api/cases/${encodeURIComponent(state.caseId)}/stop`, {});
  el("composer-status").textContent = "stopping…";
}

async function sendCheckpoint(action, extra) {
  if (!state.caseId) return;
  const body = Object.assign({ action }, extra || {});
  try {
    await post(`/api/cases/${encodeURIComponent(state.caseId)}/checkpoint`, body);
    el("checkpoint-card").hidden = true;
    el("composer-status").textContent = action === "confirm" ? "confirmed — mapping the ontology…" : action === "force-original" ? "forced framing — continuing with warning" : "continuing…";
  } catch (err) { toast("Checkpoint not open: " + err.message); }
}

function connectStream(id) {
  disconnectStream();
  const source = new EventSource(`/api/cases/${encodeURIComponent(id)}/events`);
  state.stream = source;
  source.addEventListener("message", (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch (err) { return; }
    handleEvent(data);
  });
  source.onerror = () => { source.close(); if (state.stream === source) state.stream = null; };
}

function disconnectStream() {
  if (state.stream) { state.stream.close(); state.stream = null; }
}

function handleEvent(data) {
  if (data.type === "snapshot" && data.ui) {
    state.ui = data.ui;
    state.running = ["running", "needs-confirmation"].includes(data.ui.meta.status);
    el("case-title").textContent = data.ui.meta.title;
    renderAll();
    return;
  }
  if (data.type === "step") {
    if (state.ui) {
      const entry = (state.ui.steps || []).find((s) => s.name === data.name);
      if (entry) { entry.status = data.status; entry.data = Object.assign(entry.data || {}, data.data || {}); }
      else state.ui.steps.push({ name: data.name, status: data.status, data: data.data || {} });
      renderStepper(state.ui.steps);
    }
    if (data.status === "running") el("composer-status").textContent = `${STEP_LABEL[data.name]} running…`;
    return;
  }
  if (data.type === "triage") {
    state.ui.triage = buildTriageView(data.triage);
    renderTriage();
    el("composer-status").textContent = "triage complete";
    return;
  }
  if (data.type === "checkpoint") {
    state.checkpoint = data.payload;
    renderCheckpoint(data.payload);
    el("composer-status").textContent = "waiting for your decision at the checkpoint";
    el("btn-analyze").textContent = "Review Assumptions";
    return;
  }
  if (data.type === "checkpoint-resolved") {
    el("checkpoint-card").hidden = true;
    state.checkpoint = null;
    return;
  }
  if (data.type === "note") { toast(data.text); return; }
  if (data.type === "delta") {
    appendDelib(data.text || "");
    const tokens = data.tokens || Math.round(state.delib.text.length / 4);
    const budget = data.budget_tokens ? " / " + data.budget_tokens : "";
    el("delib-meta").textContent = `${state.delib.text.length} characters · ~${tokens}${budget} tokens`;
    el("composer-status").textContent = `STATUS: GENERATING · Writer deliberating · TOKENS: ${tokens}${budget}`;
    return;
  }
  if (data.type === "final") {
    state.running = false;
    state.delib.streaming = false;
    el("btn-analyze").textContent = "Analyze";
    el("composer-status").textContent = "complete";
    setDelibOpen(false);
    loadCase(data.case_id, 5);
    return;
  }
  if (data.type === "error") {
    state.running = false;
    el("btn-analyze").textContent = "Analyze";
    el("composer-status").textContent = "failed";
    const banner = el("offline-banner");
    banner.hidden = false;
    clear(banner);
    banner.append(
      h("span", {}, (data.kind === "role-failure" ? "One role did not finish. " : "Analysis partially completed. ") + "The final answer may be incomplete. " + (data.text || "") + (data.hint ? " " + data.hint : "")),
      h("button", { class: "ghost-btn", onclick: () => retryCase() }, "Retry step"),
      h("button", { class: "ghost-btn", onclick: () => { setTab("raw"); } }, "Show raw output"),
      h("button", { class: "ghost-btn", onclick: () => { banner.hidden = true; } }, "Continue with warning")
    );
    return;
  }
  if (data.type === "cancelled") {
    state.running = false;
    el("btn-analyze").textContent = "Analyze";
    el("composer-status").textContent = "stopped";
    return;
  }
  if (data.type === "done") { disconnectStream(); refreshCases(); }
}

function loadCase(id, tries) {
  return api("/api/cases/" + encodeURIComponent(id))
    .then((ui) => { state.ui = ui; renderAll(); renderCases(); })
    .catch(async (err) => {
      if (tries > 0) {
        await new Promise((r) => setTimeout(r, 500));
        return loadCase(id, tries - 1);
      }
      toast("Case not stored yet: " + err.message);
    });
}

function setDelibOpen(open) {
  state.delib.expanded = !!open;
  el("delib-body").hidden = !open;
  el("btn-delib").textContent = open ? "Hide thinking" : "Show thinking";
  el("btn-delib").setAttribute("aria-expanded", String(!!open));
}

function toggleDeliberation() {
  state.delib.collapsedByUser = state.delib.expanded;
  setDelibOpen(!state.delib.expanded);
}

function appendDelib(text) {
  state.delib.text += text;
  state.delib.streaming = true;
  el("deliberation").hidden = false;
  el("deliberation").classList.add("streaming");
  if (!state.delib.collapsedByUser) setDelibOpen(true);
  const body = el("delib-body");
  body.appendChild(document.createTextNode(text));
  body.scrollTop = body.scrollHeight;
}

function resetDeliberation() {
  state.delib = { text: "", expanded: false, streaming: false, collapsedByUser: false };
  clear(el("delib-body"));
  el("delib-body").hidden = true;
  el("deliberation").hidden = true;
  el("deliberation").classList.remove("streaming");
  el("delib-meta").textContent = "idle";
  el("btn-delib").textContent = "Show thinking";
  el("btn-delib").setAttribute("aria-expanded", "false");
}

function renderDeliberation() {
  const d = (state.ui && state.ui.deliberation) || {};
  const text = d.text || state.delib.text || "";
  if (!text) { el("deliberation").hidden = true; return; }
  el("deliberation").hidden = false;
  el("deliberation").classList.remove("streaming");
  if (!state.delib.streaming) {
    state.delib.text = text;
    el("delib-body").textContent = text;
    setDelibOpen(false);
  }
  const chars = d.characters || text.length;
  const tokens = d.estimated_tokens || Math.round(text.length / 4);
  el("delib-meta").textContent = `${chars} characters · ~${tokens} tokens · ${d.lines || 0} lines`;
}

function buildTriageView(route) {
  const depth = (route.DEPTH_TRIGGER || "NONE");
  return {
    route: route.ROUTE, route_label: routeLabel(route.ROUTE),
    reason: route.REASON, complexity: route.COMPLEXITY, risk: route.RISK_LEVEL,
    depth_trigger: depth, depth_active: depth && depth !== "NONE",
    termination_signal: route.TERMINATION_SIGNAL, source: route._source, forced: !!route._forced,
    is_factual: route.ROUTE === "FACTUAL", refused: !!route._refused && !route._grounded,
    grounded_factual: !!route._grounded,
  };
}

function renderAll() {
  const ui = state.ui;
  if (!ui) return;
  renderStepper(ui.steps);
  renderTriage();
  renderWarnings();
  renderDeliberation();
  renderAnswer();
  renderEvidence();
  renderFollowup();
  renderInspector();
  if (ui.checkpoint && ui.checkpoint.open && ui.checkpoint.payload) renderCheckpoint(ui.checkpoint.payload);
  else el("checkpoint-card").hidden = true;
}

function renderStepper(steps) {
  const list = el("stepper");
  clear(list);
  const byName = {};
  (steps || []).forEach((s) => { byName[s.name] = s; });
  STEPS.forEach((name) => {
    const step = byName[name];
    const status = step ? step.status : "pending";
    const cls = { pending: "", running: "step-running", waiting: "step-waiting", complete: "step-complete", failed: "step-failed", overridden: "step-overridden" }[status] || "";
    const glyph = { pending: "", running: "…", waiting: "⚠", complete: "✓", failed: "✕", overridden: "↺" }[status] || "";
    const label = `${STEP_LABEL[name]} ${glyph}`.trim();
    const btn = h("li", {
      class: "step " + cls, role: "button", tabindex: "0",
      title: step && step.data ? JSON.stringify(step.data).slice(0, 220) : "not started",
      "aria-label": `${STEP_LABEL[name]}: ${status}`,
      onclick: () => { if (status !== "pending") { setTab(STEP_TAB[name]); openInspector(); } },
      onkeydown: (e) => { if (e.key === "Enter" && status !== "pending") { setTab(STEP_TAB[name]); openInspector(); } },
    }, [h("span", { class: "sdot", "aria-hidden": "true" }), label]);
    list.appendChild(btn);
  });
  const offline = state.health && !state.health.health.online;
  el("stepper-card").hidden = state.settings.show_pipeline_stepper === false;
  if (offline) el("stepper").setAttribute("aria-disabled", "true");
}

function renderTriage() {
  const ui = state.ui;
  const triage = ui.triage || buildTriageView({});
  const card = el("triage-card");
  card.hidden = false;
  clear(el("triage-body"));
  clear(el("triage-actions"));
  const routeChip = el("triage-route-chip");
  routeChip.textContent = "Route: " + (triage.route_label || routeLabel(triage.route));
  routeChip.className = "chip chip-" + routeTone(triage.route);
  const rows = [
    ["Reason", triage.reason || "-"],
    ["Complexity", triage.complexity || "-"],
    ["Risk", triage.risk || "-"],
    ["Epistemic depth trigger", triage.depth_active ? triage.depth_trigger : "None"],
    ["Termination signal", triage.termination_signal || "NONE"],
    ["Source", triage.source || "-"],
  ];
  if (triage.forced) rows.push(["Mode", "Engine forced by the interface"]);
  if (triage.is_factual) rows.push(["Knowledge boundary", triage.grounded_factual ? "Answered from the local index" : "Refused: the answer is outside the parametric boundary"]);
  rows.forEach(([k, v]) => el("triage-body").append(h("dt", {}, k), h("dd", {}, String(v))));
  if (triage.depth_active) {
    el("triage-body").append(h("dt", {}, "Depth note"), h("dd", {}, "Deep epistemic frame: a direct answer cannot settle this frame."));
  }
  if (triage.route === "SIMPLE") {
    el("triage-actions").append(
      h("button", { class: "ghost-btn", onclick: () => el("answer").scrollIntoView({ behavior: "smooth" }) }, "Show answer"),
      h("button", { class: "ghost-btn", onclick: () => forceEngine() }, "Force Engine analysis")
    );
  }
  if (triage.is_factual) {
    el("triage-actions").append(
      h("button", { class: "ghost-btn", onclick: () => { showView("settings"); renderSettings(); } }, "Add a source document"),
      h("button", { class: "ghost-btn", onclick: () => forceEngine() }, "Force Engine analysis")
    );
  }
}

function forceEngine() {
  const prompt = el("prompt-input").value.trim();
  if (!prompt) return;
  disconnectStream();
  state.running = true;
  post("/api/cases", { prompt, overrides: { mode: "engine" } })
    .then((data) => { state.caseId = data.id; localStorage.setItem("ee.case", data.id); connectStream(data.id); })
    .catch((err) => toast("Force engine failed: " + err.message));
}

function renderCheckpoint(payload) {
  const card = el("checkpoint-card");
  card.hidden = false;
  el("checkpoint-neutral").textContent = "The system will analyze: “" + (payload.neutral_question || "") + "”";
  clear(el("checkpoint-assumptions"));
  (payload.assumptions || []).forEach((a) => el("checkpoint-assumptions").appendChild(h("span", { class: "chip-x" }, a)));
  clear(el("checkpoint-meta"));
  const rows = [
    ["Epistemic stance", payload.epistemic_stance],
    ["Causal attribution", (payload.causal_attributions || []).join(" ")],
    ["Modal boundaries", (payload.modal_boundaries || []).join(" ")],
    ["Excluded options", (payload.excluded_options || []).join(" ")],
    ["Requested register", payload.requested_register],
    ["Requested task shape", payload.requested_task_shape],
    ["Fallacy flags", (payload.fallacy_flags || []).join(" ")],
    ["Termination signal", payload.termination_signal],
    ["Depth trigger", payload.depth_trigger && payload.depth_trigger !== "NONE" ? payload.depth_trigger : "None"],
  ].filter((r) => r[1]);
  rows.forEach(([k, v]) => el("checkpoint-meta").append(h("dt", {}, k), h("dd", {}, v)));
  clear(el("checkpoint-loaded"));
  (payload.loaded_terms || []).forEach((term) => {
    const issue = (state.ui.trace.vocabulary.find((row) => row.term === term) || {}).issue || "Carries an unverified commitment.";
    el("checkpoint-loaded").appendChild(h("span", { class: "chip-x", tabindex: "0", title: issue, "aria-label": `${term}: ${issue}` }, term));
  });
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  wireCheckpointButtons();
}

function wireCheckpointButtons() {
  el("cp-confirm").onclick = () => sendCheckpoint("confirm");
  el("cp-force").onclick = () => sendCheckpoint("force-original", { mode: "user-forced", note: "User requested continuation with the original frame despite detected assumptions." });
  el("cp-skip").onclick = () => { saveSettings({ checkpoint_policy: "never" }); sendCheckpoint("auto-continue"); };
  el("cp-cancel").onclick = () => stopCase();
  el("cp-edit").onclick = () => {
    el("cp-edit-area").hidden = false;
    el("cp-edit-actions").hidden = false;
    const lines = (state.checkpoint.assumptions || []).join("\n");
    el("cp-edit-area").value = lines;
    el("cp-edit-area").focus();
  };
  el("cp-edit-cancel").onclick = () => { el("cp-edit-area").hidden = true; el("cp-edit-actions").hidden = true; };
  el("cp-edit-save").onclick = () => {
    const items = el("cp-edit-area").value.split("\n").map((s) => s.trim()).filter(Boolean);
    el("cp-edit-area").hidden = true;
    el("cp-edit-actions").hidden = true;
    sendCheckpoint("edit", { edited_assumptions: items, note: "user edited assumptions" });
  };
}

function renderWarnings() {
  const box = el("warning-center");
  clear(box);
  box.appendChild(h("h2", { class: "visually-hidden" }, "Warnings"));
  const show = state.settings.warnings || {};
  (state.ui.warnings || []).forEach((w) => {
    if (w.type !== "termination" && show[w.type] === false) return;
    const actions = {
      "add-evidence": () => { el("followup").hidden = false; el("followup-input").value = "New evidence: "; el("followup-input").focus(); },
      reframe: () => { el("prompt-input").focus(); },
      "continue-anyway": () => { el("followup-mode").value = "hypothetical"; el("followup").hidden = false; el("followup-input").value = "Continue anyway with the requested framing."; sendFollowup(); },
      dismiss: () => card.remove(),
    };
    const card = h("div", { class: "warning sev-" + (w.severity === "soft" ? "soft" : w.severity === "medium" ? "medium" : "hard"), role: "status" }, [
      h("div", { class: "w-title" }, [h("span", { class: "tag" }, w.type), w.title]),
      h("div", { class: "w-text" }, w.text),
      h("div", { class: "w-actions" }, (w.actions || []).map((a) => h("button", { class: "ghost-btn", onclick: actions[a] }, labelAction(a)))),
    ]);
    box.appendChild(card);
  });
}
const labelAction = (a) => ({ "add-evidence": "Add new evidence", reframe: "Reframe question", "continue-anyway": "Continue anyway", dismiss: "Dismiss" }[a] || a);

function renderAnswer() {
  const answer = state.ui.answer || {};
  const body = el("answer-body");
  clear(body);
  const annotations = answer.annotations || [];
  if (answer.banner && answer.banner.text) {
    body.appendChild(h("div", { class: "answer-banner" }, annotate(answer.banner.text, annotations)));
  }
  (answer.sections || []).forEach((section) => {
    const node = h("div", { class: "answer-section" }, [h("h3", {}, section.title)]);
    if (section.items) node.appendChild(h("ul", {}, section.items.map((i) => h("li", {}, annotate(i, annotations)))));
    else node.appendChild(h("div", { class: "txt" }, annotate(section.text, annotations)));
    body.appendChild(node);
  });
  const footer = el("answer-footer");
  clear(footer);
  const f = answer.footer || {};
  footer.append(
    chip(`Tier: ${f.tier || "-"}`, f.tier === "Established Consensus" ? "green" : (f.tier || "").startsWith("Architecturally") || f.tier === "Highly Contested" ? "amber" : "blue"),
    chip(`Anchor modal status: ${f.anchor_modal_status || "unlabelled"}`, f.anchor_modal_status ? "muted" : "amber"),
    chip(`Scope: ${f.scope || "-"}`, "muted"),
    chip(`Stakes: ${f.stakes || "-"}`, (f.stakes || "").startsWith("High") ? "purple" : "muted"),
    chip(`Mode: ${f.mode || "-"}`, ["user-forced", "hypothetical", "speculative", "preference-based"].includes(f.mode) ? "purple" : "blue"),
    h("span", { class: "muted" }, f.label || "")
  );
  el("answer-mode-chip").textContent = f.mode ? `mode: ${f.mode}` : "";
  el("answer-mode-chip").className = "chip " + (["user-forced", "hypothetical", "speculative", "preference-based"].includes(f.mode) ? "chip-purple" : "chip-muted");
  el("followup").hidden = false;
  if (state.ui.continuation && state.ui.continuation.pushback) {
    el("followup-input").placeholder = state.ui.continuation.placeholder_pushback || "Add new evidence or specify a new constraint.";
  } else {
    el("followup-input").placeholder = (state.ui.continuation || {}).placeholder || "Add new evidence, constraints, or ask for a deeper analysis.";
  }
}

function annotate(text, annotations) {
  const container = h("span", {}, text);
  const terms = (annotations || []).filter((a) => a.term && a.term.length > 2);
  if (!terms.length) return container;
  const pattern = new RegExp("\\b(" + terms.map((t) => t.term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\b", "gi");
  const nodes = [];
  const walk = (node) => {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === 3 && pattern.test(child.textContent)) {
        pattern.lastIndex = 0;
        const text2 = child.textContent;
        const frag = document.createDocumentFragment();
        let last = 0;
        let match;
        while ((match = pattern.exec(text2))) {
          frag.appendChild(document.createTextNode(text2.slice(last, match.index)));
          const ann = terms.find((t) => t.term.toLowerCase() === match[1].toLowerCase());
          frag.appendChild(h("span", { class: "mark loaded", title: ann ? ann.note : "", tabindex: "0", "aria-label": ann ? ann.note : "" }, match[1]));
          last = match.index + match[0].length;
        }
        frag.appendChild(document.createTextNode(text2.slice(last)));
        node.replaceChild(frag, child);
      } else if (child.nodeType === 1) walk(child);
    }
  };
  walk(container);
  return container;
}

function renderEvidence() {
  const ev = state.ui.evidence || {};
  const box = el("evidence-body");
  clear(box);
  if (state.settings.show_confidence_meter !== false) {
    const tier = ev.tier || "";
    const active = TIERS.indexOf(tier);
    const tone = active <= 0 ? "" : active >= 3 ? "vlow" : "low";
    box.append(
      h("div", {}, [
        h("div", { class: "muted", style: "font-size:12.5px" }, "Qualitative tier of the central claim (no numbers)"),
        h("div", { class: "meter", role: "img", "aria-label": `Tier: ${tier || "untiered"}` }, TIERS.map((_, i) => h("span", { class: i <= active && active >= 0 ? "on " + tone : "" }))),
        h("div", { class: "meter-legend" }, TIERS.map((t) => h("span", {}, t.split(" ")[0]))),
        h("div", { class: "chip-row" }, [
          chip(`Tier: ${tier || "untiered"}`, active <= 0 ? "green" : active >= 3 ? "amber" : "blue"),
          chip("Intent scope: " + (ev.scope || "-"), "muted"),
          chip("Stakes: " + (ev.stakes || "-"), (ev.stakes || "").startsWith("High") ? "purple" : "muted"),
        ]),
      ])
    );
  }
  if (ev.summary) box.appendChild(h("p", { class: "muted", style: "margin-top:10px" }, ev.summary));
  const grid = h("div", { class: "grid-2", style: "margin-top:12px" }, [
    listCard("Theories with their predictions", ev.theories, "evidence"),
    listCard("Landmark concepts engaged or dismissed", ev.landmarks, "evidence"),
    listCard("Active frontiers", ev.frontiers, "evidence"),
    listCard("Live options", ev.live_options, "modal"),
    listCard("Rejected or weak options", ev.weak_options, "modal"),
    listCard("Key unknowns", ev.key_unknowns, "evidence"),
    listCard("Claims to avoid", ev.claims_to_avoid, "evidence"),
    listCard("Proposed tests", ev.proposed_tests, "evidence"),
    listCard("What would change this answer", ev.what_would_change, "evidence"),
  ]);
  box.appendChild(grid);
  const tiers = ev.tier_table || [];
  if (tiers.length) {
    const table = h("table", { class: "kv", style: "grid-template-columns: 1fr 190px 1fr" }, [
      h("dt", {}, "Claim"), h("dd", { style: "grid-column: span 1" }, "Tier"), h("dd", {}, "One-line justification"),
    ]);
    tiers.forEach((row) => table.append(
      h("dt", {}, row.claim),
      h("dd", { style: "grid-column: span 1" }, row.tier),
      h("dd", {}, row.justification)
    ));
    box.appendChild(h("div", { style: "margin-top:12px" }, [h("h4", {}, "Tier table"), table]));
  }
  box.appendChild(h("div", { class: "setting-note" }, "Purely parametric: no retrieval and no live search. Every claim is tiered, modal-tagged, and bounded by the knowledge cutoff."));
}

function listCard(title, items, tab, limit) {
  const list = (items || []).slice(0, limit || 10);
  const card = h("div", { class: "list-card" }, [
    h("h4", {}, title),
    list.length ? h("ul", {}, list.map((i) => h("li", {}, i))) : h("div", { class: "muted" }, "None"),
  ]);
  if (tab) {
    card.style.cursor = "pointer";
    card.title = "Open the " + TAB_LABEL[tab] + " tab";
    card.tabIndex = 0;
    card.addEventListener("click", () => { openInspector(); setTab(tab); });
    card.addEventListener("keydown", (e) => { if (e.key === "Enter") { openInspector(); setTab(tab); } });
  }
  return card;
}

function renderFollowup() {
  const row = el("followup-chips");
  clear(row);
  ((state.ui.continuation || {}).suggestions || []).forEach((s) => {
    row.appendChild(h("button", {
      class: "chip-btn",
      onclick: () => applySuggestion(s),
    }, s.label));
  });
}

function applySuggestion(s) {
  el("followup").hidden = false;
  const input = el("followup-input");
  if (s.id === "show-trace") { openInspector(); setTab("overview"); return; }
  if (s.id === "show-deliberation") {
    setDelibOpen(true);
    el("deliberation").scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }
  const seeds = {
    "add-evidence": "New evidence: ",
    "challenge-premise": "I challenge this premise: ",
    "expand-options": "Expand the solution space beyond ",
    "compare-theories": "Compare the leading theories and what each predicts: ",
    "continue-anyway": "Continue anyway with the requested framing.",
    "more-concise": "Make the answer more concise, keeping the evidence boundary and the warnings.",
  };
  input.value = seeds[s.id] || "";
  input.focus();
  if (s.mode) el("followup-mode").value = s.mode;
}

function buildTabs() {
  const tabs = el("tabs");
  const panels = el("tab-panels");
  clear(tabs);
  clear(panels);
  TABS.forEach((tab) => {
    const btn = h("button", {
      class: "tab", role: "tab", "aria-selected": String(tab === state.tab), "aria-controls": "panel-" + tab,
      onclick: () => setTab(tab),
      onkeydown: (e) => {
        const idx = TABS.indexOf(tab);
        if (e.key === "ArrowRight") setTab(TABS[(idx + 1) % TABS.length]);
        if (e.key === "ArrowLeft") setTab(TABS[(idx + TABS.length - 1) % TABS.length]);
      },
    }, TAB_LABEL[tab]);
    tabs.appendChild(btn);
    panels.appendChild(h("div", { class: "tab-panel" + (tab === state.tab ? " active" : ""), id: "panel-" + tab, role: "tabpanel", "aria-labelledby": tab }, []));
  });
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.textContent === TAB_LABEL[tab])));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + tab));
  renderInspector();
}
function openInspector() { state.inspector = true; document.querySelector(".shell").classList.remove("inspector-collapsed"); }
function toggleInspector(force) {
  state.inspector = force === undefined ? !state.inspector : force;
  document.querySelector(".shell").classList.toggle("inspector-collapsed", !state.inspector);
}

function renderInspector() {
  buildTabs();
  setTabContent(state.tab);
}

function setTabContent(tab) {
  const panel = el("panel-" + tab);
  if (!panel) return;
  clear(panel);
  const ui = state.ui;
  if (!ui) { panel.appendChild(h("div", { class: "muted" }, "No case open.")); return; }
  const trace = ui.trace || {};
  if (tab === "overview") {
    const o = trace.overview || {};
    panel.appendChild(kvBlock([
      ["Route", o.route], ["Route class", o.route_label], ["Complexity", o.complexity], ["Risk", o.risk],
      ["Depth trigger", o.depth_trigger],
      ["Intent scope", o.intent_scope], ["Stakes", o.stakes],
      ["Central claim tier", o.tier],
      ["Logic anchor", o.logic_anchor],
      ["Overthinking monitor", o.overthinking],
      ["Writer negative prompt", o.negative_prompt],
      ["Surgical section repairs", o.repairs],
      ["Knowledge-boundary refusal", o.refusal],
      ["Continuation count", o.continuation_count], ["Current mode", o.mode],
      ["Checkpoint", o.checkpoint], ["Editor source", o.editor_source],
      ["Ontology packet", o.ontology_packet],
      ["Dialectical engine", o.dialectic],
      ["Deterministic guardrails", o.guardrails],
      ["Raw prompt exposed to Writer", o.raw_prompt_exposed_to_writer ? "Yes" : "No"],
      ["Shared word run with raw prompt", o.shared_word_run],
      ["Final status", o.final_status],
    ]));
    panel.appendChild(h("h3", {}, "Overrides"));
    panel.appendChild(listCard("Register override", [trace.register_override].filter(Boolean)));
    panel.appendChild(listCard("Premise rejections", (trace.editor || {}).PREMISE_WARNINGS, "assumptions"));
    panel.appendChild(listCard("Allowed task shape", [(trace.editor || {}).ALLOWED_TASK_SHAPE].filter(Boolean), "modal"));
    panel.appendChild(listCard("Causal corrections", (trace.editor || {}).CAUSAL_CORRECTIONS, "causal"));
    panel.appendChild(listCard("Modal corrections", (trace.editor || {}).MODAL_CORRECTIONS, "modal"));
  }
  if (tab === "assumptions") {
    panel.appendChild(h("h3", {}, "Neutral question"));
    panel.appendChild(h("div", { class: "neutral" }, ((ui.checkpoint || {}).payload || {}).neutral_question || "-"));
    panel.appendChild(h("h3", {}, "Axioms with status"));
    (trace.assumptions || []).forEach((a) => panel.appendChild(h("div", { class: "chip-group" }, h("span", { class: "chip-x status-" + a.status }, [a.text, " ", h("b", {}, a.status)]))));
    const p = (ui.checkpoint || {}).payload || {};
    panel.appendChild(kvBlock([
      ["Requested register", p.requested_register],
      ["Requested task shape", p.requested_task_shape],
      ["Epistemic stance", p.epistemic_stance],
      ["Fallacy flags", (p.fallacy_flags || []).join(" ")],
      ["Termination signal", p.termination_signal],
    ].filter((r) => r[1])));
    panel.appendChild(h("h3", {}, "Checkpoint decision"));
    panel.appendChild(kvBlock([["Action", ((ui.checkpoint || {}).record || {}).action], ["Auto-approved", String(((ui.checkpoint || {}).record || {}).auto_approved)], ["Mode", ((ui.checkpoint || {}).record || {}).mode], ["Note", ((ui.checkpoint || {}).record || {}).note]].filter((r) => r[1])));
  }
  if (tab === "evidence") {
    const ev = trace.evidence || {};
    if (ev.degraded) panel.appendChild(h("div", { class: "warning sev-medium" }, [
      h("div", { class: "w-title" }, "Ontology pass degraded"),
      h("div", { class: "w-text" }, ev.degraded + " — the packet is a bounded fallback built from the neutral question and the partial ontology map, so no claim can be tiered above 'Conceptually Open'."),
    ]));
    panel.appendChild(kvBlock([
      ["Central claim tier", ev.tier],
      ["Tier reading", ev.tier_label],
      ["Anchor modal status", ev.anchor_modal_status || "unlabelled"],
      ["Intent scope", ev.scope],
      ["Stakes", ev.stakes],
    ].filter((r) => r[1])));
    panel.appendChild(listCard("Summary", [ev.summary].filter(Boolean)));
    panel.appendChild(h("h3", {}, "Formal logic anchor (" + (ev.logic_anchor || "missing") + ")"));
    if ((ev.logic_lines || []).length) {
      (ev.logic_lines || []).forEach((line) => panel.appendChild(h("div", { class: "chip-x" }, line)));
    } else {
      panel.appendChild(h("div", { class: "muted" }, "No LOGICAL_SYLLOGISM and no PSEUDO_CODE: the causal chain is unstated, so the answer carries it as a hypothesis."));
    }
    panel.appendChild(listCard(ev.map_raw > ev.map_used
      ? `Ontology map: ${ev.map_used} distinct entries kept out of ${ev.map_raw}`
      : `Ontology map: ${ev.map_used} theories, landmarks, frontiers, and options`, ev.ontology_map, null, 40));
    panel.appendChild(listCard("Theories: core, requires, excludes, predicts", ev.theories, null, 12));
    panel.appendChild(listCard("Landmark concepts (ENGAGED / DISMISSED)", ev.landmarks, null, 12));
    panel.appendChild(listCard("Active frontiers", ev.frontiers, null, 12));
    panel.appendChild(listCard("Modal claims and their tags", ev.modal_claims, "modal", 12));
    panel.appendChild(listCard("Key unknowns", ev.key_unknowns));
    panel.appendChild(listCard("Claims to avoid", ev.claims_to_avoid));
    panel.appendChild(listCard("Proposed tests", ev.proposed_tests));
    panel.appendChild(listCard("What would change this answer", ev.what_would_change));
    panel.appendChild(listCard("Theory objections (symmetry input)", ev.theory_objections, "critic", 12));
  }
  if (tab === "dialectic") {
    const d = trace.dialectic || {};
    if (d.degraded) panel.appendChild(h("div", { class: "warning sev-medium" }, [
      h("div", { class: "w-title" }, "Dialectical engine degraded"),
      h("div", { class: "w-text" }, d.degraded + " — no antithesis was generated, so the thesis was never negated and the tension is recorded as UNRESOLVED."),
    ]));
    panel.appendChild(kvBlock([
      ["Tension status", d.tension_status || "UNRESOLVED"],
      ["Antithesis forced", d.forced_antithesis ? "yes: the generator produced no negation, so the strongest standard negation was used" : "no"],
      ["Thesis/antithesis similarity cap", d.similarity],
    ].filter((r) => r[1] !== undefined && r[1] !== "")));
    panel.appendChild(h("h3", {}, "Thesis"));
    panel.appendChild(h("div", { class: "neutral" }, d.thesis || "-"));
    panel.appendChild(listCard("Thesis axioms", d.thesis_axioms));
    panel.appendChild(h("h3", {}, "Antithesis (negates at least one axiom)"));
    panel.appendChild(h("div", { class: "neutral" }, d.antithesis || "-"));
    panel.appendChild(listCard("Axioms denied", d.antithesis_negates));
    panel.appendChild(h("h3", {}, "Synthesis: what survives, and the tension that remains"));
    panel.appendChild(h("div", { class: "neutral" }, d.synthesis || "-"));
    panel.appendChild(h("div", { class: "setting-note" }, "The tension is never hidden: PRESERVED and UNRESOLVED tensions are carried into the answer."));
  }
  if (tab === "causal") {
    const c = trace.causal || {};
    const g = trace.guardrails || {};
    panel.appendChild(h("h3", {}, "User-proposed cause"));
    (c.user_causes || []).forEach((u) => panel.appendChild(h("div", { class: "chip-x status-unverified" }, u)));
    panel.appendChild(h("h3", {}, "Theories selected for the answer"));
    panel.appendChild(listCard("Selected by the Editor", c.selected, "deliberation"));
    panel.appendChild(h("h3", {}, "Mapped theories and their placement"));
    (c.models || []).forEach((m) => panel.appendChild(h("div", { class: "list-card", style: "margin-bottom:8px" }, [
      h("div", { class: "s-head" }, [h("b", {}, m.name), chip(m.status, m.status.includes("rejected") ? "red" : m.status.includes("live") ? "green" : "amber"), chip(m.placement || "trace", m.placement === "argued in the answer" ? "blue" : "muted")]),
      h("div", { class: "muted", style: "font-size:13px" }, m.note),
    ])));
    panel.appendChild(h("h3", {}, "Theory-indexed feasibility matrix"));
    const rows = g.matrix || [];
    if (!rows.length) panel.appendChild(h("div", { class: "muted" }, "No matrix rows: the guardrail has no options to index."));
    rows.forEach((row) => panel.appendChild(h("div", { class: "list-card", style: "margin-bottom:8px" }, [
      h("div", { class: "s-head" }, [h("b", {}, row.option), chip((row.viable || []).length + " viable", (row.viable || []).length ? "green" : "amber")]),
      h("div", { class: "muted", style: "font-size:13px" }, "Viable under: " + ((row.viable || []).join("; ") || "-")),
      h("div", { class: "muted", style: "font-size:13px" }, "Excluded under: " + ((row.excluded || []).join("; ") || "-")),
      h("div", { class: "muted", style: "font-size:13px" }, "Irrelevant under: " + ((row.irrelevant || []).join("; ") || "-")),
    ])));
    panel.appendChild(h("h3", {}, "Substrate-test compatibility"));
    const flags = g.test_flags || [];
    if (!flags.length) panel.appendChild(h("div", { class: "muted" }, "No proposed tests to flag."));
    flags.forEach((f) => panel.appendChild(h("div", { class: "chip-x " + (f.status === "BLOCK" ? "status-rejected" : f.status === "OK" ? "status-accepted" : "status-unverified"), style: "margin-bottom:6px" }, [
      h("b", {}, f.status), " ", h("span", {}, f.test + (f.reason ? " — " + f.reason : "")),
    ])));
    panel.appendChild(h("h3", {}, "Symmetry filter: universal objections"));
    panel.appendChild(listCard("Flaws shared by the favored theories", (g.universal_objections || []).map((o) => o.acknowledgment || o.flaw)));
    if ((g.violations || []).length) {
      panel.appendChild(h("h3", {}, "Guardrail violations carried into the directive"));
      panel.appendChild(listCard("Violations", g.violations, "critic"));
    }
    panel.appendChild(h("h3", {}, "Archived theories (trace only)"));
    panel.appendChild(listCard("Archived", c.archived));
    panel.appendChild(h("h3", {}, "Editor causal corrections"));
    panel.appendChild(listCard("Corrections", c.corrections));
  }
  if (tab === "modal") {
    const m = trace.modal || {};
    panel.appendChild(h("h3", {}, "User's live options"));
    panel.appendChild(listCard("Treated as live", m.user_options));
    panel.appendChild(h("h3", {}, "System-expanded options"));
    panel.appendChild(listCard("Added by the Editor", m.expanded));
    panel.appendChild(h("h3", {}, "Excluded by the user"));
    panel.appendChild(listCard("Excluded", m.excluded));
    panel.appendChild(listCard("Modal claims and their tags", m.modal_claims, "evidence", 12));
    if ((m.modal_gaps || []).length) panel.appendChild(listCard("Modal tags missing or invalid", m.modal_gaps, "critic"));
    panel.appendChild(kvBlock([["Allowed task shape", m.allowed_task_shape]].filter((r) => r[1])));
  }
  if (tab === "vocabulary") {
    const rows = trace.vocabulary || [];
    const table = h("table", { class: "kv", style: "grid-template-columns: 150px 1fr 1fr" }, []);
    rows.forEach((row) => table.append(
      h("dt", {}, row.term),
      h("dd", { style: "grid-column: span 1" }, row.issue),
      h("dd", {}, row.substitution)
    ));
    panel.appendChild(table);
    panel.appendChild(h("div", { class: "setting-note" }, "The final response uses neutral substitutions unless the user explicitly requests the original terms."));
  }
  if (tab === "termination") {
    const rows = trace.termination || [];
    if (!rows.length) panel.appendChild(h("div", { class: "muted" }, "No termination signals."));
    rows.forEach((row) => panel.appendChild(h("div", { class: "list-card", style: "margin-bottom:8px" }, [
      h("div", { class: "s-head" }, [h("b", {}, row.signal), chip("severity: " + row.severity, row.severity === "medium" ? "amber" : "blue")]),
      h("div", { style: "font-size:13.5px", text: row.reason }),
      h("div", { class: "muted", style: "font-size:12.5px" }, row.action),
      h("div", { class: "muted", style: "font-size:12.5px" }, "User override: " + row.override),
    ])));
    panel.appendChild(h("div", { class: "setting-note" }, "Termination warnings are advisory. They do not block continuation."));
  }
  if (tab === "critic") {
    const c = trace.critic || {};
    panel.appendChild(kvBlock([
      ["Critic mode", c.mode],
      ["Critic result", c.result],
      ["Checklist items", c.checks_total],
      ["Items failed", c.checks_failed],
      ["Sections rewritten", (c.failed_sections || []).join(", ") || "none"],
      ["Violations", (c.violations || []).join(" ") || "None"],
      ["Rewrite required", c.rewrite_required], ["Rewrite instruction", c.rewrite_instruction],
      ["Critic rounds", c.rounds], ["Sections repaired", c.repairs],
    ]));
    if (c.banner) panel.appendChild(h("div", { class: "warning sev-hard" }, [h("div", { class: "w-title" }, "Unresolved after the critic rounds"), h("div", { class: "w-text" }, c.banner)]));
    const checks = c.checks || [];
    if (checks.length) {
      panel.appendChild(h("h3", {}, "Presentation checklist verdicts"));
      checks.forEach((ck) => panel.appendChild(h("div", { class: "chip-x " + (ck.pass ? "status-accepted" : "status-rejected"), style: "margin-bottom:6px" }, [
        h("b", {}, "Check " + ck.id), " ", h("span", {}, ck.name + (ck.section ? " [" + (ck.section_title || ck.section) + "]" : "")),
        ck.note ? h("div", { class: "muted", style: "font-size:12.5px" }, ck.note) : null,
      ])));
    }
    const repairs = trace.repairs || [];
    if (repairs.length) {
      panel.appendChild(h("h3", {}, "Surgical section repair (one section rewritten, the rest untouched)"));
      repairs.forEach((r) => panel.appendChild(h("div", { class: "list-card", style: "margin-bottom:8px" }, [
        h("div", { class: "s-head" }, [h("b", {}, (r.title || r.section) + " · round " + (r.round || 1)), chip("checks " + (r.checks || []).join(", "), "purple")]),
        h("div", { style: "font-size:13.5px" }, r.reason),
        h("details", { class: "raw-item" }, [h("summary", {}, "Before / after"), h("pre", {}, "BEFORE:\n" + r.before + "\n\nAFTER:\n" + r.after)]),
      ])));
    }
    if (c.drift_note) panel.appendChild(h("div", { class: "warning sev-hard" }, [h("div", { class: "w-title" }, "Epistemic drift remaining"), h("div", { class: "w-text" }, c.drift_note)]));
  }
  if (tab === "deliberation") {
    const d = trace.deliberation || {};
    panel.appendChild(kvBlock([
      ["Characters", d.characters],
      ["Estimated tokens", d.estimated_tokens],
      ["Lines", d.lines],
    ]));
    panel.appendChild(h("details", { class: "raw-item", open: true }, [
      h("summary", {}, "Writer private deliberation"),
      h("pre", {}, d.text || state.delib.text || "none"),
    ]));
    if (d.rewrite_text) panel.appendChild(h("details", { class: "raw-item" }, [h("summary", {}, "Rewrite deliberation"), h("pre", {}, d.rewrite_text)]));
    panel.appendChild(h("div", { class: "setting-note" }, "The deliberation stays in the trace. The Answer canvas renders only the Writer's final response."));
  }
  if (tab === "raw") {
    const packets = ui.raw_packets || {};
    Object.entries(packets).forEach(([name, text]) => {
      panel.appendChild(h("details", { class: "raw-item" }, [h("summary", {}, pretty(name)), h("pre", {}, String(text))]));
    });
    const meta = ui.packet_meta || {};
    if (Object.keys(meta).length) {
      panel.appendChild(h("details", { class: "raw-item" }, [h("summary", {}, "call log"), h("pre", {}, JSON.stringify(meta, null, 2))]));
    }
    panel.appendChild(h("div", { class: "setting-note" }, "Prompt distillation: the Writer input contains the Editor directive, the ontology packet, and the dialectical result — not the raw prompt."));
  }
}

function kvBlock(rows) {
  const dl = h("dl", { class: "kv" }, []);
  rows.forEach(([k, v]) => dl.append(h("dt", {}, k), h("dd", {}, String(v === null || v === undefined ? "-" : v))));
  return dl;
}

const SETTING_GROUPS = [
  {
    title: "Model", rows: [
      { kind: "info", label: "Local LLM endpoint", render: () => `<span class="locked">🔒 ${((state.health || {}).health || {}).endpoint || "http://127.0.0.1:8082"}</span>` },
      { kind: "note", text: "This deployment uses one local LLM on port 8082. External LLM APIs are disabled." },
      { kind: "info", label: "Status", render: () => (((state.health || {}).health || {}).online ? "Online" : "Offline") },
      { kind: "info", label: "Latency estimate", render: () => (((state.health || {}).health || {}).latency_ms || "-") + " ms" },
      { kind: "info", label: "Last successful call", render: () => new Date((((state.health || {}).health || {}).last_ok || 0) * 1000).toLocaleTimeString() },
      { kind: "button", label: "Retry connection", action: () => refreshHealth(true) },
    ],
  },
  {
    title: "Pipeline", rows: [
      { kind: "note", text: "Purely parametric: no retrieval, no live search, no external LLM APIs. The Editor is a deterministic Python rule layer, never an LLM." },
      { kind: "select", key: "checkpoint_policy", label: "Require human checkpoint", options: [["always", "Always"], ["assumptions", "Only when assumptions detected"], ["never", "Never"]] },
      { kind: "toggle", key: "use_critic", label: "Enable the presentation Critic (checklist over the final text)" },
      { kind: "toggle", key: "allow_rewrite", label: "Allow one automatic rewrite" },
      { kind: "toggle", key: "logic_anchor", label: "Force the formal logic anchor (LOGICAL_SYLLOGISM / PSEUDO_CODE)" },
      { kind: "toggle", key: "negative_prompt", label: "Penalize the user's biased framing at the API (negative_prompt)" },
      { kind: "toggle", key: "overthinking_monitor", label: "Overthinking monitor: abort the stream past the hesitation limit" },
      { kind: "toggle", key: "section_repair", label: "Surgical repair: rewrite only the sections that failed the checklist" },
      { kind: "info", label: "Hesitation limit", render: () => `${state.settings.overthinking_threshold || 3} markers per ${state.settings.overthinking_window || 500} tokens` },
      { kind: "info", label: "Triage routes", render: () => (((state.health || {}).modules || {}).knowledge_boundary) || "on" },
      { kind: "toggle", key: "show_trace", label: "Show Trace Inspector by default" },
      { kind: "select", key: "analysis_depth", label: "Analysis depth", options: [["light", "Light"], ["standard", "Standard"], ["deep", "Deep"]] },
      {
        kind: "select",
        key: "thinking_reserve",
        label: "Thinking reserve: share of the output budget kept for the answer",
        options: [["0", "None: the model may spend the whole budget thinking"], ["0.25", "25% (recommended)"], ["0.5", "50%"], ["0.7", "70%"]],
      },
      { kind: "info", label: "Output budget per role", render: () => `${state.settings.max_tokens || 16384} tokens` },
      { kind: "info", label: "Structured output enforcement", render: () => ((state.health || {}).modules || {}).structured_output || "on" },
      { kind: "info", label: "Deterministic editor", render: () => ((state.health || {}).modules || {}).deterministic_editor || "on" },
    ],
  },
  {
    title: "Warnings", rows: [
      { kind: "toggle-locked", key: "termination", label: "Show termination warnings", note: "Termination warnings are advisory. They do not block continuation." },
      { kind: "toggle-warn", key: "epistemic", label: "Show epistemic warnings" },
      { kind: "toggle-warn", key: "vocabulary", label: "Show vocabulary warnings" },
      { kind: "toggle-warn", key: "causal", label: "Show causal warnings" },
      { kind: "toggle-warn", key: "modal", label: "Show modal warnings" },
    ],
  },
  {
    title: "Continuation", rows: [
      { kind: "select", key: "continuation_default", label: "Default continuation mode", options: [["evidence-bounded", "Evidence-bounded"], ["hypothetical", "Hypothetical when overridden"], ["user-forced", "User-forced when explicitly requested"]] },
    ],
  },
  {
    title: "Display", rows: [
      { kind: "select", key: "theme", label: "Theme", options: [["system", "System"], ["light", "Light"], ["dark", "Dark"]] },
      { kind: "select", key: "density", label: "Density", options: [["comfortable", "Comfortable"], ["compact", "Compact"]] },
      { kind: "toggle", key: "show_confidence_meter", label: "Show the tier strip" },
      { kind: "toggle", key: "inline_annotations", label: "Show inline annotations" },
      { kind: "toggle", key: "show_pipeline_stepper", label: "Show pipeline stepper" },
    ],
  },
];

function renderSettings() {
  const box = el("settings-body");
  clear(box);
  for (const group of SETTING_GROUPS) {
    const card = h("div", { class: "setting-card" }, [h("h3", {}, group.title)]);
    for (const row of group.rows) {
      if (row.kind === "note") { card.appendChild(h("div", { class: "setting-note" }, row.text)); continue; }
      if (row.kind === "info") card.appendChild(row_(row.label, row.render()));
      if (row.kind === "button") card.appendChild(h("div", { class: "setting-row" }, [h("span", {}, row.label), h("button", { class: "ghost-btn", onclick: row.action }, row.label)]));
      if (row.kind === "toggle") {
        const input = h("input", { type: "checkbox", checked: !!state.settings[row.key], "aria-label": row.label });
        input.addEventListener("change", () => saveSettings({ [row.key]: input.checked }));
        card.appendChild(h("div", { class: "setting-row" }, [h("span", {}, row.label), input]));
      }
      if (row.kind === "toggle-locked") {
        const input = h("input", { type: "checkbox", checked: true, disabled: true, "aria-label": row.label });
        card.appendChild(h("div", { class: "setting-row" }, [h("span", {}, row.label), input]));
        card.appendChild(h("div", { class: "setting-note" }, row.note));
      }
      if (row.kind === "toggle-warn") {
        const input = h("input", { type: "checkbox", checked: (state.settings.warnings || {})[row.key] !== false, "aria-label": row.label });
        input.addEventListener("change", () => saveSettings({ warnings: { [row.key]: input.checked } }));
        card.appendChild(h("div", { class: "setting-row" }, [h("span", {}, row.label), input]));
      }
      if (row.kind === "select") {
        const select = h("select", { "aria-label": row.label }, row.options.map(([value, label]) => h("option", { value, selected: String(state.settings[row.key]) === value }, label)));
        select.addEventListener("change", () => saveSettings({ [row.key]: select.value }));
        card.appendChild(h("div", { class: "setting-row" }, [h("span", {}, row.label), select]));
      }
    }
    box.appendChild(card);
  }
}

function renderStatus() {
  const box = el("status-body");
  clear(box);
  const health = (state.health || {}).health || {};
  const modules = (state.health || {}).modules || {};
  box.appendChild(h("div", { class: "banner", style: health.online ? "background:var(--green-soft);color:var(--green)" : "" }, health.online ? "Local LLM is online. Analysis available." : "Local LLM on port 8082 is offline. The system cannot generate analysis until the model is available."));
  box.appendChild(kvBlock([
    ["Endpoint", health.endpoint],
    ["Model", health.model],
    ["Max context", health.max_context],
    ["Latency", (health.latency_ms || "-") + " ms"],
    ["Last successful call", health.last_ok ? new Date(health.last_ok * 1000).toLocaleString() : "-"],
    ["Error", health.error || "none"],
  ]));
  box.appendChild(h("h2", {}, "Local modules"));
  box.appendChild(kvBlock(Object.entries(modules).map(([k, v]) => [k, String(v)])));
  box.appendChild(h("p", { class: "setting-note" }, "All LLM inference is routed to the single local model on port 8082. No external LLM APIs are used."));
  box.appendChild(h("button", { class: "ghost-btn", onclick: () => refreshHealth(true) }, "Retry connection"));
}

async function renderLibrary() {
  const body = el("library-body");
  clear(body);
  const table = h("div", { class: "grid-2" }, state.cases.map((c) => h("div", { class: "list-card" }, [
    h("h4", {}, c.title),
    h("div", { class: "muted", style: "font-size:12.5px" }, new Date(c.created_at * 1000).toLocaleString()),
    h("div", { class: "chip-row" }, [chip(routeLabel(c.route), routeTone(c.route)), chip(labelStatus(c.status), "muted"), chip(c.mode, "muted")]),
    h("div", { class: "card-actions" }, h("button", { class: "ghost-btn", onclick: () => { showView("workspace"); openCase(c.id, true); } }, "Open")),
  ])));
  body.appendChild(state.cases.length ? table : h("p", { class: "muted" }, "No saved cases yet."));
  const corpus = el("corpus-body");
  clear(corpus);
  try {
    const data = await api("/api/documents");
    if (!data.documents.length) corpus.appendChild(h("p", { class: "muted" }, "Library is empty. The engine is purely parametric: attached documents stay in this local library and are never fed to the model."));
    data.documents.forEach((doc) => corpus.appendChild(h("div", { class: "source-card" }, [
      h("div", { class: "s-head" }, [h("b", {}, doc.name), chip(doc.bytes + " bytes", "muted")]),
      h("div", { class: "card-actions" }, h("button", { class: "ghost-btn", onclick: async () => { await post("/api/documents/delete", { name: doc.name }); renderLibrary(); refreshHealth(true); } }, "Remove")),
    ])));
  } catch (err) { corpus.appendChild(h("p", { class: "muted" }, "Corpus unavailable: " + err.message)); }
}

function showAttach() {
  el("modal-title").textContent = "Attach local documents";
  clear(el("modal-body"));
  el("modal-body").append(
    h("p", { class: "muted" }, "Documents are stored locally under .epistemic/corpus as a personal reference library. The engine runs purely on parametric weights: no retrieval, no live search, nothing from this library reaches the model."),
    h("label", { class: "field" }, [h("span", {}, "Document name"), h("input", { id: "doc-name", value: "notes.txt" })]),
    h("label", { class: "field", style: "margin-top:8px" }, [h("span", {}, "Document text"), h("textarea", { id: "doc-text", rows: "10", placeholder: "Paste baselines, metrics, meeting notes, prior reports…" })]),
    h("div", { class: "card-actions" }, [
      h("button", {
        class: "primary-btn", onclick: async () => {
          const name = el("doc-name").value || "notes.txt";
          const text = el("doc-text").value;
          if (!text.trim()) { toast("Paste document text first."); return; }
          await post("/api/documents", { name, text });
          el("modal").hidden = true;
          toast("Document indexed.");
          refreshHealth(true);
        },
      }, "Save and index"),
      h("button", { class: "ghost-btn", onclick: () => el("modal").hidden = true }, "Cancel"),
    ])
  );
  el("modal").hidden = false;
}

function showHelp() {
  el("modal-title").textContent = "How this tool works";
  clear(el("modal-body"));
  el("modal-body").append(
    h("p", {}, "The Epistemic Engine answers while visibly correcting frame capture, false certainty, weak evidence standards, unverified causal claims, narrow solution spaces, loaded vocabulary, and forced closure."),
    h("ul", {}, [
      h("li", {}, "One local LLM on port 8082 performs the isolated roles: Triage, Skeptic, Ontology Mapper (map then distill), Dialectical Engine, Editor, Writer, Presentation Critic. No external LLM APIs, no retrieval, no live search."),
      h("li", {}, "The raw prompt is shown only to Triage and Skeptic. The Ontology pass sees the neutral question and the decomposed sub-questions; the Writer sees the Editor directive, not the raw prompt. The raw wording is used only as the API negative_prompt. This quarantines the user's frame."),
      h("li", {}, "The Editor is deterministic Python: guardrails, the theory-indexed feasibility matrix, substrate-test flags, and the symmetry filter. The LLM is never asked to make those calls."),
      h("li", {}, "Termination detection is a warning, never a block. The system continues in a labeled mode: evidence-bounded, user-forced, hypothetical, speculative, preference-based."),
      h("li", {}, "Pushback is not evidence. Repeating a demand without new information produces a repetition warning and keeps the same evidence boundary."),
      h("li", {}, "Modes: Auto decides simple vs engine; Fast answers directly unless the request is loaded; Engine always runs the full pipeline."),
    ]),
    h("p", { class: "muted" }, "Keyboard: Ctrl/Cmd+Enter submits. Stepper chips and tabs are keyboard navigable.")
  );
  el("modal").hidden = false;
}

function showView(view) {
  state.view = view;
  el("view-settings").hidden = view !== "settings";
  el("view-library").hidden = view !== "library";
  el("view-status").hidden = view !== "status";
  ["composer", "stepper-card", "triage-card", "checkpoint-card", "warning-center", "deliberation", "answer", "evidence", "followup"].forEach((id) => {
    const node = el(id);
    if (view === "workspace") {
      if (id === "triage-card") { node.hidden = !(state.ui && state.ui.triage); return; }
      if (id === "checkpoint-card") { node.hidden = !(state.ui && state.ui.checkpoint && state.ui.checkpoint.open); return; }
      if (id === "deliberation") { node.hidden = !(state.ui && (state.ui.deliberation || {}).text); return; }
      node.hidden = false;
    } else node.hidden = true;
  });
  if (view === "workspace" && !state.ui) { el("triage-card").hidden = true; el("checkpoint-card").hidden = true; }
  ["nav-cases", "nav-library", "nav-settings", "nav-status"].forEach((id) => el(id).removeAttribute("aria-current"));
  el({ workspace: "nav-cases", library: "nav-library", settings: "nav-settings", status: "nav-status" }[view]).setAttribute("aria-current", "page");
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("beforeend", `<div class="toast show">UI failed to start: ${esc(err.message)}</div>`);
});
