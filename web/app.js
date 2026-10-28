"use strict";

const STEP_ORDER = [
  "1.secure_frontend", "1b.data_frontend", "2.s2a_pass1", "3.hybrid_retrieval",
  "4.context_assembly", "5.pass2_generation", "6.epistemic_backstop", "7.case_state_update",
];
const STEP_LABEL = {
  "1.secure_frontend": "Secure Front-End",
  "1b.data_frontend": "Data Front-End (inline)",
  "2.s2a_pass1": "S2A Pass 1",
  "3.hybrid_retrieval": "Hybrid Retrieval",
  "4.context_assembly": "Context Assembly",
  "5.pass2_generation": "Pass 2 Generation",
  "6.epistemic_backstop": "Epistemic Backstop",
  "7.case_state_update": "Case State Update",
};
const TAGS = ["FACT", "PREDICTION", "HYPOTHETICAL", "POSSIBILITY", "CONSTRAINT", "AUTHOR_CLAIM"];
const TAG_NOTE = {
  FACT: "states a fact", PREDICTION: "predicts a future outcome",
  HYPOTHETICAL: "reasons hypothetically", POSSIBILITY: "flags a possibility",
  CONSTRAINT: "states a limit on the claim", AUTHOR_CLAIM: "reports the author's claim, not a fact",
};

const CRITICAL_MARKS = ["BLOCKED", "QUARANTINED", "SCRUBBED", "UNVERIFIED",
  "QUOTE MISMATCH", "ERROR", "FALLBACK", "DEGRADED", "INCONSISTENT", "INJECTION",
  "NO HITS", "TRUNCATED", "OVER BUDGET", "DIALECTIC", "INCOMPLETE", "CRITIC",
  "CONTRADICT", "UNSUPPORTED", "CONSTRAINT RULE"];
const isCriticalFlag = (f) =>
  CRITICAL_MARKS.some((m) => String(f).toUpperCase().includes(m));
const splitFlags = (flags) => {
  const list = (flags || []).filter((f) => String(f).includes(":"));
  return { critical: list.filter(isCriticalFlag), soft: list.filter((f) => !isCriticalFlag(f)) };
};

function renderBottomLine(text) {
  if (!text) return "";
  return `<div class="bottom-line"><div class="bl-label">Bottom line</div>` +
    `<div class="bl-body">${renderMarkdown(text)}</div></div>`;
}

function renderDialectic(d) {
  if (!d || (!d.thesis && !d.antithesis && !d.synthesis)) return "";
  const block = (label, text, cls) =>
    `<div class="d-block ${cls}"><div class="d-label">${label}</div>` +
    `<div class="d-body">${renderMarkdown(text || "(the model left this empty)")}</div></div>`;
  return `<div class="dialectic">` +
    block("Thesis", d.thesis, "d-thesis") +
    block("Antithesis", d.antithesis, "d-antithesis") +
    block("Synthesis", d.synthesis, "d-synthesis") + `</div>`;
}

function rowsTable(rows, cols) {
  if (!rows || !rows.length) return `<div class="v muted">none</div>`;
  const head = cols.map(([k]) => `<th>${esc(k)}</th>`).join("");
  const body = rows.map((r) => `<tr>${cols.map(([, get]) => {
    const v = typeof get === "function" ? get(r) : r[get];
    return `<td>${esc(Array.isArray(v) ? v.join(", ") : (v == null ? "" : v))}</td>`;
  }).join("")}</tr>`).join("");
  return `<table class="mini-table"><tr>${head}</tr>${body}</table>`;
}

function renderReply(text) {
  const parts = String(text).split(
    /\[(FACT|PREDICTION|HYPOTHETICAL|POSSIBILITY|CONSTRAINT|AUTHOR_CLAIM)\]/g);
  const secs = [];
  let cur = { tag: null, text: parts.shift() || "" };
  for (let i = 0; i < parts.length; i += 2) {
    if (cur.text.trim()) secs.push(cur);
    cur = { tag: parts[i], text: parts[i + 1] || "" };
  }
  secs.push(cur);
  const many = secs.length > 1;
  return secs.map((sec) => {
    const head = sec.tag
      ? `<div class="sec-head"><span class="tag-badge t-${sec.tag}" title="${TAG_NOTE[sec.tag]}">${sec.tag}</span></div>`
      : (many ? `<div class="sec-head"><span class="sec-label">Answer</span></div>` : "");
    return `<div class="sec ${sec.tag ? "sec-" + sec.tag : ""}">${head}` +
      `<div class="sec-body">${renderMarkdown(sec.text)}</div></div>`;
  }).join("");
}

function renderAlerts(flags) {
  const box = el("div", "alerts");
  flags.forEach((f) => {
    const head = (f.split(":").slice(1).join(":").trim().split(/[\s(]/)[0] || "FLAG")
      .toUpperCase().replace(/[^A-Z_ ]/g, "");
    const card = el("div", "alert" +
      (/BLOCKED|ERROR|QUARANTINED|INJECTION/.test(f) ? " alert-bad" : ""));
    card.innerHTML = `<div class="alert-title">${esc(head || "FLAG")}</div>` +
      `<div class="alert-detail">${esc(f)}</div>`;
    box.appendChild(card);
  });
  return box;
}

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

let citationMap = new Map();
let fullContext = "";
let busy = false;
let status = null;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch (e) { data = { detail: text }; }
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
  return data;
}

function txt(v) {
  if (v == null) return "";
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.map(txt).filter(Boolean).join("; ");
  if (typeof v === "object") {
    const probes = ["response", "note", "summary", "text", "answer", "status"];
    for (const k of probes) if (v[k]) return txt(v[k]);
    return Object.entries(v).map(([k, x]) => `${k}: ${txt(x)}`).filter((s) => s.length > 3).join("; ");
  }
  return String(v);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------------ status */
async function loadStatus() {
  status = await api("/api/status");
  const p = status.provider_health || {};
  $("modelPill").textContent = `Model: ${status.settings.model}` +
    (p.model_present === false ? " (NOT LOADED)" : "");
  $("densePill").textContent = `dense: ${status.index.embed_last_mode}`;
  $("contextPill").textContent = "Context: 0/" + Math.round(status.settings.context_window / 1024) + "k";
  const dead = p.ok === false || p.model_present === false;
  if (dead) {
    $("modelBanner").classList.remove("hidden");
    $("modelBanner").innerHTML =
      `<strong>The configured local model is not reachable.</strong> ` +
      `provider=<code>${esc(status.settings.provider)}</code> ` +
      `model=<code>${esc(status.settings.model)}</code> at ` +
      `<code>${esc(status.settings.base_url)}</code>. ` +
      `${esc(p.detail || (p.model_present === false ? "model not served on this endpoint" : ""))} ` +
      `Available: <code>${esc((p.available || []).join(", ") || "none")}</code>. ` +
      `Pass 1 falls back deterministically and Pass 2 fails loudly.`;
  } else {
    $("modelBanner").classList.add("hidden");
  }
  await loadDocs();
}

async function loadDocs() {
  const { documents } = await api("/api/documents");
  const list = $("docList");
  list.innerHTML = "";
  if (!documents.length) {
    list.appendChild(el("em", null, "no documents attached — answers are parametric only"));
    return;
  }
  documents.forEach((d) => {
    const chip = el("span", "docchip");
    chip.innerHTML = `<b>${esc(d.doc_id)}</b> ${esc((d.title || "").slice(0, 40))} ` +
      `<span class="muted">· ${d.n_leaves} leaves · ${d.n_sections} sections</span>` +
      `<span class="x" title="delete">✕</span>`;
    chip.onclick = (ev) => {
      if (ev.target.classList.contains("x")) {
        api(`/api/documents/${d.doc_id}`, { method: "DELETE" }).then(loadDocs);
      }
    };
    list.appendChild(chip);
  });
}


function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  $("themeBtn").textContent = mode === "day" ? "\u2600 Day" : "\u263e Night";
  try { localStorage.setItem("oee-theme", mode); } catch (e) { /* private mode */ }
}
$("themeBtn").onclick = () =>
  applyTheme(document.documentElement.dataset.theme === "day" ? "night" : "day");
let savedTheme = "night";
try { savedTheme = localStorage.getItem("oee-theme") || "night"; } catch (e) { /* noop */ }
applyTheme(savedTheme);
function showView(which) {
  const chat = which === "chat";
  $("viewChat").classList.toggle("active", chat);
  $("viewTrans").classList.toggle("active", !chat);
  $("tabChat").classList.toggle("active", chat);
  $("tabTrans").classList.toggle("active", !chat);
}
$("tabChat").onclick = () => showView("chat");
$("tabTrans").onclick = () => showView("trans");

/* ------------------------------------------------------------------ panes */
document.querySelectorAll(".collapse").forEach((b) => {
  b.onclick = () => $(b.dataset.pane).classList.toggle("collapsed");
});
document.querySelectorAll("[data-close]").forEach((b) => {
  b.onclick = () => $(b.dataset.close).classList.add("hidden");
});
document.querySelectorAll(".modal").forEach((m) => {
  m.addEventListener("click", (e) => { if (e.target === m) m.classList.add("hidden"); });
});

/* ------------------------------------------------------------ left pane fill */
function setBlock(id, text, cls) {
  const block = $(id);
  const body = block.querySelector(".block-body");
  body.textContent = text === "" ? "…" : text;
  body.classList.toggle("empty", false);
  body.classList.remove("pending");
  block.classList.toggle("flagged", !!cls);
}

function resetTransparency(rawText, dataPresent) {
  Object.keys(liveSteps).forEach((k) => delete liveSteps[k]);
  $("blkContext").querySelector("#viewFullBtn").classList.add("hidden");
  fullContext = "";
  setBlock("blkRaw", rawText);
  $("blkData").classList.toggle("hidden", !dataPresent);
  ["blkSecure", "blkPass1Sys", "blkPass1Out", "blkPass2Sys", "blkContext"].forEach((id) => {
    const body = $(id).querySelector(".block-body");
    body.textContent = "running…";
    body.classList.add("pending");
    $(id).classList.remove("flagged");
  });
  if (dataPresent) {
    const body = $("blkData").querySelector(".block-body");
    body.textContent = "running…";
    body.classList.add("pending");
  }
}

const LEFT_FILLERS = {
  "1.secure_frontend": (t) => {
    const sec = t["1.secure_frontend"];
    setBlock("blkSecure",
      (sec.flags.length ? sec.flags.join("\n") + "\n\n" : "no control tokens found\n") +
      `clean text sent to Pass 1 (${sec.output_chars} chars):\n` + sec.output_preview);
  },
  "1b.data_frontend": (t) => {
    const d = t["1b.data_frontend"];
    setBlock("blkData",
      (d.flags.length ? d.flags.join("\n") + "\n\n" : "no injection/control-token flags\n") +
      `inline data passed to Pass 2 as inert <user_data> (${d.output_chars} chars of ` +
      `${d.input_chars}; not sent to S2A, not ingested):\n` + (d.output_preview || ""),
      d.flags.length);
  },
  "2.s2a_pass1": (t) => {
    const p1 = t["2.s2a_pass1"];
    setBlock("blkPass1Sys",
      p1.system_prompt + "\n\n[user message as sent]\n" + p1.user_prompt, p1.flags.length);
    setBlock("blkPass1Out",
      `sanitized: "${p1.sanitized_request}"\n` +
      `intent_preserved: ${p1.intent_preserved}\n` +
      `bias_removed: ${p1.bias_removed.length ? p1.bias_removed.join(", ") : "(none reported)"}\n` +
      `deterministic lexicon cross-check: ${p1.deterministic_bias.join(", ") || "(clean)"}\n` +
      (p1.flags.length ? "\nFLAGS:\n" + p1.flags.join("\n") + "\n" : "") +
      `\n[raw model output]\n${p1.raw_output}`, p1.flags.length);
  },
  "4.context_assembly": (t, res) => {
    const asm = t["4.context_assembly"];
    setBlock("blkPass2Sys", asm.system_prompt, asm.flags.length);
    const shown = res.prompts.assembled_context.slice(0, 18000);
    setBlock("blkContext",
      shown + (res.prompts.assembled_context.length > 18000
        ? "\n…[truncated in pane — click View Full]" : ""), asm.flags.length);
    if (asm.tokens > 2000 || res.prompts.assembled_context.length > 18000) {
      $("viewFullBtn").classList.remove("hidden");
    }
  },
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const liveSteps = {};

function liveStep(ev) {
  /* The transparency pane follows the pipeline step by step: stage events
     carry the same payloads the finished trace carries, so prompts, outputs
     and trace cards appear while the turn is still running. */
  liveSteps[ev.step] = Object.assign(liveSteps[ev.step] || {}, ev);
  const step = liveSteps[ev.step];
  const idx = STEP_ORDER.indexOf(ev.step);
  const card = traceCard(step, idx >= 0 ? idx : 0);
  card.id = "step-" + ev.step;
  const existing = document.getElementById("step-" + ev.step);
  if (existing) existing.replaceWith(card); else $("trace").appendChild(card);
  const filler = LEFT_FILLERS[ev.step];
  if (filler) {
    try {
      filler(liveSteps, { prompts: { assembled_context: step.assembled_context || "" } });
    } catch (e) { /* fields the step has not produced yet: filled at the end */ }
  }
  card.scrollIntoView({ block: "nearest" });
}

async function revealPipeline(res) {
  const t = {};
  res.trace.forEach((s) => { t[s.step] = s; });
  fullContext = res.prompts.assembled_context;
  for (let i = 0; i < res.trace.length; i++) {
    const step = res.trace[i];
    const filler = LEFT_FILLERS[step.step];
    if (filler) filler(t, res);
    const card = traceCard(step, i);
    card.id = "step-" + step.step;
    const live = document.getElementById("step-" + step.step);
    if (live) live.replaceWith(card); else $("trace").appendChild(card);
    card.scrollIntoView({ block: "nearest" });
    if (!live) await sleep(90);
  }
}

$("viewFullBtn").onclick = () => {
  const blk = $("blkContext").querySelector(".block-body");
  blk.textContent = fullContext;
  blk.style.maxHeight = "none";
  $("viewFullBtn").classList.add("hidden");
};

/* ------------------------------------------------------------------- trace */
function renderPendingTrace(dataPresent) {
  const box = $("trace");
  box.innerHTML = "";
  STEP_ORDER
    .filter((step) => dataPresent || step !== "1b.data_frontend")
    .forEach((step, i) => {
      const card = el("div", "step pending");
      card.id = "step-" + step;
      card.innerHTML =
        `<div class="step-top"><span class="step-idx">${i + 1}</span>` +
        `<span class="step-name">${STEP_LABEL[step]}</span>` +
        `<span class="step-status s-empty">running</span></div>` +
        `<div class="step-detail">…</div>`;
      box.appendChild(card);
    });
}

function traceCard(s, i) {
  const card = el("div", "step");
  const statusCls = s.status === "ok" ? "s-ok"
    : s.status === "flagged" ? "s-flagged" : "s-error";
  card.innerHTML =
    `<div class="step-top"><span class="step-idx">${i + 1}</span>` +
    `<span class="step-name">${esc(s.step)}</span>` +
    `<span class="step-status ${statusCls}">${esc(s.status)} · ${s.latency_ms}ms</span></div>` +
    `<div class="step-detail">${esc(s.detail || "")}</div>`;
  if (s.flags && s.flags.length) {
    const fl = el("div", "step-flags");
    s.flags.forEach((f) => {
      const cls = f.startsWith("INFO") ? "info" : f.startsWith("POLICY") ? "pol" : "";
      fl.appendChild(el("div", cls, f));
    });
    card.appendChild(fl);
  }
  const extras = [];
  if (s.ranked) extras.push("ranked: " + s.ranked.map((r) =>
    `${r.node_id} rrf=${r.rrf} bm25=${r.bm25} cos=${r.cosine} via=${r.via || "rrf"}`).join("\n"));
  if (s.explain_top && s.explain_top.matched) extras.push(
    "bm25 terms in top doc: " + s.explain_top.matched.map((m) =>
      `${m.term}(tf${m.tf},+${m.contrib})`).join(" "));
  if (s.bias_removed) extras.push("bias_removed: " + s.bias_removed.join(", "));
  if (s.tier) extras.push(`tier=${s.tier} basis=${(s.basis || []).join(",")}`);
  if (s.repairs && s.repairs.length) extras.push("schema repairs: " + s.repairs.join(" | "));
  if (s.system_prompt) extras.push("system prompt as sent:\n" + s.system_prompt);
  if (s.user_prompt) extras.push("user message as sent:\n" + s.user_prompt);
  if (s.assembled_context) extras.push("assembled context:\n" + s.assembled_context);
  if (s.raw_output) extras.push("raw model output:\n" + s.raw_output);
  if (s.reply_after_scrub && (s.flags || []).length) {
    extras.push("reply before scrub:\n" + s.reply_before_scrub +
      "\n\nreply after scrub:\n" + s.reply_after_scrub);
  }
  if (s.case_state_rendered) extras.push("case state now:\n" + s.case_state_rendered);
  if (extras.length) {
    const pre = el("div", "step-extra", extras.join("\n\n"));
    pre.style.display = "none";
    const toggle = el("div", "step-toggle", "\u25b8 show prompt / raw output");
    toggle.onclick = () => {
      const shown = pre.style.display === "none";
      pre.style.display = shown ? "block" : "none";
      toggle.textContent = (shown ? "\u25be hide" : "\u25b8 show") + " prompt / raw output";
    };
    card.appendChild(toggle);
    card.appendChild(pre);
  }
  return card;
}

/* ------------------------------------------------------------------ render */
function renderMarkdown(text) {
  const lines = String(text).split("\n");
  const out = [];
  let para = [];
  const flush = () => {
    if (!para.length) return;
    out.push(`<p>${para.join(" ")}</p>`);
    para = [];
  };
  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) { flush(); return; }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      flush();
      out.push(`<div style="margin:0 0 6px">• ${inline(line.slice(2))}</div>`);
      return;
    }
    if (line.startsWith("> ")) { flush(); out.push(`<blockquote>${inline(line.slice(2))}</blockquote>`); return; }
    if (line.startsWith("### ")) { flush(); out.push(`<p><b>${inline(line.slice(4))}</b></p>`); return; }
    para.push(inline(line));
  });
  flush();
  return out.join("");
}

function inline(text) {
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  s = s.replace(/\[FACT\]|\[PREDICTION\]|\[HYPOTHETICAL\]|\[POSSIBILITY\]|\[CONSTRAINT\]|\[AUTHOR_CLAIM\]/g,
    (m) => {
      const tag = m.slice(1, -1);
      return `<span class="tag-badge t-${tag}" title="${TAG_NOTE[tag]}">${tag}</span>`;
    });
  s = s.replace(/\[(R\d+-[CS]\d+)\]/g, (m, id) => {
    const row = citationMap.get(id);
    const bad = row && !row.verified ? " unverified" : "";
    return `<span class="cite${bad}" data-cite="${id}">${id}</span>`;
  });
  return s;
}

function attachCiteHandlers(root) {
  root.querySelectorAll(".cite").forEach((chip) => {
    chip.onclick = (ev) => {
      const row = citationMap.get(chip.dataset.cite) || { quote: "(quote not in sidecar)", verified: false };
      const tip = $("tooltip");
      tip.innerHTML =
        `<span class="tid">${chip.dataset.cite}</span> ` +
        `<span class="tmeta">${esc(row.page_hint || "")}</span>\n` +
        `${esc(row.quote || row.text || "")}\n` +
        (row.verified ? "" :
          `<div class="q muted">stored chunk: “${esc(row.chunk_excerpt || row.text || "")}”</div>`) +
        `<div class="tmeta">${row.verified
          ? "quote verified against the stored chunk"
          : "NOT verbatim in stored chunk — treat as paraphrase"}</div>`;
      tip.classList.remove("hidden");
      const r = chip.getBoundingClientRect();
      tip.style.left = Math.min(window.innerWidth - 440, r.left) + "px";
      tip.style.top = (r.bottom + 8) + "px";
      ev.stopPropagation();
    };
  });
}

const TIERS = ["Established Consensus", "Theoretical Plausibility", "Highly Contested",
  "Conceptually Open", "Architecturally Implausible"];

function renderSidecar(sc, softFlags) {
  const wrap = el("div", "sidecar hidden");
  wrap.innerHTML = sidecarHTML(sc, softFlags);
  return wrap;
}

function sidecarHTML(sc, softFlags) {
  const idx = TIERS.indexOf(sc.confidence_tier);
  const ladder = TIERS.map((t, i) => `<div class="tier-seg ${i === idx ? "on" : ""}" title="${t}${sc.tier_reported === false ? " (defaulted)" : ""}"></div>`).join("");
  const list = (items) => items.length
    ? `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`
    : `<div class="v muted">none</div>`;
  const cites = (sc.citations || []).map((c) =>
    `<div class="cite-row"><b>${esc(c.chunk_id)}</b>
      <span class="${c.verified ? "ok" : "no"}">${c.verified ? "✓ verified" : "⚠ not verbatim in chunk"}</span>
      ${c.nli_label && c.nli_label !== "unavailable"
        ? `<span class="${c.nli_label === "entailment" ? "ok" : "no"}">NLI ${esc(c.nli_label)} ${(c.nli_score ?? 0).toFixed(2)}</span>` : ""}
      ${c.auto_derived ? `<span class="muted"> · derived from inline citation</span>` : ""}
      <span class="muted"> ${esc(c.page_hint || "")}</span>
      <div class="q">“${esc(c.quote)}”</div>
      ${c.verified ? "" : `<div class="q muted">stored chunk: “${esc(c.chunk_excerpt || c.text || "")}”</div>`}
    </div>`).join("");
  const fs = sc.field_sources || {};
  const gear = (k) => (fs[k] && fs[k] !== "model_declared")
    ? `<span class="gear" title="${fs[k] === "engine_default"
        ? "engine default — the model did not report this field"
        : "engine derived — reconstructed by the engine, not the model"}">\u2699</span>`
    : "";
  const tierLabel = sc.confidence_tier_display || sc.confidence_tier;
  return `<h4>Epistemic Sidecar</h4>
     <div class="sidecar-grid">
       <div class="sc" data-f="confidence_tier"><div class="k">Confidence tier ${gear("confidence_tier")}</div>
         <div class="v"><b>${esc(tierLabel)}</b>
           ${sc.tier_reported === false
             ? `<div class="muted">\u26a0 UNDECLARED \u2014 model did not report. ` +
               `Engine default: ${esc(sc.confidence_tier)}</div>` : ""}</div>
         <div class="tierbar">${ladder}</div>
         <div class="tier-ladder">${TIERS.map((t, i) =>
            i === idx ? `<b>${t}</b>` : `<span>${t}</span>`).join(" → ")}</div></div>
       <div class="sc" data-f="basis"><div class="k">Basis ${gear("basis")}</div><div class="v">
         ${(sc.basis || []).map((b) => `<span class="basis-chip">${esc(b)}</span>`).join("") || "—"}
       </div></div>
       <div class="sc" data-f="assumptions_used"><div class="k">Assumptions used</div>${list(sc.assumptions_used || [])}</div>
       <div class="sc" data-f="unknowns"><div class="k">Unknowns</div>${list(sc.unknowns || [])}</div>
       <div class="sc" data-f="what_would_change"><div class="k">What would change this</div>${list(sc.what_would_change || [])}</div>
       <div class="sc"><div class="k">Warnings &amp; engine derivations</div>${list(sc.warnings || [])}</div>
       ${(softFlags || []).length ? `<div class="sc"><div class="k">Soft flags (policy / info)</div>${list(softFlags)}</div>` : ""}
       ${sc.pushback ? `<div class="sc"><div class="k">Pushback handling</div><div class="v">${esc(txt(sc.pushback))}</div></div>` : ""}
       ${(sc.model_status || []).length ? `<div class="sc"><div class="k">Model status (structured)</div><div class="v">${(sc.model_status || []).map((l) => `<div class="muted">${esc(l)}</div>`).join("")}</div></div>` : ""}
       ${sc.epistemic_analysis ? `<div class="sc"><div class="k">Epistemic analysis (model)</div><div class="v">${esc(txt(sc.epistemic_analysis).slice(0, 900))}</div></div>` : ""}
       ${(sc.fields_omitted_by_model || []).length ? `<div class="sc"><div class="k">Fields the model omitted</div><div class="v muted">${esc((sc.fields_omitted_by_model || []).join(", "))}</div></div>` : ""}
       ${(sc.question_decomposition || []).length ? `<div class="sc" data-f="question_decomposition"><div class="k">Question decomposition</div>${list(sc.question_decomposition)}</div>` : ""}
       ${(sc.vocabulary_contract || []).length ? `<div class="sc" data-f="vocabulary_contract"><div class="k">Vocabulary contract</div>${rowsTable(sc.vocabulary_contract, [["term", "original"], ["read as", "interpreted_as"], ["why", "reason"]])}</div>` : ""}
       ${(sc.theory_matrix || []).length ? `<div class="sc" data-f="theory_matrix"><div class="k">Theory matrix</div>${rowsTable(sc.theory_matrix, [["option", "option"], ["viable under", "viable_under"], ["excluded under", "excluded_under"]])}</div>` : ""}
       ${(sc.claim_typology || []).length ? `<div class="sc" data-f="claim_typology"><div class="k">Claim typology</div>${rowsTable(sc.claim_typology, [["claim", "claim"], ["type", "type"]])}</div>` : ""}
       ${(sc.methodological_constraints || []).length ? `<div class="sc" data-f="methodological_constraints"><div class="k">Methodological constraints</div>${list(sc.methodological_constraints)}</div>` : ""}
       ${(sc.stakes_and_policy || "").trim() ? `<div class="sc" data-f="stakes_and_policy"><div class="k">Stakes &amp; policy</div><div class="v">${esc(sc.stakes_and_policy)}</div></div>` : ""}
       <div class="sc"><div class="k">Provenance</div><div class="v">${esc(sc.provenance_note || "")}</div></div>
       <div class="sc"><div class="k">Retrieval</div><div class="v">
         leaves ${(sc.retrieval && sc.retrieval.leaves || []).join(", ") || "none"} ·
         sections ${(sc.retrieval && sc.retrieval.sections || []).join(", ") || "none"}
         <div class="muted">BM25 hits ${sc.retrieval && sc.retrieval.channels ? sc.retrieval.channels.bm25_hits : 0}
           · dense hits ${sc.retrieval && sc.retrieval.channels ? sc.retrieval.channels.dense_hits : 0}
           · rrf_k ${sc.retrieval && sc.retrieval.channels ? sc.retrieval.channels.rrf_k : 60}</div>
       </div>
     </div>
     <div data-f="citations"><div class="k" style="margin:8px 0 4px">Citations ${gear("citations")}</div>
     ${cites || `<div class="cite-row muted">no citations in this turn</div>`}</div>
     <div class="cite-row muted">model ${esc(sc.model || "")} · completion ${sc.completion_tokens || 0} tok
       · context ${sc.context_tokens || 0} tok</div>`;
}

function scanCompletedFields(buf) {
  /* The Pass 2 stream is one JSON object in schema order. Every top-level
     field whose value is complete is parsed and rendered immediately, so the
     sidecar pane fills in live while generation is still running. */
  const out = {};
  let i = buf.indexOf("{");
  if (i < 0) return out;
  i++;
  while (i < buf.length) {
    while (i < buf.length && /[\s,]/.test(buf[i])) i++;
    if (i >= buf.length || buf[i] === "}") break;
    if (buf[i] !== '"') break;
    const ke = scanStringEnd(buf, i);
    if (ke < 0) break;
    let key;
    try { key = JSON.parse(buf.slice(i, ke + 1)); } catch (e) { break; }
    let j = ke + 1;
    while (j < buf.length && /\s/.test(buf[j])) j++;
    if (buf[j] !== ":") break;
    j++;
    while (j < buf.length && /\s/.test(buf[j])) j++;
    const ve = scanValueEnd(buf, j);
    if (ve < 0) break;
    try { out[key] = JSON.parse(buf.slice(j, ve + 1)); } catch (e) { break; }
    i = ve + 1;
  }
  return out;
}

function scanStringEnd(buf, from) {
  for (let j = from + 1; j < buf.length; j++) {
    if (buf[j] === "\\") { j++; continue; }
    if (buf[j] === '"') return j;
  }
  return -1;
}

function scanValueEnd(buf, from) {
  const c = buf[from];
  if (c === '"') return scanStringEnd(buf, from);
  if (c === "{" || c === "[") {
    let depth = 0, inStr = false, esc = false;
    for (let j = from; j < buf.length; j++) {
      const ch = buf[j];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === "{" || ch === "[") depth++;
      else if (ch === "}" || ch === "]") {
        depth--;
        if (depth === 0) return j;
        if (depth < 0) return -1;
      }
    }
    return -1;
  }
  const m = /^(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/.exec(buf.slice(from));
  return m ? from + m[0].length - 1 : -1;
}

let liveSidecarAt = 0;
let liveDraft = null;

const STATE_NOTE = {
  streaming: "Streaming \u2014 unverified draft from the token stream. Citations, " +
             "NLI checks, tier defaults and engine derivations have NOT run yet.",
  complete: "Verified \u2014 engine sidecar. Fields that changed from the streamed " +
            "draft are highlighted.",
};

function renderLiveSidecar(sc, mode) {
  const banner = `<div class="sidecar-state ${mode}">${STATE_NOTE[mode] || ""}</div>`;
  $("liveSidecar").innerHTML =
    `<div class="sidecar live">${banner}${sidecarHTML(sc, [])}</div>`;
  const pill = $("sidecarLivePill");
  pill.textContent = mode === "streaming" ? "Streaming \u2014 unverified"
    : mode === "complete" ? "Verified \u2014 engine sidecar" : mode;
  pill.classList.toggle("warn", mode === "streaming");
  pill.classList.toggle("ok", mode === "complete");
  if (mode === "complete" && liveDraft) {
    requestAnimationFrame(() => {
      document.querySelectorAll("#liveSidecar [data-f]").forEach((node) => {
        const k = node.dataset.f;
        const before = JSON.stringify(liveDraft[k] ?? null);
        const after = JSON.stringify(sc[k] ?? null);
        if (before !== after) {
          node.classList.add("field-changed");
          setTimeout(() => node.classList.remove("field-changed"), 2600);
        }
      });
      liveDraft = null;
    });
  }
}

function renderTurn(res, rawText) {
  citationMap = new Map();
  (res.sidecar.citations || []).forEach((c) => citationMap.set(c.chunk_id, c));
  const { critical, soft } = splitFlags(res.flags);
  const turn = el("div", "turn");

  const you = el("div", "turn-user");
  const dataChars = res.data_chars || (res.sidecar && res.sidecar.data_chars) || 0;
  const dataBadge = dataChars
    ? `<div class="data-badge" title="Inline data analyzed this turn: not sanitized by S2A, not ingested, not persisted">\u{1f4ce} inline data (${dataChars} chars)</div>`
    : "";
  you.innerHTML = `<div class="who2">You</div><div class="bubble user">${esc(rawText)}${dataBadge}</div>`;
  turn.appendChild(you);

  const asst = el("div", "turn-asst");
  asst.innerHTML = `<div class="who2">Assistant</div>` +
    `<div class="bubble asst">` +
    renderBottomLine(res.sidecar.bottom_line) +
    renderReply(res.chat_reply) +
    renderDialectic(res.sidecar.dialectic) +
    `</div>`;
  if (critical.length) asst.appendChild(renderAlerts(critical));
  const usage = el("div", "muted",
    `context ${res.usage.context_tokens}/${Math.round(res.usage.window_tokens / 1024)}k tok · ` +
    `${res.usage.completion_tokens} tok out · ${res.usage.total_ms} ms · ` +
    `${res.usage.provider}:${res.usage.model}` +
    (res.sidecar.tier_reported === false ? " · tier defaulted by the engine" : ""));
  usage.style.marginTop = "6px";
  asst.appendChild(usage);
  turn.appendChild(asst);

  const scToggle = el("button", "ghost sidecar-toggle", "\u{1f50d} Epistemic sidecar");
  const sidecar = renderSidecar(res.sidecar, soft);
  scToggle.onclick = () => {
    sidecar.classList.toggle("hidden");
    scToggle.textContent = sidecar.classList.contains("hidden")
      ? "\u{1f50d} Epistemic sidecar" : "\u{1f50d} Hide epistemic sidecar";
  };
  turn.appendChild(scToggle);
  turn.appendChild(sidecar);

  const transToggle = el("button", "ghost trans-toggle",
    `\u2699 Transparency: ${res.trace.length} pipeline steps, prompts and raw output`);
  transToggle.onclick = () => {
    showView("trans");
  };
  turn.appendChild(transToggle);

  $("chat").appendChild(turn);
  attachCiteHandlers(turn);
  $("chat").scrollTop = $("chat").scrollHeight;
}

/* ---------------------------------------------------------------- history */
function aggregateFlags(trace) {
  return (trace || []).reduce((acc, step) =>
    acc.concat((step.flags || []).map((f) => `${step.step}: ${f}`)), []);
}

async function loadHistory() {
  if ($("chat").childElementCount) return;
  const { turns } = await api("/api/turns?session=default&limit=20");
  turns.forEach((t) => {
    const sc = t.sidecar || {};
    renderTurn({
      chat_reply: t.chat_reply,
      data_chars: sc.data_chars || 0,
      sidecar: sc,
      flags: aggregateFlags(t.trace),
      trace: t.trace || [],
      prompts: { assembled_context: "" },
      usage: {
        context_tokens: t.context_tokens || sc.context_tokens || 0,
        window_tokens: (status && status.settings.context_window) || 0,
        completion_tokens: sc.completion_tokens || 0,
        total_ms: (t.trace || []).reduce((a, x) => a + (x.latency_ms || 0), 0),
        provider: (status && status.settings.provider) || "?",
        model: t.model || sc.model || "?",
      },
    }, t.raw || "(no stored input)");
  });
}

/* -------------------------------------------------------------------- ask */
function jsonStringAt(buf, key) {
  /* Progressive read of one JSON string field: the stream is a JSON object
     generated in schema order (bottom_line, chat_reply first), so the prose
     can be rendered token by token before the object closes. */
  const k = `"${key}"`;
  let i = buf.indexOf(k);
  if (i < 0) return "";
  i = buf.indexOf(":", i + k.length);
  if (i < 0) return "";
  i = buf.indexOf('"', i + 1);
  if (i < 0) return "";
  let out = "";
  for (let j = i + 1; j < buf.length; j++) {
    const c = buf[j];
    if (c === "\\") {
      const n = buf[j + 1];
      if (n === undefined) break;
      if (n === "n") { out += "\n"; j += 1; }
      else if (n === "t") { out += " "; j += 1; }
      else if (n === "u") {
        const hex = buf.slice(j + 2, j + 6);
        if (hex.length < 4) break;
        out += String.fromCharCode(parseInt(hex, 16)); j += 5;
      } else { out += n; j += 1; }
      continue;
    }
    if (c === '"') break;
    out += c;
  }
  return out;
}

function stageLabel(ev, count) {
  const name = (ev.step || "").replace(/^\d+\./, "").replace(/_/g, " ");
  const mark = ev.status === "ok" ? "\u2713" : (ev.status === "flagged" ? "\u26a0" : "\u2026");
  return `pipeline \u00b7 ${count} step(s) \u00b7 ${name} ${mark}`;
}

function composerNotice(msg) {
  const box = el("div", "composer-notice", esc(msg));
  $("chat").appendChild(box);
  $("chat").scrollTop = $("chat").scrollHeight;
  setTimeout(() => box.remove(), 6000);
}

async function send() {
  const raw = $("input").value.trim();
  const data = $("dataInput").value.trim();
  if (busy) return;
  if (!raw && !data) return;
  if (data && !raw) {
    composerNotice("You added data but no question. Add a question or remove the data.");
    return;
  }
  busy = true;
  const dataChars = data.length;
  $("input").value = "";
  $("dataInput").value = "";
  $("dataBox").classList.add("hidden");
  $("sendBtn").disabled = true;

  const turn = el("div", "turn");
  const you = el("div", "turn-user");
  const dataBadge = dataChars
    ? `<div class="data-badge" title="not sanitized by S2A, not ingested, not persisted">\u{1f4ce} inline data (${dataChars} chars)</div>`
    : "";
  you.innerHTML = `<div class="who2">You</div><div class="bubble user">${esc(raw)}${dataBadge}</div>`;
  turn.appendChild(you);
  const asst = el("div", "turn-asst");
  asst.innerHTML = `<div class="who2">Assistant</div>
    <div class="bubble asst">
      <div class="stream-progress muted">running pipeline \u2026</div>
      <div class="stream-bottom"></div>
      <div class="stream-body"></div>
    </div>`;
  turn.appendChild(asst);
  $("chat").appendChild(turn);
  $("chat").scrollTop = $("chat").scrollHeight;
  resetTransparency(raw, dataChars > 0);
  renderPendingTrace(dataChars > 0);
  liveDraft = null;
  $("sidecarLivePill").textContent = "running";
  $("sidecarLivePill").classList.remove("warn", "ok");

  const prog = asst.querySelector(".stream-progress");
  const bottomEl = asst.querySelector(".stream-bottom");
  const bodyEl = asst.querySelector(".stream-body");
  let buf = "", stages = 0, result = null, streamError = null;

  const finish = async (res) => {
    renderLiveSidecar(res.sidecar, "complete");
    turn.remove();
    await revealPipeline(res);
    renderTurn(res, raw);
    $("contextPill").textContent =
      `Context: ${res.usage.context_tokens}/${Math.round(res.usage.window_tokens / 1024)}k`;
  };

  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: raw, session: "default", data: data }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let pending = "";
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      pending += dec.decode(chunk.value, { stream: true });
      let cut;
      while ((cut = pending.indexOf("\n\n")) >= 0) {
        const frame = pending.slice(0, cut);
        pending = pending.slice(cut + 2);
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") continue;
        let ev;
        try { ev = JSON.parse(payload); } catch (e) { continue; }
        if (ev.type === "stage") {
          stages += 1;
          prog.textContent = stageLabel(ev, stages);
          liveStep(ev);
        } else if (ev.type === "delta") {
          buf += ev.text;
          const bottom = jsonStringAt(buf, "bottom_line");
          const reply = jsonStringAt(buf, "chat_reply");
          bottomEl.innerHTML = bottom ? renderBottomLine(bottom) : "";
          bodyEl.innerHTML = reply ? renderReply(reply) : "";
          $("chat").scrollTop = $("chat").scrollHeight;
          const now = Date.now();
          if (now - liveSidecarAt > 250) {
            liveSidecarAt = now;
            liveDraft = scanCompletedFields(buf);
            renderLiveSidecar(liveDraft, "streaming");
          }
        } else if (ev.type === "result") {
          result = ev.data;
        } else if (ev.type === "error") {
          streamError = ev.message;
        }
      }
    }
  } catch (err) {
    streamError = String((err && err.message) || err);
  }

  if (result) {
    await finish(result);
  } else {
    try {
      const res2 = await api("/api/ask", { method: "POST",
        body: { text: raw, session: "default", data: data } });
      await finish(res2);
    } catch (err2) {
      turn.remove();
      const errTurn = el("div", "turn");
      const you2 = el("div", "turn-user");
      you2.innerHTML = `<div class="who2">You</div><div class="bubble user">${esc(raw)}</div>`;
      errTurn.appendChild(you2);
      const asst2 = el("div", "turn-asst");
      asst2.innerHTML = `<div class="who2">Assistant</div>
        <div class="bubble asst"><span class="chip-flag">ENGINE ERROR: ${esc(String((err2 && err2.message) || err2))}${streamError ? " \u00b7 stream: " + esc(streamError) : ""}</span></div>`;
      errTurn.appendChild(asst2);
      $("chat").appendChild(errTurn);
      const card = el("div", "step");
      card.innerHTML = `<div class="step-top"><span class="step-idx">!</span>
        <span class="step-name">engine error</span>
        <span class="step-status s-error">error</span></div>`;
      const fl = el("div", "step-flags");
      fl.appendChild(el("div", "", String((err2 && err2.message) || err2)));
      card.appendChild(fl);
      $("trace").appendChild(card);
    }
  }
  busy = false;
  $("sendBtn").disabled = false;
  $("input").focus();
}

$("sendBtn").onclick = send;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});
$("dataBtn").onclick = () => {
  const box = $("dataBox");
  box.classList.toggle("hidden");
  if (!box.classList.contains("hidden")) $("dataInput").focus();
};
$("dataClear").onclick = (e) => {
  e.preventDefault();
  $("dataInput").value = "";
  $("dataBox").classList.add("hidden");
};
$("dataInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});

/* ------------------------------------------------------------------ ingest */
$("ingestBtn").onclick = $("attachBtn").onclick = () => $("ingestModal").classList.remove("hidden");
$("ingestGo").onclick = async () => {
  const text = $("docText").value.trim();
  if (!text) return;
  $("ingestGo").disabled = true;
  $("ingestStatus").textContent = `queued (${text.length} chars)…`;
  try {
    const start = await api("/api/documents", { method: "POST", body: { text } });
    let job;
    for (let i = 0; i < 240; i++) {
      await new Promise((r) => setTimeout(r, 500));
      job = await api(`/api/jobs/${start.job_id}`);
      $("ingestStatus").textContent = `${job.status} · ${job.percent}%`;
      if (job.status === "done" || job.status === "error") break;
    }
    if (job.status === "error") {
      $("ingestStatus").textContent = "ingest failed: " + job.error;
    } else {
      const r = job.result;
      $("ingestStatus").textContent =
        `ingested ${r.doc_id}: ${r.leaves} leaves, ${r.sections} sections — "${r.title}"`;
      const box = $("trace");
      box.innerHTML = "";
      const head = el("div", "step");
      head.innerHTML = `<div class="step-top"><span class="step-name">ingestion ${esc(r.doc_id)}</span>` +
        `<span class="step-status s-ok">${r.leaves} leaves · ${r.sections} sections</span></div>` +
        `<div class="step-detail">${esc(r.title)}</div>`;
      box.appendChild(head);
      (r.trace || []).forEach((s, i) => box.appendChild(traceCard(s, i)));
      if ((r.warnings || []).length) {
        const w = el("div", "step");
        w.innerHTML = `<div class="step-top"><span class="step-name">ingest warnings</span>` +
          `<span class="step-status s-flagged">${r.warnings.length}</span></div>`;
        const fl = el("div", "step-flags");
        r.warnings.forEach((f) => fl.appendChild(el("div", "info", f)));
        w.appendChild(fl);
        box.appendChild(w);
      }
      await loadDocs();
    }
  } catch (err) {
    $("ingestStatus").textContent = "ingest failed: " + err.message;
  } finally {
    $("ingestGo").disabled = false;
  }
};

/* ---------------------------------------------------------------- settings */
$("settingsBtn").onclick = async () => {
  $("settingsModal").classList.remove("hidden");
  const s = await api("/api/status");
  const prompts = await api("/api/prompts");
  const h = s.provider_health || {};
  const kv = [
    ["provider", s.settings.provider],
    ["model (single)", s.settings.model],
    ["base_url", s.settings.base_url],
    ["embedding model", s.settings.embed_model],
    ["provider health", h.ok ? "ok" : "unreachable"],
    ["model present", h.model_present === undefined ? "n/a" : (h.model_present ? "yes" : "NO")],
    ["available models", (h.available || []).join(", ") || "—"],
    ["leaf chunk tokens", s.settings.leaf_chunk_tokens],
    ["cluster size", s.settings.cluster_size],
    ["leaf_k / section_k", `${s.settings.leaf_k} / ${s.settings.section_k}`],
    ["rrf_k", s.settings.rrf_k],
    ["context window", s.settings.context_window],
    ["max context tokens", s.settings.max_context_tokens],
    ["nodes indexed", s.index.nodes_indexed],
    ["dense vectors", s.index.dense_vectors],
    ["vector modes", JSON.stringify(s.index.vector_modes)],
    ["counts", JSON.stringify(s.counts)],
    ["web access", "none (by design)"],
  ];
  $("settingsBody").innerHTML =
    `<div class="kv">${kv.map(([k, v]) =>
      `<div class="k">${esc(k)}</div><div>${esc(String(v))}</div>`).join("")}</div>
     <div class="kv"><div class="k">modal tags</div><div>${prompts.modal_tags.join(" ")}</div>
       <div class="k">confidence tiers</div><div>${prompts.confidence_tiers.join(" → ")}</div>
       <div class="k">Pass 1 system</div><div>${esc(prompts.pass1_system.slice(0, 320))}…</div>
       <div class="k">Pass 2 system</div><div>${esc(prompts.pass2_system.slice(0, 320))}…</div></div>`;
};

$("evalBtn").onclick = async () => {
  $("evalBtn").disabled = true;
  $("evalOut").classList.remove("hidden");
  $("evalOut").textContent = "running 4 tests (in-process, throwaway db)…";
  try {
    const rep = await api("/api/eval", { method: "POST" });
    const lines = [`RESULT: ${rep.passed}/${rep.total} passed — provider=${rep.provider}`];
    rep.tests.forEach((t) => {
      lines.push(`\n${t.name}  [${t.passed ? "PASS" : "FAIL"}]  ${t.wall_ms}ms`);
      t.checks.forEach((c) => lines.push(`   [${c.ok ? "x" : " "}] ${c.label}`));
    });
    $("evalOut").textContent = lines.join("\n");
  } catch (err) {
    $("evalOut").textContent = "eval failed: " + err.message;
  } finally {
    $("evalBtn").disabled = false;
  }
};

$("resetBtn").onclick = async () => {
  await api("/api/session/reset", { method: "POST", body: { session: "default" } });
  $("chat").innerHTML = "";
  $("trace").innerHTML = `<div class="stepper-empty">case state reset</div>`;
};

$("deleteBtn").onclick = async () => {
  const ok = confirm(
    "Delete ALL case state, turn history, and the job cache?\n" +
    "Attached documents are kept. This cannot be undone.");
  if (!ok) return;
  const res = await api("/api/session/delete", { method: "POST", body: { session: "" } });
  $("chat").innerHTML = "";
  $("trace").innerHTML =
    `<div class="stepper-empty">case deleted \u00b7 ${res.turns} turns and ` +
    `${res.case_state} case state(s) removed</div>`;
  $("liveSidecar").innerHTML = `<div class="stepper-empty">case deleted</div>`;
  $("sidecarLivePill").textContent = "awaiting turn";
  $("sidecarLivePill").classList.remove("warn", "ok");
  $("contextPill").textContent = "Context: 0/32k";
  ["blkRaw", "blkPass1Sys", "blkPass1Out", "blkPass2Sys", "blkContext"].forEach((id) => {
    const b = $(id).querySelector(".block-body");
    b.textContent = "awaiting submit";
    b.classList.add("empty");
    $(id).classList.remove("flagged");
  });
  $("blkData").classList.add("hidden");
};

$("deleteAllBtn").onclick = async () => {
  const ok = confirm(
    "Delete the ENTIRE memory: every attached document, chunk, vector, card, all " +
    "turns, and all case state?\nAttached documents are removed too. This cannot " +
    "be undone.");
  if (!ok) return;
  const res = await api("/api/store/delete_all", { method: "POST", body: {} });
  const d = res.deleted || {};
  $("chat").innerHTML = "";
  $("trace").innerHTML =
    `<div class="stepper-empty">ALL memory deleted \u00b7 ${d.documents || 0} doc(s), ` +
    `${d.nodes || 0} node(s), ${d.vectors || 0} vector(s), ${d.turns || 0} turn(s) removed</div>`;
  $("liveSidecar").innerHTML = `<div class="stepper-empty">memory deleted</div>`;
  $("sidecarLivePill").textContent = "awaiting turn";
  $("sidecarLivePill").classList.remove("warn", "ok");
  $("contextPill").textContent = "Context: 0/32k";
  $("densePill").textContent = "dense: \u2026";
  $("docList").innerHTML =
    "<em>no documents attached — answers are parametric only</em>";
  ["blkRaw", "blkPass1Sys", "blkPass1Out", "blkPass2Sys", "blkContext"].forEach((id) => {
    const b = $(id).querySelector(".block-body");
    b.textContent = "awaiting submit";
    b.classList.add("empty");
    $(id).classList.remove("flagged");
  });
  $("blkData").classList.add("hidden");
};

document.addEventListener("click", (e) => {
  if (!e.target.classList.contains("cite")) $("tooltip").classList.add("hidden");
});

loadStatus().then(() => loadHistory()).catch((e) => {
  $("modelBanner").classList.remove("hidden");
  $("modelBanner").textContent = "engine unreachable: " + e.message;
});