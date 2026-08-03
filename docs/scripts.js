/* Re-Matrix browser pipeline: Stages 2-6 run client-side against the Groq API via the
   Cloudflare Worker proxy (key is stored server-side; see workers/proxy). */

"use strict";

// Point this at your deployed worker.
const PROXY_URL = "https://re-matrix-proxy.irhoseapori.workers.dev";
const APP_PASSWORD_HINT = "Contrari-matrix";
const SESSION_KEY = "rematrix_unlocked";
const PASS_KEY = "rematrix_pass";
const DATA_VERSION = "4"; // bump to force browsers off a cached corpus

// Per-level caps on principles sent to the reasoning model (free-tier TPM guard).
const PROMPT_LEVEL_CAPS = { L0_axiom: 10, L1_mechanistic_pathway: 18, L2_context_modifier: 10, L3_known_exception: 4 };
const SEED_CANDIDATE_CAP = 40; // principles offered to the seed-selection model
const LEVEL_LABELS = {
  "L0_axiom": "L0 - Axioms (Foundational)",
  "L1_mechanistic_pathway": "L1 - Mechanistic Pathways",
  "L2_context_modifier": "L2 - Context Modifiers",
  "L3_known_exception": "L3 - Known Exceptions / Resistance"
};

let CORPUS = null;
let lastReport = null;

const PROMPTS = {
  understanding: `You are an expert cancer immunologist analyzing a researcher's query.

Classify the query type:
- mechanistic: "what happens if X binds Y?" / "how does X lead to Y?"
- therapeutic_hypothesis: "would blocking X improve outcomes in condition Y?"
- edge_case_exploration: "under what conditions would X fail?" / "when does Y not work?"
- comparative: "how does mechanism A differ from B in context C?"

Extract entities by category (cell_types, molecules, pathways, interventions, contexts, tumor_types).
Extract the implied causal question: what is the researcher actually asking to predict?

Return JSON only:
{
  "query_type": "mechanistic | therapeutic_hypothesis | edge_case_exploration | comparative",
  "entities": {"cell_types": [], "molecules": [], "pathways": [], "interventions": [], "contexts": [], "tumor_types": []},
  "causal_question": "string",
  "context": {}
}`,

  seed_selection: `You are an expert cancer immunologist selecting the most relevant principles for reasoning about a research query.

Given the query and a numbered list of principles (each with id, content, entities, hierarchy_level), select the IDs of the 4-8 most relevant principles that a rigorous causal analysis of this query MUST ground itself in. Prefer:
- Axioms (L0) that directly apply
- Mechanistic pathways (L1) touching the query's entities
- Any L2 context modifiers or L3 exceptions that clearly touch the same entities

Return JSON only: {"seed_ids": ["id1", "id2", ...]}`,

  thought_experiment: `You are an expert cancer immunologist conducting a rigorous causal simulation.

Given a research query with its causal question, entities, and a set of principles organized by hierarchy level (each with an id, content, entities, and citation), construct an explicit step-by-step causal chain simulating what happens.

RULES:
1. State initial conditions explicitly: what is present, what intervention/perturbation is introduced, what is held constant.
2. For EACH step: name the principle invoked (use its exact ID from the provided list), quote its content, state the mechanistic consequence, and carry that consequence forward as input to the next step. Make every inferential link explicit.
3. Predict the most likely outcome with an explicit confidence qualifier: "high", "medium", "low", or "speculative", and a one-line justification for that confidence.
4. Explicitly flag ALL assumptions the query underspecified.

IMPORTANT: Every step must reference a principle_id from the provided list. If you must cite something outside the list, put it in flagged_assumptions instead. Return JSON only:
{
  "initial_conditions": "string",
  "intervention": "string",
  "held_constant": ["..."],
  "causal_chain": [
    {"step_number": 1, "principle_id": "exact id from list", "principle_content": "...", "citation": "...", "mechanistic_consequence": "...", "confidence": "high|medium|low|speculative"}
  ],
  "predicted_outcome": "string",
  "outcome_confidence": "high|medium|low|speculative",
  "confidence_justification": "string",
  "flagged_assumptions": ["..."]
}`,

  edge_cases: `You are a deliberately ADVERSARIAL cancer immunologist whose ONLY job is to stress-test predictions. Standard immunology reasoning is already baked into the prediction; your job is to find the cracks.

You are given a predicted thought experiment (its causal chain and predicted outcome) and a set of principles including L2 context modifiers and L3 known exceptions.

For each relevant L2/L3 principle, actively ask: "Does this change, weaken, or reverse the predicted outcome?" Only keep cases that MATERIALLY do.

Search aggressively for:
- Known resistance/evasion mechanisms (antigen loss, MHC-I downregulation, alternative pathway compensation)
- Tumor heterogeneity (only a subclone responds)
- Context variants (hypoxia, stromal density, immunocompromised states, prior treatment)
- Timing issues and combination-therapy interactions
- Cold-tumor / low-T-cell-infiltrate scenarios

For EACH edge case return:
{"condition": "...", "mechanism_of_deviation": "...", "citation": "...", "principle_id": "...", "hierarchy_level": "L2_context_modifier|L3_known_exception", "severity": "reverses|weakens|qualifies"}

Return JSON only: {"edge_cases": [...]}`,

  followups: `Given a query, predicted outcome, its edge cases, and flagged assumptions, suggest specific follow-up questions or experiments that would distinguish the predicted outcome from the edge cases.

Return JSON only: {"suggested_followups": ["...", "..."]}`
};

/* ---------- Groq API helpers ---------- */

function getSettings() {
  return {
    reasoning: document.getElementById("reasoning-model").value,
    fast: document.getElementById("fast-model").value
  };
}

function getAccessPassword() {
  return sessionStorage.getItem(PASS_KEY) || "";
}

async function groqChat(model, system, user, temperature = 0.2) {
  const password = getAccessPassword();
  if (!password) throw new Error("Access password missing. Reload the page and unlock the app.");
  const resp = await fetch(PROXY_URL + "/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-App-Password": password
    },
    body: JSON.stringify({
      model,
      temperature,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ]
    })
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error("Proxy error " + resp.status + ": " + body.slice(0, 400));
  }
  const data = await resp.json();
  return JSON.parse(data.choices[0].message.content);
}

/* ---------- Pipeline ---------- */

async function loadCorpus() {
  if (CORPUS) return CORPUS;
  const resp = await fetch("data/principles.json?v=" + DATA_VERSION);
  CORPUS = await resp.json();
  CORPUS.idx = indexById();
  return CORPUS;
}

function indexById() {
  const map = {};
  for (const p of CORPUS.principles) map[p.id] = p;
  return map;
}

function byLevel(ids) {
  const out = {};
  for (const id of ids) {
    const p = CORPUS.idx[id];
    if (!p) continue;
    (out[p.hierarchy_level] = out[p.hierarchy_level] || []).push(p);
  }
  return out;
}

function walkGraph(seedIds) {
  // Replicates Stage 3 hierarchy-aware traversal in the browser.
  const retrieved = {};
  const seen = new Set();

  const add = (id) => {
    const p = CORPUS.idx[id];
    if (!p || seen.has(id)) return;
    seen.add(id);
    (retrieved[p.hierarchy_level] = retrieved[p.hierarchy_level] || []).push(p);
  };

  for (const id of seedIds) add(id);

  // Walk UP to L0 axioms
  const walkUp = (id, depth = 0) => {
    if (depth > 3) return;
    const parents = CORPUS.depends_on[id] || [];
    for (const parentId of parents) {
      add(parentId);
      walkUp(parentId, depth + 1);
    }
  };

  // Walk DOWN/SIDEWAYS to L2 context modifiers and L3 exceptions
  const walkDown = (id, depth = 0) => {
    if (depth > 3) return;
    const children = CORPUS.dependents[id] || [];
    for (const childId of children) {
      add(childId);
      walkDown(childId, depth + 1);
    }
  };

  for (const id of seedIds) {
    walkUp(id);
    walkDown(id);
  }

  return retrieved;
}

function entityOverlap(queryEntities, principle) {
  const q = new Set();
  for (const arr of Object.values(queryEntities)) for (const e of arr) q.add(e.toLowerCase());
  if (!q.size) return 0;
  let hits = 0;
  for (const e of principle.entities) if (q.has(e.toLowerCase())) hits++;
  return hits;
}

function rankByEntity(queryEntities, ids) {
  return ids.slice().sort((a, b) =>
    entityOverlap(queryEntities, CORPUS.idx[b]) - entityOverlap(queryEntities, CORPUS.idx[a]));
}

async function stageUnderstanding(query, statusEl) {
  setStep(statusEl, "running");
  const data = await groqChat(getSettings().fast, PROMPTS.understanding,
    "Query: " + query, 0.1);
  setStep(statusEl, "done");
  return data;
}

async function stageRetrieval(understanding, statusEl) {
  setStep(statusEl, "running");
  // Pre-filter to the top candidates by entity overlap, then compact each
  // principle to a short line. This keeps the seed-selection prompt small
  // (free-tier TPM guard) without asking the model to read all 142.
  const candidates = CORPUS.principles
    .slice()
    .sort((a, b) => entityOverlap(understanding.entities, b) - entityOverlap(understanding.entities, a))
    .slice(0, SEED_CANDIDATE_CAP);
  const list = candidates.map(p =>
    `ID: ${p.id}\nContent: ${p.content.slice(0, 60)}${p.content.length > 60 ? "..." : ""}\nEntities: ${p.entities.join(", ")}\nLevel: ${p.hierarchy_level}`).join("\n\n");
  const data = await groqChat(getSettings().fast, PROMPTS.seed_selection,
    `QUERY: ${understanding.causal_question}\n\nContext: ${JSON.stringify(understanding.context)}\nEntities: ${JSON.stringify(understanding.entities)}\n\nPRINCIPLES (top ${SEED_CANDIDATE_CAP} candidates by entity relevance):\n${list}`, 0.1);

  let seedIds = data.seed_ids || [];
  seedIds = seedIds.filter(id => CORPUS.idx[id]);
  seedIds = rankByEntity(understanding.entities, seedIds);

  const retrieved = walkGraph(seedIds);
  setStep(statusEl, "done");
  return retrieved;
}

function formatPrinciplesForPrompt(retrieved, understanding) {
  const order = ["L0_axiom", "L1_mechanistic_pathway", "L2_context_modifier", "L3_known_exception"];
  const entities = (understanding && understanding.entities) || {};
  const lines = [];
  for (const lvl of order) {
    const ps = retrieved[lvl] || [];
    if (!ps.length) continue;
    const cap = PROMPT_LEVEL_CAPS[lvl] || ps.length;
    const picked = ps.slice().sort((a, b) =>
      entityOverlap(entities, b) - entityOverlap(entities, a)).slice(0, cap);
    lines.push(`=== ${LEVEL_LABELS[lvl]} (showing ${picked.length} of ${ps.length} retrieved) ===`);
    for (const p of picked) {
      lines.push(`ID: ${p.id}`);
      lines.push(`Content: ${p.content}`);
      lines.push(`Entities: ${p.entities.join(", ")}`);
      lines.push(`Citation: ${p.source_citation}`);
      lines.push("");
    }
  }
  return lines.join("\n");
}

async function stageThoughtExperiment(understanding, retrieved, statusEl) {
  setStep(statusEl, "running");
  const user = `Query: ${understanding.original_query}
Causal Question: ${understanding.causal_question}
Context: ${JSON.stringify(understanding.context)}
Entities: ${JSON.stringify(understanding.entities)}

PRINCIPLES:
${formatPrinciplesForPrompt(retrieved, understanding)}`;
  const data = await groqChat(getSettings().reasoning, PROMPTS.thought_experiment, user, 0.2);
  setStep(statusEl, "done");
  return data;
}

async function stageEdgeCases(understanding, thoughtExp, retrieved, statusEl) {
  setStep(statusEl, "running");
  const chain = (thoughtExp.causal_chain || []).map(s =>
    `Step ${s.step_number}: [${s.principle_id}] ${s.mechanistic_consequence} (conf: ${s.confidence})`).join("\n");
  const user = `QUERY: ${understanding.original_query}
CAUSAL QUESTION: ${understanding.causal_question}

PREDICTED THOUGHT EXPERIMENT:
Initial Conditions: ${thoughtExp.initial_conditions}
Intervention: ${thoughtExp.intervention}
Predicted Outcome: ${thoughtExp.predicted_outcome}
Outcome Confidence: ${thoughtExp.outcome_confidence}

Causal Chain:
${chain}

PRINCIPLES (focus on L2 context modifiers and L3 known exceptions):
${formatPrinciplesForPrompt(retrieved, understanding)}`;
  const data = await groqChat(getSettings().reasoning, PROMPTS.edge_cases, user, 0.5);
  setStep(statusEl, "done");
  return data.edge_cases || [];
}

async function stageFollowups(understanding, thoughtExp, edgeCases, assumptions, statusEl) {
  setStep(statusEl, "running");
  const user = `Query: ${understanding.original_query}
Predicted outcome: ${thoughtExp.predicted_outcome} (confidence: ${thoughtExp.outcome_confidence})
Edge cases: ${(edgeCases || []).map(e => e.condition).join("; ")}
Flagged assumptions: ${(assumptions || []).join("; ")}`;
  const data = await groqChat(getSettings().fast, PROMPTS.followups, user, 0.3);
  setStep(statusEl, "done");
  return data.suggested_followups || [];
}

async function runPipeline() {
  const query = document.getElementById("query-input").value.trim();
  if (!query) return;
  if (!getAccessPassword()) {
    showError("Access password missing. Reload the page and unlock the app.");
    return;
  }

  const steps = [
    "Stage 2: Query understanding",
    "Stage 3: Hierarchy-aware retrieval",
    "Stage 4: Thought experiment (causal simulation)",
    "Stage 5: Edge case generation (adversarial)",
    "Stage 6: Building report"
  ];
  const stepEls = {};
  renderPipelineSteps(steps, stepEls);

  const runBtn = document.getElementById("run-btn");
  runBtn.disabled = true;
  hideError();
  document.getElementById("report-panel").hidden = true;

  try {
    await loadCorpus();
    setHeaderStats();

    const understanding = await stageUnderstanding(query, stepEls[0]);
    const retrieved = await stageRetrieval(understanding, stepEls[1]);
    const thoughtExp = await stageThoughtExperiment(understanding, retrieved, stepEls[2]);
    const edgeCases = await stageEdgeCases(understanding, thoughtExp, retrieved, stepEls[3]);

    const assumptions = [...(thoughtExp.flagged_assumptions || [])];
    for (const s of thoughtExp.causal_chain || []) {
      if (s.assumptions) for (const a of s.assumptions) if (!assumptions.includes(a)) assumptions.push(a);
    }

    const followups = await stageFollowups(understanding, thoughtExp, edgeCases, assumptions, stepEls[4]);

    const report = {
      query: understanding.original_query,
      queryType: understanding.query_type,
      understanding,
      retrieved,
      thoughtExp,
      edgeCases,
      assumptions,
      followups
    };
    renderReport(report);
    document.getElementById("report-panel").hidden = false;
  } catch (err) {
    for (const [i, el] of Object.entries(stepEls)) {
      if (el && el.dataset.state !== "done") setStep(el, "error");
    }
    showError(err.message || String(err));
  } finally {
    runBtn.disabled = false;
  }
}

/* ---------- Rendering ---------- */

function renderPipelineSteps(steps, stepEls) {
  const wrap = document.getElementById("pipeline-steps");
  wrap.innerHTML = "";
  const panel = document.getElementById("pipeline-panel");
  panel.hidden = false;
  steps.forEach((label, i) => {
    const el = document.createElement("div");
    el.className = "pstep";
    el.dataset.state = "pending";
    el.innerHTML = `<div class="dot"></div><div><div class="pstep-label">${label}</div><div class="pstep-detail"></div></div>`;
    wrap.appendChild(el);
    stepEls[i] = el;
  });
}

function setStep(el, state, detail) {
  if (!el) return;
  el.dataset.state = state;
  el.className = "pstep " + state;
  const dot = el.querySelector(".dot");
  if (state === "done") dot.textContent = "\u2713";
  if (state === "error") dot.textContent = "!";
  if (detail) el.querySelector(".pstep-detail").textContent = detail;
}

function esc(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

function levelBadge(lvl) {
  const cls = lvl.toLowerCase().startsWith("l0") ? "l0"
    : lvl.toLowerCase().startsWith("l1") ? "l1"
    : lvl.toLowerCase().startsWith("l2") ? "l2" : "l3";
  return `<span class="lvl-badge ${cls}">${esc(lvl.split("_")[0].toUpperCase())}</span>`;
}

function renderPrinciplesSection(retrieved) {
  const order = ["L0_axiom", "L1_mechanistic_pathway", "L2_context_modifier", "L3_known_exception"];
  const html = [];
  for (const lvl of order) {
    const ps = retrieved[lvl] || [];
    if (!ps.length) continue;
    const cls = lvl.toLowerCase().startsWith("l0") ? "l0" : lvl.toLowerCase().startsWith("l1") ? "l1" : lvl.toLowerCase().startsWith("l2") ? "l2" : "l3";
    html.push(`<div class="report-section"><h3>${LEVEL_LABELS[lvl]} (${ps.length})</h3>`);
    for (const p of ps) {
      html.push(`<div class="principle ${cls}">
        <div class="p-content">${esc(p.content)} ${levelBadge(p.hierarchy_level)}</div>
        <div class="p-meta">Entities: ${esc(p.entities.join(", "))}</div>
        <div class="p-cite">${esc(p.source_citation)}</div>
      </div>`);
    }
    html.push("</div>");
  }
  return html.join("");
}

function confBadge(conf) {
  const c = String(conf || "").toUpperCase();
  return `<span class="conf-badge conf-${esc(c)}">${esc(c)}</span>`;
}

function renderThoughtExperiment(te) {
  const html = [];
  html.push(`<div class="report-section"><h3>3. Thought Experiment</h3>`);
  html.push(`<div class="outcome-box"><strong>Initial conditions:</strong> ${esc(te.initial_conditions)}<br>
    <strong>Intervention:</strong> ${esc(te.intervention)}<br>
    <strong>Held constant:</strong> ${esc((te.held_constant || []).join(", "))}</div><br>`);

  for (const s of te.causal_chain || []) {
    const p = CORPUS.idx[s.principle_id];
    const levelTag = p ? levelBadge(p.hierarchy_level) : "";
    html.push(`<div class="step-card">
      <div class="step-head"><span>Step ${s.step_number}</span><span>${confBadge(s.confidence)}</span></div>
      <div class="step-body">
        <div><strong>Principle:</strong> ${esc(s.principle_content)} ${levelTag}</div>
        <div class="consequence"><strong>Mechanistic consequence:</strong> ${esc(s.mechanistic_consequence)}</div>
        <div class="cite">Citation: ${esc(s.citation || (p && p.source_citation))}</div>
      </div>
    </div>`);
  }

  html.push(`<div class="outcome-box"><strong>PREDICTED OUTCOME:</strong><br>${esc(te.predicted_outcome)}<br>
    <div class="conf">Confidence: ${confBadge(te.outcome_confidence)}
    ${te.confidence_justification ? `<span class="mono"> &mdash; ${esc(te.confidence_justification)}</span>` : ""}</div></div>`);
  html.push("</div>");
  return html.join("");
}

function renderEdgeCases(edgeCases) {
  if (!edgeCases || !edgeCases.length) {
    return `<div class="report-section"><h3>4. Edge Cases</h3><p class="mono">None identified across the retrieved conditions.</p></div>`;
  }
  const html = [`<div class="report-section"><h3>4. Edge Cases (${edgeCases.length})</h3>`];
  edgeCases.forEach((e, i) => {
    const sev = (e.severity || "qualifies").toLowerCase();
    html.push(`<div class="edge-card sev-${sev === "reverses" ? "reverses" : sev === "weakens" ? "weakens" : "qualifies"}">
      <span class="sev-badge ${sev === "reverses" ? "reverses" : sev === "weakens" ? "weakens" : "qualifies"}">${esc((e.severity || "qualifies").toUpperCase())}</span>
      <div class="e-cond">#${i + 1}. ${esc(e.condition)}</div>
      <div class="e-mech"><strong>Why it changes:</strong> ${esc(e.mechanism_of_deviation)}</div>
      <div class="e-cite">${esc(e.hierarchy_level || "")} &middot; ${esc(e.citation || "")}</div>
    </div>`);
  });
  html.push("</div>");
  return html.join("");
}

function renderAssumptions(assumptions) {
  const html = [`<div class="report-section"><h3>5. Flagged Assumptions / Underspecified Variables</h3>`];
  if (assumptions && assumptions.length) {
    html.push(`<ul class="assume-list">${assumptions.map(a => `<li>${esc(a)}</li>`).join("")}</ul>`);
  } else {
    html.push(`<p class="mono">None flagged.</p>`);
  }
  html.push("</div>");
  return html.join("");
}

function renderFollowups(followups) {
  const html = [`<div class="report-section"><h3>6. Suggested Follow-Up Questions / Experiments</h3>`];
  if (followups && followups.length) {
    html.push(`<ol class="followup-list">${followups.map(f => `<li>${esc(f)}</li>`).join("")}</ol>`);
  } else {
    html.push(`<p class="mono">None suggested.</p>`);
  }
  html.push("</div>");
  return html.join("");
}

function shortLevel(lvl) {
  return String(lvl || "").split("_")[0].toUpperCase();
}

function mdEscape(s) {
  if (s == null) return "";
  return String(s).replace(/\|/g, "\\|").replace(/\r?\n/g, " ").trim();
}

function buildReportMarkdown(report) {
  const md = [];
  const typeLabels = {
    mechanistic: "Mechanistic",
    therapeutic_hypothesis: "Therapeutic hypothesis",
    edge_case_exploration: "Edge-case exploration",
    comparative: "Comparative"
  };
  const now = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";

  md.push("# Re-Matrix Reasoning Report");
  md.push("");
  md.push(`**Generated:** ${now}`);
  md.push("");
  md.push(`**Query:** ${mdEscape(report.query)}`);
  md.push("");

  md.push("## 1. Query Restatement");
  md.push("");
  md.push(mdEscape(report.query));
  md.push("");
  md.push(`**Classified as:** ${typeLabels[report.queryType] || report.queryType}`);
  md.push("");
  md.push(`**Causal question:** ${mdEscape(report.understanding.causal_question || "")}`);
  md.push("");

  md.push("## 2. Principles Invoked");
  const order = ["L0_axiom", "L1_mechanistic_pathway", "L2_context_modifier", "L3_known_exception"];
  for (const lvl of order) {
    const ps = (report.retrieved && report.retrieved[lvl]) || [];
    if (!ps.length) continue;
    md.push("");
    md.push(`### ${LEVEL_LABELS[lvl]}`);
    md.push("");
    for (const p of ps) {
      md.push(`- **${mdEscape(p.content)}** (${shortLevel(p.hierarchy_level)})`);
      md.push(`  - Entities: ${mdEscape(p.entities.join(", "))}`);
      md.push(`  - Source: ${mdEscape(p.source_citation)}`);
    }
  }
  md.push("");

  const te = report.thoughtExp || {};
  md.push("## 3. Thought Experiment");
  md.push("");
  md.push(`- **Initial conditions:** ${mdEscape(te.initial_conditions)}`);
  md.push(`- **Intervention:** ${mdEscape(te.intervention)}`);
  md.push(`- **Held constant:** ${mdEscape((te.held_constant || []).join(", "))}`);
  md.push("");
  md.push("### Causal chain");
  md.push("");
  for (const s of te.causal_chain || []) {
    const p = CORPUS.idx[s.principle_id];
    const lvl = shortLevel((p && p.hierarchy_level) || "");
    const cite = s.citation || (p && p.source_citation) || "";
    md.push(`1. **${mdEscape(s.principle_content)}** (${lvl})`);
    md.push(`   - Mechanistic consequence: ${mdEscape(s.mechanistic_consequence)}`);
    md.push(`   - Confidence: ${mdEscape(s.confidence)}`);
    if (cite) md.push(`   - Citation: ${mdEscape(cite)}`);
  }
  md.push("");
  md.push(`**PREDICTED OUTCOME:** ${mdEscape(te.predicted_outcome)}`);
  md.push("");
  md.push(`**Outcome confidence:** ${mdEscape(te.outcome_confidence)}`);
  if (te.confidence_justification) md.push(`\n**Confidence justification:** ${mdEscape(te.confidence_justification)}`);
  md.push("");

  const edgeCases = report.edgeCases || [];
  md.push("## 4. Edge Cases");
  md.push("");
  if (!edgeCases.length) {
    md.push("None identified across the retrieved conditions.");
  } else {
    edgeCases.forEach((e, i) => {
      md.push(`${i + 1}. **${mdEscape((e.severity || "qualifies").toUpperCase())}** ${mdEscape(e.condition)}`);
      md.push(`   - Why it changes: ${mdEscape(e.mechanism_of_deviation)}`);
      const ecite = [e.hierarchy_level, e.citation].filter(Boolean).join(" · ");
      if (ecite) md.push(`   - ${mdEscape(ecite)}`);
    });
  }
  md.push("");

  const assumptions = report.assumptions || [];
  md.push("## 5. Flagged Assumptions / Underspecified Variables");
  md.push("");
  if (assumptions.length) {
    assumptions.forEach((a) => md.push(`- ${mdEscape(a)}`));
  } else {
    md.push("None flagged.");
  }
  md.push("");

  const followups = report.followups || [];
  md.push("## 6. Suggested Follow-Up Questions / Experiments");
  md.push("");
  if (followups.length) {
    followups.forEach((f, i) => md.push(`${i + 1}. ${mdEscape(f)}`));
  } else {
    md.push("None suggested.");
  }
  md.push("");

  return md.join("\n");
}

function renderReport(report) {
  lastReport = report;
  const body = document.getElementById("report-body");
  const typeLabels = {
    mechanistic: "Mechanistic",
    therapeutic_hypothesis: "Therapeutic hypothesis",
    edge_case_exploration: "Edge-case exploration",
    comparative: "Comparative"
  };
  const html = [];
  html.push(`<div class="report-section"><h3>1. Query Restatement</h3>
    <p>${esc(report.query)}</p>
    <p class="mono">Classified as: ${esc(typeLabels[report.queryType] || report.queryType)}</p>
    <p class="mono">Causal question: ${esc(report.understanding.causal_question || "")}</p></div>`);
  html.push(`<div class="report-section"><h3>2. Principles Invoked</h3>${renderPrinciplesSection(report.retrieved)}</div>`);
  html.push(renderThoughtExperiment(report.thoughtExp));
  html.push(renderEdgeCases(report.edgeCases));
  html.push(renderAssumptions(report.assumptions));
  html.push(renderFollowups(report.followups));
  body.innerHTML = html.join("");
}

function setHeaderStats() {
  const stats = document.getElementById("header-stats");
  const counts = {};
  for (const p of CORPUS.principles) counts[p.hierarchy_level] = (counts[p.hierarchy_level] || 0) + 1;
  stats.innerHTML = `Corpus: ${CORPUS.principles.length} principles &middot; ` +
    `L0: ${counts["L0_axiom"] || 0} &middot; L1: ${counts["L1_mechanistic_pathway"] || 0} &middot; ` +
    `L2: ${counts["L2_context_modifier"] || 0} &middot; L3: ${counts["L3_known_exception"] || 0}`;
}

function showError(msg) {
  document.getElementById("error-panel").hidden = false;
  document.getElementById("error-body").textContent = msg;
}

function hideError() {
  document.getElementById("error-panel").hidden = true;
}

/* ---------- Init ---------- */

document.addEventListener("DOMContentLoaded", async () => {
  // Access gate
  const gate = document.getElementById("gate");
  const gateForm = document.getElementById("gate-form");
  const gatePass = document.getElementById("gate-password");
  const gateError = document.getElementById("gate-error");

  function unlock() {
    sessionStorage.setItem(SESSION_KEY, "1");
    sessionStorage.setItem(PASS_KEY, gatePass.value.trim());
    gate.hidden = true;
    document.getElementById("query-input").focus();
  }

  if (sessionStorage.getItem(SESSION_KEY) === "1" && getAccessPassword()) {
    gate.hidden = true;
  } else {
    gateForm.addEventListener("submit", (e) => {
      e.preventDefault();
      if (gatePass.value.trim() === APP_PASSWORD_HINT) unlock();
      else {
        gateError.hidden = false;
        gatePass.value = "";
        gatePass.focus();
      }
    });
    gatePass.focus();
  }

  // Toggle settings
  const toggleBtn = document.getElementById("toggle-settings");
  const settingsBody = document.getElementById("settings-body");
  toggleBtn.addEventListener("click", () => {
    const hidden = settingsBody.hidden;
    settingsBody.hidden = !hidden;
    toggleBtn.textContent = hidden ? "Hide" : "Show";
  });

  // Test connection (via proxy)
  document.getElementById("test-key").addEventListener("click", async () => {
    const status = document.getElementById("key-status");
    status.className = "status-text";
    status.textContent = "Testing...";
    try {
      const password = getAccessPassword();
      const resp = await fetch(PROXY_URL + "/ping", {
        method: "GET",
        headers: { "X-App-Password": password }
      });
      if (!resp.ok) throw new Error("Proxy rejected: " + resp.status);
      const data = await resp.json();
      status.className = "status-text ok";
      status.textContent = data.ok ? "Connection OK" : "Proxy error";
    } catch (e) {
      status.className = "status-text err";
      status.textContent = "Failed: " + (e.message || "error");
    }
  });

  // Run button
  document.getElementById("run-btn").addEventListener("click", runPipeline);

  // Example chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("query-input").value = chip.dataset.q;
    });
  });

  // Copy report as a formatted Markdown document
  const copyBtn = document.getElementById("copy-report");
  copyBtn.addEventListener("click", async () => {
    if (!lastReport) return;
    const doc = buildReportMarkdown(lastReport);
    try {
      await navigator.clipboard.writeText(doc);
      copyBtn.textContent = "Copied";
      setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
    } catch (e) { /* ignore */ }
  });

  // Download report as a .md file
  document.getElementById("download-report").addEventListener("click", () => {
    if (!lastReport) return;
    const doc = buildReportMarkdown(lastReport);
    const slug = (lastReport.query || "report").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
    const blob = new Blob([doc], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `re-matrix-report-${slug || "report"}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // Load corpus for header stats
  try {
    await loadCorpus();
    setHeaderStats();
  } catch (e) { /* corpus will load on run */ }
});
