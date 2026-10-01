/* ASTRA INTEL — console logic.
   No framework, no build step. The API already does the grounding work, so the
   job here is: render the answer faithfully, keep the citation markers clickable,
   and maintain the KPI counters. */

"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};

const state = {
  queries: 0,
  grounded: 0,
  confidenceSum: 0,
  files: [],
};

/* ── status vocabulary ─────────────────────────────────────────────── */

const STATUS = {
  grounded:             { label: "Grounded",             cls: "badge-ok" },
  partially_grounded:   { label: "Partial",              cls: "badge-warn" },
  ungrounded:           { label: "Ungrounded",           cls: "badge-danger" },
  no_context:           { label: "No Evidence",          cls: "badge-danger" },
};

/* ── tiny markdown renderer ─────────────────────────────────────────
   The model already emits [S1] markers and light markdown. A full parser is
   out of scope, so this handles the subset that matters: paragraphs, bullets,
   bold/italic/code, and citation markers. Anything else is escaped and shown
   literally rather than injected as HTML. */

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  // [S1] / [S2, S3] become hoverable markers.
  html = html.replace(/\[(S\s?\d{1,3}(?:\s*,\s*S?\s?\d{1,3})*)\]/g, (match, body) => {
    const ids = body.split(",").map((s) => s.replace(/\s+/g, "").toUpperCase());
    return ids
      .map((id) => `<span class="cite-marker" title="Source ${id}">${id}</span>`)
      .join(" ");
  });
  return html;
}

function renderMarkdown(source) {
  const blocks = [];
  const lines = String(source || "").split(/\r?\n/);
  let list = null;

  const closeList = () => { if (list) { blocks.push(`</${list}>`); list = null; } };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }

    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      if (list !== "ul") { closeList(); blocks.push("<ul>"); list = "ul"; }
      blocks.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (numbered) {
      if (list !== "ol") { closeList(); blocks.push("<ol>"); list = "ol"; }
      blocks.push(`<li>${inline(numbered[1])}</li>`);
      continue;
    }
    closeList();
    blocks.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return blocks.join("");
}

/* ── toasts ────────────────────────────────────────────────────────── */

function toast(message, kind = "") {
  const node = el("div", `toast ${kind}`.trim(), message);
  $("#toastHost").appendChild(node);
  setTimeout(() => node.remove(), 5200);
}

/* ── network ───────────────────────────────────────────────────────── */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch (_) { /* empty body */ }
  if (!response.ok) {
    throw new Error((payload && payload.detail) || `Request failed (${response.status})`);
  }
  return payload;
}

/* ── KPI + register ────────────────────────────────────────────────── */

function paintKpis() {
  $("#kpiQueries").textContent = state.queries;
  const answered = state.grounded + state.confidenceCount || 0;
  $("#kpiGrounded").textContent = answered ? `${Math.round((state.grounded / answered) * 100)}%` : "—";
  $("#kpiConfidence").textContent = answered
    ? (state.confidenceSum / answered).toFixed(2)
    : "—";
}

function paintRegister(documents) {
  const body = $("#docBody");
  body.replaceChildren();
  if (!documents.length) {
    const row = el("tr", "grid-empty");
    const cell = el("td", null, "No documents indexed.");
    cell.colSpan = 5;
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  documents.forEach((doc) => {
    const row = el("tr");

    const title = el("td", "ttl");
    title.appendChild(document.createTextNode(doc.title));
    title.appendChild(el("span", "sub", doc.filename));
    row.appendChild(title);

    row.appendChild(el("td", "num", doc.page_count));
    row.appendChild(el("td", "num", doc.chunk_count));

    const status = el("td");
    const live = doc.chunk_count > 0;
    status.appendChild(el("span", `badge ${live ? "badge-ok" : "badge-muted"}`,
      live ? "INDEXED" : "EMPTY"));
    row.appendChild(status);

    const actions = el("td");
    const drop = el("button", "btn btn-ghost", "Remove");
    drop.title = `Remove ${doc.title} from the index`;
    drop.addEventListener("click", async () => {
      drop.disabled = true;
      try {
        await api(`/api/documents/${encodeURIComponent(doc.document_id)}`, { method: "DELETE" });
        toast(`Removed "${doc.title}".`, "ok");
        await refresh();
      } catch (err) {
        toast(err.message, "err");
        drop.disabled = false;
      }
    });
    actions.appendChild(drop);
    row.appendChild(actions);

    body.appendChild(row);
  });
}

async function refresh() {
  const data = await api("/api/state");
  const stats = data.stats || {};

  $("#kpiDocs").textContent = stats.documents ?? 0;
  $("#kpiChunks").textContent = stats.chunks ?? 0;
  const avg = stats.documents ? Math.round((stats.chunks || 0) / stats.documents) : 0;
  $("#kpiDocsFoot").textContent = stats.documents ? `avg ${avg} passages each` : "index is empty";
  $("#kpiChunksFoot").textContent = `${stats.vectors ?? 0} vectors stored`;

  $("#metaProvider").textContent = stats.llm_provider ?? "—";
  $("#metaModel").textContent = stats.llm_model ?? "—";
  $("#metaEmbed").textContent = (stats.embedding_model ?? "—").split("/").pop();

  const badge = $("#llmBadge");
  badge.className = `badge ${data.llm_ready ? "badge-ok" : "badge-warn"}`;
  badge.textContent = data.llm_ready ? "MODEL ONLINE" : "EVIDENCE ONLY";

  paintRegister(data.documents || []);
  paintKpis();
  return data;
}

/* ── suggestions ───────────────────────────────────────────────────── */

const FALLBACK_QUESTIONS = [
  "What are the main categories or classes of UAVs described in the documents?",
  "How does the Electronic warfare document distinguish jamming from deception?",
  "Compare unmanned aerial and unmanned ground vehicles: what roles do they share?",
];

async function loadSuggestions() {
  const host = $("#suggestions");
  host.replaceChildren();
  let questions = FALLBACK_QUESTIONS;
  try {
    const response = await fetch("/api/suggestions");
    if (response.ok) {
      const payload = await response.json();
      if (Array.isArray(payload.questions) && payload.questions.length) questions = payload.questions;
    }
  } catch (_) { /* keep the fallbacks */ }

  questions.slice(0, 8).forEach((question) => {
    const button = el("button", null, question.length > 78 ? `${question.slice(0, 75)}…` : question);
    button.title = question;
    button.addEventListener("click", () => {
      $("#queryInput").value = question;
      $("#queryInput").focus();
    });
    host.appendChild(button);
  });
}

/* ── console log ───────────────────────────────────────────────────── */

function renderCitations(citations) {
  const wrap = el("div", "evidence");
  wrap.appendChild(Object.assign(el("div", "evidence-head"), {}));
  wrap.lastChild.appendChild(el("span", null, "Evidence Register"));
  wrap.lastChild.appendChild(el("span", null, `${citations.length} passage(s)`));

  citations.forEach((citation) => {
    const card = el("div", "cite");
    const top = el("div", "cite-top");
    top.appendChild(el("span", "cite-mk", `[${citation.marker || "S"}]`));
    top.appendChild(el("span", "cite-title", citation.title));
    top.appendChild(el("span", "cite-where",
      `page ${citation.page}${citation.section ? ` · ${citation.section}` : ""}`));
    top.appendChild(el("span", "cite-score", Number(citation.score).toFixed(3)));
    card.appendChild(top);
    card.appendChild(el("div", "cite-text", citation.excerpt));
    wrap.appendChild(card);
  });
  return wrap;
}

function renderRetrieved(retrieved) {
  if (!retrieved || !retrieved.length) return null;
  const details = el("details", "retrieved");
  details.appendChild(el("summary", null, `Retrieved passages (${retrieved.length})`));
  const table = el("table");
  const tbody = el("tbody");
  retrieved.forEach((item) => {
    const row = el("tr");
    row.appendChild(el("td", "mono", Number(item.score).toFixed(3)));
    const cell = el("td");
    cell.textContent = `${item.chunk.title} — p.${item.chunk.page}`
      + (item.chunk.section ? ` · ${item.chunk.section}` : "");
    row.appendChild(cell);
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  details.appendChild(table);
  return details;
}

function appendEntry({ question, badge, meta, answerHtml, citations, retrieved, notes }) {
  const log = $("#consoleLog");
  const placeholder = log.querySelector(".empty-state");
  if (placeholder) placeholder.remove();

  const entry = el("div", "entry");

  const head = el("div", "entry-head");
  head.appendChild(el("span", "entry-q", question));
  const right = el("div", "entry-meta");
  if (badge) right.appendChild(badge);
  right.appendChild(el("span", null, meta));
  head.appendChild(right);
  entry.appendChild(head);

  const body = el("div", "entry-body");
  const answer = el("div", "answer");
  answer.innerHTML = answerHtml;
  body.appendChild(answer);

  (notes || []).forEach((note) => body.appendChild(el("div", "note", note)));

  if (citations && citations.length) body.appendChild(renderCitations(citations));

  const extras = renderRetrieved(retrieved);
  if (extras) body.appendChild(extras);

  if (citations && citations.length) {
    const actions = el("div", "entry-actions");
    const copy = el("button", "btn btn-ghost", "Copy with sources");
    copy.addEventListener("click", async () => {
      const text = [question, "", answer.textContent, "", ...citations.map(
        (c) => `[${c.marker || "S"}] ${c.title}, page ${c.page}\n${c.excerpt}`)].join("\n");
      try {
        await navigator.clipboard.writeText(text);
        toast("Copied answer and sources to the clipboard.", "ok");
      } catch (_) {
        toast("Clipboard blocked by the browser.", "err");
      }
    });
    actions.appendChild(copy);
    body.appendChild(actions);
  }

  entry.appendChild(body);
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function statusBadge(status) {
  const spec = STATUS[status] || { label: status || "unknown", cls: "badge-muted" };
  return el("span", `badge ${spec.cls}`, spec.label);
}

/* ── actions ───────────────────────────────────────────────────────── */

async function submit(path, question, label) {
  const trimmed = question.trim();
  if (!trimmed) { toast("Enter a question first.", "err"); return; }

  document.body.classList.add("busy");
  const log = $("#consoleLog");
  const placeholder = log.querySelector(".empty-state");
  if (placeholder) placeholder.remove();
  const spinner = el("div", "spinner-row", label);
  log.appendChild(spinner);
  log.scrollTop = log.scrollHeight;

  const started = performance.now();
  try {
    const data = await api(path, { method: "POST", body: JSON.stringify({ question: trimmed }) });
    spinner.remove();
    const seconds = ((performance.now() - started) / 1000).toFixed(1);

    if (path === "/api/ask") {
      state.queries += 1;
      state.confidenceCount = (state.confidenceCount || 0) + 1;
      state.confidenceSum += Number(data.confidence || 0);
      if (data.status === "grounded") state.grounded += 1;

      appendEntry({
        question: trimmed,
        badge: statusBadge(data.status),
        meta: `conf ${Number(data.confidence).toFixed(2)} · ${seconds}s`
          + (data.llm_model ? ` · ${data.llm_model}` : ""),
        answerHtml: renderMarkdown(data.answer),
        citations: data.citations,
        retrieved: data.retrieved,
        notes: data.notes,
      });
    } else {
      state.queries += 1;
      appendEntry({
        question: trimmed,
        badge: el("span", "badge badge-info", "EVIDENCE"),
        meta: `${(data.citations || []).length} passage(s) · ${seconds}s · no model call`,
        answerHtml: `<p>Retrieval only — the corpus was searched for passages above the
          relevance threshold without calling the language model.</p>`,
        citations: data.citations,
      });
    }
    paintKpis();
  } catch (err) {
    spinner.remove();
    toast(err.message, "err");
  } finally {
    document.body.classList.remove("busy");
  }
}

/* ── wiring ────────────────────────────────────────────────────────── */

function wire() {
  $("#queryForm").addEventListener("submit", (event) => {
    event.preventDefault();
    submit("/api/ask", $("#queryInput").value, "Retrieving and composing a grounded answer…");
  });

  $("#btnEvidence").addEventListener("click", () => {
    submit("/api/evidence", $("#queryInput").value, "Retrieving passages…");
  });

  $("#btnClearChat").addEventListener("click", () => {
    $("#consoleLog").replaceChildren();
    state.queries = 0; state.grounded = 0;
    state.confidenceSum = 0; state.confidenceCount = 0;
    const empty = el("div", "empty-state");
    empty.appendChild(el("h3", null, "No queries logged"));
    empty.appendChild(el("p", null,
      "Every response is verified against the retrieved pages. If the indexed "
      + "documents do not cover a question, the console reports that rather "
      + "than composing an unsupported answer."));
    $("#consoleLog").appendChild(empty);
    paintKpis();
  });

  $("#fileInput").addEventListener("change", (event) => {
    state.files = Array.from(event.target.files || []);
    $("#btnIndex").disabled = state.files.length === 0;
  });

  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault(); dropzone.classList.add("drag");
    }));
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault(); dropzone.classList.remove("drag");
    }));
  dropzone.addEventListener("drop", (event) => {
    state.files = Array.from(event.dataTransfer.files || []);
    $("#btnIndex").disabled = state.files.length === 0;
    if (state.files.length) toast(`${state.files.length} file(s) staged.`);
  });

  $("#btnIndex").addEventListener("click", async () => {
    if (!state.files.length) return;
    document.body.classList.add("busy");
    try {
      const form = new FormData();
      state.files.forEach((file) => form.append("files", file));
      const data = await api("/api/upload", { method: "POST", body: form });
      const added = (data.added || []).length;
      if (added) toast(`Indexed ${added} document(s), +${data.chunk_count} passages.`, "ok");
      (data.duplicates || []).forEach((name) => toast(`Already indexed: ${name}`));
      Object.entries(data.failures || {}).forEach(([name, why]) => toast(`${name}: ${why}`, "err"));
      state.files = [];
      $("#fileInput").value = "";
      $("#btnIndex").disabled = true;
      await refresh();
    } catch (err) {
      toast(err.message, "err");
    } finally {
      document.body.classList.remove("busy");
    }
  });

  $("#btnRebuild").addEventListener("click", async () => {
    document.body.classList.add("busy");
    try {
      const data = await api("/api/rebuild", { method: "POST" });
      toast(`Rebuilt sample index: ${(data.added || []).length} document(s), +${data.chunk_count} passages.`, "ok");
      await refresh();
    } catch (err) {
      toast(err.message, "err");
    } finally {
      document.body.classList.remove("busy");
    }
  });

  $("#btnWipe").addEventListener("click", async () => {
    if (!window.confirm("Remove every indexed document? This cannot be undone.")) return;
    try {
      await api("/api/clear", { method: "POST" });
      toast("Index wiped.", "ok");
      await refresh();
    } catch (err) {
      toast(err.message, "err");
    }
  });

  $("#btnBriefings").addEventListener("click", async () => {
    const button = $("#btnBriefings");
    button.disabled = true;
    button.textContent = "Generating…";
    try {
      const data = await api("/api/briefings", { method: "POST" });
      const host = $("#briefingList");
      host.replaceChildren();
      (data.summaries || []).forEach((item) => {
        const card = el("div", "brief-item");
        card.appendChild(el("h4", null, item.title));
        card.appendChild(el("p", null, item.summary));
        if (item.topics && item.topics.length) {
          card.appendChild(el("div", "brief-topics",
            `Sections: ${item.topics.slice(0, 8).join(", ")}`));
        }
        host.appendChild(card);
      });
      toast(`Generated ${(data.summaries || []).length} briefing(s).`, "ok");
    } catch (err) {
      toast(err.message, "err");
    } finally {
      button.disabled = false;
      button.textContent = "Generate briefings";
    }
  });
}

(async function start() {
  wire();
  await loadSuggestions();
  try {
    await refresh();
  } catch (err) {
    toast(`Could not reach the API: ${err.message}`, "err");
  }
})();