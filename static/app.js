/* ASTRA INTEL - corpus operations console.
   Vanilla ES2020, no build step and no CDN: charts are hand-drawn SVG so the
   console still works on an isolated network where Chart.js could not load. */

const $ = (id) => document.getElementById(id);

const STATUS = {
  grounded:            { label: "Grounded",      cls: "badge-grounded", color: "#38a169" },
  partially_grounded:  { label: "Partial",       cls: "badge-warn",     color: "#ed8936" },
  no_context:          { label: "No Context",    cls: "badge-no_context", color: "#e53e3e" },
  error:               { label: "Failed",        cls: "badge-error",    color: "#e53e3e" },
};

const state = {
  documents: [],
  settings: {},
  llmReady: false,
  stats: {},
  log: [],
  filter: "all",
  sort: { key: "chunk_count", dir: -1 },
  selected: null,
};

// ------------------------------------------------------------------ utils --

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* non-JSON error body; keep the status text */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

const jsonPost = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

function toast(message, kind = "") {
  const node = document.createElement("div");
  node.className = "toast " + kind;
  node.textContent = message;
  $("toasts").appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function badge(status) {
  const meta = STATUS[status] || { label: status || "unknown", cls: "badge-mute" };
  return `<span class="badge ${meta.cls}">${esc(meta.label)}</span>`;
}

/* Minimal markdown: the model answers with short prose plus bullet lists, so
   bold and inline code are the only inline marks worth handling. Escaping
   happens first, which keeps this safe against injected HTML. */
function md(text) {
  const safe = esc(text || "");
  return safe
    .split(/\n{2,}/)
    .map((block) => {
      const lines = block.split("\n").filter((l) => l.trim());
      if (!lines.length) return "";
      if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
        return "<ul>" + lines.map((l) =>
          "<li>" + inline(l.replace(/^\s*[-*]\s+/, "")) + "</li>").join("") + "</ul>";
      }
      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        return "<ol>" + lines.map((l) =>
          "<li>" + inline(l.replace(/^\s*\d+[.)]\s+/, "")) + "</li>").join("") + "</ol>";
      }
      return "<p>" + lines.map(inline).join("<br>") + "</p>";
    })
    .join("");
}

function inline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])_(.+?)_(?=$|[\s.,;:)])/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Citation markers, normalised server-side, rendered as real superscripts.
    .replace(/【\s*S(\d+)\s*】|\[\s*S(\d+)\s*\]/g,
      (_, a, b) => `<sup class="cite-ref">S${a || b}</sup>`);
}

// ------------------------------------------------------------------- donut --

function drawDonut() {
  const svg = $("donut");
  const total = state.log.length;
  const counts = { grounded: 0, partially_grounded: 0, no_context: 0, error: 0 };
  for (const row of state.log) {
    if (counts[row.status] === undefined) counts[row.status] = 0;
    counts[row.status] += 1;
  }

  if (!total) {
    svg.innerHTML =
      '<circle cx="100" cy="100" r="70" fill="none" stroke="#edf2f7" stroke-width="26"/>' +
      '<text x="100" y="97" text-anchor="middle" fill="#718096" font-size="15" font-weight="600">0</text>' +
      '<text x="100" y="116" text-anchor="middle" fill="#a0aec0" font-size="10">queries</text>';
    $("donut-legend").innerHTML = "";
    return;
  }

  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  let rings = "";
  Object.keys(STATUS).forEach((key) => {
    const count = counts[key] || 0;
    if (!count) return;
    const length = (count / total) * circumference;
    rings += `<circle cx="100" cy="100" r="${radius}" fill="none"
      stroke="${STATUS[key].color}" stroke-width="26"
      stroke-dasharray="${length} ${circumference - length}"
      stroke-dashoffset="${-offset}" transform="rotate(-90 100 100)">
      <title>${STATUS[key].label}: ${count}</title></circle>`;
    offset += length;
  });

  svg.innerHTML = rings +
    `<text x="100" y="99" text-anchor="middle" fill="#1a202c" font-size="24" font-weight="700">${total}</text>` +
    `<text x="100" y="117" text-anchor="middle" fill="#718096" font-size="10">queries</text>`;

  $("donut-legend").innerHTML = Object.keys(STATUS)
    .filter((key) => counts[key])
    .map((key) => `<li><span class="sw" style="background:${STATUS[key].color}"></span>
        ${STATUS[key].label}<span class="n">${counts[key]}</span></li>`)
    .join("");
}

// -------------------------------------------------------------------- bars --

function drawBars() {
  const rows = [...state.documents].sort((a, b) => b.chunk_count - a.chunk_count);
  if (!rows.length) {
    $("bars").innerHTML = '<p class="muted tiny">No documents indexed.</p>';
    return;
  }
  const max = Math.max(...rows.map((d) => d.chunk_count), 1);
  $("bars").innerHTML = rows.map((d) => {
    const ratio = d.chunk_count / max;
    // Same 70%-of-capability read as a bin fill level, but relative to the
    // largest document rather than an invented capacity.
    const tier = ratio >= 0.7 ? "t-high" : ratio >= 0.3 ? "t-mid" : "t-low";
    return `<div class="bar-row">
      <span class="name" title="${esc(d.title)}">${esc(d.title)}</span>
      <span class="bar-track"><span class="bar-fill ${tier}" style="width:${(ratio * 100).toFixed(1)}%"></span></span>
      <span class="val">${d.chunk_count}</span>
    </div>`;
  }).join("");
}

// -------------------------------------------------------------- document status --

function docStatus(doc) {
  if (!doc.chunk_count) return { key: "none", label: "Unindexed", cls: "badge-no_context" };
  const coverage = doc.page_count ? doc.indexed_pages / doc.page_count : 1;
  if (coverage >= 1) return { key: "full", label: "Fully indexed", cls: "badge-grounded" };
  if (coverage >= 0.6) return { key: "partial", label: "Part indexed", cls: "badge-warn" };
  return { key: "sparse", label: "Sparse", cls: "badge-mute" };
}

// ---------------------------------------------------------------- register --

function drawRegister() {
  const body = $("register-table").tBodies[0];
  const key = state.sort.key;
  const rows = [...state.documents].sort((a, b) => {
    if (key === "coverage") {
      const ca = a.page_count ? a.indexed_pages / a.page_count : 0;
      const cb = b.page_count ? b.indexed_pages / b.page_count : 0;
      return (ca - cb) * state.sort.dir;
    }
    const av = a[key] ?? 0;
    const bv = b[key] ?? 0;
    if (typeof av === "string") return av.localeCompare(bv) * state.sort.dir;
    return (av - bv) * state.sort.dir;
  });

  $("register-count").textContent =
    `${state.documents.length} document${state.documents.length === 1 ? "" : "s"}`;

  if (!rows.length) {
    body.innerHTML = '<tr class="empty"><td colspan="6">The index is empty. Ingest a PDF to begin.</td></tr>';
    return;
  }

  body.innerHTML = rows.map((d) => {
    const meta = docStatus(d);
    const coverage = d.page_count ? d.indexed_pages / d.page_count : 0;
    const pct = (coverage * 100).toFixed(0);
    return `<tr data-doc="${esc(d.document_id)}" class="${state.selected === d.document_id ? "is-selected" : ""}">
      <td>
        <div class="cell-title">${esc(d.title)}</div>
        <div class="cell-sub">${esc(d.filename)}</div>
      </td>
      <td class="num">${d.chunk_count}</td>
      <td class="num">${d.page_count}</td>
      <td>
        <span class="coverage">
          <span class="bar-track"><span class="bar-fill ${pct >= 99 ? "t-high" : "t-mid"}" style="width:${pct}%"></span></span>
          <span class="pct">${pct}%</span>
        </span>
      </td>
      <td><span class="badge ${meta.cls}">${esc(meta.label)}</span></td>
      <td class="num">
        <button class="btn btn-ghost btn-mini btn-danger-ghost" data-remove="${esc(d.document_id)}">Remove</button>
      </td>
    </tr>`;
  }).join("");

  document.querySelectorAll("#register-table thead th[data-sort]").forEach((th) => {
    th.classList.toggle("is-sorted", th.dataset.sort === state.sort.key);
  });
}

// ------------------------------------------------------------- corpus grid --

function drawCorpus() {
  const grid = $("corpus-grid");
  const legend = $("scale-legend");
  if (!state.documents.length) {
    grid.innerHTML = '<p class="muted tiny">Nothing indexed yet.</p>';
    return "";
  }

  // A single scale across every document, so a pale cell in a small document
  // means the same thing as a pale cell in a large one.
  const peak = Math.max(
    1,
    ...state.documents.flatMap((d) => (d.pages || []).map((p) => p.passages)),
  );
  const steps = [0.08, 0.3, 0.55, 0.78, 1].map((f) => {
    const mix = Math.round(f * 255);
    return `<i style="background:rgba(49,130,206,${0.1 + f * 0.85})"></i>`;
  }).join("");
  legend.innerHTML = `<span>1 passage</span>
    <span class="scale-steps">${steps}</span>
    <span>${peak} passages / page</span>
    <span class="muted">&middot; each square is one page</span>`;

  grid.innerHTML = state.documents.map((d) => {
    const cells = (d.pages || []).map((p) => {
      const alpha = 0.1 + (p.passages / peak) * 0.85;
      const hint = p.passages
        ? ` data-hint title="${esc(d.title)} p.${p.page} - ${p.passages} passage${p.passages === 1 ? "" : "s"}"`
        : ` title="${esc(d.title)} p.${p.page} - no passages extracted"`;
      return `<span class="cell" style="background:rgba(49,130,206,${alpha.toFixed(2)})"${hint}></span>`;
    }).join("");
    return `<div class="corpus-row">
      <span class="corpus-label" title="${esc(d.title)}">${esc(d.title)}</span>
      <span class="corpus-cells">${cells}</span>
    </div>`;
  }).join("");
}

// ------------------------------------------------------------ ops / config --

function drawOps() {
  const panel = $("ops-panel");
  const doc = state.documents.find((d) => d.document_id === state.selected);
  if (!doc) {
    panel.innerHTML = '<p class="muted">Select a row in the register to inspect its density profile.</p>';
    return;
  }
  const pages = doc.pages || [];
  const withPassages = pages.filter((p) => p.passages > 0);
  const peak = Math.max(1, ...pages.map((p) => p.passages));
  const busiest = pages.slice().sort((a, b) => b.passages - a.passages).slice(0, 5);

  panel.innerHTML = `
    <h3>${esc(doc.title)}</h3>
    <dl>
      <dt>Pages parsed</dt><dd>${doc.page_count}</dd>
      <dt>Pages with passages</dt><dd>${withPassages.length}</dd>
      <dt>Passages</dt><dd>${doc.chunk_count}</dd>
      <dt>Mean passages / page</dt><dd>${doc.page_count ? (doc.chunk_count / doc.page_count).toFixed(1) : "0"}</dd>
      <dt>Peak page density</dt><dd>${peak}</dd>
    </dl>
    <h3>Densest pages</h3>
    <ul class="key">${busiest.map((p) => `<li>
        <span class="sw" style="background:rgba(49,130,206,${(0.1 + (p.passages / peak) * 0.85).toFixed(2)})"></span>
        Page ${p.page}<span style="margin-left:auto;font-variant-numeric:tabular-nums">${p.passages}</span>
      </li>`).join("")}</ul>`;
}

function drawConfig() {
  $("config-json").textContent = JSON.stringify({
    settings: state.settings,
    stats: state.stats,
    llm_ready: state.llmReady,
  }, null, 2);
}

function drawKpis() {
  // The store reports `chunks`; fall back to the per-document sum so the KPI
  // still reads correctly if that key ever changes or is absent.
  const summed = state.documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
  const chunks = state.stats.chunks ?? state.stats.chunk_count ?? summed;
  const pages = state.documents.reduce((sum, d) => sum + (d.page_count || 0), 0);
  $("kpi-documents").textContent = state.documents.length;
  $("kpi-documents-foot").textContent = `${pages} pages parsed`;
  $("kpi-passages").textContent = chunks;
  $("kpi-pages").textContent = pages;

  const counts = { grounded: 0, partially_grounded: 0, no_context: 0 };
  for (const row of state.log) if (counts[row.status] !== undefined) counts[row.status] += 1;
  $("kpi-grounded").textContent = counts.grounded;
  $("kpi-partial").textContent = counts.partially_grounded;
  $("kpi-nocontext").textContent = counts.no_context;
}

function drawLog() {
  const body = $("log-table").tBodies[0];
  const rows = state.filter === "all"
    ? state.log
    : state.log.filter((r) => r.status === state.filter);

  $("log-count").textContent = state.log.length
    ? `${rows.length} of ${state.log.length} shown`
    : "no queries yet";

  const note = $("filter-note");
  if (state.filter === "all") {
    note.hidden = true;
  } else {
    const meta = STATUS[state.filter];
    note.hidden = false;
    note.innerHTML = `<span>Filtered to <strong>${esc(meta ? meta.label : state.filter)}</strong> answers.</span>
      <button class="btn btn-ghost btn-mini" id="clear-filter">Clear filter</button>`;
    $("clear-filter").onclick = () => setFilter("all");
  }

  if (!rows.length) {
    body.innerHTML = `<tr class="empty"><td colspan="5">${
      state.log.length ? "No queries match this filter." : "No queries recorded in this session."
    }</td></tr>`;
    return;
  }

  body.innerHTML = rows.map((r) => `<tr>
    <td><div class="cell-title">${esc(r.question)}</div></td>
    <td>${badge(r.status)}</td>
    <td class="num">${r.confidence === null ? "&mdash;" : r.confidence.toFixed(2)}</td>
    <td class="num">${r.sources}</td>
    <td class="num">${(r.latency_ms / 1000).toFixed(1)}s</td>
  </tr>`).join("");
}

function setFilter(filter) {
  state.filter = filter;
  document.querySelectorAll(".kpi").forEach((card) => {
    card.classList.toggle("is-active",
      card.dataset.filter === filter || (filter === "all" && card.dataset.filter === "all"));
  });
  drawLog();
}

function renderAll() {
  drawKpis();
  drawDonut();
  drawBars();
  drawRegister();
  drawCorpus();
  drawOps();
  drawConfig();
  drawLog();
}

// ------------------------------------------------------------------ actions --

function renderAnswer(payload, question) {
  const box = $("ask-result");
  const citations = payload.citations || [];
  const meta = [
    badge(payload.status || "error"),
    `<span class="muted">confidence ${payload.confidence === null || payload.confidence === undefined
      ? "n/a" : payload.confidence.toFixed(2)}</span>`,
    payload.latency_ms ? `<span class="muted">${(payload.latency_ms / 1000).toFixed(1)}s</span>` : "",
    payload.llm_model ? `<span class="muted">${esc(payload.llm_model)}</span>` : "",
  ].filter(Boolean).join("");

  const cites = citations.length ? `
    <div class="cites">
      <h3>Evidence (${citations.length})</h3>
      ${citations.map((c) => `<div class="cite">
        <div class="cite-marker">S${esc(c.marker)}</div>
        <div>
          <div class="cite-title">${esc(c.title)}</div>
          <div class="cite-meta">page ${esc(c.page)}${c.section ? " &middot; " + esc(c.section) : ""}
            &middot; score ${Number(c.score).toFixed(3)}</div>
          <div class="cite-excerpt">${esc(c.excerpt)}</div>
        </div>
      </div>`).join("")}
    </div>` : "";

  const notes = (payload.notes || []).length
    ? `<div class="notes">${payload.notes.map((n) => esc(n)).join(" &middot; ")}</div>`
    : "";

  box.innerHTML = `<div class="answer st-${esc(payload.status || "error")}">
    <div class="answer-meta">${meta}</div>
    <div class="answer-body">${md(payload.answer || "No answer was produced.")}</div>
    ${notes}${cites}
  </div>`;

  state.log.unshift({
    question,
    status: payload.status || "error",
    confidence: payload.confidence ?? null,
    sources: new Set(citations.map((c) => c.filename)).size,
    latency_ms: payload.latency_ms || 0,
  });
  renderAll();
}

async function loadState() {
  const data = await api("/api/state");
  state.documents = data.documents || [];
  state.settings = data.settings || {};
  state.stats = data.stats || {};
  state.llmReady = !!data.llm_ready;

  const conn = $("conn");
  conn.className = "conn " + (state.llmReady ? "is-live" : "is-off");
  conn.querySelector(".conn-label").textContent = state.llmReady
    ? (state.settings.llm_model || "model ready")
    : "no model configured";

  $("engine-line").textContent = state.llmReady
    ? `retrieval + grounded synthesis via ${state.settings.llm_model}`
    : "retrieval only - no language model configured";

  renderAll();
}

async function loadSuggestions() {
  try {
    const { questions } = await api("/api/suggestions");
    $("suggestions").innerHTML = questions.slice(0, 6).map((q) =>
      `<button class="chip" type="button">${esc(q)}</button>`).join("");
    $("suggestions").querySelectorAll(".chip").forEach((chip) => {
      chip.onclick = () => { $("ask-input").value = chip.textContent; $("ask-input").focus(); };
    });
  } catch (_) { /* suggestions are optional chrome */ }
}

// -------------------------------------------------------------------- wiring --

function bind() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.onclick = () => {
      document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("is-active"));
      item.classList.add("is-active");
      const view = item.dataset.view;
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
      $("view-" + view).classList.add("is-active");
      const titles = {
        dashboard: ["Operations Dashboard", "Live status across the indexed document corpus."],
        documents: ["Document Register", "Index composition, coverage and intake."],
        corpus: ["Corpus Grid", "Where the indexed material actually sits, page by page."],
        analytics: ["Briefings", "Per-document summaries generated from the corpus."],
        config: ["Configuration", "Runtime settings reported by the backend."],
      };
      $("view-title").textContent = titles[view][0];
      $("view-sub").textContent = titles[view][1];
    };
  });

  document.querySelectorAll(".kpi").forEach((card) => {
    card.onclick = () => setFilter(card.dataset.filter);
  });

  document.querySelectorAll("#register-table thead th[data-sort]").forEach((th) => {
    th.onclick = () => {
      const key = th.dataset.sort;
      state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : (key === "title" ? 1 : -1) };
      drawRegister();
    };
  });

  $("register-table").tBodies[0].addEventListener("click", async (event) => {
    const removeId = event.target.closest("[data-remove]");
    if (removeId) {
      event.stopPropagation();
      const id = removeId.dataset.remove;
      if (!confirm("Remove this document and its passages from the index?")) return;
      try {
        await api("/api/documents/" + encodeURIComponent(id), { method: "DELETE" });
        if (state.selected === id) state.selected = null;
        toast("Document removed.", "ok");
        await loadState();
      } catch (err) { toast(err.message, "err"); }
      return;
    }
    const row = event.target.closest("tr[data-doc]");
    if (!row) return;
    state.selected = state.selected === row.dataset.doc ? null : row.dataset.doc;
    drawRegister();
    drawOps();
  });

  $("ask-form").onsubmit = async (event) => {
    event.preventDefault();
    const input = $("ask-input");
    const question = input.value.trim();
    if (!question) return;
    const submit = $("ask-submit");
    submit.disabled = true;
    submit.textContent = $("ask-mode").value === "evidence" ? "Retrieving" : "Thinking";
    try {
      const mode = $("ask-mode").value;
      const payload = mode === "evidence"
        ? await jsonPost("/api/evidence", { question })
        : await jsonPost("/api/ask", { question });
      if (mode === "evidence") {
        renderAnswer({
          status: payload.citations.length ? "grounded" : "no_context",
          answer: payload.citations.length
            ? `Retrieved ${payload.citations.length} supporting passage(s) without calling the model.`
            : "No passage in the index matched that question.",
          citations: payload.citations,
          notes: ["Evidence-only mode: no model call, no synthesised text."],
          latency_ms: null,
        }, question);
      } else {
        renderAnswer(payload, question);
      }
      input.value = "";
    } catch (err) {
      renderAnswer({ status: "error", answer: "The request failed: " + err.message, citations: [] }, question);
      toast(err.message, "err");
    } finally {
      submit.disabled = false;
      submit.textContent = "Run Query";
    }
  };

  $("upload-form").onsubmit = async (event) => {
    event.preventDefault();
    const input = $("upload-input");
    if (!input.files.length) { toast("Choose at least one PDF first.", "err"); return; }
    const form = new FormData();
    for (const file of input.files) form.append("files", file);
    const note = $("upload-note");
    note.textContent = "Ingesting " + input.files.length + " file(s)...";
    try {
      const result = await api("/api/upload", { method: "POST", body: form });
      const added = result.added ? result.added.length : 0;
      note.textContent = `Added ${added}, skipped ${(result.duplicates || []).length} duplicate(s).`;
      toast(`Indexed ${added} new document(s).`, "ok");
      input.value = "";
      await loadState();
    } catch (err) {
      note.textContent = "";
      toast(err.message, "err");
    }
  };

  $("btn-refresh").onclick = () => loadState().then(() => toast("State refreshed.")).catch((e) => toast(e.message, "err"));

  $("btn-rebuild").onclick = async () => {
    if (!confirm("Rebuild the index from the starter documents?")) return;
    try {
      toast("Rebuilding the index...");
      const result = await jsonPost("/api/rebuild", {});
      toast(`Rebuilt: ${result.chunk_count} passages.`, "ok");
      state.selected = null;
      await loadState();
    } catch (err) { toast(err.message, "err"); }
  };

  $("btn-clear").onclick = async () => {
    if (!confirm("Clear every document from the index? This cannot be undone.")) return;
    try {
      await jsonPost("/api/clear", {});
      state.selected = null;
      toast("Index cleared.", "ok");
      await loadState();
    } catch (err) { toast(err.message, "err"); }
  };

  $("btn-briefings").onclick = async () => {
    const button = $("btn-briefings");
    button.disabled = true;
    button.textContent = "Generating...";
    $("briefings").innerHTML = '<p class="muted">Summarising each document from the corpus...</p>';
    try {
      const { summaries } = await jsonPost("/api/briefings", {});
      $("briefings").innerHTML = summaries.map((s) => `<div class="brief">
        <h3>${esc(s.title || s.filename || "Document")}</h3>
        <div class="body">${md(s.summary || s.text || "")}</div>
      </div>`).join("") || '<p class="muted">No summaries were produced.</p>';
    } catch (err) {
      $("briefings").innerHTML = `<p class="muted">Briefings unavailable: ${esc(err.message)}</p>`;
    } finally {
      button.disabled = false;
      button.textContent = "Generate Briefings";
    }
  };
}

bind();
loadState().then(loadSuggestions).catch((err) => toast("Could not load state: " + err.message, "err"));