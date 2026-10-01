/* ASTRA INTEL - corpus operations console.
   Vanilla ES2020, no build step and no CDN: charts are hand-drawn SVG so the
   console still works on an isolated network where Chart.js could not load.

   The console has two destinations: Query (chat-first, the operator's main
   surface) and Documents (intake, register, per-document briefing and
   extraction QA). Everything reads from the real index. */

const $ = (id) => document.getElementById(id);

const STATUS = {
  grounded:           { label: "Grounded",   cls: "badge-grounded",     color: "#34d399" },
  partially_grounded: { label: "Partial",    cls: "badge-partially_grounded", color: "#fbbf24" },
  no_context:         { label: "No Context", cls: "badge-no_context",  color: "#fb7185" },
  error:              { label: "Failed",     cls: "badge-error",       color: "#fb7185" },
};

const STATUS_ORDER = ["grounded", "partially_grounded", "no_context", "error"];

const state = {
  documents: [],
  settings: {},
  runtime: {},
  llmReady: false,
  stats: {},
  log: [],
  filter: "all",
  sort: { key: "chunk_count", dir: -1 },
  selected: null,
  briefingFor: null,
};

// Upper bound on retained query-log rows, so a long session cannot grow the
// DOM without limit.
const MAX_LOG_ROWS = 200;

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ------------------------------------------------------------------ utils --

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* non-JSON error body; keep the status text */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
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

/** Count query-log rows per status, always returning every known key. */
function statusCounts() {
  const counts = { grounded: 0, partially_grounded: 0, no_context: 0, error: 0 };
  for (const row of state.log) {
    if (counts[row.status] === undefined) counts[row.status] = 0;
    counts[row.status] += 1;
  }
  return counts;
}

/* Minimal markdown: the model answers with short prose plus bullet lists, so
   bold, italics and inline code are the only inline marks worth handling.
   Escaping happens first, which keeps this safe against injected HTML. */
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
    // Citation markers, normalised server-side, rendered as superscripts.
    .replace(/\[?\s*S(\d+)\s*\]?/g, '<sup class="cite-ref">S$1</sup>');
}

// ------------------------------------------------------------------- donut --

/* The centre figure sits on a near-black panel, so it must be near-white and
   the empty track a dim slate. Drawing them light-on-light, or dark-on-dark,
   made the count unreadable in the previous version. */
function drawDonut() {
  const svg = $("donut");
  const total = state.log.length;
  const counts = statusCounts();

  if (!total) {
    svg.innerHTML =
      '<circle class="d-track" cx="100" cy="100" r="70" fill="none" stroke-width="26"/>' +
      '<text class="d-total" x="100" y="99" text-anchor="middle">0</text>' +
      '<text class="d-cap" x="100" y="117" text-anchor="middle">QUERIES</text>';
    $("donut-legend").innerHTML = "";
    return;
  }

  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  let rings = "";
  STATUS_ORDER.forEach((key) => {
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
    `<text class="d-total" x="100" y="99" text-anchor="middle">${total}</text>` +
    `<text class="d-cap" x="100" y="117" text-anchor="middle">QUERIES</text>`;

  $("donut-legend").innerHTML = STATUS_ORDER
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
    // Relative to the largest document rather than an invented capacity.
    const tier = ratio >= 0.7 ? "t-high" : ratio >= 0.3 ? "t-mid" : "t-low";
    return `<div class="bar-row">
      <span class="name" title="${esc(d.title)}">${esc(d.title)}</span>
      <span class="bar-track"><span class="bar-fill ${tier}" style="width:${(ratio * 100).toFixed(1)}%"></span></span>
      <span class="val">${d.chunk_count}</span>
    </div>`;
  }).join("");
}

// ------------------------------------------------------------ document status --

function docStatus(doc) {
  if (!doc.chunk_count) return { key: "none", label: "Unindexed", cls: "badge-no_context" };
  const coverage = doc.page_count ? doc.indexed_pages / doc.page_count : 1;
  if (coverage >= 1) return { key: "full", label: "Fully indexed", cls: "badge-grounded" };
  if (coverage >= 0.6) return { key: "partial", label: "Part indexed", cls: "badge-partially_grounded" };
  return { key: "sparse", label: "Sparse", cls: "badge-mute" };
}

function coverageOf(doc) {
  return doc.page_count ? doc.indexed_pages / doc.page_count : 0;
}

// ---------------------------------------------------------------- register --

function drawRegister() {
  const body = $("register-table").tBodies[0];
  const key = state.sort.key;
  const rows = [...state.documents].sort((a, b) => {
    if (key === "coverage") return (coverageOf(a) - coverageOf(b)) * state.sort.dir;
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
    const pct = (coverageOf(d) * 100).toFixed(0);
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
      <td class="num row-actions">
        <button class="btn btn-mini btn-ghost" data-brief="${esc(d.document_id)}"
          title="Summarise this document from the corpus">Brief</button>
        <button class="btn btn-mini btn-ghost btn-danger" data-remove="${esc(d.document_id)}">Remove</button>
      </td>
    </tr>`;
  }).join("");

  $("register-table").querySelectorAll("thead th[data-sort]").forEach((th) => {
    th.classList.toggle("is-sorted", th.dataset.sort === state.sort.key);
  });
}

// ------------------------------------------------------------- corpus grid --

function drawCorpus() {
  const grid = $("corpus-grid");
  const legend = $("scale-legend");
  if (!state.documents.length) {
    grid.innerHTML = '<p class="muted tiny">Nothing indexed yet.</p>';
    legend.innerHTML = "";
    drawOps();
    return;
  }

  // One scale across every document, so a pale cell in a small document means
  // the same thing as a pale cell in a large one.
  const peak = Math.max(
    1,
    ...state.documents.flatMap((d) => (d.pages || []).map((p) => p.passages)),
  );
  const steps = [0.08, 0.3, 0.55, 0.78, 1]
    .map((f) => `<i style="background:rgba(56,189,248,${(0.1 + f * 0.85).toFixed(2)})"></i>`)
    .join("");
  legend.innerHTML = `<span>1 passage</span>
    <span class="scale-steps">${steps}</span>
    <span>${peak} passages / page</span>
    <span class="muted">&middot; each square is one page</span>`;

  grid.innerHTML = state.documents.map((d) => {
    const cells = (d.pages || []).map((p) => {
      const alpha = 0.1 + (p.passages / peak) * 0.85;
      const hint = p.passages
        ? ` title="${esc(d.title)} p.${p.page} - ${p.passages} passage${p.passages === 1 ? "" : "s"}"`
        : ` title="${esc(d.title)} p.${p.page} - no passages extracted"`;
      return `<span class="cell" style="background:rgba(56,189,248,${alpha.toFixed(2)})"${hint}></span>`;
    }).join("");
    return `<div class="corpus-row">
      <span class="corpus-label" title="${esc(d.title)}">${esc(d.title)}</span>
      <span class="corpus-cells">${cells}</span>
    </div>`;
  }).join("");
}

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
        <span class="sw" style="background:rgba(56,189,248,${(0.1 + (p.passages / peak) * 0.85).toFixed(2)})"></span>
        Page ${p.page}<span class="n">${p.passages}</span>
      </li>`).join("")}</ul>`;
}

// ---------------------------------------------------------------- readouts --

/* The instrument strip and the sidebar status both read from
   /api/state.runtime, which carries stable snake_case keys. The `settings`
   block is display-only and its keys are human labels ("LLM model"), so it is
   never read programmatically. */
function drawReadouts() {
  const rt = state.runtime;
  const chunks = totalChunks();
  const docs = state.documents.length;

  $("rd-corpus").textContent = docs
    ? `${docs} doc${docs === 1 ? "" : "s"} · ${chunks} passages`
    : "empty";
  $("rd-model").textContent = rt.llm_model || "—";
  $("rd-provider").textContent = rt.llm_provider || "—";
  // The model id is long; show the family, not the full repo path.
  $("rd-embed").textContent = shortModel(rt.embedding_model);

  const conn = $("conn");
  conn.className = "conn " + (state.llmReady ? "is-live" : "is-off");
  conn.querySelector(".conn-label").textContent = state.llmReady
    ? (rt.llm_model || "model ready")
    : "no model configured";

  $("engine-line").textContent = state.llmReady
    ? `retrieval + grounded synthesis via ${rt.llm_model}`
    : "retrieval only - no language model configured";
}

function shortModel(name) {
  if (!name) return "—";
  const tail = String(name).split("/").pop();
  return tail.length > 26 ? tail.slice(0, 24) + "…" : tail;
}

function totalChunks() {
  const summed = state.documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
  return state.stats.chunks ?? state.stats.chunk_count ?? summed;
}

function totalPages() {
  return state.documents.reduce((sum, d) => sum + (d.page_count || 0), 0);
}

// ------------------------------------------------------------------- KPIs --

function setNum(id, value) {
  const el = $(id);
  const target = Number(value) || 0;
  const from = Number(el.dataset.value || 0);
  if (reduceMotion.matches || from === target) {
    el.textContent = target;
    el.dataset.value = String(target);
    return;
  }
  el.dataset.value = String(target);
  const started = performance.now();
  const duration = 620;
  const step = (now) => {
    const t = Math.min(1, (now - started) / duration);
    el.textContent = Math.round(from + (target - from) * (1 - Math.pow(1 - t, 3)));
    if (t < 1 && el.dataset.value === String(target)) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function drawKpis() {
  const pages = totalPages();
  const counts = statusCounts();
  setNum("kpi-documents", state.documents.length);
  $("kpi-documents-foot").textContent = `${pages} pages parsed`;
  setNum("kpi-passages", totalChunks());
  setNum("kpi-grounded", counts.grounded);
  setNum("kpi-partial", counts.partially_grounded);
  setNum("kpi-nocontext", counts.no_context);
  // Derived, not clickable: mean passages per parsed page.
  $("kpi-density").textContent = pages ? (totalChunks() / pages).toFixed(1) : "0.0";
}

// --------------------------------------------------------------------- log --

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
    note.innerHTML = "";
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
  document.querySelectorAll(".tile[data-filter]").forEach((tile) => {
    tile.classList.toggle("is-active", tile.dataset.filter === filter);
  });
  drawLog();
}

// -------------------------------------------------------------- rendering --

function renderAll() {
  drawReadouts();
  drawKpis();
  drawDonut();
  drawBars();
  drawRegister();
  drawCorpus();
  drawLog();
}

function renderViews() {
  const current = document.querySelector(".view.is-active");
  renderAll();
  if (current) drawOps();
}

// ------------------------------------------------------------------ answer --

function renderAnswer(payload, question) {
  const box = $("ask-result");
  const citations = payload.citations || [];
  const confidence = payload.confidence;

  // Confidence as a measured meter; the number alone read as decoration.
  const meter = confidence === null || confidence === undefined
    ? `<span class="muted">confidence n/a</span>`
    : `<span class="meter">
         <span class="meter-track">
           <span class="meter-fill m-${
             payload.status === "grounded" ? "good"
               : payload.status === "partially_grounded" ? "warn" : "bad"
           }" style="width:${Math.round(confidence * 100)}%"></span>
         </span>
         <span>${confidence.toFixed(2)}</span>
       </span>`;

  const meta = [
    badge(payload.status || "error"),
    meter,
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
    confidence: confidence ?? null,
    sources: new Set(citations.map((c) => c.filename)).size,
    latency_ms: payload.latency_ms || 0,
  });
  // Bounded: an operator session can run for hours, and an unbounded log
  // grows the DOM on every answer.
  if (state.log.length > MAX_LOG_ROWS) state.log.length = MAX_LOG_ROWS;
  renderAll();
}

// ----------------------------------------------------------------- loading --

async function loadState() {
  const data = await api("/api/state");
  state.documents = data.documents || [];
  state.settings = data.settings || {};
  state.runtime = data.runtime || {};
  state.stats = data.stats || {};
  state.llmReady = !!data.llm_ready;

  // A document that was removed underneath an open briefing panel.
  if (state.briefingFor && !state.documents.some((d) => d.document_id === state.briefingFor)) {
    $("brief-panel").hidden = true;
    state.briefingFor = null;
  }
  renderViews();
}

async function loadSuggestions() {
  try {
    const { questions } = await api("/api/suggestions");
    $("suggestions").innerHTML = questions.slice(0, 6).map((q) =>
      `<button class="chip" type="button">${esc(q)}</button>`).join("");
    $("suggestions").querySelectorAll(".chip").forEach((chip) => {
      chip.onclick = () => {
        $("ask-input").value = chip.textContent;
        $("ask-input").focus();
      };
    });
  } catch (_) { /* suggestions are optional chrome */ }
}

// ------------------------------------------------------------------ intake --

function intakeStatus(text, kind = "") {
  const note = $("upload-note");
  note.textContent = text;
  note.className = "intake-status " + (kind ? "is-" + kind : "");
}

/* Preflight the batch against the same limits the server enforces, so an
   oversized or over-count upload fails in the browser with a specific message
   instead of round-tripping and returning a bare 400. */
function preflight(files) {
  const form = $("upload-form");
  const maxFiles = Number(form.dataset.maxFiles) || 20;
  const maxMb = Number(form.dataset.maxMb) || 50;
  if (files.length > maxFiles) {
    return `Too many files: ${files.length}. Upload at most ${maxFiles} at once.`;
  }
  const oversized = files.filter((f) => f.size > maxMb * 1024 * 1024);
  if (oversized.length) {
    return `${oversized[0].name} is larger than the ${maxMb} MB limit.`;
  }
  const wrongType = files.filter((f) => f.type && f.type !== "application/pdf");
  if (wrongType.length) {
    return `${wrongType[0].name} is not a PDF.`;
  }
  return null;
}

function describeFiles(files) {
  const mb = files.reduce((sum, f) => sum + f.size, 0) / (1024 * 1024);
  const size = mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(mb * 1024)} KB`;
  return `${files.length} file${files.length === 1 ? "" : "s"} · ${size}`;
}

/* Drag-and-drop onto the existing file input. The input's own files list is
   assigned rather than tracked separately, so submitting reads one source of
   truth. */
function bindDropzone() {
  const zone = $("dropzone");
  const input = $("upload-input");
  let depth = 0;

  const setFiles = (list) => {
    const transfer = new DataTransfer();
    for (const file of list) transfer.items.add(file);
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };

  zone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    zone.classList.add("is-over");
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  zone.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) zone.classList.remove("is-over");
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    zone.classList.remove("is-over");
    const files = event.dataTransfer.files;
    if (files.length) setFiles(files);
  });

  input.addEventListener("change", () => {
    if (!input.files.length) {
      $("dz-sub").textContent = "drop PDFs here, or browse";
      return;
    }
    const problem = preflight(input.files);
    if (problem) {
      $("dz-sub").textContent = problem;
      $("dz-sub").style.color = "var(--bad)";
    } else {
      $("dz-sub").textContent = describeFiles(input.files);
      $("dz-sub").style.color = "";
    }
  });
}

// ---------------------------------------------------------------- briefing --

async function briefDocument(documentId) {
  const panel = $("brief-panel");
  const body = $("brief-body");
  const meta = $("brief-meta");

  state.briefingFor = documentId;
  panel.hidden = false;
  meta.textContent = "Generating from the corpus...";
  body.innerHTML = '<span class="briefing-status">' +
    '<span class="think-pulse" aria-hidden="true"></span> Summarising...</span>';

  try {
    const { summaries } = await jsonPost(
      "/api/briefings?document_id=" + encodeURIComponent(documentId), {}
    );
    const s = summaries && summaries[0];
    if (!s) throw new Error("No summary was produced.");

    const provenance = [
      s.llm_generated ? `generated by ${esc(s.model || "the model")}` : "structural fallback (no model)",
      s.pages && s.pages.length ? `pages ${s.pages.slice(0, 12).join(", ")}${s.pages.length > 12 ? "…" : ""}` : null,
    ].filter(Boolean).join(" &middot; ");

    meta.textContent = provenance;
    body.innerHTML = `
      <div class="answer-body">${md(s.summary || "")}</div>
      ${(s.topics || []).length ? `<div class="brief-topics">${
        s.topics.map((t) => `<span class="topic">${esc(t)}</span>`).join("")
      }</div>` : ""}`;
    panel.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "nearest" });
  } catch (err) {
    meta.textContent = "Unavailable";
    body.innerHTML = `<p class="muted">${esc(err.message)}</p>`;
    toast(err.message, "err");
  }
}

// ------------------------------------------------------------------ wiring --

function showThinking(on, text) {
  $("ask-thinking").hidden = !on;
  if (text) $("think-text").textContent = text;
}

function bind() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.onclick = () => {
      document.querySelectorAll(".nav-item").forEach((n) => {
        n.classList.remove("is-active");
        n.removeAttribute("aria-current");
      });
      item.classList.add("is-active");
      item.setAttribute("aria-current", "page");
      const view = item.dataset.view;
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
      $("view-" + view).classList.add("is-active");
      const titles = {
        query: ["Query", "Ask the corpus, with every claim cited."],
        documents: ["Documents", "Intake, the register, and extraction quality."],
      };
      $("view-title").textContent = titles[view][0];
      $("view-sub").textContent = titles[view][1];
      drawOps();
    };
  });

  document.querySelectorAll(".tile[data-filter]").forEach((tile) => {
    tile.onclick = () => setFilter(tile.dataset.filter);
  });

  // Segmented mode control: a native select here read as a browser dropdown
  // dropped into a designed panel.
  $("ask-mode").querySelectorAll(".seg-btn").forEach((btn) => {
    btn.onclick = () => {
      const group = $("ask-mode");
      group.dataset.mode = btn.dataset.mode;
      group.querySelectorAll(".seg-btn").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("is-on", on);
        b.setAttribute("aria-pressed", String(on));
      });
    };
  });

  $("register-table").querySelectorAll("thead th[data-sort]").forEach((th) => {
    th.onclick = () => {
      const key = th.dataset.sort;
      state.sort = {
        key,
        dir: state.sort.key === key ? -state.sort.dir : (key === "title" ? 1 : -1),
      };
      drawRegister();
    };
  });

  $("register-table").tBodies[0].addEventListener("click", async (event) => {
    const briefId = event.target.closest("[data-brief]");
    if (briefId) {
      event.stopPropagation();
      briefDocument(briefId.dataset.brief);
      return;
    }
    const removeId = event.target.closest("[data-remove]");
    if (removeId) {
      event.stopPropagation();
      const id = removeId.dataset.remove;
      if (!confirm("Remove this document and its passages from the index?")) return;
      try {
        await api("/api/documents/" + encodeURIComponent(id), { method: "DELETE" });
        if (state.selected === id) state.selected = null;
        if (state.briefingFor === id) {
          $("brief-panel").hidden = true;
          state.briefingFor = null;
        }
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

  $("brief-close").onclick = () => {
    $("brief-panel").hidden = true;
    state.briefingFor = null;
  };

  $("ask-form").onsubmit = async (event) => {
    event.preventDefault();
    const input = $("ask-input");
    const question = input.value.trim();
    if (!question) return;
    const submit = $("ask-submit");
    const mode = $("ask-mode").dataset.mode;

    submit.disabled = true;
    $("ask-result").innerHTML = "";
    showThinking(true, mode === "evidence"
      ? "Retrieving passages from the index…"
      : "Retrieving passages, then synthesising a cited answer…");

    try {
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
      showThinking(false);
    }
  };

  $("upload-form").onsubmit = async (event) => {
    event.preventDefault();
    const input = $("upload-input");
    if (!input.files.length) { toast("Choose at least one PDF first.", "err"); return; }

    const problem = preflight(input.files);
    if (problem) {
      intakeStatus(problem, "err");
      toast(problem, "err");
      return;
    }

    const submit = $("upload-submit");
    submit.disabled = true;
    intakeStatus("Ingesting " + describeFiles(input.files) + "…", "busy");

    const form = new FormData();
    for (const file of input.files) form.append("files", file);
    try {
      const result = await api("/api/upload", { method: "POST", body: form });
      const added = result.added ? result.added.length : 0;
      const skipped = (result.duplicates || []).length;
      intakeStatus(
        `Indexed ${added} new document${added === 1 ? "" : "s"}; skipped ${skipped} duplicate${skipped === 1 ? "" : "s"}.`,
        "ok"
      );
      toast(`Indexed ${added} new document(s).`, "ok");
      input.value = "";
      $("dz-sub").textContent = "drop PDFs here, or browse";
      $("dz-sub").style.color = "";
      await loadState();
    } catch (err) {
      // A per-file failure surfaces as a 400 now, so it has to be shown.
      intakeStatus(err.message, "err");
      toast(err.message, "err");
    } finally {
      submit.disabled = false;
    }
  };

  bindDropzone();

  $("btn-refresh").onclick = () =>
    loadState().then(() => toast("State refreshed.")).catch((e) => toast(e.message, "err"));

  $("btn-rebuild").onclick = async () => {
    if (!confirm(
      "Rebuilding discards EVERY indexed document, including your uploads, and re-indexes the starter PDFs only.\n\nContinue?"
    )) return;
    try {
      toast("Rebuilding the index...");
      const result = await jsonPost("/api/rebuild?confirm=rebuild", {});
      toast(`Rebuilt: ${result.chunk_count} passages.`, "ok");
      state.selected = null;
      $("brief-panel").hidden = true;
      state.briefingFor = null;
      await loadState();
    } catch (err) { toast(err.message, "err"); }
  };

  $("btn-clear").onclick = async () => {
    if (!confirm("Clear every document from the index? This cannot be undone.")) return;
    try {
      await jsonPost("/api/clear?confirm=clear", {});
      state.selected = null;
      $("brief-panel").hidden = true;
      state.briefingFor = null;
      toast("Index cleared.", "ok");
      await loadState();
    } catch (err) { toast(err.message, "err"); }
  };

  // Enter submits, Escape clears the field.
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== $("ask-input")) {
      const active = document.activeElement;
      const typing = active && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName);
      if (!typing) {
        event.preventDefault();
        $("ask-input").focus();
      }
    }
  });
}

/* Pointer parallax: nudge the aurora orbs against the cursor. The CSS keyframes
   own the drift, so this writes the `translate` property and composes with
   them rather than fighting the animation. */
function startAmbient() {
  if (reduceMotion.matches) return;
  const orbs = [...document.querySelectorAll(".aurora .orb")];
  if (!orbs.length) return;
  const strengths = [14, -18, 10];

  let frame = null;
  let px = 0, py = 0, cx = 0, cy = 0;

  const loop = () => {
    cx += (px - cx) * 0.045;
    cy += (py - cy) * 0.045;
    orbs.forEach((orb, i) => {
      orb.style.translate = `${(cx * strengths[i]) / 10}px ${(cy * strengths[i]) / 10}px`;
    });
    frame = requestAnimationFrame(loop);
  };

  window.addEventListener("pointermove", (event) => {
    px = (event.clientX / window.innerWidth - 0.5) * 2;
    py = (event.clientY / window.innerHeight - 0.5) * 2;
    if (!frame) loop();
  }, { passive: true });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (frame) { cancelAnimationFrame(frame); frame = null; }
    } else if (!frame) {
      loop();
    }
  });
}

bind();
startAmbient();
loadState().then(loadSuggestions).catch((err) => toast("Could not load state: " + err.message, "err"));