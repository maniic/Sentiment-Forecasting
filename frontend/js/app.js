/* App state + rendering. No framework — small, readable, dependency-free. */

import {
  drawdownChart,
  embeddingCloud,
  equityChart,
  sentimentBars,
  sentimentScatter,
  topoTimeline,
} from "./charts.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  mode: "demo",
  presets: {},
  lastResult: null,
};

/* ---------------- formatting helpers ---------------- */
const pct = (v, digits = 1) =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
const signedPct = (v, digits = 1) =>
  v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;
const num = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits));

const fmtWhen = (iso) => {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
};

/* ---------------- controls ---------------- */
function initControls(config) {
  state.presets = config.presets;
  const preset = $("#preset");
  Object.entries(config.presets).forEach(([name, tks]) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = `${name} (${tks.join(", ")})`;
    preset.appendChild(opt);
  });
  const custom = document.createElement("option");
  custom.value = "__custom__";
  custom.textContent = "Custom…";
  preset.appendChild(custom);

  preset.addEventListener("change", () => {
    $("#custom-tickers-field").hidden = preset.value !== "__custom__";
  });

  $("#mode-seg").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-mode]");
    if (!btn) return;
    state.mode = btn.dataset.mode;
    $$("#mode-seg button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    $("#mode-hint").textContent =
      state.mode === "demo"
        ? "Instant, offline, reproducible synthetic market — the generated headlines genuinely predict next-day moves."
        : "Pulls real headlines (Google News / Yahoo) and real prices (yfinance). Needs internet.";
    $("#source-field").hidden = state.mode !== "live";
  });

  // live-updating slider labels
  const bind = (rangeId, labelId, fmt = (v) => v) => {
    const r = $(rangeId);
    r.addEventListener("input", () => ($(labelId).textContent = fmt(r.value)));
  };
  bind("#threshold", "#threshold-val", (v) => Number(v).toFixed(2));
  bind("#news-days", "#news-days-val");
  bind("#lookback", "#lookback-val");

  if (!config.finbert_available) {
    const engine = $("#engine");
    engine.querySelector('[value="finbert"]').disabled = true;
    engine.querySelector('[value="auto"]').textContent = "Auto — lexicon (FinBERT not installed)";
  }

  $("#run").addEventListener("click", run);

  // tabs
  $("#tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    $$(".panel").forEach((p) => (p.hidden = p.id !== tab.dataset.panel));
    // Plotly needs a resize poke when a hidden panel becomes visible
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  });

  $("#topo-ticker").addEventListener("change", renderTopology);
}

function currentTickers() {
  const preset = $("#preset").value;
  if (preset === "__custom__") {
    return $("#custom-tickers")
      .value.split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
  }
  return state.presets[preset] ?? [];
}

/* ---------------- run ---------------- */
async function run() {
  const btn = $("#run");
  btn.classList.add("busy");
  btn.textContent = "Running…";
  $("#error-banner").hidden = true;
  $$(".stat .value").forEach((el) => (el.textContent = ""));

  const body = {
    mode: state.mode,
    tickers: currentTickers(),
    news_days: Number($("#news-days").value),
    price_lookback: Number($("#lookback").value),
    threshold: Number($("#threshold").value),
    require_momentum: $("#momentum").checked,
    cost_bps: Number($("#cost").value),
    engine: $("#engine").value,
    news_source: $("#news-source").value,
  };

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail;
      throw new Error(detail || `Server error (${res.status})`);
    }
    state.lastResult = await res.json();
    renderAll(state.lastResult);
  } catch (err) {
    const banner = $("#error-banner");
    banner.textContent = `⚠ ${err.message}`;
    banner.hidden = false;
  } finally {
    btn.classList.remove("busy");
    btn.textContent = "▶ Run analysis";
  }
}

/* ---------------- rendering ---------------- */
function renderAll(r) {
  renderKpis(r.metrics, r.meta);
  renderMeta(r.meta);
  equityChart($("#chart-equity"), r.perf);
  drawdownChart($("#chart-drawdown"), r.perf);
  sentimentBars($("#chart-sent"), r.sentiment_by_ticker);
  sentimentScatter($("#chart-scatter"), r.scatter);
  renderHeadlines(r.headlines);
  renderPicks(r.picks);
  initTopoSelect(r.topology);
  renderTopology();
}

function renderKpis(m, meta) {
  const set = (key, text) => ($(`[data-kpi="${key}"]`).textContent = text);
  set("total_return", signedPct(m.total_return));
  set("sharpe", num(m.sharpe));
  set("max_dd", m.max_dd == null ? "—" : `−${Math.abs(m.max_dd * 100).toFixed(1)}%`);
  set("win_rate", m.win_rate == null ? "—" : `${Math.round(m.win_rate * 100)}%`);
  set("days", `${m.trades} / ${m.n_days}`);
  set("threshold-note", `threshold ${meta.threshold.toFixed(2)}`);

  const deltaEl = $('[data-kpi="delta"]');
  if (m.total_return != null && m.bench_total_return != null) {
    const d = m.total_return - m.bench_total_return;
    deltaEl.textContent = `${d >= 0 ? "▲" : "▼"} ${signedPct(d)} vs buy & hold`;
    deltaEl.className = `delta num ${d >= 0 ? "up" : "down"}`;
  } else {
    deltaEl.textContent = "";
  }
}

function renderMeta(meta) {
  const engine = { finbert: "FinBERT", lexicon: "Lexicon" }[meta.engine_used] ?? meta.engine_used;
  const source = meta.mode === "demo" ? "Demo data" : "Live data";
  const chip = $("#engine-chip");
  chip.textContent = `${engine} · ${source}`;
  chip.hidden = false;

  const strip = $("#meta-strip");
  strip.innerHTML = `
    <span><b>Headlines</b> ${meta.n_headlines}</span>
    <span><b>Last headline</b> ${meta.last_headline_ts ? fmtWhen(meta.last_headline_ts) : "—"}</span>
    <span><b>As-of</b> ${meta.asof_date}</span>
    <span><b>Fetched</b> ${fmtWhen(meta.fetched_at)}</span>
    <span><b>Engine</b> Rule blend · ${engine} · ${source}</span>`;
  strip.hidden = false;
}

const BADGE = {
  POSITIVE: '<span class="badge badge-pos">▲ Positive</span>',
  NEGATIVE: '<span class="badge badge-neg">▼ Negative</span>',
  NEUTRAL: '<span class="badge badge-neu">• Neutral</span>',
};

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

function renderHeadlines(rows) {
  const tbody = $("#headlines-table tbody");
  tbody.innerHTML = rows
    .map(
      (h) => `<tr>
        <td style="color:var(--ink);font-weight:650">${esc(h.ticker)}</td>
        <td class="num" style="white-space:nowrap">${fmtWhen(h.ts)}</td>
        <td style="color:var(--ink)">${esc(h.headline)}</td>
        <td>${BADGE[h.sent_label] ?? esc(h.sent_label)}</td>
        <td class="num-cell">${h.sent_score >= 0 ? "+" : "−"}${Math.abs(h.sent_score).toFixed(2)}</td>
        <td class="muted">${esc(h.source || "—")}</td>
      </tr>`
    )
    .join("");
}

function renderPicks(picks) {
  const table = $("#picks-table");
  const empty = $("#picks-empty");
  const sub = $("#picks-sub");
  if (!picks.length) {
    table.hidden = true;
    empty.hidden = false;
    sub.textContent = "";
    return;
  }
  table.hidden = false;
  empty.hidden = true;
  sub.textContent = `${picks.length} ticker${picks.length > 1 ? "s" : ""} cleared the bar for tomorrow`;
  table.querySelector("tbody").innerHTML = picks
    .map(
      (p) => `<tr>
        <td style="color:var(--ink);font-weight:650">${esc(p.ticker)}</td>
        <td><div class="prob"><div class="bar"><span style="width:${Math.round(p.p_up * 100)}%"></span></div>
            <span class="pct">${p.p_up.toFixed(2)}</span></div></td>
        <td class="num-cell" style="color:${p.ret_5 >= 0 ? "var(--good)" : "var(--bad)"}">${signedPct(p.ret_5, 2)}</td>
        <td class="num-cell">${p.sent_mean >= 0 ? "+" : "−"}${Math.abs(p.sent_mean).toFixed(2)}</td>
        <td class="num-cell">${p.n}</td>
      </tr>`
    )
    .join("");
}

function initTopoSelect(topology) {
  const sel = $("#topo-ticker");
  const prev = sel.value;
  sel.innerHTML = "";
  Object.keys(topology).forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  });
  if (prev && topology[prev]) sel.value = prev;
}

function renderTopology() {
  const r = state.lastResult;
  if (!r) return;
  const t = $("#topo-ticker").value;
  const data = r.topology[t];
  if (!data) return;
  $("#topo-3d-title").textContent = `${t} — daily returns as a 3D point cloud (delay embedding)`;
  $("#topo-ts-title").textContent = `${t} — ${data.topo_label}`;
  embeddingCloud($("#chart-embed"), data.returns);
  topoTimeline($("#chart-topo"), data.dates, data.topo);
}

/* ---------------- boot ---------------- */
(async function boot() {
  try {
    const config = await (await fetch("/api/config")).json();
    initControls(config);
    await run(); // auto-run the demo so the page opens alive
  } catch (err) {
    const banner = $("#error-banner");
    banner.textContent = `⚠ Could not reach the API: ${err.message}`;
    banner.hidden = false;
  }
})();
