/* Plotly chart builders — one shared dark theme, thin marks, recessive grid. */

const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const C = {
  strategy: css("--chart-1") || "#3987e5",
  negative: css("--chart-2") || "#e66767",
  topology: css("--chart-3") || "#9085e9",
  bench: css("--chart-bench") || "#898781",
  grid: css("--chart-grid") || "#2c2c2a",
  ink: css("--ink") || "#fff",
  ink2: css("--ink-2") || "#c3c2b7",
  ink3: css("--ink-3") || "#898781",
  surface: css("--surface-1") || "#1a1a19",
  seqLo: css("--chart-seq-lo") || "#cde2fb",
  seqHi: css("--chart-seq-hi") || "#104281",
};

const FONT = { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12.5, color: C.ink2 };

const BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: FONT,
  margin: { l: 44, r: 110, t: 8, b: 34 },
  hovermode: "x unified",
  hoverlabel: { bgcolor: C.surface, bordercolor: C.grid, font: { ...FONT, color: C.ink } },
  xaxis: { gridcolor: C.grid, zeroline: false, tickfont: { color: C.ink3 }, linecolor: C.grid },
  yaxis: { gridcolor: C.grid, zeroline: false, tickfont: { color: C.ink3 }, linecolor: C.grid },
  showlegend: false,
};

const CONFIG = { displayModeBar: false, responsive: true };

const clone = (o) => JSON.parse(JSON.stringify(o));

export function equityChart(el, perf) {
  const last = perf.equity.at(-1);
  const lastBench = perf.bench_equity.at(-1);
  const layout = clone(BASE);
  layout.yaxis.tickformat = ".2f";
  layout.annotations = [
    {
      x: perf.date.at(-1), y: last, xanchor: "left", xshift: 8, showarrow: false,
      text: `<b>Strategy ${last.toFixed(2)}×</b>`, font: { color: C.strategy, size: 12.5 },
    },
    {
      x: perf.date.at(-1), y: lastBench, xanchor: "left", xshift: 8, showarrow: false,
      text: `Buy &amp; hold ${lastBench.toFixed(2)}×`, font: { color: C.bench, size: 12 },
    },
  ];
  Plotly.react(el, [
    {
      x: perf.date, y: perf.bench_equity, name: "Buy & hold",
      line: { color: C.bench, width: 2, dash: "dash" },
      hovertemplate: "$%{y:.3f}<extra>Buy & hold</extra>",
    },
    {
      x: perf.date, y: perf.equity, name: "Strategy",
      line: { color: C.strategy, width: 2.5 },
      hovertemplate: "$%{y:.3f}<extra>Strategy</extra>",
    },
  ], layout, CONFIG);
}

export function drawdownChart(el, perf) {
  const layout = clone(BASE);
  layout.margin.r = 16;
  layout.yaxis.tickformat = ".0%";
  Plotly.react(el, [
    {
      x: perf.date, y: perf.drawdown, name: "Drawdown",
      line: { color: C.negative, width: 2 },
      fill: "tozeroy", fillcolor: "rgba(230,103,103,0.12)",
      hovertemplate: "%{y:.1%}<extra>Drawdown</extra>",
    },
  ], layout, CONFIG);
}

export function sentimentBars(el, byTicker) {
  const entries = Object.entries(byTicker).sort((a, b) => a[1] - b[1]);
  const y = entries.map(([t]) => t);
  const x = entries.map(([, v]) => v);
  const layout = clone(BASE);
  layout.margin = { l: 64, r: 56, t: 8, b: 34 };
  layout.hovermode = "closest";
  const span = Math.max(0.05, ...x.map(Math.abs)) * 1.45;
  layout.xaxis.range = [-span, span];
  Plotly.react(el, [
    {
      type: "bar", orientation: "h", x, y,
      marker: { color: x.map((v) => (v >= 0 ? C.strategy : C.negative)) },
      text: x.map((v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2)),
      textposition: "outside", cliponaxis: false,
      textfont: { color: C.ink2 },
      width: 0.55,
      hovertemplate: "%{y}: %{x:+.3f}<extra></extra>",
    },
  ], layout, CONFIG);
}

export function sentimentScatter(el, points) {
  const x = points.map((p) => p.sent_mean);
  const y = points.map((p) => p.next_ret);
  const traces = [
    {
      mode: "markers", x, y,
      marker: { color: C.strategy, size: 9, opacity: 0.6, line: { color: C.surface, width: 1 } },
      customdata: points.map((p) => [p.ticker, p.date.slice(0, 10)]),
      hovertemplate: "%{customdata[0]} %{customdata[1]}<br>sentiment %{x:.2f} → next day %{y:.2%}<extra></extra>",
    },
  ];
  if (x.length >= 3) {
    // least-squares trend
    const n = x.length;
    const mx = x.reduce((a, b) => a + b, 0) / n;
    const my = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) ** 2; }
    const slope = den ? num / den : 0;
    const b = my - slope * mx;
    const xs = [Math.min(...x), Math.max(...x)];
    traces.push({
      mode: "lines", x: xs, y: xs.map((v) => slope * v + b),
      line: { color: C.ink3, width: 2, dash: "dot" }, hoverinfo: "skip",
    });
  }
  const layout = clone(BASE);
  layout.margin = { l: 52, r: 16, t: 8, b: 44 };
  layout.hovermode = "closest";
  layout.xaxis.title = { text: "Headline sentiment (−1 to +1)", font: { size: 11.5, color: C.ink3 } };
  layout.yaxis.title = { text: "Next-day return", font: { size: 11.5, color: C.ink3 } };
  layout.yaxis.tickformat = ".1%";
  Plotly.react(el, traces, layout, CONFIG);
}

export function embeddingCloud(el, returns) {
  if (!returns || returns.length < 3) { Plotly.purge(el); return; }
  const x = returns.slice(0, -2), y = returns.slice(1, -1), z = returns.slice(2);
  const t = [...x.keys()];
  Plotly.react(el, [
    {
      type: "scatter3d", mode: "markers+lines", x, y, z,
      marker: {
        size: 3.5, color: t,
        colorscale: [[0, C.seqLo], [1, C.seqHi]],
        colorbar: { title: { text: "time →", font: { size: 11, color: C.ink3 } }, thickness: 10, tickvals: [], outlinewidth: 0 },
      },
      line: { color: "rgba(57,135,229,0.25)", width: 2 },
      hovertemplate: "r(t)=%{x:.2%}<br>r(t+1)=%{y:.2%}<br>r(t+2)=%{z:.2%}<extra></extra>",
    },
  ], {
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { ...FONT, size: 11 },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      camera: { eye: { x: 1.15, y: 1.15, z: 0.6 } },
      xaxis: { title: "return (t)",   gridcolor: C.grid, zerolinecolor: C.grid, backgroundcolor: "rgba(0,0,0,0)", tickfont: { color: C.ink3 } },
      yaxis: { title: "return (t+1)", gridcolor: C.grid, zerolinecolor: C.grid, backgroundcolor: "rgba(0,0,0,0)", tickfont: { color: C.ink3 } },
      zaxis: { title: "return (t+2)", gridcolor: C.grid, zerolinecolor: C.grid, backgroundcolor: "rgba(0,0,0,0)", tickfont: { color: C.ink3 } },
    },
  }, CONFIG);
}

export function topoTimeline(el, dates, values) {
  const layout = clone(BASE);
  layout.margin.r = 16;
  Plotly.react(el, [
    {
      x: dates, y: values,
      line: { color: C.topology, width: 2 },
      fill: "tozeroy", fillcolor: "rgba(144,133,233,0.10)",
      hovertemplate: "%{y:.4f}<extra>topology</extra>",
    },
  ], layout, CONFIG);
}

export function purgeAll(ids) {
  ids.forEach((id) => { const el = document.getElementById(id); if (el) Plotly.purge(el); });
}
