"""HTML rendering helpers shared by reporting modules."""

from __future__ import annotations

import base64
import io
import json
import warnings
from uuid import uuid4

from mlcraft.reporting.palette import css_variables, get_report_palette

BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
      :root {
{{ css_variables }}
      }
      * { box-sizing: border-box; }
      html { scroll-behavior: smooth; }
      body {
        margin: 0;
        color: var(--text-main);
        background:
          radial-gradient(circle at top left, rgba(30, 64, 175, 0.10), transparent 28%),
          radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 32%),
          linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
        font-family: var(--font-body);
      }
      .page {
        max-width: var(--space-page-max);
        margin: 0 auto;
        padding: 34px 26px 72px;
      }
      h1, h2, h3, h4 {
        color: var(--text-main);
        line-height: 1.04;
        margin: 0 0 12px;
      }
      h1, h2 {
        font-family: var(--font-heading);
        font-weight: 700;
      }
      h3, h4 {
        font-family: var(--font-body);
        font-weight: 600;
      }
      h1 { font-size: clamp(2.1rem, 2.5vw, 3rem); letter-spacing: -0.05em; }
      h2 { font-size: clamp(1.35rem, 1.65vw, 2rem); letter-spacing: -0.04em; }
      h3 { font-size: 1.05rem; letter-spacing: -0.02em; }
      h4 { font-size: 0.95rem; letter-spacing: -0.01em; }
      p { margin: 0; line-height: 1.6; }
      section { margin: 0 0 24px; }
      .section-stack {
        display: grid;
        gap: 22px;
      }
      .section-head {
        display: grid;
        gap: 6px;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius-panel);
        padding: 24px;
        box-shadow: var(--shadow);
      }
      .hero-panel {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(241, 246, 255, 0.96));
        border-color: var(--border-strong);
      }
      .viz-grid {
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }
      .kpi-grid {
        display: grid;
        gap: 14px;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      }
      .two-column-grid {
        display: grid;
        gap: 18px;
        grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
      }
      .card {
        border: 1px solid var(--border);
        border-radius: var(--radius-card);
        padding: 22px;
        background: var(--panel-strong);
        min-height: 100%;
      }
      .card--wide { grid-column: 1 / -1; }
      .metric-card {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(241, 246, 255, 0.96));
      }
      .card--ghost {
        background: rgba(255, 255, 255, 0.78);
      }
      .eyebrow {
        display: inline-block;
        margin-bottom: 10px;
        color: var(--text-soft);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .metric-big {
        display: block;
        font-size: clamp(1.45rem, 1.8vw, 2.3rem);
        font-weight: 700;
        letter-spacing: -0.04em;
        margin-bottom: 4px;
      }
      .metric-subtle {
        color: var(--text-soft);
        font-size: 0.95rem;
      }
      .muted { color: var(--text-soft); }
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.38rem 0.78rem;
        border-radius: var(--radius-pill);
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.84rem;
        font-weight: 700;
      }
      .badge--alert {
        background: var(--danger-soft);
        color: var(--danger);
      }
      .chip-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .chip {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 0.65rem 0.9rem;
        border-radius: var(--radius-pill);
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.85);
      }
      .chip strong { font-size: 0.9rem; }
      .segmented {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 6px;
        border-radius: var(--radius-pill);
        background: rgba(233, 239, 247, 0.92);
        border: 1px solid var(--border);
      }
      .segmented-button {
        appearance: none;
        border: 0;
        border-radius: var(--radius-pill);
        background: transparent;
        color: var(--text-soft);
        cursor: pointer;
        font: inherit;
        font-size: 0.93rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.7rem 1rem;
        transition: background 140ms ease, color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
      }
      .segmented-button:hover {
        color: var(--text-main);
        transform: translateY(-1px);
      }
      .segmented-button.is-active {
        background: var(--panel-strong);
        color: var(--accent);
        box-shadow: 0 8px 20px rgba(30, 64, 175, 0.14);
      }
      .summary-table {
        overflow-x: auto;
      }
      .summary-table table {
        min-width: 760px;
      }
      .summary-table tbody tr:hover {
        background: rgba(219, 231, 255, 0.26);
      }
      .table-action {
        appearance: none;
        border: 0;
        background: transparent;
        color: var(--accent);
        cursor: pointer;
        font: inherit;
        font-weight: 700;
        padding: 0;
        text-decoration: underline;
        text-underline-offset: 0.16em;
      }
      .delta {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-weight: 700;
      }
      .delta--positive { color: var(--accent-2); }
      .delta--negative { color: var(--danger); }
      .toggle-panel[hidden] {
        display: none !important;
      }
      .metric-panel {
        display: grid;
        gap: 18px;
      }
      .metric-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .metric-meta .badge {
        background: rgba(241, 245, 252, 0.98);
        color: var(--text-main);
      }
      .viz-shell {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 248, 252, 0.96));
        padding: 16px;
      }
      .viz-shell svg {
        display: block;
        width: 100%;
        height: auto;
      }
      .d3-chart {
        width: 100%;
      }
      .d3-tooltip {
        position: fixed;
        z-index: 9999;
        pointer-events: none;
        opacity: 0;
        transform: translateY(6px);
        transition: opacity 100ms ease, transform 100ms ease;
        min-width: 160px;
        max-width: 280px;
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.98);
        color: var(--text-main);
        box-shadow: var(--shadow-strong);
        font-size: 0.88rem;
        line-height: 1.45;
      }
      .d3-tooltip.is-visible {
        opacity: 1;
        transform: translateY(0);
      }
      .d3-axis text {
        fill: var(--text-soft);
        font-size: 12px;
      }
      .d3-axis path,
      .d3-axis line {
        stroke: var(--line-soft);
      }
      .d3-grid line {
        stroke: var(--line-soft);
        stroke-dasharray: 2 4;
      }
      .d3-grid path {
        display: none;
      }
      .legend-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 12px;
      }
      .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--text-soft);
        font-size: 0.92rem;
      }
      .legend-swatch {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        flex: 0 0 auto;
      }
      .comparison-list {
        display: grid;
        gap: 12px;
      }
      .comparison-row {
        display: grid;
        gap: 8px;
      }
      .comparison-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        flex-wrap: wrap;
      }
      .comparison-title {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
      }
      .comparison-track {
        position: relative;
        height: 12px;
        border-radius: var(--radius-pill);
        background: rgba(220, 229, 239, 0.88);
        overflow: hidden;
      }
      .comparison-fill {
        position: absolute;
        inset: 0 auto 0 0;
        border-radius: var(--radius-pill);
      }
      .mini-grid {
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      }
      .mini-stat {
        padding: 14px 16px;
        border-radius: 18px;
        background: var(--panel-alt);
        border: 1px solid var(--border);
      }
      .mini-stat strong {
        display: block;
        margin-top: 6px;
        font-size: 1.05rem;
      }
      .plot-frame {
        overflow: hidden;
        border-radius: 18px;
        background: white;
        padding: 10px;
        border: 1px solid var(--border);
      }
      .plot-frame img {
        width: 100%;
        display: block;
        height: auto;
      }
      .notes-list {
        display: grid;
        gap: 12px;
      }
      .note-item {
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: var(--panel-alt);
      }
      code, pre {
        font-family: var(--font-mono);
      }
      code {
        background: rgba(226, 232, 240, 0.9);
        padding: 0.16rem 0.38rem;
        border-radius: 8px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.96rem;
      }
      th, td {
        padding: 0.8rem 0.9rem;
        border-bottom: 1px solid var(--border);
        text-align: left;
        vertical-align: middle;
      }
      th {
        color: var(--text-soft);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      @media (max-width: 720px) {
        .page { padding: 20px 14px 40px; }
        .panel { padding: 18px; border-radius: 22px; }
        .viz-grid { grid-template-columns: 1fr; }
        .kpi-grid { grid-template-columns: 1fr; }
        .two-column-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="page">{{ body | safe }}</div>
    <script>
      (() => {
        const tooltip = (() => {
          const node = document.createElement("div");
          node.className = "d3-tooltip";
          document.body.appendChild(node);
          return {
            show(event, html) {
              node.innerHTML = html;
              node.classList.add("is-visible");
              this.move(event);
            },
            move(event) {
              node.style.left = `${event.clientX + 16}px`;
              node.style.top = `${event.clientY + 16}px`;
            },
            hide() {
              node.classList.remove("is-visible");
            },
          };
        })();

        function ensureD3(container) {
          if (window.d3) {
            return true;
          }
          if (container) {
            container.innerHTML = "<p class='muted'>D3.js could not be loaded in this environment.</p>";
          }
          return false;
        }

        function createSvg(container, config = {}) {
          const width = Math.max(config.minWidth || 640, container.clientWidth || config.minWidth || 640);
          const height = config.height || 360;
          const margin = Object.assign({ top: 24, right: 24, bottom: 48, left: 64 }, config.margin || {});
          const svg = window.d3
            .select(container)
            .append("svg")
            .attr("viewBox", `0 0 ${width} ${height}`)
            .attr("class", "d3-chart");
          return { svg, width, height, margin, innerWidth: width - margin.left - margin.right, innerHeight: height - margin.top - margin.bottom };
        }

        function mountBarChart(containerId, payload) {
          const container = document.getElementById(containerId);
          if (!container || !ensureD3(container)) return;
          container.innerHTML = "";
          const d3 = window.d3;
          const rows = payload.rows || [];
          if (!rows.length) {
            container.innerHTML = "<p class='muted'>No data available.</p>";
            return;
          }
          const chart = createSvg(container, { height: Math.max(260, rows.length * 48 + 70), margin: { top: 18, right: 90, bottom: 34, left: 160 } });
          const { svg, width, height, margin, innerWidth, innerHeight } = chart;
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const extent = d3.extent(rows, (d) => Number(d.value));
          const min = Math.min(0, extent[0] ?? 0);
          const max = Math.max(0, extent[1] ?? 1);
          const x = d3.scaleLinear().domain([min, max]).nice().range([0, innerWidth]);
          const y = d3.scaleBand().domain(rows.map((d) => d.label)).range([0, innerHeight]).padding(0.28);
          g.append("g").attr("class", "d3-grid").call(d3.axisBottom(x).ticks(5).tickSize(innerHeight)).attr("transform", "translate(0,0)");
          g.selectAll(".bar")
            .data(rows)
            .enter()
            .append("rect")
            .attr("x", (d) => Math.min(x(0), x(d.value)))
            .attr("y", (d) => y(d.label))
            .attr("width", (d) => Math.abs(x(d.value) - x(0)))
            .attr("height", y.bandwidth())
            .attr("rx", 10)
            .attr("fill", (d) => d.color || "var(--accent)")
            .on("mousemove", (event, d) => tooltip.show(event, `<strong>${d.label}</strong><br>${payload.metricLabel || "value"}: ${Number(d.value).toFixed(6)}`))
            .on("mouseleave", () => tooltip.hide());
          g.append("g").attr("class", "d3-axis").call(d3.axisLeft(y).tickSize(0)).select(".domain").remove();
          g.append("g").attr("class", "d3-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
          g.selectAll(".bar-label")
            .data(rows)
            .enter()
            .append("text")
            .attr("x", (d) => x(d.value) + (d.value >= 0 ? 8 : -8))
            .attr("y", (d) => y(d.label) + y.bandwidth() / 2 + 4)
            .attr("text-anchor", (d) => (d.value >= 0 ? "start" : "end"))
            .attr("fill", "var(--text-main)")
            .attr("font-size", 12)
            .attr("font-weight", 700)
            .text((d) => Number(d.value).toFixed(6));
        }

        function mountHeatmap(containerId, payload) {
          const container = document.getElementById(containerId);
          if (!container || !ensureD3(container)) return;
          container.innerHTML = "";
          const d3 = window.d3;
          const xLabels = payload.xLabels || [];
          const yLabels = payload.yLabels || [];
          const matrix = payload.matrix || [];
          if (!xLabels.length || !yLabels.length) {
            container.innerHTML = "<p class='muted'>No heatmap data available.</p>";
            return;
          }
          const cellSize = 44;
          const chart = createSvg(container, {
            height: Math.max(250, yLabels.length * cellSize + 100),
            minWidth: Math.max(560, xLabels.length * cellSize + 180),
            margin: { top: 18, right: 24, bottom: 90, left: 160 },
          });
          const { svg, margin, innerWidth, innerHeight } = chart;
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const x = d3.scaleBand().domain(xLabels).range([0, innerWidth]).padding(0.08);
          const y = d3.scaleBand().domain(yLabels).range([0, innerHeight]).padding(0.08);
          const values = matrix.flat().filter((value) => value !== null && !Number.isNaN(Number(value)));
          const color = d3.scaleLinear().domain([d3.min(values) ?? 0, d3.max(values) ?? 1]).range([payload.lowColor || "#eff6ff", payload.highColor || "#1d4ed8"]);
          const rows = [];
          matrix.forEach((row, rowIndex) => {
            row.forEach((value, colIndex) => {
              rows.push({ rowLabel: yLabels[rowIndex], colLabel: xLabels[colIndex], value });
            });
          });
          g.selectAll("rect")
            .data(rows)
            .enter()
            .append("rect")
            .attr("x", (d) => x(d.colLabel))
            .attr("y", (d) => y(d.rowLabel))
            .attr("width", x.bandwidth())
            .attr("height", y.bandwidth())
            .attr("rx", 8)
            .attr("fill", (d) => (d.value === null || Number.isNaN(Number(d.value)) ? "#f8fafc" : color(Number(d.value))))
            .on("mousemove", (event, d) => tooltip.show(event, `<strong>${d.rowLabel}</strong><br>${d.colLabel}: ${d.value === null || Number.isNaN(Number(d.value)) ? "n/a" : Number(d.value).toFixed(6)}`))
            .on("mouseleave", () => tooltip.hide());
          g.selectAll(".cell-label")
            .data(rows.filter((d) => d.value !== null && !Number.isNaN(Number(d.value))))
            .enter()
            .append("text")
            .attr("x", (d) => x(d.colLabel) + x.bandwidth() / 2)
            .attr("y", (d) => y(d.rowLabel) + y.bandwidth() / 2 + 4)
            .attr("text-anchor", "middle")
            .attr("font-size", 11)
            .attr("font-weight", 700)
            .attr("fill", "white")
            .text((d) => Number(d.value).toFixed(3));
          g.append("g").attr("class", "d3-axis").call(d3.axisLeft(y).tickSize(0)).select(".domain").remove();
          g.append("g").attr("class", "d3-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).tickSize(0)).selectAll("text").attr("transform", "rotate(-25)").style("text-anchor", "end");
        }

        function mountSeriesChart(containerId, payload) {
          const container = document.getElementById(containerId);
          if (!container || !ensureD3(container)) return;
          container.innerHTML = "";
          const d3 = window.d3;
          const series = payload.series || [];
          if (!series.length) {
            container.innerHTML = "<p class='muted'>No series data available.</p>";
            return;
          }
          const points = series.flatMap((item) => item.points || []);
          const chart = createSvg(container, { height: payload.height || 340, margin: { top: 20, right: 26, bottom: 46, left: 58 } });
          const { svg, margin, innerWidth, innerHeight } = chart;
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const xValues = points.map((point) => Number(point.x));
          const yValues = points.map((point) => Number(point.y));
          const x = d3.scaleLinear().domain(d3.extent(xValues)).nice().range([0, innerWidth]);
          const y = d3.scaleLinear().domain(d3.extent(yValues)).nice().range([innerHeight, 0]);
          g.append("g").attr("class", "d3-grid").call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth));
          g.append("g").attr("class", "d3-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
          g.append("g").attr("class", "d3-axis").call(d3.axisLeft(y).ticks(5));
          const line = d3.line().x((d) => x(Number(d.x))).y((d) => y(Number(d.y)));
          if (payload.diagonal) {
            g.append("line")
              .attr("x1", x(payload.diagonal[0][0]))
              .attr("y1", y(payload.diagonal[0][1]))
              .attr("x2", x(payload.diagonal[1][0]))
              .attr("y2", y(payload.diagonal[1][1]))
              .attr("stroke", payload.diagonalColor || "var(--grid-soft)")
              .attr("stroke-dasharray", "4 4")
              .attr("stroke-width", 1.5);
          }
          series.forEach((item) => {
            g.append("path")
              .datum(item.points)
              .attr("fill", "none")
              .attr("stroke", item.color)
              .attr("stroke-width", 2.6)
              .attr("stroke-dasharray", item.dash || null)
              .attr("d", line);
            g.selectAll(`circle-${item.name}`)
              .data(item.points)
              .enter()
              .append("circle")
              .attr("cx", (d) => x(Number(d.x)))
              .attr("cy", (d) => y(Number(d.y)))
              .attr("r", 4.2)
              .attr("fill", item.color)
              .on("mousemove", (event, d) => tooltip.show(event, `<strong>${item.name}</strong><br>${payload.xLabel || "x"}: ${Number(d.x).toFixed(4)}<br>${payload.yLabel || "y"}: ${Number(d.y).toFixed(4)}`))
              .on("mouseleave", () => tooltip.hide());
          });
        }

        function mountDumbbell(containerId, payload) {
          const container = document.getElementById(containerId);
          if (!container || !ensureD3(container)) return;
          container.innerHTML = "";
          const d3 = window.d3;
          const rows = payload.rows || [];
          if (!rows.length) {
            container.innerHTML = "<p class='muted'>No fold data available.</p>";
            return;
          }
          const chart = createSvg(container, { height: Math.max(250, rows.length * 44 + 60), margin: { top: 18, right: 80, bottom: 38, left: 130 } });
          const { svg, margin, innerWidth, innerHeight } = chart;
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const values = rows.flatMap((row) => [Number(row.train_value), Number(row.val_value)]).filter((value) => !Number.isNaN(value));
          const x = d3.scaleLinear().domain(d3.extent(values)).nice().range([0, innerWidth]);
          const y = d3.scaleBand().domain(rows.map((row) => row.label)).range([0, innerHeight]).padding(0.32);
          g.append("g").attr("class", "d3-grid").call(d3.axisBottom(x).ticks(5).tickSize(innerHeight)).attr("transform", "translate(0,0)");
          g.selectAll(".segment")
            .data(rows)
            .enter()
            .append("line")
            .attr("x1", (d) => x(Number(d.train_value)))
            .attr("x2", (d) => x(Number(d.val_value)))
            .attr("y1", (d) => y(d.label) + y.bandwidth() / 2)
            .attr("y2", (d) => y(d.label) + y.bandwidth() / 2)
            .attr("stroke", (d) => d.color || "var(--accent-2)")
            .attr("stroke-width", 4)
            .attr("stroke-linecap", "round");
          g.selectAll(".train-dot")
            .data(rows)
            .enter()
            .append("circle")
            .attr("cx", (d) => x(Number(d.train_value)))
            .attr("cy", (d) => y(d.label) + y.bandwidth() / 2)
            .attr("r", 5.4)
            .attr("fill", payload.trainColor || "var(--accent)")
            .on("mousemove", (event, d) => tooltip.show(event, `<strong>${d.label}</strong><br>Train: ${Number(d.train_value).toFixed(6)}<br>Validation: ${Number(d.val_value).toFixed(6)}`))
            .on("mouseleave", () => tooltip.hide());
          g.selectAll(".val-dot")
            .data(rows)
            .enter()
            .append("circle")
            .attr("cx", (d) => x(Number(d.val_value)))
            .attr("cy", (d) => y(d.label) + y.bandwidth() / 2)
            .attr("r", 5.4)
            .attr("fill", payload.validationColor || "var(--series-2)");
          g.append("g").attr("class", "d3-axis").call(d3.axisLeft(y).tickSize(0)).select(".domain").remove();
          g.append("g").attr("class", "d3-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
        }

        function mountScatter(containerId, payload) {
          const container = document.getElementById(containerId);
          if (!container || !ensureD3(container)) return;
          container.innerHTML = "";
          const d3 = window.d3;
          const points = payload.points || [];
          if (!points.length) {
            container.innerHTML = "<p class='muted'>No scatter data available.</p>";
            return;
          }
          const chart = createSvg(container, { height: payload.height || 340, margin: { top: 20, right: 24, bottom: 42, left: 56 } });
          const { svg, margin, innerWidth, innerHeight } = chart;
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const x = d3.scaleLinear().domain(d3.extent(points, (d) => Number(d.x))).nice().range([0, innerWidth]);
          const y = d3.scaleLinear().domain(d3.extent(points, (d) => Number(d.y))).nice().range([innerHeight, 0]);
          g.append("g").attr("class", "d3-grid").call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth));
          g.append("g").attr("class", "d3-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
          g.append("g").attr("class", "d3-axis").call(d3.axisLeft(y).ticks(5));
          g.selectAll("circle")
            .data(points)
            .enter()
            .append("circle")
            .attr("cx", (d) => x(Number(d.x)))
            .attr("cy", (d) => y(Number(d.y)))
            .attr("r", 4.2)
            .attr("fill", payload.color || "var(--series-2)")
            .attr("opacity", 0.72)
            .on("mousemove", (event, d) => tooltip.show(event, `<strong>${payload.label || "Point"}</strong><br>${payload.xLabel || "x"}: ${Number(d.x).toFixed(4)}<br>${payload.yLabel || "y"}: ${Number(d.y).toFixed(4)}`))
            .on("mouseleave", () => tooltip.hide());
        }

        function mountBeeswarm(containerId, payload) {
          const container = document.getElementById(containerId);
          if (!container || !ensureD3(container)) return;
          container.innerHTML = "";
          const d3 = window.d3;
          const groups = payload.groups || [];
          if (!groups.length) {
            container.innerHTML = "<p class='muted'>No SHAP values available.</p>";
            return;
          }
          const chart = createSvg(container, { height: Math.max(280, groups.length * 42 + 60), margin: { top: 18, right: 22, bottom: 36, left: 140 } });
          const { svg, margin, innerWidth, innerHeight } = chart;
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const values = groups.flatMap((group) => group.values.map((value) => Number(value)));
          const x = d3.scaleLinear().domain(d3.extent(values)).nice().range([0, innerWidth]);
          const y = d3.scaleBand().domain(groups.map((group) => group.label)).range([0, innerHeight]).padding(0.3);
          g.append("g").attr("class", "d3-grid").call(d3.axisBottom(x).ticks(5).tickSize(innerHeight)).attr("transform", "translate(0,0)");
          groups.forEach((group, groupIndex) => {
            const baseY = y(group.label) + y.bandwidth() / 2;
            g.selectAll(`circle-${groupIndex}`)
              .data(group.values.map((value, index) => ({ value: Number(value), offset: ((index % 9) - 4) * 2.2 })))
              .enter()
              .append("circle")
              .attr("cx", (d) => x(d.value))
              .attr("cy", (d) => baseY + d.offset)
              .attr("r", 3.6)
              .attr("fill", group.color || "var(--accent)")
              .attr("opacity", 0.68)
              .on("mousemove", (event, d) => tooltip.show(event, `<strong>${group.label}</strong><br>SHAP: ${d.value.toFixed(4)}`))
              .on("mouseleave", () => tooltip.hide());
          });
          g.append("g").attr("class", "d3-axis").call(d3.axisLeft(y).tickSize(0)).select(".domain").remove();
          g.append("g").attr("class", "d3-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
        }

        const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        function activateToggle(groupName, targetId) {
          const buttons = Array.from(document.querySelectorAll(`[data-toggle-button][data-toggle-group="${groupName}"]`));
          const panels = Array.from(document.querySelectorAll(`[data-toggle-panel][data-toggle-group="${groupName}"]`));
          for (const button of buttons) {
            const isActive = button.dataset.toggleTarget === targetId;
            button.classList.toggle("is-active", isActive);
            button.setAttribute("aria-pressed", isActive ? "true" : "false");
          }
          for (const panel of panels) {
            panel.hidden = panel.dataset.togglePanel !== targetId;
          }
        }

        const buttons = Array.from(document.querySelectorAll("[data-toggle-button][data-toggle-group][data-toggle-target]"));
        for (const button of buttons) {
          button.addEventListener("click", () => {
            const groupName = button.dataset.toggleGroup;
            const targetId = button.dataset.toggleTarget;
            activateToggle(groupName, targetId);
            const scrollId = button.dataset.toggleScroll;
            if (scrollId) {
              const node = document.getElementById(scrollId);
              if (node) {
                node.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
              }
            }
          });
        }

        const groups = new Set(buttons.map((button) => button.dataset.toggleGroup));
        for (const groupName of groups) {
          const groupButtons = buttons.filter((button) => button.dataset.toggleGroup === groupName);
          const defaultButton = groupButtons.find((button) => button.dataset.toggleDefault === "true") || groupButtons[0];
          if (defaultButton) {
            activateToggle(groupName, defaultButton.dataset.toggleTarget);
          }
        }

        function flushPendingCharts() {
          const queue = window.mlcraftPendingCharts || [];
          while (queue.length) {
            const item = queue.shift();
            const mount = window.mlcraftD3 && window.mlcraftD3[item.chartType];
            if (typeof mount === "function") {
              mount(item.containerId, item.payload || {});
            }
          }
        }

        window.mlcraftD3 = {
          mountBarChart,
          mountHeatmap,
          mountSeriesChart,
          mountDumbbell,
          mountScatter,
          mountBeeswarm,
          flushPendingCharts,
        };
        flushPendingCharts();
      })();
    </script>
  </body>
</html>
"""


def make_environment(*, palette=None):
    """Create the shared Jinja environment used by HTML renderers.

    Args:
        palette: Optional palette override used by the base template.

    Returns:
        Environment: Configured Jinja environment with the base template
        registered in memory.
    """

    from jinja2 import DictLoader, Environment, select_autoescape

    return Environment(
        loader=DictLoader({"base.html": BASE_TEMPLATE}),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def figure_to_data_uri(figure) -> str:
    """Encode a matplotlib figure as a PNG data URI.

    Args:
        figure: Matplotlib figure instance to encode.

    Returns:
        str: `data:` URI that can be embedded directly into HTML.
    """

    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        figure.tight_layout()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    buffer.seek(0)
    payload = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def wrap_html(title: str, body: str, *, palette=None) -> str:
    """Wrap rendered HTML fragments into the shared document shell.

    Args:
        title: Document title.
        body: Pre-rendered HTML body content.
        palette: Optional palette override.

    Returns:
        str: Standalone HTML document.
    """

    env = make_environment(palette=palette)
    template = env.get_template("base.html")
    return template.render(title=title, body=body, css_variables=css_variables(get_report_palette(palette)))


def render_d3_card(
    title: str,
    chart_type: str,
    payload: dict,
    *,
    wide: bool = False,
    chart_id: str | None = None,
) -> str:
    """Render a D3-driven chart card with inline payload mounting code.

    Args:
        title: Visible card title.
        chart_type: Registered chart type name understood by `window.mlcraftD3`.
        payload: JSON-serializable payload used to mount the chart.
        wide: Whether the card should span the full section width.
        chart_id: Optional explicit chart container identifier.

    Returns:
        str: HTML fragment containing the D3 chart container and mount script.
    """

    resolved_chart_id = chart_id or f"chart-{uuid4().hex}"
    wide_class = " card--wide" if wide else ""
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return (
        f"<div class='card{wide_class}'>"
        f"<span class='eyebrow'>{title}</span>"
        f"<div id='{resolved_chart_id}' class='viz-shell'></div>"
        "<script>"
        "window.mlcraftPendingCharts = window.mlcraftPendingCharts || [];"
        f"window.mlcraftPendingCharts.push({{chartType:'{chart_type}',containerId:'{resolved_chart_id}',payload:{payload_json}}});"
        "if (window.mlcraftD3 && typeof window.mlcraftD3['flushPendingCharts'] === 'function') { window.mlcraftD3.flushPendingCharts(); }"
        "</script>"
        "</div>"
    )
