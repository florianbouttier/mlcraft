"""HTML rendering helpers shared by reporting modules."""

from __future__ import annotations

import base64
import io
import warnings

from mlcraft.reporting.palette import css_variables, get_report_palette

BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    <style>
      :root {
{{ css_variables }}
        --plot-scale: 0.92;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--text-main);
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
          radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 30%),
          linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
        font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      }
      .page {
        max-width: 1820px;
        margin: 0 auto;
        padding: 32px 28px 72px;
      }
      .page-tools {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin: 0 0 24px;
        flex-wrap: wrap;
      }
      h1, h2, h3 {
        color: var(--text-main);
        line-height: 1.05;
        margin: 0 0 12px;
      }
      h1 { font-size: clamp(2.3rem, 3vw, 3.4rem); letter-spacing: -0.04em; }
      h2 { font-size: clamp(1.5rem, 2vw, 2.15rem); letter-spacing: -0.03em; }
      h3 { font-size: 1.05rem; letter-spacing: -0.02em; }
      p { margin: 0; line-height: 1.6; }
      section { margin: 0 0 26px; }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
      }
      .hero-panel {
        background:
          linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(247, 251, 252, 0.92) 48%, rgba(224, 247, 243, 0.92) 100%);
      }
      .section-stack {
        display: grid;
        gap: 26px;
      }
      .viz-grid {
        display: grid;
        gap: 28px;
        grid-template-columns: 1fr;
      }
      .viz-grid--compact {
        grid-template-columns: 1fr;
      }
      .kpi-grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }
      .card {
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 22px;
        background: var(--panel-strong);
        min-height: 100%;
      }
      .card--wide { grid-column: 1 / -1; }
      .metric-card {
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(240, 249, 250, 0.92));
      }
      .card--ghost {
        background: rgba(255, 255, 255, 0.72);
      }
      .eyebrow {
        display: inline-block;
        margin-bottom: 10px;
        color: var(--text-soft);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .metric-big {
        display: block;
        font-size: clamp(1.55rem, 2vw, 2.5rem);
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
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
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
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.85);
      }
      .chip strong { font-size: 0.9rem; }
      .plot-frame {
        overflow: auto;
        border-radius: 18px;
        background: white;
        padding: 10px;
      }
      .plot-frame img {
        width: calc(100% * var(--plot-scale, 0.92));
        display: block;
        height: auto;
        max-width: none;
        margin: 0 auto;
        transition: width 120ms ease-out;
      }
      .plot-scale-toolbar {
        display: inline-flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
      }
      .plot-scale-button {
        appearance: none;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.92);
        color: var(--text-main);
        cursor: pointer;
        font: inherit;
        font-size: 0.9rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.6rem 0.9rem;
        transition: background 120ms ease-out, border-color 120ms ease-out, color 120ms ease-out;
      }
      .plot-scale-button:hover {
        border-color: var(--accent);
      }
      .plot-scale-button.is-active {
        background: var(--accent-soft);
        border-color: var(--accent);
        color: var(--accent);
      }
      .plotly-card .js-plotly-plot,
      .plotly-card .plot-container {
        width: 100% !important;
      }
      .plotly-card .main-svg {
        border-radius: 18px;
      }
      .section-divider {
        height: 1px;
        border: 0;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 8px 0;
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
      }
    </style>
  </head>
  <body>
    <div class="page">
      <section class="panel card--ghost page-tools">
        <div>
          <span class="eyebrow">Plot Zoom</span>
          <p class="muted">Reduce plot height or zoom back to full size directly in the report.</p>
        </div>
        <div class="plot-scale-toolbar" role="group" aria-label="Plot scale controls">
          <button class="plot-scale-button" type="button" data-plot-scale="0.85">85%</button>
          <button class="plot-scale-button" type="button" data-plot-scale="0.92">92%</button>
          <button class="plot-scale-button" type="button" data-plot-scale="1">100%</button>
          <button class="plot-scale-button" type="button" data-plot-scale="1.15">115%</button>
        </div>
      </section>
      {{ body | safe }}
    </div>
    <script>
      (() => {
        const root = document.documentElement;
        const buttons = Array.from(document.querySelectorAll("[data-plot-scale]"));

        function setPlotScale(scale) {
          root.style.setProperty("--plot-scale", String(scale));
          for (const button of buttons) {
            button.classList.toggle("is-active", Math.abs(Number(button.dataset.plotScale) - scale) < 1e-9);
          }
        }

        for (const button of buttons) {
          button.addEventListener("click", () => {
            setPlotScale(Number(button.dataset.plotScale));
          });
        }

        setPlotScale(0.92);
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
