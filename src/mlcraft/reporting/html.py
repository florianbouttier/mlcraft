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
