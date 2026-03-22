"""HTML rendering helpers shared by reporting modules."""

from __future__ import annotations

import base64
import io

BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>
      body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; margin: 2rem; color: #1f2933; }
      h1, h2, h3 { color: #102a43; }
      table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
      th, td { border: 1px solid #d9e2ec; padding: 0.5rem; text-align: left; }
      th { background: #f0f4f8; }
      .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
      .card { border: 1px solid #d9e2ec; border-radius: 8px; padding: 1rem; background: white; }
      .muted { color: #52606d; }
      img { max-width: 100%; }
      code { background: #f0f4f8; padding: 0.1rem 0.3rem; border-radius: 4px; }
    </style>
  </head>
  <body>
    {{ body | safe }}
  </body>
</html>
"""


def make_environment():
    """Create the shared Jinja environment used by HTML renderers.

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
    figure.tight_layout()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    payload = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def wrap_html(title: str, body: str) -> str:
    """Wrap rendered HTML fragments into the shared document shell.

    Args:
        title: Document title.
        body: Pre-rendered HTML body content.

    Returns:
        str: Standalone HTML document.
    """

    env = make_environment()
    template = env.get_template("base.html")
    return template.render(title=title, body=body)
