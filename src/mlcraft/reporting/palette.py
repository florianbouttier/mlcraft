"""Centralized palette and chart colors for HTML reporting."""

from __future__ import annotations

from typing import Mapping

DEFAULT_REPORT_PALETTE: dict[str, str] = {
    "bg_top": "#f6f8fb",
    "bg_bottom": "#eef3f8",
    "panel": "rgba(255, 255, 255, 0.92)",
    "panel_strong": "rgba(255, 255, 255, 0.98)",
    "panel_alt": "rgba(245, 248, 252, 0.92)",
    "border": "rgba(97, 118, 138, 0.16)",
    "border_strong": "rgba(45, 64, 89, 0.18)",
    "text_main": "#10233a",
    "text_soft": "#5d7188",
    "text_inverse": "#f8fbff",
    "accent": "#1e40af",
    "accent_soft": "#dbe7ff",
    "accent_2": "#0f766e",
    "accent_2_soft": "#dbf4ef",
    "danger": "#c2410c",
    "danger_soft": "#fff1e8",
    "shadow": "0 24px 64px rgba(16, 35, 58, 0.08)",
    "shadow_strong": "0 30px 90px rgba(16, 35, 58, 0.12)",
    "series_1": "#1e40af",
    "series_2": "#3b82f6",
    "series_3": "#f59e0b",
    "series_4": "#0f766e",
    "series_5": "#c2410c",
    "series_6": "#7c3aed",
    "series_muted": "#91a8c6",
    "line_soft": "#dce5ef",
    "grid_soft": "#94a3b8",
    "interaction_low": "#ffffff",
    "interaction_mid": "#fecaca",
    "interaction_high": "#b91c1c",
    "font_heading": "\"Fira Code\", \"IBM Plex Mono\", monospace",
    "font_body": "\"Fira Sans\", \"IBM Plex Sans\", \"Segoe UI\", sans-serif",
    "font_mono": "\"Fira Code\", \"IBM Plex Mono\", monospace",
    "radius_panel": "28px",
    "radius_card": "22px",
    "radius_pill": "999px",
    "space_page_max": "1680px",
}


def get_report_palette(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the active report palette.

    Args:
        overrides: Optional palette overrides applied on top of the defaults.

    Returns:
        dict[str, str]: Palette values used by HTML and chart renderers.
    """

    palette = dict(DEFAULT_REPORT_PALETTE)
    if overrides:
        palette.update({str(key): str(value) for key, value in overrides.items()})
    return palette


def chart_colors(palette: Mapping[str, str] | None = None) -> list[str]:
    """Return the chart color sequence used by matplotlib renderers.

    Args:
        palette: Optional resolved palette.

    Returns:
        list[str]: Ordered chart colors.
    """

    resolved = get_report_palette(palette)
    return [resolved[f"series_{index}"] for index in range(1, 7)]


def css_variables(palette: Mapping[str, str] | None = None) -> str:
    """Render the palette as CSS custom properties.

    Args:
        palette: Optional resolved palette.

    Returns:
        str: CSS variable declarations ready to inject in `:root`.
    """

    resolved = get_report_palette(palette)
    return "\n".join(f"        --{key.replace('_', '-')}: {value};" for key, value in resolved.items())
