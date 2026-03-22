"""Centralized palette and chart colors for HTML reporting."""

from __future__ import annotations

from typing import Mapping

DEFAULT_REPORT_PALETTE: dict[str, str] = {
    "bg_top": "#f4efe7",
    "bg_bottom": "#e9f2f5",
    "panel": "rgba(255, 255, 255, 0.88)",
    "panel_strong": "rgba(255, 255, 255, 0.96)",
    "border": "rgba(127, 154, 173, 0.22)",
    "text_main": "#16324f",
    "text_soft": "#5b7083",
    "accent": "#0f766e",
    "accent_soft": "#d9f3ef",
    "danger": "#c2410c",
    "danger_soft": "#fff1e8",
    "shadow": "0 24px 60px rgba(22, 50, 79, 0.08)",
    "series_1": "#0f766e",
    "series_2": "#2563eb",
    "series_3": "#d97706",
    "series_4": "#7c3aed",
    "series_5": "#c2410c",
    "series_6": "#0891b2",
    "series_muted": "#7aa6c2",
    "line_soft": "#d7e3ec",
    "grid_soft": "#94a3b8",
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
