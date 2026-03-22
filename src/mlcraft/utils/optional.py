"""Helpers for optional dependencies."""

from __future__ import annotations

import importlib

from mlcraft.errors import OptionalDependencyError


def dependency_available(module_name: str) -> bool:
    """Return whether a module can be imported."""

    return importlib.util.find_spec(module_name) is not None


def optional_import(module_name: str, *, extra_name: str | None = None):
    """Import an optional dependency or raise a clear package-specific error."""

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        suffix = f" Install the '{extra_name}' extra." if extra_name else ""
        raise OptionalDependencyError(f"Optional dependency '{module_name}' is required.{suffix}") from exc

