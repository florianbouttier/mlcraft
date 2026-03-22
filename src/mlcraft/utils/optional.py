"""Helpers for optional dependencies."""

from __future__ import annotations

import importlib

from mlcraft.errors import OptionalDependencyError


def dependency_available(module_name: str) -> bool:
    """Return whether an optional dependency can be imported.

    Args:
        module_name: Importable module name.

    Returns:
        bool: `True` when the dependency is available.
    """

    return importlib.util.find_spec(module_name) is not None


def optional_import(module_name: str, *, extra_name: str | None = None):
    """Import an optional dependency or raise a package-specific error.

    Args:
        module_name: Importable module name.
        extra_name: Optional extras name suggested in the error message.

    Returns:
        module: Imported Python module.

    Raises:
        OptionalDependencyError: If the dependency is missing.
    """

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        suffix = f" Install the '{extra_name}' extra." if extra_name else ""
        raise OptionalDependencyError(f"Optional dependency '{module_name}' is required.{suffix}") from exc
