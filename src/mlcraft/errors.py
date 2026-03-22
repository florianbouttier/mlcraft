"""Custom exceptions for mlcraft."""


class MlcraftError(Exception):
    """Base exception for the package."""


class OptionalDependencyError(MlcraftError):
    """Raised when an optional dependency is required but unavailable."""


class SchemaInferenceError(MlcraftError):
    """Raised when schema inference fails."""


class BackendNotAvailableError(OptionalDependencyError):
    """Raised when a model backend is not installed."""


class InvalidConfigurationError(MlcraftError):
    """Raised when a public configuration payload is inconsistent."""

