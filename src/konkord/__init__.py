"""Konkord — LLM eval harness with judge calibration."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("konkord")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
