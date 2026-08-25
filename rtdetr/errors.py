# Apache-2.0
"""Errors this package raises. Kept in one place so they are easy to catch."""

from __future__ import annotations


class RTDETRError(Exception):
    """Base class for every error raised by rtdetr."""


class ModelNotFoundError(RTDETRError, FileNotFoundError):
    """A model name could not be resolved to weights (locally or on the mirror)."""


class DownloadError(RTDETRError, OSError):
    """A weight download failed."""
