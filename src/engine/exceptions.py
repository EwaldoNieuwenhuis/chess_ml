"""
Custom exceptions for the Stockfish UCI Engine Manager.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base exception for all chess engine manager operations."""


class InvalidFENError(EngineError):
    """Raised when a FEN string is invalid, malformed, or represents an illegal position."""


class EngineNotFoundError(EngineError):
    """Raised when a Stockfish executable binary cannot be found or downloaded."""


class EngineProcessError(EngineError):
    """Raised when the UCI engine process encounters an unexpected crash, pipe failure, or error."""
