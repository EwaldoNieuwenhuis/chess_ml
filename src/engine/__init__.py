"""
Stockfish UCI Engine Integration Module.

Provides asynchronous and synchronous UCI communication with Stockfish binaries,
automated binary discovery, on-demand downloads, and python-chess minimax fallbacks.
"""

from __future__ import annotations

from src.engine.discovery import discover_stockfish_binary, download_stockfish_binary
from src.engine.exceptions import (
    EngineError,
    EngineNotFoundError,
    EngineProcessError,
    InvalidFENError,
)
from src.engine.fallback import PythonChessFallbackEngine
from src.engine.manager import AsyncStockfishManager, StockfishManager, validate_fen_and_get_board

__all__ = [
    "StockfishManager",
    "AsyncStockfishManager",
    "PythonChessFallbackEngine",
    "discover_stockfish_binary",
    "download_stockfish_binary",
    "validate_fen_and_get_board",
    "EngineError",
    "InvalidFENError",
    "EngineNotFoundError",
    "EngineProcessError",
]
