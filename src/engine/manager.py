"""
Stockfish UCI Engine Manager & Query Interface.

Provides persistent synchronous and asynchronous UCI process session management,
Windows popup suppression, multi-tier binary discovery, automated downloads,
board validation, terminal state pre-evaluation, and fallback minimax execution.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import chess
import chess.engine
from src.engine.discovery import discover_stockfish_binary, download_stockfish_binary
from src.engine.exceptions import (
    EngineNotFoundError,
    EngineProcessError,
    InvalidFENError,
)
from src.engine.fallback import PythonChessFallbackEngine
from src.schemas.contracts import EngineEvaluation

logger = logging.getLogger(__name__)


def _get_subprocess_popen_args() -> dict[str, Any]:
    """
    Returns platform-specific subprocess creation flags to suppress console popups on Windows.
    """
    popen_args: dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        popen_args["startupinfo"] = startupinfo
        popen_args["creationflags"] = subprocess.CREATE_NO_WINDOW
    return popen_args


def validate_fen_and_get_board(fen: str) -> chess.Board:
    """
    Validates a FEN string and ensures it represents a legal chess board state.

    Args:
        fen: Forsyth-Edwards Notation string.

    Returns:
        chess.Board instance.

    Raises:
        InvalidFENError: If FEN is syntactically invalid or describes an illegal chess position.
    """
    fen_clean = fen.strip()
    if not fen_clean:
        raise InvalidFENError("FEN string cannot be empty.")

    try:
        board = chess.Board(fen_clean)
    except ValueError as e:
        raise InvalidFENError(f"Malformed FEN syntax: '{fen_clean}' ({e})") from e

    if not board.is_valid():
        status = board.status()
        raise InvalidFENError(f"Illegal chess board position for FEN '{fen_clean}' (status: {status})")

    return board


class StockfishManager:
    """
    Synchronous Stockfish UCI Engine Manager with persistent process pooling.
    """

    def __init__(
        self,
        binary_path: Path | str | None = None,
        auto_download: bool = True,
        fallback_to_minimax: bool = True,
        threads: int = 1,
        hash_size_mb: int = 64,
        skill_level: int = 20,
        default_depth: int = 15,
        default_time_limit: float | None = None,
    ) -> None:
        self.auto_download = auto_download
        self.fallback_to_minimax = fallback_to_minimax
        self.threads = threads
        self.hash_size_mb = hash_size_mb
        self.skill_level = max(0, min(20, skill_level))
        self.default_depth = default_depth
        self.default_time_limit = default_time_limit

        self.binary_path: Path | None = None
        self._engine: chess.engine.SimpleEngine | None = None
        self._fallback_engine = PythonChessFallbackEngine(default_depth=min(default_depth, 3))
        self._is_fallback_mode = False

        self._resolve_and_init(binary_path)

    def _resolve_and_init(self, custom_path: Path | str | None) -> None:
        # 1. Discover existing binary
        found_path = discover_stockfish_binary(custom_path)

        # 2. Try on-demand download if missing and enabled
        if not found_path and self.auto_download:
            try:
                found_path = download_stockfish_binary()
            except Exception as e:
                logger.warning("Automated Stockfish download failed: %s", e)

        # 3. Fallback or error
        if not found_path or not found_path.is_file():
            if self.fallback_to_minimax:
                logger.info("Stockfish binary not available; initializing in python-chess minimax fallback mode.")
                self._is_fallback_mode = True
                return
            raise EngineNotFoundError(
                "Stockfish binary could not be found or downloaded, and fallback_to_minimax is disabled."
            )

        self.binary_path = found_path
        self._init_engine()

    def _init_engine(self) -> None:
        if not self.binary_path:
            return

        popen_args = _get_subprocess_popen_args()
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                str(self.binary_path),
                **popen_args,
            )
            # Configure UCI options
            options: dict[str, Any] = {
                "Threads": self.threads,
                "Hash": self.hash_size_mb,
            }
            if "Skill Level" in self._engine.options:
                options["Skill Level"] = self.skill_level

            self._engine.configure(options)
            self._is_fallback_mode = False
            logger.info("Initialized Stockfish UCI engine session at %s", self.binary_path)
        except Exception as e:
            if self.fallback_to_minimax:
                logger.warning(
                    "Failed to launch Stockfish binary at %s (%s). Falling back to internal minimax engine.",
                    self.binary_path,
                    e,
                )
                self._is_fallback_mode = True
                self._engine = None
            else:
                raise EngineProcessError(f"Failed to spawn Stockfish UCI subprocess: {e}") from e

    @property
    def is_fallback_mode(self) -> bool:
        """Returns True if operating in pure-python minimax fallback mode."""
        return self._is_fallback_mode

    def evaluate(
        self,
        fen: str,
        depth: int | None = None,
        time_limit: float | None = None,
    ) -> EngineEvaluation:
        """
        Evaluates a chess position given its FEN string.

        Args:
            fen: Forsyth-Edwards Notation string.
            depth: Search depth (defaults to self.default_depth).
            time_limit: Search time limit in seconds (defaults to self.default_time_limit).

        Returns:
            EngineEvaluation contract with best move, score (centipawns or mate), ponder, and PV.

        Raises:
            InvalidFENError: If FEN is malformed or represents an illegal position.
            EngineProcessError: If communication with the UCI engine fails.
        """
        board = validate_fen_and_get_board(fen)
        search_depth = depth if depth is not None else self.default_depth
        search_time = time_limit if time_limit is not None else self.default_time_limit

        # Terminal state handling
        if board.is_game_over():
            if board.is_checkmate():
                return EngineEvaluation(
                    best_move_uci="",
                    best_move_san="",
                    score_cp=None,
                    score_mate=0,
                    depth=0,
                    ponder_move_uci=None,
                    pv=[],
                )
            return EngineEvaluation(
                best_move_uci="",
                best_move_san="",
                score_cp=0,
                score_mate=None,
                depth=0,
                ponder_move_uci=None,
                pv=[],
            )

        # Minimax fallback
        if self._is_fallback_mode or self._engine is None:
            return self._fallback_engine.evaluate(board, depth=min(search_depth, 3))

        # UCI Engine evaluation
        try:
            limit = chess.engine.Limit(depth=search_depth, time=search_time)
            info = self._engine.analyse(board, limit)
            return self._parse_analysis_info(board, info, search_depth)
        except Exception as e:
            logger.error("Error during UCI engine analysis: %s", e)
            if self.fallback_to_minimax:
                logger.warning("Recovering from engine error using minimax fallback.")
                return self._fallback_engine.evaluate(board, depth=min(search_depth, 3))
            raise EngineProcessError(f"UCI engine query failed: {e}") from e

    def get_top_moves(
        self,
        fen: str,
        multipv: int = 3,
        depth: int | None = None,
        time_limit: float | None = None,
    ) -> list[EngineEvaluation]:
        """
        Calculates the top N alternative moves for a chess position.

        Args:
            fen: Forsyth-Edwards Notation string.
            multipv: Number of principal variations (PVs) to return.
            depth: Search depth.
            time_limit: Search time limit in seconds.

        Returns:
            List of EngineEvaluation objects ordered from best to worst.
        """
        board = validate_fen_and_get_board(fen)
        search_depth = depth if depth is not None else self.default_depth
        search_time = time_limit if time_limit is not None else self.default_time_limit

        if board.is_game_over() or self._is_fallback_mode or self._engine is None:
            # Fallback or terminal: return single best move evaluation
            return [self.evaluate(fen, depth=search_depth, time_limit=search_time)]

        try:
            limit = chess.engine.Limit(depth=search_depth, time=search_time)
            infos = self._engine.analyse(board, limit, multipv=multipv)
            if isinstance(infos, dict):
                infos = [infos]

            results: list[EngineEvaluation] = []
            for info in infos:
                results.append(self._parse_analysis_info(board, info, search_depth))
            return results
        except Exception as e:
            logger.error("Error during MultiPV UCI analysis: %s", e)
            if self.fallback_to_minimax:
                return [self._fallback_engine.evaluate(board, depth=min(search_depth, 3))]
            raise EngineProcessError(f"MultiPV UCI analysis failed: {e}") from e

    def _parse_analysis_info(
        self,
        board: chess.Board,
        info: chess.engine.InfoDict,
        requested_depth: int,
    ) -> EngineEvaluation:
        pv_moves: list[chess.Move] = info.get("pv", [])
        if not pv_moves:
            first_legal = next(iter(board.legal_moves), None)
            if first_legal:
                pv_moves = [first_legal]

        best_move_uci = pv_moves[0].uci() if pv_moves else ""
        try:
            best_move_san = board.san(pv_moves[0]) if pv_moves else ""
        except Exception:
            best_move_san = best_move_uci

        ponder_move_uci = pv_moves[1].uci() if len(pv_moves) > 1 else None
        pv_uci = [m.uci() for m in pv_moves]

        pov_score: chess.engine.PovScore | None = info.get("score")
        score_cp: int | None = None
        score_mate: int | None = None

        if pov_score is not None:
            rel_score = pov_score.relative
            if rel_score.is_mate():
                score_mate = rel_score.mate()
            else:
                score_cp = rel_score.score()

        reached_depth = info.get("depth", requested_depth)

        return EngineEvaluation(
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            score_cp=score_cp,
            score_mate=score_mate,
            depth=reached_depth,
            ponder_move_uci=ponder_move_uci,
            pv=pv_uci,
        )

    def close(self) -> None:
        """Closes the underlying UCI engine process cleanly."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def quit(self) -> None:
        """Alias for close()."""
        self.close()

    def __enter__(self) -> StockfishManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class AsyncStockfishManager:
    """
    Asynchronous Stockfish UCI Engine Manager with non-blocking evaluation methods.
    """

    def __init__(
        self,
        binary_path: Path | str | None = None,
        auto_download: bool = True,
        fallback_to_minimax: bool = True,
        threads: int = 1,
        hash_size_mb: int = 64,
        skill_level: int = 20,
        default_depth: int = 15,
        default_time_limit: float | None = None,
    ) -> None:
        self.custom_path = binary_path
        self.auto_download = auto_download
        self.fallback_to_minimax = fallback_to_minimax
        self.threads = threads
        self.hash_size_mb = hash_size_mb
        self.skill_level = max(0, min(20, skill_level))
        self.default_depth = default_depth
        self.default_time_limit = default_time_limit

        self.binary_path: Path | None = None
        self._transport: asyncio.SubprocessTransport | None = None
        self._protocol: chess.engine.UciProtocol | None = None
        self._fallback_engine = PythonChessFallbackEngine(default_depth=min(default_depth, 3))
        self._is_fallback_mode = False
        self._initialized = False

    async def initialize(self) -> None:
        """Asynchronously initializes the UCI engine process."""
        if self._initialized:
            return

        found_path = discover_stockfish_binary(self.custom_path)
        if not found_path and self.auto_download:
            try:
                found_path = download_stockfish_binary()
            except Exception as e:
                logger.warning("Automated Stockfish download failed: %s", e)

        if not found_path or not found_path.is_file():
            if self.fallback_to_minimax:
                self._is_fallback_mode = True
                self._initialized = True
                return
            raise EngineNotFoundError("Stockfish binary not found and fallback disabled.")

        self.binary_path = found_path
        popen_args = _get_subprocess_popen_args()

        try:
            transport, protocol = await chess.engine.popen_uci(
                str(self.binary_path),
                **popen_args,
            )
            self._transport = transport
            self._protocol = protocol

            options: dict[str, Any] = {
                "Threads": self.threads,
                "Hash": self.hash_size_mb,
            }
            if "Skill Level" in protocol.options:
                options["Skill Level"] = self.skill_level

            await protocol.configure(options)
            self._is_fallback_mode = False
            self._initialized = True
        except Exception as e:
            if self.fallback_to_minimax:
                logger.warning("Async engine spawn failed (%s); switching to fallback mode.", e)
                self._is_fallback_mode = True
                self._initialized = True
            else:
                raise EngineProcessError(f"Failed to spawn async Stockfish engine: {e}") from e

    async def evaluate_async(
        self,
        fen: str,
        depth: int | None = None,
        time_limit: float | None = None,
    ) -> EngineEvaluation:
        """
        Asynchronously evaluates a FEN string position.
        """
        if not self._initialized:
            await self.initialize()

        board = validate_fen_and_get_board(fen)
        search_depth = depth if depth is not None else self.default_depth
        search_time = time_limit if time_limit is not None else self.default_time_limit

        # Terminal state
        if board.is_game_over():
            if board.is_checkmate():
                return EngineEvaluation(
                    best_move_uci="",
                    best_move_san="",
                    score_cp=None,
                    score_mate=0,
                    depth=0,
                    ponder_move_uci=None,
                    pv=[],
                )
            return EngineEvaluation(
                best_move_uci="",
                best_move_san="",
                score_cp=0,
                score_mate=None,
                depth=0,
                ponder_move_uci=None,
                pv=[],
            )

        if self._is_fallback_mode or self._protocol is None:
            return self._fallback_engine.evaluate(board, depth=min(search_depth, 3))

        try:
            limit = chess.engine.Limit(depth=search_depth, time=search_time)
            info = await self._protocol.analyse(board, limit)
            return self._parse_analysis_info(board, info, search_depth)
        except Exception as e:
            if self.fallback_to_minimax:
                return self._fallback_engine.evaluate(board, depth=min(search_depth, 3))
            raise EngineProcessError(f"Async UCI evaluation error: {e}") from e

    def _parse_analysis_info(
        self,
        board: chess.Board,
        info: chess.engine.InfoDict,
        requested_depth: int,
    ) -> EngineEvaluation:
        pv_moves: list[chess.Move] = info.get("pv", [])
        if not pv_moves:
            first_legal = next(iter(board.legal_moves), None)
            if first_legal:
                pv_moves = [first_legal]

        best_move_uci = pv_moves[0].uci() if pv_moves else ""
        try:
            best_move_san = board.san(pv_moves[0]) if pv_moves else ""
        except Exception:
            best_move_san = best_move_uci

        ponder_move_uci = pv_moves[1].uci() if len(pv_moves) > 1 else None
        pv_uci = [m.uci() for m in pv_moves]

        pov_score: chess.engine.PovScore | None = info.get("score")
        score_cp: int | None = None
        score_mate: int | None = None

        if pov_score is not None:
            rel_score = pov_score.relative
            if rel_score.is_mate():
                score_mate = rel_score.mate()
            else:
                score_cp = rel_score.score()

        reached_depth = info.get("depth", requested_depth)

        return EngineEvaluation(
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            score_cp=score_cp,
            score_mate=score_mate,
            depth=reached_depth,
            ponder_move_uci=ponder_move_uci,
            pv=pv_uci,
        )

    async def close_async(self) -> None:
        """Asynchronously terminates the UCI engine process."""
        if self._protocol is not None:
            try:
                await self._protocol.quit()
            except Exception:
                pass
            self._protocol = None
            self._transport = None
            self._initialized = False

    async def __aenter__(self) -> AsyncStockfishManager:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()
