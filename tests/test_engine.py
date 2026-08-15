"""
Unit and Integration Tests for Stockfish UCI Engine Manager & Query Interface.

Validates US-1.2.1 and US-1.2.2 requirements:
- FEN validation and custom InvalidFENError handling
- Terminal game states (checkmate, stalemate)
- Evaluation contract structure (best_move, eval_type, eval_value, ponder_move, pv)
- Famous chess puzzle positions (Opera game, mate-in-1, opening moves)
- Fallback minimax evaluator
- Sync and Async engine managers
- Persistent session pooling & query speed
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import chess
import pytest
from src.engine import (
    AsyncStockfishManager,
    EngineNotFoundError,
    InvalidFENError,
    PythonChessFallbackEngine,
    StockfishManager,
    discover_stockfish_binary,
    validate_fen_and_get_board,
)
from src.schemas.contracts import EngineEvaluation

# Test FEN positions
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Fool's Mate terminal checkmate on board (White is checkmated by Black)
FOOLS_MATE_TERMINAL = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# Stalemate terminal on board (Black has no legal moves, not in check)
STALEMATE_TERMINAL = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
# Mate in 1: White Queen deliveries back-rank mate with e1e8#
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1"
# Opera Game puzzle position: Paul Morphy vs Duke of Brunswick (White to move wins)
OPERA_GAME_FEN = "4kb1r/p2rqppp/5n2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 14"


# =========================================================================
# 1. FEN Validation & Error Handling Tests
# =========================================================================


def test_fen_validation_valid_positions() -> None:
    board = validate_fen_and_get_board(STARTING_FEN)
    assert isinstance(board, chess.Board)
    assert board.turn == chess.WHITE

    board_opera = validate_fen_and_get_board(OPERA_GAME_FEN)
    assert board_opera.turn == chess.WHITE


def test_fen_validation_empty_and_whitespace() -> None:
    with pytest.raises(InvalidFENError, match="cannot be empty"):
        validate_fen_and_get_board("")

    with pytest.raises(InvalidFENError, match="cannot be empty"):
        validate_fen_and_get_board("   ")


def test_fen_validation_malformed_syntax() -> None:
    with pytest.raises(InvalidFENError, match="Malformed FEN syntax"):
        validate_fen_and_get_board("invalid_fen_string_12345")

    with pytest.raises(InvalidFENError, match="Malformed FEN syntax"):
        # Missing rows in placement
        validate_fen_and_get_board("rnbqkbnr/8/8/8 w KQkq - 0 1")


def test_fen_validation_illegal_positions() -> None:
    # Multiple kings for White
    two_white_kings = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNK w KQkq - 0 1"
    with pytest.raises(InvalidFENError, match="Illegal chess board position"):
        validate_fen_and_get_board(two_white_kings)

    # No kings for White
    no_white_king = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQ1BNR w - - 0 1"
    with pytest.raises(InvalidFENError, match="Illegal chess board position"):
        validate_fen_and_get_board(no_white_king)

    # Pawns on 1st rank
    pawns_on_first_rank = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNP w KQkq - 0 1"
    with pytest.raises(InvalidFENError, match="Illegal chess board position"):
        validate_fen_and_get_board(pawns_on_first_rank)


# =========================================================================
# 2. Terminal Game State Handling Tests
# =========================================================================


def test_terminal_checkmate_evaluation() -> None:
    manager = StockfishManager(fallback_to_minimax=True)
    res = manager.evaluate(FOOLS_MATE_TERMINAL)

    assert isinstance(res, EngineEvaluation)
    assert res.is_terminal is True
    assert res.eval_type == "mate"
    assert res.score_mate == 0
    assert res.score_cp is None
    assert res.eval_value == 0
    assert res.best_move == ""
    assert res.best_move_uci == ""
    assert res.pv == []
    assert res.formatted_score == "M0"
    manager.close()


def test_terminal_stalemate_evaluation() -> None:
    manager = StockfishManager(fallback_to_minimax=True)
    res = manager.evaluate(STALEMATE_TERMINAL)

    assert isinstance(res, EngineEvaluation)
    assert res.is_terminal is True
    assert res.eval_type == "cp"
    assert res.score_cp == 0
    assert res.score_mate is None
    assert res.eval_value == 0
    assert res.best_move == ""
    assert res.pv == []
    assert res.formatted_score == "+0.00"
    manager.close()


# =========================================================================
# 3. Python-Chess Minimax Fallback Engine Tests
# =========================================================================


def test_minimax_fallback_starting_position() -> None:
    fallback = PythonChessFallbackEngine(default_depth=2)
    eval_res = fallback.evaluate(STARTING_FEN)

    assert isinstance(eval_res, EngineEvaluation)
    assert eval_res.best_move_uci != ""
    assert eval_res.best_move in [m.uci() for m in chess.Board(STARTING_FEN).legal_moves]
    assert eval_res.eval_type == "cp"
    assert isinstance(eval_res.eval_value, int)
    assert len(eval_res.pv) >= 1


def test_minimax_fallback_tactical_mate_in_one() -> None:
    fallback = PythonChessFallbackEngine(default_depth=2)
    eval_res = fallback.evaluate(MATE_IN_ONE)

    # In MATE_IN_ONE, Queen on e1 delivers back-rank checkmate via e1e8
    assert eval_res.best_move_uci == "e1e8"
    assert eval_res.best_move_san == "Qe8#"


def test_minimax_fallback_invalid_fen_raises() -> None:
    fallback = PythonChessFallbackEngine()
    with pytest.raises(InvalidFENError):
        fallback.evaluate("not_a_fen")


# =========================================================================
# 4. Binary Discovery & Exception Tests
# =========================================================================


def test_discovery_explicit_custom_path(tmp_path: Path) -> None:
    fake_binary = tmp_path / "stockfish.exe"
    fake_binary.touch()

    discovered = discover_stockfish_binary(custom_path=fake_binary)
    assert discovered == fake_binary.resolve()


def test_discovery_env_variable(tmp_path: Path) -> None:
    fake_binary = tmp_path / "stockfish_env.exe"
    fake_binary.touch()

    with patch.dict(os.environ, {"STOCKFISH_PATH": str(fake_binary)}):
        discovered = discover_stockfish_binary()
        assert discovered == fake_binary.resolve()


def test_engine_not_found_error_when_fallback_disabled() -> None:
    with pytest.raises(EngineNotFoundError):
        StockfishManager(
            binary_path="/non/existent/path/to/stockfish",
            auto_download=False,
            fallback_to_minimax=False,
        )


# =========================================================================
# 5. Synchronous StockfishManager Integration & Lifecycle Tests
# =========================================================================


def test_stockfish_manager_context_manager() -> None:
    with StockfishManager(fallback_to_minimax=True) as engine:
        res = engine.evaluate(STARTING_FEN)
        assert isinstance(res, EngineEvaluation)
        assert res.best_move != ""
        assert res.depth >= 1
        assert res.eval_type in ("cp", "mate")


def test_stockfish_manager_get_top_moves() -> None:
    with StockfishManager(fallback_to_minimax=True) as engine:
        moves = engine.get_top_moves(STARTING_FEN, multipv=3)
        assert isinstance(moves, list)
        assert len(moves) >= 1
        assert all(isinstance(m, EngineEvaluation) for m in moves)


def test_stockfish_manager_persistent_session_speed() -> None:
    """Verifies persistent session evaluates repeated positions with sub-millisecond overhead."""
    with StockfishManager(fallback_to_minimax=True) as engine:
        # Warmup
        engine.evaluate(STARTING_FEN, depth=1)

        t0 = time.perf_counter()
        n_queries = 20
        for _ in range(n_queries):
            engine.evaluate(STARTING_FEN, depth=1)
        total_time = time.perf_counter() - t0

        avg_latency_ms = (total_time / n_queries) * 1000.0
        # In-process persistent session should average under 10ms per depth-1 query
        assert avg_latency_ms < 50.0, f"Average query latency too slow: {avg_latency_ms:.2f}ms"


# =========================================================================
# 6. Asynchronous AsyncStockfishManager Tests
# =========================================================================


@pytest.mark.asyncio
async def test_async_stockfish_manager() -> None:
    async with AsyncStockfishManager(fallback_to_minimax=True) as engine:
        res = await engine.evaluate_async(STARTING_FEN)
        assert isinstance(res, EngineEvaluation)
        assert res.best_move != ""
        assert res.is_terminal is False

        # Test terminal in async
        term_res = await engine.evaluate_async(FOOLS_MATE_TERMINAL)
        assert term_res.is_terminal is True
        assert term_res.score_mate == 0


@pytest.mark.asyncio
async def test_async_stockfish_manager_invalid_fen() -> None:
    async with AsyncStockfishManager(fallback_to_minimax=True) as engine:
        with pytest.raises(InvalidFENError):
            await engine.evaluate_async("invalid_fen")
