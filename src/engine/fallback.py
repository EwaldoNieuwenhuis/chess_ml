"""
Deterministic Python-Chess Minimax Fallback Evaluator.

Provides a standalone, pure-Python evaluation engine using alpha-beta minimax search
with piece-square tables and material heuristics. Used for CI environments, unit tests,
and headless platforms without a native Stockfish binary.
"""

from __future__ import annotations

import chess
from src.engine.exceptions import InvalidFENError
from src.schemas.contracts import EngineEvaluation

# Piece base material values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Simplified Piece-Square Tables (PST) for positional evaluation (from White's perspective)
# fmt: off
PAWN_TABLE = [
    0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    10,  10,  20,  30,  30,  20,  10,  10,
    5,   5,  10,  25,  25,  10,   5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    5,  10,  10, -20, -20,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    0,   0,   0,   5,   5,   0,   0,   0,
]

QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_MIDDLE_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,   0,   0,   0,   0,  20,  20,
    20,  30,  10,   0,   0,  10,  30,  20,
]
# fmt: on

PST_MAP = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MIDDLE_TABLE,
}


def evaluate_board_static(board: chess.Board) -> int:
    """
    Static heuristic evaluation of the board position from White's perspective (in centipawns).
    """
    if board.is_checkmate():
        return -20000 if board.turn == chess.WHITE else 20000
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_fivefold_repetition()
    ):
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        base_val = PIECE_VALUES[piece.piece_type]
        pst = PST_MAP[piece.piece_type]
        # For White, standard rank mapping; for Black, mirror square vertically
        pst_idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        pos_val = pst[pst_idx]

        piece_total = base_val + pos_val
        if piece.color == chess.WHITE:
            score += piece_total
        else:
            score -= piece_total

    return score


class PythonChessFallbackEngine:
    """
    Pure Python minimax evaluator with alpha-beta pruning and move ordering.
    """

    def __init__(self, default_depth: int = 2) -> None:
        self.default_depth = default_depth

    def _minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        is_maximizing: bool,
    ) -> tuple[int, list[chess.Move]]:
        if depth == 0 or board.is_game_over():
            return evaluate_board_static(board), []

        best_pv: list[chess.Move] = []

        # Order moves: captures first
        legal_moves = sorted(
            board.legal_moves,
            key=lambda m: (board.is_capture(m), PIECE_VALUES.get(board.piece_type_at(m.to_square) or 0, 0)),
            reverse=True,
        )

        if is_maximizing:
            max_eval = -999999
            for move in legal_moves:
                board.push(move)
                eval_score, sub_pv = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_pv = [move] + sub_pv

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_pv
        else:
            min_eval = 999999
            for move in legal_moves:
                board.push(move)
                eval_score, sub_pv = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_pv = [move] + sub_pv

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_pv

    def evaluate(
        self,
        board_or_fen: str | chess.Board,
        depth: int | None = None,
    ) -> EngineEvaluation:
        """
        Evaluates the chess position and returns an EngineEvaluation contract.

        Args:
            board_or_fen: Either a FEN string or a chess.Board instance.
            depth: Search depth (defaults to self.default_depth).

        Returns:
            EngineEvaluation structured object.

        Raises:
            InvalidFENError: If FEN is invalid or position is illegal.
        """
        search_depth = depth if depth is not None else self.default_depth

        if isinstance(board_or_fen, str):
            try:
                board = chess.Board(board_or_fen)
            except ValueError as e:
                raise InvalidFENError(f"Invalid FEN string: '{board_or_fen}' ({e})") from e
        else:
            board = board_or_fen.copy()

        if not board.is_valid():
            raise InvalidFENError(f"Illegal chess board position for FEN: '{board.fen()}'")

        # Terminal state check
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
            # Stalemate / draw
            return EngineEvaluation(
                best_move_uci="",
                best_move_san="",
                score_cp=0,
                score_mate=None,
                depth=0,
                ponder_move_uci=None,
                pv=[],
            )

        # Minimax evaluation
        is_white = board.turn == chess.WHITE
        score_val, pv = self._minimax(
            board,
            depth=search_depth,
            alpha=-999999,
            beta=999999,
            is_maximizing=is_white,
        )

        # Convert score to current player's POV
        pov_score = score_val if is_white else -score_val

        # Best move
        if pv:
            best_move = pv[0]
            best_move_uci = best_move.uci()
            try:
                best_move_san = board.san(best_move)
            except Exception:
                best_move_san = best_move_uci
            ponder_move_uci = pv[1].uci() if len(pv) > 1 else None
            pv_uci = [m.uci() for m in pv]
        else:
            # Fallback to first legal move
            first_legal = next(iter(board.legal_moves))
            best_move_uci = first_legal.uci()
            best_move_san = board.san(first_legal)
            ponder_move_uci = None
            pv_uci = [best_move_uci]

        return EngineEvaluation(
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            score_cp=pov_score,
            score_mate=None,
            depth=search_depth,
            ponder_move_uci=ponder_move_uci,
            pv=pv_uci,
        )
