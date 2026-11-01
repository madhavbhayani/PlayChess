#!/usr/bin/env python3
"""
Dataset V2.1 / v0.4-ready generator — DEFICIT-BALANCED ELO & ENDGAME-ENRICHED BUILD

Fixes & Upgrades over previous build:
  1. Deficit-Aware Water-Filling (--balance-mode deficit): Automatically inspects
     the existing dataset's Elo distribution and allocates the 500 new games to
     under-represented Elo buckets (e.g., 1200..1800, especially 1700) while
     skipping already-saturated buckets (1900, 2000).
  2. Proper Bucket-Boundary Filtering & --max-elo-diff: Allows full [bucket - step/2,
     bucket + step/2] coverage at boundaries while rejecting lopsided rating
     mismatches so both players genuinely belong to the target Elo tier.
  3. Quality & Endgame Filtering (--min-plies 24, --allow-time-forfeit off by default):
     Excludes random bullet flagging / time-forfeits and ultra-short games, and
     rewards games that reach real checkmates or endgame conversions.
  4. Fixed UnboundLocalError on `game_map` during partial repair.
  5. Optimized `_stockfish_novelty_probe` to avoid running a redundant 2nd search
     when the played move is already in the top-2 MultiPV results.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import heapq
import io
import json
import math
import os
import pickle
import shutil
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn
import zstandard as zstd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

try:
    import psutil
except ImportError:
    psutil = None


SCHEMA_VERSION = "2.1"
FEATURE_SCHEMA_VERSION = "v0.4-ready"
MAX_HISTORY_PLIES = 50
CACHE_VERSION = 4

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PROMOTION_NAMES = {
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}

CENTER_SQUARES = {
    chess.D4, chess.E4, chess.D5, chess.E5,
}

_WORKER_ENGINE = None
_WORKER_CONFIG = None


def find_stockfish(explicit: Optional[str]) -> str:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Stockfish not found: {p}")
        return str(p)

    common = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        shutil.which("stockfish"),
    ]
    for p in common:
        if p and Path(p).exists():
            return str(p)

    raise FileNotFoundError(
        "Stockfish was not found. Use --stockfish /path/to/stockfish."
    )


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _progress_line(done: int, total: int, start: float, label: str) -> str:
    elapsed = max(0.001, time.perf_counter() - start)
    rate = done / elapsed
    pct = (100.0 * done / total) if total else 0.0
    eta = (total - done) / rate if rate > 0 and total else 0.0
    return (
        f"{label}: {done:,}/{total:,} ({pct:5.1f}%) | "
        f"rate={rate:,.2f}/s | elapsed={_fmt_seconds(elapsed)} | ETA={_fmt_seconds(eta)}"
    )


def material_difference(board: chess.Board) -> int:
    white = 0
    black = 0
    for piece_type, value in PIECE_VALUES.items():
        white += len(board.pieces(piece_type, chess.WHITE)) * value
        black += len(board.pieces(piece_type, chess.BLACK)) * value
    return white - black


def count_pawn_features(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    pawns = list(board.pieces(chess.PAWN, color))
    files = Counter(chess.square_file(sq) for sq in pawns)
    doubled = sum(max(0, count - 1) for count in files.values())

    isolated = 0
    passed_squares = []

    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        adjacent_files = {af for af in (f - 1, f + 1) if 0 <= af < 8}
        has_neighbor = any(files.get(af, 0) > 0 for af in adjacent_files)
        if not has_neighbor:
            isolated += 1

        enemy = not color
        passed = True

        for enemy_sq in board.pieces(chess.PAWN, enemy):
            ef = chess.square_file(enemy_sq)
            er = chess.square_rank(enemy_sq)
            if abs(ef - f) <= 1:
                if color == chess.WHITE and er > r:
                    passed = False
                    break
                if color == chess.BLACK and er < r:
                    passed = False
                    break

        if passed:
            passed_squares.append(sq)

    passed_files = {chess.square_file(sq) for sq in passed_squares}
    connected_passed = sum(
        1
        for sq in passed_squares
        if (
            chess.square_file(sq) - 1 in passed_files
            or chess.square_file(sq) + 1 in passed_files
        )
    )

    return {
        "isolated_pawns": isolated,
        "doubled_pawns": doubled,
        "passed_pawns": len(passed_squares),
        "connected_passed_pawns": connected_passed,
    }


def piece_activity_summary(
    piece_move_counts: Dict[str, Dict[str, int]],
    board: chess.Board,
    move_counts_by_piece: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        counts = move_counts_by_piece[color_name]
        minors_undeveloped = 0

        start_squares = {
            chess.B1 if color == chess.WHITE else chess.B8: chess.KNIGHT,
            chess.G1 if color == chess.WHITE else chess.G8: chess.KNIGHT,
            chess.C1 if color == chess.WHITE else chess.C8: chess.BISHOP,
            chess.F1 if color == chess.WHITE else chess.F8: chess.BISHOP,
        }

        for sq, piece_type in start_squares.items():
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type == piece_type:
                if counts.get(PIECE_NAMES[piece_type], 0) == 0:
                    minors_undeveloped += 1

        king_moves = counts.get("king", 0)
        queen_moves = counts.get("queen", 0)
        rook_moves = counts.get("rook", 0)
        minor_moves = counts.get("knight", 0) + counts.get("bishop", 0)

        result[color_name] = {
            "undeveloped_minors": minors_undeveloped,
            "king_moved": king_moves > 0,
            "queen_moves": queen_moves,
            "rook_moves": rook_moves,
            "minor_piece_moves": minor_moves,
        }

    return result


def board_key(board: chess.Board):
    return board._transposition_key()


def _safe_king_square(board: chess.Board, color: chess.Color) -> Optional[int]:
    try:
        kings = board.pieces(chess.KING, color)
        for square in kings:
            return int(square)
    except Exception:
        return None
    return None


def move_features(board: chess.Board, move: chess.Move) -> Dict[str, Any]:
    moved_piece = board.piece_at(move.from_square)

    captured_piece = None
    if board.is_en_passant(move):
        captured_square = (
            move.to_square - 8
            if board.turn == chess.WHITE
            else move.to_square + 8
        )
        captured_piece = board.piece_at(captured_square)
    else:
        captured_piece = board.piece_at(move.to_square)

    is_capture = bool(board.is_capture(move))
    is_castling = bool(board.is_castling(move))
    is_en_passant = bool(board.is_en_passant(move))
    is_promotion = bool(move.promotion is not None)

    board.push(move)
    try:
        is_check = bool(board.is_check())
    finally:
        board.pop()

    move_distance = max(
        abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)),
        abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)),
    )
    center_activity = bool(
        move.from_square in CENTER_SQUARES or move.to_square in CENTER_SQUARES
    )

    return {
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "from_square_index": int(move.from_square),
        "to_square_index": int(move.to_square),
        "is_capture": is_capture,
        "is_check": is_check,
        "is_castling": is_castling,
        "is_castle": is_castling,
        "is_en_passant": is_en_passant,
        "is_promotion": is_promotion,
        "captured_piece": captured_piece.symbol() if captured_piece else None,
        "captured_piece_type": int(captured_piece.piece_type if captured_piece else 0),
        "captured_piece_value": int(PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0),
        "moved_piece": moved_piece.symbol() if moved_piece else None,
        "moved_piece_type": int(moved_piece.piece_type if moved_piece else 0),
        "moved_piece_value": int(PIECE_VALUES.get(moved_piece.piece_type, 0) if moved_piece else 0),
        "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
        "promotion_piece": int(move.promotion or 0),
        "promotion_piece_name": PROMOTION_NAMES.get(move.promotion) if move.promotion else None,
        "promotion_piece_value": int(PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0),
        "move_distance": int(move_distance),
        "center_activity": center_activity,
    }


def history_move_features(
    before: chess.Board,
    move: chess.Move,
    after: chess.Board,
    strategic_event_count: int,
) -> Dict[str, Any]:
    mf = move_features(before, move)

    side_id = 0 if before.turn == chess.WHITE else 1
    opponent_king = _safe_king_square(after, not before.turn)
    destination_to_opponent_king_distance = 8
    if opponent_king is not None:
        destination_to_opponent_king_distance = max(
            abs(chess.square_file(move.to_square) - chess.square_file(opponent_king)),
            abs(chess.square_rank(move.to_square) - chess.square_rank(opponent_king)),
        )

    return {
        **mf,
        "side_id": int(side_id),
        "strategic_event_count": int(strategic_event_count),
        "destination_to_opponent_king_distance": int(destination_to_opponent_king_distance),
        "is_check_after": bool(after.is_check()),
    }


def _count_hanging_non_king(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for sq in (
        board.pieces(chess.PAWN, color)
        | board.pieces(chess.KNIGHT, color)
        | board.pieces(chess.BISHOP, color)
        | board.pieces(chess.ROOK, color)
        | board.pieces(chess.QUEEN, color)
    ):
        if board.attackers(not color, sq) and not board.attackers(color, sq):
            count += 1
    return count


def _count_pinned_non_king(board: chess.Board, color: chess.Color) -> int:
    return sum(
        1
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for sq in board.pieces(piece_type, color)
        if board.is_pinned(color, sq)
    )


def _king_ring_attack_pressure(board: chess.Board, king_square: Optional[int], attacker_color: chess.Color) -> int:
    if king_square is None:
        return 0
    kf = chess.square_file(king_square)
    kr = chess.square_rank(king_square)
    pressure = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f = kf + df
            r = kr + dr
            if 0 <= f < 8 and 0 <= r < 8:
                sq = chess.square(f, r)
                pressure += len(board.attackers(attacker_color, sq))
    return pressure


def _center_control(board: chess.Board, color: chess.Color) -> int:
    return sum(len(board.attackers(color, sq)) for sq in CENTER_SQUARES)


def build_v04_position_summary(board: chess.Board) -> Dict[str, Any]:
    current = board.turn
    opponent = not current
    c_center = _center_control(board, current)
    o_center = _center_control(board, opponent)
    return {
        "current_side_hanging_non_king": _count_hanging_non_king(board, current),
        "opponent_side_hanging_non_king": _count_hanging_non_king(board, opponent),
        "current_side_pinned_non_king": _count_pinned_non_king(board, current),
        "opponent_side_pinned_non_king": _count_pinned_non_king(board, opponent),
        "current_king_ring_attack_pressure": _king_ring_attack_pressure(board, _safe_king_square(board, current), opponent),
        "opponent_king_ring_attack_pressure": _king_ring_attack_pressure(board, _safe_king_square(board, opponent), current),
        "current_castling_right_count": int(
            board.has_kingside_castling_rights(current)
            + board.has_queenside_castling_rights(current)
        ),
        "opponent_castling_right_count": int(
            board.has_kingside_castling_rights(opponent)
            + board.has_queenside_castling_rights(opponent)
        ),
        "current_center_control": c_center,
        "opponent_center_control": o_center,
        "center_control_delta": c_center - o_center,
        "current_legal_move_count": sum(1 for _ in board.legal_moves),
        "in_check": bool(board.is_check()),
    }


def add_v04_memory_signals(
    long_term_memory: Dict[str, Any],
    material_history: Sequence[int],
    repetition_counts: Dict[Any, int],
    check_counts: Dict[str, int],
    capture_counts: Dict[str, int],
    castle_counts: Dict[str, int],
    promotion_counts: Dict[str, int],
    board: chess.Board,
) -> None:
    current_key = board_key(board)
    current_rep = repetition_counts.get(current_key, 1)
    current_side = "white" if board.turn == chess.WHITE else "black"
    opponent_side = "black" if board.turn == chess.WHITE else "white"
    current_material = material_difference(board)

    if material_history:
        initial = material_history[0]
        reference_index = max(0, len(material_history) - 10)
        last_10_reference = material_history[reference_index]
    else:
        initial = current_material
        last_10_reference = current_material

    long_term_memory["v04_explicit_signals"] = {
        "material_advantage_side_to_move_cp": int(
            current_material if board.turn == chess.WHITE else -current_material
        ),
        "last_10_ply_material_change_cp": int(current_material - last_10_reference),
        "current_position_repetition_count": int(current_rep),
        "threefold_occurred": bool(current_rep >= 3),
        "check_count_difference": int(check_counts[current_side] - check_counts[opponent_side]),
        "capture_count_difference": int(capture_counts[current_side] - capture_counts[opponent_side]),
        "castle_count_difference": int(castle_counts[current_side] - castle_counts[opponent_side]),
        "promotion_count_difference": int(promotion_counts[current_side] - promotion_counts[opponent_side]),
        "initial_material_difference_white_cp": int(initial),
    }


def move_event_types(
    before: chess.Board,
    move: chess.Move,
    after: chess.Board,
    moved_piece_type: int,
    move_was_first_for_type: bool,
    before_pawn_files: set[int],
    after_pawn_files: set[int],
    before_passed: Dict[chess.Color, int],
    after_passed: Dict[chess.Color, int],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    side = "white" if before.turn == chess.WHITE else "black"

    if before.is_castling(move):
        events.append({"type": "CASTLED", "side": side})

    if before.is_capture(move):
        events.append({"type": "CAPTURE", "side": side})

    if after.is_check():
        events.append({"type": "CHECK", "side": side})

    if move.promotion:
        events.append({"type": "PROMOTION", "side": side})

    if before.is_en_passant(move):
        events.append({"type": "EN_PASSANT", "side": side})

    if moved_piece_type == chess.PAWN:
        if before_pawn_files != after_pawn_files:
            events.append({"type": "PAWN_STRUCTURE_CHANGE", "side": side})
        elif before.fen().split()[0] != after.fen().split()[0]:
            events.append({"type": "PAWN_STRUCTURE_CHANGE", "side": side})

    if move_was_first_for_type and moved_piece_type in (chess.KNIGHT, chess.BISHOP):
        events.append({"type": "PIECE_DEVELOPMENT", "side": side})

    before_open = {f for f in range(8) if f not in before_pawn_files}
    after_open = {f for f in range(8) if f not in after_pawn_files}
    if after_open - before_open:
        for f in sorted(after_open - before_open):
            events.append({
                "type": "OPEN_FILE",
                "side": side,
                "file": chess.FILE_NAMES[f],
            })

    for color in (chess.WHITE, chess.BLACK):
        if after_passed[color] > before_passed[color]:
            events.append({
                "type": "PASSED_PAWN_CREATED",
                "side": "white" if color == chess.WHITE else "black",
            })

    if move.to_square in CENTER_SQUARES or move.from_square in CENTER_SQUARES:
        events.append({"type": "CENTER_ACTIVITY", "side": side})

    return events


def wdl_dict(pov_score: chess.engine.PovScore, ply: int) -> Optional[Dict[str, int]]:
    try:
        wdl = pov_score.wdl(model="sf", ply=ply, normalize=True)
        return {
            "win": int(wdl.wins),
            "draw": int(wdl.draws),
            "loss": int(wdl.losses),
        }
    except Exception:
        try:
            wdl = pov_score.wdl()
            return {
                "win": int(wdl.wins),
                "draw": int(wdl.draws),
                "loss": int(wdl.losses),
            }
        except Exception:
            return None


def score_to_label(score: chess.engine.PovScore) -> Tuple[Optional[int], Optional[int], int]:
    pov = score
    mate = pov.mate()
    cp = pov.score()

    if mate is not None:
        if mate > 0:
            ordering = 100000 - abs(mate) * 100
        else:
            ordering = -100000 + abs(mate) * 100
        return None, int(mate), int(ordering)

    cp_value = int(cp) if cp is not None else 0
    return cp_value, None, cp_value


def sanitize_header_value(value: Any) -> str:
    return str(value) if value is not None else ""


def game_id_from_headers(headers: chess.pgn.Headers) -> str:
    site = sanitize_header_value(headers.get("Site"))
    if site and site != "?":
        return site
    blob = "|".join([
        sanitize_header_value(headers.get("UTCDate")),
        sanitize_header_value(headers.get("UTCTime")),
        sanitize_header_value(headers.get("White")),
        sanitize_header_value(headers.get("Black")),
        sanitize_header_value(headers.get("Event")),
        sanitize_header_value(headers.get("Result")),
    ])
    return "pgngame:" + hashlib.sha1(blob.encode("utf-8")).hexdigest()


def game_move_list(game: chess.pgn.Game) -> List[chess.Move]:
    return list(game.mainline_moves())


def build_long_term_memory(
    board: chess.Board,
    ply: int,
    material_history: List[int],
    repetition_counts: Dict[Any, int],
    move_counts_by_piece: Dict[str, Dict[str, int]],
    check_counts: Dict[str, int],
    capture_counts: Dict[str, int],
    castle_counts: Dict[str, int],
    promotion_counts: Dict[str, int],
    strategic_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_material = material_difference(board)
    initial_material = material_history[0] if material_history else 0
    largest_swing = max(
        (abs(x - initial_material) for x in material_history),
        default=0,
    )

    last_10 = material_history[-10:] if material_history else [current_material]
    last_10_swing = current_material - last_10[0]

    current_key = board_key(board)
    current_rep = repetition_counts.get(current_key, 1)

    pawn_white = count_pawn_features(board, chess.WHITE)
    pawn_black = count_pawn_features(board, chess.BLACK)

    activity = piece_activity_summary(
        move_counts_by_piece,
        board,
        move_counts_by_piece,
    )

    total_repetitions = sum(1 for n in repetition_counts.values() if n >= 3)

    past_plies = max(0, ply - 1)
    position_count_in_history = len(material_history)

    return {
        "material_trajectory": {
            "initial_difference_white_cp": initial_material,
            "current_difference_white_cp": current_material,
            "current_advantage_side_to_move_cp": (
                current_material if board.turn == chess.WHITE else -current_material
            ),
            "largest_swing_from_initial_cp": largest_swing,
            "last_10_ply_change_cp": last_10_swing,
            "past_plies": past_plies,
            "position_count_in_history": position_count_in_history,
        },
        "pawn_structure": {
            "white": pawn_white,
            "black": pawn_black,
        },
        "piece_activity": activity,
        "repetition": {
            "current_position_count": current_rep,
            "threefold_occurred": current_rep >= 3,
            "repetition_events": total_repetitions,
        },
        "historical_patterns": {
            "checks": {
                "white": check_counts["white"],
                "black": check_counts["black"],
            },
            "captures": {
                "white": capture_counts["white"],
                "black": capture_counts["black"],
            },
            "castles": {
                "white": castle_counts["white"],
                "black": castle_counts["black"],
            },
            "promotions": {
                "white": promotion_counts["white"],
                "black": promotion_counts["black"],
            },
        },
        "strategic_events": strategic_events[-1000:],
    }


def candidate_engine_labels(
    board: chess.Board,
    analysis: Sequence[Dict[str, Any]],
    legal_moves: Sequence[chess.Move],
    depth: int,
    nodes: Optional[int],
) -> List[Dict[str, Any]]:
    by_uci: Dict[str, Dict[str, Any]] = {}

    for info in analysis:
        pv = info.get("pv") or []
        if not pv:
            continue

        move = pv[0]
        if move not in legal_moves:
            continue

        uci = move.uci()
        pov_score = info.get("score")
        if pov_score is None:
            continue

        pov_score = pov_score.pov(board.turn)
        score_cp, mate, ordering_score = score_to_label(pov_score)

        by_uci[uci] = {
            "uci": uci,
            "score_cp": score_cp,
            "mate": mate,
            "ordering_score": ordering_score,
            "wdl": wdl_dict(pov_score, board.ply()),
            "depth": safe_int(info.get("depth"), depth),
            "nodes": safe_int(info.get("nodes"), nodes or 0),
            "time": info.get("time"),
            "principal_variation": [
                m.uci() for m in (pv[:10] if pv else [])
            ],
        }

    missing = [m for m in legal_moves if m.uci() not in by_uci]
    if missing:
        raise RuntimeError(
            f"Stockfish returned {len(by_uci)}/{len(legal_moves)} legal moves "
            f"for position {board.fen()}; missing={len(missing)}"
        )

    return [by_uci[m.uci()] for m in legal_moves]


def make_position_record(
    game: chess.pgn.Game,
    board: chess.Board,
    move_history: List[Dict[str, Any]],
    long_term_memory: Dict[str, Any],
    legal_moves: List[chess.Move],
    engine_labels: List[Dict[str, Any]],
    played_move: Optional[chess.Move],
    ply: int,
    selected_depth: int,
    stockfish_nodes: Optional[int],
) -> Dict[str, Any]:
    controlled_side = "white" if board.turn == chess.WHITE else "black"

    candidates: List[Dict[str, Any]] = []
    label_by_uci = {x["uci"]: x for x in engine_labels}

    for move in legal_moves:
        uci = move.uci()
        label = label_by_uci[uci]
        mf = move_features(board, move)

        candidates.append({
            "uci": uci,
            "san": board.san(move),
            "move_features": mf,
            "training_label": {
                "score_cp": label["score_cp"],
                "mate": label["mate"],
                "ordering_score": label["ordering_score"],
                "wdl": label["wdl"],
                "depth": label["depth"],
                "nodes": label["nodes"],
                "time": label["time"],
                "principal_variation": label["principal_variation"],
            },
        })

    best_ordering_score = max(
        c["training_label"]["ordering_score"] for c in candidates
    )

    optimal_moves = [
        c["uci"]
        for c in candidates
        if c["training_label"]["ordering_score"] == best_ordering_score
    ]

    best_move = sorted(optimal_moves)[0]
    best_candidate = next(c for c in candidates if c["uci"] == best_move)

    for candidate in candidates:
        cand_score = candidate["training_label"]["ordering_score"]
        candidate["training_label"]["relative_to_best"] = cand_score - best_ordering_score
        candidate["training_label"]["centipawn_loss"] = max(
            0, best_ordering_score - cand_score
        )
        candidate["training_label"]["is_optimal"] = (
            cand_score == best_ordering_score
        )

    played_meta = None
    if played_move is not None:
        played_uci = played_move.uci()
        played_candidate = next(c for c in candidates if c["uci"] == played_uci)

        played_meta = {
            "uci": played_uci,
            "san": board.san(played_move),
            "engine_score_cp": played_candidate["training_label"]["score_cp"],
            "engine_mate": played_candidate["training_label"]["mate"],
            "centipawn_loss": played_candidate["training_label"]["centipawn_loss"],
            "is_optimal": played_candidate["training_label"]["is_optimal"],
        }

    position_fen = board.fen()
    fen_parts = position_fen.split()
    serialized_ep = fen_parts[3] if len(fen_parts) >= 4 else "-"
    serialized_ep = None if serialized_ep == "-" else serialized_ep

    headers = game.headers
    result = sanitize_header_value(headers.get("Result"))

    if result == "1-0":
        controlled_outcome = "win" if controlled_side == "white" else "loss"
    elif result == "0-1":
        controlled_outcome = "loss" if controlled_side == "white" else "win"
    elif result == "1/2-1/2":
        controlled_outcome = "draw"
    else:
        controlled_outcome = "unknown"

    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,

        "game": {
            "game_id": game_id_from_headers(headers),
            "event": sanitize_header_value(headers.get("Event")),
            "site": sanitize_header_value(headers.get("Site")),
            "white_player": sanitize_header_value(headers.get("White")),
            "black_player": sanitize_header_value(headers.get("Black")),
            "white_elo": sanitize_header_value(headers.get("WhiteElo")),
            "black_elo": sanitize_header_value(headers.get("BlackElo")),
            "result": result,
            "termination": sanitize_header_value(headers.get("Termination")),
        },

        "position": {
            "ply": ply,
            "fullmove_number": board.fullmove_number,
            "fen": position_fen,
            "side_to_move": controlled_side,
            "controlled_side": controlled_side,
            "board_8x8": [
                [
                    (
                        0
                        if (piece := board.piece_at(chess.square(file, 7 - rank))) is None
                        else (
                            piece.piece_type
                            if piece.color == chess.WHITE
                            else -piece.piece_type
                        )
                    )
                    for file in range(8)
                ]
                for rank in range(8)
            ],
            "position_state": {
                "castling_rights": {
                    "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
                    "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
                    "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
                    "black_queenside": board.has_queenside_castling_rights(chess.BLACK),
                },
                "en_passant_square": serialized_ep,
                "halfmove_clock": board.halfmove_clock,
                "fullmove_number": board.fullmove_number,
                "is_check": board.is_check(),
                "is_checkmate": board.is_checkmate(),
                "is_stalemate": board.is_stalemate(),
                "legal_move_count": len(legal_moves),
                "v04_tactical_summary": build_v04_position_summary(board),
            },
        },

        "recent_history": {
            "max_plies": MAX_HISTORY_PLIES,
            "plies_available": len(move_history),
            "moves": move_history[-MAX_HISTORY_PLIES:],
        },

        "long_term_memory": long_term_memory,

        "candidate_moves": candidates,

        "training_target": {
            "best_move": best_move,
            "best_move_san": best_candidate["san"],
            "optimal_moves": sorted(optimal_moves),
            "best_score_cp": best_candidate["training_label"]["score_cp"],
            "best_mate": best_candidate["training_label"]["mate"],
            "candidate_count": len(candidates),
            "candidate_source": "all_legal_moves",
            "engine_perspective": controlled_side,
            "engine_depth": selected_depth,
            "engine_nodes_limit": stockfish_nodes,
        },

        "metadata": {
            "played_move": played_meta,
            "game_result": result,
            "game_result_for_controlled_side": controlled_outcome,
        },
    }


def prepare_game_for_worker(game: chess.pgn.Game) -> Dict[str, Any]:
    return {
        "pgn": str(game),
        "game_id": game_id_from_headers(game.headers),
    }


def init_worker(
    stockfish_path: str,
    depth: int,
    nodes: Optional[int],
    hash_mb: int,
    threads: int,
):
    global _WORKER_ENGINE, _WORKER_CONFIG

    _WORKER_CONFIG = {
        "depth": depth,
        "nodes": nodes,
        "stockfish_path": stockfish_path,
        "threads": threads,
        "hash_mb": hash_mb,
    }

    _WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    try:
        _WORKER_ENGINE.configure({
            "Threads": threads,
            "Hash": hash_mb,
        })
    except Exception:
        try:
            _WORKER_ENGINE.configure({"Threads": threads})
        except Exception:
            pass

    def cleanup():
        global _WORKER_ENGINE
        if _WORKER_ENGINE is not None:
            try:
                _WORKER_ENGINE.quit()
            except Exception:
                pass
            _WORKER_ENGINE = None

    atexit.register(cleanup)


def reconstruct_game(pgn_text: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Failed to reconstruct game from serialized PGN.")
    return game


def analyze_all_legal_moves_with_retry(
    board: chess.Board,
    legal_moves: Sequence[chess.Move],
    limit_kwargs: Dict[str, Any],
) -> Sequence[Dict[str, Any]]:
    global _WORKER_ENGINE
    last_exc: Optional[BaseException] = None
    for attempt in (1, 2):
        try:
            return _WORKER_ENGINE.analyse(
                board,
                chess.engine.Limit(**limit_kwargs),
                multipv=len(legal_moves),
            )
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                break
            try:
                _WORKER_ENGINE.quit()
            except Exception:
                pass
            _WORKER_ENGINE = None
            _WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(
                _WORKER_CONFIG["stockfish_path"]
            )
            try:
                _WORKER_ENGINE.configure({
                    "Threads": _WORKER_CONFIG["threads"],
                    "Hash": _WORKER_CONFIG["hash_mb"],
                })
            except Exception:
                try:
                    _WORKER_ENGINE.configure({"Threads": _WORKER_CONFIG["threads"]})
                except Exception:
                    pass
    assert last_exc is not None
    raise last_exc


def process_game(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER_ENGINE, _WORKER_CONFIG

    stage = "worker_start"
    current_ply = 0
    current_fen = None
    try:
        if _WORKER_ENGINE is None:
            raise RuntimeError("Stockfish worker engine was not initialized.")

        stage = "reconstruct_game"
        game = reconstruct_game(payload["pgn"])
        moves = game_move_list(game)
        total_plies = len(moves)

        if total_plies == 0:
            return {
                "game_id": payload["game_id"],
                "records": [],
                "total_plies": 0,
                "error": None,
            }

        board = game.board()
        move_history: List[Dict[str, Any]] = []

        material_history: List[int] = [material_difference(board)]
        repetition_counts: Dict[Any, int] = defaultdict(int)
        repetition_counts[board_key(board)] = 1

        move_counts_by_piece = {
            "white": defaultdict(int),
            "black": defaultdict(int),
        }
        check_counts = {"white": 0, "black": 0}
        capture_counts = {"white": 0, "black": 0}
        castle_counts = {"white": 0, "black": 0}
        promotion_counts = {"white": 0, "black": 0}
        strategic_events: List[Dict[str, Any]] = []

        records: List[Dict[str, Any]] = []

        for ply, move in enumerate(moves, start=1):
            current_ply = ply
            current_fen = board.fen()
            stage = "legal_move_generation"
            legal_moves = sorted(board.legal_moves, key=lambda m: m.uci())
            if not legal_moves:
                raise RuntimeError(
                    f"No legal moves at decision position ply={ply} in {payload['game_id']}"
                )

            stage = "stockfish_analyse"
            limit_kwargs: Dict[str, Any] = {}
            if _WORKER_CONFIG["depth"] is not None:
                limit_kwargs["depth"] = _WORKER_CONFIG["depth"]
            if _WORKER_CONFIG["nodes"] is not None:
                limit_kwargs["nodes"] = _WORKER_CONFIG["nodes"]

            analysis = analyze_all_legal_moves_with_retry(
                board,
                legal_moves,
                limit_kwargs,
            )

            stage = "engine_label_extraction"
            labels = candidate_engine_labels(
                board,
                analysis,
                legal_moves,
                _WORKER_CONFIG["depth"] or 0,
                _WORKER_CONFIG["nodes"],
            )

            played_move = move

            stage = "memory_build"
            long_term_memory = build_long_term_memory(
                board=board,
                ply=ply,
                material_history=material_history,
                repetition_counts=repetition_counts,
                move_counts_by_piece=move_counts_by_piece,
                check_counts=check_counts,
                capture_counts=capture_counts,
                castle_counts=castle_counts,
                promotion_counts=promotion_counts,
                strategic_events=strategic_events,
            )

            stage = "v04_memory_signals"
            add_v04_memory_signals(
                long_term_memory=long_term_memory,
                material_history=material_history,
                repetition_counts=repetition_counts,
                check_counts=check_counts,
                capture_counts=capture_counts,
                castle_counts=castle_counts,
                promotion_counts=promotion_counts,
                board=board,
            )

            stage = "make_position_record"
            record = make_position_record(
                game=game,
                board=board.copy(stack=False),
                move_history=move_history.copy(),
                long_term_memory=long_term_memory,
                legal_moves=legal_moves,
                engine_labels=labels,
                played_move=played_move,
                ply=ply,
                selected_depth=_WORKER_CONFIG["depth"] or 0,
                stockfish_nodes=_WORKER_CONFIG["nodes"],
            )
            records.append(record)

            stage = "historical_state_update"
            before = board.copy(stack=False)
            moved_piece = before.piece_at(move.from_square)
            moved_piece_type = moved_piece.piece_type if moved_piece else chess.PAWN
            side_name = "white" if before.turn == chess.WHITE else "black"

            before_pawn_files = {
                chess.square_file(sq)
                for color in (chess.WHITE, chess.BLACK)
                for sq in before.pieces(chess.PAWN, color)
            }
            before_passed = {
                color: count_pawn_features(before, color)["passed_pawns"]
                for color in (chess.WHITE, chess.BLACK)
            }

            piece_name = PIECE_NAMES.get(moved_piece_type, "unknown")
            was_first_piece_move = move_counts_by_piece[side_name][piece_name] == 0

            is_capture = before.is_capture(move)
            is_castle = before.is_castling(move)
            is_promotion = move.promotion is not None

            board.push(move)

            after_pawn_files = {
                chess.square_file(sq)
                for color in (chess.WHITE, chess.BLACK)
                for sq in board.pieces(chess.PAWN, color)
            }
            after_passed = {
                color: count_pawn_features(board, color)["passed_pawns"]
                for color in (chess.WHITE, chess.BLACK)
            }

            events = move_event_types(
                before=before,
                move=move,
                after=board,
                moved_piece_type=moved_piece_type,
                move_was_first_for_type=was_first_piece_move,
                before_pawn_files=before_pawn_files,
                after_pawn_files=after_pawn_files,
                before_passed=before_passed,
                after_passed=after_passed,
            )
            if events:
                for ev in events:
                    ev = dict(ev)
                    ev["ply"] = ply
                    strategic_events.append(ev)

            move_counts_by_piece[side_name][piece_name] += 1

            if board.is_check():
                check_counts[side_name] += 1
            if is_capture:
                capture_counts[side_name] += 1
            if is_castle:
                castle_counts[side_name] += 1
            if is_promotion:
                promotion_counts[side_name] += 1

            stage = "history_feature_build"
            history_features = history_move_features(
                before=before,
                move=move,
                after=board,
                strategic_event_count=len(events),
            )

            move_history.append({
                "ply": ply,
                "uci": move.uci(),
                "san": before.san(move),
                "side": side_name,
                "move_features": history_features,
                "strategic_events": events,
            })

            stage = "history_checkpoint"
            material_history.append(material_difference(board))
            repetition_counts[board_key(board)] += 1

        return {
            "game_id": payload["game_id"],
            "records": records,
            "total_plies": total_plies,
            "error": None,
        }
    except Exception as exc:
        return {
            "game_id": payload.get("game_id"),
            "records": [],
            "total_plies": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "stage": stage,
            "ply": current_ply,
            "fen": current_fen,
            "traceback": traceback.format_exc(),
        }


def board_to_numeric_8x8(board: chess.Board) -> List[List[int]]:
    return [
        [
            (
                0
                if (piece := board.piece_at(chess.square(file, 7 - rank))) is None
                else (
                    piece.piece_type
                    if piece.color == chess.WHITE
                    else -piece.piece_type
                )
            )
            for file in range(8)
        ]
        for rank in range(8)
    ]


def validate_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    try:
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        if record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            errors.append(
                f"feature_schema_version must be {FEATURE_SCHEMA_VERSION}"
            )

        position = record["position"]
        board = chess.Board(position["fen"])

        if position["board_8x8"] != board_to_numeric_8x8(board):
            errors.append("board_8x8 does not match FEN")

        controlled_side = position.get("controlled_side")
        side_to_move = position.get("side_to_move")

        if controlled_side not in ("white", "black"):
            errors.append("invalid controlled_side")
        if controlled_side != side_to_move:
            errors.append("controlled_side must equal side_to_move")
        if controlled_side != ("white" if board.turn == chess.WHITE else "black"):
            errors.append("controlled_side does not match FEN side to move")

        legal = {m.uci() for m in board.legal_moves}
        candidates = record["candidate_moves"]
        candidate_uci = [c["uci"] for c in candidates]

        if set(candidate_uci) != legal:
            errors.append(
                f"candidate set != legal move set ({len(candidate_uci)} vs {len(legal)})"
            )
        if len(candidate_uci) != len(set(candidate_uci)):
            errors.append("duplicate candidate UCI moves")

        for candidate in candidates:
            mf = candidate.get("move_features", {})
            numeric_keys = (
                "from_square_index",
                "to_square_index",
                "captured_piece_type",
                "moved_piece_type",
                "promotion_piece",
            )
            for key in numeric_keys:
                value = mf.get(key, 0)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"candidate {candidate.get('uci')}: {key} must be numeric")

        target = record["training_target"]
        optimal_moves = target.get("optimal_moves", [])
        if not optimal_moves:
            errors.append("optimal_moves is empty")

        for move_uci in optimal_moves:
            if move_uci not in legal:
                errors.append(f"optimal move is not legal: {move_uci}")

        best_move = target.get("best_move")
        if best_move not in optimal_moves:
            errors.append("best_move must be one of optimal_moves")

        labels = {c["uci"]: c["training_label"] for c in candidates}
        max_score = max(labels[m]["ordering_score"] for m in candidate_uci)
        for move_uci in optimal_moves:
            if labels[move_uci]["ordering_score"] != max_score:
                errors.append("optimal_moves contains a non-optimal move")

        played = (record.get("metadata") or {}).get("played_move")
        if played and played.get("uci") not in legal:
            errors.append("played_move is not legal")

        expected_past_plies = max(0, position["ply"] - 1)
        hist = record["recent_history"]["moves"]
        if len(hist) != min(expected_past_plies, MAX_HISTORY_PLIES):
            errors.append("recent_history length mismatch")
        if hist and hist[-1]["ply"] != position["ply"] - 1:
            errors.append("recent_history does not end at ply-1")

        for hmove in hist:
            if "move_features" not in hmove:
                errors.append("history move missing move_features")
                continue
            hf = hmove["move_features"]
            for key in ("captured_piece_type", "moved_piece_type", "promotion_piece"):
                value = hf.get(key, 0)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"history {hmove.get('uci')}: {key} must be numeric")
            if "features" in hmove:
                errors.append("legacy history field 'features' should not be present")
            if "is_castle" not in hf or "is_castling" not in hf:
                errors.append(f"history {hmove.get('uci')}: missing castling flag")

        mt = record["long_term_memory"]["material_trajectory"]
        if mt.get("past_plies") != expected_past_plies:
            errors.append("material_trajectory.past_plies mismatch")
        if mt.get("position_count_in_history") != position["ply"]:
            errors.append("material_trajectory.position_count_in_history mismatch")

        if "v04_explicit_signals" not in record.get("long_term_memory", {}):
            errors.append("missing long_term_memory.v04_explicit_signals")
        if "v04_tactical_summary" not in position.get("position_state", {}):
            errors.append("missing position_state.v04_tactical_summary")

        for color in ("white", "black"):
            if "backward_pawns" in record["long_term_memory"]["pawn_structure"].get(color, {}):
                errors.append("backward_pawns is still present")

        game_meta = record.get("game", {})
        if "round" in game_meta or "date" in game_meta:
            errors.append("game metadata still contains round/date")

    except Exception as exc:
        errors.append(f"validation exception: {exc}")

    return errors


def record_is_v04_structurally_complete(record: Dict[str, Any]) -> bool:
    try:
        if record.get("schema_version") != SCHEMA_VERSION:
            return False
        if record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            return False

        position = record.get("position") or {}
        state = position.get("position_state") or {}
        history = record.get("recent_history") or {}
        memory = record.get("long_term_memory") or {}

        if "v04_tactical_summary" not in state:
            return False
        if "v04_explicit_signals" not in memory:
            return False

        moves = history.get("moves")
        if not isinstance(moves, list):
            return False

        for hmove in moves:
            if not isinstance(hmove, dict):
                return False
            mf = hmove.get("move_features")
            if not isinstance(mf, dict):
                return False
            for key in ("captured_piece_type", "moved_piece_type", "promotion_piece"):
                value = mf.get(key, 0)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return False
            if "is_castling" not in mf:
                return False

        candidates = record.get("candidate_moves")
        if not isinstance(candidates, list) or not candidates:
            return False
        target = record.get("training_target") or {}
        if target.get("candidate_source") != "all_legal_moves":
            return False

        return True
    except Exception:
        return False


def find_partial_games_needing_upgrade(partial_path: Path) -> set[str]:
    if not partial_path.exists():
        return set()

    bad_games: set[str] = set()
    with partial_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Partial dataset contains invalid JSON at line {line_no}: {exc}"
                ) from exc
            gid = record.get("game", {}).get("game_id")
            if gid and not record_is_v04_structurally_complete(record):
                bad_games.add(gid)

    return bad_games


def _legacy_engine_label(raw_candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = raw_candidate.get("training_label")
    if not isinstance(label, dict):
        return None

    ordering_score = label.get("ordering_score")
    score_cp = label.get("score_cp")
    mate = label.get("mate")

    if ordering_score is None:
        if mate is not None:
            try:
                mate_int = int(mate)
            except (TypeError, ValueError):
                return None
            ordering_score = 100000 - abs(mate_int) * 100 if mate_int > 0 else -100000 + abs(mate_int) * 100
        elif score_cp is not None:
            try:
                ordering_score = int(score_cp)
            except (TypeError, ValueError):
                return None
        else:
            return None

    return {
        "uci": str(raw_candidate.get("uci", "")),
        "score_cp": int(score_cp) if score_cp is not None else None,
        "mate": int(mate) if mate is not None else None,
        "ordering_score": int(ordering_score),
        "wdl": label.get("wdl"),
        "depth": safe_int(label.get("depth"), 0),
        "nodes": safe_int(label.get("nodes"), 0),
        "time": label.get("time"),
        "principal_variation": list(label.get("principal_variation") or []),
    }


def _rebuild_history_state_and_append(
    board: chess.Board,
    move: chess.Move,
    ply: int,
    move_history: List[Dict[str, Any]],
    material_history: List[int],
    repetition_counts: Dict[Any, int],
    move_counts_by_piece: Dict[str, Dict[str, int]],
    check_counts: Dict[str, int],
    capture_counts: Dict[str, int],
    castle_counts: Dict[str, int],
    promotion_counts: Dict[str, int],
    strategic_events: List[Dict[str, Any]],
) -> None:
    before = board.copy(stack=False)
    moved_piece = before.piece_at(move.from_square)
    moved_piece_type = moved_piece.piece_type if moved_piece else chess.PAWN
    side_name = "white" if before.turn == chess.WHITE else "black"

    before_pawn_files = {
        chess.square_file(sq)
        for color in (chess.WHITE, chess.BLACK)
        for sq in before.pieces(chess.PAWN, color)
    }
    before_passed = {
        color: count_pawn_features(before, color)["passed_pawns"]
        for color in (chess.WHITE, chess.BLACK)
    }

    piece_name = PIECE_NAMES.get(moved_piece_type, "unknown")
    was_first_piece_move = move_counts_by_piece[side_name][piece_name] == 0
    is_capture = before.is_capture(move)
    is_castle = before.is_castling(move)
    is_promotion = move.promotion is not None

    board.push(move)

    after_pawn_files = {
        chess.square_file(sq)
        for color in (chess.WHITE, chess.BLACK)
        for sq in board.pieces(chess.PAWN, color)
    }
    after_passed = {
        color: count_pawn_features(board, color)["passed_pawns"]
        for color in (chess.WHITE, chess.BLACK)
    }

    events = move_event_types(
        before=before,
        move=move,
        after=board,
        moved_piece_type=moved_piece_type,
        move_was_first_for_type=was_first_piece_move,
        before_pawn_files=before_pawn_files,
        after_pawn_files=after_pawn_files,
        before_passed=before_passed,
        after_passed=after_passed,
    )
    strategic_events.extend({**ev, "ply": ply} for ev in events)

    move_counts_by_piece[side_name][piece_name] += 1
    if board.is_check():
        check_counts[side_name] += 1
    if is_capture:
        capture_counts[side_name] += 1
    if is_castle:
        castle_counts[side_name] += 1
    if is_promotion:
        promotion_counts[side_name] += 1

    move_history.append({
        "ply": ply,
        "uci": move.uci(),
        "san": before.san(move),
        "side": side_name,
        "move_features": history_move_features(
            before=before,
            move=move,
            after=board,
            strategic_event_count=len(events),
        ),
        "strategic_events": events,
    })

    material_history.append(material_difference(board))
    repetition_counts[board_key(board)] += 1


def upgrade_existing_game_records(
    game: chess.pgn.Game,
    legacy_records: Sequence[Dict[str, Any]],
    default_depth: int,
    default_nodes: Optional[int],
) -> List[Dict[str, Any]]:
    moves = game_move_list(game)
    by_ply = {
        safe_int(r.get("position", {}).get("ply"), -1): r
        for r in legacy_records
    }

    if sorted(by_ply) != list(range(1, len(moves) + 1)):
        raise RuntimeError("legacy game does not contain a complete 1..N ply sequence")

    board = game.board()
    move_history: List[Dict[str, Any]] = []
    material_history: List[int] = [material_difference(board)]
    repetition_counts: Dict[Any, int] = defaultdict(int)
    repetition_counts[board_key(board)] = 1
    move_counts_by_piece = {"white": defaultdict(int), "black": defaultdict(int)}
    check_counts = {"white": 0, "black": 0}
    capture_counts = {"white": 0, "black": 0}
    castle_counts = {"white": 0, "black": 0}
    promotion_counts = {"white": 0, "black": 0}
    strategic_events: List[Dict[str, Any]] = []
    upgraded: List[Dict[str, Any]] = []

    for ply, played_move in enumerate(moves, start=1):
        legacy = by_ply[ply]
        legal_moves = sorted(board.legal_moves, key=lambda m: m.uci())
        raw_candidates = legacy.get("candidate_moves")
        if not isinstance(raw_candidates, list):
            raise RuntimeError(f"ply {ply}: candidate_moves is missing")

        raw_by_uci = {str(c.get("uci")): c for c in raw_candidates if isinstance(c, dict)}
        legal_uci = {m.uci() for m in legal_moves}
        if set(raw_by_uci) != legal_uci:
            raise RuntimeError(
                f"ply {ply}: existing candidate set does not exactly match all legal moves"
            )

        labels: List[Dict[str, Any]] = []
        for move in legal_moves:
            label = _legacy_engine_label(raw_by_uci[move.uci()])
            if label is None:
                raise RuntimeError(
                    f"ply {ply}: candidate {move.uci()} has incomplete engine supervision"
                )
            labels.append(label)

        long_term_memory = build_long_term_memory(
            board=board,
            ply=ply,
            material_history=material_history,
            repetition_counts=repetition_counts,
            move_counts_by_piece=move_counts_by_piece,
            check_counts=check_counts,
            capture_counts=capture_counts,
            castle_counts=castle_counts,
            promotion_counts=promotion_counts,
            strategic_events=strategic_events,
        )
        add_v04_memory_signals(
            long_term_memory=long_term_memory,
            material_history=material_history,
            repetition_counts=repetition_counts,
            check_counts=check_counts,
            capture_counts=capture_counts,
            castle_counts=castle_counts,
            promotion_counts=promotion_counts,
            board=board,
        )

        old_target = legacy.get("training_target") or {}
        selected_depth = safe_int(old_target.get("engine_depth"), default_depth)
        selected_nodes = old_target.get("engine_nodes_limit")
        if selected_nodes is None:
            selected_nodes = default_nodes
        elif safe_int(selected_nodes, -1) < 0:
            selected_nodes = default_nodes

        upgraded.append(
            make_position_record(
                game=game,
                board=board.copy(stack=False),
                move_history=move_history.copy(),
                long_term_memory=long_term_memory,
                legal_moves=legal_moves,
                engine_labels=labels,
                played_move=played_move,
                ply=ply,
                selected_depth=selected_depth,
                stockfish_nodes=selected_nodes,
            )
        )

        _rebuild_history_state_and_append(
            board=board,
            move=played_move,
            ply=ply,
            move_history=move_history,
            material_history=material_history,
            repetition_counts=repetition_counts,
            move_counts_by_piece=move_counts_by_piece,
            check_counts=check_counts,
            capture_counts=capture_counts,
            castle_counts=castle_counts,
            promotion_counts=promotion_counts,
            strategic_events=strategic_events,
        )

    return upgraded


def upgrade_partial_dataset(
    partial_path: Path,
    selected_games: Sequence[chess.pgn.Game],
    default_depth: int,
    default_nodes: Optional[int],
) -> Tuple[set[str], set[str], int]:
    if not partial_path.exists():
        return set(), set(), 0

    selected_map = {game_id_from_headers(g.headers): g for g in selected_games}
    upgraded_games: set[str] = set()
    stockfish_required: set[str] = set()
    records_written = 0
    temp_path = partial_path.with_suffix(partial_path.suffix + ".upgrade.tmp")
    backup_path = partial_path.with_suffix(partial_path.suffix + ".legacy.bak")

    current_gid: Optional[str] = None
    current_records: List[Dict[str, Any]] = []

    def flush_group(out) -> None:
        nonlocal current_gid, current_records, records_written
        if current_gid is None:
            return
        game = selected_map.get(current_gid)
        if game is None:
            raise RuntimeError(
                f"Partial dataset contains game {current_gid!r} which is not in the selected manifest."
            )

        needs_upgrade = not all(
            record_is_v04_structurally_complete(r) for r in current_records
        )

        if not needs_upgrade:
            for rec in current_records:
                out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                records_written += 1
        else:
            try:
                upgraded = upgrade_existing_game_records(
                    game=game,
                    legacy_records=current_records,
                    default_depth=default_depth,
                    default_nodes=default_nodes,
                )
            except Exception:
                stockfish_required.add(current_gid)
                upgraded = []

            if upgraded:
                for rec in upgraded:
                    out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    records_written += 1
                upgraded_games.add(current_gid)

        current_gid = None
        current_records = []

    with partial_path.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cannot upgrade partial dataset: invalid JSON at line {line_no}: {exc}"
                ) from exc

            gid = record.get("game", {}).get("game_id")
            if not gid:
                raise RuntimeError(f"Cannot upgrade partial dataset: line {line_no} has no game_id")

            if current_gid is None:
                current_gid = gid
            elif gid != current_gid:
                flush_group(out)
                current_gid = gid

            current_records.append(record)

        flush_group(out)

    if backup_path.exists():
        backup_path.unlink()
    shutil.copyfile(partial_path, backup_path)
    os.replace(temp_path, partial_path)

    return upgraded_games, stockfish_required, records_written


def load_completed_game_ids(
    partial_path: Path,
    expected_records_by_game: Optional[Dict[str, int]] = None,
) -> set[str]:
    expected_records_by_game = expected_records_by_game or {}

    if not partial_path.exists():
        return set()

    game_counts: Dict[str, int] = defaultdict(int)
    game_plies: Dict[str, List[int]] = defaultdict(list)
    game_schema_ok: Dict[str, bool] = defaultdict(lambda: True)

    with partial_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            gid = record.get("game", {}).get("game_id")
            if not gid:
                continue

            game_counts[gid] += 1
            game_plies[gid].append(
                safe_int(record.get("position", {}).get("ply"), -1)
            )
            if not record_is_v04_structurally_complete(record):
                game_schema_ok[gid] = False

    completed = set()
    for gid, count in game_counts.items():
        expected = expected_records_by_game.get(gid)
        if expected is None:
            if game_schema_ok[gid]:
                completed.add(gid)
            continue

        if (
            game_schema_ok[gid]
            and count == expected
            and sorted(game_plies[gid]) == list(range(1, expected + 1))
        ):
            completed.add(gid)

    return completed


def append_records(
    path: Path,
    records: Sequence[Dict[str, Any]],
    do_fsync: bool = False,
) -> None:
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ProgressTracker:
    def __init__(
        self,
        total_games: int,
        resume_completed: int,
        progress_path: Path,
        status_path: Path,
        interval_seconds: int = 30,
    ) -> None:
        self.total_games = total_games
        self.resume_completed = resume_completed
        self.progress_path = progress_path
        self.status_path = status_path
        self.interval_seconds = max(5, interval_seconds)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.completed_total = resume_completed
        self.successes_current_run = 0
        self.failures_current_run = 0
        self.attempts_current_run = 0
        self.last_game_duration = None
        self.run_started = time.time()
        self.last_checkpoint = None

        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def snapshot(self, reason: str) -> Dict[str, Any]:
        with self.lock:
            now = time.time()
            elapsed = now - self.run_started
            successes = self.successes_current_run
            failures = self.failures_current_run
            attempts = self.attempts_current_run
            completed = self.completed_total

            cpu = None
            if psutil is not None:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                except Exception:
                    cpu = None

            return {
                "timestamp": iso_now(),
                "timestamp_epoch": now,
                "reason": reason,
                "progress": {
                    "completed_games": completed,
                    "target_games": self.total_games,
                    "remaining_games": max(0, self.total_games - completed),
                    "percent": (completed / self.total_games * 100.0)
                    if self.total_games else 100.0,
                },
                "successes_current_run": successes,
                "failures_current_run": failures,
                "attempts_current_run": attempts,
                "time_elapsed_sec": elapsed,
                "time_elapsed_h": elapsed / 3600.0,
                "average_time_per_attempt_sec": (
                    elapsed / attempts if attempts else None
                ),
                "average_time_per_successful_game_sec": (
                    elapsed / successes if successes else None
                ),
                "last_game_time_sec": self.last_game_duration,
                "throughput_games_per_hour": (
                    attempts / elapsed * 3600.0 if elapsed > 0 else None
                ),
                "cpu_utilization_percent": cpu,
                "cpu_utilization_source": "psutil.cpu_percent"
                if psutil is not None else "unavailable",
            }

    def write_checkpoint(self, reason: str) -> None:
        snapshot = self.snapshot(reason)
        with self.lock:
            with self.progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                f.flush()

            tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
            tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            os.replace(tmp, self.status_path)
            self.last_checkpoint = snapshot

    def start(self) -> None:
        self.write_checkpoint("start")

        def heartbeat() -> None:
            while not self.stop_event.wait(self.interval_seconds):
                try:
                    self.write_checkpoint("heartbeat")
                except Exception as exc:
                    print(f"[WARN] Progress checkpoint failed: {exc}", file=sys.stderr)

        self.thread = threading.Thread(
            target=heartbeat,
            name="dataset-progress-heartbeat",
            daemon=True,
        )
        self.thread.start()

    def record_success(self, game_id: str, game_duration: float, positions: int) -> None:
        with self.lock:
            self.completed_total += 1
            self.successes_current_run += 1
            self.attempts_current_run += 1
            self.last_game_duration = game_duration
        self.write_checkpoint("game_success")

    def record_failure(self, game_id: str, game_duration: float) -> None:
        with self.lock:
            self.failures_current_run += 1
            self.attempts_current_run += 1
            self.last_game_duration = game_duration
        self.write_checkpoint("game_failure")

    def stop(self, reason: str = "stop") -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.write_checkpoint(reason)


def build_manifest_from_entries(
    args: argparse.Namespace,
    stockfish_path: str,
    game_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_records_total = sum(safe_int(e.get("expected_records"), 0) for e in game_entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_mode": "all_played_moves_per_game",
        "source_pgn": str(args.pgn.resolve()),
        "requested_games": args.games,
        "qualifying_games": len(game_entries),
        "expected_total_records": expected_records_total,
        "controlled_side_policy": "controlled_side == side_to_move",
        "candidate_policy": "all_legal_moves",
        "tie_policy": "optimal_moves contains every exact best-score tie",
        "history_feature_dim": 16,
        "board_tactical_dim": 20,
        "global_tactical_dim": 18,
        "candidate_tactical_dim": 32,
        "memory_explicit_signal_dim": 8,
        "memory_hash_vector_dim": 32,
        "history_max_plies": MAX_HISTORY_PLIES,
        "engine_labels_are_supervision_only": True,
        "pawn_structure_features": [
            "isolated_pawns",
            "doubled_pawns",
            "passed_pawns",
            "connected_passed_pawns",
        ],
        "elo_min": args.elo_min,
        "elo_max": args.elo_max,
        "elo_step": args.elo_step,
        "balance_mode": args.balance_mode,
        "max_elo_diff": args.max_elo_diff,
        "min_plies": args.min_plies,
        "allow_time_forfeit": bool(args.allow_time_forfeit),
        "novelty_enabled": True,
        "novelty_pool_factor": args.novelty_pool_factor,
        "novelty_probe_depth": args.novelty_probe_depth,
        "novelty_probe_positions": args.novelty_probe_positions,
        "novelty_workers": args.novelty_workers,
        "stockfish": stockfish_path,
        "depth": args.depth,
        "nodes": args.nodes,
        "workers": args.workers,
        "threads_per_engine": args.threads_per_engine,
        "hash_mb": args.hash_mb,
        "allow_bots": bool(args.allow_bots),
        "games": game_entries,
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_complete_game_coverage(
    records_by_game: Dict[str, List[Dict[str, Any]]],
    manifest: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    expected_games = {
        entry["game_id"]: entry
        for entry in manifest.get("games", [])
    }

    actual_ids = set(records_by_game)
    expected_ids = set(expected_games)

    missing_games = expected_ids - actual_ids
    extra_games = actual_ids - expected_ids

    for gid in sorted(missing_games):
        errors.append(f"missing game entirely: {gid}")
    for gid in sorted(extra_games):
        errors.append(f"unexpected game in dataset: {gid}")

    for gid in sorted(expected_ids & actual_ids):
        recs = records_by_game[gid]
        expected_count = safe_int(expected_games[gid].get("expected_records"), -1)
        plies = sorted(
            safe_int(r.get("position", {}).get("ply"), -1)
            for r in recs
        )

        if len(recs) != expected_count:
            errors.append(
                f"{gid}: expected {expected_count} records, found {len(recs)}"
            )

        expected_plies = list(range(1, expected_count + 1))
        if plies != expected_plies:
            errors.append(
                f"{gid}: ply coverage mismatch; expected 1..{expected_count}, "
                f"found first/last={plies[:1]}/{plies[-1:] if plies else []}"
            )

    return errors


def run_final_validation(
    partial_path: Path,
    report_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)

    total_records = 0
    counts = Counter()
    tie_counts = Counter()
    validation_errors: List[str] = []
    records_by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with partial_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                validation_errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue

            total_records += 1
            candidates = record.get("candidate_moves", [])
            counts[len(candidates)] += 1
            tie_counts[
                len((record.get("training_target") or {}).get("optimal_moves", []))
            ] += 1

            gid = record.get("game", {}).get("game_id")
            if gid:
                records_by_game[gid].append(record)
            else:
                validation_errors.append(f"line {line_no}: missing game_id")

            validation_errors.extend(
                f"line {line_no}: {err}"
                for err in validate_record(record)
            )

    validation_errors.extend(
        validate_complete_game_coverage(records_by_game, manifest)
    )

    expected_total_records = safe_int(
        manifest.get("expected_total_records"),
        -1,
    )
    if total_records != expected_total_records:
        validation_errors.append(
            f"total record count mismatch: expected {expected_total_records}, "
            f"found {total_records}"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_mode": "all_played_moves_per_game",
        "records": total_records,
        "unique_games": len(records_by_game),
        "expected_games": manifest.get("qualifying_games"),
        "expected_total_records": expected_total_records,
        "candidate_count_distribution": dict(sorted(counts.items())),
        "optimal_move_tie_distribution": dict(sorted(tie_counts.items())),
        "all_moves_covered_for_every_game": not validation_errors,
        "validation_errors": validation_errors[:100],
        "validation_error_count": len(validation_errors),
        "source_pgn": manifest.get("source_pgn"),
        "stockfish": manifest.get("stockfish"),
        "depth": manifest.get("depth"),
        "nodes": manifest.get("nodes"),
        "game_record_counts": {
            gid: len(records_by_game[gid])
            for gid in sorted(records_by_game)
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_existing_dataset_inventory(
    path: Path,
) -> Tuple[set[str], Optional[str], Optional[str], int]:
    game_ids: set[str] = set()
    first_schema: Optional[str] = None
    first_feature_schema: Optional[str] = None
    total_records = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Existing dataset contains invalid JSON at line {line_no}: {exc}"
                ) from exc

            total_records += 1
            schema = record.get("schema_version")
            feature_schema = record.get("feature_schema_version")

            if first_schema is None:
                first_schema = str(schema)
                first_feature_schema = str(feature_schema)
            elif schema != first_schema or str(feature_schema) != first_feature_schema:
                raise RuntimeError(
                    "Existing dataset contains mixed schema versions. "
                    f"First record={first_schema}/{first_feature_schema}, "
                    f"line {line_no}={schema}/{feature_schema}."
                )

            gid = record.get("game", {}).get("game_id")
            if gid:
                game_ids.add(gid)

    if total_records == 0:
        raise RuntimeError("Existing dataset file is empty.")

    return game_ids, first_schema, first_feature_schema, total_records


def game_elo_values(game: chess.pgn.Game) -> Tuple[int, int, float]:
    white = safe_int(game.headers.get("WhiteElo"), 0)
    black = safe_int(game.headers.get("BlackElo"), 0)
    avg = (white + black) / 2.0 if white > 0 and black > 0 else 0.0
    return white, black, avg


def nearest_elo_bucket(avg_elo: float, elo_min: int, elo_max: int, elo_step: int) -> int:
    if avg_elo <= 0:
        return elo_min
    bucket = elo_min + int((avg_elo - elo_min) / elo_step + 0.5) * elo_step
    return max(elo_min, min(elo_max, bucket))


def is_qualifying_game_elo_range(
    game: chess.pgn.Game,
    elo_min: int,
    elo_max: int,
    elo_step: int,
    active_buckets: set[int],
    require_human_players: bool,
    max_elo_diff: int = 150,
    min_plies: int = 24,
    allow_time_forfeit: bool = False,
) -> Optional[Tuple[int, float, List[chess.Move]]]:
    """
    Fast header + mainline check that fixes the boundary-clipping, player-mismatch,
    and time-forfeit biases:
      - Both players must be within [elo_min - step/2, elo_max + step/2]
      - |white_elo - black_elo| <= max_elo_diff so both players genuinely represent
        the target Elo bucket
      - Game average Elo must map to one of the `active_buckets` that still needs games
      - Termination must be Normal (unless --allow-time-forfeit is passed)
      - Mainline must have at least `min_plies` moves
    """
    headers = game.headers
    white, black, avg_elo = game_elo_values(game)
    half_step = elo_step / 2.0

    if white <= 0 or black <= 0:
        return None
    if avg_elo < (elo_min - half_step) or avg_elo > (elo_max + half_step):
        return None
    if white < (elo_min - half_step) or white > (elo_max + half_step):
        return None
    if black < (elo_min - half_step) or black > (elo_max + half_step):
        return None
    if max_elo_diff > 0 and abs(white - black) > max_elo_diff:
        return None

    bucket = nearest_elo_bucket(avg_elo, elo_min, elo_max, elo_step)
    if bucket not in active_buckets:
        return None

    variant = sanitize_header_value(headers.get("Variant")).lower()
    if variant and variant not in ("standard", "chess960"):
        return None

    if require_human_players:
        white_ai = sanitize_header_value(headers.get("WhiteAI")).lower()
        black_ai = sanitize_header_value(headers.get("BlackAI")).lower()
        white_title = sanitize_header_value(headers.get("WhiteTitle")).upper()
        black_title = sanitize_header_value(headers.get("BlackTitle")).upper()
        if white_ai in ("true", "1", "yes") or black_ai in ("true", "1", "yes"):
            return None
        if white_title == "BOT" or black_title == "BOT":
            return None

    result = sanitize_header_value(headers.get("Result"))
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return None

    if not allow_time_forfeit:
        termination = sanitize_header_value(headers.get("Termination")).lower()
        if termination and termination != "normal":
            return None

    moves = list(game.mainline_moves())
    if len(moves) < max(1, min_plies):
        return None

    return bucket, avg_elo, moves


def _count_bucket(value: float, edges: Sequence[int]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def _eval_bucket(score_cp: Optional[int], mate: Optional[int]) -> str:
    if mate is not None:
        return "M+" if mate > 0 else "M-"
    cp = int(score_cp or 0)
    return str(_count_bucket(cp, (-500, -300, -150, -50, 50, 150, 300, 500)))


def coarse_position_signature(board: chess.Board, ply: int) -> Tuple[Any, ...]:
    summary = build_v04_position_summary(board)
    current = board.turn
    opponent = not current
    current_material = sum(len(board.pieces(pt, current)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    opponent_material = sum(len(board.pieces(pt, opponent)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    material_bucket = _count_bucket(current_material - opponent_material, (-900, -500, -250, -100, 100, 250, 500, 900))
    non_pawn = sum(
        len(board.pieces(pt, color)) * PIECE_VALUES[pt]
        for color in (chess.WHITE, chess.BLACK)
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    phase = "opening" if ply <= 12 else ("endgame" if non_pawn <= 2200 else "middlegame")
    mobility_bucket = _count_bucket(len(list(board.legal_moves)), (5, 10, 20, 35, 50))
    pressure_bucket = _count_bucket(
        summary["current_king_ring_attack_pressure"] - summary["opponent_king_ring_attack_pressure"],
        (-4, -2, -1, 1, 2, 4),
    )
    center_bucket = _count_bucket(summary["center_control_delta"], (-3, -1, 0, 1, 3))
    counts = (
        tuple(len(board.pieces(pt, current)) for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        + tuple(len(board.pieces(pt, opponent)) for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    )
    return (
        phase, int(current), material_bucket, mobility_bucket, bool(summary["in_check"]),
        int(summary["current_side_hanging_non_king"]), int(summary["opponent_side_hanging_non_king"]),
        int(summary["current_side_pinned_non_king"]), int(summary["opponent_side_pinned_non_king"]),
        pressure_bucket, center_bucket, int(summary["current_castling_right_count"]),
        int(summary["opponent_castling_right_count"]), counts,
    )


def move_pattern_signature(board: chess.Board, move: chess.Move) -> Tuple[int, int, int, int, int, int]:
    moved = board.piece_at(move.from_square)
    return (
        int(moved.piece_type if moved else 0),
        int(board.is_capture(move)),
        int(board.is_castling(move)),
        int(board.is_en_passant(move)),
        int(move.promotion or 0),
        int(move.from_square in CENTER_SQUARES or move.to_square in CENTER_SQUARES),
    )


def cheap_game_signature(
    game: chess.pgn.Game,
    moves: Optional[List[chess.Move]] = None,
    sample_positions: int = 24,
) -> Dict[str, Any]:
    board = game.board()
    if moves is None:
        moves = game_move_list(game)
    total = len(moves)
    if not moves:
        return {"opening": "", "positions": [], "move_patterns": [], "ends_in_checkmate": False, "reaches_endgame": False}

    if total <= sample_positions:
        sample_plies = set(range(1, total + 1))
    else:
        sample_plies = {1 + int(round(i * (total - 1) / (sample_positions - 1))) for i in range(sample_positions)}

    positions = []
    move_patterns = []
    opening_moves = []
    reaches_endgame = False

    for ply, move in enumerate(moves, start=1):
        if ply <= 8:
            opening_moves.append(move.uci())
        if ply in sample_plies:
            sig = coarse_position_signature(board, ply)
            if sig[0] == "endgame":
                reaches_endgame = True
            positions.append(sig)
        move_patterns.append(move_pattern_signature(board, move))
        board.push(move)

    ends_in_checkmate = bool(board.is_checkmate())
    return {
        "opening": " ".join(opening_moves),
        "positions": positions,
        "move_patterns": move_patterns[:32],
        "ends_in_checkmate": ends_in_checkmate,
        "reaches_endgame": reaches_endgame,
    }


def _path_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _ids_digest(ids: Sequence[str]) -> str:
    blob = "\n".join(sorted(str(x) for x in ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_key(config: Dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _atomic_pickle_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_pickle_cache(path: Path, config_key: str) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            envelope = pickle.load(f)
        if (
            envelope.get("cache_version") != CACHE_VERSION
            or envelope.get("config_key") != config_key
        ):
            return None
        return envelope.get("payload")
    except Exception:
        return None


def _save_pickle_cache(path: Path, config_key: str, payload: Any) -> None:
    _atomic_pickle_write(
        path,
        {
            "cache_version": CACHE_VERSION,
            "config_key": config_key,
            "payload": payload,
        },
    )


def _probe_partial_header(config_key: str) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "type": "novelty_probe_partial",
        "config_key": config_key,
    }


def _load_probe_partial(path: Path, config_key: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    found: Dict[str, Dict[str, Any]] = {}
    try:
        with path.open("rb") as f:
            first = pickle.load(f)
            if first != _probe_partial_header(config_key):
                return {}
            while True:
                try:
                    record = pickle.load(f)
                except EOFError:
                    break
                if not isinstance(record, dict):
                    continue
                gid = record.get("game_id")
                if gid:
                    found[str(gid)] = record
    except Exception:
        return {}
    return found


def _append_probe_partial(path: Path, record: Dict[str, Any], flush: bool = True) -> None:
    with path.open("ab") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        if flush:
            f.flush()


def _init_probe_partial(path: Path, config_key: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(_probe_partial_header(config_key), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _safe_shutdown_process_pool(
    executor: Optional[ProcessPoolExecutor],
    *,
    terminate_after_sec: float = 2.0,
) -> None:
    if executor is None:
        return

    processes = []
    try:
        processes = list(getattr(executor, "_processes", {}).values())
    except Exception:
        processes = []

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass

    deadline = time.time() + max(0.1, terminate_after_sec)
    for proc in processes:
        try:
            remaining = max(0.0, deadline - time.time())
            if proc.is_alive() and remaining > 0:
                proc.join(timeout=remaining)
        except Exception:
            pass

    for proc in processes:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass


def _novelty_base_config(
    output_path: Path,
    pgn_path: Path,
    existing_ids: set[str],
    existing_records: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "output_fingerprint": _path_fingerprint(output_path),
        "source_pgn_fingerprint": _path_fingerprint(pgn_path),
        "existing_game_count": len(existing_ids),
        "existing_record_count": int(existing_records),
        "existing_ids_digest": _ids_digest(existing_ids),
        "requested_games": int(args.games),
        "elo_min": int(args.elo_min),
        "elo_max": int(args.elo_max),
        "elo_step": int(args.elo_step),
        "balance_mode": str(getattr(args, "balance_mode", "deficit")),
        "max_elo_diff": int(getattr(args, "max_elo_diff", 150)),
        "min_plies": int(getattr(args, "min_plies", 24)),
        "allow_time_forfeit": bool(getattr(args, "allow_time_forfeit", False)),
        "novelty_pool_factor": int(args.novelty_pool_factor),
        "novelty_probe_depth": int(args.novelty_probe_depth),
        "novelty_probe_positions": int(args.novelty_probe_positions),
        "novelty_workers": int(args.novelty_workers),
        "novelty_threads_per_engine": int(args.novelty_threads_per_engine),
        "novelty_hash_mb": int(args.novelty_hash_mb),
        "allow_bots": bool(args.allow_bots),
    }


def build_existing_pattern_index(path: Path, log_interval_records: int = 10000) -> Dict[str, Any]:
    opening_counts = Counter()
    position_counts = Counter()
    move_pattern_counts = Counter()
    engine_pattern_counts = Counter()
    game_ids = set()
    game_openings = defaultdict(list)
    game_meta = {}
    game_record_counts = Counter()
    records = 0
    started = time.perf_counter()
    next_log = max(1, log_interval_records)

    print(
        f"[INDEX] Starting read-only novelty index pass; logging every {log_interval_records:,} records",
        flush=True,
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rec = json.loads(line)
            records += 1

            gid = rec.get("game", {}).get("game_id")
            if gid:
                game_ids.add(gid)
                game_record_counts[gid] += 1
                if gid not in game_meta:
                    gm = rec.get("game", {}) or {}
                    game_meta[gid] = {
                        "game_id": gid,
                        "white": sanitize_header_value(gm.get("white_player")),
                        "black": sanitize_header_value(gm.get("black_player")),
                        "white_elo": sanitize_header_value(gm.get("white_elo")),
                        "black_elo": sanitize_header_value(gm.get("black_elo")),
                        "result": sanitize_header_value(gm.get("result")),
                    }

            pos = rec.get("position", {}) or {}
            ply = safe_int(pos.get("ply"), 0)
            played = (rec.get("metadata") or {}).get("played_move") or {}

            if gid and ply <= 8 and played.get("uci"):
                game_openings[gid].append(str(played["uci"]))

            fen = pos.get("fen")
            if isinstance(fen, str):
                try:
                    position_counts[
                        coarse_position_signature(chess.Board(fen), ply)
                    ] += 1
                except Exception:
                    pass

            for c in rec.get("candidate_moves", []):
                mf = c.get("move_features") or {}
                move_pattern_counts[
                    (
                        safe_int(mf.get("moved_piece_type"), 0),
                        int(bool(mf.get("is_capture"))),
                        int(bool(mf.get("is_castling"))),
                        int(bool(mf.get("is_en_passant"))),
                        safe_int(mf.get("promotion_piece"), 0),
                        int(bool(mf.get("center_activity"))),
                    )
                ] += 1

            target = rec.get("training_target") or {}
            candidates = rec.get("candidate_moves") or []
            labels = []
            for c in candidates:
                label = c.get("training_label") or {}
                ordering = label.get("ordering_score")
                if ordering is not None:
                    labels.append(int(ordering))
            labels.sort(reverse=True)

            best_score_cp = target.get("best_score_cp")
            best_mate = target.get("best_mate")

            second_gap_bucket = 0
            if len(labels) >= 2:
                second_gap_bucket = _count_bucket(
                    abs(labels[0] - labels[1]),
                    (5, 15, 30, 60, 120, 250),
                )

            played_loss = ((rec.get("metadata") or {}).get("played_move") or {}).get(
                "centipawn_loss"
            )
            candidate_count = len(candidates)
            side_to_move = pos.get("side_to_move")
            side_id = 1 if side_to_move == "white" else 0
            is_check = int(
                bool((pos.get("position_state") or {}).get("is_check", False))
            )

            engine_pattern_counts[
                (
                    _eval_bucket(best_score_cp, best_mate),
                    second_gap_bucket,
                    _count_bucket(
                        safe_int(played_loss, 0),
                        (1, 10, 30, 80, 150, 300, 600),
                    ),
                    _count_bucket(
                        candidate_count,
                        (5, 10, 20, 30, 45, 60),
                    ),
                    is_check,
                    side_id,
                )
            ] += 1

            if records >= next_log:
                elapsed = time.perf_counter() - started
                rate = records / max(0.001, elapsed)
                print(
                    f"[INDEX] records={records:,} | games={len(game_ids):,} | "
                    f"position_patterns={len(position_counts):,} | "
                    f"engine_patterns={len(engine_pattern_counts):,} | "
                    f"rate={rate:,.0f} rec/s | elapsed={_fmt_seconds(elapsed)}",
                    flush=True,
                )
                next_log += max(1, log_interval_records)

    print(
        f"[INDEX] Complete: records={records:,} games={len(game_ids):,} | "
        f"position_patterns={len(position_counts):,} "
        f"engine_patterns={len(engine_pattern_counts):,} | "
        f"elapsed={_fmt_seconds(time.perf_counter() - started)}",
        flush=True,
    )

    for gid, moves in game_openings.items():
        if moves:
            opening_counts[" ".join(moves)] += 1

    return {
        "game_ids": game_ids,
        "opening_counts": opening_counts,
        "position_counts": position_counts,
        "move_pattern_counts": move_pattern_counts,
        "engine_pattern_counts": engine_pattern_counts,
        "records": records,
        "game_meta": game_meta,
        "game_record_counts": game_record_counts,
    }


def novelty_against_index(signature: Dict[str, Any], index: Dict[str, Any]) -> float:
    positions = signature.get("positions") or []
    moves = signature.get("move_patterns") or []
    pos_novel = (
        sum(
            1.0 if index["position_counts"].get(p, 0) == 0
            else 1.0 / math.sqrt(1.0 + index["position_counts"][p])
            for p in positions
        )
        / max(1, len(positions))
    )
    move_novel = (
        sum(
            1.0 if index["move_pattern_counts"].get(m, 0) == 0
            else 1.0 / math.sqrt(1.0 + index["move_pattern_counts"][m])
            for m in moves
        )
        / max(1, len(moves))
    )
    opening = signature.get("opening", "")
    opening_novel = 1.0 if opening and index["opening_counts"].get(opening, 0) == 0 else 0.0

    # Reward games that end in clean checkmate or reach true endgames so the
    # model learns endgame conversion and checkmate patterns.
    finish_bonus = 0.0
    if signature.get("ends_in_checkmate"):
        finish_bonus += 0.08
    if signature.get("reaches_endgame"):
        finish_bonus += 0.05

    return 0.55 * pos_novel + 0.30 * move_novel + 0.15 * opening_novel + finish_bonus


def _stockfish_novelty_probe(payload: Dict[str, Any]) -> Dict[str, Any]:
    game = reconstruct_game(payload["pgn"])
    moves = game_move_list(game)
    total = len(moves)
    requested = max(1, min(payload["probe_positions"], total))
    if total <= requested:
        sample_plies = list(range(1, total + 1))
    else:
        sample_plies = sorted({1 + int(round(i * (total - 1) / (requested - 1))) for i in range(requested)})
    board = game.board()
    signatures = []
    for ply, move in enumerate(moves, start=1):
        if ply in sample_plies:
            limit = chess.engine.Limit(depth=payload["probe_depth"])
            analysis = _WORKER_ENGINE.analyse(board, limit, multipv=2)
            if isinstance(analysis, dict):
                analysis = [analysis]
            if not analysis:
                board.push(move)
                continue
            best_score = analysis[0].get("score")
            if best_score is None:
                board.push(move)
                continue
            best_cp, best_mate, _ = score_to_label(best_score.pov(board.turn))
            second_gap_bucket = 0
            if len(analysis) >= 2 and analysis[1].get("score") is not None and best_mate is None:
                second_cp, _, _ = score_to_label(analysis[1]["score"].pov(board.turn))
                second_gap_bucket = _count_bucket(abs(int(best_cp or 0) - int(second_cp or 0)), (5, 15, 30, 60, 120, 250))

            # Optimization: Reuse score from multipv=2 if `move` is already in top-2 PVs!
            played_score = None
            for info_item in analysis:
                pv_list = info_item.get("pv") or []
                if pv_list and pv_list[0] == move:
                    played_score = info_item.get("score")
                    break
            if played_score is None:
                played_info = _WORKER_ENGINE.analyse(board, limit, root_moves=[move])
                played_score = played_info.get("score") if isinstance(played_info, dict) else None

            played_cp = played_mate = None
            if played_score is not None:
                played_cp, played_mate, _ = score_to_label(played_score.pov(board.turn))
            if best_mate is not None or played_mate is not None:
                loss_bucket = 6 if best_mate is not None and best_mate > 0 and played_mate != best_mate else 0
            else:
                loss_bucket = _count_bucket(max(0, int(best_cp or 0) - int(played_cp or 0)), (1, 10, 30, 80, 150, 300, 600))
            signatures.append((
                _eval_bucket(best_cp, best_mate), second_gap_bucket, loss_bucket,
                _count_bucket(len(list(board.legal_moves)), (5, 10, 20, 30, 45, 60)),
                int(board.is_check()), int(board.turn),
            ))
        board.push(move)
    return {
        "game_id": payload["game_id"], "pgn": payload["pgn"], "avg_elo": payload["avg_elo"],
        "elo_bucket": payload["elo_bucket"], "cheap_novelty": payload["cheap_novelty"],
        "engine_signatures": signatures,
    }


def engine_novelty_against_index(result: Dict[str, Any], index: Dict[str, Any]) -> float:
    signatures = result.get("engine_signatures") or []
    if not signatures:
        return 0.0
    return (
        sum(
            1.0 if index["engine_pattern_counts"].get(sig, 0) == 0
            else 1.0 / math.sqrt(1.0 + index["engine_pattern_counts"][sig])
            for sig in signatures
        )
        / len(signatures)
    )


def recover_games_by_ids_from_pgn(pgn_path: Path, game_ids: set[str]) -> Dict[str, chess.pgn.Game]:
    found: Dict[str, chess.pgn.Game] = {}
    if not game_ids:
        return found
    with pgn_path.open("rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                gid = game_id_from_headers(game.headers)
                if gid in game_ids:
                    found[gid] = game
                    if len(found) == len(game_ids):
                        break
    missing = game_ids - set(found)
    if missing:
        raise RuntimeError(
            "Could not recover selected game IDs from PGN. "
            + ", ".join(sorted(missing)[:10])
        )
    return found


def compute_elo_bucket_quotas(
    levels: Sequence[int],
    needed_new: int,
    existing_index: Dict[str, Any],
    elo_min: int,
    elo_max: int,
    elo_step: int,
    balance_mode: str = "deficit",
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Computes how many new games to allocate to each Elo bucket in `levels`.
      - "deficit" (water-filling, default): Counts existing games per Elo bucket
        and iteratively allocates each of the `needed_new` games to whichever
        bucket currently has the lowest total `(existing + allocated)`.
      - "uniform": Divides `needed_new` equally across `levels`.
    """
    existing_bucket_counts: Dict[int, int] = {level: 0 for level in levels}
    game_meta = existing_index.get("game_meta") or {}

    for meta in game_meta.values():
        w_elo = safe_int(meta.get("white_elo"), 0)
        b_elo = safe_int(meta.get("black_elo"), 0)
        if w_elo <= 0 or b_elo <= 0:
            continue
        avg_elo = (w_elo + b_elo) / 2.0
        if avg_elo < elo_min or avg_elo > elo_max:
            continue
        b = nearest_elo_bucket(avg_elo, elo_min, elo_max, elo_step)
        if b in existing_bucket_counts:
            existing_bucket_counts[b] += 1

    if balance_mode == "uniform":
        quota = {
            level: needed_new // len(levels) + (1 if i < needed_new % len(levels) else 0)
            for i, level in enumerate(levels)
        }
        return quota, existing_bucket_counts

    # Water-filling algorithm for deficit balancing
    quota = {level: 0 for level in levels}
    for _ in range(needed_new):
        # Pick bucket with minimum (existing + quota); tie-break by lower Elo first
        best_level = min(
            levels,
            key=lambda lvl: (existing_bucket_counts[lvl] + quota[lvl], lvl),
        )
        quota[best_level] += 1

    return quota, existing_bucket_counts


def select_novel_games_covering_existing(
    pgn_path: Path,
    target_games: int,
    elo_min: int,
    elo_max: int,
    elo_step: int,
    balance_mode: str,
    max_elo_diff: int,
    min_plies: int,
    allow_time_forfeit: bool,
    require_human_players: bool,
    required_existing_ids: set[str],
    existing_index: Dict[str, Any],
    novelty_pool_factor: int,
    novelty_probe_depth: int,
    novelty_probe_positions: int,
    novelty_workers: int,
    novelty_hash_mb: int,
    novelty_threads_per_engine: int,
    novelty_stockfish_path: str,
    output_path: Path,
    existing_records: int,
    force_recompute_novelty: bool,
    scan_log_interval_games: int = 10000,
    scan_log_interval_seconds: int = 15,
    probe_log_every: int = 10,
    selection_log_every: int = 25,
) -> Tuple[List[chess.pgn.Game], Dict[str, Any], int]:
    needed_new = target_games - len(required_existing_ids)
    if needed_new <= 0:
        return [], {"new_games": 0, "candidate_pool": 0, "selected": []}, 0

    levels = list(range(elo_min, elo_max + 1, elo_step))
    quota, existing_bucket_counts = compute_elo_bucket_quotas(
        levels=levels,
        needed_new=needed_new,
        existing_index=existing_index,
        elo_min=elo_min,
        elo_max=elo_max,
        elo_step=elo_step,
        balance_mode=balance_mode,
    )

    active_levels = [lvl for lvl in levels if quota.get(lvl, 0) > 0]
    active_buckets_set = set(active_levels)
    pool_by_bucket = {
        lvl: (max(20, quota[lvl] * max(2, novelty_pool_factor)) if quota[lvl] > 0 else 0)
        for lvl in levels
    }
    total_target_pool = sum(pool_by_bucket.values())

    print("\n[ELO DEFICIT-FILLING PLAN]")
    for lvl in levels:
        ex_c = existing_bucket_counts.get(lvl, 0)
        add_c = quota.get(lvl, 0)
        pool_c = pool_by_bucket.get(lvl, 0)
        print(
            f"  Elo {lvl:4d}: existing={ex_c:4d} + new_quota={add_c:4d} -> "
            f"projected_total={ex_c + add_c:4d} (candidate_pool_target={pool_c:4d})",
            flush=True,
        )

    base_config = _novelty_base_config(
        output_path,
        pgn_path,
        required_existing_ids,
        existing_records,
        argparse.Namespace(
            games=target_games,
            elo_min=elo_min,
            elo_max=elo_max,
            elo_step=elo_step,
            balance_mode=balance_mode,
            max_elo_diff=max_elo_diff,
            min_plies=min_plies,
            allow_time_forfeit=allow_time_forfeit,
            novelty_pool_factor=novelty_pool_factor,
            novelty_probe_depth=novelty_probe_depth,
            novelty_probe_positions=novelty_probe_positions,
            novelty_workers=novelty_workers,
            novelty_threads_per_engine=novelty_threads_per_engine,
            novelty_hash_mb=novelty_hash_mb,
            allow_bots=require_human_players is False,
        ),
    )
    base_config["required_existing_ids_digest"] = _ids_digest(required_existing_ids)
    base_config["quota_by_bucket"] = {str(k): int(v) for k, v in quota.items()}
    base_config["pool_by_bucket"] = {str(k): int(v) for k, v in pool_by_bucket.items()}
    base_config["candidate_pool_total"] = total_target_pool

    candidate_config = {**base_config, "cache_type": "candidate_pool"}
    candidate_key = _cache_key(candidate_config)
    candidate_cache_path = Path(str(output_path) + ".novelty_candidates.pkl")

    selected_config = {**base_config, "cache_type": "selected_games"}
    selected_key = _cache_key(selected_config)
    selected_cache_path = Path(str(output_path) + ".novelty_selected.pkl")

    probe_config = {**base_config, "cache_type": "probe_results"}
    probe_key = _cache_key(probe_config)
    probe_cache_path = Path(str(output_path) + ".novelty_probes.pkl")
    probe_partial_path = Path(str(output_path) + ".novelty_probes.partial.pklstream")

    scan_started = time.perf_counter()

    # ------------------------------------------------------------
    # Phase 1: bounded PGN scan.
    # ------------------------------------------------------------
    candidate_payload = None
    if not force_recompute_novelty:
        candidate_payload = _load_pickle_cache(candidate_cache_path, candidate_key)

    if candidate_payload:
        candidates = list(candidate_payload["candidates"])
        scanned = int(candidate_payload["scanned"])
        eligible = int(candidate_payload["eligible"])
        skipped_existing = int(candidate_payload["skipped_existing"])
        bucket_seen = Counter(candidate_payload["bucket_seen"])
        print(
            f"[CACHE] Reusing candidate pool: {len(candidates):,} candidates "
            f"from bounded scan of {scanned:,} PGN games.",
            flush=True,
        )
    else:
        heaps: Dict[int, List[Any]] = {level: [] for level in active_levels}
        scanned = 0
        serial = 0
        eligible = 0
        skipped_existing = 0
        bucket_seen: Counter = Counter()
        next_game_log = max(1, scan_log_interval_games)
        next_time_log = time.perf_counter() + max(1, scan_log_interval_seconds)

        print("\n[SELECT] Phase 1/4: bounded PGN scan + cheap novelty", flush=True)
        print(
            f"[SELECT] Need {needed_new:,} new games across active buckets {active_levels}; "
            f"maximum_candidates={total_target_pool:,}",
            flush=True,
        )

        with pgn_path.open("rb") as raw:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw) as reader:
                text_stream = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                    errors="replace",
                )

                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break

                    scanned += 1
                    gid = game_id_from_headers(game.headers)

                    if gid in required_existing_ids:
                        skipped_existing += 1
                        continue

                    qual = is_qualifying_game_elo_range(
                        game=game,
                        elo_min=elo_min,
                        elo_max=elo_max,
                        elo_step=elo_step,
                        active_buckets=active_buckets_set,
                        require_human_players=require_human_players,
                        max_elo_diff=max_elo_diff,
                        min_plies=min_plies,
                        allow_time_forfeit=allow_time_forfeit,
                    )
                    if qual is None:
                        now = time.perf_counter()
                        if scanned >= next_game_log or now >= next_time_log:
                            rate = scanned / max(0.001, now - scan_started)
                            print(
                                f"[SELECT] scanned={scanned:,} | eligible={eligible:,} | "
                                f"existing-skipped={skipped_existing:,} | "
                                f"rate={rate:,.1f} games/s | "
                                f"elapsed={_fmt_seconds(now - scan_started)} | "
                                f"pool={sum(len(h) for h in heaps.values()):,}/{total_target_pool:,}",
                                flush=True,
                            )
                            next_game_log = scanned + max(1, scan_log_interval_games)
                            next_time_log = now + max(1, scan_log_interval_seconds)
                        continue

                    bucket, avg_elo, moves = qual
                    eligible += 1
                    bucket_seen[bucket] += 1

                    sig = cheap_game_signature(game, moves=moves)
                    cheap = novelty_against_index(sig, existing_index)
                    serial += 1

                    item_payload = {
                        "game_id": gid,
                        "pgn": str(game),
                        "avg_elo": avg_elo,
                        "elo_bucket": bucket,
                        "cheap_novelty": cheap,
                        "positions": sig.get("positions", []),
                        "move_patterns": sig.get("move_patterns", []),
                        "opening": sig.get("opening", ""),
                        "ends_in_checkmate": sig.get("ends_in_checkmate", False),
                        "reaches_endgame": sig.get("reaches_endgame", False),
                    }
                    item = (cheap, serial, item_payload)

                    heap = heaps[bucket]
                    bucket_cap = pool_by_bucket[bucket]
                    if len(heap) < bucket_cap:
                        heapq.heappush(heap, item)
                    elif cheap > heap[0][0]:
                        heapq.heapreplace(heap, item)

                    if all(
                        len(heaps[level]) >= pool_by_bucket[level]
                        for level in active_levels
                    ):
                        total_pool = sum(len(h) for h in heaps.values())
                        print(
                            f"[SELECT] Candidate pool quota reached for ALL "
                            f"{len(active_levels)} active Elo buckets at scanned={scanned:,}; "
                            f"pool={total_pool:,}/{total_target_pool:,}. "
                            "Stopping PGN scan immediately.",
                            flush=True,
                        )
                        break

                    now = time.perf_counter()
                    if scanned >= next_game_log or now >= next_time_log:
                        rate = scanned / max(0.001, now - scan_started)
                        print(
                            f"[SELECT] scanned={scanned:,} | eligible={eligible:,} | "
                            f"existing-skipped={skipped_existing:,} | "
                            f"rate={rate:,.1f} games/s | "
                            f"elapsed={_fmt_seconds(now - scan_started)} | "
                            f"pool={sum(len(h) for h in heaps.values()):,}/{total_target_pool:,}",
                            flush=True,
                        )
                        next_game_log = scanned + max(1, scan_log_interval_games)
                        next_time_log = now + max(1, scan_log_interval_seconds)

        candidates = [
            item[2]
            for heap in heaps.values()
            for item in heap
        ]

        if not candidates:
            raise RuntimeError("No new games matched the requested Elo range.")

        _save_pickle_cache(
            candidate_cache_path,
            candidate_key,
            {
                "candidates": candidates,
                "scanned": scanned,
                "eligible": eligible,
                "skipped_existing": skipped_existing,
                "bucket_seen": dict(bucket_seen),
            },
        )
        print(f"[CACHE] Candidate pool checkpoint saved: {candidate_cache_path}", flush=True)

    print(
        f"[SELECT] Phase 1/4 complete: scanned={scanned:,} | "
        f"eligible={eligible:,} | existing-skipped={skipped_existing:,} | "
        f"elapsed={_fmt_seconds(time.perf_counter() - scan_started)}",
        flush=True,
    )
    for level in levels:
        count = sum(1 for c in candidates if c["elo_bucket"] == level)
        print(
            f"[SELECT] Elo {level}: pool={count:,}/{pool_by_bucket[level]:,} | "
            f"final quota={quota[level]:,} | seen={bucket_seen[level]:,}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Phase 2: shallow Stockfish novelty probes, resumable.
    # ------------------------------------------------------------
    results_by_id: Dict[str, Dict[str, Any]] = {}

    if not force_recompute_novelty:
        final_probe_payload = _load_pickle_cache(probe_cache_path, probe_key)
        if final_probe_payload:
            for result in final_probe_payload.get("results", []):
                results_by_id[result["game_id"]] = result
            print(
                f"[CACHE] Reusing completed novelty probes: "
                f"{len(results_by_id):,}/{len(candidates):,}",
                flush=True,
            )
        else:
            partial_results = _load_probe_partial(probe_partial_path, probe_key)
            if partial_results:
                results_by_id.update(partial_results)
                print(
                    f"[CACHE] Resuming novelty probes from checkpoint: "
                    f"{len(results_by_id):,}/{len(candidates):,}",
                    flush=True,
                )

    missing_candidates = [
        c for c in candidates if c["game_id"] not in results_by_id
    ]

    if missing_candidates:
        print(
            f"[SELECT] Phase 2/4: shallow Stockfish novelty probes "
            f"({len(missing_candidates):,} remaining)",
            flush=True,
        )

        if force_recompute_novelty or not probe_partial_path.exists():
            _init_probe_partial(probe_partial_path, probe_key)

        executor: Optional[ProcessPoolExecutor] = None
        probe_started = time.perf_counter()
        probe_completed_before = len(results_by_id)

        try:
            import multiprocessing as mp

            kwargs = {
                "max_workers": max(1, novelty_workers),
                "initializer": init_worker,
                "initargs": (
                    novelty_stockfish_path,
                    novelty_probe_depth,
                    None,
                    novelty_hash_mb,
                    novelty_threads_per_engine,
                ),
            }
            try:
                kwargs["mp_context"] = mp.get_context("fork")
            except Exception:
                pass

            executor = ProcessPoolExecutor(**kwargs)
            payloads = [
                {
                    **candidate,
                    "probe_depth": novelty_probe_depth,
                    "probe_positions": novelty_probe_positions,
                }
                for candidate in missing_candidates
            ]

            chunksize = max(1, min(16, max(1, len(payloads) // max(1, novelty_workers * 8))))
            print(
                f"[PROBE] Starting {len(payloads):,} probe jobs with "
                f"workers={novelty_workers}, chunksize={chunksize}",
                flush=True,
            )

            iterator = executor.map(
                _stockfish_novelty_probe,
                payloads,
                chunksize=chunksize,
            )

            for local_done, result in enumerate(iterator, start=1):
                results_by_id[result["game_id"]] = result
                _append_probe_partial(
                    probe_partial_path,
                    result,
                    flush=(local_done % max(1, probe_log_every) == 0),
                )

                overall_done = probe_completed_before + local_done
                if local_done % max(1, probe_log_every) == 0 or local_done == len(payloads):
                    print(
                        "[PROBE] "
                        + _progress_line(
                            overall_done,
                            len(candidates),
                            probe_started,
                            "completed",
                        ),
                        flush=True,
                    )

            print(
                "[PROBE] All probe jobs returned; closing novelty worker pool...",
                flush=True,
            )
        except Exception as exc:
            print(f"[PROBE][ERROR] {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            _safe_shutdown_process_pool(executor)

        if len(results_by_id) == len(candidates):
            _save_pickle_cache(
                probe_cache_path,
                probe_key,
                {"results": [results_by_id[c["game_id"]] for c in candidates]},
            )
            try:
                probe_partial_path.unlink()
            except FileNotFoundError:
                pass
            print(f"[PROBE] Complete probe cache saved: {probe_cache_path}", flush=True)
    else:
        print(
            "[SELECT] Phase 2/4: all novelty probes already cached; no Stockfish probe work required.",
            flush=True,
        )

    if len(results_by_id) != len(candidates):
        raise RuntimeError(
            f"Novelty probe result count mismatch: {len(results_by_id)} != {len(candidates)}"
        )

    results = []
    for candidate in candidates:
        result = dict(results_by_id[candidate["game_id"]])
        result["_positions"] = candidate.get("positions", [])
        result["_moves"] = candidate.get("move_patterns", [])
        result["opening"] = candidate.get("opening", "")
        result["cheap_novelty"] = candidate["cheap_novelty"]
        results.append(result)

    print(f"[SELECT] Phase 2/4 complete: {len(results):,} probe results", flush=True)

    # ------------------------------------------------------------
    # Phase 3: novelty scoring.
    # ------------------------------------------------------------
    print("[SELECT] Phase 3/4: scoring candidate novelty", flush=True)
    score_started = time.perf_counter()

    for n, result in enumerate(results, start=1):
        result["engine_novelty"] = engine_novelty_against_index(result, existing_index)
        result["overall_novelty"] = (
            0.45 * result["cheap_novelty"]
            + 0.55 * result["engine_novelty"]
        )
        if n % max(1, probe_log_every * 10) == 0 or n == len(results):
            print(
                f"[SCORE] scored={n:,}/{len(results):,} | "
                f"elapsed={_fmt_seconds(time.perf_counter() - score_started)}",
                flush=True,
            )

    # ------------------------------------------------------------
    # Phase 4: diversity-aware exact Elo-quota selection.
    # ------------------------------------------------------------
    print("[SELECT] Phase 4/4: diversity-aware Elo-balanced selection", flush=True)

    temp_pos = Counter(existing_index["position_counts"])
    temp_move = Counter(existing_index["move_pattern_counts"])
    temp_eng = Counter(existing_index["engine_pattern_counts"])

    available = set(range(len(results)))
    remaining = {k: v for k, v in quota.items() if v > 0}
    selected: List[Dict[str, Any]] = []

    selection_started = time.perf_counter()

    while len(selected) < needed_new and available:
        best = None
        best_score = -1.0

        for idx in available:
            result = results[idx]
            bucket = result["elo_bucket"]

            if remaining.get(bucket, 0) <= 0:
                continue

            positions = result.get("_positions") or []
            moves = result.get("_moves") or []
            engine_signatures = result.get("engine_signatures") or []

            pg = (
                sum(
                    1.0 if temp_pos[x] == 0 else 1.0 / math.sqrt(1.0 + temp_pos[x])
                    for x in positions
                )
                / max(1, len(positions))
            )
            mg = (
                sum(
                    1.0 if temp_move[x] == 0 else 1.0 / math.sqrt(1.0 + temp_move[x])
                    for x in moves
                )
                / max(1, len(moves))
            )
            eg = (
                sum(
                    1.0 if temp_eng[x] == 0 else 1.0 / math.sqrt(1.0 + temp_eng[x])
                    for x in engine_signatures
                )
                / max(1, len(engine_signatures))
            )

            score = 0.30 * pg + 0.20 * mg + 0.50 * eg

            if score > best_score:
                best_score = score
                best = idx

        if best is None:
            break

        result = results[best]
        selected.append(result)

        bucket = result["elo_bucket"]
        remaining[bucket] = max(0, remaining.get(bucket, 0) - 1)
        available.remove(best)

        for x in result.get("_positions") or []:
            temp_pos[x] += 1
        for x in result.get("_moves") or []:
            temp_move[x] += 1
        for x in result.get("engine_signatures") or []:
            temp_eng[x] += 1

        if len(selected) % max(1, selection_log_every) == 0 or len(selected) == needed_new:
            elapsed = time.perf_counter() - selection_started
            rate = len(selected) / max(0.001, elapsed)
            eta = (needed_new - len(selected)) / rate if rate > 0 else 0
            print(
                f"[CHOOSE] selected={len(selected):,}/{needed_new:,} "
                f"({100.0 * len(selected) / needed_new:5.1f}%) | "
                f"rate={rate:,.2f}/s | "
                f"elapsed={_fmt_seconds(elapsed)} | "
                f"ETA={_fmt_seconds(eta)} | "
                f"remaining_buckets={dict(remaining)}",
                flush=True,
            )

    if len(selected) != needed_new:
        raise RuntimeError(
            f"Could select only {len(selected)} novel games but need {needed_new}. "
            f"Remaining quotas={dict(remaining)}"
        )

    bucket_counts = dict(Counter(r["elo_bucket"] for r in selected))
    expected_non_zero_quota = {k: v for k, v in quota.items() if v > 0}
    if bucket_counts != expected_non_zero_quota:
        raise RuntimeError(
            f"Elo quota violation: selected={bucket_counts}, required={expected_non_zero_quota}"
        )

    report = {
        "new_games": needed_new,
        "balance_mode": balance_mode,
        "existing_bucket_counts": existing_bucket_counts,
        "elo_bucket_counts": bucket_counts,
        "projected_total_bucket_counts": {
            lvl: existing_bucket_counts.get(lvl, 0) + bucket_counts.get(lvl, 0)
            for lvl in levels
        },
        "candidate_pool": len(candidates),
        "candidate_pool_by_bucket": pool_by_bucket,
        "selected": [
            {
                "game_id": r["game_id"],
                "elo_bucket": r["elo_bucket"],
                "avg_elo": r["avg_elo"],
                "cheap_novelty": r["cheap_novelty"],
                "engine_novelty": r["engine_novelty"],
                "overall_novelty": r["overall_novelty"],
            }
            for r in selected
        ],
    }

    _save_pickle_cache(
        selected_cache_path,
        selected_key,
        {
            "selected": selected,
            "report": report,
            "scanned": scanned,
        },
    )

    print(
        f"[SELECT] Complete: selected={len(selected):,} | "
        f"buckets={bucket_counts} | "
        f"total elapsed={_fmt_seconds(time.perf_counter() - scan_started)}",
        flush=True,
    )

    return (
        [reconstruct_game(r["pgn"]) for r in selected],
        report,
        scanned,
    )


def copy_final_to_partial(final_path: Path, partial_path: Path) -> None:
    if partial_path.exists():
        return
    shutil.copyfile(final_path, partial_path)


def manifest_matches_run(
    manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and manifest.get("dataset_mode") == "all_played_moves_per_game"
        and manifest.get("requested_games") == args.games
        and manifest.get("source_pgn") == str(args.pgn.resolve())
        and manifest.get("elo_min") == args.elo_min
        and manifest.get("elo_max") == args.elo_max
        and manifest.get("elo_step") == args.elo_step
        and manifest.get("balance_mode", "deficit") == args.balance_mode
        and manifest.get("max_elo_diff", 150) == args.max_elo_diff
        and manifest.get("min_plies", 24) == args.min_plies
        and bool(manifest.get("allow_time_forfeit", False)) == bool(args.allow_time_forfeit)
        and manifest.get("novelty_enabled", False) is True
        and manifest.get("novelty_pool_factor") == args.novelty_pool_factor
        and manifest.get("novelty_probe_depth") == args.novelty_probe_depth
        and manifest.get("novelty_probe_positions") == args.novelty_probe_positions
        and manifest.get("novelty_workers") == args.novelty_workers
        and manifest.get("depth") == args.depth
        and manifest.get("nodes") == args.nodes
        and manifest.get("workers") == args.workers
        and manifest.get("threads_per_engine") == args.threads_per_engine
        and manifest.get("hash_mb") == args.hash_mb
        and bool(manifest.get("allow_bots", False)) == bool(args.allow_bots)
    )


def run_runtime_self_test() -> None:
    board = chess.Board()
    zero_counts = {"white": 0, "black": 0}
    zero_reps = defaultdict(int)
    zero_reps[board_key(board)] = 1

    for history in ([], [0], [0] * 10, [0] * 11):
        ltm = {
            "material_trajectory": {},
            "pawn_structure": {},
            "piece_activity": {},
            "repetition": {},
            "historical_patterns": {},
            "strategic_events": [],
        }
        add_v04_memory_signals(
            long_term_memory=ltm,
            material_history=history,
            repetition_counts=zero_reps,
            check_counts=zero_counts,
            capture_counts=zero_counts,
            castle_counts=zero_counts,
            promotion_counts=zero_counts,
            board=board,
        )
        signals = ltm.get("v04_explicit_signals")
        if not isinstance(signals, dict):
            raise RuntimeError("Runtime self-test: v04_explicit_signals was not created")

    print("[SELF-TEST] Runtime feature paths: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate/extend the v0.4-ready chess policy dataset with deficit-aware "
            "Elo balancing and endgame/checkmate enrichment."
        )
    )
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--elo-min", type=int, default=1200)
    parser.add_argument("--elo-max", type=int, default=2000)
    parser.add_argument("--elo-step", type=int, default=100)
    parser.add_argument(
        "--balance-mode",
        type=str,
        choices=("deficit", "uniform"),
        default="deficit",
        help="Elo bucket allocation mode: 'deficit' water-fills under-represented buckets (default); 'uniform' splits equally.",
    )
    parser.add_argument(
        "--max-elo-diff",
        type=int,
        default=150,
        help="Maximum rating gap between White and Black players (default: 150).",
    )
    parser.add_argument(
        "--min-plies",
        type=int,
        default=24,
        help="Minimum game length in plies to exclude early aborts/resignations (default: 24).",
    )
    parser.add_argument(
        "--allow-time-forfeit",
        action="store_true",
        help="Allow Time forfeit / non-Normal terminations (by default only Normal terminations are selected).",
    )
    parser.add_argument("--novelty-pool-factor", type=int, default=6)
    parser.add_argument("--novelty-probe-depth", type=int, default=5)
    parser.add_argument("--novelty-probe-positions", type=int, default=5)
    parser.add_argument("--novelty-workers", type=int, default=1)
    parser.add_argument("--novelty-threads-per-engine", type=int, default=1)
    parser.add_argument("--novelty-hash-mb", type=int, default=64)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-engine", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--fsync-every-game", action="store_true")
    parser.add_argument("--status-interval", type=int, default=30)
    parser.add_argument("--allow-bots", action="store_true")
    parser.add_argument("--no-auto-upgrade-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--batch-pause-sec", type=int, default=20)
    parser.add_argument("--scan-log-interval-games", type=int, default=10000)
    parser.add_argument("--scan-log-interval-seconds", type=int, default=15)
    parser.add_argument("--probe-log-every", type=int, default=10)
    parser.add_argument("--selection-log-every", type=int, default=25)
    parser.add_argument("--force-recompute-novelty", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.elo_min < 0 or args.elo_max < args.elo_min:
        raise ValueError("Invalid Elo range: require 0 <= --elo-min <= --elo-max")
    if args.elo_step <= 0:
        raise ValueError("--elo-step must be positive")
    if not args.pgn.exists():
        raise FileNotFoundError(f"PGN not found: {args.pgn}")

    run_runtime_self_test()
    stockfish_path = find_stockfish(args.stockfish)

    manifest_path = args.output.with_suffix(".manifest.json")
    validation_path = args.output.with_suffix(".validation.json")
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    failures_path = args.output.with_suffix(".failures.jsonl")
    progress_path = args.output.with_suffix(".progress.jsonl")
    status_path = args.output.with_suffix(".status.json")

    print("=" * 72)
    print("CHESS DATASET GENERATOR v0.4.3 — DEFICIT-BALANCED ELO & ENDGAME BUILD")
    print("=" * 72)
    print(f"PGN                 : {args.pgn}")
    print(f"Output              : {args.output}")
    print(f"Target games        : {args.games}")
    print(f"Stockfish           : {stockfish_path}")
    print(f"Depth               : {args.depth}")
    print(f"Elo policy          : {args.elo_min}-{args.elo_max} step {args.elo_step} (mode={args.balance_mode})")
    print(f"Game quality filter : max_elo_diff={args.max_elo_diff}, min_plies={args.min_plies}, normal_termination_only={not args.allow_time_forfeit}")
    print(f"Novelty pool factor : {args.novelty_pool_factor}x per active Elo bucket")
    print(f"Novelty probe       : depth {args.novelty_probe_depth}, positions {args.novelty_probe_positions}, workers {args.novelty_workers}")
    print(f"Workers             : {args.workers} (threads/engine={args.threads_per_engine}, hash={args.hash_mb}MB)")
    print(f"Schema              : {SCHEMA_VERSION} / {FEATURE_SCHEMA_VERSION}")

    existing_ids: set[str] = set()
    existing_schema: Optional[str] = None
    existing_feature_schema: Optional[str] = None
    existing_records = 0
    if args.output.exists():
        (
            existing_ids,
            existing_schema,
            existing_feature_schema,
            existing_records,
        ) = read_existing_dataset_inventory(args.output)
        print(f"Existing games      : {len(existing_ids):,}")
        print(f"Existing records    : {existing_records:,}")
        print(f"Existing schema     : {existing_schema}/{existing_feature_schema}")
        if existing_schema not in (SCHEMA_VERSION, "2.0"):
            raise RuntimeError(
                f"Unsupported existing dataset schema {existing_schema!r}."
            )
        if len(existing_ids) > args.games:
            raise RuntimeError(
                f"Existing dataset already contains {len(existing_ids)} games, "
                f"which is greater than requested --games {args.games}."
            )
    else:
        raise RuntimeError(
            "This extension generator expects an existing dataset at --output."
        )

    # ------------------------------------------------------------
    # Existing dataset novelty index.
    # ------------------------------------------------------------
    index_config = {
        "cache_version": CACHE_VERSION,
        "cache_type": "existing_dataset_index",
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_fingerprint": _path_fingerprint(args.output),
        "records": existing_records,
        "games": len(existing_ids),
        "game_ids_digest": _ids_digest(existing_ids),
    }
    index_key = _cache_key(index_config)
    index_cache_path = Path(str(args.output) + ".novelty_index.pkl")

    existing_index = None
    if not args.force_recompute_novelty:
        existing_index = _load_pickle_cache(index_cache_path, index_key)

    if existing_index is not None:
        print(
            f"[CACHE] Reusing existing novelty index: games={len(existing_index['game_ids']):,} "
            f"records={existing_index['records']:,}",
            flush=True,
        )
    else:
        print("\nBuilding existing-dataset novelty index (read-only pass)...")
        existing_index = build_existing_pattern_index(
            args.output,
            log_interval_records=max(1, args.scan_log_interval_games),
        )
        _save_pickle_cache(index_cache_path, index_key, existing_index)
        print(f"[CACHE] Existing novelty index saved: {index_cache_path}", flush=True)

    # ------------------------------------------------------------
    # Novelty selection manifest/cache.
    # ------------------------------------------------------------
    manifest: Optional[Dict[str, Any]] = None
    if manifest_path.exists() and partial_path.exists():
        try:
            candidate_manifest = load_manifest(manifest_path)
            if manifest_matches_run(candidate_manifest, args):
                manifest = candidate_manifest
                print("Reusing matching manifest for resume.")
            else:
                print(
                    "Existing manifest does not match current arguments; it will not be reused.",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] Could not reuse existing manifest: {exc}", flush=True)

    if manifest is None and partial_path.exists():
        stale_partial = partial_path.with_name(
            partial_path.name + f".stale-{int(time.time())}.bak"
        )
        shutil.move(partial_path, stale_partial)
        print(f"[CACHE] Preserved stale partial dataset: {stale_partial}", flush=True)

    if manifest is None:
        print("\nSelecting novel games with Deficit-Aware Elo balancing...")
        new_games, novelty_report, scanned = select_novel_games_covering_existing(
            pgn_path=args.pgn,
            target_games=args.games,
            elo_min=args.elo_min,
            elo_max=args.elo_max,
            elo_step=args.elo_step,
            balance_mode=args.balance_mode,
            max_elo_diff=args.max_elo_diff,
            min_plies=args.min_plies,
            allow_time_forfeit=args.allow_time_forfeit,
            require_human_players=not args.allow_bots,
            required_existing_ids=existing_ids,
            existing_index=existing_index,
            novelty_pool_factor=args.novelty_pool_factor,
            novelty_probe_depth=args.novelty_probe_depth,
            novelty_probe_positions=args.novelty_probe_positions,
            novelty_workers=args.novelty_workers,
            novelty_hash_mb=args.novelty_hash_mb,
            novelty_threads_per_engine=args.novelty_threads_per_engine,
            novelty_stockfish_path=stockfish_path,
            output_path=args.output,
            existing_records=existing_records,
            force_recompute_novelty=args.force_recompute_novelty,
            scan_log_interval_games=args.scan_log_interval_games,
            scan_log_interval_seconds=args.scan_log_interval_seconds,
            probe_log_every=args.probe_log_every,
            selection_log_every=args.selection_log_every,
        )
        print(f"PGN games scanned   : {scanned:,}", flush=True)
        print(f"Novel games selected: {len(new_games):,}", flush=True)

        existing_entries = []
        for gid in sorted(existing_ids):
            meta = (existing_index.get("game_meta") or {}).get(gid)
            if not meta:
                raise RuntimeError(f"Existing game metadata missing from novelty index: {gid}")
            expected_records = safe_int(
                (existing_index.get("game_record_counts") or {}).get(gid),
                0,
            )
            if expected_records <= 0:
                raise RuntimeError(f"Existing game has no records in novelty index: {gid}")
            existing_entries.append({
                **meta,
                "total_plies": expected_records,
                "expected_records": expected_records,
            })

        new_entries = []
        for game in new_games:
            total_plies = len(game_move_list(game))
            new_entries.append({
                "game_id": game_id_from_headers(game.headers),
                "white": sanitize_header_value(game.headers.get("White")),
                "black": sanitize_header_value(game.headers.get("Black")),
                "white_elo": sanitize_header_value(game.headers.get("WhiteElo")),
                "black_elo": sanitize_header_value(game.headers.get("BlackElo")),
                "result": sanitize_header_value(game.headers.get("Result")),
                "total_plies": total_plies,
                "expected_records": total_plies,
            })

        game_entries = existing_entries + new_entries
        if len(game_entries) != args.games:
            raise RuntimeError(f"Manifest game count mismatch: {len(game_entries)} != {args.games}")

        manifest = build_manifest_from_entries(args, stockfish_path, game_entries)
        manifest["novelty_selection"] = novelty_report
        manifest["existing_game_count"] = len(existing_ids)
        manifest["new_game_count"] = len(new_games)
        manifest["novelty_cache_version"] = CACHE_VERSION

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    selected_game_ids = {entry["game_id"] for entry in manifest.get("games", [])}
    if len(selected_game_ids) != args.games:
        raise RuntimeError("Manifest does not contain exactly the requested number of unique games.")

    expected_records_by_game = {
        entry["game_id"]: safe_int(entry.get("expected_records"), -1)
        for entry in manifest.get("games", [])
    }

    if args.output.exists() and not partial_path.exists():
        copy_final_to_partial(args.output, partial_path)
        print(f"Seeded partial dataset from existing final output: {partial_path}")

    # Initialize game_map BEFORE the repair block (fixes UnboundLocalError!)
    game_map: Dict[str, chess.pgn.Game] = {}

    if partial_path.exists() and not args.no_auto_upgrade_existing:
        bad_game_ids = find_partial_games_needing_upgrade(partial_path)
        immutable_bad = bad_game_ids & existing_ids
        if immutable_bad:
            raise RuntimeError(
                "Existing final dataset is immutable for extension runs. "
                "Structurally incomplete existing-game records found: "
                + ", ".join(sorted(immutable_bad)[:10])
            )

        repair_new_ids = bad_game_ids - existing_ids
        if repair_new_ids:
            print(
                f"New/partial games requiring deterministic schema upgrade: {len(repair_new_ids):,}",
                flush=True,
            )
            repair_map = recover_games_by_ids_from_pgn(args.pgn, repair_new_ids)
            repaired_ids, stockfish_repair_ids, repaired_records = upgrade_partial_dataset(
                partial_path=partial_path,
                selected_games=list(repair_map.values()),
                default_depth=args.depth,
                default_nodes=args.nodes,
            )
            print(f"Deterministically upgraded games : {len(repaired_ids):,}", flush=True)
            print(f"Records rewritten                : {repaired_records:,}", flush=True)
            print(f"Games needing Stockfish repair   : {len(stockfish_repair_ids):,}", flush=True)

    completed_ids = load_completed_game_ids(
        partial_path,
        expected_records_by_game=expected_records_by_game,
    )
    completed_ids &= selected_game_ids

    print(f"Completed games     : {len(completed_ids):,}/{args.games:,}")

    if len(completed_ids) == args.games:
        print("All requested games are already present and complete.")
        print("Running final validation...")
        report = run_final_validation(partial_path, validation_path, manifest_path)
        if report["validation_error_count"]:
            raise RuntimeError(
                f"Final validation failed with {report['validation_error_count']} errors. "
                f"See {validation_path}"
            )
        os.replace(partial_path, args.output)
        print(f"Published/updated final dataset: {args.output}")
        print(f"Validation report   : {validation_path}")
        return

    unfinished_new_ids = sorted((selected_game_ids - existing_ids) - completed_ids)

    # Recover selected PGN games from the novelty-selected cache first
    selected_cache_path = Path(str(args.output) + ".novelty_selected.pkl")
    if selected_cache_path.exists() and not args.force_recompute_novelty:
        try:
            with selected_cache_path.open("rb") as f:
                env = pickle.load(f)
            payload_data = env.get("payload") or {}
            for sel_item in payload_data.get("selected", []):
                gid = sel_item.get("game_id")
                if gid and gid in unfinished_new_ids:
                    game_map[gid] = reconstruct_game(sel_item["pgn"])
            if game_map:
                print(
                    f"[CACHE] Recovered {len(game_map):,}/{len(unfinished_new_ids):,} "
                    "unfinished new games from novelty-selection cache.",
                    flush=True,
                )
        except Exception:
            pass

    missing_new_ids = set(unfinished_new_ids) - set(game_map)
    if missing_new_ids:
        print(f"Recovering {len(missing_new_ids):,} missing selected games from PGN...", flush=True)
        game_map.update(recover_games_by_ids_from_pgn(args.pgn, missing_new_ids))

    unfinished_ids = sorted(selected_game_ids - completed_ids)
    if any(gid in existing_ids for gid in unfinished_ids):
        raise RuntimeError(
            "Invariant violation: an existing final-dataset game is marked unfinished."
        )

    payloads = [prepare_game_for_worker(game_map[gid]) for gid in unfinished_ids]
    print(f"Games to process     : {len(payloads):,}")
    print("Existing completed games will NOT be sent to Stockfish again.", flush=True)

    tracker = ProgressTracker(
        total_games=args.games,
        resume_completed=len(completed_ids),
        progress_path=progress_path,
        status_path=status_path,
        interval_seconds=args.status_interval,
    )
    tracker.start()

    executor: Optional[ProcessPoolExecutor] = None
    interrupted = False

    try:
        try:
            import multiprocessing as mp
            mp_context = mp.get_context("fork")
        except Exception:
            mp_context = None

        executor_kwargs: Dict[str, Any] = {
            "max_workers": args.workers,
            "initializer": init_worker,
            "initargs": (
                stockfish_path,
                args.depth,
                args.nodes,
                args.hash_mb,
                args.threads_per_engine,
            ),
        }
        if mp_context is not None:
            executor_kwargs["mp_context"] = mp_context

        executor = ProcessPoolExecutor(**executor_kwargs)
        active_futures: Dict[Any, str] = {}

        for batch_start in range(0, len(payloads), args.batch_size):
            batch = payloads[batch_start:batch_start + args.batch_size]
            active_futures = {
                executor.submit(process_game, payload): payload["game_id"]
                for payload in batch
            }
            game_started = {gid: time.time() for gid in active_futures.values()}

            for future in as_completed(active_futures):
                gid = active_futures[future]
                duration = time.time() - game_started[gid]

                try:
                    result = future.result()
                    if result.get("error"):
                        stage_name = result.get("stage", "unknown_stage")
                        error_text = result.get("error")
                        ply_info = result.get("ply")
                        fen_info = result.get("fen")
                        tb = result.get("traceback", "")
                        raise RuntimeError(
                            f"Worker failed at stage={stage_name}, ply={ply_info}, "
                            f"fen={fen_info}: {error_text}\n{tb}"
                        )

                    records = result["records"]
                    expected_count = safe_int(result.get("total_plies"), -1)

                    if expected_count <= 0 or len(records) != expected_count:
                        raise RuntimeError(
                            f"Game {gid} generated {len(records)} records, "
                            f"but expected {expected_count}."
                        )

                    validation_errors: List[str] = []
                    plies = []
                    for rec in records:
                        validation_errors.extend(validate_record(rec))
                        plies.append(rec.get("position", {}).get("ply"))

                    if plies != list(range(1, expected_count + 1)):
                        validation_errors.append(
                            f"ply sequence mismatch: expected 1..{expected_count}"
                        )

                    if validation_errors:
                        raise RuntimeError(
                            f"Validation failed for {gid}: "
                            + "; ".join(validation_errors[:20])
                        )

                    append_records(
                        partial_path,
                        records,
                        do_fsync=args.fsync_every_game,
                    )

                    tracker.record_success(gid, duration, len(records))
                    snapshot = tracker.snapshot("game_success")
                    print(
                        f"[{snapshot['progress']['completed_games']}/{args.games}] "
                        f"SUCCESS {gid} | {len(records)} moves | "
                        f"game={duration:.1f}s | "
                        f"avg={snapshot['average_time_per_successful_game_sec']:.1f}s | "
                        f"CPU={snapshot['cpu_utilization_percent']}% | "
                        f"progress={snapshot['progress']['percent']:.2f}%"
                    )

                except Exception as exc:
                    with failures_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "timestamp": iso_now(),
                                    "game_id": gid,
                                    "error": repr(exc),
                                    "traceback": traceback.format_exc(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    tracker.record_failure(gid, duration)
                    snapshot = tracker.snapshot("game_failure")
                    print(
                        f"[FAILED] {gid} | game={duration:.1f}s | "
                        f"failures={snapshot['failures_current_run']} | "
                        f"error={exc}",
                        file=sys.stderr,
                    )

            has_more_batches = batch_start + args.batch_size < len(payloads)
            if has_more_batches and args.batch_pause_sec:
                print(
                    f"[PAUSE] Batch of {len(batch)} games completed. "
                    f"Sleeping {args.batch_pause_sec}s..."
                )
                time.sleep(args.batch_pause_sec)

    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[INTERRUPTED] Stopping workers and preserving partial dataset...",
            file=sys.stderr,
        )
        _safe_shutdown_process_pool(executor)
        tracker.stop("interrupted")
        print(f"Partial dataset preserved: {partial_path}")
        print("Rerun the same command to resume automatically.")
        return

    finally:
        if executor is not None and not interrupted:
            print(
                "[ENGINE] Full-generation worker pool finished; closing without blocking wait...",
                flush=True,
            )
            _safe_shutdown_process_pool(executor)
        if not interrupted:
            tracker.stop("generation_finished")

    final_completed = (
        load_completed_game_ids(
            partial_path,
            expected_records_by_game=expected_records_by_game,
        )
        & selected_game_ids
    )
    print(f"\nCompleted games in partial: {len(final_completed):,}/{args.games:,}")

    if len(final_completed) != args.games:
        print("\nGeneration is incomplete. Partial file preserved:")
        print(partial_path)
        print("Rerun the same command to retry only unfinished games.")
        return

    print("\nRunning final dataset validation...")
    report = run_final_validation(
        partial_path,
        validation_path,
        manifest_path,
    )

    if report["validation_error_count"]:
        print(
            f"Final validation FAILED with {report['validation_error_count']} errors. "
            f"See {validation_path}"
        )
        print(f"Partial file preserved: {partial_path}")
        tracker.write_checkpoint("final_validation_failed")
        return

    if args.output.exists():
        backup_path = args.output.with_suffix(args.output.suffix + ".previous.bak")
        shutil.copyfile(args.output, backup_path)
        print(f"Previous dataset backup: {backup_path}")
    os.replace(partial_path, args.output)
    tracker.write_checkpoint("final_dataset_published")

    print("\n=== DATASET V2.1 / v0.4-READY COMPLETE ===")
    print(f"Games              : {report['unique_games']:,}")
    print(f"Position records   : {report['records']:,}")
    print(f"Candidate counts   : {report['candidate_count_distribution']}")
    print(f"Output             : {args.output}")
    print(f"Manifest           : {manifest_path}")
    print(f"Validation report  : {validation_path}")
    print(f"Progress log       : {progress_path}")
    print(f"Latest status      : {status_path}")
    print(f"Failures log       : {failures_path}")


if __name__ == "__main__":
    main()
