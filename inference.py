#!/usr/bin/env python3
"""
CHESS POLICY MODEL v0.4 — STANDALONE COLAB & LOCAL INFERENCE + FULL-WIDTH WEB ARENA (SCRIPT 5/5)

Features:
  1. Complete v0.4 feature engineering (20x64 board, 18 global, 32 candidate,
     16 history, 40 perspective-corrected memory, 12 position, 5 flags, 6 geometry)
     + Exchange-Aware Tactical Refinement (Problem #3 Fix).
  2. Dual-Stream FiLM + Legacy v0.4 Elo Checkpoint Auto-Detection.
  3. Elo-Stratified Multi-Tier Reasoning Engine:
     - Easy   : 0-ply raw intuition + soft human-like move sampling
     - Medium : 1-ply tactical guard + controlled opening variety
     - Hard   : 2-ply tactical minimax lookahead + sharp candidate selection
     - Expert : Deep 2-ply minimax lookahead + endgame passed-pawn push + master book variety
  4. Interactive input() selector at startup: 1 = Google Colab (default), 2 = Local PC.
  5. Full-width responsive Web Chess Arena with Royal Blue primary theme and
     Dark / Light theme toggle.
  6. AI Move Suggestion ("Ask AI Suggestion") at Master 2200-Elo depth with
     Approve / Reject confirmation and automatic next-best move cycling on rejection.
  7. Interactive PGN Move Scrubber & Saved Game Reviewer + Detailed chess-rule
     termination explanations + separated `human_vs_model` and `model_vs_model` folders.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urlparse


# ============================================================
# 1. DEPENDENCY BOOTSTRAP
# ============================================================

def ensure_package(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


ensure_package("chess", "python-chess>=1.999")
ensure_package("cloudpickle", "cloudpickle>=3.0")
ensure_package("orjson", "orjson>=3.10")
ensure_package("matplotlib", "matplotlib>=3.8")

import chess
import cloudpickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from google.colab import drive as colab_drive  # type: ignore
    from google.colab import output as colab_output  # type: ignore
    COLAB_AVAILABLE = True
except Exception:
    colab_drive = None
    colab_output = None
    COLAB_AVAILABLE = False


# ============================================================
# 2. CONSTANTS & v0.4 FEATURE DIMENSIONS
# ============================================================

MODEL_VERSION = "0.4"
FEATURE_VERSION = "v0.4"
TRAINING_SAMPLE_SCHEMA = "0.4-training-sample"
MAX_HISTORY_PLIES = 50
BOARD_ROW_ZERO_IS_RANK_8 = True

ELO_MIN = 1200.0
ELO_MAX = 2000.0
ELO_STEP = 100.0
EXPECTED_ELO_BUCKETS = [
    "<1200", "1200", "1300", "1400", "1500", "1600",
    "1700", "1800", "1900", "2000", ">2000", "unknown",
]
ELO_BUCKET_TO_ID = {bucket: idx for idx, bucket in enumerate(EXPECTED_ELO_BUCKETS)}
ELO_ID_TO_BUCKET = {idx: bucket for bucket, idx in ELO_BUCKET_TO_ID.items()}
ELO_CONDITION_CENTER = (ELO_MIN + ELO_MAX) / 2.0
ELO_CONDITION_SCALE = max(ELO_STEP, (ELO_MAX - ELO_MIN) / 2.0)
ELO_NORMALIZED_CLAMP = 4.0
DEFAULT_TARGET_ELO = 1600.0
AI_COACH_SUGGESTION_ELO = 2200.0
MODEL_MOVE_DELAY_SECONDS = 0.8
GAME_EXPIRY_SECONDS = 24 * 60 * 60
GAME_STATUSES = ("COMPLETED", "EXPIRED", "RUNNING", "RESIGNED")

DIFFICULTY_OFFSETS = {
    "Easy": -200.0,
    "Medium": 0.0,
    "Hard": +200.0,
    "Expert": +350.0,
}

FEATURE_DIMS = {
    "board_tactical": 20,
    "global_tactical": 18,
    "candidate_tactical": 32,
    "history": 16,
    "memory": 40,
    "position": 12,
    "candidate_move_flags": 5,
    "candidate_geometry": 6,
}

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

MEMORY_PIECE_VALUES = {
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

SOLID_PIECE_GLYPHS = {
    chess.PAWN: "♟",
    chess.KNIGHT: "♞",
    chess.BISHOP: "♝",
    chess.ROOK: "♜",
    chess.QUEEN: "♛",
    chess.KING: "♚",
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)


def map_difficulty_to_model_elo(human_elo: float, difficulty: str) -> float:
    diff_key = difficulty.strip().title() if difficulty else "Medium"
    if diff_key == "Expert":
        return float(max(2000.0, min(2400.0, float(human_elo) + 350.0)))
    offset = DIFFICULTY_OFFSETS.get(diff_key, 0.0)
    return float(max(1000.0, min(2400.0, float(human_elo) + offset)))


def infer_difficulty_from_elo(target_elo: float, explicit_difficulty: Optional[str] = None) -> str:
    if explicit_difficulty and explicit_difficulty.strip().title() in DIFFICULTY_OFFSETS:
        return explicit_difficulty.strip().title()
    val = float(target_elo)
    if val < 1480.0:
        return "Easy"
    if val < 1720.0:
        return "Medium"
    if val < 1950.0:
        return "Hard"
    return "Expert"


def describe_game_termination(board: chess.Board) -> Dict[str, Any]:
    """Returns a detailed, human-readable chess rule conclusion when the game ends."""
    if not board.is_game_over(claim_draw=True):
        return {
            "is_over": False,
            "result": "*",
            "winner": None,
            "reason_code": "in_play",
            "title": "Game in Play",
            "explanation": "The game is currently active.",
        }

    result = board.result(claim_draw=True)
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        loser = "Black" if winner == "White" else "White"
        return {
            "is_over": True,
            "result": result,
            "winner": winner.lower(),
            "reason_code": "checkmate",
            "title": f"Checkmate — {winner} Wins ({result})",
            "explanation": f"{loser}'s king is in check and has no legal escape squares.",
        }

    if board.is_stalemate():
        stm = "White" if board.turn == chess.WHITE else "Black"
        return {
            "is_over": True,
            "result": "1/2-1/2",
            "winner": "draw",
            "reason_code": "stalemate",
            "title": "Draw by Stalemate (1/2-1/2)",
            "explanation": f"{stm} to move is not in check but has no legal moves available.",
        }

    if board.is_fivefold_repetition():
        return {
            "is_over": True,
            "result": "1/2-1/2",
            "winner": "draw",
            "reason_code": "fivefold_repetition",
            "title": "Draw by Fivefold Repetition (1/2-1/2)",
            "explanation": "The exact same board position has occurred five times.",
        }

    if board.can_claim_threefold_repetition():
        return {
            "is_over": True,
            "result": "1/2-1/2",
            "winner": "draw",
            "reason_code": "threefold_repetition",
            "title": "Draw by Threefold Repetition (1/2-1/2)",
            "explanation": "The exact same position with the same side to move and rights has been repeated three times.",
        }

    if board.is_seventyfive_moves():
        return {
            "is_over": True,
            "result": "1/2-1/2",
            "winner": "draw",
            "reason_code": "seventyfive_moves",
            "title": "Draw by 75-Move Rule (1/2-1/2)",
            "explanation": "75 consecutive full moves have been played without a pawn advance or piece capture.",
        }

    if board.can_claim_fifty_moves():
        return {
            "is_over": True,
            "result": "1/2-1/2",
            "winner": "draw",
            "reason_code": "fifty_moves",
            "title": "Draw by 50-Move Rule (1/2-1/2)",
            "explanation": "50 consecutive full moves have been played without a pawn advance or piece capture.",
        }

    if board.is_insufficient_material():
        return {
            "is_over": True,
            "result": "1/2-1/2",
            "winner": "draw",
            "reason_code": "insufficient_material",
            "title": "Draw by Insufficient Material (1/2-1/2)",
            "explanation": "Neither player has sufficient mating material left on the board to force checkmate.",
        }

    return {
        "is_over": True,
        "result": result,
        "winner": "draw",
        "reason_code": "draw",
        "title": f"Game Over ({result})",
        "explanation": "The game concluded by standard chess rules.",
    }


# ============================================================
# 3. ENVIRONMENT SELECTION (1 = COLAB DEFAULT, 2 = LOCAL PC)
# ============================================================

def find_local_checkpoint_candidates() -> List[Path]:
    search_roots = [
        Path.cwd(),
        Path.cwd() / "checkpoints",
        Path.cwd() / "chess" / "model_v0_4" / "checkpoints",
        Path("/content/chess/model_v0_4/checkpoints"),
        Path("/content/drive/MyDrive/chess/model_v0_4/checkpoints"),
        Path.home() / "Downloads",
    ]
    preferred_names = [
        "model_v0_4_elo_final.pt",
        "model_v0_4_elo_best.pt",
        "model_v0_4_elo_last.pt",
        "model_v0_4_initial.pt",
    ]
    found: List[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        if not root.exists():
            continue
        for name in preferred_names:
            candidate = root / name
            if candidate.exists() and candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(candidate)
        for pt_file in sorted(root.glob("*.pt")):
            if pt_file.is_file():
                resolved = pt_file.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(pt_file)
    return found


def print_required_drive_downloads() -> None:
    print("\n" + "=" * 74)
    print("REQUIRED FILES TO DOWNLOAD FROM GOOGLE DRIVE FOR LOCAL EXECUTION")
    print("=" * 74)
    print("Please download at least item #1 from your Google Drive folder:")
    print("  Drive Root: MyDrive/chess/model_v0_4/")
    print("")
    print("  1. Trained Model Checkpoint (REQUIRED):")
    print("     - MyDrive/chess/model_v0_4/checkpoints/model_v0_4_elo_final.pt")
    print("       (or model_v0_4_elo_best.pt)")
    print("  2. Elo Conditioning & Feature Specs (OPTIONAL - embedded in checkpoint):")
    print("     - MyDrive/chess/model_v0_4/reports/elo_conditioning_spec_v04.json")
    print("     - MyDrive/chess/model_v0_4/reports/v04_feature_spec.json")
    print("=" * 74 + "\n")


def configure_runtime_environment() -> Tuple[str, Path, Path, Path]:
    print("=" * 74)
    print("CHESS POLICY MODEL v0.4 — RUNTIME ENVIRONMENT SETUP")
    print("=" * 74)
    raw_mode = input(
        "Select execution environment [1 = Google Colab (default), 2 = Local PC]: "
    ).strip()
    mode = "2" if raw_mode == "2" else "1"

    if mode == "1":
        if COLAB_AVAILABLE and colab_drive is not None:
            try:
                colab_drive.mount("/content/drive", force_remount=False)
            except Exception as exc:
                print(f"Drive mount note: {exc}")

        drive_root = Path(os.environ.get("CHESS_DATA_ROOT", "/content/drive/MyDrive/chess"))
        if not drive_root.exists() and Path("./chess").exists():
            drive_root = Path("./chess").resolve()

        v04_root = drive_root / "model_v0_4"
        ckpt_dir = v04_root / "checkpoints"
        inference_dir = v04_root / "inference"
        local_cache_dir = Path("/content/chess/model_v0_4/checkpoints")
        inference_dir.mkdir(parents=True, exist_ok=True)
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            ckpt_dir / "model_v0_4_elo_final.pt",
            ckpt_dir / "model_v0_4_elo_best.pt",
            ckpt_dir / "model_v0_4_elo_last.pt",
            ckpt_dir / "model_v0_4_initial.pt",
        ]
        selected_drive_ckpt = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)

        if selected_drive_ckpt is None:
            print(f"\nNo checkpoint automatically found in {ckpt_dir}.")
            print_required_drive_downloads()
            custom_ckpt = input("Enter full path to your v0.4 .pt checkpoint file: ").strip()
            if not custom_ckpt or not Path(custom_ckpt).expanduser().exists():
                raise FileNotFoundError(f"Checkpoint not found: {custom_ckpt!r}")
            selected_drive_ckpt = Path(custom_ckpt).expanduser().resolve()

        local_ckpt = local_cache_dir / selected_drive_ckpt.name
        if (
            selected_drive_ckpt.resolve() != local_ckpt.resolve()
            and (not local_ckpt.exists() or local_ckpt.stat().st_size != selected_drive_ckpt.stat().st_size)
        ):
            print(f"Caching checkpoint {selected_drive_ckpt.name} -> {local_ckpt} ...")
            shutil.copy2(selected_drive_ckpt, local_ckpt)
        else:
            local_ckpt = selected_drive_ckpt if not local_ckpt.exists() else local_ckpt

        bundle_path = inference_dir / "v04_runtime_bundle.pkl"
        return "colab", local_ckpt, inference_dir, bundle_path

    # Mode 2: Local PC
    print("\nScanning local machine for v0.4 model checkpoints...")
    discovered = find_local_checkpoint_candidates()
    if discovered:
        print("Found local checkpoint candidate(s):")
        for idx, path in enumerate(discovered, start=1):
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f"  [{idx}] {path} ({size_mb:.1f} MB)")
        default_ckpt = discovered[0]
        prompt_text = (
            f"Enter checkpoint number [1-{len(discovered)}] or custom file path "
            f"[default = {default_ckpt}]: "
        )
    else:
        print_required_drive_downloads()
        default_ckpt = None
        prompt_text = "Enter path to downloaded v0.4 checkpoint (.pt): "

    while True:
        user_ckpt_input = input(prompt_text).strip()
        if not user_ckpt_input and default_ckpt is not None:
            chosen_ckpt = default_ckpt
            break
        if user_ckpt_input.isdigit() and 1 <= int(user_ckpt_input) <= len(discovered):
            chosen_ckpt = discovered[int(user_ckpt_input) - 1]
            break
        candidate_path = Path(user_ckpt_input).expanduser()
        if candidate_path.is_dir():
            dir_pts = list(candidate_path.glob("*.pt"))
            if dir_pts:
                chosen_ckpt = dir_pts[0]
                break
        if candidate_path.exists() and candidate_path.is_file():
            chosen_ckpt = candidate_path.resolve()
            break
        print(f"File not found: {user_ckpt_input!r}")
        print_required_drive_downloads()

    default_log_dir = Path.cwd() / "inference"
    user_log_input = input(
        f"Enter directory path to store inference logs & boards [default = {default_log_dir}]: "
    ).strip()
    inference_dir = Path(user_log_input).expanduser().resolve() if user_log_input else default_log_dir.resolve()
    inference_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = inference_dir / "v04_runtime_bundle.pkl"
    return "local", chosen_ckpt, inference_dir, bundle_path


# ============================================================
# 4. COMPLETE v0.4 FEATURE ENGINEERING (EXACT SCRIPT 2/3/4 ALIGNMENT)
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def side_to_id(side: str) -> int:
    return 0 if side == "white" else 1


def piece_value(piece: Optional[chess.Piece]) -> float:
    return 0.0 if piece is None else PIECE_VALUES[piece.piece_type]


def controlled_material(board: chess.Board, color: chess.Color) -> float:
    occ = board.occupied_co[color]
    return (
        1.0 * chess.popcount(board.pawns & occ)
        + 3.0 * chess.popcount((board.knights | board.bishops) & occ)
        + 5.0 * chess.popcount(board.rooks & occ)
        + 9.0 * chess.popcount(board.queens & occ)
    )


def count_legal_moves_for_color(board: chess.Board, color: chess.Color) -> int:
    if board.turn == color:
        return board.legal_moves.count()
    temp = board.copy(stack=False)
    temp.turn = color
    return temp.legal_moves.count()


def center_control_count(board: chess.Board, color: chess.Color) -> int:
    return sum(chess.popcount(board.attackers_mask(color, sq)) for sq in CENTER_SQUARES)


def king_ring_pressure(board: chess.Board, king_square: Optional[int], attacker: chess.Color) -> int:
    if king_square is None:
        return 0
    ring_mask = chess.BB_KING_ATTACKS[king_square]
    pressure = 0
    for sq in chess.scan_forward(ring_mask):
        pressure += chess.popcount(board.attackers_mask(attacker, sq))
    return pressure


def hanging_non_king(board: chess.Board, color: chess.Color) -> int:
    non_king_mask = board.occupied_co[color] & ~board.kings
    opp = not color
    count = 0
    for square in chess.scan_forward(non_king_mask):
        if board.attackers_mask(opp, square) and not board.attackers_mask(color, square):
            count += 1
    return count


def pinned_non_king(board: chess.Board, color: chess.Color) -> int:
    non_king_mask = board.occupied_co[color] & ~board.kings
    count = 0
    for square in chess.scan_forward(non_king_mask):
        if board.is_pinned(color, square):
            count += 1
    return count


def compute_board_tactical_features(board: chess.Board) -> List[List[float]]:
    """v0.4 20-dimensional per-square tactical features."""
    turn = board.turn
    is_white_turn = (turn == chess.WHITE)
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    pawns_bb = board.pawns
    piece_map = board.piece_map()

    legal_from_count = [0] * 64
    for move in board.legal_moves:
        legal_from_count[move.from_square] += 1

    features: List[List[float]] = []
    for square in range(64):
        piece = piece_map.get(square)
        w_mask = board.attackers_mask(chess.WHITE, square)
        b_mask = board.attackers_mask(chess.BLACK, square)

        white_attack_count = chess.popcount(w_mask)
        black_attack_count = chess.popcount(b_mask)
        current_attack_count = white_attack_count if is_white_turn else black_attack_count
        opponent_attack_count = black_attack_count if is_white_turn else white_attack_count

        pinned_white = 0.0
        pinned_black = 0.0
        signed_piece = 0.0
        is_current_side_piece = 0.0
        is_opponent_piece = 0.0
        current_side_defended = 0.0
        opponent_side_defended = 0.0
        if piece is not None:
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                signed_piece = val / 9.0
                pinned_white = float(board.is_pinned(chess.WHITE, square))
            else:
                signed_piece = -val / 9.0
                pinned_black = float(board.is_pinned(chess.BLACK, square))
            if piece.color == turn:
                is_current_side_piece = 1.0
                current_side_defended = float(current_attack_count > 0)
            else:
                is_opponent_piece = 1.0
                opponent_side_defended = float(opponent_attack_count > 0)

        current_side_attacked = float(opponent_attack_count > 0 and is_current_side_piece > 0.5)
        current_square_attacked = float(current_attack_count > 0)

        w_pawn_attacks = chess.popcount(w_mask & pawns_bb)
        b_pawn_attacks = chess.popcount(b_mask & pawns_bb)

        dist_wk = 8.0 if white_king is None else float(chess.square_distance(square, white_king))
        dist_bk = 8.0 if black_king is None else float(chess.square_distance(square, black_king))

        features.append([
            signed_piece,
            white_attack_count / 8.0,
            black_attack_count / 8.0,
            current_attack_count / 8.0,
            opponent_attack_count / 8.0,
            pinned_white,
            pinned_black,
            min(legal_from_count[square], 32) / 32.0,
            dist_wk / 7.0,
            dist_bk / 7.0,
            float(dist_wk <= 1.0),
            float(dist_bk <= 1.0),
            is_current_side_piece,
            is_opponent_piece,
            current_side_attacked,
            current_square_attacked,
            w_pawn_attacks / 2.0,
            b_pawn_attacks / 2.0,
            current_side_defended,
            opponent_side_defended,
        ])
    return features


def compute_global_tactical_features(board: chess.Board) -> List[float]:
    """v0.4 18-dimensional global tactical vector."""
    current = board.turn
    opponent = not current
    current_king = board.king(current)
    opponent_king = board.king(opponent)
    white_material = controlled_material(board, chess.WHITE)
    black_material = controlled_material(board, chess.BLACK)
    piece_count = chess.popcount(board.occupied)
    pawn_count = chess.popcount(board.pawns)
    legal_moves = board.legal_moves.count()
    current_attackers_on_king = (
        chess.popcount(board.attackers_mask(opponent, current_king))
        if current_king is not None else 0
    )
    opponent_king_attackers = (
        chess.popcount(board.attackers_mask(current, opponent_king))
        if opponent_king is not None else 0
    )

    return [
        white_material / 39.0,
        black_material / 39.0,
        (white_material - black_material) / 39.0,
        min(legal_moves, 100) / 100.0,
        float(board.is_check()),
        min(current_attackers_on_king, 8) / 8.0,
        min(opponent_king_attackers, 8) / 8.0,
        piece_count / 32.0,
        pawn_count / 16.0,
        float(current == chess.WHITE),
        hanging_non_king(board, current) / 15.0,
        hanging_non_king(board, opponent) / 15.0,
        pinned_non_king(board, current) / 15.0,
        pinned_non_king(board, opponent) / 15.0,
        min(king_ring_pressure(board, current_king, opponent), 24) / 24.0,
        min(king_ring_pressure(board, opponent_king, current), 24) / 24.0,
        float(
            board.has_kingside_castling_rights(current)
            + board.has_queenside_castling_rights(current)
        ) / 2.0,
        float(
            board.has_kingside_castling_rights(opponent)
            + board.has_queenside_castling_rights(opponent)
        ) / 2.0,
    ]


def precompute_board_context(board: chess.Board) -> Dict[str, Any]:
    current = board.turn
    opponent = not current
    current_king_before = board.king(current)
    opponent_king_before = board.king(opponent)

    own_legal_before = board.legal_moves.count()
    opponent_legal_before = count_legal_moves_for_color(board, opponent)
    current_center_before = center_control_count(board, current)

    enemy_king_attackers_before = (
        chess.popcount(board.attackers_mask(current, opponent_king_before))
        if opponent_king_before is not None else 0
    )
    own_king_attackers_before = (
        chess.popcount(board.attackers_mask(opponent, current_king_before))
        if current_king_before is not None else 0
    )
    material_before = controlled_material(board, current) - controlled_material(board, opponent)

    return {
        "current_king_before": current_king_before,
        "opponent_king_before": opponent_king_before,
        "own_legal_before": own_legal_before,
        "opponent_legal_before": opponent_legal_before,
        "current_center_before": current_center_before,
        "enemy_king_attackers_before": enemy_king_attackers_before,
        "own_king_attackers_before": own_king_attackers_before,
        "material_before": material_before,
    }


def compute_candidate_tactical_features(
    board: chess.Board,
    move: chess.Move,
    ctx: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """v0.4 32-dimensional candidate consequence tactical features."""
    if ctx is None:
        ctx = precompute_board_context(board)

    current = board.turn
    opponent = not current
    moving_piece = board.piece_at(move.from_square)

    is_ep = bool(board.is_en_passant(move))
    if is_ep:
        captured_square = move.to_square - 8 if current == chess.WHITE else move.to_square + 8
        captured_piece = board.piece_at(captured_square)
        is_capture = True
    else:
        captured_piece = board.piece_at(move.to_square)
        is_capture = captured_piece is not None

    is_castling = bool(board.is_castling(move))
    is_promotion = move.promotion is not None
    gives_check = bool(board.gives_check(move))

    from_current_attackers = chess.popcount(board.attackers_mask(current, move.from_square))
    from_opponent_attackers = chess.popcount(board.attackers_mask(opponent, move.from_square))
    to_current_attackers = chess.popcount(board.attackers_mask(current, move.to_square))
    to_opponent_attackers = chess.popcount(board.attackers_mask(opponent, move.to_square))
    pinned = float(board.is_pinned(current, move.from_square))

    own_legal_before = ctx["own_legal_before"]
    opponent_legal_before = ctx["opponent_legal_before"]
    current_center_before = ctx["current_center_before"]
    enemy_king_attackers_before = ctx["enemy_king_attackers_before"]
    own_king_attackers_before = ctx["own_king_attackers_before"]
    material_before = ctx["material_before"]

    is_king_move = moving_piece is not None and moving_piece.piece_type == chess.KING
    after_opponent_king = ctx.get("opponent_king_before", board.king(opponent))

    board.push(move)
    try:
        after_current_king = move.to_square if is_king_move else ctx.get("current_king_before", board.king(current))
        to_current_attackers_after = chess.popcount(board.attackers_mask(current, move.to_square))
        to_opponent_attackers_after = chess.popcount(board.attackers_mask(opponent, move.to_square))
        opponent_legal_after = board.legal_moves.count()
        own_legal_after = count_legal_moves_for_color(board, current)
        enemy_king_attackers_after = (
            chess.popcount(board.attackers_mask(current, after_opponent_king))
            if after_opponent_king is not None else 0
        )
        own_king_attackers_after = (
            chess.popcount(board.attackers_mask(opponent, after_current_king))
            if after_current_king is not None else 0
        )
        current_center_after = center_control_count(board, current)
        material_after = controlled_material(board, current) - controlled_material(board, opponent)
        current_king_pressure_after = king_ring_pressure(board, after_current_king, opponent)
        opponent_king_pressure_after = king_ring_pressure(board, after_opponent_king, current)
    finally:
        board.pop()

    material_delta = material_after - material_before
    own_mobility_delta = own_legal_after - own_legal_before
    opponent_mobility_delta = opponent_legal_after - opponent_legal_before
    enemy_king_pressure_delta = enemy_king_attackers_after - enemy_king_attackers_before
    own_king_pressure_delta = own_king_attackers_after - own_king_attackers_before
    destination_defense_balance = to_current_attackers_after - to_opponent_attackers_after
    center_control_delta = current_center_after - current_center_before

    return [
        from_current_attackers / 8.0,
        from_opponent_attackers / 8.0,
        to_current_attackers / 8.0,
        to_opponent_attackers / 8.0,
        to_current_attackers_after / 8.0,
        to_opponent_attackers_after / 8.0,
        pinned,
        float(gives_check),
        float(is_capture),
        float(is_castling),
        float(is_ep),
        float(is_promotion),
        piece_value(captured_piece) / 9.0,
        piece_value(moving_piece) / 9.0,
        (PIECE_VALUES.get(move.promotion, 0.0) if move.promotion else 0.0) / 9.0,
        min(opponent_legal_after, 100) / 100.0,
        enemy_king_attackers_before / 8.0,
        enemy_king_attackers_after / 8.0,
        own_king_attackers_before / 8.0,
        own_king_attackers_after / 8.0,
        enemy_king_pressure_delta / 8.0,
        own_king_pressure_delta / 8.0,
        clamp(material_delta / 9.0, -1.0, 1.0),
        own_legal_before / 100.0,
        own_legal_after / 100.0,
        opponent_legal_before / 100.0,
        own_mobility_delta / 50.0,
        opponent_mobility_delta / 50.0,
        min(current_king_pressure_after, 24) / 24.0,
        min(opponent_king_pressure_after, 24) / 24.0,
        clamp(destination_defense_balance / 8.0, -1.0, 1.0),
        clamp(center_control_delta / 4.0, -1.0, 1.0),
    ]


def candidate_geometry(from_square: int, to_square: int) -> List[float]:
    from_file, from_rank = from_square & 7, from_square >> 3
    to_file, to_rank = to_square & 7, to_square >> 3
    dx, dy = to_file - from_file, to_rank - from_rank
    abs_dx, abs_dy = abs(dx), abs(dy)
    cheb = max(abs_dx, abs_dy)
    manhattan = abs_dx + abs_dy
    return [dx / 7.0, dy / 7.0, abs_dx / 7.0, abs_dy / 7.0, cheb / 7.0, manhattan / 14.0]


def history_move_to_vector(move: Dict[str, Any]) -> List[float]:
    """v0.4 16-dimensional history move vector."""
    features = move.get("move_features") or {}
    uci = move.get("uci")
    if not isinstance(uci, str):
        raise ValueError("History move missing uci")
    parsed = chess.Move.from_uci(uci)

    captured_value = float(features.get("captured_piece_value", 0) or 0) / 900.0
    moved_value = float(features.get("moved_piece_value", 0) or 0) / 900.0
    promotion_value = float(features.get("promotion_piece_value", 0) or 0) / 900.0
    distance = float(features.get("move_distance", 0) or 0) / 7.0
    king_distance = float(features.get("destination_to_opponent_king_distance", 8) or 8) / 8.0
    events = float(features.get("strategic_event_count", len(move.get("strategic_events", []))) or 0) / 8.0

    return [
        parsed.from_square / 63.0,
        parsed.to_square / 63.0,
        float(move.get("side_id", 0 if move.get("side") == "white" else 1)),
        float(bool(features.get("is_capture", False))),
        float(bool(features.get("is_check", False))),
        float(bool(features.get("is_castling", features.get("is_castle", False)))),
        float(bool(features.get("is_en_passant", False))),
        float(bool(features.get("is_promotion", False))),
        clamp(captured_value, 0.0, 1.0),
        clamp(moved_value, 0.0, 1.0),
        clamp(promotion_value, 0.0, 1.0),
        clamp(distance, 0.0, 1.0),
        float(bool(features.get("center_activity", False))),
        clamp(king_distance, 0.0, 1.0),
        float(bool(features.get("is_check_after", features.get("is_check", False)))),
        clamp(events, 0.0, 1.0),
    ]


def position_to_vector(position_state: Dict[str, Any]) -> List[float]:
    """v0.4 12-dimensional position state vector."""
    if not isinstance(position_state, dict):
        position_state = {}
    rights = position_state.get("castling_rights") or {}
    ep = position_state.get("en_passant_square")
    ep_index = 0.0
    ep_exists = 0.0
    if isinstance(ep, str) and len(ep) == 2:
        try:
            ep_index = chess.parse_square(ep) / 63.0
            ep_exists = 1.0
        except Exception:
            pass
    return [
        clamp(float(position_state.get("legal_move_count", 0) or 0) / 100.0, 0.0, 1.0),
        float(bool(position_state.get("is_check", False))),
        float(bool(position_state.get("is_checkmate", False))),
        float(bool(position_state.get("is_stalemate", False))),
        clamp(float(position_state.get("halfmove_clock", 0) or 0) / 100.0, 0.0, 1.0),
        clamp(float(position_state.get("fullmove_number", 1) or 1) / 150.0, 0.0, 1.0),
        float(bool(rights.get("white_kingside", False))),
        float(bool(rights.get("white_queenside", False))),
        float(bool(rights.get("black_kingside", False))),
        float(bool(rights.get("black_queenside", False))),
        ep_exists,
        clamp(ep_index, 0.0, 1.0),
    ]


_BUCKET_CACHE: Dict[Tuple[str, int], int] = {}


def stable_bucket(text: str, modulo: int) -> int:
    key = (text, modulo)
    cached = _BUCKET_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.md5(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], byteorder="little", signed=False) % modulo
    if len(_BUCKET_CACHE) < 16384:
        _BUCKET_CACHE[key] = value
    return value


def memory_scalar(value: Any, key_path: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not finite_number(value):
        return 0.0
    number = float(value)
    path_lower = key_path.lower()
    if "past_plies" in path_lower or "move_number" in path_lower:
        number /= 100.0
    elif "count" in path_lower or "moves" in path_lower:
        number /= 100.0
    number = math.copysign(math.log1p(abs(number)), number)
    number /= 6.0
    return clamp(number, -1.0, 1.0)


def memory_to_vector32(memory: Dict[str, Any]) -> List[float]:
    vector = [0.0] * 32
    counts = [0.0] * 32

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = f"{path}.{key}" if path else str(key)
                visit(item, key_path)
            return
        if isinstance(value, list):
            bucket = stable_bucket(path + ".length", 32)
            vector[bucket] += clamp(math.log1p(len(value)) / 6.0, -1.0, 1.0)
            counts[bucket] += 1.0
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, bool):
            bucket = stable_bucket(path, 32)
            vector[bucket] += 1.0 if value else 0.0
            counts[bucket] += 1.0
        elif isinstance(value, (int, float)):
            bucket = stable_bucket(path, 32)
            vector[bucket] += memory_scalar(value, path)
            counts[bucket] += 1.0

    if isinstance(memory, dict):
        visit(memory, "")
    for index in range(32):
        if counts[index] > 1.0:
            vector[index] /= math.sqrt(counts[index])
        vector[index] = clamp(vector[index], -4.0, 4.0)
    return vector


def memory_to_vector40(memory: Dict[str, Any], side_to_move: str = "white") -> List[float]:
    """
    v0.4 40-dimensional memory vector (32 hashed + 8 explicit v0.4 signals).
    Fix 1 Applied: `last_10_ply_material_change_cp` is oriented to `side_to_move`
    just like `material_advantage_side_to_move_cp`.
    """
    hashed = memory_to_vector32(memory)
    signals = memory.get("v04_explicit_signals", {}) if isinstance(memory, dict) else {}
    sign = 1.0 if side_to_move == "white" else -1.0
    explicit = [
        float(signals.get("material_advantage_side_to_move_cp", 0) or 0) / 3900.0,
        (sign * float(signals.get("last_10_ply_material_change_cp", 0) or 0)) / 3900.0,
        clamp(float(signals.get("current_position_repetition_count", 1) or 1) / 3.0, 0.0, 1.0),
        float(bool(signals.get("threefold_occurred", False))),
        clamp(float(signals.get("check_count_difference", 0) or 0) / 10.0, -1.0, 1.0),
        clamp(float(signals.get("capture_count_difference", 0) or 0) / 10.0, -1.0, 1.0),
        clamp(float(signals.get("castle_count_difference", 0) or 0) / 4.0, -1.0, 1.0),
        clamp(float(signals.get("promotion_count_difference", 0) or 0) / 4.0, -1.0, 1.0),
    ]
    return hashed + explicit


def target_elo_bucket_label(target_elo: float) -> str:
    value = float(target_elo)
    if not math.isfinite(value):
        return "1800"
    if value < ELO_MIN:
        return "<1200"
    if value > ELO_MAX:
        return ">2000"
    bucket_steps = int(math.floor((value - ELO_MIN) / ELO_STEP + 0.5))
    bucket_value = max(ELO_MIN, min(ELO_MAX, ELO_MIN + bucket_steps * ELO_STEP))
    return str(int(round(bucket_value)))


# ============================================================
# 5. LONG-TERM MEMORY & RUNTIME RECORD BUILDERS (v0.4)
# ============================================================

def material_difference(board: chess.Board) -> int:
    white = 0
    black = 0
    for piece_type, value in MEMORY_PIECE_VALUES.items():
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
        if not any(files.get(af, 0) > 0 for af in adjacent_files):
            isolated += 1

        enemy = not color
        passed = True
        for enemy_sq in board.pieces(chess.PAWN, enemy):
            ef = chess.square_file(enemy_sq)
            er = chess.square_rank(enemy_sq)
            if abs(ef - f) <= 1:
                if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                    passed = False
                    break
        if passed:
            passed_squares.append(sq)

    passed_files = {chess.square_file(sq) for sq in passed_squares}
    connected_passed = sum(
        1 for sq in passed_squares
        if (chess.square_file(sq) - 1 in passed_files or chess.square_file(sq) + 1 in passed_files)
    )
    return {
        "isolated_pawns": isolated,
        "doubled_pawns": doubled,
        "passed_pawns": len(passed_squares),
        "connected_passed_pawns": connected_passed,
    }


def piece_activity_summary(
    board: chess.Board,
    move_counts_by_piece: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        counts = move_counts_by_piece[color_name]
        minors_undeveloped = 0
        start_squares = {
            (chess.B1 if color == chess.WHITE else chess.B8): chess.KNIGHT,
            (chess.G1 if color == chess.WHITE else chess.G8): chess.KNIGHT,
            (chess.C1 if color == chess.WHITE else chess.C8): chess.BISHOP,
            (chess.F1 if color == chess.WHITE else chess.F8): chess.BISHOP,
        }
        for sq, piece_type in start_squares.items():
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type == piece_type:
                if counts.get(PIECE_NAMES[piece_type], 0) == 0:
                    minors_undeveloped += 1

        result[color_name] = {
            "undeveloped_minors": minors_undeveloped,
            "king_moved": counts.get("king", 0) > 0,
            "queen_moves": counts.get("queen", 0),
            "rook_moves": counts.get("rook", 0),
            "minor_piece_moves": counts.get("knight", 0) + counts.get("bishop", 0),
        }
    return result


def board_key(board: chess.Board) -> Any:
    return board._transposition_key()


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
    largest_swing = max((abs(x - initial_material) for x in material_history), default=0)
    last_10 = material_history[-10:] if material_history else [current_material]
    last_10_swing = current_material - last_10[0]

    current_key = board_key(board)
    current_rep = repetition_counts.get(current_key, 1)
    total_repetitions = sum(1 for n in repetition_counts.values() if n >= 3)

    stm_is_white = (board.turn == chess.WHITE)
    stm_name = "white" if stm_is_white else "black"
    opp_name = "black" if stm_is_white else "white"
    mat_adv_stm = current_material if stm_is_white else -current_material

    return {
        "material_trajectory": {
            "initial_difference_white_cp": initial_material,
            "current_difference_white_cp": current_material,
            "current_advantage_side_to_move_cp": mat_adv_stm,
            "largest_swing_from_initial_cp": largest_swing,
            "last_10_ply_change_cp": last_10_swing,
            "past_plies": max(0, ply - 1),
            "position_count_in_history": len(material_history),
        },
        "pawn_structure": {
            "white": count_pawn_features(board, chess.WHITE),
            "black": count_pawn_features(board, chess.BLACK),
        },
        "piece_activity": piece_activity_summary(board, move_counts_by_piece),
        "repetition": {
            "current_position_count": current_rep,
            "threefold_occurred": current_rep >= 3,
            "repetition_events": total_repetitions,
        },
        "historical_patterns": {
            "checks": {"white": check_counts["white"], "black": check_counts["black"]},
            "captures": {"white": capture_counts["white"], "black": capture_counts["black"]},
            "castles": {"white": castle_counts["white"], "black": castle_counts["black"]},
            "promotions": {"white": promotion_counts["white"], "black": promotion_counts["black"]},
        },
        "v04_explicit_signals": {
            "material_advantage_side_to_move_cp": mat_adv_stm,
            "last_10_ply_material_change_cp": last_10_swing,
            "current_position_repetition_count": current_rep,
            "threefold_occurred": bool(current_rep >= 3 or total_repetitions > 0),
            "check_count_difference": check_counts[stm_name] - check_counts[opp_name],
            "capture_count_difference": capture_counts[stm_name] - capture_counts[opp_name],
            "castle_count_difference": castle_counts[stm_name] - castle_counts[opp_name],
            "promotion_count_difference": promotion_counts[stm_name] - promotion_counts[opp_name],
            "initial_material_difference_white_cp": initial_material,
        },
        "strategic_events": strategic_events[-1000:],
    }


def history_move_from_board(board: chess.Board, move: chess.Move) -> Dict[str, Any]:
    """Populates all v0.4 history move fields."""
    moving_piece = board.piece_at(move.from_square)
    if board.is_en_passant(move):
        captured_piece = board.piece_at(
            move.to_square - 8 if board.turn == chess.WHITE else move.to_square + 8
        )
    else:
        captured_piece = board.piece_at(move.to_square)

    moved_piece_type = int(moving_piece.piece_type if moving_piece else 0)
    captured_piece_type = int(captured_piece.piece_type if captured_piece else 0)
    promotion_piece_type = int(move.promotion or 0)

    moved_val_cp = MEMORY_PIECE_VALUES.get(moved_piece_type, 0)
    captured_val_cp = MEMORY_PIECE_VALUES.get(captured_piece_type, 0)
    promo_val_cp = MEMORY_PIECE_VALUES.get(promotion_piece_type, 0)

    move_dist = chess.square_distance(move.from_square, move.to_square)
    opp_king_sq = board.king(not board.turn)
    dest_to_opp_king = chess.square_distance(move.to_square, opp_king_sq) if opp_king_sq is not None else 8
    center_act = bool(move.to_square in CENTER_SQUARES or move.from_square in CENTER_SQUARES)

    is_cap = bool(board.is_capture(move))
    gives_chk = bool(board.gives_check(move))
    is_castle = bool(board.is_castling(move))
    is_ep = bool(board.is_en_passant(move))
    is_promo = bool(move.promotion is not None)
    side_str = "white" if board.turn == chess.WHITE else "black"

    strategic_events: List[Dict[str, Any]] = []
    if is_cap:
        strategic_events.append({"type": "CAPTURE"})
    if gives_chk:
        strategic_events.append({"type": "CHECK"})
    if is_castle:
        strategic_events.append({"type": "CASTLE"})
    if is_promo:
        strategic_events.append({"type": "PROMOTION"})
    if center_act:
        strategic_events.append({"type": "CENTER_ACTIVITY"})

    mf = {
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "is_capture": is_cap,
        "is_check": gives_chk,
        "is_check_after": gives_chk,
        "is_castle": is_castle,
        "is_castling": is_castle,
        "is_en_passant": is_ep,
        "is_promotion": is_promo,
        "captured_piece": PIECE_NAMES.get(captured_piece.piece_type) if captured_piece else None,
        "captured_piece_type": captured_piece_type,
        "captured_piece_value": captured_val_cp,
        "moved_piece": PIECE_NAMES.get(moving_piece.piece_type) if moving_piece else None,
        "moved_piece_type": moved_piece_type,
        "moved_piece_value": moved_val_cp,
        "promotion": PROMOTION_NAMES.get(move.promotion) if move.promotion else None,
        "promotion_piece": promotion_piece_type,
        "promotion_piece_value": promo_val_cp,
        "move_distance": move_dist,
        "center_activity": center_act,
        "destination_to_opponent_king_distance": dest_to_opp_king,
        "strategic_event_count": len(strategic_events),
    }

    return {
        "uci": move.uci(),
        "san": board.san(move),
        "side": side_str,
        "side_id": 0 if side_str == "white" else 1,
        "move_features": mf,
        "strategic_events": strategic_events,
    }


def board_to_8x8(board: chess.Board) -> List[List[int]]:
    out: List[List[int]] = []
    for rank in range(7, -1, -1):
        row: List[int] = []
        for file_index in range(8):
            piece = board.piece_at(chess.square(file_index, rank))
            if piece is None:
                row.append(0)
            else:
                row.append(piece.piece_type if piece.color == chess.WHITE else -piece.piece_type)
        out.append(row)
    return out


def make_position_state(board: chess.Board) -> Dict[str, Any]:
    return {
        "castling_rights": {
            "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
            "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
            "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
            "black_queenside": board.has_queenside_castling_rights(chess.BLACK),
        },
        "en_passant_square": chess.square_name(board.ep_square) if board.ep_square is not None else None,
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "legal_move_count": board.legal_moves.count(),
    }


def determine_phase(board: chess.Board) -> str:
    queens = chess.popcount(board.queens)
    rooks = chess.popcount(board.rooks)
    minors = chess.popcount(board.bishops | board.knights)
    pawns = chess.popcount(board.pawns)
    if queens >= 1 and pawns >= 8:
        return "opening"
    if queens >= 1 and (rooks + minors) >= 5:
        return "middlegame"
    if pawns <= 8 or queens == 0 or rooks <= 1:
        return "endgame"
    return "middlegame"


def runtime_candidate(
    move: chess.Move,
    board: chess.Board,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    moving_piece = board.piece_at(move.from_square)
    is_ep = bool(board.is_en_passant(move))
    if is_ep:
        captured_piece = board.piece_at(
            move.to_square - 8 if board.turn == chess.WHITE else move.to_square + 8
        )
        is_cap = True
    else:
        captured_piece = board.piece_at(move.to_square)
        is_cap = captured_piece is not None

    is_chk = bool(board.gives_check(move))
    is_castle = bool(board.is_castling(move))
    is_promo = bool(move.promotion is not None)

    promotion_piece = int(move.promotion or 0)
    captured_piece_type = int(captured_piece.piece_type if captured_piece else 0)
    moved_piece_type = int(moving_piece.piece_type if moving_piece else 0)

    flags = [float(is_cap), float(is_chk), float(is_castle), float(is_ep), float(is_promo)]

    return {
        "uci": move.uci(),
        "san": board.san(move),
        "from_square": int(move.from_square),
        "to_square": int(move.to_square),
        "promotion_piece": promotion_piece,
        "captured_piece_type": captured_piece_type,
        "moved_piece_type": moved_piece_type,
        "move_features": {
            "is_capture": is_cap,
            "is_check": is_chk,
            "is_castling": is_castle,
            "is_castle": is_castle,
            "is_en_passant": is_ep,
            "is_promotion": is_promo,
            "captured_piece": PIECE_NAMES.get(captured_piece.piece_type) if captured_piece else None,
            "moved_piece": PIECE_NAMES.get(moving_piece.piece_type) if moving_piece else None,
            "promotion": PROMOTION_NAMES.get(move.promotion) if move.promotion else None,
        },
        "move_flags": flags,
        "geometry": candidate_geometry(move.from_square, move.to_square),
        "tactical_features": compute_candidate_tactical_features(board, move, ctx=ctx),
    }


def make_runtime_record(
    board: chess.Board,
    history_moves: List[Dict[str, Any]],
    memory: Dict[str, Any],
    target_elo: float = DEFAULT_TARGET_ELO,
) -> Dict[str, Any]:
    controlled_side = "white" if board.turn == chess.WHITE else "black"
    ctx = precompute_board_context(board)
    candidates = [
        runtime_candidate(move, board, ctx=ctx)
        for move in sorted(board.legal_moves, key=lambda m: m.uci())
    ]
    if not candidates:
        raise RuntimeError("Position has no legal moves.")

    state = make_position_state(board)
    history_vectors = [
        history_move_to_vector(m)
        for m in history_moves[-MAX_HISTORY_PLIES:]
    ]
    labels = [{"relative_to_best": 0.0, "is_optimal": False} for _ in candidates]
    elo_bucket = target_elo_bucket_label(target_elo)

    return {
        "schema_version": TRAINING_SAMPLE_SCHEMA,
        "feature_schema_version": FEATURE_VERSION,
        "game_id": "live-inference",
        "ply": len(history_moves) + 1,
        "input": {
            "fen": board.fen(),
            "board_8x8": board_to_8x8(board),
            "side_to_move": controlled_side,
            "side_to_move_id": side_to_id(controlled_side),
            "controlled_side": controlled_side,
            "controlled_side_id": side_to_id(controlled_side),
            "position_state": state,
            "position_features": position_to_vector(state),
            "recent_history": {
                "max_plies": MAX_HISTORY_PLIES,
                "plies_available": len(history_moves),
                "vectors": history_vectors,
            },
            "history": history_vectors,
            "history_lengths": len(history_vectors),
            "memory_features": memory_to_vector40(memory, side_to_move=controlled_side),
            "global_tactical_features": compute_global_tactical_features(board),
            "board_tactical_features": compute_board_tactical_features(board),
            "phase": determine_phase(board),
            "candidate_moves": candidates,
        },
        "target": {
            "candidate_uci": [c["uci"] for c in candidates],
            "candidate_san": [c["san"] for c in candidates],
            "best_move": candidates[0]["uci"],
            "optimal_moves": [candidates[0]["uci"]],
            "candidate_labels": labels,
        },
        "metadata": {
            "game_weight": 1.0,
            "controlled_side_elo": float(target_elo),
            "elo": {
                "white_elo": float(target_elo),
                "black_elo": float(target_elo),
                "controlled_side_elo": float(target_elo),
                "opponent_elo": float(target_elo),
                "elo_difference": 0.0,
                "average_elo": float(target_elo),
                "elo_bucket": elo_bucket,
            },
        },
    }


# ============================================================
# 6. v0.4 COLLATE, EXCHANGE REFINEMENT & DUAL-STREAM FiLM MODELS
# ============================================================

def apply_exchange_aware_tactical_refinement(
    board_tactical: torch.Tensor,
    global_tactical: torch.Tensor,
    candidate_tactical: torch.Tensor,
    controlled_side: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Problem #3 Fix (matches Script 4 training): Upgrades naive undefended piece
    counts and candidate destination defense balance with outnumbered piece
    detection, enemy pawn threats on minor/major pieces, and Static Exchange (SEE)
    destination risk.
    """
    bt = board_tactical.clone()
    gt = global_tactical.clone()
    ct = candidate_tactical.clone()

    piece_mag = bt[:, :, 0].abs() * 9.0
    curr_att = (bt[:, :, 3] * 8.0).round()
    opp_att = (bt[:, :, 4] * 8.0).round()
    is_curr_piece = (bt[:, :, 12] > 0.5) & (piece_mag > 0.5)
    is_opp_piece = (bt[:, :, 13] > 0.5) & (piece_mag > 0.5)

    w_pawn_att = (bt[:, :, 16] * 2.0).round()
    b_pawn_att = (bt[:, :, 17] * 2.0).round()
    is_white_stm = (controlled_side == 0).unsqueeze(-1)

    opp_pawn_att = torch.where(is_white_stm, b_pawn_att, w_pawn_att)
    curr_pawn_att = torch.where(is_white_stm, w_pawn_att, b_pawn_att)

    curr_undefended = is_curr_piece & (opp_att >= 1.0) & (curr_att < 0.5)
    curr_pawn_threatened = is_curr_piece & (piece_mag >= 2.5) & (opp_pawn_att >= 1.0)
    curr_outnumbered = is_curr_piece & (opp_att > curr_att) & (curr_att >= 1.0)

    curr_vuln_score = torch.zeros_like(piece_mag)
    curr_vuln_score = torch.where(curr_outnumbered, torch.full_like(curr_vuln_score, 0.65), curr_vuln_score)
    curr_vuln_score = torch.where(curr_pawn_threatened, torch.full_like(curr_vuln_score, 0.85), curr_vuln_score)
    curr_vuln_score = torch.where(curr_undefended, torch.ones_like(curr_vuln_score), curr_vuln_score)

    opp_undefended = is_opp_piece & (curr_att >= 1.0) & (opp_att < 0.5)
    opp_pawn_threatened = is_opp_piece & (piece_mag >= 2.5) & (curr_pawn_att >= 1.0)
    opp_outnumbered = is_opp_piece & (curr_att > opp_att) & (opp_att >= 1.0)

    opp_vuln_score = torch.zeros_like(piece_mag)
    opp_vuln_score = torch.where(opp_outnumbered, torch.full_like(opp_vuln_score, 0.65), opp_vuln_score)
    opp_vuln_score = torch.where(opp_pawn_threatened, torch.full_like(opp_vuln_score, 0.85), opp_vuln_score)
    opp_vuln_score = torch.where(opp_undefended, torch.ones_like(opp_vuln_score), opp_vuln_score)

    bt[:, :, 14] = torch.maximum(bt[:, :, 14] * 0.35, curr_vuln_score)
    gt[:, 10] = (curr_vuln_score.sum(dim=-1) / 15.0).clamp(0.0, 1.0)
    gt[:, 11] = (opp_vuln_score.sum(dim=-1) / 15.0).clamp(0.0, 1.0)

    to_curr_after = ct[:, :, 4] * 8.0
    to_opp_after = ct[:, :, 5] * 8.0
    captured_val = ct[:, :, 12] * 9.0
    moving_val = ct[:, :, 13] * 9.0

    net_exposure = (moving_val - captured_val).clamp_min(0.0) / 9.0
    undefended_dest = (to_opp_after >= 0.9) & (to_curr_after < 0.5)
    outnumbered_dest = (to_opp_after > to_curr_after) & (to_curr_after >= 0.5)

    see_penalty = torch.zeros_like(ct[:, :, 30])
    see_penalty = torch.where(outnumbered_dest, 0.35 * net_exposure, see_penalty)
    see_penalty = torch.where(undefended_dest, 0.70 * (moving_val / 9.0), see_penalty)
    ct[:, :, 30] = (ct[:, :, 30] - see_penalty).clamp(-1.0, 1.0)

    return bt, gt, ct


def v04_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_candidates = max(len(sample["input"]["candidate_moves"]) for sample in batch)
    max_history = max(1, min(MAX_HISTORY_PLIES, max(len(sample["input"].get("history", [])) for sample in batch)))

    board_rows: List[List[int]] = []
    board_tactical_rows: List[Any] = []
    global_tactical_rows: List[Any] = []
    position_rows: List[Any] = []
    memory_rows: List[Any] = []
    controlled_side_rows: List[int] = []
    history_length_rows: List[int] = []
    target_elo_rows: List[float] = []
    target_elo_bucket_ids: List[int] = []

    history_tensor = torch.zeros((batch_size, max_history, FEATURE_DIMS["history"]), dtype=torch.float32)
    candidate_from = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    candidate_to = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    candidate_promotion = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    candidate_captured = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    candidate_moved = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    candidate_move_flags = torch.zeros((batch_size, max_candidates, FEATURE_DIMS["candidate_move_flags"]), dtype=torch.float32)
    candidate_geometry_t = torch.zeros((batch_size, max_candidates, FEATURE_DIMS["candidate_geometry"]), dtype=torch.float32)
    candidate_tactical = torch.zeros((batch_size, max_candidates, FEATURE_DIMS["candidate_tactical"]), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)

    for b_idx, sample in enumerate(batch):
        inp = sample["input"]
        meta = sample.get("metadata", {})
        board_8x8 = inp["board_8x8"]
        board_rows.append([int(v) + 6 for row in reversed(board_8x8) for v in row])
        board_tactical_rows.append(inp["board_tactical_features"])
        global_tactical_rows.append(inp["global_tactical_features"])
        position_rows.append(inp["position_features"])
        memory_rows.append(inp["memory_features"])
        controlled_side_rows.append(int(inp["controlled_side_id"]))

        t_elo = float((meta.get("elo") or {}).get("controlled_side_elo", meta.get("controlled_side_elo", DEFAULT_TARGET_ELO)))
        t_bucket = target_elo_bucket_label(t_elo)
        target_elo_rows.append(t_elo)
        target_elo_bucket_ids.append(ELO_BUCKET_TO_ID[t_bucket])

        h_vecs = inp.get("history", [])[-MAX_HISTORY_PLIES:]
        history_length_rows.append(len(h_vecs))
        if h_vecs:
            sliced = h_vecs[-max_history:]
            history_tensor[b_idx, :len(sliced)] = torch.tensor(sliced, dtype=torch.float32)

        cands = inp["candidate_moves"]
        ncands = len(cands)
        candidate_from[b_idx, :ncands] = torch.tensor([int(c["from_square"]) for c in cands], dtype=torch.long)
        candidate_to[b_idx, :ncands] = torch.tensor([int(c["to_square"]) for c in cands], dtype=torch.long)
        candidate_promotion[b_idx, :ncands] = torch.tensor([int(c.get("promotion_piece", 0) or 0) for c in cands], dtype=torch.long)
        candidate_captured[b_idx, :ncands] = torch.tensor([int(c.get("captured_piece_type", 0) or 0) for c in cands], dtype=torch.long)
        candidate_moved[b_idx, :ncands] = torch.tensor([int(c.get("moved_piece_type", 0) or 0) for c in cands], dtype=torch.long)
        candidate_move_flags[b_idx, :ncands] = torch.tensor([c["move_flags"] for c in cands], dtype=torch.float32)
        candidate_geometry_t[b_idx, :ncands] = torch.tensor([c["geometry"] for c in cands], dtype=torch.float32)
        candidate_tactical[b_idx, :ncands] = torch.tensor([c["tactical_features"] for c in cands], dtype=torch.float32)
        candidate_mask[b_idx, :ncands] = True

    return {
        "board": torch.tensor(board_rows, dtype=torch.long),
        "board_tactical": torch.tensor(board_tactical_rows, dtype=torch.float32),
        "global_tactical": torch.tensor(global_tactical_rows, dtype=torch.float32),
        "history": history_tensor,
        "history_lengths": torch.tensor(history_length_rows, dtype=torch.long),
        "position": torch.tensor(position_rows, dtype=torch.float32),
        "memory": torch.tensor(memory_rows, dtype=torch.float32),
        "controlled_side": torch.tensor(controlled_side_rows, dtype=torch.long),
        "target_elo": torch.tensor(target_elo_rows, dtype=torch.float32),
        "target_elo_bucket_id": torch.tensor(target_elo_bucket_ids, dtype=torch.long),
        "candidate_from": candidate_from,
        "candidate_to": candidate_to,
        "candidate_promotion": candidate_promotion,
        "candidate_captured": candidate_captured,
        "candidate_moved": candidate_moved,
        "candidate_move_flags": candidate_move_flags,
        "candidate_geometry": candidate_geometry_t,
        "candidate_tactical": candidate_tactical,
        "candidate_mask": candidate_mask,
    }


class SpatialBoardEncoderV04(nn.Module):
    def __init__(self, model_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, 48)
        self.square_embedding = nn.Embedding(64, 32)
        self.tactical_projection = nn.Sequential(
            nn.Linear(FEATURE_DIMS["board_tactical"], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_projection = nn.Linear(48 + 32 + 32, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, board: torch.Tensor, board_tactical: torch.Tensor):
        batch_size = board.shape[0]
        square_ids = torch.arange(64, device=board.device).unsqueeze(0).expand(batch_size, -1)
        tokens = torch.cat([
            self.piece_embedding(board),
            self.square_embedding(square_ids),
            self.tactical_projection(board_tactical),
        ], dim=-1)
        tokens = self.output_norm(self.transformer(self.input_projection(tokens)))
        return tokens, tokens.mean(dim=1)


class HistoryEncoderV04(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(FEATURE_DIMS["history"], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history: torch.Tensor, lengths: torch.Tensor):
        outputs, _ = self.gru(self.input_projection(history))
        result = torch.zeros((history.shape[0], outputs.shape[-1]), device=history.device, dtype=outputs.dtype)
        non_empty = lengths > 0
        if non_empty.any():
            result[non_empty] = outputs[non_empty, lengths[non_empty] - 1]
        return self.output_norm(result)


class PositionEncoderV04(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIMS["position"], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, position: torch.Tensor):
        return self.net(position)


class MemoryEncoderV04(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIMS["memory"], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, memory: torch.Tensor):
        return self.net(memory)


class GlobalTacticalEncoderV04(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIMS["global_tactical"], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor):
        return self.net(value)


class CandidateEncoderV04(nn.Module):
    def __init__(self, model_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.from_embedding = nn.Embedding(64, 32)
        self.to_embedding = nn.Embedding(64, 32)
        self.promotion_embedding = nn.Embedding(7, 8)
        self.captured_embedding = nn.Embedding(7, 8)
        self.moved_embedding = nn.Embedding(7, 8)
        self.move_flag_projection = nn.Sequential(
            nn.Linear(FEATURE_DIMS["candidate_move_flags"], 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(FEATURE_DIMS["candidate_geometry"], 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.tactical_projection = nn.Sequential(
            nn.Linear(FEATURE_DIMS["candidate_tactical"], 48),
            nn.LayerNorm(48),
            nn.GELU(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(424, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        board_tokens: torch.Tensor,
        candidate_from: torch.Tensor,
        candidate_to: torch.Tensor,
        candidate_promotion: torch.Tensor,
        candidate_captured: torch.Tensor,
        candidate_moved: torch.Tensor,
        candidate_move_flags: torch.Tensor,
        candidate_geometry: torch.Tensor,
        candidate_tactical: torch.Tensor,
    ):
        model_dim = board_tokens.shape[-1]
        from_board_token = board_tokens.gather(1, candidate_from.unsqueeze(-1).expand(-1, -1, model_dim))
        to_board_token = board_tokens.gather(1, candidate_to.unsqueeze(-1).expand(-1, -1, model_dim))
        representation = torch.cat([
            from_board_token,
            to_board_token,
            self.from_embedding(candidate_from),
            self.to_embedding(candidate_to),
            self.promotion_embedding(candidate_promotion.clamp(0, 6)),
            self.captured_embedding(candidate_captured.clamp(0, 6)),
            self.moved_embedding(candidate_moved.clamp(0, 6)),
            self.move_flag_projection(candidate_move_flags),
            self.geometry_projection(candidate_geometry),
            self.tactical_projection(candidate_tactical),
        ], dim=-1)
        return self.output_projection(representation)


class EloConditionEncoderV04(nn.Module):
    def __init__(self, bucket_count: int = len(EXPECTED_ELO_BUCKETS), output_dim: int = 32):
        super().__init__()
        self.scalar_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.bucket_embedding = nn.Embedding(bucket_count, 16)
        self.output_net = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, target_elo: torch.Tensor, bucket_id: torch.Tensor) -> torch.Tensor:
        normalized = (target_elo.float() - ELO_CONDITION_CENTER) / ELO_CONDITION_SCALE
        normalized = normalized.clamp(-ELO_NORMALIZED_CLAMP, ELO_NORMALIZED_CLAMP).unsqueeze(-1)
        scalar_features = self.scalar_net(normalized)
        bucket_features = self.bucket_embedding(bucket_id.clamp(0, len(EXPECTED_ELO_BUCKETS) - 1))
        return self.output_net(torch.cat([scalar_features, bucket_features], dim=-1))


class ChessPolicyRankerV04Elo(nn.Module):
    """
    Script 4 Elo-Conditioned v0.4 Policy Model with Exchange-Aware SEE Refinement.
    Supports both Dual-Stream FiLM (`use_film=True`) and legacy (`use_film=False`)
    v0.4 Elo checkpoints automatically.
    """
    def __init__(self, dropout: float = 0.10, use_film: bool = True):
        super().__init__()
        self.use_film = use_film
        self.board_encoder = SpatialBoardEncoderV04(model_dim=128, dropout=dropout)
        self.history_encoder = HistoryEncoderV04(hidden_dim=128, dropout=dropout)
        self.position_encoder = PositionEncoderV04(hidden_dim=64)
        self.memory_encoder = MemoryEncoderV04(hidden_dim=64)
        self.global_tactical_encoder = GlobalTacticalEncoderV04(hidden_dim=64)
        self.controlled_side_embedding = nn.Embedding(2, 16)
        self.elo_encoder = EloConditionEncoderV04(bucket_count=len(EXPECTED_ELO_BUCKETS), output_dim=32)
        self.context_projection = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 64 + 16 + 32, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_film:
            self.context_film = nn.Linear(32, 192 * 2)
            self.candidate_film = nn.Linear(32, 128 * 2)
        self.candidate_encoder = CandidateEncoderV04(model_dim=128, dropout=dropout)
        self.score_head = nn.Sequential(
            nn.Linear(128 + 192, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        board_tact, global_tact, cand_tact = apply_exchange_aware_tactical_refinement(
            batch["board_tactical"],
            batch["global_tactical"],
            batch["candidate_tactical"],
            batch["controlled_side"],
        )

        board_tokens, board_global = self.board_encoder(batch["board"], board_tact)
        history_vector = self.history_encoder(batch["history"], batch["history_lengths"])
        position_vector = self.position_encoder(batch["position"])
        memory_vector = self.memory_encoder(batch["memory"])
        global_tactical_vector = self.global_tactical_encoder(global_tact)
        controlled_side_vector = self.controlled_side_embedding(batch["controlled_side"])
        elo_vector = self.elo_encoder(batch["target_elo"], batch["target_elo_bucket_id"])

        context = self.context_projection(torch.cat([
            board_global,
            history_vector,
            position_vector,
            memory_vector,
            global_tactical_vector,
            controlled_side_vector,
            elo_vector,
        ], dim=-1))

        if self.use_film and hasattr(self, "context_film"):
            gamma_ctx, beta_ctx = self.context_film(elo_vector).chunk(2, dim=-1)
            context = context * (1.0 + torch.tanh(gamma_ctx)) + beta_ctx

        candidate_representation = self.candidate_encoder(
            board_tokens,
            batch["candidate_from"],
            batch["candidate_to"],
            batch["candidate_promotion"],
            batch["candidate_captured"],
            batch["candidate_moved"],
            batch["candidate_move_flags"],
            batch["candidate_geometry"],
            cand_tact,
        )

        if self.use_film and hasattr(self, "candidate_film"):
            gamma_cand, beta_cand = self.candidate_film(elo_vector).unsqueeze(1).chunk(2, dim=-1)
            candidate_representation = candidate_representation * (1.0 + torch.tanh(gamma_cand)) + beta_cand

        candidate_count = candidate_representation.shape[1]
        expanded_context = context.unsqueeze(1).expand(-1, candidate_count, -1)
        scores = self.score_head(torch.cat([candidate_representation, expanded_context], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~batch["candidate_mask"], torch.finfo(scores.dtype).min)
        return {"scores": scores, "candidate_mask": batch["candidate_mask"]}


class ChessPolicyRankerV04(nn.Module):
    """Script 3 Non-Elo v0.4 fallback model."""
    def __init__(self, dropout: float = 0.10):
        super().__init__()
        self.board_encoder = SpatialBoardEncoderV04(model_dim=128, dropout=dropout)
        self.history_encoder = HistoryEncoderV04(hidden_dim=128, dropout=dropout)
        self.position_encoder = PositionEncoderV04(hidden_dim=64)
        self.memory_encoder = MemoryEncoderV04(hidden_dim=64)
        self.global_tactical_encoder = GlobalTacticalEncoderV04(hidden_dim=64)
        self.controlled_side_embedding = nn.Embedding(2, 16)
        self.context_projection = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 64 + 16, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_encoder = CandidateEncoderV04(model_dim=128, dropout=dropout)
        self.score_head = nn.Sequential(
            nn.Linear(128 + 160, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        board_tact, global_tact, cand_tact = apply_exchange_aware_tactical_refinement(
            batch["board_tactical"],
            batch["global_tactical"],
            batch["candidate_tactical"],
            batch["controlled_side"],
        )

        board_tokens, board_global = self.board_encoder(batch["board"], board_tact)
        history_vector = self.history_encoder(batch["history"], batch["history_lengths"])
        position_vector = self.position_encoder(batch["position"])
        memory_vector = self.memory_encoder(batch["memory"])
        global_tactical_vector = self.global_tactical_encoder(global_tact)
        controlled_side_vector = self.controlled_side_embedding(batch["controlled_side"])

        context = self.context_projection(torch.cat([
            board_global,
            history_vector,
            position_vector,
            memory_vector,
            global_tactical_vector,
            controlled_side_vector,
        ], dim=-1))

        candidate_representation = self.candidate_encoder(
            board_tokens,
            batch["candidate_from"],
            batch["candidate_to"],
            batch["candidate_promotion"],
            batch["candidate_captured"],
            batch["candidate_moved"],
            batch["candidate_move_flags"],
            batch["candidate_geometry"],
            cand_tact,
        )
        candidate_count = candidate_representation.shape[1]
        expanded_context = context.unsqueeze(1).expand(-1, candidate_count, -1)
        scores = self.score_head(torch.cat([candidate_representation, expanded_context], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~batch["candidate_mask"], torch.finfo(scores.dtype).min)
        return {"scores": scores, "candidate_mask": batch["candidate_mask"]}


# ============================================================
# 7. COMPACT BOARD RENDERER (SOLID WHITE & BLACK PIECES)
# ============================================================

BOARD_LIGHT = "#EAEFF6"
BOARD_DARK = "#4B7399"
MOVE_BLUE = "#BFE3FF"
MOVE_BLUE_DARK = "#2563EB"
CHECK_RED = "#F8C4C4"
CHECK_RED_DARK = "#C62828"
BOARD_EDGE = "#1E293B"
TEXT_DARK = "#0F172A"

LIVE_BOARD_DISPLAY = None


def _captured_lines(logger: Optional["GameLogger"]) -> Tuple[str, str]:
    if logger is None:
        return "Captured by White: —", "Captured by Black: —"
    symbols = {"pawn": "♟", "knight": "♞", "bishop": "♝", "rook": "♜", "queen": "♛", "king": "♚"}

    def fmt(counter: Counter) -> str:
        if not counter:
            return "—"
        order = ["queen", "rook", "bishop", "knight", "pawn", "king"]
        parts = [f"{symbols[name]}×{counter[name]}" for name in order if counter.get(name, 0) > 0]
        return "  ".join(parts) if parts else "—"

    return (
        "Captured by White: " + fmt(logger.captures["white"]),
        "Captured by Black: " + fmt(logger.captures["black"]),
    )


def render_board_matplotlib(
    board: chess.Board,
    output_svg: Path,
    output_png: Optional[Path] = None,
    *,
    last_move: Optional[chess.Move] = None,
    move_number: int = 0,
    san: str = "",
    target_elo: float = DEFAULT_TARGET_ELO,
    captured_piece: Optional[str] = None,
    logger: Optional["GameLogger"] = None,
    inline: bool = True,
) -> str:
    global LIVE_BOARD_DISPLAY

    fig = plt.figure(figsize=(6.2, 6.8), dpi=100, facecolor="#F8FAFC")
    ax = fig.add_axes([0.09, 0.16, 0.82, 0.73])
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    for rank in range(8):
        for file_index in range(8):
            sq_color = BOARD_LIGHT if (rank + file_index) % 2 != 0 else BOARD_DARK
            ax.add_patch(
                Rectangle((file_index, rank), 1, 1, facecolor=sq_color, edgecolor=BOARD_EDGE, linewidth=0.6)
            )

    if last_move is not None:
        for sq, lw in ((last_move.from_square, 1.4), (last_move.to_square, 2.2)):
            fx, rr = chess.square_file(sq), chess.square_rank(sq)
            ax.add_patch(
                Rectangle(
                    (fx, rr), 1, 1,
                    facecolor=MOVE_BLUE,
                    edgecolor=MOVE_BLUE_DARK,
                    linewidth=lw,
                    alpha=0.82,
                )
            )

    if board.is_check():
        checked_king_sq = board.king(board.turn)
        if checked_king_sq is not None:
            fx, rr = chess.square_file(checked_king_sq), chess.square_rank(checked_king_sq)
            ax.add_patch(
                Rectangle(
                    (fx + 0.04, rr + 0.04), 0.92, 0.92,
                    facecolor=CHECK_RED,
                    edgecolor=CHECK_RED_DARK,
                    linewidth=2.4,
                    alpha=0.95,
                )
            )

    for square, piece in board.piece_map().items():
        fx, rr = chess.square_file(square), chess.square_rank(square)
        glyph = SOLID_PIECE_GLYPHS[piece.piece_type]
        if piece.color == chess.WHITE:
            fill_color = "#FFFFFF"
            stroke_color = "#0F172A"
            stroke_width = 2.2
        else:
            fill_color = "#141414"
            stroke_color = "#F8FAFC"
            stroke_width = 1.2

        if board.is_check() and piece.piece_type == chess.KING and piece.color == board.turn:
            stroke_color = CHECK_RED_DARK
            stroke_width = 2.6

        txt = ax.text(
            fx + 0.5,
            rr + 0.48,
            glyph,
            ha="center",
            va="center",
            fontsize=28,
            fontfamily="DejaVu Sans",
            color=fill_color,
            fontweight="bold",
            zorder=5,
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=stroke_width, foreground=stroke_color),
            path_effects.Normal(),
        ])

    for i, file_name in enumerate("abcdefgh"):
        ax.text(i + 0.5, -0.22, file_name, ha="center", va="center", fontsize=10, color=TEXT_DARK, fontweight="bold")
        ax.text(i + 0.5, 8.22, file_name, ha="center", va="center", fontsize=10, color=TEXT_DARK, fontweight="bold")
    for i, rank_name in enumerate("12345678"):
        ax.text(-0.22, i + 0.5, rank_name, ha="center", va="center", fontsize=10, color=TEXT_DARK, fontweight="bold")
        ax.text(8.22, i + 0.5, rank_name, ha="center", va="center", fontsize=10, color=TEXT_DARK, fontweight="bold")

    title_side = "White" if board.turn == chess.WHITE else "Black"
    fig.text(
        0.5, 0.955,
        f"CHESS POLICY MODEL v0.4 (Elo {int(round(target_elo))})",
        ha="center", va="center", fontsize=13, color=TEXT_DARK, fontweight="bold",
    )
    fig.text(
        0.5, 0.922,
        f"Move {move_number}  •  Turn: {title_side}" + (f"  •  Last: {san}" if san else ""),
        ha="center", va="center", fontsize=10, color="#334155",
    )

    cap_white, cap_black = _captured_lines(logger)
    fig.text(0.10, 0.102, cap_white, ha="left", va="center", fontsize=9.5, color=TEXT_DARK, fontweight="bold")
    fig.text(0.10, 0.070, cap_black, ha="left", va="center", fontsize=9.5, color=TEXT_DARK, fontweight="bold")

    term_info = describe_game_termination(board)
    status = term_info["title"] if term_info["is_over"] else (
        f"CHECK — {title_side} in check" if board.is_check() else "In play"
    )
    if captured_piece:
        status += f"  •  Captured: {captured_piece}"
    fig.text(
        0.10, 0.036, status,
        ha="left", va="center", fontsize=9.5,
        color=(CHECK_RED_DARK if board.is_check() else TEXT_DARK),
        fontweight="bold",
    )

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    if output_png is not None:
        fig.savefig(output_png, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)

    if inline:
        try:
            from IPython import get_ipython
            from IPython.display import SVG, display
            if get_ipython() is not None:
                svg_obj = SVG(filename=str(output_svg))
                if LIVE_BOARD_DISPLAY is None:
                    LIVE_BOARD_DISPLAY = display(svg_obj, display_id=True)
                else:
                    LIVE_BOARD_DISPLAY.update(svg_obj)
        except Exception:
            pass

    return str(output_svg)


def print_terminal_board(board: chess.Board, last_move: Optional[chess.Move] = None) -> None:
    light = (234, 239, 246)
    dark = (75, 115, 153)
    blue = (191, 227, 255)
    files = "  a   b   c   d   e   f   g   h"
    print("\n    " + files)
    print("   " + "─" * 34)
    for rank in range(7, -1, -1):
        row = [f" {rank + 1} │"]
        for file_index in range(8):
            square = chess.square(file_index, rank)
            is_last = last_move is not None and square in (last_move.from_square, last_move.to_square)
            bg = blue if is_last else (light if (rank + file_index) % 2 != 0 else dark)
            piece = board.piece_at(square)
            glyph = SOLID_PIECE_GLYPHS[piece.piece_type] if piece else " "
            fg = "255;255;255" if (piece and piece.color == chess.WHITE) else "20;20;20"
            if board.is_check() and square == board.king(board.turn):
                bg = (248, 196, 196)
                fg = "198;40;40"
            r, g, b = bg
            row.append(f"\033[48;2;{r};{g};{b}m\033[38;2;{fg}m {glyph} \033[0m")
        row.append(f"│ {rank + 1}")
        print("".join(row))
    print("   " + "─" * 34)
    print("    " + files)


# ============================================================
# 8. SEPARATED FOLDER HIERARCHY (`human_vs_model` & `model_vs_model`)
# ============================================================

class GameLogger:
    """
    Organizes games into dedicated sub-directories under `inference/`:
      - `inference/human_vs_model/<timestamp>_hvm_human_<side>_elo<h_elo>_<diff>_model_elo<m_elo>/`
      - `inference/model_vs_model/<timestamp>_mvm_white_elo<w_elo>_vs_black_elo<b_elo>/`
      - `inference/position_analysis/<timestamp>_analysis_elo<elo>/`
    """

    def __init__(
        self,
        inference_root: Path,
        mode: str,
        target_elo: float = DEFAULT_TARGET_ELO,
        human_color: str = "white",
        human_elo: float = 1500.0,
        difficulty: str = "Medium",
        white_elo: float = DEFAULT_TARGET_ELO,
        black_elo: float = DEFAULT_TARGET_ELO,
    ):
        now = dt.datetime.now().astimezone()
        self.started_at = now.isoformat()
        stamp = now.strftime("%Y%m%d_%H%M%S")
        self.mode = mode
        self.target_elo = float(target_elo)
        self.human_color = human_color
        self.human_elo = float(human_elo)
        self.difficulty = difficulty
        self.white_elo = float(white_elo)
        self.black_elo = float(black_elo)

        mode_lower = mode.lower()
        if "human" in mode_lower:
            category_dir = inference_root / "human_vs_model"
            diff_slug = difficulty.lower().replace(" ", "_")
            self.game_id = (
                f"{stamp}_hvm_human_{human_color}_elo{int(round(human_elo))}"
                f"_{diff_slug}_model_elo{int(round(target_elo))}"
            )
        elif "model vs model" in mode_lower or "model_vs_model" in mode_lower:
            category_dir = inference_root / "model_vs_model"
            self.game_id = (
                f"{stamp}_mvm_white_elo{int(round(white_elo))}"
                f"_vs_black_elo{int(round(black_elo))}"
            )
        else:
            category_dir = inference_root / "position_analysis"
            self.game_id = f"{stamp}_analysis_elo{int(round(target_elo))}"

        self.game_dir = category_dir / self.game_id
        self.board_dir = self.game_dir / "boards"
        self.board_dir.mkdir(parents=True, exist_ok=True)

        self.moves_jsonl_path = self.game_dir / "moves.jsonl"
        self.thinking_log_path = self.game_dir / "thinking.log"
        self.thinking_jsonl_path = self.game_dir / "thinking.jsonl"
        self.pgn_path = self.game_dir / "game.pgn"
        self.summary_json_path = self.game_dir / "game_summary.json"
        # Deliberately separate review-only data from the immutable game summary.
        # This lets a completed game be starred/noted without altering its result.
        self.review_metadata_path = self.game_dir / "review_metadata.json"

        self.move_number = 0
        self.captures = {"white": Counter(), "black": Counter()}
        self.san_moves: List[str] = []
        self.move_records: List[Dict[str, Any]] = []
        self.thinking_history: List[Dict[str, Any]] = []

        start_payload = {
            "event": "game_start",
            "game_id": self.game_id,
            "mode": mode,
            "human_color": human_color if "human" in mode_lower else None,
            "human_elo": self.human_elo if "human" in mode_lower else None,
            "difficulty": difficulty if "human" in mode_lower else None,
            "model_target_elo": self.target_elo,
            "white_elo": self.white_elo,
            "black_elo": self.black_elo,
            "started_at": self.started_at,
            "status": "RUNNING",
            "game_dir": str(self.game_dir),
        }
        self.write_move_event(start_payload)
        self.summary_json_path.write_text(json.dumps(start_payload, indent=2), encoding="utf-8")
        self.log_thinking_text(
            f"=== GAME STARTED | id={self.game_id} | mode={mode} | "
            f"white_elo={self.white_elo:.0f} | black_elo={self.black_elo:.0f} ==="
        )

    def write_move_event(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with self.moves_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def update_progress(self, **fields: Any) -> None:
        """Persist the live lifecycle/timing fields without ending the game."""
        try:
            current = json.loads(self.summary_json_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        current.update(fields)
        current["status"] = "RUNNING"
        self.summary_json_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def log_thinking_text(self, text: str) -> None:
        stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted = f"[{stamp}] {text}"
        with self.thinking_log_path.open("a", encoding="utf-8") as f:
            f.write(formatted + "\n")
            f.flush()

    def record_thinking(self, analysis: Dict[str, Any], also_print: bool = True) -> List[str]:
        now_iso = dt.datetime.now().astimezone().isoformat()
        payload = {"timestamp": now_iso, "game_id": self.game_id, **analysis}
        self.thinking_history.append(payload)
        with self.thinking_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            f.flush()

        purpose = analysis.get("purpose", "MODEL_MOVE")
        tier = analysis.get("difficulty_tier", "Medium")
        lines = [
            f"[{purpose} | Tier={tier}] MODEL THINKING (ply={analysis['ply']}, move={analysis['fullmove_number']}, side={analysis['side']})",
            f"  Conditioning : target_elo={analysis['target_elo']:.0f} (bucket={analysis['target_elo_bucket']})",
            f"  Strategy Mode: {analysis.get('strategic_mode', 'BALANCED')} "
            f"(material_adv={analysis.get('material_advantage_cp', 0):+d} cp, phase={analysis['phase']})",
            f"  Board State  : in_check={analysis['in_check']} | "
            f"legal_moves={analysis['legal_move_count']} | history_used={analysis['history_plies_used']}",
            f"  Score Stats  : range=[{analysis['score_min']:+.4f}, {analysis['score_max']:+.4f}] | "
            f"mean={analysis['score_mean']:+.4f} | std={analysis['score_std']:.4f} | "
            f"top1_prob={analysis['top1_prob']*100:.1f}% | gap1_2={analysis['best_second_gap']:+.4f}",
            "  Top Candidates (Adjusted = Raw + Elo-Stratified Tactical/Minimax Bonus):",
        ]
        for item in analysis["top_candidates"]:
            tags = ", ".join(item["tags"]) if item["tags"] else "quiet"
            moved = item["moved_piece"] or "?"
            captured = item["captured_piece"] or "-"
            raw_sc = item.get("raw_score", item["score"])
            adj_sc = item.get("tactical_adjustment", 0.0)
            lines.append(
                f"    #{item['rank']:>2} {item['san']:<8} ({item['uci']:<5}) | "
                f"score={item['score']:+.4f} (raw={raw_sc:+.4f}, Δadj={adj_sc:+.2f}) | "
                f"prob={item['prob']*100:5.1f}% | moved={moved:<6} cap={captured:<6} [{tags}]"
            )

        for line in lines:
            self.log_thinking_text(line)
            if also_print:
                print(line)
        return lines

    def record_move(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        move: chess.Move,
        san: str,
        actor: str,
        ranking: Optional[List[Dict[str, Any]]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        human_move_time_seconds: Optional[float] = None,
        coach_requested: bool = False,
    ) -> Dict[str, Any]:
        self.move_number += 1
        side = "white" if board_before.turn == chess.WHITE else "black"
        if board_before.is_en_passant(move):
            captured_piece = board_before.piece_at(
                move.to_square - 8 if board_before.turn == chess.WHITE else move.to_square + 8
            )
        else:
            captured_piece = board_before.piece_at(move.to_square)

        captured_name = PIECE_NAMES.get(captured_piece.piece_type) if captured_piece else None
        if captured_name:
            self.captures[side][captured_name] += 1

        self.san_moves.append(san)
        sel_policy = (analysis or {}).get("selection_policy", "direct")
        self.log_thinking_text(
            f"MOVE PLAYED #{self.move_number}: {actor.upper()} ({side}) played {san} ({move.uci()}) "
            f"[policy={sel_policy}]"
            + (f" [captured {captured_name}]" if captured_name else "")
        )

        svg_path = self.board_dir / f"board_{self.move_number:04d}.svg"
        png_path = self.board_dir / f"board_{self.move_number:04d}.png"
        render_board_matplotlib(
            board_after,
            svg_path,
            png_path,
            last_move=move,
            move_number=self.move_number,
            san=san,
            target_elo=self.target_elo,
            captured_piece=captured_name,
            logger=self,
            inline=True,
        )
        shutil.copy2(svg_path, self.board_dir / "current_board.svg")
        shutil.copy2(png_path, self.board_dir / "current_board.png")

        event = {
            "event": "move",
            "game_id": self.game_id,
            "move_number": self.move_number,
            "ply": self.move_number,
            "timestamp": dt.datetime.now().astimezone().isoformat(),
            "model_move_completed_at": (
                dt.datetime.now().astimezone().isoformat() if actor == "model" else None
            ),
            "human_decision_completed_at": (
                dt.datetime.now().astimezone().isoformat() if actor == "human" else None
            ),
            "side": side,
            "actor": actor,
            "san": san,
            "uci": move.uci(),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "target_elo": self.target_elo,
            "fen_before": board_before.fen(),
            "fen_after": board_after.fen(),
            "captured_piece": captured_name,
            "captures_by_white": dict(self.captures["white"]),
            "captures_by_black": dict(self.captures["black"]),
            "ranking": ranking or [],
            "model_decision_analysis": analysis,
            "human_move_time_seconds": human_move_time_seconds,
            "coach_requested": bool(coach_requested),
            "board_svg": str(svg_path),
            "board_png": str(png_path),
        }
        self.move_records.append(event)
        self.write_move_event(event)
        self.update_progress(
            game_id=self.game_id,
            mode=self.mode,
            started_at=self.started_at,
            status="RUNNING",
            last_move_at=event["timestamp"],
            moves=len(self.san_moves),
        )
        self._write_pgn()
        return event

    def _write_pgn(self, result_str: str = "*") -> None:
        tokens: List[str] = []
        for index in range(0, len(self.move_records), 2):
            move_no = index // 2 + 1
            white = self._pgn_move_token(self.move_records[index])
            if index + 1 < len(self.move_records):
                black = self._pgn_move_token(self.move_records[index + 1])
                tokens.append(f"{move_no}. {white} {black}")
            else:
                tokens.append(f"{move_no}. {white}")
        with self.pgn_path.open("w", encoding="utf-8") as f:
            f.write(f'[Event "{self.mode}"]\n')
            f.write(f'[White "Elo {int(round(self.white_elo))}"]\n')
            f.write(f'[Black "Elo {int(round(self.black_elo))}"]\n')
            f.write(f'[Result "{result_str}"]\n\n')
            f.write(" ".join(tokens) + f" {result_str}\n")
            f.flush()

    def _pgn_move_token(self, event: Dict[str, Any]) -> str:
        """Return a portable PGN token with the requested human/model details."""
        actor = str(event.get("actor", "model")).title()
        side = str(event.get("side", "")).title()
        details = [
            f"White Elo: {int(round(self.white_elo))}",
            f"Black Elo: {int(round(self.black_elo))}",
            f"Played by {actor} ({side})",
            f"UCI: {event.get('uci', '')}",
        ]
        if event.get("human_move_time_seconds") is not None:
            details.append(f"Human time: {float(event['human_move_time_seconds']):.2f}s")
        if event.get("coach_requested"):
            details.append("Coach help requested")
        return f"{event.get('san', '')} {{" + "; ".join(details) + "}"

    def finish(self, board: chess.Board, termination_info: Dict[str, Any]) -> None:
        result = termination_info.get("result", "*")
        title = termination_info.get("title", "Game Over")
        explanation = termination_info.get("explanation", "")
        self.log_thinking_text(
            f"=== GAME FINISHED | {title} | {explanation} | total_moves={len(self.san_moves)} ==="
        )
        self._write_pgn(result_str=result)
        end_payload = {
            "event": "game_end",
            "game_id": self.game_id,
            "mode": self.mode,
            "timestamp": dt.datetime.now().astimezone().isoformat(),
            "status": str(termination_info.get("status", "COMPLETED")).upper(),
            "result": result,
            "termination": termination_info,
            "final_fen": board.fen(),
            "moves": len(self.san_moves),
            "san_moves": self.san_moves,
            "white_elo": self.white_elo,
            "black_elo": self.black_elo,
            "human_color": self.human_color,
            "human_elo": self.human_elo,
            "difficulty": self.difficulty,
            "captures_by_white": dict(self.captures["white"]),
            "captures_by_black": dict(self.captures["black"]),
        }
        self.write_move_event(end_payload)
        self.summary_json_path.write_text(json.dumps(end_payload, indent=2), encoding="utf-8")


# ============================================================
# 9. MODEL LOADING & ELO-STRATIFIED 2-PLY MINIMAX REASONING ENGINE
# ============================================================

LAST_MODEL_ANALYSIS: Dict[str, Any] = {}


def load_v04_runtime(checkpoint_path: Path, inference_dir: Path, bundle_path: Path) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    is_elo_conditioned = (
        checkpoint.get("model_variant") == "v0.4-elo-conditioned"
        or any(k.startswith("elo_encoder.") for k in state_dict.keys())
    )
    has_film = any(k.startswith("context_film.") for k in state_dict.keys())

    if is_elo_conditioned:
        model: nn.Module = ChessPolicyRankerV04Elo(dropout=0.10, use_film=has_film)
        variant_name = "v0.4-elo-film" if has_film else "v0.4-elo-conditioned"
    else:
        model = ChessPolicyRankerV04(dropout=0.10)
        variant_name = "v0.4-base"

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    bundle = {
        "bundle_version": "v0.4-standalone-inference-4.0-stratified-arena",
        "model_version": MODEL_VERSION,
        "model_variant": variant_name,
        "is_elo_conditioned": is_elo_conditioned,
        "has_film": has_film,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch", "?")),
        "best_val_top1": checkpoint.get("best_val_top1"),
        "best_val_regret": checkpoint.get("best_val_regret"),
        "inference_dir": str(inference_dir),
        "device": device,
        "model": model.to(device).eval(),
    }

    try:
        serializable_bundle = {
            **{k: v for k, v in bundle.items() if k not in ("model", "device")},
            "model": model_cpu,
        }
        with bundle_path.open("wb") as f:
            cloudpickle.dump(serializable_bundle, f, protocol=5)
    except Exception as exc:
        print(f"Bundle save note: {exc}")

    return bundle


def _move_tags(candidate: Dict[str, Any]) -> List[str]:
    mf = candidate.get("move_features", {})
    tags: List[str] = []
    if mf.get("is_capture"):
        tags.append(f"capture:{mf.get('captured_piece') or 'piece'}")
    if mf.get("is_check"):
        tags.append("check")
    if mf.get("is_castling") or mf.get("is_castle"):
        tags.append("castle")
    if mf.get("is_en_passant"):
        tags.append("en-passant")
    if mf.get("is_promotion"):
        tags.append(f"promotion:{mf.get('promotion') or 'queen'}")
    return tags


def _static_position_eval_cp(board: chess.Board, perspective: chess.Color) -> float:
    """
    Fast centipawn + positional evaluation from `perspective` side:
      - Material balance (P=100, N=320, B=330, R=500, Q=900)
      - Bishop pair bonus (+35 cp)
      - Center control (+12 cp per attack on d4/e4/d5/e5)
      - King safety / king-ring pressure (-18 cp per enemy attacker near king)
      - Hanging / undefended piece penalty (-110 cp)
    """
    if board.is_checkmate():
        return -10000.0 if board.turn == perspective else 10000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
        return 0.0

    mat_white = material_difference(board)
    mat_persp = float(mat_white if perspective == chess.WHITE else -mat_white)

    own_bishops = chess.popcount(board.bishops & board.occupied_co[perspective])
    opp_bishops = chess.popcount(board.bishops & board.occupied_co[not perspective])
    bishop_pair_cp = (35.0 if own_bishops >= 2 else 0.0) - (35.0 if opp_bishops >= 2 else 0.0)

    center_cp = 12.0 * (
        center_control_count(board, perspective) - center_control_count(board, not perspective)
    )

    own_king = board.king(perspective)
    opp_king = board.king(not perspective)
    king_safety_cp = 18.0 * (
        king_ring_pressure(board, opp_king, perspective)
        - king_ring_pressure(board, own_king, not perspective)
    )

    own_hanging = hanging_non_king(board, perspective)
    opp_hanging = hanging_non_king(board, not perspective)
    if board.turn == perspective:
        hanging_cp = 110.0 * opp_hanging - 45.0 * own_hanging
    else:
        hanging_cp = 45.0 * opp_hanging - 110.0 * own_hanging

    return mat_persp + bishop_pair_cp + center_cp + king_safety_cp + hanging_cp


def evaluate_candidate_tactical_consequences(
    board: chess.Board,
    move: chess.Move,
    repetition: Dict[Any, int],
    history_moves: List[Dict[str, Any]],
    material_adv_cp: int,
    phase_str: str,
    target_elo: float = DEFAULT_TARGET_ELO,
    difficulty: str = "Medium",
) -> Tuple[float, List[str]]:
    """
    Elo-Stratified Tactical & Positional Consequence Evaluator:
      - Easy   (<1480 Elo): 0-ply shallow intuition (recognizes immediate Mate-in-1
        and stalemate traps when ahead, without 2-ply blunder shield or deep SEE).
      - Medium (1480-1719): 1-ply tactical guard + repetition contempt + avoids
        immediate Mate-in-1 blunders.
      - Hard   (1720-1949): Full 2-ply tactical minimax lookahead (evaluates
        opponent's strongest capture/check/promotion refutation and material swing).
      - Expert (>=1950 Elo): Deep 2-ply minimax lookahead + passed-pawn endgame
        push bonuses + aggressive win-conversion contempt.
    """
    extra_tags: List[str] = []
    score_adjustment = 0.0
    elo_val = float(target_elo)
    diff_upper = (difficulty or "").strip().upper()

    is_easy = (elo_val < 1480.0) or (diff_upper == "EASY" and elo_val < 1650.0)
    is_expert = (elo_val >= 1950.0) or (diff_upper == "EXPERT")
    is_hard = (not is_easy and not is_expert) and ((elo_val >= 1720.0) or (diff_upper == "HARD"))

    our_color = board.turn
    is_capture = board.is_capture(move)
    moving_piece = board.piece_at(move.from_square)
    is_pawn_move = moving_piece is not None and moving_piece.piece_type == chess.PAWN
    base_static_cp = _static_position_eval_cp(board, our_color)

    is_ping_pong = False
    if len(history_moves) >= 2 and not is_capture and not is_pawn_move:
        prev_own_uci = history_moves[-2].get("uci", "")
        if len(prev_own_uci) >= 4:
            try:
                prev_own_move = chess.Move.from_uci(prev_own_uci)
                if (
                    move.from_square == prev_own_move.to_square
                    and move.to_square == prev_own_move.from_square
                ):
                    is_ping_pong = True
            except Exception:
                pass

    board.push(move)
    try:
        if board.is_checkmate():
            extra_tags.append("CHECKMATE!")
            return 12.0, extra_tags

        if is_easy:
            if board.is_stalemate() and material_adv_cp >= 200:
                score_adjustment -= 2.5
                extra_tags.append("stalemate-trap")
            return score_adjustment, extra_tags

        new_key = board_key(board)
        prior_occurrences = repetition.get(new_key, 0)
        is_stalemate = board.is_stalemate()
        is_insufficient = board.is_insufficient_material()
        is_fifty_moves = board.can_claim_fifty_moves()
        is_draw_rule = is_stalemate or is_insufficient or is_fifty_moves
        is_threefold = prior_occurrences >= 2
        is_twofold = prior_occurrences == 1

        if is_stalemate:
            extra_tags.append("stalemate-draw")
        elif is_threefold:
            extra_tags.append("3fold-draw")
        elif is_twofold:
            extra_tags.append("repeats-pos")
        elif is_draw_rule:
            extra_tags.append("rule-draw")

        if is_ping_pong:
            extra_tags.append("ping-pong")

        contempt_scale = 1.35 if is_expert else (1.10 if is_hard else 0.80)
        if is_draw_rule or is_threefold:
            if material_adv_cp >= 50:
                score_adjustment -= contempt_scale * (5.0 + (material_adv_cp / 200.0))
            elif material_adv_cp >= -50:
                score_adjustment -= contempt_scale * 0.90
            else:
                score_adjustment += contempt_scale * (3.0 + (abs(material_adv_cp) / 300.0))
        elif is_twofold:
            if material_adv_cp >= 50:
                score_adjustment -= contempt_scale * (2.2 + (material_adv_cp / 400.0))
            elif material_adv_cp >= -50:
                score_adjustment -= contempt_scale * 0.45
            else:
                score_adjustment += contempt_scale * 0.90

        if is_ping_pong and not (is_draw_rule or is_threefold):
            if material_adv_cp >= 50:
                score_adjustment -= contempt_scale * 1.00
            elif material_adv_cp >= -50:
                score_adjustment -= contempt_scale * 0.35

        opp_moves = list(board.legal_moves)
        opp_can_mate_in_1 = False
        all_opp_replies_get_mated = bool(opp_moves) and (board.is_check() or len(opp_moves) <= 6)
        worst_static_after_opp_reply_cp = 10000.0

        for opp_mv in opp_moves:
            is_opp_tactical = board.is_capture(opp_mv) or board.gives_check(opp_mv) or (opp_mv.promotion is not None)
            board.push(opp_mv)
            try:
                if board.is_checkmate():
                    opp_can_mate_in_1 = True
                    all_opp_replies_get_mated = False
                    break

                if all_opp_replies_get_mated:
                    has_mate_reply = False
                    for our_reply in board.legal_moves:
                        if board.gives_check(our_reply):
                            board.push(our_reply)
                            is_m = board.is_checkmate()
                            board.pop()
                            if is_m:
                                has_mate_reply = True
                                break
                    if not has_mate_reply:
                        all_opp_replies_get_mated = False

                if (is_hard or is_expert) and (is_opp_tactical or len(opp_moves) <= 10):
                    ev_cp = _static_position_eval_cp(board, our_color)
                    if ev_cp < worst_static_after_opp_reply_cp:
                        worst_static_after_opp_reply_cp = ev_cp
            finally:
                board.pop()

        if opp_can_mate_in_1:
            score_adjustment -= 10.0
            extra_tags.append("BLUNDERS-MATE!")
        elif all_opp_replies_get_mated:
            score_adjustment += 6.5
            extra_tags.append("FORCED-MATE-2!")
        elif (is_hard or is_expert) and worst_static_after_opp_reply_cp < 9999.0:
            net_swing_cp = worst_static_after_opp_reply_cp - base_static_cp
            minimax_weight = 0.0045 if is_expert else 0.0025
            swing_adj = clamp(net_swing_cp * minimax_weight, -3.2, 2.2)
            score_adjustment += swing_adj
            if net_swing_cp <= -220:
                extra_tags.append(f"loses-exchange({int(net_swing_cp)}cp)")
            elif net_swing_cp >= 180:
                extra_tags.append(f"wins-tactic(+{int(net_swing_cp)}cp)")

        if is_expert:
            if phase_str == "endgame" and is_pawn_move:
                to_rank = chess.square_rank(move.to_square)
                adv_rank = to_rank if our_color == chess.WHITE else (7 - to_rank)
                if adv_rank >= 4:
                    score_adjustment += 0.18 * (adv_rank - 3)
                    extra_tags.append("passed-push")
            if board.is_check() and material_adv_cp >= 0 and not is_twofold:
                score_adjustment += 0.15

    finally:
        board.pop()

    return score_adjustment, extra_tags


def select_move_by_elo_and_difficulty(
    ranking: List[Dict[str, Any]],
    ply: int,
    target_elo: float,
    difficulty: str = "Medium",
    purpose: str = "MODEL_MOVE",
    excluded_ucis: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Selects a move from `ranking` according to the difficulty / Elo tier:
      - AI_SUGGESTION_FOR_HUMAN: Returns the highest-ranked candidate not in
        `excluded_ucis` (so rejecting a suggestion and asking again cycles to
        the next strongest move!).
      - Expert (>=1950 Elo): Opening variety in the first 6 plies among master book
        candidates within 0.12 score of #1; strict Top-1 optimal play thereafter.
      - Hard (1720-1949 Elo): Opening variety in the first 8 plies within 0.22 of #1;
        near-optimal in middlegame/endgame.
      - Medium (1480-1719 Elo): Softmax sampling over top 3 moves within 0.45 of #1.
      - Easy (<1480 Elo): Human-like softmax sampling over top 4 moves within 1.25 of #1.
    """
    if not ranking:
        raise ValueError("Empty ranking list.")

    if purpose != "MODEL_MOVE":
        if excluded_ucis:
            for cand in ranking:
                if cand["uci"] not in excluded_ucis:
                    return cand, f"coach_alternative(rank={cand['rank']})"
        return ranking[0], "strict_top1_coach"

    if len(ranking) == 1:
        return ranking[0], "single_legal_move"

    top1 = ranking[0]
    top1_tags = set(top1.get("tags") or [])
    if "CHECKMATE!" in top1_tags or "FORCED-MATE-2!" in top1_tags:
        return top1, "forced_mate_top1"

    elo_val = float(target_elo)
    diff_upper = (difficulty or "").strip().upper()
    is_easy = (elo_val < 1480.0) or (diff_upper == "EASY" and elo_val < 1650.0)
    is_expert = (elo_val >= 1950.0) or (diff_upper == "EXPERT")
    is_hard = (not is_easy and not is_expert) and ((elo_val >= 1720.0) or (diff_upper == "HARD"))

    if is_expert:
        if ply <= 6:
            pool = [
                c for c in ranking[:3]
                if (top1["score"] - c["score"]) <= 0.14
                and "BLUNDERS-MATE!" not in (c.get("tags") or [])
                and not any(t.startswith("loses-exchange") for t in (c.get("tags") or []))
            ]
            if len(pool) > 1:
                chosen = random.choice(pool)
                return chosen, f"expert_book_variety(rank={chosen['rank']},pool={len(pool)})"
        return top1, "expert_strict_top1"

    if is_hard:
        max_gap = 0.24 if ply <= 8 else 0.08
        max_k = 3 if ply <= 8 else 2
        pool = [
            c for c in ranking[:max_k]
            if (top1["score"] - c["score"]) <= max_gap
            and "BLUNDERS-MATE!" not in (c.get("tags") or [])
        ]
        if len(pool) <= 1:
            return top1, "hard_strict_top1"
        temp = 0.16
    elif not is_easy:
        max_gap = 0.48 if ply <= 10 else 0.28
        pool = [
            c for c in ranking[:3]
            if (top1["score"] - c["score"]) <= max_gap
            and "BLUNDERS-MATE!" not in (c.get("tags") or [])
        ]
        if len(pool) <= 1:
            return top1, "medium_top1"
        temp = 0.30
    else:
        pool = [c for c in ranking[:4] if (top1["score"] - c["score"]) <= 1.30]
        if len(pool) <= 1:
            return top1, "easy_single"
        temp = 0.65

    weights = [math.exp((c["score"] - top1["score"]) / max(0.05, temp)) for c in pool]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    chosen = random.choices(pool, weights=probs, k=1)[0]
    tier_name = "easy" if is_easy else ("hard" if is_hard else "medium")
    return chosen, f"elo_sampled(tier={tier_name},rank={chosen['rank']},pool={len(pool)})"


def infer_ranked_moves(
    bundle: Dict[str, Any],
    board: chess.Board,
    history_moves: List[Dict[str, Any]],
    memory: Dict[str, Any],
    target_elo: float = DEFAULT_TARGET_ELO,
    difficulty: Optional[str] = None,
    top_n: int = 5,
    logger: Optional[GameLogger] = None,
    print_thoughts: bool = True,
    repetition: Optional[Dict[Any, int]] = None,
    purpose: str = "MODEL_MOVE",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    global LAST_MODEL_ANALYSIS

    device: torch.device = bundle["device"]
    model: nn.Module = bundle["model"]
    resolved_difficulty = infer_difficulty_from_elo(target_elo, difficulty)

    record = make_runtime_record(board, history_moves, memory, target_elo=target_elo)
    batch = v04_collate([record])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.inference_mode():
        outputs = model(batch)
        raw_scores = outputs["scores"].float()[0]
        mask = outputs["candidate_mask"][0]

    candidates = record["input"]["candidate_moves"]
    phase_str = record["input"]["phase"]
    mat_diff_white = material_difference(board)
    material_adv_cp = mat_diff_white if board.turn == chess.WHITE else -mat_diff_white

    if repetition is None:
        repetition = Counter()
        replay = chess.Board()
        repetition[board_key(replay)] += 1
        for hm in history_moves:
            try:
                replay.push_uci(hm["uci"])
                repetition[board_key(replay)] += 1
            except Exception:
                break
        repetition[board_key(board)] = max(1, repetition.get(board_key(board), 1))

    adjusted_scores = raw_scores.clone()
    consequence_tags_by_idx: Dict[int, List[str]] = {}
    adjustments_by_idx: Dict[int, float] = {}

    valid = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    for c_idx_t in valid:
        c_idx = int(c_idx_t.item())
        cand = candidates[c_idx]
        mv = chess.Move.from_uci(cand["uci"])
        adj, extra_tags = evaluate_candidate_tactical_consequences(
            board=board,
            move=mv,
            repetition=repetition,
            history_moves=history_moves,
            material_adv_cp=material_adv_cp,
            phase_str=phase_str,
            target_elo=target_elo,
            difficulty=resolved_difficulty,
        )
        adjusted_scores[c_idx] += adj
        adjustments_by_idx[c_idx] = adj
        consequence_tags_by_idx[c_idx] = extra_tags

    probs = torch.softmax(
        adjusted_scores.masked_fill(~mask, torch.finfo(torch.float32).min), dim=-1
    )
    ordered = valid[torch.argsort(adjusted_scores[valid], descending=True)]

    valid_scores = adjusted_scores[ordered].detach().cpu().tolist()
    valid_probs = probs[ordered].detach().cpu().tolist()
    best_score = float(valid_scores[0])
    second_score = float(valid_scores[1]) if len(valid_scores) > 1 else best_score
    mean_score = sum(valid_scores) / max(1, len(valid_scores))
    variance = sum((x - mean_score) ** 2 for x in valid_scores) / max(1, len(valid_scores))
    std_score = math.sqrt(variance)

    detailed: List[Dict[str, Any]] = []
    for rank_idx, cand_idx_t in enumerate(ordered[:max(top_n, 10)]):
        c_idx = int(cand_idx_t.item())
        cand = candidates[c_idx]
        sc = float(adjusted_scores[c_idx].item())
        raw_sc = float(raw_scores[c_idx].item())
        adj_sc = float(adjustments_by_idx.get(c_idx, 0.0))
        pr = float(probs[c_idx].item())
        tags = _move_tags(cand) + consequence_tags_by_idx.get(c_idx, [])
        detailed.append({
            "rank": rank_idx + 1,
            "uci": cand["uci"],
            "san": cand["san"],
            "from": chess.square_name(cand["from_square"]),
            "to": chess.square_name(cand["to_square"]),
            "from_square": cand["from_square"],
            "to_square": cand["to_square"],
            "score": sc,
            "raw_score": raw_sc,
            "tactical_adjustment": adj_sc,
            "prob": pr,
            "delta_from_best": sc - best_score,
            "moved_piece": cand.get("move_features", {}).get("moved_piece"),
            "captured_piece": cand.get("move_features", {}).get("captured_piece"),
            "tags": tags,
        })

    ranking = [item.copy() for item in detailed[:top_n]]
    side = "white" if board.turn == chess.WHITE else "black"
    elo_bucket = target_elo_bucket_label(target_elo)

    strategic_mode = (
        "PLAY_FOR_WIN_AVOID_DRAW" if material_adv_cp >= 50
        else "SEEK_DRAW_OR_PERPETUAL" if material_adv_cp <= -150
        else "BALANCED_FIGHTING"
    )

    analysis = {
        "purpose": purpose,
        "difficulty_tier": resolved_difficulty,
        "side": side,
        "phase": phase_str,
        "ply": len(history_moves) + 1,
        "fullmove_number": board.fullmove_number,
        "material_advantage_cp": material_adv_cp,
        "strategic_mode": strategic_mode,
        "target_elo": float(target_elo),
        "target_elo_bucket": elo_bucket,
        "in_check": bool(board.is_check()),
        "legal_move_count": len(candidates),
        "candidate_count": len(candidates),
        "history_plies_used": min(len(history_moves), MAX_HISTORY_PLIES),
        "score_min": float(min(valid_scores)),
        "score_max": float(max(valid_scores)),
        "score_mean": float(mean_score),
        "score_std": float(std_score),
        "best_score": best_score,
        "second_score": second_score,
        "best_second_gap": float(best_score - second_score),
        "top1_prob": float(valid_probs[0]) if valid_probs else 0.0,
        "top_candidates": detailed[:10],
    }
    LAST_MODEL_ANALYSIS = analysis

    if logger is not None:
        logger.record_thinking(analysis, also_print=print_thoughts)

    return ranking, analysis


# ============================================================
# 10. GAME SESSION MANAGER (WITH AI SUGGESTIONS & PGN REVIEW)
# ============================================================

def parse_user_move(board: chess.Board, text: str) -> chess.Move:
    text = text.strip()
    if not text:
        raise ValueError("Enter a UCI move such as e2e4 (or a legal SAN move such as Nf3).")
    normalized = text.lower()
    try:
        move = chess.Move.from_uci(normalized)
        if board.is_legal(move):
            return move
        if len(normalized) in (4, 5):
            raise ValueError(
                f"'{text}' is not legal in this position. Your king may be in check, the piece may not move that way, "
                "or the move may leave your king in check."
            )
    except Exception:
        if len(normalized) in (4, 5) and normalized[:2].isalnum() and normalized[2:4].isalnum():
            raise
    try:
        return board.parse_san(text)
    except Exception as exc:
        raise ValueError(
            f"'{text}' is neither a valid UCI move nor a legal SAN move for the current position. "
            "Use UCI such as e2e4, g1f3, or e7e8q for a promotion."
        ) from exc


def rebuild_memory_state_from_moves(
    board: chess.Board,
    history_moves: List[Dict[str, Any]],
    repetition: Counter,
    material_history: List[int],
) -> Dict[str, Any]:
    move_counts_by_piece: Dict[str, Dict[str, int]] = {"white": defaultdict(int), "black": defaultdict(int)}
    check_counts = {"white": 0, "black": 0}
    capture_counts = {"white": 0, "black": 0}
    castle_counts = {"white": 0, "black": 0}
    promotion_counts = {"white": 0, "black": 0}
    strategic_events: List[Dict[str, Any]] = []

    for item in history_moves:
        side = item["side"]
        mf = item.get("move_features", {})
        if mf.get("is_check"):
            check_counts[side] += 1
        if mf.get("is_capture"):
            capture_counts[side] += 1
        if mf.get("is_castle") or mf.get("is_castling"):
            castle_counts[side] += 1
        if mf.get("is_promotion"):
            promotion_counts[side] += 1
        moved_name = mf.get("moved_piece")
        if moved_name:
            move_counts_by_piece[side][str(moved_name).lower()] += 1
        strategic_events.extend(item.get("strategic_events", []))

    return build_long_term_memory(
        board=board,
        ply=len(history_moves) + 1,
        material_history=material_history,
        repetition_counts=repetition,
        move_counts_by_piece=move_counts_by_piece,
        check_counts=check_counts,
        capture_counts=capture_counts,
        castle_counts=castle_counts,
        promotion_counts=promotion_counts,
        strategic_events=strategic_events,
    )


def serialize_board_pieces(board: chess.Board) -> Dict[str, Dict[str, str]]:
    pieces_map: Dict[str, Dict[str, str]] = {}
    for sq, piece in board.piece_map().items():
        pieces_map[chess.square_name(sq)] = {
            "color": "white" if piece.color == chess.WHITE else "black",
            "type": PIECE_NAMES[piece.piece_type],
            "symbol": piece.symbol(),
            "glyph": SOLID_PIECE_GLYPHS[piece.piece_type],
        }
    return pieces_map


class LiveGameSession:
    """Thread-safe game state manager backing the REST endpoints and Web UI."""

    def __init__(self, bundle: Dict[str, Any], inference_dir: Path):
        self.bundle = bundle
        self.inference_dir = inference_dir
        self.lock = threading.Lock()

        self.mode = "Human vs Model"
        self.human_color = "white"
        self.human_elo = 1500.0
        self.difficulty = "Medium"
        self.target_elo = map_difficulty_to_model_elo(self.human_elo, self.difficulty)
        self.white_elo = self.human_elo
        self.black_elo = self.target_elo

        self.board = chess.Board()
        self.history_moves: List[Dict[str, Any]] = []
        self.repetition: Counter = Counter({board_key(self.board): 1})
        self.material_history: List[int] = [material_difference(self.board)]
        self.last_move: Optional[chess.Move] = None
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.pending_suggestion: Optional[Dict[str, Any]] = None
        self.rejected_suggestion_ucis: Set[str] = set()
        self.ply_snapshots: List[Dict[str, Any]] = []
        self.human_turn_started_at: Optional[dt.datetime] = None
        self.model_move_completed_at: Optional[dt.datetime] = None
        self.game_status: Optional[str] = None
        self.coach_requested_this_turn = False
        self.manual_termination: Optional[Dict[str, Any]] = None

        self.logger: Optional[GameLogger] = None

    def _record_ply_snapshot_unlocked(
        self,
        san: Optional[str],
        actor: str,
        analysis: Optional[Dict[str, Any]],
    ) -> None:
        self.ply_snapshots.append({
            "ply": len(self.san_moves_unlocked()),
            "move_number": self.logger.move_number,
            "fullmove_number": self.board.fullmove_number,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "san": san,
            "actor": actor,
            "is_check": self.board.is_check(),
            "last_move": {
                "from": chess.square_name(self.last_move.from_square),
                "to": chess.square_name(self.last_move.to_square),
                "uci": self.last_move.uci(),
            } if self.last_move else None,
            "pieces": serialize_board_pieces(self.board),
            "captures": {
                "white": dict(self.logger.captures["white"]),
                "black": dict(self.logger.captures["black"]),
            },
            "analysis": copy.deepcopy(analysis) if analysis else None,
            "human_move_time_seconds": None,
            "coach_requested": False,
        })

    def san_moves_unlocked(self) -> List[str]:
        return list(self.logger.san_moves)

    def start_new_game(
        self,
        mode: str = "Human vs Model",
        human_color: str = "white",
        human_elo: float = 1500.0,
        difficulty: str = "Medium",
        white_elo: Optional[float] = None,
        black_elo: Optional[float] = None,
        target_elo: Optional[float] = None,
        starting_fen: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if self.game_status == "RUNNING":
                self._expire_if_needed_unlocked()
            if self.game_status == "RUNNING":
                raise ValueError("The current game is still RUNNING. Resign or complete it before starting a new game.")
            self.mode = mode
            self.human_color = human_color.lower() if human_color.lower() in ("white", "black") else "white"
            self.human_elo = float(human_elo or 1500.0)
            self.difficulty = difficulty if difficulty in DIFFICULTY_OFFSETS else "Medium"

            if "model vs model" in mode.lower():
                self.white_elo = float(white_elo if white_elo is not None else (target_elo or 1800.0))
                self.black_elo = float(black_elo if black_elo is not None else (target_elo or 1800.0))
                self.target_elo = self.white_elo
            else:
                mapped_elo = map_difficulty_to_model_elo(self.human_elo, self.difficulty)
                self.target_elo = float(target_elo) if target_elo is not None else mapped_elo
                if self.human_color == "white":
                    self.white_elo = self.human_elo
                    self.black_elo = self.target_elo
                else:
                    self.white_elo = self.target_elo
                    self.black_elo = self.human_elo

            self.board = chess.Board(starting_fen) if starting_fen else chess.Board()
            self.history_moves = []
            self.repetition = Counter({board_key(self.board): 1})
            self.material_history = [material_difference(self.board)]
            self.last_move = None
            self.last_analysis = None
            self.pending_suggestion = None
            self.rejected_suggestion_ucis.clear()
            self.ply_snapshots = []
            self.human_turn_started_at = (
                dt.datetime.now(dt.timezone.utc)
                if "model vs model" not in self.mode.lower() and self.board.turn == (self.human_color == "white")
                else None
            )
            self.model_move_completed_at = None
            self.game_status = "RUNNING"
            self.coach_requested_this_turn = False
            self.manual_termination = None

            self.logger = GameLogger(
                self.inference_dir,
                self.mode,
                target_elo=self.target_elo,
                human_color=self.human_color,
                human_elo=self.human_elo,
                difficulty=self.difficulty,
                white_elo=self.white_elo,
                black_elo=self.black_elo,
            )
            self.logger.update_progress(
                human_turn_started_at=(
                    self.human_turn_started_at.isoformat() if self.human_turn_started_at else None
                ),
                model_move_completed_at=None,
            )
            self._record_ply_snapshot_unlocked(san=None, actor="start", analysis=None)

            svg_path = self.logger.board_dir / "board_0000.svg"
            render_board_matplotlib(
                self.board, svg_path, move_number=0, target_elo=self.target_elo, logger=self.logger, inline=False
            )
            return self._serialize_state_unlocked()

    def _expire_if_needed_unlocked(self) -> None:
        if self.game_status != "RUNNING" or self.logger is None:
            return
        started = self._parse_timestamp(self.logger.started_at)
        if started is None or (dt.datetime.now(dt.timezone.utc) - started).total_seconds() <= GAME_EXPIRY_SECONDS:
            return
        self.manual_termination = {
            "is_over": True,
            "result": "*",
            "winner": None,
            "reason_code": "expired",
            "title": "Game Expired",
            "explanation": "The game expired after 24 hours without completion.",
            "status": "EXPIRED",
        }
        self.game_status = "EXPIRED"
        self.human_turn_started_at = None
        self.logger.finish(self.board, self.manual_termination)

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[dt.datetime]:
        if not value:
            return None
        try:
            parsed = dt.datetime.fromisoformat(str(value))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return None

    def request_ai_suggestion(self, suggestion_elo: Optional[float] = None) -> Dict[str, Any]:
        """
        Computes a Master-tier (2200 Elo, Expert 2-ply minimax) move suggestion
        for the human player without playing it on the board. If the human rejected
        a previous suggestion on this same ply, automatically proposes the next
        strongest candidate move!
        """
        with self.lock:
            self._ensure_active_unlocked()
            self._expire_if_needed_unlocked()
            if self.manual_termination or self.board.is_game_over(claim_draw=True):
                raise ValueError("Game is already over.")
            if "model vs model" in self.mode.lower() or (
                ("white" if self.board.turn == chess.WHITE else "black") != self.human_color
            ):
                raise ValueError("AI coaching is available only while it is the human player's turn.")
            self.coach_requested_this_turn = True

            elo_to_use = float(
                suggestion_elo
                if suggestion_elo is not None
                else max(AI_COACH_SUGGESTION_ELO, self.human_elo, self.target_elo)
            )
            memory = rebuild_memory_state_from_moves(
                self.board, self.history_moves, self.repetition, self.material_history
            )
            ranking, analysis = infer_ranked_moves(
                self.bundle,
                self.board,
                self.history_moves,
                memory,
                target_elo=elo_to_use,
                difficulty="Expert",
                top_n=10,
                logger=self.logger,
                print_thoughts=False,
                repetition=self.repetition,
                purpose="AI_SUGGESTION_FOR_HUMAN",
            )
            best_cand, sel_policy = select_move_by_elo_and_difficulty(
                ranking=ranking,
                ply=len(self.history_moves) + 1,
                target_elo=elo_to_use,
                difficulty="Expert",
                purpose="AI_SUGGESTION_FOR_HUMAN",
                excluded_ucis=self.rejected_suggestion_ucis,
            )
            analysis["selection_policy"] = sel_policy
            analysis["selected_move_rank"] = best_cand["rank"]

            self.pending_suggestion = {
                "rank": best_cand["rank"],
                "uci": best_cand["uci"],
                "san": best_cand["san"],
                "from": best_cand["from"],
                "to": best_cand["to"],
                "score": best_cand["score"],
                "raw_score": best_cand.get("raw_score", best_cand["score"]),
                "tactical_adjustment": best_cand.get("tactical_adjustment", 0.0),
                "prob": best_cand["prob"],
                "tags": best_cand.get("tags", []),
                "suggestion_elo": elo_to_use,
                "strategic_mode": analysis.get("strategic_mode", "BALANCED"),
                "rejected_count": len(self.rejected_suggestion_ucis),
                "top_alternatives": ranking[:5],
            }
            self.last_analysis = analysis
            return self._serialize_state_unlocked()

    def reject_ai_suggestion(self) -> Dict[str, Any]:
        with self.lock:
            if self.pending_suggestion and self.pending_suggestion.get("uci"):
                self.rejected_suggestion_ucis.add(str(self.pending_suggestion["uci"]))
            self.pending_suggestion = None
            return self._serialize_state_unlocked()

    def apply_human_move(self, move_str: str) -> Dict[str, Any]:
        with self.lock:
            self._ensure_active_unlocked()
            self._expire_if_needed_unlocked()
            if self.manual_termination or self.board.is_game_over(claim_draw=True):
                raise ValueError("Game is already over.")
            if "model vs model" in self.mode.lower():
                raise ValueError("Human moves are unavailable in Model vs Model games.")
            side_to_move = "white" if self.board.turn == chess.WHITE else "black"
            if side_to_move != self.human_color:
                raise ValueError("It is the model's turn. Wait for the model move before entering your UCI move.")
            move = parse_user_move(self.board, move_str)
            now = dt.datetime.now(dt.timezone.utc)
            elapsed = max(0.0, (now - self.human_turn_started_at).total_seconds()) if self.human_turn_started_at else 0.0
            board_before = self.board.copy(stack=False)
            san = self.board.san(move)
            self.history_moves.append(history_move_from_board(self.board, move))
            self.board.push(move)
            self.last_move = move
            self.pending_suggestion = None
            self.rejected_suggestion_ucis.clear()
            self.repetition[board_key(self.board)] += 1
            self.material_history.append(material_difference(self.board))
            self.logger.record_move(
                board_before, self.board, move, san, actor="human",
                human_move_time_seconds=elapsed,
                coach_requested=self.coach_requested_this_turn,
            )
            self.logger.update_progress(
                human_turn_started_at=None,
                model_move_completed_at=None,
                last_human_move_seconds=elapsed,
            )
            self._record_ply_snapshot_unlocked(san=san, actor="human", analysis=None)
            self.ply_snapshots[-1]["human_move_time_seconds"] = elapsed
            self.ply_snapshots[-1]["coach_requested"] = self.coach_requested_this_turn
            self.human_turn_started_at = None
            self.model_move_completed_at = None
            self.coach_requested_this_turn = False

            if self.board.is_game_over(claim_draw=True):
                term_info = describe_game_termination(self.board)
                term_info["status"] = "COMPLETED"
                self.game_status = "COMPLETED"
                self.logger.finish(self.board, term_info)
            return self._serialize_state_unlocked()

    def play_model_move(self, target_elo_override: Optional[float] = None) -> Dict[str, Any]:
        with self.lock:
            self._ensure_active_unlocked()
            self._expire_if_needed_unlocked()
            if self.manual_termination or self.board.is_game_over(claim_draw=True):
                return self._serialize_state_unlocked()

            is_white_turn = (self.board.turn == chess.WHITE)
            if "model vs model" in self.mode.lower():
                active_elo = self.white_elo if is_white_turn else self.black_elo
                active_difficulty = infer_difficulty_from_elo(active_elo)
            else:
                active_elo = float(target_elo_override) if target_elo_override is not None else self.target_elo
                active_difficulty = self.difficulty

            self.target_elo = active_elo
            self.logger.target_elo = active_elo
            self.pending_suggestion = None
            self.rejected_suggestion_ucis.clear()

            memory = rebuild_memory_state_from_moves(
                self.board, self.history_moves, self.repetition, self.material_history
            )
            ranking, analysis = infer_ranked_moves(
                self.bundle,
                self.board,
                self.history_moves,
                memory,
                target_elo=active_elo,
                difficulty=active_difficulty,
                top_n=8,
                logger=self.logger,
                print_thoughts=False,
                repetition=self.repetition,
                purpose="MODEL_MOVE",
            )
            selected, selection_policy = select_move_by_elo_and_difficulty(
                ranking=ranking,
                ply=len(self.history_moves) + 1,
                target_elo=active_elo,
                difficulty=active_difficulty,
                purpose="MODEL_MOVE",
            )
            analysis["selection_policy"] = selection_policy
            analysis["selected_move_rank"] = selected["rank"]

            move = chess.Move.from_uci(selected["uci"])
            board_before = self.board.copy(stack=False)
            san = self.board.san(move)
            self.history_moves.append(history_move_from_board(self.board, move))
            self.board.push(move)
            self.last_move = move
            self.last_analysis = analysis
            self.repetition[board_key(self.board)] += 1
            self.material_history.append(material_difference(self.board))
            self.logger.record_move(
                board_before, self.board, move, san, actor="model", ranking=ranking[:5], analysis=analysis
            )
            self._record_ply_snapshot_unlocked(san=san, actor="model", analysis=analysis)
            self.model_move_completed_at = dt.datetime.now(dt.timezone.utc)
            self.logger.update_progress(
                human_turn_started_at=None,
                model_move_completed_at=self.model_move_completed_at.isoformat(),
            )

            if "model vs model" not in self.mode.lower() and (
                ("white" if self.board.turn == chess.WHITE else "black") == self.human_color
            ):
                self.human_turn_started_at = dt.datetime.now(dt.timezone.utc)
                self.logger.update_progress(
                    human_turn_started_at=self.human_turn_started_at.isoformat(),
                )
                self.coach_requested_this_turn = False

            if self.board.is_game_over(claim_draw=True):
                term_info = describe_game_termination(self.board)
                term_info["status"] = "COMPLETED"
                self.game_status = "COMPLETED"
                self.logger.finish(self.board, term_info)
            return self._serialize_state_unlocked()

    def resign_human_game(self) -> Dict[str, Any]:
        """Finish the active human-vs-model game as a human resignation."""
        with self.lock:
            self._ensure_active_unlocked()
            self._expire_if_needed_unlocked()
            if "model vs model" in self.mode.lower():
                raise ValueError("Resignation is available only in Human vs Model games.")
            if self.manual_termination or self.board.is_game_over(claim_draw=True):
                raise ValueError("Game is already over.")
            model_color = "black" if self.human_color == "white" else "white"
            self.manual_termination = {
                "is_over": True,
                "result": "0-1" if self.human_color == "white" else "1-0",
                "winner": model_color,
                "reason_code": "human_resignation",
                "title": "You resigned",
                "explanation": f"The human player ({self.human_color.title()}) resigned. {model_color.title()} (Model) wins.",
                "status": "RESIGNED",
            }
            self.game_status = "RESIGNED"
            self.pending_suggestion = None
            self.human_turn_started_at = None
            self.logger.finish(self.board, self.manual_termination)
            return self._serialize_state_unlocked()

    def _ensure_active_unlocked(self) -> None:
        if self.logger is None or self.game_status is None:
            raise ValueError("No game is running. Choose New Game to start one.")

    def _find_game_dir(self, game_id: str) -> Path:
        """Resolve an archive id without allowing a request to escape the log root."""
        if not game_id or Path(game_id).name != game_id:
            raise FileNotFoundError("Invalid saved game id.")
        for subdir_name in ("human_vs_model", "model_vs_model", "position_analysis"):
            gdir = self.inference_dir / subdir_name / game_id
            if gdir.is_dir() and (gdir / "moves.jsonl").exists():
                return gdir
        raise FileNotFoundError(f"Saved game not found: {game_id}")

    @staticmethod
    def _read_review_metadata(gdir: Path) -> Dict[str, Any]:
        path = gdir / "review_metadata.json"
        if not path.exists():
            return {"starred": False, "notes": ""}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"starred": bool(data.get("starred", False)), "notes": str(data.get("notes", ""))}
        except Exception:
            return {"starred": False, "notes": ""}

    @staticmethod
    def _read_game_events(gdir: Path) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for line in (gdir / "moves.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    @staticmethod
    def _is_completed_events(events: List[Dict[str, Any]]) -> bool:
        return any(event.get("event") == "game_end" for event in events)

    @staticmethod
    def _format_pgn_move(event: Dict[str, Any], white_elo: Any, black_elo: Any) -> str:
        actor = str(event.get("actor", "model")).title()
        side = str(event.get("side", "")).title()
        details = [
            f"White Elo: {int(round(float(white_elo or 0)))}",
            f"Black Elo: {int(round(float(black_elo or 0)))}",
            f"Played by {actor} ({side})",
            f"UCI: {event.get('uci', '')}",
        ]
        if event.get("human_move_time_seconds") is not None:
            details.append(f"Human time: {float(event['human_move_time_seconds']):.2f}s")
        if event.get("coach_requested"):
            details.append("Coach help requested")
        return f"{event.get('san', '')} {{" + "; ".join(details) + "}"

    def get_annotated_pgn(self, game_id: Optional[str] = None) -> Dict[str, Any]:
        """Build a copy-ready PGN for either the live game or a saved review."""
        with self.lock:
            if game_id:
                gdir = self._find_game_dir(game_id)
                events = self._read_game_events(gdir)
            else:
                gdir = self.logger.game_dir
                events = self._read_game_events(gdir)
            start = next((event for event in events if event.get("event") == "game_start"), {})
            end = next((event for event in reversed(events) if event.get("event") == "game_end"), {})
            moves = [event for event in events if event.get("event") == "move"]
            white_elo = start.get("white_elo", self.white_elo)
            black_elo = start.get("black_elo", self.black_elo)
            result = end.get("result", "*")
            tokens: List[str] = []
            for index in range(0, len(moves), 2):
                move_no = index // 2 + 1
                white = self._format_pgn_move(moves[index], white_elo, black_elo)
                black = self._format_pgn_move(moves[index + 1], white_elo, black_elo) if index + 1 < len(moves) else ""
                tokens.append(f"{move_no}. {white}" + (f" {black}" if black else ""))
            pgn = (
                f'[Event "{start.get("mode", self.mode)}"]\n'
                f'[White "Elo {int(round(float(white_elo or 0)))}"]\n'
                f'[Black "Elo {int(round(float(black_elo or 0)))}"]\n'
                f'[Result "{result}"]\n\n'
                + " ".join(tokens) + f" {result}\n"
            )
            return {"game_id": start.get("game_id", gdir.name), "pgn": pgn}

    def update_game_review(self, game_id: str, starred: Optional[bool] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            gdir = self._find_game_dir(game_id)
            events = self._read_game_events(gdir)
            if not self._is_completed_events(events):
                raise ValueError("Stars and notes unlock once the game has finished.")
            metadata = self._read_review_metadata(gdir)
            if starred is not None:
                metadata["starred"] = bool(starred)
            if notes is not None:
                metadata["notes"] = str(notes).strip()[:10000]
            (gdir / "review_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return metadata

    def list_saved_games(self) -> List[Dict[str, Any]]:
        games_list: List[Dict[str, Any]] = []
        for subdir_name in ("human_vs_model", "model_vs_model"):
            cat_dir = self.inference_dir / subdir_name
            if not cat_dir.exists():
                continue
            for gdir in sorted(cat_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not gdir.is_dir():
                    continue
                moves_file = gdir / "moves.jsonl"
                if not moves_file.exists():
                    continue
                summary_file = gdir / "game_summary.json"
                meta: Dict[str, Any] = {}
                if summary_file.exists():
                    try:
                        meta = json.loads(summary_file.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                try:
                    events = self._read_game_events(gdir)
                except Exception:
                    events = []
                review = self._read_review_metadata(gdir)
                games_list.append({
                    "game_id": gdir.name,
                    "category": subdir_name,
                    "mode": meta.get("mode", subdir_name.replace("_", " ").title()),
                    "result": meta.get("result", "*"),
                    "status": str(meta.get("status", "COMPLETED" if self._is_completed_events(events) else "RUNNING")).upper(),
                    "moves": meta.get("moves", 0),
                    "white_elo": meta.get("white_elo"),
                    "black_elo": meta.get("black_elo"),
                    "difficulty": meta.get("difficulty"),
                    "is_completed": self._is_completed_events(events),
                    "starred": review["starred"],
                    "notes": review["notes"],
                    "path": str(gdir),
                })
                if len(games_list) >= 40:
                    break
        return sorted(games_list, key=lambda game: (not game["starred"], game["game_id"]), reverse=False)

    def load_saved_game_for_review(self, game_id: str) -> Dict[str, Any]:
        gdir = self._find_game_dir(game_id)

        board = chess.Board()
        snapshots: List[Dict[str, Any]] = [{
            "ply": 0,
            "move_number": 0,
            "fullmove_number": 1,
            "fen": board.fen(),
            "turn": "white",
            "san": None,
            "actor": "start",
            "is_check": False,
            "last_move": None,
            "pieces": serialize_board_pieces(board),
            "captures": {"white": {}, "black": {}},
            "analysis": None,
        }]
        san_history: List[str] = []
        game_meta: Dict[str, Any] = {"game_id": game_id, "game_dir": str(gdir)}
        saved_termination: Optional[Dict[str, Any]] = None

        for line in (gdir / "moves.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("event") == "game_start":
                game_meta.update(ev)
            elif ev.get("event") == "game_end":
                saved_termination = ev.get("termination")
            elif ev.get("event") == "move":
                mv = chess.Move.from_uci(ev["uci"])
                board.push(mv)
                san_history.append(ev["san"])
                snapshots.append({
                    "ply": len(san_history),
                    "move_number": ev.get("move_number", len(san_history)),
                    "fullmove_number": board.fullmove_number,
                    "fen": board.fen(),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "san": ev["san"],
                    "actor": ev.get("actor", "model"),
                    "is_check": board.is_check(),
                    "last_move": {
                        "from": chess.square_name(mv.from_square),
                        "to": chess.square_name(mv.to_square),
                        "uci": mv.uci(),
                    },
                    "pieces": serialize_board_pieces(board),
                    "captures": {
                        "white": ev.get("captures_by_white", {}),
                        "black": ev.get("captures_by_black", {}),
                    },
                    "analysis": ev.get("model_decision_analysis"),
                    "human_move_time_seconds": ev.get("human_move_time_seconds"),
                    "coach_requested": bool(ev.get("coach_requested", False)),
                })

        thinking_tail = ""
        tlog = gdir / "thinking.log"
        if tlog.exists():
            lines = tlog.read_text(encoding="utf-8").splitlines()
            thinking_tail = "\n".join(lines[-80:])

        term_info = saved_termination or describe_game_termination(board)
        review = self._read_review_metadata(gdir)
        return {
            "game_id": game_id,
            "game_dir": str(gdir),
            "mode": game_meta.get("mode", "Saved Game Review"),
            "white_elo": game_meta.get("white_elo"),
            "black_elo": game_meta.get("black_elo"),
            "san_history": san_history,
            "ply_snapshots": snapshots,
            "termination": term_info,
            "thinking_log_tail": thinking_tail,
            "starred": review["starred"],
            "notes": review["notes"],
            "is_completed": self._is_completed_events(self._read_game_events(gdir)),
            "human_move_timings": [
                {
                    "ply": snap["ply"],
                    "move_number": (int(snap["ply"]) + 1) // 2,
                    "seconds": float(snap["human_move_time_seconds"]),
                    "coach_requested": bool(snap["coach_requested"]),
                    "san": snap["san"],
                }
                for snap in snapshots
                if snap.get("actor") == "human" and snap.get("human_move_time_seconds") is not None
            ],
        }

    def analyze_custom_position(
        self,
        fen: str,
        move_tokens: List[str],
        target_elo: float = DEFAULT_TARGET_ELO,
    ) -> Dict[str, Any]:
        with self.lock:
            board = chess.Board(fen) if fen.strip() else chess.Board()
            replay = chess.Board()
            hist_moves: List[Dict[str, Any]] = []
            rep = Counter({board_key(replay): 1})
            mat_hist = [material_difference(replay)]

            for tok in move_tokens:
                tok = tok.strip()
                if not tok:
                    continue
                mv = parse_user_move(replay, tok)
                hist_moves.append(history_move_from_board(replay, mv))
                replay.push(mv)
                rep[board_key(replay)] += 1
                mat_hist.append(material_difference(replay))

            if not move_tokens:
                rep = Counter({board_key(board): 1})
                mat_hist = [material_difference(board)]

            memory = rebuild_memory_state_from_moves(board, hist_moves, rep, mat_hist)
            analysis_logger = GameLogger(self.inference_dir, "Position_Analysis", target_elo=target_elo)
            ranking, analysis = infer_ranked_moves(
                self.bundle,
                board,
                hist_moves,
                memory,
                target_elo=float(target_elo),
                difficulty=infer_difficulty_from_elo(target_elo),
                top_n=10,
                logger=analysis_logger,
                print_thoughts=False,
                repetition=rep,
                purpose="CUSTOM_FEN_ANALYSIS",
            )
            svg_path = analysis_logger.board_dir / "analysis_board.svg"
            render_board_matplotlib(
                board, svg_path, move_number=len(hist_moves), target_elo=target_elo, logger=analysis_logger, inline=False
            )
            return {
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "target_elo": float(target_elo),
                "ranking": ranking,
                "analysis": analysis,
                "game_dir": str(analysis_logger.game_dir),
                "thinking_log_path": str(analysis_logger.thinking_log_path),
            }

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            self._expire_if_needed_unlocked()
            return self._serialize_state_unlocked()

    def _serialize_state_unlocked(self) -> Dict[str, Any]:
        if self.logger is None:
            return {
                "game_id": None,
                "game_dir": None,
                "mode": None,
                "status": None,
                "is_game_over": False,
                "termination": {"is_over": False, "title": "No Game Running", "explanation": "Choose New Game to begin."},
                "human_turn_started_at": None,
                "model_move_completed_at": None,
                "san_history": [],
                "ply_snapshots": [],
                "pieces": {},
                "legal_moves": [],
                "thinking_log_tail": "",
                "human_move_timings": [],
            }
        legal_moves_list = []
        for mv in self.board.legal_moves:
            legal_moves_list.append({
                "uci": mv.uci(),
                "san": self.board.san(mv),
                "from": chess.square_name(mv.from_square),
                "to": chess.square_name(mv.to_square),
                "promotion": chess.piece_symbol(mv.promotion) if mv.promotion else None,
            })

        thinking_tail = ""
        if self.logger.thinking_log_path.exists():
            lines = self.logger.thinking_log_path.read_text(encoding="utf-8").splitlines()
            thinking_tail = "\n".join(lines[-55:])

        term_info = self.manual_termination or describe_game_termination(self.board)
        review = self._read_review_metadata(self.logger.game_dir)
        human_move_timings = [
            {
                "ply": event.get("ply"),
                "move_number": (int(event.get("ply", 0)) + 1) // 2,
                "seconds": float(event.get("human_move_time_seconds", 0.0)),
                "coach_requested": bool(event.get("coach_requested", False)),
                "san": event.get("san", ""),
            }
            for event in self.logger.move_records
            if event.get("actor") == "human" and event.get("human_move_time_seconds") is not None
        ]
        return {
            "game_id": self.logger.game_id,
            "game_dir": str(self.logger.game_dir),
            "thinking_log_file": str(self.logger.thinking_log_path),
            "mode": self.mode,
            "human_color": self.human_color,
            "human_elo": self.human_elo,
            "difficulty": self.difficulty,
            "target_elo": self.target_elo,
            "white_elo": self.white_elo,
            "black_elo": self.black_elo,
            "target_elo_bucket": target_elo_bucket_label(self.target_elo),
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "move_number": self.logger.move_number,
            "fullmove_number": self.board.fullmove_number,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": term_info["is_over"],
            "status": self.game_status,
            "result": term_info["result"] if term_info["is_over"] else None,
            "termination": term_info,
            "last_move": {
                "from": chess.square_name(self.last_move.from_square),
                "to": chess.square_name(self.last_move.to_square),
                "uci": self.last_move.uci(),
            } if self.last_move else None,
            "san_history": self.logger.san_moves,
            "ply_snapshots": self.ply_snapshots,
            "captures": {
                "white": dict(self.logger.captures["white"]),
                "black": dict(self.logger.captures["black"]),
            },
            "pieces": serialize_board_pieces(self.board),
            "legal_moves": legal_moves_list,
            "pending_suggestion": self.pending_suggestion,
            "last_analysis": self.last_analysis,
            "thinking_log_tail": thinking_tail,
            "starred": review["starred"],
            "notes": review["notes"],
            "human_turn_started_at": self.human_turn_started_at.isoformat() if self.human_turn_started_at else None,
            "model_move_completed_at": self.model_move_completed_at.isoformat() if self.model_move_completed_at else None,
            "human_move_timings": human_move_timings,
        }


# ============================================================
# 11. FULL-WIDTH WEB ARENA HTML/JS + REST API ENDPOINTS
# ============================================================

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chess Policy Model v0.4 — Pro Chess Arena</title>
<style>
  /* Prefer a locally installed Manrope; the system stack keeps the UI readable offline. */
  @font-face {
    font-family: "Manrope";
    src: local("Manrope"), local("Manrope Regular");
    font-display: swap;
  }
  :root, [data-theme="dark"] {
    --bg: #0b1120;
    --panel: #111827;
    --panel-alt: #1e293b;
    --border: #26334d;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-soft: rgba(37, 99, 235, 0.18);
    --accent-cyan: #06b6d4;
    --success: #16a34a;
    --danger: #dc2626;
    --warning: #f59e0b;
    --text: #f8fafc;
    --muted: #94a3b8;
    --light-sq: #e2e8f0;
    --dark-sq: #3b6998;
    --move-hl: rgba(59, 130, 246, 0.45);
    --sel-hl: rgba(16, 185, 129, 0.55);
    --suggest-hl: rgba(245, 158, 11, 0.65);
    --check-hl: rgba(220, 38, 38, 0.72);
  }
  [data-theme="light"] {
    --bg: #f1f5f9;
    --panel: #ffffff;
    --panel-alt: #f8fafc;
    --border: #cbd5e1;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-soft: rgba(37, 99, 235, 0.12);
    --accent-cyan: #0284c7;
    --success: #15803d;
    --danger: #b91c1c;
    --warning: #d97706;
    --text: #0f172a;
    --muted: #475569;
    --light-sq: #f0f4f8;
    --dark-sq: #4b7399;
    --move-hl: rgba(37, 99, 235, 0.38);
    --sel-hl: rgba(22, 163, 74, 0.48);
    --suggest-hl: rgba(245, 158, 11, 0.60);
    --check-hl: rgba(220, 38, 38, 0.68);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Manrope", "Manrope Fallback", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    width: 100%;
    min-height: 100vh;
    padding: 12px 18px;
    transition: background 0.2s, color 0.2s;
  }
  .topbar {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
    color: #ffffff;
    padding: 12px 20px;
    border-radius: 12px;
    margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.28);
  }
  .topbar-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .topbar-title h1 {
    font-size: 1.22rem;
    font-weight: 800;
    letter-spacing: 0.2px;
  }
  .topbar-sub {
    font-size: 0.76rem;
    color: #dbeafe;
    font-family: monospace;
    margin-top: 2px;
  }
  .topbar-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .badge {
    background: rgba(255, 255, 255, 0.16);
    color: #fff;
    font-size: 0.76rem;
    padding: 5px 11px;
    border-radius: 999px;
    font-weight: 600;
    font-family: monospace;
  }
  .arena-grid {
    width: 100%;
    display: grid;
    grid-template-columns: 340px minmax(460px, 560px) 1fr;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 1280px) {
    .arena-grid {
      grid-template-columns: minmax(430px, 520px) 1fr;
    }
    .left-col { grid-column: 1 / -1; }
  }
  @media (max-width: 860px) {
    .arena-grid { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
  }
  .card-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 7px;
  }
  .board-container {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .player-banner {
    width: 100%;
    max-width: 496px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 8px;
    margin: 6px 0;
    font-size: 0.86rem;
    font-weight: 600;
  }
  .player-elo-pill {
    background: var(--primary);
    color: #fff;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-family: monospace;
  }
  .board-grid {
    width: 488px;
    height: 488px;
    max-width: 92vw;
    max-height: 92vw;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    border: 4px solid #1e3a8a;
    border-radius: 6px;
    user-select: none;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
  }
  .sq {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
    cursor: pointer;
    line-height: 1;
  }
  .sq.light { background: var(--light-sq); }
  .sq.dark { background: var(--dark-sq); }
  .sq.last-move { box-shadow: inset 0 0 0 999px var(--move-hl); }
  .sq.selected { box-shadow: inset 0 0 0 999px var(--sel-hl); }
  .sq.suggested { box-shadow: inset 0 0 0 999px var(--suggest-hl); }
  .sq.in-check { box-shadow: inset 0 0 0 999px var(--check-hl); }
  .sq .coord-file {
    position: absolute; bottom: 2px; right: 4px;
    font-size: 10px; font-weight: 800; color: #0f172a; pointer-events: none; opacity: 0.75;
  }
  .sq .coord-rank {
    position: absolute; top: 2px; left: 4px;
    font-size: 10px; font-weight: 800; color: #0f172a; pointer-events: none; opacity: 0.75;
  }
  .legal-dot {
    position: absolute;
    width: 15px; height: 15px;
    background: rgba(37, 99, 235, 0.78);
    border-radius: 50%;
    pointer-events: none;
    z-index: 3;
  }
  .legal-ring {
    position: absolute;
    inset: 3px;
    border: 3.5px solid rgba(220, 38, 38, 0.82);
    border-radius: 50%;
    pointer-events: none;
    z-index: 3;
  }
  .piece {
    font-family: "DejaVu Sans", "Segoe UI Symbol", "Noto Sans Symbols 2", sans-serif;
    font-weight: 700;
    pointer-events: none;
    z-index: 2;
  }
  .piece.white {
    color: #FFFFFF;
    -webkit-text-stroke: 2.1px #0f172a;
    paint-order: stroke fill;
    filter: drop-shadow(0 2px 2px rgba(0, 0, 0, 0.40));
  }
  .piece.black {
    color: #141414;
    -webkit-text-stroke: 1.3px #f8fafc;
    paint-order: stroke fill;
    filter: drop-shadow(0 2px 2px rgba(0, 0, 0, 0.35));
  }
  .scrubber-bar {
    width: 100%;
    max-width: 496px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 8px;
  }
  .scrubber-bar button {
    flex: 1;
    padding: 6px 8px;
    font-size: 0.82rem;
  }
  button {
    padding: 8px 13px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.84rem;
    cursor: pointer;
    transition: background 0.15s, transform 0.08s;
  }
  button:hover:not(:disabled) { background: var(--primary-hover); }
  button:active:not(:disabled) { transform: scale(0.98); }
  button.secondary { background: var(--panel-alt); color: var(--text); border: 1px solid var(--border); }
  button.secondary:hover:not(:disabled) { background: var(--border); }
  button.success { background: var(--success); }
  button.warning { background: var(--warning); color: #0f172a; font-weight: 700; }
  button.danger { background: var(--danger); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }

  .pgn-scroll {
    max-height: 250px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel-alt);
    margin-bottom: 10px;
  }
  .pgn-row {
    display: grid;
    grid-template-columns: 44px 1fr 1fr;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
  }
  .pgn-no {
    padding: 5px 8px;
    color: var(--muted);
    font-family: monospace;
    background: rgba(0, 0, 0, 0.12);
  }
  .pgn-ply {
    padding: 5px 10px;
    cursor: pointer;
    font-weight: 600;
  }
  .pgn-ply:hover { background: var(--primary-soft); }
  .pgn-ply.active-ply {
    background: var(--primary);
    color: #ffffff;
  }
  .suggestion-banner {
    display: none;
    background: var(--primary-soft);
    border: 2px solid var(--warning);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
  }
  .suggestion-banner.visible { display: block; }
  .conclusion-banner {
    display: none;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.22), rgba(22, 163, 74, 0.22));
    border: 2px solid var(--primary);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
  }
  .conclusion-banner.visible { display: block; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.81rem;
    margin-bottom: 12px;
  }
  th, td {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }
  th { color: var(--muted); font-weight: 600; }
  .log-box {
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.77rem;
    line-height: 1.45;
    white-space: pre-wrap;
    max-height: 270px;
    overflow-y: auto;
    color: #dbeafe;
  }
  select, input[type="number"], input[type="text"], textarea {
    width: 100%;
    padding: 7px 9px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    font-size: 0.86rem;
    font-family: inherit;
  }
  label { display: block; font-size: 0.77rem; color: var(--muted); margin-bottom: 4px; font-weight: 600; }
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(5, 10, 22, 0.76);
    z-index: 100;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-backdrop.open { display: flex; }
  .modal-card {
    width: 100%;
    max-width: 480px;
    background: var(--panel);
    border: 2px solid var(--primary);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65);
  }
  .form-group { margin-bottom: 12px; }
  textarea { min-height: 104px; resize: vertical; font-family: inherit; line-height: 1.45; }
  .icon-button { padding: 3px 8px; font-size: 1rem; line-height: 1.2; }
  .engagement-note { font-size: 0.81rem; color: var(--muted); line-height: 1.45; margin-bottom: 9px; }
  .move-feedback { display:none; margin-top:8px; border-radius:7px; padding:7px 9px; font-size:0.79rem; line-height:1.4; }
  .move-feedback.error { display:block; background:rgba(220,38,38,.16); color:#fecaca; border:1px solid rgba(220,38,38,.7); }
  [data-theme="light"] .move-feedback.error { color:#991b1b; }
  .human-clock { font-size:1.35rem; font-weight:800; color:var(--accent-cyan); font-variant-numeric:tabular-nums; margin:4px 0 9px; }
  .chart-card { margin-top:14px; padding:11px; }
  .chart-frame { background:#fff; border-radius:8px; border:1px solid #dbe3ee; padding:6px; cursor:pointer; min-height:155px; }
  .chart-frame:hover { border-color:#7c3aed; box-shadow:0 0 0 2px rgba(124,58,237,.18); }
  .chart-legend { display:flex; gap:10px; flex-wrap:wrap; font-size:.7rem; color:var(--muted); margin:6px 0 0; }
  .legend-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }
  .thinking-line { display:block; white-space:pre-wrap; }
  .thinking-line.prediction { color:#4ade80; }
  .thinking-line.coach-prediction { color:#d8b4fe; }
  .thinking-line.request { color:#f8fafc; }
  .chart-modal-card { width:min(94vw, 1100px); max-width:none; max-height:92vh; overflow:auto; }
  .chart-modal-card .chart-frame { min-height:520px; padding:16px; }
</style>
</head>
<body data-theme="dark">

<!-- TOP BAR -->
<div class="topbar">
  <div class="topbar-title">
    <div>
      <h1>♔ Chess Policy Model v0.4 — Elo-Conditioned Pro Arena</h1>
      <div class="topbar-sub" id="logPathBanner">Initializing game folder...</div>
    </div>
  </div>
  <div class="topbar-actions">
    <span class="badge" id="runtimeBadge">v0.4 Ready</span>
    <button onclick="openNewGameModal()" style="background:#ffffff; color:#1e3a8a; font-weight:800;">+ New Game</button>
    <button class="secondary" onclick="toggleTheme()" id="themeBtn" style="background:rgba(255,255,255,0.16); color:#fff; border:none;">☀️ Light Theme</button>
  </div>
</div>

<!-- 3-COLUMN FULL-WIDTH ARENA -->
<div class="arena-grid">

  <!-- LEFT COLUMN: GAME SETUP INFO, PGN MOVE REVIEWER & ARCHIVE -->
  <div class="card left-col">
    <div class="card-title">
      <span>📜 PGN Move Reviewer & History</span>
      <div style="display:flex; align-items:center; gap:7px;">
        <button class="secondary icon-button" onclick="copyAnnotatedPgn()" title="Copy annotated PGN">⧉</button>
        <span id="reviewStatusBadge" style="font-size:0.74rem; color:var(--accent-cyan);">Live</span>
      </div>
    </div>

    <div class="pgn-scroll" id="pgnContainer"></div>

    <div class="btn-row">
      <button class="secondary" onclick="returnToLive()" id="btnReturnLive" style="width:100%; display:none;">⏮ Return to Live Game</button>
    </div>

    <div class="card-title" style="margin-top:10px;">
      <span>📂 Saved Games Archive (HvM & MvM)</span>
      <button class="secondary" style="padding:3px 8px; font-size:0.74rem;" onclick="refreshSavedGames()">Refresh</button>
    </div>
    <div class="form-group">
      <select id="savedGamesSelect" onchange="onSelectSavedGame()">
        <option value="">-- Select a past game to review on board --</option>
      </select>
    </div>

    <div class="card-title" style="margin-top:10px;">
      <span id="engagementTitle">📝 Game Notes</span>
    </div>
    <div id="engagementPanel"></div>

    <div class="card chart-card" id="humanChartCard" style="display:none;">
      <div class="card-title" style="margin-bottom:7px;">
        <span>⏱ Human Move Time</span>
        <button class="secondary icon-button" onclick="openChartModal()" title="Open full-screen chart">⛶</button>
      </div>
      <div class="chart-frame" id="humanTimingChart" onclick="openChartModal()"></div>
      <div class="chart-legend">
        <span><i class="legend-dot" style="background:#2563eb;"></i>Independent move</span>
        <span><i class="legend-dot" style="background:#8b5cf6;"></i>Coach requested</span>
      </div>
    </div>
  </div>

  <!-- CENTER COLUMN: INTERACTIVE CHESS BOARD + TIMELINE SCRUBBER -->
  <div class="card board-container">
    <div class="player-banner">
      <span id="topPlayerLabel">Black (Model)</span>
      <span class="player-elo-pill" id="topPlayerElo">Elo 1600</span>
    </div>

    <div class="board-grid" id="boardGrid"></div>

    <div class="player-banner">
      <span id="bottomPlayerLabel">White (Human)</span>
      <span class="player-elo-pill" id="bottomPlayerElo">Elo 1500</span>
    </div>

    <div class="scrubber-bar">
      <button class="secondary" onclick="scrubTo(0)" title="First Move">|&lt;&lt;</button>
      <button class="secondary" onclick="scrubDelta(-1)" title="Previous Move">&lt; Prev</button>
      <span id="plyIndicator" style="font-size:0.8rem; font-weight:700; min-width:90px; text-align:center;">Ply 0 / 0</span>
      <button class="secondary" onclick="scrubDelta(1)" title="Next Move">Next &gt;</button>
      <button class="secondary" onclick="scrubTo('latest')" title="Latest Position">&gt;&gt;|</button>
      <button class="secondary" onclick="flipBoard()" title="Flip Board">⇅ Flip</button>
    </div>

    <div class="player-banner" style="flex-direction:column; align-items:flex-start; gap:4px; margin-top:8px;">
      <div style="width:100%; display:flex; justify-content:space-between;">
        <span id="turnIndicator" style="color:var(--accent-cyan);">Turn: WHITE</span>
        <span id="modeIndicator" style="color:var(--muted);">Human vs Model</span>
      </div>
      <div id="capWhite" style="font-size:0.8rem;">Captured by White: —</div>
      <div id="capBlack" style="font-size:0.8rem;">Captured by Black: —</div>
    </div>
  </div>

  <!-- RIGHT COLUMN: AI SUGGESTION APPROVAL, CONTROLS, CANDIDATE TABLE & THINKING LOGS -->
  <div class="card">
    <div class="conclusion-banner" id="conclusionBanner">
      <div style="font-size:1.02rem; font-weight:800; margin-bottom:4px;" id="conclusionTitle">Game Over</div>
      <div style="font-size:0.85rem; color:var(--muted);" id="conclusionText"></div>
    </div>

    <!-- AI Move Suggestion Banner (Approve / Reject / Next Alternative) -->
    <div class="suggestion-banner" id="suggestionBanner">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <strong style="font-size:0.92rem;">💡 AI Suggested Move: <span id="sugMoveSan" style="color:var(--warning);"></span></strong>
        <span class="player-elo-pill" id="sugConfidencePill">Prob: --%</span>
      </div>
      <div style="font-size:0.81rem; margin-bottom:8px; color:var(--muted);" id="sugReasoningText"></div>
      <div class="btn-row" style="margin-bottom:0;">
        <button class="success" onclick="approveSuggestion()">✅ Approve & Play Move</button>
        <button class="danger" onclick="rejectSuggestion()">✖ Reject (Ask Next Alternative)</button>
      </div>
    </div>

    <div class="card-title">
      <span>🎮 Action & AI Controls</span>
      <span id="strategyModeBadge" style="font-size:0.75rem; color:var(--accent-cyan);">BALANCED</span>
    </div>

    <div class="btn-row">
      <button class="warning" id="btnAskSuggestion" onclick="askAiSuggestion()">💡 Ask AI Suggestion</button>
      <button id="btnForceModel" onclick="triggerModelMove()">🤖 Force Model Move</button>
      <button class="secondary" id="btnAutoPlay" onclick="toggleAutoPlay()">▶ Start MvM Auto-Play</button>
      <button class="danger" id="btnResign" onclick="resignGame()" style="display:none;">🏳 Resign Game</button>
      <div style="display:flex; align-items:center; gap:6px; margin-left:auto;">
        <label style="margin:0;">Pace(ms):</label>
        <input type="number" id="delayInput" value="950" min="250" max="5000" step="100" style="width:82px;">
      </div>
    </div>

    <div class="card-title">
      <span id="rankingTableTitle">📊 Top Model Candidate Rankings</span>
    </div>
    <table>
      <thead>
        <tr><th>#</th><th>Move</th><th>UCI</th><th>Final</th><th>Raw</th><th>ΔAdj</th><th>Prob</th><th>Tactical Tags</th></tr>
      </thead>
      <tbody id="rankingBody">
        <tr><td colspan="8" style="color:var(--muted);">Play a move or click "Ask AI Suggestion" to inspect model reasoning.</td></tr>
      </tbody>
    </table>

    <div class="card-title">
      <span>🧠 Timestamped Model Thinking Stream</span>
    </div>
    <div class="log-box" id="thinkingLogBox">Waiting for model thoughts...</div>
  </div>
</div>

<!-- FULL-SCREEN HUMAN MOVE-TIME CHART -->
<div class="modal-backdrop" id="chartModal" onclick="if(event.target===this) closeChartModal()">
  <div class="modal-card chart-modal-card">
    <div class="card-title" style="font-size:1.05rem;">
      <span>⏱ Human Decision Time</span>
      <button class="secondary" style="padding:3px 9px;" onclick="closeChartModal()">✕</button>
    </div>
    <div class="chart-frame" id="humanTimingChartLarge"></div>
    <div class="chart-legend">
      <span><i class="legend-dot" style="background:#2563eb;"></i>Independent move</span>
      <span><i class="legend-dot" style="background:#8b5cf6;"></i>Coach requested</span>
      <span>Hover a point for exact time and move.</span>
    </div>
  </div>
</div>

<!-- NEW GAME CONFIGURATION MODAL -->
<div class="modal-backdrop" id="newGameModal">
  <div class="modal-card">
    <div class="card-title" style="font-size:1.05rem;">
      <span>♔ Start New Chess Game</span>
      <button class="secondary" style="padding:3px 9px;" onclick="closeNewGameModal()">✕</button>
    </div>

    <div class="form-group">
      <label>Select Game Mode</label>
      <select id="modalModeSelect" onchange="onModalModeChange()">
        <option value="Human vs Model">Human vs Model (Play against AI + AI Coach)</option>
        <option value="Model vs Model">Model vs Model (Watch AI vs AI Arena)</option>
      </select>
    </div>

    <div id="hvmOptionsBox">
      <div class="form-group">
        <label>Your Playing Side</label>
        <select id="modalHumanColor">
          <option value="white">Play as White (Move First)</option>
          <option value="black">Play as Black</option>
        </select>
      </div>
      <div class="form-group">
        <label>Your Chess Elo / Expertness Score (1000–2400)</label>
        <input type="number" id="modalHumanElo" value="1500" min="800" max="2600" step="50" oninput="updateMappedEloPreview()">
      </div>
      <div class="form-group">
        <label>AI Difficulty Level</label>
        <select id="modalDifficulty" onchange="updateMappedEloPreview()">
          <option value="Easy">Easy (Model Elo = Your Elo - 200 • 0-Ply Human-Like)</option>
          <option value="Medium" selected>Medium (Model Elo = Your Elo • 1-Ply Club Guard)</option>
          <option value="Hard">Hard (Model Elo = Your Elo + 200 • 2-Ply Tactical Minimax)</option>
          <option value="Expert">Expert (Master Tier: max(2000, Your Elo + 350) • Deep 2-Ply Minimax)</option>
        </select>
      </div>
      <div style="background:var(--primary-soft); border:1px solid var(--primary); padding:8px 12px; border-radius:8px; font-size:0.83rem; margin-bottom:12px;">
        Mapped AI Model Conditioning Elo: <strong id="mappedModelEloText">1500</strong>
      </div>
    </div>

    <div id="mvmOptionsBox" style="display:none;">
      <div class="form-group">
        <label>White Model Expertness Score (Elo 1000–2400)</label>
        <input type="number" id="modalWhiteElo" value="1800" min="800" max="2600" step="50">
      </div>
      <div class="form-group">
        <label>Black Model Expertness Score (Elo 1000–2400)</label>
        <input type="number" id="modalBlackElo" value="1800" min="800" max="2600" step="50">
      </div>
    </div>

    <div class="btn-row" style="justify-content:flex-end; margin-top:14px; margin-bottom:0;">
      <button class="secondary" onclick="closeNewGameModal()">Cancel</button>
      <button id="startNewGameButton" onclick="confirmStartNewGame()">Start Game</button>
    </div>
  </div>
</div>

<script>
let gameState = null;
let reviewData = null;
let viewedPly = null;
let selectedSquare = null;
let boardFlipped = false;
let autoPlayTimer = null;
let busy = false;
let humanClockTimer = null;

const FILES = ['a','b','c','d','e','f','g','h'];

async function apiCall(path, method='GET', body=null) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function activeSource() {
  return reviewData || gameState;
}

function isHumanVsModel(src) {
  return !!src && (src.mode || '').toLowerCase().includes('human vs model');
}

function isCompleted(src) {
  return !!src && !!((src.termination && src.termination.is_over) || src.is_completed);
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds || 0));
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function updateHumanClock() {
  const clock = document.getElementById('humanTurnClock');
  if (!clock || !gameState || reviewData || !gameState.human_turn_started_at) return;
  const elapsed = (Date.now() - new Date(gameState.human_turn_started_at).getTime()) / 1000;
  clock.textContent = formatDuration(elapsed);
}

function syncHumanClock() {
  if (humanClockTimer) clearInterval(humanClockTimer);
  humanClockTimer = null;
  if (gameState && !reviewData && gameState.human_turn_started_at && !gameState.is_game_over) {
    updateHumanClock();
    humanClockTimer = setInterval(updateHumanClock, 250);
  }
}

function showMoveError(message) {
  const box = document.getElementById('uciMoveError');
  if (!box) return;
  box.textContent = message;
  box.className = 'move-feedback error';
}

function clearMoveError() {
  const box = document.getElementById('uciMoveError');
  if (box) { box.textContent = ''; box.className = 'move-feedback'; }
}

function renderEngagementPanel(src) {
  const panel = document.getElementById('engagementPanel');
  const title = document.getElementById('engagementTitle');
  if (!src) return;
  const inLiveHumanTurn = !reviewData && isHumanVsModel(src) && !src.is_game_over && src.turn === src.human_color;
  if (inLiveHumanTurn) {
    title.textContent = '⌨️ Play a UCI Move';
    panel.innerHTML = `
      <div class="engagement-note">Your turn. Enter UCI notation (for example <code>e2e4</code>, <code>g1f3</code>, or <code>e7e8q</code>) and press Enter. Every move is checked against the current legal position.</div>
      <div class="human-clock" id="humanTurnClock">00:00:00</div>
      <label for="uciMoveInput">UCI move</label>
      <div style="display:flex; gap:7px;">
        <input id="uciMoveInput" type="text" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="e2e4" aria-label="UCI move">
        <button id="uciMoveButton" style="white-space:nowrap;">Play</button>
      </div>
      <div class="move-feedback" id="uciMoveError" role="alert"></div>`;
    const input = document.getElementById('uciMoveInput');
    const play = () => submitUciFromInput();
    input.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); play(); } });
    document.getElementById('uciMoveButton').addEventListener('click', play);
    syncHumanClock();
    return;
  }

  if (isCompleted(src)) {
    title.textContent = '📝 Game Notes';
    panel.innerHTML = `
      <div class="btn-row" style="align-items:center;">
        <button class="secondary" id="starGameButton" title="Mark this completed game as important"></button>
        <span class="engagement-note" style="margin:0;">Notes and star are saved with this completed game.</span>
      </div>
      <label for="gameNotesInput">Review notes</label>
      <textarea id="gameNotesInput" maxlength="10000" placeholder="What should you remember about this game?"></textarea>
      <div class="btn-row" style="margin-top:8px; margin-bottom:0;"><button class="success" id="saveNotesButton">Save Notes</button></div>`;
    document.getElementById('starGameButton').textContent = src.starred ? '★ Starred' : '☆ Star Game';
    document.getElementById('gameNotesInput').value = src.notes || '';
    document.getElementById('starGameButton').addEventListener('click', () => saveGameReview({starred: !src.starred}));
    document.getElementById('saveNotesButton').addEventListener('click', () => saveGameReview({notes: document.getElementById('gameNotesInput').value}));
    syncHumanClock();
    return;
  }

  title.textContent = '📝 Game Notes';
  panel.innerHTML = '<div class="engagement-note">Notes unlock after the game finishes. During your turn, this panel becomes a validated UCI move entry with a live decision timer.</div>';
  syncHumanClock();
}

async function submitUciFromInput() {
  const input = document.getElementById('uciMoveInput');
  if (!input || busy) return;
  clearMoveError();
  const move = input.value.trim();
  if (!move) { showMoveError('Enter a UCI move such as e2e4.'); return; }
  await submitHumanMove(move, true);
}

async function saveGameReview(change) {
  const src = activeSource();
  if (!src || !isCompleted(src)) return;
  try {
    const payload = {game_id: src.game_id};
    if (Object.prototype.hasOwnProperty.call(change, 'starred')) payload.starred = change.starred;
    if (Object.prototype.hasOwnProperty.call(change, 'notes')) payload.notes = change.notes;
    const metadata = await apiCall('/api/update_game_review', 'POST', payload);
    src.starred = metadata.starred;
    src.notes = metadata.notes;
    await refreshSavedGames();
    renderAll();
  } catch (err) {
    alert(err.message);
  }
}

async function copyAnnotatedPgn() {
  try {
    const gid = reviewData ? `?game_id=${encodeURIComponent(reviewData.game_id)}` : '';
    const res = await apiCall('/api/annotated_pgn' + gid);
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(res.pgn);
    } else {
      const helper = document.createElement('textarea');
      helper.value = res.pgn;
      helper.style.position = 'fixed'; helper.style.opacity = '0';
      document.body.appendChild(helper); helper.select();
      const copied = document.execCommand('copy');
      helper.remove();
      if (!copied) throw new Error('Browser clipboard access was denied');
    }
    const badge = document.getElementById('reviewStatusBadge');
    const old = badge.textContent;
    badge.textContent = 'PGN copied';
    setTimeout(() => { if (badge.textContent === 'PGN copied') badge.textContent = old; }, 1600);
  } catch (err) {
    alert('Could not copy PGN: ' + err.message);
  }
}

function chartData(src) {
  return (src && src.human_move_timings) ? src.human_move_timings : [];
}

function renderTimingChart(containerId, expanded=false) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const data = chartData(activeSource());
  if (!data.length) {
    container.innerHTML = '<div style="color:#475569; font-size:.78rem; padding:24px 8px; text-align:center;">Human move times will appear here after your first move.</div>';
    return;
  }
  const width = expanded ? 980 : 300;
  const height = expanded ? 470 : 145;
  const pad = expanded ? {l:58,r:24,t:25,b:48} : {l:35,r:12,t:16,b:30};
  const maxY = Math.max(5, ...data.map(d => Number(d.seconds || 0))) * 1.12;
  const x = i => data.length === 1 ? (pad.l + width - pad.r) / 2 : pad.l + i * ((width - pad.l - pad.r) / (data.length - 1));
  const y = value => height - pad.b - (Number(value || 0) / maxY) * (height - pad.t - pad.b);
  const points = data.map((d, i) => `${x(i)},${y(d.seconds)}`).join(' ');
  const yTicks = [0, maxY / 2, maxY];
  const grid = yTicks.map(v => `<g><line x1="${pad.l}" y1="${y(v)}" x2="${width-pad.r}" y2="${y(v)}" stroke="#e2e8f0"/><text x="${pad.l-6}" y="${y(v)+4}" text-anchor="end" fill="#64748b" font-size="${expanded ? 13 : 9}">${v.toFixed(v < 10 ? 1 : 0)}s</text></g>`).join('');
  const circles = data.map((d, i) => {
    const color = d.coach_requested ? '#8b5cf6' : '#2563eb';
    return `<circle cx="${x(i)}" cy="${y(d.seconds)}" r="${expanded ? 7 : 4.5}" fill="${color}" stroke="#fff" stroke-width="2"><title>Move ${d.move_number}: ${d.san || ''} — ${Number(d.seconds).toFixed(2)}s${d.coach_requested ? ' (coach requested)' : ''}</title></circle>`;
  }).join('');
  const labels = data.map((d, i) => `<text x="${x(i)}" y="${height-10}" text-anchor="middle" fill="#475569" font-size="${expanded ? 12 : 8}">${d.move_number}</text>`).join('');
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${expanded ? 500 : 155}" role="img" aria-label="Human move decision times"><rect width="100%" height="100%" fill="#fff"/>${grid}<polyline points="${points}" fill="none" stroke="#2563eb" stroke-width="${expanded ? 3 : 2}" stroke-linecap="round" stroke-linejoin="round"/>${circles}${labels}<text x="${width/2}" y="${height-2}" text-anchor="middle" fill="#475569" font-size="${expanded ? 12 : 8}">Human move number</text></svg>`;
}

function openChartModal() {
  if (!isHumanVsModel(activeSource())) return;
  document.getElementById('chartModal').classList.add('open');
  renderTimingChart('humanTimingChartLarge', true);
}

function closeChartModal() {
  document.getElementById('chartModal').classList.remove('open');
}

function toggleTheme() {
  const body = document.body;
  const next = body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  body.setAttribute('data-theme', next);
  document.getElementById('themeBtn').textContent = next === 'dark' ? '☀️ Light Theme' : '🌙 Dark Theme';
}

function openNewGameModal() {
  updateMappedEloPreview();
  document.getElementById('newGameModal').classList.add('open');
}

function closeNewGameModal() {
  document.getElementById('newGameModal').classList.remove('open');
}

function onModalModeChange() {
  const mode = document.getElementById('modalModeSelect').value;
  const isMvm = mode === 'Model vs Model';
  document.getElementById('hvmOptionsBox').style.display = isMvm ? 'none' : 'block';
  document.getElementById('mvmOptionsBox').style.display = isMvm ? 'block' : 'none';
}

function computeMappedElo(humanElo, difficulty) {
  if (difficulty === 'Expert') return Math.max(2000, Math.min(2400, humanElo + 350));
  const offsets = { 'Easy': -200, 'Medium': 0, 'Hard': 200 };
  return Math.max(1000, Math.min(2400, humanElo + (offsets[difficulty] || 0)));
}

function updateMappedEloPreview() {
  const hElo = parseFloat(document.getElementById('modalHumanElo').value || '1500');
  const diff = document.getElementById('modalDifficulty').value;
  const mapped = computeMappedElo(hElo, diff);
  document.getElementById('mappedModelEloText').textContent = `${Math.round(mapped)} (${diff})`;
}

function activeSnapshots() {
  const src = reviewData || gameState;
  return (src && src.ply_snapshots) ? src.ply_snapshots : [];
}

function currentViewedSnapshot() {
  const snaps = activeSnapshots();
  if (!snaps.length) return null;
  if (viewedPly === null || viewedPly >= snaps.length) {
    return snaps[snaps.length - 1];
  }
  return snaps[Math.max(0, viewedPly)];
}

function isViewingHistoricalPly() {
  if (reviewData !== null) return true;
  const snaps = activeSnapshots();
  return viewedPly !== null && viewedPly < snaps.length - 1;
}

function scrubTo(target) {
  const snaps = activeSnapshots();
  if (!snaps.length) return;
  if (target === 'latest') {
    viewedPly = reviewData ? (snaps.length - 1) : null;
  } else {
    viewedPly = Math.max(0, Math.min(snaps.length - 1, parseInt(target, 10)));
    if (!reviewData && viewedPly === snaps.length - 1) viewedPly = null;
  }
  selectedSquare = null;
  renderAll();
}

function scrubDelta(delta) {
  const snaps = activeSnapshots();
  if (!snaps.length) return;
  const cur = (viewedPly === null) ? (snaps.length - 1) : viewedPly;
  scrubTo(cur + delta);
}

function returnToLive() {
  reviewData = null;
  viewedPly = null;
  document.getElementById('savedGamesSelect').value = '';
  renderAll();
}

function renderPgnTable() {
  const container = document.getElementById('pgnContainer');
  container.innerHTML = '';
  const src = reviewData || gameState;
  if (!src || !src.san_history) return;

  const moves = src.san_history;
  const snaps = activeSnapshots();
  const activePly = (viewedPly === null) ? (snaps.length - 1) : viewedPly;

  for (let i = 0; i < moves.length; i += 2) {
    const row = document.createElement('div');
    row.className = 'pgn-row';

    const noSpan = document.createElement('span');
    noSpan.className = 'pgn-no';
    noSpan.textContent = `${Math.floor(i / 2) + 1}.`;
    row.appendChild(noSpan);

    const wSpan = document.createElement('span');
    wSpan.className = 'pgn-ply' + (activePly === i + 1 ? ' active-ply' : '');
    wSpan.textContent = moves[i];
    wSpan.onclick = () => scrubTo(i + 1);
    row.appendChild(wSpan);

    const bSpan = document.createElement('span');
    bSpan.className = 'pgn-ply' + (activePly === i + 2 ? ' active-ply' : '');
    if (i + 1 < moves.length) {
      bSpan.textContent = moves[i + 1];
      bSpan.onclick = () => scrubTo(i + 2);
    } else {
      bSpan.textContent = '';
    }
    row.appendChild(bSpan);

    container.appendChild(row);
  }
}

function renderThinkingLog(logText) {
  const box = document.getElementById('thinkingLogBox');
  box.innerHTML = '';
  let currentKind = 'request';
  for (const line of (logText || 'Waiting for model thoughts...').split('\\n')) {
    if (line.includes('AI_SUGGESTION_FOR_HUMAN')) currentKind = 'coach-prediction';
    else if (line.includes('MODEL_MOVE')) currentKind = 'prediction';
    else if (line.includes('MOVE PLAYED') || line.includes('GAME STARTED') || line.includes('GAME FINISHED')) currentKind = 'request';
    const span = document.createElement('span');
    span.className = `thinking-line ${currentKind}`;
    span.textContent = line;
    box.appendChild(span);
  }
  box.scrollTop = box.scrollHeight;
}

function renderAll() {
  const grid = document.getElementById('boardGrid');
  grid.innerHTML = '';
  const src = reviewData || gameState;
  if (!src) return;
  const liveHumanGame = !reviewData && isHumanVsModel(src) && src.status === 'RUNNING';
  document.getElementById('btnResign').style.display = liveHumanGame ? 'inline-block' : 'none';
  const startButton = document.getElementById('startNewGameButton');
  if (startButton) startButton.disabled = !!(!reviewData && gameState && gameState.status === 'RUNNING');

  const snap = currentViewedSnapshot();
  const pieces = (snap ? snap.pieces : src.pieces) || {};
  const lastMove = snap ? snap.last_move : src.last_move;
  const isCheck = snap ? snap.is_check : src.is_check;
  const turn = snap ? snap.turn : src.turn;
  const captures = (snap ? snap.captures : src.captures) || {white: {}, black: {}};
  const snaps = activeSnapshots();
  const curPly = snap ? snap.ply : src.move_number;
  const maxPly = Math.max(0, snaps.length - 1);

  document.getElementById('plyIndicator').textContent = `Ply ${curPly} / ${maxPly}`;
  const inReview = isViewingHistoricalPly();
  document.getElementById('btnReturnLive').style.display = inReview ? 'block' : 'none';
  document.getElementById('reviewStatusBadge').textContent = reviewData
    ? 'Archived Game Review'
    : (inReview ? `History Preview (Ply ${curPly})` : 'Live Board');

  const wElo = Math.round(src.white_elo || src.target_elo || 1600);
  const bElo = Math.round(src.black_elo || src.target_elo || 1600);
  const isMvm = (src.mode || '').toLowerCase().includes('model vs model');
  const wDesc = isMvm ? 'White (Model)' : (src.human_color === 'white' ? 'White (Human)' : `White (Model • ${src.difficulty || ''})`);
  const bDesc = isMvm ? 'Black (Model)' : (src.human_color === 'black' ? 'Black (Human)' : `Black (Model • ${src.difficulty || ''})`);

  if (boardFlipped) {
    document.getElementById('topPlayerLabel').textContent = wDesc;
    document.getElementById('topPlayerElo').textContent = `Elo ${wElo}`;
    document.getElementById('bottomPlayerLabel').textContent = bDesc;
    document.getElementById('bottomPlayerElo').textContent = `Elo ${bElo}`;
  } else {
    document.getElementById('topPlayerLabel').textContent = bDesc;
    document.getElementById('topPlayerElo').textContent = `Elo ${bElo}`;
    document.getElementById('bottomPlayerLabel').textContent = wDesc;
    document.getElementById('bottomPlayerElo').textContent = `Elo ${wElo}`;
  }

  const ranks = boardFlipped ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const fileIndices = boardFlipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];

  const legalTargets = {};
  if (!inReview && selectedSquare && gameState) {
    for (const mv of gameState.legal_moves) {
      if (mv.from === selectedSquare) {
        legalTargets[mv.to] = mv;
      }
    }
  }

  const pendingSug = (!inReview && gameState) ? gameState.pending_suggestion : null;

  for (const r of ranks) {
    for (const f of fileIndices) {
      const sqName = FILES[f] + (r + 1);
      const isLight = (r + f) % 2 !== 0;
      const cell = document.createElement('div');
      cell.className = 'sq ' + (isLight ? 'light' : 'dark');

      if (lastMove && (lastMove.from === sqName || lastMove.to === sqName)) {
        cell.classList.add('last-move');
      }
      if (pendingSug && (pendingSug.from === sqName || pendingSug.to === sqName)) {
        cell.classList.add('suggested');
      }
      if (!inReview && selectedSquare === sqName) {
        cell.classList.add('selected');
      }

      const piece = pieces[sqName];
      if (piece && isCheck && piece.type === 'king' && piece.color === turn) {
        cell.classList.add('in-check');
      }

      if (f === fileIndices[0]) {
        const rk = document.createElement('span');
        rk.className = 'coord-rank';
        rk.textContent = (r + 1);
        cell.appendChild(rk);
      }
      if (r === ranks[7]) {
        const fl = document.createElement('span');
        fl.className = 'coord-file';
        fl.textContent = FILES[f];
        cell.appendChild(fl);
      }

      if (piece) {
        const pSpan = document.createElement('span');
        pSpan.className = 'piece ' + piece.color;
        pSpan.textContent = piece.glyph;
        cell.appendChild(pSpan);
      }

      if (legalTargets[sqName]) {
        const indicator = document.createElement('div');
        indicator.className = piece ? 'legal-ring' : 'legal-dot';
        cell.appendChild(indicator);
      }

      cell.onclick = () => onSquareClick(sqName);
      grid.appendChild(cell);
    }
  }

  const term = src.termination || { is_over: false };
  const concBanner = document.getElementById('conclusionBanner');
  if (term.is_over) {
    concBanner.classList.add('visible');
    document.getElementById('conclusionTitle').textContent = `🏁 ${term.title}`;
    document.getElementById('conclusionText').textContent = term.explanation;
  } else {
    concBanner.classList.remove('visible');
  }

  const sugBanner = document.getElementById('suggestionBanner');
  if (pendingSug) {
    sugBanner.classList.add('visible');
    const rkNum = pendingSug.rank || 1;
    document.getElementById('sugMoveSan').textContent =
      `#${rkNum} ${pendingSug.san} (${pendingSug.uci})`;
    document.getElementById('sugConfidencePill').textContent =
      `Confidence: ${(pendingSug.prob * 100).toFixed(1)}% • Score: ${pendingSug.score >= 0 ? '+' : ''}${pendingSug.score.toFixed(3)}`;
    const tagStr = (pendingSug.tags || []).join(', ') || 'positional';
    document.getElementById('sugReasoningText').textContent =
      `Strategy: ${pendingSug.strategic_mode} | Tactical tags: [${tagStr}] | Coach Elo: ${Math.round(pendingSug.suggestion_elo)}`;
  } else {
    sugBanner.classList.remove('visible');
  }

  document.getElementById('turnIndicator').textContent = term.is_over
    ? `${term.title}`
    : `Turn: ${turn.toUpperCase()}${isCheck ? ' (CHECK!)' : ''}`;
  document.getElementById('modeIndicator').textContent = src.mode || '';

  const fmtCap = (obj) => {
    const entries = Object.entries(obj || {});
    return entries.length ? entries.map(([k,v]) => `${k} ×${v}`).join(', ') : '—';
  };
  document.getElementById('capWhite').textContent = 'Captured by White: ' + fmtCap(captures.white);
  document.getElementById('capBlack').textContent = 'Captured by Black: ' + fmtCap(captures.black);
  document.getElementById('logPathBanner').textContent = 'Active Folder: ' + (src.game_dir || '');
  renderEngagementPanel(src);
  const chartCard = document.getElementById('humanChartCard');
  chartCard.style.display = isHumanVsModel(src) ? 'block' : 'none';
  if (isHumanVsModel(src)) renderTimingChart('humanTimingChart');

  const activeAnalysis = (snap && snap.analysis) ? snap.analysis : src.last_analysis;
  const tbody = document.getElementById('rankingBody');
  if (activeAnalysis && activeAnalysis.top_candidates) {
    const modeStr = activeAnalysis.strategic_mode || 'BALANCED';
    const matAdv = activeAnalysis.material_advantage_cp || 0;
    const selPol = activeAnalysis.selection_policy ? ` • ${activeAnalysis.selection_policy}` : '';
    document.getElementById('strategyModeBadge').textContent =
      `${modeStr} (${matAdv >= 0 ? '+' : ''}${matAdv} cp)${selPol}`;
    document.getElementById('rankingTableTitle').textContent =
      `📊 Model Candidate Rankings (Ply ${activeAnalysis.ply} • ${activeAnalysis.side.toUpperCase()} @ Elo ${Math.round(activeAnalysis.target_elo)})`;

    tbody.innerHTML = '';
    for (const c of activeAnalysis.top_candidates.slice(0, 7)) {
      const tr = document.createElement('tr');
      const rawSc = (c.raw_score !== undefined) ? c.raw_score : c.score;
      const adjSc = (c.tactical_adjustment !== undefined) ? c.tactical_adjustment : 0.0;
      tr.innerHTML = `
        <td>${c.rank}</td>
        <td><b>${c.san}</b></td>
        <td><code>${c.uci}</code></td>
        <td>${c.score >= 0 ? '+' : ''}${c.score.toFixed(4)}</td>
        <td>${rawSc >= 0 ? '+' : ''}${rawSc.toFixed(4)}</td>
        <td>${adjSc >= 0 ? '+' : ''}${adjSc.toFixed(2)}</td>
        <td>${(c.prob * 100).toFixed(1)}%</td>
        <td>${(c.tags || []).join(', ') || 'quiet'}</td>
      `;
      tbody.appendChild(tr);
    }
  } else {
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted);">No model analysis recorded for this ply yet.</td></tr>';
  }

  renderThinkingLog(src.thinking_log_tail);

  renderPgnTable();
}

async function onSquareClick(sqName) {
  if (busy || !gameState || gameState.is_game_over || isViewingHistoricalPly()) return;
  if (isHumanVsModel(gameState) && gameState.turn !== gameState.human_color) return;

  if (selectedSquare) {
    const matching = gameState.legal_moves.filter(m => m.from === selectedSquare && m.to === sqName);
    if (matching.length > 0) {
      let chosenUci = matching[0].uci;
      const queenPromo = matching.find(m => m.promotion === 'q');
      if (queenPromo) chosenUci = queenPromo.uci;
      selectedSquare = null;
      await submitHumanMove(chosenUci);
      return;
    }
  }

  const piece = gameState.pieces[sqName];
  if (piece && piece.color === gameState.turn) {
    selectedSquare = (selectedSquare === sqName) ? null : sqName;
  } else {
    selectedSquare = null;
  }
  renderAll();
}

async function askAiSuggestion() {
  if (busy || !gameState || gameState.is_game_over || isViewingHistoricalPly()) return;
  busy = true;
  try {
    gameState = await apiCall('/api/suggest_move', 'POST', {});
    renderAll();
  } catch (err) {
    alert(err.message);
  } finally {
    busy = false;
  }
}

async function approveSuggestion() {
  if (!gameState || !gameState.pending_suggestion) return;
  const uci = gameState.pending_suggestion.uci;
  await submitHumanMove(uci);
}

async function rejectSuggestion() {
  if (!gameState) return;
  busy = true;
  try {
    gameState = await apiCall('/api/reject_suggestion', 'POST', {});
    renderAll();
  } catch (err) {
    console.error(err);
  } finally {
    busy = false;
  }
}

async function submitHumanMove(uci, showInlineError=false) {
  busy = true;
  try {
    gameState = await apiCall('/api/human_move', 'POST', { move: uci });
    viewedPly = null;
    renderAll();
    if (!gameState.is_game_over && gameState.mode === 'Human vs Model' && gameState.turn !== gameState.human_color) {
      const delay = parseInt(document.getElementById('delayInput').value || '850', 10);
      setTimeout(() => triggerModelMove(), Math.min(1200, Math.max(250, delay)));
    }
  } catch (err) {
    if (showInlineError) showMoveError(err.message);
    else alert(err.message);
  } finally {
    busy = false;
  }
}

async function triggerModelMove() {
  if (busy || !gameState || gameState.is_game_over) return;
  busy = true;
  try {
    reviewData = null;
    viewedPly = null;
    gameState = await apiCall('/api/model_move', 'POST', {});
    renderAll();
  } catch (err) {
    console.error(err);
  } finally {
    busy = false;
  }
}

async function resignGame() {
  if (!gameState || !isHumanVsModel(gameState) || gameState.is_game_over || isViewingHistoricalPly()) return;
  if (!window.confirm('Resign this game? The result will be saved and cannot be resumed.')) return;
  busy = true;
  try {
    gameState = await apiCall('/api/resign', 'POST', {});
    viewedPly = null;
    await refreshSavedGames();
    renderAll();
  } catch (err) {
    alert(err.message);
  } finally {
    busy = false;
  }
}

async function confirmStartNewGame() {
  stopAutoPlay();
  closeNewGameModal();
  const mode = document.getElementById('modalModeSelect').value;
  const humanColor = document.getElementById('modalHumanColor').value;
  const humanElo = parseFloat(document.getElementById('modalHumanElo').value || '1500');
  const difficulty = document.getElementById('modalDifficulty').value;
  const whiteElo = parseFloat(document.getElementById('modalWhiteElo').value || '1800');
  const blackElo = parseFloat(document.getElementById('modalBlackElo').value || '1800');

  reviewData = null;
  viewedPly = null;
  selectedSquare = null;
  boardFlipped = (mode === 'Human vs Model' && humanColor === 'black');

  try {
    gameState = await apiCall('/api/new_game', 'POST', {
      mode: mode,
      human_color: humanColor,
      human_elo: humanElo,
      difficulty: difficulty,
      white_elo: whiteElo,
      black_elo: blackElo,
    });
  } catch (err) {
    alert(err.message);
    openNewGameModal();
    return;
  }
  await refreshSavedGames();
  renderAll();

  if (mode === 'Human vs Model' && humanColor === 'black') {
    setTimeout(() => triggerModelMove(), 500);
  } else if (mode === 'Model vs Model') {
    toggleAutoPlay();
  }
}

function flipBoard() {
  boardFlipped = !boardFlipped;
  renderAll();
}

function stopAutoPlay() {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer);
    autoPlayTimer = null;
  }
  document.getElementById('btnAutoPlay').textContent = '▶ Start MvM Auto-Play';
}

function toggleAutoPlay() {
  if (autoPlayTimer) {
    stopAutoPlay();
    return;
  }
  const delay = Math.max(300, parseInt(document.getElementById('delayInput').value || '950', 10));
  document.getElementById('btnAutoPlay').textContent = '⏸ Stop Auto-Play';
  autoPlayTimer = setInterval(async () => {
    if (!gameState || gameState.is_game_over || isViewingHistoricalPly()) {
      stopAutoPlay();
      return;
    }
    if (!busy) {
      await triggerModelMove();
    }
  }, delay);
}

async function refreshSavedGames() {
  try {
    const res = await apiCall('/api/list_saved_games');
    const sel = document.getElementById('savedGamesSelect');
    const currentVal = sel.value;
    sel.innerHTML = '<option value="">-- Select a past game to review on board --</option>';
    for (const g of (res.games || [])) {
      const opt = document.createElement('option');
      opt.value = g.game_id;
      opt.textContent = `${g.starred ? '★ ' : ''}[${g.status || 'UNKNOWN'}] [${g.category === 'human_vs_model' ? 'HvM' : 'MvM'}] ${g.game_id} (${g.result})`;
      sel.appendChild(opt);
    }
    sel.value = currentVal;
  } catch (err) {
    console.error(err);
  }
}

async function onSelectSavedGame() {
  const gid = document.getElementById('savedGamesSelect').value;
  if (!gid) {
    returnToLive();
    return;
  }
  stopAutoPlay();
  try {
    reviewData = await apiCall('/api/load_saved_game?game_id=' + encodeURIComponent(gid));
    viewedPly = reviewData.ply_snapshots.length - 1;
    renderAll();
  } catch (err) {
    alert(err.message);
  }
}

async function analyzeCustomFen() {
  const fen = document.getElementById('customFenInput').value.trim();
  busy = true;
  try {
    const res = await apiCall('/api/analyze', 'POST', {
      fen: fen || (gameState ? gameState.fen : ''),
      target_elo: gameState ? gameState.target_elo : 1800,
    });
    if (gameState) {
      gameState.last_analysis = res.analysis;
      renderAll();
    }
  } catch (err) {
    alert(err.message);
  } finally {
    busy = false;
  }
}

async function init() {
  const status = await apiCall('/api/status');
  document.getElementById('runtimeBadge').textContent =
    `${status.model_variant} • ${status.device.toUpperCase()}`;
  gameState = await apiCall('/api/game_state');
  await refreshSavedGames();
  renderAll();
  if (!gameState.game_id) openNewGameModal();
}

init();
</script>
</body>
</html>
"""


def create_api_handler(session: LiveGameSession):
    class ChessAPIHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = WEB_UI_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/status":
                self._send_json({
                    "status": "ok",
                    "model_version": MODEL_VERSION,
                    "model_variant": session.bundle["model_variant"],
                    "is_elo_conditioned": session.bundle["is_elo_conditioned"],
                    "has_film": session.bundle.get("has_film", False),
                    "checkpoint_path": session.bundle["checkpoint_path"],
                    "checkpoint_epoch": session.bundle["checkpoint_epoch"],
                    "device": str(session.bundle["device"]),
                    "inference_dir": str(session.inference_dir),
                    "elo_buckets": EXPECTED_ELO_BUCKETS,
                })
                return

            if parsed.path == "/api/game_state":
                self._send_json(session.get_state())
                return

            if parsed.path == "/api/list_saved_games":
                self._send_json({"games": session.list_saved_games()})
                return

            if parsed.path == "/api/load_saved_game":
                qs = parse_qs(parsed.query)
                gid = (qs.get("game_id") or [""])[0]
                try:
                    self._send_json(session.load_saved_game_for_review(gid))
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status_code=404)
                return

            if parsed.path == "/api/annotated_pgn":
                qs = parse_qs(parsed.query)
                gid = (qs.get("game_id") or [""])[0]
                try:
                    self._send_json(session.get_annotated_pgn(gid or None))
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status_code=404)
                return

            self._send_json({"error": f"Unknown endpoint {parsed.path}"}, status_code=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                data = self._read_json_body()
                if parsed.path == "/api/new_game":
                    state = session.start_new_game(
                        mode=str(data.get("mode", "Human vs Model")),
                        human_color=str(data.get("human_color", "white")),
                        human_elo=float(data.get("human_elo", 1500.0)),
                        difficulty=str(data.get("difficulty", "Medium")),
                        white_elo=float(data["white_elo"]) if data.get("white_elo") is not None else None,
                        black_elo=float(data["black_elo"]) if data.get("black_elo") is not None else None,
                        target_elo=float(data["target_elo"]) if data.get("target_elo") is not None else None,
                        starting_fen=data.get("fen"),
                    )
                    self._send_json(state)
                    return

                if parsed.path == "/api/suggest_move":
                    elo_val = data.get("target_elo")
                    state = session.request_ai_suggestion(
                        suggestion_elo=float(elo_val) if elo_val is not None else None
                    )
                    self._send_json(state)
                    return

                if parsed.path == "/api/reject_suggestion":
                    state = session.reject_ai_suggestion()
                    self._send_json(state)
                    return

                if parsed.path in ("/api/human_move", "/api/move"):
                    move_str = str(data.get("move", "")).strip()
                    state = session.apply_human_move(move_str)
                    self._send_json(state)
                    return

                if parsed.path == "/api/model_move":
                    elo_val = data.get("target_elo")
                    state = session.play_model_move(
                        target_elo_override=float(elo_val) if elo_val is not None else None
                    )
                    self._send_json(state)
                    return

                if parsed.path == "/api/resign":
                    self._send_json(session.resign_human_game())
                    return

                if parsed.path == "/api/update_game_review":
                    game_id = str(data.get("game_id", "")).strip()
                    starred = data.get("starred") if "starred" in data else None
                    notes = data.get("notes") if "notes" in data else None
                    metadata = session.update_game_review(
                        game_id,
                        starred=bool(starred) if starred is not None else None,
                        notes=str(notes) if notes is not None else None,
                    )
                    self._send_json(metadata)
                    return

                if parsed.path == "/api/analyze":
                    fen = str(data.get("fen", "")).strip()
                    moves = data.get("moves", [])
                    if isinstance(moves, str):
                        moves = moves.split()
                    elo_val = float(data.get("target_elo", DEFAULT_TARGET_ELO))
                    result = session.analyze_custom_position(fen, list(moves), target_elo=elo_val)
                    self._send_json(result)
                    return

                self._send_json({"error": f"Unknown POST endpoint {parsed.path}"}, status_code=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status_code=400)

    return ChessAPIHandler


def find_free_port(preferred: int = 7860) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_web_and_api_server(session: LiveGameSession, env_mode: str) -> Tuple[ThreadingHTTPServer, str]:
    port = find_free_port(7860)
    handler_cls = create_api_handler(session)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    web_url = f"http://127.0.0.1:{port}"
    if env_mode == "colab" and COLAB_AVAILABLE and colab_output is not None:
        try:
            proxy_url = colab_output.eval_js(f"google.colab.kernel.proxyPort({port})")
            if proxy_url:
                web_url = str(proxy_url)
        except Exception:
            pass

    return server, web_url


# ============================================================
# 12. INTERACTIVE TERMINAL FLOWS (WITH DIFFICULTY & SUGGESTIONS)
# ============================================================

def prompt_color(default: str = "white") -> str:
    while True:
        val = input(f"Choose your color (white/black) [{default}]: ").strip().lower()
        if not val:
            return default
        if val in ("white", "black"):
            return val
        print("Please enter 'white' or 'black'.")


def prompt_difficulty(default: str = "Medium") -> str:
    options = {"1": "Easy", "2": "Medium", "3": "Hard", "4": "Expert",
               "easy": "Easy", "medium": "Medium", "hard": "Hard", "expert": "Expert"}
    while True:
        raw = input(f"Choose Difficulty [1=Easy, 2=Medium, 3=Hard, 4=Expert] [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return options[raw]
        print("Please enter Easy, Medium, Hard, or Expert (or 1-4).")


def prompt_target_elo(prompt_label: str = "Target Elo", default: float = 1500.0) -> float:
    while True:
        raw = input(f"{prompt_label} (1000-2400) [{int(default)}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if 600 <= val <= 3200:
                return val
        except ValueError:
            pass
        print("Please enter a valid numeric Elo (e.g. 1200, 1500, 1800, 2000).")


def play_human_vs_model_cli(bundle: Dict[str, Any], inference_dir: Path, web_url: str) -> None:
    human_color = prompt_color("white")
    model_color = "black" if human_color == "white" else "white"
    human_elo = prompt_target_elo("Enter Your Human Elo (Expertness Score)", 1500.0)
    difficulty = prompt_difficulty("Medium")
    target_elo = map_difficulty_to_model_elo(human_elo, difficulty)
    white_elo = human_elo if human_color == "white" else target_elo
    black_elo = target_elo if human_color == "white" else human_elo

    board = chess.Board()
    history_moves: List[Dict[str, Any]] = []
    repetition = Counter({board_key(board): 1})
    material_history = [material_difference(board)]
    rejected_ucis: Set[str] = set()
    logger = GameLogger(
        inference_dir,
        "Human vs Model",
        target_elo=target_elo,
        human_color=human_color,
        human_elo=human_elo,
        difficulty=difficulty,
        white_elo=white_elo,
        black_elo=black_elo,
    )

    init_svg = logger.board_dir / "board_0000.svg"
    render_board_matplotlib(board, init_svg, move_number=0, target_elo=target_elo, logger=logger, inline=True)
    print_terminal_board(board)

    print("\n" + "=" * 74)
    print(
        f"HUMAN ({human_color.title()} @ Elo {human_elo:.0f}) vs "
        f"MODEL ({model_color.title()} • {difficulty} @ Elo {target_elo:.0f})"
    )
    print(f"Dedicated HvM folder       : {logger.game_dir}")
    print(f"Interactive Web Arena link : {web_url}")
    print("Commands: SAN (e4, Nf3) or UCI (e2e4), 'suggest' for AI move suggestion, 'quit'")
    print("=" * 74)

    while True:
        if board.is_game_over(claim_draw=True):
            term_info = describe_game_termination(board)
            logger.finish(board, term_info)
            print(f"\n🏁 {term_info['title']}")
            print(f"   Reason: {term_info['explanation']}")
            print(f"   Saved in: {logger.game_dir}")
            return

        side = "white" if board.turn == chess.WHITE else "black"
        if side == model_color:
            print(f"\nModel ({model_color.title()}, {difficulty} @ Elo {target_elo:.0f}) is thinking...")
            time.sleep(MODEL_MOVE_DELAY_SECONDS)
            memory = rebuild_memory_state_from_moves(board, history_moves, repetition, material_history)
            ranking, analysis = infer_ranked_moves(
                bundle, board, history_moves, memory,
                target_elo=target_elo, difficulty=difficulty,
                top_n=8, logger=logger, print_thoughts=True,
                repetition=repetition,
            )
            selected, sel_policy = select_move_by_elo_and_difficulty(
                ranking=ranking,
                ply=len(history_moves) + 1,
                target_elo=target_elo,
                difficulty=difficulty,
                purpose="MODEL_MOVE",
            )
            analysis["selection_policy"] = sel_policy
            analysis["selected_move_rank"] = selected["rank"]

            move = chess.Move.from_uci(selected["uci"])
            board_before = board.copy(stack=False)
            san = board.san(move)
            history_moves.append(history_move_from_board(board, move))
            board.push(move)
            repetition[board_key(board)] += 1
            material_history.append(material_difference(board))
            rejected_ucis.clear()
            logger.record_move(board_before, board, move, san, actor="model", ranking=ranking[:5], analysis=analysis)
            print_terminal_board(board, last_move=move)
            print(f"Model played: {san} ({move.uci()}) [policy={sel_policy}]")
            continue

        cmd = input(f"\nYour move ({human_color.title()}) [or 'suggest' / 'quit']: ").strip()
        if cmd.lower() == "quit":
            logger.finish(board, {"result": "*", "title": "Game Aborted", "explanation": "Player exited."})
            print(f"Game saved to: {logger.game_dir}")
            return
        if cmd.lower() in ("suggest", "hint"):
            memory = rebuild_memory_state_from_moves(board, history_moves, repetition, material_history)
            coach_elo = max(AI_COACH_SUGGESTION_ELO, human_elo, target_elo)
            ranking, _ = infer_ranked_moves(
                bundle, board, history_moves, memory,
                target_elo=coach_elo, difficulty="Expert", top_n=8,
                logger=logger, print_thoughts=True, repetition=repetition,
                purpose="AI_SUGGESTION_FOR_HUMAN",
            )
            sug, _ = select_move_by_elo_and_difficulty(
                ranking=ranking,
                ply=len(history_moves) + 1,
                target_elo=coach_elo,
                difficulty="Expert",
                purpose="AI_SUGGESTION_FOR_HUMAN",
                excluded_ucis=rejected_ucis,
            )
            confirm = input(
                f"💡 AI Coach (#{sug['rank']}) suggests {sug['san']} ({sug['uci']}) [prob={(sug['prob']*100):.1f}%]. "
                "Approve and play this move? (y/n): "
            ).strip().lower()
            if confirm in ("y", "yes"):
                cmd = sug["uci"]
            else:
                rejected_ucis.add(sug["uci"])
                print("Suggestion rejected. Type 'suggest' again for the next alternative, or enter your own move.")
                continue

        try:
            move = parse_user_move(board, cmd)
        except ValueError as exc:
            print(exc)
            continue

        board_before = board.copy(stack=False)
        san = board.san(move)
        history_moves.append(history_move_from_board(board, move))
        board.push(move)
        repetition[board_key(board)] += 1
        material_history.append(material_difference(board))
        rejected_ucis.clear()
        logger.record_move(board_before, board, move, san, actor="human")
        print_terminal_board(board, last_move=move)


def play_model_vs_model_cli(bundle: Dict[str, Any], inference_dir: Path) -> None:
    white_elo = prompt_target_elo("Enter Expertness Score (Elo) for White Model", 1800.0)
    black_elo = prompt_target_elo("Enter Expertness Score (Elo) for Black Model", 1800.0)

    board = chess.Board()
    history_moves: List[Dict[str, Any]] = []
    repetition = Counter({board_key(board): 1})
    material_history = [material_difference(board)]
    logger = GameLogger(
        inference_dir,
        "Model vs Model",
        target_elo=(white_elo + black_elo) / 2.0,
        white_elo=white_elo,
        black_elo=black_elo,
    )

    init_svg = logger.board_dir / "board_0000.svg"
    render_board_matplotlib(board, init_svg, move_number=0, target_elo=white_elo, logger=logger, inline=True)
    print_terminal_board(board)

    print(f"\nMODEL (White @ Elo {white_elo:.0f}) vs MODEL (Black @ Elo {black_elo:.0f})")
    print(f"Dedicated MvM folder: {logger.game_dir}")

    while True:
        if board.is_game_over(claim_draw=True):
            term_info = describe_game_termination(board)
            logger.finish(board, term_info)
            print(f"\n🏁 {term_info['title']}")
            print(f"   Reason: {term_info['explanation']}")
            return

        is_white = (board.turn == chess.WHITE)
        active_elo = white_elo if is_white else black_elo
        active_diff = infer_difficulty_from_elo(active_elo)
        logger.target_elo = active_elo

        time.sleep(MODEL_MOVE_DELAY_SECONDS)
        memory = rebuild_memory_state_from_moves(board, history_moves, repetition, material_history)
        ranking, analysis = infer_ranked_moves(
            bundle, board, history_moves, memory,
            target_elo=active_elo, difficulty=active_diff, top_n=8,
            logger=logger, print_thoughts=True,
            repetition=repetition,
        )
        selected, sel_policy = select_move_by_elo_and_difficulty(
            ranking=ranking,
            ply=len(history_moves) + 1,
            target_elo=active_elo,
            difficulty=active_diff,
            purpose="MODEL_MOVE",
        )
        analysis["selection_policy"] = sel_policy
        analysis["selected_move_rank"] = selected["rank"]

        move = chess.Move.from_uci(selected["uci"])
        board_before = board.copy(stack=False)
        san = board.san(move)
        history_moves.append(history_move_from_board(board, move))
        board.push(move)
        repetition[board_key(board)] += 1
        material_history.append(material_difference(board))
        logger.record_move(board_before, board, move, san, actor="model", ranking=ranking[:5], analysis=analysis)
        print_terminal_board(board, last_move=move)

        cmd = input("Press Enter for next move (or type 'quit' to stop): ").strip().lower()
        if cmd == "quit":
            logger.finish(board, {"result": "*", "title": "Game Aborted", "explanation": "Stopped by user."})
            print(f"Game saved to: {logger.game_dir}")
            return


def analyze_position_cli(session: LiveGameSession) -> None:
    print("\nANALYZE POSITION (v0.4 ELO-CONDITIONED + EXCHANGE SEE + 2-PLY MINIMAX)")
    fen = input("FEN (blank = starting position): ").strip()
    raw_moves = input("Previous moves, oldest to newest (space-separated SAN/UCI; blank = none): ").strip()
    target_elo = prompt_target_elo("Target Elo for analysis", DEFAULT_TARGET_ELO)
    tokens = raw_moves.split() if raw_moves else []

    try:
        res = session.analyze_custom_position(fen, tokens, target_elo=target_elo)
        board = chess.Board(res["fen"])
        print_terminal_board(board)
        print(f"\nAnalysis saved to folder: {res['game_dir']}")
        print(f"Thinking log file       : {res['thinking_log_path']}")
        log_text = Path(res["thinking_log_path"]).read_text(encoding="utf-8")
        print("\n" + log_text)
    except Exception as exc:
        print(f"Analysis failed: {exc}")


# ============================================================
# 13. MAIN ENTRY POINT
# ============================================================

def main() -> None:
    env_mode, checkpoint_path, inference_dir, bundle_path = configure_runtime_environment()
    bundle = load_v04_runtime(checkpoint_path, inference_dir, bundle_path)
    session = LiveGameSession(bundle, inference_dir)
    server, web_url = start_web_and_api_server(session, env_mode)

    print("\n" + "=" * 76)
    print("CHESS POLICY MODEL v0.4 — PRO CHESS ARENA & API SERVER READY")
    print("=" * 76)
    print(f"Environment         : {env_mode.upper()}")
    print(f"Inference device    : {bundle['device']}")
    print(f"Loaded checkpoint   : {bundle['checkpoint_path']} (epoch {bundle['checkpoint_epoch']})")
    print(f"Model variant       : {bundle['model_variant']} (FiLM={bundle.get('has_film', False)})")
    print("Reasoning Engine    : Elo-Stratified 2-Ply Minimax + Exchange SEE + Contempt")
    print(f"HvM Logs Folder     : {inference_dir / 'human_vs_model'}")
    print(f"MvM Logs Folder     : {inference_dir / 'model_vs_model'}")
    print("-" * 76)
    print("🌐 OPEN THIS LINK IN A NEW BROWSER TAB FOR THE FULL-WIDTH INTERACTIVE UI:")
    print(f"   👉 {web_url}")
    print("=" * 76)

    try:
        while True:
            print("\nTERMINAL MENU (Web Arena remains live in the background at the link above!):")
            print("  1. Play Human vs Model (Terminal + Difficulty Mapping + AI Move Suggestions)")
            print("  2. Watch Model vs Model (Terminal + Independent White/Black Elos)")
            print("  3. Analyze Custom FEN + Move History + Target Elo")
            print("  4. Keep Web UI Server Running (Play in Browser Tab)")
            print("  5. Exit")
            choice = input("Select option [1-5]: ").strip()
            if choice == "1":
                play_human_vs_model_cli(bundle, inference_dir, web_url)
            elif choice == "2":
                play_model_vs_model_cli(bundle, inference_dir)
            elif choice == "3":
                analyze_position_cli(session)
            elif choice == "4":
                print(f"\nWeb Arena is active! Open this link in a new tab:\n  👉 {web_url}")
                input("Press Enter here anytime to return to the terminal menu...")
            elif choice == "5":
                print("Shutting down server. Goodbye!")
                break
            else:
                print("Please choose 1, 2, 3, 4, or 5.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
