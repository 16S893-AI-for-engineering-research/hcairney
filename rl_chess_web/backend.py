"""Lightweight backend to play chess vs a policy RL agent from RLChessBot.

This server loads the policy network checkpoint from RLChessBot and serves
one move at a time via a simple HTTP API suitable for a minimal frontend UI.

Usage:
  python rl_chess_web/backend.py

Notes:
- The client frontend (public/easter-egg.js) calls /move with a JSON body:
  {"fen": "<FEN>"}
  and expects a response with {"move": "<uci>", "san": "<san>"}.
- The policy itself is loaded from the specified RLChessBot checkpoint.
- If the checkpoint cannot be loaded, the server falls back to a random legal move
  selector for robustness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import chess
import torch
from flask import Flask, jsonify, request

# Ensure RLChessBot is importable when this script runs on the user's machine.
RL_DIR = Path("/Users/hughcairney/ALD/RLChessBot").expanduser()
if RL_DIR.exists():
    sys.path.insert(0, str(RL_DIR))

try:
    from rl_chess.play.checkpoint import load_policy_network
    from rl_chess.play.agent import PolicyNetworkMoveSelector, RandomLegalMoveSelector
except Exception:
    load_policy_network = None  # type: ignore
    PolicyNetworkMoveSelector = None  # type: ignore
    RandomLegalMoveSelector = None  # type: ignore

# Checkpoint path provided by user
CHECKPOINT_PATH = Path("/Users/hughcairney/ALD/RLChessBot/runs/ppo/opening-extended-v2/seed-0/checkpoints/iteration-00003020.pt").expanduser()

app = Flask(__name__, static_folder=None, static_url_path=None)

selector = None


def build_selector() -> object:
    # Try to load a real policy network; otherwise fall back to a random selector.
    if load_policy_network is None or PolicyNetworkMoveSelector is None:
        print("Warning: RLChessBot policy modules not available. Falling back to random moves.")
        return RandomLegalMoveSelector(seed=42) if RandomLegalMoveSelector else None
    try:
        # Provide a reasonable architecture in case the checkpoint does not carry it.
        architecture = {
            "channels": 192,
            "num_residual_blocks": 5,
            "value_channels": 128,
            "value_hidden_size": 256,
        }
        model = load_policy_network(CHECKPOINT_PATH, device=torch.device("cpu"), fallback_architecture=architecture)
        selector = PolicyNetworkMoveSelector(model, device=torch.device("cpu"), temperature=0.0)
        print(f"Loaded RLChessBot policy from {CHECKPOINT_PATH}")
        return selector
    except Exception as exc:
        print(f"Warning: could not load policy checkpoint: {exc}")
        return RandomLegalMoveSelector(seed=42) if RandomLegalMoveSelector else None


def ensure_selector():
    global selector
    if selector is None:
        selector = build_selector()
    return selector


@app.route("/move", methods=["POST"])
def move_endpoint():  # pragma: no cover
    data = request.get_json(force=True, silent=True) or {}
    fen = data.get("fen")
    if fen is None:
        # If no FEN provided, use the standard starting position
        board = chess.Board()
    else:
        board = chess.Board(fen)

    sel = ensure_selector()
    if sel is None:
        return jsonify({"error": "selector not available"}), 500

    try:
        move = sel.choose_move(board)
        san = board.san(move)
        uci = move.uci()
        return jsonify({"move": uci, "san": san})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":  # pragma: no cover
    # Bind to all interfaces for local testing; port 5000 by default.
    app.run(host="0.0.0.0", port=5000)
