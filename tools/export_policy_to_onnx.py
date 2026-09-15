"""One-time export tool: converts an RLChessBot PPO policy checkpoint into a
portable ONNX model that can run entirely in the browser via onnxruntime-web.

This script is a *developer* utility. It is NOT required by anyone who clones
this Astro site — the exported .onnx file is committed under
rl_chess_web/model/policy.onnx and loaded directly by the browser at runtime.
Only the site owner needs to re-run this script if the checkpoint changes.

Usage (run with the RLChessBot conda/virtualenv that has torch + python-chess
installed):

    python tools/export_policy_to_onnx.py \
        --rlchessbot-dir /Users/hughcairney/ALD/RLChessBot \
        --checkpoint runs/ppo/opening-extended-v2/seed-0/checkpoints/iteration-00003020.pt \
        --output rl_chess_web/model/policy.onnx

The script:
  1. Imports rl_chess.model.network.PolicyValueNetwork from the RLChessBot
     checkout (path supplied via --rlchessbot-dir).
  2. Loads the checkpoint's architecture + state_dict via the same
     rl_chess.play.checkpoint.load_policy_network helper RLChessBot uses for
     interactive play, so the exported model is bit-for-bit the same network.
  3. Exports the model to ONNX with a fixed input shape [1, 19, 8, 8]
     (batch, channel, row, column), matching rl_chess.encoding.board.encode_board.
  4. Verifies the ONNX model's outputs match the PyTorch model on a handful of
     random legal positions before writing the final file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rlchessbot-dir",
        type=Path,
        default=Path("/Users/hughcairney/ALD/RLChessBot"),
        help="Path to a local RLChessBot checkout (for imports only).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pt checkpoint (absolute, or relative to --rlchessbot-dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "rl_chess_web"
        / "model"
        / "policy.onnx",
        help="Where to write the exported ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to target.",
    )
    args = parser.parse_args()

    rlchessbot_dir = args.rlchessbot_dir.expanduser().resolve()
    if not rlchessbot_dir.is_dir():
        raise SystemExit(f"RLChessBot directory not found: {rlchessbot_dir}")
    sys.path.insert(0, str(rlchessbot_dir))

    try:
        import chess  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "python-chess is not installed in this interpreter. Run this "
            "script with the RLChessBot project's environment."
        ) from exc

    from rl_chess.play.checkpoint import load_policy_network

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = rlchessbot_dir / checkpoint_path
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    model = load_policy_network(checkpoint_path, device=device)
    model.eval()

    dummy_input = torch.zeros((1, 19, 8, 8), dtype=torch.float32)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting checkpoint {checkpoint_path} -> {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    _verify(model, output_path, device)
    print("Export verified. Wrote", output_path)


def _verify(model: torch.nn.Module, onnx_path: Path, device: torch.device) -> None:
    """Compare PyTorch and ONNX Runtime outputs on random legal positions."""
    import chess
    import onnxruntime as ort

    from rl_chess.encoding.board import encode_board

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    board = chess.Board()
    boards = [board.copy()]
    for _ in range(8):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = moves[int(rng.integers(0, len(moves)))]
        board.push(move)
        boards.append(board.copy())

    max_policy_diff = 0.0
    max_value_diff = 0.0
    with torch.inference_mode():
        for sample_board in boards:
            encoded = encode_board(sample_board, device=device).unsqueeze(0)
            torch_policy, torch_value = model(encoded)
            torch_policy = torch_policy.numpy()
            torch_value = torch_value.numpy()

            onnx_policy, onnx_value = session.run(
                ["policy_logits", "value"],
                {"board": encoded.numpy()},
            )

            max_policy_diff = max(
                max_policy_diff, float(np.abs(torch_policy - onnx_policy).max())
            )
            max_value_diff = max(
                max_value_diff, float(np.abs(torch_value - onnx_value).max())
            )

    print(f"Max |policy diff| across {len(boards)} positions: {max_policy_diff:.3e}")
    print(f"Max |value diff| across {len(boards)} positions: {max_value_diff:.3e}")
    if max_policy_diff > 1e-3 or max_value_diff > 1e-3:
        raise SystemExit(
            "ONNX export diverges from PyTorch model beyond tolerance; aborting."
        )


if __name__ == "__main__":
    main()
