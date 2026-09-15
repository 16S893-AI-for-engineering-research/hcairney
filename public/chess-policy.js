// Pure-JS port of RLChessBot's board/move encoding (rl_chess/encoding/board.py
// and rl_chess/encoding/move.py). This lets the ONNX-exported policy network
// run entirely client-side with no backend, matching the exact tensor layout
// the model was trained on.
//
// Exposed as window.ChessPolicy = { encodeBoardTensor, encodeMoveAction,
// NUM_ACTIONS }.
(function () {
  "use strict";

  const NUM_CHANNELS = 19;
  const BOARD_SIZE = 8;
  const NUM_MOVE_PLANES = 73;
  const NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE * NUM_MOVE_PLANES; // 4672

  const PIECE_TYPE_OFFSET = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

  // Tensor-coordinate deltas, in the exact order used by the Python encoder.
  const SLIDING_DIRECTIONS = [
    [-1, 0], // north
    [-1, 1], // northeast
    [0, 1], // east
    [1, 1], // southeast
    [1, 0], // south
    [1, -1], // southwest
    [0, -1], // west
    [-1, -1], // northwest
  ];

  const KNIGHT_OFFSETS = [
    [-2, -1],
    [-2, 1],
    [-1, 2],
    [1, 2],
    [2, 1],
    [2, -1],
    [1, -2],
    [-1, -2],
  ];

  const UNDERPROMOTION_FILE_DELTAS = [-1, 0, 1];
  const UNDERPROMOTION_PIECES = ["n", "b", "r"];

  const SLIDING_PLANE_COUNT = 56;
  const KNIGHT_PLANE_START = 56;
  const UNDERPROMOTION_PLANE_START = 64;

  function squareStrToIndex(square) {
    // square like "e4" -> python-chess-style index in [0, 63].
    const fileIndex = square.charCodeAt(0) - 97; // 'a' -> 0
    const rank = parseInt(square[1], 10);
    return (rank - 1) * 8 + fileIndex;
  }

  function orientSquare(squareIndex, isWhitePerspective) {
    return isWhitePerspective ? squareIndex : 63 - squareIndex;
  }

  function orientedToRowCol(orientedSquareIndex) {
    const fileIndex = orientedSquareIndex % 8;
    const rankIndex = Math.floor(orientedSquareIndex / 8);
    const row = 7 - rankIndex;
    const column = fileIndex;
    return [row, column];
  }

  function fillPlane(tensor, channel, value) {
    const offset = channel * BOARD_SIZE * BOARD_SIZE;
    for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
      tensor[offset + i] = value;
    }
  }

  /**
   * Encode a chess.js Chess instance into a Float32Array of shape
   * [19, 8, 8] (flattened, channel-major), matching
   * rl_chess.encoding.board.encode_board exactly.
   *
   * @param {Chess} chessInstance - a chess.js Chess() instance.
   * @param {string[]} [positionHistoryKeys] - array of "piece placement +
   *   turn + castling + en passant" FEN prefixes for every position reached
   *   so far in the game, INCLUDING the current position. Used to compute
   *   the repetition channel. If omitted, repetition is always 0.
   */
  function encodeBoardTensor(chessInstance, positionHistoryKeys) {
    const tensor = new Float32Array(NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE);
    const turn = chessInstance.turn(); // 'w' or 'b'
    const isWhite = turn === "w";
    const boardArray = chessInstance.board(); // row0=rank8 ... row7=rank1

    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const cell = boardArray[row][col];
        if (!cell) continue;

        const rank = 8 - row; // row0 -> rank8
        const fileIndex = col;
        const squareIndex = (rank - 1) * 8 + fileIndex;
        const orientedSquare = orientSquare(squareIndex, isWhite);
        const [tRow, tCol] = orientedToRowCol(orientedSquare);

        const pieceOffset = PIECE_TYPE_OFFSET[cell.type];
        const isCurrentPlayerPiece = cell.color === turn;
        const channel = isCurrentPlayerPiece
          ? pieceOffset
          : 6 + pieceOffset;

        tensor[channel * 64 + tRow * 8 + tCol] = 1.0;
      }
    }

    const opponentColor = isWhite ? "b" : "w";
    const ourRights = chessInstance.getCastlingRights(turn);
    const oppRights = chessInstance.getCastlingRights(opponentColor);
    fillPlane(tensor, 12, ourRights.k ? 1 : 0);
    fillPlane(tensor, 13, ourRights.q ? 1 : 0);
    fillPlane(tensor, 14, oppRights.k ? 1 : 0);
    fillPlane(tensor, 15, oppRights.q ? 1 : 0);

    const fenParts = chessInstance.fen().split(" ");

    // chess.js only reports an en passant target square in its FEN output
    // when a capture is immediately legal there (matching python-chess's
    // default "legal" FEN mode). python-chess's `board.ep_square` (used by
    // the original encoder) is set after *any* two-square pawn push,
    // regardless of capturability. We reproduce that broader behavior by
    // inspecting the last played move directly instead of relying on FEN.
    const history = chessInstance.history({ verbose: true });
    if (history.length > 0) {
      const lastMove = history[history.length - 1];
      const isDoublePawnPush =
        lastMove.piece === "p" &&
        Math.abs(
          parseInt(lastMove.to[1], 10) - parseInt(lastMove.from[1], 10)
        ) === 2;
      if (isDoublePawnPush) {
        const fromRank = parseInt(lastMove.from[1], 10);
        const toRank = parseInt(lastMove.to[1], 10);
        const epRank = (fromRank + toRank) / 2;
        const epSquareStr = lastMove.from[0] + epRank;
        const epSquareIndex = squareStrToIndex(epSquareStr);
        const orientedEp = orientSquare(epSquareIndex, isWhite);
        const [tRow, tCol] = orientedToRowCol(orientedEp);
        tensor[16 * 64 + tRow * 8 + tCol] = 1.0;
      }
    }

    const halfmoveClock = parseInt(fenParts[4], 10) || 0;
    const normalizedClock = Math.min(halfmoveClock, 100) / 100.0;
    fillPlane(tensor, 17, normalizedClock);

    let repetition = 0;
    if (positionHistoryKeys && positionHistoryKeys.length) {
      const key = fenParts.slice(0, 4).join(" ");
      let occurrences = 0;
      for (const existing of positionHistoryKeys) {
        if (existing === key) occurrences++;
      }
      repetition = occurrences >= 2 ? 1 : 0;
    }
    fillPlane(tensor, 18, repetition);

    return tensor;
  }

  function encodeSlidingPlane(rowDelta, columnDelta) {
    const distance = Math.max(Math.abs(rowDelta), Math.abs(columnDelta));
    if (distance === 0 || distance > 7) {
      throw new Error("Move has an invalid sliding distance.");
    }
    const isStraight = rowDelta === 0 || columnDelta === 0;
    const isDiagonal = Math.abs(rowDelta) === Math.abs(columnDelta);
    if (!isStraight && !isDiagonal) {
      throw new Error("Move is neither sliding nor knight-shaped.");
    }
    const sign = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);
    const direction = [sign(rowDelta), sign(columnDelta)];
    const directionIndex = SLIDING_DIRECTIONS.findIndex(
      (d) => d[0] === direction[0] && d[1] === direction[1]
    );
    if (directionIndex < 0) {
      throw new Error("Move has an unsupported direction.");
    }
    return directionIndex * 7 + (distance - 1);
  }

  function encodeUnderpromotionPlane(rowDelta, columnDelta, promotion) {
    if (rowDelta !== -1 || !UNDERPROMOTION_FILE_DELTAS.includes(columnDelta)) {
      throw new Error("Underpromotion has an invalid movement direction.");
    }
    const directionIndex = UNDERPROMOTION_FILE_DELTAS.indexOf(columnDelta);
    const pieceIndex = UNDERPROMOTION_PIECES.indexOf(promotion);
    return (
      UNDERPROMOTION_PLANE_START +
      directionIndex * UNDERPROMOTION_PIECES.length +
      pieceIndex
    );
  }

  /**
   * Encode a chess.js verbose move (with .from, .to, and optional
   * .promotion) as an integer action index in [0, 4672), matching
   * rl_chess.encoding.move.encode_move. `turnColor` must be the color of the
   * player making the move ('w' or 'b').
   */
  function encodeMoveAction(move, turnColor) {
    const isWhite = turnColor === "w";
    const fromIndex = squareStrToIndex(move.from);
    const toIndex = squareStrToIndex(move.to);
    const fromOriented = orientSquare(fromIndex, isWhite);
    const toOriented = orientSquare(toIndex, isWhite);
    const [fromRow, fromCol] = orientedToRowCol(fromOriented);
    const [toRow, toCol] = orientedToRowCol(toOriented);
    const rowDelta = toRow - fromRow;
    const columnDelta = toCol - fromCol;

    let plane;
    const promotion = move.promotion;
    if (promotion && UNDERPROMOTION_PIECES.includes(promotion)) {
      plane = encodeUnderpromotionPlane(rowDelta, columnDelta, promotion);
    } else {
      const knightIndex = KNIGHT_OFFSETS.findIndex(
        (d) => d[0] === rowDelta && d[1] === columnDelta
      );
      if (knightIndex >= 0) {
        plane = KNIGHT_PLANE_START + knightIndex;
      } else {
        plane = encodeSlidingPlane(rowDelta, columnDelta);
      }
    }

    const originIndex = fromRow * BOARD_SIZE + fromCol;
    return originIndex * NUM_MOVE_PLANES + plane;
  }

  window.ChessPolicy = {
    NUM_ACTIONS,
    NUM_CHANNELS,
    BOARD_SIZE,
    encodeBoardTensor,
    encodeMoveAction,
  };
})();
