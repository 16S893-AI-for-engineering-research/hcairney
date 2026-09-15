// Easter egg: click the word "chess" on the About page to play a real game
// of chess against an RL policy network — entirely in the browser, no
// backend required. The opponent is an ONNX export of a PPO-trained
// PolicyValueNetwork from RLChessBot (see /tools/export_policy_to_onnx.py in
// the repo for provenance). Board rules are enforced by chess.js; the board
// tensor/action encoding in chess-policy.js is a verified port of
// RLChessBot's Python encoders (see tools/export_policy_to_onnx.py and the
// dev log for verification details).
(function () {
  "use strict";

  const BASE_URL = window.__SITE_BASE_URL__ || "/";

  function assetUrl(path) {
    return BASE_URL.replace(/\/$/, "") + "/" + path.replace(/^\//, "");
  }

  let scriptsLoadingPromise = null;
  let sessionPromise = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[data-easter-src="${src}"]`);
      if (existing) {
        if (existing.dataset.loaded === "true") {
          resolve();
        } else {
          existing.addEventListener("load", () => resolve());
          existing.addEventListener("error", reject);
        }
        return;
      }
      const script = document.createElement("script");
      script.src = src;
      script.dataset.easterSrc = src;
      script.addEventListener("load", () => {
        script.dataset.loaded = "true";
        resolve();
      });
      script.addEventListener("error", reject);
      document.head.appendChild(script);
    });
  }

  function ensureScriptsLoaded() {
    if (!scriptsLoadingPromise) {
      scriptsLoadingPromise = Promise.all([
        loadScript(assetUrl("vendor/chess.iife.js")),
        loadScript(assetUrl("chess-policy.js")),
        loadScript(assetUrl("ort/ort.wasm.min.js")),
      ]).then(() => {
        if (!window.ort) {
          throw new Error("onnxruntime-web failed to load.");
        }
        window.ort.env.wasm.wasmPaths = assetUrl("ort/");
        // Single-threaded WASM avoids requiring cross-origin isolation
        // headers, which static hosts like GitHub Pages do not send.
        window.ort.env.wasm.numThreads = 1;
      });
    }
    return scriptsLoadingPromise;
  }

  function ensureSession() {
    if (!sessionPromise) {
      sessionPromise = ensureScriptsLoaded().then(() =>
        window.ort.InferenceSession.create(assetUrl("model/policy.onnx"), {
          executionProviders: ["wasm"],
        })
      );
    }
    return sessionPromise;
  }

  function bindChessTrigger() {
    const el = document.getElementById("easter-chess");
    if (!el) return;
    el.style.cursor = "pointer";
    el.addEventListener("click", () => {
      openChessPopup();
    });
  }

  function positionKey(chessInstance) {
    return chessInstance.fen().split(" ").slice(0, 4).join(" ");
  }

  async function chooseAgentMove(chessInstance, positionHistoryKeys) {
    const session = await ensureSession();
    const ChessPolicy = window.ChessPolicy;

    const tensor = ChessPolicy.encodeBoardTensor(chessInstance, positionHistoryKeys);
    const inputTensor = new window.ort.Tensor("float32", tensor, [
      1,
      ChessPolicy.NUM_CHANNELS,
      ChessPolicy.BOARD_SIZE,
      ChessPolicy.BOARD_SIZE,
    ]);

    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const policyLogits = results[session.outputNames[0]].data;

    const legalMoves = chessInstance.moves({ verbose: true });
    if (legalMoves.length === 0) {
      throw new Error("No legal moves available.");
    }

    const turnColor = chessInstance.turn();
    let bestMove = null;
    let bestLogit = -Infinity;
    for (const move of legalMoves) {
      const candidateMoves =
        move.promotion || !isPromotionCandidate(move)
          ? [move]
          : ["q", "n", "b", "r"].map((p) => ({ ...move, promotion: p }));
      for (const candidate of candidateMoves) {
        let action;
        try {
          action = ChessPolicy.encodeMoveAction(candidate, turnColor);
        } catch (err) {
          continue;
        }
        const logit = policyLogits[action];
        if (logit > bestLogit) {
          bestLogit = logit;
          bestMove = candidate;
        }
      }
    }

    if (!bestMove) {
      throw new Error("Policy could not select a legal move.");
    }
    return bestMove;
  }

  function isPromotionCandidate(move) {
    // chess.js verbose moves already set .promotion for pawn moves that
    // reach the final rank when generated via .moves({verbose:true}), but
    // it defaults to queen. We need every underpromotion action's logit too,
    // so re-derive candidates whenever a pawn move lands on rank 1 or 8.
    if (move.piece !== "p") return false;
    const rank = move.to[1];
    return rank === "1" || rank === "8";
  }

  function pieceUnicode(piece) {
    const blackMap = {
      p: "♟",
      n: "♞",
      b: "♝",
      r: "♜",
      q: "♛",
      k: "♚",
    };
    const whiteMap = {
      p: "♙",
      n: "♘",
      b: "♗",
      r: "♖",
      q: "♕",
      k: "♔",
    };
    const map = piece.color === "w" ? whiteMap : blackMap;
    return map[piece.type] || "";
  }

  function openChessPopup() {
    if (document.getElementById("easter-chess-modal")) return;

    const overlay = document.createElement("div");
    overlay.id = "easter-chess-modal";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(2,6,23,0.92)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.zIndex = "1000";
    overlay.style.padding = "16px";

    const container = document.createElement("div");
    container.style.width = "min(92vw, 520px)";
    container.style.background = "#111827";
    container.style.border = "1px solid #334155";
    container.style.borderRadius = "10px";
    container.style.padding = "14px";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "10px";
    container.style.fontFamily =
      "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

    const title = document.createElement("div");
    title.style.display = "flex";
    title.style.justifyContent = "space-between";
    title.style.alignItems = "center";
    title.style.color = "#f1f5f9";
    title.style.fontWeight = "600";
    title.innerHTML =
      '<span>Chess vs. an RL policy \u2014 you are White</span>' +
      '<button id="close-easter-chess" style="padding:6px 10px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;cursor:pointer;font-family:inherit;">Close</button>';

    const statusLine = document.createElement("div");
    statusLine.id = "easter-chess-status";
    statusLine.style.color = "#94a3b8";
    statusLine.style.fontSize = "13px";
    statusLine.style.minHeight = "18px";
    statusLine.textContent = "Loading policy network\u2026";

    const boardDiv = document.createElement("div");
    boardDiv.style.display = "grid";
    boardDiv.style.gridTemplateColumns = "repeat(8, 1fr)";
    boardDiv.style.gridTemplateRows = "repeat(8, 1fr)";
    boardDiv.style.border = "2px solid #475569";
    boardDiv.style.borderRadius = "6px";
    boardDiv.style.overflow = "hidden";
    boardDiv.style.width = "100%";
    boardDiv.style.aspectRatio = "1 / 1";

    const helpLine = document.createElement("div");
    helpLine.style.color = "#64748b";
    helpLine.style.fontSize = "12px";
    helpLine.textContent =
      "Click a piece, then click a destination square. Promotions are auto-queen.";

    container.appendChild(title);
    container.appendChild(statusLine);
    container.appendChild(boardDiv);
    container.appendChild(helpLine);
    overlay.appendChild(container);
    document.body.appendChild(overlay);

    document.getElementById("close-easter-chess").addEventListener("click", () => {
      overlay.remove();
    });

    ensureScriptsLoaded()
      .then(() => runGame(boardDiv, statusLine))
      .catch((err) => {
        console.error(err);
        statusLine.textContent =
          "Failed to load the chess engine. See console for details.";
        statusLine.style.color = "#f87171";
      });
  }

  function runGame(boardDiv, statusLine) {
    const Chess = window.ChessJS.Chess;
    const game = new Chess();
    let positionHistoryKeys = [positionKey(game)];
    let selectedSquare = null;
    let agentThinking = false;

    function squareAt(row, col) {
      const file = "abcdefgh"[col];
      const rank = 8 - row;
      return file + rank;
    }

    function renderBoard() {
      boardDiv.innerHTML = "";
      const boardArray = game.board();
      const legalFromSelected = selectedSquare
        ? game.moves({ square: selectedSquare, verbose: true }).map((m) => m.to)
        : [];

      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const square = squareAt(row, col);
          const cell = document.createElement("div");
          cell.style.position = "relative";
          cell.style.width = "100%";
          cell.style.paddingBottom = "100%";
          cell.style.boxSizing = "border-box";
          const isLight = (row + col) % 2 === 0;
          cell.style.background = isLight ? "#cdeaff" : "#1e3a8a";
          if (square === selectedSquare) {
            cell.style.outline = "3px solid #38bdf8";
            cell.style.outlineOffset = "-3px";
          } else if (legalFromSelected.includes(square)) {
            cell.style.boxShadow = "inset 0 0 0 3px rgba(56, 189, 248, 0.55)";
          }

          const inner = document.createElement("div");
          inner.style.position = "absolute";
          inner.style.inset = "0";
          inner.style.display = "flex";
          inner.style.alignItems = "center";
          inner.style.justifyContent = "center";
          inner.style.fontSize = "min(6vw, 32px)";
          inner.style.userSelect = "none";
          inner.style.cursor = "pointer";

          const piece = boardArray[row][col];
          if (piece) {
            inner.textContent = pieceUnicode(piece);
            inner.style.color = isLight ? "#000" : "#fff";
          }

          cell.appendChild(inner);
          cell.addEventListener("click", () => onSquareClick(square));
          boardDiv.appendChild(cell);
        }
      }
    }

    function setStatus(text, isError) {
      statusLine.textContent = text;
      statusLine.style.color = isError ? "#f87171" : "#94a3b8";
    }

    function describeGameOver() {
      if (game.isCheckmate()) {
        return game.turn() === "w"
          ? "Checkmate \u2014 the RL policy wins."
          : "Checkmate \u2014 you win!";
      }
      if (game.isStalemate()) return "Draw by stalemate.";
      if (game.isThreefoldRepetition()) return "Draw by threefold repetition.";
      if (game.isInsufficientMaterial()) return "Draw by insufficient material.";
      if (game.isDraw()) return "Draw by the fifty-move rule.";
      return "Game over.";
    }

    async function onSquareClick(square) {
      if (agentThinking || game.isGameOver()) return;
      if (game.turn() !== "w") return;

      if (selectedSquare === square) {
        selectedSquare = null;
        renderBoard();
        return;
      }

      if (selectedSquare) {
        const legalMoves = game.moves({ square: selectedSquare, verbose: true });
        const target = legalMoves.find((m) => m.to === square);
        if (target) {
          game.move({ from: selectedSquare, to: square, promotion: "q" });
          positionHistoryKeys.push(positionKey(game));
          selectedSquare = null;
          renderBoard();

          if (game.isGameOver()) {
            setStatus(describeGameOver());
            return;
          }

          await playAgentMove();
          return;
        }

        const piece = game.get(square);
        if (piece && piece.color === "w") {
          selectedSquare = square;
        } else {
          selectedSquare = null;
        }
        renderBoard();
        return;
      }

      const piece = game.get(square);
      if (piece && piece.color === "w") {
        selectedSquare = square;
        renderBoard();
      }
    }

    async function playAgentMove() {
      agentThinking = true;
      setStatus("RL policy is thinking\u2026");
      try {
        const move = await chooseAgentMove(game, positionHistoryKeys);
        game.move({ from: move.from, to: move.to, promotion: move.promotion });
        positionHistoryKeys.push(positionKey(game));
        renderBoard();
        if (game.isGameOver()) {
          setStatus(describeGameOver());
        } else {
          setStatus("Your move.");
        }
      } catch (err) {
        console.error(err);
        setStatus("The policy failed to move. See console for details.", true);
      } finally {
        agentThinking = false;
      }
    }

    renderBoard();
    setStatus("Your move.");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindChessTrigger);
  } else {
    bindChessTrigger();
  }
})();
