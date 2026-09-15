// Easter egg: open a tiny chess popup when user clicks the word "chess" on the About page.
(function () {
  // Attach click handler to the element with id 'easter-chess' if present.
  function bindChessTrigger() {
    const el = document.getElementById("easter-chess");
    if (!el) return;
    el.style.cursor = "pointer";
    el.style.textDecoration = "none";
    el.addEventListener("click", () => {
      openChessPopup();
    });
  }

  function openChessPopup() {
    // If already open, do nothing.
    if (document.getElementById("easter-chess-modal")) return;

    const overlay = document.createElement("div");
    overlay.id = "easter-chess-modal";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(2,6,23,0.9)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.zIndex = "1000";

    const container = document.createElement("div");
    container.style.width = "min(90vw, 720px)";
    container.style.height = "min(70vh, 520px)";
    container.style.background = "#111";
    container.style.border = "1px solid #444";
    container.style.borderRadius = "8px";
    container.style.padding = "12px";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "8px";

    // Title bar
    const title = document.createElement("div");
    title.style.display = "flex";
    title.style.justifyContent = "space-between";
    title.style.alignItems = "center";
    title.style.color = "#eee";
    title.style.fontFamily = "monospace";
    title.style.fontWeight = "bold";
    title.innerHTML = `<span>Chess vs RL policy</span><button id="close-easter-chess" style="padding:6px 10px;border-radius:6px;border:1px solid #555;background:#333;color:#eee;cursor:pointer">Close</button>`;

    // Board container
    const boardDiv = document.createElement("div");
    boardDiv.id = "easter-board";
    boardDiv.style.display = "grid";
    boardDiv.style.gridTemplateColumns = "repeat(8, 1fr)";
    boardDiv.style.gridTemplateRows = "repeat(8, 1fr)";
    boardDiv.style.border = "1px solid #555";
    boardDiv.style.borderRadius = "6px";
    boardDiv.style.overflow = "hidden";
    boardDiv.style.width = "100%";
    boardDiv.style.height = "auto";

    // Build initial board (standard chess starting position)
    const initial = [
      ["r","n","b","q","k","b","n","r"],
      ["p","p","p","p","p","p","p","p"],
      [".",".",".",".",".",".",".","."],
      [".",".",".",".",".",".",".","."],
      [".",".",".",".",".",".",".","."],
      [".",".",".",".",".",".",".","."],
      ["P","P","P","P","P","P","P","P"],
      ["R","N","B","Q","K","B","N","R"],
    ];

    // Board state variable: 2D array of characters
    let board = initial.map((row) => row.slice());
    let selected = null; // {r,c}
    let turnWhite = true; // White to move first

    const pieceUnicode = {
      'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
      'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚',
      '.': ''
    };

    function renderBoard() {
      boardDiv.innerHTML = '';
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const cell = document.createElement("div");
          cell.style.width = "100%";
          cell.style.height = "0";
          cell.style.paddingBottom = "12.5%"; // square
          cell.style.boxSizing = "border-box";
          const isLight = (r + c) % 2 === 0;
          cell.style.background = isLight ? "#f0d9b5" : "#b58863";
          cell.style.display = "flex";
          cell.style.alignItems = "center";
          cell.style.justifyContent = "center";
          cell.style.fontSize = "28px";
          cell.style.cursor = (board[r][c] !== '.') ? "pointer" : "default";
          cell.dataset.r = r;
          cell.dataset.c = c;

          const piece = board[r][c];
          cell.textContent = pieceUnicode[piece] || '';

          cell.addEventListener("click", () => {
            onCellClick(r, c);
          });

          boardDiv.appendChild(cell);
        }
      }
    }

    function onCellClick(r, c) {
      if (selected) {
        // attempt to move from selected to (r,c)
        const fr = selected.r, fc = selected.c;
        const piece = board[fr][fc];
        if (piece === '.' || piece === undefined) {
          selected = null; renderBoard(); return;
        }
        // only allow moving white pieces for user
        if (piece !== piece.toUpperCase()) {
          selected = null; renderBoard(); return;
        }
        const dst = board[r][c];
        if (dst === '.' || dst === dst.toLowerCase()) {
          // perform move
          board[r][c] = piece;
          board[fr][fc] = '.';
          selected = null;
          renderBoard();
          // after white move, request black move from server
          turnWhite = false; // now black to move
          requestBlackMove();
        } else {
          // can't capture own piece, ignore
          selected = null; renderBoard();
        }
      } else {
        // select piece if it's white
        if (board[r][c] && board[r][c] !== '.' && board[r][c] === board[r][c].toUpperCase()) {
          selected = { r, c };
        }
      }
    }

    function toFen() {
      // simple FEN generator from board (no castling rights, en passant, etc.)
      let fen = '';
      for (let r = 0; r < 8; r++) {
        let empty = 0;
        for (let c = 0; c < 8; c++) {
          const ch = board[r][c];
          if (ch === '.') {
            empty++;
          } else {
            if (empty) { fen += empty; empty = 0; }
            fen += ch;
          }
        }
        if (empty) fen += empty;
        if (r < 7) fen += '/';
      }
      fen += turnWhite ? ' w - - 0 1' : ' b - - 0 1';
      return fen;
    }

    async function requestBlackMove() {
      // send current FEN to server and apply response move (black's move) if provided
      const fen = toFen();
      try {
        const res = await fetch("http://127.0.0.1:5000/move", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fen: fen }),
        });
        const data = await res.json();
        if (data && data.move) {
          applyMoveFromUci(data.move);
        }
      } catch (e) {
        // fallback: simple random move for black if server unavailable
        fallbackBlackMove();
      }
      // After black move, white to move again
      turnWhite = true;
    }

    function applyMoveFromUci(uci) {
      if (!uci || uci.length < 4) return;
      const fileToIndex = { a:0, b:1, c:2, d:3, e:4, f:5, g:6, h:7 };
      const fromFile = uci.charAt(0);
      const fromRank = parseInt(uci.charAt(1), 10);
      const toFile = uci.charAt(2);
      const toRank = parseInt(uci.charAt(3), 10);
      const fr = 8 - fromRank;
      const fc = fileToIndex[fromFile];
      const tr = 8 - toRank;
      const tc = fileToIndex[toFile];
      const pieceFrom = board[fr][fc];
      if (pieceFrom && pieceFrom !== '.') {
        board[tr][tc] = pieceFrom;
        board[fr][fc] = '.';
      }
      // Blue: after black move, white to move again
      turnWhite = true;
      renderBoard();
    }

    function fallbackBlackMove() {
      // naive fallback: move any black piece forward if possible
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const piece = board[r][c];
          if (piece && piece !== '.' && piece === piece.toLowerCase()) {
            // try simple forward by 1
            const nr = r - 1;
            if (nr >= 0 && board[nr][c] === '.') {
              board[nr][c] = piece; board[r][c] = '.'; renderBoard(); return;
            }
          }
        }
      }
    }

    function onClose() {
      overlay.remove();
    }

    // Build DOM
    container.appendChild(title);
    container.appendChild(boardDiv);
    const help = document.createElement("div");
    help.style.fontFamily = "monospace";
    help.style.fontSize = "12px";
    help.style.color = "#ddd";
    help.textContent = "White to move: click a white piece, then a destination. Black will respond automatically.";
    container.appendChild(help);

    overlay.appendChild(container);

    document.getElementById("close-easter-chess")?.addEventListener("click", onClose);

    renderBoard();
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindChessTrigger);
  } else {
    bindChessTrigger();
  }
})();
