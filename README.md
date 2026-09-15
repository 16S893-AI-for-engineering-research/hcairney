# hcairney — 16.S893 portfolio

Personal site for MIT 16.S893: AI Agents for Engineering Research. Built with
[Astro](https://astro.build/) + Tailwind CSS.

## Structure

- `src/pages/index.astro` — home
- `src/pages/about.astro` — bio
- `src/pages/project.astro` — semester project one-pager (placeholder until advisor sign-off)
- `src/pages/dev-log/` — weekly log of how I used an AI coding agent; new entries
  are added as Markdown files in `src/content/dev-log/`, no page code needed
- `src/layouts/BaseLayout.astro` — shared nav/header/footer
- `public/easter-egg.js` — click the word "chess" on the About page to play
  against an RL policy network, entirely client-side (see below)

## Development

```sh
npm install
npm run dev
npm run build
npm run preview
```

## Adding a dev-log entry

Add a new file to `src/content/dev-log/`, e.g. `week-02.md`:

```md
---
title: "Week 2: ..."
date: 2026-09-18
summary: "One-line summary for the index page."
---

Notes go here.
```

It will automatically appear on `/dev-log/` and get its own page.

## Chess easter egg

Clicking the word "chess" on the About page opens a small chess board where
you play White against an RL policy network. Everything runs client-side in
the browser — no backend server is required, so it works the same way for
anyone who clones or visits this site:

- `public/model/policy.onnx` — an ONNX export of a PPO-trained
  `PolicyValueNetwork` checkpoint from a private RL chess project. Move
  selection is greedy (highest policy logit among legal moves).
- `public/ort/` — a single-threaded build of
  [onnxruntime-web](https://github.com/microsoft/onnxruntime) (WASM), vendored
  so the site works on static hosts like GitHub Pages without
  cross-origin-isolation headers.
- `public/vendor/chess.iife.js` — [chess.js](https://github.com/jhlywa/chess.js),
  bundled as a browser global, used for legal move generation and game-over
  detection.
- `public/chess-policy.js` — a hand-written JS port of the training
  repository's board/action tensor encoders. It was verified to produce
  bit-identical tensors and action indices versus the original Python
  implementation across hundreds of randomly played games (including
  castling, en passant, promotions, and threefold repetition).
- `public/easter-egg.js` — wires the above together: click handler, board
  rendering, and the human/agent turn loop.

### Re-exporting the policy checkpoint

`tools/export_policy_to_onnx.py` is a one-time developer utility (not needed
by anyone just running the site) that converts a `.pt` checkpoint from the RL
chess training repo into `public/model/policy.onnx`. It requires that repo's
Python environment (PyTorch + python-chess + onnx + onnxruntime) and is run
like:

```sh
python tools/export_policy_to_onnx.py \
  --rlchessbot-dir /path/to/RLChessBot \
  --checkpoint runs/ppo/<run>/checkpoints/iteration-XXXXXXXX.pt \
  --output public/model/policy.onnx
```

The script verifies the exported ONNX model's outputs against the original
PyTorch model on sampled positions before writing the file.

## Deployment

Deployed to GitHub Pages via `.github/workflows/deploy.yml` on push to `main`.

Live at: https://16s893-ai-for-engineering-research.github.io/hcairney/
