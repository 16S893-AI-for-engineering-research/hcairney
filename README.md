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
- `public/easter-egg.js` — Konami-code easter egg (try ↑ ↑ ↓ ↓ ← → ← →)

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

## Deployment

Deployed to GitHub Pages via `.github/workflows/deploy.yml` on push to `main`.

Live at: https://16s893-ai-for-engineering-research.github.io/hcairney/
