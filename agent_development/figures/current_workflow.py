"""Diagram of the current review_sweep.sh workflow. Run: uv run --project <svg-figs> python current_workflow.py"""
import sys; sys.path.insert(0, "/Users/hughcairney/GradSchool/16.S893/hcairney/.pi/skills/svg-figs")
from svgfig import Figure

f = Figure("deck", palette="color")
f.t["line"]["border"] = 3.0  # thicker box outlines (token default is 1.0)
f.t["space"]["pad"] = 16  # text inset: clears the thicker outline (lint measures from the stroke edge)
M, W = 80, 1760
f.text(M, 100, "Current workflow: review_sweep.sh", "title", "ink")
f.text(M, 144, "One sweep in, one review out. The agent only reads and proposes. You launch every run.", "subtitle", "muted")

NH, GAP, PAD = 136, 56, 20
PH = 62 + NH + 20           # panel height
ROW_GAP = 52
Y = [175 + i * (PH + ROW_GAP) for i in range(3)]


def code(s):
    return (s, "code", "muted")


def row(i, title, caption, caption_style, specs):
    y0 = Y[i]
    f.box(M, y0, W, PH)
    f.text(M + PAD, y0 + 40, title, "heading", "ink")
    if caption:
        f.text(M + W - PAD, y0 + 40, caption, caption_style, "muted", anchor="end")
    n = len(specs)
    w = (W - 2 * PAD - (n - 1) * GAP) / n
    nodes = [f.node(M + PAD + k * (w + GAP), y0 + 62, rows, cat, w=w, h=NH) for k, (cat, rows) in enumerate(specs)]
    for a, b in zip(nodes, nodes[1:]):
        f.arrow(a.r, b.l)
    return nodes


A = row(0, "1. Prepare", "review_sweep.sh", "code", [
    (3, ["Sweep folder", code("runs/<sweep>")]),
    (0, ["Check folder", "Stop if it is missing"]),
    (0, ["Archive old review", code("review-<timestamp>.md")]),
    (0, ["Start the agent", code("pi -p --no-session")]),
])
B = row(1, "2. Agent", "Opus 5.5, high thinking, read and bash only", "small", [
    (1, ["Get summary", "Build it if missing", code("summarize_runs.py")]),
    (1, ["Read summary", "Tables and plots", code("summary.md")]),
    (1, ["Check numbers", "Per-seed values", code("summary.json")]),
    (1, ["Read base config", "Real setting names", code("submitted.toml")]),
    (1, ["Write report", "Five sections", "JSON proposals"]),
])
C = row(2, "3. Finish", "review_sweep.sh, then you", "small", [
    (0, ["Add header", "Model, date, sweep"]),
    (3, ["Review", code("review.md")]),
    (0, ["Check report", code("check_review.py")]),
    (3, ["Check result", code("check.txt")]),
    (2, ["You decide", "Read the review,", "launch runs"]),
])


def wrap(src, dst, i):
    """Return connector from the end of row i to the start of row i+1, through the gap."""
    p0 = src.at("b", 0.3)
    p3 = dst.at("t", 0.85)
    gy = Y[i] + PH + ROW_GAP / 2
    f.connector([p0, (p0[0], gy), (p3[0], gy), p3])


wrap(A[-1], B[0], 0)
wrap(B[-1], C[0], 1)

# Legend
ly = 972
x = M
for cat, name in [(0, "Script step"), (1, "Agent step"), (2, "You"), (3, "File")]:
    f.box(x, ly - 24, 32, 32, cat)
    wd = f.text(x + 44, ly, name, "heading", f"cat{cat}")
    x += 44 + wd + 56

f.t["space"]["pad"] = 14  # lint minimum is the token default
f.save("current_workflow.svg")
