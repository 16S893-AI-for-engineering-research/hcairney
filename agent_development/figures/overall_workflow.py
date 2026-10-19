"""Diagram of the intended overall agentic workflow (implementation-plan.md)."""
import sys; sys.path.insert(0, "/Users/hughcairney/GradSchool/16.S893/hcairney/.pi/skills/svg-figs")
from svgfig import Figure

f = Figure("deck", palette="color")
M, W = 80, 1760
f.text(M, 100, "Intended workflow: review and propose loop", "title", "ink")
f.text(M, 144, "Scripts do the numbers, the agent interprets them, and you launch every run.", "subtitle", "muted")


def code(s):
    return (s, "code", "muted")


specs = [
    (0, ["Summarize", "Phase 1: built", code("summarize_runs.py")]),
    (1, ["Review", "Phase 2: built", "Opus 5.5, high"]),
    (1, ["Second look", "Phase 6: optional", "Other provider"]),
    (0, ["Check", "Phase 3: partly built", code("check_review.py")]),
    (0, ["Make configs", "Phase 3: planned", "3 TOML per setup"]),
    (2, ["Approve", "Phase 4: planned", "You choose"]),
    (2, ["Launch", "Phase 5: planned", "You run jobs"]),
]
GAP = 40
NW = (W - GAP * (len(specs) - 1)) / len(specs)
NY, NH = 330, 190
nodes = [f.node(M + k * (NW + GAP), NY, rows, cat, w=NW, h=NH, align="start")
         for k, (cat, rows) in enumerate(specs)]
for a, b in zip(nodes, nodes[1:]):
    f.arrow(a.r, b.l)
# the second look can be skipped: draw the bypass over the top of that node
n2, n3, n4 = nodes[1], nodes[2], nodes[3]

# Return path over the top
ry = 250
f.connector([nodes[-1].t, (nodes[-1].cx, ry), (nodes[0].cx, ry), nodes[0].t])
lab = "Next round: new runs go into the next sweep"
f.text((nodes[0].cx + nodes[-1].cx) / 2, ry - 16, lab, "label", "ink2", anchor="middle")

# Annotation under the optional node
f.text(n3.cx, NY + NH + 40, "Critiques proposals", "small", "muted", anchor="middle")
f.text(n3.cx, NY + NH + 66, "before you see them", "small", "muted", anchor="middle")

# History panel
HY, HH = 640, 300
panel = f.box(M, HY, W, HH)
f.text(M + 24, HY + 44, "History folder", "heading", "ink")
f.text(M + 24 + f.width("History folder", "heading") + 16, HY + 44, "Phase 4: planned", "label", "muted")
c1, c2 = M + 24, M + 700
f.text(c1, HY + 96, "campaigns/<name>/campaign.json", "code", "muted")
f.text(c2, HY + 96, "goal, start group, allowed settings, stop rule", "label", "ink2")
f.text(c1, HY + 140, "round-001/summary.json", "code", "muted")
f.text(c2, HY + 140, "summary script output", "label", "ink2")
f.text(c1, HY + 184, "round-001/review.md, proposals.json", "code", "muted")
f.text(c2, HY + 184, "agent report and fixed-form proposals", "label", "ink2")
f.text(c1, HY + 228, "round-001/decision.json, configs/", "code", "muted")
f.text(c2, HY + 228, "what you approved, and the TOML files", "label", "ink2")
f.text(c1, HY + 272, "history.md", "code", "muted")
f.text(c2, HY + 272, "one line per round, read by the next review", "label", "ink2")

# Arrows between the loop and the history
f.connector([n2.at("b", 0.5), (n2.cx, HY)], both=True)
f.connector([nodes[5].at("b", 0.5), (nodes[5].cx, HY)])
f.text(n2.cx + 20, NY + NH + 62, "Reads history,", "small", "muted")
f.text(n2.cx + 20, NY + NH + 86, "saves report", "small", "muted")
f.text(nodes[5].cx + 20, NY + NH + 62, "Saves decision", "small", "muted")
f.text(nodes[5].cx + 20, NY + NH + 86, "and configs", "small", "muted")

# Legend
ly = 1010
x = M
for cat, name in [(0, "Script"), (1, "Agent"), (2, "You")]:
    f.box(x, ly - 24, 32, 32, cat)
    wd = f.text(x + 44, ly, name, "heading", f"cat{cat}")
    x += 44 + wd + 56
f.text(x, ly, "Run data stays in runs/. The history folder only points to it.", "label", "muted")

f.save("overall_workflow.svg")
