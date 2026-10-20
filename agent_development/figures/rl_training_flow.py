import sys; sys.path.insert(0, "/home/gridsan/hcairney/.pi/agent/skills/svg-figs")
from svgfig import Figure

f = Figure("deck", palette="color")
THICK = 3.5
f.t["line"]["border"] = THICK
f.t["space"]["pad"] = 17
W, H = 370, 156
X = [120 + i * 440 for i in range(4)]
CODE = lambda s: (s, "code", "muted")

f.text(80, 95, "RL Training Loop", "title", "ink", "start")

# --- row 0: set up and legend
setup = f.node(X[0], 130, ["Set Up", "Build actor, critic, replay", "Check the starting actor",
                           CODE("train.py: execute")], 3, w=W, h=H)
legend = [(0, "Collect data: actor and flow solver"), (1, "Learn from replay"),
          (2, "Check and save"), (3, "Start and finish")]
f.text(1000, 168, "Legend", "heading", "ink")
for i, (c, s) in enumerate(legend):
    x, y = 1000 + (i % 2) * 450, 200 + (i // 2) * 44
    f.box(x, y, 28, 28, c)
    f.text(x + 42, y + 22, s, "label", "ink2")

# --- loop panel
PY, PH = 325, 490
f.t["line"]["border"] = 2.5
f.box(80, PY, 1760, PH, None, stroke="muted")
f.t["line"]["border"] = THICK
f.text(X[1], PY + 36, "Repeat Until the Solve Limit", "heading", "ink")

R1, R2 = 385, 620
snap = f.node(X[0], R1, ["1  Copy the Actor", "Add noise $\\sigma$ to weights", "Gives the behaviour actor",
                         CODE("exploration.snapshot")], 0, w=W, h=H)
act = f.node(X[1], R1, ["2  Choose $C$", "Actor reads the start profile", "Sets $C$ in each cell",
                        CODE("scripted evaluate")], 0, w=W, h=H)
solve = f.node(X[2], R1, ["3  Solve to Steady State", "Flow solver uses the $C$ field",
                          ("Not converged: run pauses", "label", "emph"),
                          CODE("solve_to_steady")], 0, w=W, h=H)
score = f.node(X[3], R1, ["4  Score", "Compare with target profile", "Return from profile error",
                          CODE("rewards.py")], 0, w=W, h=H)
add = f.node(X[3], R2, ["5  Add to Replay", "Whole evaluation at once", "Oldest one leaves when full",
                        CODE("ReplayBuffer.admit")], 1, w=W, h=H)
samp = f.node(X[2], R2, ["6  Sample a Batch", "Random evaluation, then cell", "Skipped until warm-up ends",
                         CODE("ReplayBuffer.sample")], 1, w=W, h=H)
crit = f.node(X[1], R2, ["7  Update Critic", "Fit predicted return to return", "Mean squared error",
                         CODE("TerminalLearner")], 1, w=W, h=H)
actu = f.node(X[0], R2, ["8  Update Actor", "Change $C$ to raise return", "Same batch, after critic",
                         CODE("TerminalLearner")], 1, w=W, h=H)

# --- bottom row
B = 870
chk = f.node(X[0], B, ["Performance Check", "Unperturbed actor, new solve", "Keep the lowest volume error",
                       CODE("selection.py")], 2, w=W, h=H)
ckp = f.node(X[1], B, ["Save Checkpoint", "Models, replay, random states", "Resume gives the same run",
                       CODE("checkpoint.py")], 2, w=W, h=H)
fin = f.node(X[3], B, ["Finish", "Final check and checkpoint", "Write the run report",
                       CODE("run_stop")], 3, w=W, h=H)

# --- arrows
f.arrow(setup.b, snap.t)
f.arrow(snap.r, act.l)
f.arrow(act.r, solve.l)
f.arrow(solve.r, score.l)
f.arrow(score.b, add.t)
f.arrow(add.l, samp.r)
f.arrow(samp.l, crit.r)
f.arrow(crit.l, actu.r)
f.arrow(actu.t, snap.b)
f.connector([actu.at("b", 0.25), chk.at("t", 0.25)], dash="6 5")
f.text(actu.at("b", 0.25)[0] + 14, 853, "every $N$ rounds", "small", "muted")
f.text(ckp.cx + 14, 860, "every $M$ rounds", "small", "muted")
f.connector([actu.at("b", 0.75), (actu.at("b", 0.75)[0], 835), (ckp.cx, 835), ckp.t], dash="6 5")
f.connector([(X[3] + W / 2, PY + PH), fin.t])
f.text(X[3] + W / 2 + 14, 852, "solve limit reached", "small", "muted")

f.t["space"]["pad"] = 14   # lint minimum
f.save("rl_training_flow.svg")
