---
title: "Week 3: figure reproduction assignment"
date: 2026-10-01
summary: "Used the agent to independently re-derive a relation in Pope (1975)"
---

# What I added

I added some python code to [`reproduce_figure/`](https://github.com/16S893-AI-for-engineering-research/hcairney/tree/main/reproduce_figure) that verifies the relation derived in 
Stephen Pope's 1975 paper about more general non-linear turbulent viscosity closures.

This is an old paper, but it has gained newfound relevance since data-driven methods
have become more popular since the explicit relation Pope derives is easy to implement
in solvers and data-driven methods can maintain physical invariances by predicting the
non-dimensional coefficients in the tensor basis expansion.

# Why I added it

This was for the second class assignment

# What I added

I asked the agent to use uv and the test-driven development skill to verify the 
derivation in pope. One way it did this was by directly solving the implicit relation
numerically and then comparing that with the explicit formula. The agent had no trouble
doing this all in one shot. I manually reviewed the code to make sure everything looked
right.

Interestingly, I noticed that the figure labels in the reproduced plot were not matched
to the labels in the original figure. GPT-5.6 Sol did not notice this when it reproduced
the figure (even though it must have seen the original to know that it needed to reproduce
it). I started a new session and asked it about the discrepancy and with some prompting it 
produced the figure 2 included in the [`derivation.pdf`](https://github.com/16S893-AI-for-engineering-research/hcairney/blob/main/reproduce_figure/derivation.pdf) that I had it write up. It turns out
this very famous paper (about 1000 citations) has a label offset causing the contours to be
labeled with the wrong value!