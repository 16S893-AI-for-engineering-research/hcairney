---
title: "Week 5: taking a step back to the simpler RANS case"
date: 2026-10-16
summary: "Initial LES training runs using V-RACER failed, looking into the simpler RANS case as a proof of concept"
---

## What I added



## Why I added it

My initial work using the V-RACER RL algorithm to train SGS models proved unsucessful. This forced me to re-
evaluate how I would proceed with the project. My advisor and I decided to take a step back to the simpler steady
RANS case which does not have the same turbulent fluctuations as in LES. One hypothesis was that trying to encourage
"good" actions under a stochastic policy is simply not possible given that it is not clear that any particular action
will have a meaningful and lasting effect on the flow.

In the steady RANS case, each rollout is instead essentially one converged simulation. This made V-RACER a bit 
awkward since each simulation is an independent realization and so trying to calculate returns over an episode
is not appropriate. Therefore, I switched to a deterministic policy gradient approach where a critic network is
fitted using the data collected from different policies and then an actor network is optimized by differentiating
through the critic (which is possible since the critic takes the action as one of its inputs).

## How I used the agent

I used the agent to help make a detailed implementation plan for building a simple finite-volume solver for the
1D RANS equation for the case of turbulent channel flow. I used skills such as grill-me to make this plan which 
helped me to flesh out exactly what needed to be decided. After a few rounds of back-and-forth, I had the agent
turn that implementation plan into a set of backlog tasks (originally 58, it has since expanded from there) that
covered implementing the standalone solver in C++, compiling that C++ solver into a Python extension module with
pybind11, and the custom Python RL training code implemented with torch that can call the solver through the
extension module.

Once the backlog tasks were in place, I started going through them one by one. I found that the agent that wrote
the original backlog tasks did it in a bit of a "quick and dirty" way that left some things underspecified. For
this reason, my workflow essentially became a 2 step process for each task. First, I asked the agent to read the
task, clarify what needed to be added for the task and tell me if anthing was left that needed a decision. Once
these details were ironed out, I would ask it to update the backlog task with the more specific details. Then, I
would start a new session and ask it to implement the task. 

I found this break down to be very effective. However, it did make the process much more tedious.
