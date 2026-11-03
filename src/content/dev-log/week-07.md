---
title: "Week 7: "
date: 2026-11-03
summary: ""
---

## What I added

This is a bit late, but for the class clinic on 10/20/26 I added a small experiment 
summary and then agent review workflow.

## Why I added it

I wanted to see how an agent would do at proposing new experiments based on the results
of the previous experiment sweep. I also was considering making this a part of a larger
agentic workflow that allows the agent to take the results of an experiment, interpret
the results, propose new experiments and repeat the process.

## How I used the agent

I used the agent to help plan the workflow. The design actually uses a [script](/agent_development/scripts/review_sweep.sh)
to make sure the information is in place and then launches a pi session with an actual
agent to do the review.
