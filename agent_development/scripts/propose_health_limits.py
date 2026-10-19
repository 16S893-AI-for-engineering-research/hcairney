#!/usr/bin/env python3
"""Propose health limits from the distribution of run health across sweep summaries.

Usage: propose_health_limits.py SUMMARY_JSON [SUMMARY_JSON ...] [--out FILE]
Run summarize_runs.py with --no-limits first. Method:
  * For each health measure, take every run in every summary.
  * Count measures (rejected steps, failed solves, unavailable solver work, failed checks):
    if every run is zero, the limit is 0 (any occurrence is unusual).
  * Otherwise limit = the 99th percentile over runs, rounded up to 2 significant figures.
    So about 1% of runs (those in the extreme tail) would exceed it.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

MEASURES = ("rejected_steps", "failed_solves", "unavailable_solver_work", "failed_checks",
            "nonlinear_iterations_per_solve")
PERCENTILE = 99


def round_up(value, figures=2):
    if value == 0:
        return 0
    scale = 10 ** (math.floor(math.log10(abs(value))) - figures + 1)
    return math.ceil(value / scale) * scale


def propose(summaries):
    runs = [run for summary in summaries for group in summary["groups"] for run in group["runs"]]
    limits, distribution = {}, {}
    for measure in MEASURES:
        values = np.array([r["health"][measure] for r in runs if r["health"][measure] is not None], float)
        if not len(values):
            continue
        percentiles = {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99, 100)}
        distribution[measure] = dict(runs=len(values), **percentiles)
        limits[measure] = 0 if values.max() == 0 else round_up(percentiles[f"p{PERCENTILE}"])
    return limits, distribution, len(runs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    limits, distribution, n_runs = propose(summaries)
    result = dict(
        note=(f"Proposed from {n_runs} runs in sweeps {', '.join(s['sweep'] for s in summaries)}. "
              "Measures that were zero in every run get limit 0; others use the 99th percentile "
              "rounded up to 2 significant figures. Edit the numbers freely; groups whose worst "
              "seed is above a limit are flagged unreliable."),
        limits=limits, distribution=distribution)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
