#!/usr/bin/env python3
"""Summarize a finished RL training sweep: seed-group scores, trends and convergence health.

Reads run folders (run.json, performance.jsonl, the end of history.jsonl,
inputs/submitted.toml) without modifying them, and writes summary.json, summary.md and
plots to a separate folder. See docs/summarize-runs.md for field definitions and the
trend rule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from statistics import mean, median, stdev
import sys
import time

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # rank_training_runs.py lives here
from rank_training_runs import SEEDS_OUTPUT_NAME, seed_group, successful_score  # noqa: E402

STYLE = PROJECT_ROOT / "scripts" / "nonlinear-print.mplstyle"
DEFAULT_LIMITS = PROJECT_ROOT / "agent_development" / "health_limits.json"

EXPECTED_SEEDS = 3
# Settings that always differ between runs and are not experimental choices.
IGNORED_SETTINGS = ("run.name", "run.output", "seeds.")
HEALTH_KEYS = ("rejected_steps", "failed_solves", "nonlinear_iterations",
               "nonlinear_iterations_per_solve", "unavailable_solver_work", "failed_checks")
CLEAN_STOP_REASONS = ("attempted_solve_limit",)

# Trend rule (documented in docs/summarize-runs.md).
TREND_MIN_CHECKS = 10
TREND_WINDOW_FRACTION = 0.2
TREND_MIN_WINDOW = 5
TREND_IMPROVING_BELOW = 0.8
TREND_WORSE_ABOVE = 1.25


class SummaryError(Exception):
    """A problem that stops the whole summary (as opposed to one unreadable run)."""


def trend_label(values):
    """Compare the median of the last window of checks with the median of the window before.

    Returns (label, ratio). Window = max(5, 20% of checks). ratio = last / previous.
    ratio < 0.8 -> improving, ratio > 1.25 -> worse, otherwise flat; fewer than 10
    checks -> unknown.
    """
    count = len(values)
    if count < TREND_MIN_CHECKS:
        return "unknown", None
    window = max(TREND_MIN_WINDOW, round(count * TREND_WINDOW_FRACTION))
    last = median(values[-window:])
    previous = median(values[-2 * window:-window])
    if previous == 0:
        return ("flat", 1.0) if last == 0 else ("worse", None)
    ratio = last / previous
    if ratio < TREND_IMPROVING_BELOW:
        return "improving", ratio
    if ratio > TREND_WORSE_ABOVE:
        return "worse", ratio
    return "flat", ratio


def flatten(data, prefix=""):
    for key, value in data.items():
        if isinstance(value, dict):
            yield from flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def read_settings(run):
    """Flattened submitted.toml without seeds, names and paths."""
    with (run / "inputs" / "submitted.toml").open("rb") as stream:
        data = tomllib.load(stream)
    return {key: value for key, value in flatten(data)
            if not key.startswith(IGNORED_SETTINGS)}


def last_line(path, block=1 << 16):
    """Last non-empty line of a possibly huge file, reading only its end."""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        size = block
        while True:
            start = max(0, end - size)
            stream.seek(start)
            lines = stream.read(end - start).rstrip(b"\r\n").split(b"\n")
            if len(lines) > 1 or start == 0:
                return lines[-1].decode("utf-8") if lines[-1] else None
            size *= 4


def read_stop(run):
    """Return (stop_record or None, problem or None) from the end of history.jsonl."""
    try:
        line = last_line(run / "history.jsonl")
    except OSError as exc:
        return None, f"history.jsonl unreadable: {exc.strerror or exc}"
    try:
        record = json.loads(line) if line else None
    except ValueError:
        return None, "history.jsonl ends with an unreadable line"
    if not isinstance(record, dict) or record.get("event") != "run_stop":
        return None, "history.jsonl does not end with a run_stop record"
    return record, None


def read_checks(run):
    """Stream performance.jsonl: successful E_volume series and where the best occurred."""
    series = []
    checks = failed = 0
    best = best_index = best_solves = None
    with (run / "performance.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                score = successful_score(event)
                is_result = event.get("event") == "selection_result"
                boundary = event.get("completed_boundary") if is_result else None
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
                raise ValueError(f"performance.jsonl line {line_number}: {exc}") from exc
            if not is_result:
                continue
            index = checks
            checks += 1
            if score is None:
                failed += 1
                continue
            series.append(score)
            if best is None or score < best:
                best, best_index, best_solves = score, index, boundary
    return dict(series=series, n_checks=checks, failed_checks=failed,
                best=best, best_index=best_index, best_solves=best_solves)


def health_from_stop(stop):
    work = stop.get("actual_work") or {}
    iterations = work.get("returned_nonlinear_iterations")
    solves = (stop.get("cost") or {}).get("combined", {}).get("returned_solves")
    per_solve = iterations / solves if iterations is not None and solves else None
    return dict(rejected_steps=work.get("returned_rejected_steps"),
                failed_solves=work.get("failed_returned_solves"),
                nonlinear_iterations=iterations,
                nonlinear_iterations_per_solve=per_solve,
                unavailable_solver_work=work.get("unavailable_solver_work"))


def summarize_run(task):
    """Summarize one run folder; never raises for problems with the run's own files."""
    sweep, relative = task
    run = Path(sweep) / relative
    problems = []
    record = dict(run=relative, best_E_volume=None, best_check_index=None, best_solves=None,
                  final_E_volume=None, final_over_best=None, n_checks=None, trend="unknown",
                  trend_ratio=None, stop_reason=None, work_completeness=None,
                  health={key: None for key in HEALTH_KEYS}, errors=problems)
    settings = None
    try:
        settings = read_settings(run)
    except (OSError, ValueError) as exc:  # TOMLDecodeError is a ValueError
        problems.append(f"unreadable_settings: {exc}")
    try:
        checks = read_checks(run)
    except (OSError, ValueError) as exc:
        problems.append(f"unreadable_performance: {str(exc).replace(chr(10), ' ')}")
    else:
        series = checks["series"]
        record.update(n_checks=checks["n_checks"], best_E_volume=checks["best"],
                      best_check_index=checks["best_index"], best_solves=checks["best_solves"])
        record["health"]["failed_checks"] = checks["failed_checks"]
        if series:
            record["final_E_volume"] = series[-1]
            record["final_over_best"] = series[-1] / checks["best"] if checks["best"] else None
            record["trend"], record["trend_ratio"] = trend_label(series)
        else:
            problems.append("no_successful_results")
    stop, problem = read_stop(run)
    if stop is None:
        problems.append(f"no_stop_record: {problem}")
    else:
        record["stop_reason"] = stop.get("reason")
        record["work_completeness"] = (stop.get("actual_work") or {}).get("work_completeness")
        record["health"].update(health_from_stop(stop))
        if record["stop_reason"] not in CLEAN_STOP_REASONS:
            problems.append(f"stopped_early: {record['stop_reason']}")
        if record["work_completeness"] != "returned_work_available":
            problems.append(f"work_incomplete: {record['work_completeness']}")
    return record, settings


def find_runs(sweep):
    """Directories containing run.json, relative to the sweep (does not enter run folders)."""
    runs = []
    for directory, subdirectories, filenames in os.walk(sweep):
        if "run.json" in filenames:
            runs.append(Path(directory).relative_to(sweep).as_posix())
            subdirectories.clear()
    return sorted(runs)


def load_limits(path):
    if path is None:
        return {}, None
    data = json.loads(Path(path).read_text())
    limits = data.get("limits", {})
    unknown = set(limits) - set(HEALTH_KEYS)
    if unknown:
        raise SummaryError(f"unknown health limit(s) {sorted(unknown)}; known: {list(HEALTH_KEYS)}")
    return limits, data.get("note")


def worst_health(runs):
    worst = {}
    for key in HEALTH_KEYS:
        values = [r["health"][key] for r in runs if r["health"][key] is not None]
        worst[key] = max(values) if values else None
    return worst


def build_group(name, members, settings_by_run, limits, expected_seeds):
    members = sorted(members, key=lambda r: r["run"])
    scores = [r["best_E_volume"] for r in members]
    ranked = all(score is not None for score in scores)
    reasons = []
    if len(members) < expected_seeds:
        reasons.append(f"missing_seeds: {len(members)} of {expected_seeds} runs found")
    for run in members:
        reasons.extend(f"{run['run']}: {problem}" for problem in run["errors"])
    known = [s for s in scores if s is not None]
    present = [settings_by_run[r["run"]] for r in members if r["run"] in settings_by_run]
    mismatched = []
    if present:
        mismatched = sorted(key for key in set().union(*present)
                            if len({json.dumps(s.get(key), sort_keys=True) for s in present}) > 1)
    worst = worst_health(members)
    exceeded = sorted(key for key, limit in limits.items()
                      if worst.get(key) is not None and worst[key] > limit)
    return dict(
        group=name, family=Path(name).parent.as_posix(), n_runs=len(members),
        rank=None, runs=members,
        mean_best_E_volume=mean(scores) if ranked else None,
        std_best_E_volume=stdev(scores) if ranked and len(scores) > 1 else None,
        min_best_E_volume=min(known) if known else None,
        max_best_E_volume=max(known) if known else None,
        worst_health=worst, exceeded_limits=exceeded,
        flags=dict(incomplete=bool(reasons), unreliable=bool(exceeded),
                   seed_mismatch=bool(mismatched)),
        incomplete_reasons=reasons, mismatched_settings=mismatched,
        settings=dict(present[0]) if present else {})


def check_ranking(sweep, groups):
    """Compare group order and means with rank_training_runs.py output, if it exists."""
    path = Path(sweep) / SEEDS_OUTPUT_NAME
    if not path.is_file():
        return dict(status="not_available", compared=0, detail=f"{SEEDS_OUTPUT_NAME} not found")
    reference = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].isdigit():
            reference.append((parts[3], float(parts[1])))
    mine = [(g["group"], g["mean_best_E_volume"])
            for g in sorted((g for g in groups if g["rank"] is not None), key=lambda g: g["rank"])]
    if [name for name, _ in mine] != [name for name, _ in reference]:
        first = next((i for i, (a, b) in enumerate(zip(mine, reference)) if a[0] != b[0]),
                     min(len(mine), len(reference)))
        return dict(status="mismatch", compared=len(reference),
                    detail=f"group order differs from {SEEDS_OUTPUT_NAME} at position {first + 1}")
    worst = max((abs(a - b) / max(abs(b), 1e-300) for (_, a), (_, b) in zip(mine, reference)),
                default=0.0)
    if worst > 1e-9:
        return dict(status="mismatch", compared=len(reference),
                    detail=f"means differ from {SEEDS_OUTPUT_NAME} (relative {worst:.2g})")
    return dict(status="match", compared=len(reference),
                detail=f"same order and means as {SEEDS_OUTPUT_NAME}")


def _sort_key(value):
    return (type(value).__name__, json.dumps(value) if isinstance(value, (list, dict)) else value)


def varied_settings(groups):
    """Settings whose value differs between groups: key -> sorted list of distinct values."""
    values = {}
    for group in groups:
        for key, value in group["settings"].items():
            values.setdefault(key, {})[json.dumps(value, sort_keys=True)] = value
    keys = {key for key, seen in values.items() if len(seen) > 1}
    # A setting that is absent from some groups also counts as varied.
    for group in groups:
        keys |= {key for key in values if key not in group["settings"]}
    return {key: sorted(values[key].values(), key=_sort_key) for key in sorted(keys)}


def linked_settings(groups, varied):
    """Varied settings that always change together (same split of groups): lists of keys.

    Their separate effects cannot be told apart from this sweep.
    """
    # Two keys are linked if their values map one-to-one onto each other.
    merged = {}
    for key in varied:
        for other in varied:
            if other <= key:
                continue
            pairs = {(json.dumps(g["settings"].get(key), sort_keys=True),
                      json.dumps(g["settings"].get(other), sort_keys=True)) for g in groups}
            if (len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})):
                merged.setdefault(key, {key}).add(other)
    sets = []
    for members in merged.values():
        for existing in sets:
            if existing & members:
                existing |= members
                break
        else:
            sets.append(set(members))
    return sorted(sorted(members) for members in sets)


def summarize_sweep(sweep, limits=None, workers=None, expected_seeds=EXPECTED_SEEDS,
                    limits_note=None, progress=None):
    sweep = Path(sweep).resolve()
    if not sweep.is_dir():
        raise SummaryError(f"Not a sweep directory: {sweep}")
    runs = find_runs(sweep)
    if not runs:
        raise SummaryError(f"No training runs (run.json) found in {sweep}")
    limits = limits or {}
    tasks = [(str(sweep), relative) for relative in runs]
    workers = workers or min(32, os.cpu_count() or 1)
    results = []
    if workers <= 1:
        for number, task in enumerate(tasks, 1):
            results.append(summarize_run(task))
            if progress:
                progress(number, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for number, result in enumerate(pool.map(summarize_run, tasks, chunksize=4), 1):
                results.append(result)
                if progress and number % 100 == 0:
                    progress(number, len(tasks))
    members = {}
    settings_by_run = {}
    for record, settings in results:
        members.setdefault(seed_group(record["run"]), []).append(record)
        if settings is not None:
            settings_by_run[record["run"]] = settings
    groups = [build_group(name, group_runs, settings_by_run, limits, expected_seeds)
              for name, group_runs in members.items()]
    ranked = sorted((g for g in groups if g["mean_best_E_volume"] is not None),
                    key=lambda g: (g["mean_best_E_volume"], g["group"]))
    for rank, group in enumerate(ranked, 1):
        group["rank"] = rank
    groups.sort(key=lambda g: (g["rank"] is None, g["rank"] or 0, g["group"]))
    varied = varied_settings(groups)
    for group in groups:  # keep only the settings that differ between groups
        group["settings"] = {key: group["settings"][key] for key in varied
                             if key in group["settings"]}
    summary = dict(
        sweep=sweep.name, sweep_path=str(sweep), n_runs=len(runs), n_groups=len(groups),
        n_ranked=len(ranked), expected_seeds=expected_seeds,
        health_limits=limits, health_limits_note=limits_note,
        varied_settings=varied, linked_settings=linked_settings(groups, list(varied)),
        groups=groups)
    summary["ranking_check"] = check_ranking(sweep, groups)
    summary["counts"] = {flag: sum(g["flags"][flag] for g in groups)
                         for flag in ("incomplete", "unreliable", "seed_mismatch")}
    return summary


# ---------------------------------------------------------------- markdown ----

def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}e}" if isinstance(value, float) else str(value)


def short(value):
    if isinstance(value, str) and "/" in value:
        return value.rsplit("/", 1)[-1]
    if isinstance(value, float):
        return f"{value:g}"
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def settings_text(group, varied):
    return ", ".join(f"{key}={short(group['settings'][key])}"
                     for key in varied if key in group["settings"])


def trend_text(group):
    counts = {}
    for run in group["runs"]:
        counts[run["trend"]] = counts.get(run["trend"], 0) + 1
    return ", ".join(f"{n} {label}" for label, n in sorted(counts.items()))


def flags_text(group):
    return ", ".join(name for name, on in group["flags"].items() if on)


def health_text(group):
    worst = group["worst_health"]
    per_solve = worst["nonlinear_iterations_per_solve"]
    per_solve = "-" if per_solve is None else f"{per_solve:.0f}"
    return (f"rej {fmt(worst['rejected_steps'])}, fail {fmt(worst['failed_solves'])}, "
            f"unavail {fmt(worst['unavailable_solver_work'])}, it/solve {per_solve}")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


GROUP_HEADER = ["Rank", "Group", "Mean best E_volume", "Std", "Min / max",
                "Trend (seeds)", "Worst-seed health", "Flags", "Varied settings"]


def group_row(group, varied):
    return [group["rank"] if group["rank"] is not None else "-", group["group"],
            fmt(group["mean_best_E_volume"]), fmt(group["std_best_E_volume"]),
            f"{fmt(group['min_best_E_volume'])} / {fmt(group['max_best_E_volume'])}",
            trend_text(group), health_text(group), flags_text(group), settings_text(group, varied)]


def setting_effects(summary):
    """Median of group means for each value of each varied setting."""
    rows = []
    for key, values in summary["varied_settings"].items():
        for value in values:
            members = [g for g in summary["groups"] if g["settings"].get(key) == value
                       and g["mean_best_E_volume"] is not None]
            if not members:
                continue
            means = [g["mean_best_E_volume"] for g in members]
            rows.append([key, short(value), len(members), fmt(median(means)), fmt(min(means)),
                         sum(g["flags"]["unreliable"] for g in members)])
    return rows


def write_markdown(summary, path, top=15, max_flagged=30):
    groups = summary["groups"]
    varied = list(summary["varied_settings"])
    counts = summary["counts"]
    check = summary["ranking_check"]
    lines = [f"# Sweep summary: {summary['sweep']}", "",
             f"- Runs: {summary['n_runs']}; seed groups: {summary['n_groups']} "
             f"(ranked: {summary['n_ranked']}; expected seeds per group: {summary['expected_seeds']})",
             f"- Flagged groups: incomplete {counts['incomplete']}, unreliable {counts['unreliable']}, "
             f"seed_mismatch {counts['seed_mismatch']}",
             f"- Ranking check against {SEEDS_OUTPUT_NAME}: {check['status']} ({check['detail']})",
             "- Score = best successful unperturbed `E_volume` per seed (lower is better); "
             "groups are ranked by the mean over seeds. Std is the sample standard deviation.",
             "- Full per-run detail is in `summary.json`.", "", "## Varied settings", ""]
    if summary["varied_settings"]:
        lines += [f"- `{key}`: {', '.join(short(v) for v in values)}"
                  for key, values in summary["varied_settings"].items()]
    else:
        lines.append("None (all groups use the same settings).")
    for linked in summary["linked_settings"]:
        lines.append(f"- Linked (always change together, effects cannot be separated): "
                     + ", ".join(f"`{key}`" for key in linked))
    effects = setting_effects(summary)
    if effects:
        lines += ["", "## Effect of each setting value (over ranked groups)", "",
                  "Each row pools all groups with that value, ignoring the other settings."]
        lines.append("")
        lines += table(["Setting", "Value", "Groups", "Median of group means", "Best group mean",
                        "Unreliable groups"], effects)
    lines += ["", f"## Top {min(top, summary['n_ranked'])} groups", ""]
    lines += table(GROUP_HEADER, [group_row(g, varied) for g in groups[:top] if g["rank"]])
    families = {}
    for group in groups:
        if group["rank"] is not None:
            families.setdefault(group["family"], group)
    if len(families) > 1:
        lines += ["", "## Best group per family", ""]
        lines += table(GROUP_HEADER, [group_row(g, varied) for _, g in sorted(families.items())])
    flagged = [g for g in groups if any(g["flags"].values())]
    lines += ["", f"## Flagged groups ({len(flagged)})", ""]
    if flagged:
        rows = []
        for group in flagged[:max_flagged]:
            reason = []
            if group["flags"]["unreliable"]:
                reason.append("over limit: " + ", ".join(group["exceeded_limits"]))
            if group["flags"]["incomplete"]:
                reason.append(group["incomplete_reasons"][0])
            if group["flags"]["seed_mismatch"]:
                reason.append("settings differ: " + ", ".join(group["mismatched_settings"]))
            rows.append([group["rank"] or "-", group["group"], flags_text(group), "; ".join(reason)])
        lines += table(["Rank", "Group", "Flags", "Reason"], rows)
        if len(flagged) > max_flagged:
            lines += ["", f"{len(flagged) - max_flagged} more flagged groups are listed in "
                          "`summary.json` (best-ranked flagged groups are shown first)."]
    else:
        lines.append("None.")
    lines += ["", "## Health limits", ""]
    if summary["health_limits"]:
        lines.append("A group is `unreliable` if its worst seed is above any limit "
                     "(counts cover the whole run: training plus checks):")
        lines += [f"- `{key}` <= {limit}" for key, limit in summary["health_limits"].items()]
        if summary["health_limits_note"]:
            lines.append(f"- Note: {summary['health_limits_note']}")
    else:
        lines.append("No limits loaded, so no group is flagged `unreliable`.")
    lines += ["", "## Notes", "",
              "- Trend compares the median of the last 20% of checks with the median of the 20% before: "
              "ratio < 0.8 improving, > 1.25 worse, otherwise flat (fewer than 10 checks: unknown).",
              "- `rej` = rejected steps; `fail` = failed solves; `unavail` = solver work with "
              "unavailable counts; `it/solve` = nonlinear iterations per solve.",
              "- `incomplete`: missing seed, no successful result, unreadable file, missing stop record, "
              "early stop, or incomplete work counts. Such groups are still ranked if every seed has a score."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- plots ----

def make_plots(summary, out, top_curves=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if STYLE.is_file():
        plt.style.use(str(STYLE))
    ranked = [g for g in summary["groups"] if g["rank"] is not None]
    written = []
    if not ranked:
        return written
    families = sorted({g["family"] for g in ranked})
    colours = {family: f"C{i % 8}" for i, family in enumerate(families)}
    varied = list(summary["varied_settings"])
    if varied:
        columns = min(3, len(varied))
        rows = -(-len(varied) // columns)
        fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.3 * rows), squeeze=False)
        rng = np.random.default_rng(0)
        for axis, key in zip(axes.flat, varied):
            values = summary["varied_settings"][key]
            for position, value in enumerate(values):
                members = [g for g in ranked if g["settings"].get(key) == value]
                x = position + rng.uniform(-0.25, 0.25, len(members))
                y = [g["mean_best_E_volume"] for g in members]
                axis.scatter(x, y, s=6, alpha=0.6, linewidths=0,
                             c=[colours[g["family"]] for g in members])
                bad = [(xi, yi) for xi, yi, g in zip(x, y, members) if g["flags"]["unreliable"]]
                if bad:
                    axis.scatter(*zip(*bad), s=14, marker="x", c="k", linewidths=0.5)
                if y:
                    axis.hlines(median(y), position - 0.35, position + 0.35, colors="k", linewidth=1)
            axis.set_xticks(range(len(values)))
            axis.set_xticklabels([short(v) for v in values], rotation=30 if len(values) > 3 else 0)
            axis.set_yscale("log")
            axis.set_xlabel(key)
            axis.set_ylabel("mean best $E_{volume}$")
        for axis in list(axes.flat)[len(varied):]:
            axis.set_visible(False)
        if len(families) > 1:
            handles = [plt.Line2D([], [], marker="o", ls="", color=colours[f], label=f)
                       for f in families]
            fig.legend(handles=handles, loc="upper center", ncol=min(4, len(families)),
                       bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        path = out / "score_vs_settings.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    # Training curves of the best groups, one line per seed.
    sweep = Path(summary["sweep_path"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), squeeze=False, sharex=True, sharey=True)
    for axis, group in zip(axes.flat, ranked[:top_curves]):
        for run in group["runs"]:
            points = []
            try:
                with (sweep / run["run"] / "performance.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        event = json.loads(line)
                        if successful_score(event) is not None:
                            points.append((event["completed_boundary"], event["metrics"]["E_volume"]))
            except (OSError, ValueError, KeyError):
                continue
            if points:
                axis.plot(*zip(*points), linewidth=0.7)
        axis.set_yscale("log")
        axis.set_title(f"Rank {group['rank']}, Average Best $E_{{volume}}$ = "
                       f"{group['mean_best_E_volume']:.2e}", fontsize=6)
    for axis in list(axes.flat)[min(top_curves, len(ranked)):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("training solves")
    for axis in axes[:, 0]:
        axis.set_ylabel("$E_{volume}$")
    fig.tight_layout()
    path = out / "training_curves_top_groups.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)
    return written


# --------------------------------------------------------------------- CLI ----

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path, help="directory containing nested training runs")
    parser.add_argument("--out", type=Path, help="output folder (default: <sweep>/agent_summary)")
    parser.add_argument("--limits", type=Path,
                        help=f"health limits JSON (default: {DEFAULT_LIMITS} if it exists)")
    parser.add_argument("--no-limits", action="store_true", help="do not load any health limits")
    parser.add_argument("--expected-seeds", type=int, default=EXPECTED_SEEDS)
    parser.add_argument("--workers", type=int, help="parallel processes (default: up to 32)")
    parser.add_argument("--top", type=int, default=15, help="groups listed in summary.md")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        sweep = args.sweep.resolve()
        out = (args.out or sweep / "agent_summary").resolve()
        if sweep.is_dir():
            for run in find_runs(sweep):
                run_dir = sweep / run
                if out == run_dir or run_dir in out.parents:
                    raise SummaryError(f"Output {out} is inside run folder {run_dir}")
        limits_path = None
        if not args.no_limits:
            limits_path = args.limits or (DEFAULT_LIMITS if DEFAULT_LIMITS.is_file() else None)
        limits, note = load_limits(limits_path)
        started = time.time()
        summary = summarize_sweep(
            sweep, limits=limits, limits_note=note, workers=args.workers,
            expected_seeds=args.expected_seeds,
            progress=lambda done, total: print(f"  {done}/{total} runs read", file=sys.stderr))
        summary["elapsed_s"] = round(time.time() - started, 1)
        summary["limits_file"] = str(limits_path) if limits_path else None
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        write_markdown(summary, out / "summary.md", top=args.top)
        plots = [] if args.no_plots else make_plots(summary, out)
    except (SummaryError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Wrote {out}: {summary['n_groups']} groups, {summary['n_runs']} runs, "
          f"{len(plots)} plots, {summary['elapsed_s']} s; "
          f"ranking check: {summary['ranking_check']['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
