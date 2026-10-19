#!/usr/bin/env python
"""Check a sweep review report: section layout and the final proposal block.

Usage: python agent_development/scripts/check_review.py REPORT.md SWEEP_DIR

Prints each problem and exits with 1 if there are any, otherwise prints OK.
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

SECTIONS = [
    (1, "what was tested"),
    (2, "key results"),
    (3, "health concerns"),
    (4, "verdict"),
    (5, "proposed next setups"),
]
FIELDS = ["id", "kind", "based_on", "question", "changes", "expected_result",
          "would_change_my_mind", "evidence", "cost", "priority"]
TEXT_FIELDS = ["question", "expected_result", "would_change_my_mind"]
SEED_GROUP_RUNS = 3
FORBIDDEN_PREFIXES = ("run.", "seeds.")  # names, paths, seeds are set by config generation


def flatten(data, prefix=""):
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def kind_of(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def split_report(text):
    """Return (body before the last fenced block, JSON text or None, text after the block)."""
    blocks = list(re.finditer(r"^```json[ \t]*\n(.*?)^```[ \t]*$", text, re.S | re.M))
    if not blocks:
        return text, None, ""
    last = blocks[-1]
    return text[:last.start()], last.group(1), text[last.end():]


def check_layout(body, problems):
    headings = [(m.group(1), m.group(2).strip().lower(), m.start())
                for m in re.finditer(r"^##\s+(\d+)[.)]?\s+(.+?)\s*$", body, re.M)]
    found = []
    for number, title in SECTIONS:
        hit = [h for h in headings if int(h[0]) == number]
        if not hit:
            problems.append(f"missing section {number} ('## {number}. ...' for: {title})")
        elif title not in hit[0][1]:
            problems.append(f"section {number} heading should mention '{title}', got '{hit[0][1]}'")
        if hit:
            found.append(hit[0][2])
    if found != sorted(found):
        problems.append("sections are out of order")
    for number, _title in SECTIONS:  # sections must not be empty
        m = re.search(rf"^##\s+{number}[.)]?\s+.+?$(.*?)(?=^##\s+\d+[.)]?\s|\Z)", body, re.S | re.M)
        if m and len(m.group(1).strip()) < 20:
            problems.append(f"section {number} is empty")


def base_files(sweep, based_on):
    group = sweep / based_on
    runs = sorted(sweep.glob(f"{based_on}-s*/inputs/submitted.toml"))
    return group, runs


def check_proposal(index, p, sweep, problems):
    tag = f"proposal {index + 1}"
    if not isinstance(p, dict):
        problems.append(f"{tag}: not an object")
        return
    for field in FIELDS:
        if field not in p:
            problems.append(f"{tag}: missing field '{field}'")
    for field in p:
        if field not in FIELDS:
            problems.append(f"{tag}: unknown field '{field}'")
    if p.get("kind") != "settings":
        problems.append(f"{tag}: kind must be 'settings'")
    for field in TEXT_FIELDS:
        if field in p and not (isinstance(p[field], str) and p[field].strip()):
            problems.append(f"{tag}: '{field}' must be non-empty text")
    ev = p.get("evidence")
    if "evidence" in p and not (isinstance(ev, list) and ev and all(isinstance(e, str) and e.strip() for e in ev)):
        problems.append(f"{tag}: 'evidence' must be a non-empty list of text")
    if "id" in p and not (isinstance(p["id"], str) and p["id"].strip()):
        problems.append(f"{tag}: 'id' must be text")
    prio = p.get("priority")
    if "priority" in p and not (isinstance(prio, int) and not isinstance(prio, bool) and prio >= 1):
        problems.append(f"{tag}: 'priority' must be an integer >= 1")

    cost = p.get("cost")
    cost_ok = isinstance(cost, dict) and set(cost) == {"runs", "attempted_solve_limit"}
    if "cost" in p and not cost_ok:
        problems.append(f"{tag}: 'cost' must have exactly 'runs' and 'attempted_solve_limit'")
    if cost_ok and cost["runs"] != SEED_GROUP_RUNS:
        problems.append(f"{tag}: cost.runs must be {SEED_GROUP_RUNS} (one seed group)")

    based_on = p.get("based_on")
    if not isinstance(based_on, str):
        return
    group, files = base_files(sweep, based_on)
    if not files:
        problems.append(f"{tag}: based_on '{based_on}' is not a seed group in {sweep} (no inputs/submitted.toml found)")
        return
    try:
        with files[0].open("rb") as stream:
            base = tomllib.load(stream)
    except Exception as err:  # unreadable or invalid file
        problems.append(f"{tag}: cannot read {files[0]}: {err}")
        return
    flat = flatten(base)

    if cost_ok:
        limit = flat.get("run.attempted_solve_limit")
        if cost["attempted_solve_limit"] != limit:
            problems.append(f"{tag}: cost.attempted_solve_limit is {cost['attempted_solve_limit']}, "
                            f"base run has {limit}")

    changes = p.get("changes")
    if "changes" not in p:
        return
    if not isinstance(changes, dict) or not changes:
        problems.append(f"{tag}: 'changes' must be a non-empty object")
        return
    changed = False
    bad = False
    for key, value in changes.items():
        if key.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{tag}: '{key}' may not be changed (names, paths, seeds, solve limit)")
            bad = True
        elif key not in flat:
            problems.append(f"{tag}: setting '{key}' does not exist in the base submitted.toml")
            bad = True
        elif kind_of(value) != kind_of(flat[key]):
            problems.append(f"{tag}: '{key}' is a {kind_of(flat[key])} in the base, got a {kind_of(value)}")
            bad = True
        elif value != flat[key]:
            changed = True
    if not changed and not bad:
        problems.append(f"{tag}: no change differs from the base values")


def check_report(text, sweep):
    problems = []
    sweep = Path(sweep)
    body, block, after = split_report(text)
    check_layout(body, problems)
    if block is None:
        problems.append("no final fenced ```json block")
        return problems
    if after.strip():
        problems.append("text after the final json block (the block must be last)")
    try:
        proposals = json.loads(block)
    except json.JSONDecodeError as err:
        problems.append(f"final json block is not valid JSON: {err}")
        return problems
    if not isinstance(proposals, list):
        problems.append("final json block must be a list of proposals")
        return problems
    if not 1 <= len(proposals) <= 3:
        problems.append(f"expected 1-3 proposals, got {len(proposals)}")
    for i, p in enumerate(proposals):
        check_proposal(i, p, sweep, problems)
    ids = [p.get("id") for p in proposals if isinstance(p, dict)]
    if len(set(map(str, ids))) != len(ids):
        problems.append("proposal ids are not unique")
    prios = [p.get("priority") for p in proposals if isinstance(p, dict)]
    if len(set(map(str, prios))) != len(prios):
        problems.append("proposal priorities are not unique")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("sweep", type=Path)
    args = ap.parse_args(argv)
    problems = check_report(args.report.read_text(), args.sweep)
    for problem in problems:
        print("PROBLEM:", problem)
    if not problems:
        print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
