# Sweep review agent

Reads a finished RL sweep and writes a review with 1–3 proposed next setups. It only reviews;
you launch every run. Design: `implementation-plan.md`. Numbers come from `agent_development/scripts/summarize_runs.py`
(`docs/summarize-runs.md`).

## Run a review

```sh
agent_development/scripts/review_sweep.sh runs/<sweep>
```

This runs `pi -p` with Claude Opus 5.5 (`MIT-parley/bedrock/claude-opus-5-5`), high thinking, and
only the `read` and `bash` tools, using the skill in `.pi/skills/rl-run-review/`. It takes about
1.5 minutes. In an interactive pi session you can use `/skill:rl-run-review runs/<sweep>` instead
(then you must save the answer yourself).

If the sweep has no `agent_summary/` folder, the agent runs the summary script first and puts the
output under `agent_development/reviews/<sweep>/summary/`.

## Outputs

`agent_development/reviews/<sweep-name>/`

| File | Content |
|---|---|
| `review.md` | The report. First line records model, thinking level, date and sweep. An older `review.md` is renamed `review-<timestamp>.md`. |
| `check.txt` | Result of the format checker (`OK` or a list of problems) |

Run folders under `runs/` are never written to.

## Report layout

1. What was tested  2. Key results  3. Health concerns  4. Verdict  5. Proposed next setups
then a final `json` block with the proposals. The proposal fields are in
`.pi/skills/rl-run-review/references/proposal-form.md`.

## Format checker

```sh
python agent_development/scripts/check_review.py agent_development/reviews/<sweep>/review.md runs/<sweep>
python tests/test_check_review.py        # tests
```

It checks the five sections, the JSON block, that every proposal uses the fixed fields,
`kind = "settings"`, 3 runs, the solve limit of the base run, and that every changed setting exists
(with the same value type) in the base group's `inputs/submitted.toml`. It cannot judge whether the
reasoning or the numbers are right; read the report yourself.
