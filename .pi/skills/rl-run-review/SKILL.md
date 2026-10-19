---
name: rl-run-review
description: Review a finished RL training sweep in this project (a runs/<sweep> folder of seed groups) and propose 1-3 next setups as a fixed-format report with a final JSON block. Use when asked to review, summarise or propose next runs for an RL sweep; not for launching jobs or for single runs.
---

# RL sweep review

Review one finished sweep, say what the numbers show, and propose next setups. You only
review and propose. The user launches every run.

Input: a sweep folder such as `runs/task-40_scaled_inputs_sweep`. If none is given, ask.
Run commands from the project root (`/home/gridsan/hcairney/ALD/channel_rans`).

## Hard rules

- Never launch jobs: no `LLSUB`, `sbatch`, `srun`, `submit_training.sh`, or any training/solver command.
- Never edit, move or delete anything under `runs/`. Never write files. Your report is your final
  answer; the wrapper `agent_development/scripts/review_sweep.sh` saves it.
  The one exception: if the summary is missing (below), you may run the summary script with `--out`
  pointing to `agent_development/reviews/<sweep-name>/summary/`.
- Numbers come from the summary (`summary.md`, `summary.json`). Do not compute statistics from raw
  logs (`performance.jsonl`, `history.jsonl`). Simple arithmetic on summary numbers (ratios,
  differences) is fine. Never read a whole `history.jsonl` (very large).
- Judge setups only by seed groups (3 seeds), never by a single run or one lucky seed.
- A difference between two groups is **clear** only if the gap between their means is larger than the
  seed spread of both (use std and min/max). Otherwise call it **unclear**. Say which rule you applied.
- A group flagged `unreliable` or `incomplete` can never be called "best" without saying so in the same sentence.
- Say plainly when evidence is thin. Do not invent explanations the numbers do not support.
- Keep the language plain. Avoid jargon; explain a term once if you must use it.

## Steps

1. **Find the summary.** Look for `<sweep>/agent_summary/summary.md` and `summary.json`. If missing, run
   `python agent_development/scripts/summarize_runs.py <sweep> --out agent_development/reviews/<sweep-name>/summary`
   and use that folder. Read `docs/summarize-runs.md` if any term (score, trend, health, flags) is unclear.
2. **Read `summary.md` whole.** It lists varied settings, the effect of each setting value, top groups,
   best group per family, flagged groups and health limits.
3. **Look at the plots** (`score_vs_settings.png`, `training_curves_top_groups.png`) with `read`.
4. **Dig into numbers when needed** from `summary.json` using short `python -c` or `jq` queries (per-seed
   scores, trend, `final_over_best`, health). Use this to check your claims, for example the seed values of
   the groups you compare, or whether an effect holds in more than one family.
5. **Pick the base group** for each proposal and read its `inputs/submitted.toml`
   (`<sweep>/<group>-s1/inputs/submitted.toml`) to see the real setting names, current values and
   `run.attempted_solve_limit`.
6. **Read `references/proposal-form.md`** (in this skill folder) before writing proposals.
7. **Write the report** in the layout below as your final answer.

## Things to watch

- `final_over_best` far above 1 means the run's last check was much worse than its best: the score
  is the best check, and the policy may be unstable late in training. Mention it when it is large for top groups.
- Best check near the end of training plus an `improving` trend suggests a longer run could help
  (but the solve limit cannot be proposed as a change; only mention it in the verdict).
- The "effect of each setting value" table pools all other settings, so treat it as a rough guide and confirm with the group table.
- Linked settings cannot be separated by this sweep. Do not claim an effect for one of them alone.
- Best groups at the edge of a tested range suggest the next value to try beyond the edge.
- A seed group with large spread (std close to the mean) is not better than a neighbour just because its mean is lower.

## Report layout (fixed)

Start with the heading `# Review: <sweep name>`. Then exactly these five sections, with these
heading texts, in this order, then the JSON block. Nothing after the block.

```
## 1. What was tested
## 2. Key results
## 3. Health concerns
## 4. Verdict
## 5. Proposed next setups
```

1. **What was tested**: settings varied and their values, number of groups and runs, families, which settings are linked.
2. **Key results**: a small table of the groups that matter (name, mean, std, min–max, flags) and the main setting effects, each with its spread across seeds.
3. **Health concerns**: unreliable/incomplete groups and why, trends, `final_over_best` problems, anything odd (e.g. the best group has large spread). Write "none found" if so.
4. **Verdict**: two lists, "Clearly better" (gap larger than spread) and "Unclear" (gap within spread or evidence too thin). Name any unreliable/incomplete group you mention.
5. **Proposed next setups**: 1–3, in priority order, each explained in two or three plain sentences (what changes, why, what result you expect). The structured version goes in the JSON block.

Finally add one fenced `json` block: a list with the same proposals in the fixed form described in
`references/proposal-form.md`. `kind` is `"settings"`; `changes` use only settings that exist in the base
group's `inputs/submitted.toml`; `cost.runs` is 3; `cost.attempted_solve_limit` is copied from the base run.

## Self-check before answering

- Every number in the report appears in the summary (or is simple arithmetic on summary numbers).
- Every claim of "better" passes the spread rule.
- Every `changes` key was checked against the base `submitted.toml`, and `based_on` is a group name from `summary.md`.
- The wrapper runs a format checker afterwards; follow the layout and proposal form exactly.
