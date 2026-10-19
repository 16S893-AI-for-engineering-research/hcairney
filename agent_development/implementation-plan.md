# RL experiment review agent: implementation plan

Status: draft, reflects decisions agreed so far. Backlog tasks not yet created.

## 1. Goal

Build an assistant that reviews a finished RL training experiment, explains what
happened, and proposes reasoned next runs. Later it becomes one step in a loop:

propose → (user launches) → summarize → review → propose …

For now the agent **only reviews and proposes**. The user launches all runs.

## 2. Guiding rules

1. **Scripts handle the numbers; the agent interprets them.** Reading logs, all
   arithmetic, checking settings and writing configs are done by plain, tested code.
2. **Judge setups by their seed group, never by a single run.** Each run sets three
   seeds (`network_initialization`, `parameter_space_exploration`,
   `replay_sampling`). A *seed group* is the set of 3 runs (`-s1`, `-s2`, `-s3`)
   that differ only in these seeds. Every proposed setup is run as a full seed
   group, and is compared only with other full seed groups. Differences smaller
   than the seed-to-seed spread are reported as "unclear".
3. **Proposals use a fixed form** (section 6), so they can be checked and turned
   into configs without interpretation.
4. **The user stays in charge.** The agent never launches jobs.
5. **Keep a written history** (section 7) so the agent does not repeat itself.

## 3. Decisions

| Topic | Decision |
|---|---|
| Main score | Best successful, unperturbed `E_volume` (as in `scripts/rank_training_runs.py`) |
| Health checks | Convergence measures, e.g. rejected steps, failed solves, retries, nonlinear iterations. A setup with high counts is marked "unreliable" even if its score is good. Limits for "high" are set from past runs (section 4.3). |
| What the agent may change | Existing configurable settings only: number values (rates, noise, sizes, tolerances) **and** existing options (activation, input features, replay type, …). No code changes for now. The proposal form has a `kind` field so `"code_change"` can be added later. |
| Seeds | Every proposed setup runs as one seed group (3 seed sets). Evaluation uses mean, spread and worst case across the group. |
| History | A dedicated folder organised by campaign and round. Not stored in Backlog. |
| Agent permissions | Review and propose only. Submission may be revisited later. |
| Config generation | Needed, because the user launches runs: proposals are turned into ready-to-launch TOML files (one per seed set). |
| Model | Claude Opus 5.5 (`MIT-parley/bedrock/claude-opus-5-5`), high reasoning, pinned and recorded in each round. |
| Second opinion | Later addition: a model from another provider critiques proposals before the user sees them. |
| First campaign | Deferred. Build and test the summary script and review skill first. |

## 4. Phase 1: run summary script

**File:** `agent_development/scripts/summarize_runs.py` (with tests under `tests/`).

### 4.1 Input and grouping
- Input: a sweep folder, e.g. `runs/task-40_scaled_inputs_sweep`.
- Finds every run folder (contains `run.json`, `performance.jsonl`, `history.jsonl`).
- Groups runs into seed groups by removing the final `-s<number>`, matching
  `scripts/rank_training_runs.py`, and checks that group members differ only in seeds.
  Groups with a missing or failed member are reported as incomplete.

### 4.2 Output per seed group
- **Settings:** the values that differ between groups in this sweep (read from
  `run.json` / `inputs/submitted.toml`), so the agent sees what was varied.
- **Score:** best `E_volume` per seed, then mean, spread (min–max and standard
  deviation) and worst case; when the best value occurred.
- **Trend:** still improving at the end, levelled off, or got worse.
- **Health:** convergence counts per seed and their worst case, with flags against
  the limits in 4.3. Also failed or interrupted runs and whether the step limit was reached.

### 4.3 Health limits
- Measure the distribution of each convergence count across all runs in the two
  test sweeps.
- Propose starting limits (e.g. based on typical and extreme values), document
  them in a small config file, and let the user adjust them.

### 4.4 Output files
- `summary.json`: full machine-readable result.
- `summary.md`: short readable table, ranked by mean best `E_volume`, with
  "unreliable" and "incomplete" flags. Kept short enough for the agent to read whole.
- A few plots (training curves for the top and notable groups, score vs. main
  varied settings).

### 4.5 Done when
- Tests pass on both test sweeps.
- Group rankings agree with the existing `training_run_ranking_seeds.txt` files.
- The user finds `summary.md` correct and readable.

## 5. Phase 2: review skill

**Folder:** `.pi/skills/rl-run-review/` (contains `SKILL.md` and any helper notes).

- Use: `/skill:rl-run-review runs/<sweep>`, or without the interface:
  ```sh
  pi -p --no-session --model MIT-parley/bedrock/claude-opus-5-5 --thinking high \
     --tools read,bash --skill .pi/skills/rl-run-review "Review runs/<sweep>"
  ```
- The agent can only read files and run commands. It does not edit files.
- Steps: run the summary script, read `summary.md` and plots, then write a report in a
  fixed layout:
  1. What was tested (which settings varied)
  2. Key results, with spread across seeds
  3. Health concerns and anything odd
  4. Verdict: what is clearly better, what is unclear
  5. 1–3 proposed next setups, in the fixed form (section 6)
- **Done when:** reviews of the two test sweeps match the user's own conclusions,
  and the proposals are specific and sensible.

## 6. Proposal form

Each proposal is one JSON object. Draft fields:

```json
{
  "id": "P-001",
  "kind": "settings",
  "based_on": "<seed group directory>",
  "question": "What this tests",
  "changes": { "<setting path>": "<new value>" },
  "expected_result": "Predicted effect on mean best E_volume and on health",
  "would_change_my_mind": "Result that would show the idea is wrong",
  "evidence": ["Specific numbers from the summary that motivate it"],
  "cost": { "runs": 3, "attempted_solve_limit": 10000 },
  "priority": 1
}
```

- `kind`: only `"settings"` allowed for now.
- `changes`: only existing settings, with values in allowed ranges or options.
- `runs` is always a multiple of 3 (one seed group per setup).
- In phase 2 the form is a JSON block at the end of the review. A later upgrade is a
  pi tool (`submit_proposal`) that rejects badly formed proposals automatically.

## 7. Later phases

### Phase 3: proposal checker and config generator
- Checks that each proposal is well formed and that the settings in `changes` exist.
- Copies the base group's `inputs/submitted.toml`, applies the changes, and writes
  three TOML files (one per seed set, with new seed values).
- Runs the existing configuration checks (e.g. `--dry-run`) on each file.
- Output: ready-to-launch configs for the user.

### Phase 4: history folder
```
agent_development/campaigns/<campaign-name>/   # exact location to confirm
  campaign.json        # goal, starting seed group, allowed settings, stopping point
  round-001/
    summary.json       # summary script output
    review.md          # agent report
    proposals.json     # fixed-form proposals
    decision.json      # what the user approved or rejected
    configs/           # generated TOML files
    runs.txt           # links to run folders in runs/
  round-002/ …
  history.md           # one line per round; read by the next review
```
Run data stays in `runs/`; the history folder only points to it.

### Phase 5: loop driver
A plain script running: summarize → review → propose → check → generate configs →
save for user approval. Each step uses a fresh agent run (`pi -p --no-session …`).
The user launches runs and starts the next round.

### Phase 6 (optional)
- Second-opinion check by a model from another provider.
- Allow `"code_change"` proposals.
- Revisit whether the agent may submit jobs.

## 8. Test data

| Sweep | Contents |
|---|---|
| `runs/task-40_scaled_inputs_sweep` | 4 families (`relu_cell`, `relu_volume`, `tanh_cell`, `tanh_volume`), 324 seed groups, 972 runs |
| `runs/task-40_pretrained_actor_sweep` | Same 4 families, 324 seed groups, 972 runs, starting from a pretrained actor |

Both already contain `training_run_ranking_seeds.txt`, which is used to check the
summary script's rankings.

## 9. Open items

- **First campaign:** starting seed group, goal, allowed settings, stopping point, name. Deferred.
- **Health limits:** to be proposed from the test sweeps in phase 1.
- **Exact convergence fields:** to be confirmed from `history.jsonl` and run diagnostics.
- **History folder location:** inside `agent_development/` or elsewhere.

## 10. Main risks

| Risk | Safeguard |
|---|---|
| Treating seed noise as a real effect | Seed groups and the "smaller than spread is unclear" rule |
| Good score through poor convergence | Health checks the agent cannot override |
| Wrong or incomplete summaries | Tested script, checked against existing rankings |
| Vague or unrunnable proposals | Fixed form, checker, dry-run validation |
| Wasted compute | User launches every run |
