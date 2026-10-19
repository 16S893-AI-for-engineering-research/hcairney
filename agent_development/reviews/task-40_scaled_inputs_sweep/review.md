<!-- model: MIT-parley/bedrock/claude-opus-5-5 | thinking: high | date: 2026-10-19T21:55:35Z | sweep: runs/task-40_scaled_inputs_sweep -->

# Review: task-40_scaled_inputs_sweep

## 1. What was tested
- **Runs:** 972 runs in 324 seed groups, with 3 seeds per group. Every group got a score. The ranking matches `training_run_ranking_seeds.txt`.
- **Families:** `relu_cell`, `relu_volume`, `tanh_cell`, `tanh_volume`.
- **Settings varied:**
  - `actor.activation` / `critic.activation`: relu, tanh
  - `exploration.sigma`: 0.003, 0.01, 0.03
  - `learning.critic_updates_per_evaluation`: 4, 8, 16
  - `replay.batch_size`: 64, 128, 256
  - `replay.capacity_evaluations` / `replay.warmup_evaluations`: 32/8, 64/16, 128/32
  - `replay.cell_sampling`: cells, volume
- **Linked pairs:** two pairs always changed together, so the sweep can't tell their effects apart:
  - actor and critic activation
  - replay capacity and warmup
- **Score:** the best `E_volume` per seed. Lower is better.

## 2. Key results

| Group | Mean | Std | Min–max | Flags |
|---|---|---|---|---|
| relu_volume/…-sigma-0p003-b256-u16-w8-c32 (rank 1) | 1.070e-05 | 4.106e-06 | 6.03e-06 – 1.375e-05 | none |
| relu_cell/…-sigma-0p003-b128-u16-w32-c128 (rank 2) | 1.953e-05 | 1.958e-05 | 5.17e-06 – 4.183e-05 | none |
| relu_volume/…-sigma-0p003-b256-u16-w16-c64 (rank 3) | 2.118e-05 | 9.870e-06 | 9.84e-06 – 2.784e-05 | none |
| relu_volume/…-sigma-0p003-b256-u8-w8-c32 (rank 18) | 7.573e-05 | 8.460e-05 | 2.38e-05 – 1.734e-04 | none |
| tanh_volume/…-sigma-0p003-b128-u16-w32-c128 (rank 22, best tanh) | 7.915e-05 | 5.197e-05 | 2.41e-05 – 1.274e-04 | none |
| relu_volume/…-sigma-0p003-b256-u4-w8-c32 (rank 26) | 8.966e-05 | 9.542e-05 | 1.88e-05 – 1.982e-04 | none |
| relu_volume/…-sigma-0p01-b256-u16-w8-c32 (rank 49) | 1.265e-04 | 2.161e-05 | 1.017e-04 – 1.415e-04 | none |

What the settings did:
- **Activation:** relu beats tanh in 159 of 162 matched pairs of groups. Median group mean is 2.01e-04 for relu and 1.24e-02 for tanh. No tanh group is in the top 21.
- **Exploration noise (sigma):** 0.003 gives ranks 1–19. The best groups at 0.01 and 0.03 are rank 20 and rank 100.
- **Critic updates:** 16 gives ranks 1–15. The best groups with 8 and 4 are rank 16 and rank 26. In matched pairs, 16 beats 8 only 67 times out of 108, so the effect is mostly seen among the best setups.
- **Batch size, cell sampling, replay capacity/warmup:** no consistent effect.
  - Batch 256 beats 128 in 54 of 108 pairs.
  - Volume sampling beats cell sampling in 71 of 162 pairs.
  - Capacity 32 beats 128 in 49 of 108 pairs.

## 3. Health concerns
- **15 unreliable groups.** They are all tanh with sigma 0.03. All were flagged for more than 450 inner solver steps per solve (`nonlinear_iterations_per_solve`). They rank 242–320, so none is a candidate for best.
- **Top groups are healthy.** No rejected steps or failed solves, and about 210 solver steps per solve.
- **Late-run drops.** `final_over_best` is how much worse the last check was than the best check. For the rank-1 seeds it is 4.0, 5.0 and 7.1, so the policy gets noticeably worse between checks. One rank-10 seed has 106.7 and one rank-2 seed has 12.8.
- **Still improving at the end.** In the top 15 groups, the best check usually comes near the end (check 81–100 of 101, with one at 11). Most seeds are still "improving". Rank 1 is 2 improving and 1 flat, with best checks at 89, 96 and 98.

## 4. Verdict
Rule: "clearly better" means the gap between group means is larger than the seed std of both groups, and the min–max ranges do not overlap.

**Clearly better**
- **Relu over tanh.** Best relu is 1.070e-05 against best tanh at 7.915e-05. The gap of 6.85e-05 is larger than both stds (4.1e-06 and 5.2e-05). The ranges don't overlap: the tanh minimum of 2.41e-05 is above the relu maximum of 1.375e-05. This also holds in 159 of 162 matched pairs.
- **Sigma 0.003 over 0.01 at the rank-1 settings.** 1.070e-05 against 1.265e-04. The gap of 1.16e-04 is larger than both stds, and the ranges don't overlap.

**Unclear**
- **Rank 1 vs rank 2.** The gap is 8.8e-06, smaller than rank 2's std of 1.96e-05.
- **Rank 1 vs rank 3.** The gap of 1.05e-05 is larger than both stds, but the ranges overlap.
- **16 vs 8 critic updates at the rank-1 settings.** The gap is 6.5e-05, smaller than the std of the 8-update group (8.46e-05).
- **Batch size, cell sampling and replay capacity/warmup:** no clear effect anywhere.

Overall, rank 1 (relu_volume …-sigma-0p003-b256-u16-w8-c32) is the best and steadiest group, but it isn't clearly ahead of ranks 2–3. Both of its strong settings, sigma 0.003 and 16 critic updates, are at the edge of the tested range. Scores were still improving at the end and late checks dropped, so a longer run might help. That can't be proposed as a settings change here.

## 5. Proposed next setups
1. **Lower sigma to 0.001 on the rank-1 group.** Sigma 0.003 is the smallest value tried and holds ranks 1–19, so a smaller value is the obvious next test. This goes beyond the tested range. I expect a similar or slightly lower mean and perhaps smaller late-run drops.
2. **Raise critic updates per evaluation to 32 on the rank-1 group.** 16 is the largest value tried and holds ranks 1–15, so this also goes beyond the tested range. I expect a modestly lower mean with unchanged health, though each run will take longer. The evidence for this is weaker than for sigma.

```json
[
  {
    "id": "P-001",
    "kind": "settings",
    "based_on": "relu_volume/task40-scaled-relu_volume-sigma-0p003-b256-u16-w8-c32",
    "question": "Does exploration noise below the tested range (sigma 0.001) improve the best relu_volume setup?",
    "changes": {"exploration.sigma": 0.001},
    "expected_result": "Mean best E_volume at or below 1.07e-05 (low confidence, extrapolating beyond tested range); no rejected steps or failed solves, about 212 it/solve.",
    "would_change_my_mind": "Mean above 2.1e-05 (worse than rank 3) or any seed above 4e-05.",
    "evidence": [
      "sigma 0.003 groups hold ranks 1-19; best sigma 0.01 group is rank 20 (7.688e-05), best sigma 0.03 group is rank 100 (2.665e-04)",
      "same base settings with sigma 0.01: mean 1.265e-04, min 1.017e-04, versus 1.070e-05, max 1.375e-05 at sigma 0.003"
    ],
    "cost": {"runs": 3, "attempted_solve_limit": 10000},
    "priority": 1
  },
  {
    "id": "P-002",
    "kind": "settings",
    "based_on": "relu_volume/task40-scaled-relu_volume-sigma-0p003-b256-u16-w8-c32",
    "question": "Does doubling critic updates per evaluation beyond the tested maximum (16 to 32) improve the best relu_volume setup?",
    "changes": {"learning.critic_updates_per_evaluation": 32},
    "expected_result": "Mean best E_volume around 8e-06 to 1.07e-05 (low confidence, extrapolating); health unchanged at about 212 it/solve with no failures.",
    "would_change_my_mind": "Mean at or above 2e-05, or any seed worse than 4e-05.",
    "evidence": [
      "u16 groups hold ranks 1-15; best u8 group is rank 16 (6.664e-05), best u4 group is rank 26 (8.966e-05)",
      "at base settings, u8 gives mean 7.573e-05 (std 8.460e-05) and u4 gives 8.966e-05 (std 9.542e-05) versus 1.070e-05 at u16; gap within u8/u4 spread, so unclear",
      "u16 beats u8 in 67 of 108 matched pairs (summary.json)"
    ],
    "cost": {"runs": 3, "attempted_solve_limit": 10000},
    "priority": 2
  }
]
```
