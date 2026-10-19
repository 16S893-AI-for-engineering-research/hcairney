# Proposal form

The report ends with one fenced `json` block holding a list of 1–3 proposals. Nothing may
follow the block. Each proposal is one object with exactly these fields:

```json
{
  "id": "P-001",
  "kind": "settings",
  "based_on": "relu_volume/task40-scaled-relu_volume-sigma-0p003-b256-u16-w8-c32",
  "question": "Does a larger critic update count help the best relu_volume setup?",
  "changes": {"learning.critic_updates_per_evaluation": 32},
  "expected_result": "Mean best E_volume falls below 1.0e-05; health unchanged.",
  "would_change_my_mind": "Mean at or above 1.3e-05, or any seed worse than 2e-05.",
  "evidence": ["u16 groups hold ranks 1-15 while u4/u8 appear later (summary.md top 15)"],
  "cost": {"runs": 3, "attempted_solve_limit": 10000},
  "priority": 1
}
```

| Field | Rule |
|---|---|
| `id` | Short unique text, e.g. `P-001`. |
| `kind` | Always `"settings"`. |
| `based_on` | Seed group name exactly as in `summary.md` (family folder + group name, no `-s1`). It must be a group in this sweep. |
| `question` | One sentence: what the run tests. |
| `changes` | Setting path → new value. Paths are the flattened `section.key` names of the base group's `inputs/submitted.toml` (e.g. `replay.batch_size`). Only settings that already exist there. A value keeps the same type as the base (number for number, text for text, list for list). At least one value must differ from the base. Never change `run.*` or `seeds.*`. |
| `expected_result` | Predicted effect on mean best `E_volume` and on health, with numbers. |
| `would_change_my_mind` | A result that would show the idea is wrong, with numbers. |
| `evidence` | Non-empty list of text; each item quotes numbers from `summary.md` / `summary.json`. |
| `cost` | `runs`: always 3 (one seed group). `attempted_solve_limit`: copy `run.attempted_solve_limit` from the base group's `inputs/submitted.toml`; do not guess. |
| `priority` | Integer from 1 (do first); unique across proposals. |

Rules of thumb
- Change one setting, or a few that belong together (e.g. linked settings). Say why in `question`.
- Linked settings (listed in `summary.md`) must be changed together, otherwise the config may be invalid.
- Stay near values already tested unless the evidence points beyond them; say so when extrapolating.
- Pick a base group that is reliable and complete.
- If the sweep has nothing worth proposing, still give at least one proposal and explain the low confidence in `expected_result`.
