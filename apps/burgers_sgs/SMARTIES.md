# Burgers SMARTIES executable

The optional `burgers_smarties` target is configured from the repository root
with `BURGERS_ENABLE_SMARTIES=ON`. Standalone builds of `apps/burgers_sgs`
leave the option off and retain no SMARTIES, MPI, or OpenMP dependency.

The application arguments must be supplied through SMARTIES `--appSettings`.
The example argument file is
[`configs/burgers_rl_app_settings.txt`](configs/burgers_rl_app_settings.txt),
and the strict versioned environment document is
[`configs/burgers_rl.json`](configs/burgers_rl.json).

Before launching, assemble a flat setup directory containing:

- `burgers_rl.json`;
- the referenced solver configuration as `les_test.json`;
- the example [SMARTIES learner configuration](configs/settings.json) as
  `settings.json` (or a complete replacement); and
- `dns_target_metadata.json`, `dns_target.csv`, and `dns_spectrum.csv` from
  `runs/finalization`.

Pass that directory to SMARTIES with `--setupFolder`. SMARTIES copies its files
into each simulation directory before the callback starts, so all paths in the
example RL configuration resolve locally. The callback requires the explicit
`settings.json`, and users should specify every learner field rather than rely
on implicit SMARTIES defaults.

Each simulation directory receives `burgers_rl_resolved.json`,
`burgers_solver_resolved.json`, the original `settings.json`, and
`burgers_rl_episodes.csv`. Training episode seeds come from the reproducible
SMARTIES communicator random stream. Evaluation consumes the configured seed
list in order and initially requires one SMARTIES evaluation environment.

Detailed evaluation output is opt-in through `output.write_evaluation_output`
in `burgers_rl.json`; omitting it preserves the original rollout path and
defaults to `false`. When enabled, each held-out episode is written beneath
`output.evaluation_directory` in a directory named from its evaluation index
and seed. Each directory contains the same scalar history, profile history,
final profile, optional mean spectrum, and run metadata written by
`run_solver`. The scalar and profile clocks come from the referenced solver
configuration. Reaching those clocks may subdivide a policy decision, but the
selected action remains fixed for the entire decision interval.

Three additional CSV files retain policy-specific diagnostics:

- `burgers_rl_history.csv` contains one scalar row per decision, including
  reward, action-projection, stability, and advancement diagnostics;
- `burgers_rl_fields.csv` contains cell-resolved actions, rewards, running-mean
  estimates, and target values in long form; and
- `burgers_rl_spectra.csv` contains the raw and applied action spectra when
  `environment.record_action_spectra` is enabled.
