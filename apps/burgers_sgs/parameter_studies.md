# Phase 5 parameter-study workflow

## Configuration

`run_solver` accepts a complete, versioned JSON configuration. The checked-in
baseline is `configs/phase5_validation.json`. Configuration documents are
strict: all fields are required, unknown fields are rejected, and the existing
scientific validation is applied after parsing.

Typical use from `apps/burgers_sgs` is:

```sh
mkdir -p runs/stationarity/baseline/seed_5489
build/run_solver \
  --config configs/phase5_validation.json \
  --output-directory runs/stationarity/baseline/seed_5489 \
  --seed 5489
```

The output directory must exist before starting the solver. To inspect the
fully resolved configuration without advancing the PDE or creating run output:

```sh
build/run_solver --config configs/phase5_validation.json --dry-run
```

The command-line seed and output directory overrides are deliberately narrow.
Scientific parameters should remain in versioned JSON files rather than a long
shell command. Copy the baseline and change one factor at a time for the clock,
amplitude, correlation-time, resolution, and forcing-band studies.

The Chekhlov--Yakhot-type cases use `spectral_exponent: 1.0` and explicit
`wavenumbers` arrays containing every mode from 1 through the selected maximum.
For `k_max=32`, use at least 512 cells for the coarsest comparison unless the
finite-volume attenuation on a 256-cell grid is explicitly accepted.

## Output sampling

`history_interval` controls scalar samples. `profile_interval` independently
controls profile snapshots and online-spectrum samples. The time integrator
advances to the union of these physical-time schedules, while energy-budget
terms are accumulated between scalar-history rows.

For production parameter studies, use:

```json
"write_profile_history": true,
"write_online_spectrum": false
```

and calculate spectra offline. The C++ direct DFT remains available as a
reference implementation and for small verification runs.

Rows after the initial scalar-history row contain interval-averaged power and
dissipation. `interval_start_time` and `interval_duration` identify that
interval. The initial row has zero interval duration; its power and molecular
dissipation values are instantaneous and should not enter interval-weighted
stationary averages.

## Python analysis

Create a Python environment and install:

```sh
python3 -m pip install -r requirements-analysis.txt
```

Plot the kinetic-energy history and energy injection/dissipation balance for
one run with:

```sh
python3 scripts/plot_history.py \
  runs/stationarity/baseline/seed_5489/burgers_history.csv
```

This creates `KE.png` and `balance.png` beside the input CSV. Pass
`--output-directory DIRECTORY` to write both figures elsewhere.

Analyze one run over a selected stationary window:

```sh
python3 scripts/analyze_run.py \
  runs/stationarity/baseline/seed_5489 \
  --window 10:50 \
  --stationarity-split 30
```

This writes:

- `analysis_summary.json`: scalar means, autocorrelation estimates, block
  standard errors, stationarity comparisons, and resolution indicators;
- `analysis_arrays.npz`: pointwise temporal means and variances, their block
  uncertainties, and raw and compensated spectra with block uncertainties.

Plot the temporal mean velocity profile and its pointwise temporal block
standard error with:

```sh
python3 scripts/plot_mean_velocity_profile.py \
  runs/stationarity/baseline/seed_5489/analysis_arrays.npz
```

This writes `mean_velocity_profile.png` beside the input NPZ file.

If `--block-duration` is omitted, the script uses five times the largest
estimated integrated autocorrelation time among the principal scalar
observables. Always inspect the resulting block count. Fewer than ten complete
blocks means that the uncertainty estimate is weak and the run should normally
be extended.

The stationarity comparison reports a Welch statistic based on independent
block means. It is a screening diagnostic, not proof of stationarity. Define
practical equivalence tolerances before interpreting a small statistic as
scientific convergence.

Combine independently seeded analyses with:

```sh
python3 scripts/analyze_ensemble.py \
  runs/candidate/seed_5489 \
  runs/candidate/seed_13007 \
  runs/candidate/seed_26003 \
  runs/candidate/seed_52009 \
  -o runs/candidate/ensemble_summary.json
```

This also writes `ensemble_arrays.npz` beside the ensemble summary. It contains
the pointwise mean of the seed mean profiles and the standard error calculated
from their between-seed sample variability. The same plotting command accepts
this archive and labels its uncertainty as between-seed:

```sh
python3 scripts/plot_mean_velocity_profile.py \
  runs/candidate/ensemble_arrays.npz
```

The ensemble report contains the Student-t interval across seed means and a
hierarchical bootstrap interval that resamples both seeds and within-run time
blocks. Four seeds is a minimum; more are preferable.

Compare two analyzed resolutions or parameter values with:

```sh
python3 scripts/compare_analyses.py \
  runs/resolution/n512/seed_5489 \
  runs/resolution/n1024/seed_5489 \
  -o runs/resolution/n512_vs_n1024.json
```

Fine pointwise fields are conservatively restricted to the common coarse grid.
By default, spectra are compared only through one quarter of the common cell
count, rather than through the coarse-grid Nyquist mode. Use `--maximum-mode`
to set a study-specific common resolved band.

## Recommended sequence

1. Run a long baseline and determine spin-up and flow autocorrelation time.
2. Repeat the baseline across at least four seeds.
3. Compare forcing-clock intervals `tau/5`, `tau/10`, and `tau/20` across
   seeds; changing the clock does not produce pathwise-coupled realizations.
4. Sweep stochastic RMS and deterministic amplitude separately.
5. Sweep correlation time while maintaining `clock_interval=tau/10`.
6. At the selected forcing parameters, compare 256, 512, and 1024 cells at
   fixed molecular viscosity.
7. Run the forcing-band study only on grids that comfortably represent every
   forced mode.

Do not use the cumulative work and dissipation totals in run metadata as
stationary averages: those totals include spin-up. Use interval-weighted scalar
history over the selected analysis window.
