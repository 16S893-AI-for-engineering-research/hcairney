# Phase 6 DNS-reference finalization

## Purpose

Phase 6 converts verified forced-solver runs into a portable offline DNS target.
Finalization is an acceptance workflow, not another numerical solver. It checks
grid convergence, physical-scale resolution, numerical dissipation, spin-up,
and sampling convergence before writing a target that a later SMARTIES
environment can load.

The versioned default study is `configs/phase6_study.json`. All paths in that
file are resolved relative to the study file, so the workflow does not depend
on the shell's current directory.

## Files

- `configs/phase6_grid_n1024.json` and `phase6_grid_n2048.json` define the
  paired seed-5489 grid study through `t=200`.
- `configs/phase6_target_n1024.json` defines the `t=320` target runs. Override
  only the output directory and seed on the command line.
- `scripts/analyze_run.py` computes scalar, profile, variance, and spectrum
  statistics. Its NPZ output includes the complete-block profile arrays used by
  the finalizer.
- `scripts/finalize_phase6.py` evaluates the study definition and writes the
  acceptance report and, when every required check passes, the DNS target.

Install the analysis dependencies before using the Python tools:

```sh
python3 -m pip install -r requirements-analysis.txt
```

## Run-directory safety

Always start a solve in a new, empty directory. `run_solver` streams the profile
history while it is advancing and writes final metadata only at termination.
Reusing a directory can therefore leave an old completed metadata file or old
analysis beside a new partial profile. The finalizer checks completion status,
configured cell count, analysis shape, windows, seeds, and scientific
configuration, but a unique directory remains the clearest run-lifecycle
boundary.

Do not analyze the active `n2048` run until it has completed. After completion,
confirm that its metadata reports 2048 cells, status `completed`, and final time
200, then regenerate its analysis so the retained Phase 6 block arrays are
present.

## 1. Produce or retain the paired grid runs

The checked-in study definition initially points at the existing
`runs/test/grid_refinement` directories. To produce fresh equivalent runs:

```sh
mkdir -p runs/finalization/grid/n1024/seed_5489
build/run_solver \
  --config configs/phase6_grid_n1024.json \
  --output-directory runs/finalization/grid/n1024/seed_5489 \
  --seed 5489

mkdir -p runs/finalization/grid/n2048/seed_5489
build/run_solver \
  --config configs/phase6_grid_n2048.json \
  --output-directory runs/finalization/grid/n2048/seed_5489 \
  --seed 5489
```

If fresh paths are used, update `grid_candidate_analysis` and
`grid_reference_analysis` in `configs/phase6_study.json`.

Analyze both runs over the identical stationary window and block duration:

```sh
python3 scripts/analyze_run.py \
  runs/test/grid_refinement/n1024/seed_5489 \
  --window 20:200 --block-duration 15

python3 scripts/analyze_run.py \
  runs/test/grid_refinement/n2048/seed_5489 \
  --window 20:200 --block-duration 15
```

These commands replace `analysis_summary.json` and `analysis_arrays.npz` in
each run directory. The finalizer requires the block arrays written by the
current `analyze_run.py`; older NPZ files are rejected with an instruction to
reanalyze.

## 2. Produce the target ensemble

The default target uses four independent seeds at 1024 cells:

```text
5489 13007 26003 52009
```

For each seed, create a unique directory and run the same versioned
configuration. For example:

```sh
mkdir -p runs/finalization/target/n1024/seed_5489
build/run_solver \
  --config configs/phase6_target_n1024.json \
  --output-directory runs/finalization/target/n1024/seed_5489 \
  --seed 5489
```

Repeat with seeds 13007, 26003, and 52009. The target configuration runs to
`t=320`, discards `t<20`, keeps scalar output every 0.05 time units, and writes
profiles every 0.25 time units. The profile interval is an integer multiple of
the forcing clock and supplies 60 snapshots per 15-unit block while reducing
raw CSV volume.

Analyze every target run over the complete target window:

```sh
python3 scripts/analyze_run.py \
  runs/phase6/target/n1024/seed_5489 \
  --window 20:320 --block-duration 15
```

Repeat for the other three seeds. One full-window analysis is sufficient: the
finalizer uses retained block start/end times to form the `20:170` early window
and the `170:320` late window without rereading the profile CSV.

To use a single long realization instead, remove the other target analyses
from the study file and set `minimum_target_seed_count` to 1. This is supported
but is weaker evidence than the default ensemble.

## 3. Review the study definition

Before finalizing, review and commit `configs/phase6_study.json`. It records:

- candidate and finer-grid resolutions and paired seed;
- spin-up, grid-comparison, early, late, and target windows;
- block duration, confidence level, and deterministic bootstrap seed;
- the common dynamically important spectrum band;
- every numerical acceptance threshold; and
- the expected input and output paths.

The default acceptance limits are:

| Check | Maximum |
| --- | ---: |
| Grid mean-profile relative L2 change | 0.1% |
| Grid variance-profile relative L2 change | 0.5% |
| Grid kinetic-energy relative change | 0.1% |
| Grid molecular-dissipation relative change | 2% |
| Grid mean-square-gradient relative change | 2% |
| Spectrum relative L1 change, modes 1 through 128 | 0.5% |
| Candidate numerical/molecular dissipation | 1% |
| Candidate energy above mode 128 | 0.001% |
| Sampling mean-profile relative L2 upper bound | 2% |

Each early and late sampling window must contain at least ten complete blocks,
and the default target must contain four distinct seeds. The finalizer also
requires the observed early/late profile change to be consistent with the 95%
null block-bootstrap distribution. Change thresholds only for a documented
scientific reason, not in response to whether a completed run passed.

## 4. Finalize

Run from `apps/burgers_sgs`:

```sh
python3 scripts/finalize_phase6.py configs/phase6_study.json
```

Use a temporary output directory while testing a modified definition:

```sh
python3 scripts/finalize_phase6.py \
  configs/phase6_study.json \
  --output-directory runs/phase6/finalized_trial
```

The command always writes `phase6_report.json` after it has loaded and
validated all inputs. Its process exit status is zero only when every acceptance
check passes. A failed scientific check writes the report, returns nonzero, and
does not write or overwrite a DNS target.

When every check passes, the finalizer additionally writes:

- `dns_target.csv`, with cell centers, mean velocity, within-run temporal
  standard error, between-seed standard error, hierarchical-bootstrap combined
  standard error, temporal variance, and variance standard error;
- `dns_target_metadata.json`, with sampling counts and duration, uncertainty
  method, complete run/build/configuration metadata, hashes of every source
  analysis, the target CSV hash, and an embedded copy of the acceptance report.

The CSV is the portable field file intended for the future C++ environment.
The metadata JSON is the evidence and provenance record that makes it an
accepted DNS target rather than an unqualified mean profile.

## Statistical interpretation

Grid runs share the same seed and physical-time forcing clock. When their block
times match, the finalizer conservatively restricts fine cell averages to the
candidate grid and reports a paired block standard error for the profile
difference. This removes most forcing-realization variability from the grid
comparison.

For sampling convergence, complete temporal blocks preserve the spatial
correlations within each mean profile. The finalizer performs a hierarchical
bootstrap across seeds and within-run blocks. It reports the observed
early/late relative L2 difference, its null-distribution uncertainty, the 95%
upper bound on the resampled early/late difference, and the final target's
relative sampling uncertainty.

Pointwise errors are retained in the target CSV, but the pass/fail decision is
based on whole-profile norms rather than percentages near zero crossings.

## Tests

Run the dependency-light C++ suite and Python analysis tests with:

```sh
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

The Python interpreter must have NumPy and Matplotlib installed.
