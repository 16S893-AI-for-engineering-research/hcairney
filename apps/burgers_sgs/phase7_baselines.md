# Phase 7 static SGS baseline

## Implemented scope

Phase 7 adds `NoClosure`, `StaticSmagorinsky`, and
`PrescribedCoefficientField` behind one closure interface. Dynamic
Smagorinsky is deliberately deferred and a configuration selecting it is
rejected with a descriptive error.

Configuration and metadata express the conventional nonnegative coefficient
\(C_S\). Internally the closure stores

\[
  q_i=C_{S,i}^2,
\]

and evaluates

\[
  \nu_{\mathrm{sgs},i}
  =q_i\Delta x^2\left|
    \frac{u_{i+1}-u_{i-1}}{2\Delta x}
  \right|.
\]

The initial production convention interpolates total effective viscosity to
faces arithmetically. The resulting viscous flux remains conservative. Eddy
viscosity is recalculated from the current velocity at every SSP-RK3 stage,
while a prescribed coefficient field is held until explicitly replaced.

The public prescribed-field setter accepts one \(C_{S,i}\) per cell, validates
the configured bounds, and rejects use before the first complete field has
been supplied. This API is independent of SMARTIES.

## Diagnostics

Scalar histories now separate molecular, SGS, and numerical dissipation. The
numerical term remains the residual of the time-discrete SSP-RK3 balance after
subtracting both physical dissipation mechanisms. Histories also record the
minimum, mean, and maximum \(C_S\), plus mean and maximum eddy viscosity.

The Phase 6 finalizer exports `dns_spectrum.csv` and ensemble scalar energy and
dissipation statistics in `dns_target_metadata.json`, in addition to the
existing mean and variance fields. Existing Phase 5 and Phase 6 histories are
read as zero-SGS data by the analysis script.

After analyzing a baseline run, compare it directly with the portable bundle:

```sh
python3 scripts/compare_dns_target.py \
  runs/phase7/example/seed_5489 \
  runs/phase6/finalized/dns_target_metadata.json \
  -o runs/phase7/example/seed_5489/dns_comparison.json
```

The comparison verifies both target hashes before conservatively restricting
DNS fields to the LES grid and comparing the common resolved spectrum band.

## Later LES study

The implementation does not select an LES resolution or tune \(C_S\). Once the
Phase 6 target is accepted, use a short single-seed pilot to select a useful
coefficient range, then run the final no-model and static cases with identical
LES resolution, numerical method, forcing, seeds, spin-up, and sampling
duration. Compare relative mean-profile error together with variance, energy,
spectrum, molecular dissipation, SGS dissipation, and numerical dissipation.

The checked-in Phase 7 CTest target covers closure kernels, coefficient
representation and bounds, periodic face interpolation, conservative updates,
the SGS timestep restriction, unsupported dynamic selection, and the
time-discrete SGS energy budget.
