# Burgers SGS implementation guide

## 1. Purpose

This application is a precursor to reinforcement-learning (RL) subgrid-scale
(SGS) modeling for large-eddy simulation (LES). The immediate objective is to
build and verify a reliable finite-volume solver for the one-dimensional
viscous Burgers equation. Turbulent forcing, classical SGS baselines, offline
generation of a DNS mean profile, and SMARTIES integration will be added only
after the unclosed solver has passed its verification suite.

The solver must remain independent of SMARTIES. The eventual RL environment
will wrap the solver and supply a field of closure coefficients in the same way
that a classical closure model does. This separation should allow the numerical
method, forcing, closures, and RL interface to be tested independently.

The initial governing equation is

\[
  \frac{\partial u}{\partial t}
  + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right)
  = \frac{\partial}{\partial x}
    \left[\left(\nu + \nu_{\mathrm{sgs}}\right)
    \frac{\partial u}{\partial x}\right]
  + F_{\mathrm{mean}}(x) + f'(x,t),
\]

on the periodic domain \(x\in[0,2\pi)\). The molecular viscosity \(\nu\) is
constant. The solver stores cell averages on a uniform mesh and uses double
precision.

## 2. Fixed design decisions

The following decisions define the initial scope:

- Implement the solver in C++14, consistent with the SMARTIES library and its
  existing applications.
- Begin with a serial solver. Keep numerical kernels independent of SMARTIES,
  MPI, and OpenMP so that parallelism can be added later without changing the
  mathematical interfaces.
- Use a uniform, cell-centered, periodic finite-volume grid on \([0,2\pi)\).
- Start with piecewise-constant reconstruction and the exact scalar Godunov
  flux for the advective term.
- Use a centered, second-order face gradient for the viscous term.
- Use explicit SSP-RK3 time integration and enforce both advective and
  diffusive timestep restrictions.
- Add MUSCL reconstruction after the first-order solver is verified. Keep
  reconstruction and Riemann-flux selection independent so their numerical
  dissipation can be compared systematically. A Rusanov flux may be added as
  another comparison.
- Define all SGS coefficients and eddy viscosities at cell centers. Interpolate
  viscosity to faces only when constructing the conservative viscous flux.
- Restrict the first closures to a single Smagorinsky-like coefficient per
  cell. The future RL policy will output one local coefficient for each cell,
  with all cells using the same policy.
- Generate the target DNS mean profile offline using a high-resolution,
  demonstrably grid-converged configuration of this solver. No DNS will evolve
  alongside the LES during RL training.
- Use mean and stochastic forcing without a spatial zero mode. This permits an
  inhomogeneous mean velocity profile without a bulk-force/drag balance.

The following choices are deliberately deferred until the preceding components
can inform them: the MUSCL limiter, the exact deterministic forcing modes, the
stochastic forcing process and parameters, DNS and LES resolutions, SGS
coefficient bounds, dynamic-model averaging, and the RL observation and reward
aggregation. They must remain configurable rather than becoming implicit
constants in numerical kernels.

## 3. Proposed source layout

The application should be organized approximately as follows. Exact filenames
may change as the implementation becomes clearer.

```text
apps/burgers_sgs/
  CMakeLists.txt
  implementation_guide.md

  include/burgers/
    Config.h
    Grid.h
    State.h
    Reconstruction.h
    ConvectiveFlux.h
    Forcing.h
    ClosureModel.h
    BurgersSolver.h
    Diagnostics.h

  src/
    Grid.cpp
    Reconstruction.cpp
    ConvectiveFlux.cpp
    Forcing.cpp
    ClosureModel.cpp
    BurgersSolver.cpp
    Diagnostics.cpp

  tests/
    test_grid.cpp
    test_flux.cpp
    test_conservation.cpp
    test_manufactured_solution.cpp
    test_energy_budget.cpp
    test_waves.cpp
    test_forcing.cpp

  tools/
    run_solver.cpp
    generate_dns_mean.cpp

  environment/          # added after the solver and closures are verified
    SGSEnvironment.*
    smarties_main.cpp
```

Build the numerical code as a library target such as `burgers_core`. Solver
tests and standalone tools should link only against this library. A future
SMARTIES executable should be a separate target that links both `burgers_core`
and `smarties`.

Use CMake and CTest. The core verification suite should not depend on SMARTIES,
MPI, plotting libraries, or external test frameworks. Small assertion helpers
are sufficient initially. Tests must return a nonzero exit code and print a
useful diagnostic when a tolerance is violated.

## 4. Numerical formulation

### 4.1 Finite-volume update

For cell average \(u_i\), write the semidiscrete update in conservative form:

\[
  \frac{d u_i}{dt}
  = -\frac{F^a_{i+1/2}-F^a_{i-1/2}}{\Delta x}
    +\frac{F^\nu_{i+1/2}-F^\nu_{i-1/2}}{\Delta x}
    + f_i.
\]

Here \(F^a\) is a numerical approximation to the advective flux
\(f(u)=u^2/2\), and

\[
  F^\nu_{i+1/2}
  = \nu_{\mathrm{eff},i+1/2}
    \frac{u_{i+1}-u_i}{\Delta x}.
\]

All periodic indexing should be centralized in the grid or state abstraction;
individual numerical kernels should not contain special-case boundary logic.

### 4.2 Convective fluxes and reconstruction

Implement the exact Godunov flux for scalar Burgers first. Unit-test its shock,
rarefaction, transonic-rarefaction, equal-state, and sign-changing cases.

Keep two separate choices in the configuration:

- **Reconstruction:** piecewise constant initially, followed by MUSCL with a
  documented limiter.
- **Riemann flux:** Godunov initially, with Rusanov as an optional comparison.

The first-order Godunov scheme is the robust reference implementation, not the
intended final turbulence discretization. MUSCL-Godunov should become the main
candidate for forced simulations once its smooth-solution convergence and
nonoscillatory behavior have been verified.

When adding MUSCL:

1. Implement slope calculation independently of flux evaluation.
2. Confirm that constant data reconstructs exactly.
3. Confirm second-order convergence for smooth solutions before enabling a
   limiter-sensitive discontinuous test.
4. Make the limiter selectable so its dissipation can be measured rather than
   hidden in the implementation.

### 4.3 Molecular and SGS viscosity

The closure coefficient and eddy viscosity are cell-centered. For a
one-dimensional Smagorinsky analog, use

\[
  \nu_{\mathrm{sgs},i}
  = (C_{S,i}\Delta)^2
    \left|\left(\frac{\partial u}{\partial x}\right)_i\right|,
\]

where \(\Delta=\Delta x\) initially and the cell-centered gradient is computed
with a documented consistent stencil. The effective face viscosity is initially
formed with an arithmetic average:

\[
  \nu_{\mathrm{eff},i+1/2}
  = \nu + \frac{\nu_{\mathrm{sgs},i}
                    +\nu_{\mathrm{sgs},i+1}}{2}.
\]

This preserves the cell-centered interface expected by the eventual LES code
while retaining a conservative finite-volume update. Harmonic averaging can be
added later if strongly discontinuous viscosity fields make it useful.

Use one common closure interface with implementations for:

- `NoClosure`: \(\nu_{\mathrm{sgs}}=0\).
- `StaticSmagorinsky`: a prescribed constant \(C_S\), evaluated locally to
  produce a spatially varying eddy viscosity.
- `DynamicSmagorinsky`: a test-filter-based dynamic coefficient with its
  averaging, regularization, and clipping choices exposed in configuration.
- `PrescribedCoefficientField`: accepts one cell-centered \(C_{S,i}\) per cell;
  this will be used by the future RL environment without putting RL code in the
  solver.

The first RL model will be restricted to nonnegative, bounded coefficients.
Since \(C_S\) enters the model squared, consider representing and bounding
\(C_S^2\) internally to avoid a redundant action sign.

Do not implement the dynamic model until the unclosed solver, filtering
operations required by the dynamic procedure, and static model have been
verified. Before using it as a baseline, document the one-dimensional Burgers
version of the Germano identity, the definition of grid and test filters, how
the least-squares coefficient is averaged, and how small denominators and
negative coefficients are handled.

### 4.4 Time integration and timestep selection

Implement SSP-RK3 as a method-of-lines integrator. Recompute state-dependent
fluxes and eddy viscosity at every RK stage. A prescribed RL coefficient may be
held fixed over an eventual RL decision interval, but the eddy viscosity still
changes with the stage velocity gradient.

Select the timestep from both constraints:

\[
  \Delta t_a = C_a\frac{\Delta x}{\max_i |u_i|},
  \qquad
  \Delta t_\nu = C_\nu
  \frac{\Delta x^2}{\max_i(\nu+\nu_{\mathrm{sgs},i})}.
\]

Use the smaller value and handle zero denominators explicitly. Put the CFL
numbers in configuration and begin conservatively. Allow the last step to be
shortened to reach a requested output or final time exactly. Later, define RL
decision intervals in physical time rather than in a fixed number of adaptive
PDE steps.

## 5. Phased implementation plan

### Phase 0: scaffolding

1. Add the local CMake build and a `burgers_core` library.
2. Add a standalone solver executable and CTest integration.
3. Define typed configuration objects for the grid, viscosity, numerical
   method, time integration, forcing, closure, output, and random seed.
4. Ensure Debug builds enable useful warnings, assertions, and sanitizers when
   available.
5. Make every run write enough metadata to reproduce it.

**Completion criterion:** a no-op test and standalone executable build without
linking SMARTIES.

### Phase 1: grid and state

1. Construct uniform cell centers and cell widths on \([0,2\pi)\).
2. Implement and test periodic indexing.
3. Store cell averages in a simple contiguous container.
4. Implement discrete integrals, means, and norms.
5. Define initial-condition callbacks, including constant, sinusoidal, random,
   and periodic two-state data.

**Completion criterion:** periodic indexing, coordinates, cell-average
integration, and constant-state tests pass.

### Phase 2: first-order unclosed solver

1. Implement piecewise-constant face states.
2. Implement and unit-test exact Godunov flux.
3. Implement centered molecular-viscous fluxes.
4. Assemble the conservative right-hand side with zero forcing and zero SGS
   viscosity.
5. Implement SSP-RK3 and CFL selection.
6. Add finite-value checks and descriptive failure messages.

**Completion criterion:** constant preservation, mean conservation, smooth
decay, and basic wave-propagation tests pass.

### Phase 3: formal verification

Apply the verification plan in Section 6 incrementally to the capabilities of
the Phase 2 solver: piecewise-constant reconstruction, Godunov flux, molecular
viscosity, SSP-RK3, periodic boundaries, and no SGS model. Make the checks
automated rather than relying only on plots.

1. Complete the unit-level checks applicable to the Phase 2 numerical kernels,
   including Fourier-mode checks of centered gradients and the resulting
   viscous operator.
2. Strengthen constant-preservation and periodic-mean-conservation tests with
   multi-step smooth and random cases, reporting absolute and relative drift
   against documented, scale-aware tolerances.
3. Add the minimal time-dependent manufactured-source capability needed for
   the analytic solution in Section 6.4. Evaluate the source as finite-volume
   cell averages and at the correct physical time for every SSP-RK3 stage. This
   is a narrow exception to the Phase 5 ordering: the general composable
   forcing interface, production deterministic forcing, and stochastic forcing
   remain Phase 5 work.
4. Run an independent spatial refinement study for the manufactured solution,
   comparing numerical and exact cell averages in the \(L_1\), \(L_2\), and
   \(L_\infty\) norms. Verify that the piecewise-constant method approaches
   first order while temporal error is kept negligible.
5. Run an independent fixed-timestep refinement study on a fixed, sufficiently
   fine grid. Compare against the manufactured solution or a much smaller-step
   reference and verify third-order SSP-RK3 convergence before spatial error
   dominates.
6. Verify the unforced energy balance, including monotone energy decay,
   molecular dissipation, and the numerical-dissipation residual of a
   documented time-discrete balance on a smooth refinement sequence.
7. Extend the nonlinear shock/rarefaction test into a refinement study of shock
   speed, rarefaction extent, integral error, wave-location error, entropy
   behavior, and mean conservation before the periodic waves interact.
8. Complete the Phase 2 CFL and failure-handling matrix, including timesteps
   below and near the configured limits, zero velocity, zero viscosity, large
   viscosity, invalid sizes and parameters, and non-finite states.
9. Version the configurations, observed convergence slopes, conservation
   drift, and energy-budget residuals used as small regression references.

Section 6 checks that require features not yet implemented are deferred to the
phase that introduces those features. These include MUSCL and Rusanov variants,
general deterministic and stochastic forcing (including stochastic restart),
spatially varying and SGS viscosity, SGS coefficient arrays and timestep
bounds, and forced/SGS energy-budget terms. Each applicable verification check
must be added when its corresponding feature is introduced; deferral from
Phase 3 does not remove it from the overall verification suite.

**Completion criterion:** the first-order method achieves its expected spatial
order, SSP-RK3 achieves its expected temporal order in an appropriately
isolated study, and the applicable conservation, energy, nonlinear-wave, CFL,
and failure-handling tests meet documented tolerances. The manufactured-source
path must exercise the production solver and its time integration rather than
exist only in test-side integration code.

### Phase 4: higher-order and flux comparison

1. Add MUSCL reconstruction and at least one documented limiter.
2. Verify smooth second-order spatial convergence.
3. Add Rusanov flux if useful for the dissipation study.
4. Run identical smooth, steepening-wave, and forced cases with each supported
   reconstruction/flux combination.
5. Measure numerical energy dissipation using the discrete energy budget.

**Completion criterion:** numerical choices are interchangeable by
configuration, retain conservation, achieve their expected order, and have a
quantified dissipation difference.

### Phase 5: deterministic and stochastic forcing

1. Introduce a composable forcing interface.
2. Add manufactured forcing for verification.
3. Add zero-mean deterministic spatial forcing for the inhomogeneous mean
   profile. A sinusoid or a small set of nonzero Fourier modes is a suitable
   initial choice.
4. Add seeded low-wavenumber stochastic forcing, initially using an
   Ornstein-Uhlenbeck process for Fourier coefficients or another explicitly
   documented process with a finite correlation time.
5. Remove the discrete spatial mean of every forcing realization as a defensive
   guarantee that the zero mode is absent.
6. Define stochastic forcing in physical time and Fourier space so the same
   realization can be evaluated consistently on different grids.
7. Record separate power input from deterministic and stochastic forcing.

For all deterministic spatial sources, use finite-volume cell averages of the
source when available analytically. Otherwise use a quadrature rule accurate
enough not to reduce the measured order of the spatial discretization.

**Completion criterion:** forcing is reproducible for a fixed seed, changes
appropriately for different seeds, has zero discrete spatial mean, and produces
a statistically stationary solution without bulk-velocity drift.

### Phase 6: offline DNS target generation

1. Establish a high-resolution configuration for which the physical viscous
   scales are resolved.
2. Demonstrate grid convergence of the mean profile, variance, energy,
   dissipation, and resolved spectrum using at least one finer grid.
3. Quantify numerical dissipation and require it to be acceptably small relative
   to molecular dissipation over the dynamically important range.
4. Run through a documented spin-up period before collecting statistics.
5. Accumulate temporal means and uncertainties; use multiple forcing seeds if
   needed to reduce sampling error.
6. Write \(\overline u^{DNS}(x)\), sample counts, averaging duration,
   uncertainty estimates, and complete run metadata to a reusable file.

Calling this result “DNS” requires the resolution and statistical convergence
evidence above. Using the same code is acceptable because the analytic and
manufactured-solution tests independently verify the implementation, while the
DNS resolution study verifies the chosen configuration.

**Completion criterion:** repeating or extending the averaging interval changes
the target profile by less than its documented sampling uncertainty, and grid
refinement does not materially change it.

### Phase 7: classical SGS baselines

1. Implement static Smagorinsky with a configurable constant \(C_S\).
2. Sweep \(C_S\) to establish mean-profile and turbulence-statistic baselines.
3. Implement the dynamic Smagorinsky analog only after its test filtering and
   Germano-identity calculations have dedicated tests.
4. Compare no-model, static, and dynamic results at identical LES resolution,
   forcing parameters, and sampling duration.
5. Report mean-profile error together with energy, variance, spectrum,
   molecular dissipation, SGS dissipation, and numerical dissipation.

**Completion criterion:** both baseline models are reproducible, conservative,
stable under documented bounds, and evaluated against the same offline DNS
statistics intended for the future RL model.

### Phase 8: SMARTIES environment

This phase is intentionally deferred until the solver and baselines are
verified. Its expected form is recorded in Section 8 so current solver APIs do
not obstruct it.

## 6. Solver verification plan

Verification tests should be fast enough for routine execution where possible.
Long resolution and statistical studies may be separate labeled tests or
scripts, but their configurations and expected results must remain versioned.

### 6.1 Unit-level checks

- Periodic wraparound for positive and negative neighbor offsets.
- Exact cell coordinates and total domain length.
- Flux consistency: \(F^a(u,u)=u^2/2\).
- Godunov flux for left-going and right-going shocks, rarefactions crossing
  zero, and constant states.
- Reconstruction of a constant state.
- Centered gradients and Laplacians on periodic Fourier modes.
- Face interpolation of constant and spatially varying viscosity.
- Zero, deterministic, manufactured, and stochastic forcing components.
- Fixed-seed reproducibility and serialization/restart of stochastic forcing
  state if restart support is added.

### 6.2 Constant-state preservation

With constant initial data and zero forcing, the state must remain constant to
roundoff for both inviscid and viscous configurations and for every supported
flux/reconstruction combination. This test exercises periodic boundaries,
flux cancellation, RK stages, and timestep logic simultaneously.

### 6.3 Conservation of the periodic mean

For zero-mean forcing,

\[
  \frac{d}{dt}\left(\frac{1}{N}\sum_i u_i\right)=0.
\]

Test this using smooth and random states. Report absolute and relative drift
over many steps. Tolerances should scale sensibly with machine precision, the
number of operations, and the magnitude of the initial state rather than being
chosen only to make a particular run pass.

### 6.4 Manufactured solution

Use the smooth periodic manufactured solution

\[
  u(x,t)=A e^{-t}\sin(kx).
\]

For

\[
  u_t + u u_x = \nu u_{xx} + f,
\]

the required source is

\[
  f(x,t)=(-1+\nu k^2)u
    + A^2 k e^{-2t}\sin(kx)\cos(kx).
\]

Compare numerical cell averages against exact cell averages, not merely
pointwise values at cell centers. Measure \(L_1\), \(L_2\), and \(L_\infty\)
errors on a refinement sequence such as \(N=32,64,128,256\), and compute

\[
  p=\log_2(E_N/E_{2N}).
\]

Use a sufficiently small timestep to isolate spatial convergence. The
piecewise-constant method should approach first order and the smooth MUSCL
method should approach second order.

### 6.5 Temporal convergence

Use a smooth problem on a sufficiently fine grid and repeat it with successively
smaller fixed timesteps. Compare all runs to an exact manufactured solution or
a much smaller-timestep reference. Confirm third-order SSP-RK3 convergence
before spatial error dominates.

### 6.6 Energy budget and numerical dissipation

For an unforced periodic solution without SGS viscosity,

\[
  E=\frac{1}{2}\int u^2\,dx,
  \qquad
  \frac{dE}{dt}=-\nu\int u_x^2\,dx.
\]

Energy must not grow. With forcing and SGS viscosity, track the discrete analogs
of

\[
  \frac{dE}{dt}
  = P_{\mathrm{mean}} + P_{\mathrm{stochastic}}
    -\varepsilon_\nu-\varepsilon_{\mathrm{sgs}}
    -\varepsilon_{\mathrm{num}}.
\]

Calculate \(\varepsilon_{\mathrm{num}}\) as the residual of a carefully
time-discretized energy balance. Verify the diagnostic on smooth refinement
studies before interpreting it physically. Use this diagnostic to compare
first-order Godunov, MUSCL-Godunov, and any later Rusanov configurations.

### 6.7 Nonlinear waves and entropy behavior

Use periodic two-state initial data that produces a shock and a rarefaction.
Stop the comparison before periodic waves interact. Check:

- the Rankine-Hugoniot shock speed
  \(s=(u_L+u_R)/2\),
- the direction and extent of the rarefaction fan,
- absence of expansion shocks,
- conservation of the mean, and
- convergence of integral and wave-location errors.

For viscous steep fronts, perform a refinement study that resolves the physical
shock thickness. Do not interpret an under-resolved shock-capturing run as a
DNS result.

### 6.8 CFL and failure handling

- Test timesteps below and near the configured stability limits.
- Exercise zero velocity, zero viscosity, and large-viscosity cases without
  dividing by zero.
- Confirm that SGS viscosity participates in the diffusive timestep bound.
- Reject negative molecular viscosity, invalid grid sizes, non-finite state,
  and coefficient arrays of the wrong size with descriptive errors.
- Ensure failures are observable by the future environment rather than silently
  propagating `NaN` values.

### 6.9 Independent checks and regression data

Manufactured solutions provide an analytic check independent of the numerical
implementation and are mandatory. An additional comparison against a simple
Fourier pseudospectral solution for smooth periodic cases would be valuable but
need not block the first solver milestone.

Once a test passes, store its configuration and small reference quantities such
as final norms, convergence slopes, conservation drift, and energy-budget
residual. Avoid storing large binary reference fields when an analytic or
property-based check is available.

## 7. Forced-turbulence validation before RL

After deterministic verification, establish that the forced system is suitable
for closure experiments:

1. Run deterministic mean forcing without stochastic forcing and verify a
   bounded, reproducible inhomogeneous response.
2. Run stochastic forcing without mean forcing and verify stationarity,
   reproducibility, zero bulk drift, and expected symmetry.
3. Combine both forcings and monitor spin-up, autocorrelation time, and sampling
   uncertainty.
4. Compare multiple grid resolutions and numerical schemes using identical
   Fourier-space forcing definitions.
5. Confirm that energy input balances molecular, SGS, and numerical
   dissipation within the diagnostic residual.
6. Record the pointwise temporal mean, variance, energy spectrum, gradient
   statistics, and dissipation.
7. Choose the DNS and LES resolutions only after these comparisons reveal where
   physical resolution ends and numerical dissipation becomes important.

A suitable zero-mean inhomogeneous forcing can be formed from nonzero Fourier
modes, for example \(F_{\mathrm{mean}}(x)=A\sin(kx+\phi)\), or a documented
sum of such modes. The target profile may itself have a nonzero domain mean if
that mean is established by the initial condition; it is the net forcing that
must vanish to avoid secular acceleration in the periodic domain.

## 8. Future RL formulation and constraints on current APIs

The expected SMARTIES environment uses one agent per cell and a single shared
policy. Every agent produces one local, bounded \(C_{S,i}\) (or equivalently
\(C_{S,i}^2\)). The solver receives the complete coefficient field through
`PrescribedCoefficientField`, computes cell-centered SGS viscosity, interpolates
it to faces, and advances conservatively.

The target \(\overline u^{DNS}(x)\) is loaded from the offline statistics file.
The LES mean is estimated during an episode, likely with an exponential moving
average (EMA). A candidate local linear reward is

\[
  r_i^n =
  -\operatorname{sign}(\widehat U_i^n-U_i^{DNS})
   (u_i^{n+1}-U_i^{DNS}).
\]

To avoid using the same fluctuation to determine the sign and the error:

1. Form the sign from the existing EMA.
2. Apply the action and advance the solver.
3. Evaluate the reward from the new sample or decision-block average.
4. Update the EMA afterward.

The exact observations, reward aggregation, EMA timescale, agent synchronization,
and SMARTIES settings are intentionally deferred. The solver must nevertheless
support the following without redesign:

- setting a full cell-centered coefficient field before an advance,
- holding that coefficient field over a physical-time decision interval,
- exposing local stencils and diagnostics without exposing internal mutable
  storage,
- resetting deterministically from a seed,
- detecting numerical failure, and
- advancing to a requested physical time despite adaptive PDE timesteps.

Mean-profile recovery alone does not uniquely identify correct SGS physics.
Therefore, even if the reward uses only the mean profile, comparisons must also
report variance, energy, spectra, molecular and SGS dissipation, numerical
dissipation, coefficient distributions, and stability. Evaluate trained models
on held-out forcing amplitudes, viscosities, grids, and random seeds before
claiming closure generalization.

## 9. Reproducibility and output requirements

Every standalone run should record:

- domain, grid, molecular viscosity, initial condition, and final time;
- flux, reconstruction, limiter, time integrator, and CFL values;
- forcing type, modes, amplitudes, correlation times, and random seed;
- closure type and all coefficient bounds or regularization parameters;
- timestep counts, rejected/final shortened steps, and failure status;
- time histories of mean, energy, power input, molecular dissipation, SGS
  dissipation, and estimated numerical dissipation; and
- code/build metadata sufficient to reproduce the executable.

CSV is sufficient for initial scalar histories and profiles. Keep file output
outside the numerical kernels, and ensure verification tests can run without
creating output files. Add a more scalable format only if DNS data volume makes
CSV demonstrably inadequate.

## 10. Overall definition of readiness for RL

SMARTIES integration should begin only when all of the following are true:

1. The unclosed first- and higher-order solvers pass the automated verification
   suite at their expected convergence orders.
2. Conservation, CFL handling, periodic boundaries, and energy diagnostics are
   verified.
3. Numerical dissipation has been quantified for the candidate LES schemes.
4. The stochastic and mean forcing produce a stationary, reproducible,
   inhomogeneous turbulent problem without bulk acceleration.
5. The high-resolution offline target is resolution- and sampling-converged.
6. Static and dynamic Smagorinsky analogs provide tested comparison baselines.
7. The prescribed local coefficient field uses the same cell-centered closure
   path as the classical models.

At that point the RL work should primarily add an environment and a new source
of \(C_{S,i}\), rather than modify the PDE solver.
