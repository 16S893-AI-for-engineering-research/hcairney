# Burgers SGS implementation guide

## 1. Purpose

This application is a precursor to reinforcement-learning (RL) subgrid-scale
(SGS) modeling for implicitly filtered large-eddy simulation (LES). The
verified solver, turbulent forcing, classical SGS baselines, and offline DNS
reference remain the numerical foundation. The first RL problem is now to learn
an additive, zero-net forcing that makes the LES reproduce the DNS mean velocity
profile. It is not to reproduce the deterministic DNS forcing point by point,
nor to output a Smagorinsky coefficient.

This change follows the parameter studies in which the mean profile was much
less sensitive to grid resolution than expected. That makes mean-profile error
a weak discriminator between conventional coefficient fields and leaves the
original coefficient-learning problem poorly conditioned for an initial RL
integration test. An additive forcing gives the policy direct control over the
quantity used for success while the instantaneous turbulent state and the
finite-time mean estimate remain noisy. It is therefore a useful test of the
SMARTIES coupling and of learning in an unsteady LES, even though it is not a
claim that the learned forcing is the unique or physically exact SGS term.

The solver must remain independent of SMARTIES. The environment will wrap the
solver, collect one raw scalar action per cell from a single shared policy,
project the resulting field to zero discrete mean, and supply it as an additive
source held over a physical-time decision interval. This separation keeps the
numerical method, reference forcing, classical closures, action projection, and
RL interface independently testable.

The reference calculation uses

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

The initial controlled LES instead uses

\[
  \frac{\partial u}{\partial t}
  + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right)
  = \frac{\partial}{\partial x}
    \left[\left(\nu + \nu_{\mathrm{sgs}}\right)
    \frac{\partial u}{\partial x}\right]
  + f'(x,t) + f_{\mathrm{RL}}(x,t).
\]

The known deterministic component \(F_{\mathrm{mean}}\) is withheld in this
environment. It may be applied in an oracle/debug baseline, but it is not a
training label. The learned term may differ from it because the coarse
discretization, numerical dissipation, and any retained classical closure are
part of the controlled dynamics.

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
- Retain the verified Smagorinsky-like closures as classical baselines. The
  first RL policy acts through an additive cell-centered source instead of a
  prescribed eddy-viscosity coefficient.
- Use one agent per LES cell and one shared, memoryless policy initially. The
  policy parameters, observation definition, normalization, and action bounds
  are identical at every cell. Do not provide cell index or absolute position
  as an observation.
- Remove the volume-weighted discrete mean of every learned action field before
  it is applied. Spatial smoothing and amplitude limiting must preserve this
  constraint.
- Generate the target DNS mean profile offline using a high-resolution,
  demonstrably grid-converged configuration of this solver. No DNS will evolve
  alongside the LES during RL training.
- Use reference deterministic forcing, stochastic forcing, and learned forcing
  without a spatial zero mode. This permits an inhomogeneous mean velocity
  profile without a bulk-force/drag balance.

The following choices are deliberately deferred until the preceding components
can inform them: the MUSCL limiter, the exact deterministic forcing modes, the
stochastic forcing process and parameters, DNS and LES resolutions, learned
forcing bounds and smoothing, the complete dynamic SGS model, and the final RL
observation, reward structure, and reward aggregation. They must remain
configurable rather than becoming implicit constants in numerical kernels.

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
    ActionProjection.*
    MeanProfileEstimator.*
    ObservationBuilder.*
    RewardModel.*
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

The Phase 4 MUSCL implementation uses the backward and forward cell-average
differences

\[
  \delta^-_i=u_i-u_{i-1}, \qquad \delta^+_i=u_{i+1}-u_i,
\]

and supports three selectable limited slopes:

\[
\begin{aligned}
  \sigma_i^{\mathrm{minmod}}
    &=\mathrm{minmod}(\delta^-_i,\delta^+_i),\\
  \sigma_i^{\mathrm{MC}}
    &=\mathrm{minmod}\left(2\delta^-_i,
       \frac{\delta^-_i+\delta^+_i}{2},2\delta^+_i\right),\\
  \sigma_i^{\mathrm{VL}}
    &=\begin{cases}
       \dfrac{2\delta^-_i\delta^+_i}{\delta^-_i+\delta^+_i},
         & \delta^-_i\delta^+_i>0,\\
       0, & \text{otherwise}.
      \end{cases}
\end{aligned}
\]

At face \(i+1/2\), reconstruct

\[
  u^L_{i+1/2}=u_i+\frac{\sigma_i}{2}, \qquad
  u^R_{i+1/2}=u_{i+1}-\frac{\sigma_{i+1}}{2}.
\]

This is a method-of-lines reconstruction; no separate MUSCL-Hancock predictor
is applied because SSP-RK3 supplies the temporal integration. The Rusanov
comparison uses the local scalar Burgers speed

\[
  F^{\mathrm{Rus}}(u_L,u_R)=\frac{f(u_L)+f(u_R)}{2}
  -\frac{\max(|u_L|,|u_R|)}{2}(u_R-u_L).
\]

The distinct Phase 4 comparison matrix consists of piecewise-constant Godunov
and Rusanov, plus each of Minmod, MC, and Van Leer MUSCL reconstruction paired
with both fluxes. A limiter is not applied to piecewise-constant data, so this
is an eight-configuration matrix rather than twelve distinct methods. Forced
Phase 4 verification remains restricted to the manufactured source; general
deterministic and stochastic forcing remains Phase 5 work.

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
- `DynamicSmagorinsky`: reserved for a possible later test-filter-based model;
  it is explicitly unsupported in the initial Phase 7 implementation.
- `PrescribedCoefficientField`: accepts one cell-centered \(C_{S,i}\) per cell;
  this remains useful for solver verification and non-RL coefficient-field
  experiments.

Configuration and reports use \(C_S\), while Phase 7 closure kernels store
\(C_S^2\) internally to avoid a redundant sign and repeated squaring. These
coefficient conventions do not constrain the learned additive forcing, which
is signed and has its own physical units and bounds.

The initial Phase 7 baseline deliberately omits the dynamic model. If it is
revisited, first add independently verified filtering operations and document
the one-dimensional Burgers version of the Germano identity, the definition of
grid and test filters, how the least-squares coefficient is averaged, and how
small denominators and negative coefficients are handled.

### 4.4 Time integration and timestep selection

Implement SSP-RK3 as a method-of-lines integrator. Recompute state-dependent
fluxes and eddy viscosity at every RK stage. The projected learned forcing is
held fixed over an RL decision interval and evaluated as the same additive
field at every RK stage within that interval.

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

### 4.5 Phase 5 forcing convention

The production deterministic forcing is a configurable sum of periodic modes,

\[
  F_{\mathrm{mean}}(x)=\sum_m A_m
  \sin\left(\frac{2\pi k_m(x-x_{\mathrm{begin}})}{L}+\phi_m\right),
  \qquad L=x_{\mathrm{end}}-x_{\mathrm{begin}},
\]

with positive, unique, non-Nyquist integer mode indices. The initial validation
case uses the single mode \(0.1\sin x\). Source values supplied to the solver are
analytic finite-volume cell averages, followed by removal of their discrete
mean.

The stochastic forcing uses independent cosine and sine coefficients,

\[
  f'(x,t)=\sum_{k\in\mathcal K}
  \left[a_k(t)\cos(kx)+b_k(t)\sin(kx)\right],
\]

with the corresponding domain-scaled basis on a general periodic interval.
The stationary quadrature variances are

\[
  V_k=f_{\mathrm{rms}}^2
  \frac{k^{-p}}{\sum_{j\in\mathcal K}j^{-p}}.
\]

Thus the continuum pointwise stationary RMS is independent of the number of
forced modes. The low-mode OU preset uses \(p=0\) and modes \(1{:}3\). The
finite-correlation Chekhlov--Yakhot-type preset uses \(p=1\) and modes
\(1{:}8\). Both the exponent and explicit mode list remain configurable; the
presets are validation defaults rather than claims of an inertial-range
spectrum.

The stochastic process is an exact OU transition on a fixed physical-time
clock, held piecewise constant between clock times:

\[
  a_{k,j+1}=\rho a_{k,j}
    +\sqrt{V_k(1-\rho^2)}\,\xi^a_{k,j},
  \qquad
  \rho=\exp(-\Delta t_f/\tau),
\]

with the same independent update for \(b_k\). Coefficients are initialized from
their stationary distributions. The default clock interval is
\(\Delta t_f=\tau/10\). PDE steps end at clock boundaries and all SSP-RK3
stages in one PDE step use one immutable forcing snapshot. The clock advances
only after a successful PDE step. This makes the realization a function of
seed and physical time rather than grid-dependent adaptive timestep calls.

Restart data contains the clock index and time, Fourier coefficients, random
engine and distribution state, and the stochastic parameters needed to reject
an incompatible restart. Initial-condition and forcing randomness use
independent streams derived from the run seed.

Deterministic and stochastic fields have their discrete means removed
separately before composition. Their discrete power inputs are also retained
separately. Per-step work and molecular dissipation use the SSP-RK3 weights
\(1/6,1/6,2/3\), and numerical dissipation is the residual of the resulting
time-discrete energy balance.

### 4.6 Learned additive forcing

SMARTIES supplies one raw scalar action \(a_i\) for every LES cell. The same
policy maps local observations to \(a_i\) at all cells; there are no learned
cell-specific constants. The environment, not the policy, converts the raw
actions into the applied field. First apply any configured periodic,
constant-preserving smoothing to obtain \(s_i\). For a nonuniform
generalization, form the volume-weighted mean and projection

\[
  \langle s\rangle_V =
  \frac{\sum_i V_i s_i}{\sum_i V_i},
  \qquad p_i=s_i-\langle s\rangle_V.
\]

Repeat the mean removal defensively after any later field transformation.
Finally apply a common physical scale and, if needed, a single field-wide
rescaling to satisfy \(|f_i|\leq A_f\):

\[
  f_{\mathrm{RL},i}
  = A_f\,\frac{p_i}{\max(1,\|p\|_\infty)}.
\]

Here \(p\) is nondimensional and \(A_f\) has forcing units. A uniform raw
action field correctly projects to zero. Do not clip cells independently after
projection because that generally reintroduces a nonzero mean. Record raw,
smoothed, projected, and applied action summaries so action saturation and
projection effects are visible.

The learned field is a separate forcing component in the solver energy budget:

\[
  P_{\mathrm{RL}}=\int u f_{\mathrm{RL}}\,dx.
\]

Zero net force does not imply zero power, so this term must not be folded into
numerical dissipation. The initial environment disables the known deterministic
reference forcing, retains the same stochastic forcing process, and uses
\(f_{\mathrm{RL}}\) in its place. Alternative compositions must be explicit in
configuration and metadata.

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
3. Implement and test `PrescribedCoefficientField` through the same
   cell-centered \(C_S^2\) and eddy-viscosity path used by the static model.
4. Compare no-model and static results at identical LES resolution,
   forcing parameters, and sampling duration.
5. Report mean-profile error together with energy, variance, spectrum,
   molecular dissipation, SGS dissipation, and numerical dissipation.

The LES resolution, coefficient sweep, seeds, and sampling duration are chosen
in a separate study after the implementation and verification path is in
place. Dynamic Smagorinsky is not a Phase 7 completion requirement.

**Completion criterion:** the static baseline and prescribed-field path are
reproducible, conservative, stable under documented bounds, and the static
baseline is evaluated against the same offline DNS statistics intended for the
controlled learned-forcing model.

### Phase 8: SMARTIES environment

This phase is intentionally deferred until the solver and baselines are
verified. Add the learned additive-forcing interface, its projection and
diagnostics, a SMARTIES-independent environment core, and finally the SMARTIES
communication loop. The implementation sequence and acceptance tests are in
[the SMARTIES integration guide](implement_rl.md).

**Completion criterion:** the environment conserves the periodic mean under
the learned action, reproduces seeded rollouts, exchanges one action and
transition per cell per decision, terminates cleanly on timeout or failure, and
passes short training and evaluation smoke tests without exposing cell
position to the shared policy.

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
- Zero, deterministic, manufactured, stochastic, and prescribed learned
  forcing components.
- Volume-weighted action projection, periodic smoothing, field-wide amplitude
  limiting, and invariance under cyclic permutation of cells.
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
    + P_{\mathrm{RL}}
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
- Reject non-finite learned actions and additive fields of the wrong size;
  terminate an episode cleanly if a bounded action still produces an invalid
  state.
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

## 8. RL formulation and constraints on current APIs

The SMARTIES environment uses one agent per cell and one shared policy. It must
not call `agentsDefineDifferentMDP()`: all cells have the same observation and
action definitions and contribute experience to the same policy. The policy
receives translation-equivariant local features, initially a periodic velocity
stencil and derived local gradients. Cell index, absolute \(x\), and the local
DNS target value are excluded so the learned mapping cannot memorize a spatial
forcing profile.

Every agent produces one signed raw action. After all actions have been
collected, the environment smooths them if configured, removes their
volume-weighted mean, applies a field-wide bound-preserving scale, and holds the
resulting \(f_{\mathrm{RL}}\) fixed over one physical-time decision interval.
Independent per-agent exploration noise is permitted; shared exploration noise
would largely disappear under the zero-mean projection. The solver must support
the following without putting SMARTIES types in `burgers_core`:

- setting and clearing a complete cell-centered additive forcing field;
- holding that field over a requested physical-time advance;
- accounting separately for learned-forcing work;
- exposing immutable local stencils and diagnostics;
- resetting the PDE, stochastic forcing, running mean, and episode counters
  deterministically from a seed;
- detecting numerical failure; and
- advancing to a requested time despite adaptive PDE timesteps and forcing
  clock boundaries.

The target \(\overline u^{DNS}(x)\) is loaded from the accepted offline bundle
and conservatively restricted to the LES grid. Success is defined by the
normalized error of the statistically converged LES mean profile, not by
forcing-field error. During training, a block mean or an EMA supplies a noisy
finite-time estimate. Reward structure must remain configurable and be tested
rather than being treated as part of the additive-forcing definition. One
local candidate uses the direction of the pre-action EMA error and the
time-weighted velocity block generated after applying the action:

\[
  r_{i,\mathrm{dir}}^n =
  -\operatorname{sign}(\widehat U_i^n-U_i^{DNS})
   \frac{U_{i,\mathrm{block}}^n-U_i^{DNS}}{U_{\mathrm{scale}}}
  -\lambda_a\left(\frac{f_{\mathrm{RL},i}^n}{f_{\max}}\right)^2.
\]

Here \(U_{i,\mathrm{block}}^n\) is the physical-time-weighted average at cell
\(i\) over the decision interval following action \(n\). Cache
\(\widehat U_i^n\), apply the action, accumulate the block, compute the reward,
and only then update the EMA. This ordering prevents the same fluctuation from
setting both the sign and the signed error. A second candidate directly
penalizes the updated mean estimate:

\[
  r_{i,\mathrm{EMA2}}^n =
  -w_i\left(\frac{\widehat U_i^{n+1}-U_i^{DNS}}{U_{\mathrm{scale}}}\right)^2
  -\lambda_a\left(\frac{f_{\mathrm{RL},i}^n}{f_{\max}}\right)^2.
\]

The directional reward offers more immediate but noisier credit and can be
temporarily misdirected when its lagged sign crosses the target. The squared-
EMA reward is smoother and directly reflects mean-squared error, but spreads
credit across the estimator history and retains a finite-sample variance
penalty. Test both with identical training and held-out seeds. Also compare
local rewards with their shared, volume-weighted global aggregations. Because
the action projection couples cells, no local reward provides perfect
individual credit assignment; that limitation must be reported rather than
hidden. Reward choice does not change the statistically converged mean-profile
error used as the primary evaluation metric.

Mean-profile recovery does not uniquely identify SGS physics. Variance, energy,
spectra, molecular/SGS/numerical dissipation, learned-forcing power, action
spectra, and stability are mandatory evaluation diagnostics even when they are
not reward terms. Generalization claims require held-out seeds, initial
conditions, grids, viscosities, and forcing parameters. The detailed build
order, SMARTIES calls, tests, and ablations are maintained in
[the SMARTIES integration guide](implement_rl.md).

## 9. Reproducibility and output requirements

Every standalone run should record:

- domain, grid, molecular viscosity, initial condition, and final time;
- flux, reconstruction, limiter, time integrator, and CFL values;
- forcing type, modes, amplitudes, correlation times, and random seed;
- closure type, learned-forcing composition, projection, smoothing, action
  bounds, reward structure and aggregation, mean-estimator timescale, reward
  scaling, and regularization parameters;
- timestep counts, rejected/final shortened steps, and failure status;
- time histories of mean, energy, every forcing power contribution, molecular
  dissipation, SGS dissipation, and estimated numerical dissipation;
- raw and applied action norms, projection residual, saturation fraction, and
  learned-forcing spectrum at configured intervals; and
- code/build metadata sufficient to reproduce the executable.

CSV is sufficient for initial scalar histories and profiles. Keep file output
outside the numerical kernels, and ensure verification tests can run without
creating output files. Profile-history snapshots use the same prescribed
physical-time clock as configured profile samples; scalar and profile clocks
may use different intervals, and the final adaptive PDE step is shortened to
reach their union. Add a more scalable format only if DNS data volume makes CSV
demonstrably inadequate. Production parameter studies may disable the online
reference DFT and compute block-resolved spectra from profile snapshots with
the documented NumPy analysis tools.

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
6. Static Smagorinsky provides a tested classical comparison baseline; a
   dynamic analog remains optional future work.
7. A SMARTIES-independent additive-field setter, zero-mean projection, action
   bounds, and learned-work diagnostic pass their tests.
8. Applying the known deterministic forcing through the new additive-field
   path reproduces the corresponding coarse reference run within tolerance.

At that point the remaining RL work should primarily add the environment state,
reward, episode lifecycle, and SMARTIES communication loop rather than alter
the verified numerical kernels.
