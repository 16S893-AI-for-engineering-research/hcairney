# Channel RANS implementation guide

## 1. Objective and document status

Build a one-dimensional, fully developed plane-channel RANS application in
`apps/channel_rans`. First verify a standalone mean-momentum solver against
laminar and prescribed mixing-length reference solutions. Then integrate an RL
environment with SMARTIES to recover the wall-normal variation of a
mixing-length coefficient from velocity-profile rewards.

The initial experiment is a controlled recovery problem. A known, van
Driest-damped mixing length generates a synthetic reference. The learned
closure receives neither the damping formula nor its values: it must produce
the effective coefficient through its actions. The target coefficient remains
available for offline evaluation, but is not a training label.

This document is an implementation plan, not a record of implemented features.
File names, interfaces, configuration keys, commands, and numerical defaults
below are proposed unless explicitly identified as existing Burgers code.
No channel solver or SMARTIES executable exists yet.

### 1.1 Agreed scope

- Evolve only the mean streamwise velocity, `U(y,t)`, on a half-channel.
- Use a mixing-length eddy viscosity; do not introduce `k`, `omega`, `epsilon`,
  DNS evolution, or tensor-basis coefficients in the first implementation.
- Verify the laminar profile, steady total-stress balance, and a prescribed
  mixing-length reference before introducing learning.
- Use van Driest damping in reference generation. A separate outer correction
  may be added later, but its formula is not yet chosen.
- Learn a nonnegative coefficient multiplying wall distance:
  `ell_RL = y * C_theta`. Do not secretly multiply the policy output by the
  reference damping or outer correction.
- Keep core numerics and environment logic usable without SMARTIES.
- Preserve the existing `apps/burgers_sgs` application.

This is a synthetic RANS closure benchmark. Recovery of its generating model
does not establish accuracy against real turbulence or transfer to LES.

### 1.2 Decisions still to be made

The recommendations below permit a concrete implementation plan without
treating every proposed numerical choice as an already accepted requirement.
Resolve each decision before its dependent phase and record the choice in this
guide and the resolved configuration.

| ID | Decision | Recommended starting choice | Resolve before |
| --- | --- | --- | --- |
| D1 | Driving and units | Fixed positive pressure-gradient acceleration `G`; configure dimensional `h, nu, G`, derive wall units | Phase 0 |
| D2 | Reference parameters and cases | Explicit `kappa=0.4`, `A_plus=26`; choose initial `Re_tau` and a separate sufficiently high-Re wall-law check | Phase 0 |
| D3 | Outer mixing-length correction | `f_outer=1` initially; introduce a named, explicit model only in a later experiment | Phase 4 |
| D4 | Mesh and wall resolution | Uniform mesh for verification; smooth wall clustering for reference and training; establish adequacy by refinement | Phase 1 |
| D5 | Closure coefficient location | Cell-centred coefficients, mapped consistently to face mixing lengths; evaluate fluxes from face gradients | Phase 3 |
| D6 | Production time integration | Backward Euler with converged Picard iterations; retain explicit stepping for verification | Phase 5 |
| D7 | First RL transition semantics | One joint coefficient field per episode, solve to convergence, then reward; sequential transient mode is an extension | Phase 7 |
| D8 | Policy observations | Scaled `y_plus` and `y/h` for the first stationary coefficient policy | Phase 7 |
| D9 | Action map and bounds | A direct bounded nonnegative `C`, including zero; choose `C_max` after baseline sensitivity tests | Phase 6 |
| D10 | Reward weights and regularization | Global volume-weighted mean squared error in `U_plus`; no coefficient labels; initially zero regularization | Phase 6 |
| D11 | Agent placement | One cell-agent per grid cell, with one shared policy; keep grid count fixed within a SMARTIES job | Phase 8 |
| D12 | Training distribution | Establish single-case recovery first; subsequently vary `Re_tau` with held-out cases and meshes | Phase 9 |
| D13 | Accuracy and convergence thresholds | Establish explicit tolerances from verification, action sensitivity, and refinement studies | Phases 2–6 |
| D14 | Initial coefficient and velocity baselines | Undamped `C=kappa` and its converged velocity, plus laminar/startup verification cases | Phase 6 |

Do not implement both alternatives for every decision immediately. The first
delivery should be the smallest verified path through these phases.

## 2. Mathematical specification

### 2.1 Domain, variables, and driving

Let `y=0` denote the stationary wall and `y=h` the channel centreline. Statistical
homogeneity in streamwise and spanwise directions and fully developed mean flow
give mean velocity `(U(y,t),0,0)`. This dimensional reduction applies to the mean
flow; the modeled turbulence is still three-dimensional.

Use constant density and constant molecular kinematic viscosity `nu>0`:

\[
\partial_t U=G+\partial_y F,\qquad
F=(\nu+\nu_t)\partial_y U,
\qquad G=-\rho^{-1}\partial_x\overline p>0.
\]

Here `F` is total shear stress divided by density, with a positive sign on the
lower half-channel. Equivalently,

\[
-\overline{u'v'}=\nu_t U_y,\qquad
\nu_t=\ell_m^2|U_y|.
\]

Boundary conditions are

\[
U(0,t)=0,\qquad U_y(h,t)=0.
\]

There is no Burgers convective flux, periodic boundary, stochastic forcing,
or policy-controlled additive source. The specified `G` is part of the problem,
not an action.

For fixed `G`, define the equilibrium reference scales

\[
u_\tau=\sqrt{Gh},\quad Re_\tau=\frac{u_\tau h}{\nu},\quad
y^+=\frac{yu_\tau}{\nu},\quad U^+=\frac{U}{u_\tau},\quad
t^*=\frac{tu_\tau}{h}.
\]

Use these fixed scales throughout startup and learning. Instantaneous wall
stress during a transient need not equal `rho*G*h`; do not silently replace the
reference scale by an instantaneous estimate.

If inputs are later expressed through `Re_tau`, define an unambiguous conversion
to `h,nu,G`. Do not accept independently inconsistent values for all four.
Constant-bulk-flow driving is outside the initial scope.

### 2.2 Reference and learned closures

The reference closure is

\[
\ell_{\mathrm{ref}}(y)=\kappa y
\left[1-\exp\left(-y^+/A^+\right)\right]f_{\mathrm{outer}}(y/h).
\]

For the first experiment set `f_outer=1`. This does not remove channel-scale
effects: total stress still decreases toward the centreline. It only removes an
additional empirical outer mixing-length treatment.

The learned closure is

\[
\ell_{\mathrm{RL}}(y)=yC_\theta(\mathbf o(y)),\qquad
\nu_{t,\mathrm{RL}}=y^2C_\theta^2|U_y|.
\]

The generating coefficient, used only for diagnostics and scripted oracle
evaluation, is

\[
C_{\mathrm{ref}}(y)=\kappa
\left[1-\exp\left(-y^+/A^+\right)\right]f_{\mathrm{outer}}(y/h).
\]

Distinguish `C`, `C^2`, `ell_m`, and `nu_t` in APIs and output. `C` is not a
Smagorinsky coefficient, and the grid spacing is not the mixing length. At the
wall, `ell_m=0` follows from geometry for any finite coefficient. Do not impose
`C=0` in the first interior cell to encode the hidden damping by hand.

The RL execution path must contain no call to the reference damping evaluator.
It may load the velocity target and physical case scales. Exact coefficient
fields are available only through an explicit reference/oracle path.

### 2.3 Independent steady references

For any prescribed nonnegative mixing length, steady momentum balance gives

\[
F(y)=G(h-y),\qquad
\nu s+\ell_m^2s^2=T(y),\quad s=U_y\geq0,\quad T=G(h-y).
\]

Use the cancellation-resistant root

\[
s(y)=\frac{2T(y)}{\nu+\sqrt{\nu^2+4\ell_m(y)^2T(y)}},\qquad
U(y)=\int_0^y s(\eta)\,d\eta.
\]

It remains valid at `ell_m=0` and at the centreline. Implement the quadrature
reference separately from the PDE flux and time integrator. Verify its
tolerance independently. Produce finite-volume target cell averages, not just
point samples at cell centres.

For the laminar case,

\[
U_{\mathrm{lam}}(y)=\frac{G}{\nu}\left(hy-\frac{y^2}{2}\right),\qquad
U_b=\frac1h\int_0^h U\,dy=\frac{Gh^2}{3\nu}.
\]

The exact cell average on `[a,b]` is

\[
\overline U_{[a,b]}=\frac G\nu
\left[\frac{h(a+b)}2-\frac{a^2+ab+b^2}{6}\right].
\]

### 2.4 Wall-law and identifiability checks

With `s_plus=dU_plus/dy_plus` and `ell_plus=ell_m*u_tau/nu`,

\[
s^+ +(\ell_m^+)^2(s^+)^2=1-y/h.
\]

The model should approach `U_plus=y_plus` close to the wall and exhibit
`y_plus*dU_plus/dy_plus` approximately equal to `1/kappa` in an overlap region
when `Re_tau` is sufficiently large. The logarithmic law is not an exact
whole-channel target. Its intercept follows from the specified damping model;
do not independently impose a chosen intercept and expect exact agreement.

Velocity is weakly sensitive to mixing length where molecular transport
dominates and where mean shear vanishes. At the centreline the local coefficient
is unidentifiable from shear stress. For interior nonzero shear, inversion gives

\[
\ell_m^2=\frac{G(h-y)-\nu U_y}{U_y^2}.
\]

Use this only as a diagnostic: differentiation, cancellation near the wall,
and division near the centreline make it poorly conditioned. Do not reward
relative coefficient error at locations where the reference tends to zero.

## 3. Reuse from Burgers and application structure

Read existing components as implementation references, not as a specification
that every Burgers feature should be copied.

| Existing source | Reuse | Replace or omit |
| --- | --- | --- |
| `../burgers_sgs/include/burgers/State.h` | Scalar cell-average storage and checks | Namespace and physical field names |
| `../burgers_sgs/include/burgers/BurgersSolver.h` | `advanceTo`, accepted-step observer, prescribed-field lifecycle | Convective RHS, forcing clocks, Burgers budgets |
| `../burgers_sgs/src/ViscousFlux.cpp` | Conservative face transport concept | Uniform periodic geometry and `N`-face convention |
| `../burgers_sgs/include/burgers/ClosureModel.h` | Closure abstraction and held coefficient fields | `C_S` semantics, grid-scale SGS laws |
| `../burgers_sgs/tools/run_solver.cpp` | CLI, resolved config, output scheduling, status/metadata | Burgers defaults, statistics, forcing options |
| `../burgers_sgs/src/RunOutputRecorder.cpp` | Output-event scheduling and reusable recording | Spectra and stochastic statistics |
| `../burgers_sgs/src/environment/SGSEnvironment.cpp` | Reset/step separation and failure handling | Additive actions, mandatory running means, forcing rewards |
| `../burgers_sgs/src/environment/smarties_main.cpp` | Communicator lifecycle and evaluation plumbing | Application fields, action semantics, scripted Fourier forcing |
| `../burgers_sgs/CMakeLists.txt` | Standalone core and optional SMARTIES target pattern | Globally conflicting target names |

Initially keep the channel code self-contained. Extract a common library only
after stable, genuinely shared functionality is apparent; do not make the new
solver depend on Burgers physics types merely to avoid small amounts of code.

Proposed layout:

```text
apps/channel_rans/
  implementation_guide.md
  CMakeLists.txt
  cmake/BuildInfo.h.in
  include/channel_rans/
    Config.h, ConfigIO.h, Grid.h, State.h
    ClosureModel.h, MomentumFlux.h, ChannelSolver.h
    ReferenceSolution.h, InitialCondition.h, Diagnostics.h
    RunMetadata.h, RunOutputRecorder.h
    environment/
      EnvironmentConfigIO.h, ChannelEnvironment.h
      ActionMapping.h, ObservationBuilder.h, RewardModel.h, TargetProfile.h
  src/                         # corresponding implementation files
    environment/
      smarties_main.cpp
  tools/run_solver.cpp
  tools/generate_reference.cpp
  tools/run_environment.cpp     # scripted environment verification
  configs/
  scripts/
  tests/
    fixtures/
```

Proposed build targets are `channel_rans_core`, `channel_rans_environment`,
`channel_rans_run_solver`, `channel_rans_generate_reference`,
`channel_rans_run_environment`, and optional `channel_rans_smarties`.
Use aliases such as `channel_rans::core`. Prefix test executable names too.

The existing Burgers application already defines a CMake target `run_solver`.
Use a unique channel target; optionally set its output name to `run_solver` in
the channel binary directory. Both applications must configure together.

## 4. Finite-volume and advancement contracts

### 4.1 Grid and boundaries

Store `N+1` face coordinates spanning `[0,h]`, `N` cell centres, positive cell
widths, and centre-to-centre distances. Provide uniform and smoothly stretched
construction. Validate monotonicity and reject zero-width cells. No API should
wrap neighbour indices periodically.

For an interior face,

\[
s_{i+1/2}=\frac{U_{i+1}-U_i}{y_{i+1}-y_i},\quad
F_{i+1/2}=\nu s_{i+1/2}
             +\ell_{i+1/2}^2|s_{i+1/2}|s_{i+1/2},
\]

\[
\dot U_i=G+\frac{F_{i+1/2}-F_{i-1/2}}{\Delta y_i}.
\]

Recommended first wall flux: `F_wall=nu*(U_0-0)/(y_0-0)`; turbulent wall flux is
zero because the wall mixing length vanishes. At the centreline set total flux
to zero exactly. This simple boundary approximation must be assessed with
cell-average convergence tests; do not claim exact discrete reproduction of
the laminar polynomial or a formal boundary order without verification.

For D5, a practical prescription is linear, distance-weighted interpolation of
cell coefficients to interior faces followed by `ell_face=y_face*C_face`.
Use that same representation for prescribed and learned fields. Alternatively,
face-native actions can be considered later. Document interpolation and wall
handling explicitly: interpolating `C`, `ell`, and `nu_t` are different schemes.

The continuum reference remains independent of this representation. A scripted
oracle sampled on the numerical grid may have interpolation error; refine it
rather than assuming exact equality to the continuum solution.

### 4.2 Solver API responsibilities

Proposed operations:

```cpp
setPrescribedCoefficientField(coefficients); // C, not C squared
closureFields(state, fields);
rightHandSide(state, derivative);
stableTimeStep(state);
advanceTo(state, initial_time, target_time, limits, observer);
solveToSteady(state, convergence_config, observer);
```

The coefficient field is held until replaced. Eddy viscosity is recomputed
from the current gradient at each explicit stage or nonlinear iteration.
The core solver has no policy, target-profile, or reward dependency.

Advancement results should include achieved time, accepted/rejected steps,
nonlinear iterations, residuals, convergence status, and a failure reason.
Reaching a requested time is distinct from reaching a steady state.

For explicit stepping, derive a safe bound from the actual nonuniform operator
and its boundary conductances. The nonlinear turbulent flux has derivative
`nu+2*ell^2*abs(s)` with respect to `s`; a frozen-viscosity diffusion bound alone
must not be assumed sufficient. Test stability as coefficient magnitude and
wall refinement increase.

For implicit stepping, lag positive face viscosities to form a tridiagonal
backward-Euler system and iterate to a defined nonlinear tolerance. Check the
original nonlinear residual, not only the lagged linear-system residual.
Bound iteration counts and provide step reduction or a clearly reported
failure when convergence is not obtained.

### 4.3 Convergence and conservation

Use dimensionless diagnostics, including:

- Momentum residual, e.g. `max(abs(G+div(F)))/G`.
- Profile change over a specified interval, scaled by `u_tau`.
- Steady face-stress error `max(abs(F-G*(h-y_face)))/(G*h)`.
- Bulk velocity and wall stress.

Require residual and profile-change criteria together for sustained convergence.
Small timesteps alone must not trigger a false steady-state declaration. A
steady residual criterion is a numerical condition, not a requirement that the
profile match the reference target.

The integrated momentum equation is

\[
\frac{d}{dt}\int_0^h U\,dy=Gh-F(0).
\]

Check its discrete counterpart. For this initial nonnegative eddy viscosity,
the continuous mean-kinetic-energy balance is

\[
\frac{d}{dt}\int_0^h\frac{U^2}{2}\,dy
=G\int_0^h U\,dy-\int_0^h(\nu+\nu_t)U_y^2\,dy.
\]

Use the matching discrete flux work for numerical budgets. Record time
integration defects separately; mean-flow turbulent dissipation here is transfer
out of the mean field, not resolved turbulent kinetic energy.

## 5. Standalone executable and reference artifacts

### 5.1 `run_solver` workflow

Follow the useful structure of the existing Burgers runner:

1. Parse arguments and load a strict versioned JSON configuration.
2. Apply explicit CLI overrides and validate the resolved configuration.
3. For `--dry-run`, print the resolved configuration without advancing or
   creating simulation output.
4. Construct grid, closure, initial state, and output recorder.
5. Advance to requested output times or solve to steady, respecting step and
   nonlinear iteration budgets.
6. Write the final state and run metadata, including meaningful failure status.
7. Return a nonzero code on invalid input or a failed requested solve.

Recommended CLI: `--config`, `--output-directory`, `--dry-run`, and `--help`.
Add `--seed` only when randomized initial conditions are implemented. Keep
physics choices in configuration rather than proliferating CLI switches.

Proposed configuration groups:

| Group | Contents |
| --- | --- |
| `schema_version` | Explicit supported version |
| `physics` | `h`, `nu`, `G`; derived scales recorded separately |
| `grid` | Cell count, mesh kind, stretching parameters |
| `closure` | `laminar`, `constant_coefficient`, `van_driest`, or `prescribed_coefficient`; type-specific parameters |
| `initial_condition` | Zero, analytic laminar, or validated profile input |
| `numerics` | Flux representation, integrator, timestep and nonlinear controls |
| `termination` | Fixed final time or steady solve; tolerances and budgets |
| `output` | Directory, profile/scalar intervals, final output controls |

Reject unknown keys, invalid enum values, nonfinite parameters, incompatible
field lengths, and malformed profiles. Store enough metadata to reproduce the
mesh and all normalization choices.

### 5.2 Outputs and analysis

Write resolved configuration, build revision/dirty status, and run status.
Proposed data products:

- `channel_history.csv`: time, bulk velocity, wall stress, residuals, stress
  error, step counts, and nonlinear iteration counts.
- `channel_profiles.csv`: time, cell geometry, `U`, `U_plus`, `C`, mixing
  length, and clearly located diagnostic gradients/eddy viscosities.
- `channel_faces.csv`: face coordinate, gradient, mixing length, molecular
  and turbulent flux, and total flux.
- `channel_final_profile.csv`: final cell-average solution.
- `channel_metadata.json`: physical scales, conventions, config, convergence,
  provenance, and failure information.

Output names are proposed; keep schemas versioned. Distinguish face and cell
quantities rather than writing ambiguous columns. Use metadata to state whether
profiles are point values or cell averages.

Initial plotting scripts should compare `U_plus(y_plus)`, the logarithmic
diagnostic `y_plus*dU_plus/dy_plus`, normalized total stress, coefficient and
mixing-length profiles, and convergence histories. Fourier spectra and
turbulent-time averaging are not required.

### 5.3 Reference generation

Provide a small standalone reference generator using the quadrature solution.
Record `h,nu,G,kappa,A_plus`, the outer model identifier/parameters, quadrature
tolerance, and reference resolution. Export velocity targets separately from
optional coefficient diagnostics so the environment need not load the latter.

Target loading must validate geometry, units, physical case, and representation.
Do not reuse the Burgers target's periodic interpolation or require its energy
spectrum. Integrate or conservatively project the reference onto each training
mesh; document any interpolation error.

Maintain two comparisons:

1. Continuum quadrature versus PDE under refinement, which verifies numerics.
2. Sampled reference coefficient run versus learned coefficient run on the same
   mesh, which establishes the attainable numerical baseline for RL.

Do not hide continuum discretization error by reporting only the second.

## 6. RL environment specification

### 6.1 Action and observation contracts

One scalar coefficient action per cell is sufficient. The initial action map
should cover `0 <= C <= C_max`, with a documented baseline action corresponding
to `C=kappa`. Zero raw action need not mean zero physical coefficient; record
both values and the exact map. A linear bounded map is the simplest option.

Choose `C_max` large enough for the intended reference but not arbitrarily
large. Do not use an exponential map with a substantial positive lower bound
that prevents recovery of small near-wall coefficients. Report clipping and
saturation. Remove Burgers' zero-mean projection and periodic smoothing.

Initially use no action smoothing. If later enabled, define its physical-space
metric, wall treatment, and nonuniform-grid behavior; it changes the class of
recoverable coefficient fields.

Recommended observations are scaled versions of `y_plus` and `y/h`. Wall
coordinates are allowed; neither reference damping values nor target velocity
errors are policy inputs. Record feature transformations such as
`log1p(y_plus)` if used, and hold them fixed during evaluation.

Coordinates alone describe the desired stationary coefficient map. They do
not describe the evolving velocity state of a sequential transient environment.
If D7 selects that mode, explicitly accept partial observability or add suitable
velocity/shear/history features. A local stencil still does not generally expose
the full globally coupled flow state.

### 6.2 Transition semantics: choose deliberately

**Recommended initial mode: one joint action and a converged solve.**

1. Reset the physical case and initial velocity deterministically from the
   episode configuration/seed.
2. Supply observations to all cell-agents.
3. Collect a complete coefficient field before changing the solver.
4. Hold that field fixed while solving the momentum equation to steady state.
5. Compute and deliver the global profile reward, then end the episode.

This is a cooperative one-step optimization problem, not a long-horizon control
problem. It is a useful first test of SMARTIES coefficient learning with a
stationary target. Coordinate-only observations are consistent with it. Compare
against simple coefficient optimization as a baseline when practical.

**Later alternative: sequential transient decisions.** Hold each action field
for a specified physical interval, retain the evolving velocity state, and
return a reward and observations after the interval. Keep physical time distinct
from solver iterations. Reuse the Burgers action-holding/output-event contract.
Determine warmup, horizon, observations, and reward timing explicitly before
implementing this mode; do not inherit the Burgers values.

For either mode, final evaluation must run the frozen policy to a verified
equilibrium. A profile that briefly crosses the target during startup is not
successful steady closure recovery.

### 6.3 Reward and evaluation metrics

Start with the shared global reward

\[
r=-\sum_i w_i
\left(\frac{\overline U_i-\overline U_{\mathrm{target},i}}{u_\tau}\right)^2,
\qquad w_i=\Delta y_i/h,\quad \sum_iw_i=1.
\]

Every agent receives the same scalar. More near-wall weighting is an explicit
experiment, not an accidental consequence of counting clustered cells equally.
Do not divide by local target velocity, which approaches zero at the wall.

Optional regularization can penalize coefficient roughness or departure from a
constant baseline. Define it in a mesh-consistent coordinate, state its weight,
and report its contribution separately. It biases coefficient recovery; it must
not become supervision toward the exact damped coefficient.

Record velocity error, bulk velocity error, coefficient error in absolute units,
stress balance, convergence, and saturation separately. Assess coefficient
recovery away from the least sensitive wall/centreline regions as well as over
the complete profile. Compare absolute errors when `C_ref` is near zero.

At fixed `G`, equilibrium wall stress is fixed by `G*h`; it is a conservation
check, not a sufficient learning objective. `U` already is the Reynolds mean;
an EMA is not needed to estimate it in this deterministic benchmark.

### 6.4 Failure and reproducibility

Detect nonfinite actions/states, negative prescribed coefficients, invalid
targets, linear/nonlinear solve failures, and exhausted convergence budgets.
Log the case and reason. Never reward a failed steady solve as though its final
transient profile were a valid equilibrium.

Define the failure penalty before training and verify it does not make solver
failure preferable to a valid poor closure. Keep numerical failure, normal
task termination, and time truncation distinct in the environment API.

Reset all solver state, coefficient fields, counters, and optional history.
Randomness is for reproducible case/initial-condition sampling and policy
exploration; stochastic forcing is not part of the channel equation.

## 7. Phased implementation and acceptance gates

### Phase 0 — Scaffolding and contracts

**Implement:** standalone C++14 CMake project, core target, prefixed test target,
configuration types/validation/serialization, metadata skeleton, runner
`--help` and `--dry-run`. Document D1/D2 choices and establish output units.

**Gate:** configure and build the channel app without MPI or SMARTIES; valid
configuration round-trips; invalid physical values fail clearly. No solver
advancement is claimed yet.

### Phase 1 — Grid, state, and boundary geometry

**Implement:** cell-average state, uniform/stretched grids, `N+1` face layout,
wall and centreline boundary treatment, field-size checks.

**Gate:** verify domain coverage, positive volumes, face/centre distances,
integration of constant fields, and absence of periodic coupling. Establish
mesh-family definitions for refinement before measuring convergence.

### Phase 2 — Laminar momentum solver and usable runner

**Implement:** molecular flux, constant pressure source, explicit advancement,
step limits, accepted-step observer, fixed-time/steady termination, initial
conditions, output scheduling, scalar/profile output, final metadata.

**Verify:** analytic laminar cell averages and bulk velocity under refinement;
integrated transient momentum balance; steady linear stress; wall conditions;
failure on invalid timesteps/budgets. A useful transient reference is a decaying
perturbation proportional to `sin(pi*y/(2*h))` about the laminar solution, with
decay rate `nu*(pi/(2*h))^2`; compare its cell averages.

**Gate:** `run_solver --config <laminar case>` produces a verified solution and
meaningful diagnostics. Set numerical tolerances from measured errors rather
than requiring roundoff agreement from an approximate boundary discretization.

### Phase 3 — Mixing-length closures and prescribed actions in the core

**Implement:** laminar, constant `C`, van Driest, and prescribed `C` modes;
coefficient interpolation; nonlinear turbulent flux; closure diagnostics;
stage-by-stage viscosity updates; safe explicit step selection.

**Verify:** `C=0` recovers laminar; constant prescribed `C` equals the constant
model; sampled reference coefficients use the same numerical closure path;
nonnegative coefficients give nonnegative viscosity and dissipative transport;
negative/nonfinite fields fail; prescribed fields remain held during advances.

**Gate:** standalone prescribed-coefficient solves work without an RL environment.
No forcing-action code is required for this phase.

### Phase 4 — Independent reference and wall-law verification

**Implement:** stable quadratic root, independent quadrature, reference metadata,
cell-average projection, reference generation CLI, comparison plots.

**Verify:** laminar limit of quadrature; quadrature tolerance refinement; PDE
convergence toward the nonlinear reference; total-stress identity; viscous
sublayer and overlap diagnostics at appropriate `Re_tau`.

**Gate:** freeze a documented reference case and tolerances. Record continuum
error and sampled-oracle error separately. D3 is recorded as `none` unless an
outer treatment has actually been specified and verified.

### Phase 5 — Efficient steady solves

**Implement:** tridiagonal backward-Euler diffusion, Picard iteration,
nonlinear residual checks, bounded retries/step reduction, and reliable
`solveToSteady`. Preserve explicit advancement as a verification option.

**Verify:** implicit and explicit steady solutions agree within established
accuracy; timestep refinement checks transient accuracy; final solutions do not
depend materially on nonlinear tolerance; timestep reduction does not fabricate
convergence. Exercise strongly damped and large-coefficient cases on clustered
grids.

**Gate:** reference and baseline solves are affordable and reliable enough for
repeated environment evaluations. Do not call a one-pass lagged solve fully
nonlinearly converged.

### Phase 6 — Baselines and sensitivity before RL

**Implement/run:** laminar, undamped `C=kappa`, and sampled reference-coefficient
baselines on the intended environment grid. Perturb coefficients in wall,
buffer, overlap, and centreline regions and measure velocity response.

**Gate:** demonstrate a meaningful velocity-error gap between baseline and
oracle; select D9/D10/D13/D14; confirm action bounds can express the reference
and solver noise is below meaningful action-induced differences. Document the
regions where coefficients are weakly identifiable.

### Phase 7 — Environment without SMARTIES

**Implement:** `ChannelEnvironment`, target loader, action map, observations,
reward, reset/step/status, and a scripted rollout executable. Resolve D7/D8
before defining the transition API. Reuse the standalone output recorder.

**Verify:** exact reward calculation on known profiles; coordinate normalization;
deterministic reset; coefficient holding; reference data excluded from policy
inputs; baseline and oracle actions reproduce standalone results; malformed
actions and solve failures produce the intended status and penalty.

For sequential mode, additionally verify that output subdivisions do not change
the held action or transition endpoint, and test the actual chosen reward timing.

**Gate:** run complete scripted episodes without SMARTIES and show that the
oracle outperforms the baseline by the expected amount. This gate isolates
physics/environment defects from communicator or learner defects.

### Phase 8 — SMARTIES integration

**Implement:** optional root/app `CHANNEL_RANS_ENABLE_SMARTIES` build option,
`channel_rans_smarties`, application-settings file, complete learner settings,
resolved environment output, episode CSV, and deterministic evaluation cases.

Follow the existing Burgers communicator pattern:

1. Construct and validate the environment before communication.
2. Set `N` agents, the observation dimension, and one scalar action dimension.
3. Set raw action bounds and finalize the problem description.
4. Send each initial observation, collect all actions, then advance the joint
   environment once.
5. Send each agent its shared reward and next/final observation.
6. Honor asynchronous shutdown after sends/receives without continuing a
   partially shut-down communication loop.

Check the existing library's learner-sharing configuration so all cell-agents
actually use one policy. Do not assume `setNumAgents` alone documents sharing.
Maintain a fixed number of cell-agents within a job; perform mesh-transfer
evaluation through separately configured jobs unless dynamic resizing is
explicitly supported and tested.

Use `sendTermState` for genuine terminal transitions, including successful
one-step episodes; use `sendLastState` for a continuing task truncated by an
artificial time limit, following the current library semantics. Verify this
mapping in a smoke run. Numerical failure is terminal with a failure record.

Retain `--appSettings`, `--setupFolder`, and resolved-settings provenance as in
Burgers. Package target velocity data and metadata in the setup folder; keep
oracle coefficient data separate or explicitly evaluation-only. If an oracle
action override exists, reject it during training.

Audit build flags: root `libsmarties` currently exports Release `-ffast-math`.
Ensure channel finite/nonfinite validation remains reliable in integrated
builds; inspect actual compile flags and test failure behavior rather than
assuming standalone behavior carries over.

**Gate:** standalone channel build still works; root builds with Burgers and
channel enabled do not collide; one-worker training/evaluation smoke runs
terminate correctly; scripted environment and integrated evaluation agree;
checkpoint/restart and resolved settings are recorded. Only then scale worker
counts.

### Phase 9 — Learning and held-out evaluation

**Run:** single-reference recovery first, multiple reproducible training seeds,
then a documented `Re_tau` distribution. Evaluate frozen policies on held-out
cases, initial conditions, and separately configured meshes.

**Report:** velocity and bulk errors against baseline/oracle, coefficient and
mixing-length profiles, stress/residual checks, clipping, failure rates,
variation across seeds, and inference inputs. Compare to simple optimization
where useful, especially in one-step mode.

**Gate:** sustained converged velocity improvement over the undamped baseline;
quantified coefficient recovery in sensitive regions; no dependence on hidden
reference inputs or merely transient target crossings. Set quantitative success
thresholds using Phase 6 sensitivity results before inspecting training success.

### Phase 10 — Optional outer treatment and richer problems

Only after the first benchmark works, choose a reference outer function,
document its positivity/centreline behavior, regenerate references, and ask the
same learned coefficient to recover both inner and outer behavior. Keep the
outer formula absent from the learned closure.

Other independent extensions are state-dependent policy inputs, sequential
transient decisions, bulk-flow driving, DNS targets, transported turbulence
variables, and richer geometries. Tensor-basis identification is not a small
extension of the scalar velocity-only objective; it requires a separate design.

## 8. Intended build and run workflow

These commands describe the planned interface and will become executable as
the corresponding phases land. All paths are relative to the repository root.

```bash
cmake -S apps/channel_rans -B apps/channel_rans/build \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build apps/channel_rans/build
ctest --test-dir apps/channel_rans/build --output-on-failure

apps/channel_rans/build/run_solver \
  --config apps/channel_rans/configs/laminar.json --dry-run
apps/channel_rans/build/run_solver \
  --config apps/channel_rans/configs/laminar.json
apps/channel_rans/build/run_solver \
  --config apps/channel_rans/configs/van_driest.json
```

The `run_solver` binary spelling above assumes the proposed `OUTPUT_NAME` choice
is adopted. The build command builds the runner and tests. For targeted builds,
use phase-specific test targets and filters; building only the runner does not
build test executables.

For integrated builds, after Phase 8 adds the root option:

```bash
cmake -S . -B build-channel-rans \
  -DCHANNEL_RANS_ENABLE_SMARTIES=ON -DCOMPILE_PY_SO=OFF
cmake --build build-channel-rans --target channel_rans_smarties
```

Document the tested SMARTIES launch command, setup-folder contents, and
checkpoint/evaluation invocation in a new `SMARTIES.md` during Phase 8, based on
the repository's actual launcher behavior. Do not invent an untested launch
command in advance.

## 9. Completion checklist

- Standalone solver and independent reference agree under refinement.
- Laminar, total-stress, transient momentum, and wall-law checks pass.
- Reference damping is absent from the prescribed/learned closure path.
- Prescribed coefficients control physical mixing length, never grid length.
- Coefficient representation, target averaging, and boundary conventions match
  their documented contracts.
- Scripted environment reproduces standalone baseline and oracle results.
- SMARTIES remains optional and coexists with the Burgers build.
- RL transition semantics, reward, action range, and termination are explicit.
- Frozen-policy results include steady convergence and held-out evaluation.
- Remaining weak identifiability and numerical errors are reported separately
  from learning performance.

## 10. Background references

- [Van Driest mixing-length model documentation](https://www.cham.co.uk/phoenics/d_polis/d_enc/turmod/enc_t314.htm): exponential damping convention and typical constants.
- [Universal velocity profile research](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/universal-velocity-profile-for-turbulent-wall-flows-including-adverse-pressure-gradient-boundary-layers/C7E193ACA13CEF1C87043A2BE6E3686A): broader mixing-length/wall-profile context; not an additional initial model requirement.
- [Existing Burgers implementation guide](../burgers_sgs/implementation_guide.md), [SMARTIES notes](../burgers_sgs/SMARTIES.md), and [communication flowchart](../burgers_sgs/SMARTIES_FLOWCHART.md): repository-specific implementation references. Burgers forcing and periodic assumptions do not apply to the channel problem.
