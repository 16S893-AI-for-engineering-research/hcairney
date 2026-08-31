# SMARTIES integration for learned additive forcing

## 1. Objective and scope

This document expands Phase 8 of the
[Burgers SGS implementation guide](implementation_guide.md). It should be kept
consistent with the accepted-reference workflow in
[finalize_reference.md](finalize_reference.md) and the classical baselines in
[phase7_baselines.md](phase7_baselines.md).

The first RL environment should learn a numerics-coupled additive forcing that
makes an implicitly filtered Burgers LES reproduce the accepted DNS mean
velocity profile. Exact recovery of the deterministic forcing used by the DNS
is not an objective.

The reference and controlled equations differ as follows:

\[
\begin{aligned}
  \text{DNS reference:}\quad
  u_t + (u^2/2)_x
    &= [ (\nu+\nu_{\mathrm{sgs}})u_x ]_x
       + F_{\mathrm{mean}}(x)+f'(x,t),\\
  \text{controlled LES:}\quad
  u_t + (u^2/2)_x
    &= [ (\nu+\nu_{\mathrm{sgs}})u_x ]_x
       + f'(x,t)+f_{\mathrm{RL}}(x,t).
\end{aligned}
\]

The controlled LES normally uses `NoClosure` initially so the learned source
is the only new model term. A fixed classical closure may later be retained as
background physics, but its choice must be part of the environment
configuration. The stochastic forcing realization follows the existing
physical-time clock. The known deterministic forcing is disabled except in an
oracle/debug baseline.

The primary success metric is the statistically converged mean-profile error,
not action error:

\[
  E_U =
  \left[
    \frac{\sum_i V_i(\overline u_i^{LES}-\overline u_i^{DNS})^2}
         {\sum_i V_i(\overline u_i^{DNS})^2}
  \right]^{1/2}.
\]

This is deliberately an effective-model problem. The learned forcing is
allowed to couple tightly to the LES grid, flux, limiter, timestepper, and
numerical dissipation. Claims should therefore be about mean-profile recovery
and tested generalization, not identification of a unique SGS term.

## 2. Invariance and multi-agent design

Use one logical agent per LES cell and one shared policy. For every cell,

\[
  \widetilde f_i^n=\pi(o_i^n;\theta),
\]

with the same parameters \(\theta\), observation layout, normalization, and
action bounds. Different cells may produce different actions only because
their local flow states differ.

The initial policy must not observe:

- cell or agent index;
- absolute coordinate \(x_i\);
- a cell-specific learned parameter;
- the local DNS target value \(U_i^{DNS}\); or
- a cell-specific observation normalization.

These exclusions prevent the policy from memorizing the deterministic spatial
forcing. A cyclic shift of the velocity field must cyclically shift the raw and
applied actions. Global physical parameters such as \(\nu\), \(\Delta x\), and
the decision interval may be supplied to every agent if cross-configuration
generalization requires them.

In SMARTIES, call `setNumAgents(N)` and define one common state/action
descriptor. Do not call `agentsDefineDifferentMDP()`. With the common
descriptor, all cells use the same MDP and policy. Independent exploration
noise is preferable initially: `agentsShareExplorationNoise()` would give all
cells the same perturbation, which the zero-mean action projection would mostly
remove.

## 3. Action transformation

SMARTIES should expose one bounded, nondimensional continuous action per cell,
initially \(a_i\in[-1,1]\). The environment transforms the complete raw action
vector, not each cell independently:

1. Validate its size and finite values.
2. Optionally apply a periodic, constant-preserving smoothing filter.
3. Remove the volume-weighted mean.
4. Apply one field-wide rescaling if the projected field exceeds its bound.
5. Multiply by the configured physical forcing scale \(A_f\).

For uniform cells,

\[
  p_i=a_i-\frac{1}{N}\sum_j a_j,
  \qquad
  f_i=A_f\frac{p_i}{\max(1,\max_j|p_j|)}.
\]

For a nonuniform extension, replace the arithmetic mean by
\(\sum_jV_ja_j/\sum_jV_j\). If smoothing is used, repeat the mean removal after
smoothing to protect against roundoff or a non-ideal filter implementation.
Do not clip cells independently after the projection: independent clipping can
break the zero-net-force constraint.

The transformation must satisfy, to roundoff:

- \(\sum_i V_i f_i=0\);
- \(|f_i|\leq A_f\);
- a uniform raw action maps to zero;
- projection is idempotent before amplitude scaling;
- cyclic shifts commute with the transformation; and
- the input vector is not mutated.

Start with either no smoothing or the periodic three-point filter
\(s_i=(a_{i-1}+2a_i+a_{i+1})/4\). It preserves constants and the arithmetic
mean on the uniform grid. Independent agent exploration can otherwise create
grid-scale forcing, so an action-spectrum diagnostic is required before
deciding to omit smoothing.

## 4. Solver changes before linking SMARTIES

Add the forcing path to `burgers_core` without including `smarties.h`.

### 4.1 Public solver API

Add an interface with semantics similar to:

```cpp
void setPrescribedAdditiveForcingField(
  const std::vector<double>& forcing);
void clearPrescribedAdditiveForcingField();
```

The setter accepts physical forcing values, requires exactly one finite value
per cell, and holds the field until it is replaced or cleared. It should not
perform the RL action projection; projection belongs to the environment and is
tested separately.

### 4.2 Time advancement and budgets

Add the held field to every right-hand-side evaluation and every SSP-RK3 stage
within a decision interval. Extend `ForcingFields`, `SspRk3StepBudget`, and
`AdvanceResult` with a separate learned/prescribed forcing component and work
term. The discrete energy diagnostic should resolve

\[
  \Delta E = W_{\mathrm{det}}+W_{\mathrm{stoch}}+W_{\mathrm{RL}}
             -D_\nu-D_{\mathrm{sgs}}-D_{\mathrm{num}}.
\]

Do not classify \(W_{\mathrm{RL}}\) as numerical dissipation. A zero-mean force
can still inject or remove energy because \(\int u f_{\mathrm{RL}}\,dx\) is
generally nonzero.

### 4.3 Solver-level tests

Before adding an environment, test:

- zero field equivalence to the current solver;
- rejection of wrong-sized and non-finite fields;
- persistence until replacement and correct clearing;
- zero-mean preservation of the periodic velocity mean;
- constant prescribed field behavior in a manufactured-source test;
- SSP-RK3 work accounting and energy-budget closure; and
- oracle equivalence: disable configured deterministic forcing, inject its
  analytic cell averages through the prescribed field, and reproduce the
  deterministic-forcing trajectory to tolerance.

The oracle test verifies the new source path. It does not make exact forcing
recovery an RL metric.

## 5. SMARTIES-independent environment core

Implement the environment logic as ordinary C++ classes before adding the
communication loop. A useful split is:

```text
environment/
  ActionProjection.*       # smoothing, zero mean, field-wide scaling
  MeanProfileEstimator.*   # block integration and/or EMA
  ObservationBuilder.*     # periodic local features
  RewardModel.*            # selectable target errors and regularization
  SGSEnvironment.*         # reset, advance one decision, termination
  smarties_main.cpp        # only SMARTIES-dependent file
```

`SGSEnvironment` should expose operations conceptually equivalent to:

```cpp
reset(seed);
observations();
step(raw_actions);  // returns observations, rewards, diagnostics, status
```

This makes deterministic rollout tests possible without starting a learner.

### 5.1 Episode reset

A reset must initialize from a recorded episode seed:

- the LES state and physical time;
- stochastic forcing and its clock;
- the prescribed learned forcing, initially zero;
- the block accumulator and running mean;
- action and reward histories; and
- timeout/failure status.

Use separate derived random streams for initial-condition and stochastic
forcing randomness. Training may vary seeds by episode, while evaluation uses
a fixed held-out seed list.

### 5.2 Decision interval

Specify decisions in physical time, \(\Delta t_{RL}\), rather than a number of
adaptive PDE steps. For decision \(n\):

1. Cache the pre-action mean estimate \(\widehat U^n\), then collect all raw
   actions.
2. Project and install the complete forcing field.
3. Advance from \(t_n\) to \(t_{n+1}=t_n+\Delta t_{RL}\), stopping exactly at
   the endpoint and respecting stochastic-forcing clock boundaries.
4. Accumulate a time-weighted block mean during the advance.
5. Compute any block-velocity reward using the cached pre-action estimate.
6. Update the finite-time mean estimator and compute any reward defined from
   the updated estimate.
7. Compute the remaining diagnostics and build the next observations.

Choose \(\Delta t_{RL}\) only after comparing it with the PDE timestep, forcing
correlation time, and flow autocorrelation time. It should contain multiple PDE
steps but remain short enough for feedback to affect the evolving state.

### 5.3 Initial observations

The initial observation must be Galilean invariant and therefore must not
contain \(u\) itself. Define the dimensionless local gradient

\[
  z_i = \frac{\sqrt{(u_{x,i})^2}\,\Delta x^2}{\nu}
      = \frac{|u_{x,i}|\,\Delta x^2}{\nu}
\]

and transformed feature

\[
  q_i = \ln(1+z_i).
\]

For the first implementation, give each agent a configurable small periodic
stencil of this feature,

\[
  o_i = [q_{i-r},\ldots,q_i,\ldots,q_{i+r}],
\]

where \(r\geq0\) is a configuration parameter shared by all agents. Thus
\(r=0\) gives agent \(i\) only \(q_i\); larger values add neighboring gradient
features, with periodic wrapping at the domain boundary. Keep \(r\) fixed and
small relative to the number of grid cells. The stencil orientation is
meaningful in one dimension and may distinguish flow features without
revealing absolute position. Use the same documented discrete derivative as
the numerical diagnostics.

This logarithm has an argument greater than or equal to one, but requires
\(\nu>0\). The observation builder must reject a nonpositive viscosity. The
absolute gradient deliberately makes compression and expansion with the same
magnitude indistinguishable to the initial policy.

Start with a memoryless feed-forward policy. Add appended past observations or
a recurrent network only if a controlled comparison shows that the
instantaneous local stencil is insufficient. Explicit time is not an
observation; state dependence already makes the applied forcing time varying.
If identical local stencils occur at locations requiring systematically
different corrections, enlarge the translation-equivariant receptive field or
add other Galilean-invariant local derivatives before considering any
absolute-coordinate feature. Such conflicting local transitions are evidence
of partial observability, not a reason to assign cell-specific parameters.

### 5.4 Mean estimator and reward

The accepted DNS profile is loaded once, its hashes and metadata are verified,
and it is conservatively restricted to the LES grid. Do not interpolate it
pointwise when a conservative restriction is available.

A physical-time EMA can be updated from the decision-block mean:

\[
  U_{i,\mathrm{block}}^n
  =\frac{1}{\Delta t_{RL}}
    \int_{t_n}^{t_{n+1}}u_i(t)\,dt,
  \qquad
  \widehat U_i^{n+1}
  =(1-\alpha)\widehat U_i^n+\alpha U_{i,\mathrm{block}}^n,
  \qquad
  \alpha=1-\exp(-\Delta t_{RL}/\tau_U).
\]

With adaptive PDE timesteps, integrate \(U_{i,\mathrm{block}}^n\) with
physical-time or RK-consistent weights; do not average stored solver states
without accounting for their unequal time intervals.

Reward structure is an experimental choice rather than a fixed part of the
additive-forcing formulation. The first local candidate uses the direction of
the pre-action mean error and the velocity block generated by the current
action:

\[
  r_{i,\mathrm{dir}}^n =
  -\operatorname{sign}(\widehat U_i^n-U_i^{DNS})
   \frac{U_{i,\mathrm{block}}^n-U_i^{DNS}}{U_{\mathrm{scale}}}
  -\lambda_a\left(\frac{f_i^n}{A_f}\right)^2.
\]

The ordering is essential: cache \(\widehat U_i^n\), apply the action, form the
block average, evaluate \(r_{i,\mathrm{dir}}^n\), and only then update the EMA.
This prevents the same velocity fluctuations from determining both the sign
and the signed error. If the cached sign represents the long-time mean error,
the expected unregularized reward is proportional to
\(-|\overline u_i-U_i^{DNS}|\); zero-mean fluctuations cancel in expectation.
The block average retains relatively immediate action credit while reducing
the variance of a single-sample reward.

A second transparent local candidate directly penalizes the updated mean
estimate:

\[
  r_{i,\mathrm{EMA2}}^n =
  -w_i\left(
    \frac{\widehat U_i^{n+1}-U_i^{DNS}}{U_{\mathrm{scale}}}
  \right)^2
  -\lambda_a\left(\frac{f_i^n}{A_f}\right)^2.
\]

Use one global \(U_{\mathrm{scale}}\), not a per-cell scale that can become
singular near target zero crossings. Initially set \(\lambda_a=0\) to verify
that the task is controllable, then increase it only if the policy uses
unnecessarily large forcing. Any smoothing or forcing-power penalty should be
introduced one at a time.

Neither local candidate is universally preferable. The directional block
reward is noisier and its lagged sign can temporarily give the wrong incentive
after the mean crosses the target. The squared-EMA reward is smoother and
directly matches a mean-squared profile objective, but it diffuses temporal
credit across the EMA history and weakly penalizes finite-sample estimator
variance. Implement both behind configuration, record the selected type and
parameters, and compare them using identical seeds and evaluation metrics.

For each local form, a shared global aggregation should also be tested. For the
squared form this can be the negative normalized profile error; for the
directional form use the volume-weighted sum of the local directional terms.
Local rewards offer sharper credit but ignore some effects of the globally
projected action; a shared reward is better aligned with the coupled system but
has more diffuse credit. Reward selection must not change the primary frozen-
policy evaluation metric \(E_U\).

The running mean is part of the environment state. Either expose enough history
for an approximately Markov observation or explicitly describe the task as
partially observable. Do not initialize every training episode from the target
mean, which would leak the answer and suppress the transient learning signal.

### 5.5 Termination

Use `sendLastState` for a normal finite-horizon timeout because the policy did
not cause it. Use `sendTermState` with a documented finite penalty when the
action leads to solver failure or a configured physical safety limit. Because
all agents act on one coupled PDE, terminate or truncate all cell agents at the
same decision boundary.

## 6. SMARTIES communication loop

The executable should link `burgers_core`, the environment-core library, and
SMARTIES as a separate target. Retain `run_solver` and all core tests without a
SMARTIES dependency. The SMARTIES target will also require the same MPI and
OpenMP link dependencies used by existing C++ applications.

The initial problem description is approximately:

```cpp
comm->setNumAgents(n_cells);
comm->setStateActionDims(observation_size, 1);
comm->setActionScales({1.0}, {-1.0}, true);
```

Do not call `agentsDefineDifferentMDP()` or
`agentsShareExplorationNoise()`. Call `setStateScales` only with common scales
for the shared observation definition. Finalize the description before the
first episode if useful for catching setup errors early.

The episode loop should follow this ordering:

```text
reset environment
for every cell: sendInitState(local observation, cell ID)

repeat:
  for every cell:
    recvAction(cell ID)
    if SMARTIES requests termination: return
  project complete action vector
  advance one physical-time decision interval
  compute all next observations and rewards
  for every cell:
    sendState, sendLastState, or sendTermState consistently
until episode ends
```

Do not advance the PDE while only a subset of the new cell actions has been
received. The zero-mean projection requires the complete synchronous action
field.

Use the standard C++ entry pattern already present in `apps/cart_pole_cpp` and
`apps/cart_pole_many`:

```cpp
smarties::Engine engine(argc, argv);
if (engine.parse()) return 1;
engine.run(app_main);
```

Pass the Burgers environment configuration through SMARTIES `--appSettings` or
the setup directory rather than embedding scientific settings in the wrapper.
Record both the resolved Burgers configuration and the SMARTIES learner
settings in each run directory.

## 7. Build and test sequence

Implement in the following order so failures remain localized:

1. Add the prescribed additive-field solver API and energy diagnostics.
2. Add `ActionProjection` and its property tests.
3. Add target loading/restriction and `MeanProfileEstimator` tests.
4. Add `ObservationBuilder` invariance and periodic-stencil tests.
5. Add `RewardModel` tests for both reward structures, pre-action-sign timing,
   time-weighted block integration, scaling, aggregation, and finite values.
6. Add `SGSEnvironment` deterministic reset and rollout tests with scripted
   actions.
7. Add the optional SMARTIES CMake target and communication wrapper.
8. Run a zero-action communication smoke test.
9. Run a scripted/oracle-action smoke test through the same environment path.
10. Run a short RL training job, restart from a checkpoint, and perform frozen
    policy evaluation on held-out seeds.

The core CTest suite must still configure and run when SMARTIES integration is
disabled. Add an option such as `BURGERS_ENABLE_SMARTIES` rather than making the
verified numerical library depend unconditionally on the parent project.

## 8. Baselines and ablations

The minimum comparison set is:

1. **No learned forcing:** controlled LES with \(f_{RL}=0\).
2. **Shared linear local law:** one translation-equivariant local mapping such as
   \(\widetilde f_i=\boldsymbol\beta^T\phi(o_i)\), with a small global parameter
   vector optimized outside RL and the same projection as the RL actions.
3. **Shared nonlinear RL law:** the SMARTIES policy with the same observations,
   projection, bounds, and evaluation protocol.

“Static” in the second baseline means that \(\boldsymbol\beta\) is fixed, not
that each cell owns a different constant. Its output remains state dependent
and therefore changes in space and time. No baseline may contain a learned
per-cell constant or absolute-coordinate input.

Also run the known deterministic DNS forcing through the prescribed additive
path as an oracle/debug case. It establishes whether the coarse numerical
system can approach the target under the original forcing, but it is spatially
prescribed and is not a generalizable closure baseline.

After the first successful training run, ablate one factor at a time:

- directional block-velocity versus squared-EMA reward;
- local versus shared global aggregation for each reward structure;
- EMA timescale and, for the directional reward, decision-block duration;
- no action penalty versus justified nonzero \(\lambda_a\);
- no smoothing versus periodic smoothing;
- action amplitude and decision interval;
- instantaneous stencil versus appended history;
- no classical closure versus a fixed static Smagorinsky background; and
- training grid versus held-out grids.

There is no need for an explicitly time-indexed open-loop forcing baseline.
State dependence already produces time-varying actions and is the mechanism
that can generalize across locations and realizations.

## 9. Evaluation and reporting

Evaluate frozen policies over stationary windows and held-out stochastic seeds
using the existing block/ensemble analysis. The primary report contains
\(E_U\), its sampling uncertainty, and the absolute profile error. Also report:

- selected reward structure and aggregation, \(U_{\mathrm{scale}}\),
  \(\tau_U\), \(\lambda_a\), and the block-integration convention;
- training-return histories alongside mean-profile-error histories;
- temporal variance and resolved energy spectrum;
- kinetic energy and molecular, SGS, and numerical dissipation;
- deterministic, stochastic, and learned forcing power separately;
- raw and applied action mean, RMS, maximum, saturation fraction, and spectrum;
- zero-mean projection residual;
- numerical failures and safety-limit terminations; and
- wall-clock cost per decision and per simulated physical-time unit.

Test at least held-out seeds and initial conditions before calling a policy
successful. Claims of closure generalization additionally require held-out
grids, viscosities, forcing parameters, and, eventually, discretizations. A
policy that matches the mean while severely corrupting other statistics has met
the narrow training objective, but that behavior must be made visible rather
than described as recovery of SGS physics.

## 10. Initial completion criteria

The first integration milestone is complete when:

1. The additive forcing and work diagnostics pass unit and energy-budget tests.
2. Action projection is bounded, zero mean, and cyclically equivariant.
3. Scripted rollouts are reproducible for a fixed seed.
4. All cells use one common SMARTIES policy description with no spatial
   identifiers in the observations.
5. Normal timeouts and action-induced failures use the correct SMARTIES episode
   status for every agent.
6. A short training job improves held-out mean-profile error over the no-action
   baseline without numerical failure.
7. Frozen-policy evaluation records the primary metric, uncertainty, secondary
   flow statistics, and action diagnostics.
8. The shared nonlinear policy is compared against the shared linear local-law
   baseline, so any benefit from RL is separated from the benefit of simply
   adding a controllable forcing term.
9. Matched short-training and frozen-policy comparisons of the directional
   block-velocity and squared-EMA rewards have been recorded before selecting a
   default reward.
