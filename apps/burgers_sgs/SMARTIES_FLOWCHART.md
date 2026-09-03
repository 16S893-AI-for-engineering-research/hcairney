# SMARTIES/Burgers interaction flowchart

## Runtime architecture

```mermaid
flowchart LR
  CLI["burgers_smarties command<br/>SMARTIES options + app settings"]

  subgraph S["SMARTIES"]
    ENG["Engine / launcher<br/>parse settings and create N workloads"]
    ROUTER["Communication handlers<br/>multiplex state/action messages"]
    POLICY["Shared VRACER learner<br/>policy/value network"]
    MEMORY["Replay memory<br/>transitions from every cell and environment"]

    ROUTER -->|"observations + rewards<br/>+ episode status"| MEMORY
    MEMORY -->|"training batches"| POLICY
    POLICY -->|"one action per cell agent"| ROUTER
  end

  CLI --> ENG
  ENG --> ROUTER

  subgraph E0["Environment process 0 — simulation_000_..."]
    APP0["appMain callback<br/>one agent per grid cell"]
    ENV0["SGSEnvironment 0<br/>episode/reset, observations,<br/>action projection, reward"]
    SOL0["BurgersSolver 0<br/>grid, PDE state, numerics,<br/>closure and forcing"]
    APP0 --> ENV0
    ENV0 -->|"projected cell-wise<br/>additive forcing field"| SOL0
    SOL0 -->|"advanced PDE state"| ENV0
  end

  subgraph E1["Environment process 1 — simulation_001_..."]
    APP1["appMain callback<br/>one agent per grid cell"]
    ENV1["SGSEnvironment 1"]
    SOL1["BurgersSolver 1"]
    APP1 --> ENV1
    ENV1 -->|"projected action field"| SOL1
    SOL1 -->|"advanced PDE state"| ENV1
  end

  subgraph EN["Environment process N-1 — simulation_N-1_..."]
    APPN["appMain callback<br/>one agent per grid cell"]
    ENVN["SGSEnvironment N-1"]
    SOLN["BurgersSolver N-1"]
    APPN --> ENVN
    ENVN -->|"projected action field"| SOLN
    SOLN -->|"advanced PDE state"| ENVN
  end

  ROUTER <-->|"socket messages<br/>(independent asynchronous rollout)"| APP0
  ROUTER <-->|"socket messages"| APP1
  ROUTER <-->|"..."| APPN
```

`--nEnvironments N` creates `N` independent rollout processes. Each process
has its own `SGSEnvironment`, `BurgersSolver`, PDE state, random stream,
episode clock, and simulation directory. They do not exchange PDE state with
one another. They only interact indirectly by contributing experience to, and
requesting actions from, the shared SMARTIES learner.

For a Burgers grid with `C` cells, every environment registers `C` SMARTIES
agents. Agent `i` observes the local stencil around cell `i`, returns one scalar
action, and receives cell `i`'s reward. The supplied `settings.json` creates a
single learner, so one policy is reused across all `N * C` logical agents.

## One environment's decision loop

The following loop runs independently in every environment process.

```mermaid
flowchart TD
  RESET["Reset episode<br/>choose seed, create BurgersSolver,<br/>initialize PDE state and running mean"]
  OBS["ObservationBuilder<br/>build one local observation per grid cell"]
  SEND0["sendInitState for each cell agent"]
  ACT["SMARTIES shared policy<br/>select one scalar raw action per cell"]
  RECV["appMain recvAction loop<br/>assemble the complete raw action field"]
  PROJ["ActionProjection<br/>bound/scale/smooth and enforce constraints"]
  APPLY["BurgersSolver<br/>install projected additive forcing field"]
  ADV["advanceTo<br/>integrate PDE for one decision interval<br/>(adaptive internal timesteps)"]
  METRICS["SGSEnvironment<br/>update running mean, compare with DNS target,<br/>compute per-cell rewards and next observations"]
  STATUS{"Episode status?"}
  STEP["sendState for each cell<br/>SMARTIES stores transition and returns next action"]
  LAST["sendLastState<br/>normal time-limit completion"]
  FAIL["sendTermState<br/>solver/projection/safety failure<br/>with failure penalty"]
  SUMMARY["Write episode summary<br/>and optional evaluation diagnostics"]
  STOP{"Global SMARTIES<br/>shutdown signal?"}

  RESET --> OBS --> SEND0 --> ACT --> RECV --> PROJ --> APPLY --> ADV --> METRICS --> STATUS
  STATUS -->|"running"| STEP
  STEP --> ACT
  STATUS -->|"timeout"| SUMMARY --> LAST --> STOP
  STATUS -->|"failure"| SUMMARY --> FAIL --> STOP
  STOP -->|"no: start another episode"| RESET
  STOP -->|"yes"| DONE["Environment process exits"]
```

The application sends and receives cell-agent messages one at a time, but the
`N` environment processes progress independently. Consequently a slow Burgers
rollout does not require all other environments to reach the same physical
time before their next policy request.

During training, SMARTIES adds the returned transitions to replay memory and
updates the shared policy. During evaluation, the network is frozen; configured
held-out seeds are used, and optional detailed Burgers output is recorded.

## Code map

- SMARTIES entry point and cell-agent communication:
  [`src/environment/smarties_main.cpp`](src/environment/smarties_main.cpp)
- Environment reset, action projection, solver advancement, and rewards:
  [`src/environment/SGSEnvironment.cpp`](src/environment/SGSEnvironment.cpp)
- PDE solver owned by each environment:
  [`src/BurgersSolver.cpp`](src/BurgersSolver.cpp)
- SMARTIES process creation and per-simulation directories:
  [`../../source/smarties/Core/Launcher.cpp`](../../source/smarties/Core/Launcher.cpp)
- SMARTIES state/action routing and learner selection:
  [`../../source/smarties/Core/Worker.cpp`](../../source/smarties/Core/Worker.cpp)

