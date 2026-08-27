#pragma once

#include "burgers/Config.h"
#include "burgers/Forcing.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstddef>
#include <string>

namespace burgers {

struct AdvanceResult {
  double initial_time = 0.0;
  double final_time = 0.0;
  std::size_t timestep_count = 0;
  std::size_t shortened_final_step_count = 0;
  std::size_t forcing_clock_step_count = 0;
  double deterministic_work = 0.0;
  double stochastic_work = 0.0;
  double manufactured_work = 0.0;
  double molecular_dissipation = 0.0;
  double numerical_dissipation = 0.0;
  double energy_change = 0.0;
};

struct SspRk3StepBudget {
  double time_step = 0.0;
  double deterministic_work = 0.0;
  double stochastic_work = 0.0;
  double manufactured_work = 0.0;
  double molecular_dissipation = 0.0;
  double numerical_dissipation = 0.0;
  double energy_change = 0.0;
  bool advanced_forcing_clock = false;
};

class BurgersSolver {
public:
  explicit BurgersSolver(const RunConfig& config);

  const Grid& grid() const noexcept;
  double molecularViscosity() const noexcept;

  // Autonomous convenience overload. Any forcing requires the explicit-time
  // overload below.
  void rightHandSide(const State& state, State& derivative) const;
  void rightHandSide(const State& state,
                     double time,
                     State& derivative) const;
  double stableTimeStep(const State& state) const;
  double stableTimeStep(const State& state, double time) const;

  void forcingFields(double time, ForcingFields& fields) const;
  std::string serializeStochasticForcingState() const;
  void restoreStochasticForcingState(const std::string& serialized);

  // Autonomous convenience overload. Any forcing requires the explicit-time
  // overload below.
  SspRk3StepBudget advanceSspRk3(State& state, double time_step) const;
  SspRk3StepBudget advanceSspRk3(State& state,
                                 double time,
                                 double time_step) const;

  AdvanceResult advanceTo(State& state,
                          double initial_time,
                          double target_time) const;
  AdvanceResult advanceTo(State& state,
                          double initial_time,
                          double target_time,
                          std::size_t maximum_steps) const;

private:
  Grid grid_;
  double molecular_viscosity_;
  Reconstruction reconstruction_;
  ConvectiveFlux convective_flux_;
  Limiter limiter_;
  mutable Forcing forcing_;
  double advective_cfl_;
  double diffusive_cfl_;
  std::size_t maximum_steps_;
};

}  // namespace burgers
