#pragma once

#include "burgers/Config.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstddef>

namespace burgers {

struct AdvanceResult {
  double initial_time = 0.0;
  double final_time = 0.0;
  std::size_t timestep_count = 0;
  std::size_t shortened_final_step_count = 0;
};

class BurgersSolver {
public:
  explicit BurgersSolver(const RunConfig& config);

  const Grid& grid() const noexcept;
  double molecularViscosity() const noexcept;

  void rightHandSide(const State& state, State& derivative) const;
  double stableTimeStep(const State& state) const;
  void advanceSspRk3(State& state, double time_step) const;

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
  double advective_cfl_;
  double diffusive_cfl_;
  std::size_t maximum_steps_;
};

}  // namespace burgers
