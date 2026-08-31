#pragma once

#include "burgers/Config.h"
#include "burgers/ClosureModel.h"
#include "burgers/Forcing.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
  double sgs_dissipation = 0.0;
  double numerical_dissipation = 0.0;
  double energy_change = 0.0;
};

struct SspRk3StepBudget {
  double time_step = 0.0;
  double deterministic_work = 0.0;
  double stochastic_work = 0.0;
  double manufactured_work = 0.0;
  double molecular_dissipation = 0.0;
  double sgs_dissipation = 0.0;
  double numerical_dissipation = 0.0;
  double energy_change = 0.0;
  bool advanced_forcing_clock = false;
};

class BurgersSolver {
public:
  // Called after each accepted step. The index is relative to the current
  // advanceTo call, and the time and state are those at the end of the step.
  using StepObserver =
    std::function<void(std::size_t, double, const State&)>;

  explicit BurgersSolver(const RunConfig& config);

  const Grid& grid() const noexcept;
  double molecularViscosity() const noexcept;
  FaceViscosityAveraging faceViscosityAveraging() const noexcept;
  ClosureType closureType() const noexcept;

  // Values supplied here are C_S, not C_S^2. The field is held until the
  // next call and eddy viscosity is still recomputed at every RK stage.
  void setPrescribedCoefficientField(
    const std::vector<double>& coefficients);
  void closureFields(const State& state, ClosureFields& fields) const;

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
  AdvanceResult advanceTo(State& state,
                          double initial_time,
                          double target_time,
                          std::size_t maximum_steps,
                          const StepObserver& observer) const;

private:
  void rightHandSideWithClosure(const State& state,
                                double time,
                                State& derivative,
                                ClosureFields* closure_fields) const;

  Grid grid_;
  double molecular_viscosity_;
  FaceViscosityAveraging face_viscosity_averaging_;
  Reconstruction reconstruction_;
  ConvectiveFlux convective_flux_;
  Limiter limiter_;
  std::unique_ptr<ClosureModel> closure_;
  mutable Forcing forcing_;
  double advective_cfl_;
  double diffusive_cfl_;
  std::size_t maximum_steps_;
};

}  // namespace burgers
