#include "burgers/BurgersSolver.h"

#include "burgers/ConvectiveFlux.h"
#include "burgers/Diagnostics.h"
#include "burgers/Reconstruction.h"
#include "burgers/ViscousFlux.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace burgers {
namespace {

std::string joinErrors(const std::vector<std::string>& errors) {
  std::ostringstream message;
  for(std::size_t index = 0; index < errors.size(); ++index) {
    if(index != 0) {
      message << "; ";
    }
    message << errors[index];
  }
  return message.str();
}

Grid makeSolverGrid(const RunConfig& config) {
  const std::vector<std::string> errors = validate(config);
  if(!errors.empty()) {
    throw std::invalid_argument("invalid solver configuration: " +
                                joinErrors(errors));
  }
  if(config.time_integration.integrator != TimeIntegrator::SspRk3) {
    throw std::invalid_argument(
      "solver supports only SSP-RK3 time integration");
  }
  if(config.closure.type == ClosureType::DynamicSmagorinsky) {
    throw std::invalid_argument(
      "dynamic_smagorinsky is intentionally unsupported in Phase 7");
  }
  return Grid(config.grid);
}

void requireCompatibleFiniteState(const Grid& grid,
                                  const State& state,
                                  const char* context) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(std::string(context) +
                                " state size does not match solver grid");
  }
  for(std::size_t cell = 0; cell < state.size(); ++cell) {
    if(std::isfinite(state[cell]) == 0) {
      throw std::runtime_error(std::string(context) +
                               " state contains a non-finite value at cell " +
                               std::to_string(cell));
    }
  }
}

void requireFiniteTime(double value, const char* name) {
  if(std::isfinite(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

}  // namespace

BurgersSolver::BurgersSolver(const RunConfig& config)
  : grid_(makeSolverGrid(config)),
    molecular_viscosity_(config.viscosity.molecular),
    face_viscosity_averaging_(config.viscosity.face_averaging),
    reconstruction_(config.numerical_method.reconstruction),
    convective_flux_(config.numerical_method.convective_flux),
    limiter_(config.numerical_method.limiter),
    closure_(makeClosureModel(grid_, config.closure)),
    forcing_(grid_, config.forcing, config.viscosity.molecular,
             config.random.seed),
    prescribed_additive_forcing_(grid_, 0.0),
    prescribed_additive_forcing_is_set_(false),
    advective_cfl_(config.time_integration.advective_cfl),
    diffusive_cfl_(config.time_integration.diffusive_cfl),
    maximum_steps_(config.time_integration.maximum_steps) {}

const Grid& BurgersSolver::grid() const noexcept {
  return grid_;
}

double BurgersSolver::molecularViscosity() const noexcept {
  return molecular_viscosity_;
}

FaceViscosityAveraging
BurgersSolver::faceViscosityAveraging() const noexcept {
  return face_viscosity_averaging_;
}

ClosureType BurgersSolver::closureType() const noexcept {
  return closure_->type();
}

void BurgersSolver::setPrescribedCoefficientField(
  const std::vector<double>& coefficients) {
  closure_->setCoefficientField(coefficients);
}

void BurgersSolver::closureFields(const State& state,
                                  ClosureFields& fields) const {
  requireCompatibleFiniteState(grid_, state, "closure input");
  closure_->evaluate(grid_, state, fields);
}

void BurgersSolver::setPrescribedAdditiveForcingField(
  const std::vector<double>& forcing) {
  if(forcing.size() != grid_.cellCount()) {
    throw std::invalid_argument(
      "prescribed additive forcing field size does not match the solver "
      "grid");
  }
  State validated(grid_, 0.0);
  for(std::size_t cell = 0; cell < forcing.size(); ++cell) {
    if(std::isfinite(forcing[cell]) == 0) {
      throw std::invalid_argument(
        "prescribed additive forcing at cell " + std::to_string(cell) +
        " must be finite");
    }
    validated[cell] = forcing[cell];
  }
  prescribed_additive_forcing_ = std::move(validated);
  prescribed_additive_forcing_is_set_ = true;
}

void BurgersSolver::clearPrescribedAdditiveForcingField() noexcept {
  prescribed_additive_forcing_.fill(0.0);
  prescribed_additive_forcing_is_set_ = false;
}

bool BurgersSolver::hasActiveForcing() const noexcept {
  return forcing_.type() != ForcingType::None ||
         prescribed_additive_forcing_is_set_;
}

void BurgersSolver::evaluateForcingFields(
  double time, ForcingFields& fields) const {
  forcing_.evaluate(time, fields);
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    fields.prescribed[cell] = prescribed_additive_forcing_[cell];
    fields.total[cell] += fields.prescribed[cell];
    if(std::isfinite(fields.total[cell]) == 0) {
      throw std::runtime_error(
        "composed forcing produced a non-finite value at cell " +
        std::to_string(cell));
    }
  }
}

void BurgersSolver::rightHandSide(const State& state,
                                  State& derivative) const {
  if(hasActiveForcing()) {
    throw std::invalid_argument(
      "right-hand side with forcing requires physical time");
  }
  rightHandSide(state, 0.0, derivative);
}

void BurgersSolver::rightHandSide(const State& state,
                                  double time,
                                  State& derivative) const {
  rightHandSideWithClosure(state, time, derivative, nullptr);
}

void BurgersSolver::rightHandSideWithClosure(
  const State& state,
  double time,
  State& derivative,
  ClosureFields* closure_fields) const {
  requireCompatibleFiniteState(grid_, state, "right-hand-side input");
  requireFiniteTime(time, "right-hand-side time");
  if(derivative.size() != grid_.cellCount()) {
    throw std::invalid_argument(
      "right-hand-side output state size does not match solver grid");
  }

  FaceStates face_states(grid_.cellCount());
  reconstruct(grid_, state, reconstruction_, limiter_, face_states);

  std::vector<double> advective_fluxes(grid_.cellCount(), 0.0);
  std::vector<double> viscous_fluxes(grid_.cellCount(), 0.0);
  computeConvectiveFluxes(face_states,
                          convective_flux_,
                          advective_fluxes);
  ClosureFields evaluated_closure(grid_);
  closure_->evaluate(grid_, state, evaluated_closure);
  computeViscousFluxes(grid_,
                       state,
                       molecular_viscosity_,
                       evaluated_closure.eddy_viscosity,
                       face_viscosity_averaging_,
                       viscous_fluxes);
  if(closure_fields != nullptr) {
    *closure_fields = evaluated_closure;
  }

  ForcingFields forcing_fields(grid_);
  evaluateForcingFields(time, forcing_fields);

  const double inverse_width = 1.0 / grid_.cellWidth();
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    const std::size_t left_face = grid_.neighbor(cell, -1);
    derivative[cell] =
      (-(advective_fluxes[cell] - advective_fluxes[left_face]) +
       (viscous_fluxes[cell] - viscous_fluxes[left_face])) *
      inverse_width + forcing_fields.total[cell];
    if(std::isfinite(derivative[cell]) == 0) {
      throw std::runtime_error(
        "right-hand side produced a non-finite value at cell " +
        std::to_string(cell));
    }
  }
}

double BurgersSolver::stableTimeStep(const State& state) const {
  requireCompatibleFiniteState(grid_, state, "timestep-selection input");

  double maximum_speed = 0.0;
  for(const double value : state) {
    maximum_speed = std::max(maximum_speed, std::abs(value));
  }

  ClosureFields closure_fields(grid_);
  closure_->evaluate(grid_, state, closure_fields);
  double maximum_effective_viscosity = molecular_viscosity_;
  for(const double value : closure_fields.eddy_viscosity) {
    maximum_effective_viscosity =
      std::max(maximum_effective_viscosity,
               molecular_viscosity_ + value);
  }

  const double infinity = std::numeric_limits<double>::infinity();
  const double advective_limit =
    maximum_speed > 0.0
      ? advective_cfl_ * grid_.cellWidth() / maximum_speed
      : infinity;
  const double diffusive_limit =
    maximum_effective_viscosity > 0.0
      ? diffusive_cfl_ * grid_.cellWidth() * grid_.cellWidth() /
          maximum_effective_viscosity
      : infinity;
  const double time_step = std::min(advective_limit, diffusive_limit);

  if(std::isnan(time_step) != 0 || time_step <= 0.0) {
    throw std::runtime_error(
      "CFL selection did not produce a positive timestep");
  }
  return time_step;
}

double BurgersSolver::stableTimeStep(const State& state, double time) const {
  requireFiniteTime(time, "timestep-selection time");
  return std::min(stableTimeStep(state),
                  forcing_.timeUntilNextClock(time));
}

void BurgersSolver::forcingFields(double time, ForcingFields& fields) const {
  requireFiniteTime(time, "forcing field time");
  evaluateForcingFields(time, fields);
}

std::string BurgersSolver::serializeStochasticForcingState() const {
  return forcing_.serializeStochasticState();
}

void BurgersSolver::restoreStochasticForcingState(
  const std::string& serialized) {
  forcing_.restoreStochasticState(serialized);
}

SspRk3StepBudget BurgersSolver::advanceSspRk3(
  State& state, double time_step) const {
  if(hasActiveForcing()) {
    throw std::invalid_argument(
      "SSP-RK3 with forcing requires physical time");
  }
  return advanceSspRk3(state, 0.0, time_step);
}

SspRk3StepBudget BurgersSolver::advanceSspRk3(
  State& state, double time, double time_step) const {
  requireCompatibleFiniteState(grid_, state, "SSP-RK3 input");
  requireFiniteTime(time, "SSP-RK3 time");
  requireFiniteTime(time_step, "SSP-RK3 timestep");
  if(time_step <= 0.0) {
    throw std::invalid_argument("SSP-RK3 timestep must be positive");
  }

  const std::size_t initial_clock_index = forcing_.currentClockIndex();
  forcing_.prepareStep(time, time_step);

  const State initial = state;
  State derivative(grid_);
  State stage_one(grid_);
  State stage_two(grid_);

  const double stage_one_time = time + time_step;
  const double stage_two_time = time + 0.5 * time_step;
  requireFiniteTime(stage_one_time, "SSP-RK3 stage-one time");
  requireFiniteTime(stage_two_time, "SSP-RK3 stage-two time");

  ForcingFields forcing_zero(grid_);
  ForcingFields forcing_one(grid_);
  ForcingFields forcing_two(grid_);
  ForcingPower power_zero;
  ForcingPower power_one;
  ForcingPower power_two;
  ClosureFields closure_zero(grid_);
  ClosureFields closure_one(grid_);
  ClosureFields closure_two(grid_);
  const double initial_energy = kineticEnergy(grid_, initial);

  try {
    evaluateForcingFields(time, forcing_zero);
    power_zero = forcingPower(grid_, initial, forcing_zero);
    rightHandSideWithClosure(
      initial, time, derivative, &closure_zero);
    for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
      stage_one[cell] = initial[cell] + time_step * derivative[cell];
    }
    requireCompatibleFiniteState(grid_, stage_one, "SSP-RK3 stage one");

    evaluateForcingFields(stage_one_time, forcing_one);
    power_one = forcingPower(grid_, stage_one, forcing_one);
    rightHandSideWithClosure(
      stage_one, stage_one_time, derivative, &closure_one);
    for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
      stage_two[cell] =
        0.75 * initial[cell] +
        0.25 * (stage_one[cell] + time_step * derivative[cell]);
    }
    requireCompatibleFiniteState(grid_, stage_two, "SSP-RK3 stage two");

    evaluateForcingFields(stage_two_time, forcing_two);
    power_two = forcingPower(grid_, stage_two, forcing_two);
    rightHandSideWithClosure(
      stage_two, stage_two_time, derivative, &closure_two);
    for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
      state[cell] =
        (1.0 / 3.0) * initial[cell] +
        (2.0 / 3.0) *
          (stage_two[cell] + time_step * derivative[cell]);
    }
    requireCompatibleFiniteState(grid_, state, "SSP-RK3 result");
    forcing_.commitStep(stage_one_time);
  } catch(...) {
    forcing_.cancelPreparedStep();
    state = initial;
    throw;
  }

  const double one_sixth = 1.0 / 6.0;
  const double two_thirds = 2.0 / 3.0;
  const double weighted_molecular_dissipation =
    one_sixth * molecularDissipation(
      grid_, initial, molecular_viscosity_) +
    one_sixth * molecularDissipation(
      grid_, stage_one, molecular_viscosity_) +
    two_thirds * molecularDissipation(
      grid_, stage_two, molecular_viscosity_);
  const double weighted_sgs_dissipation =
    one_sixth * sgsDissipation(
      grid_, initial, closure_zero.eddy_viscosity,
      face_viscosity_averaging_) +
    one_sixth * sgsDissipation(
      grid_, stage_one, closure_one.eddy_viscosity,
      face_viscosity_averaging_) +
    two_thirds * sgsDissipation(
      grid_, stage_two, closure_two.eddy_viscosity,
      face_viscosity_averaging_);

  SspRk3StepBudget budget;
  budget.time_step = time_step;
  budget.deterministic_work = time_step * (
    one_sixth * power_zero.deterministic +
    one_sixth * power_one.deterministic +
    two_thirds * power_two.deterministic);
  budget.stochastic_work = time_step * (
    one_sixth * power_zero.stochastic +
    one_sixth * power_one.stochastic +
    two_thirds * power_two.stochastic);
  budget.prescribed_work = time_step * (
    one_sixth * power_zero.prescribed +
    one_sixth * power_one.prescribed +
    two_thirds * power_two.prescribed);
  budget.manufactured_work = time_step * (
    one_sixth * power_zero.manufactured +
    one_sixth * power_one.manufactured +
    two_thirds * power_two.manufactured);
  budget.molecular_dissipation =
    time_step * weighted_molecular_dissipation;
  budget.sgs_dissipation = time_step * weighted_sgs_dissipation;
  budget.energy_change = kineticEnergy(grid_, state) - initial_energy;
  budget.numerical_dissipation =
    budget.deterministic_work + budget.stochastic_work +
    budget.prescribed_work + budget.manufactured_work -
    budget.molecular_dissipation - budget.sgs_dissipation -
    budget.energy_change;
  budget.advanced_forcing_clock =
    forcing_.currentClockIndex() != initial_clock_index;
  return budget;
}

AdvanceResult BurgersSolver::advanceTo(State& state,
                                       double initial_time,
                                       double target_time) const {
  return advanceTo(state, initial_time, target_time, maximum_steps_);
}

AdvanceResult BurgersSolver::advanceTo(State& state,
                                       double initial_time,
                                       double target_time,
                                       std::size_t maximum_steps) const {
  return advanceTo(
    state, initial_time, target_time, maximum_steps, StepObserver{});
}

AdvanceResult BurgersSolver::advanceTo(State& state,
                                       double initial_time,
                                       double target_time,
                                       std::size_t maximum_steps,
                                       const StepObserver& observer) const {
  requireCompatibleFiniteState(grid_, state, "advance input");
  requireFiniteTime(initial_time, "initial time");
  requireFiniteTime(target_time, "target time");
  if(target_time < initial_time) {
    throw std::invalid_argument("target time must not precede initial time");
  }

  AdvanceResult result;
  result.initial_time = initial_time;
  result.final_time = initial_time;
  if(target_time == initial_time) {
    return result;
  }
  if(maximum_steps == 0) {
    throw std::runtime_error(
      "maximum timestep count reached before advancing toward target time");
  }

  double time = initial_time;
  while(time < target_time) {
    if(result.timestep_count == maximum_steps) {
      throw std::runtime_error(
        "maximum timestep count reached before target time");
    }

    const double stability_limit = stableTimeStep(state, time);
    const double remaining = target_time - time;
    const bool shortened =
      std::isfinite(stability_limit) != 0 && remaining < stability_limit;
    const double time_step = std::min(stability_limit, remaining);
    if(std::isfinite(time_step) == 0 || time_step <= 0.0) {
      throw std::runtime_error(
        "selected timestep is not finite and positive");
    }
    if(time + time_step <= time) {
      throw std::runtime_error(
        "selected timestep is too small to advance physical time");
    }

    const SspRk3StepBudget budget =
      advanceSspRk3(state, time, time_step);
    ++result.timestep_count;
    result.deterministic_work += budget.deterministic_work;
    result.stochastic_work += budget.stochastic_work;
    result.prescribed_work += budget.prescribed_work;
    result.manufactured_work += budget.manufactured_work;
    result.molecular_dissipation += budget.molecular_dissipation;
    result.sgs_dissipation += budget.sgs_dissipation;
    result.numerical_dissipation += budget.numerical_dissipation;
    result.energy_change += budget.energy_change;
    if(budget.advanced_forcing_clock) {
      ++result.forcing_clock_step_count;
    }
    if(shortened) {
      ++result.shortened_final_step_count;
    }

    if(time_step == remaining) {
      time = target_time;
    } else {
      time += time_step;
    }
    if(observer) {
      observer(result.timestep_count, time, state);
    }
  }

  result.final_time = time;
  return result;
}

}  // namespace burgers
