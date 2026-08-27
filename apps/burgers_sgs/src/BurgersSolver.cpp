#include "burgers/BurgersSolver.h"

#include "burgers/ConvectiveFlux.h"
#include "burgers/ManufacturedSolution.h"
#include "burgers/Reconstruction.h"
#include "burgers/ViscousFlux.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

Grid makePhase3Grid(const RunConfig& config) {
  const std::vector<std::string> errors = validate(config);
  if(!errors.empty()) {
    throw std::invalid_argument("invalid solver configuration: " +
                                joinErrors(errors));
  }
  if(config.numerical_method.reconstruction !=
       Reconstruction::PiecewiseConstant) {
    throw std::invalid_argument(
      "Phase 3 solver supports only piecewise-constant reconstruction");
  }
  if(config.numerical_method.convective_flux != ConvectiveFlux::Godunov) {
    throw std::invalid_argument(
      "Phase 3 solver supports only the Godunov convective flux");
  }
  if(config.time_integration.integrator != TimeIntegrator::SspRk3) {
    throw std::invalid_argument(
      "Phase 3 solver supports only SSP-RK3 time integration");
  }
  if(config.forcing.type != ForcingType::None &&
     config.forcing.type != ForcingType::Manufactured) {
    throw std::invalid_argument(
      "Phase 3 solver supports only zero or manufactured forcing");
  }
  if(config.closure.type != ClosureType::NoClosure) {
    throw std::invalid_argument("Phase 3 solver does not support SGS closure");
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
  : grid_(makePhase3Grid(config)),
    molecular_viscosity_(config.viscosity.molecular),
    forcing_type_(config.forcing.type),
    manufactured_forcing_(config.forcing.manufactured),
    advective_cfl_(config.time_integration.advective_cfl),
    diffusive_cfl_(config.time_integration.diffusive_cfl),
    maximum_steps_(config.time_integration.maximum_steps) {}

const Grid& BurgersSolver::grid() const noexcept {
  return grid_;
}

double BurgersSolver::molecularViscosity() const noexcept {
  return molecular_viscosity_;
}

void BurgersSolver::rightHandSide(const State& state,
                                  State& derivative) const {
  if(forcing_type_ != ForcingType::None) {
    throw std::invalid_argument(
      "right-hand side with manufactured forcing requires physical time");
  }
  rightHandSide(state, 0.0, derivative);
}

void BurgersSolver::rightHandSide(const State& state,
                                  double time,
                                  State& derivative) const {
  requireCompatibleFiniteState(grid_, state, "right-hand-side input");
  requireFiniteTime(time, "right-hand-side time");
  if(derivative.size() != grid_.cellCount()) {
    throw std::invalid_argument(
      "right-hand-side output state size does not match solver grid");
  }

  FaceStates face_states(grid_.cellCount());
  reconstructPiecewiseConstant(grid_, state, face_states);

  std::vector<double> advective_fluxes(grid_.cellCount(), 0.0);
  std::vector<double> viscous_fluxes(grid_.cellCount(), 0.0);
  computeGodunovFluxes(face_states, advective_fluxes);
  computeMolecularViscousFluxes(
    grid_, state, molecular_viscosity_, viscous_fluxes);

  State forcing(grid_, 0.0);
  if(forcing_type_ == ForcingType::Manufactured) {
    computeManufacturedSourceCellAverages(grid_,
                                          manufactured_forcing_,
                                          molecular_viscosity_,
                                          time,
                                          forcing);
  }

  const double inverse_width = 1.0 / grid_.cellWidth();
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    const std::size_t left_face = grid_.neighbor(cell, -1);
    derivative[cell] =
      (-(advective_fluxes[cell] - advective_fluxes[left_face]) +
       (viscous_fluxes[cell] - viscous_fluxes[left_face])) *
      inverse_width + forcing[cell];
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

  const double infinity = std::numeric_limits<double>::infinity();
  const double advective_limit =
    maximum_speed > 0.0
      ? advective_cfl_ * grid_.cellWidth() / maximum_speed
      : infinity;
  const double diffusive_limit =
    molecular_viscosity_ > 0.0
      ? diffusive_cfl_ * grid_.cellWidth() * grid_.cellWidth() /
          molecular_viscosity_
      : infinity;
  const double time_step = std::min(advective_limit, diffusive_limit);

  if(std::isnan(time_step) != 0 || time_step <= 0.0) {
    throw std::runtime_error(
      "CFL selection did not produce a positive timestep");
  }
  return time_step;
}

void BurgersSolver::advanceSspRk3(State& state, double time_step) const {
  if(forcing_type_ != ForcingType::None) {
    throw std::invalid_argument(
      "SSP-RK3 with manufactured forcing requires physical time");
  }
  advanceSspRk3(state, 0.0, time_step);
}

void BurgersSolver::advanceSspRk3(State& state,
                                  double time,
                                  double time_step) const {
  requireCompatibleFiniteState(grid_, state, "SSP-RK3 input");
  requireFiniteTime(time, "SSP-RK3 time");
  requireFiniteTime(time_step, "SSP-RK3 timestep");
  if(time_step <= 0.0) {
    throw std::invalid_argument("SSP-RK3 timestep must be positive");
  }

  const State initial = state;
  State derivative(grid_);
  State stage_one(grid_);
  State stage_two(grid_);

  const double stage_one_time = time + time_step;
  const double stage_two_time = time + 0.5 * time_step;
  requireFiniteTime(stage_one_time, "SSP-RK3 stage-one time");
  requireFiniteTime(stage_two_time, "SSP-RK3 stage-two time");

  rightHandSide(initial, time, derivative);
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    stage_one[cell] = initial[cell] + time_step * derivative[cell];
  }
  requireCompatibleFiniteState(grid_, stage_one, "SSP-RK3 stage one");

  rightHandSide(stage_one, stage_one_time, derivative);
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    stage_two[cell] =
      0.75 * initial[cell] +
      0.25 * (stage_one[cell] + time_step * derivative[cell]);
  }
  requireCompatibleFiniteState(grid_, stage_two, "SSP-RK3 stage two");

  rightHandSide(stage_two, stage_two_time, derivative);
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    state[cell] =
      (1.0 / 3.0) * initial[cell] +
      (2.0 / 3.0) *
        (stage_two[cell] + time_step * derivative[cell]);
  }
  requireCompatibleFiniteState(grid_, state, "SSP-RK3 result");
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

    const double stability_limit = stableTimeStep(state);
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

    advanceSspRk3(state, time, time_step);
    ++result.timestep_count;
    if(shortened) {
      ++result.shortened_final_step_count;
    }

    if(time_step == remaining) {
      time = target_time;
    } else {
      time += time_step;
    }
  }

  result.final_time = time;
  return result;
}

}  // namespace burgers
