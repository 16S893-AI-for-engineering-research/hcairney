#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/ConvectiveFlux.h"
#include "burgers/Diagnostics.h"
#include "burgers/Grid.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"
#include "burgers/ViscousFlux.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if(!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expectNear(double actual,
                double expected,
                double tolerance,
                const std::string& message) {
  if(std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " (expected " << expected
              << ", got " << actual << ", tolerance " << tolerance
              << ")\n";
    ++failures;
  }
}

template<typename Exception, typename Function>
void expectThrows(Function function, const std::string& message) {
  try {
    function();
  } catch(const Exception&) {
    return;
  } catch(const std::exception& error) {
    std::cerr << "FAIL: " << message << " (wrong exception: "
              << error.what() << ")\n";
    ++failures;
    return;
  }
  std::cerr << "FAIL: " << message << " (no exception)\n";
  ++failures;
}

burgers::RunConfig unforcedConfig(std::size_t cell_count,
                                  double molecular_viscosity) {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = cell_count;
  config.viscosity.molecular = molecular_viscosity;
  config.forcing.type = burgers::ForcingType::None;
  config.closure.type = burgers::ClosureType::NoClosure;
  return config;
}

double convergenceOrder(double coarse_error, double fine_error) {
  if(coarse_error <= 0.0 || fine_error <= 0.0) {
    return 0.0;
  }
  return std::log(coarse_error / fine_error) / std::log(2.0);
}

void checkFourierViscousOperator() {
  const double pi = 3.1415926535897932384626433832795;
  const burgers::Grid grid(0.0, 2.0 * pi, 128);
  const int mode = 3;
  const double viscosity = 0.07;
  burgers::State state(grid);
  burgers::sinusoidalInitialCondition(0.0, 0.8, mode, 0.31)(grid, state);

  std::vector<double> fluxes(grid.cellCount(), 0.0);
  burgers::computeMolecularViscousFluxes(
    grid, state, viscosity, fluxes);
  const double discrete_eigenvalue =
    -4.0 * std::pow(std::sin(0.5 * static_cast<double>(mode) *
                            grid.cellWidth()), 2.0) /
    (grid.cellWidth() * grid.cellWidth());
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t left_face = grid.neighbor(cell, -1);
    const double viscous_operator =
      (fluxes[cell] - fluxes[left_face]) / grid.cellWidth();
    expectNear(viscous_operator,
               viscosity * discrete_eigenvalue * state[cell],
               2.0e-11,
               "periodic centered viscous operator must have its exact "
               "Fourier eigenvalue");
  }
}

void checkFluxConsistency() {
  for(const double state : {-4.0, -1.25, 0.0, 0.75, 3.5}) {
    expectNear(burgers::godunovFlux(state, state),
               burgers::burgersPhysicalFlux(state),
               2.0e-15,
               "Godunov flux must be consistent for every sampled state");
  }
}

void checkLongConservation() {
  burgers::RunConfig config = unforcedConfig(128, 0.02);
  config.time_integration.advective_cfl = 0.25;
  config.time_integration.diffusive_cfl = 0.15;
  const burgers::BurgersSolver solver(config);

  burgers::State random_state(solver.grid());
  burgers::randomInitialCondition(0.35, 0.6, 48291u)(
    solver.grid(), random_state);
  const double initial_mean = burgers::mean(solver.grid(), random_state);
  const double initial_scale =
    std::max(1.0, burgers::l1Norm(solver.grid(), random_state));
  const burgers::AdvanceResult result =
    solver.advanceTo(random_state, 0.0, 1.0);
  const double drift =
    burgers::mean(solver.grid(), random_state) - initial_mean;
  const double tolerance =
    256.0 * std::numeric_limits<double>::epsilon() * initial_scale *
    static_cast<double>(solver.grid().cellCount()) *
    static_cast<double>(std::max<std::size_t>(1, result.timestep_count));
  std::cout << "random mean drift: absolute=" << std::abs(drift)
            << " relative=" << std::abs(drift) / initial_scale
            << " tolerance=" << tolerance << '\n';
  expect(std::abs(drift) <= tolerance,
         "long random evolution must conserve its mean within a "
         "scale-aware roundoff bound");

  burgers::State smooth_state(solver.grid());
  burgers::sinusoidalInitialCondition(0.4, 0.5, 2, 0.1)(
    solver.grid(), smooth_state);
  const double smooth_initial_mean =
    burgers::mean(solver.grid(), smooth_state);
  const burgers::AdvanceResult smooth_result =
    solver.advanceTo(smooth_state, 0.0, 1.0);
  const double smooth_scale =
    std::max(1.0, burgers::l1Norm(solver.grid(), smooth_state));
  const double smooth_drift =
    burgers::mean(solver.grid(), smooth_state) - smooth_initial_mean;
  const double smooth_tolerance =
    256.0 * std::numeric_limits<double>::epsilon() * smooth_scale *
    static_cast<double>(solver.grid().cellCount()) *
    static_cast<double>(
      std::max<std::size_t>(1, smooth_result.timestep_count));
  std::cout << "smooth mean drift: absolute=" << std::abs(smooth_drift)
            << " relative=" << std::abs(smooth_drift) / smooth_scale
            << " tolerance=" << smooth_tolerance << '\n';
  expect(std::abs(smooth_drift) <= smooth_tolerance,
         "long smooth evolution must conserve its mean within a "
         "scale-aware roundoff bound");
}

struct EnergyStudyResult {
  double residual = 0.0;
  bool monotone = true;
};

EnergyStudyResult runEnergyStudy(double time_step) {
  burgers::RunConfig config = unforcedConfig(128, 0.03);
  const burgers::BurgersSolver solver(config);
  burgers::State state(solver.grid());
  burgers::sinusoidalInitialCondition(0.1, 0.4, 2, 0.2)(
    solver.grid(), state);

  const double final_time = 0.04;
  const std::size_t step_count = static_cast<std::size_t>(
    std::llround(final_time / time_step));
  const double initial_energy =
    burgers::kineticEnergy(solver.grid(), state);
  double integrated_dissipation = 0.0;
  bool monotone = true;

  for(std::size_t step = 0; step < step_count; ++step) {
    burgers::State derivative_before(solver.grid());
    solver.rightHandSide(state, derivative_before);
    const burgers::UnforcedEnergyBudgetRate before =
      burgers::unforcedEnergyBudgetRate(solver.grid(),
                                        state,
                                        derivative_before,
                                        solver.molecularViscosity());
    expect(before.numerical_dissipation >= -2.0e-12,
           "Godunov numerical dissipation must be nonnegative to roundoff");

    const double energy_before =
      burgers::kineticEnergy(solver.grid(), state);
    const double time = static_cast<double>(step) * time_step;
    solver.advanceSspRk3(state, time, time_step);
    const double energy_after =
      burgers::kineticEnergy(solver.grid(), state);
    monotone = monotone && energy_after <= energy_before + 2.0e-14;

    burgers::State derivative_after(solver.grid());
    solver.rightHandSide(state, derivative_after);
    const burgers::UnforcedEnergyBudgetRate after =
      burgers::unforcedEnergyBudgetRate(solver.grid(),
                                        state,
                                        derivative_after,
                                        solver.molecularViscosity());
    expect(after.numerical_dissipation >= -2.0e-12,
           "Godunov numerical dissipation must remain nonnegative");
    integrated_dissipation +=
      0.5 * time_step *
      (before.molecular_dissipation + before.numerical_dissipation +
       after.molecular_dissipation + after.numerical_dissipation);
  }

  EnergyStudyResult result;
  result.residual = std::abs(
    burgers::kineticEnergy(solver.grid(), state) - initial_energy +
    integrated_dissipation);
  result.monotone = monotone;
  return result;
}

void checkEnergyBudget() {
  const EnergyStudyResult coarse = runEnergyStudy(0.002);
  const EnergyStudyResult fine = runEnergyStudy(0.001);
  const double residual_order =
    convergenceOrder(coarse.residual, fine.residual);
  std::cout << "energy residual dt=0.002: " << coarse.residual << '\n'
            << "energy residual dt=0.001: " << fine.residual << '\n'
            << "energy residual order: " << residual_order << '\n';
  expect(coarse.monotone && fine.monotone,
         "unforced viscous energy must be monotone at both timesteps");
  expect(fine.residual < coarse.residual,
         "time-discrete energy residual must decrease with timestep");
  expect(residual_order >= 1.5,
         "trapezoidal energy-budget residual must approach second order");
  expect(fine.residual <= 1.0e-5,
         "refined energy-budget residual must meet its regression bound");
}

void checkCflAndFailures() {
  burgers::RunConfig config = unforcedConfig(4, 0.5);
  config.grid.x_end = 4.0;
  const burgers::BurgersSolver solver(config);
  const burgers::State initial(
    burgers::State::Container{0.0, -2.0, 1.0, 0.5});
  const double limit = solver.stableTimeStep(initial);
  for(const double fraction : {0.5, 0.99}) {
    burgers::State state = initial;
    solver.advanceSspRk3(state, 0.0, fraction * limit);
    for(const double value : state) {
      expect(std::isfinite(value) != 0,
             "a timestep at or below the CFL limit must remain finite");
    }
  }

  burgers::RunConfig large_viscosity_config = unforcedConfig(8, 1.0e6);
  large_viscosity_config.grid.x_end = 8.0;
  const burgers::BurgersSolver large_viscosity_solver(
    large_viscosity_config);
  const burgers::State zero(large_viscosity_solver.grid(), 0.0);
  expectNear(large_viscosity_solver.stableTimeStep(zero),
             2.0e-7,
             1.0e-21,
             "large viscosity must produce a finite diffusive timestep");

  burgers::RunConfig negative_viscosity = unforcedConfig(16, -0.1);
  expectThrows<std::invalid_argument>(
    [&negative_viscosity]() {
      const burgers::BurgersSolver invalid(negative_viscosity);
      static_cast<void>(invalid);
    },
    "negative molecular viscosity must be rejected by solver construction");

  burgers::State wrong_size(3);
  burgers::State derivative(solver.grid());
  expectThrows<std::invalid_argument>(
    [&solver, &wrong_size, &derivative]() {
      solver.rightHandSide(wrong_size, derivative);
    },
    "right-hand side must reject a state of the wrong size");
  expectThrows<std::invalid_argument>(
    [&solver, &initial, &derivative]() {
      solver.rightHandSide(
        initial,
        std::numeric_limits<double>::quiet_NaN(),
        derivative);
    },
    "right-hand side must reject non-finite physical time");
  expectThrows<std::invalid_argument>(
    [&solver, &initial]() {
      burgers::State state = initial;
      solver.advanceSspRk3(state, 0.0, 0.0);
    },
    "SSP-RK3 must reject a zero timestep");

  burgers::RunConfig invalid_manufactured = unforcedConfig(16, 0.01);
  invalid_manufactured.forcing.type = burgers::ForcingType::Manufactured;
  invalid_manufactured.forcing.manufactured.wavenumber = 0;
  expectThrows<std::invalid_argument>(
    [&invalid_manufactured]() {
      const burgers::BurgersSolver invalid(invalid_manufactured);
      static_cast<void>(invalid);
    },
    "manufactured forcing must reject a zero wavenumber");

  burgers::RunConfig nonperiodic_manufactured = unforcedConfig(16, 0.01);
  nonperiodic_manufactured.grid.x_end = 4.0;
  nonperiodic_manufactured.forcing.type =
    burgers::ForcingType::Manufactured;
  expectThrows<std::invalid_argument>(
    [&nonperiodic_manufactured]() {
      const burgers::BurgersSolver invalid(nonperiodic_manufactured);
      static_cast<void>(invalid);
    },
    "manufactured mode must be periodic on the configured domain");
}

}  // namespace

int main() {
  checkFluxConsistency();
  checkFourierViscousOperator();
  checkLongConservation();
  checkEnergyBudget();
  checkCflAndFailures();

  if(failures != 0) {
    std::cerr << failures << " Phase 3 invariant check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 3 invariant checks passed\n";
  return 0;
}
