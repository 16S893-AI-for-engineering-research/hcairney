#include "Phase4Methods.h"

#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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

burgers::RunConfig compositeConfig() {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = 96;
  config.viscosity.molecular = 0.02;
  config.forcing.type = burgers::ForcingType::Composite;
  config.forcing.deterministic.modes = {{1, 0.1, 0.0}};
  config.forcing.stochastic = burgers::makeLowModeOuForcingConfig();
  config.forcing.stochastic.stationary_rms = 0.1;
  config.forcing.stochastic.correlation_time = 0.5;
  config.forcing.stochastic.clock_interval = 0.05;
  config.time_integration.advective_cfl = 0.2;
  config.time_integration.diffusive_cfl = 0.1;
  return config;
}

double maximumDifference(const burgers::State& left,
                         const burgers::State& right) {
  double result = 0.0;
  for(std::size_t cell = 0; cell < left.size(); ++cell) {
    result = std::max(result, std::abs(left[cell] - right[cell]));
  }
  return result;
}

void checkMeanConservationForMethodMatrix() {
  for(const phase4_test::MethodCase& method : phase4_test::methodCases()) {
    burgers::RunConfig config = compositeConfig();
    phase4_test::selectMethod(config, method);
    const burgers::BurgersSolver solver(config);
    burgers::State state(solver.grid());
    burgers::randomInitialCondition(0.3, 0.25, 4567u)(solver.grid(), state);
    const double initial_mean = burgers::mean(solver.grid(), state);
    const burgers::AdvanceResult advance =
      solver.advanceTo(state, 0.0, 0.12);
    const double drift =
      burgers::mean(solver.grid(), state) - initial_mean;
    const double scale =
      std::max(1.0, burgers::l1Norm(solver.grid(), state));
    const double tolerance =
      1024.0 * std::numeric_limits<double>::epsilon() * scale *
      static_cast<double>(solver.grid().cellCount()) *
      static_cast<double>(std::max<std::size_t>(
        1, advance.timestep_count));
    std::cout << method.name << " forced_mean_drift=" << std::abs(drift)
              << " tolerance=" << tolerance
              << " forcing_clock_steps="
              << advance.forcing_clock_step_count << '\n';
    expect(std::abs(drift) <= tolerance,
           std::string(method.name) +
             " must conserve bulk velocity under zero-mean composite "
             "forcing");
    expect(advance.forcing_clock_step_count == 2,
           std::string(method.name) +
             " must advance the forcing clock at t=0.05 and t=0.10");
  }
}

void checkClockBoundAndEnergyBudget() {
  burgers::RunConfig config = compositeConfig();
  config.viscosity.molecular = 0.0;
  const burgers::BurgersSolver solver(config);
  burgers::State zero(solver.grid(), 0.0);
  expectNear(solver.stableTimeStep(zero, 0.0), 0.05, 2.0e-16,
             "forcing clock must bound an otherwise unlimited PDE step");
  burgers::ForcingFields constant_forcing(solver.grid());
  burgers::State constant_derivative(solver.grid());
  solver.forcingFields(0.0, constant_forcing);
  solver.rightHandSide(zero, 0.0, constant_derivative);
  const burgers::ForcedEnergyBudgetRate constant_budget =
    burgers::forcedEnergyBudgetRate(solver.grid(),
                                    zero,
                                    constant_derivative,
                                    constant_forcing,
                                    0.0);
  expectNear(constant_budget.energy_rate, 0.0, 2.0e-16,
             "zero velocity must have zero instantaneous forcing power");
  expectNear(constant_budget.numerical_dissipation, 0.0, 2.0e-16,
             "forcing a zero state must not be classified as numerical "
             "dissipation");

  burgers::State state(solver.grid());
  burgers::sinusoidalInitialCondition(0.2, 0.3, 2, 0.17)(
    solver.grid(), state);
  const burgers::SspRk3StepBudget budget =
    solver.advanceSspRk3(state, 0.0, 0.04);
  const double reconstructed_energy_change =
    budget.deterministic_work + budget.stochastic_work +
    budget.manufactured_work - budget.molecular_dissipation -
    budget.numerical_dissipation;
  std::cout << "forced_step deterministic_work="
            << budget.deterministic_work
            << " stochastic_work=" << budget.stochastic_work
            << " molecular_dissipation=" << budget.molecular_dissipation
            << " numerical_dissipation=" << budget.numerical_dissipation
            << " energy_change=" << budget.energy_change << '\n';
  expectNear(reconstructed_energy_change, budget.energy_change, 2.0e-15,
             "forced SSP-RK3 energy-budget terms must close per step");
  expect(std::isfinite(budget.deterministic_work) != 0 &&
           std::isfinite(budget.stochastic_work) != 0 &&
           std::isfinite(budget.numerical_dissipation) != 0,
         "forced energy-budget terms must remain finite");

  expectThrows<std::invalid_argument>(
    [&solver, &state]() {
      burgers::State copy = state;
      solver.advanceSspRk3(copy, 0.04, 0.02);
    },
    "a direct RK step must not cross a forcing-clock boundary");
}

void checkSolverRestart() {
  burgers::RunConfig config = compositeConfig();
  config.random.seed = 1881u;
  const burgers::BurgersSolver uninterrupted(config);
  burgers::State state(uninterrupted.grid());
  burgers::sinusoidalInitialCondition(0.15, 0.25, 1, 0.2)(
    uninterrupted.grid(), state);
  uninterrupted.advanceTo(state, 0.0, 0.10);
  const std::string forcing_restart =
    uninterrupted.serializeStochasticForcingState();
  const burgers::State checkpoint = state;
  uninterrupted.advanceTo(state, 0.10, 0.20);

  burgers::BurgersSolver restarted(config);
  burgers::State restarted_state = checkpoint;
  restarted.restoreStochasticForcingState(forcing_restart);
  restarted.advanceTo(restarted_state, 0.10, 0.20);
  expectNear(maximumDifference(state, restarted_state), 0.0, 0.0,
             "solver restart must reproduce the uninterrupted state exactly");
  expect(uninterrupted.serializeStochasticForcingState() ==
           restarted.serializeStochasticForcingState(),
         "solver restart must reproduce the uninterrupted forcing state");
}

void checkConfigurationFailures() {
  burgers::RunConfig duplicate = compositeConfig();
  duplicate.forcing.stochastic.wavenumbers = {1, 1};
  expectThrows<std::invalid_argument>(
    [&duplicate]() {
      const burgers::BurgersSolver invalid(duplicate);
      static_cast<void>(invalid);
    },
    "duplicate stochastic modes must be rejected");

  burgers::RunConfig nyquist = compositeConfig();
  nyquist.grid.cell_count = 16;
  nyquist.forcing.stochastic.wavenumbers = {8};
  expectThrows<std::invalid_argument>(
    [&nyquist]() {
      const burgers::BurgersSolver invalid(nyquist);
      static_cast<void>(invalid);
    },
    "a stochastic Nyquist mode must be rejected");

  burgers::RunConfig invalid_tau = compositeConfig();
  invalid_tau.forcing.stochastic.correlation_time = 0.0;
  expectThrows<std::invalid_argument>(
    [&invalid_tau]() {
      const burgers::BurgersSolver invalid(invalid_tau);
      static_cast<void>(invalid);
    },
    "nonpositive stochastic correlation time must be rejected");

  burgers::RunConfig invalid_mean_policy = compositeConfig();
  invalid_mean_policy.forcing.remove_discrete_mean = false;
  expectThrows<std::invalid_argument>(
    [&invalid_mean_policy]() {
      const burgers::BurgersSolver invalid(invalid_mean_policy);
      static_cast<void>(invalid);
    },
    "production forcing must require defensive mean removal");
}

}  // namespace

int main() {
  checkMeanConservationForMethodMatrix();
  checkClockBoundAndEnergyBudget();
  checkSolverRestart();
  checkConfigurationFailures();

  if(failures != 0) {
    std::cerr << failures << " Phase 5 solver check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 5 solver checks passed\n";
  return 0;
}
