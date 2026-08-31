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

burgers::RunConfig baseConfig() {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = 32;
  config.viscosity.molecular = 0.01;
  config.forcing.type = burgers::ForcingType::None;
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

void expectField(const burgers::State& actual,
                 const std::vector<double>& expected,
                 double tolerance,
                 const std::string& message) {
  expect(actual.size() == expected.size(), message + " size");
  if(actual.size() != expected.size()) {
    return;
  }
  for(std::size_t cell = 0; cell < actual.size(); ++cell) {
    expectNear(actual[cell], expected[cell], tolerance,
               message + " at cell " + std::to_string(cell));
  }
}

void checkValidationPersistenceReplacementAndClearing() {
  burgers::BurgersSolver solver(baseConfig());
  const std::size_t cell_count = solver.grid().cellCount();
  const std::vector<double> first(cell_count, 0.25);
  solver.setPrescribedAdditiveForcingField(first);

  burgers::ForcingFields fields(solver.grid());
  solver.forcingFields(0.0, fields);
  expectField(fields.prescribed, first, 0.0,
              "prescribed field must retain its supplied values");
  expectField(fields.total, first, 0.0,
              "prescribed field must contribute to total forcing");

  expectThrows<std::invalid_argument>(
    [&solver, cell_count]() {
      solver.setPrescribedAdditiveForcingField(
        std::vector<double>(cell_count - 1, 0.0));
    },
    "prescribed forcing must reject a field with the wrong size");
  std::vector<double> nonfinite = first;
  nonfinite[cell_count / 2] =
    std::numeric_limits<double>::quiet_NaN();
  expectThrows<std::invalid_argument>(
    [&solver, &nonfinite]() {
      solver.setPrescribedAdditiveForcingField(nonfinite);
    },
    "prescribed forcing must reject NaN values");
  nonfinite[cell_count / 2] = std::numeric_limits<double>::infinity();
  expectThrows<std::invalid_argument>(
    [&solver, &nonfinite]() {
      solver.setPrescribedAdditiveForcingField(nonfinite);
    },
    "prescribed forcing must reject infinite values");
  solver.forcingFields(0.0, fields);
  expectField(fields.prescribed, first, 0.0,
              "a rejected update must leave the held field unchanged");

  std::vector<double> replacement(cell_count, 0.0);
  for(std::size_t cell = 0; cell < cell_count; ++cell) {
    replacement[cell] = cell % 2 == 0 ? 0.5 : -0.5;
  }
  solver.setPrescribedAdditiveForcingField(replacement);
  solver.forcingFields(1.0, fields);
  expectField(fields.prescribed, replacement, 0.0,
              "a complete update must replace the held field");

  burgers::State state(solver.grid(), 0.0);
  expectThrows<std::invalid_argument>(
    [&solver, &state]() {
      burgers::State derivative(solver.grid());
      solver.rightHandSide(state, derivative);
    },
    "active prescribed forcing must require explicit physical time");

  solver.clearPrescribedAdditiveForcingField();
  solver.forcingFields(2.0, fields);
  const std::vector<double> zero(cell_count, 0.0);
  expectField(fields.prescribed, zero, 0.0,
              "clearing must zero the prescribed component");
  expectField(fields.total, zero, 0.0,
              "clearing must remove prescribed forcing from the total");
  burgers::State derivative(solver.grid());
  solver.rightHandSide(state, derivative);
}

void checkZeroFieldEquivalence() {
  burgers::BurgersSolver unforced(baseConfig());
  burgers::BurgersSolver prescribed(baseConfig());
  prescribed.setPrescribedAdditiveForcingField(
    std::vector<double>(prescribed.grid().cellCount(), 0.0));

  burgers::State unforced_state(unforced.grid());
  burgers::sinusoidalInitialCondition(0.1, 0.3, 2, 0.17)(
    unforced.grid(), unforced_state);
  burgers::State prescribed_state = unforced_state;
  unforced.advanceTo(unforced_state, 0.0, 0.1);
  prescribed.advanceTo(prescribed_state, 0.0, 0.1);
  expectNear(maximumDifference(unforced_state, prescribed_state), 0.0, 0.0,
             "an installed zero field must reproduce the unforced solver");
}

void checkConstantManufacturedSource() {
  burgers::RunConfig config = baseConfig();
  config.grid.cell_count = 16;
  config.viscosity.molecular = 0.0;
  burgers::BurgersSolver solver(config);
  const double initial_value = 0.2;
  const double source = 0.4;
  const double final_time = 0.1;
  solver.setPrescribedAdditiveForcingField(
    std::vector<double>(solver.grid().cellCount(), source));
  burgers::State state(solver.grid(), initial_value);
  solver.advanceTo(state, 0.0, final_time);
  for(const double value : state) {
    expectNear(value, initial_value + source * final_time, 2.0e-15,
               "a uniform prescribed source must evolve a uniform state "
               "at the manufactured rate");
  }
}

void checkZeroMeanPreservationAndEnergyAccounting() {
  burgers::BurgersSolver solver(baseConfig());
  std::vector<double> forcing(solver.grid().cellCount(), 0.0);
  for(std::size_t cell = 0; cell < forcing.size(); ++cell) {
    forcing[cell] = cell % 2 == 0 ? 0.15 : -0.15;
  }
  solver.setPrescribedAdditiveForcingField(forcing);

  burgers::State state(solver.grid());
  for(std::size_t cell = 0; cell < state.size(); ++cell) {
    state[cell] = 0.3 + (cell % 2 == 0 ? 0.25 : -0.25);
  }
  const double initial_mean = burgers::mean(solver.grid(), state);
  const burgers::SspRk3StepBudget budget =
    solver.advanceSspRk3(state, 0.0, 0.01);
  const double reconstructed_energy_change =
    budget.deterministic_work + budget.stochastic_work +
    budget.prescribed_work + budget.manufactured_work -
    budget.molecular_dissipation - budget.sgs_dissipation -
    budget.numerical_dissipation;
  expectNear(reconstructed_energy_change, budget.energy_change, 2.0e-15,
             "prescribed work must close the SSP-RK3 energy budget");
  expect(std::isfinite(budget.prescribed_work) != 0,
         "prescribed work must remain finite");
  expect(std::abs(budget.prescribed_work) > 1.0e-6,
         "zero-mean prescribed forcing may perform nonzero work");

  const burgers::AdvanceResult advance =
    solver.advanceTo(state, 0.01, 0.2);
  const double scale = std::max(1.0, burgers::l1Norm(solver.grid(), state));
  const double tolerance =
    1024.0 * std::numeric_limits<double>::epsilon() * scale *
    static_cast<double>(solver.grid().cellCount()) *
    static_cast<double>(std::max<std::size_t>(1, advance.timestep_count));
  expect(std::abs(burgers::mean(solver.grid(), state) - initial_mean) <=
           tolerance,
         "zero-mean prescribed forcing must preserve periodic velocity "
         "mean");
}

void checkStochasticComposition() {
  burgers::RunConfig config = baseConfig();
  config.forcing.type = burgers::ForcingType::StochasticFourierOu;
  config.forcing.stochastic = burgers::makeLowModeOuForcingConfig();
  config.random.seed = 9482u;
  burgers::BurgersSolver solver(config);
  std::vector<double> prescribed(solver.grid().cellCount(), 0.0);
  for(std::size_t cell = 0; cell < prescribed.size(); ++cell) {
    prescribed[cell] = cell % 2 == 0 ? 0.05 : -0.05;
  }
  solver.setPrescribedAdditiveForcingField(prescribed);

  burgers::ForcingFields fields(solver.grid());
  solver.forcingFields(0.0, fields);
  for(std::size_t cell = 0; cell < solver.grid().cellCount(); ++cell) {
    expectNear(fields.total[cell],
               fields.stochastic[cell] + fields.prescribed[cell],
               2.0e-16,
               "stochastic and prescribed forcing must compose "
               "independently");
  }

  burgers::State state(solver.grid(), 0.0);
  const burgers::AdvanceResult advance =
    solver.advanceTo(state, 0.0, 0.06);
  expect(advance.forcing_clock_step_count == 1,
         "prescribed forcing must not interfere with the stochastic clock");
  expect(std::isfinite(advance.stochastic_work) != 0 &&
           std::isfinite(advance.prescribed_work) != 0,
         "combined stochastic and prescribed work must remain separate and "
         "finite");
}

void checkDeterministicOracleEquivalence() {
  burgers::RunConfig deterministic_config = baseConfig();
  deterministic_config.grid.cell_count = 48;
  deterministic_config.forcing.type =
    burgers::ForcingType::DeterministicFourier;
  deterministic_config.forcing.deterministic.modes = {
    {1, 0.1, 0.17}, {3, 0.025, -0.31}};
  burgers::BurgersSolver deterministic(deterministic_config);

  burgers::RunConfig oracle_config = deterministic_config;
  oracle_config.forcing.type = burgers::ForcingType::None;
  burgers::BurgersSolver oracle(oracle_config);
  burgers::ForcingFields analytic(deterministic.grid());
  deterministic.forcingFields(0.0, analytic);
  oracle.setPrescribedAdditiveForcingField(
    analytic.deterministic.cellAverages());

  burgers::State deterministic_state(deterministic.grid());
  burgers::sinusoidalInitialCondition(0.15, 0.2, 2, 0.11)(
    deterministic.grid(), deterministic_state);
  burgers::State oracle_state = deterministic_state;
  const burgers::AdvanceResult deterministic_result =
    deterministic.advanceTo(deterministic_state, 0.0, 0.12);
  const burgers::AdvanceResult oracle_result =
    oracle.advanceTo(oracle_state, 0.0, 0.12);
  expectNear(maximumDifference(deterministic_state, oracle_state),
             0.0, 2.0e-15,
             "analytic deterministic forcing injected through the "
             "prescribed path must reproduce its trajectory");
  expectNear(deterministic_result.deterministic_work,
             oracle_result.prescribed_work, 2.0e-15,
             "oracle prescribed work must reproduce deterministic work");
  expectNear(oracle_result.deterministic_work, 0.0, 0.0,
             "the oracle solver must keep configured deterministic forcing "
             "disabled");
}

}  // namespace

int main() {
  checkValidationPersistenceReplacementAndClearing();
  checkZeroFieldEquivalence();
  checkConstantManufacturedSource();
  checkZeroMeanPreservationAndEnergyAccounting();
  checkStochasticComposition();
  checkDeterministicOracleEquivalence();

  if(failures != 0) {
    std::cerr << failures << " Phase 8 prescribed-forcing check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 8 prescribed-forcing checks passed\n";
  return 0;
}
