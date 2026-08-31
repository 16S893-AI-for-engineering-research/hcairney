#include "burgers/BurgersSolver.h"
#include "burgers/ClosureModel.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/Grid.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"
#include "burgers/ViscousFlux.h"

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
  config.time_integration.advective_cfl = 0.2;
  config.time_integration.diffusive_cfl = 0.1;
  return config;
}

void checkCenteredGradientAndStaticClosure() {
  burgers::RunConfig config = baseConfig();
  config.closure.type = burgers::ClosureType::StaticSmagorinsky;
  config.closure.static_coefficient = 0.3;
  burgers::BurgersSolver solver(config);
  burgers::State velocity(solver.grid());
  burgers::sinusoidalInitialCondition(0.0, 0.4, 2, 0.17)(
    solver.grid(), velocity);

  burgers::State gradient(solver.grid());
  burgers::computeCellCenteredGradients(
    solver.grid(), velocity, gradient);
  burgers::ClosureFields fields(solver.grid());
  solver.closureFields(velocity, fields);
  const double coefficient_squared = 0.09;
  const double delta_squared =
    solver.grid().cellWidth() * solver.grid().cellWidth();
  for(std::size_t cell = 0; cell < solver.grid().cellCount(); ++cell) {
    const std::size_t left = solver.grid().neighbor(cell, -1);
    const std::size_t right = solver.grid().neighbor(cell, 1);
    const double expected_gradient =
      (velocity[right] - velocity[left]) /
      (2.0 * solver.grid().cellWidth());
    expectNear(gradient[cell], expected_gradient, 2.0e-15,
               "cell-centered closure gradient must use the periodic "
               "centered stencil");
    expectNear(fields.coefficient_squared[cell], coefficient_squared,
               2.0e-16,
               "static closure must store C_S squared internally");
    expectNear(fields.eddy_viscosity[cell],
               coefficient_squared * delta_squared *
                 std::abs(expected_gradient),
               2.0e-15,
               "static closure must evaluate local eddy viscosity");
  }

  const burgers::ClosureStatistics statistics =
    burgers::closureStatistics(solver.grid(), fields);
  expectNear(statistics.minimum_coefficient, 0.3, 2.0e-15,
             "reported minimum coefficient must use C_S");
  expectNear(statistics.mean_coefficient, 0.3, 2.0e-15,
             "reported mean coefficient must use C_S");
  expectNear(statistics.maximum_coefficient, 0.3, 2.0e-15,
             "reported maximum coefficient must use C_S");
}

void checkFaceAveraging() {
  burgers::GridConfig grid_config;
  grid_config.x_begin = 0.0;
  grid_config.x_end = 4.0;
  grid_config.cell_count = 4;
  const burgers::Grid grid(grid_config);
  const burgers::State eddy(std::vector<double>{0.0, 0.2, 0.4, 0.6});
  std::vector<double> faces(4, 0.0);
  burgers::interpolateEffectiveViscosityToFaces(
    grid, 0.1, eddy, burgers::FaceViscosityAveraging::Arithmetic, faces);
  expectNear(faces[0], 0.2, 2.0e-16,
             "arithmetic face viscosity must average adjacent effective "
             "cell viscosities");
  expectNear(faces[1], 0.4, 2.0e-16,
             "arithmetic face viscosity must include molecular viscosity");
  expectNear(faces[3], 0.4, 2.0e-16,
             "face averaging must wrap periodically");
}

void checkPrescribedField() {
  burgers::RunConfig config = baseConfig();
  config.closure.type = burgers::ClosureType::PrescribedCoefficientField;
  config.closure.minimum_coefficient = 0.0;
  config.closure.maximum_coefficient = 0.5;
  burgers::BurgersSolver solver(config);
  burgers::State velocity(solver.grid());
  burgers::sinusoidalInitialCondition(0.0, 0.2, 1, 0.0)(
    solver.grid(), velocity);

  expectThrows<std::runtime_error>(
    [&solver, &velocity]() {
      static_cast<void>(solver.stableTimeStep(velocity));
    },
    "prescribed closure must reject evaluation before a field is set");
  expectThrows<std::invalid_argument>(
    [&solver]() {
      solver.setPrescribedCoefficientField(std::vector<double>(3, 0.1));
    },
    "prescribed closure must reject a field with the wrong size");

  std::vector<double> coefficients(solver.grid().cellCount(), 0.2);
  coefficients[3] = 0.4;
  solver.setPrescribedCoefficientField(coefficients);
  burgers::ClosureFields fields(solver.grid());
  solver.closureFields(velocity, fields);
  expectNear(fields.coefficient_squared[0], 0.04, 2.0e-16,
             "prescribed C_S must be squared internally");
  expectNear(fields.coefficient_squared[3], 0.16, 2.0e-16,
             "prescribed field must retain local values");

  coefficients[5] = 0.6;
  expectThrows<std::invalid_argument>(
    [&solver, &coefficients]() {
      solver.setPrescribedCoefficientField(coefficients);
    },
    "prescribed closure must enforce configured C_S bounds");

  burgers::State derivative(solver.grid());
  solver.rightHandSide(velocity, derivative);
  expectNear(burgers::integral(solver.grid(), derivative), 0.0, 2.0e-14,
             "a spatially varying prescribed closure must remain "
             "conservative");
}

void checkTimestepAndEnergyBudget() {
  burgers::RunConfig static_config = baseConfig();
  static_config.viscosity.molecular = 0.0;
  static_config.closure.type = burgers::ClosureType::StaticSmagorinsky;
  static_config.closure.static_coefficient = 0.5;
  burgers::BurgersSolver solver(static_config);
  burgers::State state(solver.grid());
  burgers::sinusoidalInitialCondition(0.0, 0.8, 3, 0.1)(
    solver.grid(), state);

  burgers::ClosureFields fields(solver.grid());
  solver.closureFields(state, fields);
  double maximum_viscosity = 0.0;
  double maximum_speed = 0.0;
  for(std::size_t cell = 0; cell < state.size(); ++cell) {
    maximum_viscosity =
      std::max(maximum_viscosity, fields.eddy_viscosity[cell]);
    maximum_speed = std::max(maximum_speed, std::abs(state[cell]));
  }
  const double advective = static_config.time_integration.advective_cfl *
    solver.grid().cellWidth() / maximum_speed;
  const double diffusive = static_config.time_integration.diffusive_cfl *
    solver.grid().cellWidth() * solver.grid().cellWidth() /
    maximum_viscosity;
  expectNear(solver.stableTimeStep(state), std::min(advective, diffusive),
             2.0e-15,
             "SGS viscosity must participate in the diffusive CFL bound");

  burgers::State derivative(solver.grid());
  solver.rightHandSide(state, derivative);
  const burgers::UnforcedEnergyBudgetRate instantaneous =
    burgers::unforcedEnergyBudgetRate(
      solver.grid(), state, derivative, solver.molecularViscosity(),
      fields.eddy_viscosity, solver.faceViscosityAveraging());
  expect(instantaneous.sgs_dissipation > 0.0,
         "instantaneous energy diagnostics must expose SGS dissipation");
  expectNear(-instantaneous.energy_rate,
             instantaneous.molecular_dissipation +
               instantaneous.sgs_dissipation +
               instantaneous.numerical_dissipation,
             2.0e-14,
             "instantaneous unforced energy-budget terms must close");

  const double time_step = 0.1 * solver.stableTimeStep(state);
  const burgers::SspRk3StepBudget budget =
    solver.advanceSspRk3(state, time_step);
  expect(budget.sgs_dissipation > 0.0,
         "nonconstant static-closure state must have positive SGS "
         "dissipation");
  const double reconstructed_change =
    budget.deterministic_work + budget.stochastic_work +
    budget.manufactured_work - budget.molecular_dissipation -
    budget.sgs_dissipation - budget.numerical_dissipation;
  expectNear(reconstructed_change, budget.energy_change, 2.0e-15,
             "SSP-RK3 energy budget must close after separating SGS "
             "dissipation");
}

void checkConfigurationAndUnsupportedDynamicModel() {
  burgers::RunConfig dynamic = baseConfig();
  dynamic.closure.type = burgers::ClosureType::DynamicSmagorinsky;
  expectThrows<std::invalid_argument>(
    [&dynamic]() {
      const burgers::BurgersSolver solver(dynamic);
      static_cast<void>(solver);
    },
    "dynamic Smagorinsky must remain explicitly unsupported");

  burgers::RunConfig representation = baseConfig();
  representation.closure.store_squared_coefficient = false;
  expect(!burgers::validate(representation).empty(),
         "Phase 7 must require the documented internal C_S-squared "
         "representation");

  burgers::RunConfig no_closure = baseConfig();
  burgers::BurgersSolver solver(no_closure);
  expectThrows<std::logic_error>(
    [&solver]() {
      solver.setPrescribedCoefficientField(
        std::vector<double>(solver.grid().cellCount(), 0.1));
    },
    "only the prescribed closure may accept a coefficient field");
}

}  // namespace

int main() {
  checkCenteredGradientAndStaticClosure();
  checkFaceAveraging();
  checkPrescribedField();
  checkTimestepAndEnergyBudget();
  checkConfigurationAndUnsupportedDynamicModel();

  if(failures != 0) {
    std::cerr << failures << " Phase 7 closure check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 7 closure checks passed\n";
  return 0;
}
