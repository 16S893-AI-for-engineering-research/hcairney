#include "Phase4Methods.h"

#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

const double pi = 3.1415926535897932384626433832795;
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

burgers::RunConfig methodConfig(const phase4_test::MethodCase& method,
                                std::size_t cell_count,
                                double viscosity) {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = cell_count;
  config.viscosity.molecular = viscosity;
  config.forcing.type = burgers::ForcingType::None;
  config.closure.type = burgers::ClosureType::NoClosure;
  config.time_integration.advective_cfl = 0.2;
  config.time_integration.diffusive_cfl = 0.1;
  phase4_test::selectMethod(config, method);
  return config;
}

double overlap(double left,
               double right,
               double interval_left,
               double interval_right) {
  return std::max(0.0,
                  std::min(right, interval_right) -
                    std::max(left, interval_left));
}

void exactInviscidWave(const burgers::Grid& grid,
                       double time,
                       burgers::State& exact) {
  const double rarefaction_right = 2.0 * time;
  const double shock = pi + time;
  const double width = grid.cellWidth();
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double left =
      grid.xBegin() + static_cast<double>(cell) * width;
    const double right = left + width;
    const double fan_left = std::max(left, 0.0);
    const double fan_right = std::min(right, rarefaction_right);
    double integral = 0.0;
    if(fan_right > fan_left) {
      integral +=
        (fan_right * fan_right - fan_left * fan_left) / (2.0 * time);
    }
    integral += 2.0 * overlap(left, right, rarefaction_right, shock);
    exact[cell] = integral / width;
  }
}

double thresholdShockLocation(const burgers::Grid& grid,
                              const burgers::State& state) {
  for(std::size_t cell = grid.cellCount() / 2;
      cell + 1 < grid.cellCount(); ++cell) {
    if(state[cell] >= 1.0 && state[cell + 1] < 1.0) {
      return grid.xBegin() +
             static_cast<double>(cell + 1) * grid.cellWidth();
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

double conservativeShockLocation(const burgers::Grid& grid,
                                 const burgers::State& state) {
  const std::size_t first = grid.cellCount() / 4;
  const std::size_t last = 3 * grid.cellCount() / 4;
  double mass = 0.0;
  for(std::size_t cell = first; cell < last; ++cell) {
    mass += state[cell] * grid.cellWidth();
  }
  return 0.5 * pi + 0.5 * mass;
}

void checkNonlinearWaves() {
  const std::array<std::size_t, 3> cell_counts{{128, 256, 512}};
  const double final_time = 0.08;
  const double expected_shock = pi + final_time;

  for(const phase4_test::MethodCase& method : phase4_test::methodCases()) {
    std::array<double, 3> l1_errors{{0.0, 0.0, 0.0}};
    for(std::size_t level = 0; level < cell_counts.size(); ++level) {
      const burgers::RunConfig config =
        methodConfig(method, cell_counts[level], 0.0);
      const burgers::BurgersSolver solver(config);
      burgers::State numerical(solver.grid());
      burgers::periodicTwoStateInitialCondition(2.0, 0.0)(
        solver.grid(), numerical);
      const double initial_mean = burgers::mean(solver.grid(), numerical);
      const burgers::AdvanceResult result =
        solver.advanceTo(numerical, 0.0, final_time);

      burgers::State exact(solver.grid());
      exactInviscidWave(solver.grid(), final_time, exact);
      l1_errors[level] =
        burgers::errorNorms(solver.grid(), numerical, exact).l1;
      const double detected_shock =
        thresholdShockLocation(solver.grid(), numerical);
      expect(std::isfinite(detected_shock) != 0,
             std::string(method.name) + " must retain a detectable shock");
      expect(std::abs(detected_shock - expected_shock) <=
               4.0 * solver.grid().cellWidth(),
             std::string(method.name) +
               " shock location must converge at the Rankine-Hugoniot speed");

      const double mean_tolerance =
        512.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(solver.grid().cellCount()) *
        static_cast<double>(
          std::max<std::size_t>(1, result.timestep_count));
      expectNear(burgers::mean(solver.grid(), numerical),
                 initial_mean,
                 mean_tolerance,
                 std::string(method.name) +
                   " wave evolution must conserve its mean");
      expectNear(conservativeShockLocation(solver.grid(), numerical),
                 expected_shock,
                 mean_tolerance,
                 std::string(method.name) +
                   " conservative shock location must move at the "
                   "Rankine-Hugoniot speed");

      const std::size_t fan_cells = static_cast<std::size_t>(
        std::floor(2.0 * final_time / solver.grid().cellWidth()));
      for(std::size_t cell = 0;
          cell + 1 < fan_cells && cell + 1 < numerical.size(); ++cell) {
        expect(numerical[cell + 1] + 5.0e-13 >= numerical[cell],
               std::string(method.name) +
                 " must not create an expansion shock");
      }
      for(const double value : numerical) {
        expect(value >= -5.0e-13 && value <= 2.0 + 5.0e-13,
               std::string(method.name) +
                 " must not create a new wave extremum");
      }

      std::cout << method.name << " wave N=" << cell_counts[level]
                << " L1=" << l1_errors[level]
                << " shock_error="
                << std::abs(detected_shock - expected_shock) << '\n';
    }
    expect(l1_errors[1] < l1_errors[0] &&
             l1_errors[2] < l1_errors[1],
           std::string(method.name) +
             " shock/rarefaction L1 error must decrease under refinement");
    const double final_order =
      std::log(l1_errors[1] / l1_errors[2]) / std::log(2.0);
    std::cout << method.name << " wave final_L1_order="
              << final_order << '\n';
    expect(final_order >= 0.3,
           std::string(method.name) +
             " shock/rarefaction integral error must converge");
  }
}

struct DissipationResult {
  double initial_energy = 0.0;
  double final_energy = 0.0;
  double energy_loss = 0.0;
  double integrated_numerical_dissipation = 0.0;
  double balance_residual = 0.0;
};

DissipationResult runDissipationStudy(
  const phase4_test::MethodCase& method,
  std::size_t cell_count) {
  const burgers::RunConfig config = methodConfig(method, cell_count, 0.0);
  const burgers::BurgersSolver solver(config);
  burgers::State state(solver.grid());
  burgers::sinusoidalInitialCondition(0.15, 0.6, 2, 0.19)(
    solver.grid(), state);

  const double final_time = 0.2;
  const double requested_time_step =
    0.01 * solver.grid().cellWidth();
  const std::size_t step_count = static_cast<std::size_t>(
    std::ceil(final_time / requested_time_step));
  const double time_step =
    final_time / static_cast<double>(step_count);
  DissipationResult result;
  result.initial_energy = burgers::kineticEnergy(solver.grid(), state);

  for(std::size_t step = 0; step < step_count; ++step) {
    burgers::State derivative_before(solver.grid());
    solver.rightHandSide(state, derivative_before);
    const burgers::UnforcedEnergyBudgetRate before =
      burgers::unforcedEnergyBudgetRate(
        solver.grid(), state, derivative_before, 0.0);

    solver.advanceSspRk3(
      state, static_cast<double>(step) * time_step, time_step);

    burgers::State derivative_after(solver.grid());
    solver.rightHandSide(state, derivative_after);
    const burgers::UnforcedEnergyBudgetRate after =
      burgers::unforcedEnergyBudgetRate(
        solver.grid(), state, derivative_after, 0.0);
    result.integrated_numerical_dissipation +=
      0.5 * time_step *
      (before.numerical_dissipation + after.numerical_dissipation);
  }

  result.final_energy = burgers::kineticEnergy(solver.grid(), state);
  result.energy_loss = result.initial_energy - result.final_energy;
  result.balance_residual = std::abs(
    result.final_energy - result.initial_energy +
    result.integrated_numerical_dissipation);
  return result;
}

void checkDissipationComparison() {
  const std::array<std::size_t, 3> cell_counts{{64, 128, 256}};
  std::array<std::vector<DissipationResult>, 3> results;
  for(std::vector<DissipationResult>& level_results : results) {
    level_results.reserve(phase4_test::methodCases().size());
  }

  std::cout << std::setprecision(12)
            << "method,cell_count,initial_energy,final_energy,energy_loss,"
               "integrated_numerical_dissipation,balance_residual\n";
  for(std::size_t level = 0; level < cell_counts.size(); ++level) {
    for(const phase4_test::MethodCase& method : phase4_test::methodCases()) {
      const DissipationResult result =
        runDissipationStudy(method, cell_counts[level]);
      results[level].push_back(result);
      std::cout << method.name << ',' << cell_counts[level] << ','
                << result.initial_energy << ',' << result.final_energy << ','
                << result.energy_loss << ','
                << result.integrated_numerical_dissipation << ','
                << result.balance_residual << '\n';

      expect(std::isfinite(result.integrated_numerical_dissipation) != 0,
             std::string(method.name) +
               " integrated numerical dissipation must be finite");
      expect(result.energy_loss >= -2.0e-12,
             std::string(method.name) +
               " must not increase energy in an unforced inviscid run");
      expect(result.integrated_numerical_dissipation >= -2.0e-10,
             std::string(method.name) +
               " integrated numerical dissipation must be nonnegative");
      expect(result.balance_residual <= 2.0e-6,
             std::string(method.name) +
               " time-discrete energy balance must close");
    }
  }

  for(std::size_t method = 0;
      method < phase4_test::methodCases().size(); ++method) {
    expect(results[1][method].energy_loss < results[0][method].energy_loss &&
             results[2][method].energy_loss < results[1][method].energy_loss,
           std::string(phase4_test::methodCases()[method].name) +
             " smooth-wave numerical dissipation must decrease under "
             "grid refinement");
  }

  const std::vector<DissipationResult>& comparison = results[1];
  double minimum_loss = std::numeric_limits<double>::infinity();
  double maximum_loss = 0.0;
  for(const DissipationResult& result : comparison) {
    minimum_loss = std::min(minimum_loss, result.energy_loss);
    maximum_loss = std::max(maximum_loss, result.energy_loss);
  }
  expect(maximum_loss > minimum_loss + 1.0e-8,
         "the Phase 4 matrix must resolve a nonzero dissipation spread");

  // The first two cases differ only by Riemann flux. Requiring distinct
  // measured losses guards against accidentally ignoring flux selection.
  expect(std::abs(comparison[0].energy_loss -
                  comparison[1].energy_loss) > 1.0e-9,
         "piecewise Godunov and Rusanov must produce distinct dissipation");

  // Within each flux group, the three MUSCL cases differ only by limiter.
  const double godunov_limiter_spread =
    std::max({comparison[2].energy_loss,
              comparison[3].energy_loss,
              comparison[4].energy_loss}) -
    std::min({comparison[2].energy_loss,
              comparison[3].energy_loss,
              comparison[4].energy_loss});
  const double rusanov_limiter_spread =
    std::max({comparison[5].energy_loss,
              comparison[6].energy_loss,
              comparison[7].energy_loss}) -
    std::min({comparison[5].energy_loss,
              comparison[6].energy_loss,
              comparison[7].energy_loss});
  expect(godunov_limiter_spread > 1.0e-10,
         "Godunov MUSCL limiters must have a measurable dissipation spread");
  expect(rusanov_limiter_spread > 1.0e-10,
         "Rusanov MUSCL limiters must have a measurable dissipation spread");
}

}  // namespace

int main() {
  checkNonlinearWaves();
  checkDissipationComparison();

  if(failures != 0) {
    std::cerr << failures << " Phase 4 wave/dissipation check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 4 wave and dissipation checks passed\n";
  return 0;
}
