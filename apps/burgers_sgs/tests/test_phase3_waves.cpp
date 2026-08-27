#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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

burgers::RunConfig waveConfig(std::size_t cell_count,
                              double molecular_viscosity) {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = cell_count;
  config.viscosity.molecular = molecular_viscosity;
  config.time_integration.advective_cfl = 0.25;
  config.time_integration.diffusive_cfl = 0.15;
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
    integral += 2.0 * overlap(left,
                              right,
                              rarefaction_right,
                              shock);
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

void checkInviscidWaveConvergence() {
  const std::array<std::size_t, 3> cell_counts{{128, 256, 512}};
  std::array<double, 3> l1_errors{{0.0, 0.0, 0.0}};
  std::array<double, 3> location_errors{{0.0, 0.0, 0.0}};
  const double final_time = 0.08;
  const double expected_shock = pi + final_time;

  for(std::size_t level = 0; level < cell_counts.size(); ++level) {
    const burgers::RunConfig config = waveConfig(cell_counts[level], 0.0);
    const burgers::BurgersSolver solver(config);
    burgers::State numerical(solver.grid());
    burgers::periodicTwoStateInitialCondition(2.0, 0.0)(
      solver.grid(), numerical);
    const double initial_mean = burgers::mean(solver.grid(), numerical);
    const burgers::AdvanceResult advance =
      solver.advanceTo(numerical, 0.0, final_time);

    burgers::State exact(solver.grid());
    exactInviscidWave(solver.grid(), final_time, exact);
    l1_errors[level] =
      burgers::errorNorms(solver.grid(), numerical, exact).l1;
    const double detected_shock =
      thresholdShockLocation(solver.grid(), numerical);
    expect(std::isfinite(detected_shock) != 0,
           "the discontinuous shock must remain detectable");
    location_errors[level] = std::abs(detected_shock - expected_shock);
    expect(location_errors[level] <= 3.0 * solver.grid().cellWidth(),
           "threshold shock location must converge within three cells");

    const double mass_location =
      conservativeShockLocation(solver.grid(), numerical);
    const double mean_tolerance =
      256.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(solver.grid().cellCount()) *
      static_cast<double>(std::max<std::size_t>(1, advance.timestep_count));
    expectNear(mass_location,
               expected_shock,
               mean_tolerance,
               "conservative shock motion must have Rankine-Hugoniot speed");
    expectNear(burgers::mean(solver.grid(), numerical),
               initial_mean,
               mean_tolerance,
               "shock/rarefaction evolution must conserve periodic mean");

    const std::size_t fan_cells = static_cast<std::size_t>(
      std::floor(2.0 * final_time / solver.grid().cellWidth()));
    for(std::size_t cell = 0;
        cell + 1 < fan_cells && cell + 1 < numerical.size(); ++cell) {
      expect(numerical[cell + 1] + 2.0e-13 >= numerical[cell],
             "the rarefaction fan must not contain an expansion shock");
    }
    expect(numerical[0] > 0.0 && numerical[0] < 2.0,
           "the periodic expansion must form an intermediate fan state");

    std::cout << "wave N=" << cell_counts[level]
              << " L1=" << l1_errors[level]
              << " shock_error=" << location_errors[level] << '\n';
    if(level != 0) {
      expect(l1_errors[level] < l1_errors[level - 1],
             "inviscid wave L1 error must decrease under refinement");
    }
  }

  expect(location_errors.back() < location_errors.front(),
         "fine-grid shock-location error must improve over the coarse grid");
  const double final_l1_order =
    std::log(l1_errors[1] / l1_errors[2]) / std::log(2.0);
  std::cout << "wave final L1 order: " << final_l1_order << '\n';
  expect(final_l1_order >= 0.35,
         "shock/rarefaction integral error must show convergence");
}

burgers::State restrictCellAverages(const burgers::Grid& coarse_grid,
                                    const burgers::State& fine_state) {
  if(fine_state.size() % coarse_grid.cellCount() != 0) {
    throw std::invalid_argument(
      "fine reference size must be divisible by coarse cell count");
  }
  const std::size_t ratio = fine_state.size() / coarse_grid.cellCount();
  burgers::State restricted(coarse_grid);
  for(std::size_t coarse = 0; coarse < coarse_grid.cellCount(); ++coarse) {
    double sum = 0.0;
    for(std::size_t offset = 0; offset < ratio; ++offset) {
      sum += fine_state[coarse * ratio + offset];
    }
    restricted[coarse] = sum / static_cast<double>(ratio);
  }
  return restricted;
}

void checkViscousFrontRefinement() {
  const double viscosity = 0.05;
  const double final_time = 0.08;
  const burgers::RunConfig reference_config = waveConfig(1024, viscosity);
  const burgers::BurgersSolver reference_solver(reference_config);
  burgers::State fine_reference(reference_solver.grid());
  burgers::periodicTwoStateInitialCondition(2.0, 0.0)(
    reference_solver.grid(), fine_reference);
  reference_solver.advanceTo(fine_reference, 0.0, final_time);

  const std::array<std::size_t, 3> cell_counts{{128, 256, 512}};
  std::array<double, 3> errors{{0.0, 0.0, 0.0}};
  for(std::size_t level = 0; level < cell_counts.size(); ++level) {
    const burgers::RunConfig config =
      waveConfig(cell_counts[level], viscosity);
    const burgers::BurgersSolver solver(config);
    burgers::State state(solver.grid());
    burgers::periodicTwoStateInitialCondition(2.0, 0.0)(
      solver.grid(), state);
    solver.advanceTo(state, 0.0, final_time);
    const burgers::State restricted_reference =
      restrictCellAverages(solver.grid(), fine_reference);
    errors[level] =
      burgers::errorNorms(solver.grid(), state, restricted_reference).l1;
    std::cout << "viscous front N=" << cell_counts[level]
              << " reference L1=" << errors[level] << '\n';
    if(level != 0) {
      expect(errors[level] < errors[level - 1],
             "resolved viscous-front error must decrease under refinement");
    }
  }
}

}  // namespace

int main() {
  checkInviscidWaveConvergence();
  checkViscousFrontRefinement();

  if(failures != 0) {
    std::cerr << failures << " Phase 3 wave check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 3 wave checks passed\n";
  return 0;
}
