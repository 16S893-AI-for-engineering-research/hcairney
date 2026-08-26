#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/Grid.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
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
              << ", got " << actual << ")\n";
    ++failures;
  }
}

template<typename Function>
void expectInvalidArgument(Function function, const std::string& message) {
  try {
    function();
  } catch(const std::invalid_argument&) {
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

}  // namespace

int main() {
  const double pi = 3.1415926535897932384626433832795;
  const double tolerance = 1.0e-14;
  const burgers::Grid grid(0.0, 2.0 * pi, 4);

  expect(grid.cellCount() == 4, "grid must expose its cell count");
  expectNear(grid.length(), 2.0 * pi, tolerance,
             "grid must expose the domain length");
  expectNear(grid.cellWidth(), 0.5 * pi, tolerance,
             "uniform grid spacing must be correct");
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    expectNear(grid.cellWidth(cell), 0.5 * pi, tolerance,
               "each cell width must equal the uniform spacing");
    expectNear(grid.cellCenter(cell),
               (static_cast<double>(cell) + 0.5) * 0.5 * pi,
               tolerance,
               "cell center must be located at the cell midpoint");
  }

  expect(grid.periodicIndex(-9) == 3,
         "periodic indexing must wrap multiple periods to the left");
  expect(grid.periodicIndex(-1) == 3,
         "periodic indexing must wrap one cell to the left");
  expect(grid.periodicIndex(4) == 0,
         "periodic indexing must wrap at the right boundary");
  expect(grid.periodicIndex(9) == 1,
         "periodic indexing must wrap multiple periods to the right");
  expect(grid.neighbor(0, -1) == 3,
         "neighbor lookup must use periodic indexing");
  expect(grid.neighbor(3, 1) == 0,
         "neighbor lookup must wrap the last cell");

  burgers::State constant(grid, 3.0);
  expect(constant.size() == grid.cellCount(),
         "state size must match its grid");
  expect(&constant[1] == constant.data() + 1,
         "state cell averages must be contiguous");
  expectNear(burgers::integral(grid, constant), 6.0 * pi, tolerance,
             "cell-average integral must include cell width");
  expectNear(burgers::mean(grid, constant), 3.0, tolerance,
             "constant-state mean must be preserved");
  expectNear(burgers::l1Norm(grid, constant), 3.0, tolerance,
             "normalized L1 norm of a constant must be its magnitude");
  expectNear(burgers::l2Norm(grid, constant), 3.0, tolerance,
             "normalized L2 norm of a constant must be its magnitude");
  expectNear(burgers::lInfinityNorm(grid, constant), 3.0, tolerance,
             "L-infinity norm of a constant must be its magnitude");

  burgers::State initialized(grid);
  burgers::constantInitialCondition(-2.0)(grid, initialized);
  for(const double value : initialized) {
    expectNear(value, -2.0, 0.0,
               "constant initialization must fill every cell exactly");
  }

  burgers::sinusoidalInitialCondition(0.0, 1.0, 1, 0.0)(grid, initialized);
  const double sine_average = 2.0 / pi;
  expectNear(initialized[0], sine_average, tolerance,
             "sinusoidal data must use exact first-cell averages");
  expectNear(initialized[1], sine_average, tolerance,
             "sinusoidal data must use exact second-cell averages");
  expectNear(initialized[2], -sine_average, tolerance,
             "sinusoidal data must use exact third-cell averages");
  expectNear(initialized[3], -sine_average, tolerance,
             "sinusoidal data must use exact fourth-cell averages");

  burgers::periodicTwoStateInitialCondition(1.5, -0.5)(grid, initialized);
  expectNear(initialized[0], 1.5, 0.0,
             "two-state data must use the left state in the first half");
  expectNear(initialized[1], 1.5, 0.0,
             "two-state data must use the left state before the midpoint");
  expectNear(initialized[2], -0.5, 0.0,
             "two-state data must use the right state after the midpoint");
  expectNear(initialized[3], -0.5, 0.0,
             "two-state data must use the right state in the second half");

  const burgers::Grid odd_grid(0.0, 2.0 * pi, 3);
  burgers::State odd_state(odd_grid);
  burgers::periodicTwoStateInitialCondition(1.0, -1.0)(odd_grid, odd_state);
  expectNear(odd_state[0], 1.0, tolerance,
             "odd-grid two-state data must preserve the left plateau");
  expectNear(odd_state[1], 0.0, tolerance,
             "a midpoint-cut cell must contain the exact mixed average");
  expectNear(odd_state[2], -1.0, tolerance,
             "odd-grid two-state data must preserve the right plateau");

  burgers::State random_a(grid);
  burgers::State random_b(grid);
  burgers::State random_c(grid);
  burgers::randomInitialCondition(2.0, 0.25, 1234u)(grid, random_a);
  burgers::randomInitialCondition(2.0, 0.25, 1234u)(grid, random_b);
  burgers::randomInitialCondition(2.0, 0.25, 1235u)(grid, random_c);
  bool different_seed_changed_value = false;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    expect(random_a[cell] == random_b[cell],
           "random initialization must be reproducible for a fixed seed");
    expect(random_a[cell] >= 1.75 && random_a[cell] <= 2.25,
           "uniform random initialization must respect its amplitude");
    different_seed_changed_value =
      different_seed_changed_value || random_a[cell] != random_c[cell];
  }
  expect(different_seed_changed_value,
         "changing the random seed must change the initialized state");

  burgers::InitialConditionConfig initial_config;
  initial_config.type = burgers::InitialConditionType::PeriodicTwoState;
  initial_config.left_state = 4.0;
  initial_config.right_state = -3.0;
  burgers::initializeState(grid, initialized, initial_config, 99u);
  expectNear(initialized[0], 4.0, 0.0,
             "configuration dispatch must select periodic two-state data");
  expectNear(initialized[3], -3.0, 0.0,
             "configuration dispatch must pass both two-state values");

  expectInvalidArgument(
    []() { const burgers::Grid invalid(0.0, 1.0, 0); },
    "a grid with no cells must be rejected");
  expectInvalidArgument(
    [&grid]() {
      burgers::State wrong_size(3);
      static_cast<void>(burgers::integral(grid, wrong_size));
    },
    "diagnostics must reject mismatched grid and state sizes");
  expectInvalidArgument(
    [&grid]() {
      burgers::State wrong_size(3);
      burgers::constantInitialCondition(1.0)(grid, wrong_size);
    },
    "initialization must reject mismatched grid and state sizes");

  if(failures != 0) {
    std::cerr << failures << " Phase 1 grid/state check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 1 grid/state checks passed\n";
  return 0;
}
