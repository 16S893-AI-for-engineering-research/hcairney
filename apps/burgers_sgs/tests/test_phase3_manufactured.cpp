#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/ManufacturedSolution.h"
#include "burgers/State.h"

#include <array>
#include <cmath>
#include <cstddef>
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

double convergenceOrder(double coarse_error, double fine_error) {
  if(coarse_error <= 0.0 || fine_error <= 0.0) {
    return 0.0;
  }
  return std::log(coarse_error / fine_error) / std::log(2.0);
}

burgers::RunConfig manufacturedConfig(std::size_t cell_count) {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = cell_count;
  config.viscosity.molecular = 0.05;
  config.forcing.type = burgers::ForcingType::Manufactured;
  config.forcing.manufactured.amplitude = 0.5;
  config.forcing.manufactured.wavenumber = 1;
  config.forcing.manufactured.phase = 0.2;
  config.forcing.manufactured.decay_rate = 1.0;
  config.time_integration.advective_cfl = 0.1;
  config.time_integration.diffusive_cfl = 0.1;
  return config;
}

void initializeExact(const burgers::BurgersSolver& solver,
                     const burgers::ManufacturedForcingConfig& manufactured,
                     double time,
                     burgers::State& state) {
  burgers::computeManufacturedSolutionCellAverages(
    solver.grid(), manufactured, time, state);
}

void advanceFixed(const burgers::BurgersSolver& solver,
                  burgers::State& state,
                  double initial_time,
                  double final_time,
                  std::size_t step_count) {
  const double time_step =
    (final_time - initial_time) / static_cast<double>(step_count);
  for(std::size_t step = 0; step < step_count; ++step) {
    const double time =
      initial_time + static_cast<double>(step) * time_step;
    solver.advanceSspRk3(state, time, time_step);
  }
}

void checkManufacturedSourceInterface() {
  burgers::RunConfig config = manufacturedConfig(64);
  const burgers::BurgersSolver solver(config);
  burgers::State exact(solver.grid());
  burgers::State source(solver.grid());
  initializeExact(solver, config.forcing.manufactured, 0.3, exact);
  burgers::computeManufacturedSourceCellAverages(
    solver.grid(),
    config.forcing.manufactured,
    config.viscosity.molecular,
    0.3,
    source);

  const double expected_amplitude =
    config.forcing.manufactured.amplitude *
    std::exp(-config.forcing.manufactured.decay_rate * 0.3);
  expectNear(burgers::lInfinityNorm(solver.grid(), exact),
             expected_amplitude,
             2.0 * expected_amplitude * solver.grid().cellWidth(),
             "manufactured exact cell averages must have the expected scale");
  expectNear(burgers::mean(solver.grid(), exact),
             0.0,
             2.0e-15,
             "manufactured exact solution must have zero periodic mean");
  expectNear(burgers::mean(solver.grid(), source),
             0.0,
             2.0e-15,
             "manufactured source must have zero periodic mean");

  burgers::State derivative(solver.grid());
  expectThrows<std::invalid_argument>(
    [&solver, &exact, &derivative]() {
      solver.rightHandSide(exact, derivative);
    },
    "manufactured forcing must reject a right-hand side without time");
  expectThrows<std::invalid_argument>(
    [&solver, &exact]() {
      burgers::State copy = exact;
      solver.advanceSspRk3(copy, 1.0e-3);
    },
    "manufactured forcing must reject an RK step without physical time");

  solver.rightHandSide(exact, 0.3, derivative);
  for(const double value : derivative) {
    expect(std::isfinite(value) != 0,
           "time-aware manufactured right-hand side must remain finite");
  }
}

void checkSpatialConvergence() {
  const std::array<std::size_t, 4> cell_counts{{32, 64, 128, 256}};
  std::array<burgers::ErrorNorms, 4> errors;
  const double initial_time = 0.0;
  const double final_time = 0.1;

  for(std::size_t level = 0; level < cell_counts.size(); ++level) {
    const burgers::RunConfig config =
      manufacturedConfig(cell_counts[level]);
    const burgers::BurgersSolver solver(config);
    burgers::State numerical(solver.grid());
    burgers::State exact(solver.grid());
    initializeExact(solver,
                    config.forcing.manufactured,
                    initial_time,
                    numerical);
    solver.advanceTo(numerical, initial_time, final_time);
    initializeExact(solver,
                    config.forcing.manufactured,
                    final_time,
                    exact);
    errors[level] = burgers::errorNorms(solver.grid(), numerical, exact);

    std::cout << "spatial N=" << cell_counts[level]
              << " L1=" << errors[level].l1
              << " L2=" << errors[level].l2
              << " Linf=" << errors[level].l_infinity << '\n';
    if(level != 0) {
      expect(errors[level].l1 < errors[level - 1].l1,
             "manufactured L1 error must decrease under grid refinement");
      expect(errors[level].l2 < errors[level - 1].l2,
             "manufactured L2 error must decrease under grid refinement");
      expect(errors[level].l_infinity < errors[level - 1].l_infinity,
             "manufactured Linf error must decrease under grid refinement");
    }
  }

  const double l1_order = convergenceOrder(errors[2].l1, errors[3].l1);
  const double l2_order = convergenceOrder(errors[2].l2, errors[3].l2);
  const double linfinity_order = convergenceOrder(
    errors[2].l_infinity, errors[3].l_infinity);
  std::cout << "spatial final orders: L1=" << l1_order
            << " L2=" << l2_order
            << " Linf=" << linfinity_order << '\n';
  expect(l1_order >= 0.8 && l1_order <= 1.2,
         "piecewise-constant Godunov L1 order must approach one");
  expect(l2_order >= 0.8 && l2_order <= 1.2,
         "piecewise-constant Godunov L2 order must approach one");
  expect(linfinity_order >= 0.7 && linfinity_order <= 1.3,
         "piecewise-constant Godunov Linf order must approach one");
}

void checkTemporalConvergence() {
  burgers::RunConfig config = manufacturedConfig(256);
  config.viscosity.molecular = 0.002;
  config.forcing.manufactured.amplitude = 0.8;
  config.forcing.manufactured.wavenumber = 2;
  config.forcing.manufactured.phase = 0.17;
  const burgers::BurgersSolver solver(config);
  const double initial_time = 0.0;
  const double final_time = 0.04;

  burgers::State reference(solver.grid());
  initializeExact(solver,
                  config.forcing.manufactured,
                  initial_time,
                  reference);
  advanceFixed(solver, reference, initial_time, final_time, 512);

  const std::array<std::size_t, 4> step_counts{{4, 8, 16, 32}};
  std::array<double, 4> errors{{0.0, 0.0, 0.0, 0.0}};
  for(std::size_t level = 0; level < step_counts.size(); ++level) {
    burgers::State numerical(solver.grid());
    initializeExact(solver,
                    config.forcing.manufactured,
                    initial_time,
                    numerical);
    advanceFixed(solver,
                 numerical,
                 initial_time,
                 final_time,
                 step_counts[level]);
    errors[level] =
      burgers::errorNorms(solver.grid(), numerical, reference).l2;
    std::cout << "temporal steps=" << step_counts[level]
              << " L2=" << errors[level] << '\n';
    if(level != 0) {
      expect(errors[level] < errors[level - 1],
             "fixed-step error must decrease under timestep refinement");
    }
  }

  for(std::size_t level = 0; level + 1 < step_counts.size(); ++level) {
    const double order = convergenceOrder(errors[level], errors[level + 1]);
    std::cout << "temporal order " << step_counts[level] << "->"
              << step_counts[level + 1] << ": " << order << '\n';
    expect(order >= 2.6 && order <= 3.4,
           "SSP-RK3 temporal order must approach three");
  }
}

}  // namespace

int main() {
  checkManufacturedSourceInterface();
  checkSpatialConvergence();
  checkTemporalConvergence();

  if(failures != 0) {
    std::cerr << failures << " Phase 3 manufactured check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 3 manufactured checks passed\n";
  return 0;
}
