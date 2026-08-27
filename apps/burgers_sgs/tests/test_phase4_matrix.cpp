#include "Phase4Methods.h"

#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/ManufacturedSolution.h"
#include "burgers/State.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if(!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

double convergenceOrder(double coarse_error, double fine_error) {
  if(coarse_error <= 0.0 || fine_error <= 0.0) {
    return 0.0;
  }
  return std::log(coarse_error / fine_error) / std::log(2.0);
}

burgers::RunConfig baseConfig(const phase4_test::MethodCase& method,
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

void checkConstantPreservation() {
  for(const phase4_test::MethodCase& method : phase4_test::methodCases()) {
    for(const double viscosity : {0.0, 0.04}) {
      const burgers::RunConfig config =
        baseConfig(method, 64, viscosity);
      const burgers::BurgersSolver solver(config);
      burgers::State state(solver.grid(), -0.75);
      solver.advanceTo(state, 0.0, 0.1);
      double maximum_error = 0.0;
      for(const double value : state) {
        maximum_error = std::max(maximum_error, std::abs(value + 0.75));
      }
      std::cout << method.name << " constant nu=" << viscosity
                << " Linf=" << maximum_error << '\n';
      expect(maximum_error <= 8.0e-15,
             std::string(method.name) +
               " must preserve constant data to roundoff");
    }
  }
}

void checkMeanConservationAndCfl() {
  for(const phase4_test::MethodCase& method : phase4_test::methodCases()) {
    const burgers::RunConfig config = baseConfig(method, 96, 0.025);
    const burgers::BurgersSolver solver(config);

    for(const bool use_random_state : {false, true}) {
      burgers::State state(solver.grid());
      if(use_random_state) {
        burgers::randomInitialCondition(0.3, 0.4, 9137u)(
          solver.grid(), state);
      } else {
        burgers::sinusoidalInitialCondition(0.3, 0.4, 2, 0.17)(
          solver.grid(), state);
      }
      const double initial_mean = burgers::mean(solver.grid(), state);
      const double initial_scale =
        std::max(1.0, burgers::l1Norm(solver.grid(), state));
      const burgers::AdvanceResult result =
        solver.advanceTo(state, 0.0, 0.08);
      const double drift =
        std::abs(burgers::mean(solver.grid(), state) - initial_mean);
      const double tolerance =
        512.0 * std::numeric_limits<double>::epsilon() * initial_scale *
        static_cast<double>(solver.grid().cellCount()) *
        static_cast<double>(
          std::max<std::size_t>(1, result.timestep_count));
      std::cout << method.name
                << (use_random_state ? " random" : " smooth")
                << " mean_drift=" << drift
                << " tolerance=" << tolerance << '\n';
      expect(drift <= tolerance,
             std::string(method.name) +
               " must conserve the periodic mean for " +
               (use_random_state ? "random" : "smooth") + " data");
    }

    burgers::State cfl_state(solver.grid());
    burgers::sinusoidalInitialCondition(0.2, 0.7, 1, 0.0)(
      solver.grid(), cfl_state);
    const double stable_step = solver.stableTimeStep(cfl_state);
    solver.advanceSspRk3(cfl_state, 0.0, 0.99 * stable_step);
    for(const double value : cfl_state) {
      expect(std::isfinite(value) != 0,
             std::string(method.name) +
               " must remain finite near its configured CFL limit");
    }
  }
}

burgers::ErrorNorms runManufactured(
  const phase4_test::MethodCase& method,
  std::size_t cell_count) {
  burgers::RunConfig config = baseConfig(method, cell_count, 0.04);
  config.forcing.type = burgers::ForcingType::Manufactured;
  config.forcing.manufactured.amplitude = 0.45;
  config.forcing.manufactured.wavenumber = 1;
  config.forcing.manufactured.phase = 0.23;
  config.forcing.manufactured.decay_rate = 1.0;
  config.time_integration.advective_cfl = 0.08;
  config.time_integration.diffusive_cfl = 0.08;

  const burgers::BurgersSolver solver(config);
  burgers::State numerical(solver.grid());
  burgers::State exact(solver.grid());
  burgers::computeManufacturedSolutionCellAverages(
    solver.grid(), config.forcing.manufactured, 0.0, numerical);
  solver.advanceTo(numerical, 0.0, 0.08);
  burgers::computeManufacturedSolutionCellAverages(
    solver.grid(), config.forcing.manufactured, 0.08, exact);
  return burgers::errorNorms(solver.grid(), numerical, exact);
}

void checkManufacturedSpatialConvergence() {
  const std::array<std::size_t, 3> cell_counts{{32, 64, 128}};
  for(const phase4_test::MethodCase& method : phase4_test::methodCases()) {
    std::array<burgers::ErrorNorms, 3> errors;
    for(std::size_t level = 0; level < cell_counts.size(); ++level) {
      errors[level] = runManufactured(method, cell_counts[level]);
      std::cout << method.name << " manufactured N=" << cell_counts[level]
                << " L1=" << errors[level].l1
                << " L2=" << errors[level].l2
                << " Linf=" << errors[level].l_infinity << '\n';
      if(level != 0) {
        expect(errors[level].l1 < errors[level - 1].l1,
               std::string(method.name) +
                 " manufactured L1 error must decrease");
        expect(errors[level].l2 < errors[level - 1].l2,
               std::string(method.name) +
                 " manufactured L2 error must decrease");
        expect(errors[level].l_infinity < errors[level - 1].l_infinity,
               std::string(method.name) +
                 " manufactured Linf error must decrease");
      }
    }

    const double l1_order =
      convergenceOrder(errors[1].l1, errors[2].l1);
    const double l2_order =
      convergenceOrder(errors[1].l2, errors[2].l2);
    const double linfinity_order = convergenceOrder(
      errors[1].l_infinity, errors[2].l_infinity);
    std::cout << method.name << " manufactured final_orders L1="
              << l1_order << " L2=" << l2_order
              << " Linf=" << linfinity_order << '\n';

    if(method.expected_spatial_order == 1) {
      expect(l1_order >= 0.75 && l1_order <= 1.35,
             std::string(method.name) + " L1 order must approach one");
      expect(l2_order >= 0.75 && l2_order <= 1.35,
             std::string(method.name) + " L2 order must approach one");
      expect(linfinity_order >= 0.65 && linfinity_order <= 1.45,
             std::string(method.name) + " Linf order must approach one");
    } else {
      expect(l1_order >= 1.6 && l1_order <= 2.4,
             std::string(method.name) + " L1 order must approach two");
      expect(l2_order >= 1.55 && l2_order <= 2.4,
             std::string(method.name) + " L2 order must approach two");
      expect(linfinity_order >= 1.35 && linfinity_order <= 2.5,
             std::string(method.name) + " Linf order must approach two");
    }
  }
}

}  // namespace

int main() {
  checkConstantPreservation();
  checkMeanConservationAndCfl();
  checkManufacturedSpatialConvergence();

  if(failures != 0) {
    std::cerr << failures << " Phase 4 matrix check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 4 matrix checks passed\n";
  return 0;
}
