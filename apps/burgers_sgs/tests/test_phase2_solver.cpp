#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/State.h"

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

burgers::RunConfig phase2Config(std::size_t cells,
                                double molecular_viscosity) {
  burgers::RunConfig config = burgers::makeDefaultRunConfig();
  config.grid.cell_count = cells;
  config.viscosity.molecular = molecular_viscosity;
  config.time_integration.final_time = 0.1;
  return config;
}

void checkConstantPreservation(double molecular_viscosity) {
  burgers::RunConfig config = phase2Config(64, molecular_viscosity);
  const burgers::BurgersSolver solver(config);
  burgers::State state(solver.grid(), 1.25);
  const burgers::AdvanceResult result = solver.advanceTo(state, 0.0, 0.2);

  expect(result.timestep_count > 0,
         "constant-state advancement must take at least one timestep");
  expectNear(result.final_time, 0.2, 0.0,
             "constant-state advancement must reach target time exactly");
  for(const double value : state) {
    expectNear(value, 1.25, 2.0e-14,
               "constant state must be preserved through every RK stage");
  }
}

}  // namespace

int main() {
  const double pi = 3.1415926535897932384626433832795;

  checkConstantPreservation(0.0);
  checkConstantPreservation(0.1);

  burgers::RunConfig conservation_config = phase2Config(96, 0.03);
  const burgers::BurgersSolver conservation_solver(conservation_config);
  burgers::State random_state(conservation_solver.grid());
  burgers::randomInitialCondition(0.4, 0.7, 9182u)(
    conservation_solver.grid(), random_state);
  const double initial_mean =
    burgers::mean(conservation_solver.grid(), random_state);
  conservation_solver.advanceTo(random_state, 0.0, 0.25);
  expectNear(burgers::mean(conservation_solver.grid(), random_state),
             initial_mean,
             2.0e-13,
             "periodic conservative advancement must preserve random-state mean");

  burgers::RunConfig decay_config = phase2Config(128, 0.08);
  const burgers::BurgersSolver decay_solver(decay_config);
  burgers::State smooth_state(decay_solver.grid());
  burgers::sinusoidalInitialCondition(0.0, 0.5, 1, 0.0)(
    decay_solver.grid(), smooth_state);
  const double initial_smooth_mean =
    burgers::mean(decay_solver.grid(), smooth_state);
  const double initial_energy =
    burgers::kineticEnergy(decay_solver.grid(), smooth_state);
  const double initial_dissipation = burgers::molecularDissipation(
    decay_solver.grid(), smooth_state, decay_solver.molecularViscosity());
  decay_solver.advanceTo(smooth_state, 0.0, 0.2);
  const double final_energy =
    burgers::kineticEnergy(decay_solver.grid(), smooth_state);
  expect(initial_dissipation > 0.0,
         "a nonconstant smooth state must have positive molecular dissipation");
  expect(final_energy < initial_energy,
         "an unforced viscous smooth solution must lose kinetic energy");
  expectNear(burgers::mean(decay_solver.grid(), smooth_state),
             initial_smooth_mean,
             2.0e-13,
             "periodic conservative advancement must preserve smooth-state mean");

  burgers::RunConfig cfl_config = phase2Config(4, 0.5);
  cfl_config.grid.x_end = 4.0;
  const burgers::BurgersSolver cfl_solver(cfl_config);
  burgers::State cfl_state(
    burgers::State::Container{0.0, -2.0, 1.0, 0.5});
  expectNear(cfl_solver.stableTimeStep(cfl_state), 0.2, 1.0e-14,
             "CFL selection must choose the smaller advective limit");

  burgers::RunConfig diffusion_config = phase2Config(4, 2.0);
  diffusion_config.grid.x_end = 4.0;
  const burgers::BurgersSolver diffusion_solver(diffusion_config);
  burgers::State zero_state(diffusion_solver.grid(), 0.0);
  expectNear(diffusion_solver.stableTimeStep(zero_state), 0.1, 1.0e-14,
             "zero velocity must leave the diffusive timestep active");

  burgers::RunConfig static_config = phase2Config(4, 0.0);
  static_config.grid.x_end = 4.0;
  const burgers::BurgersSolver static_solver(static_config);
  burgers::State static_state(static_solver.grid(), 0.0);
  expect(std::isinf(static_solver.stableTimeStep(static_state)) != 0,
         "zero velocity and zero viscosity must have no finite CFL bound");
  const burgers::AdvanceResult static_advance =
    static_solver.advanceTo(static_state, 1.0, 1.5);
  expect(static_advance.timestep_count == 1,
         "an unconstrained static state must advance in one requested step");
  expectNear(static_advance.final_time, 1.5, 0.0,
             "unconstrained advancement must reach target time exactly");

  burgers::RunConfig shortened_config = phase2Config(4, 0.0);
  shortened_config.grid.x_end = 4.0;
  const burgers::BurgersSolver shortened_solver(shortened_config);
  burgers::State shortened_state(shortened_solver.grid(), 1.0);
  const burgers::AdvanceResult shortened_advance =
    shortened_solver.advanceTo(shortened_state, 0.0, 1.1);
  expect(shortened_advance.timestep_count == 3,
         "adaptive advancement must use the expected number of CFL steps");
  expect(shortened_advance.shortened_final_step_count == 1,
         "adaptive advancement must report a shortened final step");
  expectNear(shortened_advance.final_time, 1.1, 0.0,
             "a shortened final step must reach the requested time exactly");

  burgers::State nonfinite_state(static_solver.grid(), 0.0);
  nonfinite_state[2] = std::numeric_limits<double>::quiet_NaN();
  expectThrows<std::runtime_error>(
    [&static_solver, &nonfinite_state]() {
      static_cast<void>(static_solver.stableTimeStep(nonfinite_state));
    },
    "timestep selection must reject a non-finite state descriptively");

  burgers::RunConfig maximum_step_config = phase2Config(8, 0.0);
  maximum_step_config.grid.x_end = 8.0;
  maximum_step_config.time_integration.maximum_steps = 1;
  const burgers::BurgersSolver maximum_step_solver(maximum_step_config);
  burgers::State moving_state(maximum_step_solver.grid(), 1.0);
  expectThrows<std::runtime_error>(
    [&maximum_step_solver, &moving_state]() {
      static_cast<void>(
        maximum_step_solver.advanceTo(moving_state, 0.0, 1.0));
    },
    "advancement must report exhaustion of the configured timestep budget");

  burgers::RunConfig unsupported_config = phase2Config(32, 0.01);
  unsupported_config.numerical_method.reconstruction =
    burgers::Reconstruction::Muscl;
  unsupported_config.numerical_method.limiter = burgers::Limiter::Minmod;
  expectThrows<std::invalid_argument>(
    [&unsupported_config]() {
      const burgers::BurgersSolver unsupported(unsupported_config);
      static_cast<void>(unsupported);
    },
    "future reconstruction choices must not be silently ignored");

  burgers::RunConfig wave_config = phase2Config(256, 0.0);
  wave_config.time_integration.advective_cfl = 0.3;
  const burgers::BurgersSolver wave_solver(wave_config);
  burgers::State wave_state(wave_solver.grid());
  burgers::periodicTwoStateInitialCondition(2.0, 0.0)(
    wave_solver.grid(), wave_state);
  const double wave_initial_mean = burgers::mean(wave_solver.grid(), wave_state);
  const double wave_time = 0.04;
  wave_solver.advanceTo(wave_state, 0.0, wave_time);
  expectNear(burgers::mean(wave_solver.grid(), wave_state),
             wave_initial_mean,
             2.0e-13,
             "shock-rarefaction evolution must conserve the periodic mean");
  expect(wave_state[0] > 0.0 && wave_state[0] < 2.0,
         "the periodic expansion must form an intermediate rarefaction state");
  expect(wave_state[1] >= wave_state[0],
         "the rarefaction must not become an expansion shock");

  const double midpoint = pi;
  const double expected_shock = midpoint + wave_time;
  double detected_shock = midpoint;
  bool shock_found = false;
  for(std::size_t cell = wave_solver.grid().cellCount() / 2;
      cell + 1 < wave_solver.grid().cellCount(); ++cell) {
    if(wave_state[cell] >= 1.0 && wave_state[cell + 1] < 1.0) {
      detected_shock = 0.5 * (wave_solver.grid().cellCenter(cell) +
                              wave_solver.grid().cellCenter(cell + 1));
      shock_found = true;
      break;
    }
  }
  expect(shock_found, "the moving shock must remain detectable");
  expectNear(detected_shock,
             expected_shock,
             3.0 * wave_solver.grid().cellWidth(),
             "the shock must propagate at its Rankine-Hugoniot speed");

  if(failures != 0) {
    std::cerr << failures << " Phase 2 solver check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 2 solver checks passed\n";
  return 0;
}
