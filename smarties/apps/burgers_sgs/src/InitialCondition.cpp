#include "burgers/InitialCondition.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace burgers {
namespace {

void requireCompatible(const Grid& grid, const State& state) {
  if(grid.cellCount() != state.size()) {
    throw std::invalid_argument(
      "initial-condition state size does not match the grid");
  }
}

void requireFinite(double value, const char* name) {
  if(std::isfinite(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

}  // namespace

InitialCondition constantInitialCondition(double value) {
  requireFinite(value, "constant initial value");
  return [value](const Grid& grid, State& state) {
    requireCompatible(grid, state);
    state.fill(value);
  };
}

InitialCondition sinusoidalInitialCondition(double mean_value,
                                             double amplitude,
                                             int wavenumber,
                                             double phase) {
  requireFinite(mean_value, "sinusoidal initial mean");
  requireFinite(amplitude, "sinusoidal initial amplitude");
  requireFinite(phase, "sinusoidal initial phase");
  if(wavenumber == 0) {
    throw std::invalid_argument(
      "sinusoidal initial wavenumber must be nonzero");
  }

  return [mean_value, amplitude, wavenumber, phase](const Grid& grid,
                                                    State& state) {
    requireCompatible(grid, state);
    const double width = grid.cellWidth();
    const double wave_number = static_cast<double>(wavenumber);
    for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
      const double left = grid.xBegin() + static_cast<double>(cell) * width;
      const double right = left + width;
      const double sine_average =
        (std::cos(wave_number * left + phase) -
         std::cos(wave_number * right + phase)) /
        (wave_number * width);
      state[cell] = mean_value + amplitude * sine_average;
    }
  };
}

InitialCondition randomInitialCondition(double mean_value,
                                         double amplitude,
                                         std::uint64_t seed) {
  requireFinite(mean_value, "random initial mean");
  requireFinite(amplitude, "random initial amplitude");
  if(amplitude < 0.0) {
    throw std::invalid_argument("random initial amplitude must be nonnegative");
  }

  return [mean_value, amplitude, seed](const Grid& grid, State& state) {
    requireCompatible(grid, state);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for(double& value : state) {
      value = mean_value + amplitude * distribution(generator);
    }
  };
}

InitialCondition periodicTwoStateInitialCondition(double left_state,
                                                   double right_state) {
  requireFinite(left_state, "periodic two-state left value");
  requireFinite(right_state, "periodic two-state right value");

  return [left_state, right_state](const Grid& grid, State& state) {
    requireCompatible(grid, state);
    const double midpoint = grid.xBegin() + 0.5 * grid.length();
    const double width = grid.cellWidth();
    for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
      const double cell_left =
        grid.xBegin() + static_cast<double>(cell) * width;
      const double cell_right = cell_left + width;
      const double left_overlap =
        std::max(0.0, std::min(cell_right, midpoint) - cell_left);
      const double right_overlap = width - left_overlap;
      state[cell] =
        (left_state * left_overlap + right_state * right_overlap) / width;
    }
  };
}

InitialCondition makeInitialCondition(const InitialConditionConfig& config,
                                      std::uint64_t seed) {
  switch(config.type) {
    case InitialConditionType::Constant:
      return constantInitialCondition(config.mean);
    case InitialConditionType::Sinusoidal:
      return sinusoidalInitialCondition(config.mean,
                                         config.amplitude,
                                         config.wavenumber,
                                         config.phase);
    case InitialConditionType::Random:
      return randomInitialCondition(config.mean, config.amplitude, seed);
    case InitialConditionType::PeriodicTwoState:
      return periodicTwoStateInitialCondition(config.left_state,
                                               config.right_state);
  }
  throw std::invalid_argument("unknown initial-condition type");
}

void initializeState(const Grid& grid,
                     State& state,
                     const InitialConditionConfig& config,
                     std::uint64_t seed) {
  makeInitialCondition(config, seed)(grid, state);
}

}  // namespace burgers
