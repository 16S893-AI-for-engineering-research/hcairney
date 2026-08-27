#include "burgers/ManufacturedSolution.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace burgers {
namespace {

void requireCompatible(const Grid& grid,
                       const State& state,
                       const char* quantity) {
  if(grid.cellCount() != state.size()) {
    throw std::invalid_argument(std::string("manufactured ") + quantity +
                                " state size does not match grid");
  }
}

void requireFinite(double value, const char* name) {
  if(std::isfinite(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

void validateInputs(const ManufacturedForcingConfig& config,
                    double time) {
  requireFinite(config.amplitude, "manufactured amplitude");
  requireFinite(config.phase, "manufactured phase");
  requireFinite(config.decay_rate, "manufactured decay rate");
  requireFinite(time, "manufactured time");
  if(config.wavenumber == 0) {
    throw std::invalid_argument("manufactured wavenumber must be nonzero");
  }
  if(config.decay_rate <= 0.0) {
    throw std::invalid_argument("manufactured decay rate must be positive");
  }
}

double sineCellAverage(double left,
                       double right,
                       double width,
                       double wavenumber,
                       double phase) {
  return (std::cos(wavenumber * left + phase) -
          std::cos(wavenumber * right + phase)) /
         (wavenumber * width);
}

double sineCosineCellAverage(double left,
                             double right,
                             double width,
                             double wavenumber,
                             double phase) {
  return (std::cos(2.0 * (wavenumber * left + phase)) -
          std::cos(2.0 * (wavenumber * right + phase))) /
         (4.0 * wavenumber * width);
}

}  // namespace

void computeManufacturedSolutionCellAverages(
  const Grid& grid,
  const ManufacturedForcingConfig& config,
  double time,
  State& solution) {
  requireCompatible(grid, solution, "solution");
  validateInputs(config, time);

  const double width = grid.cellWidth();
  const double wavenumber = static_cast<double>(config.wavenumber);
  const double time_factor = std::exp(-config.decay_rate * time);
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double left =
      grid.xBegin() + static_cast<double>(cell) * width;
    const double right = left + width;
    solution[cell] = config.amplitude * time_factor *
                     sineCellAverage(left,
                                     right,
                                     width,
                                     wavenumber,
                                     config.phase);
    if(std::isfinite(solution[cell]) == 0) {
      throw std::runtime_error(
        "manufactured solution produced a non-finite value at cell " +
        std::to_string(cell));
    }
  }
}

void computeManufacturedSourceCellAverages(
  const Grid& grid,
  const ManufacturedForcingConfig& config,
  double molecular_viscosity,
  double time,
  State& source) {
  requireCompatible(grid, source, "source");
  validateInputs(config, time);
  requireFinite(molecular_viscosity, "manufactured molecular viscosity");
  if(molecular_viscosity < 0.0) {
    throw std::invalid_argument(
      "manufactured molecular viscosity must be nonnegative");
  }

  const double width = grid.cellWidth();
  const double wavenumber = static_cast<double>(config.wavenumber);
  const double time_factor = std::exp(-config.decay_rate * time);
  const double linear_coefficient =
    -config.decay_rate + molecular_viscosity * wavenumber * wavenumber;
  const double nonlinear_coefficient =
    config.amplitude * config.amplitude * wavenumber *
    time_factor * time_factor;

  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double left =
      grid.xBegin() + static_cast<double>(cell) * width;
    const double right = left + width;
    const double solution_average = config.amplitude * time_factor *
      sineCellAverage(left,
                      right,
                      width,
                      wavenumber,
                      config.phase);
    const double nonlinear_average = sineCosineCellAverage(
      left, right, width, wavenumber, config.phase);
    source[cell] = linear_coefficient * solution_average +
                   nonlinear_coefficient * nonlinear_average;
    if(std::isfinite(source[cell]) == 0) {
      throw std::runtime_error(
        "manufactured source produced a non-finite value at cell " +
        std::to_string(cell));
    }
  }
}

}  // namespace burgers
