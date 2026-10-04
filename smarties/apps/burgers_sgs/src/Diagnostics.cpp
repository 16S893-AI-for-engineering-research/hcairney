#include "burgers/Diagnostics.h"

#include "burgers/ViscousFlux.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace burgers {
namespace {

void requireCompatible(const Grid& grid, const State& state) {
  if(grid.cellCount() != state.size()) {
    throw std::invalid_argument(
      "grid and state must have the same number of cells");
  }
}

template<typename Transform>
double compensatedSum(const State& state, Transform transform) {
  double sum = 0.0;
  double correction = 0.0;
  for(const double value : state) {
    const double term = transform(value) - correction;
    const double updated = sum + term;
    correction = (updated - sum) - term;
    sum = updated;
  }
  return sum;
}

}  // namespace

double integral(const Grid& grid, const State& state) {
  requireCompatible(grid, state);
  return grid.cellWidth() *
         compensatedSum(state, [](double value) { return value; });
}

double mean(const Grid& grid, const State& state) {
  return integral(grid, state) / grid.length();
}

double l1Norm(const Grid& grid, const State& state) {
  requireCompatible(grid, state);
  const double absolute_sum = compensatedSum(
    state, [](double value) { return std::abs(value); });
  return grid.cellWidth() * absolute_sum / grid.length();
}

double l2Norm(const Grid& grid, const State& state) {
  requireCompatible(grid, state);
  const double square_sum = compensatedSum(
    state, [](double value) { return value * value; });
  return std::sqrt(grid.cellWidth() * square_sum / grid.length());
}

double lInfinityNorm(const Grid& grid, const State& state) {
  requireCompatible(grid, state);
  double maximum = 0.0;
  for(const double value : state) {
    maximum = std::max(maximum, std::abs(value));
  }
  return maximum;
}

double kineticEnergy(const Grid& grid, const State& state) {
  const double norm = l2Norm(grid, state);
  return 0.5 * grid.length() * norm * norm;
}

double spatialVariance(const Grid& grid, const State& state) {
  const double state_mean = mean(grid, state);
  return 2.0 * kineticEnergy(grid, state) / grid.length() -
         state_mean * state_mean;
}

std::vector<double> energySpectrum(const Grid& grid, const State& state) {
  requireCompatible(grid, state);
  const std::size_t maximum_mode = grid.cellCount() / 2;
  std::vector<double> spectrum(maximum_mode + 1, 0.0);
  const double two_pi = 6.283185307179586476925286766559;
  const double inverse_count = 1.0 / static_cast<double>(grid.cellCount());

  for(std::size_t mode = 0; mode <= maximum_mode; ++mode) {
    double real_part = 0.0;
    double imaginary_part = 0.0;
    for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
      const double angle = two_pi * static_cast<double>(mode) *
                           static_cast<double>(cell) * inverse_count;
      real_part += state[cell] * std::cos(angle);
      imaginary_part -= state[cell] * std::sin(angle);
    }
    real_part *= inverse_count;
    imaginary_part *= inverse_count;
    const double coefficient_square =
      real_part * real_part + imaginary_part * imaginary_part;
    const bool self_conjugate = mode == 0 ||
      (grid.cellCount() % 2 == 0 && mode == maximum_mode);
    spectrum[mode] =
      (self_conjugate ? 0.5 : 1.0) * grid.length() * coefficient_square;
  }
  return spectrum;
}

double molecularDissipation(const Grid& grid,
                            const State& state,
                            double molecular_viscosity) {
  requireCompatible(grid, state);
  if(std::isfinite(molecular_viscosity) == 0 ||
     molecular_viscosity < 0.0) {
    throw std::invalid_argument(
      "molecular viscosity must be finite and nonnegative");
  }
  if(molecular_viscosity == 0.0) {
    return 0.0;
  }

  double gradient_square_sum = 0.0;
  double correction = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t right = grid.neighbor(cell, 1);
    const double gradient =
      (state[right] - state[cell]) / grid.cellWidth();
    const double term = gradient * gradient - correction;
    const double updated = gradient_square_sum + term;
    correction = (updated - gradient_square_sum) - term;
    gradient_square_sum = updated;
  }
  return molecular_viscosity * grid.cellWidth() * gradient_square_sum;
}

double sgsDissipation(const Grid& grid,
                      const State& state,
                      const State& eddy_viscosity,
                      FaceViscosityAveraging averaging) {
  requireCompatible(grid, state);
  requireCompatible(grid, eddy_viscosity);
  std::vector<double> face_effective(grid.cellCount(), 0.0);
  interpolateEffectiveViscosityToFaces(
    grid, 0.0, eddy_viscosity, averaging, face_effective);

  double dissipation = 0.0;
  double correction = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t right = grid.neighbor(cell, 1);
    const double gradient =
      (state[right] - state[cell]) / grid.cellWidth();
    const double term =
      face_effective[cell] * gradient * gradient - correction;
    const double updated = dissipation + term;
    correction = (updated - dissipation) - term;
    dissipation = updated;
  }
  return grid.cellWidth() * dissipation;
}

ClosureStatistics closureStatistics(const Grid& grid,
                                    const ClosureFields& fields) {
  requireCompatible(grid, fields.coefficient_squared);
  requireCompatible(grid, fields.eddy_viscosity);
  ClosureStatistics statistics;
  statistics.minimum_coefficient =
    std::numeric_limits<double>::infinity();
  double coefficient_sum = 0.0;
  double viscosity_sum = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double squared = fields.coefficient_squared[cell];
    const double viscosity = fields.eddy_viscosity[cell];
    if(std::isfinite(squared) == 0 || squared < 0.0 ||
       std::isfinite(viscosity) == 0 || viscosity < 0.0) {
      throw std::invalid_argument(
        "closure fields must contain finite nonnegative values");
    }
    const double coefficient = std::sqrt(squared);
    statistics.minimum_coefficient =
      std::min(statistics.minimum_coefficient, coefficient);
    statistics.maximum_coefficient =
      std::max(statistics.maximum_coefficient, coefficient);
    statistics.maximum_eddy_viscosity =
      std::max(statistics.maximum_eddy_viscosity, viscosity);
    coefficient_sum += coefficient;
    viscosity_sum += viscosity;
  }
  const double inverse_count = 1.0 / static_cast<double>(grid.cellCount());
  statistics.mean_coefficient = coefficient_sum * inverse_count;
  statistics.mean_eddy_viscosity = viscosity_sum * inverse_count;
  return statistics;
}

double powerInput(const Grid& grid,
                  const State& state,
                  const State& forcing) {
  requireCompatible(grid, state);
  requireCompatible(grid, forcing);
  double sum = 0.0;
  double correction = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double term = state[cell] * forcing[cell] - correction;
    const double updated = sum + term;
    correction = (updated - sum) - term;
    sum = updated;
  }
  return grid.cellWidth() * sum;
}

ForcingPower forcingPower(const Grid& grid,
                          const State& state,
                          const ForcingFields& forcing) {
  ForcingPower power;
  power.manufactured = powerInput(grid, state, forcing.manufactured);
  power.deterministic = powerInput(grid, state, forcing.deterministic);
  power.stochastic = powerInput(grid, state, forcing.stochastic);
  power.prescribed = powerInput(grid, state, forcing.prescribed);
  power.total = powerInput(grid, state, forcing.total);
  return power;
}

ForcedEnergyBudgetRate forcedEnergyBudgetRate(
  const Grid& grid,
  const State& state,
  const State& derivative,
  const ForcingFields& forcing,
  double molecular_viscosity) {
  requireCompatible(grid, state);
  requireCompatible(grid, derivative);
  ForcedEnergyBudgetRate budget;
  budget.energy_rate = powerInput(grid, state, derivative);
  budget.power = forcingPower(grid, state, forcing);
  budget.molecular_dissipation =
    molecularDissipation(grid, state, molecular_viscosity);
  budget.numerical_dissipation =
    budget.power.total - budget.molecular_dissipation - budget.energy_rate;
  return budget;
}

ForcedEnergyBudgetRate forcedEnergyBudgetRate(
  const Grid& grid,
  const State& state,
  const State& derivative,
  const ForcingFields& forcing,
  double molecular_viscosity,
  const State& eddy_viscosity,
  FaceViscosityAveraging averaging) {
  ForcedEnergyBudgetRate budget = forcedEnergyBudgetRate(
    grid, state, derivative, forcing, molecular_viscosity);
  budget.sgs_dissipation =
    sgsDissipation(grid, state, eddy_viscosity, averaging);
  budget.numerical_dissipation -= budget.sgs_dissipation;
  return budget;
}

ErrorNorms errorNorms(const Grid& grid,
                      const State& numerical,
                      const State& reference) {
  requireCompatible(grid, numerical);
  requireCompatible(grid, reference);

  double absolute_sum = 0.0;
  double absolute_correction = 0.0;
  double square_sum = 0.0;
  double square_correction = 0.0;
  double maximum = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double difference = numerical[cell] - reference[cell];
    const double absolute_term = std::abs(difference) - absolute_correction;
    const double updated_absolute = absolute_sum + absolute_term;
    absolute_correction =
      (updated_absolute - absolute_sum) - absolute_term;
    absolute_sum = updated_absolute;

    const double square_term =
      difference * difference - square_correction;
    const double updated_square = square_sum + square_term;
    square_correction = (updated_square - square_sum) - square_term;
    square_sum = updated_square;
    maximum = std::max(maximum, std::abs(difference));
  }

  ErrorNorms errors;
  errors.l1 = grid.cellWidth() * absolute_sum / grid.length();
  errors.l2 =
    std::sqrt(grid.cellWidth() * square_sum / grid.length());
  errors.l_infinity = maximum;
  return errors;
}

UnforcedEnergyBudgetRate unforcedEnergyBudgetRate(
  const Grid& grid,
  const State& state,
  const State& derivative,
  double molecular_viscosity) {
  requireCompatible(grid, state);
  requireCompatible(grid, derivative);

  double product_sum = 0.0;
  double correction = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double term = state[cell] * derivative[cell] - correction;
    const double updated = product_sum + term;
    correction = (updated - product_sum) - term;
    product_sum = updated;
  }

  UnforcedEnergyBudgetRate budget;
  budget.energy_rate = grid.cellWidth() * product_sum;
  budget.molecular_dissipation =
    molecularDissipation(grid, state, molecular_viscosity);
  budget.numerical_dissipation =
    -budget.energy_rate - budget.molecular_dissipation;
  return budget;
}

UnforcedEnergyBudgetRate unforcedEnergyBudgetRate(
  const Grid& grid,
  const State& state,
  const State& derivative,
  double molecular_viscosity,
  const State& eddy_viscosity,
  FaceViscosityAveraging averaging) {
  UnforcedEnergyBudgetRate budget = unforcedEnergyBudgetRate(
    grid, state, derivative, molecular_viscosity);
  budget.sgs_dissipation =
    sgsDissipation(grid, state, eddy_viscosity, averaging);
  budget.numerical_dissipation -= budget.sgs_dissipation;
  return budget;
}

}  // namespace burgers
