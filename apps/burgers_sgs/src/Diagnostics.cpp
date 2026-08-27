#include "burgers/Diagnostics.h"

#include <algorithm>
#include <cmath>
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

}  // namespace burgers
