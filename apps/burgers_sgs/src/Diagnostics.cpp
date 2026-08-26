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

}  // namespace burgers
