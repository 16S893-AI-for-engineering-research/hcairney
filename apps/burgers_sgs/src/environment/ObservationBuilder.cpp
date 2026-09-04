#include "burgers/environment/ObservationBuilder.h"

#include "burgers/ClosureModel.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace burgers {
namespace environment {

ObservationBuilder::ObservationBuilder(const Grid& grid,
                                       double molecular_viscosity,
                                       std::size_t stencil_radius)
  : grid_(grid),
    molecular_viscosity_(molecular_viscosity),
    stencil_radius_(stencil_radius) {
  if(std::isfinite(molecular_viscosity_) == 0 ||
     molecular_viscosity_ <= 0.0) {
    throw std::invalid_argument(
      "observation molecular viscosity must be positive");
  }
  if(stencil_radius_ >
     (std::numeric_limits<std::size_t>::max() - 1u) / 2u) {
    throw std::invalid_argument("observation stencil radius is too large");
  }
  if(observationSize() > grid_.cellCount()) {
    throw std::invalid_argument(
      "observation stencil must not repeat cells on the periodic grid");
  }
}

std::size_t ObservationBuilder::stencilRadius() const noexcept {
  return stencil_radius_;
}

std::size_t ObservationBuilder::observationSize() const noexcept {
  return 2u * stencil_radius_ + 1u;
}

std::vector<double> ObservationBuilder::featureField(
  const State& velocity) const {
  State gradients(grid_, 0.0);
  computeCellCenteredGradients(grid_, velocity, gradients);
  std::vector<double> features(grid_.cellCount(), 0.0);
  const double width_squared = grid_.cellWidth() * grid_.cellWidth();
  for(std::size_t cell = 0; cell < features.size(); ++cell) {
    features[cell] =
      gradients[cell] * width_squared / molecular_viscosity_;
    if(std::isfinite(features[cell]) == 0) {
      throw std::runtime_error(
        "observation feature became non-finite at cell " +
        std::to_string(cell));
    }
  }
  return features;
}

std::vector<std::vector<double>> ObservationBuilder::build(
  const State& velocity) const {
  const std::vector<double> features = featureField(velocity);
  std::vector<std::vector<double>> observations(
    grid_.cellCount(), std::vector<double>(observationSize(), 0.0));
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    for(std::size_t entry = 0; entry < observationSize(); ++entry) {
      const std::ptrdiff_t offset =
        static_cast<std::ptrdiff_t>(entry) -
        static_cast<std::ptrdiff_t>(stencil_radius_);
      observations[cell][entry] = features[grid_.neighbor(cell, offset)];
    }
  }
  return observations;
}

}  // namespace environment
}  // namespace burgers
