#include "burgers/ViscousFlux.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace burgers {

namespace {

void requireNonnegativeFinite(double value, const char* name) {
  if(std::isfinite(value) == 0 || value < 0.0) {
    throw std::invalid_argument(
      std::string(name) + " must be finite and nonnegative");
  }
}

}  // namespace

void computeMolecularViscousFluxes(const Grid& grid,
                                   const State& state,
                                   double molecular_viscosity,
                                   std::vector<double>& fluxes) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "molecular-viscous flux state size does not match grid");
  }
  if(fluxes.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "molecular-viscous flux array size does not match grid");
  }
  requireNonnegativeFinite(molecular_viscosity, "molecular viscosity");
  if(molecular_viscosity == 0.0) {
    std::fill(fluxes.begin(), fluxes.end(), 0.0);
    return;
  }

  const double inverse_width = 1.0 / grid.cellWidth();
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t right = grid.neighbor(cell, 1);
    fluxes[cell] = molecular_viscosity *
                   (state[right] - state[cell]) * inverse_width;
  }
}

void interpolateEffectiveViscosityToFaces(
  const Grid& grid,
  double molecular_viscosity,
  const State& eddy_viscosity,
  FaceViscosityAveraging averaging,
  std::vector<double>& face_viscosity) {
  requireNonnegativeFinite(molecular_viscosity, "molecular viscosity");
  if(eddy_viscosity.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "eddy-viscosity field size does not match grid");
  }
  if(face_viscosity.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "face-viscosity array size does not match grid");
  }

  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t right = grid.neighbor(cell, 1);
    requireNonnegativeFinite(eddy_viscosity[cell], "eddy viscosity");
    requireNonnegativeFinite(eddy_viscosity[right], "eddy viscosity");
    const double left_effective = molecular_viscosity + eddy_viscosity[cell];
    const double right_effective = molecular_viscosity + eddy_viscosity[right];
    if(averaging == FaceViscosityAveraging::Arithmetic) {
      face_viscosity[cell] = 0.5 * (left_effective + right_effective);
    } else {
      face_viscosity[cell] =
        left_effective > 0.0 && right_effective > 0.0
        ? 2.0 / (1.0 / left_effective + 1.0 / right_effective)
        : 0.0;
    }
    if(std::isfinite(face_viscosity[cell]) == 0) {
      throw std::runtime_error(
        "face interpolation produced a non-finite effective viscosity");
    }
  }
}

void computeViscousFluxes(const Grid& grid,
                          const State& state,
                          double molecular_viscosity,
                          const State& eddy_viscosity,
                          FaceViscosityAveraging averaging,
                          std::vector<double>& fluxes) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "viscous flux state size does not match grid");
  }
  if(fluxes.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "viscous flux array size does not match grid");
  }
  std::vector<double> face_viscosity(grid.cellCount(), 0.0);
  interpolateEffectiveViscosityToFaces(
    grid, molecular_viscosity, eddy_viscosity, averaging, face_viscosity);
  const double inverse_width = 1.0 / grid.cellWidth();
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t right = grid.neighbor(cell, 1);
    fluxes[cell] = face_viscosity[cell] *
                   (state[right] - state[cell]) * inverse_width;
  }
}

}  // namespace burgers
