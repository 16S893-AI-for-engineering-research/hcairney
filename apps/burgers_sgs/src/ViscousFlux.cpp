#include "burgers/ViscousFlux.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace burgers {

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
  if(std::isfinite(molecular_viscosity) == 0 ||
     molecular_viscosity < 0.0) {
    throw std::invalid_argument(
      "molecular viscosity must be finite and nonnegative");
  }
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

}  // namespace burgers
