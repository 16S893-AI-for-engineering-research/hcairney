#pragma once

#include "burgers/Grid.h"
#include "burgers/State.h"

#include <vector>

namespace burgers {

// Flux i is the molecular-viscous flux through the interface between cell i
// and its periodic right neighbor.
void computeMolecularViscousFluxes(const Grid& grid,
                                   const State& state,
                                   double molecular_viscosity,
                                   std::vector<double>& fluxes);

}  // namespace burgers
