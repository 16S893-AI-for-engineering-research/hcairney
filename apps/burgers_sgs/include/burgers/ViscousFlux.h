#pragma once

#include "burgers/Config.h"
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

// Eddy viscosity is cell-centered. Face interpolation is kept here so the
// closure models and the future environment never need boundary logic.
void interpolateEffectiveViscosityToFaces(
  const Grid& grid,
  double molecular_viscosity,
  const State& eddy_viscosity,
  FaceViscosityAveraging averaging,
  std::vector<double>& face_viscosity);

void computeViscousFluxes(const Grid& grid,
                          const State& state,
                          double molecular_viscosity,
                          const State& eddy_viscosity,
                          FaceViscosityAveraging averaging,
                          std::vector<double>& fluxes);

}  // namespace burgers
