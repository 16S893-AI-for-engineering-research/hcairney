#pragma once

#include "burgers/Config.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

namespace burgers {

// Fill exact finite-volume cell averages for
//   u(x,t) = A exp(-decay_rate * t) sin(wavenumber * x + phase).
void computeManufacturedSolutionCellAverages(
  const Grid& grid,
  const ManufacturedForcingConfig& config,
  double time,
  State& solution);

// Fill exact finite-volume cell averages of the source that makes the
// manufactured solution satisfy viscous Burgers:
//   u_t + u u_x = molecular_viscosity * u_xx + source.
void computeManufacturedSourceCellAverages(
  const Grid& grid,
  const ManufacturedForcingConfig& config,
  double molecular_viscosity,
  double time,
  State& source);

}  // namespace burgers
