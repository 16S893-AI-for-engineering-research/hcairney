#pragma once

#include "burgers/Reconstruction.h"

#include <vector>

namespace burgers {

double burgersPhysicalFlux(double state);
double godunovFlux(double left_state, double right_state);

void computeGodunovFluxes(const FaceStates& face_states,
                          std::vector<double>& fluxes);

}  // namespace burgers
