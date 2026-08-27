#include "burgers/ConvectiveFlux.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace burgers {
namespace {

void requireFinite(double value, const char* name) {
  if(std::isfinite(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

}  // namespace

double burgersPhysicalFlux(double state) {
  requireFinite(state, "Burgers flux state");
  const double flux = 0.5 * state * state;
  if(std::isfinite(flux) == 0) {
    throw std::overflow_error("Burgers physical flux is not finite");
  }
  return flux;
}

double godunovFlux(double left_state, double right_state) {
  requireFinite(left_state, "Godunov left state");
  requireFinite(right_state, "Godunov right state");

  if(left_state <= right_state) {
    if(left_state >= 0.0) {
      return burgersPhysicalFlux(left_state);
    }
    if(right_state <= 0.0) {
      return burgersPhysicalFlux(right_state);
    }
    return 0.0;
  }

  const double shock_speed = 0.5 * (left_state + right_state);
  return shock_speed >= 0.0 ? burgersPhysicalFlux(left_state)
                            : burgersPhysicalFlux(right_state);
}

void computeGodunovFluxes(const FaceStates& face_states,
                          std::vector<double>& fluxes) {
  if(face_states.left.empty() ||
     face_states.left.size() != face_states.right.size()) {
    throw std::invalid_argument(
      "Godunov left and right face-state arrays must have equal positive size");
  }
  if(fluxes.size() != face_states.size()) {
    throw std::invalid_argument(
      "Godunov flux array size does not match face-state size");
  }

  for(std::size_t face = 0; face < face_states.size(); ++face) {
    fluxes[face] =
      godunovFlux(face_states.left[face], face_states.right[face]);
  }
}

}  // namespace burgers
