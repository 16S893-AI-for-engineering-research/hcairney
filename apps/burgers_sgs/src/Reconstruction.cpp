#include "burgers/Reconstruction.h"

#include <stdexcept>

namespace burgers {

FaceStates::FaceStates(std::size_t face_count)
  : left(face_count, 0.0), right(face_count, 0.0) {
  if(face_count == 0) {
    throw std::invalid_argument("face-state count must be positive");
  }
}

std::size_t FaceStates::size() const noexcept {
  return left.size();
}

void reconstructPiecewiseConstant(const Grid& grid,
                                  const State& state,
                                  FaceStates& face_states) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "piecewise-constant reconstruction state size does not match grid");
  }
  if(face_states.left.size() != grid.cellCount() ||
     face_states.right.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "piecewise-constant face-state size does not match grid");
  }

  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    face_states.left[cell] = state[cell];
    face_states.right[cell] = state[grid.neighbor(cell, 1)];
  }
}

}  // namespace burgers
