#pragma once

#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstddef>
#include <vector>

namespace burgers {

// Face i is the interface between cell i and its periodic right neighbor.
struct FaceStates {
  explicit FaceStates(std::size_t face_count);

  std::size_t size() const noexcept;

  std::vector<double> left;
  std::vector<double> right;
};

void reconstructPiecewiseConstant(const Grid& grid,
                                  const State& state,
                                  FaceStates& face_states);

}  // namespace burgers
