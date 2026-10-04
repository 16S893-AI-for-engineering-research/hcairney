#pragma once

#include "burgers/Config.h"
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

// Return the limited cell-to-cell difference used as the MUSCL slope in a
// cell. The inputs are u_i-u_{i-1} and u_{i+1}-u_i, respectively.
double musclLimitedSlope(double backward_difference,
                         double forward_difference,
                         Limiter limiter);

void computeMusclSlopes(const Grid& grid,
                        const State& state,
                        Limiter limiter,
                        std::vector<double>& slopes);

void reconstructMuscl(const Grid& grid,
                      const State& state,
                      Limiter limiter,
                      FaceStates& face_states);

void reconstruct(const Grid& grid,
                 const State& state,
                 Reconstruction reconstruction,
                 Limiter limiter,
                 FaceStates& face_states);

}  // namespace burgers
