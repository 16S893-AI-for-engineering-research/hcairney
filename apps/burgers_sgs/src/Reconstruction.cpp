#include "burgers/Reconstruction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace burgers {
namespace {

void requireCompatible(const Grid& grid,
                       const State& state,
                       const FaceStates& face_states,
                       const char* method) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(std::string(method) +
                                " reconstruction state size does not "
                                "match grid");
  }
  if(face_states.left.size() != grid.cellCount() ||
     face_states.right.size() != grid.cellCount()) {
    throw std::invalid_argument(std::string(method) +
                                " face-state size does not match grid");
  }
}

void requireFinite(double value, const char* name) {
  if(std::isfinite(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double minmod(double first, double second) {
  if(first == 0.0 || second == 0.0 ||
     std::signbit(first) != std::signbit(second)) {
    return 0.0;
  }
  return std::copysign(std::min(std::abs(first), std::abs(second)), first);
}

double minmod(double first, double second, double third) {
  return minmod(first, minmod(second, third));
}

double vanLeerSlope(double backward_difference,
                    double forward_difference) {
  if(std::signbit(backward_difference) !=
       std::signbit(forward_difference) ||
     backward_difference == 0.0 || forward_difference == 0.0) {
    return 0.0;
  }

  // This form of the harmonic mean avoids multiplying the two differences,
  // which can overflow for otherwise finite input.
  const double smaller =
    std::min(std::abs(backward_difference),
             std::abs(forward_difference));
  const double larger =
    std::max(std::abs(backward_difference),
             std::abs(forward_difference));
  const double magnitude =
    smaller * (2.0 / (1.0 + smaller / larger));
  return std::copysign(magnitude, backward_difference);
}

}  // namespace

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
  requireCompatible(grid, state, face_states, "piecewise-constant");

  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    face_states.left[cell] = state[cell];
    face_states.right[cell] = state[grid.neighbor(cell, 1)];
  }
}

double musclLimitedSlope(double backward_difference,
                         double forward_difference,
                         Limiter limiter) {
  requireFinite(backward_difference, "MUSCL backward difference");
  requireFinite(forward_difference, "MUSCL forward difference");

  double slope = 0.0;
  switch(limiter) {
    case Limiter::Minmod:
      slope = minmod(backward_difference, forward_difference);
      break;
    case Limiter::MonotonizedCentral: {
      const double centered =
        0.5 * (backward_difference + forward_difference);
      const double twice_backward = 2.0 * backward_difference;
      const double twice_forward = 2.0 * forward_difference;
      if(std::isfinite(centered) == 0 ||
         std::isfinite(twice_backward) == 0 ||
         std::isfinite(twice_forward) == 0) {
        throw std::overflow_error(
          "monotonized-central slope calculation overflowed");
      }
      slope = minmod(centered, twice_backward, twice_forward);
      break;
    }
    case Limiter::VanLeer:
      slope = vanLeerSlope(backward_difference, forward_difference);
      break;
    case Limiter::None:
      throw std::invalid_argument(
        "MUSCL reconstruction requires a slope limiter");
    default:
      throw std::invalid_argument("unknown MUSCL slope limiter");
  }

  if(std::isfinite(slope) == 0) {
    throw std::overflow_error("MUSCL limited slope is not finite");
  }
  return slope;
}

void computeMusclSlopes(const Grid& grid,
                        const State& state,
                        Limiter limiter,
                        std::vector<double>& slopes) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "MUSCL slope state size does not match grid");
  }
  if(slopes.size() != grid.cellCount()) {
    throw std::invalid_argument(
      "MUSCL slope array size does not match grid");
  }
  if(limiter == Limiter::None) {
    throw std::invalid_argument(
      "MUSCL reconstruction requires a slope limiter");
  }

  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double value = state[cell];
    const double backward = value - state[grid.neighbor(cell, -1)];
    const double forward = state[grid.neighbor(cell, 1)] - value;
    if(std::isfinite(backward) == 0 || std::isfinite(forward) == 0) {
      throw std::overflow_error(
        "MUSCL cell difference is not finite at cell " +
        std::to_string(cell));
    }
    slopes[cell] = musclLimitedSlope(backward, forward, limiter);
  }
}

void reconstructMuscl(const Grid& grid,
                      const State& state,
                      Limiter limiter,
                      FaceStates& face_states) {
  requireCompatible(grid, state, face_states, "MUSCL");
  std::vector<double> slopes(grid.cellCount(), 0.0);
  computeMusclSlopes(grid, state, limiter, slopes);

  for(std::size_t face = 0; face < grid.cellCount(); ++face) {
    const std::size_t right_cell = grid.neighbor(face, 1);
    face_states.left[face] = state[face] + 0.5 * slopes[face];
    face_states.right[face] =
      state[right_cell] - 0.5 * slopes[right_cell];
    if(std::isfinite(face_states.left[face]) == 0 ||
       std::isfinite(face_states.right[face]) == 0) {
      throw std::overflow_error(
        "MUSCL reconstruction produced a non-finite face state at face " +
        std::to_string(face));
    }
  }
}

void reconstruct(const Grid& grid,
                 const State& state,
                 Reconstruction reconstruction,
                 Limiter limiter,
                 FaceStates& face_states) {
  switch(reconstruction) {
    case Reconstruction::PiecewiseConstant:
      if(limiter != Limiter::None) {
        throw std::invalid_argument(
          "piecewise-constant reconstruction requires limiter none");
      }
      reconstructPiecewiseConstant(grid, state, face_states);
      return;
    case Reconstruction::Muscl:
      reconstructMuscl(grid, state, limiter, face_states);
      return;
  }
  throw std::invalid_argument("unknown reconstruction method");
}

}  // namespace burgers
