#include "burgers/environment/ActionProjection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace burgers {
namespace environment {
namespace {

void requireFinite(double value, const char* name) {
  if(std::isfinite(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

}  // namespace

ActionProjection::ActionProjection(
  const Grid& grid, const ActionProjectionConfig& config)
  : grid_(grid), config_(config) {
  requireFinite(config_.forcing_scale, "forcing scale");
  requireFinite(config_.minimum_raw_action, "minimum raw action");
  requireFinite(config_.maximum_raw_action, "maximum raw action");
  if(config_.forcing_scale <= 0.0) {
    throw std::invalid_argument("forcing scale must be positive");
  }
  if(config_.maximum_raw_action <= config_.minimum_raw_action) {
    throw std::invalid_argument(
      "maximum raw action must exceed minimum raw action");
  }
}

const ActionProjectionConfig& ActionProjection::config() const noexcept {
  return config_;
}

double ActionProjection::weightedMean(
  const std::vector<double>& values) const {
  double weighted_sum = 0.0;
  double volume = 0.0;
  for(std::size_t cell = 0; cell < values.size(); ++cell) {
    weighted_sum += grid_.cellWidth(cell) * values[cell];
    volume += grid_.cellWidth(cell);
  }
  return weighted_sum / volume;
}

ActionStatistics ActionProjection::statistics(
  const std::vector<double>& values) const {
  ActionStatistics result;
  result.mean = weightedMean(values);
  double square_sum = 0.0;
  double volume = 0.0;
  for(std::size_t cell = 0; cell < values.size(); ++cell) {
    const double width = grid_.cellWidth(cell);
    square_sum += width * values[cell] * values[cell];
    volume += width;
    result.maximum_absolute =
      std::max(result.maximum_absolute, std::abs(values[cell]));
  }
  result.rms = std::sqrt(square_sum / volume);
  return result;
}

ActionProjectionResult ActionProjection::apply(
  const std::vector<double>& raw_actions) const {
  if(raw_actions.size() != grid_.cellCount()) {
    throw std::invalid_argument(
      "raw action field size does not match the environment grid");
  }
  for(std::size_t cell = 0; cell < raw_actions.size(); ++cell) {
    const double action = raw_actions[cell];
    if(std::isfinite(action) == 0) {
      throw std::invalid_argument(
        "raw action at cell " + std::to_string(cell) + " must be finite");
    }
    if(action < config_.minimum_raw_action ||
       action > config_.maximum_raw_action) {
      throw std::invalid_argument(
        "raw action at cell " + std::to_string(cell) +
        " is outside the configured bounds");
    }
  }

  ActionProjectionResult result;
  result.nondimensional = raw_actions;
  result.raw_statistics = statistics(raw_actions);

  if(config_.smoothing == ActionSmoothing::PeriodicThreePoint) {
    std::vector<double> smoothed(raw_actions.size(), 0.0);
    for(std::size_t cell = 0; cell < raw_actions.size(); ++cell) {
      smoothed[cell] =
        0.25 * (raw_actions[grid_.neighbor(cell, -1)] +
                2.0 * raw_actions[cell] +
                raw_actions[grid_.neighbor(cell, 1)]);
    }
    result.nondimensional.swap(smoothed);
  }

  const double first_mean = weightedMean(result.nondimensional);
  for(double& value : result.nondimensional) {
    value -= first_mean;
  }

  // Repeat mean removal after all configured filtering. This is redundant for
  // the three-point filter on the uniform grid but protects the invariant from
  // accumulated roundoff.
  const double residual_mean = weightedMean(result.nondimensional);
  for(double& value : result.nondimensional) {
    value -= residual_mean;
  }

  double maximum = 0.0;
  for(const double value : result.nondimensional) {
    maximum = std::max(maximum, std::abs(value));
  }
  if(maximum > 1.0) {
    for(double& value : result.nondimensional) {
      value /= maximum;
    }
  }

  result.applied.resize(result.nondimensional.size(), 0.0);
  std::size_t saturated = 0;
  const double saturation_tolerance =
    64.0 * std::numeric_limits<double>::epsilon() * config_.forcing_scale;
  for(std::size_t cell = 0; cell < result.nondimensional.size(); ++cell) {
    result.applied[cell] =
      config_.forcing_scale * result.nondimensional[cell];
    if(std::abs(result.applied[cell]) >=
       config_.forcing_scale - saturation_tolerance) {
      ++saturated;
    }
  }
  result.applied_statistics = statistics(result.applied);
  result.applied_mean_residual = weightedMean(result.applied);
  result.saturation_fraction = static_cast<double>(saturated) /
                               static_cast<double>(result.applied.size());
  return result;
}

}  // namespace environment
}  // namespace burgers
