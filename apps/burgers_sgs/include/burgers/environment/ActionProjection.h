#pragma once

#include "burgers/Grid.h"

#include <vector>

namespace burgers {
namespace environment {

enum class ActionSmoothing {
  None,
  PeriodicThreePoint
};

struct ActionProjectionConfig {
  double forcing_scale = 0.5;
  double minimum_raw_action = -1.0;
  double maximum_raw_action = 1.0;
  ActionSmoothing smoothing = ActionSmoothing::None;
};

struct ActionStatistics {
  double mean = 0.0;
  double rms = 0.0;
  double maximum_absolute = 0.0;
};

struct ActionProjectionResult {
  std::vector<double> nondimensional;
  std::vector<double> applied;
  ActionStatistics raw_statistics;
  ActionStatistics applied_statistics;
  double applied_mean_residual = 0.0;
  double saturation_fraction = 0.0;
};

class ActionProjection {
public:
  ActionProjection(const Grid& grid, const ActionProjectionConfig& config);

  const ActionProjectionConfig& config() const noexcept;

  // Returns a new field and never mutates raw_actions. The nondimensional
  // result is smoothed as configured, has zero volume-weighted mean, and is
  // rescaled field-wide to have maximum absolute value at most one.
  ActionProjectionResult apply(
    const std::vector<double>& raw_actions) const;

private:
  ActionStatistics statistics(const std::vector<double>& values) const;
  double weightedMean(const std::vector<double>& values) const;

  Grid grid_;
  ActionProjectionConfig config_;
};

}  // namespace environment
}  // namespace burgers
