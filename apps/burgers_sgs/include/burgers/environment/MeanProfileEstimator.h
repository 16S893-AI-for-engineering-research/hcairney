#pragma once

#include "burgers/Grid.h"
#include "burgers/State.h"

#include <vector>

namespace burgers {
namespace environment {

class MeanProfileEstimator {
public:
  MeanProfileEstimator(const Grid& grid, double time_scale);

  double timeScale() const noexcept;
  const std::vector<double>& estimate() const noexcept;

  void reset(const std::vector<double>& initial_estimate);

  // Block integration uses the trapezoidal rule over accepted solver states,
  // preserving the physical duration of unequal adaptive timesteps.
  void beginBlock(double time, const State& state);
  void observeAcceptedState(double time, const State& state);
  std::vector<double> completeBlock() const;
  double blockDuration() const;

  std::vector<double> previewUpdate(
    const std::vector<double>& block_mean,
    double duration) const;
  void commitUpdate(const std::vector<double>& updated_estimate);

private:
  void requireCompatible(const std::vector<double>& values,
                         const char* name) const;

  Grid grid_;
  double time_scale_;
  std::vector<double> estimate_;
  std::vector<double> integral_;
  State previous_state_;
  double block_start_time_;
  double previous_time_;
  bool block_active_;
};

}  // namespace environment
}  // namespace burgers
