#include "burgers/environment/MeanProfileEstimator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace burgers {
namespace environment {

MeanProfileEstimator::MeanProfileEstimator(const Grid& grid,
                                           double time_scale)
  : grid_(grid),
    time_scale_(time_scale),
    estimate_(grid.cellCount(), 0.0),
    integral_(grid.cellCount(), 0.0),
    previous_state_(grid, 0.0),
    block_start_time_(0.0),
    previous_time_(0.0),
    block_active_(false) {
  if(std::isfinite(time_scale_) == 0 || time_scale_ <= 0.0) {
    throw std::invalid_argument("mean-profile EMA time scale must be positive");
  }
}

double MeanProfileEstimator::timeScale() const noexcept {
  return time_scale_;
}

const std::vector<double>& MeanProfileEstimator::estimate() const noexcept {
  return estimate_;
}

void MeanProfileEstimator::requireCompatible(
  const std::vector<double>& values, const char* name) const {
  if(values.size() != grid_.cellCount()) {
    throw std::invalid_argument(std::string(name) +
                                " size does not match the grid");
  }
  for(std::size_t cell = 0; cell < values.size(); ++cell) {
    if(std::isfinite(values[cell]) == 0) {
      throw std::invalid_argument(
        std::string(name) + " contains a non-finite value at cell " +
        std::to_string(cell));
    }
  }
}

void MeanProfileEstimator::reset(
  const std::vector<double>& initial_estimate) {
  requireCompatible(initial_estimate, "initial mean estimate");
  estimate_ = initial_estimate;
  std::fill(integral_.begin(), integral_.end(), 0.0);
  block_start_time_ = 0.0;
  previous_time_ = 0.0;
  block_active_ = false;
}

void MeanProfileEstimator::beginBlock(double time, const State& state) {
  if(block_active_) {
    throw std::logic_error("a mean-profile block is already active");
  }
  if(std::isfinite(time) == 0) {
    throw std::invalid_argument("mean-profile block start time must be finite");
  }
  requireCompatible(state.cellAverages(), "mean-profile block state");
  std::fill(integral_.begin(), integral_.end(), 0.0);
  previous_state_ = state;
  block_start_time_ = time;
  previous_time_ = time;
  block_active_ = true;
}

void MeanProfileEstimator::observeAcceptedState(double time,
                                                const State& state) {
  if(!block_active_) {
    throw std::logic_error("no mean-profile block is active");
  }
  if(std::isfinite(time) == 0 || time <= previous_time_) {
    throw std::invalid_argument(
      "accepted-state time must advance within a mean-profile block");
  }
  requireCompatible(state.cellAverages(), "accepted mean-profile state");
  const double duration = time - previous_time_;
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    integral_[cell] +=
      0.5 * duration * (previous_state_[cell] + state[cell]);
  }
  previous_state_ = state;
  previous_time_ = time;
}

double MeanProfileEstimator::blockDuration() const {
  if(!block_active_) {
    throw std::logic_error("no mean-profile block is active");
  }
  return previous_time_ - block_start_time_;
}

std::vector<double> MeanProfileEstimator::completeBlock() const {
  const double duration = blockDuration();
  if(std::isfinite(duration) == 0 || duration <= 0.0) {
    throw std::logic_error("mean-profile block has zero duration");
  }
  std::vector<double> result(integral_.size(), 0.0);
  for(std::size_t cell = 0; cell < result.size(); ++cell) {
    result[cell] = integral_[cell] / duration;
  }
  return result;
}

std::vector<double> MeanProfileEstimator::previewUpdate(
  const std::vector<double>& block_mean, double duration) const {
  requireCompatible(block_mean, "block mean");
  if(std::isfinite(duration) == 0 || duration <= 0.0) {
    throw std::invalid_argument("EMA update duration must be positive");
  }
  const double alpha = 1.0 - std::exp(-duration / time_scale_);
  std::vector<double> updated(estimate_.size(), 0.0);
  for(std::size_t cell = 0; cell < updated.size(); ++cell) {
    updated[cell] =
      (1.0 - alpha) * estimate_[cell] + alpha * block_mean[cell];
  }
  return updated;
}

void MeanProfileEstimator::commitUpdate(
  const std::vector<double>& updated_estimate) {
  requireCompatible(updated_estimate, "updated mean estimate");
  estimate_ = updated_estimate;
  block_active_ = false;
}

}  // namespace environment
}  // namespace burgers
