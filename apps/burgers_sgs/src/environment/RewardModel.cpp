#include "burgers/environment/RewardModel.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace burgers {
namespace environment {
namespace {

double sign(double value) noexcept {
  return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

}  // namespace

RewardModel::RewardModel(const Grid& grid,
                         const std::vector<double>& target,
                         double forcing_scale,
                         const RewardConfig& config)
  : grid_(grid),
    target_(target),
    forcing_scale_(forcing_scale),
    config_(config),
    velocity_scale_(config.velocity_scale),
    target_square_integral_(0.0) {
  requireCompatible(target_, "reward target");
  if(std::isfinite(forcing_scale_) == 0 || forcing_scale_ <= 0.0) {
    throw std::invalid_argument("reward forcing scale must be positive");
  }
  if(std::isfinite(config_.action_penalty) == 0 ||
     config_.action_penalty < 0.0) {
    throw std::invalid_argument(
      "reward action penalty must be finite and nonnegative");
  }
  if(!config_.local_squared_weights.empty()) {
    requireCompatible(config_.local_squared_weights,
                      "local squared reward weights");
    for(const double weight : config_.local_squared_weights) {
      if(weight < 0.0) {
        throw std::invalid_argument(
          "local squared reward weights must be nonnegative");
      }
    }
  }

  double volume = 0.0;
  for(std::size_t cell = 0; cell < target_.size(); ++cell) {
    const double width = grid_.cellWidth(cell);
    target_square_integral_ += width * target_[cell] * target_[cell];
    volume += width;
  }
  if(std::isfinite(target_square_integral_) == 0 ||
     target_square_integral_ <= 0.0) {
    throw std::invalid_argument(
      "reward target must have a finite positive spatial square integral");
  }
  if(velocity_scale_ <= 0.0) {
    velocity_scale_ = std::sqrt(target_square_integral_ / volume);
  }
  if(std::isfinite(velocity_scale_) == 0 || velocity_scale_ <= 0.0) {
    throw std::invalid_argument(
      "reward velocity scale must be positive; the target RMS is zero");
  }
}

double RewardModel::velocityScale() const noexcept {
  return velocity_scale_;
}

const RewardConfig& RewardModel::config() const noexcept {
  return config_;
}

void RewardModel::requireCompatible(const std::vector<double>& values,
                                    const char* name) const {
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

double RewardModel::weightedMean(
  const std::vector<double>& values) const {
  double sum = 0.0;
  double volume = 0.0;
  for(std::size_t cell = 0; cell < values.size(); ++cell) {
    sum += grid_.cellWidth(cell) * values[cell];
    volume += grid_.cellWidth(cell);
  }
  return sum / volume;
}

std::vector<double> RewardModel::evaluate(
  const std::vector<double>& pre_action_mean,
  const std::vector<double>& block_mean,
  const std::vector<double>& updated_mean,
  const std::vector<double>& applied_forcing) const {
  requireCompatible(pre_action_mean, "pre-action mean");
  requireCompatible(block_mean, "block mean");
  requireCompatible(updated_mean, "updated mean");
  requireCompatible(applied_forcing, "applied forcing");

  std::vector<double> profile_terms(grid_.cellCount(), 0.0);
  std::vector<double> penalty_terms(grid_.cellCount(), 0.0);
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    const double normalized_forcing =
      applied_forcing[cell] / forcing_scale_;
    penalty_terms[cell] =
      config_.action_penalty * normalized_forcing * normalized_forcing;
    if(config_.type == RewardType::DirectionalBlockVelocity) {
      profile_terms[cell] =
        -sign(pre_action_mean[cell] - target_[cell]) *
        (block_mean[cell] - target_[cell]) / velocity_scale_;
    } else {
      const double normalized_error =
        (updated_mean[cell] - target_[cell]) / velocity_scale_;
      const double weight = config_.local_squared_weights.empty()
                              ? 1.0
                              : config_.local_squared_weights[cell];
      profile_terms[cell] = -weight * normalized_error * normalized_error;
    }
  }

  std::vector<double> rewards(grid_.cellCount(), 0.0);
  if(config_.aggregation == RewardAggregation::Local) {
    for(std::size_t cell = 0; cell < rewards.size(); ++cell) {
      rewards[cell] = profile_terms[cell] - penalty_terms[cell];
      if(std::isfinite(rewards[cell]) == 0) {
        throw std::runtime_error(
          "local reward became non-finite at cell " +
          std::to_string(cell));
      }
    }
    return rewards;
  }

  double shared_profile = 0.0;
  if(config_.type == RewardType::DirectionalBlockVelocity) {
    shared_profile = weightedMean(profile_terms);
  } else {
    double error_square_integral = 0.0;
    for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
      const double error = updated_mean[cell] - target_[cell];
      error_square_integral += grid_.cellWidth(cell) * error * error;
    }
    shared_profile = -error_square_integral / target_square_integral_;
  }
  const double shared_reward =
    shared_profile - weightedMean(penalty_terms);
  if(std::isfinite(shared_reward) == 0) {
    throw std::runtime_error("shared reward became non-finite");
  }
  for(double& reward : rewards) {
    reward = shared_reward;
  }
  return rewards;
}

}  // namespace environment
}  // namespace burgers
