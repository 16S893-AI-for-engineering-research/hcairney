#pragma once

#include "burgers/Grid.h"

#include <vector>

namespace burgers {
namespace environment {

enum class RewardType {
  DirectionalBlockVelocity,
  SquaredEma
};

enum class RewardAggregation {
  Local,
  SharedGlobal
};

struct RewardConfig {
  RewardType type = RewardType::SquaredEma;
  RewardAggregation aggregation = RewardAggregation::Local;
  // A nonpositive value selects the volume-weighted RMS of the DNS target.
  double velocity_scale = 0.0;
  double action_penalty = 0.0;
  // Empty selects unit weights. Nonempty weights apply only to local EMA2.
  std::vector<double> local_squared_weights;
};

class RewardModel {
public:
  RewardModel(const Grid& grid,
              const std::vector<double>& target,
              double forcing_scale,
              const RewardConfig& config);

  double velocityScale() const noexcept;
  const RewardConfig& config() const noexcept;

  std::vector<double> evaluate(
    const std::vector<double>& pre_action_mean,
    const std::vector<double>& block_mean,
    const std::vector<double>& updated_mean,
    const std::vector<double>& applied_forcing) const;

private:
  void requireCompatible(const std::vector<double>& values,
                         const char* name) const;
  double weightedMean(const std::vector<double>& values) const;

  Grid grid_;
  std::vector<double> target_;
  double forcing_scale_;
  RewardConfig config_;
  double velocity_scale_;
  double target_square_integral_;
};

}  // namespace environment
}  // namespace burgers
