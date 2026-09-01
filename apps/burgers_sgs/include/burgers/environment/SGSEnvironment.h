#pragma once

#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/State.h"
#include "burgers/environment/ActionProjection.h"
#include "burgers/environment/MeanProfileEstimator.h"
#include "burgers/environment/ObservationBuilder.h"
#include "burgers/environment/RewardModel.h"
#include "burgers/environment/TargetProfile.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace burgers {
namespace environment {

enum class EpisodeStatus {
  Running,
  Timeout,
  Failure
};

struct SGSEnvironmentConfig {
  // The complete existing solver configuration remains the source of grid,
  // numerics, closure, initial-condition, and stochastic-forcing choices.
  // Episode duration below supersedes solver.time_integration.final_time for
  // environment rollouts without mutating the supplied RunConfig.
  RunConfig solver;
  std::string target_metadata_path;
  ActionProjectionConfig action;
  RewardConfig reward;
  double decision_interval = 0.1;
  double warmup_duration = 20.0;
  double control_duration = 180.0;
  double mean_time_scale = 15.0;
  std::size_t observation_radius = 1;
  // Candidate rewards are still computed and recorded during warm-up. Only
  // the learner-facing values are zeroed; a failure penalty is never masked.
  bool zero_warmup_rewards = true;
  double maximum_absolute_velocity = 5.0;
  double failure_penalty = -100.0;
  bool record_action_spectra = true;
};

struct ActionHistoryEntry {
  double start_time = 0.0;
  double end_time = 0.0;
  std::vector<double> raw;
  std::vector<double> applied;
};

struct RewardHistoryEntry {
  double start_time = 0.0;
  double end_time = 0.0;
  bool warmup = false;
  std::vector<double> candidate;
  std::vector<double> delivered;
};

struct StepDiagnostics {
  std::uint64_t episode_seed = 0;
  std::size_t decision_index = 0;
  double start_time = 0.0;
  double end_time = 0.0;
  bool warmup = false;
  bool reward_masked = false;
  ActionProjectionResult action;
  std::vector<double> raw_action_spectrum;
  std::vector<double> applied_action_spectrum;
  std::vector<double> pre_action_mean;
  std::vector<double> block_mean;
  std::vector<double> updated_mean;
  std::vector<double> candidate_rewards;
  double mean_profile_relative_l2 = 0.0;
  double maximum_absolute_velocity = 0.0;
  AdvanceResult advance;
  std::string failure_message;
};

struct StepResult {
  std::vector<std::vector<double>> observations;
  std::vector<double> rewards;
  StepDiagnostics diagnostics;
  EpisodeStatus status = EpisodeStatus::Running;
};

class SGSEnvironment {
public:
  explicit SGSEnvironment(const SGSEnvironmentConfig& config);

  const SGSEnvironmentConfig& config() const noexcept;
  const Grid& grid() const noexcept;
  const State& state() const noexcept;
  std::size_t observationSize() const noexcept;
  double rewardVelocityScale() const noexcept;
  double time() const noexcept;
  double episodeEndTime() const noexcept;
  std::uint64_t episodeSeed() const noexcept;
  EpisodeStatus status() const noexcept;
  const std::string& failureMessage() const noexcept;
  const TargetProfile& targetProfile() const noexcept;
  const std::vector<double>& targetMean() const noexcept;
  const std::vector<double>& runningMean() const noexcept;
  const std::vector<ActionHistoryEntry>& actionHistory() const noexcept;
  const std::vector<RewardHistoryEntry>& rewardHistory() const noexcept;

  void reset(std::uint64_t episode_seed);
  std::vector<std::vector<double>> observations() const;
  StepResult step(const std::vector<double>& raw_actions);

private:
  void validateConfiguration() const;
  bool transitionIsWarmup(double end_time) const noexcept;
  bool reachedEpisodeEnd(double time) const noexcept;
  double profileError(const std::vector<double>& estimate) const;
  double maximumAbsoluteVelocity() const;
  std::vector<double> actionSpectrum(const std::vector<double>& values) const;
  StepResult failureResult(const std::string& message,
                           StepDiagnostics diagnostics);

  SGSEnvironmentConfig config_;
  TargetProfile target_profile_;
  Grid grid_;
  std::vector<double> target_mean_;
  ActionProjection action_projection_;
  MeanProfileEstimator mean_estimator_;
  ObservationBuilder observation_builder_;
  RewardModel reward_model_;
  std::unique_ptr<BurgersSolver> solver_;
  State state_;
  double time_;
  double episode_end_time_;
  std::uint64_t episode_seed_;
  std::size_t decision_index_;
  EpisodeStatus status_;
  std::string failure_message_;
  std::vector<ActionHistoryEntry> action_history_;
  std::vector<RewardHistoryEntry> reward_history_;
};

}  // namespace environment
}  // namespace burgers
