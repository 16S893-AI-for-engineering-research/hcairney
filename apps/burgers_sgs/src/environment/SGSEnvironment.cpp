#include "burgers/environment/SGSEnvironment.h"

#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace burgers {
namespace environment {
namespace {

std::uint64_t deriveSeed(std::uint64_t episode_seed,
                         std::uint64_t stream_tag) noexcept {
  std::uint64_t value = episode_seed ^ stream_tag;
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

bool durationIsWholeIntervals(double duration, double interval) noexcept {
  const double count = duration / interval;
  const double nearest = std::round(count);
  const double scale = std::max(1.0, std::abs(count));
  return std::abs(count - nearest) <=
         256.0 * std::numeric_limits<double>::epsilon() * scale;
}

double timeTolerance(double value) noexcept {
  return 256.0 * std::numeric_limits<double>::epsilon() *
         std::max(1.0, std::abs(value));
}

bool hasNonzeroDeterministicForcing(const RunConfig& config) {
  if(config.forcing.type != ForcingType::DeterministicFourier &&
     config.forcing.type != ForcingType::Composite) {
    return false;
  }
  for(const FourierModeConfig& mode : config.forcing.deterministic.modes) {
    if(mode.amplitude != 0.0) return true;
  }
  return false;
}

std::string joinErrors(const std::vector<std::string>& errors) {
  std::ostringstream message;
  for(std::size_t index = 0; index < errors.size(); ++index) {
    if(index != 0u) message << "; ";
    message << errors[index];
  }
  return message.str();
}

}  // namespace

SGSEnvironment::SGSEnvironment(const SGSEnvironmentConfig& config)
  : config_(config),
    target_profile_(TargetProfile::loadAccepted(config.target_metadata_path)),
    grid_(config.solver.grid),
    target_mean_(target_profile_.restrictMeanVelocity(grid_)),
    action_projection_(grid_, config.action),
    mean_estimator_(grid_, config.mean_time_scale),
    observation_builder_(grid_, config.solver.viscosity.molecular,
                         config.observation_radius),
    reward_model_(grid_, target_mean_, config.action.forcing_scale,
                  config.reward),
    solver_(),
    state_(grid_, 0.0),
    time_(config.solver.time_integration.initial_time),
    episode_end_time_(time_),
    episode_seed_(0u),
    decision_index_(0u),
    status_(EpisodeStatus::Running) {
  validateConfiguration();
  reset(config_.solver.random.seed);
}

const SGSEnvironmentConfig& SGSEnvironment::config() const noexcept {
  return config_;
}

const Grid& SGSEnvironment::grid() const noexcept {
  return grid_;
}

const State& SGSEnvironment::state() const noexcept {
  return state_;
}

std::size_t SGSEnvironment::observationSize() const noexcept {
  return observation_builder_.observationSize();
}

double SGSEnvironment::rewardVelocityScale() const noexcept {
  return reward_model_.velocityScale();
}

double SGSEnvironment::time() const noexcept {
  return time_;
}

double SGSEnvironment::episodeEndTime() const noexcept {
  return episode_end_time_;
}

std::uint64_t SGSEnvironment::episodeSeed() const noexcept {
  return episode_seed_;
}

EpisodeStatus SGSEnvironment::status() const noexcept {
  return status_;
}

const std::string& SGSEnvironment::failureMessage() const noexcept {
  return failure_message_;
}

const TargetProfile& SGSEnvironment::targetProfile() const noexcept {
  return target_profile_;
}

const std::vector<double>& SGSEnvironment::targetMean() const noexcept {
  return target_mean_;
}

const std::vector<double>& SGSEnvironment::runningMean() const noexcept {
  return mean_estimator_.estimate();
}

const std::vector<ActionHistoryEntry>&
SGSEnvironment::actionHistory() const noexcept {
  return action_history_;
}

const std::vector<RewardHistoryEntry>&
SGSEnvironment::rewardHistory() const noexcept {
  return reward_history_;
}

void SGSEnvironment::validateConfiguration() const {
  const std::vector<std::string> solver_errors = validate(config_.solver);
  if(!solver_errors.empty()) {
    throw std::invalid_argument(
      "invalid environment solver configuration: " +
      joinErrors(solver_errors));
  }
  if(config_.solver.forcing.type == ForcingType::Manufactured) {
    throw std::invalid_argument(
      "the controlled environment cannot enable manufactured forcing");
  }
  if(hasNonzeroDeterministicForcing(config_)) {
    throw std::invalid_argument(
      "the controlled environment must disable configured deterministic "
      "forcing; use the prescribed field for an oracle rollout");
  }
  if(config_.solver.closure.type == ClosureType::PrescribedCoefficientField) {
    throw std::invalid_argument(
      "the environment core does not supply prescribed closure coefficients");
  }
  if(std::isfinite(config_.decision_interval) == 0 ||
     config_.decision_interval <= 0.0) {
    throw std::invalid_argument(
      "environment decision interval must be positive");
  }
  if(std::isfinite(config_.warmup_duration) == 0 ||
     config_.warmup_duration < 0.0) {
    throw std::invalid_argument(
      "environment warm-up duration must be finite and nonnegative");
  }
  if(std::isfinite(config_.control_duration) == 0 ||
     config_.control_duration <= 0.0) {
    throw std::invalid_argument(
      "environment control duration must be positive");
  }
  if(!durationIsWholeIntervals(config_.warmup_duration,
                               config_.decision_interval) ||
     !durationIsWholeIntervals(config_.control_duration,
                               config_.decision_interval)) {
    throw std::invalid_argument(
      "warm-up and control durations must contain whole decision intervals");
  }
  if(std::isfinite(config_.maximum_absolute_velocity) == 0 ||
     config_.maximum_absolute_velocity <= 0.0) {
    throw std::invalid_argument(
      "maximum absolute velocity must be positive");
  }
  if(std::isfinite(config_.failure_penalty) == 0) {
    throw std::invalid_argument("failure penalty must be finite");
  }
  const double final_time =
    config_.solver.time_integration.initial_time +
    config_.warmup_duration + config_.control_duration;
  if(std::isfinite(final_time) == 0) {
    throw std::invalid_argument("environment episode end time is not finite");
  }
}

void SGSEnvironment::reset(std::uint64_t episode_seed) {
  RunConfig episode_config = config_.solver;
  const std::uint64_t initial_condition_seed =
    deriveSeed(episode_seed, 0x243f6a8885a308d3ULL);
  const std::uint64_t stochastic_forcing_seed =
    deriveSeed(episode_seed, 0x13198a2e03707344ULL);
  episode_config.random.seed = stochastic_forcing_seed;
  solver_.reset(new BurgersSolver(episode_config));
  initializeState(grid_, state_, episode_config.initial_condition,
                  initial_condition_seed);
  solver_->setPrescribedAdditiveForcingField(
    std::vector<double>(grid_.cellCount(), 0.0));

  time_ = episode_config.time_integration.initial_time;
  episode_end_time_ =
    time_ + config_.warmup_duration + config_.control_duration;
  episode_seed_ = episode_seed;
  decision_index_ = 0u;
  status_ = EpisodeStatus::Running;
  failure_message_.clear();
  action_history_.clear();
  reward_history_.clear();
  mean_estimator_.reset(state_.cellAverages());
}

std::vector<std::vector<double>> SGSEnvironment::observations() const {
  return observation_builder_.build(state_);
}

bool SGSEnvironment::transitionIsWarmup(double end_time) const noexcept {
  const double warmup_end =
    config_.solver.time_integration.initial_time + config_.warmup_duration;
  return config_.warmup_duration > 0.0 &&
         end_time <= warmup_end + timeTolerance(warmup_end);
}

bool SGSEnvironment::reachedEpisodeEnd(double time) const noexcept {
  return time >= episode_end_time_ - timeTolerance(episode_end_time_);
}

double SGSEnvironment::profileError(
  const std::vector<double>& estimate) const {
  double error_square = 0.0;
  double target_square = 0.0;
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    const double difference = estimate[cell] - target_mean_[cell];
    error_square += grid_.cellWidth(cell) * difference * difference;
    target_square +=
      grid_.cellWidth(cell) * target_mean_[cell] * target_mean_[cell];
  }
  return std::sqrt(error_square / target_square);
}

double SGSEnvironment::maximumAbsoluteVelocity() const {
  double maximum = 0.0;
  for(const double velocity : state_) {
    if(std::isfinite(velocity) == 0) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(maximum, std::abs(velocity));
  }
  return maximum;
}

std::vector<double> SGSEnvironment::actionSpectrum(
  const std::vector<double>& values) const {
  if(!config_.record_action_spectra) return std::vector<double>();
  return energySpectrum(grid_, State(values));
}

StepResult SGSEnvironment::failureResult(
  const std::string& message, StepDiagnostics diagnostics) {
  status_ = EpisodeStatus::Failure;
  failure_message_ = message;
  diagnostics.failure_message = message;
  diagnostics.maximum_absolute_velocity = maximumAbsoluteVelocity();
  std::vector<double> penalties(grid_.cellCount(), config_.failure_penalty);

  RewardHistoryEntry reward_history;
  reward_history.start_time = diagnostics.start_time;
  reward_history.end_time = diagnostics.end_time;
  reward_history.warmup = diagnostics.warmup;
  reward_history.candidate = diagnostics.candidate_rewards;
  reward_history.delivered = penalties;
  reward_history_.push_back(reward_history);

  StepResult result;
  try {
    result.observations = observations();
  } catch(const std::exception&) {
    result.observations.clear();
  }
  result.rewards = penalties;
  result.diagnostics = std::move(diagnostics);
  result.status = status_;
  ++decision_index_;
  return result;
}

StepResult SGSEnvironment::step(
  const std::vector<double>& raw_actions) {
  if(status_ != EpisodeStatus::Running) {
    throw std::logic_error("cannot step an environment after episode end");
  }
  if(raw_actions.size() != grid_.cellCount()) {
    throw std::invalid_argument(
      "raw action field size does not match the environment grid");
  }

  StepDiagnostics diagnostics;
  diagnostics.episode_seed = episode_seed_;
  diagnostics.decision_index = decision_index_;
  diagnostics.start_time = time_;
  const double requested_end = std::min(
    episode_end_time_, diagnostics.start_time + config_.decision_interval);
  diagnostics.end_time = requested_end;
  diagnostics.warmup = transitionIsWarmup(requested_end);
  diagnostics.pre_action_mean = mean_estimator_.estimate();

  try {
    diagnostics.action = action_projection_.apply(raw_actions);
    diagnostics.raw_action_spectrum = actionSpectrum(raw_actions);
    diagnostics.applied_action_spectrum =
      actionSpectrum(diagnostics.action.applied);
  } catch(const std::exception& error) {
    ActionHistoryEntry history;
    history.start_time = diagnostics.start_time;
    history.end_time = diagnostics.start_time;
    history.raw = raw_actions;
    action_history_.push_back(history);
    diagnostics.end_time = diagnostics.start_time;
    diagnostics.warmup = transitionIsWarmup(diagnostics.start_time);
    return failureResult(error.what(), std::move(diagnostics));
  }

  solver_->setPrescribedAdditiveForcingField(diagnostics.action.applied);
  mean_estimator_.beginBlock(time_, state_);
  double last_observed_time = time_;
  try {
    diagnostics.advance = solver_->advanceTo(
      state_, time_, requested_end,
      config_.solver.time_integration.maximum_steps,
      [this, &last_observed_time](std::size_t, double observed_time,
                                  const State& observed_state) {
        mean_estimator_.observeAcceptedState(observed_time, observed_state);
        last_observed_time = observed_time;
      });
    time_ = diagnostics.advance.final_time;
  } catch(const std::exception& error) {
    time_ = last_observed_time;
    diagnostics.end_time = time_;
    ActionHistoryEntry history;
    history.start_time = diagnostics.start_time;
    history.end_time = diagnostics.end_time;
    history.raw = raw_actions;
    history.applied = diagnostics.action.applied;
    action_history_.push_back(history);
    return failureResult(error.what(), std::move(diagnostics));
  }

  diagnostics.end_time = time_;
  diagnostics.warmup = transitionIsWarmup(time_);
  ActionHistoryEntry action_history;
  action_history.start_time = diagnostics.start_time;
  action_history.end_time = diagnostics.end_time;
  action_history.raw = raw_actions;
  action_history.applied = diagnostics.action.applied;
  action_history_.push_back(action_history);

  try {
    diagnostics.block_mean = mean_estimator_.completeBlock();
    const double block_duration = mean_estimator_.blockDuration();
    diagnostics.updated_mean =
      mean_estimator_.previewUpdate(diagnostics.block_mean, block_duration);
    // Reward evaluation receives the cached estimate and the previewed update;
    // the estimator itself remains at the pre-action value until this call
    // finishes, preserving the directional-reward timing contract.
    diagnostics.candidate_rewards = reward_model_.evaluate(
      diagnostics.pre_action_mean,
      diagnostics.block_mean,
      diagnostics.updated_mean,
      diagnostics.action.applied);
    mean_estimator_.commitUpdate(diagnostics.updated_mean);
    diagnostics.mean_profile_relative_l2 =
      profileError(diagnostics.updated_mean);
    diagnostics.maximum_absolute_velocity = maximumAbsoluteVelocity();
  } catch(const std::exception& error) {
    return failureResult(error.what(), std::move(diagnostics));
  }

  if(diagnostics.maximum_absolute_velocity >
     config_.maximum_absolute_velocity) {
    return failureResult(
      "maximum absolute velocity exceeded the configured safety limit",
      std::move(diagnostics));
  }

  std::vector<double> delivered = diagnostics.candidate_rewards;
  diagnostics.reward_masked =
    diagnostics.warmup && config_.zero_warmup_rewards;
  if(diagnostics.reward_masked) {
    std::fill(delivered.begin(), delivered.end(), 0.0);
  }

  RewardHistoryEntry reward_history;
  reward_history.start_time = diagnostics.start_time;
  reward_history.end_time = diagnostics.end_time;
  reward_history.warmup = diagnostics.warmup;
  reward_history.candidate = diagnostics.candidate_rewards;
  reward_history.delivered = delivered;
  reward_history_.push_back(reward_history);

  if(reachedEpisodeEnd(time_)) {
    time_ = episode_end_time_;
    status_ = EpisodeStatus::Timeout;
  }

  StepResult result;
  result.observations = observations();
  result.rewards = delivered;
  result.diagnostics = std::move(diagnostics);
  result.status = status_;
  ++decision_index_;
  return result;
}

}  // namespace environment
}  // namespace burgers
