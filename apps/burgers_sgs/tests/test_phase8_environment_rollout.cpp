#include "burgers/Config.h"
#include "burgers/environment/SGSEnvironment.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if(!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expectNear(double actual,
                double expected,
                double tolerance,
                const std::string& message) {
  if(std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " (expected " << expected
              << ", got " << actual << ", tolerance " << tolerance
              << ")\n";
    ++failures;
  }
}

template<typename Exception, typename Function>
void expectThrows(Function function, const std::string& message) {
  try {
    function();
  } catch(const Exception&) {
    return;
  } catch(const std::exception& error) {
    std::cerr << "FAIL: " << message << " (wrong exception: "
              << error.what() << ")\n";
    ++failures;
    return;
  }
  std::cerr << "FAIL: " << message << " (no exception)\n";
  ++failures;
}

burgers::environment::SGSEnvironmentConfig environmentConfig() {
  burgers::environment::SGSEnvironmentConfig config;
  config.solver = burgers::makeDefaultRunConfig();
  config.solver.grid.cell_count = 16u;
  config.solver.initial_condition.type =
    burgers::InitialConditionType::Random;
  config.solver.initial_condition.mean = 0.0;
  config.solver.initial_condition.amplitude = 0.05;
  config.solver.viscosity.molecular = 0.01;
  config.solver.forcing.type = burgers::ForcingType::StochasticFourierOu;
  config.solver.forcing.stochastic = burgers::makeLowModeOuForcingConfig();
  config.solver.closure.type = burgers::ClosureType::NoClosure;
  config.solver.time_integration.initial_time = 0.0;
  config.solver.time_integration.final_time = 0.03;
  config.solver.time_integration.maximum_steps = 10000u;
  config.target_metadata_path =
    std::string(BURGERS_TEST_SOURCE_DIR) +
    "/runs/finalization/dns_target_metadata.json";
  config.action.forcing_scale = 0.5;
  config.action.smoothing = burgers::environment::ActionSmoothing::None;
  config.reward.type = burgers::environment::RewardType::SquaredEma;
  config.reward.aggregation = burgers::environment::RewardAggregation::Local;
  config.reward.action_penalty = 0.0;
  config.decision_interval = 0.01;
  config.warmup_duration = 0.01;
  config.control_duration = 0.02;
  config.mean_time_scale = 0.02;
  config.observation_radius = 1u;
  config.zero_warmup_rewards = true;
  config.maximum_absolute_velocity = 5.0;
  config.failure_penalty = -100.0;
  config.record_action_spectra = true;
  return config;
}

bool allEqual(const std::vector<double>& values, double expected) {
  for(const double value : values) {
    if(value != expected) return false;
  }
  return true;
}

double maximumDifference(const burgers::State& left,
                         const burgers::State& right) {
  double difference = 0.0;
  for(std::size_t cell = 0; cell < left.size(); ++cell) {
    difference = std::max(difference, std::abs(left[cell] - right[cell]));
  }
  return difference;
}

void checkDefaults() {
  const burgers::environment::SGSEnvironmentConfig config;
  expectNear(config.action.forcing_scale, 0.5, 0.0,
             "default forcing scale must match the provisional pilot bound");
  expect(config.action.smoothing ==
           burgers::environment::ActionSmoothing::None,
         "action smoothing must be disabled by default");
  expectNear(config.decision_interval, 0.1, 0.0,
             "default decision interval must be physical time 0.1");
  expectNear(config.warmup_duration, 20.0, 0.0,
             "default warm-up duration must be 20");
  expectNear(config.control_duration, 180.0, 0.0,
             "default controlled duration must be 180");
  expect(config.zero_warmup_rewards,
         "warm-up rewards must be masked by default");
}

void checkDeterministicResetAndRollout() {
  burgers::environment::SGSEnvironment environment(environmentConfig());
  environment.reset(778899u);
  const burgers::State initial = environment.state();
  const std::vector<std::vector<double>> initial_observations =
    environment.observations();
  expect(environment.status() == burgers::environment::EpisodeStatus::Running,
         "reset environment must be running");
  expect(environment.actionHistory().empty() &&
         environment.rewardHistory().empty(),
         "reset must clear action and reward histories");

  std::vector<double> actions(environment.grid().cellCount(), 0.0);
  for(std::size_t cell = 0; cell < actions.size(); ++cell) {
    actions[cell] = cell % 2u == 0u ? 0.4 : -0.4;
  }
  const burgers::environment::StepResult warmup = environment.step(actions);
  expect(warmup.status == burgers::environment::EpisodeStatus::Running,
         "warm-up transition must not truncate the episode");
  expect(warmup.diagnostics.warmup && warmup.diagnostics.reward_masked,
         "first transition must be a masked policy-controlled warm-up");
  expect(allEqual(warmup.rewards, 0.0),
         "normal warm-up rewards must be zero");
  expect(warmup.diagnostics.candidate_rewards.size() == actions.size(),
         "masked warm-up must retain candidate rewards in diagnostics");
  expect(warmup.diagnostics.action.applied_statistics.maximum_absolute > 0.0,
         "policy actions must be applied during warm-up");

  const burgers::environment::StepResult controlled = environment.step(actions);
  expect(!controlled.diagnostics.warmup &&
         !controlled.diagnostics.reward_masked,
         "reward must activate immediately after warm-up");
  expect(!allEqual(controlled.rewards, 0.0),
         "controlled transition must deliver its candidate reward");

  const burgers::environment::StepResult timeout = environment.step(actions);
  expect(timeout.status == burgers::environment::EpisodeStatus::Timeout,
         "last finite-horizon transition must report a normal timeout");
  expectNear(environment.time(), 0.03, 2.0e-15,
             "episode must stop at its exact physical-time endpoint");
  expect(environment.actionHistory().size() == 3u &&
         environment.rewardHistory().size() == 3u,
         "complete rollout must retain per-decision histories");
  expectThrows<std::logic_error>(
    [&environment, &actions]() { environment.step(actions); },
    "environment must reject steps after timeout");

  environment.reset(778899u);
  expectNear(maximumDifference(initial, environment.state()), 0.0, 0.0,
             "reset with the same episode seed must reproduce the state");
  expect(initial_observations == environment.observations(),
         "reset with the same episode seed must reproduce observations");
  const burgers::environment::StepResult repeated = environment.step(actions);
  expectNear(maximumDifference(
               burgers::State(warmup.diagnostics.updated_mean),
               burgers::State(repeated.diagnostics.updated_mean)),
             0.0, 0.0,
             "scripted rollout must reproduce the mean estimator exactly");

  environment.reset(778900u);
  expect(maximumDifference(initial, environment.state()) > 0.0,
         "a different episode seed must select an independent reset stream");
}

void checkFailuresAndValidation() {
  burgers::environment::SGSEnvironment environment(environmentConfig());
  environment.reset(11u);
  std::vector<double> invalid(environment.grid().cellCount(), 0.0);
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  const burgers::environment::StepResult failed = environment.step(invalid);
  expect(failed.status == burgers::environment::EpisodeStatus::Failure,
         "non-finite action must terminate every coupled agent");
  expect(allEqual(failed.rewards, -100.0),
         "action failure must override warm-up masking with finite penalty");
  expect(!failed.diagnostics.failure_message.empty(),
         "failure result must retain a diagnostic message");

  environment.reset(12u);
  expect(environment.status() == burgers::environment::EpisodeStatus::Running &&
         environment.failureMessage().empty(),
         "reset must clear failure status");
  expectThrows<std::invalid_argument>(
    [&environment]() {
      environment.step(
        std::vector<double>(environment.grid().cellCount() - 1u, 0.0));
    },
    "wrong action count must remain a caller contract error");

  burgers::environment::SGSEnvironmentConfig invalid_config =
    environmentConfig();
  invalid_config.solver.forcing.type = burgers::ForcingType::Composite;
  invalid_config.solver.forcing.deterministic.modes = {{1, 0.1, 0.0}};
  expectThrows<std::invalid_argument>(
    [&invalid_config]() {
      burgers::environment::SGSEnvironment invalid_environment(invalid_config);
      static_cast<void>(invalid_environment);
    },
    "controlled environment must reject known deterministic forcing");

  invalid_config = environmentConfig();
  invalid_config.warmup_duration = 0.015;
  expectThrows<std::invalid_argument>(
    [&invalid_config]() {
      burgers::environment::SGSEnvironment invalid_environment(invalid_config);
      static_cast<void>(invalid_environment);
    },
    "episode phases must contain whole decision intervals");
}

}  // namespace

int main() {
  checkDefaults();
  checkDeterministicResetAndRollout();
  checkFailuresAndValidation();

  if(failures != 0) {
    std::cerr << failures << " environment-rollout check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 8 environment-rollout checks passed\n";
  return 0;
}
