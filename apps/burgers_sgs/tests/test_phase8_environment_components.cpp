#include "burgers/Grid.h"
#include "burgers/State.h"
#include "burgers/environment/ActionProjection.h"
#include "burgers/environment/MeanProfileEstimator.h"
#include "burgers/environment/ObservationBuilder.h"
#include "burgers/environment/RewardModel.h"
#include "burgers/environment/TargetProfile.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
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

double maximumDifference(const std::vector<double>& left,
                         const std::vector<double>& right) {
  if(left.size() != right.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double difference = 0.0;
  for(std::size_t index = 0; index < left.size(); ++index) {
    difference = std::max(difference, std::abs(left[index] - right[index]));
  }
  return difference;
}

std::vector<double> cyclicShift(const std::vector<double>& values,
                                std::size_t shift) {
  std::vector<double> result(values.size(), 0.0);
  for(std::size_t cell = 0; cell < values.size(); ++cell) {
    result[(cell + shift) % values.size()] = values[cell];
  }
  return result;
}

void checkActionProjection() {
  const burgers::Grid grid(0.0, 1.0, 8);
  burgers::environment::ActionProjectionConfig config;
  config.forcing_scale = 0.5;
  burgers::environment::ActionProjection projection(grid, config);
  const std::vector<double> raw = {
    -1.0, -0.8, -0.2, 0.0, 0.1, 0.4, 0.8, 1.0};
  const std::vector<double> saved = raw;
  const burgers::environment::ActionProjectionResult result =
    projection.apply(raw);
  expect(raw == saved, "action projection must not mutate its input");
  expectNear(result.applied_mean_residual, 0.0, 2.0e-16,
             "applied action must have zero mean");
  expect(result.applied_statistics.maximum_absolute <= 0.5 + 1.0e-15,
         "applied action must obey the physical bound");

  const burgers::environment::ActionProjectionResult uniform =
    projection.apply(std::vector<double>(grid.cellCount(), 0.7));
  expect(uniform.applied_statistics.maximum_absolute <= 1.0e-15,
         "uniform raw action must project to zero");

  const burgers::environment::ActionProjectionResult repeated =
    projection.apply(result.nondimensional);
  expectNear(maximumDifference(result.nondimensional,
                               repeated.nondimensional),
             0.0, 4.0e-16,
             "nondimensional action projection must be idempotent");

  const std::vector<double> shifted_raw = cyclicShift(raw, 3u);
  const burgers::environment::ActionProjectionResult shifted =
    projection.apply(shifted_raw);
  expectNear(maximumDifference(cyclicShift(result.applied, 3u),
                               shifted.applied),
             0.0, 4.0e-16,
             "action projection must commute with cyclic shifts");

  config.smoothing =
    burgers::environment::ActionSmoothing::PeriodicThreePoint;
  burgers::environment::ActionProjection smoothed(grid, config);
  const burgers::environment::ActionProjectionResult filtered =
    smoothed.apply(raw);
  expectNear(filtered.applied_mean_residual, 0.0, 2.0e-16,
             "smoothed action must retain zero mean");
  expect(filtered.applied_statistics.maximum_absolute <= 0.5 + 1.0e-15,
         "smoothed action must retain the field-wide bound");

  expectThrows<std::invalid_argument>(
    [&projection]() { projection.apply(std::vector<double>(7u, 0.0)); },
    "projection must reject a wrong-sized action field");
  std::vector<double> nonfinite(8u, 0.0);
  nonfinite[2] = std::numeric_limits<double>::quiet_NaN();
  expectThrows<std::invalid_argument>(
    [&projection, &nonfinite]() { projection.apply(nonfinite); },
    "projection must reject non-finite actions");
  std::vector<double> outside(8u, 0.0);
  outside[2] = 1.01;
  expectThrows<std::invalid_argument>(
    [&projection, &outside]() { projection.apply(outside); },
    "projection must reject actions outside the declared bounds");
}

void checkMeanEstimator() {
  const burgers::Grid grid(0.0, 1.0, 4);
  burgers::environment::MeanProfileEstimator estimator(grid, 2.0);
  estimator.reset(std::vector<double>(4u, 0.0));
  burgers::State state(grid, 0.0);
  estimator.beginBlock(0.0, state);
  state.fill(2.0);
  estimator.observeAcceptedState(1.0, state);
  state.fill(4.0);
  estimator.observeAcceptedState(3.0, state);
  const std::vector<double> block = estimator.completeBlock();
  for(const double value : block) {
    expectNear(value, 7.0 / 3.0, 2.0e-15,
               "block mean must use physical-time trapezoidal weights");
  }
  const std::vector<double> updated = estimator.previewUpdate(block, 3.0);
  const double expected =
    (1.0 - std::exp(-3.0 / 2.0)) * (7.0 / 3.0);
  for(const double value : updated) {
    expectNear(value, expected, 2.0e-15,
               "EMA update must use its physical-time coefficient");
  }
  estimator.commitUpdate(updated);
  expectNear(estimator.estimate()[0], expected, 2.0e-15,
             "EMA commit must retain the previewed estimate");
}

void checkObservations() {
  const burgers::Grid grid(0.0, 1.0, 8);
  burgers::environment::ObservationBuilder builder(grid, 0.01, 1u);
  burgers::State velocity(std::vector<double>{
    0.0, 0.1, 0.3, 0.2, -0.1, -0.4, -0.2, 0.0});
  burgers::State translated = velocity;
  for(double& value : translated) value += 7.5;
  const std::vector<std::vector<double>> original = builder.build(velocity);
  const std::vector<std::vector<double>> galilean =
    builder.build(translated);
  for(std::size_t cell = 0; cell < original.size(); ++cell) {
    expectNear(maximumDifference(original[cell], galilean[cell]),
               0.0, 2.0e-13,
               "gradient observations must be Galilean invariant");
  }
  expect(original.size() == grid.cellCount() && original[0].size() == 3u,
         "radius-one observation must have three entries per cell");

  const std::vector<double> shifted_values =
    cyclicShift(velocity.cellAverages(), 2u);
  const std::vector<std::vector<double>> shifted =
    builder.build(burgers::State(shifted_values));
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    expect(original[cell] == shifted[(cell + 2u) % grid.cellCount()],
           "periodic observations must commute with cyclic shifts");
  }
  expectThrows<std::invalid_argument>(
    [&grid]() {
      burgers::environment::ObservationBuilder invalid(grid, 0.0, 1u);
      static_cast<void>(invalid);
    },
    "observation builder must reject nonpositive viscosity");
}

void checkRewards() {
  const burgers::Grid grid(0.0, 1.0, 4);
  const std::vector<double> target{1.0, -1.0, 1.0, -1.0};
  const std::vector<double> pre{0.5, -0.5, 0.5, -0.5};
  const std::vector<double> block{0.75, -0.75, 0.75, -0.75};
  const std::vector<double> updated{0.8, -0.8, 0.8, -0.8};
  const std::vector<double> forcing{0.25, -0.25, 0.25, -0.25};

  burgers::environment::RewardConfig config;
  config.type = burgers::environment::RewardType::SquaredEma;
  config.aggregation = burgers::environment::RewardAggregation::Local;
  burgers::environment::RewardModel local_squared(
    grid, target, 0.5, config);
  expectNear(local_squared.velocityScale(), 1.0, 2.0e-15,
             "automatic velocity scale must equal target RMS");
  const std::vector<double> local = local_squared.evaluate(
    pre, block, updated, forcing);
  for(const double reward : local) {
    expectNear(reward, -0.04, 2.0e-15,
               "local EMA2 reward must expose squared target error");
  }

  config.aggregation = burgers::environment::RewardAggregation::SharedGlobal;
  burgers::environment::RewardModel shared_squared(
    grid, target, 0.5, config);
  const std::vector<double> shared = shared_squared.evaluate(
    pre, block, updated, forcing);
  for(const double reward : shared) {
    expectNear(reward, -0.04, 2.0e-15,
               "shared EMA2 reward must be normalized profile error");
  }

  config.type =
    burgers::environment::RewardType::DirectionalBlockVelocity;
  config.aggregation = burgers::environment::RewardAggregation::Local;
  burgers::environment::RewardModel directional(grid, target, 0.5, config);
  const std::vector<double> direction = directional.evaluate(
    pre, block, updated, forcing);
  for(const double reward : direction) {
    expectNear(reward, -0.25, 2.0e-15,
               "directional reward must use cached pre-action error sign");
  }
}

void checkAcceptedTarget() {
  const std::string metadata =
    std::string(BURGERS_TEST_SOURCE_DIR) +
    "/runs/finalization/dns_target_metadata.json";
  const burgers::environment::TargetProfile target =
    burgers::environment::TargetProfile::loadAccepted(metadata);
  expect(target.cellCount() == 1024u,
         "accepted target loader must honor metadata cell count");
  expect(target.profileHash().size() == 64u &&
         target.spectrumHash().size() == 64u,
         "accepted target loader must retain verified SHA-256 hashes");
  const burgers::Grid les(0.0, 6.283185307179586, 64u);
  const std::vector<double> restricted = target.restrictMeanVelocity(les);
  expect(restricted.size() == les.cellCount(),
         "accepted target must conservatively restrict to the LES grid");
  double expected_first = 0.0;
  for(std::size_t cell = 0; cell < 16u; ++cell) {
    expected_first += target.meanVelocity()[cell];
  }
  expected_first /= 16.0;
  expectNear(restricted[0], expected_first, 0.0,
             "restriction must average complete fine-cell groups");
}

}  // namespace

int main() {
  checkActionProjection();
  checkMeanEstimator();
  checkObservations();
  checkRewards();
  checkAcceptedTarget();

  if(failures != 0) {
    std::cerr << failures << " environment-component check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 8 environment-component checks passed\n";
  return 0;
}
