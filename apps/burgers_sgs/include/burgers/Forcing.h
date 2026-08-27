#pragma once

#include "burgers/Config.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace burgers {

struct ForcingFields {
  explicit ForcingFields(const Grid& grid);

  State manufactured;
  State deterministic;
  State stochastic;
  State total;
};

// Composes the Phase 5 forcing components and owns the clocked OU state.
// Stochastic coefficients are updated only at fixed physical-time clock
// boundaries, never as a side effect of right-hand-side evaluation.
class Forcing {
public:
  Forcing(const Grid& grid,
          const ForcingConfig& config,
          double molecular_viscosity,
          std::uint64_t run_seed);

  ForcingType type() const noexcept;
  bool hasStochasticComponent() const noexcept;
  double clockInterval() const noexcept;
  double currentClockTime() const noexcept;
  std::size_t currentClockIndex() const noexcept;

  // Synchronize an idle forcing object to time and return the time remaining
  // in its current stochastic interval. Returns infinity without stochastic
  // forcing. This does not prepare an RK step.
  double timeUntilNextClock(double time);

  // Freeze the current stochastic coefficient field for a successful RK
  // step. The step must not cross a forcing-clock boundary. commitStep()
  // advances the OU process only after the PDE state update succeeds.
  void prepareStep(double time, double time_step);
  void commitStep(double final_time);
  void cancelPreparedStep() noexcept;

  void evaluate(double time, ForcingFields& fields) const;

  std::string serializeStochasticState() const;
  void restoreStochasticState(const std::string& serialized);

private:
  void synchronizeIdleTo(double time);
  void advanceClock();
  void fillDeterministic(State& field) const;
  void fillStochastic(State& field) const;
  void removeDiscreteMean(State& field) const;
  void requireFieldCompatibility(const ForcingFields& fields) const;
  double clockTolerance(double time) const noexcept;

  Grid grid_;
  ForcingConfig config_;
  double molecular_viscosity_;
  bool deterministic_enabled_;
  bool stochastic_enabled_;
  bool manufactured_enabled_;
  std::vector<double> stochastic_variances_;
  std::vector<double> cosine_coefficients_;
  std::vector<double> sine_coefficients_;
  std::mt19937_64 random_engine_;
  std::normal_distribution<double> standard_normal_;
  std::uint64_t run_seed_;
  std::size_t clock_index_;
  double current_clock_time_;
  bool step_prepared_;
  double prepared_final_time_;
};

}  // namespace burgers
