#include "burgers/Forcing.h"

#include "burgers/ManufacturedSolution.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace burgers {
namespace {

const double pi = 3.1415926535897932384626433832795;
const char* const restart_magic = "BURGERS_FORCING_STATE_V2";

bool enablesDeterministic(ForcingType type) {
  return type == ForcingType::DeterministicFourier ||
         type == ForcingType::Composite;
}

bool enablesStochastic(ForcingType type) {
  return type == ForcingType::StochasticFourierOu ||
         type == ForcingType::Composite;
}

double sineCellAverage(double left,
                       double right,
                       double width,
                       double wavenumber,
                       double phase) {
  return (std::cos(wavenumber * left + phase) -
          std::cos(wavenumber * right + phase)) /
         (wavenumber * width);
}

double cosineCellAverage(double left,
                         double right,
                         double width,
                         double wavenumber) {
  return (std::sin(wavenumber * right) -
          std::sin(wavenumber * left)) /
         (wavenumber * width);
}

std::uint64_t forcingSeed(std::uint64_t run_seed) {
  // SplitMix64 finalizer with a fixed stream tag. This keeps forcing
  // randomness independent of the initial-condition generator.
  std::uint64_t value = run_seed ^ 0x6a09e667f3bcc909ULL;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

}  // namespace

ForcingFields::ForcingFields(const Grid& grid)
  : manufactured(grid, 0.0),
    deterministic(grid, 0.0),
    stochastic(grid, 0.0),
    prescribed(grid, 0.0),
    total(grid, 0.0) {}

Forcing::Forcing(const Grid& grid,
                 const ForcingConfig& config,
                 double molecular_viscosity,
                 std::uint64_t run_seed)
  : grid_(grid),
    config_(config),
    molecular_viscosity_(molecular_viscosity),
    deterministic_enabled_(enablesDeterministic(config.type)),
    stochastic_enabled_(enablesStochastic(config.type)),
    manufactured_enabled_(config.type == ForcingType::Manufactured),
    random_engine_(forcingSeed(run_seed)),
    standard_normal_(0.0, 1.0),
    run_seed_(run_seed),
    clock_index_(0),
    current_clock_time_(config.stochastic.clock_reference_time),
    step_prepared_(false),
    prepared_final_time_(config.stochastic.clock_reference_time) {
  if(!stochastic_enabled_) {
    return;
  }

  const std::size_t mode_count = config_.stochastic.wavenumbers.size();
  stochastic_variances_.resize(mode_count, 0.0);
  cosine_coefficients_.resize(mode_count, 0.0);
  sine_coefficients_.resize(mode_count, 0.0);

  double weight_sum = 0.0;
  for(const int mode : config_.stochastic.wavenumbers) {
    weight_sum += std::pow(static_cast<double>(mode),
                           -config_.stochastic.spectral_exponent);
  }
  if(!std::isfinite(weight_sum) || weight_sum <= 0.0) {
    throw std::invalid_argument(
      "stochastic forcing spectral weights must have a finite positive sum");
  }

  const double variance_sum =
    config_.stochastic.stationary_rms *
    config_.stochastic.stationary_rms;
  for(std::size_t mode = 0; mode < mode_count; ++mode) {
    const double weight = std::pow(
      static_cast<double>(config_.stochastic.wavenumbers[mode]),
      -config_.stochastic.spectral_exponent);
    stochastic_variances_[mode] = variance_sum * weight / weight_sum;
    const double deviation = std::sqrt(stochastic_variances_[mode]);
    cosine_coefficients_[mode] = deviation * standard_normal_(random_engine_);
    sine_coefficients_[mode] = deviation * standard_normal_(random_engine_);
  }
}

ForcingType Forcing::type() const noexcept {
  return config_.type;
}

bool Forcing::hasStochasticComponent() const noexcept {
  return stochastic_enabled_;
}

double Forcing::clockInterval() const noexcept {
  return config_.stochastic.clock_interval;
}

double Forcing::currentClockTime() const noexcept {
  return current_clock_time_;
}

std::size_t Forcing::currentClockIndex() const noexcept {
  return clock_index_;
}

double Forcing::clockTolerance(double time) const noexcept {
  return 64.0 * std::numeric_limits<double>::epsilon() *
         std::max({1.0, std::abs(time), std::abs(current_clock_time_),
                   config_.stochastic.clock_interval});
}

void Forcing::advanceClock() {
  const double rho = std::exp(
    -config_.stochastic.clock_interval /
    config_.stochastic.correlation_time);
  const double innovation_factor = std::sqrt(
    std::max(0.0, 1.0 - rho * rho));
  for(std::size_t mode = 0; mode < stochastic_variances_.size(); ++mode) {
    const double innovation_deviation =
      std::sqrt(stochastic_variances_[mode]) * innovation_factor;
    cosine_coefficients_[mode] =
      rho * cosine_coefficients_[mode] +
      innovation_deviation * standard_normal_(random_engine_);
    sine_coefficients_[mode] =
      rho * sine_coefficients_[mode] +
      innovation_deviation * standard_normal_(random_engine_);
  }
  ++clock_index_;
  current_clock_time_ =
    config_.stochastic.clock_reference_time +
    static_cast<double>(clock_index_) * config_.stochastic.clock_interval;
}

void Forcing::synchronizeIdleTo(double time) {
  if(!stochastic_enabled_) {
    return;
  }
  if(step_prepared_) {
    throw std::logic_error(
      "cannot synchronize stochastic forcing while an RK step is prepared");
  }
  if(!std::isfinite(time)) {
    throw std::invalid_argument("forcing time must be finite");
  }
  const double tolerance = clockTolerance(time);
  if(time < current_clock_time_ - tolerance) {
    throw std::invalid_argument(
      "stochastic forcing cannot move backward in physical time; restore a "
      "saved forcing state before restarting from an earlier time");
  }

  double next_clock =
    current_clock_time_ + config_.stochastic.clock_interval;
  while(time >= next_clock - tolerance) {
    advanceClock();
    next_clock = current_clock_time_ + config_.stochastic.clock_interval;
  }
}

double Forcing::timeUntilNextClock(double time) {
  if(!stochastic_enabled_) {
    return std::numeric_limits<double>::infinity();
  }
  synchronizeIdleTo(time);
  const double remaining =
    current_clock_time_ + config_.stochastic.clock_interval - time;
  if(remaining <= 0.0 || !std::isfinite(remaining)) {
    throw std::runtime_error(
      "stochastic forcing clock did not produce a positive interval");
  }
  return remaining;
}

void Forcing::prepareStep(double time, double time_step) {
  if(!stochastic_enabled_) {
    return;
  }
  if(step_prepared_) {
    throw std::logic_error("a stochastic forcing step is already prepared");
  }
  synchronizeIdleTo(time);
  const double final_time = time + time_step;
  const double next_clock =
    current_clock_time_ + config_.stochastic.clock_interval;
  if(final_time > next_clock + clockTolerance(final_time)) {
    throw std::invalid_argument(
      "an SSP-RK3 step must not cross a stochastic forcing-clock boundary");
  }
  step_prepared_ = true;
  prepared_final_time_ = final_time;
}

void Forcing::commitStep(double final_time) {
  if(!stochastic_enabled_) {
    return;
  }
  if(!step_prepared_) {
    throw std::logic_error("no stochastic forcing step is prepared");
  }
  if(std::abs(final_time - prepared_final_time_) >
     clockTolerance(final_time)) {
    throw std::invalid_argument(
      "committed PDE time does not match the prepared forcing step");
  }
  const double next_clock =
    current_clock_time_ + config_.stochastic.clock_interval;
  step_prepared_ = false;
  if(std::abs(final_time - next_clock) <= clockTolerance(final_time)) {
    advanceClock();
  }
}

void Forcing::cancelPreparedStep() noexcept {
  step_prepared_ = false;
}

void Forcing::requireFieldCompatibility(const ForcingFields& fields) const {
  if(fields.manufactured.size() != grid_.cellCount() ||
     fields.deterministic.size() != grid_.cellCount() ||
     fields.stochastic.size() != grid_.cellCount() ||
     fields.prescribed.size() != grid_.cellCount() ||
     fields.total.size() != grid_.cellCount()) {
    throw std::invalid_argument(
      "forcing output fields must match the forcing grid");
  }
}

void Forcing::removeDiscreteMean(State& field) const {
  double sum = 0.0;
  double correction = 0.0;
  for(const double value : field) {
    const double term = value - correction;
    const double updated = sum + term;
    correction = (updated - sum) - term;
    sum = updated;
  }
  const double discrete_mean = sum / static_cast<double>(field.size());
  for(double& value : field) {
    value -= discrete_mean;
  }
}

void Forcing::fillDeterministic(State& field) const {
  field.fill(0.0);
  const double width = grid_.cellWidth();
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    const double left = static_cast<double>(cell) * width;
    const double right = left + width;
    for(const FourierModeConfig& mode : config_.deterministic.modes) {
      const double physical_wavenumber =
        2.0 * pi * static_cast<double>(mode.wavenumber) / grid_.length();
      field[cell] += mode.amplitude * sineCellAverage(
        left, right, width, physical_wavenumber, mode.phase);
    }
  }
  if(config_.remove_discrete_mean) {
    removeDiscreteMean(field);
  }
}

void Forcing::fillStochastic(State& field) const {
  field.fill(0.0);
  const double width = grid_.cellWidth();
  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    const double left = static_cast<double>(cell) * width;
    const double right = left + width;
    for(std::size_t mode = 0;
        mode < config_.stochastic.wavenumbers.size(); ++mode) {
      const double physical_wavenumber =
        2.0 * pi *
        static_cast<double>(config_.stochastic.wavenumbers[mode]) /
        grid_.length();
      field[cell] +=
        cosine_coefficients_[mode] * cosineCellAverage(
          left, right, width, physical_wavenumber) +
        sine_coefficients_[mode] * sineCellAverage(
          left, right, width, physical_wavenumber, 0.0);
    }
  }
  if(config_.remove_discrete_mean) {
    removeDiscreteMean(field);
  }
}

void Forcing::evaluate(double time, ForcingFields& fields) const {
  if(!std::isfinite(time)) {
    throw std::invalid_argument("forcing evaluation time must be finite");
  }
  requireFieldCompatibility(fields);
  fields.manufactured.fill(0.0);
  fields.deterministic.fill(0.0);
  fields.stochastic.fill(0.0);
  fields.prescribed.fill(0.0);
  fields.total.fill(0.0);

  if(manufactured_enabled_) {
    computeManufacturedSourceCellAverages(
      grid_, config_.manufactured, molecular_viscosity_, time,
      fields.manufactured);
  }
  if(deterministic_enabled_) {
    fillDeterministic(fields.deterministic);
  }
  if(stochastic_enabled_) {
    const double tolerance = clockTolerance(time);
    const double next_clock =
      current_clock_time_ + config_.stochastic.clock_interval;
    if(!step_prepared_ &&
       (time < current_clock_time_ - tolerance ||
        time >= next_clock - tolerance)) {
      throw std::invalid_argument(
        "stochastic forcing evaluation time is outside the current clock "
        "interval; synchronize through timeUntilNextClock or prepareStep");
    }
    fillStochastic(fields.stochastic);
  }

  for(std::size_t cell = 0; cell < grid_.cellCount(); ++cell) {
    fields.total[cell] = fields.manufactured[cell] +
                         fields.deterministic[cell] +
                         fields.stochastic[cell];
    if(!std::isfinite(fields.total[cell])) {
      throw std::runtime_error(
        "forcing produced a non-finite value at cell " +
        std::to_string(cell));
    }
  }
}

std::string Forcing::serializeStochasticState() const {
  if(!stochastic_enabled_) {
    throw std::logic_error(
      "cannot serialize stochastic state when stochastic forcing is disabled");
  }
  if(step_prepared_) {
    throw std::logic_error(
      "cannot serialize stochastic state during a prepared RK step");
  }

  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << restart_magic << '\n'
         << run_seed_ << '\n'
         << clock_index_ << '\n'
         << current_clock_time_ << '\n'
         << config_.stochastic.stationary_rms << ' '
         << config_.stochastic.spectral_exponent << ' '
         << config_.stochastic.correlation_time << ' '
         << config_.stochastic.clock_interval << ' '
         << config_.stochastic.clock_reference_time << '\n'
         << cosine_coefficients_.size() << '\n';
  for(std::size_t mode = 0; mode < cosine_coefficients_.size(); ++mode) {
    output << config_.stochastic.wavenumbers[mode] << ' '
           << stochastic_variances_[mode] << ' '
           << cosine_coefficients_[mode] << ' '
           << sine_coefficients_[mode] << '\n';
  }
  output << random_engine_ << '\n' << standard_normal_ << '\n';
  return output.str();
}

void Forcing::restoreStochasticState(const std::string& serialized) {
  if(!stochastic_enabled_) {
    throw std::logic_error(
      "cannot restore stochastic state when stochastic forcing is disabled");
  }
  if(step_prepared_) {
    throw std::logic_error(
      "cannot restore stochastic state during a prepared RK step");
  }

  std::istringstream input(serialized);
  std::string magic;
  std::uint64_t restored_seed = 0;
  std::size_t restored_index = 0;
  double restored_time = 0.0;
  double restored_rms = 0.0;
  double restored_exponent = 0.0;
  double restored_correlation_time = 0.0;
  double restored_interval = 0.0;
  double restored_reference_time = 0.0;
  std::size_t restored_count = 0;
  input >> magic >> restored_seed >> restored_index >> restored_time
        >> restored_rms >> restored_exponent >> restored_correlation_time
        >> restored_interval >> restored_reference_time >> restored_count;
  if(!input || magic != restart_magic) {
    throw std::invalid_argument("invalid stochastic forcing restart header");
  }
  if(restored_seed != run_seed_) {
    throw std::invalid_argument(
      "stochastic forcing restart seed does not match configuration");
  }
  if(restored_count != cosine_coefficients_.size()) {
    throw std::invalid_argument(
      "stochastic forcing restart mode count does not match configuration");
  }

  std::vector<double> restored_cosine(restored_count, 0.0);
  std::vector<double> restored_sine(restored_count, 0.0);
  for(std::size_t mode = 0; mode < restored_count; ++mode) {
    int restored_wavenumber = 0;
    double restored_variance = 0.0;
    input >> restored_wavenumber >> restored_variance
          >> restored_cosine[mode] >> restored_sine[mode];
    if(!input ||
       restored_wavenumber != config_.stochastic.wavenumbers[mode] ||
       restored_variance != stochastic_variances_[mode] ||
       !std::isfinite(restored_cosine[mode]) ||
       !std::isfinite(restored_sine[mode])) {
      throw std::invalid_argument(
        "stochastic forcing restart modes do not match configuration");
    }
  }

  std::mt19937_64 restored_engine;
  std::normal_distribution<double> restored_normal;
  input >> restored_engine >> restored_normal;
  if(!input || !std::isfinite(restored_time)) {
    throw std::invalid_argument("invalid stochastic forcing restart state");
  }
  if(restored_rms != config_.stochastic.stationary_rms ||
     restored_exponent != config_.stochastic.spectral_exponent ||
     restored_correlation_time != config_.stochastic.correlation_time ||
     restored_interval != config_.stochastic.clock_interval ||
     restored_reference_time !=
       config_.stochastic.clock_reference_time) {
    throw std::invalid_argument(
      "stochastic forcing restart parameters do not match configuration");
  }
  const double expected_time =
    config_.stochastic.clock_reference_time +
    static_cast<double>(restored_index) * config_.stochastic.clock_interval;
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
    std::max({1.0, std::abs(expected_time), std::abs(restored_time)});
  if(std::abs(restored_time - expected_time) > tolerance) {
    throw std::invalid_argument(
      "stochastic forcing restart clock is inconsistent with configuration");
  }

  clock_index_ = restored_index;
  current_clock_time_ = restored_time;
  cosine_coefficients_ = restored_cosine;
  sine_coefficients_ = restored_sine;
  random_engine_ = restored_engine;
  standard_normal_ = restored_normal;
}

}  // namespace burgers
