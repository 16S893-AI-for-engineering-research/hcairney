#include "burgers/RunOutputRecorder.h"

#include "burgers/Diagnostics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace burgers {
namespace {

struct HistoryRow {
  double time;
  double mean;
  double kinetic_energy;
  double spatial_variance;
  double molecular_dissipation;
  double sgs_dissipation;
  double minimum_coefficient;
  double mean_coefficient;
  double maximum_coefficient;
  double mean_eddy_viscosity;
  double maximum_eddy_viscosity;
  double deterministic_power;
  double stochastic_power;
  double prescribed_power;
  double manufactured_power;
  double interval_numerical_dissipation_rate;
  double interval_start_time;
  double interval_duration;
  double total_power;
  double interval_energy_change_rate;
  double interval_budget_residual_rate;
};

std::string outputFilePath(const OutputConfig& output,
                           const std::string& filename) {
  if(output.directory.empty() || output.directory == ".") {
    return std::string("./") + filename;
  }
  const char last = output.directory.back();
  return last == '/' || last == '\\'
    ? output.directory + filename
    : output.directory + '/' + filename;
}

bool sameTime(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
    32.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool finiteState(const State& state) {
  for(const double value : state) {
    if(std::isfinite(value) == 0) return false;
  }
  return true;
}

HistoryRow makeHistoryRow(double time,
                          const BurgersSolver& solver,
                          const State& state,
                          const AdvanceResult* interval = nullptr) {
  ForcingFields forcing(solver.grid());
  ClosureFields closure(solver.grid());
  solver.forcingFields(time, forcing);
  solver.closureFields(state, closure);
  const ForcingPower power = forcingPower(solver.grid(), state, forcing);
  const ClosureStatistics closure_statistics =
    closureStatistics(solver.grid(), closure);
  double molecular_dissipation = molecularDissipation(
    solver.grid(), state, solver.molecularViscosity());
  double sgs_dissipation = burgers::sgsDissipation(
    solver.grid(), state, closure.eddy_viscosity,
    solver.faceViscosityAveraging());
  double deterministic_power = power.deterministic;
  double stochastic_power = power.stochastic;
  double prescribed_power = power.prescribed;
  double manufactured_power = power.manufactured;
  double numerical_dissipation = 0.0;
  double interval_start_time = time;
  double interval_duration = 0.0;
  double total_power = power.total;
  double energy_change_rate = 0.0;
  double budget_residual_rate = 0.0;
  if(interval != nullptr && interval->final_time > interval->initial_time) {
    const double duration = interval->final_time - interval->initial_time;
    interval_start_time = interval->initial_time;
    interval_duration = duration;
    molecular_dissipation = interval->molecular_dissipation / duration;
    sgs_dissipation = interval->sgs_dissipation / duration;
    deterministic_power = interval->deterministic_work / duration;
    stochastic_power = interval->stochastic_work / duration;
    prescribed_power = interval->prescribed_work / duration;
    manufactured_power = interval->manufactured_work / duration;
    numerical_dissipation = interval->numerical_dissipation / duration;
    total_power = deterministic_power + stochastic_power + prescribed_power +
      manufactured_power;
    energy_change_rate = interval->energy_change / duration;
    budget_residual_rate = total_power - molecular_dissipation -
      sgs_dissipation - numerical_dissipation - energy_change_rate;
  }
  return HistoryRow{
    time,
    mean(solver.grid(), state),
    kineticEnergy(solver.grid(), state),
    spatialVariance(solver.grid(), state),
    molecular_dissipation,
    sgs_dissipation,
    closure_statistics.minimum_coefficient,
    closure_statistics.mean_coefficient,
    closure_statistics.maximum_coefficient,
    closure_statistics.mean_eddy_viscosity,
    closure_statistics.maximum_eddy_viscosity,
    deterministic_power,
    stochastic_power,
    prescribed_power,
    manufactured_power,
    numerical_dissipation,
    interval_start_time,
    interval_duration,
    total_power,
    energy_change_rate,
    budget_residual_rate};
}

void writeHistory(const std::string& path,
                  const std::vector<HistoryRow>& history) {
  std::ofstream output(path);
  if(!output) {
    throw std::runtime_error("unable to open scalar history file: " + path);
  }
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "time,mean,kinetic_energy,spatial_variance,molecular_dissipation,"
            "sgs_dissipation,minimum_coefficient,mean_coefficient,"
            "maximum_coefficient,mean_eddy_viscosity,"
            "maximum_eddy_viscosity,"
            "deterministic_power,stochastic_power,prescribed_power,"
            "manufactured_power,"
            "interval_numerical_dissipation_rate,interval_start_time,"
            "interval_duration,total_power,interval_energy_change_rate,"
            "interval_budget_residual_rate\n";
  for(const HistoryRow& row : history) {
    output << row.time << ',' << row.mean << ',' << row.kinetic_energy << ','
           << row.spatial_variance << ',' << row.molecular_dissipation << ','
           << row.sgs_dissipation << ',' << row.minimum_coefficient << ','
           << row.mean_coefficient << ',' << row.maximum_coefficient << ','
           << row.mean_eddy_viscosity << ','
           << row.maximum_eddy_viscosity << ','
           << row.deterministic_power << ',' << row.stochastic_power << ','
           << row.prescribed_power << ',' << row.manufactured_power << ','
           << row.interval_numerical_dissipation_rate << ','
           << row.interval_start_time << ',' << row.interval_duration << ','
           << row.total_power << ',' << row.interval_energy_change_rate << ','
           << row.interval_budget_residual_rate << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write scalar history file: " + path);
  }
}

void writeProfileSnapshot(std::ostream& output,
                          std::size_t step,
                          double time,
                          const Grid& grid,
                          const State& state) {
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    output << step << ',' << time << ',' << grid.cellCenter(cell) << ','
           << state[cell] << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write profile-history CSV");
  }
}

void writeFinalProfile(const std::string& path,
                       double time,
                       const Grid& grid,
                       const State& state) {
  std::ofstream output(path);
  if(!output) {
    throw std::runtime_error("unable to open final profile file: " + path);
  }
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "time,x,cell_average\n";
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    output << time << ',' << grid.cellCenter(cell) << ',' << state[cell]
           << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write final profile file: " + path);
  }
}

void accumulateSpectrum(const Grid& grid,
                        const State& state,
                        std::vector<double>& sum) {
  const std::vector<double> sample = energySpectrum(grid, state);
  if(sum.size() != sample.size()) {
    throw std::invalid_argument(
      "spectrum accumulator size does not match sampled spectrum");
  }
  for(std::size_t mode = 0; mode < sample.size(); ++mode) {
    sum[mode] += sample[mode];
  }
}

void writeMeanSpectrum(const std::string& path,
                       const std::vector<double>& spectrum_sum,
                       std::size_t sample_count) {
  if(sample_count == 0u) {
    throw std::runtime_error(
      "cannot write a mean spectrum without post-spin-up samples");
  }
  std::ofstream output(path);
  if(!output) {
    throw std::runtime_error("unable to open spectrum file: " + path);
  }
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "wavenumber,mean_energy,k_five_thirds_mean_energy,sample_count\n";
  for(std::size_t mode = 0; mode < spectrum_sum.size(); ++mode) {
    const double mean_energy =
      spectrum_sum[mode] / static_cast<double>(sample_count);
    const double compensated = mode == 0u
      ? 0.0
      : std::pow(static_cast<double>(mode), 5.0 / 3.0) * mean_energy;
    output << mode << ',' << mean_energy << ',' << compensated << ','
           << sample_count << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write spectrum file: " + path);
  }
}

}  // namespace

struct RunOutputRecorder::Implementation {
  explicit Implementation(const RunConfig& input_config)
    : config(input_config) {}

  RunConfig config;
  std::vector<HistoryRow> history;
  std::vector<double> spectrum_sum;
  std::ofstream profile_history;
  std::string profile_history_path;
  AdvanceResult history_interval;
  AdvanceResult total_advance;
  std::size_t history_index = 1u;
  std::size_t profile_index = 1u;
  std::size_t statistics_sample_count = 0u;
  double initial_time = 0.0;
  double final_time = 0.0;
  double last_history_time = 0.0;
  double last_profile_time = 0.0;
  double last_statistics_time = 0.0;
  bool has_profile_sample = false;
  bool has_statistics_sample = false;
  bool begun = false;
  bool finished = false;

  bool profileEventsEnabled() const noexcept {
    return config.output.write_profile_history ||
      config.output.write_online_spectrum;
  }

  double nextHistoryTime() const noexcept {
    return initial_time + static_cast<double>(history_index) *
      config.output.history_interval;
  }

  double nextProfileTime() const noexcept {
    return profileEventsEnabled()
      ? initial_time + static_cast<double>(profile_index) *
          config.output.profile_interval
      : std::numeric_limits<double>::infinity();
  }
};

RunOutputRecorder::RunOutputRecorder(const RunConfig& config)
  : implementation_(new Implementation(config)) {}

RunOutputRecorder::~RunOutputRecorder() = default;

void RunOutputRecorder::begin(const BurgersSolver& solver,
                              const State& state,
                              double initial_time,
                              double final_time) {
  Implementation& data = *implementation_;
  if(data.begun) throw std::logic_error("run output recorder already begun");
  if(final_time < initial_time) {
    throw std::invalid_argument(
      "run output final time must not precede initial time");
  }
  data.initial_time = initial_time;
  data.final_time = final_time;
  data.last_history_time = initial_time;
  data.last_profile_time = initial_time;
  data.last_statistics_time = initial_time;
  data.spectrum_sum.assign(solver.grid().cellCount() / 2u + 1u, 0.0);

  if(data.config.output.write_profile_history) {
    data.profile_history_path = outputFilePath(
      data.config.output, data.config.output.profile_history_filename);
    data.profile_history.open(data.profile_history_path);
    if(!data.profile_history) {
      throw std::runtime_error(
        "unable to open profile-history file: " + data.profile_history_path);
    }
    data.profile_history.imbue(std::locale::classic());
    data.profile_history
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "step,time,x,cell_average\n";
  }
  if(data.config.output.write_profile_history &&
     data.config.output.write_initial_profile) {
    writeProfileSnapshot(
      data.profile_history, 0u, initial_time, solver.grid(), state);
    data.has_profile_sample = true;
  }

  data.history.push_back(makeHistoryRow(initial_time, solver, state));
  const bool initial_statistics_sample =
    data.config.output.write_online_spectrum ||
    (data.config.output.write_profile_history &&
     data.config.output.write_initial_profile);
  if(initial_statistics_sample &&
     initial_time >= data.config.output.statistics_start_time) {
    if(data.config.output.write_online_spectrum) {
      accumulateSpectrum(solver.grid(), state, data.spectrum_sum);
    }
    ++data.statistics_sample_count;
    data.has_statistics_sample = true;
  }

  data.history_interval.initial_time = initial_time;
  data.history_interval.final_time = initial_time;
  data.total_advance.initial_time = initial_time;
  data.total_advance.final_time = initial_time;
  data.begun = true;
}

double RunOutputRecorder::nextEventTime() const noexcept {
  const Implementation& data = *implementation_;
  if(!data.begun || data.finished) {
    return std::numeric_limits<double>::infinity();
  }
  return std::min(
    data.final_time, std::min(data.nextHistoryTime(), data.nextProfileTime()));
}

void RunOutputRecorder::observeAdvance(const BurgersSolver& solver,
                                       const State& state,
                                       const AdvanceResult& advance) {
  Implementation& data = *implementation_;
  if(!data.begun || data.finished) {
    throw std::logic_error("run output recorder is not active");
  }
  accumulateAdvanceResult(data.history_interval, advance);
  accumulateAdvanceResult(data.total_advance, advance);
  const double time = advance.final_time;
  const double next_history_time = data.nextHistoryTime();
  const double next_profile_time = data.nextProfileTime();
  const bool history_due = sameTime(time, next_history_time) ||
    sameTime(time, data.final_time);
  const bool profile_due = data.profileEventsEnabled() &&
    (sameTime(time, next_profile_time) || sameTime(time, data.final_time));

  if(history_due) {
    data.history.push_back(
      makeHistoryRow(time, solver, state, &data.history_interval));
    data.last_history_time = time;
    data.history_interval = AdvanceResult{};
    data.history_interval.initial_time = time;
    data.history_interval.final_time = time;
  }
  if(profile_due && data.config.output.write_profile_history) {
    writeProfileSnapshot(data.profile_history,
                         data.total_advance.timestep_count,
                         time,
                         solver.grid(),
                         state);
    data.last_profile_time = time;
    data.has_profile_sample = true;
  }
  if(profile_due && time >= data.config.output.statistics_start_time) {
    if(data.config.output.write_online_spectrum) {
      accumulateSpectrum(solver.grid(), state, data.spectrum_sum);
    }
    ++data.statistics_sample_count;
    data.last_statistics_time = time;
    data.has_statistics_sample = true;
  }
  if(sameTime(time, next_history_time)) ++data.history_index;
  if(data.profileEventsEnabled() && sameTime(time, next_profile_time)) {
    ++data.profile_index;
  }
}

void RunOutputRecorder::finish(const BurgersSolver& solver,
                               const State& state,
                               double time) {
  Implementation& data = *implementation_;
  if(!data.begun || data.finished) {
    throw std::logic_error("run output recorder is not active");
  }
  if(finiteState(state)) {
    if(!sameTime(time, data.last_history_time)) {
      const AdvanceResult* interval =
        data.history_interval.final_time > data.history_interval.initial_time
        ? &data.history_interval
        : nullptr;
      data.history.push_back(makeHistoryRow(time, solver, state, interval));
      data.last_history_time = time;
    }
    if(data.config.output.write_profile_history &&
       (!data.has_profile_sample || !sameTime(time, data.last_profile_time))) {
      writeProfileSnapshot(data.profile_history,
                           data.total_advance.timestep_count,
                           time,
                           solver.grid(),
                           state);
      data.last_profile_time = time;
      data.has_profile_sample = true;
    }
    if(data.profileEventsEnabled() &&
       time >= data.config.output.statistics_start_time &&
       (!data.has_statistics_sample ||
        !sameTime(time, data.last_statistics_time))) {
      if(data.config.output.write_online_spectrum) {
        accumulateSpectrum(solver.grid(), state, data.spectrum_sum);
      }
      ++data.statistics_sample_count;
      data.last_statistics_time = time;
      data.has_statistics_sample = true;
    }
  }

  if(data.config.output.write_profile_history) {
    data.profile_history.close();
    if(!data.profile_history) {
      throw std::runtime_error(
        "unable to write profile-history file: " + data.profile_history_path);
    }
  }
  writeHistory(outputFilePath(data.config.output,
                              data.config.output.scalar_history_filename),
               data.history);
  if(finiteState(state)) {
    writeFinalProfile(outputFilePath(
                        data.config.output,
                        data.config.output.final_profile_filename),
                      time, solver.grid(), state);
  }
  if(data.config.output.write_online_spectrum) {
    writeMeanSpectrum(outputFilePath(data.config.output,
                                     data.config.output.spectrum_filename),
                      data.spectrum_sum,
                      data.statistics_sample_count);
  }
  data.finished = true;
}

const AdvanceResult& RunOutputRecorder::totalAdvance() const noexcept {
  return implementation_->total_advance;
}

std::size_t RunOutputRecorder::statisticsSampleCount() const noexcept {
  return implementation_->statistics_sample_count;
}

}  // namespace burgers
