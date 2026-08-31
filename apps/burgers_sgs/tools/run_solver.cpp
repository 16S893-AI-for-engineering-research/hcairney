#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/ConfigIO.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/RunMetadata.h"
#include "burgers/State.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string joinErrors(const std::vector<std::string>& errors) {
  std::ostringstream message;
  for(std::size_t index = 0; index < errors.size(); ++index) {
    if(index != 0) {
      message << "; ";
    }
    message << errors[index];
  }
  return message.str();
}

std::string outputFilePath(const burgers::OutputConfig& output,
                           const std::string& filename) {
  if(output.directory.empty() || output.directory == ".") {
    return std::string("./") + filename;
  }
  const char last = output.directory.back();
  if(last == '/' || last == '\\') {
    return output.directory + filename;
  }
  return output.directory + '/' + filename;
}

void printUsage(std::ostream& output, const char* executable) {
  output
    << "Usage: " << executable << " [options]\n"
    << "Options:\n"
    << "  --config FILE              Load a complete versioned JSON config\n"
    << "  --output-directory DIR     Override the configured output directory\n"
    << "  --seed INTEGER             Override the configured random seed\n"
    << "  --dry-run                  Validate and print the resolved config\n"
    << "  --profiles FILE            Profile-history CSV filename\n"
    << "                             (default: burgers_profiles.csv)\n"
    << "  --no-initial-profile       Do not save the initial condition\n"
    << "  -h, --help                 Show this help message\n";
}

struct CommandLine {
  std::string config_path;
  std::string output_directory;
  std::string profiles_filename;
  std::uint64_t seed = 0;
  bool has_seed = false;
  bool disable_initial_profile = false;
  bool dry_run = false;
  bool show_help = false;
};

std::uint64_t parseSeed(const std::string& value) {
  if(value.empty() || value[0] == '-') {
    throw std::invalid_argument("--seed requires a nonnegative integer");
  }
  std::size_t parsed = 0;
  unsigned long long converted = 0;
  try {
    converted = std::stoull(value, &parsed, 10);
  } catch(const std::exception&) {
    throw std::invalid_argument("--seed requires a nonnegative integer");
  }
  if(parsed != value.size()) {
    throw std::invalid_argument("--seed requires a nonnegative integer");
  }
  return static_cast<std::uint64_t>(converted);
}

CommandLine parseCommandLine(int argc, char* argv[]) {
  CommandLine options;
  for(int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if(argument == "-h" || argument == "--help") {
      options.show_help = true;
      continue;
    }
    if(argument == "--no-initial-profile") {
      options.disable_initial_profile = true;
      continue;
    }
    if(argument == "--dry-run") {
      options.dry_run = true;
      continue;
    }
    if(argument == "--config" || argument == "--output-directory" ||
       argument == "--seed" || argument == "--profiles") {
      if(index + 1 >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      const std::string value(argv[++index]);
      if(value.empty()) {
        throw std::invalid_argument(argument + " requires a value");
      }
      if(argument == "--config") {
        options.config_path = value;
      } else if(argument == "--output-directory") {
        options.output_directory = value;
      } else if(argument == "--seed") {
        options.seed = parseSeed(value);
        options.has_seed = true;
      } else {
        options.profiles_filename = value;
      }
      continue;
    }
    throw std::invalid_argument("unknown option: " + argument);
  }
  return options;
}

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

HistoryRow makeHistoryRow(double time,
                          const burgers::BurgersSolver& solver,
                          const burgers::State& state,
                          const burgers::AdvanceResult* interval = nullptr) {
  burgers::ForcingFields forcing(solver.grid());
  burgers::ClosureFields closure(solver.grid());
  solver.forcingFields(time, forcing);
  solver.closureFields(state, closure);
  const burgers::ForcingPower power =
    burgers::forcingPower(solver.grid(), state, forcing);
  const burgers::ClosureStatistics closure_statistics =
    burgers::closureStatistics(solver.grid(), closure);
  double molecular_dissipation = burgers::molecularDissipation(
    solver.grid(), state, solver.molecularViscosity());
  double sgs_dissipation = burgers::sgsDissipation(
    solver.grid(),
    state,
    closure.eddy_viscosity,
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
    burgers::mean(solver.grid(), state),
    burgers::kineticEnergy(solver.grid(), state),
    burgers::spatialVariance(solver.grid(), state),
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

void accumulateSpectrum(const burgers::Grid& grid,
                        const burgers::State& state,
                        std::vector<double>& sum) {
  const std::vector<double> sample = burgers::energySpectrum(grid, state);
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
  if(sample_count == 0) {
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
    const double compensated = mode == 0
      ? 0.0
      : std::pow(static_cast<double>(mode), 5.0 / 3.0) * mean_energy;
    output << mode << ',' << mean_energy << ',' << compensated << ','
           << sample_count << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write spectrum file: " + path);
  }
}

void writeFinalProfile(const std::string& path,
                       double time,
                       const burgers::Grid& grid,
                       const burgers::State& state) {
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

void writeProfileSnapshot(std::ostream& output,
                          std::size_t step,
                          double time,
                          const burgers::Grid& grid,
                          const burgers::State& state) {
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    output << step << ',' << time << ',' << grid.cellCenter(cell) << ','
           << state[cell] << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write profile-history CSV");
  }
}

void accumulateAdvance(burgers::AdvanceResult& total,
                       const burgers::AdvanceResult& increment) {
  if(total.timestep_count == 0 && total.final_time == total.initial_time) {
    total.initial_time = increment.initial_time;
  }
  total.final_time = increment.final_time;
  total.timestep_count += increment.timestep_count;
  total.shortened_final_step_count += increment.shortened_final_step_count;
  total.forcing_clock_step_count += increment.forcing_clock_step_count;
  total.deterministic_work += increment.deterministic_work;
  total.stochastic_work += increment.stochastic_work;
  total.prescribed_work += increment.prescribed_work;
  total.manufactured_work += increment.manufactured_work;
  total.molecular_dissipation += increment.molecular_dissipation;
  total.sgs_dissipation += increment.sgs_dissipation;
  total.numerical_dissipation += increment.numerical_dissipation;
  total.energy_change += increment.energy_change;
}

bool sameTime(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
    32.0 * std::numeric_limits<double>::epsilon() * scale;
}

}  // namespace

int main(int argc, char* argv[]) {
  CommandLine command_line;
  try {
    command_line = parseCommandLine(argc, argv);
    if(command_line.show_help) {
      printUsage(std::cout, argv[0]);
      return 0;
    }
  } catch(const std::exception& error) {
    std::cerr << "burgers_sgs: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  }

  burgers::RunConfig config;
  try {
    config = command_line.config_path.empty()
      ? burgers::makePhase5ValidationRunConfig()
      : burgers::loadRunConfig(command_line.config_path);
    if(!command_line.output_directory.empty()) {
      config.output.directory = command_line.output_directory;
    }
    if(command_line.has_seed) {
      config.random.seed = command_line.seed;
    }
    if(!command_line.profiles_filename.empty()) {
      config.output.profile_history_filename =
        command_line.profiles_filename;
      config.output.write_profile_history = true;
    }
    if(command_line.disable_initial_profile) {
      config.output.write_initial_profile = false;
    }
  } catch(const std::exception& error) {
    std::cerr << "burgers_sgs: " << error.what() << '\n';
    return 2;
  }

  const std::vector<std::string> errors = burgers::validate(config);
  if(command_line.dry_run) {
    if(!errors.empty()) {
      std::cerr << "burgers_sgs: invalid configuration: "
                << joinErrors(errors) << '\n';
      return 1;
    }
    std::cout << burgers::serializeRunConfig(config);
    return 0;
  }

  burgers::RunResultMetadata result;
  result.phase = 7;
  result.final_time = config.time_integration.initial_time;
  if(!errors.empty()) {
    result.status = burgers::RunStatus::InvalidConfiguration;
    result.message = joinErrors(errors);
  }

  if(!errors.empty()) {
    const std::string metadata_path = burgers::metadataFilePath(config.output);
    try {
      burgers::writeRunMetadata(metadata_path, config, result);
    } catch(const std::exception& error) {
      std::cerr << "burgers_sgs: " << error.what() << '\n';
      return 2;
    }
    std::cerr << "burgers_sgs: invalid configuration: "
              << result.message << '\n'
              << "Metadata written to " << metadata_path << '\n';
    return 1;
  }

  try {
    const burgers::BurgersSolver solver(config);
    burgers::State state(solver.grid());
    burgers::initializeState(solver.grid(),
                             state,
                             config.initial_condition,
                             config.random.seed);
    static_cast<void>(solver.stableTimeStep(
      state, config.time_integration.initial_time));

    std::vector<HistoryRow> history;
    std::vector<double> spectrum_sum(solver.grid().cellCount() / 2 + 1,
                                     0.0);
    double time = config.time_integration.initial_time;
    const double final_time = config.time_integration.final_time;
    const bool profile_events_enabled =
      config.output.write_profile_history ||
      config.output.write_online_spectrum;
    std::string profile_history_path;
    std::ofstream profile_history;
    if(config.output.write_profile_history) {
      profile_history_path = outputFilePath(
        config.output, config.output.profile_history_filename);
      profile_history.open(profile_history_path);
      if(!profile_history) {
        throw std::runtime_error(
          "unable to open profile-history file: " + profile_history_path);
      }
      profile_history.imbue(std::locale::classic());
      profile_history
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "step,time,x,cell_average\n";
    }
    if(config.output.write_profile_history &&
       config.output.write_initial_profile) {
      writeProfileSnapshot(
        profile_history, 0, time, solver.grid(), state);
    }

    history.push_back(makeHistoryRow(time, solver, state));
    const bool initial_statistics_sample =
      config.output.write_online_spectrum ||
      (config.output.write_profile_history &&
       config.output.write_initial_profile);
    if(initial_statistics_sample &&
       time >= config.output.statistics_start_time) {
      if(config.output.write_online_spectrum) {
        accumulateSpectrum(solver.grid(), state, spectrum_sum);
      }
      ++result.statistics_sample_count;
    }

    burgers::AdvanceResult history_interval;
    history_interval.initial_time = time;
    history_interval.final_time = time;
    std::size_t history_index = 1;
    std::size_t profile_index = 1;
    while(time < final_time) {
      const double next_history_time =
        config.time_integration.initial_time +
        static_cast<double>(history_index) * config.output.history_interval;
      const double scheduled_profile_time =
        config.time_integration.initial_time +
        static_cast<double>(profile_index) * config.output.profile_interval;
      const double next_profile_time = profile_events_enabled
        ? scheduled_profile_time
        : std::numeric_limits<double>::infinity();
      double target_time = final_time;
      if(std::isfinite(next_history_time) != 0) {
        target_time = std::min(target_time, next_history_time);
      }
      if(std::isfinite(next_profile_time) != 0) {
        target_time = std::min(target_time, next_profile_time);
      }
      if(target_time <= time) {
        throw std::runtime_error(
          "an output interval is too small to advance physical time");
      }
      if(result.timestep_count >= config.time_integration.maximum_steps) {
        throw std::runtime_error(
          "maximum timestep count reached before final time");
      }
      const std::size_t remaining_steps =
        config.time_integration.maximum_steps - result.timestep_count;
      const burgers::AdvanceResult advance =
        solver.advanceTo(state, time, target_time, remaining_steps);
      accumulateAdvance(history_interval, advance);
      result.timestep_count += advance.timestep_count;
      result.shortened_final_step_count +=
        advance.shortened_final_step_count;
      result.forcing_clock_step_count += advance.forcing_clock_step_count;
      result.deterministic_work += advance.deterministic_work;
      result.stochastic_work += advance.stochastic_work;
      result.prescribed_work += advance.prescribed_work;
      result.manufactured_work += advance.manufactured_work;
      result.molecular_dissipation += advance.molecular_dissipation;
      result.sgs_dissipation += advance.sgs_dissipation;
      result.numerical_dissipation += advance.numerical_dissipation;
      result.energy_change += advance.energy_change;
      time = advance.final_time;
      result.final_time = time;
      result.numerical_advancement_performed =
        result.timestep_count != 0;

      const bool history_due = sameTime(time, next_history_time) ||
        sameTime(time, final_time);
      const bool profile_due = profile_events_enabled &&
        (sameTime(time, next_profile_time) || sameTime(time, final_time));
      if(history_due) {
        history.push_back(
          makeHistoryRow(time, solver, state, &history_interval));
        history_interval = burgers::AdvanceResult{};
        history_interval.initial_time = time;
        history_interval.final_time = time;
      }
      if(profile_due && config.output.write_profile_history) {
        writeProfileSnapshot(profile_history,
                             result.timestep_count,
                             time,
                             solver.grid(),
                             state);
      }
      if(profile_due && time >= config.output.statistics_start_time) {
        if(config.output.write_online_spectrum) {
          accumulateSpectrum(solver.grid(), state, spectrum_sum);
        }
        ++result.statistics_sample_count;
      }
      if(sameTime(time, next_history_time)) {
        ++history_index;
      }
      if(profile_events_enabled && sameTime(time, next_profile_time)) {
        ++profile_index;
      }
    }

    result.status = burgers::RunStatus::Completed;
    result.message = "Phase 7 closure-capable forced solve completed.";
    result.numerical_advancement_performed = result.timestep_count != 0;
    result.final_time = time;

    if(config.output.write_profile_history) {
      profile_history.close();
      if(!profile_history) {
        throw std::runtime_error(
          "unable to write profile-history file: " + profile_history_path);
      }
    }

    const std::string history_path = outputFilePath(
      config.output, config.output.scalar_history_filename);
    const std::string profile_path = outputFilePath(
      config.output, config.output.final_profile_filename);
    writeHistory(history_path, history);
    writeFinalProfile(profile_path, time, solver.grid(), state);
    if(config.output.write_online_spectrum) {
      const std::string spectrum_path = outputFilePath(
        config.output, config.output.spectrum_filename);
      writeMeanSpectrum(spectrum_path,
                        spectrum_sum,
                        result.statistics_sample_count);
    }
  } catch(const std::exception& error) {
    result.status = burgers::RunStatus::Failed;
    result.message = error.what();
  }

  const std::string metadata_path = burgers::metadataFilePath(config.output);
  try {
    burgers::writeRunMetadata(metadata_path, config, result);
  } catch(const std::exception& error) {
    std::cerr << "burgers_sgs: " << error.what() << '\n';
    return 2;
  }

  if(result.status != burgers::RunStatus::Completed) {
    std::cerr << "burgers_sgs: solve failed: " << result.message << '\n'
              << "Metadata written to " << metadata_path << '\n';
    return 1;
  }

  std::cout << result.message << '\n'
            << "Advanced " << result.timestep_count << " timestep(s) to t="
            << result.final_time << ". Metadata written to "
            << metadata_path << '\n';
  return 0;
}
