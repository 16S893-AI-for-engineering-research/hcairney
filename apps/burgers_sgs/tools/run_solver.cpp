#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/InitialCondition.h"
#include "burgers/RunMetadata.h"
#include "burgers/State.h"

#include <algorithm>
#include <cmath>
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
    << "  --profiles FILE            Profile-history CSV filename\n"
    << "                             (default: burgers_profiles.csv)\n"
    << "  --no-initial-profile       Do not save the initial condition\n"
    << "  -h, --help                 Show this help message\n";
}

bool parseCommandLine(int argc,
                      char* argv[],
                      burgers::RunConfig& config) {
  for(int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if(argument == "-h" || argument == "--help") {
      return false;
    }
    if(argument == "--no-initial-profile") {
      config.output.write_initial_profile = false;
      continue;
    }
    if(argument == "--profiles") {
      if(index + 1 >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      const std::string value(argv[++index]);
      if(value.empty()) {
        throw std::invalid_argument(argument + " requires a filename");
      }
      config.output.profile_history_filename = value;
      continue;
    }
    throw std::invalid_argument("unknown option: " + argument);
  }
  return true;
}

struct HistoryRow {
  double time;
  double mean;
  double kinetic_energy;
  double spatial_variance;
  double molecular_dissipation;
  double deterministic_power;
  double stochastic_power;
  double manufactured_power;
  double interval_numerical_dissipation_rate;
};

HistoryRow makeHistoryRow(double time,
                          const burgers::BurgersSolver& solver,
                          const burgers::State& state,
                          const burgers::AdvanceResult* interval = nullptr) {
  burgers::ForcingFields forcing(solver.grid());
  solver.forcingFields(time, forcing);
  const burgers::ForcingPower power =
    burgers::forcingPower(solver.grid(), state, forcing);
  double molecular_dissipation = burgers::molecularDissipation(
    solver.grid(), state, solver.molecularViscosity());
  double deterministic_power = power.deterministic;
  double stochastic_power = power.stochastic;
  double manufactured_power = power.manufactured;
  double numerical_dissipation = 0.0;
  if(interval != nullptr && interval->final_time > interval->initial_time) {
    const double duration = interval->final_time - interval->initial_time;
    molecular_dissipation = interval->molecular_dissipation / duration;
    deterministic_power = interval->deterministic_work / duration;
    stochastic_power = interval->stochastic_work / duration;
    manufactured_power = interval->manufactured_work / duration;
    numerical_dissipation = interval->numerical_dissipation / duration;
  }
  return HistoryRow{
    time,
    burgers::mean(solver.grid(), state),
    burgers::kineticEnergy(solver.grid(), state),
    burgers::spatialVariance(solver.grid(), state),
    molecular_dissipation,
    deterministic_power,
    stochastic_power,
    manufactured_power,
    numerical_dissipation};
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
            "deterministic_power,stochastic_power,manufactured_power,"
            "interval_numerical_dissipation_rate\n";
  for(const HistoryRow& row : history) {
    output << row.time << ',' << row.mean << ',' << row.kinetic_energy << ','
           << row.spatial_variance << ',' << row.molecular_dissipation << ','
           << row.deterministic_power << ',' << row.stochastic_power << ','
           << row.manufactured_power << ','
           << row.interval_numerical_dissipation_rate << '\n';
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

}  // namespace

int main(int argc, char* argv[]) {
  burgers::RunConfig config = burgers::makePhase5ValidationRunConfig();
  try {
    if(!parseCommandLine(argc, argv, config)) {
      printUsage(std::cout, argv[0]);
      return 0;
    }
  } catch(const std::exception& error) {
    std::cerr << "burgers_sgs: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  }

  burgers::RunResultMetadata result;
  result.phase = 5;
  result.final_time = config.time_integration.initial_time;
  const std::vector<std::string> errors = burgers::validate(config);

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
    const std::string profile_history_path = outputFilePath(
      config.output, config.output.profile_history_filename);
    std::ofstream profile_history(profile_history_path);
    if(!profile_history) {
      throw std::runtime_error(
        "unable to open profile-history file: " + profile_history_path);
    }
    profile_history.imbue(std::locale::classic());
    profile_history
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "step,time,x,cell_average\n";
    if(config.output.write_initial_profile) {
      writeProfileSnapshot(
        profile_history, 0, time, solver.grid(), state);
    }

    history.push_back(makeHistoryRow(time, solver, state));
    if(time >= config.output.statistics_start_time) {
      accumulateSpectrum(solver.grid(), state, spectrum_sum);
      ++result.statistics_sample_count;
    }

    std::size_t output_index = 1;
    while(time < final_time) {
      const double scheduled_time =
        config.time_integration.initial_time +
        static_cast<double>(output_index) * config.output.history_interval;
      const double target_time =
        std::isfinite(scheduled_time) != 0
          ? std::min(scheduled_time, final_time)
          : final_time;
      if(target_time <= time) {
        throw std::runtime_error(
          "history interval is too small to advance physical time");
      }
      if(result.timestep_count >= config.time_integration.maximum_steps) {
        throw std::runtime_error(
          "maximum timestep count reached before final time");
      }
      const std::size_t remaining_steps =
        config.time_integration.maximum_steps - result.timestep_count;
      const burgers::AdvanceResult advance =
        solver.advanceTo(state, time, target_time, remaining_steps);
      result.timestep_count += advance.timestep_count;
      result.shortened_final_step_count +=
        advance.shortened_final_step_count;
      result.forcing_clock_step_count += advance.forcing_clock_step_count;
      result.deterministic_work += advance.deterministic_work;
      result.stochastic_work += advance.stochastic_work;
      result.manufactured_work += advance.manufactured_work;
      result.molecular_dissipation += advance.molecular_dissipation;
      result.numerical_dissipation += advance.numerical_dissipation;
      result.energy_change += advance.energy_change;
      time = advance.final_time;
      result.final_time = time;
      result.numerical_advancement_performed =
        result.timestep_count != 0;
      history.push_back(makeHistoryRow(time, solver, state, &advance));
      writeProfileSnapshot(profile_history,
                           result.timestep_count,
                           time,
                           solver.grid(),
                           state);
      if(time >= config.output.statistics_start_time) {
        accumulateSpectrum(solver.grid(), state, spectrum_sum);
        ++result.statistics_sample_count;
      }
      ++output_index;
    }

    result.status = burgers::RunStatus::Completed;
    result.message = "Phase 5 configurable forced solve completed.";
    result.numerical_advancement_performed = result.timestep_count != 0;
    result.final_time = time;

    profile_history.close();
    if(!profile_history) {
      throw std::runtime_error(
        "unable to write profile-history file: " + profile_history_path);
    }

    const std::string history_path = outputFilePath(
      config.output, config.output.scalar_history_filename);
    const std::string profile_path = outputFilePath(
      config.output, config.output.final_profile_filename);
    const std::string spectrum_path = outputFilePath(
      config.output, config.output.spectrum_filename);
    writeHistory(history_path, history);
    writeFinalProfile(profile_path, time, solver.grid(), state);
    writeMeanSpectrum(spectrum_path,
                      spectrum_sum,
                      result.statistics_sample_count);
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
