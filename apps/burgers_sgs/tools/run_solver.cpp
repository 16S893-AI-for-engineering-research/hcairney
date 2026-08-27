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

struct HistoryRow {
  double time;
  double mean;
  double kinetic_energy;
  double molecular_dissipation;
};

HistoryRow makeHistoryRow(double time,
                          const burgers::BurgersSolver& solver,
                          const burgers::State& state) {
  return HistoryRow{
    time,
    burgers::mean(solver.grid(), state),
    burgers::kineticEnergy(solver.grid(), state),
    burgers::molecularDissipation(
      solver.grid(), state, solver.molecularViscosity())};
}

void writeHistory(const std::string& path,
                  const std::vector<HistoryRow>& history) {
  std::ofstream output(path);
  if(!output) {
    throw std::runtime_error("unable to open scalar history file: " + path);
  }
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "time,mean,kinetic_energy,molecular_dissipation\n";
  for(const HistoryRow& row : history) {
    output << row.time << ',' << row.mean << ',' << row.kinetic_energy << ','
           << row.molecular_dissipation << '\n';
  }
  if(!output) {
    throw std::runtime_error("unable to write scalar history file: " + path);
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

}  // namespace

int main() {
  const burgers::RunConfig config = burgers::makeDefaultRunConfig();
  burgers::RunResultMetadata result;
  result.phase = 4;
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
    static_cast<void>(solver.stableTimeStep(state));

    std::vector<HistoryRow> history;
    double time = config.time_integration.initial_time;
    const double final_time = config.time_integration.final_time;
    history.push_back(makeHistoryRow(time, solver, state));

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
      time = advance.final_time;
      result.final_time = time;
      result.numerical_advancement_performed =
        result.timestep_count != 0;
      history.push_back(makeHistoryRow(time, solver, state));
      ++output_index;
    }

    result.status = burgers::RunStatus::Completed;
    result.message = "Phase 4 configurable unclosed solve completed.";
    result.numerical_advancement_performed = result.timestep_count != 0;
    result.final_time = time;

    const std::string history_path = outputFilePath(
      config.output, config.output.scalar_history_filename);
    const std::string profile_path = outputFilePath(
      config.output, config.output.final_profile_filename);
    writeHistory(history_path, history);
    writeFinalProfile(profile_path, time, solver.grid(), state);
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
