#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/ConfigIO.h"
#include "burgers/InitialCondition.h"
#include "burgers/RunMetadata.h"
#include "burgers/RunOutputRecorder.h"
#include "burgers/State.h"

#include <cstdint>
#include <exception>
#include <iostream>
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

    double time = config.time_integration.initial_time;
    const double final_time = config.time_integration.final_time;
    burgers::RunOutputRecorder output(config);
    output.begin(solver, state, time, final_time);
    while(time < final_time) {
      const double target_time = output.nextEventTime();
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
      output.observeAdvance(solver, state, advance);
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
    }

    result.status = burgers::RunStatus::Completed;
    result.message = "Phase 7 closure-capable forced solve completed.";
    result.numerical_advancement_performed = result.timestep_count != 0;
    result.final_time = time;

    output.finish(solver, state, time);
    result.statistics_sample_count = output.statisticsSampleCount();
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
