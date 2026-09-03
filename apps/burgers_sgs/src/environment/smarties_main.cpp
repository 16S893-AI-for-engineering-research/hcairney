#include "burgers/ConfigIO.h"
#include "burgers/environment/EnvironmentConfigIO.h"
#include "burgers/environment/SGSEnvironment.h"

#include "smarties.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using burgers::environment::EnvironmentApplicationConfig;
using burgers::environment::EpisodeStatus;
using burgers::environment::SGSEnvironment;
using burgers::environment::StepResult;

std::string parseEnvironmentConfigPath(int argc, char** argv) {
  std::string result;
  for(int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    std::string candidate;
    if(argument == "--burgers-config") {
      if(index + 1 >= argc) {
        throw std::invalid_argument("--burgers-config requires a value");
      }
      candidate = argv[++index];
    } else {
      const std::string prefix = "--burgers-config=";
      if(argument.compare(0u, prefix.size(), prefix) == 0) {
        candidate = argument.substr(prefix.size());
      }
    }
    if(candidate.empty()) continue;
    if(!result.empty() && result != candidate) {
      throw std::invalid_argument(
        "multiple conflicting --burgers-config values were supplied");
    }
    result = candidate;
  }
  if(result.empty()) {
    throw std::invalid_argument(
      "the SMARTIES app settings must provide --burgers-config FILE");
  }
  return result;
}

std::string readRequiredFile(const std::string& path) {
  std::ifstream input(path);
  if(!input) throw std::runtime_error("unable to open required file: " + path);
  std::ostringstream contents;
  contents << input.rdbuf();
  if(!input.good() && !input.eof()) {
    throw std::runtime_error("unable to read required file: " + path);
  }
  if(contents.str().empty()) {
    throw std::runtime_error("required file is empty: " + path);
  }
  return contents.str();
}

void writeTextFile(const std::string& path, const std::string& contents) {
  std::ofstream output(path);
  if(!output) throw std::runtime_error("unable to open output file: " + path);
  output << contents;
  if(!output) throw std::runtime_error("unable to write output file: " + path);
}

bool fileHasContent(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if(!input) return false;
  return input.peek() != std::ifstream::traits_type::eof();
}

std::string csvField(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2u);
  escaped.push_back('"');
  for(const char character : value) {
    if(character == '"') escaped.push_back('"');
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

const char* statusName(EpisodeStatus status) {
  switch(status) {
    case EpisodeStatus::Running: return "running";
    case EpisodeStatus::Timeout: return "timeout";
    case EpisodeStatus::Failure: return "failure";
  }
  return "unknown";
}

double meanValue(const std::vector<double>& values) {
  if(values.empty()) {
    throw std::invalid_argument("cannot average an empty reward vector");
  }
  double sum = 0.0;
  for(const double value : values) {
    if(std::isfinite(value) == 0) {
      throw std::runtime_error("SMARTIES-facing reward became non-finite");
    }
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

bool validObservations(const std::vector<std::vector<double>>& observations,
                       std::size_t agent_count,
                       std::size_t observation_size) {
  if(observations.size() != agent_count) return false;
  for(const std::vector<double>& observation : observations) {
    if(observation.size() != observation_size) return false;
    for(const double value : observation) {
      if(std::isfinite(value) == 0) return false;
    }
  }
  return true;
}

std::uint64_t drawTrainingSeed(std::mt19937& generator) {
  const std::uint64_t high = static_cast<std::uint64_t>(generator());
  const std::uint64_t low = static_cast<std::uint64_t>(generator());
  return (high << 32u) | low;
}

std::uint64_t episodeSeed(smarties::Communicator* comm,
                          const EnvironmentApplicationConfig& config,
                          std::size_t evaluation_episode) {
  if(comm->isTraining()) return drawTrainingSeed(comm->getPRNG());
  if(evaluation_episode >= config.evaluation_seeds.size()) {
    throw std::runtime_error(
      "evaluation requested more episodes than the configured held-out "
      "seed list; use one SMARTIES evaluation environment and provide at "
      "least --nEvalEpisodes seeds");
  }
  return config.evaluation_seeds[evaluation_episode];
}

void appendEpisodeSummary(const std::string& path,
                          std::size_t episode_index,
                          bool training,
                          std::uint64_t seed,
                          EpisodeStatus status,
                          std::size_t decision_count,
                          double initial_time,
                          double final_time,
                          double cumulative_mean_reward,
                          double final_profile_error,
                          double maximum_absolute_velocity,
                          const std::string& failure_message) {
  const bool write_header = !fileHasContent(path);
  std::ofstream output(path, std::ios::app);
  if(!output) {
    throw std::runtime_error(
      "unable to open episode-summary output: " + path);
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  if(write_header) {
    output
      << "episode_index,mode,episode_seed,status,decision_count,"
         "initial_time,final_time,cumulative_mean_reward,"
         "final_mean_profile_relative_l2,maximum_absolute_velocity,"
         "failure_message\n";
  }
  output << episode_index << ',' << (training ? "training" : "evaluation")
         << ',' << seed << ',' << statusName(status) << ',' << decision_count
         << ',' << initial_time << ',' << final_time << ','
         << cumulative_mean_reward << ',' << final_profile_error << ','
         << maximum_absolute_velocity << ','
         << csvField(failure_message) << '\n';
  if(!output) {
    throw std::runtime_error(
      "unable to write episode-summary output: " + path);
  }
}

bool sendInitialObservations(
  smarties::Communicator* comm,
  const std::vector<std::vector<double>>& observations) {
  for(std::size_t cell = 0; cell < observations.size(); ++cell) {
    comm->sendInitState(observations[cell], static_cast<int>(cell));
    if(comm->terminateTraining()) return false;
  }
  return true;
}

bool sendStepResult(smarties::Communicator* comm,
                    const StepResult& result,
                    const std::vector<std::vector<double>>& observations) {
  if(result.rewards.size() != observations.size()) {
    throw std::runtime_error(
      "environment reward count does not match the SMARTIES agent count");
  }
  for(std::size_t cell = 0; cell < observations.size(); ++cell) {
    const int agent = static_cast<int>(cell);
    if(result.status == EpisodeStatus::Running) {
      comm->sendState(observations[cell], result.rewards[cell], agent);
    } else if(result.status == EpisodeStatus::Timeout) {
      comm->sendLastState(observations[cell], result.rewards[cell], agent);
    } else {
      comm->sendTermState(observations[cell], result.rewards[cell], agent);
    }
    // SMARTIES attaches its shutdown signal to the action returned by a send.
    // Stop immediately rather than attempting another agent communication.
    if(comm->terminateTraining()) return false;
  }
  return true;
}

void appMain(smarties::Communicator* const comm, int argc, char** argv) {
  const std::string config_path = parseEnvironmentConfigPath(argc, argv);
  const EnvironmentApplicationConfig application_config =
    burgers::environment::loadEnvironmentApplicationConfig(config_path);

  // SMARTIES reads this fixed filename internally. Requiring an explicit file
  // ensures that every run retains a learner-settings record; setupFolder
  // leaves the same file in each simulation directory.
  static_cast<void>(readRequiredFile("settings.json"));

  SGSEnvironment environment(application_config.environment);
  const std::size_t agent_count = environment.grid().cellCount();
  const std::size_t observation_size = environment.observationSize();
  if(agent_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
     observation_size >
       static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
      "Burgers grid or observation size exceeds the SMARTIES integer range");
  }

  writeTextFile(
    application_config.output.resolved_environment_filename,
    burgers::environment::serializeEnvironmentApplicationConfig(
      application_config));
  writeTextFile(
    application_config.output.resolved_solver_filename,
    burgers::serializeRunConfig(application_config.environment.solver));

  comm->setNumAgents(static_cast<int>(agent_count));
  comm->setStateActionDims(static_cast<int>(observation_size), 1);
  comm->setActionScales(
    {application_config.environment.action.maximum_raw_action},
    {application_config.environment.action.minimum_raw_action}, true);
  // The initial policy is intentionally memoryless. In this SMARTIES version,
  // setIsPartiallyObservable() selects a recurrent approximator, so partial
  // observability is documented rather than changing the network here.
  comm->finalizeProblemDescription();

  std::size_t episode_index = 0u;
  std::size_t evaluation_episode = 0u;
  while(true) {
    const bool training = comm->isTraining();
    const bool evaluation_seeds_exhausted =
      !training &&
      evaluation_episode >= application_config.evaluation_seeds.size();
    // SMARTIES decides that evaluation is complete asynchronously after the
    // final terminal state.  It may therefore need one more application
    // communication to deliver its shutdown signal.  Seed zero is used only
    // to construct a valid initial-state probe; it is never recorded as an
    // evaluation episode.  If SMARTIES did request more episodes than were
    // configured, the probe completes normally and the error below remains
    // diagnostic rather than being hidden as a successful shutdown.
    const std::uint64_t seed = evaluation_seeds_exhausted
      ? 0u
      : episodeSeed(comm, application_config, evaluation_episode);
    if(!training && !evaluation_seeds_exhausted) ++evaluation_episode;
    environment.reset(seed);

    std::vector<std::vector<double>> observations =
      environment.observations();
    if(!validObservations(observations, agent_count, observation_size)) {
      throw std::runtime_error(
        "environment produced an invalid initial observation field");
    }
    if(!sendInitialObservations(comm, observations)) return;
    if(evaluation_seeds_exhausted) {
      throw std::runtime_error(
        "evaluation requested more episodes than the configured held-out "
        "seed list; use one SMARTIES evaluation environment and provide at "
        "least --nEvalEpisodes seeds");
    }

    const double initial_time = environment.time();
    std::size_t decision_count = 0u;
    double cumulative_mean_reward = 0.0;
    double final_profile_error =
      std::numeric_limits<double>::quiet_NaN();
    double maximum_absolute_velocity = 0.0;
    EpisodeStatus final_status = EpisodeStatus::Running;
    std::string failure_message;

    while(final_status == EpisodeStatus::Running) {
      std::vector<double> raw_actions(agent_count, 0.0);
      for(std::size_t cell = 0; cell < agent_count; ++cell) {
        if(comm->terminateTraining()) return;
        const std::vector<double> action =
          comm->recvAction(static_cast<int>(cell));
        if(action.size() != 1u) {
          throw std::runtime_error(
            "SMARTIES returned an action with the wrong dimension");
        }
        raw_actions[cell] = action[0];
      }

      const std::vector<std::vector<double>> last_finite_observations =
        observations;
      StepResult result = environment.step(raw_actions);
      ++decision_count;
      cumulative_mean_reward += meanValue(result.rewards);
      if(!result.diagnostics.updated_mean.empty()) {
        final_profile_error = result.diagnostics.mean_profile_relative_l2;
      }
      maximum_absolute_velocity = std::max(
        maximum_absolute_velocity,
        result.diagnostics.maximum_absolute_velocity);
      final_status = result.status;
      failure_message = result.diagnostics.failure_message;

      if(validObservations(result.observations,
                           agent_count, observation_size)) {
        observations = result.observations;
      } else if(result.status == EpisodeStatus::Failure) {
        // A failed PDE state can be non-finite. SMARTIES still requires a
        // finite terminal vector of the declared dimension, so report the last
        // valid pre-action observation with the environment's failure penalty.
        observations = last_finite_observations;
      } else {
        throw std::runtime_error(
          "environment produced invalid nonterminal observations");
      }

      // Record a completed environment episode before its terminal send. The
      // learner can attach the global shutdown signal to that send, in which
      // case control will not return here to write the summary afterward.
      if(final_status != EpisodeStatus::Running) {
        appendEpisodeSummary(
          application_config.output.episode_summary_filename,
          episode_index, training, seed, final_status, decision_count,
          initial_time, environment.time(), cumulative_mean_reward,
          final_profile_error, maximum_absolute_velocity, failure_message);
        ++episode_index;
      }
      if(!sendStepResult(comm, result, observations)) return;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    smarties::Engine engine(argc, argv);
    if(engine.parse()) return 1;
    engine.run(appMain);
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "Burgers SMARTIES error: " << error.what() << '\n';
    return 1;
  }
}
