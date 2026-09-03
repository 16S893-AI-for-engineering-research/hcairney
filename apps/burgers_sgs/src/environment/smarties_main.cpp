#include "burgers/ConfigIO.h"
#include "burgers/RunMetadata.h"
#include "burgers/RunOutputRecorder.h"
#include "burgers/environment/EnvironmentConfigIO.h"
#include "burgers/environment/SGSEnvironment.h"

#include "smarties.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {

using burgers::environment::EnvironmentApplicationConfig;
using burgers::environment::EnvironmentAdvanceObserver;
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

std::string joinPath(const std::string& directory,
                     const std::string& filename) {
  if(directory.empty() || directory == ".") return filename;
  const char last = directory.back();
  return last == '/' || last == '\\'
    ? directory + filename
    : directory + '/' + filename;
}

bool isDirectory(const std::string& path) {
  struct stat information;
  return stat(path.c_str(), &information) == 0 &&
    S_ISDIR(information.st_mode);
}

void createOneDirectory(const std::string& path) {
  if(path.empty() || path == "." || path == "/") return;
  if(mkdir(path.c_str(), 0777) == 0) return;
  if(errno == EEXIST && isDirectory(path)) return;
  throw std::runtime_error(
    "unable to create evaluation output directory " + path + ": " +
    std::strerror(errno));
}

void createDirectories(const std::string& path) {
  if(path.empty()) {
    throw std::invalid_argument(
      "evaluation output directory must not be empty");
  }
  std::size_t position = path[0] == '/' ? 1u : 0u;
  while((position = path.find('/', position)) != std::string::npos) {
    createOneDirectory(path.substr(0u, position));
    ++position;
  }
  createOneDirectory(path);
}

std::string episodeDirectory(const std::string& root,
                             std::size_t episode_index,
                             std::uint64_t seed) {
  std::ostringstream name;
  name << "episode_" << std::setw(4) << std::setfill('0') << episode_index
       << "_seed_" << seed;
  return joinPath(root, name.str());
}

void writeOptionalValue(std::ostream& output,
                        const std::vector<double>& values,
                        std::size_t index) {
  output << ',';
  if(index < values.size() && std::isfinite(values[index]) != 0) {
    output << values[index];
  }
}

void writeOptionalMean(std::ostream& output,
                       const std::vector<double>& values) {
  output << ',';
  if(values.empty()) return;
  double sum = 0.0;
  for(const double value : values) {
    if(std::isfinite(value) == 0) return;
    sum += value;
  }
  output << sum / static_cast<double>(values.size());
}

class EvaluationRecorder final : public EnvironmentAdvanceObserver {
public:
  EvaluationRecorder(std::size_t episode_index,
                     std::uint64_t seed,
                     const EnvironmentApplicationConfig& application_config,
                     const std::vector<double>& target_mean)
    : episode_index_(episode_index),
      seed_(seed),
      run_config_(application_config.environment.solver),
      target_mean_(target_mean),
      write_action_spectra_(
        application_config.environment.record_action_spectra),
      directory_(episodeDirectory(
        application_config.output.evaluation_directory,
        episode_index,
        seed)) {
    createDirectories(directory_);
    run_config_.output.directory = directory_;
    run_config_.random.seed = seed_;
  }

  void beginEpisode(const burgers::BurgersSolver& solver,
                    const burgers::State& state,
                    double initial_time,
                    double final_time) override {
    solver_ = &solver;
    run_config_.time_integration.initial_time = initial_time;
    run_config_.time_integration.final_time = final_time;

    openRlOutput();
    standard_output_.reset(new burgers::RunOutputRecorder(run_config_));
    standard_output_->begin(solver, state, initial_time, final_time);
  }

  double nextEventTime() const noexcept override {
    return standard_output_ == nullptr
      ? std::numeric_limits<double>::infinity()
      : standard_output_->nextEventTime();
  }

  void observeAdvance(const burgers::BurgersSolver& solver,
                      const burgers::State& state,
                      const burgers::AdvanceResult& advance) override {
    standard_output_->observeAdvance(solver, state, advance);
  }

  void recordDecision(const std::vector<double>& raw_actions,
                      const StepResult& result,
                      const burgers::Grid& grid) {
    const burgers::environment::StepDiagnostics& diagnostics =
      result.diagnostics;
    const bool projected =
      diagnostics.action.applied.size() == grid.cellCount();

    decision_history_
      << episode_index_ << ',' << seed_ << ','
      << diagnostics.decision_index << ',' << diagnostics.start_time << ','
      << diagnostics.end_time << ',' << (diagnostics.warmup ? 1 : 0) << ','
      << (diagnostics.reward_masked ? 1 : 0) << ','
      << statusName(result.status);
    writeOptionalMean(decision_history_, diagnostics.candidate_rewards);
    writeOptionalMean(decision_history_, result.rewards);
    if(projected) {
      decision_history_
        << ',' << diagnostics.action.raw_statistics.mean
        << ',' << diagnostics.action.raw_statistics.rms
        << ',' << diagnostics.action.raw_statistics.maximum_absolute
        << ',' << diagnostics.action.applied_statistics.mean
        << ',' << diagnostics.action.applied_statistics.rms
        << ',' << diagnostics.action.applied_statistics.maximum_absolute
        << ',' << diagnostics.action.applied_mean_residual
        << ',' << diagnostics.action.saturation_fraction;
    } else {
      decision_history_ << ",,,,,,,,";
    }
    if(diagnostics.updated_mean.empty()) {
      decision_history_ << ',';
    } else {
      decision_history_ << ',' << diagnostics.mean_profile_relative_l2;
    }
    decision_history_
      << ',' << diagnostics.maximum_absolute_velocity
      << ',' << diagnostics.advance.timestep_count
      << ',' << diagnostics.advance.shortened_final_step_count
      << ',' << diagnostics.advance.forcing_clock_step_count
      << ',' << csvField(diagnostics.failure_message) << '\n';

    for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
      field_history_
        << episode_index_ << ',' << seed_ << ','
        << diagnostics.decision_index << ',' << diagnostics.start_time << ','
        << diagnostics.end_time << ',' << cell << ','
        << grid.cellCenter(cell);
      writeOptionalValue(field_history_, raw_actions, cell);
      writeOptionalValue(
        field_history_, diagnostics.action.nondimensional, cell);
      writeOptionalValue(field_history_, diagnostics.action.applied, cell);
      writeOptionalValue(
        field_history_, diagnostics.candidate_rewards, cell);
      writeOptionalValue(field_history_, result.rewards, cell);
      writeOptionalValue(field_history_, diagnostics.pre_action_mean, cell);
      writeOptionalValue(field_history_, diagnostics.block_mean, cell);
      writeOptionalValue(field_history_, diagnostics.updated_mean, cell);
      writeOptionalValue(field_history_, target_mean_, cell);
      field_history_ << '\n';
    }

    if(write_action_spectra_) {
      const std::size_t spectrum_size = std::max(
        diagnostics.raw_action_spectrum.size(),
        diagnostics.applied_action_spectrum.size());
      for(std::size_t mode = 0; mode < spectrum_size; ++mode) {
        spectrum_history_
          << episode_index_ << ',' << seed_ << ','
          << diagnostics.decision_index << ',' << diagnostics.end_time << ','
          << mode;
        writeOptionalValue(
          spectrum_history_, diagnostics.raw_action_spectrum, mode);
        writeOptionalValue(
          spectrum_history_, diagnostics.applied_action_spectrum, mode);
        spectrum_history_ << '\n';
      }
    }
    if(!decision_history_ || !field_history_ ||
       (write_action_spectra_ && !spectrum_history_)) {
      throw std::runtime_error(
        "unable to write SMARTIES evaluation diagnostic CSV");
    }
  }

  void finishEpisode(EpisodeStatus status,
                     const SGSEnvironment& environment) {
    standard_output_->finish(
      solver(), environment.state(), environment.time());
    closeRlOutput();

    const burgers::AdvanceResult& advance =
      standard_output_->totalAdvance();
    burgers::RunResultMetadata result;
    result.phase = 8u;
    result.status = status == EpisodeStatus::Timeout
      ? burgers::RunStatus::Completed
      : burgers::RunStatus::Failed;
    result.message = status == EpisodeStatus::Timeout
      ? "SMARTIES evaluation episode completed."
      : environment.failureMessage();
    result.smarties_linked = true;
    result.numerical_advancement_performed = advance.timestep_count != 0u;
    result.final_time = environment.time();
    result.timestep_count = advance.timestep_count;
    result.shortened_final_step_count = advance.shortened_final_step_count;
    result.forcing_clock_step_count = advance.forcing_clock_step_count;
    result.statistics_sample_count =
      standard_output_->statisticsSampleCount();
    result.deterministic_work = advance.deterministic_work;
    result.stochastic_work = advance.stochastic_work;
    result.prescribed_work = advance.prescribed_work;
    result.manufactured_work = advance.manufactured_work;
    result.molecular_dissipation = advance.molecular_dissipation;
    result.sgs_dissipation = advance.sgs_dissipation;
    result.numerical_dissipation = advance.numerical_dissipation;
    result.energy_change = advance.energy_change;
    burgers::writeRunMetadata(
      burgers::metadataFilePath(run_config_.output), run_config_, result);
  }

private:
  void openRlOutput() {
    decision_history_.open(joinPath(directory_, "burgers_rl_history.csv"));
    field_history_.open(joinPath(directory_, "burgers_rl_fields.csv"));
    if(write_action_spectra_) {
      spectrum_history_.open(
        joinPath(directory_, "burgers_rl_spectra.csv"));
    }
    if(!decision_history_ || !field_history_ ||
       (write_action_spectra_ && !spectrum_history_)) {
      throw std::runtime_error(
        "unable to open SMARTIES evaluation diagnostic CSV");
    }
    decision_history_.imbue(std::locale::classic());
    field_history_.imbue(std::locale::classic());
    if(write_action_spectra_) {
      spectrum_history_.imbue(std::locale::classic());
    }
    const int precision = std::numeric_limits<double>::max_digits10;
    decision_history_ << std::setprecision(precision);
    field_history_ << std::setprecision(precision);
    if(write_action_spectra_) {
      spectrum_history_ << std::setprecision(precision);
    }
    decision_history_
      << "episode_index,episode_seed,decision_index,start_time,end_time,"
         "warmup,reward_masked,status,candidate_reward_mean,"
         "delivered_reward_mean,raw_action_mean,raw_action_rms,"
         "raw_action_maximum_absolute,applied_action_mean,applied_action_rms,"
         "applied_action_maximum_absolute,applied_mean_residual,"
         "saturation_fraction,mean_profile_relative_l2,"
         "maximum_absolute_velocity,timestep_count,"
         "shortened_final_step_count,forcing_clock_step_count,"
         "failure_message\n";
    field_history_
      << "episode_index,episode_seed,decision_index,start_time,end_time,cell,x,"
         "raw_action,nondimensional_action,applied_action,candidate_reward,"
         "delivered_reward,pre_action_mean,block_mean,updated_mean,target_mean\n";
    if(write_action_spectra_) {
      spectrum_history_
        << "episode_index,episode_seed,decision_index,end_time,wavenumber,"
           "raw_action_energy,applied_action_energy\n";
    }
  }

  void closeRlOutput() {
    decision_history_.close();
    field_history_.close();
    if(write_action_spectra_) spectrum_history_.close();
    if(!decision_history_ || !field_history_ ||
       (write_action_spectra_ && !spectrum_history_)) {
      throw std::runtime_error(
        "unable to close SMARTIES evaluation diagnostic CSV");
    }
  }

  const burgers::BurgersSolver& solver() const {
    if(solver_ == nullptr) {
      throw std::logic_error("evaluation recorder has no solver");
    }
    return *solver_;
  }

  std::size_t episode_index_;
  std::uint64_t seed_;
  burgers::RunConfig run_config_;
  std::vector<double> target_mean_;
  bool write_action_spectra_;
  std::string directory_;
  const burgers::BurgersSolver* solver_ = nullptr;
  std::unique_ptr<burgers::RunOutputRecorder> standard_output_;
  std::ofstream decision_history_;
  std::ofstream field_history_;
  std::ofstream spectrum_history_;
};

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
    const std::size_t current_evaluation_episode = evaluation_episode;
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

    std::unique_ptr<EvaluationRecorder> evaluation_recorder;
    if(!training && application_config.output.write_evaluation_output) {
      evaluation_recorder.reset(new EvaluationRecorder(
        current_evaluation_episode,
        seed,
        application_config,
        environment.targetMean()));
      environment.beginObservation(*evaluation_recorder);
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
      StepResult result = evaluation_recorder == nullptr
        ? environment.step(raw_actions)
        : environment.step(raw_actions, evaluation_recorder.get());
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
      if(evaluation_recorder != nullptr) {
        evaluation_recorder->recordDecision(
          raw_actions, result, environment.grid());
      }

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
        if(evaluation_recorder != nullptr) {
          evaluation_recorder->finishEpisode(final_status, environment);
        }
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
