#include "burgers/environment/EnvironmentConfigIO.h"

#include "burgers/ConfigIO.h"

#include "json.hpp"

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace burgers {
namespace environment {
namespace {

using Json = nlohmann::json;

const std::uint64_t environment_schema_version = 1u;

std::invalid_argument configError(const std::string& message) {
  return std::invalid_argument(
    "invalid Burgers RL configuration: " + message);
}

void requireObject(const Json& value, const std::string& path) {
  if(!value.is_object()) {
    throw configError(path + " must be an object");
  }
}

void requireKeys(const Json& object,
                 std::initializer_list<const char*> allowed,
                 const std::string& path) {
  std::set<std::string> keys;
  for(const char* key : allowed) keys.insert(key);
  for(Json::const_iterator item = object.begin(); item != object.end(); ++item) {
    if(keys.count(item.key()) == 0u) {
      throw configError(path + " contains unknown field " + item.key());
    }
  }
}

const Json& member(const Json& object,
                   const std::string& name,
                   const std::string& path) {
  const Json::const_iterator found = object.find(name);
  if(found == object.end()) {
    throw configError(path + "." + name + " is required");
  }
  return *found;
}

const Json& objectMember(const Json& object,
                         const std::string& name,
                         const std::string& path) {
  const Json& value = member(object, name, path);
  requireObject(value, path + "." + name);
  return value;
}

std::string stringMember(const Json& object,
                         const std::string& name,
                         const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_string()) {
    throw configError(path + "." + name + " must be a string");
  }
  const std::string result = value.get<std::string>();
  if(result.empty()) {
    throw configError(path + "." + name + " must not be empty");
  }
  return result;
}

double doubleMember(const Json& object,
                    const std::string& name,
                    const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_number()) {
    throw configError(path + "." + name + " must be a number");
  }
  return value.get<double>();
}

bool boolMember(const Json& object,
                const std::string& name,
                const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_boolean()) {
    throw configError(path + "." + name + " must be a boolean");
  }
  return value.get<bool>();
}

std::uint64_t unsignedMember(const Json& object,
                             const std::string& name,
                             const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_number_integer()) {
    throw configError(path + "." + name +
                      " must be a nonnegative integer");
  }
  if(value.is_number_unsigned()) return value.get<std::uint64_t>();
  const std::int64_t converted = value.get<std::int64_t>();
  if(converted < 0) {
    throw configError(path + "." + name +
                      " must be a nonnegative integer");
  }
  return static_cast<std::uint64_t>(converted);
}

std::size_t checkedSize(std::uint64_t value, const std::string& path) {
  if(value > static_cast<std::uint64_t>(
               std::numeric_limits<std::size_t>::max())) {
    throw configError(path + " is outside the supported size range");
  }
  return static_cast<std::size_t>(value);
}

std::string directoryName(const std::string& path) {
  const std::string::size_type separator = path.find_last_of("/\\");
  if(separator == std::string::npos) return ".";
  if(separator == 0u) return path.substr(0u, 1u);
  return path.substr(0u, separator);
}

bool isAbsolutePath(const std::string& path) {
  if(path.empty()) return false;
  if(path[0] == '/' || path[0] == '\\') return true;
  return path.size() >= 3u &&
         ((path[0] >= 'A' && path[0] <= 'Z') ||
          (path[0] >= 'a' && path[0] <= 'z')) &&
         path[1] == ':' && (path[2] == '/' || path[2] == '\\');
}

std::string resolveReferencedPath(const std::string& document_path,
                                  const std::string& referenced_path) {
  if(isAbsolutePath(referenced_path)) return referenced_path;
  const std::string directory = directoryName(document_path);
  if(directory == ".") return referenced_path;
  const char last = directory.back();
  if(last == '/' || last == '\\') return directory + referenced_path;
  return directory + '/' + referenced_path;
}

ActionSmoothing parseSmoothing(const std::string& value,
                               const std::string& path) {
  if(value == "none") return ActionSmoothing::None;
  if(value == "periodic_three_point") {
    return ActionSmoothing::PeriodicThreePoint;
  }
  throw configError(path + " has unsupported value " + value);
}

RewardType parseRewardType(const std::string& value,
                           const std::string& path) {
  if(value == "directional_block_velocity") {
    return RewardType::DirectionalBlockVelocity;
  }
  if(value == "squared_ema") return RewardType::SquaredEma;
  throw configError(path + " has unsupported value " + value);
}

RewardAggregation parseRewardAggregation(const std::string& value,
                                         const std::string& path) {
  if(value == "local") return RewardAggregation::Local;
  if(value == "shared_global") return RewardAggregation::SharedGlobal;
  throw configError(path + " has unsupported value " + value);
}

const char* smoothingName(ActionSmoothing value) {
  switch(value) {
    case ActionSmoothing::None: return "none";
    case ActionSmoothing::PeriodicThreePoint:
      return "periodic_three_point";
  }
  return "unknown";
}

const char* rewardTypeName(RewardType value) {
  switch(value) {
    case RewardType::DirectionalBlockVelocity:
      return "directional_block_velocity";
    case RewardType::SquaredEma: return "squared_ema";
  }
  return "unknown";
}

const char* rewardAggregationName(RewardAggregation value) {
  switch(value) {
    case RewardAggregation::Local: return "local";
    case RewardAggregation::SharedGlobal: return "shared_global";
  }
  return "unknown";
}

std::vector<double> doubleArrayMember(const Json& object,
                                      const std::string& name,
                                      const std::string& path) {
  const Json& values = member(object, name, path);
  if(!values.is_array()) {
    throw configError(path + "." + name + " must be an array");
  }
  std::vector<double> result;
  result.reserve(values.size());
  for(std::size_t index = 0; index < values.size(); ++index) {
    if(!values[index].is_number()) {
      throw configError(path + "." + name + "[" +
                        std::to_string(index) + "] must be a number");
    }
    result.push_back(values[index].get<double>());
  }
  return result;
}

std::vector<std::uint64_t> seedArray(const Json& root) {
  const Json& values = member(root, "evaluation_seeds", "document");
  if(!values.is_array() || values.empty()) {
    throw configError(
      "document.evaluation_seeds must be a nonempty array");
  }
  std::vector<std::uint64_t> result;
  result.reserve(values.size());
  for(std::size_t index = 0; index < values.size(); ++index) {
    const std::string path = "document.evaluation_seeds[" +
      std::to_string(index) + "]";
    if(!values[index].is_number_integer()) {
      throw configError(path + " must be a nonnegative integer");
    }
    if(values[index].is_number_unsigned()) {
      result.push_back(values[index].get<std::uint64_t>());
    } else {
      const std::int64_t value = values[index].get<std::int64_t>();
      if(value < 0) throw configError(path + " must be nonnegative");
      result.push_back(static_cast<std::uint64_t>(value));
    }
  }
  return result;
}

}  // namespace

EnvironmentApplicationConfig loadEnvironmentApplicationConfig(
  const std::string& path) {
  std::ifstream input(path);
  if(!input) {
    throw std::runtime_error(
      "unable to open Burgers RL configuration: " + path);
  }

  Json document;
  try {
    input >> document;
  } catch(const Json::exception& error) {
    throw configError(std::string("unable to parse JSON: ") + error.what());
  }

  requireObject(document, "document");
  requireKeys(document,
              {"schema_version", "solver_config", "target_metadata",
               "environment", "evaluation_seeds", "output"},
              "document");
  const std::uint64_t schema = unsignedMember(
    document, "schema_version", "document");
  if(schema != environment_schema_version) {
    throw configError("document.schema_version must be " +
                      std::to_string(environment_schema_version));
  }

  EnvironmentApplicationConfig result;
  result.source_path = path;
  result.solver_config_path = resolveReferencedPath(
    path, stringMember(document, "solver_config", "document"));
  result.environment.solver = loadRunConfig(result.solver_config_path);
  result.environment.target_metadata_path = resolveReferencedPath(
    path, stringMember(document, "target_metadata", "document"));
  result.evaluation_seeds = seedArray(document);

  const Json& environment = objectMember(document, "environment", "document");
  requireKeys(environment,
              {"action", "reward", "decision_interval", "warmup_duration",
               "control_duration", "mean_time_scale", "observation_radius",
               "zero_warmup_rewards", "maximum_absolute_velocity",
               "failure_penalty", "record_action_spectra"},
              "document.environment");

  const Json& action = objectMember(
    environment, "action", "document.environment");
  requireKeys(action,
              {"forcing_scale", "minimum_raw_action", "maximum_raw_action",
               "smoothing"},
              "document.environment.action");
  result.environment.action.forcing_scale = doubleMember(
    action, "forcing_scale", "document.environment.action");
  result.environment.action.minimum_raw_action = doubleMember(
    action, "minimum_raw_action", "document.environment.action");
  result.environment.action.maximum_raw_action = doubleMember(
    action, "maximum_raw_action", "document.environment.action");
  result.environment.action.smoothing = parseSmoothing(
    stringMember(action, "smoothing", "document.environment.action"),
    "document.environment.action.smoothing");

  const Json& reward = objectMember(
    environment, "reward", "document.environment");
  requireKeys(reward,
              {"type", "aggregation", "velocity_scale", "action_penalty",
               "local_squared_weights"},
              "document.environment.reward");
  result.environment.reward.type = parseRewardType(
    stringMember(reward, "type", "document.environment.reward"),
    "document.environment.reward.type");
  result.environment.reward.aggregation = parseRewardAggregation(
    stringMember(reward, "aggregation", "document.environment.reward"),
    "document.environment.reward.aggregation");
  result.environment.reward.velocity_scale = doubleMember(
    reward, "velocity_scale", "document.environment.reward");
  result.environment.reward.action_penalty = doubleMember(
    reward, "action_penalty", "document.environment.reward");
  result.environment.reward.local_squared_weights = doubleArrayMember(
    reward, "local_squared_weights", "document.environment.reward");

  result.environment.decision_interval = doubleMember(
    environment, "decision_interval", "document.environment");
  result.environment.warmup_duration = doubleMember(
    environment, "warmup_duration", "document.environment");
  result.environment.control_duration = doubleMember(
    environment, "control_duration", "document.environment");
  result.environment.mean_time_scale = doubleMember(
    environment, "mean_time_scale", "document.environment");
  result.environment.observation_radius = checkedSize(
    unsignedMember(environment, "observation_radius", "document.environment"),
    "document.environment.observation_radius");
  result.environment.zero_warmup_rewards = boolMember(
    environment, "zero_warmup_rewards", "document.environment");
  result.environment.maximum_absolute_velocity = doubleMember(
    environment, "maximum_absolute_velocity", "document.environment");
  result.environment.failure_penalty = doubleMember(
    environment, "failure_penalty", "document.environment");
  result.environment.record_action_spectra = boolMember(
    environment, "record_action_spectra", "document.environment");

  const Json& output = objectMember(document, "output", "document");
  requireKeys(output,
              {"resolved_environment_filename", "resolved_solver_filename",
               "episode_summary_filename"},
              "document.output");
  result.output.resolved_environment_filename = stringMember(
    output, "resolved_environment_filename", "document.output");
  result.output.resolved_solver_filename = stringMember(
    output, "resolved_solver_filename", "document.output");
  result.output.episode_summary_filename = stringMember(
    output, "episode_summary_filename", "document.output");

  return result;
}

std::string serializeEnvironmentApplicationConfig(
  const EnvironmentApplicationConfig& config) {
  Json document;
  document["schema_version"] = environment_schema_version;
  document["solver_config"] = config.solver_config_path;
  document["target_metadata"] =
    config.environment.target_metadata_path;
  document["evaluation_seeds"] = config.evaluation_seeds;
  document["environment"] = {
    {"action", {
      {"forcing_scale", config.environment.action.forcing_scale},
      {"minimum_raw_action",
       config.environment.action.minimum_raw_action},
      {"maximum_raw_action",
       config.environment.action.maximum_raw_action},
      {"smoothing", smoothingName(config.environment.action.smoothing)}}},
    {"reward", {
      {"type", rewardTypeName(config.environment.reward.type)},
      {"aggregation",
       rewardAggregationName(config.environment.reward.aggregation)},
      {"velocity_scale", config.environment.reward.velocity_scale},
      {"action_penalty", config.environment.reward.action_penalty},
      {"local_squared_weights",
       config.environment.reward.local_squared_weights}}},
    {"decision_interval", config.environment.decision_interval},
    {"warmup_duration", config.environment.warmup_duration},
    {"control_duration", config.environment.control_duration},
    {"mean_time_scale", config.environment.mean_time_scale},
    {"observation_radius", config.environment.observation_radius},
    {"zero_warmup_rewards", config.environment.zero_warmup_rewards},
    {"maximum_absolute_velocity",
     config.environment.maximum_absolute_velocity},
    {"failure_penalty", config.environment.failure_penalty},
    {"record_action_spectra",
     config.environment.record_action_spectra}};
  document["output"] = {
    {"resolved_environment_filename",
     config.output.resolved_environment_filename},
    {"resolved_solver_filename", config.output.resolved_solver_filename},
    {"episode_summary_filename", config.output.episode_summary_filename}};
  return document.dump(2) + "\n";
}

}  // namespace environment
}  // namespace burgers
