#pragma once

#include "burgers/environment/SGSEnvironment.h"

#include <cstdint>
#include <string>
#include <vector>

namespace burgers {
namespace environment {

struct EnvironmentOutputConfig {
  std::string resolved_environment_filename =
    "burgers_rl_resolved.json";
  std::string resolved_solver_filename =
    "burgers_solver_resolved.json";
  std::string episode_summary_filename =
    "burgers_rl_episodes.csv";
  // Detailed per-episode output is intentionally opt-in so existing training
  // and evaluation runs retain their original numerical advancement path.
  bool write_evaluation_output = false;
  std::string evaluation_directory = "evaluation";
};

struct EnvironmentApplicationConfig {
  SGSEnvironmentConfig environment;
  // These are retained for provenance. The environment and solver paths are
  // resolved relative to the source RL configuration when it is loaded.
  std::string source_path;
  std::string solver_config_path;
  std::vector<std::uint64_t> evaluation_seeds;
  EnvironmentOutputConfig output;
};

// Load the strict, versioned application document used by the SMARTIES
// wrapper. Relative solver and target paths are interpreted relative to the
// directory containing this document.
EnvironmentApplicationConfig loadEnvironmentApplicationConfig(
  const std::string& path);

// Serialize all environment-level settings after path resolution. The solver
// configuration is deliberately serialized separately with serializeRunConfig
// so both resolved documents remain directly readable.
std::string serializeEnvironmentApplicationConfig(
  const EnvironmentApplicationConfig& config);

}  // namespace environment
}  // namespace burgers
