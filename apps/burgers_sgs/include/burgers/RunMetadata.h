#pragma once

#include "burgers/Config.h"

#include <cstddef>
#include <string>

namespace burgers {

enum class RunStatus {
  ScaffoldComplete,
  InvalidConfiguration,
  Failed
};

struct BuildMetadata {
  std::string project_version;
  std::string compiler_id;
  std::string compiler_version;
  std::string build_configuration;
  std::string cmake_version;
  std::string cmake_generator;
  std::string source_revision;
  std::string source_dirty;
  bool assertions_enabled = false;
  bool sanitizers_enabled = false;
};

struct RunResultMetadata {
  RunStatus status = RunStatus::ScaffoldComplete;
  std::string message = "Phase 0 scaffold completed; no PDE was advanced.";
  std::size_t timestep_count = 0;
  std::size_t rejected_step_count = 0;
  std::size_t shortened_final_step_count = 0;
};

BuildMetadata currentBuildMetadata();
const char* toString(RunStatus value);
std::string metadataFilePath(const OutputConfig& output);
std::string serializeRunMetadata(const RunConfig& config,
                                 const RunResultMetadata& result);
void writeRunMetadata(const std::string& path,
                      const RunConfig& config,
                      const RunResultMetadata& result);

}  // namespace burgers

